import jax
from jax.nn import softmax
import jax.numpy as jnp


def attention(
    Q: jax.Array, K: jax.Array, V: jax.Array, mask: jax.Array = None
) -> tuple[jax.Array, jax.Array]:
    d_k = K.shape[-1]

    # (..., seq_length, d_k) @ (..., d_k, seq_length) = (..., seq_length, seq_length)
    scaled = (Q @ K.mT) / (d_k**0.5)

    scaled += mask

    att = softmax(scaled, axis=-1)

    # (..., seq_length, seq_length) @ (..., seq_length, d_k) = (..., seq_length, d_k)
    output = att @ V

    return output, att


def multihead_self_attention(
    config,
    params,
    x,
    mask,
):
    # x.shape = (seq_length, d_model), where d_model is the embedding dimension
    seq_length, d_model = x.shape

    # define Q, K and V sizes
    d_k = d_model // config.num_heads

    # Compute QKV from data using the parameters
    # (seq_length, d_model) @ (d_model, d_model) = (seq_length, d_model)
    Q = x @ params["W_q"]
    K = x @ params["W_k"]
    V = x @ params["W_v"]

    # Split data for multiple heads
    # reshape to split heads resulting in
    # (num_heads, seq_length, d_k) d_k = d_model / num_heads (Ex.: (64 / 8) = 8)
    Q = Q.reshape(seq_length, config.num_heads, d_k).swapaxes(-2, -3)
    K = K.reshape(seq_length, config.num_heads, d_k).swapaxes(-2, -3)
    V = V.reshape(seq_length, config.num_heads, d_k).swapaxes(-2, -3)

    # output.shape = (num_heads, seq_length, d_k)
    output, att = attention(Q, K, V, mask=mask)

    # Concatenate all heads output and get d_model back
    # concat_output.shape = (seq_length, d_model)
    concat_output = output.mT.reshape(seq_length, config.num_heads * d_k)

    # (seq_length, d_model) @ (d_model, d_model) = (seq_length, d_model)
    multihead_att = concat_output @ params["W_up_O"]

    return multihead_att


def layer_normalization(params, X):
    # params are learnable gamma and beta of shape (..., seq_length)
    # X.shape = (..., seq_length, layer_dim)

    layer_mean = X.mean(axis=[-3, -1])[jnp.newaxis, :, jnp.newaxis]
    layer_std = X.std(axis=[-3, -1])[jnp.newaxis, :, jnp.newaxis]

    X += layer_mean
    X /= layer_std + 1e-5  # plus 1e-5 just to avoid div by zero

    # print(f"{params['gamma'].shape=}")
    # print(f"{params['beta'].shape=}")
    # print(f"{X.shape=}")
    # Add scaling and bias trainable params to layer norm
    out = X * params["gamma"] + params["beta"]

    return out


def positional_encoding(seq_length, d_model):
    # create an axis with even indices
    even_i = jnp.arange(0, d_model, 2).astype(jnp.float32)

    # the result of the denominator would be the same (see formula)
    # for even_i and odd_i, so only one can be used

    # Using aˆb = eˆ(b*ln(a)) propriety to transfor that paper
    # equation into the log space (better for computing)
    denominator = jnp.exp(even_i * (-jnp.log(1e4) / d_model))

    # Add a dimention to further interleave even and odd
    position = jnp.arange(seq_length)[:, jnp.newaxis]

    even_PE = jnp.sin(position * denominator)
    odd_PE = jnp.cos(position * denominator)

    # Create the empty matrix
    pe = jnp.zeros((seq_length, d_model))

    # Interleave
    pe = pe.at[:, 0::2].set(even_PE)
    pe = pe.at[:, 1::2].set(odd_PE)

    return pe


def linear(params, X):
    out = X @ params["W"] + params["b"]

    return out


def dropout(X, key, rate, training=False):
    if not training or rate == 0.0:
        return X

    keep_prob = 1 - rate

    mask = jax.random.bernoulli(key, p=keep_prob, shape=X.shape)

    return (X * mask) / keep_prob


def feed_forward(params, X):
    # print(f"{X.shape=}")
    # print(f"{params['linear_in']['W'].shape=}")
    # print(f"{params['linear_in']['b'].shape=}")
    feed_forwarded_in = jax.nn.relu(linear(params["linear_in"], X))
    # droppedout = dropout(feed_forwarded_in, params["dropout_key"], 0.2, True)
    feed_forwarded_out = linear(params["linear_out"], feed_forwarded_in)
    return feed_forwarded_out


def decoder_only(params, x, config, attention_mask):
    # x.shape = (batch_size, seq_length, vocab_size)

    # Embedding
    # embeddings.shape = (batch_size, seq_length, emb_size)
    embeddings = params["embeddings"]
    # print(f"{x.shape=}")
    # print(f"{embeddings.shape=}")
    projections = x @ embeddings
    # print(f"{projections.shape=}")

    # Positional Encoding
    # po.shape = (seq_length, emb_size)
    pe = positional_encoding(config.max_seq_length, config.emb_size)
    # print(f"{pe.shape=}")
    encoded_embs = projections + pe
    # print(f"{encoded_embs.shape=}")

    # Multihead Attention
    # TODO: Maybe this for-loop could be avoided with something like vmap
    for layer in range(config.num_layers):
        # print(f"Layer {layer}")
        layer_params = params["layers"][layer]

        att_params = layer_params["attention"]

        causal_mask = jnp.zeros((config.max_seq_length, config.max_seq_length))
        causal_mask = jnp.full(
            (config.max_seq_length, config.max_seq_length), float("-inf")
        )
        causal_mask = jnp.triu(causal_mask, k=1)
        pad_mask = jnp.vstack([attention_mask] * len(attention_mask))
        mask = causal_mask + pad_mask

        batch_multihead_attention = jax.vmap(
            lambda x: multihead_self_attention(
                config, att_params, x, mask=mask
            ),
            in_axes=(0,),
        )
        # print(f"{att_params['W_q'].shape=}")
        # print(f"{att_params['W_k'].shape=}")
        # print(f"{att_params['W_v'].shape=}")
        # print(f"{encoded_embs.shape=}")
        multihead_att = batch_multihead_attention(encoded_embs)
        # print(f"{multihead_att.shape=}")

        # Add & Norm
        added = multihead_att + encoded_embs
        normalized = layer_normalization(layer_params["ln1"], added)
        # print(f"{normalized.shape=}")

        # Feedforward
        feed_forwarded_out = feed_forward(
            layer_params["feed_forward"], normalized
        )

        # Add & Norm 2
        added_2 = feed_forwarded_out + normalized
        normalized_2 = layer_normalization(layer_params["ln2"], added_2)

    logits = normalized_2 @ embeddings.mT

    probs = jax.nn.softmax(logits)

    return logits, probs
