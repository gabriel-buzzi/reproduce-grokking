import jax
import jax.numpy as jnp
from grokking.model.transformer import decoder_only
import functools as ft
import itertools as it
from grokking.data.generate import generate, Tokenizer
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
from matplotlib import pyplot as plt

DTYPE = jnp.float32


def init_param_state(config: DictConfig, vocab_size: int) -> dict:
    # Define the root key to be splited
    root_key = jax.random.key(config.param_seed)
    # Defines the iterator that will split the key
    key = map(ft.partial(jax.random.fold_in, root_key), it.count())
    # Defines the factors for weights initialization
    zero_init = jax.nn.initializers.constant(0.0)
    one_init = jax.nn.initializers.constant(1.0)
    he_init = jax.nn.initializers.he_normal(1, 1)
    # Setting
    dtype = DTYPE

    # Start defining params
    params = {
        "embeddings": he_init(
            next(key), (vocab_size, config.emb_size), dtype
        ),
        "layers": {},
    }

    for layer in range(config.num_layers):
        params["layers"][layer] = {
            "attention": {
                "W_q": he_init(
                    next(key), (config.emb_size, config.emb_size), dtype
                ),
                "W_k": he_init(
                    next(key), (config.emb_size, config.emb_size), dtype
                ),
                "W_v": he_init(
                    next(key), (config.emb_size, config.emb_size), dtype
                ),
                "W_up_O": he_init(
                    next(key), (config.emb_size, config.emb_size), dtype
                ),
            },
            "ln1": {
                "gamma": one_init(next(key), (config.emb_size,), dtype),
                "beta": zero_init(next(key), (config.emb_size,)),
            },
            "feed_forward": {
                "linear_in": {
                    "W": he_init(
                        next(key),
                        (config.emb_size, config.feed_forward_size),
                        dtype,
                    ),
                    "b": zero_init(
                        next(key), (config.feed_forward_size,), dtype
                    ),
                },
                "linear_out": {
                    "W": he_init(
                        next(key),
                        (config.feed_forward_size, config.emb_size),
                        dtype,
                    ),
                    "b": zero_init(next(key), (config.emb_size,), dtype),
                },
            },
            "ln2": {
                "gamma": one_init(next(key), (config.emb_size,), dtype),
                "beta": zero_init(next(key), (config.emb_size,)),
            },
        }

    return params


def init_train_state(config: DictConfig) -> dict:
    train_state = {}
    train_state["params"] = init_param_state(config)
    return train_state


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    masked_text_data, text_data, att_mask_data = generate(
        mod_operand=config.mod_operand,
        max_operands_value=config.max_operands_value,
        max_seq_length=config.max_seq_length,
    )

    tokenizer = Tokenizer(config.max_operands_value)
    embeddings_data = tokenizer.tokenize(text_data)
    embeddings_masked_data = tokenizer.tokenize(masked_text_data)

    X = jnp.array(embeddings_masked_data)
    Y = jnp.array(embeddings_data)
    M = jnp.array(att_mask_data)

    vocab_size = tokenizer.vocab_size

    lr = config.learning_rate
    batch_size = config.batch_size

    model = ft.partial(
        decoder_only, config=config
    )
    # TODO: make this stochastic gradient descent
    params = init_param_state(config, vocab_size)
    train_losses = []
    target_token_idx = 6
    for step in tqdm(range(config.n_train_steps)):
        step_loss = 0
        n_batches = len(X) // batch_size
        for batch in range(n_batches):
            batch_slice = slice(batch * batch_size, (batch + 1) * batch_size)
            X_batch = X[batch_slice]
            Y_batch = Y[batch_slice]
            M_batch = M[batch_slice]

            def loss_fn(params, X, Y, M):
                logits, probs = model(params, X, attention_mask=M)
                #TODO: Review this loss calculations and think how to avoid this hardcoded target token index
                y_pred = probs[:, target_token_idx, :]
                y_true = Y[:, target_token_idx]
                return (1 - jnp.sum(y_true * y_pred)) ** 2

            batch_loss, grad = jax.value_and_grad(loss_fn, argnums=0)(
                params, X_batch, Y_batch, M_batch
            )

            step_loss += batch_loss #/ n_batches

            # TODO: Implement Adam instead of SGD
            params = jax.tree.map(lambda w, d: w - lr * d, params, grad)

        if step % 10 == 0:
            print(step_loss)

        train_losses.append(step_loss)

    plt.plot(train_losses, label="train loss", marker="o")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig("history.png", dpi=300)
    plt.close()

    _, y_hat = model(params=params, x=X, attention_mask=M)
    target_indice = jnp.argmax(y_hat, axis=-1)
    y_hat_one_hots = jax.nn.one_hot(target_indice, num_classes=y_hat.shape[-1])

    y_hat = tokenizer.detokenize(y_hat_one_hots.tolist())
    y_true = tokenizer.detokenize(Y.tolist())

    for i in range(len(Y)):
        print(f"Case {i}")
        print(y_true[i][target_token_idx])
        print(y_hat[i][target_token_idx])

if __name__ == "__main__":
    main()
