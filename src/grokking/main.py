import jax
import jax.numpy as jnp
from grokking.model.transformer import decoder_only
import functools as ft
import itertools as it
from grokking.data.generate import generate
from tqdm import tqdm
from omegaconf import DictConfig
import hydra

DTYPE = jnp.float32

def init_param_state(config: DictConfig) -> dict:
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
            next(key), (config.vocab_size, config.emb_size), dtype
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
    X, mask_token, attention_mask = generate(
        max_operands_value=config.max_operands_value,
        max_seq_length=config.max_seq_length,
    )
    X = jnp.array(X)
    attention_mask = jnp.array(attention_mask)

    params = init_param_state(config)

    lr = 1e-3

    print("Training: ")
    model = ft.partial(
        decoder_only, config=config, attention_mask=attention_mask
    )
    # TODO: make this stochastic gradient descent
    for step in tqdm(range(50)):
        # params = jax.tree.map(jax.ref.get, train_state["params"])
        def loss_fn(params, X, y_true):
            logits, probs = model(params, X)
            y_pred = probs[:, 3, :]
            return (1 - jnp.sum(y_true * y_pred)) ** 2

        y_true = X[:, 4]
        loss, grad = jax.value_and_grad(loss_fn, argnums=0)(params, X, y_true)

        # TODO: Implement Adam instead of SGD
        params = jax.tree.map(lambda w, d: w - lr * d, params, grad)

        if step % 10 == 0:
            print(loss)

    _, y_hat = model(params, X)
    print(f"{y_hat.shape=}")
    y_hat_idx = jnp.argmax(y_hat, axis=-1)

    op1 = jnp.argmax(X, axis=-1)[:, 1]
    op2 = jnp.argmax(X, axis=-1)[:, 2]
    y_true = jnp.argmax(X, axis=-1)[:, 4]
    y_pred = y_hat_idx[:, 4]

    for i in range(len(op1)):
        print(f"({op1[i]}, {op2[i]}) = {y_true[i]}, got {y_pred[i]}")


if __name__ == "__main__":
    main()
