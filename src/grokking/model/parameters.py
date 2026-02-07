import jax
import functools as ft
from jax.typing import DTypeLike
from omegaconf import DictConfig


def init_param_state(
    config: DictConfig, vocab_size: int, keygen: jax.Array, dtype: DTypeLike
) -> dict:
    zero_init = ft.partial(jax.nn.initializers.constant(0.0), dtype=dtype)
    one_init = ft.partial(jax.nn.initializers.constant(1.0), dtype=dtype)
    he_init = ft.partial(jax.nn.initializers.he_normal(1, 1), dtype=dtype)

    params = {
        "embeddings": he_init(next(keygen), (vocab_size, config.emb_size)),
        "layers": {},
    }

    for layer in range(config.num_layers):
        params["layers"][layer] = {
            "attention": {
                "W_q": he_init(
                    next(keygen), (config.emb_size, config.emb_size)
                ),
                "W_k": he_init(
                    next(keygen), (config.emb_size, config.emb_size)
                ),
                "W_v": he_init(
                    next(keygen), (config.emb_size, config.emb_size)
                ),
                "W_up_O": he_init(
                    next(keygen), (config.emb_size, config.emb_size)
                ),
            },
            "ln1": {
                "gamma": one_init(next(keygen), (config.emb_size,)),
                "beta": zero_init(next(keygen), (config.emb_size,)),
            },
            "feed_forward": {
                "linear_in": {
                    "W": he_init(
                        next(keygen),
                        (config.emb_size, config.feed_forward_size),
                    ),
                    "b": zero_init(next(keygen), (config.feed_forward_size,)),
                },
                "linear_out": {
                    "W": he_init(
                        next(keygen),
                        (config.feed_forward_size, config.emb_size),
                    ),
                    "b": zero_init(next(keygen), (config.emb_size,)),
                },
            },
            "ln2": {
                "gamma": one_init(next(keygen), (config.emb_size,)),
                "beta": zero_init(next(keygen), (config.emb_size,)),
            },
        }

    return params
