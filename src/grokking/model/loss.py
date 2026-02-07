import jax.numpy as jnp

def cross_entropy_loss(y_true, y_pred):
    """Per-token loss"""
    return -jnp.sum(y_true * jnp.log(y_pred))
