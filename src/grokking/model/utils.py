import jax
import jax.numpy as jnp


def accuracy(y_true, y_pred):
    """Per token accuracy

    Expects 2D arrays with shape (batch_size, vocab_size)

    """
    return jnp.mean(jnp.all(y_true == y_pred, axis=1))

def one_hot_encode(y_pred):
    target_indice = jnp.argmax(y_pred, axis=-1)
    num_classes = y_pred.shape[-1]
    y_pred_one_hots = jax.nn.one_hot(target_indice, num_classes=num_classes)

    return y_pred_one_hots

def evaluate(model, params, X, Y, M, target_token_idx):
    # TODO: Make this batched inference (consider creating a batch inference function)
    _, y_pred = model(params=params, x=X, attention_mask=M)
    y_pred_one_hots = one_hot_encode(y_pred)

    y_true = Y[:, target_token_idx, :]
    y_pred = y_pred_one_hots[:, target_token_idx, :]

    acc = accuracy(y_true, y_pred)

    return acc
