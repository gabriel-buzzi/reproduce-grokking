import jax
import jax.numpy as jnp

def evaluate(model, params, tokenizer, X, Y, M, target_token_idx):
    # TODO: Make this batched inference (consider creating a batch inference function)
    _, y_hat = model(params=params, x=X, attention_mask=M)
    target_indice = jnp.argmax(y_hat, axis=-1)
    num_classes = y_hat.shape[-1]
    y_hat_one_hots = jax.nn.one_hot(target_indice, num_classes=num_classes)

    y_hat = tokenizer.detokenize(y_hat_one_hots.tolist())
    y_true = tokenizer.detokenize(Y.tolist())

    y_trues = []
    y_preds = []
    correct_preds = 0
    all_preds = 0

    for i in range(len(Y)):
        true = int(y_true[i][target_token_idx])
        pred = int(y_hat[i][target_token_idx])

        y_trues.append(true)
        y_preds.append(pred)

        if y_true[i][target_token_idx] == y_hat[i][target_token_idx]:
            correct_preds += 1

        all_preds += 1

    cm = jnp.zeros((num_classes, num_classes), dtype=jnp.int32)
    cm = cm.at[y_trues, y_preds].add(1)

    accuracy = correct_preds / all_preds

    return accuracy, cm
