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


def init_param_state(
    config: DictConfig, vocab_size: int, key: jax.Array
) -> dict:
    # Defines the factors for weights initialization
    zero_init = jax.nn.initializers.constant(0.0)
    one_init = jax.nn.initializers.constant(1.0)
    he_init = jax.nn.initializers.he_normal(1, 1)
    # Setting
    dtype = DTYPE

    # Start defining params
    params = {
        "embeddings": he_init(next(key), (vocab_size, config.emb_size), dtype),
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


def cross_entropy_loss(y_true, y_pred):
    """Per-token loss"""
    return -jnp.sum(y_true * jnp.log(y_pred))

def loss_fn(model, params, X, Y, M):
    logits, probs = model(params, X, attention_mask=M)

    seq_loss_fnc = jax.vmap(cross_entropy_loss)
    batch_loss_fnc = jax.vmap(seq_loss_fnc)

    batch_loss = jnp.sum(batch_loss_fnc(Y, probs))

    return batch_loss


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:

    root_key = jax.random.key(config.param_seed)
    key = map(ft.partial(jax.random.fold_in, root_key), it.count())

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

    # TODO: Log data sizes as info

    # TODO: move split code to a function and add a validation split
    num_samples = X.shape[0]
    indices = jnp.arange(num_samples)
    shuffled_indices = jax.random.permutation(
        next(key), indices, independent=True
    )

    test_size = int(config.test_size * len(shuffled_indices))
    
    train_val_idx = shuffled_indices[:-test_size]
    test_idx = shuffled_indices[-test_size:]

    val_size = int(config.val_size * len(shuffled_indices))
    train_idx = train_val_idx[:-val_size]
    val_idx = train_val_idx[-val_size:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    M_train = M[train_idx]

    X_val = X[val_idx]
    Y_val = Y[val_idx]
    M_val = M[val_idx]

    X_test = X[test_idx]
    Y_test = Y[test_idx]
    M_test = M[test_idx]

    print(f"{X_train.shape=}")
    print(f"{X_val.shape=}")
    print(f"{X_test.shape=}")

    vocab_size = tokenizer.vocab_size

    lr = config.learning_rate
    batch_size = config.batch_size

    model = ft.partial(decoder_only, config=config, key=next(key))
    params = init_param_state(config, vocab_size, key)
    train_losses = []
    val_losses = []
    target_token_idx = 6
    
    for step in tqdm(range(config.n_train_steps)):
        step_train_loss = 0

        n_batches = len(X_train) // batch_size
        for batch in range(n_batches):
            batch_slice = slice(batch * batch_size, (batch + 1) * batch_size)
            X_batch = X_train[batch_slice]
            Y_batch = Y_train[batch_slice]
            M_batch = M_train[batch_slice]

            batch_loss, grad = jax.value_and_grad(loss_fn, argnums=1)(
                model, params, X_batch, Y_batch, M_batch
            )

            step_train_loss += batch_loss  # / n_batches

            # TODO: Implement Adam instead of SGD
            params = jax.tree.map(lambda w, d: w - lr * d, params, grad)

        # TODO: make batched inference
        step_val_loss = loss_fn(model, params, X_val, Y_val, M_val)

        train_losses.append(step_train_loss)
        val_losses.append(step_val_loss)

    # TODO: Same model's weights for continue training

    plt.plot(train_losses, label="train loss", marker="o")
    plt.plot(val_losses, label="val loss", marker="o")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig("history.png", dpi=300)
    plt.close()

    train_acc = evaluate(model, params, tokenizer, X_train, Y_train, M_train, target_token_idx)
    test_acc = evaluate(model, params, tokenizer, X_test, Y_test, M_test, target_token_idx)

    print(f"{train_acc=}")
    print(f"{test_acc=}")


def evaluate(model, params, tokenizer, X, Y, M, target_token_idx):
    _, y_hat = model(params=params, x=X, attention_mask=M)
    target_indice = jnp.argmax(y_hat, axis=-1)
    y_hat_one_hots = jax.nn.one_hot(target_indice, num_classes=y_hat.shape[-1])

    y_hat = tokenizer.detokenize(y_hat_one_hots.tolist())
    y_true = tokenizer.detokenize(Y.tolist())

    y_trues = []
    y_preds = []
    correct_preds = 0
    all_preds = 0

    for i in range(len(Y)):
        # print(f"Case {i}")
        # print(y_true[i][target_token_idx])
        # print(y_hat[i][target_token_idx])

        y_trues.append(y_true[i][target_token_idx])
        y_preds.append(y_hat[i][target_token_idx])

        if y_true[i][target_token_idx] == y_hat[i][target_token_idx]:
            correct_preds += 1

        all_preds += 1

    accuracy = correct_preds / all_preds

    return accuracy

if __name__ == "__main__":
    main()
