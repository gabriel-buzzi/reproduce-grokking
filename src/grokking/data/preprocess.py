import jax
import jax.numpy as jnp

def split_data(X, Y, M, val_size, test_size, keygen):
    num_samples = X.shape[0]
    indices = jnp.arange(num_samples)
    shuffled_indices = jax.random.permutation(
        next(keygen), indices, independent=True
    )

    num_test_samples = int(test_size * len(shuffled_indices))
    train_val_idx = shuffled_indices[:-num_test_samples]
    num_val_samples = int(val_size * len(train_val_idx))

    train_idx = train_val_idx[:-num_val_samples]
    val_idx = train_val_idx[-num_val_samples:]
    test_idx = shuffled_indices[-num_test_samples:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    M_train = M[train_idx]

    X_val = X[val_idx]
    Y_val = Y[val_idx]
    M_val = M[val_idx]

    X_test = X[test_idx]
    Y_test = Y[test_idx]
    M_test = M[test_idx]

    return (
        X_train,
        Y_train,
        M_train,
        X_val,
        Y_val,
        M_val,
        X_test,
        Y_test,
        M_test,
    )