import functools as ft
import itertools as it
import json
import logging
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from grokking.data.generate import generate_dataset
from grokking.data.preprocess import split_data
from grokking.data.tokenizer import Tokenizer
from grokking.model.loss import cross_entropy_loss
from grokking.model.parameters import init_param_state
from grokking.model.transformer import decoder_only
from grokking.model.utils import accuracy, evaluate, one_hot_encode

log = logging.getLogger(__name__)
logging.getLogger("orbax").setLevel(logging.WARNING)
plt.style.use("config/plotstyle.mplstyle")

DTYPE = jnp.float64


def loss_fn(model, params, X, Y, M, target_token_idx):
    _, probs = model(params, X, attention_mask=M)

    seq_loss_fnc = jax.vmap(cross_entropy_loss)
    batch_loss_fnc = jax.vmap(seq_loss_fnc)

    batch_loss = jnp.sum(batch_loss_fnc(Y, probs))

    y_pred_one_hots = one_hot_encode(probs)
    acc = accuracy(
        Y[:, target_token_idx, :], y_pred_one_hots[:, target_token_idx, :]
    )

    return batch_loss, {"accuracy": acc}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:

    log.info("Starting main application...")
    log.info(OmegaConf.to_yaml(config))

    root_key = jax.random.key(config.param_seed)
    keygen = map(ft.partial(jax.random.fold_in, root_key), it.count())

    log.info("Generating dataset...")
    masked_text_data, text_data, att_mask_data, target_token_idx = (
        generate_dataset(
            mod_operand=config.mod_operand,
            max_operands_value=config.max_operands_value,
            max_seq_length=config.max_seq_length,
        )
    )
    log.info(f"Generated {len(masked_text_data)} samples.")

    log.info("Tokenizing data...")
    tokenizer = Tokenizer(config.max_operands_value)
    embeddings_masked_data = tokenizer.tokenize(masked_text_data)
    embeddings_data = tokenizer.tokenize(text_data)

    X = jnp.array(embeddings_masked_data)
    Y = jnp.array(embeddings_data)
    M = jnp.array(att_mask_data)

    log.info("Splitting data...")
    (
        X_train,
        Y_train,
        M_train,
        X_val,
        Y_val,
        M_val,
        X_test,
        Y_test,
        M_test,
    ) = split_data(X, Y, M, config.val_size, config.test_size, keygen)

    log.info(f"Train size: {X_train.shape}")
    log.info(f"Validation size: {X_val.shape}")
    log.info(f"Test size: {X_test.shape}")

    ckpt_path = Path() / config.checkpoints_path
    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    manager = ocp.CheckpointManager(ckpt_path.absolute(), options=options)
    history_path = Path() / config.checkpoints_path / "history.json"

    log.info("Initilizing model params...")
    vocab_size = tokenizer.vocab_size
    params = init_param_state(config, vocab_size, keygen, DTYPE)
    history = {
        "loss": {"train": [], "validation": []},
        "metrics": {"accuracy": {"train": [], "validation": []}},
    }
    if manager.latest_step() is not None and config.use_checkpoint:
        log.info(
            f"Found checkpoint at step {manager.latest_step()}. Loading..."
        )
        restored_params = manager.restore(
            manager.latest_step(), args=ocp.args.StandardRestore(params)
        )
        params = restored_params

        with open(history_path) as f:
            history = json.load(f)

    model = ft.partial(decoder_only, config=config, key=next(keygen))

    log.info("Starting training...")
    lr = config.learning_rate
    batch_size = config.batch_size
    for _ in tqdm(range(config.n_train_steps)):
        # Train
        step_train_loss = 0
        step_train_accuracy = 0
        num_train_samples = len(X_train)
        num_train_batches = (
            num_train_samples // batch_size + num_train_samples % batch_size
        )
        for batch in range(num_train_batches):
            batch_slice = slice(batch * batch_size, (batch + 1) * batch_size)
            X_batch = X_train[batch_slice]
            Y_batch = Y_train[batch_slice]
            M_batch = M_train[batch_slice]

            grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)

            (batch_train_loss, batch_train_metrics), grad = grad_fn(
                model, params, X_batch, Y_batch, M_batch, target_token_idx
            )

            step_train_loss += batch_train_loss
            step_train_accuracy += (
                batch_train_metrics["accuracy"] / num_train_batches
            )

            # TODO: Implement Adam instead of SGD
            params = jax.tree.map(lambda w, d: w - lr * d, params, grad)

        # Validate
        step_val_loss = 0
        step_val_accuracy = 0
        num_val_samples = len(X_val)
        num_val_batches = (
            num_val_samples // batch_size + num_train_samples % batch_size
        )
        for batch in range(num_val_batches):
            batch_slice = slice(batch * batch_size, (batch + 1) * batch_size)
            X_batch = X_val[batch_slice]
            Y_batch = Y_val[batch_slice]
            M_batch = M_val[batch_slice]

            batch_val_loss, batch_val_metrics = loss_fn(
                model, params, X_val, Y_val, M_val, target_token_idx
            )

            step_val_loss += batch_val_loss
            step_val_accuracy += (
                batch_val_metrics["accuracy"] / num_val_batches
            )

        history["loss"]["train"].append(float(step_train_loss))
        history["loss"]["validation"].append(float(step_val_loss))
        history["metrics"]["accuracy"]["train"].append(
            float(step_train_accuracy)
        )
        history["metrics"]["accuracy"]["validation"].append(
            float(step_val_accuracy)
        )

    if config.save_checkpoint:
        log.info("Saving current model params...")
        save_args = orbax_utils.save_args_from_target(params)
        manager.save(
            step=1, args=ocp.args.StandardSave(params, save_args=save_args)
        )
        with open(history_path, "w") as f:
            json.dump(history, f)

    log.info("Saving losses plot...")
    plt.title("Loss curves")
    plt.plot(history["loss"]["train"], label="Train")
    plt.plot(history["loss"]["validation"], label="Validation")
    plt.legend()
    plt.savefig("loss_history.png")
    plt.close()

    plt.title("Accuracy curves")
    plt.plot(history["metrics"]["accuracy"]["train"], label="Train")
    plt.plot(history["metrics"]["accuracy"]["validation"], label="Validation")
    plt.legend()
    plt.savefig("accuracy_history.png")
    plt.close()

    log.info("Evaluating model...")
    train_acc = evaluate(
        model, params, X_train, Y_train, M_train, target_token_idx
    )
    test_acc = evaluate(
        model, params, X_test, Y_test, M_test, target_token_idx
    )

    log.info("Train performance:")
    log.info(f"Accuracy = {train_acc}")

    log.info("Test performance:")
    log.info(f"Accuracy = {test_acc}")


if __name__ == "__main__":
    main()
