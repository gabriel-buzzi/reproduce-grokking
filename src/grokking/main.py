import functools as ft
import itertools as it
import logging

import hydra
from pathlib import Path
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from grokking.data.generate import generate_dataset
from grokking.data.preprocess import split_data
from grokking.data.tokenizer import Tokenizer
from grokking.model.loss import cross_entropy_loss
from grokking.model.parameters import init_param_state
from grokking.model.transformer import decoder_only
from grokking.model.utils import evaluate
import orbax.checkpoint as ocp
from flax.training import orbax_utils

log = logging.getLogger(__name__)
logging.getLogger('orbax').setLevel(logging.WARNING)

DTYPE = jnp.float64


def loss_fn(model, params, X, Y, M):
    _, probs = model(params, X, attention_mask=M)

    seq_loss_fnc = jax.vmap(cross_entropy_loss)
    batch_loss_fnc = jax.vmap(seq_loss_fnc)

    batch_loss = jnp.sum(batch_loss_fnc(Y, probs))

    return batch_loss


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

    log.info("Initilizing model params...")
    vocab_size = tokenizer.vocab_size
    params = init_param_state(config, vocab_size, keygen, DTYPE)
    if manager.latest_step() is not None:
        log.info(
            f"Found checkpoint at step {manager.latest_step()}. Loading..."
        )
        restored = manager.restore(
            manager.latest_step(), args=ocp.args.StandardRestore(params)
        )
        params = restored

    model = ft.partial(decoder_only, config=config, key=next(keygen))

    log.info("Starting training...")
    lr = config.learning_rate
    batch_size = config.batch_size
    train_losses = []
    val_losses = []
    for _ in tqdm(range(config.n_train_steps)):
        # Train
        step_train_loss = 0
        num_train_batches = len(X_train) // batch_size
        for batch in range(num_train_batches):
            batch_slice = slice(batch * batch_size, (batch + 1) * batch_size)
            X_batch = X_train[batch_slice]
            Y_batch = Y_train[batch_slice]
            M_batch = M_train[batch_slice]

            batch_train_loss, grad = jax.value_and_grad(loss_fn, argnums=1)(
                model, params, X_batch, Y_batch, M_batch
            )

            step_train_loss += batch_train_loss

            # TODO: Implement Adam instead of SGD
            params = jax.tree.map(lambda w, d: w - lr * d, params, grad)

        # Validate
        step_val_loss = 0
        num_val_batches = len(X_val) // batch_size
        for batch in range(num_val_batches):
            batch_slice = slice(batch * batch_size, (batch + 1) * batch_size)
            X_batch = X_val[batch_slice]
            Y_batch = Y_val[batch_slice]
            M_batch = M_val[batch_slice]

            batch_val_loss = loss_fn(model, params, X_val, Y_val, M_val)

            step_val_loss += batch_val_loss

        train_losses.append(step_train_loss)
        val_losses.append(step_val_loss)

    # TODO: Need to figure out a way to save the lossess as well for plotting

    log.info("Saving current model params...")
    save_args = orbax_utils.save_args_from_target(params)
    manager.save(
        step=1, args=ocp.args.StandardSave(params, save_args=save_args)
    )

    log.info("Saving losses plot...")
    plt.plot(train_losses, label="train loss", marker="o")
    plt.plot(val_losses, label="val loss", marker="o")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig("history.png", dpi=300)
    plt.close()

    log.info("Evaluating model...")
    train_acc, train_cm = evaluate(
        model, params, tokenizer, X_train, Y_train, M_train, target_token_idx
    )
    test_acc, test_cm = evaluate(
        model, params, tokenizer, X_test, Y_test, M_test, target_token_idx
    )

    log.info("Train performance:")
    log.info(f"Accuracy = {train_acc}")
    log.info(f"Confusion Matrix: \n {train_cm}")

    log.info("Test performance:")
    log.info(f"Accuracy = {test_acc}")
    log.info(f"Confusion Matrix: \n {test_cm}")


if __name__ == "__main__":
    main()
