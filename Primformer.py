from __future__ import annotations

import argparse
import json
import math
import os
from typing import Optional, Sequence

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


DEFAULT_HEADS_CHANNELS = {"human1": 5544, "monkey": 3702}
DEFAULT_CHECKPOINT_DIR = "/home/users/zyfeng/DataSet/SavedModel/FullAdult/"
DEFAULT_CHECKPOINT_NAME = "MeHSA"
METRIC_NAMES = ("All", "CAGE", "RNA", "ATAC", "ChIPH", "ChIPT")

tf = None
tqdm = None
Enformer = None
evaluate_model1 = None
get_dataset = None


def load_runtime() -> None:
    global Enformer, evaluate_model1, get_dataset, tf, tqdm
    if tf is not None:
        return

    import tensorflow as tensorflow
    from tqdm import tqdm as tqdm_progress

    from src.model import Enformer as enformer_model
    from src.utils import evaluate_model1 as evaluate_human_model
    from src.utils import get_dataset as load_dataset

    tensorflow.get_logger().setLevel("ERROR")
    tf = tensorflow
    tqdm = tqdm_progress
    Enformer = enformer_model
    evaluate_model1 = evaluate_human_model
    get_dataset = load_dataset


def build_model():
    load_runtime()
    return Enformer(
        heads_channels=DEFAULT_HEADS_CHANNELS,
        channels=1536,
        num_heads=8,
        num_transformer_layers=11,
        pooling_type="max",
    )


def build_optimizer(learning_rate):
    load_runtime()
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def build_checkpoint_manager(
    model,
    optimizer,
    checkpoint_dir: str,
    checkpoint_name: str,
    max_to_keep: int,
) -> tf.train.CheckpointManager:
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    return tf.train.CheckpointManager(
        checkpoint,
        checkpoint_dir,
        checkpoint_name=checkpoint_name,
        max_to_keep=max_to_keep,
    )


def restore_latest_checkpoint(
    checkpoint_manager: tf.train.CheckpointManager,
    learning_rate,
    target_learning_rate: float,
    decay_rate: float,
    decay_steps: int,
) -> int:
    latest_checkpoint = checkpoint_manager.latest_checkpoint
    if not latest_checkpoint:
        print("No checkpoint found, starting from Enformer...")
        return 0

    checkpoint_name = os.path.basename(latest_checkpoint)
    current_step = int(checkpoint_name.split("-")[-1])
    learning_rate.assign(target_learning_rate * (decay_rate ** (current_step // decay_steps)))
    checkpoint_manager.checkpoint.restore(latest_checkpoint)
    print(f"Restored checkpoint from step {current_step}")
    return current_step


def shuffle_dataset(
    dataset,
    steps: int,
    steps_per_epoch: int,
    batch_size: int,
    buffer_size: int,
    repeat_count: int,
    prefetch_size: int,
):
    random_seed = steps // steps_per_epoch
    skip_n = steps % steps_per_epoch
    return (
        dataset.shuffle(buffer_size=buffer_size, seed=random_seed)
        .batch(batch_size)
        .repeat(repeat_count)
        .skip(skip_n)
        .prefetch(prefetch_size)
    )


def run_train(args: argparse.Namespace) -> None:
    load_runtime()
    strategy = tf.distribute.MirroredStrategy()
    batch_size = args.batch_size

    with strategy.scope():
        model = build_model()
        learning_rate = tf.Variable(0.0, trainable=False, name="learning_rate")
        learning_rate.assign(args.learning_rate)
        optimizer = build_optimizer(learning_rate)

        @tf.function
        def train_human(batch):
            def step_fn_human(batch):
                with tf.GradientTape() as tape:
                    outputs = model(batch["sequence"], is_training=True)["human1"]
                    per_example_loss = tf.reduce_mean(
                        tf.keras.losses.poisson(batch["target"], outputs)
                    )
                    loss = tf.reduce_sum(per_example_loss) / batch_size
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return loss

            per_replica_losses = strategy.run(step_fn_human, args=(batch,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        @tf.function
        def train_monkey(batch):
            def step_fn_monkey(batch):
                with tf.GradientTape() as tape:
                    outputs = model(batch["sequence"], is_training=True)["monkey"]
                    per_example_loss = tf.reduce_mean(
                        tf.keras.losses.poisson(batch["target"], outputs)
                    )
                    loss = tf.reduce_sum(per_example_loss) / batch_size
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return loss

            per_replica_losses = strategy.run(step_fn_monkey, args=(batch,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        checkpoint_manager = build_checkpoint_manager(
            model,
            optimizer,
            args.checkpoint_dir,
            args.checkpoint_name,
            args.max_to_keep,
        )
        current_step = restore_latest_checkpoint(
            checkpoint_manager,
            learning_rate,
            args.learning_rate,
            args.learning_rate_decay,
            args.learning_rate_decay_steps,
        )

    human_train_dataset = get_dataset(args.human_dataset, "train")
    monkey_train_dataset = get_dataset(args.monkey_dataset, "train")
    human_train_num = sum(1 for _ in human_train_dataset)
    steps_per_epoch = int(human_train_num / batch_size) + 1

    human_run_dataset = iter(
        strategy.experimental_distribute_dataset(
            shuffle_dataset(
                human_train_dataset,
                current_step,
                steps_per_epoch,
                batch_size,
                args.shuffle_buffer_size,
                args.repeat_count,
                args.prefetch_size,
            )
        )
    )
    monkey_run_dataset = iter(
        strategy.experimental_distribute_dataset(
            shuffle_dataset(
                monkey_train_dataset,
                current_step,
                steps_per_epoch,
                batch_size,
                args.shuffle_buffer_size,
                args.repeat_count,
                args.prefetch_size,
            )
        )
    )

    learning_rate_frac = args.learning_rate_decay ** (
        current_step // args.learning_rate_decay_steps
    )
    for global_step in tqdm(
        range(current_step + 1, args.total_steps + 1),
        initial=current_step,
        total=args.total_steps,
        desc="Training",
    ):
        batch_human = next(human_run_dataset)
        batch_monkey = next(monkey_run_dataset)

        if global_step % steps_per_epoch == 0:
            human_run_dataset = iter(
                strategy.experimental_distribute_dataset(
                    shuffle_dataset(
                        human_train_dataset,
                        global_step,
                        steps_per_epoch,
                        batch_size,
                        args.shuffle_buffer_size,
                        args.repeat_count,
                        args.prefetch_size,
                    )
                )
            )
            monkey_run_dataset = iter(
                strategy.experimental_distribute_dataset(
                    shuffle_dataset(
                        monkey_train_dataset,
                        global_step,
                        steps_per_epoch,
                        batch_size,
                        args.shuffle_buffer_size,
                        args.repeat_count,
                        args.prefetch_size,
                    )
                )
            )

        if global_step % args.learning_rate_decay_steps == 0:
            learning_rate_frac *= args.learning_rate_decay
            learning_rate.assign(args.learning_rate * learning_rate_frac)

        train_human(batch=batch_human)
        train_monkey(batch=batch_monkey)

        if global_step % args.save_every == 0:
            checkpoint_manager.save(checkpoint_number=global_step)


def selection_path(args: argparse.Namespace) -> str:
    return os.path.join(args.checkpoint_dir, "best_model", "selection.json")


def checkpoint_files(prefix: str):
    shards = sorted(tf.io.gfile.glob(prefix + ".data-*-of-*"))
    if not tf.io.gfile.exists(prefix + ".index") or not shards:
        raise FileNotFoundError(f"Checkpoint index or data shards missing: {prefix}")
    return [prefix + ".index", *shards]


def score_checkpoint(prefix: str, dataset):
    checkpoint_files(prefix)
    model = build_model()
    # Training checkpoints also contain optimizer state, which inference does not need.
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(prefix).expect_partial()
    metrics = tuple(float(value) for value in evaluate_model1(model, dataset=dataset))
    # Sonnet creates variables lazily, so verify model restoration after prediction.
    status.assert_existing_objects_matched()
    return metrics


def write_metric_header(output_file, split: str) -> None:
    output_file.write("Steps\t" + "\t".join(f"{split}_{name}" for name in METRIC_NAMES) + "\n")


def write_metrics(output_file, step: int, metrics) -> None:
    output_file.write(f"{step}\t" + "\t".join(str(value) for value in metrics) + "\n")
    output_file.flush()


def run_validation(args: argparse.Namespace) -> None:
    load_runtime()
    dataset = get_dataset(args.human_dataset, "valid").batch(1).prefetch(args.prefetch_size)
    best = None
    with open(args.val_output, "w", encoding="utf-8") as output_file:
        write_metric_header(output_file, "Val")
        for step in range(args.val_start, args.val_stop + 1, args.val_interval):
            prefix = os.path.join(args.checkpoint_dir, f"{args.checkpoint_name}-{step}")
            if not tf.io.gfile.exists(prefix + ".index"):
                print(f"Skipping missing checkpoint: {prefix}")
                continue
            metrics = score_checkpoint(prefix, dataset)
            write_metrics(output_file, step, metrics)
            print(f"Validation step {step}: Val_All={metrics[0]}")
            if not math.isfinite(metrics[0]):
                print(f"Skipping selection for step {step}: Val_All is not finite.")
                continue
            # Strict comparison keeps the earliest step when scores are equal.
            if best is None or metrics[0] > best["validation_metrics"]["All"]:
                best = {
                    "step": step,
                    "source_checkpoint": os.path.abspath(prefix),
                    "dataset": args.human_dataset,
                    "selection_metric": "Val_All",
                    "validation_metrics": dict(zip(METRIC_NAMES, metrics)),
                }

    if best is None:
        raise ValueError("No checkpoint with a finite Val_All score was found; no model selected.")

    best_dir = os.path.dirname(selection_path(args))
    os.makedirs(best_dir, exist_ok=True)
    prefix = best["source_checkpoint"]
    saved_name = os.path.basename(prefix)
    # Keep a separate copy so training checkpoint retention cannot delete the final model.
    for source in checkpoint_files(prefix):
        tf.io.gfile.copy(source, os.path.join(best_dir, saved_name) + source[len(prefix):], overwrite=True)
    best["checkpoint"] = saved_name
    # Group metrics may be undefined for constant tracks; write valid JSON nulls.
    best["validation_metrics"] = {
        name: value if math.isfinite(value) else None
        for name, value in best["validation_metrics"].items()
    }
    with open(selection_path(args) + ".tmp", "w", encoding="utf-8") as output_file:
        json.dump(best, output_file, indent=2, allow_nan=False)
        output_file.write("\n")
    os.replace(selection_path(args) + ".tmp", selection_path(args))
    print(f"Selected final model: step {best['step']}, Val_All={best['validation_metrics']['All']}")
    print(f"Selection saved to {selection_path(args)}")


def run_eval(args: argparse.Namespace) -> None:
    path = selection_path(args)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No selected model at {path}. Run --mode validation first.")
    with open(path, encoding="utf-8") as input_file:
        best = json.load(input_file)
    if best["dataset"] != args.human_dataset:
        raise ValueError("--human-dataset must match the dataset used for validation.")
    load_runtime()
    prefix = os.path.join(os.path.dirname(path), best["checkpoint"])
    checkpoint_files(prefix)
    dataset = get_dataset(args.human_dataset, "test").batch(1).prefetch(args.prefetch_size)
    metrics = score_checkpoint(prefix, dataset)
    if not math.isfinite(metrics[0]):
        raise ValueError("Test_All is not finite; final performance cannot be reported.")
    with open(args.eval_output, "w", encoding="utf-8") as output_file:
        write_metric_header(output_file, "Test")
        write_metrics(output_file, best["step"], metrics)
    print(f"Evaluation of selected step {best['step']}: Test_All={metrics[0]}")
    print(f"Test metrics saved to {args.eval_output}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, select on validation, or evaluate Primformer on test data.")
    parser.add_argument("--mode", choices=("train", "validation", "eval"), default="train")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--total-steps", type=int, default=150000)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--learning-rate-decay", type=float, default=0.95)
    parser.add_argument("--learning-rate-decay-steps", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--max-to-keep", type=int, default=500)
    parser.add_argument("--human-dataset", default="Human")
    parser.add_argument("--monkey-dataset", default="MonkeyFull")
    parser.add_argument("--shuffle-buffer-size", type=int, default=1000)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--prefetch-size", type=int, default=2)
    parser.add_argument("--val-start", type=int, default=1000)
    parser.add_argument("--val-stop", type=int, default=150000)
    parser.add_argument("--val-interval", type=int, default=1000)
    parser.add_argument("--val-output", default="validation.tsv")
    parser.add_argument("--eval-output", default="evaluation.tsv")
    args = parser.parse_args(argv)
    if args.mode == "validation":
        if args.val_start < 0 or args.val_stop < args.val_start or args.val_interval <= 0:
            parser.error("Require 0 <= --val-start <= --val-stop and --val-interval > 0.")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.mode == "train":
        run_train(args)
    elif args.mode == "validation":
        run_validation(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
