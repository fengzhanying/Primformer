from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional, Sequence

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


DEFAULT_HEADS_CHANNELS = {"human1": 5544, "monkey": 3702}
DEFAULT_CHECKPOINT_DIR = "/home/users/zyfeng/DataSet/SavedModel/FullAdult/"
DEFAULT_CHECKPOINT_NAME = "MeHSA"
DEFAULT_EVAL_OUTPUT = "Pimfomer_loss.err"
EVAL_HEADER = (
    "Steps\tVal_All\tVal_CAGE\tVal_RNA\tVal_ATAC\tVal_ChIPH\tVal_ChIPT\t"
    "Test_CAGE\tTest_RNA\tTest_ATAC\tTest_ChIPH\tTest_ChIPT\n"
)

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


def restore_checkpoint_at_step(
    checkpoint_manager: tf.train.CheckpointManager,
    checkpoint_dir: str,
    checkpoint_name: str,
    step: int,
) -> None:
    latest_checkpoint = os.path.join(checkpoint_dir, f"{checkpoint_name}-{step}")
    current_step = int(os.path.basename(latest_checkpoint).split("-")[-1])
    print(f"Restored checkpoint from step {current_step}")
    checkpoint_manager.checkpoint.restore(latest_checkpoint).expect_partial()


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


def eval_steps(args: argparse.Namespace) -> Iterable[int]:
    return range(args.eval_start, args.eval_stop + 1, args.eval_interval)


def run_eval(args: argparse.Namespace) -> None:
    load_runtime()
    human_val_dataset = get_dataset(args.human_dataset, "valid").batch(1).prefetch(args.prefetch_size)
    human_test_dataset = get_dataset(args.human_dataset, "test").batch(1).prefetch(args.prefetch_size)

    with open(args.eval_output, "a", encoding="utf-8") as output_file:
        output_file.write(EVAL_HEADER)

    for step in eval_steps(args):
        model = build_model()
        learning_rate = tf.Variable(0.0, trainable=False, name="learning_rate")
        learning_rate.assign(args.learning_rate)
        optimizer = build_optimizer(learning_rate)
        checkpoint_manager = build_checkpoint_manager(
            model,
            optimizer,
            args.checkpoint_dir,
            args.checkpoint_name,
            args.max_to_keep,
        )
        restore_checkpoint_at_step(
            checkpoint_manager,
            args.checkpoint_dir,
            args.checkpoint_name,
            step,
        )

        pcc_val = evaluate_model1(
            model,
            dataset=human_val_dataset,
            max_steps=args.max_eval_steps,
        )
        pcc_test = evaluate_model1(
            model,
            dataset=human_test_dataset,
            max_steps=args.max_eval_steps,
        )

        with open(args.eval_output, "a", encoding="utf-8") as output_file:
            output_file.write(f"{step}\t")
            output_file.write("\t".join(str(value) for value in pcc_val))
            output_file.write("\t")
            output_file.write("\t".join(str(value) for value in pcc_test[1:]))
            output_file.write("\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate Primformer.")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--total-steps", type=int, default=60000)
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
    parser.add_argument("--eval-start", type=int, default=1000)
    parser.add_argument("--eval-stop", type=int, default=60000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-output", default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.mode == "train":
        run_train(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
