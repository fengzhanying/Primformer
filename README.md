# Primformer

Primformer is a Monkey ENCODE-enhanced primate foundation model for sequence
activity prediction. The model implementation is consolidated in `src/model.py`,
and training/evaluation are controlled through `Primformer.py`.

## Model Architecture

![Primformer model architecture](Model.png)

## Repository Layout

- `src/model.py`: Primformer model definition, including the Enformer-style trunk
  and relative-position attention modules.
- `Primformer.py`: unified command-line entry point for training and
  evaluation.
- `src/utils.py`: TFRecord loading, metadata parsing, and evaluation metrics.
- `src/HumanData.txt` and `src/Human_wigs_class.txt`: metadata used by the human
  evaluation routine.
- `Model.png`: model architecture diagram.

## Requirements

Install the Python dependencies used by the training and evaluation scripts:

```bash
pip install tensorflow dm-sonnet numpy pandas tqdm
```

Use the TensorFlow build appropriate for your GPU/CPU environment.

## Data Layout

By default, `src/utils.py` reads datasets from:

```text
/home/users/zyfeng/DataSet/Input/<organism>/
```

Each organism directory is expected to contain:

```text
statistics.json
tfrecords/train-*.tfr
tfrecords/valid-*.tfr
tfrecords/test-*.tfr
```

The default training configuration uses:

- Human training dataset: `Human`
- Monkey training dataset: `Monkey`
- Human validation/test dataset for evaluation: `Human`

Edit `organism_path()` in `src/utils.py` if your TFRecord datasets are stored in a
different root directory.

## Training

Run training with the unified entry point:

```bash
python Primformer.py --mode train \
  --checkpoint-dir /home/users/zyfeng/DataSet/SavedModel/FullAdult/ \
  --checkpoint-name MeHSA \
  --total-steps 150000 \
  --batch-size 32 \
  --learning-rate 0.0001
```

The training script uses `tf.distribute.MirroredStrategy()` and trains the human
and monkey heads in each step. It restores the latest checkpoint from
`--checkpoint-dir` when available, otherwise it starts a new model.

Useful training options:

```bash
python Primformer.py --mode train --help
```

Common options include:

- `--total-steps`: total optimization steps, default `150000`.
- `--batch-size`: global batch size, default `32`.
- `--save-every`: checkpoint interval, default `1000`.
- `--learning-rate-decay-steps`: decay interval, default `10000`.
- `--learning-rate-decay`: decay factor, default `0.95`.
- `--checkpoint-dir`: checkpoint output/restore directory.

## Evaluation

Evaluate saved checkpoints with:

```bash
python Primformer.py --mode eval \
  --checkpoint-dir /home/users/zyfeng/DataSet/SavedModel/FullAdult/ \
  --checkpoint-name MeHSA \
  --eval-start 1000 \
  --eval-stop 150000 \
  --eval-interval 1000 \
  --eval-output Pimfomer_loss.err
```
Useful evaluation options:

```bash
python Primformer.py --mode eval --help
```
