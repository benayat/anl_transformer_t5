# Seq2Seq Negotiator Standalone

This is the current standalone training/evaluation package extracted from the attached project dump, with a first implementation pass for **serialization v2**.

## What was added in this pass

- Parallel **v2 dataset views** emitted by `seq2seq_negotiator/scripts/build_dataset.py`:
  - `rich_v2/`
  - `single_target_teacher_v2/`
  - `single_target_main_v2/`
  - `multitask_teacher_v2/`
  - `multitask_main_v2/`
- Compact v2 source serialization using:
  - anonymous issues `i1, i2, ...`
  - anonymous values `v1, v2, ...`
  - compact state blocks (`@S`, `@V`, `@C`, `@H`)
  - compact targets:
    - `A`
    - `O v3,v1,v2,...`
- Training can now select the dataset serialization version:
  - `--serialization-version v1`
  - `--serialization-version v2`
- Evaluation and prediction parsing now support **both** v1 and v2 formats.
- Tokenization for both recipes now uses `tokenizer.truncation_side = "left"` so the most recent context is preserved under truncation.

## Example build

```bash
uv run python seq2seq_negotiator/scripts/build_dataset.py \
  --input ./data/raw/anl_seqneg/seqneg_traces.jsonl \
  --output-dir ./data/processed/seqneg_v2 \
  --max-history-turns 64 \
  --train-frac 0.90 \
  --valid-frac 0.05 \
  --test-frac 0.05 \
  --seed 13
```

## Example single-target v2 train

```bash
uv run python seq2seq_negotiator/scripts/train.py \
  --recipe single_target \
  --stage warmstart \
  --dataset-dir ./data/processed/seqneg_v2 \
  --serialization-version v2 \
  --init-model t5-small \
  --output-dir ./models/warmstart_single_v2
```

## Example multitask v2 train

```bash
uv run python seq2seq_negotiator/scripts/train.py \
  --recipe multitask \
  --stage main \
  --dataset-dir ./data/processed/seqneg_v2 \
  --serialization-version v2 \
  --init-model t5-small \
  --output-dir ./models/main_mt_v2 \
  --action-loss-weight 1.0 \
  --offer-loss-weight 3.0
```

## Notes

This is an **implementation pass**, not a finished experiment report.
The next recommended step is:
1. build the v2 dataset
2. retrain the strongest single-target baseline on v2
3. compare token-length statistics and validation metrics against v1
