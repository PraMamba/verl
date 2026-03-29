# MOPD Teacher-Order Invariance Results

## Goal

Validate the cheap but important configuration-invariance claim for the current
multi-teacher MOPD runtime:
if the mapping is really `teacher_id -> teacher` and not an implicit YAML slot,
then permuting `algorithm.mopd.teachers[]` should not change the semantics.

This experiment keeps the student, teachers, dataset, rollout budget, and all
trainer settings fixed, and only changes the declaration order of
`algorithm.mopd.teachers[]`.

## Checked-In Experiment Surface

The current branch now contains a dedicated experiment harness plus unit tests
for the experiment contract:

- harness: `recipe/mopd/run_teacher_order_invariance.py`
- unit tests: `tests/unit/test_mopd_teacher_order_invariance.py`
- tracked metrics:
  - `training/global_step`
  - `critic/advantages/mean`
  - `mopd/is_ratio_mean`
  - `mopd/is_valid_fraction`
  - `mopd/is_zeroed_fraction`
  - `mopd/cell_type_teacher/sample_fraction`
  - `mopd/disease_state_teacher/sample_fraction`
  - `mopd/cell_type_teacher/reverse_kl_mean`
  - `mopd/disease_state_teacher/reverse_kl_mean`

## Initial Failed Run: Non-Algorithmic Batch-Shape Issue

The first paired run on 2026-03-17 used the raw smoke parquet inputs
`/tmp/mopd-zero-teacher-smoke/mopd_train.parquet` and
`/tmp/mopd-zero-teacher-smoke/mopd_test.parquet`, wrote outputs under
`/tmp/mopd-teacher-order-invariance-20260317`, and failed before producing
training metrics.

The failure was:

```text
AssertionError: 5 % 4 != 0
```

The stack pointed to teacher sub-batch balancing in
`verl/trainer/ppo/ray_trainer.py`, ultimately through
`get_seqlen_balanced_partitions(..., equal_size=True)`.

Interpretation:

- one teacher-routed sub-batch in the sampled batch had `5` examples
- teacher DP size was `4`
- this violated the runtime balancing constraint for the current smoke setup

This was treated as a batching/data-shape issue, not as evidence against order
invariance.

To isolate the intended claim, the harness was updated to prepare balanced
teacher batches and to disable shuffle for the paired rerun.

## Rerun Setup

### Balanced dataset preparation

The rerun wrote prepared data under:

- `/tmp/mopd-teacher-order-invariance-20260317-rerun/prepared_data/order_invariance_train.parquet`
- `/tmp/mopd-teacher-order-invariance-20260317-rerun/prepared_data/order_invariance_val.parquet`

Prepared train data facts:

- total rows: `32`
- teacher split: `16` `cell_type_teacher` / `16` `disease_state_teacher`
- each 8-sample batch is exactly `4/4`

### Paired rerun command

```bash
python recipe/mopd/run_teacher_order_invariance.py \
  --train-file recipe/mopd/data/mopd_train.parquet \
  --val-file recipe/mopd/data/mopd_test.parquet \
  --output-root /tmp/mopd-teacher-order-invariance-20260317-rerun \
  --train-batch-size 8 \
  --max-prompt-length 4096 \
  --max-response-length 32 \
  --rollout-n 1 \
  --rollout-gpu-memory-utilization 0.25 \
  --max-train-batches 4 \
  --max-val-batches 1 \
  --cuda-visible-devices 0,1,2,3
```

Paired modes:

- `declared_order`
- `reversed_declared_order`

Intended semantic difference:

- only the order of the two teacher specs inside `algorithm.mopd.teachers[]`

## Final Per-Run Summary

Both runs completed to `training/global_step=4`.

| Mode | `critic/advantages/mean` | `mopd/is_ratio_mean` | `mopd/is_valid_fraction` | `mopd/is_zeroed_fraction` | `cell sample_fraction` | `disease sample_fraction` | `cell reverse_kl_mean` | `disease reverse_kl_mean` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `declared_order` | -3.14385843 | 0.99404597 | 1.0 | 0.0 | 0.5 | 0.5 | -3.39302969 | -3.05770826 |
| `reversed_declared_order` | -3.33859634 | 0.99394995 | 1.0 | 0.0 | 0.5 | 0.5 | -3.83501577 | -3.05359530 |

## Step-Level Comparison

### Structural diagnostics

These are the strongest invariance signals in this smoke run:

- both runs reached the same horizon: steps `1,2,3,4`
- `training/global_step` matched exactly at every step
- `mopd/is_valid_fraction` stayed `1.0` in both runs at every step
- `mopd/is_zeroed_fraction` stayed `0.0` in both runs at every step
- both teacher sample fractions stayed exactly `0.5 / 0.5` at every step
- no teacher-name swap symptom appeared in the per-teacher metric keys

### Tracked metric table

| Step | Metric | `declared_order` | `reversed_declared_order` | Delta (`declared - reversed`) |
| --- | --- | ---: | ---: | ---: |
| 1 | `critic/advantages/mean` | -3.63968039 | -3.56733441 | -0.07234597 |
| 1 | `mopd/is_ratio_mean` | 0.99443018 | 0.99554211 | -0.00111192 |
| 1 | `mopd/cell_type_teacher/reverse_kl_mean` | -4.08709049 | -3.83620024 | -0.25089025 |
| 1 | `mopd/disease_state_teacher/reverse_kl_mean` | -3.19882631 | -3.36843824 | +0.16961193 |
| 2 | `critic/advantages/mean` | -3.38563585 | -3.05122638 | -0.33440948 |
| 2 | `mopd/is_ratio_mean` | 1.00307536 | 0.99548459 | +0.00759077 |
| 2 | `mopd/cell_type_teacher/reverse_kl_mean` | -3.26640749 | -3.10064077 | -0.16576672 |
| 2 | `mopd/disease_state_teacher/reverse_kl_mean` | -3.49452758 | -3.01165605 | -0.48287153 |
| 3 | `critic/advantages/mean` | -3.16862702 | -2.82209039 | -0.34653664 |
| 3 | `mopd/is_ratio_mean` | 0.99773651 | 1.00259900 | -0.00486249 |
| 3 | `mopd/cell_type_teacher/reverse_kl_mean` | -3.46093535 | -3.35184646 | -0.10908890 |
| 3 | `mopd/disease_state_teacher/reverse_kl_mean` | -2.90442228 | -2.27437282 | -0.63004947 |
| 4 | `critic/advantages/mean` | -3.14385843 | -3.33859634 | +0.19473791 |
| 4 | `mopd/is_ratio_mean` | 0.99404597 | 0.99394995 | +0.00009602 |
| 4 | `mopd/cell_type_teacher/reverse_kl_mean` | -3.39302969 | -3.83501577 | +0.44198608 |
| 4 | `mopd/disease_state_teacher/reverse_kl_mean` | -3.05770826 | -3.05359530 | -0.00411296 |

### Aggregate deltas over common steps

- mean absolute delta of `critic/advantages/mean`: `0.23700750`
- mean absolute delta of `mopd/is_ratio_mean`: `0.00341530`
- mean absolute delta of `mopd/cell_type_teacher/reverse_kl_mean`: `0.24193299`
- mean absolute delta of `mopd/disease_state_teacher/reverse_kl_mean`: `0.32166147`

## Interpretation

The evidence again splits into two levels.

### 1. Structural order invariance: supported

This is the strongest conclusion from the paired run.

The paired smoke rerun shows:

- identical training horizon
- identical teacher sample fractions
- identical IS mask health
- no evidence that teacher identity got tied to YAML position

That materially reduces the "the implementation still depends on hidden slot
positions" concern.

### 2. Short-horizon trajectory invariance: supportive, but not strong-overlap proof

The step-level traces are not bitwise identical, and some per-step reverse-KL
deltas are visibly non-zero.

Given the experiment boundary, that should be interpreted cautiously:

- this is only a 4-step smoke run
- the two jobs are run sequentially, not on a locked shared trajectory
- rollout sampling remains stochastic

So the correct wording is:

- structure-level invariance is directly supported
- short-horizon trajectory evidence is supportive
- this is not yet a long-horizon "curves are essentially identical" proof

## Bottom Line

For the current branch, the order-permutation experiment is now complete and the
best-supported claim is:

- `teachers[]` declaration order does not show a structural effect on routing or
  metric attribution in the current runtime
- no slot-hardcoding symptom appeared in the paired GPUs `0-3` smoke run
- longer-horizon paired evidence is still optional if a stronger empirical
  invariance claim is needed

## Verification Artifacts

- harness: `recipe/mopd/run_teacher_order_invariance.py`
- harness tests: `tests/unit/test_mopd_teacher_order_invariance.py`
- initial failed logs:
  - `/tmp/mopd-teacher-order-invariance-20260317/declared_order/console.log`
  - `/tmp/mopd-teacher-order-invariance-20260317/reversed_declared_order/console.log`
- balanced rerun logs:
  - `/tmp/mopd-teacher-order-invariance-20260317-rerun/declared_order/console.log`
  - `/tmp/mopd-teacher-order-invariance-20260317-rerun/reversed_declared_order/console.log`
