# MOPD Zero-Teacher ORM-Only Reduction Results

## Goal

Validate the complementary MOPD reduction claim:
when teacher signal is fully disabled, the implementation should collapse to the
existing ORM-only / GRPO-style baseline instead of leaving any ghost teacher
effect in the training path.

In this worktree, "teacher fully disabled" is represented by a dedicated
reduction estimator rather than `lambda=0`, because the current standard MOPD
path does not treat `lambda_val=0` as a complete teacher-off switch in every
composition path.

## Checked-In Reduction Surface

The current branch already contains the main reduction machinery:

- independent reduction estimator: `mopd_zero_teacher_orm_only`
- paired runtime harness: `recipe/mopd/run_zero_teacher_orm_reduction.py`
- deterministic non-zero ORM reward harness:
  `recipe/mopd/response_length_reward.py`
- same-batch equivalence tests in `tests/unit/test_mopd_advantage.py`
- harness command-generation tests in `tests/unit/test_mopd_zero_teacher_reduction.py`
- runtime preflight / dependency tests in `tests/unit/test_mopd_trainer_runtime.py`

## Algorithm-Level Status

Algorithm-level reduction is supported.

The current unit tests prove that, on the same batch:

- `mopd_zero_teacher_orm_only` returns the same `advantages` as `grpo`
- `mopd_zero_teacher_orm_only` returns the same `returns` as `grpo`
- the same-batch match is preserved for both default GRPO normalization and
  `norm_adv_by_std_in_grpo=False`
- dispatch does not require teacher tensors for the zero-teacher path
- `need_reference_policy()` stays `False` when the reduction is used with
  `algorithm.mopd.enabled=False`
- trainer preflight does not require MOPD teacher runtime when the reduction is
  used with `algorithm.mopd.enabled=False`

Fresh verification on the current worktree:

```bash
pytest -q \
  tests/unit/test_mopd_advantage.py \
  tests/unit/test_mopd_trainer_runtime.py \
  tests/unit/test_mopd_single_teacher_reduction.py \
  tests/unit/test_mopd_zero_teacher_reduction.py
```

Result:

```text
67 passed, 1 warning in 7.27s
```

## Runtime Experiment Setup

### Initial non-algorithmic failure

The first paired smoke run used `train_batch_size=4` and failed for both modes
with:

```text
AssertionError: only support equal chunk. Got size of DataProto 4 and chunk 8.
```

This was a runtime batching/config issue, not an algorithm mismatch, so the
reduction rerun used `train_batch_size=8`.

### Fresh paired smoke rerun on 2026-03-17

- GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3`
- output root:
  `/tmp/mopd-zero-teacher-smoke-run-b8-rerun-20260317`
- train file: `/tmp/mopd-zero-teacher-smoke/mopd_train.parquet`
- val file: `/tmp/mopd-zero-teacher-smoke/mopd_test.parquet`
- train batch size: `8`
- max prompt length: `4096`
- max response length: `32`
- rollout `n=1`
- rollout GPU memory utilization: `0.25`
- reward function: `recipe/mopd/response_length_reward.py`

Command:

```bash
python recipe/mopd/run_zero_teacher_orm_reduction.py \
  --train-file /tmp/mopd-zero-teacher-smoke/mopd_train.parquet \
  --val-file /tmp/mopd-zero-teacher-smoke/mopd_test.parquet \
  --output-root /tmp/mopd-zero-teacher-smoke-run-b8-rerun-20260317 \
  --train-batch-size 8 \
  --max-prompt-length 4096 \
  --max-response-length 32 \
  --rollout-n 1 \
  --rollout-gpu-memory-utilization 0.25 \
  --cuda-visible-devices 0,1,2,3
```

Both modes completed to `training/global_step=4`.

## Fresh Step-Level Results

### `critic/score/mean`

| Step | `mopd_zero_teacher_orm_only` | `grpo` | Delta (`mopd - grpo`) |
| --- | ---: | ---: | ---: |
| 1 | 0.75781250 | 0.75390625 | +0.00390625 |
| 2 | 0.86718750 | 0.79687500 | +0.07031250 |
| 3 | 0.80859375 | 0.80859375 | +0.00000000 |
| 4 | 0.75781250 | 0.79296875 | -0.03515625 |

### `critic/advantages/mean`

| Step | `mopd_zero_teacher_orm_only` | `grpo` | Delta (`mopd - grpo`) |
| --- | ---: | ---: | ---: |
| 1 | 0.75781167 | 0.75390553 | +0.00390613 |
| 2 | 0.86718661 | 0.79687411 | +0.07031250 |
| 3 | 0.80859286 | 0.80859286 | +0.00000000 |
| 4 | 0.75781178 | 0.79296792 | -0.03515613 |

### Aggregate

- common steps: `4`
- mean absolute score delta: `0.02734375`
- max absolute score delta: `0.07031250`
- final score delta: `-0.03515625`
- mean absolute advantage delta: `0.02734369`
- max absolute advantage delta: `0.07031250`
- final advantage delta: `-0.03515613`
- score trace correlation over 4 steps: `0.540338`

The low trace correlation should not be over-interpreted here because it is
computed from only four points.

## Interpretation

The evidence again splits into two levels.

### 1. Algorithm-level reduction: supported

This is the strongest part of the proof surface.

On the same tensors, the zero-teacher reduction matches the ORM-only baseline
exactly, and the runtime dependency checks show that the teacher/reference path
is not required for the intended reduction config:
`adv_estimator=mopd_zero_teacher_orm_only` with `algorithm.mopd.enabled=False`.

### 2. Runtime smoke evidence: supportive, but still short-horizon

The fresh paired smoke rerun on GPUs `0-3` is materially tighter than the older
smoke artifact:

- same training horizon: both sides reached `training/global_step=4`
- one step matched exactly (`step=3`)
- final delta stayed small (`-0.03515625`)
- mean absolute delta stayed small (`0.02734375`)

This is consistent with the reduction claim and does not show an obvious
systematic extra teacher term surviving in the reduced path.

However, this is still only a 4-step, 32-row smoke run. It is good enough to
support "no obvious ghost teacher effect in the current implementation", but it
is not yet strong enough to claim a paper-grade, long-horizon
"the training curves are essentially identical" result.

One more caveat matters here:
this harness runs the two jobs sequentially and uses stochastic rollout
sampling (`temperature=0.7`, `top_p=1.0`), so the step-level deltas are sanity
evidence, not a locked-step same-batch isolation of estimator effects.

## Teardown Note

Both logs show Ray worker `SIGTERM` / `SystemExit` noise after the final metric
lines and after the harness summary lines were already emitted.

Inference from the logs:
this appears to be post-run teardown noise rather than a training failure,
because both runs had already reached `training/global_step=4` and produced
their final summaries before those messages appeared.

## Bottom Line

For the current branch, the zero-teacher reduction goal is best described as:

- implementation complete
- algorithm-level proof complete for the intended reduction config
  (`adv_estimator=mopd_zero_teacher_orm_only` with `algorithm.mopd.enabled=False`)
- runtime smoke evidence supportive
- stronger long-horizon empirical reduction evidence still optional / open

So the correct claim is not "fully paper-grade closed", but also no longer
"unproven" or "possibly still leaking teacher signal".

## Verification Artifacts

- estimator implementation: `verl/trainer/ppo/core_algos.py`
- harness: `recipe/mopd/run_zero_teacher_orm_reduction.py`
- reward function: `recipe/mopd/response_length_reward.py`
- same-batch reduction tests: `tests/unit/test_mopd_advantage.py`
- harness tests: `tests/unit/test_mopd_zero_teacher_reduction.py`
- runtime/preflight tests: `tests/unit/test_mopd_trainer_runtime.py`
- fresh smoke logs:
  `/tmp/mopd-zero-teacher-smoke-run-b8-rerun-20260317/mopd_zero_teacher_orm_only/console.log`
  `/tmp/mopd-zero-teacher-smoke-run-b8-rerun-20260317/grpo/console.log`
