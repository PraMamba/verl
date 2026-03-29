# MOPD Single-Teacher Reduction Results

## Goal

Validate the reduction claim for the current verl MOPD implementation:
when MOPD is reduced to a single teacher, uses the compatible-tokenizer
token-level path, disables ORM, disables rollout IS correction, disables base
normalization, and avoids sequence-reward mixing, it should behave like an
independent single-teacher on-policy reverse-KL baseline.

This document records the requested 3-way matched comparison over
`data.seed`.

## Official Run Setup

- Harness: `recipe/mopd/run_single_teacher_reduction.py`
- Date: `2026-03-16`
- Data seeds: `42`, `43`, `44`
- GPUs: `CUDA_VISIBLE_DEVICES=4,5,6,7`
- Official output root:
  `/gpfs/Mamba/Project/Single_Cell/Training/MOPD-single-teacher-reduction-20260316-run1`
- Per-seed output pattern:
  `/gpfs/Mamba/Project/Single_Cell/Training/MOPD-single-teacher-reduction-20260316-run1/seed{seed}/{mopd|single_teacher_reverse_kl}`

Command template:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python recipe/mopd/run_single_teacher_reduction.py \
  --seed <42|43|44> \
  --output-root /gpfs/Mamba/Project/Single_Cell/Training/MOPD-single-teacher-reduction-20260316-run1/seed<seed>
```

## Frozen Harness Contract

This round intentionally kept the current harness fixed after the two runtime
fixes below. No experiment-time config drift was allowed across seeds.

- exactly one teacher: `single_teacher`
- `tokenizer_policy=compatible`
- `algorithm.mopd.orm_weight=0.0`
- `algorithm.mopd.is_correction=False`
- `algorithm.mopd.use_base_normalization=False`
- zero-reward harness via `recipe/mopd/zero_reward.py`
- `trainer.save_freq=-1`
- `trainer.test_freq=-1`

The paired comparison was:

1. reduced `mopd`
2. `single_teacher_reverse_kl`

Both modes used the same student, same teacher, same prepared dataset, same
optimizer settings, and the same rollout budget. The only intended semantic
difference was `algorithm.adv_estimator`.

## Pre-Run Blocking Issues And Fixes

### 1. Tokenizer mismatch in the original harness default

The first harness default pointed the student to a plain HuggingFace cache copy
of Qwen3. That tokenizer did not match the production teacher tokenizer
metadata.

The correct fix was to switch the harness default student model to the
explicit-token student used in production:

- `/data/Mamba/Project/Single_Cell/Model/Qwen3-4B-Thinking-2507_Manual_Resize_Block-8_Size-32_Num-256_ExplicitTokens`

Without that fix, the reduction preflight failed in the compatible-tokenizer
path and the experiment would not have been testing the intended algorithm.

### 2. Non-algorithmic checkpoint/validation interference

An earlier real run failed during checkpoint writing on a temporary local disk.
For this official reduction pass, the harness was intentionally kept on:

- `trainer.save_freq=-1`
- `trainer.test_freq=-1`

That removes checkpoint and periodic validation I/O from the experiment. The
conclusion in this document therefore rests on matched training metrics, not on
validation score.

## Tracked Metrics

The reduction harness summarized the following stable training metrics from each
`console.log`:

- `training/global_step`
- `critic/advantages/mean`
- `mopd/single_teacher/reverse_kl_mean`
- `critic/score/mean`

Because this run used `recipe/mopd/zero_reward.py`, `critic/score/mean`
stayed at `0.0`, and the useful signal is the reverse-KL / advantage trace.

## Final Metrics

### Per-run summary

| Seed | Mode | Global Step | `critic/advantages/mean` | `reverse_kl_mean` | `critic/score/mean` |
| --- | --- | ---: | ---: | ---: | ---: |
| 42 | `mopd` | 12.0 | -2.878078 | -2.878078 | 0.0 |
| 42 | `single_teacher_reverse_kl` | 12.0 | -2.329815 | -2.329815 | 0.0 |
| 43 | `mopd` | 12.0 | -2.541241 | -2.541241 | 0.0 |
| 43 | `single_teacher_reverse_kl` | 12.0 | -2.748077 | -2.748077 | 0.0 |
| 44 | `mopd` | 12.0 | -2.507900 | -2.507900 | 0.0 |
| 44 | `single_teacher_reverse_kl` | 12.0 | -2.607313 | -2.607313 | 0.0 |

### Per-seed final deltas

Delta is defined as `mopd - single_teacher_reverse_kl`.

| Seed | `delta_advantages_mean` | `delta_reverse_kl_mean` |
| --- | ---: | ---: |
| 42 | -0.548263 | -0.548263 |
| 43 | +0.206836 | +0.206836 |
| 44 | +0.099413 | +0.099413 |

### Aggregate over 3 seeds

- MOPD final mean: `-2.642407`
- MOPD final std: `0.204777`
- Baseline final mean: `-2.561735`
- Baseline final std: `0.212823`
- Mean paired delta: `-0.080671`
- Paired delta std: `0.408493`
- Mean absolute paired delta: `0.284837`
- Max absolute paired delta: `0.548263`
- Mean relative absolute paired delta vs baseline magnitude: `11.62%`

Within each run, `critic/advantages/mean` and
`mopd/single_teacher/reverse_kl_mean` matched exactly, which is expected under
the zero-reward harness.

## Step-Level Alignment

For each seed, the full 12-step `critic/advantages/mean` trace was compared
between reduced MOPD and the reverse-KL baseline.

| Seed | Common Steps | Trace Corr. | Mean Abs Step Delta | Max Abs Step Delta | Final Step Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 12 | 0.660046 | 0.224385 | 0.643119 | -0.548263 |
| 43 | 12 | 0.668601 | 0.168866 | 0.539691 | +0.206836 |
| 44 | 12 | 0.702572 | 0.193633 | 0.637274 | +0.099413 |

Interpretation:

- all six runs reached the same training horizon: `global_step=12`
- there is no consistent one-sided final offset across seeds
- the traces are positively correlated but not tightly overlapping
- `seed42` ends with a visibly larger final gap than the other two seeds

## Reduction Judgment

The evidence splits into two levels.

### 1. Algorithm-level reduction on the same batch: supported

The checked-in estimator tests in `tests/unit/test_mopd_advantage.py` prove
that reduced single-teacher MOPD matches the independent
`single_teacher_reverse_kl` estimator on the same input tensors. That is the
strongest direct algorithmic proof in this branch.

### 2. Training-trajectory reduction under the current harness: mixed

This 3-seed training-metric experiment does **not** give a strong "the curves
basically overlap" story yet.

What the experiment does support:

- both methods train to the same horizon under the same runtime contract
- there is no stable sign-consistent offset indicating an obvious extra MOPD
  term leaking in only on one side
- the average paired final offset is smaller than the across-seed final
  standard deviation of either method

What it does **not** support strongly enough:

- near-overlap of the training curves
- near-overlap of the final metric in every seed

The most important counterexample is `seed42`, where the final paired gap is
`-0.548263`, materially larger than the other two seeds.

### Bottom line

The tensor-level reduction proof is in place, but this 3-seed matched training
comparison is best described as **non-systematic but not yet tightly aligned**.

That means the current results are consistent with "no obvious systematic extra
signal" in reduced MOPD, but they are not yet strong enough to claim a clean
training-curve-level empirical reduction proof.

## Caveats

- This round intentionally disabled periodic save/test to remove non-algorithmic
  checkpoint and validation interference. The conclusion is based on matched
  training metrics, not validation score.
- `critic/score/mean` is flat by construction because `recipe/mopd/zero_reward.py`
  contributes no downstream task reward.
- This is a 3-way `data.seed` sweep under the frozen harness, not a proof that
  every stochastic source in the runtime was independently swept.
- These are separate on-policy runs rather than a shared-trajectory replay
  experiment, so rollout-level stochasticity can amplify trajectory divergence
  even when the per-batch estimator formulas are equal.

## Verification Artifacts

- Harness implementation:
  `recipe/mopd/run_single_teacher_reduction.py`
- Seed 42 logs:
  `/gpfs/Mamba/Project/Single_Cell/Training/MOPD-single-teacher-reduction-20260316-run1/seed42/{mopd,single_teacher_reverse_kl}/console.log`
- Seed 43 logs:
  `/gpfs/Mamba/Project/Single_Cell/Training/MOPD-single-teacher-reduction-20260316-run1/seed43/{mopd,single_teacher_reverse_kl}/console.log`
- Seed 44 logs:
  `/gpfs/Mamba/Project/Single_Cell/Training/MOPD-single-teacher-reduction-20260316-run1/seed44/{mopd,single_teacher_reverse_kl}/console.log`
- Harness tests:
  `tests/unit/test_mopd_single_teacher_reduction.py`
- Tensor-level reduction tests:
  `tests/unit/test_mopd_advantage.py`
