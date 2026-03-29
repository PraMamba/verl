# MOPD Teacher Order Invariance Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run a paired MOPD experiment that keeps every runtime setting fixed except `algorithm.mopd.teachers[]` declaration order, then document whether results remain invariant within short-horizon smoke noise.

**Architecture:** Reuse the existing reduction-harness pattern instead of touching trainer/runtime logic. Add one dedicated Python harness that emits two matched commands, one with the canonical teacher declaration order and one with the reversed order, then summarize the resulting console metrics for direct comparison.

**Tech Stack:** Python 3.10+, Hydra overrides, verl PPO trainer, pandas, pytest, CUDA GPUs 0-3

---

## Chunk 1: Harness And Tests

### Task 1: Lock the order-invariance harness contract with tests

**Files:**
- Create: `tests/unit/test_mopd_teacher_order_invariance.py`
- Test: `recipe/mopd/run_teacher_order_invariance.py`

- [ ] **Step 1: Write failing tests for command generation**

Cover:
- both commands share the same core training settings
- only the `algorithm.mopd.teachers=[...]` item changes
- teacher names appear in opposite orders across the paired commands

- [ ] **Step 2: Write failing tests for console metric parsing**

Cover:
- `extract_step_metrics()` parses step-level metrics containing:
  - `training/global_step`
  - `critic/advantages/mean`
  - `mopd/is_ratio_mean`
  - `mopd/cell_type_teacher/reverse_kl_mean`
  - `mopd/disease_state_teacher/reverse_kl_mean`

- [ ] **Step 3: Implement the minimal harness**

Create `recipe/mopd/run_teacher_order_invariance.py` with:
- a typed config dataclass
- paired command generation for:
  - `declared_order`
  - `reversed_declared_order`
- a `--dry-run` mode
- shared log parsing / final summary output

- [ ] **Step 4: Run the new unit tests**

Run:

```bash
pytest -q tests/unit/test_mopd_teacher_order_invariance.py
```

Expected:
- new harness tests pass

## Chunk 2: Paired Experiment Execution

### Task 2: Run the matched smoke experiment on GPUs 0-3

**Files:**
- Run: `recipe/mopd/run_teacher_order_invariance.py`
- Output: `/tmp/mopd-teacher-order-invariance-20260317`

- [ ] **Step 1: Dry-run the paired commands**

Run:

```bash
python recipe/mopd/run_teacher_order_invariance.py \
  --dry-run \
  --train-file /tmp/mopd-zero-teacher-smoke/mopd_train.parquet \
  --val-file /tmp/mopd-zero-teacher-smoke/mopd_test.parquet \
  --output-root /tmp/mopd-teacher-order-invariance-20260317 \
  --train-batch-size 8 \
  --max-prompt-length 4096 \
  --max-response-length 32 \
  --rollout-n 1 \
  --rollout-gpu-memory-utilization 0.25 \
  --cuda-visible-devices 0,1,2,3
```

Expected:
- two commands print successfully
- the only intentional config difference is teacher declaration order

- [ ] **Step 2: Launch the paired run**

Run:

```bash
python recipe/mopd/run_teacher_order_invariance.py \
  --train-file /tmp/mopd-zero-teacher-smoke/mopd_train.parquet \
  --val-file /tmp/mopd-zero-teacher-smoke/mopd_test.parquet \
  --output-root /tmp/mopd-teacher-order-invariance-20260317 \
  --train-batch-size 8 \
  --max-prompt-length 4096 \
  --max-response-length 32 \
  --rollout-n 1 \
  --rollout-gpu-memory-utilization 0.25 \
  --cuda-visible-devices 0,1,2,3
```

Expected:
- both runs finish
- both logs emit `training/global_step`

- [ ] **Step 3: Extract and compare tracked metrics**

Compare:
- `training/global_step`
- `critic/advantages/mean`
- `critic/score/mean`
- `mopd/is_ratio_mean`
- `mopd/is_valid_fraction`
- `mopd/is_zeroed_fraction`
- `mopd/cell_type_teacher/sample_fraction`
- `mopd/disease_state_teacher/sample_fraction`
- `mopd/cell_type_teacher/reverse_kl_mean`
- `mopd/disease_state_teacher/reverse_kl_mean`

Expected:
- no obvious slot-dependent swap or systematic drift
- final deltas remain small under smoke-run noise

## Chunk 3: Results And Documentation

### Task 3: Record the experiment and update status docs

**Files:**
- Create: `docs/plans/2026-03-17-mopd-teacher-order-invariance-results.md`
- Modify: `docs/plans/mopd-changes-summary.md`
- Modify: `docs/plans/mopd-n-teacher-challenge-analysis.md`
- Modify: `docs/plans/mopd-n-teacher-extension-challenges.md`
- Modify: `docs/plans/mopd-test-results.md`

- [ ] **Step 1: Write the dedicated results doc**

Include:
- experiment goal
- exact command
- fixed config vs changed variable
- step-level and final metrics
- interpretation limits

- [ ] **Step 2: Update the four requested docs**

Add:
- explicit statement that the config-invariance experiment is now run
- result summary
- residual caveats about smoke horizon / stochastic rollout

- [ ] **Step 3: Re-run the directly relevant verification**

Run:

```bash
pytest -q \
  tests/unit/test_mopd_teacher_order_invariance.py \
  tests/unit/test_teacher_routing.py \
  tests/unit/test_mopd_trainer_runtime.py
```

Expected:
- all selected tests pass

- [ ] **Step 4: Review the final claims against the logs**

Check:
- the docs do not over-claim beyond smoke evidence
- conclusions match the actual final metrics
