# MOPD Long-Run Experiment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the current MOPD recipe from a smoke-style dual-teacher runtime launcher into an experiment-grade long-run protocol with real validation, explicit pool topology, clean provenance, and interpretable baselines.

**Architecture:** Keep the current trainer-side MOPD runtime intact and improve the experiment surface around it. The main changes are at the recipe, evaluation, observability, and run-contract layers rather than inside the core MOPD advantage estimator.

**Tech Stack:** Python, Bash, Hydra overrides, Ray, vLLM, verl PPO trainer, pytest

---

### Task 1: Replace Zero Validation With A Real Evaluator

**Files:**
- Create: `recipe/mopd/real_eval_reward.py`
- Modify: `recipe/mopd/run_mopd_qwen3_4b.sh`
- Test: `tests/unit/test_mopd_run_script.py`

**Step 1: Write the failing test**

Add assertions that the experiment-grade launcher no longer points at `zero_reward.py`, and that it configures deterministic or explicitly controlled validation behavior.

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_mopd_run_script.py -q`
Expected: FAIL because the current script still references `zero_reward.py` and sampling-based validation.

**Step 3: Write minimal implementation**

- add a real evaluation reward shim that returns structured domain-specific metrics
- update the launcher to use it instead of `zero_reward.py`
- change validation decoding policy to deterministic or explicitly multi-sample

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_mopd_run_script.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add recipe/mopd/real_eval_reward.py recipe/mopd/run_mopd_qwen3_4b.sh tests/unit/test_mopd_run_script.py
git commit -m "[trainer] feat: use real validation in mopd long-run recipe"
```

### Task 2: Externalize Teacher Topology And Pool Placement

**Files:**
- Create: `recipe/mopd/mopd_longrun_teachers.yaml`
- Modify: `recipe/mopd/run_mopd_qwen3_4b.sh`
- Test: `tests/unit/test_mopd_run_script.py`

**Step 1: Write the failing test**

Add assertions that the launcher no longer hardcodes both teachers into the same inline `global_pool` override and instead resolves explicit pool topology from a config artifact.

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_mopd_run_script.py -q`
Expected: FAIL because the current script still embeds the inline teacher list.

**Step 3: Write minimal implementation**

- add a versioned teacher-topology config or resolved manifest template
- update the launcher to load teachers and pools from that artifact
- keep the default topology valid for the current 4-GPU single-node setup

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_mopd_run_script.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add recipe/mopd/mopd_longrun_teachers.yaml recipe/mopd/run_mopd_qwen3_4b.sh tests/unit/test_mopd_run_script.py
git commit -m "[ray] feat: externalize mopd teacher pool topology"
```

### Task 3: Make Run Provenance And Fresh-Run Semantics Explicit

**Files:**
- Modify: `recipe/mopd/run_mopd_qwen3_4b.sh`
- Possibly Create: `recipe/mopd/log_run_metadata.py`
- Test: `tests/unit/test_mopd_run_script.py`

**Step 1: Write the failing test**

Add assertions that the launcher records resolved metadata and defaults to a fresh checkpoint root instead of ambiguous auto-resume behavior.

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_mopd_run_script.py -q`
Expected: FAIL because the current script still uses shared `CKPTS_DIR` plus `trainer.resume_mode=auto`.

**Step 3: Write minimal implementation**

- log git SHA or dirty marker, dataset path, teacher manifest, and resolved launch metadata
- default to a unique run directory under the checkpoint root
- make resume an explicit environment override rather than the default behavior

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_mopd_run_script.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add recipe/mopd/run_mopd_qwen3_4b.sh recipe/mopd/log_run_metadata.py tests/unit/test_mopd_run_script.py
git commit -m "[cfg] feat: add explicit provenance to mopd long-run runs"
```

### Task 4: Add Validation Slicing And Teacher/Domain Health Gates

**Files:**
- Modify: `verl/trainer/ppo/ray_trainer.py`
- Modify: `recipe/mopd/run_mopd_qwen3_4b.sh`
- Test: `tests/unit/test_mopd_trainer_runtime.py`
- Test: `tests/integration/test_mopd_e2e.py`

**Step 1: Write the failing test**

Add tests that validation output can be sliced by domain or teacher-facing metadata and that MOPD health metrics are surfaced with clear acceptance thresholds.

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_mopd_trainer_runtime.py -q`
Expected: FAIL because validation currently aggregates by `data_source` plus reward only.

**Step 3: Write minimal implementation**

- add validation slicing by explicit domain metadata
- ensure the experiment recipe surfaces acceptance thresholds for IS-health metrics
- avoid changing the core MOPD estimator math unless a test proves it is necessary

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_mopd_trainer_runtime.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add verl/trainer/ppo/ray_trainer.py recipe/mopd/run_mopd_qwen3_4b.sh tests/unit/test_mopd_trainer_runtime.py tests/integration/test_mopd_e2e.py
git commit -m "[trainer] feat: add mopd long-run validation slicing"
```

### Task 5: Add A Paired Baseline Harness For Long-Run Comparison

**Files:**
- Create: `recipe/mopd/run_mopd_longrun_baselines.py`
- Modify: `recipe/mopd/README.md`
- Test: `tests/unit/test_mopd_run_script.py`

**Step 1: Write the failing test**

Add assertions that the repo provides a first-class way to launch the paired baseline matrix needed to interpret long-run MOPD results.

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_mopd_run_script.py -q`
Expected: FAIL because no baseline harness exists for the experiment-grade recipe.

**Step 3: Write minimal implementation**

- add a small harness that generates the paired run commands for:
  - student-only baseline
  - cell-type single-teacher baseline
  - disease-state single-teacher baseline
  - dual-teacher MOPD run
- document the intended comparison contract

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_mopd_run_script.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add recipe/mopd/run_mopd_longrun_baselines.py recipe/mopd/README.md tests/unit/test_mopd_run_script.py
git commit -m "[trainer] feat: add paired long-run baseline harness for mopd"
```

### Task 6: Verify The Experiment-Grade Recipe End-To-End

**Files:**
- Modify if needed: `recipe/mopd/run_mopd_qwen3_4b_preflight.sh`
- Test: `tests/unit/test_mopd_run_script.py`
- Test: `tests/unit/test_mopd_trainer_runtime.py`
- Test: `tests/integration/test_mopd_e2e.py`

**Step 1: Run unit coverage**

Run:

```bash
pytest -q tests/unit/test_mopd_run_script.py tests/unit/test_mopd_trainer_runtime.py
```

Expected: PASS

**Step 2: Run targeted CPU-safe regression**

Run:

```bash
pytest -q \
  tests/unit/test_mopd_advantage.py \
  tests/unit/test_teacher_routing.py \
  tests/unit/test_teacher_config.py \
  tests/unit/test_mopd_preflight.py \
  tests/unit/test_mopd_resource_pools.py \
  tests/unit/test_mopd_run_script.py \
  tests/unit/test_mopd_trainer_runtime.py
```

Expected: PASS

**Step 3: Run preflight on the new launcher**

Run:

```bash
bash recipe/mopd/run_mopd_qwen3_4b_preflight.sh
```

Expected: success boundary reaches first training step

**Step 4: Run opt-in GPU E2E if hardware is available**

Run:

```bash
VERL_MOPD_E2E=1 pytest tests/integration/test_mopd_e2e.py::test_mopd_training_e2e -v
```

Expected: PASS

**Step 5: Document verification outcomes**

Update the relevant plan or test-results note with:

- what was run
- what still requires real long-run cluster execution
- what remains out of scope

**Step 6: Commit**

```bash
git add recipe/mopd/run_mopd_qwen3_4b_preflight.sh tests/unit/test_mopd_run_script.py tests/unit/test_mopd_trainer_runtime.py tests/integration/test_mopd_e2e.py docs/plans
git commit -m "[trainer] test: verify experiment-grade mopd recipe"
```
