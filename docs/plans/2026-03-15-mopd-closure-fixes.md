# MOPD Closure Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close confirmed MOPD lifecycle and configuration gaps without changing one-teacher-per-sample runtime semantics.

**Architecture:** Keep the current trainer-side teacher worker graph. Fix resource finalization in `RayPPOTrainer`, then route `algorithm.mopd` through typed validation so unsupported schema-only fields fail fast before training starts.

**Tech Stack:** Python, OmegaConf, dataclass configs, pytest

---

### Task 1: Add failing lifecycle tests

**Files:**
- Modify: `tests/unit/test_mopd_trainer_runtime.py`

**Step 1: Write the failing test**

Add tests for:
- `cleanup_teacher_workers()` clearing both `teacher_wgs` and `base_policy_wg`
- `RayPPOTrainer.fit()` calling `_finalize_fit_resources()` when `_validate()` raises after tracking starts

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_mopd_trainer_runtime.py -k "cleanup_teacher_workers or fit_finalizes" -v`
Expected: FAIL because cleanup is asymmetric and `fit()` lacks `finally`.

**Step 3: Write minimal implementation**

Update `RayPPOTrainer` cleanup/finalization behavior only enough to satisfy the tests.

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_mopd_trainer_runtime.py -k "cleanup_teacher_workers or fit_finalizes" -v`
Expected: PASS

### Task 2: Add failing typed-config tests

**Files:**
- Modify: `tests/trainer/config/test_algo_config_on_cpu.py`
- Modify: `tests/unit/test_teacher_config.py`

**Step 1: Write the failing test**

Add tests for:
- `omega_conf_to_dataclass(cfg.algorithm)` producing a typed `config.mopd`
- unsupported `TeacherConfig.weight != 1.0`
- unsupported per-teacher `base_model_path`

**Step 2: Run test to verify it fails**

Run: `pytest tests/trainer/config/test_algo_config_on_cpu.py tests/unit/test_teacher_config.py -v`
Expected: FAIL because `AlgoConfig` does not expose `mopd` and unsupported fields are accepted.

**Step 3: Write minimal implementation**

Wire `MOPDConfig` into `AlgoConfig` and the validation path, then add the fail-fast checks.

**Step 4: Run test to verify it passes**

Run: `pytest tests/trainer/config/test_algo_config_on_cpu.py tests/unit/test_teacher_config.py -v`
Expected: PASS

### Task 3: Broader verification

**Files:**
- Modify: `verl/trainer/ppo/ray_trainer.py`
- Modify: `verl/trainer/config/algorithm.py`
- Modify: `verl/utils/config.py`
- Modify: `verl/workers/config/teacher.py`

**Step 1: Run targeted MOPD/config suite**

Run: `pytest tests/unit/test_mopd_trainer_runtime.py tests/unit/test_teacher_config.py tests/trainer/config/test_algo_config_on_cpu.py tests/unit/test_teacher_workers.py -v`
Expected: PASS

**Step 2: Run formatting/linting on touched files**

Run: `python -m pytest --version`
Expected: sanity check that pytest env is present before broader verification.

**Step 3: Run repo verification command for touched files if available**

Run: `pre-commit run --files verl/trainer/ppo/ray_trainer.py verl/trainer/config/algorithm.py verl/utils/config.py verl/workers/config/teacher.py tests/unit/test_mopd_trainer_runtime.py tests/unit/test_teacher_config.py tests/trainer/config/test_algo_config_on_cpu.py`
Expected: PASS, or report missing local hooks/tooling.
