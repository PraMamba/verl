# MOPD Preflight Smoke Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the MOPD preflight script default to a smoke-safe preset that avoids the known GPU binding and undersized batch pitfalls.

**Architecture:** Keep the change local to the preflight entrypoints. The shell wrapper should default to the first `NGPUS_PER_NODE` GPUs instead of the reward-server GPUs, and both the shell wrapper and Python helper should reject undersized `train_batch_size * rollout_n` combinations before launch.

**Tech Stack:** Bash, Python, pytest

---

### Task 1: Lock in smoke-safe shell defaults

**Files:**
- Modify: `recipe/mopd/run_mopd_qwen3_4b_preflight.sh`
- Test: `tests/unit/test_mopd_run_script.py`

**Step 1: Write the failing test**

Add assertions that the preflight script:
- derives `CUDA_VISIBLE_DEVICES` from `NGPUS_PER_NODE` when unset
- defaults `PREFLIGHT_TRAIN_BATCH_SIZE` to `2`
- does not hardcode `4,5,6,7`

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/unit/test_mopd_run_script.py -k preflight`

Expected: FAIL on the missing smoke-safe defaults.

**Step 3: Write minimal implementation**

Update the shell script to:
- set `NGPUS_PER_NODE` before GPU binding
- default `CUDA_VISIBLE_DEVICES` to `0..NGPUS_PER_NODE-1` when unset
- default `PREFLIGHT_TRAIN_BATCH_SIZE` to `2`

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/unit/test_mopd_run_script.py -k preflight`

Expected: PASS

### Task 2: Add explicit fail-fast for undersized smoke runs

**Files:**
- Modify: `recipe/mopd/check_mopd_first_batch.py`
- Modify: `recipe/mopd/run_mopd_qwen3_4b_preflight.sh`
- Test: `tests/unit/test_mopd_preflight.py`
- Test: `tests/unit/test_mopd_run_script.py`

**Step 1: Write the failing test**

Add tests that:
- reject `train_batch_size * rollout_n < 8` in the Python helper
- assert the shell script surfaces the same guard

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/unit/test_mopd_preflight.py tests/unit/test_mopd_run_script.py`

Expected: FAIL on the missing validation behavior.

**Step 3: Write minimal implementation**

Add a helper validator in the Python preflight module and call it before building or launching the command. Mirror the same arithmetic guard in the shell script for fast feedback.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/unit/test_mopd_preflight.py tests/unit/test_mopd_run_script.py`

Expected: PASS

### Task 3: Re-verify the smoke contract

**Files:**
- Verify only

**Step 1: Run targeted verification**

Run: `pytest -q tests/unit/test_mopd_preflight.py tests/unit/test_mopd_run_script.py`

Expected: PASS

**Step 2: Review the implementation**

Request a code review pass focused on regressions in preflight defaults and fail-fast behavior.
