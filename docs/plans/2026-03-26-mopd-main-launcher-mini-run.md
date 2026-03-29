# MOPD Main Launcher Mini-Run Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a repeatable mini-run wrapper around the main MOPD launcher and verify it against the live reward server.

**Architecture:** Keep the production launcher as the single source of truth for Hydra overrides, and add only the minimum extra env-controlled knobs needed for a bounded run. A thin wrapper script will export smoke-safe defaults and invoke `run_mopd_qwen3_4b.sh` unchanged for the core training graph.

**Tech Stack:** Bash, Hydra overrides, pytest

---

### Task 1: Expose bounded-run knobs in the main launcher

**Files:**
- Modify: `recipe/mopd/run_mopd_qwen3_4b.sh`
- Test: `tests/unit/test_mopd_run_script.py`

**Step 1: Write the failing test**

Add assertions that the main launcher exposes env overrides for:
- `TRAIN_MAX_SAMPLES`
- `VAL_MAX_SAMPLES`
- `VAL_BEFORE_TRAIN`
- `LOG_VAL_GENERATIONS`

and threads them into the Hydra command.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/unit/test_mopd_run_script.py -k "mini or max_samples or VAL_BEFORE_TRAIN or LOG_VAL_GENERATIONS"`

Expected: FAIL on missing launcher knobs.

**Step 3: Write minimal implementation**

Add the env defaults and wire them into the `python -m verl.trainer.main_ppo` invocation.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/unit/test_mopd_run_script.py -k "mini or max_samples or VAL_BEFORE_TRAIN or LOG_VAL_GENERATIONS"`

Expected: PASS

### Task 2: Add a thin mini-run wrapper

**Files:**
- Create: `recipe/mopd/run_mopd_qwen3_4b_mini.sh`
- Modify: `recipe/mopd/README.md`
- Test: `tests/unit/test_mopd_run_script.py`

**Step 1: Write the failing test**

Add assertions that the new wrapper:
- defaults to `CUDA_VISIBLE_DEVICES=0,1,2,3` when unset
- sets bounded run knobs (`TRAIN_PROMPT_BSZ=2`, `TRAIN_MAX_SAMPLES=4`, `TOTAL_EPOCHS=1`, etc.)
- defaults `REWARD_API_BASE` to the live mini-run endpoint
- delegates to `run_mopd_qwen3_4b.sh`

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/unit/test_mopd_run_script.py -k mini`

Expected: FAIL because the wrapper does not exist yet.

**Step 3: Write minimal implementation**

Create the wrapper and update the README mini-run usage snippet.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/unit/test_mopd_run_script.py -k mini`

Expected: PASS

### Task 3: Real runtime verification

**Files:**
- Verify only

**Step 1: Launch the mini-run**

Run the wrapper against the live reward server on `http://127.0.0.1:30005/v1`.

**Step 2: Verify runtime evidence**

Confirm:
- the run binds to GPUs `0,1,2,3`
- the log reaches at least `training/global_step:1`
- the process exits cleanly at the configured bounded epoch

---

## Execution Results

### Completed artifacts

- `recipe/mopd/run_mopd_qwen3_4b_mini.sh`
- `recipe/mopd/run_mopd_qwen3_4b_mini_val.sh`
- bounded-run env knobs in `recipe/mopd/run_mopd_qwen3_4b.sh`
- README launcher table and usage notes in `recipe/mopd/README.md`
- run-script coverage in `tests/unit/test_mopd_run_script.py`

### Local verification

- `pytest -q tests/unit/test_mopd_run_script.py`
  - result: `13 passed`
- `bash -n recipe/mopd/run_mopd_qwen3_4b.sh`
- `bash -n recipe/mopd/run_mopd_qwen3_4b_mini.sh`
- `bash -n recipe/mopd/run_mopd_qwen3_4b_mini_val.sh`

### Runtime milestones

1. `mini` bounded launcher run completed on the live reward path
   - bounded `4/4` run succeeded
   - expanded `8/8` run also succeeded

2. `mini_val` validation-first rehearsal completed
   - bounded `16/16` run with `VAL_BEFORE_TRAIN=true` succeeded
   - artifact log:
     `/tmp/mopd_qwen3_4b_mini_run_16x16_vbt/mini_20260326_203212/logs/model_training_20260326_20.log`

3. periodic save/test rehearsal completed on `/gpfs`
   - command:
     ```bash
     CKPTS_ROOT=/gpfs/Mamba/Project/Single_Cell/tmp/mopd_qwen3_4b_mini_val_save_test_gpfs \
     SWANLAB_MODE=disabled \
     SAVE_FREQ=2 \
     TEST_FREQ=2 \
     bash recipe/mopd/run_mopd_qwen3_4b_mini_val.sh
     ```
   - artifact root:
     `/gpfs/Mamba/Project/Single_Cell/tmp/mopd_qwen3_4b_mini_val_save_test_gpfs/mini_val_20260329_054738`
   - final log:
     `/gpfs/Mamba/Project/Single_Cell/tmp/mopd_qwen3_4b_mini_val_save_test_gpfs/mini_val_20260329_054738/logs/model_training_20260329_05.log`
   - final state:
     - `Training Progress: 100%|...| 8/8`
     - `validation generation end`
     - `Initial validation metrics`
     - `Final validation metrics`
     - complete checkpoints at `global_step_2`, `global_step_4`, `global_step_6`, and `global_step_8`
     - `latest_checkpointed_iteration.txt = 8`

### Operational conclusion

- the bounded wrappers are sufficient to exercise the real production launcher without forking the main Hydra graph
- `SAVE_FREQ=2` and `TEST_FREQ=2` are now proven on the validation-first preset under a spacious checkpoint root
- the earlier `/tmp` checkpoint failure was environmental disk exhaustion, not a launcher or checkpointing logic error
