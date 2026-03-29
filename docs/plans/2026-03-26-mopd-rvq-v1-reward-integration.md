# MOPD RVQ V1 Reward Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current lightweight MOPD evaluator with an RVQ V1-backed wrapper while preserving MOPD teacher distillation as the default training signal.

**Architecture:** Add a local wrapper under `recipe/mopd` that delegates to the external RVQ stable reward entrypoint, adapts MOPD dataset inputs, and exposes a verl-compatible `compute_score`. Update the MOPD launcher to use the wrapper plus the built-in `rate_limited` reward manager and keep `algorithm.mopd.orm_weight=0.0` by default, while exposing an explicit override knob for future ORM mixing experiments.

**Tech Stack:** Python, bash, Hydra config overrides, verl reward loop, pytest

---

### Task 1: Lock the intended launcher and wrapper contract with tests

**Files:**
- Create: `tests/unit/test_mopd_rvq_v1_reward.py`
- Modify: `tests/unit/test_mopd_run_script.py`

**Step 1: Write the failing tests**

- Add wrapper tests that assert:
  - missing RVQ root fails with a clear `FileNotFoundError`
  - the wrapper delegates to the external RVQ entrypoint and returns `score`, `acc`, and `pred`
- Update the run-script test to assert:
  - the launcher points to `recipe/mopd/rvq_v1_reward.py`
  - the launcher uses `reward.reward_manager.name=rate_limited`
  - the launcher exposes RVQ reward env knobs and `MOPD_ORM_WEIGHT`

**Step 2: Run tests to verify they fail**

Run:

```bash
pytest -q tests/unit/test_mopd_rvq_v1_reward.py tests/unit/test_mopd_run_script.py
```

Expected: failures because the wrapper file and launcher changes do not exist yet.

### Task 2: Implement the local RVQ V1 wrapper

**Files:**
- Create: `recipe/mopd/rvq_v1_reward.py`

**Step 1: Write minimal implementation**

- Load the external RVQ stable entrypoint by file path from `rvq_reward_root` or `RVQ_REWARD_ROOT`
- Normalize `ground_truth` into `list[str]`
- Ensure `extra_info["task_type"]` exists when available from MOPD data
- Call `compute_deepseek_singlecell_reward(..., reward_version="v1", ...)`
- Return the delegated result plus:
  - `acc` alias derived from `llm_judge_accuracy`, `answer_valid`, or `score`
  - lightweight `pred` string derived from the local solution text

**Step 2: Run the wrapper tests**

Run:

```bash
pytest -q tests/unit/test_mopd_rvq_v1_reward.py
```

Expected: PASS

### Task 3: Update the production MOPD launcher

**Files:**
- Modify: `recipe/mopd/run_mopd_qwen3_4b.sh`

**Step 1: Replace the reward integration path**

- Switch reward config to:
  - `reward.reward_manager.name=rate_limited`
  - `reward.num_workers=...`
  - `+reward.max_concurrent=...`
  - `+reward.max_rpm=...`
  - `+reward.max_tpm=...`
  - `+reward.estimated_tokens_per_request=...`
  - `+reward.timeout=...`
  - `reward.custom_reward_function.path="${SCRIPT_DIR}/rvq_v1_reward.py"`
  - `reward.custom_reward_function.name=compute_score`
- Add launcher env knobs for RVQ reward root/provider/API configuration
- Add `MOPD_ORM_WEIGHT=${MOPD_ORM_WEIGHT:-0.0}` and wire it to `algorithm.mopd.orm_weight`
- Keep rollout/FSDP defaults unchanged

**Step 2: Run launcher-contract tests**

Run:

```bash
pytest -q tests/unit/test_mopd_run_script.py
```

Expected: PASS

### Task 4: Refresh docs

**Files:**
- Modify: `recipe/mopd/README.md`

**Step 1: Update the validation/reward contract section**

- Explain that the production launcher now uses the local RVQ V1 wrapper
- Note that default `MOPD_ORM_WEIGHT=0.0` keeps reward eval-only
- Document the key env knobs and rate-limiting guidance

### Task 5: Verify and review

**Files:**
- Verify only

**Step 1: Run focused verification**

Run:

```bash
pytest -q tests/unit/test_mopd_rvq_v1_reward.py tests/unit/test_mopd_run_script.py
```

**Step 2: Run one broader regression slice**

Run:

```bash
pytest -q tests/unit/test_mopd_trainer_runtime.py
```

**Step 3: Request code review**

- Dispatch a reviewer/code-reviewer style subagent on the final diff

