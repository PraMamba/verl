# MOPD Deployment Closure Design

**Date:** 2026-03-14
**Branch:** `feature/mopd-implementation`
**Status:** Approved design, ready for execution

## Purpose

This design closes the gap between the current MOPD implementation and the way the branch is described and validated. The branch already contains the main runtime capabilities for N-teacher MOPD, ExOPD base normalization, heterogeneous-tokenizer support, quantized teacher backends, manifest drift detection, and per-teacher metrics. The remaining problem is that some operator-facing documents and the deployment verification surface still describe an older state.

After this work, an operator should be able to answer three questions from the repository alone:

1. What is actually implemented now.
2. Which files are still historical analysis and which files are current source of truth.
3. How to run preflight and GPU E2E verification using the real recipe assumptions in this worktree.

## Why This Slice

This branch does not currently need another feature wave before deployment verification. The highest-value work is to make the current state legible and to validate the deployment path with the existing runtime, not to reopen architecture work such as teacher parallel dispatch or checkpoint redesign.

This scope deliberately does **not** include:

- teacher parallelization across worker groups
- additional checkpoint self-description work beyond current manifest support
- new reward algorithms
- large-scale rewrite of historical planning documents

## Current True State

The current implementation already supports the following runtime capabilities:

- Explicit `algorithm.mopd.teachers` configuration with per-teacher routing metadata.
- Trainer-side N-teacher worker-group orchestration instead of actor-internal teacher slots.
- Standard MOPD reverse-KL token-level distillation.
- ExOPD base normalization through `use_base_normalization=true` and `base_model_path`.
- Per-teacher `lambda_val` overrides.
- Tokenizer policies `compatible` and `sequence_reward`.
- Quantized teacher backends through `hf_int8` and `hf_4bit`.
- Sequence-level teacher reward collection for heterogeneous-tokenizer teachers.
- Importance sampling correction and IS diagnostics.
- Per-teacher runtime metrics.
- Dataset-level `teacher_id` counting during preflight.
- MOPD manifest generation and semantic or deployment drift validation.

The strongest source-of-truth files for the current state are:

- `verl/workers/config/teacher.py`
- `verl/trainer/config/algorithm/mopd.yaml`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/ppo/core_algos.py`
- `tests/unit/test_mopd_trainer_runtime.py`
- `tests/unit/test_teacher_workers.py`
- `tests/unit/test_teacher_config.py`
- `tests/integration/test_mopd_e2e.py`

## Historical Drift To Correct

Several documents still describe a prior implementation stage and now risk misleading further development or deployment work. The main drift points are:

- documents that still say ExOPD base normalization is deferred or not wired
- test summary documents that still report the older `80 tests / 79 passed / 1 skipped` state
- recipe documentation that does not clearly separate current supported capability from historical design notes
- GPU E2E instructions that use a different environment-variable contract than the real recipe scripts

These documents should not be rewritten wholesale. Instead, the work should mark older statements as historical snapshots where needed and update the operator-facing documents that a practitioner is likely to trust first.

## Design Decisions

### Decision 1: Treat operator-facing docs and runtime tests as the primary correction surface

Only the files that directly shape deployment behavior or operator expectations should be updated in this slice. Historical analysis documents remain useful as an audit trail and should only receive minimal clarification if they currently contradict live runtime behavior in a harmful way.

### Decision 2: Align GPU E2E and recipe scripts around one environment-variable contract

The `recipe/mopd/run_mopd_qwen3_4b.sh` script is the real deployment entry point in this worktree. The full GPU E2E test should use the same naming and path assumptions, rather than inventing a separate test-only contract.

The target contract is:

- `STUDENT_MODEL_PATH`
- `CELL_TYPE_TEACHER_PATH`
- `DISEASE_STATE_TEACHER_PATH`
- `TRAIN_FILE`
- `TEST_FILE`
- `CUDA_VISIBLE_DEVICES`
- `VERL_MOPD_E2E`

The GPU E2E test may still use a reduced runtime footprint, but it should not use a semantically different set of model and data environment variables.

### Decision 3: Keep pytest and production shell execution separate

The E2E test should not directly wrap the full production shell script with `subprocess`. The production script includes environment exports, external logging integration, and site-specific assumptions that would make failures noisy and difficult to classify.

Instead:

- shell remains the human operator entry point
- pytest remains the programmatic verification entry point
- both share the same environment-variable contract and runtime assumptions

### Decision 4: Use layered verification

Verification must proceed in this order:

1. CPU unit and lightweight integration tests.
2. Recipe-aligned preflight.
3. GPU E2E.
4. Documentation result backfill.

If a lower layer fails, work stops and enters targeted debugging before moving upward.

## Files In Scope

### New

- `docs/plans/2026-03-14-mopd-deployment-closure-design.md`
- `docs/plans/2026-03-14-mopd-deployment-closure-plan.md`

### Update

- `recipe/mopd/README.md`
- `recipe/mopd/run_mopd_qwen3_4b_preflight.sh`
- `recipe/mopd/run_mopd_qwen3_4b.sh`
- `tests/integration/test_mopd_e2e.py`
- `docs/plans/mopd-test-results.md`
- `docs/plans/mopd-changes-summary.md` if needed for a high-signal correction only

## Verification Matrix

### Layer 1: CPU Verification

Run the relevant unit and integration tests that cover:

- MOPD config validation
- teacher worker surface
- trainer runtime helpers
- advantage computation
- lightweight integration path

Success means the branch remains green before any GPU-specific validation is attempted.

### Layer 2: Preflight Verification

Use the recipe-aligned preflight path to prove:

- teacher-id distribution is visible and valid
- tokenizer compatibility checks pass
- worker initialization succeeds
- first-batch or first-update validation reaches the expected success marker

Success means the deployment path is internally consistent before a full GPU E2E run.

### Layer 3: GPU E2E Verification

Use `CUDA_VISIBLE_DEVICES=4,5,6,7` and the recipe-aligned environment variables to run the full MOPD E2E path.

Success means:

- Ray initializes successfully on the chosen GPUs
- teacher worker groups initialize
- MOPD routing executes without shape or device failure
- the training loop advances at least one real step

### Layer 4: Documentation Consistency

After runtime verification, update result documents so the reported state matches the actual observed evidence. The repository should no longer simultaneously claim both "feature complete" and "ExOPD deferred".

## Failure Classification

### Environment Failures

These include:

- model paths not accessible
- data files missing
- GPU visibility or allocation issues
- Ray startup failure caused by environment
- external logger or site-local dependency failures

These should be resolved without changing algorithm code unless evidence shows a real repository bug.

### Runtime Failures

These include:

- teacher routing shape mismatches
- device mismatches
- tokenizer contract mismatches
- base normalization path failures
- worker initialization or dispatch regressions

These should trigger systematic debugging and a regression test when a repository bug is confirmed.

## Rollback and Recovery

This work is intended to be additive and safe to rerun.

- Documentation updates are low risk and can be retried directly.
- Test-entry and script changes must preserve existing environment-variable defaults where possible.
- If GPU verification fails for environmental reasons, the work can still land with corrected docs and a documented blocked verification result, but the result document must state that limitation explicitly.
- If GPU verification fails for repository reasons, the run is not considered complete until a focused fix and re-verification are performed.

## Expected Outcome

At the end of this slice, this branch should have:

- a clear design and implementation plan for deployment closure
- operator-facing documentation that reflects the real MOPD runtime
- aligned recipe and E2E environment-variable semantics
- fresh CPU verification evidence
- fresh preflight evidence
- fresh GPU E2E evidence, or a precise blocked-by-environment record

## Approved Transition

This design was presented incrementally and approved by the user before implementation. The next step is to create the execution plan and then implement the approved scope with test-first changes where behavior or test entrypoints change.
