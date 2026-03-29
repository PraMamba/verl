# MOPD Deployment Closure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align operator-facing documentation and deployment verification with the actual MOPD runtime, then produce fresh preflight and GPU E2E evidence for this worktree.

**Architecture:** Keep the current MOPD runtime intact and treat this slice as deployment closure rather than feature development. The work will first lock the intended test and script contract with focused tests, then minimally adjust the GPU E2E and recipe entrypoints to share one environment-variable vocabulary, then update only the high-signal documents that currently misstate the branch status.

**Tech Stack:** Python, PyTorch, Ray, Hydra, pytest, shell scripts, Markdown documentation

---

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan must be maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

Someone operating MOPD from this branch should be able to read the repository, run the recipe-aligned preflight path, run the GPU E2E path on `CUDA_VISIBLE_DEVICES=4,5,6,7`, and understand whether the branch is validated or blocked. Right now the core runtime is ahead of a few high-trust documents and parts of the deployment verification surface. After this plan, the repository should expose a single consistent story: what MOPD does now, how to verify it, and what evidence exists.

## Progress

- [x] (2026-03-14 00:00+08) Reviewed current source-of-truth runtime files, test files, recipe scripts, and branch analysis documents.
- [x] (2026-03-14 00:00+08) Confirmed the design scope with the user: deployment closure, not new feature work.
- [x] (2026-03-14 00:00+08) Wrote the approved design document at `docs/plans/2026-03-14-mopd-deployment-closure-design.md`.
- [x] (2026-03-14 15:33+08) Locked the deployment env contract in `tests/integration/test_mopd_e2e.py`, preferring recipe env vars and preserving legacy fallback coverage.
- [x] (2026-03-14 15:40+08) Updated the recipe/preflight teacher overrides to declare `tokenizer_compat_group=qwen3-shared`, matching the real tokenizer metadata of the student and both teachers.
- [x] (2026-03-14 16:00+08) Verified the recipe preflight on `CUDA_VISIBLE_DEVICES=4,5,6,7`; it reached the first real actor update and exited cleanly.
- [x] (2026-03-14 16:06+08) Found that the GPU E2E pytest path was producing a false failure because it asserted checkpoint-directory creation even when `trainer.save_freq=-1`.
- [x] (2026-03-14 16:09+08) Added a failing-then-passing regression test for the GPU E2E success predicate and switched the test to detect `training/global_step:1` in the subprocess log.
- [x] (2026-03-14 16:18+08) Re-ran the full GPU E2E pytest entrypoint on the real recipe contract; it passed in 338.21s.
- [x] (2026-03-14 16:19+08) Re-ran the CPU-safe regression suite after the new E2E success-predicate tests landed; result: `109 passed, 1 skipped, 1 warning in 12.72s`.
- [x] (2026-03-14 16:22+08) Confirmed the operator-facing recipe README and updated the high-signal status docs so the current runtime and verification boundary are documented consistently.
- [ ] Request code review, address findings, and refresh verification evidence if changes occur.

## Surprises & Discoveries

- Observation: the live runtime already includes more functionality than several documents claim, including ExOPD base normalization, sequence-level teacher routing, quantized teachers, and per-teacher metrics.
  Evidence: `verl/trainer/ppo/ray_trainer.py`, `verl/trainer/ppo/core_algos.py`, and `tests/unit/test_mopd_trainer_runtime.py` contain those paths and their runtime tests.

- Observation: some branch summaries still report older totals such as `80 tests / 79 passed / 1 skipped`, while newer evidence indicates a larger green suite.
  Evidence: existing plan and summary docs disagree on both implementation completeness and test counts.

- Observation: the full GPU E2E test originally used its own environment-variable contract rather than the same names used by `recipe/mopd/run_mopd_qwen3_4b.sh`.
  Evidence: the original `tests/integration/test_mopd_e2e.py` read `MOPD_TEST_*` variables, while the recipe script used `STUDENT_MODEL_PATH`, `CELL_TYPE_TEACHER_PATH`, `DISEASE_STATE_TEACHER_PATH`, `TRAIN_FILE`, and `TEST_FILE`; the current test now prefers the recipe names and keeps the legacy fallback under test.

- Observation: the recipe/preflight originally failed fast on tokenizer compatibility even though the real student and both teachers shared the same tokenizer metadata.
  Evidence: the first real preflight run failed until `tokenizer_compat_group=qwen3-shared` was declared in the recipe/preflight overrides, after which the same run passed.

- Observation: the GPU E2E runtime had already succeeded once the subprocess exited with code 0 and logged `training/global_step:1`; the failing assertion was in the test harness, not in the trainer/runtime path.
  Evidence: `/tmp/pytest-of-scbjtfy/pytest-28/test_mopd_training_e2e0/mopd-e2e.log` contains a successful step-1 metrics line and a clean training shutdown, while the old assertion failed only on `ckpt_dir.exists()`.

## Decision Log

- Decision: constrain this work to deployment closure rather than new MOPD functionality.
  Rationale: the main risk is stale operator guidance and verification drift, not missing algorithm code.
  Date/Author: 2026-03-14 / Codex

- Decision: keep the production shell entrypoint and the pytest E2E entrypoint separate, but align them on one shared environment-variable contract.
  Rationale: wrapping the production shell script from pytest would obscure failures behind unrelated environment and logging concerns.
  Date/Author: 2026-03-14 / Codex

- Decision: update only high-signal documents and mark older conclusions as historical where needed instead of rewriting all historical plans.
  Rationale: historical documents are still useful as audit trail; only operator-facing or status-facing contradictions are worth changing in this slice.
  Date/Author: 2026-03-14 / Codex

- Decision: treat the GPU E2E success boundary the same way as the recipe preflight success boundary: a real `training/global_step >= 1` line in the subprocess log.
  Rationale: `trainer.save_freq=-1` intentionally avoids checkpoint creation, so checkpoint-directory existence is not a valid success signal for this entrypoint.
  Date/Author: 2026-03-14 / Codex

## Outcomes & Retrospective

- Verification that succeeded:
  - CPU-safe regression suite: `109 passed, 1 skipped, 1 warning in 12.72s`
  - Recipe-aligned preflight on `CUDA_VISIBLE_DEVICES=4,5,6,7`: passed, with success logged at `/gpfs/Mamba/Project/Single_Cell/Training/Qwen3-4B-Thinking-2507_Manual_Resize_Block-8_Size-32_Num-256_ExplicitTokens_Continued-Pretraining-ALL_Augmented-V1_SFT-ALL_DFT-MOPD/preflight/logs/mopd_preflight_20260314_155450.log`
  - GPU E2E pytest on the same recipe contract: `1 passed, 1 warning in 338.21s`

- Repository/runtime issues fixed in this slice:
  - Declared the real tokenizer compatibility group for the Qwen3 student/teacher trio in the recipe/preflight/test surfaces
  - Corrected the GPU E2E test harness so success is based on a real completed training step rather than checkpoint-directory creation

- Current source-of-truth documents:
  - `recipe/mopd/README.md` for operator-facing deployment guidance
  - `docs/plans/mopd-test-results.md` for current verification evidence
  - `docs/plans/mopd-changes-summary.md` as historical audit trail with explicit “historical snapshot” markers on outdated sections

- Remaining work outside this slice:
  - long-run stability and convergence validation
  - full-run memory-margin characterization on production batch sizes
  - quantized-teacher quality-vs-memory study on real tasks
  - multi-node / failure-recovery validation

## Context and Orientation

`verl/trainer/ppo/ray_trainer.py` is the main controller for PPO-style training in this repository. In this branch it already includes MOPD-specific logic such as teacher worker-group setup, teacher-id preflight, base-model log-prob support, sequence-level teacher reward collection, per-teacher metrics, and manifest validation.

`verl/trainer/ppo/core_algos.py` contains `compute_mopd_advantage`, which already supports token-level MOPD, ExOPD base normalization, sequence-teacher reward composition, ORM mixing, and importance-sampling diagnostics.

`recipe/mopd/run_mopd_qwen3_4b.sh` is the deployment-oriented shell entrypoint that names the real student model, teacher models, data paths, and `CUDA_VISIBLE_DEVICES=4,5,6,7` default for this worktree. `recipe/mopd/run_mopd_qwen3_4b_preflight.sh` is the lightweight recipe-aligned validation path that should be trusted before a full run.

`tests/integration/test_mopd_e2e.py` now provides both lightweight integration coverage and the gated full GPU E2E test. The gated test consumes the same recipe-first environment-variable contract as the shell recipe (`STUDENT_MODEL_PATH`, `CELL_TYPE_TEACHER_PATH`, `DISEASE_STATE_TEACHER_PATH`, `TRAIN_FILE`, `TEST_FILE`) while preserving legacy `MOPD_TEST_*` fallback coverage for older harnesses.

`recipe/mopd/README.md` is the operator-facing documentation most likely to be read before running the recipe. `docs/plans/mopd-test-results.md` and `docs/plans/mopd-changes-summary.md` are high-signal status documents that currently contain some outdated claims and should be corrected minimally.

In this plan, "preflight" means the fast validation path that proves the first real batch can clear teacher-id checks, tokenizer checks, worker initialization, and the expected early-training milestone. "GPU E2E" means a full Ray and GPU-backed trainer execution that advances at least one real MOPD training step.

## Plan of Work

### Task 1: Lock the new deployment contract with failing tests

The first task is to encode the intended contract before touching the production-facing files. Extend `tests/integration/test_mopd_e2e.py` so the gated GPU test expects the same environment-variable names used by `recipe/mopd/run_mopd_qwen3_4b.sh`. Add focused test coverage, or update existing tests, for any helper logic that translates environment variables into a compact E2E configuration. The test must fail before the implementation change and pass afterward.

Also add coverage for any preflight helper or script-facing assumptions that are easy to check from Python. If the script contract cannot be unit-tested directly, create the narrowest possible helper surface so the contract becomes testable without shelling out to the full production script.

### Task 2: Implement minimal entrypoint alignment

Once the failing tests exist, update `tests/integration/test_mopd_e2e.py` and any required helpers so the E2E test consumes the shared environment-variable contract. Keep the test lightweight in shape: one small training run that proves the real trainer path, not the full production logging and checkpoint behavior.

Then inspect `recipe/mopd/run_mopd_qwen3_4b_preflight.sh` and `recipe/mopd/run_mopd_qwen3_4b.sh` and make only the smallest changes needed to keep defaults and comments aligned with the new documented contract. Preserve backward-safe defaults where possible.

### Task 3: Update high-signal operator-facing and result-facing docs

Update `recipe/mopd/README.md` so it clearly states the current runtime capability and the current validation boundary. It must explain that ExOPD base normalization is implemented, that heterogeneous-tokenizer support works through `sequence_reward` teachers, that quantized teachers are supported through dedicated teacher workers, and that true deployment validation still depends on explicitly running preflight and GPU E2E.

Update `docs/plans/mopd-test-results.md` with fresh numbers and dates based on the verification actually run in this session. If `docs/plans/mopd-changes-summary.md` still contains a directly harmful contradiction such as "ExOPD deferred", add a small historical clarification rather than rewriting the full document.

### Task 4: Run layered verification and capture evidence

After code and docs are updated, run the targeted CPU test suite first. Then run the recipe-aligned preflight path. Then run the gated GPU E2E path with `CUDA_VISIBLE_DEVICES=4,5,6,7` and the real recipe environment variables.

If preflight or GPU E2E fails, classify the failure first: environment or repository runtime. Only when evidence points to a repository bug should implementation change again. Any confirmed runtime regression must get a minimal regression test before re-running the verification layers.

### Task 5: Review, fix, and finalize

After the branch is stable and the verification output is captured, request code review. Fix any important findings, then rerun the necessary verification commands. Update this plan's `Progress`, `Surprises & Discoveries`, and `Outcomes & Retrospective` sections so a future contributor can resume from this file alone.

## Concrete Steps

Work from `/home/scbjtfy/verl/.worktrees/mopd-implementation`.

1. Write or update the failing tests that lock the environment-variable contract.

    pytest tests/integration/test_mopd_e2e.py -k "e2e or config" -v

   Expected before implementation: a failure or missing coverage that proves the old contract is still in place.

2. Implement the smallest code and script changes needed to satisfy the tests.

    pytest tests/integration/test_mopd_e2e.py -k "e2e or config" -v

   Expected after implementation: the targeted test selection passes.

3. Run the main CPU verification suite for this slice.

    pytest tests/unit/test_teacher_config.py tests/unit/test_teacher_workers.py tests/unit/test_mopd_advantage.py tests/unit/test_mopd_trainer_runtime.py tests/integration/test_mopd_e2e.py -v

   Expected: all selected CPU-safe tests pass, and the only GPU-gated case remains skipped until explicitly enabled.

4. Run recipe-aligned preflight.

    CUDA_VISIBLE_DEVICES=4,5,6,7 bash recipe/mopd/run_mopd_qwen3_4b_preflight.sh

   Expected: preflight reaches its declared success milestone and exits cleanly.

5. Run the full GPU E2E verification using the shared environment-variable contract.

    CUDA_VISIBLE_DEVICES=4,5,6,7 VERL_MOPD_E2E=1 pytest tests/integration/test_mopd_e2e.py::test_mopd_training_e2e -v

   Expected: the trainer initializes, worker groups come up, the training loop advances at least one real step, and the test exits `PASSED`.

6. Refresh the result documents with the observed outcomes.

   Expected: the docs and the observed evidence match exactly on dates, counts, and current capability claims.

## Validation and Acceptance

This work is complete only when all of the following are true:

1. The GPU E2E test and the recipe scripts use the same semantic environment-variable contract for model and data paths.
2. The targeted CPU-safe test suite passes after the code changes.
3. The recipe-aligned preflight path completes successfully, or a clearly documented environment blocker is recorded.
4. The gated GPU E2E test completes successfully on `CUDA_VISIBLE_DEVICES=4,5,6,7`, or a clearly documented environment blocker is recorded.
5. `recipe/mopd/README.md` no longer understates the runtime capabilities that already exist in code.
6. `docs/plans/mopd-test-results.md` reports fresh counts and outcomes from this session rather than stale numbers.
7. Any remaining contradictory line in `docs/plans/mopd-changes-summary.md` that could mislead an operator is clarified as historical.

## Idempotence and Recovery

The documentation updates are idempotent and can be rerun safely. The pytest commands can be rerun safely. The recipe scripts should remain safe to rerun as long as model and data paths are valid.

If preflight fails because of environment or path issues, fix the environment and rerun preflight before touching repository logic. If GPU E2E fails because of environment setup, capture that evidence and do not claim runtime validation succeeded. If GPU E2E fails because of a repository bug, add a regression test before fixing the implementation.

Do not broaden scope into performance optimization, worker parallelism, or new algorithm work while executing this plan.

## Artifacts and Notes

Important evidence to capture during implementation and verification:

- failing and passing targeted pytest output for the E2E contract change
- preflight success or failure marker
- GPU E2E pass or precise failure mode
- final CPU-safe verification output

Keep transcripts concise and place only the high-signal lines into the result documents.

## Interfaces and Dependencies

At the end of this plan, the repository should expose:

- a GPU E2E test in `tests/integration/test_mopd_e2e.py` that reads the same model and data environment variables as the recipe scripts
- recipe scripts whose comments and defaults match the shared contract
- operator-facing docs that describe the live MOPD runtime, not the older milestone state
- result documents that cite the verification actually executed in this session

Revision note: This plan was created after the deployment-closure design was approved. It converts that design into a novice-executable, test-first sequence focused on environment-contract alignment, verification, and high-signal documentation cleanup.
