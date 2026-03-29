# MOPD N-Teacher P0 Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `PLANS.md`.

**Goal:** Upgrade the current N-teacher MOPD branch from “teacher routing works” to “ExOPD-ready, preflight-validated, and resume-safe” without changing worker topology or resource-pool semantics.

**Architecture:** Keep the current `teachers[] + teacher_wgs + trainer-side sub-batch routing + adv_estimator="mopd"` design. Fill the missing P0 trainer/config plumbing in `verl/trainer/ppo/ray_trainer.py`, `verl/workers/config/teacher.py`, and `verl/trainer/config/algorithm/mopd.yaml`, then prove the new behavior through CPU-only unit and lightweight integration tests.

**Tech Stack:** Python, Hydra/OmegaConf, PyTorch, Ray trainer controller, verl `DataProto`, pytest

---

## Purpose / Big Picture

After this change, a user can run MOPD with a shared base model for ExOPD, resume from checkpoints without silently drifting MOPD semantics, and fail fast before worker startup when the dataset or teacher tokenizer metadata is invalid. The observable proof is that targeted pytest coverage will verify: `base_log_prob` is produced when base normalization is enabled, per-teacher lambda values override the global default only in ExOPD mode, invalid `teacher_id` or tokenizer settings fail during trainer preflight, and checkpoint manifest drift is rejected before actor/critic state is loaded.

## Progress

- [x] (2026-03-13 14:54Z) Read `PLANS.md`, the challenge analysis, the solution review, and the current MOPD code/tests to identify the P0 scope.
- [x] (2026-03-13 14:54Z) Confirmed via focused expert review that the estimator math already supports ExOPD and the remaining gap is trainer/runtime plumbing.
- [x] (2026-03-13 15:12Z) Wrote failing tests for base worker wiring, lambda precedence, tokenizer gating, base-model compatibility, preflight checks, and checkpoint manifest drift.
- [x] (2026-03-13 15:12Z) Implemented `TeacherConfig` schema updates for optional per-teacher lambda and tokenizer compatibility metadata.
- [x] (2026-03-13 15:12Z) Implemented trainer helpers for base worker config, base log-prob computation, lambda tensor construction, preflight validation, and manifest save/load.
- [x] (2026-03-13 15:12Z) Updated Hydra MOPD config comments to expose the new teacher fields.
- [x] (2026-03-13 15:12Z) Ran targeted verification: `pytest tests/unit/test_teacher_config.py tests/unit/test_teacher_workers.py tests/unit/test_teacher_routing.py tests/unit/test_mopd_advantage.py tests/unit/test_mopd_trainer_runtime.py tests/integration/test_mopd_e2e.py -v` passed with `59 passed, 1 skipped`.
- [x] (2026-03-13 15:12Z) Ran `ruff check` on all touched Python files and resolved the final review-driven tokenizer/base-model gaps.

## Surprises & Discoveries

- Observation: `compute_mopd_advantage()` in `verl/trainer/ppo/core_algos.py` already contains the ExOPD formula and `compute_advantage()` already forwards `base_log_prob` when present.
  Evidence: The current trainer only lacks runtime creation of a base reference worker and population of `batch.batch["base_log_prob"]`.
- Observation: `TeacherConfig.base_model_path` exists today but is not consumed by trainer runtime code.
  Evidence: `RayPPOTrainer._build_teacher_worker_config()` only rewrites `model.path` and ref batching settings.
- Observation: The safest insertion point for `teacher_id` and tokenizer validation is trainer preflight, not dataset construction.
  Evidence: `RLHFDataset` only knows data config and the student tokenizer, while teacher compatibility is a trainer/runtime contract.
- Observation: The tokenizer gate also has to cover the ExOPD shared base model, not only routed teachers.
  Evidence: A review pass found that `base_log_prob` could otherwise bypass fail-fast validation when `use_base_normalization=True`, so the final implementation validates the base-model tokenizer signature too.
- Observation: Comparing only tokenizer metadata like `pad_token_id` and `vocab_size` is not strong enough for the token-level log-prob contract.
  Evidence: The final gate adds a deterministic full-vocabulary hash and a unit test that proves mismatched vocabularies now fail even when the superficial metadata matches.

## Decision Log

- Decision: Keep resource-pool behavior unchanged in this round and continue failing fast on unknown teacher pools.
  Rationale: The review document classifies teacher pool expansion as P1, and changing `TaskRunner.init_resource_pool_mgr()` would widen the distributed blast radius.
  Date/Author: 2026-03-13 / Codex
- Decision: Use `algorithm.mopd.base_model_path` as the only runtime source of truth for the shared base model.
  Rationale: It matches the review document, avoids reviving the dead `TeacherConfig.base_model_path` field, and keeps ExOPD semantics shared across teachers.
  Date/Author: 2026-03-13 / Codex
- Decision: Batch-provided `lambda_val` will override `config.algorithm.mopd.lambda_val`, with the scalar config retained as the fallback.
  Rationale: This preserves backward compatibility while enabling per-teacher ExOPD without changing the estimator API again later.
  Date/Author: 2026-03-13 / Codex
- Decision: The shared ExOPD base worker keeps the normal reference batching configuration and only overrides `model.path`.
  Rationale: The base worker consumes the full batch like the regular reference path, so there is no routed-sub-batch reason to force the teacher-specific fixed micro-batch behavior.
  Date/Author: 2026-03-13 / Codex
- Decision: Tokenizer compatibility is proven by full tokenizer signatures, not only by compat-group labels or special-token metadata.
  Rationale: The runtime feeds student token IDs directly into teacher and base workers, so startup validation must reject vocabularies that differ even if high-level metadata matches.
  Date/Author: 2026-03-13 / Codex

## Outcomes & Retrospective

The intended P0 outcome has been reached in this worktree. The trainer remains architecturally identical to the current MOPD branch but now has ExOPD runtime completeness, per-teacher lambda plumbing, dataset/tokenizer fail-fast checks, and checkpoint manifest drift protection, all covered by CPU-only tests.

## Context and Orientation

The current MOPD implementation spans four important layers:

`verl/workers/config/teacher.py` defines the dataclasses for a single teacher and for the top-level `algorithm.mopd` config tree. It currently models teacher identity and a few runtime knobs, but it does not yet carry tokenizer compatibility metadata or per-teacher lambda.

`verl/trainer/config/algorithm/mopd.yaml` is the Hydra surface for those dataclasses. It already exposes `use_base_normalization` and `base_model_path`, but it does not describe tokenizer compatibility or per-teacher lambda yet.

`verl/trainer/ppo/ray_trainer.py` is the controller. This file already creates teacher worker groups, routes sub-batches by `teacher_id`, and computes `teacher_log_prob`. It also owns checkpoint save/load and has access to the instantiated train dataset, which makes it the correct place to add preflight validation, base-worker creation, and manifest drift checks.

`verl/trainer/ppo/core_algos.py` already registers the `"mopd"` estimator. In standard mode it uses `teacher_log_prob - old_log_probs`; in ExOPD mode it uses `teacher_log_prob`, `old_log_probs`, `base_log_prob`, and `lambda_val`. The math is not the problem. The missing behavior is that trainer runtime never computes `base_log_prob`, never builds a per-sample lambda tensor, and never validates that resume data and dataset routing are coherent.

The key tests already exist in `tests/unit/test_mopd_advantage.py`, `tests/unit/test_teacher_workers.py`, `tests/unit/test_teacher_routing.py`, `tests/unit/test_teacher_config.py`, and `tests/integration/test_mopd_e2e.py`. This change should extend those tests and add focused trainer-helper coverage rather than introducing GPU- or Ray-dependent checks for P0.

## Plan of Work

### Task 1: Freeze the desired behavior in tests before touching implementation

Files:
- Modify: `tests/unit/test_mopd_advantage.py`
- Modify: `tests/unit/test_teacher_workers.py`
- Modify: `tests/unit/test_teacher_config.py`
- Add or modify: `tests/unit/test_mopd_preflight.py`
- Modify: `tests/integration/test_mopd_e2e.py`

Step 1: Write the failing test for lambda precedence.

Add a test that creates a batch containing `base_log_prob` and `lambda_val` tensor values that differ across samples, passes a conflicting scalar `config.mopd.lambda_val`, and asserts that `compute_advantage()` uses the batch values instead of the config fallback.

Step 2: Run only that test to confirm it fails for the right reason.

Run in `/home/scbjtfy/verl/.worktrees/mopd-implementation`:

    pytest tests/unit/test_mopd_advantage.py -k lambda -v

Expected before implementation: the result still reflects the config scalar instead of the per-sample tensor.

Step 3: Write failing tests for base worker config and manifest/preflight helpers.

Add unit tests that exercise pure helpers on a trainer instance created via `RayPPOTrainer.__new__(RayPPOTrainer)`. The tests should assert:

- `_build_base_worker_config()` uses `algorithm.mopd.base_model_path`
- `_compute_base_log_prob()` returns the expected tensor shape from a mock worker
- preflight rejects unknown `teacher_id`, missing configured teachers, and tokenizer incompatibility
- manifest comparison hard-fails on semantic drift and tolerates deployment-only changes

Step 4: Run the new helper tests and confirm they fail because the helpers or fields do not exist yet.

Run:

    pytest tests/unit/test_teacher_workers.py tests/unit/test_mopd_preflight.py tests/unit/test_teacher_config.py -v

Step 5: Add a lightweight integration test showing standard MOPD ignores per-sample lambda when `base_log_prob` is absent, while ExOPD uses it when present.

Run:

    pytest tests/integration/test_mopd_e2e.py -k "lambda or exopd" -v

Expected before implementation: the ExOPD per-sample lambda path fails or uses the scalar fallback.

### Task 2: Extend config schema without changing the existing runtime topology

Files:
- Modify: `verl/workers/config/teacher.py`
- Modify: `verl/trainer/config/algorithm/mopd.yaml`

Step 1: Add schema fields required by the review document.

In `TeacherConfig`, add:

- `lambda_val: Optional[float] = None`
- `tokenizer_path: Optional[str] = None`
- `tokenizer_compat_group: Optional[str] = None`

Keep `resource_pool` and `log_prob_micro_batch_size` unchanged. Do not repurpose `weight`. Do not make tokenizer metadata mandatory; absence means “same tokenizer as student unless preflight proves otherwise.”

Step 2: Update validation rules.

Validate only what can be validated locally:

- `lambda_val`, when provided, must be positive
- `tokenizer_compat_group` can be any non-empty string if provided

Avoid enforcing tokenizer compatibility here because that requires the student model context held by the trainer.

Step 3: Reflect the new fields in Hydra config comments.

Update `verl/trainer/config/algorithm/mopd.yaml` so a novice can see the intended meaning of per-teacher lambda and tokenizer metadata directly in the config file.

Step 4: Run the config tests.

Run:

    pytest tests/unit/test_teacher_config.py tests/unit/test_teacher_workers.py -v

Expected after implementation: all schema-focused tests pass.

### Task 3: Complete trainer-side ExOPD plumbing and per-teacher lambda construction

Files:
- Modify: `verl/trainer/ppo/ray_trainer.py`

Step 1: Add base worker state and helper construction.

Add `self.base_policy_wg: Optional[RayWorkerGroup] = None` in `RayPPOTrainer.__init__()`.

Add `_build_base_worker_config()` beside `_build_teacher_worker_config()`. It should clone `self.config.actor_rollout_ref`, replace `model.path` with `self.config.algorithm.mopd.base_model_path`, and disable dynamic ref batching exactly as teacher workers do.

Step 2: Initialize the base worker only when `use_base_normalization` is enabled.

In `init_workers()`, after reference-policy setup and before teacher worker creation completes, create a base worker group using the existing `Role.RefPolicy` worker class and the normal reference resource pool. Do not invent new resource pools in this round.

Step 3: Add `_compute_base_log_prob(batch)`.

This helper should mirror `_compute_ref_log_prob()` in the legacy-worker path: forward the full batch to `self.base_policy_wg.compute_ref_log_prob(batch)` and return a tensor or `DataProto` that can be stored under `batch.batch["base_log_prob"]`. The helper should raise a clear error if base normalization is enabled but `self.base_policy_wg` is missing.

Step 4: Add a helper that builds the per-sample lambda tensor.

Use `teacher_id` and `self.config.algorithm.mopd.teachers` to construct a `[batch_size, 1]` or `[batch_size, response_len]` broadcastable tensor of lambda values. Each teacher uses `teacher.lambda_val` when provided, otherwise the global `config.algorithm.mopd.lambda_val`.

Step 5: Thread both values into the main training path.

In the block that currently computes `teacher_log_prob`, also populate:

- `batch.batch["base_log_prob"]` when base normalization is enabled
- `batch.batch["lambda_val"]` whenever teacher routing is active

Then update `compute_advantage()` so `adv_kwargs["lambda_val"]` prefers `data.batch["lambda_val"]` and falls back to `config.mopd.lambda_val`.

Step 6: Run the MOPD advantage and routing tests.

Run:

    pytest tests/unit/test_mopd_advantage.py tests/integration/test_mopd_e2e.py -k "mopd or exopd" -v

Acceptance: ExOPD tests pass, standard MOPD remains unchanged, and per-sample lambda only affects ExOPD cases.

### Task 4: Add trainer preflight checks for dataset routing and tokenizer compatibility

Files:
- Modify: `verl/trainer/ppo/ray_trainer.py`
- Modify: `tests/unit/test_mopd_preflight.py`

Step 1: Add `_run_mopd_preflight_checks()`.

This helper should run before worker creation in `init_workers()`. It must inspect `self.train_dataset` and the configured teacher list. The dataset may be a real `RLHFDataset` or a test double, so the helper should consume only observable attributes such as `len(dataset)`, `dataset[i]`, and `teacher_id` fields on items.

Step 2: Validate routing data.

Collect the observed `teacher_id` values and counts from the train dataset when MOPD is enabled.

Hard-fail when:

- the dataset contains an unknown teacher id
- the dataset contains only the `"default"` fallback in a multi-teacher setup
- one or more configured teachers never appear in the dataset

Log the observed distribution with `logger.info(...)`.

Step 3: Validate tokenizer compatibility.

Resolve the student tokenizer identity from `actor_rollout_ref.model.tokenizer_path` and fall back to `actor_rollout_ref.model.path`. For each teacher:

- if neither tokenizer field is set, treat the teacher tokenizer as the teacher model path
- if `teacher.tokenizer_compat_group` matches the student compat group, allow it
- otherwise require the teacher tokenizer path to equal the student tokenizer path

This keeps the rule strict and aligned with the current token-level log-prob contract.

Step 4: Run the preflight tests.

Run:

    pytest tests/unit/test_mopd_preflight.py -v

Acceptance: invalid dataset/tokenizer combinations fail before any worker startup logic is reached.

### Task 5: Persist and validate MOPD semantic manifest during save/load

Files:
- Modify: `verl/trainer/ppo/ray_trainer.py`
- Modify: `tests/unit/test_mopd_preflight.py` or add dedicated manifest tests there

Step 1: Add `_build_mopd_manifest()`.

Return a plain Python dict containing only semantic MOPD fields:

- `adv_estimator`
- ordered teacher specs with `name`, `model_path`, `lambda_val`, `tokenizer_path`, `tokenizer_compat_group`
- `use_base_normalization`
- `base_model_path`
- `orm_weight`
- `is_correction`
- `is_epsilon_low`
- `is_epsilon_high`

Optionally include deployment fields such as `resource_pool` and `log_prob_micro_batch_size` in a separate section so they can be warned on without causing hard failure.

Step 2: Save the manifest in `_save_checkpoint()`.

Write `mopd_manifest.json` into the `global_step_*` directory after creating it for dataloader state. Use `json.dump(..., indent=2)` to stay within existing stdlib dependencies.

Step 3: Validate the manifest in `_load_checkpoint()`.

Before actor/critic checkpoint loads, read the saved manifest and compare it with the current manifest. Hard-fail on semantic drift. Warn through `logger.warning(...)` when only deployment fields change.

Step 4: Run helper tests for save/load drift handling.

Run:

    pytest tests/unit/test_mopd_preflight.py -k manifest -v

Acceptance: semantic drift aborts resume before model state is loaded.

## Concrete Steps

All commands below run from `/home/scbjtfy/verl/.worktrees/mopd-implementation`.

1. Write tests first and run them red:

       pytest tests/unit/test_mopd_advantage.py -k lambda -v
       pytest tests/unit/test_teacher_workers.py tests/unit/test_mopd_preflight.py tests/unit/test_teacher_config.py -v
       pytest tests/integration/test_mopd_e2e.py -k "lambda or exopd" -v

2. Implement the config changes and rerun the schema-focused tests:

       pytest tests/unit/test_teacher_config.py tests/unit/test_teacher_workers.py -v

3. Implement trainer ExOPD plumbing and rerun the algorithm tests:

       pytest tests/unit/test_mopd_advantage.py tests/integration/test_mopd_e2e.py -k "mopd or exopd" -v

4. Implement preflight and manifest helpers and rerun their tests:

       pytest tests/unit/test_mopd_preflight.py -v

5. Run the combined targeted verification for this change set:

       pytest tests/unit/test_teacher_config.py tests/unit/test_teacher_workers.py tests/unit/test_mopd_advantage.py tests/unit/test_mopd_preflight.py tests/integration/test_mopd_e2e.py -v

Expected final outcome: all targeted tests pass on CPU without requiring Ray worker startup or GPUs.

## Validation and Acceptance

The change is acceptable only if all of the following are true:

1. `compute_advantage()` uses batch-level `lambda_val` over the config fallback, proven by a test with conflicting values.
2. Enabling `algorithm.mopd.use_base_normalization=true` causes trainer runtime to compute and supply `base_log_prob`, proven by a mock-worker test.
3. A train dataset with bad `teacher_id` or incompatible tokenizer metadata fails during trainer preflight before worker initialization.
4. Loading a checkpoint with changed semantic MOPD configuration fails before actor/critic state is loaded.
5. Standard MOPD behavior remains unchanged when `base_log_prob` is absent, proven by regression tests in `tests/unit/test_mopd_advantage.py` and `tests/integration/test_mopd_e2e.py`.

## Idempotence and Recovery

The planned edits are additive and can be repeated safely. Re-running the tests is idempotent. If the manifest comparison is implemented incorrectly and starts blocking resume, the safe recovery path is to inspect the emitted semantic diff, fix the comparison logic, and rerun only the manifest tests before touching any model state. No destructive git operations are required at any step.

## Artifacts and Notes

Important expected evidence snippets after implementation:

    tests/unit/test_mopd_advantage.py::test_batch_lambda_overrides_config_scalar_for_exopd_dispatch PASSED
    tests/unit/test_mopd_trainer_runtime.py::test_run_mopd_preflight_rejects_unknown_teacher_ids PASSED
    tests/unit/test_mopd_trainer_runtime.py::test_validate_loaded_mopd_manifest_rejects_semantic_drift PASSED

Final verification transcript:

    pytest tests/unit/test_teacher_config.py tests/unit/test_teacher_workers.py tests/unit/test_teacher_routing.py tests/unit/test_mopd_advantage.py tests/unit/test_mopd_trainer_runtime.py tests/integration/test_mopd_e2e.py -v
    ...
    ================== 59 passed, 1 skipped, 3 warnings in 6.48s ===================

    ruff check verl/workers/config/teacher.py verl/trainer/ppo/core_algos.py verl/trainer/ppo/ray_trainer.py verl/trainer/ppo/utils.py tests/unit/test_teacher_config.py tests/unit/test_teacher_workers.py tests/unit/test_teacher_routing.py tests/unit/test_mopd_advantage.py tests/unit/test_mopd_trainer_runtime.py tests/integration/test_mopd_e2e.py
    All checks passed!

Note: `verl/trainer/config/algorithm/mopd.yaml` was intentionally excluded from Ruff because Ruff parses inputs as Python source; running it directly on YAML produces false-positive `F821` errors for YAML literals such as `true`, `false`, and `null`.

If any command above fails for an unrelated reason, record the failure in `Surprises & Discoveries`, update `Progress`, and continue from the smallest failing scope rather than broadening verification prematurely.

## Interfaces and Dependencies

At the end of this plan, the following interfaces must exist:

In `verl/workers/config/teacher.py`, `TeacherConfig` must expose:

    lambda_val: Optional[float]
    tokenizer_path: Optional[str]
    tokenizer_compat_group: Optional[str]

In `verl/trainer/ppo/ray_trainer.py`, `RayPPOTrainer` must expose helper methods equivalent to:

    def _build_base_worker_config(self): ...
    def _compute_base_log_prob(self, batch: DataProto) -> DataProto | torch.Tensor: ...
    def _build_mopd_lambda_tensor(self, batch: DataProto) -> torch.Tensor: ...
    def _run_mopd_preflight_checks(self) -> None: ...
    def _build_mopd_manifest(self) -> dict[str, Any]: ...

`compute_advantage(...)` in the same file must prefer `data.batch["lambda_val"]` over `config.mopd.lambda_val`.

Update note: created this ExecPlan on 2026-03-13 because the review document identified concrete P0 implementation gaps but the repository did not yet contain a self-contained execution document for the fixes. Updated at 2026-03-13 15:12Z after implementation to record the completed runtime/tokenizer/manifest work and the final verification evidence. Updated again at 2026-03-13 15:34Z to record the final rerun of pytest and the corrected Ruff verification scope.
