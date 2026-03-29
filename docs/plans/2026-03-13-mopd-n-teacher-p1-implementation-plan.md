# MOPD N-Teacher P1 Throughput And Observability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `PLANS.md`.

**Goal:** Upgrade the current MOPD branch from P0 trainer/config completeness to P1 throughput and observability by making teacher resource pools real, adding pool-aware async teacher submission, exposing per-teacher metrics, and folding in the new blocking review fixes on routing order and tokenizer/runtime consistency.

**Architecture:** Keep the existing `teachers[] + teacher_wgs + trainer-side teacher_id routing` architecture. Extend `TaskRunner.init_resource_pool_mgr()` so teacher pools are explicitly described in the resource-pool specification, teach the trainer to overlap teacher forwards across different pools without changing per-pool sequencing semantics, and attach per-teacher metrics to the existing trainer metrics stream. While touching the same code paths, fix the review-proven correctness gaps so the new P1 behavior is built on a correct routed-teacher runtime.

**Tech Stack:** Python, Hydra/OmegaConf, Ray worker groups, verl `DataProto` / `DataProtoFuture`, pytest, Ruff

---

## Purpose / Big Picture

After this change, a user can give MOPD teachers their own named resource pools, and those pools will be instantiated instead of failing because only `global_pool` exists. The teacher phase will no longer be fully serialized when teachers sit on different pools; the trainer will overlap one in-flight teacher forward per pool and expose which teacher handled how much traffic, how strong its reverse-KL signal was, and how much IS masking survived. The observable proof is a CPU-only test suite that fails before the change and passes after it, plus resource-pool tests showing teacher pools and colocate budgets are computed explicitly.

## Progress

- [x] (2026-03-13 16:08Z) Re-read `solution-review`, `challenge-analysis`, `PLANS.md`, and the Superpowers process skills for this P1 round.
- [x] (2026-03-13 16:08Z) Re-triaged fresh code-review findings and accepted four blockers into this round: teacher tokenizer-path propagation, base-tokenizer preflight bypass, teacher routing order corruption after balancing, and base/teacher pool colocate-capacity drift.
- [x] (2026-03-13 16:12Z) Finished source inspection of `verl/trainer/main_ppo.py`, `verl/trainer/ppo/ray_trainer.py`, `verl/single_controller/ray/base.py`, and current tests to identify the minimal P1 insertion points.
- [x] (2026-03-13 16:26Z) Confirmed the red-test surface for teacher pools, routing order, tokenizer propagation/base-tokenizer validation, async overlap by pool, and per-teacher metrics. Added follow-up assertions for `adv_std` and future-vs-`TensorDict` output resolution.
- [x] (2026-03-13 16:31Z) Completed the P1 implementation path already staged in the branch: explicit `algorithm.mopd.resource_pools`, dynamic colocate budgets, async teacher worker APIs, routed scatter-index preservation, tokenizer-path propagation, base-normalization worker wiring, per-teacher metrics, and MOPD checkpoint manifest handling.
- [x] (2026-03-13 16:36Z) Fixed the remaining runtime regression in `_run_mopd_preflight_checks()` so dataset `teacher_id` errors are raised before tokenizer loading, and tightened `_resolve_teacher_log_prob_output()` so plain `TensorDict` outputs are not mis-materialized as futures.
- [x] (2026-03-13 16:44Z) Ran the targeted unit suite, integration suite, and Ruff checks for the P1 file set; all selected checks passed.
- [x] (2026-03-14 00:51+08) Removed the accidental duplicate helper definition in `ray_trainer.py` and reran the full P1 verification suite on the final diff (`71 passed, 1 skipped`, Ruff clean, compileall exit code 0).
- [x] (2026-03-13 16:46Z) Requested focused review from subagents. One review note about engine-worker `mesh_name=\"ref\"` metadata was addressed in the final cleanup by resolving the teacher DP mesh from worker method metadata before balancing/padding sub-batches.

## Surprises & Discoveries

- Observation: `teacher.resource_pool` already exists in the schema and worker init path, but `TaskRunner.init_resource_pool_mgr()` still only creates `global_pool` and optional `reward_pool`.
  Evidence: `verl/trainer/main_ppo.py` builds `resource_pool_spec` with only those two entries.
- Observation: the current `RayResourcePool(max_colocate_count=3)` assumption is no longer sufficient once ExOPD adds a base worker and MOPD adds per-teacher worker groups on the same pool.
  Evidence: `ResourcePoolManager.create_resource_pool()` hardcodes `max_colocate_count=3`, while `ray_trainer.py` now creates extra MOPD/base worker groups outside the original actor/critic/ref trio.
- Observation: `_balance_batch()` mutates the routed teacher sub-batch in place, so scattering teacher results back with the pre-balance `indices` is incorrect.
  Evidence: `batch.reorder(global_idx)` in `_balance_batch()` runs on the `DataProto` object passed to it.
- Observation: the tokenizer preflight and runtime teacher/base worker configs can diverge today.
  Evidence: preflight reads `teacher.tokenizer_path` and should read the base model tokenizer, but `_build_teacher_worker_config()` only rewrites `model.path`, and the base-tokenizer preflight currently prefers the student tokenizer path before the base model path.
- Observation: the worker-group framework already supports asynchronous, collected `DataProtoFuture` results for `blocking=False` registered methods.
  Evidence: `BatchData.concat()` turns lists of `ray.ObjectRef` into `DataProtoFuture`, and `func_generator()` skips `ray.get()` when the registered method is non-blocking.
- Observation: `TensorDict` also exposes a `.get()` method, so a naive "future if it has get()" check will corrupt plain worker outputs.
  Evidence: the engine ref worker returns `TensorDict`, and `_resolve_teacher_log_prob_output()` needed to short-circuit `DataProto` / `TensorDict` before touching future-style `.get()`.
- Observation: worker mesh metadata needed to be recovered from registered worker methods rather than hardcoding `"actor"` in the trainer.
  Evidence: `verl/workers/engine_workers.py` registers async ref log-prob on `mesh_name="ref"`, so the final patch added `_get_worker_method_mesh_name()` / `_get_teacher_dp_mesh_name()` to inspect `MAGIC_ATTR` and feed the correct mesh into `_get_dp_size()`.

## Decision Log

- Decision: Treat the fresh review blockers as part of this P1 round instead of deferring them.
  Rationale: They are high-severity correctness issues on the same MOPD routing/runtime path that P1 must extend. Adding throughput changes on top of known misrouting/tokenizer inconsistencies would make the branch harder to trust.
  Date/Author: 2026-03-13 / Codex
- Decision: Represent teacher pool specs in `algorithm.mopd.resource_pools` and lower them into `resource_pool_spec` in `TaskRunner.init_resource_pool_mgr()`.
  Rationale: The teacher-pool concept is MOPD-specific, and the `solution-review` explicitly says the real fix is to extend `resource_pool_spec`, not merely to allow arbitrary pool names on teachers.
  Date/Author: 2026-03-13 / Codex
- Decision: Generalize `ResourcePoolManager.resource_pool_spec` entries from bare `list[int]` to a back-compatible richer shape that can carry `max_colocate_count`.
  Rationale: Teacher pools and ExOPD base workers need explicit colocate capacity. Keeping only `list[int]` would force more ad-hoc hardcoding and would not solve the review-reported pool-capacity bug.
  Date/Author: 2026-03-13 / Codex
- Decision: P1 async teacher dispatch will run at most one in-flight teacher forward per resource pool, while overlapping different pools.
  Rationale: This matches the review guidance, preserves current single-pool semantics, and avoids inventing same-pool concurrency rules before measuring them.
  Date/Author: 2026-03-13 / Codex
- Decision: Per-teacher metrics will be emitted from trainer-side tensors and routing metadata, not by changing the estimator API again.
  Rationale: The trainer already owns `teacher_id`, `teacher_log_prob`, `old_log_probs`, and optional IS diagnostics. Emitting metrics there keeps the algorithm interface stable.
  Date/Author: 2026-03-13 / Codex
- Decision: Resolve teacher DP mesh names from registered worker-method metadata inside the trainer instead of hardcoding `"actor"`.
  Rationale: This keeps the async teacher path aligned with FSDP, Megatron, and engine workers without broadening the public worker-group API surface in this round.
  Date/Author: 2026-03-14 / Codex

## Outcomes & Retrospective

This round landed the full P1 runtime surface that the review document called out: teacher resource pools are now real `resource_pool_spec` entries, colocate capacity is derived from actual worker placement pressure, teacher forwards overlap across pools while preserving same-pool serialization, and trainer metrics now expose per-teacher sample share, advantage stats, reverse-KL, and IS-valid fractions.

The last failing regression after the large P1 patch was not in the new pool/runtime logic itself, but in preflight ordering: tokenizer loading still happened before dataset teacher routing validation, which could mask real dataset/configuration errors. That is now fixed, and the async-result path is also safer because `TensorDict` outputs are no longer mistaken for future-like objects just because they expose `.get()`.

The final cleanup also closed the only meaningful review follow-up: `_build_mopd_teacher_jobs()` no longer assumes teacher DP mesh `"actor"`, so the balancing/padding path can recover the correct mesh from registered worker metadata before calling `_get_dp_size()`.

## Context and Orientation

The files involved in this round span three layers.

`verl/trainer/main_ppo.py` owns `TaskRunner.init_resource_pool_mgr()`. This is where the repo converts configuration into a `resource_pool_spec`, which later becomes one or more Ray placement-group pools. Right now it only creates `global_pool` and optional `reward_pool`, so a teacher configured with `resource_pool: code_pool` is rejected at runtime because that pool never exists.

`verl/single_controller/ray/base.py` implements `ResourcePoolManager`, `RayResourcePool`, and the worker-group execution helpers. A "resource pool" here means a set of Ray placement groups over a fixed GPU allocation. `max_colocate_count` is the number of worker groups the scheduler is allowed to place on each GPU in that pool. This layer also contains the `execute_all_async()` machinery used by registered worker-group methods.

`verl/trainer/ppo/ray_trainer.py` owns MOPD runtime orchestration. It builds teacher worker configs, initializes teacher worker groups, routes batches by `teacher_id`, computes `teacher_log_prob`, and aggregates metrics. The P1 throughput and observability work belongs here, and the new review blockers also point here.

`verl/workers/fsdp_workers.py`, `verl/workers/megatron_workers.py`, and `verl/workers/engine_workers.py` expose `compute_ref_log_prob()` on the worker side. To get real non-blocking teacher submission through the existing worker-group binding layer, this round must add a non-blocking variant instead of trying to repurpose the current blocking method.

The main tests today are:

- `tests/unit/test_teacher_workers.py` for config and worker-config helpers.
- `tests/unit/test_teacher_routing.py` for routed teacher forwarding.
- `tests/unit/test_mopd_trainer_runtime.py` for trainer-side runtime helpers and tokenizer preflight.
- `tests/integration/test_mopd_e2e.py` for lightweight MOPD integration behavior.

## Plan of Work

The first milestone is test-first lock-in of the P1 surface and the review blockers. Add a new unit test around `TaskRunner.init_resource_pool_mgr()` that builds a minimal config with `algorithm.mopd.resource_pools.code_pool` and teachers assigned to `code_pool`, then asserts the returned `ResourcePoolManager.resource_pool_spec` includes that pool and that the stored colocate capacity is large enough for actor/critic/ref plus MOPD extras on pools that host them. In the same red step, extend `tests/unit/test_teacher_workers.py` and `tests/unit/test_mopd_trainer_runtime.py` to assert `teacher.tokenizer_path` is propagated into the teacher worker config and that the base-tokenizer compatibility check still loads and validates the real base model tokenizer even when the student config already defines `actor_rollout_ref.model.tokenizer_path`.

The second milestone is routed-teacher correctness before optimization. Extend `tests/unit/test_teacher_routing.py` with a non-constant mock teacher that returns per-sample sentinel outputs based on the routed sub-batch order. Use it to prove that balancing does not corrupt scatter-back alignment. Also add a focused trainer-runtime test that exercises the future async path in pure Python by stubbing teacher worker groups with a non-blocking `compute_ref_log_prob_async()`-style method and verifying the trainer overlaps different pools while keeping teachers on the same pool serialized.

The third milestone is the resource-pool implementation. In `verl/workers/config/teacher.py`, add a dataclass for teacher resource-pool specs and add a `resource_pools` field to `MOPDConfig`. In `verl/trainer/config/algorithm/mopd.yaml`, document the new structure with an example `code_pool`. In `verl/trainer/main_ppo.py`, extend `init_resource_pool_mgr()` to lower `algorithm.mopd.resource_pools` into `resource_pool_spec`, and compute a required `max_colocate_count` per pool from the role mapping plus extra MOPD worker groups (teacher workers and optional base worker). `verl/single_controller/ray/base.py` must accept the richer spec shape, still accept the old `list[int]` form for existing callers, and instantiate `RayResourcePool` with the chosen `max_colocate_count`.

The fourth milestone is the async teacher runtime. In the worker classes, add non-blocking `compute_ref_log_prob_async()` methods that reuse the same underlying implementation as `compute_ref_log_prob()`. In `verl/trainer/ppo/ray_trainer.py`, factor the teacher-routing code into smaller helpers that: build routed teacher jobs; attach the original full-batch indices into each sub-batch before balancing; submit at most one async job per pool; materialize and scatter results with the balanced original indices; and record per-teacher metrics from the returned tensors. While touching `_build_teacher_worker_config()` and `_validate_mopd_tokenizer_compatibility()`, fix the tokenizer propagation and base-tokenizer selection bugs from review.

The final milestone is observability and docs. Emit per-teacher metrics into the existing trainer metrics map under `mopd/<teacher>/...`, update the MOPD YAML comments to show teacher pools and metrics-related expectations, and record the final verification evidence in this ExecPlan.

## Concrete Steps

All commands below run from `/home/scbjtfy/verl/.worktrees/mopd-implementation`.

1. Write the failing tests first.

       pytest tests/unit/test_teacher_workers.py tests/unit/test_teacher_routing.py tests/unit/test_mopd_trainer_runtime.py -k "tokenizer or base or routing or pool or async or metric" -v

   Expected before implementation: failures around missing teacher-pool config/runtime support, missing tokenizer propagation, base-tokenizer bypass, and routing order preservation.

2. Add a focused `TaskRunner` unit test file if needed, then rerun just the pool tests.

       pytest tests/unit/test_mopd_resource_pools.py -v

   Expected before implementation: failure because `init_resource_pool_mgr()` only creates `global_pool` and `reward_pool`, and because `resource_pool_spec` cannot carry colocate metadata yet.

3. Implement the config and runtime changes incrementally, rerunning the smallest failing test after each change.

       pytest tests/unit/test_teacher_workers.py -k tokenizer -v
       pytest tests/unit/test_mopd_trainer_runtime.py -k "base or tokenizer" -v
       pytest tests/unit/test_teacher_routing.py -k "routing or async" -v
       pytest tests/unit/test_mopd_resource_pools.py -v

4. Run the full targeted verification for this round.

       pytest tests/unit/test_teacher_workers.py tests/unit/test_teacher_routing.py tests/unit/test_mopd_trainer_runtime.py tests/unit/test_mopd_resource_pools.py tests/integration/test_mopd_e2e.py -v
       ruff check verl/workers/config/teacher.py verl/trainer/main_ppo.py verl/trainer/ppo/ray_trainer.py verl/single_controller/ray/base.py verl/workers/fsdp_workers.py verl/workers/megatron_workers.py verl/workers/engine_workers.py tests/unit/test_teacher_workers.py tests/unit/test_teacher_routing.py tests/unit/test_mopd_trainer_runtime.py tests/unit/test_mopd_resource_pools.py tests/integration/test_mopd_e2e.py

5. Request final review on the diff and fix any valid findings before claiming completion.

## Validation and Acceptance

This P1 round is acceptable only if all of the following are true:

1. A config with `algorithm.mopd.resource_pools.<pool_name>` produces a real pool in `TaskRunner.init_resource_pool_mgr()`, and teachers referencing that pool no longer depend on undocumented external pool creation.
2. Pools that host extra MOPD/base worker groups compute a colocate capacity high enough that the scheduler does not silently reuse the old `max_colocate_count=3` assumption.
3. Teacher routing still preserves per-sample alignment after `_balance_batch()` reorders the sub-batch.
4. `teacher.tokenizer_path` is propagated into the teacher worker config, and the base-tokenizer compatibility gate validates the actual base model tokenizer path even when the student tokenizer path is explicitly configured.
5. The trainer overlaps teacher forwards across different pools while keeping same-pool teachers serialized in the first P1 version.
6. The trainer emits per-teacher metrics under `mopd/<teacher>/sample_fraction`, `adv_mean`, `adv_std`, `reverse_kl_mean`, and `is_valid_fraction` when the relevant tensors exist.

## Idempotence and Recovery

The plan is additive and safe to rerun. The new config fields must be optional and preserve the old behavior when `algorithm.mopd.resource_pools` is unset. If the richer `resource_pool_spec` parsing causes regressions, the safe recovery path is to keep backward compatibility for list-based specs and rerun only the new resource-pool tests before broadening verification again. If async teacher submission proves harder than expected on one backend, the safe fallback is to keep the synchronous path as a helper and guard the async route behind the presence of the new non-blocking worker method while maintaining the same public trainer behavior.

## Artifacts and Notes

Verified on 2026-03-14 with the final post-cleanup diff:

    pytest tests/unit/test_teacher_config.py tests/unit/test_teacher_workers.py tests/unit/test_teacher_routing.py tests/unit/test_mopd_advantage.py tests/unit/test_mopd_trainer_runtime.py tests/unit/test_mopd_resource_pools.py tests/integration/test_mopd_e2e.py -v
    # Result: 71 passed, 1 skipped, 3 warnings

    ruff check verl/workers/config/teacher.py verl/trainer/main_ppo.py verl/trainer/ppo/core_algos.py verl/trainer/ppo/utils.py verl/trainer/ppo/ray_trainer.py verl/single_controller/ray/base.py verl/workers/fsdp_workers.py verl/workers/megatron_workers.py verl/workers/engine_workers.py tests/unit/test_teacher_config.py tests/unit/test_teacher_workers.py tests/unit/test_teacher_routing.py tests/unit/test_mopd_advantage.py tests/unit/test_mopd_trainer_runtime.py tests/unit/test_mopd_resource_pools.py tests/integration/test_mopd_e2e.py
    # Result: All checks passed

    python -m compileall verl/workers/config/teacher.py verl/trainer/main_ppo.py verl/trainer/ppo/ray_trainer.py verl/single_controller/ray/base.py verl/workers/fsdp_workers.py verl/workers/megatron_workers.py verl/workers/engine_workers.py
    # Result: exit code 0

YAML was intentionally excluded from Ruff because `ruff check verl/trainer/config/algorithm/mopd.yaml` mis-parses Hydra YAML as Python and reports false `F821` errors on values such as `false`, `true`, and `null`.

Representative passing checks from the final run:

    tests/unit/test_teacher_routing.py::test_teacher_log_prob_preserves_sample_alignment_after_balancing PASSED
    tests/unit/test_teacher_routing.py::test_teacher_log_prob_async_overlaps_different_pools_but_serializes_same_pool PASSED
    tests/unit/test_mopd_trainer_runtime.py::test_validate_tokenizer_compatibility_checks_base_model_even_with_student_tokenizer_path PASSED
    tests/unit/test_mopd_trainer_runtime.py::test_run_mopd_preflight_rejects_unknown_teacher_ids PASSED
    tests/unit/test_mopd_resource_pools.py::test_init_resource_pool_mgr_adds_teacher_pools_and_dynamic_colocate_capacity PASSED

## Interfaces and Dependencies

At the end of this round, the following interfaces must exist:

In `verl/workers/config/teacher.py`, define a teacher-pool config structure that can be instantiated from Hydra and referenced by `MOPDConfig.resource_pools`. It must capture:

    n_gpus_per_node: int
    nnodes: int
    max_colocate_count: int | None

In `verl/trainer/main_ppo.py`, `TaskRunner.init_resource_pool_mgr()` must build a `resource_pool_spec` that accepts both legacy entries shaped like:

    "global_pool": [8, 8]

and richer entries shaped like:

    "code_pool": {
        "process_on_nodes": [4, 4],
        "max_colocate_count": 2,
    }

In `verl/single_controller/ray/base.py`, `ResourcePoolManager.create_resource_pool()` must parse both shapes without breaking existing callers.

In the worker classes (`verl/workers/fsdp_workers.py`, `verl/workers/megatron_workers.py`, and any needed engine worker), define a non-blocking ref-log-prob method equivalent to:

    @register(..., blocking=False)
    def compute_ref_log_prob_async(...): ...

In `verl/trainer/ppo/ray_trainer.py`, `RayPPOTrainer` must expose helper-level behavior equivalent to:

    def _compute_teacher_log_probs(self, batch: DataProto) -> torch.Tensor: ...
    def _compute_teacher_log_probs_async(self, batch: DataProto) -> torch.Tensor: ...
    def _record_mopd_teacher_metrics(self, batch: DataProto, metrics: dict[str, float]) -> None: ...

The trainer must keep the existing public batch contract (`teacher_log_prob` stays a tensor on the batch) so downstream advantage code does not change shape.

Update note: created this ExecPlan on 2026-03-13 to cover the `solution-review` P1 scope and the newly reported high-severity review blockers on the same MOPD runtime path.
