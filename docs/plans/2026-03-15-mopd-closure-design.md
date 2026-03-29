# MOPD Closure Design

**Date:** 2026-03-15

## Context

Three user-reported MOPD risks are real on the current worktree:

1. `RayPPOTrainer.fit()` finalization is not protected by a `finally`, so exceptions can bypass
   trainer finalization entirely.
2. `cleanup_teacher_workers()` only clears `teacher_wgs`; it does not symmetrically release
   `base_policy_wg`.
3. `TeacherConfig.weight` and `TeacherConfig.base_model_path` exist in schema but are not consumed by
   the runtime. `MOPDConfig` also exists without end-to-end typed validation through `AlgoConfig` and
   `validate_config()`.

## Goals

- Make MOPD worker lifecycle cleanup deterministic on success, `val_only`, and exception paths.
- Reject unsupported teacher fields before runtime instead of silently accepting misleading config.
- Restore typed validation coverage for `algorithm.mopd` through the standard config validation path.

## Non-Goals

- Implement runtime weighted aggregation across multiple teachers.
- Implement per-teacher base-model normalization.
- Change MOPD routing semantics away from one teacher per sample.

## Options

### Option A: Closure Fixes + Fail-Fast Validation

- Add `try/finally` around `RayPPOTrainer.fit()`.
- Extend cleanup to release `base_policy_wg` as well as `teacher_wgs`.
- Wire `MOPDConfig` into `AlgoConfig` and instantiate `config.algorithm` in `validate_config()`.
- Fail fast when unsupported ghost fields are configured.

Pros:
- Matches current paper/runtime semantics.
- Removes silent misconfiguration.
- Smallest safe change set.

Cons:
- Keeps deprecated schema fields around for compatibility.

### Option B: Implement Ghost Fields as Real Runtime Features

- Add weighted multi-teacher aggregation.
- Add per-teacher base normalization workers.

Pros:
- Makes schema fully literal.

Cons:
- Changes algorithm semantics.
- High-risk multi-file refactor outside the audited bug scope.

### Option C: Documentation-Only Cleanup

- Update docs and comments without runtime changes.

Pros:
- Smallest code change.

Cons:
- Leaves confirmed lifecycle leaks and exception-path cleanup gaps in place.

## Decision

Choose **Option A**.

It fixes the confirmed bugs, preserves current MOPD semantics from the paper notes, and avoids
quietly accepting unsupported configuration that users could mistake for working features.

## Test Strategy

- Add unit coverage for `cleanup_teacher_workers()` symmetry and idempotence.
- Add unit coverage that `fit()` finalizes resources when an exception is raised after tracking starts.
- Add config tests proving `algorithm.mopd` is typed via `AlgoConfig`.
- Add config tests proving unsupported `TeacherConfig.weight != 1.0` and per-teacher
  `base_model_path` are rejected.
