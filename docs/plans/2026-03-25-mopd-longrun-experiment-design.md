# MOPD Long-Run Experiment Design

**Date:** 2026-03-25

## Goal

Turn the current MOPD recipe into an experiment-grade protocol that can support a real
claim about multi-teacher learning quality, not just runtime closure.

The target claim is narrower and stricter than "the job runs":

- the student improves on the cell-type domain
- the student improves on the disease-state domain
- the student does not collapse one domain while fitting the other
- the comparison is reproducible across reruns and baseline pairings

## Current State

The existing recipe at `recipe/mopd/run_mopd_qwen3_4b.sh` already proves several useful
things:

- standard `verl.trainer.main_ppo` integration works
- `algorithm.mopd.teachers[]` is wired into the trainer runtime
- per-sample `teacher_id` routing works
- the dual-teacher single-node 4-GPU path can complete preflight, E2E, short full runs,
  and a conservative long-run rerun

But it is still a runtime recipe, not an experiment protocol.

The main gaps are:

1. validation is tied to `zero_reward.py`, so `val-*` metrics are not meaningful
2. the dataset assigns exactly one `teacher_id` per sample, so this proves routed
   distillation on a merged dataset, not same-sample multi-teacher fusion
3. both teachers are placed on `global_pool`, so same-pool serialization and shared-GPU
   contention contaminate long-run throughput comparisons
4. the launcher is a large inline shell override, which is weak for N-teacher extension,
   provenance capture, and clean reruns
5. `resume_mode=auto` plus shared checkpoint roots is operationally convenient but weak
   for experiment provenance

## Recommended Approach

Keep the current MOPD runtime and replace the experiment surface around it.

This means:

- preserve the existing trainer-side `teacher_wgs`, `teacher_id` routing, and MOPD
  advantage logic
- replace smoke-style reward wiring with a real held-out evaluator
- expose explicit teacher resource-pool topology instead of forcing `global_pool`
- make experiment metadata first-class at launch time
- require paired baselines before interpreting long-run curves

## Proposed Experiment Surface

### 1. Training Objective

For the first experiment-grade version, keep the current routed MOPD objective:

- compatible-tokenizer teachers
- token-level teacher log-prob path
- `lambda_val=1.0`
- `orm_weight=0.0` unless a real outcome reward is ready

This keeps the training-side diff minimal and avoids mixing a new evaluator change with
 a new algorithm variant.

### 2. Validation Protocol

Validation must stop being "always zero reward".

The new validation protocol should produce:

- cell-type held-out metric
- disease-state held-out metric
- overall aggregate metric
- optional compositional slice if the dataset contains explicit composition examples

Validation should also be deterministic or at least controlled:

- either `do_sample=False`
- or `n > 1` with an explicitly chosen aggregation policy

### 3. Dataset Interpretation

The current dataset format remains valid for routed MOPD:

- each sample has one `teacher_id`
- the student sees a mixed stream of teacher-specific samples

But the experiment docs must stop over-claiming what this setup proves.

This protocol can prove:

- the student retains and/or improves across multiple routed domains

This protocol cannot alone prove:

- same-sample weighted fusion of multiple teacher opinions

### 4. Resource Topology

Teacher placement must be explicit.

The recipe should support:

- separate named `algorithm.mopd.resource_pools`
- teacher-to-pool assignment as part of the resolved experiment config
- logs that record the final pool topology and teacher manifest

For the 4-GPU single-node environment, the first clean target is not "maximize parallelism"
but "make the topology explicit and reproducible".

### 5. Provenance and Resume Policy

Each run must record:

- git SHA or dirty-worktree marker
- dataset snapshot identifier or path
- student checkpoint path
- teacher model paths and manifest
- teacher resource-pool assignment
- resolved Hydra command/config

Experiment runs should default to a fresh checkpoint directory. Resume should be opt-in,
not implicit.

### 6. Baseline Matrix

The long-run experiment is only interpretable with paired baselines:

- student-only continued training baseline
- cell-type single-teacher baseline
- disease-state single-teacher baseline
- dual-teacher MOPD run

Optional later extension:

- N-teacher routed run
- heterogeneous-tokenizer sequence-reward run

## Verification Gates

Before treating the new recipe as experiment-grade, require:

1. preflight success on the exact new launcher
2. one short real-evaluator smoke run
3. validation metrics that separate the two domains
4. one fresh long-run paired comparison against at least one single-teacher baseline
5. explicit acceptance thresholds for `mopd/is_valid_fraction` and other health metrics

## Non-Goals

This design does not try to solve:

- weighted multi-teacher fusion on the same sample
- fully parallel N-teacher scheduling on a small single-node machine
- artifact-self-contained checkpoint packaging for all teacher/base weights
- broad multi-node failure-matrix closure

Those remain valid follow-up projects, but they should not block turning the current
recipe into a scientifically usable long-run protocol.
