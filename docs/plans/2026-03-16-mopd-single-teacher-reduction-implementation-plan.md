# MOPD Single-Teacher Reduction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prove inside this repository that single-teacher MOPD reduces to pure on-policy reverse-KL distillation when ORM, sequence-teacher mixing, and rollout correction are disabled.

**Architecture:** Keep the current trainer and teacher-routing runtime unchanged, add one independent baseline advantage estimator, and add a reduction harness that runs both modes with matched configuration so the comparison isolates algorithm semantics instead of infrastructure differences. The proof has two layers: exact tensor-level equivalence in tests and a reproducible recipe-level pair run for training-curve and final-metric inspection.

**Tech Stack:** Python 3.10+, PyTorch, Ray, Hydra, pytest, existing verl PPO trainer and MOPD recipe scripts.

---

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan must be maintained in accordance with [PLANS.md](/home/scbjtfy/verl/.worktrees/mopd-implementation/PLANS.md).

## Purpose / Big Picture

After this change, a contributor can run one repository-local experiment that
compares reduced single-teacher `mopd` against an independent reverse-KL
baseline using the same teacher, the same data, and the same rollout budget.
They can also run unit tests that prove the reduction exactly at tensor level.
This matters because the current branch demonstrates that MOPD runs, but it
does not yet demonstrate that the reduced case is the intended algorithm rather
than an accidentally entangled variant.

## Progress

- [x] (2026-03-16 17:10 CST) Read the current MOPD docs, tests, trainer code, and the companion G-OPD analysis to identify the missing proof surface.
- [x] (2026-03-16 17:18 CST) Wrote the design document at `docs/plans/2026-03-16-mopd-single-teacher-reduction-design.md`.
- [x] (2026-03-16 17:27 CST) Wrote failing unit tests for the independent baseline estimator, exact reduced-MOPD equivalence, and the reduction harness loader/command builder.
- [x] (2026-03-16 17:34 CST) Implemented `single_teacher_reverse_kl` in `verl/trainer/ppo/core_algos.py` and verified the new tests turned green.
- [x] (2026-03-16 17:36 CST) Added `recipe/mopd/run_single_teacher_reduction.py` with paired command generation, sequential run support, and console-metric parsing helpers.
- [x] (2026-03-16 17:39 CST) Updated `recipe/mopd/README.md` to document the reduction harness and success criteria.
- [x] (2026-03-16 17:43 CST) Ran targeted pytest commands and a dry-run harness check; recorded evidence below.
- [x] (2026-03-16 18:09 CST) Requested independent code review, fixed the reported runtime gating and zero-placeholder issues, and closed the remaining README consistency nit.

## Surprises & Discoveries

- Observation: the current branch already emits per-teacher reverse-KL metrics in `RayPPOTrainer`, so the reduction experiment does not need a new trainer-level metric family.
  Evidence: `verl/trainer/ppo/ray_trainer.py` records `mopd/<teacher>/reverse_kl_mean`.

- Observation: the existing test surface covers formula fragments for MOPD but not an independent single-teacher baseline or a reduction proof.
  Evidence: `tests/unit/test_mopd_advantage.py` verifies MOPD, ExOPD, ORM, sequence-teacher, and IS pieces, but no test names mention `single_teacher_reverse_kl` or reduction.

- Observation: the production console logger prints stable `step:<n> - key:value` metric lines, including MOPD teacher metrics and validation summaries, so a lightweight parser is sufficient for the reduction harness.
  Evidence: a real run log under `/gpfs/Mamba/Project/Single_Cell/Training/MOPD-longrun-closure-20260315-e3/console.log` contains keys such as `mopd/cell_type_teacher/reverse_kl_mean`, `critic/advantages/mean`, and `training/global_step` on the same line.

## Decision Log

- Decision: use an in-repo independent baseline estimator instead of treating the external G-OPD repository as the primary oracle.
  Rationale: the user asked for proof that the current implementation is the right algorithm. Reusing the same trainer/runtime while changing only the estimator makes any agreement or disagreement easier to interpret.
  Date/Author: 2026-03-16 / Codex

- Decision: keep the reduction proof two-layered, with exact tensor-level tests and a recipe-level pair run.
  Rationale: unit tests catch logic regressions quickly, while the pair run addresses the user’s explicit requirement about curves, teacher-gap signals, and final scores.
  Date/Author: 2026-03-16 / Codex

## Outcomes & Retrospective

The repository now contains both layers of the reduction proof scaffold.

At the code level, `single_teacher_reverse_kl` is an independent estimator that
computes the plain reverse-KL signal and rejects MOPD-only extras such as ORM,
sequence-teacher rewards, base normalization, and rollout IS correction. The
new tests prove that reduced MOPD matches this baseline exactly on the same
batch tensors.

At the recipe level, `recipe/mopd/run_single_teacher_reduction.py` builds two
matched commands that differ only in `algorithm.adv_estimator` and output
directory. The harness can optionally run both commands and summarize the final
tracked metrics from console logs.

What remains is external review and, when desired, an actual GPU pair run to
collect training curves and final scores under the new harness. The independent
review found one real blocker in the initial implementation: trainer-injected
zero-valued sequence placeholders caused the baseline estimator to reject real
runtime batches. The follow-up revision fixed that and also promoted the
baseline onto the same teacher-runtime gating path as `mopd`.

## Context and Orientation

The current MOPD runtime lives in two main files.

`verl/trainer/ppo/core_algos.py` contains registered advantage estimators. The
function `compute_mopd_advantage()` computes the token-level teacher advantage
and may also add sequence-level teacher rewards, ORM advantages, and rollout
importance-sampling weights. In plain language, an “advantage estimator” is the
function that turns batch tensors such as rewards and log probabilities into the
training signal consumed by the actor loss.

`verl/trainer/ppo/ray_trainer.py` contains `compute_advantage()`, which is the
driver-side dispatcher that picks an estimator and forwards tensors from
`DataProto` batches. The same file also logs MOPD teacher metrics through
`RayPPOTrainer._record_mopd_teacher_metrics()`.

The relevant tests already live in `tests/unit/test_mopd_advantage.py` and
`tests/unit/test_mopd_trainer_runtime.py`. The MOPD runtime recipes live under
`recipe/mopd/`, including smoke scripts and the preflight utility
`recipe/mopd/check_mopd_first_batch.py`.

For this work, “single-teacher reduction” means all of the following are true at
the same time:

- there is exactly one teacher model
- the tokenizer path is compatible, so the token-level teacher log-prob path is used
- `algorithm.mopd.orm_weight == 0.0`
- `algorithm.mopd.is_correction == false`
- no sequence-teacher reward path is active
- no base normalization path is active

Under those conditions, the intended advantage is simply:

    teacher_log_prob - old_log_probs

That is the reverse-KL distillation signal.

## Plan of Work

First, add failing tests in `tests/unit/test_mopd_advantage.py`. One test will
call a new baseline estimator directly and assert it returns
`teacher_log_prob - old_log_probs` masked by `response_mask`. Another test will
call both the new baseline estimator and `compute_mopd_advantage()` on the same
batch with all extra MOPD branches disabled and assert exact equality. Keep
these tests fully tensor-local so they run on CPU.

Second, implement the baseline estimator in `verl/trainer/ppo/core_algos.py`.
The function should be minimal: accept `teacher_log_prob`, `old_log_probs`, and
`response_mask`, return `advantages` and `returns`, and ignore MOPD-only extras.
The estimator should be registered with a distinct name so it can be selected
through `algorithm.adv_estimator` in a runtime config.

Third, update `verl/trainer/ppo/ray_trainer.py::compute_advantage()` only as
needed so the new estimator receives the already-existing batch fields. Avoid
special MOPD-only assumptions so the baseline can reuse the same batch plumbing.

Fourth, add a recipe-level harness in `recipe/mopd/`. The harness should build
two matched command lines, one for reduced MOPD and one for the baseline. Both
must share the same student model, teacher model, train/validation files,
response length, rollout count, and optimization settings. The harness should
launch the runs separately, collect their logs, and print a compact comparison
summary for stable scalar metrics. Use existing smoke/preflight patterns rather
than inventing a new runtime entrypoint.

Fifth, update `recipe/mopd/README.md` or `recipe/mopd/README_SMOKE_TEST.md` so
the reduction experiment has explicit instructions and success criteria.

Sixth, verify the change with targeted pytest and at least one lightweight
recipe/harness check. If a full GPU pair run is not feasible in this session,
the repository should still contain the exact command needed and the harness
must pass its local non-training checks.

## Concrete Steps

Run all commands from `/home/scbjtfy/verl/.worktrees/mopd-implementation`.

1. Write the failing unit tests.

   Run:

       pytest -q tests/unit/test_mopd_advantage.py -k "single_teacher or reduction"

   Expected before implementation:

       FAIL because the new estimator name is unknown or the new tests are absent.

2. Implement the estimator and dispatch plumbing.

   Run:

       pytest -q tests/unit/test_mopd_advantage.py -k "single_teacher or reduction"

   Expected after implementation:

       PASS for the new baseline and reduction-equivalence tests.

3. Add the recipe harness and its lightweight tests or parser checks.

   Run:

       pytest -q tests/unit/test_mopd_run_script.py

   Expected:

       PASS with coverage for the new reduction harness command generation or log parsing.

4. Run the focused MOPD CPU-safe test slice that covers the changed areas.

   Run:

       pytest -q \
         tests/unit/test_mopd_advantage.py \
         tests/unit/test_mopd_run_script.py \
         tests/unit/test_mopd_trainer_runtime.py

   Expected:

       All selected tests pass.

5. If the environment permits, run the new reduction harness in dry-run or
   command-print mode so the paired commands and output directories can be
   inspected without a long GPU job.

   Run:

       python recipe/mopd/run_single_teacher_reduction.py --dry-run

   Expected:

       Two clearly labeled commands, one for `mopd` and one for the baseline,
       with matched data/model settings and only the intended estimator-specific
       toggles differing.

## Validation and Acceptance

Acceptance is met when all of the following are true.

First, the unit tests prove exact tensor equality between reduced MOPD and the
independent baseline on the same batch. The new tests must fail before the code
change and pass after it.

Second, the repository exposes a reproducible reduction harness that a human can
run to compare a reduced single-teacher MOPD run against the baseline run. The
dry-run output must show matched configuration, and the documentation must state
that the expected outcome is statistical agreement in training curves,
teacher-gap or reverse-KL summaries, and final reward/score rather than bitwise
identity.

Third, the existing changed-area tests remain green, showing that no current
MOPD behavior regressed.

## Idempotence and Recovery

The unit tests and dry-run harness commands are safe to rerun. The runtime
harness should write each paired experiment into a dedicated output directory so
rerunning it does not silently overwrite an earlier comparison unless the user
chooses the same path explicitly. No existing checkpoints or recipe outputs
should be deleted as part of this work.

## Artifacts and Notes

Targeted pytest after implementation:

    pytest -q tests/unit/test_mopd_advantage.py -k "single_teacher or reverse_kl_baseline or reduction"
    3 passed, 15 deselected, 1 warning in 7.96s

Harness unit tests:

    pytest -q tests/unit/test_mopd_single_teacher_reduction.py
    2 passed in 0.03s

Changed-area regression slice:

    pytest -q tests/unit/test_mopd_advantage.py tests/unit/test_mopd_single_teacher_reduction.py tests/unit/test_mopd_run_script.py tests/unit/test_mopd_trainer_runtime.py
    58 passed, 1 warning in 8.52s

Reduction harness dry-run:

    python recipe/mopd/run_single_teacher_reduction.py --dry-run

    The script prints two matched `python -m verl.trainer.main_ppo` commands:
    one with `algorithm.adv_estimator=mopd` and one with
    `algorithm.adv_estimator=single_teacher_reverse_kl`. Both commands share
    the same student, teacher, data files, rollout count, and `algorithm.mopd`
    reduction settings.

## Interfaces and Dependencies

In `verl/trainer/ppo/core_algos.py`, define a new registered estimator with a
stable name such as `single_teacher_reverse_kl`. It must accept the same batch
tensor names already produced by the trainer for MOPD-compatible teachers and it
must return the standard two-tuple:

    tuple[torch.Tensor, torch.Tensor]

where the first tensor is `advantages` and the second tensor is `returns`.

In `recipe/mopd/`, add a Python entrypoint that builds and optionally runs the
two matched training commands. Reuse the current `python -m verl.trainer.main_ppo`
entrypoint and reuse the same data and teacher configuration style as the
existing smoke or preflight scripts.

Revision note: created this ExecPlan on 2026-03-16 to support a correctness-oriented
single-teacher reduction proof requested by the user. The first revision records
current context and the intended two-layer validation strategy before code changes begin.

Revision note: updated on 2026-03-16 after implementation to record the added
estimator, harness, README changes, and fresh validation evidence.
