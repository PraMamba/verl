# MOPD Single-Teacher Reduction Design

## Purpose

The current worktree proves that the MOPD estimator can run and that individual
formula components behave as expected in isolation, but it does not yet prove
the stronger claim that matters for algorithm credibility: when MOPD is reduced
to a single compatible-tokenizer teacher with ORM disabled and no rollout
correction, the system behaves like a standard on-policy reverse-KL
distillation objective rather than a nearby approximation.

This design adds an independent single-teacher reverse-KL baseline and a
reduction experiment harness so that the repository can answer one concrete
question with evidence: do single-teacher `mopd` and pure reverse-KL training
match up to normal training noise when all extra MOPD branches are disabled?

## Context

The current implementation already computes the single-teacher token signal
inside `verl/trainer/ppo/core_algos.py` as:

- standard MOPD: `teacher_log_prob - old_log_probs`
- ExOPD: base-normalized variant

But that logic is embedded in a richer estimator that may also mix:

- sequence-level teacher rewards for incompatible tokenizers
- ORM advantages
- rollout importance sampling correction

The trainer also records MOPD-specific metrics in
`verl/trainer/ppo/ray_trainer.py`, including
`mopd/<teacher>/reverse_kl_mean`, which is useful for a reduction experiment.

The missing piece is an independent baseline path that uses the same trainer and
policy-loss stack but does not reuse the MOPD estimator implementation.

## Design Goals

1. Make the reduction claim falsifiable inside this repository.
2. Keep the proof local to verl rather than depending on an external G-OPD
   runtime that has different control flow and configuration semantics.
3. Reuse the same trainer, rollout, and logging surface on both sides of the
   comparison so the only intended difference is the advantage estimator.
4. Preserve current MOPD runtime behavior for multi-teacher, sequence-reward,
   ORM, and IS-correction paths.

## Options Considered

### Option A: Independent baseline estimator plus comparison recipe

Add a new advantage estimator that computes pure single-teacher reverse-KL
distillation directly from `teacher_log_prob` and `old_log_probs`, with no ORM,
no sequence-teacher branch, no ExOPD, and no rollout correction. Then add a
recipe utility that runs:

- single-teacher `mopd` in its reduced configuration
- the new baseline estimator with the same single teacher, same data, and same
  rollout settings

This option gives the strongest in-repo evidence because both runs share the
same trainer/runtime surface while using independent advantage code.

### Option B: Unit-test-only equivalence

Add tests proving that `compute_mopd_advantage()` equals
`teacher_log_prob - old_log_probs` under the reduction preconditions.

This proves the estimator formula, but it does not prove that the trainer path
does not accidentally mix in routing, metrics, or future batch plumbing.

### Option C: External G-OPD comparison only

Use `/home/scbjtfy/G-OPD` as the sole baseline implementation.

This is useful as a conceptual cross-check, but it is weaker as the main proof
because the external code has a different architecture and a different
advantage-overwrite path. Matching or diverging there is harder to interpret.

## Chosen Design

Use Option A as the main deliverable and Option B as the low-level guardrail.

The implementation will add:

1. A new advantage estimator, tentatively named `single_teacher_reverse_kl`,
   registered in `verl/trainer/ppo/core_algos.py`.
2. Dispatch plumbing in `verl/trainer/ppo/ray_trainer.py::compute_advantage()`
   so the new estimator can consume the same teacher tensors that MOPD already
   uses.
3. Targeted unit tests that prove:
   - the new baseline returns `teacher_log_prob - old_log_probs`
   - reduced single-teacher MOPD exactly matches the baseline at advantage level
   - MOPD-only branches remain disabled in the reduction case
4. A recipe-level reduction harness under `recipe/mopd/` that can launch a pair
   of runs with matched settings and compare the most relevant logged signals.
5. Documentation describing the reduction contract, how to run it, and what
   outcome counts as success.

## What Counts as Success

At unit-test level:

- reduced MOPD and the independent baseline produce equal advantages on the same
  tensors

At trainer/runtime level:

- the baseline run and reduced-MOPD run use the same teacher, dataset, rollout
  count, and student configuration
- the comparison utility can show side-by-side summaries of:
  - `critic/advantages/mean`
  - `mopd/<teacher>/reverse_kl_mean` for the MOPD run
  - matching baseline reverse-KL statistics
  - final validation or reward metrics when present

The experiment is considered supportive if these traces are statistically close
within ordinary run-to-run noise, not if they are bitwise identical.

## Non-Goals

- Changing multi-teacher routing semantics
- Adding incompatible-tokenizer token-level distillation
- Reworking ORM or sequence-teacher behavior
- Replacing the current MOPD estimator

## Risks and Mitigations

Risk: the baseline accidentally reuses MOPD logic and becomes a tautology.

Mitigation: keep the baseline implementation minimal and isolated, and test it
against direct tensor arithmetic.

Risk: recipe comparison is too brittle because logs vary by environment.

Mitigation: compare a small set of stable scalar metrics and treat the recipe
tool as a harness for human review, not as a hard pass/fail CI gate.

## Approval Assumption

The user request explicitly asked for a correctness-oriented reduction proof.
This design assumes that the preferred path is an in-repo independent baseline
plus a reproducible reduction harness unless the user later asks to pivot to an
external G-OPD-only comparison.
