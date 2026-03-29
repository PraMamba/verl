# MOPD N-Teacher P2 Independent Teacher Backend And Sequence Distillation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents are available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `PLANS.md`.

**Goal:** Extend MOPD with the `solution-review` P2 scope by adding an independent teacher-only quantized inference backend and an explicit incompatible-tokenizer sequence-level distillation path, without modifying the existing FSDP reference-worker initialization semantics.

**Architecture:** Keep the current P1 MOPD token-level path intact for compatible-tokenizer teachers. Add a new dedicated teacher worker class for `hf_int8` and `hf_4bit` teachers, and add a parallel sequence-distillation path that operates on raw prompt messages plus decoded student responses to produce an independent `teacher_seq_reward` / `teacher_seq_advantage` signal. The trainer will route each teacher by both backend and tokenizer policy, then `compute_mopd_advantage()` will combine token-level MOPD, sequence-level teacher distillation, and ORM explicitly instead of collapsing them into one channel.

**Tech Stack:** Python, Hydra/OmegaConf, Ray worker groups, Hugging Face Transformers, BitsAndBytesConfig (optional runtime dependency for int8/4bit loading), verl `DataProto`, pytest, Ruff

---

## Purpose / Big Picture

After this change, a user can configure some MOPD teachers to run as inference-only quantized Hugging Face models instead of piggybacking on the current ref-worker lifecycle, which reduces the pressure to colocate every teacher with heavyweight training workers. In the same training run, a user can also declare tokenizer-incompatible teachers that score prompt-response text with their own tokenizer and contribute an explicit sequence-level teacher distillation signal rather than being blocked by the compatibility gate or incorrectly funneled through ORM. The observable proof is a CPU-only unit/integration test suite that fails before the change and passes after it, plus trainer/runtime tests that show backend selection, sequence-text reconstruction, and explicit estimator composition.

## Progress

- [x] (2026-03-14 00:58+08) Re-read the `solution-review` P2 section, `PLANS.md`, and the Superpowers process skills relevant to planning, TDD, and subagent-driven execution.
- [x] (2026-03-14 01:05+08) Finished codebase exploration of `verl/workers/config/teacher.py`, `verl/trainer/ppo/ray_trainer.py`, `verl/trainer/ppo/core_algos.py`, `verl/utils/dataset/rl_dataset.py`, and worker/distributed helpers to locate the cleanest P2 insertion points.
- [x] (2026-03-14 01:08+08) Settled the high-level design: add `teacher.backend` and `teacher.tokenizer_policy`, create a dedicated `HFQuantizedTeacherWorker`, reconstruct text from `raw_prompt + decoded response`, and combine sequence distillation explicitly in the MOPD estimator.
- [x] (2026-03-14 02:31+08) Added failing P2 tests for teacher backend/tokenizer policy fields, dedicated quantized worker presence, raw prompt preservation, response decoding, sequence teacher job construction, sequence reward plumbing, manifest coverage, and explicit estimator composition.
- [x] (2026-03-14 02:44+08) Implemented the dedicated teacher-only HF backend and trainer-side backend selection without altering the existing FSDP ref-worker init path.
- [x] (2026-03-14 02:44+08) Implemented incompatible-tokenizer sequence-level scoring and explicit `teacher_seq_reward` / `teacher_seq_advantage` composition in the MOPD trainer/estimator path.
- [x] (2026-03-14 02:44+08) Extended checkpoint manifest, tokenizer preflight validation, and teacher metrics to cover P2 fields.
- [x] (2026-03-14 13:02+08) Finished focused dual review and follow-up fixes: pure `sequence_reward` fleets no longer route through token log-prob, sequence-teacher jobs now respect teacher DP sizing, and sequence teacher scoring now honors per-teacher micro-batch limits.
- [x] (2026-03-14 13:02+08) Re-ran targeted pytest, Ruff, and compileall after the review-driven fixes; the P2 verification set is green.

## Surprises & Discoveries

- Observation: the current trainer already stores enough text context for sequence-level distillation.
  Evidence: `RLHFDataset.__getitem__()` writes `raw_prompt` into `non_tensor_batch` in `verl/utils/dataset/rl_dataset.py`, and `RayPPOTrainer` already keeps a tokenizer for decoding generated responses in `verl/trainer/ppo/ray_trainer.py`.
- Observation: the cleanest seam for a P2 teacher backend is not in `TaskRunner` or role mapping, but inside the explicit teacher-worker loop in `RayPPOTrainer.init_workers()`.
  Evidence: P1 teacher workers are instantiated manually there today, so backend-specific branching can stay local to MOPD instead of perturbing actor/ref bootstrap for the rest of PPO.
- Observation: the estimator already has a stable extension point for extra teacher signals because `compute_mopd_advantage()` is only used through the trainer-side `compute_advantage()` wrapper.
  Evidence: `compute_advantage()` in `verl/trainer/ppo/ray_trainer.py` already forwards MOPD-specific keyword arguments such as `teacher_log_prob`, `base_log_prob`, `lambda_val`, and `orm_weight`.
- Observation: review explicitly forbids two tempting shortcuts that would otherwise shrink the patch.
  Evidence: the `solution-review` P2 section says not to stuff `8bit + device_map="auto"` into the existing FSDP ref worker and not to disguise incompatible-tokenizer teacher scores as ORM.

## Decision Log

- Decision: Introduce `teacher.backend: legacy_ref | hf_int8 | hf_4bit` on each teacher instead of adding a global MOPD backend toggle.
  Rationale: P2 is teacher-only and mixed teacher fleets are a core use case; per-teacher backend selection keeps compatible legacy teachers on the current path while letting specific teachers opt into the new inference-only worker.
  Date/Author: 2026-03-14 / Codex
- Decision: Introduce `teacher.tokenizer_policy: compatible | sequence_reward` and make incompatible-tokenizer support opt-in rather than implicit.
  Rationale: This preserves the P1 compatibility gate by default and forces users to declare when they want the new sequence-distillation semantics.
  Date/Author: 2026-03-14 / Codex
- Decision: Implement the new backend as a dedicated worker module under `verl/workers/teacher_workers.py`.
  Rationale: This isolates quantized HF loading and inference-only logic from the existing ref workers, which keeps the P2 patch aligned with the review guidance and avoids widening the blast radius in FSDP/Megatron code.
  Date/Author: 2026-03-14 / Codex
- Decision: Model the incompatible-tokenizer signal as `teacher_seq_reward` plus an explicit estimator-side `teacher_seq_advantage`, not as ORM.
  Rationale: This preserves semantic separation between teacher distillation and reward-model outcome supervision, which the review called out as a critical correctness requirement.
  Date/Author: 2026-03-14 / Codex
- Decision: Normalize sequence-level teacher rewards in a dedicated helper instead of reusing the GRPO/ORM helper as-is.
  Rationale: Reusing the ORM path would blur the distinction the review asked to preserve. A dedicated helper keeps the math and metrics explicitly labeled as teacher sequence distillation.
  Date/Author: 2026-03-14 / Codex
- Decision: `tokenizer_policy=sequence_reward` is currently supported only through the dedicated quantized teacher backend, not the legacy ref-worker backend.
  Rationale: the P2 worker addition includes `compute_seq_scores`, while the legacy ref worker still only implements token-level `compute_ref_log_prob`. Failing fast here keeps the contract explicit instead of silently routing sequence teachers onto an unsupported path.
  Date/Author: 2026-03-14 / Codex

## Outcomes & Retrospective

P2 is no longer design-only in this worktree. `TeacherConfig` now supports per-teacher backend selection and explicit tokenizer policy, `verl/workers/teacher_workers.py` provides a dedicated inference-only quantized teacher worker, `RayPPOTrainer` routes compatible teachers through token-level log-prob and sequence teachers through `raw_prompt + decoded response text`, and `compute_mopd_advantage()` explicitly mixes token-level MOPD, sequence-level teacher advantage, and ORM without aliasing sequence signals into ORM.

The focused review loop surfaced three real follow-up issues, all of which are now addressed in-tree: pure `sequence_reward` teacher fleets were incorrectly falling back into the compatible token-logprob path, sequence-teacher jobs were not respecting the teacher DP mesh for padding/balancing, and `HFQuantizedTeacherWorker.compute_seq_scores()` ignored per-teacher micro-batch sizing. Those fixes landed with dedicated regression tests before the final verification sweep.

The main residual limitation is deliberate: `sequence_reward` is not wired onto the legacy ref worker. That backend split keeps the P2 patch aligned with the review's ban on pushing new quantized inference semantics into the existing FSDP ref initialization path.

## Context and Orientation

P2 builds directly on the P1 MOPD runtime already present in this worktree. The relevant files span four areas.

`verl/workers/config/teacher.py` defines the typed MOPD config. Today a teacher has routing, tokenizer, and lambda settings, but no independent backend selection and no tokenizer-policy declaration for sequence distillation.

`verl/trainer/ppo/ray_trainer.py` owns the MOPD runtime. It builds teacher worker configs, initializes teacher worker groups, routes each sample by `teacher_id`, computes `teacher_log_prob`, and forwards those tensors into `compute_advantage()`. This is also where the trainer still enforces tokenizer compatibility and where new sequence-level scoring must be attached to the batch before the advantage step.

`verl/trainer/ppo/core_algos.py` defines the actual MOPD estimator. Right now `compute_mopd_advantage()` only understands token-level teacher log-prob, optional base normalization, and optional ORM mixing. P2 must extend this file so sequence-level teacher distillation remains mathematically explicit and separately named.

`verl/utils/dataset/rl_dataset.py` is the key reason P2 is feasible without reworking the entire data pipeline. Each sample already carries `raw_prompt`, which is the message list needed to re-render prompt text on a teacher’s own tokenizer. `RayPPOTrainer` also holds the student tokenizer, so generated `responses` can be decoded back to text on the driver before sending text-based jobs to sequence teachers.

For worker infrastructure, the new dedicated backend should follow the same single-controller worker contract used elsewhere in `verl/workers/`. A “worker” here is a Ray-executed Python object that registers methods through `@register(...)` so the driver can dispatch `DataProto` batches across one or more GPUs. The new teacher worker only needs inference-only methods and must not join the existing actor/ref FSDP lifecycle.

## Plan of Work

The first milestone is red-test lock-in of the P2 surface. Extend `tests/unit/test_teacher_config.py` and `tests/unit/test_teacher_workers.py` so they fail until `TeacherConfig` supports backend and tokenizer-policy declarations, until the trainer can build a non-legacy teacher worker config, and until invalid combinations are rejected. Add new trainer/runtime tests in `tests/unit/test_mopd_trainer_runtime.py` for text reconstruction from `raw_prompt + responses`, sequence teacher reward collection, checkpoint manifest coverage for new fields, and explicit composition in `compute_mopd_advantage()`. Extend `tests/unit/test_mopd_advantage.py` with failing assertions that the estimator combines `teacher_seq_advantage` independently of ORM.

The second milestone is the new teacher-only backend. Add `verl/workers/teacher_workers.py` with a dedicated HF quantized worker that loads `AutoModelForCausalLM` rank-locally, uses `BitsAndBytesConfig` for `hf_int8` or `hf_4bit`, and exposes `init_model()`, `compute_ref_log_prob()`, `compute_ref_log_prob_async()`, `compute_seq_scores()`, and `compute_seq_scores_async()`. The log-prob path should reuse `verl.utils.torch_functional.log_probs_from_logits_response()`. The sequence-score path should build teacher-local prompt/response text from message lists plus response text, tokenize on the teacher side, run a forward pass, and return a scalar sequence score per sample. For this first P2 implementation, use normalized mean response log-prob as the scalar sequence reward so scores are comparable across varying response lengths without pretending to be token-aligned.

The third milestone is trainer integration. In `verl/trainer/ppo/ray_trainer.py`, branch teacher initialization on `teacher.backend`. Keep `legacy_ref` on the P1 path, and instantiate the new quantized worker directly for `hf_int8` / `hf_4bit`. Add helpers that split teachers into token-level and sequence-level jobs, decode student responses back to text, and collect either `teacher_log_prob` or `teacher_seq_reward` depending on `teacher.tokenizer_policy`. Preserve the existing routed-teacher batch contract: token teachers still populate `teacher_log_prob`; sequence teachers instead populate `teacher_seq_reward`, `teacher_seq_weight`, and an explicit token-vs-sequence mask so the estimator can combine signals samplewise.

The fourth milestone is estimator and manifest integration. Extend `compute_mopd_advantage()` in `verl/trainer/ppo/core_algos.py` with an explicit sequence-distillation branch. Add a dedicated helper that turns `teacher_seq_reward` into `teacher_seq_advantage` by normalizing active sequence-teacher rewards and broadcasting them across response tokens. Compose the final MOPD advantage as token-level MOPD plus `teacher_seq_weight * teacher_seq_advantage`, then optionally add `orm_weight * A_orm`; do not rename or reroute any sequence teacher values into the ORM channel. Update the MOPD manifest in `ray_trainer.py` so checkpoints record `backend`, `tokenizer_policy`, and `seq_reward_weight`, and extend trainer metrics to expose per-teacher sequence reward diagnostics when available.

The final milestone is verification and review. Run the smallest red/green test slices first, then the targeted MOPD suites, then Ruff and compileall. Request focused code review on the new worker, trainer routing, and estimator composition, and fix any valid findings before calling the round complete.

## Concrete Steps

All commands below run from `/home/scbjtfy/verl/.worktrees/mopd-implementation`.

1. Write the failing P2 config/runtime tests first.

       pytest tests/unit/test_teacher_config.py tests/unit/test_teacher_workers.py tests/unit/test_mopd_trainer_runtime.py tests/unit/test_mopd_advantage.py -k "backend or tokenizer_policy or sequence or quantized or seq_reward" -v

   Expected before implementation: failures about missing teacher config fields, missing quantized worker selection, missing text reconstruction helpers, and missing sequence-distillation estimator support.

2. After adding the new worker tests, run the most focused slice for the dedicated backend.

       pytest tests/unit/test_teacher_workers.py -k "quantized or backend" -v

   Expected before implementation: failure because `teacher.backend` is unknown and the trainer cannot instantiate a non-legacy teacher worker.

3. Implement the new worker and trainer selection incrementally, rerunning the smallest failing tests after each step.

       pytest tests/unit/test_teacher_config.py -k "backend or tokenizer_policy" -v
       pytest tests/unit/test_teacher_workers.py -k "backend or quantized" -v
       pytest tests/unit/test_mopd_trainer_runtime.py -k "sequence or text or manifest" -v
       pytest tests/unit/test_mopd_advantage.py -k "sequence" -v

4. Run the full targeted P2 verification on the final diff.

       pytest tests/unit/test_teacher_config.py tests/unit/test_teacher_workers.py tests/unit/test_teacher_routing.py tests/unit/test_mopd_advantage.py tests/unit/test_mopd_trainer_runtime.py tests/unit/test_mopd_resource_pools.py tests/integration/test_mopd_e2e.py -v
       ruff check verl/workers/config/teacher.py verl/workers/teacher_workers.py verl/trainer/main_ppo.py verl/trainer/ppo/core_algos.py verl/trainer/ppo/ray_trainer.py tests/unit/test_teacher_config.py tests/unit/test_teacher_workers.py tests/unit/test_mopd_advantage.py tests/unit/test_mopd_trainer_runtime.py tests/integration/test_mopd_e2e.py
       python -m compileall verl/workers/config/teacher.py verl/workers/teacher_workers.py verl/trainer/ppo/core_algos.py verl/trainer/ppo/ray_trainer.py

5. Request final review on the P2 diff and fix any valid findings before claiming completion.

## Validation and Acceptance

This P2 round is acceptable only if all of the following are true:

1. A teacher can declare `backend: hf_int8` or `backend: hf_4bit`, and the trainer initializes a dedicated inference-only teacher worker instead of the existing ref worker path.
2. The P2 implementation never inserts `device_map="auto"` or bitsandbytes loading logic into the existing FSDP ref worker initialization path.
3. A teacher can declare `tokenizer_policy: sequence_reward`, and the trainer then uses `raw_prompt + decoded response text` to request a scalar sequence reward from that teacher.
4. The trainer records sequence distillation in explicit batch fields such as `teacher_seq_reward` / `teacher_seq_weight` instead of remapping them into ORM or `token_level_rewards`.
5. `compute_mopd_advantage()` combines token-level MOPD, sequence-level teacher distillation, and ORM explicitly, and dedicated tests prove that sequence-level teachers influence the final advantage even when ORM is disabled.
6. Checkpoint manifests and preflight validation include the new P2 fields so backend/policy drift is detectable on resume.

## Idempotence and Recovery

The P2 patch must remain additive and preserve the P1 path when every teacher uses `backend: legacy_ref` and `tokenizer_policy: compatible`. The new quantized worker should fail fast with an informative error if bitsandbytes or the required HF quantization support is unavailable, rather than silently falling back to some other backend. If sequence-distillation wiring partially lands and breaks tests, the safe recovery path is to keep the new config fields but gate sequence teachers behind a validation error until the worker and estimator branches are both present.

## Artifacts and Notes

Expected acceptance evidence after implementation includes:

    tests/unit/test_teacher_workers.py::test_quantized_teacher_backend_uses_dedicated_worker PASSED
    tests/unit/test_mopd_trainer_runtime.py::test_sequence_teacher_jobs_decode_student_responses_to_text PASSED
    tests/unit/test_mopd_advantage.py::test_mopd_advantage_explicitly_adds_sequence_teacher_signal PASSED
    tests/unit/test_mopd_trainer_runtime.py::test_mopd_manifest_records_teacher_backend_and_tokenizer_policy PASSED

This section must be updated with final transcripts once implementation and verification complete.

Observed verification transcripts:

    pytest tests/unit/test_teacher_config.py tests/unit/test_teacher_workers.py tests/unit/test_mopd_trainer_runtime.py tests/unit/test_mopd_advantage.py tests/unit/test_teacher_routing.py tests/unit/test_mopd_resource_pools.py tests/integration/test_mopd_e2e.py -v
    # 90 passed, 1 skipped

    ruff check verl/workers/config/teacher.py verl/workers/teacher_workers.py verl/trainer/ppo/core_algos.py verl/trainer/ppo/ray_trainer.py tests/unit/test_teacher_config.py tests/unit/test_teacher_workers.py tests/unit/test_mopd_advantage.py tests/unit/test_mopd_trainer_runtime.py tests/unit/test_teacher_routing.py
    # All checks passed

    python -m compileall verl/workers/config/teacher.py verl/workers/teacher_workers.py verl/trainer/ppo/core_algos.py verl/trainer/ppo/ray_trainer.py
    # Exit code 0

## Interfaces and Dependencies

At the end of this round, the following interfaces must exist.

In `verl/workers/config/teacher.py`, extend `TeacherConfig` with at least:

    backend: str = "legacy_ref"
    tokenizer_policy: str = "compatible"
    seq_reward_weight: float = 1.0

where `backend` accepts `legacy_ref`, `hf_int8`, and `hf_4bit`, and `tokenizer_policy` accepts `compatible` and `sequence_reward`.

In `verl/workers/teacher_workers.py`, define a new worker class along the lines of:

    class HFQuantizedTeacherWorker(Worker):
        def __init__(self, config): ...
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def init_model(self): ...
        @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
        def compute_ref_log_prob(self, data: DataProto): ...
        @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"), blocking=False)
        def compute_ref_log_prob_async(self, data: DataProto): ...
        @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
        def compute_seq_scores(self, data: DataProto): ...
        @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"), blocking=False)
        def compute_seq_scores_async(self, data: DataProto): ...

In `verl/trainer/ppo/ray_trainer.py`, `RayPPOTrainer` must grow helper-level behavior equivalent to:

    def _build_teacher_worker(self, teacher_cfg, wg_kwargs): ...
    def _decode_mopd_response_texts(self, batch: DataProto) -> np.ndarray: ...
    def _build_mopd_sequence_teacher_jobs(self, batch: DataProto) -> tuple[list[dict[str, Any]], torch.device]: ...
    def _compute_teacher_sequence_rewards(self, batch: DataProto) -> torch.Tensor: ...

In `verl/trainer/ppo/core_algos.py`, `compute_mopd_advantage()` must accept explicit sequence-distillation inputs such as:

    teacher_seq_reward: Optional[torch.Tensor] = None
    teacher_seq_weight: float | torch.Tensor = 0.0
    teacher_token_mask: Optional[torch.Tensor] = None

and combine them without routing through ORM.

Update note: created this ExecPlan on 2026-03-14 to cover the `solution-review` P2 scope after P1 had already landed in this worktree.
