# MOPD Implementation - Current Test Results

**Date**: 2026-03-15
**Branch**: `feature/mopd-implementation` (worktree: `mopd-implementation`)
**Scope**: Current worktree source, including uncommitted follow-up changes

---

## Analysis Boundary

This document was refreshed against the current branch source, not against the older
80-test snapshot that previously lived in this file.

For this refresh, the evidence source is split into two parts:

- **Freshly executed in this update**
  - `pytest -q` over the CPU-safe MOPD suite
  - `pytest --collect-only -q` over the same suite
  - `pytest -q tests/unit/test_teardown_cleanup.py`
  - `bash recipe/mopd/run_mopd_qwen3_4b_preflight.sh` on the latest worktree
  - `VERL_MOPD_E2E=1 pytest tests/integration/test_mopd_e2e.py::test_mopd_training_e2e -v`
  - `bash recipe/mopd/run_mopd_qwen3_4b.sh` with an isolated checkpoint directory
  - conservative long-run reruns on the latest worktree, including the successful `18/18` closure run
  - real-weight `HFQuantizedTeacherWorker` profiling for `hf_int8` / `hf_4bit`
- **Not freshly exercised in this update**
  - multi-node teacher deployment
  - fault-recovery / kill-injection scenarios
  - real-weight quantized-teacher `sequence_reward` scoring path

That means this document now reflects the **current CPU-safe test surface and a fresh
single-node, 4-GPU runtime closure for both the real recipe contract and a
conservative long-run stress rerun**.
It still does **not** claim fresh multi-node or fault-recovery coverage.

---

## Executive Summary

The old detailed breakdown in this file was stale. The current branch now exposes a
larger and materially different MOPD validation surface:

- The suite has grown from the old **80 collected tests** snapshot to **116 collected tests**.
- A fresh CPU-safe rerun on the current worktree completed with:
  - **115 passed**
  - **1 skipped**
  - **1 warning**
- A separate teardown/cleanup regression file completed with:
  - **5 passed**
  - **3 warnings**
- A fresh recipe-aligned runtime closure on 2026-03-15 completed with:
  - preflight shell entrypoint success at `training/global_step:1`
  - opt-in GPU E2E outer pytest success: **1 passed, 1 warning in 345.91s**
  - full recipe shell run success with `Training Progress: 100%|...| 6/6`
  - final checkpoint marker written as `latest_checkpointed_iteration.txt = 6`
- A fresh conservative long-run rerun on 2026-03-15 completed with:
  - successful `Training Progress: 100%|...| 18/18`
  - `step:18 ... training/global_step:18`
  - `validation generation end` and `Final validation metrics`
  - routing mean `cell=0.4965` / `disease=0.5035`, matching the `100 / 100` dataset split
  - stable IS mask behavior with mean `mopd/is_valid_fraction=0.9999998278` and mean `mopd/is_zeroed_fraction=1.714958e-7`
- A fresh real-weight quantized-teacher profiling sweep on 2026-03-15 completed with:
  - `hf_int8` and `hf_4bit` both loading the real recipe teacher checkpoints successfully
  - 1 / 2 / 4 teacher cases profiled on GPUs 4-7
  - per-teacher peak reserved memory of about **5.05 GiB** (`hf_int8`) vs **3.81 GiB** (`hf_4bit`)
  - measurable first-step latency improvement when single-teacher micro-batch grows from `2` to `4`
- The newly covered areas that did not exist in the old snapshot include:
  - batch-level `lambda_val` override semantics in ExOPD
  - sequence-teacher advantage composition and dispatch plumbing
  - async teacher scheduling across resource pools
  - dedicated quantized teacher worker behavior
  - `TeacherResourcePoolConfig` validation
  - run-script import binding to the current worktree
  - GPU E2E env-contract and success-contract tests
  - trainer fit finalization and dataloader-worker shutdown checks

The full GPU E2E path still remains an opt-in pytest entrypoint gated by
`VERL_MOPD_E2E=1` and CUDA availability, but unlike the previous version of this
document it was also freshly executed in this refresh.

---

## Fresh Execution Evidence

### 1. CPU-safe regression suite

Command:

```bash
pytest -q \
  tests/unit/test_mopd_advantage.py \
  tests/unit/test_teacher_routing.py \
  tests/unit/test_teacher_config.py \
  tests/unit/test_teacher_workers.py \
  tests/unit/test_dataset_teacher_id.py \
  tests/unit/test_mopd_preflight.py \
  tests/unit/test_mopd_resource_pools.py \
  tests/unit/test_mopd_run_script.py \
  tests/unit/test_mopd_trainer_runtime.py \
  tests/integration/test_mopd_e2e.py
```

Result:

```text
115 passed, 1 skipped, 1 warning in 11.31s
```

### 2. Collected test inventory

Command:

```bash
pytest --collect-only -q \
  tests/unit/test_mopd_advantage.py \
  tests/unit/test_teacher_routing.py \
  tests/unit/test_teacher_config.py \
  tests/unit/test_teacher_workers.py \
  tests/unit/test_dataset_teacher_id.py \
  tests/unit/test_mopd_preflight.py \
  tests/unit/test_mopd_resource_pools.py \
  tests/unit/test_mopd_run_script.py \
  tests/unit/test_mopd_trainer_runtime.py \
  tests/integration/test_mopd_e2e.py
```

Result:

```text
116 tests collected in 8.57s
```

### 3. Teardown / cleanup regression file

Command:

```bash
pytest -q tests/unit/test_teardown_cleanup.py
```

Result:

```text
5 passed, 3 warnings in 11.55s
```

### 4. Recipe preflight on the real shell entrypoint

Command:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
CKPTS_DIR=/tmp/mopd-real-run-closure-20260315 \
PREFLIGHT_TIMEOUT_SECONDS=1200 \
PREFLIGHT_LOG_FILE=/tmp/mopd-real-run-closure-20260315/preflight/mopd_preflight_latest.log \
bash recipe/mopd/run_mopd_qwen3_4b_preflight.sh
```

Result:

```text
exit code 0
success boundary reached at step:1 ... training/global_step:1
```

Artifact:

- `/tmp/mopd-real-run-closure-20260315/preflight/mopd_preflight_latest.log`

### 5. Opt-in GPU E2E subprocess path

Command:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
VERL_MOPD_E2E=1 \
MOPD_E2E_TRAIN_MAX_SAMPLES=8 \
MOPD_E2E_VAL_MAX_SAMPLES=8 \
pytest tests/integration/test_mopd_e2e.py::test_mopd_training_e2e -v \
  --basetemp=/tmp/mopd-real-run-closure-20260315/pytest-e2e
```

Result:

```text
1 passed, 1 warning in 345.91s
```

Inner training artifact:

- `/tmp/mopd-real-run-closure-20260315/pytest-e2e/test_mopd_training_e2e0/mopd-e2e.log`

Observed success boundary inside the subprocess log:

```text
step:1 ... training/global_step:1
Training Progress: 100%|...| 1/1
```

### 6. Full recipe shell run

Command:

```bash
CKPTS_DIR=/gpfs/Mamba/Project/Single_Cell/Training/MOPD-real-run-closure-20260315 \
bash recipe/mopd/run_mopd_qwen3_4b.sh
```

Result:

```text
exit code 0
Training Progress: 100%|...| 6/6
step:6 ... training/global_step:6
latest_checkpointed_iteration.txt = 6
```

Artifacts:

- `/gpfs/Mamba/Project/Single_Cell/Training/MOPD-real-run-closure-20260315/logs/model_training_20260315_14.log`
- `/gpfs/Mamba/Project/Single_Cell/Training/MOPD-real-run-closure-20260315/latest_checkpointed_iteration.txt`

### 7. Conservative long-run rerun closure

Run identifier:

- `/gpfs/Mamba/Project/Single_Cell/Training/MOPD-longrun-closure-20260315-eager025-maxseq16-e3-r3`

Observed rollout overrides from the run config dump:

- `actor_rollout_ref.rollout.enforce_eager=True`
- `actor_rollout_ref.rollout.gpu_memory_utilization=0.25`
- `actor_rollout_ref.rollout.max_num_seqs=16`
- `actor_rollout_ref.rollout.max_num_batched_tokens=8192`
- `actor_rollout_ref.rollout.max_model_len=8192`

Long-run lineage:

- original run `/gpfs/Mamba/Project/Single_Cell/Training/MOPD-longrun-closure-20260315-e3`
  failed before completion with generation-side `CUDA out of memory`
- rerun #2 `/gpfs/Mamba/Project/Single_Cell/Training/MOPD-longrun-closure-20260315-eager035-maxseq32-e3-r2`
  moved the boundary past generation but failed after `step:9` on vLLM wake-up with
  `cumem_allocator.cpp:62`
- rerun #3 `/gpfs/Mamba/Project/Single_Cell/Training/MOPD-longrun-closure-20260315-eager025-maxseq16-e3-r3`
  completed successfully

Result:

```text
exit code 0
Training Progress: 100%|...| 18/18
step:18 ... training/global_step:18
validation generation end
Final validation metrics
```

Artifacts:

- `/gpfs/Mamba/Project/Single_Cell/Training/MOPD-longrun-closure-20260315-eager025-maxseq16-e3-r3/console.log`
- `/gpfs/Mamba/Project/Single_Cell/Training/MOPD-longrun-closure-20260315-eager025-maxseq16-e3-r3/metrics.jsonl`

Observed closure answers:

- teacher routing matched the `100 / 100` dataset split:
  `mopd/cell_type_teacher/sample_fraction` mean `0.4965277778`,
  `mopd/disease_state_teacher/sample_fraction` mean `0.5034722222`
- IS masking remained controlled:
  `mopd/is_valid_fraction` mean `0.9999998278`,
  `mopd/is_zeroed_fraction` mean `1.714958e-7`,
  and the final step itself stayed at `1.0 / 0.0`
- advantage and reverse-KL stayed interpretable rather than diverging:
  final `adv_mean` was `-0.5929` / `-0.7875`,
  final `reverse_kl_mean` was `-0.5919` / `-0.7870`
- `seq_reward_mean` stayed `0.0` on both teachers because this run still used the
  zero-reward recipe path, not because the sequence-teacher plumbing failed
- response-length pressure remained high even in the successful run:
  `response_length/clip_ratio` hit `1.0` at `step:17` and finished at `0.9921875`

This is the first fresh run in this update that demonstrates the full `18/18`
long-run closure on the latest worktree.

### 8. Post-completion teardown noise review

The GPU E2E log, the default `6/6` full-run log, and the conservative `18/18`
long-run log all contain late `multiprocessing.resource_tracker` `KeyError`
messages, and the E2E log also includes a Ray worker `SIGTERM` notice.
In all three artifacts these lines appear **after** the successful completion markers:

- after `Training Progress: 100%|...| 1/1` in the GPU E2E log
- after `Training Progress: 100%|...| 6/6`, final validation, and checkpoint writes in the full-run log
- after `Training Progress: 100%|...| 18/18`, `validation generation end`, and
  `Final validation metrics` in the long-run log

For this refresh they are treated as **post-completion shutdown noise**, not as
evidence of an in-run MOPD failure.

This file is **not** part of the 116-test core MOPD suite. It is listed separately because
it strengthens the current "deployment / engineering closure" picture without changing the
historical MOPD-suite comparison baseline.

### 9. Warning profile

The fresh CPU-safe MOPD suite emitted one non-blocking warning:

- `ray.util.state.util.py`: Ray state API deprecation warning

The separate teardown file emitted:

- the same Ray state API deprecation warning
- optional-dependency import warnings from the vLLM path (`SwigPyPacked`, `SwigPyObject`, `swigvarlink`)

No MOPD-specific failures were observed in the fresh reruns.

### 10. Real-weight quantized teacher profiling

Method:

- teacher-only harness that instantiated `HFQuantizedTeacherWorker` directly
- `init_model()` + first `_compute_ref_log_prob_impl()` measured on an 8-sample batch
- real checkpoints came from the recipe defaults for `CELL_TYPE_TEACHER_PATH` and `DISEASE_STATE_TEACHER_PATH`
- the `N teacher` case used 4 GPUs and repeated those two real checkpoints across 4 teacher workers
- peak memory below is `torch.cuda.max_memory_reserved()` on the busiest teacher GPU; total reserved memory is shown in parentheses

| Teacher Count | Backend | `log_prob_micro_batch_size` | Peak Reserved GiB | First-Step Latency (s) | Startup Time (s) |
|---:|---|---:|---:|---:|---:|
| 1 | `hf_int8` | 2 | `5.05` (`5.05` total) | `1.70` | `17.95` |
| 1 | `hf_int8` | 4 | `5.05` (`5.05` total) | `1.20` | `17.45` |
| 2 | `hf_int8` | 2 | `5.05` (`10.05` total) | `2.44` | `28.20` |
| 4 | `hf_int8` | 2 | `5.05` (`20.10` total) | `3.28` | `49.38` |
| 1 | `hf_4bit` | 2 | `3.81` (`3.81` total) | `1.15` | `15.37` |
| 1 | `hf_4bit` | 4 | `3.81` (`3.81` total) | `0.77` | `15.23` |
| 2 | `hf_4bit` | 2 | `3.81` (`7.63` total) | `1.54` | `27.78` |
| 4 | `hf_4bit` | 2 | `3.81` (`15.25` total) | `2.22` | `52.60` |

Observed takeaways:

- `hf_4bit` reduced per-teacher peak reserved memory from about `5.05 GiB` to `3.81 GiB` relative to `hf_int8`
- increasing single-teacher micro-batch from `2` to `4` improved first-step latency without materially changing peak memory
- total reserved memory scaled close to linearly with teacher count in this teacher-only harness
- startup time also scaled close to linearly because the current trainer path initializes teacher workers sequentially

---

## Current Suite Inventory

| File | Tests Collected | Current Focus |
|---|---:|---|
| `tests/unit/test_mopd_advantage.py` | 15 | MOPD / ExOPD formulae, IS correction, ORM, sequence-teacher mixing, dispatch plumbing |
| `tests/unit/test_teacher_routing.py` | 10 | `teacher_id` routing, balancing, scatter alignment, async scheduling, dispatch mesh |
| `tests/unit/test_teacher_config.py` | 14 | `TeacherConfig`, `MOPDConfig`, `TeacherResourcePoolConfig` validation |
| `tests/unit/test_teacher_workers.py` | 11 | worker-config building, tokenizer propagation, quantized worker path, shared base path |
| `tests/unit/test_dataset_teacher_id.py` | 5 | dataset extraction and collation of `teacher_id` |
| `tests/unit/test_mopd_preflight.py` | 9 | preflight command synthesis and success/failure log detection |
| `tests/unit/test_mopd_resource_pools.py` | 4 | teacher resource pool registration and colocate budget behavior |
| `tests/unit/test_mopd_run_script.py` | 3 | production script defaults, `PYTHONPATH` binding, self-check logging |
| `tests/unit/test_mopd_trainer_runtime.py` | 25 | tokenizer compatibility, sequence teacher runtime, metrics, manifest drift |
| `tests/integration/test_mopd_e2e.py` | 20 | lightweight integration, env contract, success contract, opt-in GPU E2E |
| **Total** | **116** | |

The single skipped test in the fresh run is the GPU-only
`tests/integration/test_mopd_e2e.py::test_mopd_training_e2e`.

The extra teardown file sits outside this inventory and currently contributes:

- `tests/unit/test_teardown_cleanup.py`: 5 tests covering tracking/logger finish idempotence, vLLM shutdown, replica shutdown, agent-loop cleanup, and trainer dataloader shutdown

---

## What The Current Tests Actually Prove

### 1. Algorithm and advantage composition

The algorithm layer is no longer only testing the original reverse-KL path.
Current coverage now includes:

- standard MOPD dispatch and output shape
- ExOPD base-normalized path
- batch-level `lambda_val` overriding config-level scalar in ExOPD
- confirmation that standard MOPD ignores `lambda_val` when no base log prob exists
- IS overflow protection and degenerate all-masked fallback
- ORM additive mixing
- sequence-teacher additive mixing, both with and without ORM
- explicit forwarding of `teacher_seq_reward`, `teacher_seq_weight`, and `teacher_token_mask`

Representative tests:

- `test_batch_lambda_overrides_config_scalar_for_exopd_dispatch`
- `test_batch_lambda_does_not_change_standard_mopd_without_base_log_prob`
- `test_mopd_advantage_sequence_teacher_signal_changes_result_when_orm_disabled`
- `test_mopd_advantage_sequence_teacher_signal_adds_on_top_of_orm`
- `test_compute_advantage_dispatch_uses_teacher_sequence_reward_and_mask_fields`

### 2. Teacher routing, balancing, and scheduling

Routing coverage is now meaningfully deeper than “shape is correct”.
Current tests exercise:

- per-sample `teacher_id` routing
- unknown-teacher fail-fast behavior
- teacher sub-batch balancing before forward
- scatter-back alignment after balanced reordering
- async overlap across different resource pools
- serialization within the same resource pool
- use of teacher dispatch mesh metadata when deriving DP size

Representative tests:

- `test_teacher_log_prob_balances_teacher_sub_batches_before_forward`
- `test_teacher_log_prob_preserves_sample_alignment_after_balancing`
- `test_teacher_log_prob_async_overlaps_different_pools_but_serializes_same_pool`
- `test_teacher_log_prob_uses_teacher_dispatch_mesh_for_dp_size`

### 3. Config schema and worker construction

The config and worker tests now reflect the current multi-teacher architecture rather
than the older dual-teacher assumptions.

Covered now:

- duplicate teacher-name rejection
- backend validation (`legacy_ref`, `hf_int8`, `hf_4bit`)
- tokenizer policy validation (`compatible`, `sequence_reward`)
- positive `seq_reward_weight`
- positive teacher resource pool dimensions and colocate budgets
- fixed ref micro-batch behavior in teacher worker configs
- propagation of teacher tokenizer overrides
- presence of dedicated `HFQuantizedTeacherWorker`
- quantized sequence-score micro-batching
- rejection of quantized teachers on the legacy ref-worker config path
- shared ExOPD base worker construction from `algorithm.mopd.base_model_path`

Representative tests:

- `test_teacher_config_rejects_unknown_backend`
- `test_teacher_config_rejects_unknown_tokenizer_policy`
- `test_teacher_resource_pool_config_validates_positive_dimensions`
- `test_quantized_teacher_backend_has_dedicated_worker_module`
- `test_quantized_teacher_sequence_scores_respect_micro_batch_size`
- `test_base_worker_config_uses_shared_base_model_path`

### 4. Data path, preflight helper, and run scripts

The source-backed test surface now validates not only internal Python helpers but
also the recipe-facing shell and preflight contracts.

Covered now:

- `teacher_id` extraction into `batch.non_tensor_batch`
- default `"default"` fallback when configured field is missing
- preflight command generation with teacher overrides and `tokenizer_compat_group`
- success detection tied to the first real training step
- failure-marker detection for NCCL timeout, actor death, fatal Python error, and segfault
- production run-script defaults for memory and batch sizing
- shell scripts binding `PYTHONPATH` to the current worktree
- shell-script self-check logging of `verl.__file__`

Representative tests:

- `test_teacher_id_end_to_end_with_collate`
- `test_build_training_command_uses_first_batch_overrides`
- `test_detect_terminal_event_recognizes_first_actor_update_success`
- `test_detect_terminal_event_recognizes_failure_markers[...]`
- `test_run_scripts_bind_python_imports_to_current_worktree`
- `test_run_scripts_log_verl_import_self_check`

### 5. Trainer runtime helpers and manifest behavior

This is the biggest single block of current coverage and the main reason the old
80-test breakdown is now obsolete.

Covered now:

- unknown configured teachers and missing configured teachers in preflight
- tokenizer compatibility metadata checks
- tokenizer vocab mismatch rejection
- base-model tokenizer compatibility checks for ExOPD
- preservation of `raw_prompt` into generation batches
- student-tokenizer decoding of response texts for sequence teachers
- construction of sequence-teacher jobs from `raw_prompt + response_text`
- DP-aware padding for sequence-teacher jobs
- materialization of `teacher_seq_reward`, `teacher_seq_weight`, and `teacher_token_mask`
- skipping token log-prob routing for sequence-only teachers
- allowing heterogeneous tokenizer sequence teachers in preflight
- per-teacher metric breakdown recording
- TensorDict-preserving resolution of teacher outputs
- manifest semantic drift rejection and deployment drift warning
- manifest recording of teacher backend, tokenizer policy, and sequence reward weight

Representative tests:

- `test_validate_tokenizer_compatibility_accepts_matching_metadata_with_compat_group`
- `test_validate_tokenizer_compatibility_rejects_base_model_vocab_mismatch`
- `test_build_mopd_sequence_teacher_jobs_uses_raw_prompt_and_response_text`
- `test_compute_teacher_sequence_rewards_builds_reward_tensor_and_teacher_token_mask`
- `test_compute_teacher_log_probs_skips_sequence_reward_only_teachers`
- `test_record_mopd_teacher_metrics_adds_teacher_breakdown`
- `test_validate_loaded_mopd_manifest_rejects_semantic_drift`
- `test_build_mopd_manifest_records_teacher_backend_and_tokenizer_policy`

### 6. Teardown and engineering-closure checks outside the core MOPD suite

The current worktree also adds a small but useful teardown-focused regression file:

- `Tracking.finish()` idempotence
- vLLM HTTP server shutdown cleanup
- vLLM replica shutdown + `ray.kill` behavior
- agent-loop manager shutdown resource release
- `RayPPOTrainer._shutdown_dataloader_workers()` idempotence

Representative tests:

- `test_tracking_finish_is_idempotent`
- `test_vllm_http_server_shutdown_stops_uvicorn_and_engine`
- `test_vllm_replica_shutdown_calls_server_shutdown_and_kills_servers`
- `test_agent_loop_manager_shutdown_releases_replicas_and_actors`
- `test_trainer_shutdown_dataloader_workers_is_idempotent`

These checks do **not** prove full MOPD lifecycle closure. They do, however, show that the
branch is now testing engineering-tail behavior in addition to the main algorithm/runtime path.

### 7. Integration-level contract coverage

The integration file is no longer just a lightweight formula smoke test.
It now has three distinct layers:

- lightweight config-to-advantage integration
- GPU E2E environment-variable contract validation
- GPU E2E success predicate validation

The file currently covers:

- deterministic config-to-advantage flow
- ExOPD end-to-end path
- ExOPD batch lambda override
- registry wiring for the `mopd` estimator
- recipe-env preference over legacy env vars:
  - `STUDENT_MODEL_PATH`
  - `CELL_TYPE_TEACHER_PATH`
  - `DISEASE_STATE_TEACHER_PATH`
  - `TRAIN_FILE`
  - `TEST_FILE`
- fallback to legacy `MOPD_TEST_*` env vars
- success detection keyed to log lines containing both `step:1` and `training/global_step:1`

Representative tests:

- `test_exopd_batch_lambda_overrides_config_scalar`
- `test_build_mopd_e2e_config_prefers_recipe_env_vars`
- `test_build_mopd_e2e_config_falls_back_to_legacy_env_vars`
- `test_log_reports_success_when_first_training_step_is_reached`

The final GPU subprocess test remains:

- `test_mopd_training_e2e`

It is guarded by:

- `torch.cuda.is_available()`
- `VERL_MOPD_E2E=1`

---

## What Changed Since The Old 80-Test Snapshot

The prior version of this document preserved a detailed **80-test / 79 passed / 1 skipped**
snapshot. That is no longer representative of the current branch.

The current delta is:

- **116 collected now**
- **+36 tests** relative to the old snapshot

The most important additions are:

- deeper advantage-path coverage around batch lambda and sequence-teacher composition
- deeper routing/scheduling tests
- expanded config coverage for backend, tokenizer policy, and resource-pool validation
- quantized teacher worker tests
- 2 new run-script contract tests
- broader trainer-runtime coverage for tokenizer compatibility, sequence jobs, metrics, and manifest
- 4 integration tests around GPU env-contract and success-contract behavior

Because of that change in scope, the old per-file counts, commit callouts, and
“production ready” conclusion in the previous version were removed rather than patched.

---

## Current Validation Boundary

Freshly validated in this update:

- current CPU-safe MOPD suite on the latest worktree state
- current collected test inventory and per-file counts
- current lightweight integration and contract tests
- current teardown/cleanup regression helpers
- recipe-aligned preflight on the real shell entrypoint
- full GPU subprocess path in `test_mopd_training_e2e`
- full `run_mopd_qwen3_4b.sh` recipe path through final checkpoint write
- conservative long-run rerun through `Training Progress: 100%|...| 18/18`,
  final validation, and `training/global_step:18`
- real-weight quantized-teacher load / memory / first-step profiling for `hf_int8` and `hf_4bit`

Not freshly re-executed in this update:

- any multi-node teacher resource pool deployment
- any deliberate fault-recovery or kill-injection scenario
- real-weight quantized-teacher `sequence_reward` scoring path

Still not proven by the current closure:

- multi-node teacher resource pools under a real cluster
- worker crash recovery or OOM recovery
- repeatability and quality gain beyond a single successful `18/18` long-run stress run
- quantized-teacher quality tradeoff and sequence-score path under real model weights
- whether the post-completion `resource_tracker` noise can be eliminated entirely at the vLLM / multiprocessing layer

---

## Recommended Next Checks Beyond Current Closure

The three core recipe-aligned commands have now been run for the latest worktree
state. The next validation gaps are no longer “run preflight / run GPU E2E”, but:

- longer-horizon or repeated long-run quality / convergence studies beyond the current `18/18` closure run
- multi-node teacher placement and resource-pool behavior
- failure-path validation such as actor kill, Ray restart, or rollout teardown interruption
- real quantized-teacher `sequence_reward` scoring and quality validation with `hf_int8` / `hf_4bit`
- explicit investigation of the post-completion `resource_tracker` noise in the vLLM path
