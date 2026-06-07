# Project Design Philosophy

> Scope: this document summarizes design principles observed in verl source code, tests, docs, GitHub PR/Issue history, review comments, and frequent-contributor commits. It is guidance for Claude Code / Codex / human contributors before modifying the project.  
> Evidence levels: **明确事实** = stated in code/docs/review; **强推断** = repeatedly supported by independent evidence; **弱推断** = limited evidence, should be confirmed by maintainers; **未知** = insufficient evidence.

## 1. 摘要

- **Control flow 与 computation flow 分离是核心设计。** Trainer/controller expresses RL algorithm dataflow; workers/engines execute distributed model computation. Do not collapse backend details into `RayPPOTrainer` or `main_ppo.py`.
- **`DataProto` is the current cross-stage data contract.** Its TensorDict/non-tensor/meta-info semantics are shared by trainer, dispatch, rollout, reward, and tests; treat changes as high risk.
- **Configuration is public API.** Hydra YAML + dataclass configs are user-facing, generated docs/configs are checked by CI, and compatibility/migration paths are expected.
- **Composition/registry/adapter boundaries are preferred over hard-coded branches.** Engines, rollout backends, reward managers, datasets, and advantage estimators all extend through explicit registries or external object loading.
- **Backend integrations should be thin and localized.** FSDP/Megatron/vLLM/SGLang/TRTLLM differences belong in engine/rollout/backend directories and config, not in generic algorithm control flow.
- **Performance claims require reproducible evidence.** Memory, dtype, weight sync, and rollout changes need unit/e2e/benchmark evidence, not only plausibility.
- **Tests and docs are design artifacts.** Non-trivial changes should update matching tests, CI workflow scope, docs, examples, and generated config artifacts.
- **Small, focused PRs are the norm.** Recent merged PRs are mostly 1-3 files; large refactors are accepted only with clear migration purpose and broad test/docs cleanup.
- **Do not optimize or extend abstractions maintainers are retiring.** Recent reviews rejected work on `DataProtoFuture` because TransferQueue/native TensorDict directions supersede it.
- **Hardware and dependency compatibility are bounded.** The project supports multiple accelerators, but direct CUDA/NCCL literals are CI-guarded and very old vLLM versions/upstream library bugs are not automatically carried in verl.

## 2. 项目目标与非目标

### 目标

- **结论级别：明确事实**
- **说明：** verl is a flexible, efficient RL post-training framework for LLMs, built around HybridFlow / hybrid-controller programming.
- **证据：**
  - 文档：`README.md#Project Overview` describes verl/HybridFlow as a flexible and efficient RL post-training framework.
  - 文档：`docs/hybrid_flow.rst:45-80` defines RL as two-level dataflow and states verl adopts separated control/computation flow.
  - 文档：`docs/hybrid_flow.rst:84-97` explains a single controller process orchestrating generator/actor/critic workers.
  - 源码：`verl/trainer/main_ppo.py::main`, `verl/trainer/main_ppo.py::TaskRunner` are the Hydra/Ray entry path.
  - 源码：`verl/trainer/ppo/ray_trainer.py::RayPPOTrainer` orchestrates rollout, critic, reward, and updates.

### 非目标 / 不鼓励方向

- **结论级别：明确事实 + 强推断**
- **说明：** verl does not aim to couple each algorithm to one distributed backend, maintain every old third-party-version workaround, or accept feature code whose correct home has moved to another repo.
- **证据：**
  - 文档：`docs/hybrid_flow.rst:63-79` contrasts coupled multi-controller designs with separated flow and chooses separation.
  - 文档：`docs/workers/model_engine.rst#Architecture` states worker/SPMD trainer level is engine-agnostic; RL trainer constructs HybridFlow rather than model engine details.
  - 测试：`tests/special_sanity/check_device_api_usage.py:15-19,67` checks irregular `.cuda`, `"cuda"`, `"nccl"` usage and points to `verl/utils/device.py`.
  - PR：#6147 closed; maintainer comment says `vllm<=0.8.2` is too old to maintain backward compatibility.
  - PR：#6081 closed to wait for Transformers upstream fix instead of carrying a tokenizer workaround.
  - PR：#6077/#6093 closed because Generative RL / vllm-omni work belongs in `verl-omni`.

## 3. 核心设计哲学

### 3.1 分离控制流与计算流

结论级别：**明确事实**

说明：RL algorithm logic should remain in single-controller/trainer control flow; neural network computation and backend placement should live in workers/engines/rollout components.

证据：

- 文档：`docs/hybrid_flow.rst:68-80` explicitly says verl adopts separated control flow and computation flow to decouple RL algorithms from computation engines.
- 文档：`docs/hybrid_flow.rst:203-206` says users can use different computation backends and placement without modifying the control process.
- 源码：`verl/trainer/ppo/ray_trainer.py:1340-1551` expresses PPO-like loop as DataProto transformations and worker calls.
- 源码：`verl/workers/engine_workers.py:130-138` creates `BaseEngine` through `EngineRegistry` rather than direct backend imports in trainer.

对后续开发的要求：

- Put algorithm ordering and batch-level orchestration in trainer/controller code.
- Put backend-specific memory, weight sync, kernels, and runtime calls in engine/rollout adapters.
- Do not add backend-specific branches to `RayPPOTrainer.fit` unless no existing adapter/config boundary can represent the behavior.

### 3.2 显式配置优先，配置即 API

结论级别：**明确事实**

说明：User behavior is driven by Hydra YAML defaults and structured dataclass configs. Config fields require comments, generated config sync, validation, and backward compatibility consideration.

证据：

- 源码：`verl/trainer/config/ppo_trainer.yaml:1-5` states CI-enforced YAML doc format rules.
- 源码：`verl/trainer/config/ppo_trainer.yaml:7-47` composes model engine, actor, data, ref, rollout, model, critic, reward, algorithm, distillation configs.
- 源码：`verl/trainer/config/rollout/rollout.yaml:1-8` exposes `rollout.name` and mode as explicit config.
- 源码：`verl/utils/config.py:23-65` converts OmegaConf to dataclass and requires `_target_` when no dataclass type is supplied.
- 测试：`tests/special_sanity/test_config_docs.py:60-87` validates comments/blank lines/no inline comments for key config YAML files.
- Commit：`4de3ecf0 [cfg] refactor: add ActorConfig, EngineConfig...` moved config toward structured dataclasses while preserving dict-like access through `BaseConfig`.

对后续开发的要求：

- Add new user-visible knobs in the correct Hydra config group and dataclass schema.
- Update generated trainer configs and docs when adding config.
- Validate incompatible/deprecated fields early in `validate_config` or dataclass validation.
- Mark breaking API/config changes in PR title with `[BREAKING]`.

### 3.3 组合/注册/适配优先，而不是重写主循环

结论级别：**强推断**

说明：The project repeatedly uses registries, factories, and role/resource mappings to compose components.

证据：

- 源码：`verl/trainer/main_ppo.py:120-190` builds `role_worker_mapping` and resource pool mapping dynamically.
- 源码：`verl/trainer/ppo/ray_trainer.py:244-257` receives worker/resource mappings instead of instantiating all implementations internally.
- 源码：`verl/workers/engine/base.py:267-337` implements `EngineRegistry`.
- 源码：`verl/workers/rollout/base.py:83-105` maps rollout `(name, mode)` to backend classes.
- 源码：`verl/workers/reward_manager/registry.py:21-55` registers reward managers.
- 测试：`tests/trainer/ppo/test_core_algos_on_cpu.py::TestRegisterAdvEst` and `tests/workers/reward_manager/test_registry_on_cpu.py::test_register_decorator` cover registry behavior.

对后续开发的要求：

- Prefer adding a registered strategy/adapter/config over special-casing one path in trainer.
- Keep registries small and explicit; do not add generic plugin layers without clear extension demand.
- Reuse existing backend names and add feature flags when a capability is a mode of an existing backend.

### 3.4 后端集成局部化

结论级别：**明确事实 + 强推断**

说明：Training and rollout backends are plugin/adapter layers. Their third-party quirks should not leak into generic code.

证据：

- 源码：`verl/workers/engine/fsdp/transformer_impl.py`, `verl/workers/engine/megatron/transformer_impl.py` register engine implementations through `EngineRegistry`.
- 源码：`verl/workers/rollout/vllm_rollout/*`, `verl/workers/rollout/sglang_rollout/*`, `verl/workers/rollout/trtllm_rollout/*` isolate rollout runtime concerns.
- PR：#6117 initially introduced `sglang_pd`/`vllm_pd`; maintainer review said “Reuse `sglang` rollout backend.” Final design used `rollout.name=sglang` + `rollout.disaggregation.enabled=True` with internal subclassing.
- Commit：`0a902a94 [megatron] fix: patch support newer mcore version`; `b9d71f9a [megatron] fix: update patch for MLA flashattn forward` show upstream Megatron changes absorbed in Megatron-specific paths.

对后续开发的要求：

- Put backend-specific version checks, patches, kernel assumptions, and memory logic under the backend directory.
- Expose only stable user-facing config names; avoid new public backend names for feature variants.
- Provide e2e or integration evidence for backend changes, especially distributed world-size/resource mapping.

### 3.5 性能优化必须可解释、可观测、可验证

结论级别：**强推断**

说明：Performance work is accepted when it explains memory/latency tradeoffs and provides tests, metrics, or benchmark context.

证据：

- 源码：`verl/workers/engine_workers.py:662-727` centralizes rollout weight sync, checkpoint engine, LoRA base sync, offload, and KV cache resume.
- 源码：`verl/trainer/config/rollout/rollout.yaml:33-95,118-140` exposes performance knobs: GPU memory utilization, TP/DP/EP/PP, KV cache, chunked prefill, prefix caching, engine kwargs.
- 文档：`docs/perf/best_practices.rst` covers rollout backend, GPU memory, dynamic batch, tensor parallel and benchmark-driven tuning.
- Issue：#6225 maintainer challenged memory/800s sync claims and supplied measured hardware/model/parallelism evidence.
- PR：#6150 FSDP mixed precision fix required small Qwen GRPO fp16 vs bf16 train/val reward verification.
- Commit：`a9b024d4 [rollout,vllm] feat: split large weight into chunks...` explained peak memory reduction in weight sync.
- Commit：`45b4ce91 [perf] feat: Add rollout longtail observation metrics`; `545f8998 [BREAKING] [perf] refactor: Profiler api refactor`.

对后续开发的要求：

- Include hardware/model/backend/parallelism details for performance claims.
- Prefer adding metrics/profiler hooks over opaque optimizations.
- Keep performance-sensitive code inside resource/engine/rollout/sync boundaries.

### 3.6 兼容性：保留 public surface when possible, but do not carry unlimited legacy

结论级别：**明确事实 + 强推断**

说明：The project preserves important public/config/subclass paths during migrations, but rejects compatibility for too-old dependencies or deprecated directions.

证据：

- 源码：`verl/trainer/main_ppo.py:14-16` notes `main` is not combined with `ray_trainer` because `ray_trainer` is used by other mains.
- 源码：`verl/trainer/main_ppo.py:212-219` keeps `add_ref_policy_worker` as a no-op for backward compatibility with subclasses.
- 源码：`verl/utils/config.py:115-147` gives explicit deprecation error for old micro-batch fields and points to `*_per_gpu` fields.
- 源码：`verl/base_config.py:20-68` makes dataclass configs dict-like for compatibility.
- PR：#6147 rejects `vllm<=0.8.2` compatibility.
- PR：#6234 rejects optimizing `DataProtoFuture` because TransferQueue is mature and native TensorDict is future direction.
- Commit：`018f8dc5 [reward] fix: backward compatibility with old reward config`.

对后续开发的要求：

- Explain compatibility impact for public APIs, config, data format, and error semantics.
- Add migration/deprecation paths for user-visible changes.
- Do not add long-term support code for obsolete third-party versions without maintainer direction.

### 3.7 错误处理：fail early on contract violations

结论级别：**明确事实**

说明：verl commonly validates config/data/registry contracts with assert/ValueError/TypeError near the boundary.

证据：

- 源码：`verl/protocol.py:454-478` asserts `DataProto` batch/non-tensor consistency.
- 源码：`verl/protocol.py:781-798` documents conflict errors for union semantics.
- 源码：`verl/utils/config.py:89-202` validates batch divisibility, deprecated fields, validation sampling, LoRA/vLLM config.
- 源码：`verl/workers/engine/base.py:313-337`, `verl/workers/reward_manager/registry.py:43-55`, `verl/workers/rollout/base.py:91-105` reject unknown registry keys.
- 源码：`verl/utils/dataset/rl_dataset.py:490-498` rejects custom dataset classes not inheriting `torch.utils.data.Dataset`.

对后续开发的要求：

- Validate user/config/data contracts at the adapter boundary.
- Prefer clear error messages that name the invalid config path.
- Do not silently coerce data shapes, truncation, or missing backend names in core paths.

### 3.8 测试策略：CPU contracts + backend/distributed/e2e coverage

结论级别：**明确事实**

说明：Core contracts are covered by CPU tests; backend/runtime behavior needs GPU/distributed/e2e or workflow-specific tests.

证据：

- 文档：`CONTRIBUTING.md:42-58` asks contributors to add CI tests where possible and choose relevant workflows.
- 文档：`tests/README.md:1-17` maps test folders to package namespaces and separates `special_distributed`, `special_e2e`, `special_npu`, `special_sanity`.
- 文档：`tests/README.md:19-30` explains CI workflow layout and manual exclusions when adding dedicated workflows.
- 测试：`tests/test_protocol_on_cpu.py` covers `DataProto`.
- 测试：`tests/trainer/ppo/test_core_algos_on_cpu.py` covers algorithm registries.
- 测试：`tests/workers/reward_manager/test_registry_on_cpu.py` covers reward manager registry.
- 测试：`tests/special_sanity/check_device_api_usage.py`, `tests/special_sanity/test_config_docs.py`, `tests/special_sanity/validate_structure.py` enforce cross-project constraints.
- PR：#6150, #5992, #6117, #6129 were accepted with tests/docs/workflows relevant to their scope.

对后续开发的要求：

- Add CPU tests for pure contract/registry/config behavior.
- Add GPU/e2e tests or documented manual experiments for backend, dtype, training, rollout, or distributed behavior.
- Update CI path filters when introducing a new workflow or test category.

## 4. 架构总览

```text
User config / examples / CLI
  │
  ▼
Hydra YAML + structured dataclass config
  │  validates public knobs and composes actor/data/ref/rollout/model/critic/reward
  ▼
verl.trainer.main_ppo::TaskRunner
  │  builds datasets, tokenizer/processor, role_worker_mapping, ResourcePoolManager
  ▼
verl.trainer.ppo.ray_trainer::RayPPOTrainer
  │  owns RL control flow: rollout -> reward/logprob/value -> advantage -> update
  │
  ├── DataProto protocol: TensorDict + non_tensor_batch + meta_info
  │
  ▼
verl.single_controller WorkerGroup + @register dispatch
  │  splits/collects DataProto and routes calls to Ray workers
  ▼
Workers / adapters
  ├── ActorRolloutRefWorker / TrainingWorker facade
  ├── BaseEngine via EngineRegistry: FSDP/FSDP2/Megatron/TorchTitan/VeOmni/...
  ├── BaseRollout via rollout registry: vLLM/SGLang/TRTLLM/HF...
  ├── Reward manager registry / custom reward function import
  └── Dataset/model utilities and external object loading
```

Allowed dependency direction: config/trainer/controller depend on abstract worker/engine/rollout/reward contracts; backend directories depend on third-party runtimes. Reverse leakage from backend-specific assumptions into generic trainer should be avoided.

## 5. 模块边界

| 边界 | 允许 | 禁止 | 证据 |
|---|---|---|---|
| `verl/protocol.py::DataProto` | Add tested fields/operations preserving batch/non-tensor/meta-info consistency | Silent shape coercion; changing union/chunk semantics without broad tests; optimizing `DataProtoFuture` without checking deprecation direction | 源码：`verl/protocol.py:317-324,454-478,781-798`; 测试：`tests/test_protocol_on_cpu.py`; PR：#6234 |
| `verl/single_controller/*` | Add dispatch/collect patterns with decorator tests | Bypass `@register` metadata; expose raw Ray actor multi-call details to trainer users | 文档：`docs/single_controller.rst:33-52,127-145`; 源码：`verl/single_controller/base/decorator.py:398-442`; 测试：`tests/single_controller/base/test_decorator.py` |
| `RayPPOTrainer` control loop | Compose existing worker/reward/algorithm calls; add algorithm branches only when core dataflow requires it | Backend-specific runtime logic, direct vLLM/SGLang/FSDP hacks, untested batch semantics | 源码：`verl/trainer/ppo/ray_trainer.py:234-257,1340-1551`; 文档：`docs/hybrid_flow.rst:180-207` |
| Training engine layer | Implement backend-specific training in `verl/workers/engine/<backend>` and register via `EngineRegistry` | Instantiating backend classes directly in trainer; scattering third-party patches across core modules | 源码：`verl/workers/engine/base.py:267-337`; `verl/workers/engine_workers.py:130-138`; docs: `docs/workers/model_engine.rst#Add a new backend` |
| Rollout layer | Add backend/mode adapters under `verl/workers/rollout/<backend>`; prefer feature flags under existing backend names | Inventing new public backend names for one backend mode; mixing rollout lifecycle into trainer | 源码：`verl/workers/rollout/base.py:29-105`; PR：#6117 review “Reuse `sglang` rollout backend” |
| Reward layer | Use `custom_reward_function.path/name`, registry, or importlib reward manager | Hard-code task reward into trainer; bypass token-level reward manager semantics | 源码：`verl/trainer/ppo/reward.py:50-151`; `verl/workers/reward_manager/registry.py:21-55`; 文档：`docs/preparation/reward_function.rst` |
| Dataset layer | Preprocess into parquet/JSON/JSONL; use `data.custom_cls` for custom loader; keep `Dataset` inheritance | Mutating DataProto/data rows in ways that break batch consistency; expensive multimodal filter in wrong layer | 源码：`verl/utils/dataset/rl_dataset.py:71-86,157-168,478-503`; Issue：#6145; PR：#5965/#6079 |
| Config layer | Add documented YAML + dataclass schema + validation + generated config sync | Undocumented fields, inline comments, missing generated config, ambiguous defaults | 源码：`verl/trainer/config/ppo_trainer.yaml:1-47`; 测试：`tests/special_sanity/test_config_docs.py`; `.pre-commit-config.yaml:17-19` |
| Device/hardware abstraction | Use `verl/utils/device.py` and workflow-specific tests | Direct `.cuda`, `"cuda"`, `"nccl"` in normal code outside whitelist | 测试：`tests/special_sanity/check_device_api_usage.py`; `.pre-commit-config.yaml:41-43`; PR：#5943 |
| Public package API | Keep `verl.__all__`, config, CLI examples stable or document breaking changes | Breaking function signature/config/data format without `[BREAKING]`, migration, tests | 源码：`verl/__init__.py::__all__`; PR template `.github/PULL_REQUEST_TEMPLATE.md:7-13,31-40`; `CONTRIBUTING.md:79-87` |

## 6. 已识别的设计模式

### 6.1 Registry + Factory

结论级别：**明确事实**

出现位置：

- `verl/workers/engine/base.py::EngineRegistry.register/new`
- `verl/workers/rollout/base.py::_ROLLOUT_REGISTRY`, `get_rollout_class`
- `verl/workers/reward_manager/registry.py::register`, `get_reward_manager_cls`
- `verl/trainer/ppo/core_algos.py::register_adv_est`, `get_adv_estimator_fn`
- `verl/models/registry.py::ModelRegistry`

解决的问题：Map explicit string/config keys to concrete implementations while keeping trainer/controller code backend-agnostic.

为什么这是项目偏好的方式：It appears across engines, rollouts, rewards, algorithms, and models; tests assert duplicate/unknown/case-sensitive behavior.

后续开发如何遵循：Register a new implementation at the boundary and configure it through the proper YAML/dataclass field.

不应该怎么用：Do not create global registries for vague extension points; do not hide required configuration or silently fall back to unknown defaults.

证据：

- 源码：`verl/workers/engine/base.py:267-337`
- 源码：`verl/workers/rollout/base.py:83-105`
- 源码：`verl/workers/reward_manager/registry.py:21-55`
- 源码：`verl/trainer/ppo/core_algos.py:88-150`
- 测试：`tests/trainer/ppo/test_core_algos_on_cpu.py::TestRegisterAdvEst`
- 测试：`tests/workers/reward_manager/test_registry_on_cpu.py::test_register_decorator`
- PR / Review：#6117 backend registry naming was narrowed to reuse `sglang` plus config flag.

### 6.2 Strategy

结论级别：**明确事实**

出现位置：

- `verl/trainer/config/actor/actor.yaml::strategy`
- `verl/trainer/config/rollout/rollout.yaml::name/mode`
- `verl/trainer/config/ppo_trainer.yaml::algorithm.adv_estimator`
- `verl/trainer/ppo/core_algos.py::AdvantageEstimator`

解决的问题：Choose algorithm estimator, training engine, rollout engine, and execution mode from config.

为什么这是项目偏好的方式：It allows the same control loop to run PPO/GRPO/RLOO-style variants and different backends.

后续开发如何遵循：Expose strategy as documented config only after implementing boundary-specific class/function and tests.

不应该怎么用：Do not add strategy flags that alter many unrelated layers or require trainer-specific backend knowledge.

证据：

- 源码：`verl/trainer/config/ppo_trainer.yaml:58-99`
- 源码：`verl/trainer/config/rollout/rollout.yaml:1-8`
- 源码：`verl/trainer/ppo/core_algos.py:88-150`
- 源码：`verl/workers/engine_workers.py:130-138,603-607`

### 6.3 Adapter / Facade

结论级别：**明确事实**

出现位置：

- `verl/workers/engine_workers.py::TrainingWorker`
- `verl/workers/engine_workers.py::ActorRolloutRefWorker`
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py::ServerAdapter`
- `verl/workers/rollout/sglang_rollout/sglang_rollout.py::ServerAdapter`

解决的问题：Expose a uniform worker/rollout API while hiding backend-specific engine/server lifecycle and weight sync details.

为什么这是项目偏好的方式：The trainer calls worker methods and DataProto operations, while adapters own backend-specific methods.

后续开发如何遵循：Add backend adapter methods behind existing abstract/facade APIs; keep lifecycle semantics (`resume`, `sleep`, `update_weights`) consistent.

不应该怎么用：Do not bypass the facade from trainer to call third-party backend internals.

证据：

- 源码：`verl/workers/engine_workers.py:130-138,586-607,662-727`
- 源码：`verl/workers/rollout/base.py:29-80`
- PR：#6117 kept `SGLangPDReplica` as internal subclass selected by config.

### 6.4 Pipeline / Dataflow

结论级别：**明确事实**

出现位置：

- `docs/hybrid_flow.rst#PPO main loop`
- `verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit`
- `verl/protocol.py::DataProto.union`

解决的问题：Represent RL training as a sequence of high-level operators with explicit data passed between stages.

为什么这是项目偏好的方式：HybridFlow documentation defines RL as two-level dataflow; PPO loop is written as single-process-style pipeline over distributed workers.

后续开发如何遵循：Make new algorithm steps explicit DataProto transformations; preserve batch and metadata semantics.

不应该怎么用：Do not hide stateful side effects in backend calls that the trainer cannot observe or test.

证据：

- 文档：`docs/hybrid_flow.rst:45-80,180-207`
- 源码：`verl/trainer/ppo/ray_trainer.py:1340-1551`
- 源码：`verl/protocol.py:781-798`

### 6.5 Configuration Object

结论级别：**明确事实**

出现位置：

- `verl/base_config.py::BaseConfig`
- `verl/utils/config.py::omega_conf_to_dataclass`, `validate_config`
- `verl/trainer/config/*.py`, `verl/workers/config/*.py`
- YAML groups under `verl/trainer/config/`

解决的问题：Keep user-visible Hydra configs structured, documented, validated, and partially backward-compatible with dict-style access.

为什么这是项目偏好的方式：Config files are shipped as package data and checked by pre-commit/CI; recent contributors migrated loose configs to dataclasses.

后续开发如何遵循：Add schema + YAML + validation + tests + generated config updates together.

不应该怎么用：Do not add loose `DictConfig` fields consumed only by stringly-typed deep access with no validation.

证据：

- 源码：`verl/base_config.py:20-68`
- 源码：`verl/utils/config.py:23-65,74-202`
- 配置：`verl/trainer/config/ppo_trainer.yaml:1-47`
- 测试：`tests/special_sanity/test_config_docs.py:60-87`
- Commit：`4de3ecf0 [cfg] refactor: add ActorConfig, EngineConfig...`

### 6.6 Callback / Hook-like RPC decorator

结论级别：**明确事实**

出现位置：

- `verl/single_controller/base/decorator.py::register`
- `verl/single_controller/base/decorator.py::Dispatch`
- `verl/single_controller/base/worker_group.py::_bind_worker_method`

解决的问题：Attach dispatch/execute/blocking metadata to worker methods so driver-side `WorkerGroup` can expose multi-actor RPC as one call.

为什么这是项目偏好的方式：The single-controller docs call binding the heart of the system.

后续开发如何遵循：When adding worker APIs, choose correct `Dispatch` and `Execute` modes and cover them with worker/controller tests.

不应该怎么用：Do not call remote workers manually in generic trainer code when the method should be a registered worker API.

证据：

- 文档：`docs/single_controller.rst:127-145`
- 源码：`verl/single_controller/base/decorator.py:398-442`
- 源码：`verl/single_controller/base/worker_group.py:185-253`
- 测试：`tests/single_controller/base/test_decorator.py`

### 6.7 External Plugin via importlib

结论级别：**明确事实**

出现位置：

- `verl/utils/import_utils.py::load_extern_object`
- `verl/trainer/ppo/reward.py::get_custom_reward_fn`, `load_reward_manager`
- `verl/utils/dataset/rl_dataset.py::get_dataset_class`

解决的问题：Let users customize reward functions and datasets without forking core trainer.

为什么这是项目偏好的方式：Docs explicitly advertise custom reward/dataset paths and code validates external object contracts.

后续开发如何遵循：Use importlib extension when customization is user/task-specific; promote to built-in registry only when generally useful.

不应该怎么用：Do not hard-code task-specific reward/data logic into `RayPPOTrainer`.

证据：

- 源码：`verl/trainer/ppo/reward.py:50-151`
- 源码：`verl/utils/dataset/rl_dataset.py:478-503`
- 文档：`docs/examples/config.rst#Customized Dataset`, `docs/examples/config.rst#Customized Reward Function`

## 7. gh CLI 变更脉络分析

Commands used:

```bash
gh repo view volcengine/verl --json name,owner,description,defaultBranchRef,repositoryTopics,licenseInfo
gh pr list -R volcengine/verl --state merged --limit 100 --json number,title,author,mergedAt,labels,files,additions,deletions
gh pr list -R volcengine/verl --state closed --limit 100 --json number,title,author,closedAt,mergedAt,labels,comments,reviews
gh issue list -R volcengine/verl --state all --limit 100 --json number,title,author,createdAt,closedAt,labels,comments
gh pr view -R volcengine/verl 6117 --json number,title,state,comments,reviews,files
gh pr view -R volcengine/verl 6150 --json number,title,state,comments,reviews,files
gh pr view -R volcengine/verl 6234 --json number,title,state,comments,reviews,files
gh pr view -R volcengine/verl 6147 --json number,title,state,comments,reviews,files
gh pr view -R volcengine/verl 6081 --json number,title,state,comments,reviews,files
gh api repos/verl-project/verl/pulls/6117/comments --paginate
gh issue view -R volcengine/verl 6225 --json number,title,state,comments
```

### Repository snapshot

- **结论级别：明确事实**
- `gh repo view` returned owner `verl-project`, repo `verl`, default branch `main`, description `verl/HybridFlow: A Flexible and Efficient RL Post-Training Framework`.

### Merged PR patterns

- **结论级别：强推断**
- Last 100 merged PRs sampled by gh CLI:
  - 74/100 changed 1-3 files.
  - 36/100 touched `tests/` or `.github/workflows/`.
  - 20/100 touched docs/README.
- Accepted work is usually focused, but large PRs are accepted when they implement a clear migration/deprecation and update docs/tests/config.
- Evidence:
  - PR：#6129 split `LLMServerManager` out of `AgentLoopManager`, with docs/tests updates.
  - PR：#6074 `[BREAKING] [env] refactor: deprecate verl/interactions`; maintainer also required removing accidental unrelated notebook edits.
  - PR：#6117 SGLang PD disaggregation added tests and generated config updates.
  - PR：#6150 FSDP mixed precision fix included `tests/models/test_engine.py` and required real GRPO convergence check.

### Closed PR / Issue design signals

- **结论级别：明确事实 + 强推断**
- Maintainers close work when it optimizes deprecated abstractions, fixes at the wrong layer, duplicates existing work, targets migrated-out domains, or carries old dependency workarounds.
- Evidence:
  - PR：#6234 `DataProtoFuture` optimization closed; maintainer said TransferQueue supersedes it and `DataProto` itself will favor native TensorDict.
  - PR：#5965 overlong prompt truncation closed; maintainer said overlong prompts should be filtered, not truncated.
  - PR：#6079 early termination in `ToolAgentLoop` closed; maintainer said filtering needs systematic prompt supplementation.
  - PR：#6147 vLLM <= 0.8.2 compatibility closed as too old.
  - PR：#6081 tokenizer workaround closed to wait for Transformers upstream fix.
  - Issue：#6145 multimodal overlong filtering discussion: maintainer said image/video must be included for correct length, but large multimodal filtering in `RLDataset` is slow; prefer a mechanism in `AgentLoopWorker`.
  - Issue：#6225 performance proposal challenged due to unverified memory/time claims.

### Review comments repeatedly emphasized

| Rule | Evidence |
|---|---|
| Reuse existing backend public names; use config flags for backend modes | Review comment on PR #6117: “Reuse `sglang` rollout backend.” |
| Numeric/training fixes require real training verification | Review comment on PR #6150 requested GRPO experiment on small Qwen and train/val reward comparison, fp16 vs bf16 |
| Do not include unrelated file edits | PR #6074 maintainer: do not change `examples/tutorial/agent_loop_get_started/agent_loop_tutorial.ipynb` |
| Overlong prompts need systematic filtering, not truncation/early exit | PR #5965/#6079 maintainer comments |
| Prefer existing value-head mechanisms over duplicate model classes | PR #6263 maintainer pointed to FSDP TRL value head and Megatron mcore bridge |
| Do not optimize soon-deprecated paths | PR #6234 maintainer comments on `DataProtoFuture` / TensorDict future |
| Performance claims need measured evidence | Issue #6225 maintainer comments |
| Hardware support must preserve behavior and backend naming | PR #5943 inline reviews on device check behavior and backend naming |

## 8. 高频 Contributor 设计习惯

Commands used:

```bash
git shortlog -sn --all
git log --author="<AUTHOR>" --stat --oneline --date=short
git log --author="<AUTHOR>" --name-only --pretty=format:"%h %ad %s" --date=short
git show --stat --oneline <commit>
```

| Contributor | 高频修改模块 | 稳定设计习惯 | 代表 commits / PR | 对后续开发的启发 |
|---|---|---|---|---|
| Chi Zhang | `verl/workers`, `verl/trainer`, `.github/workflows`, `verl/utils` | 小范围 rollout/backend/CI fixes; accepts revert when runtime risk appears | `15371c91 [vllm] feat: make seed configurable and different among replicas`; `7f4b76a8 [rollout] fix: remove dtype cast`; `d9939add Revert ... vllm separation mode` | Backend/runtime changes should be small, reversible, and localized |
| Joel | `verl/workers`, `verl/trainer`, `verl/experimental`, checkpoint/rollout server | Large architecture/perf work includes config sync and performance rationale | `866a1eaa [trainer] feat: add new trainer with TransferQueue`; `a9b024d4 [rollout,vllm] feat: split large weight into chunks...`; `2edc06a4 [cfg] refactor: unify ppo_trainer and ppo_megatron_trainer config` | For major design changes, explain data/control split, config migration, and memory/throughput tradeoffs |
| HL | `.github/workflows`, README/docs, `verl/utils`, `verl/trainer` | Infrastructure, docs, CI, and trainer API clarity are treated as core work | `52437be1 [trainer] breaking: pass dataset as required args to SFTTrainer...`; `5313d96f [CI] fix: add additional pre-commit test...`; `d882b62b tests: add import utils tests` | Do not treat docs/CI as optional afterthoughts |
| ℍ𝕠𝕝𝕝𝕠𝕨 𝕄𝕒𝕟 | Megatron/MCore, `verl/models/mcore`, `verl/workers/megatron_workers.py` | Adapts upstream Megatron changes through patches/bridges in backend-specific paths | `b9d71f9a [megatron] fix: update patch for MLA flashattn forward`; `0a902a94 [megatron] fix: patch support newer mcore version`; `e5f5ea66 [megatron] feat: fused kernel support for new model engine` | Keep third-party backend changes thin, traceable, and local |
| Blue Space | CI/workflows, Docker, Megatron utilities, profiling | Performance and environment upgrades include observability and CI updates | `545f8998 [BREAKING] [perf] refactor: Profiler api refactor`; `45b4ce91 [perf] feat: Add rollout longtail observation metrics`; `9b6a07fa [docker] feat: update to vllm 0.10.0, mcore 0.13...` | Performance design includes profiler/metrics/environment reproducibility |
| Guangming Sheng | rollout server, vLLM/SGLang, `tests/e2e`, trainer config | Rollout lifecycle/caching/abort changes are conservative and sometimes reverted | `e0c46e9c [vllm] feat: support abort generating requests in vllm server`; `da214abc [vllm, rollout] feat: support reset prefix cache after abort`; `ef268476 Revert "[vllm] feat: remove workers from vLLMHttpServer"` | Rollout state machines are high risk; preserve reset/abort/cache invariants |
| Shawn / Yuxuan Tong | `verl/trainer/ppo/core_algos.py`, reward/algorithm, recipe | Algorithm changes focus on numerical correctness, distributed semantics, and recipe separation | `3671d37f [algo] feat: Exception for agg_loss when dp_size > 1...`; `b8d91ef8 [algo] fix: seq mean...`; `ab070522 [worker] feat: custom reward_manager`; `2bb42bae [recipe] feat: migrate recipe to dedicated repo` | Algorithm PRs must specify math/distributed semantics and tests |
| H | config schema, trainer/workers/tests | Structured config migration with dict-like compatibility | `4de3ecf0 [cfg] refactor: add ActorConfig, EngineConfig...`; `332c7d53 [cfg] refactor: add flatten megatron trainer config generation...`; `0b62a6ec [cfg] feat: add critic config class` | Treat config schema as stable API; keep old access patterns working where possible |
| Yuyang Ding | `verl/experimental/reward_loop`, `agent_loop`, reward config/docs/tests | Experimental reward/agent systems may break, but then receive compatibility/docs/test cleanup | `2cd92830 [reward] refactor: migrate all reward managers to the new asynchronous reward manager`; `67dc84b9 [BREAKING][reward] refactor: the full reward configuration`; `018f8dc5 [reward] fix: backward compatibility with old reward config` | Experimental can iterate, but public migration must converge |
| Zhen | Ascend/NPU Docker, `tests/special_npu`, e2e Ascend workflows | Hardware support reuses common scripts via device abstraction and CI matrix | `392791bd [hardware] feat: Auto set device_name to npu for Ascend NPU`; `cfcdddf1 [ci] feat: Update e2e_ascend to improve CI execution efficiency`; `252d7690 [doc, ci] fix: Update Ascend doc and fix e2e_ascend CI` | Hardware differences should be absorbed by device abstraction/workflows, not script forks |

## 9. 推荐扩展方式

| 扩展目标 | 推荐位置 | 推荐模式 | 必须测试 | 禁止做法 |
|---|---|---|---|---|
| 新 PPO-like advantage / loss | `verl/trainer/ppo/core_algos.py`, `verl/trainer/config/algorithm.py`, `verl/trainer/config/ppo_trainer.yaml`, examples/docs | `register_adv_est` / `register_policy_loss`; config strategy | CPU math/registry tests in `tests/trainer/ppo/`; e2e if affects training behavior | Rewriting `RayPPOTrainer.fit` for a small estimator variant; changing enum only with no registry/test |
| 全新 algorithm control flow | `recipe/<algorithm>/` first, or `verl/trainer/<algorithm>/` if core; examples/docs | Pipeline over DataProto and WorkerGroup | CPU unit tests for math/config; special_e2e for scripts | Copy-paste entire PPO trainer with backend-specific hacks |
| 新 training backend | `verl/workers/engine/<backend>/`, `verl/workers/config/`, `verl/trainer/config/engine|actor|critic|ref|model_engine` | `BaseEngine` + `EngineRegistry`; config object | Engine unit tests; distributed/e2e workflow if possible; generated config check | Instantiating backend directly in trainer; scattering third-party patches in generic code |
| 新 rollout backend/mode | `verl/workers/rollout/<backend>/`, `rollout/base.py`, `rollout/replica.py`, rollout config | `BaseRollout` adapter; reuse backend public name with feature flags when possible | `tests/workers/rollout/`; special_e2e; resource/world-size tests | Creating new public backend name for one mode of existing backend (#6117 lesson) |
| 新 reward function | External file via `reward.custom_reward_function.path/name`; or `verl/utils/reward_score/<task>.py` for common tasks | External import / task scorer | Unit tests for scoring edge cases; docs/example config | Hard-code reward logic in trainer or rollout |
| 新 reward manager | `verl/workers/reward_manager/<name>.py` or experimental reward loop manager | Registry or importlib manager | Registry tests; token-level reward shape tests; async tests if needed | Bypassing reward manager with ad-hoc batch mutation |
| 新 dataset format | Prefer `examples/data_preprocess/` to parquet/JSON/JSONL; custom Dataset via `data.custom_cls.path/name`; common loaders in `verl/utils/dataset/` | External `Dataset` subclass / preprocessing pipeline | `tests/utils/dataset/*_on_cpu.py`; multimodal tests when applicable | Expensive/incorrect filtering in wrong layer; custom class not inheriting `Dataset` |
| 新 config field | `verl/trainer/config/**.yaml`, `verl/trainer/config/*.py`, `verl/workers/config/*.py` | Config Object + validation | `tests/special_sanity/test_config_docs.py`; generated config pre-commit; targeted config tests | Undocumented YAML field, inline comments, no validation, no generated config update |
| 新 model support | `verl/models/transformers/` for FSDP/HF; `verl/models/mcore/` for Megatron; registry/bridge | Backend-specific model bridge/registry | Model loading/forward tests; e2e if backend-specific | Duplicate model path when existing value-head/bridge mechanisms exist (#6263) |
| 新 command/example | `examples/<domain>/`, docs, CI workflow if non-trivial | Hydra override script using existing trainer entry | Minimal workload CI or documented manual run | Hardcoded local paths, huge workload, unconnected script |
| 新 dependency | `setup.py` extras / pyproject if package metadata; Docker/CI updates | Optional extra when backend-specific | Import guards; CI for extra if feasible | Unrequested global dependency; workaround for upstream bug without maintainer agreement |
| 新 docs | `docs/`, examples README, generated config docs | Sphinx docs, concise examples | `docs` build / `doc.yml`; docs time/info checks if applicable | Docs without matching config/API examples for user-facing changes |

## 10. 不应破坏的不变量

- **Data contract:** `DataProto.batch`, `non_tensor_batch`, and `meta_info` must preserve length/shape consistency and conflict semantics. Evidence: `verl/protocol.py:454-478,781-798`; `tests/test_protocol_on_cpu.py`.
- **Single-controller abstraction:** Driver-side calls should remain simple worker-group method calls; remote actor splitting/collection stays behind `@register` and WorkerGroup. Evidence: `docs/single_controller.rst:127-145`; `verl/single_controller/base/decorator.py:398-442`.
- **Backend agnosticism of trainer:** Trainer expresses algorithm stages, not FSDP/Megatron/vLLM/SGLang internals. Evidence: `docs/hybrid_flow.rst:203-206`; `verl/workers/engine_workers.py:130-138`.
- **Config documentation format:** YAML fields need comments above, blank lines, no inline comments. Evidence: `verl/trainer/config/ppo_trainer.yaml:1-5`; `tests/special_sanity/test_config_docs.py`.
- **Generated config synchronization:** `autogen-trainer-cfg` pre-commit must stay green. Evidence: `.pre-commit-config.yaml:17-19`.
- **Device abstraction:** Avoid direct CUDA/NCCL literals outside whitelist; use `verl/utils/device.py`. Evidence: `tests/special_sanity/check_device_api_usage.py`; `.pre-commit-config.yaml:41-43`.
- **Test layout:** Tests map to package namespaces; special GPU/NPU/e2e tests are separated. Evidence: `tests/README.md:1-30`; `tests/special_sanity/validate_structure.py`.
- **PR title/API breaking semantics:** Titles follow `[{modules}] {type}: ...`, `[BREAKING]` required for API/config/function-signature breaks. Evidence: `.github/PULL_REQUEST_TEMPLATE.md:7-13`; `.github/workflows/check-pr-title.yml`.
- **Public extension points:** Custom reward/dataset import paths and backend registries must keep contract validation. Evidence: `verl/trainer/ppo/reward.py:50-151`; `verl/utils/dataset/rl_dataset.py:478-503`.
- **Performance evidence:** Optimization PRs should include reproducible measurements or targeted tests. Evidence: Issue #6225; PR #6150.
- **Repository boundary:** Work migrated to `verl-omni` or deprecated internals should not be reintroduced without maintainer direction. Evidence: PR #6077/#6093/#6234.

## 11. 常见反模式

### 11.1 大而杂的 PR

表现：One PR changes backend code, trainer logic, examples, docs, notebooks, and unrelated cleanup.

为什么不符合本项目：Recent accepted PRs are usually focused; large PRs are accepted only for clear migration/deprecation with coordinated docs/tests.

维护者或历史 PR 证据：

- gh CLI: 74/100 recent merged PRs changed 1-3 files.
- PR #6074 maintainer flagged unrelated notebook edit.
- PR template asks for concise overview, scope, tests, API examples, design changes.

正确做法：Split independent backend/config/docs cleanup unless they are required for one migration.

PR 自查问题：

- Is every changed file necessary for this design?
- Can any cleanup be a separate PR?
- Does the PR title accurately name all touched modules?

### 11.2 后端细节侵入 trainer

表现：Adding vLLM/SGLang/Megatron/FSDP-specific branches in `RayPPOTrainer.fit` or `main_ppo.py`.

为什么不符合本项目：HybridFlow separates algorithm control flow from computation engines.

维护者或历史 PR 证据：

- 文档：`docs/hybrid_flow.rst:68-80,203-206`.
- 源码：`EngineRegistry` and rollout registry handle backend selection.
- PR #6117 review required reusing `sglang` backend and config flag rather than new public backend names.

正确做法：Add/modify backend adapter, registry entry, config schema, and targeted backend tests.

PR 自查问题：

- Could this be represented as engine/rollout config?
- Is the trainer still backend-agnostic?

### 11.3 无证据的性能优化

表现：Claiming memory/time wins without model/hardware/backend/parallelism data or tests.

为什么不符合本项目：Performance boundaries are high risk and maintainers challenge unsupported claims.

维护者或历史 PR 证据：

- Issue #6225 maintainer questioned 30GB serialization and 800s sync claims.
- PR #6150 required GRPO fp16 vs bf16 reward verification.
- Commits `a9b024d4`, `45b4ce91`, `545f8998` show performance changes tied to memory/metrics/profiler context.

正确做法：Provide benchmark table, logs, workload details, and/or targeted unit/e2e tests.

PR 自查问题：

- Can a reviewer reproduce this claim?
- Does the PR include metrics or tests for the changed path?

### 11.4 修错层级错误：局部 workaround 破坏全局语义

表现：Truncating overlong prompts in `ToolAgentLoop`, early-terminating samples, or filtering multimodal prompts without image/video context.

为什么不符合本项目：Batch/data semantics require systematic filtering and prompt supplementation.

维护者或历史 PR 证据：

- PR #5965: overlong prompt should be filtered, not truncated.
- PR #6079: not systematic because filtered prompts need supplementation in AgentLoop.
- Issue #6145: multimodal filtering must include image/video; large filtering may belong in `AgentLoopWorker`.

正确做法：Fix at data/agent-loop boundary with explicit replacement/sample semantics and metrics.

PR 自查问题：

- Does this preserve batch size and reward semantics?
- Are multimodal tokens computed with media fields?

### 11.5 优化即将废弃的抽象

表现：Investing in `DataProtoFuture` or legacy workers without checking current migration direction.

为什么不符合本项目：Maintainers prefer moving toward TransferQueue/native TensorDict and engine workers.

维护者或历史 PR 证据：

- PR #6234 closed: `DataProtoFuture` superseded by TransferQueue; `DataProto` itself may favor native TensorDict.
- Commit `044bbba2 [BREAKING] [misc] refactor: deprecate workers, migrate to engines (#6067)`.

正确做法：Check recent PRs/RFCs and target current abstractions.

PR 自查问题：

- Is this abstraction still on the roadmap?
- Did I search recent PRs/issues for migration direction?

### 11.6 复制已有机制

表现：Adding a separate token-classification value model when existing FSDP/Megatron value-head paths exist.

为什么不符合本项目：Project prefers reusing existing backend/model bridge mechanisms.

维护者或历史 PR 证据：

- PR #6263 maintainer pointed to FSDP TRL `load_valuehead_model` and Megatron `make_value_model` instead of adding duplicate Qwen3.5 token-classification path.

正确做法：Extend existing model bridge/value-head hook with tests.

PR 自查问题：

- Is there an existing utility/bridge/registry for this?
- Am I duplicating backend-specific logic?

### 11.7 无测试、无文档、无生成配置同步

表现：Adding user-facing config/API/backend behavior without tests, docs, or generated config updates.

为什么不符合本项目：PR template and CI explicitly require tests/docs/config checks.

维护者或历史 PR 证据：

- `.github/PULL_REQUEST_TEMPLATE.md:31-40` requires pre-commit, docs, tests/CI explanation.
- `CONTRIBUTING.md:79-87` repeats docs/tests requirements.
- `.pre-commit-config.yaml:17-19` verifies generated trainer configs.
- Accepted PRs #6117/#6129/#6153 updated generated configs/docs/tests.

正确做法：Update docs/examples/config/tests in the same PR when user behavior changes.

PR 自查问题：

- Which test proves the behavior?
- Which docs/config examples changed?
- Did `autogen-trainer-cfg` run?

### 11.8 直接写设备特定 API

表现：Using `.cuda`, literal `"cuda"`, or `"nccl"` in normal code.

为什么不符合本项目：Device abstraction is CI-guarded to support CUDA/NPU and future accelerators.

维护者或历史 PR 证据：

- `tests/special_sanity/check_device_api_usage.py:15-19,67`.
- `.pre-commit-config.yaml:41-43`.
- PR #5943 device support discussion emphasized preserving NPU behavior and backend naming.

正确做法：Use `verl/utils/device.py`, add whitelist only with strong reason, and provide hardware/e2e evidence.

PR 自查问题：

- Would this code run on NPU/non-CUDA paths?
- Is there an existing device helper?

## 12. PR 设计说明模板

```markdown
# PR Design Explanation

## Problem

这个 PR 解决什么问题？为什么需要在 verl 主仓解决，而不是用户脚本、recipe、外部库或 verl-omni？

## Scope

这个 PR 修改什么？明确不修改什么？列出受影响模块：`trainer` / `rollout` / `engine` / `cfg` / `reward` / `data` / `ci` / `doc` 等。

## Existing Design Followed

遵循了项目中的哪些既有设计模式、模块边界或 contributor 习惯？

证据：

- 源码：...
- 测试：...
- 文档：...
- PR / Issue / Review：...
- Commit：...

## Alternatives Considered

考虑过哪些方案？为什么没有选择？

- Rejected: ... | reason ...
- Rejected: ... | reason ...

## Final Design

最终设计是什么？为什么符合 HybridFlow control/computation separation、DataProto contract、config API、backend adapter boundary？

## Compatibility

是否影响 public API、CLI/config、Hydra defaults、generated config、data format、`DataProto` semantics、error messages、performance, security, or hardware behavior？

- Breaking: yes/no
- Migration path:
- Deprecated fields:

## Tests

新增或修改了哪些测试？如何运行？

- CPU/unit:
- GPU/distributed/e2e:
- Docs/config/pre-commit:
- Manual experiment/benchmark, if CI cannot cover:

## Risk

维护者需要重点审查什么？

- Runtime/backends:
- Distributed world-size/resource mapping:
- Memory/performance:
- Compatibility:
- Rollback plan:
```

## 13. 证据索引

### Source / docs / tests

- 文档：`README.md#Project Overview` — project positioning.
- 文档：`CONTRIBUTING.md:25-40` — pre-commit and generated config checks.
- 文档：`CONTRIBUTING.md:42-58` — test/CI contribution guidance.
- 文档：`CONTRIBUTING.md:79-87` — PR docs/tests/checks requirements.
- 文档：`docs/hybrid_flow.rst:45-80` — two-level RL dataflow and separated flow design.
- 文档：`docs/hybrid_flow.rst:84-97` — single controller and multi-process workers.
- 文档：`docs/hybrid_flow.rst:166-177` — `@register` dispatch requiring `DataProto`.
- 文档：`docs/hybrid_flow.rst:180-207` — PPO main loop and backend/placement takeaways.
- 文档：`docs/single_controller.rst:33-52` — WorkerGroup / ResourcePool / ClassWithArgs motivation.
- 文档：`docs/single_controller.rst:127-145` — method binding is the heart of `single_controller`.
- 配置：`verl/trainer/config/ppo_trainer.yaml:1-47` — config format and Hydra composition.
- 配置：`verl/trainer/config/rollout/rollout.yaml:1-140` — rollout backend/perf config knobs.
- 源码：`verl/protocol.py::DataProto`, `check_consistency`, `union`.
- 源码：`verl/trainer/main_ppo.py::main`, `TaskRunner`, `add_ref_policy_worker`.
- 源码：`verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit`.
- 源码：`verl/workers/engine/base.py::BaseEngine`, `EngineRegistry`.
- 源码：`verl/workers/engine_workers.py::TrainingWorker`, `ActorRolloutRefWorker`, `update_weights`.
- 源码：`verl/workers/rollout/base.py::BaseRollout`, `get_rollout_class`.
- 源码：`verl/workers/reward_manager/registry.py::register`, `get_reward_manager_cls`.
- 源码：`verl/trainer/ppo/core_algos.py::register_adv_est`, `get_adv_estimator_fn`.
- 源码：`verl/trainer/ppo/reward.py::get_custom_reward_fn`, `load_reward_manager`.
- 源码：`verl/utils/dataset/rl_dataset.py::RLHFDataset`, `get_dataset_class`.
- 源码：`verl/utils/config.py::omega_conf_to_dataclass`, `validate_config`.
- 源码：`verl/base_config.py::BaseConfig`.
- 测试：`tests/test_protocol_on_cpu.py` — DataProto behavior.
- 测试：`tests/single_controller/base/test_decorator.py` — register/decorator behavior.
- 测试：`tests/trainer/ppo/test_core_algos_on_cpu.py` — advantage registry.
- 测试：`tests/workers/reward_manager/test_registry_on_cpu.py` — reward manager registry.
- 测试：`tests/special_sanity/check_device_api_usage.py` — device abstraction boundary.
- 测试：`tests/special_sanity/test_config_docs.py` — config documentation format.
- 测试：`tests/README.md:1-30` — test and workflow layout.
- CI：`.pre-commit-config.yaml:17-19` — generated trainer config check.
- CI：`.pre-commit-config.yaml:41-55` — device API and test structure checks.
- PR template：`.github/PULL_REQUEST_TEMPLATE.md:7-13,31-40` — title, breaking, docs/tests checklist.

### gh CLI commands

- `gh repo view volcengine/verl --json name,owner,description,defaultBranchRef,repositoryTopics,licenseInfo`
- `gh pr list -R volcengine/verl --state merged --limit 100 --json number,title,author,mergedAt,labels,files,additions,deletions`
- `gh pr list -R volcengine/verl --state closed --limit 100 --json number,title,author,closedAt,mergedAt,labels,comments,reviews`
- `gh issue list -R volcengine/verl --state all --limit 100 --json number,title,author,createdAt,closedAt,labels,comments`
- `gh pr view -R volcengine/verl 6117 --json number,title,state,comments,reviews,files`
- `gh pr view -R volcengine/verl 6150 --json number,title,state,comments,reviews,files`
- `gh pr view -R volcengine/verl 6234 --json number,title,state,comments,reviews,files`
- `gh pr view -R volcengine/verl 6147 --json number,title,state,comments,reviews,files`
- `gh pr view -R volcengine/verl 6081 --json number,title,state,comments,reviews,files`
- `gh api repos/verl-project/verl/pulls/6117/comments --paginate`
- `gh issue view -R volcengine/verl 6225 --json number,title,state,comments`

### PR / Issue / Review evidence

- PR：#6117 `[sglang] feat: SGLang Prefill-Decode disaggregated rollout` — accepted as existing `sglang` backend + config flag; tests/generated config included.
- Review comment：PR #6117 — maintainer: “Reuse `sglang` rollout backend.”
- PR：#6150 `[fsdp] fix: honor mixed_precision.param_dtype in forward_step autocast` — required unit test and GRPO convergence verification.
- Review comment：PR #6150 — maintainer requested small Qwen GRPO train/val reward verification, fp16 vs bf16.
- PR：#6234 `[perf] refactor: optimizes DataProtoFuture...` — closed because TransferQueue/native TensorDict direction supersedes it.
- PR：#5965 / #6079 — overlong prompt local truncation/early exit rejected as wrong layer/systemic solution.
- Issue：#6145 — multimodal overlong filtering must include image/video; large RLDataset filtering may be wrong layer.
- PR：#6147 — vLLM <= 0.8.2 compatibility rejected as too old.
- PR：#6081 — tokenizer workaround closed to wait for Transformers upstream fix.
- PR：#6074 — accepted breaking removal, but maintainer flagged unrelated notebook change.
- PR：#6263 — duplicate value model path rejected in favor of existing FSDP TRL value head / Megatron mcore bridge.
- Issue：#6225 — performance/memory proposal challenged for unsupported claims.
- PR：#5943 — hardware support review emphasized preserving device behavior and backend naming.

### Frequent contributor commit evidence

- Commit：`15371c91 [vllm] feat: make seed configurable and different among replicas`
- Commit：`7f4b76a8 [rollout] fix: remove dtype cast`
- Commit：`d9939add Revert ... vllm separation mode`
- Commit：`866a1eaa [trainer] feat: add new trainer with TransferQueue`
- Commit：`a9b024d4 [rollout,vllm] feat: split large weight into chunks...`
- Commit：`2edc06a4 [cfg] refactor: unify ppo_trainer and ppo_megatron_trainer config`
- Commit：`52437be1 [trainer] breaking: pass dataset as required args to SFTTrainer...`
- Commit：`5313d96f [CI] fix: add additional pre-commit test...`
- Commit：`d882b62b tests: add import utils tests`
- Commit：`b9d71f9a [megatron] fix: update patch for MLA flashattn forward`
- Commit：`0a902a94 [megatron] fix: patch support newer mcore version`
- Commit：`545f8998 [BREAKING] [perf] refactor: Profiler api refactor`
- Commit：`45b4ce91 [perf] feat: Add rollout longtail observation metrics`
- Commit：`e0c46e9c [vllm] feat: support abort generating requests in vllm server`
- Commit：`da214abc [vllm, rollout] feat: support reset prefix cache after abort`
- Commit：`3671d37f [algo] feat: Exception for agg_loss when dp_size > 1...`
- Commit：`b8d91ef8 [algo] fix: seq mean...`
- Commit：`ab070522 [worker] feat: custom reward_manager`
- Commit：`4de3ecf0 [cfg] refactor: add ActorConfig, EngineConfig...`
- Commit：`018f8dc5 [reward] fix: backward compatibility with old reward config`
- Commit：`392791bd [hardware] feat: Auto set device_name to npu for Ascend NPU`
- Commit：`044bbba2 [BREAKING] [misc] refactor: deprecate workers, migrate to engines (#6067)`

## 14. 未确认问题

- **未知：** Native TensorDict replacement timing for `DataProto` is not documented as a formal roadmap in this repository snapshot; evidence comes from maintainer PR comment (#6234) and code TODOs.
- **未知：** Exact acceptance bar for new full algorithm trainers varies by algorithm; current evidence supports starting in `recipe/` or registry-based PPO-like extension, but maintainers should confirm for major control-flow changes.
- **弱推断：** Existing frequent-contributor habits imply strong preference for small PRs and localized backend changes; exceptions exist for coordinated migrations.
