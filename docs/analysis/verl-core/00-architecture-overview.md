# verl 核心包 — 源码架构分析

> 最后更新：2026-08-02（基准源码版本：上游 verl `e3573545`）
> 一句话定位：基于 Ray 单控制器架构的 LLM 强化学习后训练框架，支持可插拔训练引擎、推理引擎和奖励函数

## 1. 系统概述

verl 是一个面向大语言模型后训练（RLHF/RLAIF）的分布式强化学习框架。其核心解决的问题是：在分布式 GPU 集群上高效编排 Actor 生成、Critic 价值估计、Reference 策略对比、Reward 计算和 Policy 更新这一完整的 RL 训练循环。

在项目整体中，`verl/` 核心包是框架的运行时核心。它与 `examples/` 中的训练脚本和配置、`tests/` 中的测试套件、以及 `docs/` 中的文档共同构成完整项目。核心包实现了从数据加载到模型训练的全部运行时逻辑，外部只需提供模型权重路径、数据路径和 Hydra 配置即可启动训练。

| 指标 | 数值 |
|------|------|
| 生产文件数 | 360 个 |
| 生产代码行数 | 107,421 行 |
| 测试文件数 | 231 个 |
| 测试代码行数 | 47,326 行 |
| 支持的优势估计器数 | 14 种（GAE, GRPO, RLOO, REINFORCE++, ReMax, OPO, GPG 等，含向量化与 pass@k 变体） |
| 支持的训练引擎数 | 6 种（FSDP, Megatron, TorchTitan, VeOmni, MindSpeed, AutoModel） |
| 标准 rollout 工厂注册数 | 3 种（vLLM, SGLang, TRT-LLM；`_ROLLOUT_REGISTRY` 的 async `ServerAdapter`） |
| Checkpoint Engine 注册键数 | 6 个（`naive`, `nccl`, `nixl`, `kimi_ckpt_engine`, `mooncake`, `delta_sharded`；NCCL/HCCL 两个实现共享 `nccl` 键） |

> 注：`HFRollout`（`workers/rollout/hf_rollout.py:39`）与 `NaiveRollout`（`workers/rollout/naive/naive_rollout.py:36`）是保留的 legacy 类，但未实现 `BaseRollout` 的 `resume`/`update_weights`/`release` 抽象方法，也未进入注册表；因此不计入标准可选 rollout 数。

## 2. 总体流程图

当前入口 `trainer/main_ppo.py` 按 `config.trainer.use_v1`（默认 `true`，见 `trainer/config/ppo_trainer.yaml:222`）分派两条路径：**V1 路径**（默认）经 `TaskRunnerV1`（`main_ppo.py:104`）在 `main_ppo.py:138-156` 启用并初始化 TransferQueue，再由 `get_trainer_cls(config.trainer.v1.trainer_mode)` 选择 `PPOTrainerSync`、`PPOTrainerColocateAsync` 或 `PPOTrainerSeparateAsync`；抽象基类是 `PPOTrainer`（`trainer/ppo/v1/trainer_base.py:120`），并在 `trainer_base.py:144-190` 构建 `ReplayBuffer`/`ReplayBufferAsync`。**legacy 路径**（`use_v1=false`）经 `main_ppo_v0.TaskRunner` 进入已 `@deprecated` 的 `RayPPOTrainer`（`trainer/ppo/ray_trainer.py:286`）。下图只表示 legacy 的 batch 化十步流程；V1 是由 TransferQueue、ReplayBuffer 和 trainer mode 驱动的事件/批次混合流程，不能折叠为同一线性循环。

```
┌─────────────────────────────────────────────────────────────────────────┐
│       Legacy RayPPOTrainer.fit() · batch 化十步主训练循环                 │
│       （仅表示 use_v1=false 的兼容路径；V1 不适用此线性图）              │
└────────────┬────────────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────┐    ┌───────────────────┐    ┌──────────────────────┐
│  1. 数据加载       │───▶│  2. 生成 (Rollout)   │───▶│  3. 奖励计算          │
│                    │    │                     │    │                      │
│ 输入：Parquet/JSON/ │    │ 输入：DataProto      │    │ 输入：DataProto       │
│       JSONL 数据    │    │                     │    │                      │
│ 输出：DataProto    │    │     (prompts)       │    │     (prompts+resp)   │
│                    │    │ 输出：DataProto      │    │ 输出：DataProto       │
│ [StatefulDataLoader│    │     (prompts+resp)  │    │     (+token_scores)  │
│  + RLHFDataset]    │    │ [AgentLoopManager   │    │ [RewardLoopManager   │
│                    │    │  → LLMServerManager  │    │  + RewardManager]    │
│                    │    │  → vLLM/SGLang]      │    │                      │
└──────────────────┘    └───────────────────┘    └──────────┬───────────┘
                                                            │
             ┌──────────────────────────────────────────────┘
             ▼
┌──────────────────┐    ┌───────────────────┐    ┌──────────────────────┐
│  4. 旧策略对数概率  │───▶│  5. Ref 对数概率     │───▶│  6. 价值估计           │
│                    │    │                     │    │                      │
│ 输入：DataProto    │    │ 输入：DataProto      │    │ 输入：DataProto       │
│ 输出：DataProto    │    │ 输出：DataProto      │    │ 输出：DataProto       │
│     (+old_log_prob)│    │     (+ref_log_prob) │    │     (+values)        │
│ [ActorRolloutRef   │    │ [RefPolicy Worker   │    │ [Critic Worker       │
│  Worker]           │    │  or Actor w/ LoRA]  │    │  via TrainingWorker] │
└──────────────────┘    └───────────────────┘    └──────────┬───────────┘
                                                            │
             ┌──────────────────────────────────────────────┘
             ▼
┌──────────────────┐    ┌───────────────────┐    ┌──────────────────────┐
│  7. 优势计算       │───▶│  8. Critic 更新      │───▶│  9. Actor 更新        │
│                    │    │                     │    │                      │
│ 输入：DataProto    │    │ 输入：DataProto      │    │ 输入：DataProto       │
│ 输出：DataProto    │    │     (+advantages)   │    │     (+advantages)    │
│     (+advantages,  │    │ 输出：metrics        │    │ 输出：metrics         │
│      +returns)     │    │ [Critic Worker      │    │ [ActorRolloutRef     │
│ [legacy driver /   │    │  .train_mini_batch] │    │  Worker.update_actor]│
│  V1 trainer 控制流 │    │                     │    │                      │
│  core_algos.py]    │    │                     │    │                      │
└──────────────────┘    └───────────────────┘    └──────────┬───────────┘
                                                            │
                                                            ▼
                                               ┌──────────────────────┐
                                               │  10. 权重同步          │
                                               │                      │
                                               │ Trainer → Rollout    │
                                               │ [CheckpointEngine    │
                                               │  NCCL/NIXL/HCCL]    │
                                               └──────────────────────┘
```

## 3. 子模块导航表

| # | 模块 | 文档链接 | 源码位置 | 行数 | 一句话概述 |
|---|------|---------|---------|------|----------|
| 1 | 协议与配置基础 | [01-protocol-and-config.md](01-protocol-and-config.md) | `verl/protocol.py`, `verl/base_config.py` | 1,555 | DataProto 数据交换协议和 BaseConfig 冻结配置基类 |
| 2 | 单控制器架构 | [02-single-controller.md](02-single-controller.md) | `verl/single_controller/` | 2,251 | Ray 单控制器模式：Worker、Dispatch、WorkerGroup |
| 3 | Worker 配置体系 | [03-worker-config.md](03-worker-config.md) | `verl/workers/config/` | 2,987 | 所有 Worker 的类型化配置 dataclass 层次 |
| 4 | 训练引擎后端 | [04-engine-backends.md](04-engine-backends.md) | `verl/workers/engine/` | 8,427 | BaseEngine 抽象和 6 种可插拔训练引擎实现 |
| 5 | 统一 Worker 实现 | [05-engine-workers.md](05-engine-workers.md) | `verl/workers/engine_workers.py`, `verl/workers/utils/` | 1,265 | TrainingWorker 和 ActorRolloutRefWorker |
| 6 | 推理引擎适配 | [06-rollout-engines.md](06-rollout-engines.md) | `verl/workers/rollout/` | 9,618 | vLLM、SGLang、TRT-LLM 异步推理服务器 |
| 7 | Checkpoint 引擎 | [07-checkpoint-engine.md](07-checkpoint-engine.md) | `verl/checkpoint_engine/` | 3,654 | 训练权重到推理引擎的实时传输 |
| 8 | 核心 RL 算法 | [08-core-algorithms.md](08-core-algorithms.md) | `verl/trainer/ppo/core_algos.py` | 2,508 | 14 种优势估计器和 11 种策略损失函数 |
| 9 | 训练编排 | [09-training-orchestration.md](09-training-orchestration.md) | `verl/trainer/` | 13,971 | PPOTrainer（V1，默认）/ RayPPOTrainer（legacy）主循环、Hydra 配置、蒸馏、SFT |
| 10 | 模型层 | [10-models.md](10-models.md) | `verl/models/` | 10,128 | Megatron-Core 集成和 HuggingFace 模型补丁 |
| 11 | 工具库 | [11-utils.md](11-utils.md) | `verl/utils/` | 35,745 | FSDP 工具、数据集、检查点、性能分析、奖励评分等 |
| 12 | 实验性功能 | [12-experimental.md](12-experimental.md) | `verl/experimental/` | 11,147 | Agent 循环、全异步训练、离策略训练、训推分离 |
| 13 | 外围模块 | [13-peripheral.md](13-peripheral.md) | `verl/plugin/`, `verl/model_merger/`, `verl/tools/`, `verl/third_party/` | 5,642 | 平台插件、模型合并、工具调用、第三方依赖 |

## 4. 组件依赖拓扑

```
PPO 训练驱动 (trainer/main_ppo.py, 按 trainer.use_v1 分派)
│   ├─ V1（默认, use_v1=true）: TaskRunnerV1 → PPOTrainer (trainer/ppo/v1/trainer_base.py:120, ABC)
│   │   ├─ get_trainer_cls(trainer_mode) → PPOTrainerSync / PPOTrainerColocateAsync / PPOTrainerSeparateAsync
│   │   └─ TransferQueue + ReplayBuffer / ReplayBufferAsync（trainer_base.py:144-190）
│   └─ legacy（use_v1=false）: main_ppo_v0.TaskRunner → RayPPOTrainer (trainer/ppo/ray_trainer.py:286, @deprecated)
├── DataProto (protocol.py) ← legacy/接口层的批数据载体；V1 训练批次经 TransferQueue 的 KVBatchMeta 交换
├── RayWorkerGroup (single_controller/ray/base.py) ← 分布式工作者编排
│   ├── Worker (single_controller/base/worker.py)
│   ├── Dispatch + Execute (single_controller/base/decorator.py)
│   └── ResourcePoolManager (single_controller/ray/base.py)
├── ActorRolloutRefWorker (workers/engine_workers.py)
│   ├── BaseEngine 子类 (workers/engine/{fsdp,megatron,...}/)
│   │   └── EngineRegistry (workers/engine/base.py) ← 引擎注册表
│   ├── BaseRollout 子类 (workers/rollout/{vllm,sglang,...}/)
│   │   └── RolloutReplica (workers/rollout/replica.py)
│   └── CheckpointEngine (checkpoint_engine/base.py) ← 权重传输
│       └── NCCLCheckpointEngine / NIXLCheckpointEngine / DeltaShardedCheckpointEngine / ...
├── TrainingWorker (workers/engine_workers.py) ← Critic Worker
├── AgentLoopManager (experimental/agent_loop/) ← 生成管理
│   └── LLMServerManager (workers/rollout/llm_server.py)
├── RewardLoopManager (experimental/reward_loop/) ← 奖励计算
│   └── RewardManager 子类 (workers/reward_manager/)
├── core_algos (trainer/ppo/core_algos.py) ← 优势/损失计算
│   ├── AdvantageEstimator (GAE, GRPO, RLOO, ...)
│   └── POLICY_LOSS_REGISTRY (PPO clip, DPPO, GSPO, ...)
└── Tracking (utils/tracking.py) ← 日志记录 (W&B, MLflow, ...)
```

## 5. 关键架构决策

1. **Ray 单控制器模式而非去中心化** — 一个 CPU 驱动进程通过 RPC 集中编排 GPU Worker 的业务调用与数据收集；这不是对 Engine/FSDP/Megatron 内部 collective 的绝对禁止。Driver/V1 trainer 也会处理 TensorDict 与优势计算，而 FSDP/Megatron Worker 内部仍执行 `all_reduce` 等数据面通信，因此这里描述的是控制面集中，而非“driver 不处理 tensor”。

2. **分层数据交换协议** — legacy 路径和通用 Worker 接口使用 `DataProto`（TensorDict + numpy dict + meta_info），支持自动 padding/chunking/concat，并由 `@register` 装饰器处理分发；V1 训练路径则在 TransferQueue 中以 `KVBatchMeta`/底层 TensorDict 交换批次（`ppo/v1/trainer_base.py:1595-1603`），不能把 DataProto 概括为所有路径的唯一载体。

3. **混合引擎时分复用 GPU** — 在 hybrid/colocate 模式下，训练和推理可共享同一组 GPU：训练时 FSDP 持有权重，推理时通过 `CheckpointEngine` 将权重传输给 vLLM/SGLang，并用 sleep/wake 机制切换；`separate_async` 另建 standalone rollout 资源池（`ppo/v1/trainer_separate_async.py:77-93`），不应概括为所有模式都共享 GPU。

4. **`@register` 装饰器驱动的分发模式** — Worker 方法通过 `@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)` 声明数据分发策略，WorkerGroup 在调用时自动处理数据切分、分发、执行、收集。这把 Worker 方法的业务分发/收集逻辑与调用方解耦；引擎内部 collective 仍由各后端自行负责。

5. **统一引擎抽象（BaseEngine + EngineRegistry）** — 6 种训练引擎（FSDP、Megatron、TorchTitan、VeOmni、MindSpeed、AutoModel）共享同一个 `BaseEngine` 接口，通过 `EngineRegistry` 注册。上层 `TrainingWorker` 和 `ActorRolloutRefWorker` 不关心具体引擎类型。

6. **优势估计器和策略损失的注册表模式** — `ADV_ESTIMATOR_REGISTRY` 和 `POLICY_LOSS_REGISTRY` 允许用户通过 `@register_adv_est("name")` 和 `@register_policy_loss("name")` 扩展自定义算法，无需修改框架代码。

7. **优势计算位于训练控制流** — legacy `RayPPOTrainer` 在 driver process 调用 `compute_advantage()`（`ray_trainer.py:1637-1642`）；V1 则在 `PPOTrainer._compute_advantage()`（`trainer_base.py:1595`，调用点 `:574-576`）中从 TransferQueue 取出批次后计算。源码不保证统一的 CPU 设备位置，因此不据此宣称避免 GPU kernel。

8. **Agent Loop 管理多轮生成** — 生成不再是简单的单次 `model.generate()`，而是通过 `AgentLoopManager` 管理可能包含工具调用的多轮对话。这使得框架原生支持 Agent RL 训练。

9. **Checkpoint Engine 作为训练-推理桥梁** — 权重同步不通过文件系统，而是通过专门的 `CheckpointEngine`（支持 NCCL、NIXL 等高速传输协议）直接在 GPU 间传输。这大幅减少了权重同步延迟。

10. **experimental/ 作为功能孵化区** — 尚未稳定的功能（全异步训练、训推分离、Agent 循环等）放在 `experimental/` 下，但已被主训练循环直接导入使用。这些功能在稳定后会逐步迁移到主代码路径。

## 6. 分层隔离模型

| 数据 | 驱动层 (Driver) | Worker 层 | 推理层 (Rollout) |
|------|----------------|-----------|-----------------|
| 训练数据 | legacy 使用 DataProto；V1 使用 TransferQueue/KVBatchMeta | TensorDict 形式存在，负责前向/反向 | 不涉及 |
| 模型权重 | 不接触权重 | FSDP/Megatron 分片持有 | hybrid/colocate 由 vLLM/SGLang 持有副本；separate_async 使用独立 rollout 资源池 |
| 梯度 | 不接触梯度 | 在 Worker 内计算和通信 | 不涉及 |
| 优势/回报 | legacy driver 或 V1 PPOTrainer 控制流中计算 | 接收结果用于损失计算 | 不涉及 |
| 生成结果 | 转发 DataProto | 不涉及 | 生成 token 序列 |
| 权重同步 | 触发 checkpoint_manager 调用 | 导出分片权重 | 通过 CheckpointEngine 接收 |

## 7. 已知架构注意事项

| # | 问题描述 | 位置 | 严重性 | 影响 |
|---|---------|------|--------|------|
| 1 | `RayPPOTrainer` 已标记 `@deprecated`；legacy 入口 `main_ppo_v0.py` 与该类均计划 v0.9.0 移除。替代者为 V1 路径的 `PPOTrainer`（ABC，由原 `main_ppo_sync.py` rename 而来）；`main_ppo.py` 入口本身未 deprecated | `ray_trainer.py:285` (`@deprecated`)、`:286` (`class RayPPOTrainer`)、`trainer/ppo/v1/trainer_base.py:120` (`class PPOTrainer`) | 中 | 当前 V1（默认）与 legacy 两套训练循环并存，v0.9.0 后统一 |
| 2 | `print()` 调用未完全替换为 `logger` | 多处 | 低 | 不影响功能，影响日志管理 |
| 3 | experimental/ 中的模块已被主训练循环直接导入 | `trainer/ppo/ray_trainer.py:910-944`（`RewardLoopManager`@910、`MultiTeacherModelManager`@927、`AgentLoopManager`@944）| 中 | experimental 标签名不副实，这些功能实际上是核心路径 |
| 4 | `CLAUDE.md` 曾引用已重构掉的 `fsdp_workers.py`、`megatron_workers.py` 和 `sharding_manager/` | CLAUDE.md | 已修复 | 2026-08-02 已更正为 `engine_workers.py` 与 `engine/`，保留此项作为修复记录 |
| 5 | `transferqueue_utils.py` 中 `BatchMeta`/`KVBatchMeta` 为条件导入的 mock 类 | `utils/transferqueue_utils.py` | 低 | 依赖外部 `transferqueue` 包，不可用时降级为 mock |

## 8. 审计收敛报告

- 初始全量审计：76 严重 / 60 重要 / 47 次要，共 183 项；按用户选择全部处理。
- 定量基线：发现 4 项文件数/行数/注册表计数异常，均已逐项用 `wc -l` 或源码核对修正；其余数量、字段、常量与行号通过核对。
- 结构与代码审计第 1 轮：架构审查 1 严重 / 9 重要 / 5 次要；代码审查 9 严重 / 9 重要 / 0 次要；全部修正。
- 结构与代码审计第 2 轮：架构审查 0 严重 / 0 重要 / 3 次要；代码审查在最终修正前发现 1 严重 / 1 重要 / 1 次要，修正后复核为 0 / 0 / 0；3 项架构次要项亦已修正。
- 最终状态：通过。架构与代码复核均为 0 严重 / 0 重要 / 0 次要；`git diff --check` 通过。未运行构建或测试，未修改源码实现与用户保留的未跟踪文件。
