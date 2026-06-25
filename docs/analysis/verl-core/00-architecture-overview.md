# verl 核心包 — 源码架构分析

> 最后更新：2026-06-13
> 一句话定位：基于 Ray 单控制器架构的 LLM 强化学习后训练框架，支持可插拔训练引擎、推理引擎和奖励函数

## 1. 系统概述

verl 是一个面向大语言模型后训练（RLHF/RLAIF）的分布式强化学习框架。其核心解决的问题是：在分布式 GPU 集群上高效编排 Actor 生成、Critic 价值估计、Reference 策略对比、Reward 计算和 Policy 更新这一完整的 RL 训练循环。

在项目整体中，`verl/` 核心包是框架的运行时核心。它与 `examples/` 中的训练脚本和配置、`tests/` 中的测试套件、以及 `docs/` 中的文档共同构成完整项目。核心包实现了从数据加载到模型训练的全部运行时逻辑，外部只需提供模型权重路径、数据路径和 Hydra 配置即可启动训练。

| 指标 | 数值 |
|------|------|
| 生产文件数 | 325 个 |
| 生产代码行数 | 94,006 行 |
| 测试文件数 | 177 个 |
| 测试代码行数 | 34,215 行 |
| 支持的 RL 算法数 | 13 种（PPO, GRPO, RLOO, REINFORCE++, ReMax, OPO, GPG 等） |
| 支持的训练引擎数 | 6 种（FSDP, Megatron, TorchTitan, VeOmni, MindSpeed, AutoModel） |
| 支持的推理引擎数 | 4 种（vLLM, SGLang, TRT-LLM, HuggingFace） |
| 支持的 Checkpoint 传输后端数 | 5 种（NCCL, NIXL, HCCL, Mooncake, Kimi） |

## 2. 总体流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      RayPPOTrainer.fit() 主训练循环                       │
│                      (trainer/ppo/ray_trainer.py)                        │
└────────────┬────────────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────┐    ┌───────────────────┐    ┌──────────────────────┐
│  1. 数据加载       │───▶│  2. 生成 (Rollout)   │───▶│  3. 奖励计算          │
│                    │    │                     │    │                      │
│ 输入：JSONL 数据    │    │ 输入：DataProto      │    │ 输入：DataProto       │
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
│ [Driver CPU 计算   │    │  .train_mini_batch] │    │  Worker.update_actor]│
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
| 3 | Worker 配置体系 | [03-worker-config.md](03-worker-config.md) | `verl/workers/config/` | 2,822 | 所有 Worker 的类型化配置 dataclass 层次 |
| 4 | 训练引擎后端 | [04-engine-backends.md](04-engine-backends.md) | `verl/workers/engine/` | 6,897 | BaseEngine 抽象和 6 种可插拔训练引擎实现 |
| 5 | 统一 Worker 实现 | [05-engine-workers.md](05-engine-workers.md) | `verl/workers/engine_workers.py`, `verl/workers/utils/` | 1,201 | TrainingWorker 和 ActorRolloutRefWorker |
| 6 | 推理引擎适配 | [06-rollout-engines.md](06-rollout-engines.md) | `verl/workers/rollout/` | 8,333 | vLLM、SGLang、TRT-LLM 异步推理服务器 |
| 7 | Checkpoint 引擎 | [07-checkpoint-engine.md](07-checkpoint-engine.md) | `verl/checkpoint_engine/` | 2,611 | 训练权重到推理引擎的实时传输 |
| 8 | 核心 RL 算法 | [08-core-algorithms.md](08-core-algorithms.md) | `verl/trainer/ppo/core_algos.py` | 2,487 | 13 种优势估计器和 12 种策略损失函数 |
| 9 | 训练编排 | [09-training-orchestration.md](09-training-orchestration.md) | `verl/trainer/` | 11,942 | RayPPOTrainer 主循环、Hydra 配置、蒸馏、SFT |
| 10 | 模型层 | [10-models.md](10-models.md) | `verl/models/` | 9,478 | Megatron-Core 集成和 HuggingFace 模型补丁 |
| 11 | 工具库 | [11-utils.md](11-utils.md) | `verl/utils/` | 31,060 | FSDP 工具、数据集、检查点、性能分析、奖励评分等 |
| 12 | 实验性功能 | [12-experimental.md](12-experimental.md) | `verl/experimental/` | 8,952 | Agent 循环、全异步训练、离策略训练、训推分离 |
| 13 | 外围模块 | [13-peripheral.md](13-peripheral.md) | `verl/plugin/`, `verl/model_merger/`, `verl/tools/`, `verl/third_party/` | 5,464 | 平台插件、模型合并、工具调用、第三方依赖 |

## 4. 组件依赖拓扑

```
RayPPOTrainer (trainer/ppo/ray_trainer.py)
├── DataProto (protocol.py) ← 所有数据交换的载体
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
│       └── NCCLCheckpointEngine / NIXLCheckpointEngine / ...
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

1. **Ray 单控制器模式而非去中心化** — 一个 CPU 驱动进程通过 RPC 编排所有 GPU Worker，Worker 之间不直接通信。这使得训练循环的控制流集中在一个进程中，代码清晰且易于调试，代价是驱动进程成为潜在瓶颈（但由于驱动进程不处理 tensor 数据，实际中不是问题）。

2. **DataProto 而非直接传 Tensor** — 所有跨 Worker 数据交换都通过 `DataProto`（TensorDict + numpy dict + meta_info）。这统一了序列化/反序列化路径，支持自动 padding/chunking/concat，且让 `@register` 装饰器能透明地处理数据分发。

3. **混合引擎时分复用 GPU** — 训练和推理共享同一组 GPU：训练时 FSDP 持有权重，推理时通过 `CheckpointEngine` 将权重传输给 vLLM/SGLang，通过 sleep/wake 机制切换。这避免了为推理单独分配 GPU 的资源浪费。

4. **`@register` 装饰器驱动的分发模式** — Worker 方法通过 `@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)` 声明数据分发策略，WorkerGroup 在调用时自动处理数据切分、分发、执行、收集。这把分布式通信逻辑从业务代码中完全解耦。

5. **统一引擎抽象（BaseEngine + EngineRegistry）** — 6 种训练引擎（FSDP、Megatron、TorchTitan、VeOmni、MindSpeed、AutoModel）共享同一个 `BaseEngine` 接口，通过 `EngineRegistry` 注册。上层 `TrainingWorker` 和 `ActorRolloutRefWorker` 不关心具体引擎类型。

6. **优势估计器和策略损失的注册表模式** — `ADV_ESTIMATOR_REGISTRY` 和 `POLICY_LOSS_REGISTRY` 允许用户通过 `@register_adv_est("name")` 和 `@register_policy_loss("name")` 扩展自定义算法，无需修改框架代码。

7. **优势计算在驱动进程执行** — 优势（advantage）和回报（returns）在 CPU 驱动进程上计算，而非在 GPU Worker 上。因为这些计算是轻量级的纯 tensor 运算，在驱动进程执行可以避免一次不必要的 GPU kernel 启动。

8. **Agent Loop 管理多轮生成** — 生成不再是简单的单次 `model.generate()`，而是通过 `AgentLoopManager` 管理可能包含工具调用的多轮对话。这使得框架原生支持 Agent RL 训练。

9. **Checkpoint Engine 作为训练-推理桥梁** — 权重同步不通过文件系统，而是通过专门的 `CheckpointEngine`（支持 NCCL、NIXL 等高速传输协议）直接在 GPU 间传输。这大幅减少了权重同步延迟。

10. **experimental/ 作为功能孵化区** — 尚未稳定的功能（全异步训练、训推分离、Agent 循环等）放在 `experimental/` 下，但已被主训练循环直接导入使用。这些功能在稳定后会逐步迁移到主代码路径。

## 6. 分层隔离模型

| 数据 | 驱动层 (Driver) | Worker 层 | 推理层 (Rollout) |
|------|----------------|-----------|-----------------|
| 训练数据 | DataProto 形式存在，负责 chunk/concat/padding | TensorDict 形式存在，负责前向/反向 | 不涉及 |
| 模型权重 | 不接触权重 | FSDP/Megatron 分片持有 | vLLM/SGLang 持有副本 |
| 梯度 | 不接触梯度 | 在 Worker 内计算和通信 | 不涉及 |
| 优势/回报 | 在 Driver 上计算 | 接收结果用于损失计算 | 不涉及 |
| 生成结果 | 转发 DataProto | 不涉及 | 生成 token 序列 |
| 权重同步 | 触发 checkpoint_manager 调用 | 导出分片权重 | 通过 CheckpointEngine 接收 |

## 7. 已知架构注意事项

| # | 问题描述 | 位置 | 严重性 | 影响 |
|---|---------|------|--------|------|
| 1 | `RayPPOTrainer` 标记为 deprecated，将被 `PPOTrainer`（main_ppo_sync.py）替代 | `trainer/ppo/ray_trainer.py:283` | 中 | 当前两套训练循环并存，v0.8.0 后统一 |
| 2 | `print()` 调用未完全替换为 `logger` | 多处 | 低 | 不影响功能，影响日志管理 |
| 3 | experimental/ 中的模块已被主训练循环直接导入 | `trainer/ppo/ray_trainer.py:901-936` | 中 | experimental 标签名不副实，这些功能实际上是核心路径 |
| 4 | `fsdp_workers.py` 和 `megatron_workers.py` 在 CLAUDE.md 中被引用但已不存在 | CLAUDE.md | 低 | 文档过期，实际已重构为 `engine_workers.py` |
| 5 | `transferqueue_utils.py` 中 `BatchMeta`/`KVBatchMeta` 为条件导入的 mock 类 | `utils/transferqueue_utils.py` | 低 | 依赖外部 `transferqueue` 包，不可用时降级为 mock |
