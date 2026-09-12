# 12 - verl/experimental/ 实验性子模块架构文档

> **源码位置**: `verl/experimental/` | **文件数**: 45（含 unittest） | **生产文件**: 43 | **总行数**: 11,147
>
> **最后更新**: 2026-08-02 | **基准源码**: 上游 `e3573545`
>
> 实验性模块，包含 6 个子系统：agent_loop（多轮工具调用）、fully_async_policy（全异步训练）、one_step_off_policy（一步前瞻）、separation（训推分离基础设施）、reward_loop（异步奖励计算）、teacher_loop（多教师蒸馏）。

---

## 1. 模块定位

`verl/experimental/` 是 verl 的实验性扩展层，承载三类前沿功能：

1. **异步训练模式**：从同步 PPO 到全异步 RL 的演进链路
2. **Agent 框架**：多轮工具调用的状态机和协程执行引擎
3. **奖励/蒸馏扩展**：流式奖励计算和多教师模型蒸馏

这些模块与主线 `verl/trainer/ppo/ray_trainer.py` 的 `RayPPOTrainer` 共享大量基础设施，通过继承和模板方法模式实现差异化。

---

## 2. 文件清单与行数

### 2.1 agent_loop/（6 文件，2,907 行）

| 文件 | 行数 | 职责 |
|------|------|------|
| `agent_loop.py` | 1,283 | `AgentLoopBase`（L206）：协程抽象基类；`AgentLoopWorker`（L497）：批量并发执行；`AgentLoopManager`（L1161）：Ray 分布式调度 |
| `tool_agent_loop.py` | 552 | `ToolAgentLoop`（L100）：ReAct 模式的多轮工具调用状态机；`AgentState` 枚举（L49） |
| `tool_parser.py` | 816 | `ToolParser`（L48）：抽象解析器基类；9 种格式实现（hermes/gpt-oss/qwen3_coder/glm/seed/minimax/kimi/deepseek_v4/gemma4）|
| `single_turn_agent_loop.py` | 115 | 单轮 agent loop 实现 |
| `utils.py` | 108 | 辅助函数（配置路径解析、GPT-OSS 工具响应格式化） |
| `__init__.py` | 33 | 模块入口 |

### 2.2 fully_async_policy/（14 文件，4,335 行）

| 文件 | 行数 | 职责 |
|------|------|------|
| `fully_async_trainer.py` | 1,049 | `FullyAsyncTrainer`（L54）：全异步训练器，从 MessageQueue 获取样本，支持 staleness 管理 |
| `fully_async_rollouter.py` | 1,392 | `FullyAsyncRollouter`：异步 rollout 产生器，向 MessageQueue 发送样本 |
| `message_queue.py` | 242 | `MessageQueue`（L27）：Ray actor 消息队列；`MessageQueueClient`（L180）：异步客户端 |
| `detach_utils.py` | 466 | `MetricsAggregator`、`assemble_batch_from_rollout_samples` |
| `fully_async_main.py` | 243 | 全异步训练入口脚本 |
| `dynamic_schedule/` | 740 | 动态调度子包（6 文件），详见 §2.2.1 |
| `unittest/` | 190 | 流式测试（2 文件）：`simple_streaming_demo.py` 等 |
| `__init__.py` | 13 | 模块入口 |

#### 2.2.1 dynamic_schedule/ 动态调度子包（6 文件，740 行）

动态调度子系统，控制全异步训练中 hybrid rollout 副本的激活/停用与资源分配（另含 `README.md` / `README_zh.md`）。

| 文件 | 行数 | 公开类 / 职责 |
|------|------|------|
| `base.py` | 174 | `DynamicScheduleContext`（L57）：调度决策统一上下文；`DynamicSchedulePolicyBase`（L109）：策略抽象基类 + 注册表 |
| `default_policy.py` | 235 | `DefaultDynamicSchedulePolicy`（L37）：默认策略，hybrid 激活时停用，自适应比例 |
| `fixed_ratio_policy.py` | 89 | `FixedRatioDynamicSchedulePolicy`（L23）：固定比例策略，`deactivate_ratio` 不更新 |
| `static_fully_async_policy.py` | 54 | `StaticFullyAsyncPolicy`（L21）：静态全异步策略 |
| `dynamic_resource_controller.py` | 160 | `DynamicResourceController`（L51）：hybrid 副本生命周期管理（STANDALONE_ONLY <-> HYBRID_ACTIVE） |
| `__init__.py` | 28 | 导出已导入的策略类与注册表（不含 `FixedRatioDynamicSchedulePolicy`） |

`DynamicSchedulePolicyBase` 的 `build_policy()`（`dynamic_schedule/base.py:49-53`）只实例化已导入并触发 `@register_policy` 的策略。当前默认 `dynamic_schedule/__init__.py:15-18` 导入 `default` 与 `static_fully_async`，但没有导入 `fixed_ratio_policy.py`；全仓也没有其他默认导入，因此配置写入 `fixed_ratio` 时会在 `build_policy()` 处触发 `KeyError`，除非入口先手动导入该模块。动态调度控制器本身由 `FullyAsyncTrainer._setup_dynamic_resource_controller()`（`fully_async_trainer.py:302-327`）在 `use_dynamic_resource_scheduling=True` 时创建。

### 2.3 one_step_off_policy/（3 文件，553 行）

| 文件 | 行数 | 职责 |
|------|------|------|
| `ray_trainer.py` | 408 | `OneStepOffRayTrainer`（L49）：一步前瞻异步 rollout 训练器 |
| `main_ppo.py` | 129 | Hydra 入口 |
| `__init__.py` | 16 | 模块入口 |

### 2.4 separation/（4 文件，1,030 行）

| 文件 | 行数 | 职责 |
|------|------|------|
| `ray_trainer.py` | 763 | `SeparateRayPPOTrainer`（L52）：训推分离基类，定义模板方法 |
| `engine_workers.py` | 159 | 引擎 worker 初始化 |
| `utils.py` | 95 | 分离模式工具函数 |
| `__init__.py` | 13 | 模块入口 |

### 2.5 reward_loop/（14 文件，1,938 行）

| 文件 | 行数 | 职责 |
|------|------|------|
| `reward_loop.py` | 362 | `RewardLoopWorker`（L93）：奖励计算 worker |
| `reward_model.py` | 139 | `RewardModelManager`：奖励模型路由管理 |
| `reward_manager/base.py` | 82 | `RewardManagerBase`（L34）：奖励管理器抽象基类 |
| `reward_manager/naive.py` | 99 | `NaiveRewardManager`：规则奖励 |
| `reward_manager/dapo.py` | 119 | `DAPORewardManager`：DAPO 算法奖励 |
| `reward_manager/gdpo.py` | 92 | `GDPORewardManager`：GDPO 算法奖励 |
| `reward_manager/limited.py` | 540 | `RateLimitedRewardManager`：限速奖励 |
| `reward_manager/remote.py` | 130 | `RemoteRewardManager`：远程奖励模型 |
| `reward_manager/registry.py` | 53 | 注册表（`@register` 装饰器） |
| `router/naive_router.py` | 188 | `NaiveRouter`：简单奖励路由 |
| `router/inner_sglang_router.py` | 73 | SGLang 内部路由：`launch_router_process`（L30）启动函数（无类定义） |
| `reward_manager/__init__.py` | 30 | RewardManager 导出 |
| `router/__init__.py` | 13 | 路由器导出 |
| `__init__.py` | 18 | reward_loop 模块入口 |

### 2.6 teacher_loop/（3 文件，371 行）

| 文件 | 行数 | 职责 |
|------|------|------|
| `teacher_manager.py` | 141 | `AsyncTeacherLLMServerManager`（L78）：异步教师模型 logprob 计算 |
| `teacher_model.py` | 213 | `MultiTeacherModelManager`：多教师模型管理，按 teacher_key 路由 |
| `__init__.py` | 17 | 模块入口 |

---

## 3. 继承层次与异步模式演进

### 3.1 Trainer 继承链

```
RayPPOTrainer (verl/trainer/ppo/ray_trainer.py)
  │
  ├─── SeparateRayPPOTrainer (separation/ray_trainer.py, L52)
  │      │  公共基类：训推分离基础设施
  │      │  模板方法：init_workers() 拆分为 _init_resource_pools() / 
  │      │            _create_worker_classes() / _init_worker_groups() / _init_models()
  │      │
  │      ├─── OneStepOffRayTrainer (one_step_off_policy/ray_trainer.py, L49)
  │      │      一步前瞻：训练使用上一步 rollout 结果，当前步同时生成新 rollout
  │      │
  │      └─── FullyAsyncTrainer (fully_async_policy/fully_async_trainer.py, L54)
  │             全异步：训练从 MessageQueue 拉取样本，rollout 独立推送
  │             @ray.remote(num_cpus=10) 作为独立 Ray actor 运行
```

### 3.2 异步模式演进路线

```
同步 PPO (RayPPOTrainer)
  │  每步：rollout -> reward -> ref_log_prob -> advantage -> actor_update -> critic_update
  │  特点：GPU 在 rollout 和训练间交替空闲
  │
  ├─> 一步前瞻 (OneStepOffRayTrainer)
  │     每步：使用上一步的 rollout 训练，同时启动当前步 rollout
  │     特点：训练和 rollout 部分重叠，staleness = 1 step
  │
  └─> 全异步 (FullyAsyncTrainer + FullyAsyncRollouter)
        训练和 rollout 完全解耦，通过 MessageQueue 通信
        特点：最大 GPU 利用率，staleness 由队列深度控制
```

### 3.3 全异步架构

```
FullyAsyncRollouter (Ray actor)           MessageQueue (Ray actor)         FullyAsyncTrainer (Ray actor)
┌─────────────────────┐                   ┌────────────────────┐          ┌──────────────────────┐
│  持续生成 rollout    │                   │  deque(maxlen=N)   │          │  从队列拉取样本       │
│                     │  put_sample()     │                    │          │  训练 + 参数同步      │
│  AgentLoopManager   │ ──────────────>   │  asyncio.Lock +    │          │                      │
│  LLMServerManager   │                   │  Condition 变量    │  get()   │  MetricsAggregator   │
│                     │                   │                    │ <──────  │  CheckpointManager   │
└─────────────────────┘                   │  统计：produced,   │          └──────────────────────┘
                                          │  consumed, dropped │
                                          └────────────────────┘
```

`MessageQueue`（L27，`message_queue.py`）是核心解耦组件：
- **底层容器**：`collections.deque(maxlen=max_queue_size)`
- **并发控制**：`asyncio.Lock` + `asyncio.Condition`
- **有界丢弃/过载降级**：队列满时不阻塞生产者，而是丢弃最旧样本（FIFO 溢出）
- **验证队列**：独立的 `val_queue` 用于验证数据

---

## 4. Agent Loop 状态机

### 4.1 核心类关系

```
AgentLoopBase (agent_loop.py, L206)
  │  抽象基类：定义 run() 协程接口
  │  提供：apply_chat_template(), process_multi_modal_info()
  │
  ├─── SingleTurnAgentLoop
  │      单轮生成，无工具调用
  │
  └─── ToolAgentLoop (tool_agent_loop.py, L100)
         @register("tool_agent")
         多轮 ReAct 模式：生成 -> 解析工具调用 -> 执行工具 -> 拼接响应 -> 继续生成

AgentLoopWorker (agent_loop.py, L497)
  │  批量并发：asyncio.gather() 同时执行多个 agent loop
  │  后处理：_pad_token_ids(), _compute_multi_modal_inputs(), _compute_score()
  │
AgentLoopManager (agent_loop.py, L1161)
  │  分布式调度：多个 AgentLoopWorker 作为 Ray actor
  │  输入拆分 -> 并发执行 -> 结果合并 -> 性能指标
```

### 4.2 ToolAgentLoop 状态机

```
AgentState.PENDING
  │  准备 prompt_ids (apply_chat_template)
  │
  v
AgentState.GENERATING
  │  调用 LLM server 生成 response
  │  累积 response_ids, response_mask (mask=1)
  │  检查终止条件：
  │    - response_length 超限
  │    - assistant_turns >= max_assistant_turns
  │    - user_turns >= max_user_turns
  │  解析工具调用 (ToolParser.extract_tool_calls)
  │
  ├── 有工具调用 ────> AgentState.PROCESSING_TOOLS
  │                      │  asyncio.gather() 并行执行工具
  │                      │  拼接工具响应到 prompt_ids
  │                      │  response_mask += [0] * len(tool_response)
  │                      │  检查 response_length
  │                      │
  │                      └── 未超限 ────> AgentState.GENERATING (循环)
  │                          超限   ────> AgentState.TERMINATED
  │
  └── 无工具调用 ────> AgentState.TERMINATED
```

关键数据流：
- `prompt_ids`：持续追加，包含所有历史 token
- `response_mask`：LLM 生成 token 为 1，工具响应 token 为 0
- `response_logprobs`：仅 LLM 生成部分有值，工具响应部分填 0.0

`AgentData`（`tool_agent_loop.py:56-96`）承载该状态机的可变会话状态：消息与多模态输入、`prompt_ids`/`response_ids`、`response_mask`/`response_logprobs`、turn 计数、`tool_calls`、路由专家信息和可扩展的 `extra_fields`。工具执行可通过它读取完整历史并写入会话级附加数据。

### 4.3 ToolParser 注册表

`ToolParser`（tool_parser.py, L48）使用类级别 `_registry` 字典管理解析器：

| 格式名称（注册键） | 适用模型 | 特点 |
|----------|----------|------|
| `hermes` | Hermes/通用 | 标准 XML 标签 |
| `gpt-oss` | GPT 开源变体 | 手动格式化响应 |
| `qwen3_coder` | Qwen3-Coder/Qwen3.5（类 `Qwen3XMLToolParser`） | XML 格式，依赖 EOS 停止 |
| `glm` | GLM | GLM XML 风格函数调用 |
| `seed` | ByteDance Seed | Seed XML 风格函数调用 |
| `minimax` | MiniMax | MiniMax XML 风格函数调用 |
| `kimi` | Kimi K2 系列 | 特殊 token 函数调用 |
| `deepseek_v4` | DeepSeek-V4 | DSML 函数调用格式 |
| `gemma4` | Google Gemma 4 | 自定义标签，需显式 stop_token_ids |

---

## 5. Reward Loop 流式与批量模式

### 5.1 架构概览

```
RewardLoopManager
  │  管理 RewardLoopWorker 实例
  │
  ├── RewardLoopWorker (reward_loop.py, L93)
  │     │  核心：_init_reward_fn() 加载奖励函数
  │     │  模式：
  │     │  ├── 自定义奖励函数 -> 直接调用
  │     │  ├── 规则奖励 -> default_compute_score
  │     │  └── 模型奖励 -> 请求 RewardModelManager
  │     │
  │     └── 使用 RewardManagerBase 子类处理具体逻辑
  │
  └── RewardModelManager
        │  管理奖励模型推理引擎
        │  通过 NaiveRouter 路由请求
        └── NaiveRouter / launch_router_process（inner_sglang_router.py:30 函数，无类定义）
```

### 5.2 RewardManager 层次

```
RewardManagerBase (reward_manager/base.py, L34)
  │  抽象方法：run_single(data: DataProto)
  │  通用方法：assemble_rm_scores() — 将标量分数放置到 response 最后一个有效 token 位置
  │
  ├── NaiveRewardManager    — 规则奖励（@register("naive")）
  ├── DAPORewardManager     — DAPO 算法特定奖励处理
  ├── GDPORewardManager     — GDPO 算法特定奖励处理
  ├── RateLimitedRewardManager — 限速奖励（防止过载）
  └── RemoteRewardManager   — 远程奖励模型调用
```

流式模式（per-sample）：Agent Loop 中 `_compute_score()` 异步调用 `RewardLoopWorker.compute_score.remote()`，每个样本独立计算奖励。

批量模式：传统 `RayPPOTrainer` 中批量调用奖励函数，所有样本一次性计算。

---

## 6. Teacher Loop 多教师蒸馏

### 6.1 核心类

```
AsyncTeacherLLMServerManager (teacher_manager.py, L78)
  │  初始化：从 DistillationConfig 获取 teacher_models 配置
  │  路由：按 teacher_key（如 "model_name"）选择对应教师模型
  │  核心方法：compute_teacher_logprobs_single()
  │    - 将 student 生成的 sequence_ids 发送给教师模型
  │    - 获取 teacher 的 prompt_logprobs
  │    - 返回 (teacher_ids, teacher_logprobs) 用于蒸馏损失
  │
MultiTeacherModelManager (teacher_model.py)
  │  管理多个教师模型实例
  │  按 routing_key 将请求分发到不同教师
```

### 6.2 数据流

```
AgentLoopWorker._compute_teacher_logprobs()
  │  检查 distillation_enabled 且非 validate
  │  从 sample_kwargs 获取 routing_key
  │
  v
AsyncTeacherLLMServerManager.compute_teacher_logprobs_single()
  │  选择教师模型 (routing_key -> teacher_model_config)
  │  构造 sampling_params (temperature=1, prompt_logprobs=topk)
  │  调用 teacher_client.generate() 获取 logprobs
  │
  v
_pad_teacher_outputs()
  │  将 teacher_ids, teacher_logprobs 填充到与 prompt/response 对齐的形状
  │  返回给 AgentLoopWorker 合并到输出 batch
```

---

## 7. 设计模式与架构决策

### 7.1 模板方法模式（Separation）

`SeparateRayPPOTrainer` 将 `init_workers()` 拆分为 5 个步骤：
1. `_init_resource_pools()` — 创建 Ray 资源池
2. `_create_worker_classes()` — 创建 worker 类（由子类实现 `_create_actor_rollout_classes()`）
3. `_init_worker_groups()` — 初始化 worker 组
4. `_init_models()` — 初始化模型
5. `_init_async_rollout_manager()` — 初始化异步 rollout 管理器

这允许 `OneStepOffRayTrainer` 和 `FullyAsyncTrainer` 只覆写差异化步骤。

### 7.2 协程 + Ray 混合并发

Agent Loop 使用两层并发：
- **进程间**：`AgentLoopManager` 通过 Ray actor 分发到多个 `AgentLoopWorker`
- **进程内**：每个 `AgentLoopWorker` 使用 `asyncio.gather()` 并发执行多个 agent loop 协程

这种设计允许 IO 密集的工具调用不阻塞其他样本的生成。

### 7.3 生产者-消费者解耦（全异步）

`MessageQueue` 作为 Ray actor 独立运行，使用 `asyncio.Lock` + `asyncio.Condition` 实现线程安全的异步等待：
- 生产者（Rollouter）无需等待训练完成
- 消费者（Trainer）无需等待 rollout 完成
- 使用有界 `deque(maxlen=N)` 做丢弃式过载降级：队列满时丢弃最旧样本，不阻塞生产者

### 7.4 动态资源调度的状态机与安全顺序

当 `async_training.use_dynamic_resource_scheduling=True` 时，`FullyAsyncTrainer` 在 `_setup_dynamic_resource_controller()`（`fully_async_trainer.py:302-327`）按配置名调用 `build_policy()`；每个训练 step 在 `fully_async_trainer.py:560-595` 先询问策略 `should_deactivate()`，必要时等待样本阈值，再执行停用。

`DynamicResourceController` 的状态为 `STANDALONE_ONLY ↔ HYBRID_ACTIVE`：

1. **停用**（`dynamic_resource_controller.py:135-155`）：先从 load balancer 移除 hybrid replicas，阻止重试重新路由；再 abort 在途请求；最后 sleep/release KV cache 与权重，归还训练显存。
2. **权重同步**（`:97-110`）：先 abort，调用 hybrid checkpoint manager 的 naive backend 更新权重，再恢复 generation。
3. **激活**（`:118-133`）：权重同步完成后把 hybrid replicas 加回 load balancer，再恢复 generation；成功后切换为 `HYBRID_ACTIVE`。

若没有 hybrid replicas，控制器会跳过相应转换；策略名未注册（例如当前默认导入遗漏的 `fixed_ratio`）则在 `build_policy()` 处 fail-fast 为 `KeyError`，不是静默回退。

### 7.5 注册表模式

三处使用注册表：
1. `_agent_loop_registry`（agent_loop.py）：`@register("tool_agent")` 注册 agent loop
2. `ToolParser._registry`（tool_parser.py）：注册工具解析器格式
3. `reward_manager/registry.py`：`@register("naive")` 注册奖励管理器

---

## 8. 依赖关系

```
verl/experimental/ 的依赖方向：

外部依赖：
├── verl/trainer/ppo/ray_trainer.py  — SeparateRayPPOTrainer 继承 RayPPOTrainer
├── verl/trainer/ppo/core_algos.py   — KL 控制器、优势估计器
├── verl/protocol.py                 — DataProto 数据交换
├── verl/workers/rollout/            — LLMServerClient, LLMServerManager
├── verl/tools/                      — BaseTool, FunctionTool, ToolSchema
├── verl/utils/config.py             — omega_conf_to_dataclass
├── verl/utils/reward_score/         — default_compute_score
└── verl/utils/ray_utils.py          — auto_await, get_event_loop

被依赖：
├── verl/trainer/main_*.py           — 入口脚本引用异步训练器
├── verl/workers/rollout/            — agent_loop 集成
└── examples/                        — 示例配置引用实验性功能

内部依赖（子模块间）：
separation/ <── one_step_off_policy/ （继承）
separation/ <── fully_async_policy/  （继承）
agent_loop/ <── reward_loop/         （流式奖励计算）
agent_loop/ <── teacher_loop/        （教师蒸馏）
```
