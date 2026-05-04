# PR #6051 Multi-Teacher On-Policy Distillation (MOPD) 深度源码分析

> **PR**: [#6051 — [trainer,cfg,rollout,algo] feat: Multi-Teacher OPD](https://github.com/volcengine/verl/pull/6051)
> **Author**: JacobHelwig
> **Base PR**: #5834 (Single-Teacher OPD standalone-only refactor)
> **Additions**: 817 lines | **Deletions**: 354 lines | **Files Changed**: 22

---

## 目录

1. [概述与动机](#1-概述与动机)
2. [核心架构设计](#2-核心架构设计)
3. [配置系统详解](#3-配置系统详解)
4. [资源池分配与 GPU 拓扑](#4-资源池分配与-gpu-拓扑)
5. [样本路由机制](#5-样本路由机制)
6. [端到端数据流](#6-端到端数据流)
7. [关键代码逐文件解析](#7-关键代码逐文件解析)
8. [如何配置 Multi-Teacher OPD 训练](#8-如何配置-multi-teacher-opd-训练)
9. [架构评估与代码质量审查](#9-架构评估与代码质量审查)
10. [总结](#10-总结)

---

## 1. 概述与动机

### 1.1 什么是 Multi-Teacher OPD？

On-Policy Distillation (OPD) 是一种在线知识蒸馏方法：在 RL 训练循环中，学生模型（Student）的每一条 rollout 输出会被送给教师模型（Teacher）计算 logprobs，然后用蒸馏损失（如 forward KL）引导学生学习教师的输出分布。

**Multi-Teacher OPD (MOPD)** 将此扩展为多教师场景——不同数据集/任务由不同教师模型负责。例如：

| 数据集 | 学生模型 | 教师模型 |
|--------|----------|----------|
| GSM8K (纯文本数学) | Qwen3-VL-2B-Instruct | Qwen3-4B-Instruct-2507 |
| Geometry3K (视觉数学) | Qwen3-VL-2B-Instruct | Qwen3-VL-4B-Instruct |

每个训练样本根据其 `data_source` 字段自动路由到对应的教师模型。

### 1.2 设计目标

1. **零代码扩展**：添加新教师仅需修改配置，无需改动任何代码
2. **向后兼容**：单教师场景的配置方式基本不变
3. **资源隔离**：每个教师拥有独立的 GPU 子池、推理副本和负载均衡器
4. **灵活路由**：通过可配置的 `teacher_key` 字段将样本路由到正确的教师

---

## 2. 核心架构设计

### 2.1 整体架构图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        PPO Trainer / Sync Trainer                        │
│                                                                          │
│  ┌────────────────┐     ┌──────────────────────────────────────────┐    │
│  │ Student Pool   │     │          Teacher Pool (共享 GPU 池)        │    │
│  │ (actor/rollout │     │                                          │    │
│  │  /ref model)   │     │  ┌─────────────────────────────────────┐ │    │
│  │                │     │  │    MultiTeacherModelManager         │ │    │
│  │ n_gpus=2       │     │  │                                     │ │    │
│  │                │     │  │  split_resource_pool([ws1, ws2])    │ │    │
│  └────────────────┘     │  │         ┌───────────┐               │ │    │
│                         │  │         │           │               │ │    │
│                         │  │    ┌────▼────┐ ┌────▼────┐          │ │    │
│                         │  │    │Teacher  │ │Teacher  │          │ │    │
│                         │  │    │Manager  │ │Manager  │          │ │    │
│                         │  │    │(GSM8K)  │ │(Geo3K)  │          │ │    │
│                         │  │    │1 GPU    │ │1 GPU    │          │ │    │
│                         │  │    │1 replica│ │1 replica│          │ │    │
│                         │  │    │LB actor │ │LB actor │          │ │    │
│                         │  │    └─────────┘ └─────────┘          │ │    │
│                         │  └─────────────────────────────────────┘ │    │
│                         └──────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                    AgentLoopManager                               │    │
│  │   ┌────────────────────────────────────────────────────────┐     │    │
│  │   │ AgentLoopWorker × N                                    │     │    │
│  │   │                                                        │     │    │
│  │   │  AsyncTeacherLLMServerManager                         │     │    │
│  │   │   ├─ server_managers["openai/gsm8k"]  → AsyncLLMSvrMgr│     │    │
│  │   │   └─ server_managers["hiyouga/geo3k"] → AsyncLLMSvrMgr│     │    │
│  │   │                                                        │     │    │
│  │   │  每个样本:                                              │     │    │
│  │   │   data_source → routing_key → 选择对应教师 → 计算logprobs│     │    │
│  │   └────────────────────────────────────────────────────────┘     │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心类层次

```
DistillationConfig (配置中心)
├── teacher_models: dict[str, DistillationTeacherModelConfig]
├── distillation_loss: DistillationLossConfig
├── teacher_key: str = "data_source"
├── n_gpus_per_node / nnodes (资源池参数)
└── __post_init__ → _resolve_teacher_models()

MultiTeacherModelManager (资源管理)
├── 拥有一个共享资源池
├── split_resource_pool() → 按教师 world_size 切分
├── teacher_model_managers: dict[str, TeacherModelManager]
├── server_addresses: dict[str, list[str]]
├── server_handles: dict[str, list]
└── load_balancer_handle: dict[str, object]

    TeacherModelManager (单教师管理)
    ├── rollout_replicas: list[RolloutReplica]
    ├── _initialize_llm_servers() → 启动推理服务
    ├── _validate_replica_node_alignment() → 校验节点对齐
    └── _initialize_load_balancer_handle() → 创建负载均衡

AsyncTeacherLLMServerManager (运行时路由, 组合模式)
├── server_managers: dict[str, AsyncLLMServerManager]  # 每教师一个
├── _resolve_teacher_key(routing_key) → 路由决策
└── compute_teacher_logprobs_single() → 计算教师 logprobs
```

### 2.3 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 管理器模式 | 两层：Multi → Single | 每个教师独立管理其副本和负载均衡 |
| 服务器管理 | 组合而非继承 | `AsyncTeacherLLMServerManager` 持有 `dict[str, AsyncLLMServerManager]` 而非继承 `AsyncLLMServerManager`，避免路由逻辑污染基类 |
| 资源池 | 静态划分 | 初始化时一次性切分，避免运行时调度复杂性 |
| 路由方式 | 样本级字段匹配 | 通过 `data_source` 等 DataProto 字段精确路由 |
| 名称隔离 | Ray actor name_suffix | 教师 key（如 `openai_gsm8k`）作为后缀避免多教师间 Ray actor 名称冲突 |

---

## 3. 配置系统详解

### 3.1 配置 Dataclass 结构

#### DistillationConfig (`verl/workers/config/distillation.py:209-308`)

```python
@dataclass
class DistillationConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models", "n_gpus_per_node", "nnodes"}

    enabled: bool = False
    n_gpus_per_node: int = 0          # 教师资源池每节点 GPU 数
    nnodes: int = 0                    # 教师资源池节点数
    teacher_models: dict[str, DistillationTeacherModelConfig] = field(default_factory=dict)
    teacher_key: str = "data_source"   # 路由字段名
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)
```

**`__post_init__` 逻辑** (`distillation.py:257-275`)：
1. 调用 `_resolve_teacher_models()` 解析并标准化教师配置
2. 对每个教师调用 `validate_and_prepare_for_distillation()` 调整推理参数
3. 校验所有教师 `world_size` 之和等于 `n_gpus_per_node * nnodes`

#### DistillationTeacherModelConfig (`distillation.py:115-206`)

```python
@dataclass
class DistillationTeacherModelConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields | {"num_replicas", "key"}

    key: Optional[str] = None          # 路由键值 (如 "openai/gsm8k")
    model_path: Optional[str] = None   # HuggingFace 模型路径
    inference: RolloutConfig = field(default_factory=RolloutConfig)
    num_replicas: Optional[int] = 0    # 推理副本数

    @property
    def per_replica_world_size(self) -> int:
        # TP × DP × PP = 单个副本占用的 GPU 数
        return (self.inference.tensor_model_parallel_size
                * self.inference.data_parallel_size
                * self.inference.pipeline_model_parallel_size)

    @property
    def world_size(self) -> int:
        # 该教师的总 GPU 占用量
        return self.num_replicas * self.per_replica_world_size
```

### 3.2 _resolve_teacher_models 核心逻辑

这是 MOPD 配置系统的核心方法 (`distillation.py:277-308`)，分两条路径处理：

```
_resolve_teacher_models()
│
├── len(teacher_models) == 1  →  单教师模式
│   ├── 自动计算 num_replicas = pool_size // per_replica_world_size
│   ├── 设置 key = "default"
│   └── 用户无需指定 key 和 num_replicas
│
└── len(teacher_models) > 1  →  多教师模式
    ├── pop("teacher_model")  # 移除 YAML 默认占位符
    ├── 对每个教师调用 omega_conf_to_dataclass() 转换
    ├── 调用 check_configured() 校验 key/model_path/num_replicas
    ├── 检查 key 不重复
    └── 按 teacher_config.key 重新索引 dict
```

### 3.3 YAML 配置模板

`verl/trainer/config/distillation/distillation.yaml` 的关键结构：

```yaml
# 顶层字段
enabled: false
n_gpus_per_node: 8         # 教师资源池每节点 GPU 数
nnodes: 0                  # 教师资源池节点数
teacher_key: data_source    # 路由键名

# 蒸馏损失配置
distillation_loss:
  loss_mode: k3
  topk: 32
  use_policy_gradient: false
  # ... 更多损失参数

# 教师模型配置（默认单教师模板）
teacher_models:
  teacher_model:           # ⚠️ 保留名称，多教师时会被 pop
    _target_: verl.workers.config.DistillationTeacherModelConfig
    key: null
    model_path: null
    num_replicas: 0
    inference:
      _target_: verl.workers.config.RolloutConfig
      name: ${oc.select:actor_rollout_ref.rollout.name}     # 继承学生的推理引擎
      tensor_model_parallel_size: 2
      gpu_memory_utilization: 0.5
      # ... 更多推理参数
```

### 3.4 validate_and_prepare_for_distillation

每个教师配置在 `__post_init__` 中经过此方法处理 (`distillation.py:160-174`)：

```python
def validate_and_prepare_for_distillation(self, use_topk, topk):
    # 1. 校验 max_model_len 足够容纳 prompt + response + 1
    required_context_len = student_prompt_length + student_response_length + 1
    if max_model_len is not None and required_context_len > max_model_len:
        raise ValueError(...)

    # 2. 调整推理参数：教师把学生的完整序列作为 "prompt"
    self.inference.prompt_length = self.inference.prompt_length + self.inference.response_length
    self.inference.response_length = 1  # 教师只生成 1 个 token（仅为获取 prompt_logprobs）

    # 3. 校验 topk 与引擎的 max_logprobs 对齐
    self._validate_topk_logprobs(use_topk=use_topk, topk=topk)
```

---

## 4. 资源池分配与 GPU 拓扑

### 4.1 资源池创建

在 `main_ppo.py:176-184` 和 `main_ppo_sync.py:1674-1685` 中：

```python
# 教师资源池创建
teacher_pool = [distillation_config.n_gpus_per_node] * distillation_config.nnodes
resource_pool_spec["teacher_pool"] = teacher_pool
# 例如: n_gpus_per_node=2, nnodes=1 → teacher_pool = [2]
```

资源池通过 `Role.TeacherModel` 枚举值映射：

```python
self.mapping[Role.TeacherModel] = "teacher_pool"
```

### 4.2 两级资源池切分

```
┌─────────────────────────────────────────────────────────┐
│           Combined Teacher Resource Pool                 │
│           (n_gpus_per_node × nnodes GPUs)                │
│                                                          │
│   第一级切分: MultiTeacherModelManager                    │
│   split_resource_pool(pool, split_size=[ws1, ws2, ...]) │
│                                                          │
│   ┌──────────────────┐  ┌──────────────────┐            │
│   │ Teacher A 子池   │  │ Teacher B 子池   │            │
│   │ ws1 = num_rep_A  │  │ ws2 = num_rep_B  │            │
│   │ × per_rep_ws_A   │  │ × per_rep_ws_B   │            │
│   │                  │  │                  │            │
│   │ 第二级切分:      │  │ 第二级切分:      │            │
│   │ TeacherModelMgr  │  │ TeacherModelMgr  │            │
│   │ split(per_rep_ws)│  │ split(per_rep_ws)│            │
│   │                  │  │                  │            │
│   │ ┌──────┐┌──────┐│  │ ┌──────┐         │            │
│   │ │Rep 0 ││Rep 1 ││  │ │Rep 0 │         │            │
│   │ │1 GPU ││1 GPU ││  │ │1 GPU │         │            │
│   │ └──────┘└──────┘│  │ └──────┘         │            │
│   └──────────────────┘  └──────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

### 4.3 节点对齐校验

`_validate_replica_node_alignment` (`teacher_model.py:102-143`) 防止 `split_resource_pool` 的线性切分导致副本跨越节点边界：

```
示例 (P = n_gpus_per_node = 4, 两个教师 W=3 和 W=4):

        node 0                  node 1
        [0 1 2 3]               [4 5 6 7]       ← bundle 索引

    Teacher A (W=3):
        [A A A .]               [. . . .]        期望跨 1 节点, 实际 1  ✓
    Teacher B (W=4):
        [. . . B]               [B B B .]        期望跨 1 节点, 实际 2  ✗ 报错！
```

校验逻辑：
```python
expected_span = ceil(W / P)  # 期望跨越的节点数
first_node = start // P
last_node = (start + W - 1) // P
observed_span = last_node - first_node + 1
if observed_span != expected_span:
    raise ValueError(...)  # 提示用户调整教师顺序或副本数
```

### 4.4 GPU 分配计算示例

以 `run_qwen3_mopd_gsm8k_geo3k.sh` 为例：

```bash
STUDENT_WORLD_SIZE=2                    # 学生 2 GPU
TEACHER_NUM_REPLICAS_GSM8K=1            # GSM8K 教师 1 副本
TEACHER_NUM_REPLICAS_GEO3K=1            # Geo3K 教师 1 副本
# 每个教师 TP=DP=PP=1, 所以 per_replica_world_size=1

TEACHER_POOL_WORLD_SIZE = 1 + 1 = 2    # 教师池共 2 GPU
```

**总计需要 4 GPU**：
- GPU 0-1：学生模型 (actor + rollout + ref model)
- GPU 2：GSM8K 教师 (1 vLLM 副本)
- GPU 3：Geo3K 教师 (1 vLLM 副本)

校验：`sum(teacher_world_sizes) = 1 + 1 = 2 = n_gpus_per_node(2) × nnodes(1)` ✓

---

## 5. 样本路由机制

### 5.1 三层路由链

```
┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  Layer 1: 数据集  │     │  Layer 2: 配置绑定 │     │  Layer 3: 运行时  │
│                  │     │                   │     │  路由             │
│  parquet 文件中   │ ──▶ │  teacher_models   │ ──▶ │  _resolve_       │
│  data_source 列  │     │  .key 字段绑定     │     │  teacher_key()   │
│                  │     │                   │     │                  │
│  "openai/gsm8k"  │     │  gsm8k.key=       │     │  routing_key →   │
│  "hiyouga/geo3k" │     │  "openai/gsm8k"   │     │  server_managers │
└──────────────────┘     └───────────────────┘     └──────────────────┘
```

### 5.2 路由字段提取

在 `agent_loop.py:893-915` 的 `_compute_teacher_logprobs` 中：

```python
# 1. 从样本 kwargs 中提取路由值
routing_value = sample_kwargs.get(self.teacher_key)  # teacher_key = "data_source"

# 2. 归一化（numpy 0-d 数组 → Python 字符串）
routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value

# 3. 传给教师服务管理器
teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
    sequence_ids=prompt_ids + response_ids,
    multi_modal_data=output.multi_modal_data,
    routing_key=routing_key,  # 如 "openai/gsm8k"
)
```

### 5.3 路由决策逻辑

`AsyncTeacherLLMServerManager._resolve_teacher_key` (`teacher_manager.py:98-112`)：

```python
def _resolve_teacher_key(self, routing_key):
    if len(self.teacher_model_configs) == 1:
        # 单教师快捷路径：忽略 routing_key，直接返回唯一教师
        return next(iter(self.teacher_model_configs))

    if routing_key is None:
        raise ValueError("Multi-teacher requires routing key")

    if routing_key not in self.teacher_model_configs:
        raise ValueError(f"No teacher for routing key {routing_key!r}")

    return routing_key
```

### 5.4 路由与 Reward Manager 的统一

MOPD 的路由机制复用了 verl 已有的 `data_source` 约定——`NaiveRewardManager` 和 `DAPORewardManager` 也使用 `data_source` 来选择不同数据集的奖励函数。这提供了一致的路由范式。

---

## 6. 端到端数据流

### 6.1 初始化阶段

```
YAML config
  │
  ▼
DistillationConfig.__post_init__()          [distillation.py:257]
  │  _resolve_teacher_models()
  │    → 单教师: auto num_replicas, key="default"
  │    → 多教师: pop("teacher_model"), re-key by teacher_config.key
  │  validate_and_prepare_for_distillation()
  │    → prompt_length += response_length; response_length = 1
  │  校验 sum(world_sizes) == pool_size
  ▼
init_resource_pool_mgr()                    [main_ppo.py:176]
  │  teacher_pool = [n_gpus_per_node] * nnodes
  │  Role.TeacherModel → "teacher_pool"
  ▼
ResourcePoolManager 创建 Ray placement groups
  ▼
MultiTeacherModelManager.__init__()         [teacher_model.py:156]
  │  split_resource_pool(pool, [ws1, ws2, ...])
  │    → 每教师一个 SubRayResourcePool
  │
  ├──▶ TeacherModelManager(teacher_A)       [teacher_model.py:36]
  │    │  split_resource_pool(sub_pool, per_replica_ws)
  │    │  _validate_replica_node_alignment()
  │    │  RolloutReplica × num_replicas → init_colocated()
  │    │  GlobalRequestLoadBalancer.remote()
  │    ▼
  │    server_handles, server_addresses, load_balancer_handle
  │
  └──▶ TeacherModelManager(teacher_B)
       │  (同上)
       ▼
       server_handles, server_addresses, load_balancer_handle
```

### 6.2 训练阶段（每个样本）

```
DataProto batch (含 non_tensor_batch["data_source"])
  │
  ▼
AgentLoopWorker.run_agent_loop()            [agent_loop.py:595]
  │  kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
  │  → kwargs["data_source"] = "openai/gsm8k"
  ▼
Student rollout → AgentLoopOutput
  │  prompt_ids, response_ids, multi_modal_data
  ▼
_agent_loop_postprocess()                   [agent_loop.py:644]
  │
  ├── _compute_score() → 计算奖励
  │
  └── _compute_teacher_logprobs()           [agent_loop.py:893]
      │  routing_key = kwargs["data_source"].item()
      │    → "openai/gsm8k"
      ▼
      AsyncTeacherLLMServerManager
      .compute_teacher_logprobs_single()    [teacher_manager.py:114]
        │
        │  _resolve_teacher_key("openai/gsm8k")
        │    → teacher_key = "openai/gsm8k"
        │
        │  server_manager = self.server_managers["openai/gsm8k"]
        │  sampling_params = {max_tokens: 1, prompt_logprobs: topk}
        │
        │  teacher_output = await server_manager.generate(
        │      prompt_ids = student_prompt + student_response,
        │      sampling_params = {...}
        │  )
        │
        │  teacher_ids = tensor(teacher_output["prompt_ids"])
        │  teacher_logprobs = tensor(teacher_output["prompt_logprobs"])
        ▼
      output.extra_fields["teacher_ids"] = teacher_ids
      output.extra_fields["teacher_logprobs"] = teacher_logprobs
  │
  ▼
_pad_teacher_outputs()                      [teacher_manager.py:47]
  │  左填充 prompt 区域，右填充 response 区域
  ▼
_InternalAgentLoopOutput(teacher_ids, teacher_logprobs)
  │
  ▼
_postprocess() → torch.cat() → DataProto   [agent_loop.py:917]
  │  batch["teacher_ids"], batch["teacher_logprobs"]
  ▼
Actor training step
  │  distillation loss = f(student_logprobs, teacher_logprobs)
  │  total_loss = rl_loss + distillation_loss_coef * distillation_loss
  ▼
Parameter update
```

---

## 7. 关键代码逐文件解析

### 7.1 `verl/workers/config/distillation.py` — 配置数据类

**核心变更**：从 `teacher_model: DistillationTeacherModelConfig` 重构为 `teacher_models: dict[str, DistillationTeacherModelConfig]`

| 旧结构 | 新结构 |
|--------|--------|
| `distillation.teacher_model.model_path` | `distillation.teacher_models.{name}.model_path` |
| `distillation.teacher_model.n_gpus_per_node` | `distillation.n_gpus_per_node`（提升到顶层） |
| `distillation.teacher_model.nnodes` | `distillation.nnodes`（提升到顶层） |
| `distillation.num_workers` | 移除（改用 `num_replicas`） |

**新增属性**：
- `DistillationTeacherModelConfig.per_replica_world_size` (计算属性)：`TP * DP * PP`
- `DistillationTeacherModelConfig.world_size` (计算属性)：`num_replicas * per_replica_world_size`
- `DistillationTeacherModelConfig.check_configured()`：多教师模式下校验必填字段
- `DistillationConfig.teacher_key`：路由字段名，默认 `"data_source"`

**关键方法 `_resolve_teacher_models()`** (`distillation.py:277-308`)：

```python
def _resolve_teacher_models(self):
    assert "teacher_model" in self.teacher_models

    if len(self.teacher_models) == 1:
        # 单教师：自动推算 num_replicas 和 key
        teacher_model = self.teacher_models["teacher_model"]
        per_replica = TP * DP * PP
        pool_size = n_gpus_per_node * nnodes
        teacher_model.num_replicas = pool_size // per_replica
        teacher_model.key = "default"
    else:
        # 多教师：移除默认占位符
        self.teacher_models.pop("teacher_model")

    # 按 key 重新索引
    teacher_models = {}
    for teacher_config in self.teacher_models.values():
        teacher_config = omega_conf_to_dataclass(teacher_config, DistillationTeacherModelConfig)
        teacher_config.check_configured()
        if teacher_config.key in teacher_models:
            raise ValueError(f"Duplicate teacher key {teacher_config.key}")
        teacher_models[teacher_config.key] = teacher_config
    return teacher_models
```

### 7.2 `verl/experimental/teacher_loop/teacher_model.py` — 管理器层次

**`TeacherModelManager`** (`teacher_model.py:36-151`)：管理单个教师的推理副本

关键流程：
1. `_initialize_llm_servers()`：
   - 校验 `resource_pool.world_size == num_replicas * per_replica_world_size`
   - 获取 `rollout_replica_class`（vLLM/SGLang/TRTLLM）
   - 创建 `name_suffix` = `teacher_config.key.replace("/", "_")`（如 `"openai_gsm8k"`）
   - 实例化 `num_replicas` 个 `RolloutReplica`，每个带 `is_teacher_model=True` 和 `name_suffix`
   - `split_resource_pool()` 切分子池
   - `_validate_replica_node_alignment()` 校验节点对齐
   - `_run_all()` 并行启动所有副本

2. `_initialize_load_balancer_handle()`：为该教师创建独立的 `GlobalRequestLoadBalancer` Ray actor

**`MultiTeacherModelManager`** (`teacher_model.py:153-194`)：管理多个教师

```python
def _initialize_teacher_model_managers(self):
    split_sizes = [teacher.world_size for teacher in teacher_models.values()]
    split_pools = split_resource_pool(self.resource_pool, split_size=split_sizes)

    for (key, teacher_model_config), teacher_pool in zip(...):
        manager = TeacherModelManager(
            distillation_config=self.distillation_config,
            teacher_model_config=teacher_model_config,
            resource_pool=teacher_pool,
        )
        self.teacher_model_managers[key] = manager
        self.server_addresses[key] = manager.server_addresses
        self.server_handles[key] = manager.server_handles
        self.load_balancer_handle[key] = manager.load_balancer_handle
```

### 7.3 `verl/experimental/teacher_loop/teacher_manager.py` — 异步路由

**`AsyncTeacherLLMServerManager`** (`teacher_manager.py:66-137`)：从继承改为组合

| 旧设计 | 新设计 |
|--------|--------|
| `class AsyncTeacherLLMServerManager(AsyncLLMServerManager)` | `class AsyncTeacherLLMServerManager:` (独立类) |
| 直接调用 `self.generate()` | 通过 `self.server_managers[key].generate()` 路由 |
| 持有一个 `distillation_config` | 持有 `teacher_model_configs: dict` |

关键方法：

```python
async def compute_teacher_logprobs_single(self, sequence_ids, multi_modal_data=None, routing_key=None):
    teacher_key = self._resolve_teacher_key(routing_key)
    teacher_model_config = self.teacher_model_configs[teacher_key]
    server_manager = self.server_managers[teacher_key]

    teacher_output = await server_manager.generate(
        request_id=uuid4().hex,
        prompt_ids=sequence_ids,
        sampling_params=_get_teacher_sampling_params(teacher_model_config, self.distillation_loss_config),
        image_data=multi_modal_data.get("images"),
        video_data=multi_modal_data.get("videos"),
    )

    teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
    teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
    return teacher_ids, teacher_logprobs
```

### 7.4 `verl/experimental/agent_loop/agent_loop.py` — Agent Loop 集成

**类型签名变更**：
```python
# 旧
teacher_servers: list[tuple[str, ray.actor.ActorHandle]] = None
teacher_load_balancer_handle: ray.actor.ActorHandle = None

# 新
teacher_servers: Optional[dict[str, list[tuple[str, ray.actor.ActorHandle]]]] = None
teacher_load_balancer_handle: Optional[dict[str, ray.actor.ActorHandle]] = None
```

**`AgentLoopManager._init_agent_loop_workers`** (`agent_loop.py:1142-1183`)：构建按教师 key 索引的字典

```python
if self.distillation_enabled:
    teacher_servers = {
        key: list(zip(
            self.teacher_model_manager.server_addresses[key],
            self.teacher_model_manager.server_handles[key],
        ))
        for key in self.teacher_model_manager.server_addresses
    }
    teacher_load_balancer_handle = dict(self.teacher_model_manager.load_balancer_handle)
```

**`_compute_teacher_logprobs` 新增路由** (`agent_loop.py:893-915`)：

```python
async def _compute_teacher_logprobs(self, output, prompt_ids, response_ids, validate, sample_kwargs=None):
    if self.distillation_enabled and not validate:
        routing_key = None
        if sample_kwargs is not None:
            routing_value = sample_kwargs.get(self.teacher_key)
            if routing_value is not None:
                routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value

        teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
            sequence_ids=prompt_ids + response_ids,
            multi_modal_data=output.multi_modal_data,
            routing_key=routing_key,
        )
```

### 7.5 `verl/workers/rollout/replica.py` — 副本名称隔离

新增 `name_suffix` 参数，确保多教师的 Ray actor 名称不冲突：

```python
class RolloutReplica(ABC):
    def __init__(self, ..., name_suffix: str = ""):
        self.name_suffix = f"_{name_suffix}" if name_suffix else ""

    async def init_colocated(self, resource_pool):
        if self.is_teacher_model:
            name_prefix = f"rollout_teacher_colocate_{self.replica_rank}{self.name_suffix}"
            # 例如: "rollout_teacher_colocate_0_openai_gsm8k"
```

`name_suffix` 来自教师 key（斜杠替换为下划线）：
```python
# teacher_model.py:82
name_suffix = (teacher_model_config.key or "").replace("/", "_")
```

所有三个 Replica 子类（vLLM、SGLang、TRTLLM）都已适配此参数。

---

## 8. 如何配置 Multi-Teacher OPD 训练

### 8.1 单教师配置（向后兼容）

```bash
python3 -m verl.trainer.main_ppo \
    --config-name='ppo_trainer.yaml' \
    distillation.enabled=True \
    distillation.n_gpus_per_node=4 \
    distillation.nnodes=1 \
    distillation.teacher_models.teacher_model.model_path="Qwen/Qwen3-4B" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.3 \
    distillation.distillation_loss.loss_mode=k1 \
    distillation.distillation_loss.topk=64 \
    distillation.distillation_loss.use_policy_gradient=True \
    # ... 其他参数
```

**注意**：单教师模式下：
- 无需设置 `key` 和 `num_replicas`（自动推算）
- `num_replicas` 自动计算为 `pool_size // per_replica_world_size`

### 8.2 多教师配置

```bash
python3 -m verl.trainer.main_ppo \
    --config-name='ppo_trainer.yaml' \
    distillation.enabled=True \
    distillation.teacher_key=data_source \
    distillation.n_gpus_per_node=4 \
    distillation.nnodes=1 \
    # --- 教师 1: GSM8K (纯文本) ---
    +distillation.teacher_models.gsm8k.key="openai/gsm8k" \
    +distillation.teacher_models.gsm8k.model_path="Qwen/Qwen3-4B-Instruct-2507" \
    +distillation.teacher_models.gsm8k.num_replicas=2 \
    +distillation.teacher_models.gsm8k.inference.name=vllm \
    +distillation.teacher_models.gsm8k.inference.tensor_model_parallel_size=1 \
    +distillation.teacher_models.gsm8k.inference.gpu_memory_utilization=0.8 \
    # --- 教师 2: Geo3K (视觉语言) ---
    +distillation.teacher_models.geo3k.key="hiyouga/geometry3k" \
    +distillation.teacher_models.geo3k.model_path="Qwen/Qwen3-VL-4B-Instruct" \
    +distillation.teacher_models.geo3k.num_replicas=2 \
    +distillation.teacher_models.geo3k.inference.name=vllm \
    +distillation.teacher_models.geo3k.inference.tensor_model_parallel_size=1 \
    +distillation.teacher_models.geo3k.inference.gpu_memory_utilization=0.8 \
    +distillation.teacher_models.geo3k.inference.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    # --- 蒸馏损失 ---
    distillation.distillation_loss.loss_mode=k1 \
    distillation.distillation_loss.topk=64 \
    distillation.distillation_loss.use_policy_gradient=True
```

### 8.3 关键配置参数参考

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `distillation.enabled` | 是否启用蒸馏 | `True` |
| `distillation.teacher_key` | 路由字段名 | `data_source` |
| `distillation.n_gpus_per_node` | 教师池每节点 GPU 数 | `4` |
| `distillation.nnodes` | 教师池节点数 | `1` |
| `+distillation.teacher_models.{name}.key` | 路由键值（匹配数据集字段） | `"openai/gsm8k"` |
| `+distillation.teacher_models.{name}.model_path` | 教师模型路径 | `"Qwen/Qwen3-4B"` |
| `+distillation.teacher_models.{name}.num_replicas` | 推理副本数 | `2` |
| `+distillation.teacher_models.{name}.inference.name` | 推理引擎 | `vllm` / `sglang` |
| `+distillation.teacher_models.{name}.inference.tensor_model_parallel_size` | TP 大小 | `1` |
| `distillation.distillation_loss.loss_mode` | 蒸馏损失类型 | `k1` / `k3` |
| `distillation.distillation_loss.topk` | Top-K logprobs 数量 | `64` |
| `distillation.distillation_loss.use_policy_gradient` | 是否使用策略梯度 | `True` |

### 8.4 资源约束公式

```
总教师 GPU = distillation.n_gpus_per_node × distillation.nnodes

对每个教师 i:
  per_replica_world_size_i = TP_i × DP_i × PP_i
  world_size_i = num_replicas_i × per_replica_world_size_i

约束: Σ world_size_i = 总教师 GPU
```

### 8.5 ⚠️ 重要注意事项

#### Hydra `+` 前缀

多教师配置中，新增教师条目**必须使用 `+` 前缀**：
```bash
+distillation.teacher_models.gsm8k.key=...    # ✅ 正确：+ 表示新增
distillation.teacher_models.gsm8k.key=...     # ❌ 错误：Hydra 找不到该 key 会报错
```

#### 避免使用 `teacher_model` 作为多教师的 key

```bash
# ❌ 错误用法：teacher_model 会被 _resolve_teacher_models() 中的 pop() 移除
distillation.teacher_models.teacher_model.key=openai/gsm8k
+distillation.teacher_models.teacher2.key=hiyouga/geometry3k

# ✅ 正确用法：所有教师使用自定义名称
+distillation.teacher_models.gsm8k.key=openai/gsm8k
+distillation.teacher_models.geo3k.key=hiyouga/geometry3k
```

#### Key 值必须精确匹配数据集

教师的 `key` 必须与数据集中 `data_source` 列的值完全一致（大小写、斜杠敏感）：
```bash
# 数据集中: data_source = "openai/gsm8k"
+distillation.teacher_models.gsm8k.key="openai/gsm8k"    # ✅
+distillation.teacher_models.gsm8k.key="OpenAI/GSM8K"    # ❌ 大小写不匹配
```

---

## 9. 架构评估与代码质量审查

### 9.1 架构优势

1. **关注点分离清晰**：两层管理器（Multi → Single）映射合理，每个 `TeacherModelManager` 自包含
2. **组合优于继承**：`AsyncTeacherLLMServerManager` 使用组合而非继承 `AsyncLLMServerManager`，避免路由逻辑污染基类
3. **防御性校验**：`_validate_replica_node_alignment` 捕获了 `split_resource_pool` 线性切分的隐蔽失败模式
4. **零代码扩展**：新增教师仅需配置变更

### 9.2 架构局限

1. **静态资源分配**：教师 GPU 一旦分配不可动态重平衡，当数据分布不均时空闲 GPU 浪费
2. **配置陷阱**：`teacher_model` 作为魔术哨兵名称，多教师场景误用时静默丢弃数据
3. **无容错机制**：单个教师的 `GlobalRequestLoadBalancer` 死亡会阻塞该教师所有请求，无熔断或回退
4. **Mutable config**：`validate_and_prepare_for_distillation` 在 `__post_init__` 中原地修改 `prompt_length/response_length`，不可重入

### 9.3 代码质量发现

| 严重度 | 发现 | 位置 |
|--------|------|------|
| HIGH | `_resolve_teacher_models` 使用 `assert` 做用户输入校验（优化模式下被跳过） | `distillation.py:278` |
| HIGH | `_postprocess` 仅检查 `inputs[0]` 的 `teacher_logprobs`，混合批次可能静默丢数据或崩溃 | `agent_loop.py:932-938` |
| MEDIUM | `DistillationLossConfig.__post_init__` 用 `print()` 而非 `logger.warning()` | `distillation.py:101-106` |
| MEDIUM | `_get_teacher_sampling_params` 错误信息硬编码 "vLLM" 但函数与引擎无关 | `teacher_manager.py:37` |
| MEDIUM | `_run_all` 使用 `asyncio.gather` 只传播第一个异常 | `teacher_model.py:33` |
| MEDIUM | routing_key 的 `.item()` 可能返回 `numpy.str_` 而非 `str` | `agent_loop.py:907-908` |
| LOW | `num_replicas: Optional[int] = 0` 类型注解与默认值语义不一致 | `distillation.py:138` |
| LOW | 访问 RolloutReplica 的私有属性 `_server_handle` / `_server_address` | `teacher_model.py:99-100` |

---

## 10. 总结

PR #6051 通过以下核心设计实现了 Multi-Teacher OPD：

1. **配置层**：将 `teacher_model` 扩展为 `teacher_models` 字典，支持通过 Hydra CLI `+` 前缀动态添加任意数量的教师模型
2. **资源层**：引入 `MultiTeacherModelManager` → `TeacherModelManager` 两级管理，通过 `split_resource_pool` 将共享 GPU 池按教师 `world_size` 切分
3. **路由层**：`AsyncTeacherLLMServerManager` 使用组合模式持有多个 `AsyncLLMServerManager`，通过 `data_source` 等可配置字段将每个训练样本路由到正确的教师
4. **隔离层**：通过 `name_suffix` 机制确保多教师 Ray actor 名称唯一，通过 `_validate_replica_node_alignment` 确保 GPU 拓扑正确

该设计对其主要用例（已知固定教师集合、均衡负载分布）架构合理、扩展性良好，但在运行时容错、动态资源调度和配置安全性方面存在改进空间。
