# verl 专家 Agent 使用指南

> 本文档详细介绍 `verl/.claude/agents/` 下所有 Agent 的定位、触发条件、能力范围和使用技巧。

## 目录

- [Agent 体系概览](#agent-体系概览)
- [主动激活 Agent](#主动激活-agent)
  - [planner — 实施规划器](#planner--实施规划器)
  - [simple-code-reviewer — 快速代码审查](#simple-code-reviewer--快速代码审查)
  - [code-verifier — 代码验证器](#code-verifier--代码验证器)
- [按需请求 Agent](#按需请求-agent)
  - [algorithm-expert — RL 算法专家](#algorithm-expert--rl-算法专家)
  - [fsdp-engine-expert — FSDP2 训练专家](#fsdp-engine-expert--fsdp2-训练专家)
  - [megatron-engine-expert — Megatron 并行专家](#megatron-engine-expert--megatron-并行专家)
  - [vllm-sglang-expert — 推理引擎专家](#vllm-sglang-expert--推理引擎专家)
  - [ray-controller-expert — Ray 编排专家](#ray-controller-expert--ray-编排专家)
- [Agent 协作模式](#agent-协作模式)
- [提示词模板](#提示词模板)

---

## Agent 体系概览

verl 的 Agent 体系分为两类：**主动激活**（代码改动后自动触发）和**按需请求**（开发者主动询问时激活）。

```
┌─────────────────────────────────────────────────────────────────┐
│                      主动激活 Agent                              │
│                                                                  │
│   planner (Opus)                simple-code-reviewer (Sonnet)    │
│   ├ 多文件改动时自动规划           ├ 代码改动后自动审查               │
│   ├ 新功能/架构决策时激活          ├ 检查 verl 模式��规性            │
│   └ 只读：研究 + 出计划           └ 只读：发现问题不修复             │
│                                                                  │
│   code-verifier (Haiku)                                          │
│   ├ 代码改动后自动运行                                             │
│   ├ 跑 pre-commit / Ruff / mypy / pytest                        │
│   └ 可执行：实际运行命令                                           │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                      按需请求 Agent (均为 Opus)                   │
│                                                                  │
│   algorithm-expert     fsdp-engine-expert    megatron-engine-expert │
│   ├ PPO/GRPO/RLOO      ├ FSDP2/DTensor       ├ TP/PP/SP/CP       │
│   ├ Reward/Advantage   ├ Device Mesh          ├ Pipeline Schedule  │
│   └ Loss Functions     └ CPU Offload          └ Checkpoint Convert │
│                                                                  │
│   vllm-sglang-expert   ray-controller-expert                     │
│   ├ vLLM/SGLang         ├ 单控制器模式                             │
│   ├ Rollout Workers     ├ Dispatch Modes                          │
│   └ Multi-turn/Tools    └ Resource Management                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 主动激活 Agent

### planner — 实施规划器

**模型：** Opus | **工具：** Read, Grep, Glob, Task | **激活：** 主动

#### 何时激活

- 涉及 **3+ 个文件**的改动
- 设计**新功能**（worker、dataset、reward、算法）
- 需要**架构决策**
- 用户问"我该怎么..."或"最好的方式是什么..."

#### 不应使用

- 单文件、明显实现的改动
- 拼写修复、简单重命名、文档更新
- 纯粹的代码库探索（用 Explore agent）

#### 规划流程

**Phase 1: 理解需求**
- 澄清需求（问具体的、有选项的问题，不问开放式问题）
- 识别影响范围
- 找到现有模式

**Phase 2: 研究代码库**
- 找相似实现
- 找调用者/依赖
- 检查测试和配置

**Phase 3: 输出计划**
- 简单任务：Summary + Changes 表 + Steps
- 复杂任务：Summary + Changes 表 + Steps + Patterns + Risks + Testing

#### 使用示例

```
用户：我想给 verl 添加一个新的 DAPO 算法实现
↓ planner 自动激活
↓ 研究 verl/trainer/ppo/ 和 verl/experimental/ 的模式
↓ 输出：
   ## Summary
   添加 DAPO 算法，基于 PPO trainer 架构。

   ## Changes
   | File | Action | Purpose |
   |------|--------|---------|
   | verl/trainer/dapo/ | Create | DAPO 算法核心实现 |
   | verl/trainer/config/algorithm/dapo.yaml | Create | DAPO 配置 |
   | tests/unit/test_dapo.py | Create | 单元测试 |

   ## Steps
   1. 创建 verl/trainer/dapo/actor.py（参考 PPO actor.py）
   2. 添加 DAPO-specific 的 advantage 计算
   3. 创建 Hydra 配置
   4. 添加测试
   ...
```

---

### simple-code-reviewer — 快速代码审查

**模型：** Sonnet | **工具：** Read, Grep, Glob（只读） | **激活：** 主动

#### 何时激活

- 代码改动完成后
- 提交代码前
- 用户问"帮我看看这段代码"或"这样对吗？"

#### 审查重点

**1. verl 特有模式检查**

| 检查项 | 正确做法 |
|-------|---------|
| 日志 | `logging.getLogger(__file__)` + `VERL_LOGGING_LEVEL`，不用 `print` |
| Ray | `@ray.remote`、`ray.get()` 阻塞、`ray.wait()` 异步 |
| DataProto | 所有 worker 间数据传输用 `DataProto` |
| 配置 | 扩展 Hydra YAML 或 `verl/workers/config/` 中的 dataclass |
| 导入 | 禁止 `*` 导入；重型依赖放函数内部 |
| Worker | 命名遵循 `XxxWorker`，方法用 `@register` 标注 dispatch mode |
| Tensor | 遵循 `[batch, seq_len, ...]` 维度约定 |

**2. 常见问题**
- GPU-CPU 同步：训练循环中不必要的 `.item()`, `.cpu()`, `.numpy()`
- 缺少进程组：集合操作没有 `group=` 参数
- 类型提示缺失
- 资源泄漏

**3. 分布式代码问题**
- 缺少同步：`all_reduce`/`all_gather` 没指定 process group
- 设备不匹配
- Mesh 维度错误
- 使用了全局 process group

**4. Ray 模式问题**
- 控制器中的阻塞调用
- 大数据直传（应该用 `ray.put()` + ref）
- Worker 方法缺少 `@register`

#### 输出格式

```markdown
## Quick Review Summary

**Files Reviewed**: verl/workers/fsdp_workers.py, tests/unit/test_fsdp.py
**Issues Found**: 3 (1 critical, 2 suggestions)

### Critical Issues
1. **Missing process group in all_reduce** - `fsdp_workers.py:142`
   - Problem: dist.all_reduce(grad) 没有指定 group 参数
   - Fix: dist.all_reduce(grad, group=self.dp_group)

### Suggestions
1. **Consider using ray.put()** - `ray_trainer.py:89`
   - 大张量直接传递给 worker，建议先 ray.put() 再传 ref

### Looks Good ✓
- DataProto 使用规范
- Worker 命名符合约定
```

---

### code-verifier — 代码验证器

**模型：** Haiku | **工具：** Read, Grep, Glob, Bash | **激活：** 主动

#### 何时激活

- 代码改动后准备提交时
- 用户问"准备好提交了吗？"或"帮我检查一下"
- 实现功能或修复后
- 创建 PR 之前

#### 验证流程

**Phase 1: 识别变更文件**
```bash
git status --short
git diff --name-only HEAD
```

**Phase 2: 格式化 & Lint**
```bash
pre-commit run --all-files
# 包含：Ruff (lint + format)、mypy、auto-gen config、docstring、license、compile
```

**Phase 3: 运行测试**
```bash
pytest tests/unit/ -v --timeout=60      # 单元测试（无需 GPU）
pytest tests/special_sanity/ -v          # 健全性检查（视情况需 GPU）
pytest tests/special_e2e/ -v            # 端到端（需 GPU）
pytest tests/special_distributed/ -v    # 分布式（需多 GPU）
```

**Phase 4: 文档检查**
```bash
# 如果 Hydra config dataclass 有变
bash scripts/generate_trainer_config.sh
git diff --name-only  # 检查生成文件是否有变化
```

**Phase 5: 输出报告**

| 检查项 | 状态 | 详情 |
|-------|------|------|
| Ruff (lint) | ✅ PASS | 无问题 |
| Ruff (format) | ✅ PASS | 自动修复 2 个文件 |
| mypy | ✅ PASS | 无类型错误 |
| 单元测试 | ✅ PASS | 12 项通过 |
| GPU 测试 | ⏭ SKIP | GPU 不可用 |

---

## 按需请求 Agent

### algorithm-expert — RL 算法专家

**模型：** Opus | **工具：** Read, Grep, Glob（只读）

#### 专业领域

| 领域 | 覆盖内容 |
|------|---------|
| **RL 算法** | PPO、GRPO、RLOO、REINFORCE++、GSPO、ReMax、DAPO、VAPO |
| **Reward 设计** | 函数设计与实现、多组件 reward、归一化与缩放、KL 惩罚 |
| **Advantage 计算** | GAE、Monte Carlo、Baseline 减法、归一化策略 |
| **Loss 函数** | Policy loss (clipped/unclipped)、Value loss、Entropy bonus、KL divergence |

#### 关键文件

- `verl/trainer/ppo/`：PPO 算法核心（`actor.py`, `critic.py`, `rollout_manager.py`）
- `verl/trainer/grpo/`：GRPO 实现
- `verl/reward/`：Reward 函数实现
- `verl/workers/actor_rollout_worker.py`：Policy rollout
- `verl/workers/critic/`：Value function workers

#### 典型提问

```
"PPO 的 clip_ratio 一般设多少？"
→ 0.1-0.3（通常 0.2），KL 系数从 0.05 开始

"GRPO 和 PPO 的核心区别是什么？"
→ GRPO 用 group-based advantage normalization，不需要 critic

"训练 KL divergence 太大怎么办？"
→ 降低学习率、增大 KL 系数、减小 clip_ratio、检查 reward 缩放

"reward 函数的签名是什么？"
→ (prompt, completions, prompt_ids, completion_ids, **kwargs) -> Tensor[batch_size]
```

---

### fsdp-engine-expert — FSDP2 训练专家

**模型：** Opus | **工具：** Read, Grep, Glob（只读）

#### 专业领域

| 领域 | 覆盖内容 |
|------|---------|
| **FSDP2 架构** | 参数分片、Device Mesh、DTensor、混合精度 |
| **Worker 集成** | FSDPActorWorker、FSDPCriticWorker、FSDPReferenceWorker |
| **性能优化** | Activation Checkpointing、CPU Offload、梯度累积、通信重叠 |
| **Checkpoint** | FSDPCheckpointManager、分布式保存/加载 |

#### 关键文件

- `verl/workers/fsdp_workers.py`：主 FSDP worker 实现
- `verl/utils/fsdp_utils/`：核心工具（`fsdp_utils.py`, `checkpoint.py`, `mixed_precision.py`, `offload.py`）
- `verl/workers/config/fsdp_engine.py`：`FSDPEngineConfig` dataclass

#### 典型提问

```
"如何创建 DP + TP 的 device mesh？"
→ init_device_mesh('cuda', mesh_shape=(dp_size, tp_size), mesh_dim_names=('dp', 'tp'))

"训练 OOM 了怎么办？"
→ 依次尝试：activation checkpointing → CPU offload → 减 batch → 增 DP

"怎么给模型应用 FSDP2？"
→ apply_fsdp2(model, dp_mesh=dp_mesh, tp_mesh=tp_mesh, wrap_policy=..., mixed_precision_policy=...)

"checkpoint 保存失败？"
→ 用 FSDPCheckpointManager，确保所有 rank 参与 checkpoint 操作
```

---

### megatron-engine-expert — Megatron 并行专家

**模型：** Opus | **工具：** Read, Grep, Glob（只读）

#### 专业领域

| 领域 | 覆盖内容 |
|------|---------|
| **Megatron-Core 集成** | Pipeline Parallel、Tensor Parallel、Sequence Parallel、Context Parallel |
| **verl 集成** | `verl/models/mcore/`、`verl/utils/megatron_utils/`、checkpoint 转换 |
| **性能优化** | Pipeline schedule (1F1B, interleaved)、Activation 重计算、分布式优化器 |

#### 关键文件

- `verl/models/mcore/`：Megatron-Core 模型包装
- `verl/utils/megatron_utils/`：初始化和工具函数
- `verl/trainer/config/actor/megatron_actor.yaml`：配置

#### 典型提问

```
"如何配置 Megatron 的并行策略？"
→ tensor_model_parallel_size=4, pipeline_model_parallel_size=2, sequence_parallel=True

"Pipeline parallel 训练 hang 了？"
→ 检查所有 pipeline stage 的 batch size 一致、pipeline schedule 正确、forward/backward 调用匹配

"HF 模型怎么转 Megatron 格式？"
→ convert_hf_to_megatron(hf_state_dict, config)，注意 TP/PP 大小需匹配
```

---

### vllm-sglang-expert — 推理引擎专家

**模型：** Opus | **工具：** Read, Grep, Glob（只读）

#### 专业领域

| 领域 | 覆盖内容 |
|------|---------|
| **vLLM 集成** | 高吞吐推理、PagedAttention、连续批处理、LoRA 支持 |
| **SGLang 集成** | 结构化生成、多轮对话、Tool calling、约束生成 |
| **Rollout Workers** | VLLMRolloutWorker、SGLangRolloutWorker、异步生成 |
| **DataProto 集成** | vLLM/SGLang 输出转 DataProto |

#### 关键文件

- `verl/workers/rollout/vllm_rollout.py`：vLLM rollout worker
- `verl/workers/rollout/sglang_rollout.py`：SGLang rollout worker
- `verl/trainer/config/rollout/vllm.yaml` / `sglang.yaml`：配置
- `examples/sglang_multiturn/`：多轮对话示例

#### 典型提问

```
"vLLM 推理 OOM 了？"
→ 降 gpu_memory_utilization (0.7-0.8)、减 max_num_seqs、用更小 batch、开 TP

"SGLang 多轮对话怎么实现？"
→ 用 @sgl.function 装饰器 + sgl.gen()，支持 RadixAttention 前缀缓存

"怎么给 rollout worker 更新模型权重？"
→ vLLM 不支持热更新，需重启 worker 或用 LoRA adapter

"如何把 vLLM 输出转成 DataProto？"
→ 封装 prompts、completions、completion_ids、log_probs 到 DataProto.from_dict()
```

---

### ray-controller-expert — Ray 编排专家

**模型：** Opus | **工具：** Read, Grep, Glob（只读）

#### 专业领域

| 领域 | 覆盖内容 |
|------|---------|
| **Ray 架构** | 单控制器模式、`@ray.remote`、Object Store、Actor 生命周期 |
| **verl 控制器** | `RayPPOTrainer`、Worker 初始化与协调、DataProto 通信 |
| **资源管理** | GPU/CPU 分配、Object Store 内存、Worker 放置策略 |
| **Dispatch Modes** | ONE_TO_ALL、DP_COMPUTE、MEGATRON_COMPUTE |

#### 关键文件

- `verl/single_controller/base/`：`Worker` 基类、`@register` 装饰器
- `verl/trainer/ppo/ray_trainer.py`：`RayPPOTrainer` 实现
- `verl/trainer/main_ppo.py`：入口点

#### Dispatch Modes 详解

| Mode | 含义 | 使用场景 |
|------|------|---------|
| `ONE_TO_ALL` | 同样的数据发给所有 worker 副本 | 广播配置、同步权重 |
| `DP_COMPUTE` | 数据按 worker 分片（数据并行） | 批量训练、批量推理 |
| `MEGATRON_COMPUTE` | Megatron 风格的流水线并行执行 | 大模型流水线推理 |

#### 典型提问

```
"Worker 初始化失败怎么排查？"
→ 检查 GPU 数量是否匹配 num_gpus、模型能否放入 GPU、NCCL 初始化日志

"Object Store 满了？"
→ ray.init(object_store_memory=10*1024**3)、ray.internal.free([ref])、减小传输数据量

"如何高效传输大数据？"
→ ray.put(data) 一次，然后传 ref 给所有 worker（避免重复序列化）

"怎么定义一个新 Worker？"
→ @ray.remote(num_gpus=N) + 继承 Worker + @register(dispatch_mode=...) 标注方法
```

---

## Agent 协作模式

### 模式 1：规划 → 实施 → 审查

```
planner
  ↓ 制定实施计划
用户实施（可能咨询领域专家）
  ↓
simple-code-reviewer   ← 自动审查代码质量
  ↓
code-verifier          ← 自动跑 lint/test
  ↓
/review-pr             ← 完整 PR 审查
```

### 模式 2：分布式问题诊断

```
fsdp-engine-expert / megatron-engine-expert
  ↓ 分析问题（hang/OOM/数值错误）
  ↓ 参考 debug-distributed skill
  ↓ 提供诊断步骤和修复建议
ray-controller-expert
  ↓ 如果涉及 worker 编排问题
simple-code-reviewer
  ↓ 修复后自动审查
```

### 模式 3：推理 + 训练联合调试

```
vllm-sglang-expert     ← rollout 阶段问题
  +
fsdp-engine-expert     ← 训练阶段问题
  +
algorithm-expert       ← RL 算法层面问题
  ↓
ray-controller-expert  ← worker 协调问题
```

---

## 提示词模板

### 请求领域专家分析

```
请用 {expert-name} 帮我分析以下问题：

问题描述：{具体问题}

相关文件：
- {file1.py}
- {file2.py}

已尝试过：{已尝试的方案}

期望结果：{期望的行为}
实际结果：{实际的行为}
```

### 请求代码审查

```
请审查我刚修改的代码：

修改了以下文件：
- {file1.py}：{改了什么}
- {file2.py}：{改了什么}

这次改动的目的是：{目的}

特别关注：
- {关注点1}
- {关注点2}
```

### 请求实施规划

```
我想实现以下功能：{功能描述}

背景信息：
- {为什么需要这个功能}
- {有哪些约束条件}

参考实现：{已有的类似实现}

期望输出：详细的实施计划
```

### 请求分布式问题诊断

```
分布式训练出现了以下问题：

症状：{具体症状，如 hang/OOM/数值异常}
环境：{GPU 数量、节点数、NCCL 版本}
复现步骤：{如何复现}

日志信息：
```
{相关日志}
```

已启用的调试选项：
- NCCL_DEBUG=INFO
- TORCH_DISTRIBUTED_DEBUG=DETAIL
```
