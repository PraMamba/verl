# verl AI 辅助开发工具使用指南

> 本指南面向 verl 开发者，系统介绍 `verl/.claude/` 目录下所有 AI 辅助开发工具的用途、触发方式和最佳实践。

## 目录

- [整体架构](#整体架构)
- [快速上手](#快速上手)
- [命令 (Commands)](#命令-commands)
- [专家 Agent (Agents)](#专家-agent-agents)
- [编码规范 (Rules)](#编码规范-rules)
- [技能指南 (Skills)](#技能指南-skills)
- [数据文件 (Data)](#数据文件-data)
- [钩子 (Hooks)](#钩子-hooks)
- [典型工作流示例](#典型工作流示例)

---

## 整体架构

```
.claude/
├── agents/          ← 8 个领域专家 Agent（按需或自动激活）
│   ├── planner.md              # [Opus, 主动] 实施规划
│   ├── algorithm-expert.md     # [Opus] RL 算法专家
│   ├── fsdp-engine-expert.md   # [Opus] FSDP2 训练后端
│   ├── megatron-engine-expert.md # [Opus] Megatron-LM 集成
│   ├── vllm-sglang-expert.md   # [Opus] 推理引擎
│   ├── ray-controller-expert.md # [Opus] Ray 编排
│   ├── simple-code-reviewer.md # [Sonnet, 主动] 快速代码审查
│   └── code-verifier.md        # [Haiku, 主动] 格式/测试检查
│
├── commands/        ← 3 个用户可直接调用的命令
│   ├── create-pr.md            # /create-pr
│   ├── review-pr.md            # /review-pr
│   └── gen-commit-msg.md       # /gen-commit-msg
│
├── rules/           ← 4 套编码规范（自动加载）
│   ├── code-style.md           # 设计原则、命名、日志
│   ├── distributed.md          # 分布式训练模式
│   ├── testing.md              # 测试组织与模式
│   └── api-config.md           # Hydra 配置规范
│
��── skills/          ← 6 个步骤化技能指南
│   ├── add-reward.md           # 添加 Reward 函数
│   ├── add-dataset.md          # 添加 Dataset Loader
│   ├── add-worker.md           # 添加 Ray Worker
│   ├── add-unit-tests/SKILL.md # 添加单元测试
│   └── debug-distributed.md    # 调试分布式问题
│
├── data/            ← PR Review 的参考数据
│   ├── review-pr-change-types.md
│   └── review-pr-templates.md
│
├── hooks/           ← 自动化钩子
│   └── check-expert-update.sh
│
├── settings.json    ← 钩子配置
└── README.md
```

### 工具层级关系

```
你（开发者）
  │
  ├─ 直接调用 ──→ 命令 (Commands)     : /create-pr, /review-pr, /gen-commit-msg
  │
  ├─ 自动触发 ──→ 主动 Agent          : planner, simple-code-reviewer, code-verifier
  │              ──→ 编码规范 (Rules)   : 根据文件路径自动加载
  │              ──→ 钩子 (Hooks)      : 编辑文件后自动运行
  │
  ├─ 按需请求 ──→ 领域专家 Agent      : algorithm-expert, fsdp-engine-expert 等
  │
  └─ 引导流程 ──→ 技能 (Skills)       : add-reward, add-worker 等
```

---

## 快速上手

### 日常开发三件套

| 场景 | 该用什么 | 怎么用 |
|------|---------|--------|
| 写完代码要提交 | `/gen-commit-msg` | 自动分析 staged changes，生成规范化提交消息 |
| 准备发起 PR | `/create-pr` | 自动 rebase、squash、推送、创建 PR |
| PR 需要 review | `/review-pr` | 动态分配专家，按风险等级审查 |

### 自动化保障（无需手动操作）

| 自动行为 | 触发时机 | 做什么 |
|---------|---------|--------|
| **code-verifier** | 代码修改后 | 跑 pre-commit、Ruff、mypy、pytest |
| **simple-code-reviewer** | 代码修改后 | 检查 verl 模式、分布式代码、Ray 模式 |
| **planner** | 多文件改动 / 新功能 | 制定实施计划，识别模式 |
| **check-expert-update.sh** | 编辑文件后 | 提醒更新对应专家 Agent 文档 |

---

## 命令 (Commands)

### `/create-pr` — 一键创建 PR

**用途：** 从 rebase 到创建 GitHub PR 的完整流程。

**使用方法：**
```
/create-pr              # 标准模式
/create-pr --draft      # 创建 draft PR
/create-pr --base dev   # 指定目标分支
```

**完整流程：**
1. 检查前置条件（不在 main 分支、无未提交变更）
2. 检查是否已有 PR（已有则征求覆盖许可）
3. `git fetch origin main && git rebase origin/main`
4. 将所有 commit squash 为单个 commit
5. 生成规范化提交消息（`<type>(<scope>): <subject>`）
6. Force push 到远端
7. 通过 `gh` CLI 创建/更新 PR

**PR 标题格式：** `[{modules}] {type}: {description}`

**模块映射：**

| 文件路径 | 模块名 |
|---------|--------|
| `verl/workers/` | `worker` |
| `verl/workers/rollout/` | `rollout` |
| `verl/trainer/` | `trainer` |
| `verl/reward/` | `reward` |
| `verl/data/` | `data` |
| `verl/models/` | `model` |
| `verl/utils/fsdp_utils/` | `fsdp` |
| `verl/models/mcore/` | `megatron` |
| `verl/single_controller/` | `ray` |
| `verl/checkpoint_engine/` | `ckpt` |
| `verl/trainer/config/` | `cfg` |

**安全机制：**
- 关键步骤（squash、force push、创建 PR）前均需用户确认
- Rebase 冲突时自动 abort 并提示手动解决
- 自动创建备份分支

---

### `/review-pr` — 动态智能 PR 审查

**用途：** 根据 PR 变更内容自动组装专家团队进行代码审查。

**使用方法：**
```
/review-pr           # 审查当前分支的 PR
/review-pr 123       # 审查指定 PR 编号
/review-pr --quick   # 快速模式（仅 Phase 1 分析）
/review-pr --economy # 经济模式（降低模型配置）
```

**四阶段流程：**

| 阶段 | 职责 | 模型 |
|------|------|------|
| **Phase 1** | 深度 PR 分析：检测变更类型、识别风险 | Haiku + Sonnet |
| **Phase 2** | 动态任务规划：按风险生成审查任务 | Sonnet |
| **Phase 3** | 并行执行审查任务 | Opus / Sonnet / Haiku |
| **Phase 4** | 置信度评分 & 汇总报告 | Haiku |

**模型选择策略：**

| 风险等级 | 默认模式 | 快速模式 | 经济模式 |
|---------|---------|---------|---------|
| CRITICAL/HIGH | Opus | Sonnet | Sonnet |
| MEDIUM | Sonnet | Sonnet | Haiku |
| LOW | Haiku | Sonnet | Haiku |

**风险分类示例：**
- **CRITICAL：** FSDP 核心、Megatron 核心、DCP checkpoint、Ray controller
- **HIGH：** 分布式通信、DTensor、张量并行、Trainer 核心
- **MEDIUM：** Tensor 操作、vLLM/SGLang rollout、API 配置
- **LOW：** 测试、文档、配置

---

### `/gen-commit-msg` — 智能提交消息生成

**用途：** 分析 staged changes，生成符合 Conventional Commits 规范的提交消息。

**使用方法：**
```
/gen-commit-msg                  # 标准模式
/gen-commit-msg --amend          # 修改上一个 commit
/gen-commit-msg --scope workers  # 强制指定 scope
```

**消息格式：**
```
<type>(<scope>): <subject>

<body>

Key changes:
- change 1
- change 2

Refs: #123, #456
```

**Type 分类：**

| Type | 何时使用 |
|------|---------|
| `feat` | 新功能/能力 |
| `fix` | Bug 修复 |
| `docs` | 仅文档变更 |
| `refactor` | 不改变功能的代码重构 |
| `test` | 添加/修复测试 |
| `chore` | 构建、依赖、配置变更 |
| `perf` | 性能优化 |

---

## 专家 Agent (Agents)

> 详见 [agents-guide.md](./agents-guide.md) 获取完整 Agent 使用指南。

### 概览

| Agent | 模型 | 激活方式 | 核心能力 |
|-------|------|---------|---------|
| **planner** | Opus | 主动 | 复杂任务的实施规划 |
| **algorithm-expert** | Opus | 手动 | PPO/GRPO/RLOO 等 RL 算法 |
| **fsdp-engine-expert** | Opus | 手动 | FSDP2 参数分片、设备网格 |
| **megatron-engine-expert** | Opus | 手动 | Megatron-LM 流水线/张量并行 |
| **vllm-sglang-expert** | Opus | 手动 | vLLM/SGLang 推理引擎 |
| **ray-controller-expert** | Opus | 手动 | Ray 单控制器模式、Worker 编排 |
| **simple-code-reviewer** | Sonnet | 主动 | 快速代码质量检查 |
| **code-verifier** | Haiku | 主动 | Ruff/mypy/pytest 自动检查 |

### 何时使用哪个 Agent

```
需要规划多文件改动？          → planner（自动激活）
在修改 RL 算法/reward？       → algorithm-expert
在修改 FSDP/分布式训练？      → fsdp-engine-expert
在修改 Megatron 模型并行？    → megatron-engine-expert
在修改 vLLM/SGLang rollout？  → vllm-sglang-expert
在修改 Ray worker 编排？      → ray-controller-expert
想快速检查代码质量？          → simple-code-reviewer（自动激活）
想跑 lint/test 检查？         → code-verifier（自动激活）
```

---

## 编码规范 (Rules)

> 详见 [rules-guide.md](./rules-guide.md) 获取完整规范说明。

Rules 是**自动加载**的编码规范，根据文件匹配模式决定何时激活：

| 规范文件 | 匹配范围 | 核心内容 |
|---------|---------|---------|
| `code-style.md` | **始终加载** | 设计原则、命名约定、日志规范 |
| `distributed.md` | `verl/workers/**`, `verl/models/**`, `verl/utils/fsdp_utils/**` | 分布式训练模式、进程组管理 |
| `testing.md` | `**/tests/**`, `*_test.py`, `test_*.py` | 测试组织、Pytest marker、断言模式 |
| `api-config.md` | `verl/trainer/config/**`, `verl/workers/config/**` | Hydra 配置结构与验证 |

---

## 技能指南 (Skills)

> 详见 [skills-guide.md](./skills-guide.md) 获取完整技能指南。

Skills 提供步骤化的操作引导，适用于重复性开发任务：

| Skill | 用途 | 预计时间 |
|-------|------|---------|
| `add-reward` | 添加新的 Reward 函数 | ~20 min |
| `add-dataset` | 添加新的 Dataset Loader | ~30 min |
| `add-worker` | 添加新的 Ray Worker | ~40 min |
| `add-unit-tests` | 为新功能添加单元测试 | ~20 min |
| `debug-distributed` | 调试分布式训练问题 | 视情况 |

---

## 数据文件 (Data)

数据文件为 `/review-pr` 命令提供参考知识库：

### `review-pr-change-types.md`

定义了变更类型检测表，按风险等级分类：

- **CRITICAL：** FSDP_CORE, MEGATRON_CORE, DCP_CHECKPOINT, RAY_CONTROLLER
- **HIGH：** DISTRIBUTED_COMM, DTENSOR, TENSOR_PARALLEL, SEQUENCE_PARALLEL
- **MEDIUM：** TENSOR_OPS, NUMERICAL, ROLLOUT_VLLM, ROLLOUT_SGLANG, API_CONFIG
- **LOW：** TESTS, DOCS, CONFIG_ONLY, EXAMPLES

还包含框架级风险识别规则和风险关联规则。

### `review-pr-templates.md`

为不同变更类型提供审查任务模板：

- **框架专项：** FSDP、Megatron、Ray Controller、DCP、vLLM/SGLang、Trainer
- **通用专项：** 逻辑正确性、并发安全、张量形状、数值稳定性、TP 一致性、通信模式、API 兼容性

---

## 钩子 (Hooks)

### `check-expert-update.sh`

**触发条件：** 每次通过 Write 或 Edit 工具修改文件时自动运行。

**作用：** 检查被修改的文件是否与某个专家 Agent 文档相关，如果是，则提示开发者同步更新对应的 Agent 文档。

**映射关系：**

| 文件模式 | 对应 Agent |
|---------|-----------|
| `verl/workers/fsdp_workers.py` | fsdp-engine-expert.md |
| `verl/utils/fsdp_utils/` | fsdp-engine-expert.md |
| `verl/models/mcore/` | megatron-engine-expert.md |
| `verl/utils/megatron_utils/` | megatron-engine-expert.md |
| `verl/trainer/ppo/` | algorithm-expert.md |
| `verl/reward/` | algorithm-expert.md |
| `verl/workers/rollout/vllm*` | vllm-sglang-expert.md |
| `verl/workers/rollout/sglang*` | vllm-sglang-expert.md |
| `verl/single_controller/` | ray-controller-expert.md |

**配置位置：** `settings.json`

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/check-expert-update.sh"
          }
        ]
      }
    ]
  }
}
```

---

## 典型工作流示例

### 场景 1：添加新的 Reward 函数

```
1. 告诉 AI："我要添加一个 math accuracy reward"
   ↓ planner 自动激活，制定实施计划
2. AI 参考 add-reward skill，引导你创建文件
   ↓ code-verifier 自动跑 lint 检查
3. 完成代码后：
   /gen-commit-msg           ← 生成提交消息
   /create-pr               ← 创建 PR
   /review-pr               ← 动态审查
```

### 场景 2：修复 FSDP 分布式训练 Bug

```
1. 描述问题："训练在 all_reduce 时 hang 住了"
   ↓ fsdp-engine-expert 被请求激活
   ↓ distributed.md 规范自动加载
2. AI 参考 debug-distributed skill 提供诊断步骤
3. 修复后：
   ↓ simple-code-reviewer 自动检查分布式模式
   ↓ code-verifier 自动跑测试
4. /create-pr 提交修复
```

### 场景 3：添加新的 Ray Worker

```
1. 告诉 AI："我要添加一个 RewardModelWorker"
   ↓ planner 自动激活
   ↓ AI 参考 add-worker skill
2. 创建 worker 文件、config、测试
   ↓ ray-controller-expert 可按需咨询 dispatch mode
   ↓ code-style.md 规范自动加载
3. /create-pr 一键提交
```

### 场景 4：大规模 PR 审查

```
/review-pr 456
   ↓ Phase 1: 分析 PR，检测到 FSDP_CORE + DISTRIBUTED_COMM + TESTS
   ↓ Phase 2: 生成 6 个审查任务（3 Opus + 2 Sonnet + 1 Haiku）
   ↓ Phase 3: 并行执行所有审查任务
   ↓ Phase 4: 汇总报告，置信度评分
   → 输出结构化审查报告
```

---

## 维护指南

### 何时更新各组件

| 组件 | 更新时机 |
|------|---------|
| **Agents** | verl 架构演进、框架 API 变更、新领域专家需求 |
| **Rules** | 编码规范变更、新最佳实践、框架 API 演进 |
| **Skills** | 新的常见任务、现有流程改进 |
| **Commands** | PR 模板变更、新模块 scope、审查类型演化 |
| **Data** | 新的变更类型、审查模板优化 |
| **Hooks** | 新的文件-Agent 映射关系 |
