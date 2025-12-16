# GRPO 训练框架实现详解

**面向对象**: 基础设施（Infrastructure）初学者
**最后更新**: 2025-11-27

---

## 目录

1. [什么是 GRPO？](#1-什么是-grpo)
2. [GRPO 的核心思想](#2-grpo-的核心思想)
3. [整体架构设计](#3-整体架构设计)
4. [核心组件实现](#4-核心组件实现)
5. [训练流程详解](#5-训练流程详解)
6. [关键代码解析](#6-关键代码解析)
7. [配置与使用](#7-配置与使用)
8. [总结](#8-总结)

---

## 1. 什么是 GRPO？

### 1.1 GRPO 简介

**GRPO** (Group Relative Policy Optimization，组相对策略优化) 是 DeepSeek 在 DeepSeekMath 论文中提出的一种简化版强化学习算法。

**核心特点**：**不需要 Critic 网络！**

```
传统 PPO:
  Actor (策略网络)  +  Critic (价值网络)
  ↓                    ↓
  输出动作             估计价值
  ↓                    ↓
  ────────→ 计算 Advantage ←────────

GRPO:
  Actor (策略网络) ONLY!
  ↓
  输出多个动作（组采样）
  ↓
  组内比较 → 计算 Advantage
```

### 1.2 为什么叫"组相对"（Group Relative）？

**"组"（Group）**：
- 对同一个问题（prompt），生成多个回答
- 这些回答组成一个"组"

**"相对"（Relative）**：
- 不看绝对分数有多高
- 只看相对于组内平均水平如何

**直观例子**：

```
问题: "计算 12 × 15 = ?"

GRPO 生成 4 个回答（一组）:
  回答1: "180" → 正确 → 得分 1.0
  回答2: "160" → 错误 → 得分 0.0
  回答3: "180" → 正确 → 得分 1.0
  回答4: "175" → 错误 → 得分 0.0

组内平均分: (1.0 + 0.0 + 1.0 + 0.0) / 4 = 0.5

计算 Advantage（相对优势）:
  回答1: 1.0 - 0.5 = +0.5  ✓ 比平均好
  回答2: 0.0 - 0.5 = -0.5  ✗ 比平均差
  回答3: 1.0 - 0.5 = +0.5  ✓ 比平均好
  回答4: 0.0 - 0.5 = -0.5  ✗ 比平均差

训练效果:
  - 增强回答1和3（正确的）
  - 抑制回答2和4（错误的）
```

### 1.3 与其他算法的对比

| 算法 | Critic 网络 | Advantage 计算 | 组采样 | 显存占用 | 训练速度 |
|------|------------|---------------|--------|---------|---------|
| **PPO** | 需要 | GAE (需要 Value) | 可选 | 高 | 慢 |
| **GRPO** | 不需要 ✓ | 组内归一化 | 必须 | 低 ✓ | 快 ✓ |
| **DAPO** | 不需要 | GRPO + 过滤 | 必须 | 低 | 中等 |
| **REINFORCE** | 不需要 | 简单差分 | 不需要 | 最低 | 最快 |

**GRPO 的优势**：
- ✅ **省显存**: 不需要训练 Critic，只需要一个大模型
- ✅ **省时间**: 不需要 Critic 的前向和反向传播
- ✅ **易实现**: 算法更简单，代码更少
- ✅ **效果好**: 在数学推理任务上表现优秀

**GRPO 的劣势**：
- ❌ **需要组采样**: 每个 prompt 必须生成多个回答（增加生成成本）
- ❌ **组内依赖**: 如果组内所有回答都一样好/差，学不到东西
- ❌ **不适合某些任务**: 对话任务可能不如 PPO

---

## 2. GRPO 的核心思想

### 2.1 为什么不需要 Critic？

让我们先理解 PPO 为什么需要 Critic：

**PPO 的 Advantage 计算**：
```python
# PPO 需要估计"平均水平"
Advantage = Q(s,a) - V(s)
          = "这个动作的价值" - "平均价值"
          ↑                   ↑
      实际回报              需要 Critic 估计
```

**GRPO 的巧妙之处**：

```python
# GRPO 直接用组内平均代替 Critic 的估计
Advantage = score_i - mean(scores_in_group)
          = "这个回答的得分" - "组内平均得分"
          ↑                   ↑
      实际得分            直接计算，不需要网络估计！
```

**关键洞察**：
- 如果我们对同一个问题生成多个回答
- 这些回答的平均得分 ≈ 这个问题的"期望价值"
- 不需要 Critic 来估计！

### 2.2 组采样（Group Sampling）

GRPO 的核心机制是**组采样**：对每个 prompt 生成多个回答。

**配置**：
```yaml
actor_rollout_ref.rollout.n: 5  # 每个 prompt 生成 5 个回答
```

**数据流**：

```
输入: 1 个 prompt
  "计算 25 × 4 = ?"

     ↓ 组采样 (n=5)

输出: 5 个回答（一个组）
  回答1: "100" → 正确
  回答2: "100" → 正确
  回答3: "80"  → 错误
  回答4: "100" → 正确
  回答5: "105" → 错误

     ↓ 评分

得分: [1.0, 1.0, 0.0, 1.0, 0.0]

     ↓ 组内归一化

组内平均: 0.6
Advantages: [+0.4, +0.4, -0.6, +0.4, -0.6]

     ↓ 更新策略

增强: 回答1, 2, 4 (正确的)
抑制: 回答3, 5 (错误的)
```

**为什么需要多个回答？**

假设只生成 1 个回答：
```python
回答: "100" → 得分 1.0
组内平均: 1.0
Advantage: 1.0 - 1.0 = 0  # 没有梯度！
```

无法学习！必须有对比才能知道好坏。

### 2.3 Advantage 计算公式

GRPO 的 Advantage 计算非常简洁：

**标准 GRPO**:
```python
# 1. 计算每个回答的总得分
score_i = sum(token_rewards)  # 通常只有最后一个 token 有 reward

# 2. 按组分组
group_scores = {
    prompt_1: [score_1_1, score_1_2, ..., score_1_n],
    prompt_2: [score_2_1, score_2_2, ..., score_2_n],
    ...
}

# 3. 计算每组的均值和标准差
for group_id, scores in group_scores.items():
    mean = mean(scores)
    std = std(scores)

    # 4. 归一化（Z-score normalization）
    for i, score in enumerate(scores):
        advantage[i] = (score - mean) / (std + epsilon)
```

**数学表达**：

$$
A_i = \frac{r_i - \mu_g}{\sigma_g + \epsilon}
$$

其中：
- $A_i$: 第 $i$ 个回答的 Advantage
- $r_i$: 第 $i$ 个回答的得分
- $\mu_g$: 所属组 $g$ 的平均得分
- $\sigma_g$: 所属组 $g$ 的得分标准差
- $\epsilon$: 防止除零的小常数

**为什么除以标准差？**

```
情况1: 组内得分差异大
  scores = [1.0, 0.8, 0.2, 0.0]
  mean = 0.5, std = 0.41
  advantages = [1.22, 0.73, -0.73, -1.22]  # 拉大差距

情况2: 组内得分差异小
  scores = [0.6, 0.55, 0.5, 0.45]
  mean = 0.525, std = 0.06
  advantages = [1.25, 0.42, -0.42, -1.25]  # 同样拉大

作用: 归一化让不同难度的问题有相似的梯度尺度
```

**DrGRPO 变体**（不除以标准差）:

```python
# DrGRPO: 不归一化标准差
advantage[i] = score - mean  # 只减均值

优点: 避免"长度偏差"（后面会详细讲）
```

### 2.4 与 PPO 的详细对比

| 对比维度 | PPO | GRPO |
|---------|-----|------|
| **Advantage 来源** | Critic 估计 V(s) | 组内平均 |
| **训练的网络** | Actor + Critic | Actor only |
| **每个 prompt 生成** | 1 个回答 | n 个回答 (n>1) |
| **Advantage 计算** | GAE（复杂） | 组内归一化（简单） |
| **显存占用** | 需要额外存 Critic | 只存 Actor |
| **训练时间** | 更新 2 个网络 | 更新 1 个网络 |
| **样本复用** | 可多轮（3-5 epochs） | 通常 1 轮 |
| **适用任务** | 通用（对话、推理） | 推理任务（数学、代码） |

**计算量对比**（以 7B 模型为例）：

```
PPO (batch_size=256, n=4):
  生成: 256 * 4 = 1024 序列
  Critic 前向: 1024 序列  ← 额外开销
  Actor 更新: 1024 序列 × 4 epochs
  Critic 更新: 1024 序列 × 4 epochs  ← 额外开销
  总计: ~5x Actor 前向

GRPO (batch_size=256, n=4):
  生成: 256 * 4 = 1024 序列
  Actor 更新: 1024 序列 × 1 epoch
  总计: ~1x Actor 前向

速度提升: ~5x faster! ✓
```

---

## 3. 整体架构设计

### 3.1 系统架构图

```
┌────────────────────────────────────────────────────────────┐
│                  Ray Cluster (分布式环境)                   │
├────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           RayPPOTrainer (训练协调器)                 │   │
│  │                                                       │   │
│  │  核心方法:                                            │   │
│  │  • fit()              - 主训练循环                   │   │
│  │  • compute_advantage  - GRPO 组内归一化              │   │
│  │  • _validate()        - 验证                         │   │
│  └─────────────────────────────────────────────────────┘   │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Actor     │  │   Rollout    │  │  RefPolicy   │      │
│  │WorkerGroup  │  │ WorkerGroup  │  │ WorkerGroup  │      │
│  │             │  │              │  │              │      │
│  │• 更新策略    │  │• 生成 n 个   │  │• 计算 KL     │      │
│  │• 计算 logp  │  │  回答/prompt │  │  (可选)      │      │
│  │             │  │• 记录 logp   │  │              │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │             Reward Function (规则评分)                │  │
│  │  - 数学题: 检查答案是否正确                           │  │
│  │  - 代码: 运行测试用例                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      Driver Process (Controller)                      │  │
│  │                                                        │  │
│  │  • compute_grpo_advantage()  - 组内归一化             │  │
│  │  • apply_kl_loss()           - KL 正则化 (在 loss 中)│  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ❌ 不需要 Critic Worker Group ❌                          │
│                                                              │
└────────────────────────────────────────────────────────────┘
```

**关键差异（对比 PPO）**：
1. ❌ **没有 Critic Worker**: 省了一个大模型的显存和计算
2. ✅ **Rollout.n > 1**: 必须生成多个回答
3. ✅ **KL Loss**: 通常在 Actor loss 中加 KL，而不是在 reward 中

### 3.2 数据流

```
输入数据 (Prompts)  [batch_size = 256]
    │
    ▼
┌──────────────────┐
│  1. Rollout      │  每个 prompt 生成 n=4 个回答
│  (组采样)         │  → 256 * 4 = 1024 responses
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ 2. Compute       │  重新计算当前策略的 log_probs
│    Old LogProb   │  → old_log_probs
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ 3. Compute       │  [GRPO 特色] 计算参考策略 log_probs
│    Ref LogProb   │  → ref_log_probs (用于 KL loss)
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ 4. Reward        │  评估 1024 个回答
│  (奖励计算)       │  → token_level_rewards
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ 5. GRPO          │  [GRPO 核心] 组内归一化
│    Advantage     │  → advantages
│                  │
│  按 UID 分组:    │
│  Group 1: [r1, r2, r3, r4]  → adv1
│  Group 2: [r5, r6, r7, r8]  → adv2
│  ...             │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ 6. Update Actor  │  [GRPO 特色] 通常只更新 1 轮
│  (策略更新)       │
│                  │
│  Loss = PG Loss + KL Loss
│  (没有 Critic loss)
└──────────────────┘
```

**与 PPO 的流程对比**：

| 步骤 | PPO | GRPO |
|------|-----|------|
| Rollout | 每个 prompt 生成 1-4 个 | 每个 prompt 生成 4-8 个 ✓ |
| Compute Values | ✓ 需要 | ❌ 不需要 |
| Compute Ref LogProb | 可选 | 几乎必须（用于 KL loss） |
| Advantage 计算 | GAE | 组内归一化 ✓ |
| Update Critic | ✓ 需要 | ❌ 不需要 |
| Update Actor | 3-5 轮 | 通常 1 轮 ✓ |

### 3.3 组的概念

GRPO 中"组"（Group）的理解非常重要：

```python
# 输入 batch
prompts = [
    "计算 12 × 15 = ?",  # prompt_0
    "计算 25 × 4 = ?",   # prompt_1
]
batch_size = 2

# 每个 prompt 生成 n=4 个回答
rollout.n = 4

# 生成后的数据结构
responses = [
    # Group 0 (来自 prompt_0)
    "12 × 15 = 180",     # response_0  ← 同组
    "12 × 15 = 160",     # response_1  ← 同组
    "12 × 15 = 180",     # response_2  ← 同组
    "12 × 15 = 175",     # response_3  ← 同组

    # Group 1 (来自 prompt_1)
    "25 × 4 = 100",      # response_4  ← 同组
    "25 × 4 = 100",      # response_5  ← 同组
    "25 × 4 = 80",       # response_6  ← 同组
    "25 × 4 = 100",      # response_7  ← 同组
]

# UID (唯一标识符，标记同一组)
uid = [
    "uuid_0", "uuid_0", "uuid_0", "uuid_0",  # 同一组
    "uuid_1", "uuid_1", "uuid_1", "uuid_1",  # 同一组
]

# 得分
scores = [1, 0, 1, 0,  # Group 0
          1, 1, 0, 1]  # Group 1

# Advantage 计算（按组）
Group 0:
  mean = (1+0+1+0)/4 = 0.5
  std = 0.577
  advantages = [(1-0.5)/0.577, (0-0.5)/0.577, (1-0.5)/0.577, (0-0.5)/0.577]
             = [0.87, -0.87, 0.87, -0.87]

Group 1:
  mean = (1+1+0+1)/4 = 0.75
  std = 0.5
  advantages = [(1-0.75)/0.5, (1-0.75)/0.5, (0-0.75)/0.5, (1-0.75)/0.5]
             = [0.5, 0.5, -1.5, 0.5]
```

**关键点**：
- 同一个 prompt 的所有回答在一个组内
- 只在组内做归一化，不跨组
- UID 用于标识哪些回答属于同一组

---

## 4. 核心组件实现

### 4.1 GRPO Advantage 计算

**位置**: `verl/trainer/ppo/core_algos.py:compute_grpo_outcome_advantage`

这是 GRPO 最核心的函数！

```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,  # [bs, seq_len]
    response_mask: torch.Tensor,         # [bs, seq_len]
    index: np.ndarray,                   # [bs], 组 ID
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO 的 Advantage 计算

    核心思想: 组内归一化
    """
    # 1. 计算每个序列的总得分
    # token_level_rewards: [bs, seq_len]
    # 通常只有最后一个 token 有非零 reward
    scores = token_level_rewards.sum(dim=-1)  # [bs]

    # 2. 按组分组收集得分
    id2score = defaultdict(list)  # {group_id: [score1, score2, ...]}
    id2mean = {}   # {group_id: mean}
    id2std = {}    # {group_id: std}

    with torch.no_grad():  # 不需要梯度
        bsz = scores.shape[0]

        # 2.1 收集每个组的所有得分
        for i in range(bsz):
            group_id = index[i]  # 这个样本属于哪个组
            id2score[group_id].append(scores[i])

        # 2.2 计算每个组的统计量
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # 只有一个样本，无法归一化
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                # 正常情况：多个样本
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        # 2.3 归一化每个得分
        for i in range(bsz):
            group_id = index[i]
            if norm_adv_by_std_in_grpo:
                # 标准 GRPO: (score - mean) / std
                scores[i] = (scores[i] - id2mean[group_id]) / (id2std[group_id] + epsilon)
            else:
                # DrGRPO: score - mean (不除以 std)
                scores[i] = scores[i] - id2mean[group_id]

        # 3. 广播到所有 token（乘以 mask）
        # scores: [bs] → [bs, 1] → [bs, seq_len]
        scores = scores.unsqueeze(-1) * response_mask

    # 4. GRPO 中 advantages = returns (没有 value baseline)
    return scores, scores
```

**详细例子**：

```python
# 假设数据
token_level_rewards = [
    [0, 0, 0, 1.0],  # response_0, group_0
    [0, 0, 0, 0.0],  # response_1, group_0
    [0, 0, 0, 1.0],  # response_2, group_0
    [0, 0, 0, 0.0],  # response_3, group_0
    [0, 0, 0, 1.0],  # response_4, group_1
    [0, 0, 0, 1.0],  # response_5, group_1
    [0, 0, 0, 0.0],  # response_6, group_1
    [0, 0, 0, 1.0],  # response_7, group_1
]
response_mask = [[1,1,1,1]] * 8  # 全部有效
index = [0, 0, 0, 0, 1, 1, 1, 1]  # 组 ID

# Step 1: 计算总得分
scores = [1.0, 0.0, 1.0, 0.0,  # group 0
          1.0, 1.0, 0.0, 1.0]  # group 1

# Step 2: 按组分组
id2score = {
    0: [1.0, 0.0, 1.0, 0.0],
    1: [1.0, 1.0, 0.0, 1.0]
}

# Step 3: 计算统计量
Group 0:
  mean = (1.0 + 0.0 + 1.0 + 0.0) / 4 = 0.5
  std = sqrt(((1-0.5)^2 + (0-0.5)^2 + (1-0.5)^2 + (0-0.5)^2) / 4)
      = sqrt((0.25 + 0.25 + 0.25 + 0.25) / 4)
      = sqrt(0.25) = 0.5

Group 1:
  mean = (1.0 + 1.0 + 0.0 + 1.0) / 4 = 0.75
  std = sqrt(((1-0.75)^2 + (1-0.75)^2 + (0-0.75)^2 + (1-0.75)^2) / 4)
      = sqrt((0.0625 + 0.0625 + 0.5625 + 0.0625) / 4)
      = sqrt(0.1875) = 0.433

# Step 4: 归一化
Group 0 advantages:
  (1.0 - 0.5) / 0.5 = 1.0
  (0.0 - 0.5) / 0.5 = -1.0
  (1.0 - 0.5) / 0.5 = 1.0
  (0.0 - 0.5) / 0.5 = -1.0

Group 1 advantages:
  (1.0 - 0.75) / 0.433 = 0.577
  (1.0 - 0.75) / 0.433 = 0.577
  (0.0 - 0.75) / 0.433 = -1.732
  (1.0 - 0.75) / 0.433 = 0.577

# Step 5: 广播到所有 token
advantages = [
    [1.0, 1.0, 1.0, 1.0],      # response_0
    [-1.0, -1.0, -1.0, -1.0],  # response_1
    [1.0, 1.0, 1.0, 1.0],      # response_2
    [-1.0, -1.0, -1.0, -1.0],  # response_3
    [0.577, 0.577, 0.577, 0.577],   # response_4
    [0.577, 0.577, 0.577, 0.577],   # response_5
    [-1.732, -1.732, -1.732, -1.732], # response_6
    [0.577, 0.577, 0.577, 0.577],   # response_7
]
```

### 4.2 向量化版本（GRPO_VECTORIZED）

上面的实现用 Python 循环，效率较低。verl 还提供了向量化版本：

```python
@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    向量化 GRPO Advantage 计算

    比循环版本快 10-100 倍！
    """
    with torch.no_grad():
        # 1. 计算总得分
        scores = token_level_rewards.sum(dim=-1)  # [bs]

        # 2. 转换 index 为 torch tensor
        g = as_torch_index(index, device=scores.device)  # [bs]

        # 3. 使用 scatter 操作计算组统计量
        # group_mean_std 是一个高效的辅助函数
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon)
        # mean_g: [num_groups]
        # std_g: [num_groups]

        # 4. 归一化
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]

        # 5. 广播
        advantages = scalars.unsqueeze(-1) * response_mask

        return advantages, advantages
```

**group_mean_std 的实现**（简化版）:

```python
def group_mean_std(scores, group_indices, eps=1e-6):
    """
    高效计算每个组的均值和标准差

    使用 scatter_add 实现，避免循环
    """
    num_groups = group_indices.max() + 1

    # 计算每组的和
    group_sum = torch.zeros(num_groups, device=scores.device)
    group_sum.scatter_add_(0, group_indices, scores)

    # 计算每组的样本数
    group_count = torch.zeros(num_groups, device=scores.device)
    group_count.scatter_add_(0, group_indices, torch.ones_like(scores))

    # 计算均值
    mean_g = group_sum / (group_count + eps)

    # 计算标准差
    diff = scores - mean_g[group_indices]
    group_var_sum = torch.zeros(num_groups, device=scores.device)
    group_var_sum.scatter_add_(0, group_indices, diff ** 2)
    std_g = torch.sqrt(group_var_sum / (group_count + eps) + eps)

    return mean_g, std_g, group_count
```

**性能对比**：

```python
# 测试数据
batch_size = 10000
num_groups = 2000

# 循环版本
time: ~500ms

# 向量化版本
time: ~5ms

加速: 100x! ✓
```

### 4.3 Pass@k 变体（GRPO_PASSK）

GRPO 还有一个特殊变体，用于 Pass@k 评估：

```python
@register_adv_est(AdvantageEstimator.GRPO_PASSK)
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pass@k 风格的 GRPO

    核心思想: 只奖励每组中最好的回答
    """
    scores = token_level_rewards.sum(dim=-1)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        # 1. 收集每组的得分和索引
        for i in range(len(scores)):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        # 2. 对每个组，只奖励最好的回答
        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])

            # 找到 top-2
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group")

            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]

            # 只给最好的回答非零 advantage
            advantages[i_max] = r_max - r_second_max
            # 其他回答的 advantage = 0

        # 3. 广播
        advantages = advantages.unsqueeze(-1) * response_mask

    return advantages, advantages
```

**Pass@k 的动机**：

```
问题: "写一个排序函数"

生成 5 个回答:
  回答1: 冒泡排序 (正确) → 1.0
  回答2: 快速排序 (正确) → 1.0
  回答3: 错误代码 → 0.0
  回答4: 错误代码 → 0.0
  回答5: 错误代码 → 0.0

标准 GRPO:
  mean = 0.4
  advantages = [0.6, 0.6, -0.4, -0.4, -0.4]
  → 同时增强回答1和2

Pass@k GRPO:
  只奖励最好的一个（或者说最好的和第二好的差距）
  advantages = [0.0, 0.0, 0.0, 0.0, 0.0]  # r_max = r_second_max = 1.0
  → 鼓励"多样性"而不是"重复正确答案"
```

---

## 5. 训练流程详解

### 5.1 完整训练步骤

```
Step 0: 初始化
├─ 加载预训练模型
├─ 创建 Actor, RefPolicy Worker Groups
├─ ❌ 不创建 Critic Worker
├─ 加载训练数据
└─ 设置配置参数

Step 1: 数据加载
├─ 从 dataloader 获取一个 batch
├─ Batch 包含: {prompts, ground_truth, ...}
├─ batch_size = 256
└─ 例如: "计算 12 × 15 = ?"

Step 2: 组采样 Rollout
├─ 对每个 prompt 生成 n=4 个回答
├─ 使用 vLLM/SGLang 高效生成
├─ 记录 rollout_log_probs
├─ 添加 UID 标记（标识同组）
└─ 输出: 256 * 4 = 1024 responses

Step 3: Recompute Old Log Probs
├─ 使用当前 Actor 重新计算 log_probs
├─ 作为 PPO 裁剪的"旧策略"
└─ 输出: {old_log_probs}

Step 4: Compute Ref Log Probs
├─ [GRPO 特色] 几乎总是计算
├─ 用于 KL loss（在 loss 中，不在 reward 中）
└─ 输出: {ref_log_probs}

Step 5: Compute Rewards
├─ 使用 Reward Function 评估
├─ 通常是规则评分（数学题、代码）
└─ 输出: {token_level_rewards}

Step 6: ❌ 跳过 Compute Values
├─ PPO 需要这一步
└─ GRPO 不需要！

Step 7: GRPO Advantage 计算
├─ [GRPO 核心] 按 UID 分组
├─ 组内归一化: (score - mean) / std
├─ 广播到所有 token
└─ 输出: {advantages}

Step 8: Update Actor (单轮)
├─ [GRPO 特色] 通常只更新 1 轮
├─ 计算 PG loss (with clipping)
├─ 计算 KL loss: KL(π_new || π_ref)
├─ Total loss = PG loss + β * KL loss
└─ 反向传播，更新 Actor

Step 9: ❌ 跳过 Update Critic
├─ PPO 需要这一步
└─ GRPO 不需要！

Step 10: Validation & Checkpointing
├─ 定期验证
├─ 保存 checkpoint
└─ 记录 metrics

Step 11: 重复
└─ 回到 Step 1
```

### 5.2 时间线视图

```
时间轴 (以 7B 模型, batch_size=256, n=4 为例):
0s    3s        4s    5s      6s       7s
│     │         │     │       │        │
├─────┤         │     │       │        │
│Gen  │         │     │       │        │
│(x4) │         │     │       │        │  比 PPO 生成更多
│     │         │     │       │        │
│     ├─────────┤     │       │        │
│     │Old LogP │     │       │        │
│     │         │     │       │        │
│     │         ├─────┤       │        │
│     │         │ Ref │       │        │
│     │         │LogP │       │        │  GRPO 通常需要
│     │         │     │       │        │
│     │         │     ├───────┤        │
│     │         │     │Reward │        │
│     │         │     │+GRPO  │        │  组内归一化 (快)
│     │         │     │ Adv   │        │
│     │         │     │       │        │
│     │         │     │       ├────────┤
│     │         │     │       │ Update │
│     │         │     │       │ Actor  │  只更新 1 轮
│     │         │     │       │ (1轮)  │  ❌ 不更新 Critic
└─────┴─────────┴─────┴───────┴────────┘
```

**各阶段耗时** (7B 模型):
- Generation (x4): ~3s (生成 4 倍数据)
- Recompute LogProbs: ~1s
- Ref LogProbs: ~1s
- Reward + GRPO Adv: ~1s (GRPO adv 计算很快)
- Update Actor: ~1s (只更新 1 轮)

**总时间**: ~7s/step

**对比 PPO** (~10s/step):
- GRPO 生成更多 (+1s)
- GRPO 省 Critic 计算 (-2s)
- GRPO 少更新几轮 (-2s)
- **总体更快** ✓

### 5.3 关键差异总结

| 步骤 | PPO | GRPO | 差异 |
|------|-----|------|------|
| **Rollout** | n=1-4 | n=4-8 | GRPO 生成更多 |
| **Compute Values** | ✓ | ❌ | GRPO 省略 |
| **Advantage** | GAE (复杂) | 组内归一化 (简单) | GRPO 更快 |
| **Update Critic** | ✓ | ❌ | GRPO 省略 |
| **Update Actor** | 3-5 轮 | 1 轮 | GRPO 更少轮数 |
| **KL 约束** | Reward 中 | Loss 中 | GRPO 直接在 loss |

---

## 6. 关键代码解析

### 6.1 KL Loss 的应用

GRPO 通常在 **Loss** 中加 KL，而不是在 Reward 中：

**位置**: `verl/workers/roles/utils/losses.py:ppo_loss`

```python
def ppo_loss(config, model_output, data, dp_group=None):
    """
    计算 GRPO 的 loss

    Loss = PG Loss + KL Loss (+ Entropy Loss)
    """
    # 1. 基础数据
    log_prob = model_output["log_probs"]
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    response_mask = data["response_mask"]

    # 2. 计算 PG loss (和 PPO 一样)
    ratio = torch.exp(log_prob - old_log_prob)
    clip_range = config.clip_ratio

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1-clip_range, 1+clip_range)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    pg_loss = agg_loss(pg_losses, response_mask, config.loss_agg_mode)
    policy_loss = pg_loss

    # 3. [GRPO 特色] 添加 KL loss
    if config.use_kl_loss:  # GRPO 通常设为 True
        ref_log_prob = data["ref_log_prob"]

        # 计算 KL 散度
        kld = kl_penalty(
            logprob=log_prob,
            ref_logprob=ref_log_prob,
            kl_penalty=config.kl_loss_type  # 通常 "low_var_kl"
        )

        kl_loss = agg_loss(kld, response_mask, config.loss_agg_mode)

        # 加到总 loss
        policy_loss += kl_loss * config.kl_loss_coef

        metrics["kl_loss"] = kl_loss.detach().item()
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics
```

**为什么 GRPO 在 Loss 中加 KL？**

```
PPO 的方式 (KL in Reward):
  reward = original_reward - β * KL
  ↓
  advantage = (reward - mean) / std
  ↓
  只影响 advantage，不直接约束策略

GRPO 的方式 (KL in Loss):
  loss = PG_loss + β * KL_loss
  ↓
  直接约束策略不要偏离参考策略
  ↓
  更直接、更强的约束 ✓
```

**low_var_kl 的选择**：

GRPO 论文推荐使用 `low_var_kl`（也叫 k3）：

```python
def kl_penalty(logprob, ref_logprob, kl_penalty="low_var_kl"):
    if kl_penalty == "kl":  # k1, 标准 KL
        kl = torch.exp(logprob) * (logprob - ref_logprob)

    elif kl_penalty == "mse":  # k2, 均方误差
        kl = 0.5 * (logprob - ref_logprob) ** 2

    elif kl_penalty == "low_var_kl":  # k3, 低方差 KL
        # 结合 k1 和 k2 的优点
        kl = torch.exp(ref_logprob) * ((logprob - ref_logprob) ** 2)

    return kl
```

**为什么 low_var_kl 更好？**

详见这个分析：http://joschu.net/blog/kl-approx.html

简单来说：
- k1 (标准 KL)：无偏但高方差
- k2 (MSE)：低方差但有偏
- k3 (low_var_kl)：**平衡两者** ✓

### 6.2 DrGRPO: 解决长度偏差

**问题**：标准 GRPO 存在"长度偏差"

```
问题: "计算 12 × 15 = ?"

生成 3 个回答:
  回答1: "180" (短，正确) → 得分 1.0, 长度 3
  回答2: "我们可以这样计算：12 × 15 = 12 × (10 + 5) = 120 + 60 = 180" (长，正确)
         → 得分 1.0, 长度 20
  回答3: "160" (短，错误) → 得分 0.0, 长度 3

标准 GRPO (seq-mean-token-mean):
  每个序列的 loss 权重相同
  回答2 虽然长，但只算 1 个序列
  → 长回答的"每个 token"贡献较小

结果: 模型学会"写长回答"来减小 loss
```

**DrGRPO 的解决方案**：

```yaml
# 配置
actor_rollout_ref.actor.loss_agg_mode: "seq-mean-token-sum-norm"
algorithm.norm_adv_by_std_in_grpo: False
```

**loss_agg_mode 对比**：

```python
# 标准 GRPO: "seq-mean-token-mean"
seq_losses = sum(token_losses * mask) / sum(mask)  # token-mean
loss = mean(seq_losses)  # seq-mean
→ 长短序列权重相同

# DrGRPO: "seq-mean-token-sum-norm"
seq_losses = sum(token_losses * mask)  # token-sum (不除长度)
loss = mean(seq_losses) / global_constant  # 全局归一化
→ 长序列权重更大 ✓
```

**norm_adv_by_std_in_grpo: False**：

```python
# 标准 GRPO
advantage = (score - mean) / std

问题: 如果长回答得分方差大，std 大
     → advantage 被"压缩"
     → 梯度小，不鼓励长回答

# DrGRPO
advantage = score - mean  # 不除以 std

→ 保持原始差异 ✓
```

### 6.3 UID 的生成和使用

UID (Unique ID) 用于标识哪些回答属于同一组。

**生成 UID**：

```python
# 在 RayPPOTrainer.fit() 中
import uuid

# 1. 为每个 prompt 生成唯一 ID
batch.non_tensor_batch["uid"] = np.array(
    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
    dtype=object
)
# 例如: ["uuid_0", "uuid_1", "uuid_2", ...]

# 2. Repeat 以匹配 rollout 次数
batch = batch.repeat(
    repeat_times=self.config.actor_rollout_ref.rollout.n,
    interleave=True
)
# 例如 n=4:
# ["uuid_0", "uuid_0", "uuid_0", "uuid_0",
#  "uuid_1", "uuid_1", "uuid_1", "uuid_1",
#  ...]
```

**使用 UID 计算 Advantage**：

```python
def compute_advantage(data, adv_estimator, ...):
    # 提取 UID
    uid = data.non_tensor_batch["uid"]

    # 转换为数字索引
    unique_uids = np.unique(uid)
    uid_to_index = {u: i for i, u in enumerate(unique_uids)}
    index = np.array([uid_to_index[u] for u in uid])
    # 例如: ["uuid_0", "uuid_0", "uuid_1", "uuid_1"]
    #    → [0, 0, 1, 1]

    # 传递给 GRPO advantage 函数
    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=data.batch["response_mask"],
        index=index,  # ← 用于分组
        ...
    )

    return advantages, returns
```

**为什么需要 UID？**

如果直接用 batch 索引：
```python
# 错误做法
index = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

# 问题: 如果 batch 被打乱（如序列长度均衡）
# 打乱后: [2, 0, 1, 2, 0, 1, 0, 2, 1, 0, 2, 1]
# index 不再对应正确的组！

# 正确做法: 使用 UID
uid = ["uuid_0", "uuid_0", "uuid_0", "uuid_0",
       "uuid_1", "uuid_1", "uuid_1", "uuid_1",
       "uuid_2", "uuid_2", "uuid_2", "uuid_2"]

# 打乱后 UID 仍然跟随数据
# ["uuid_2", "uuid_0", "uuid_1", "uuid_2", ...]
# 转换为 index 仍然正确！
```

---

## 7. 配置与使用

### 7.1 核心配置项

**完整 GRPO 配置示例**:

```yaml
# GRPO 配置要点

# === 数据配置 ===
data:
  train_batch_size: 1024       # prompt 数量
  max_prompt_length: 512
  max_response_length: 1024

# === 算法配置 ===
algorithm:
  # [关键] 使用 GRPO advantage 估计器
  adv_estimator: grpo          # gae → grpo

  # [可选] 不归一化标准差 (DrGRPO)
  norm_adv_by_std_in_grpo: true  # true: 标准 GRPO, false: DrGRPO

  # [关键] 不在 reward 中使用 KL
  use_kl_in_reward: false      # GRPO 在 loss 中用 KL

# === Actor 配置 ===
actor_rollout_ref:
  rollout:
    # [关键] 组采样：每个 prompt 生成多个回答
    n: 5                       # 通常 4-8

  actor:
    # PPO 参数（仍然适用）
    ppo_epochs: 1              # [关键] GRPO 通常只 1 轮
    ppo_mini_batch_size: 256
    ppo_micro_batch_size_per_gpu: 32

    # 裁剪参数
    clip_ratio: 0.2

    # [关键] KL Loss (在 loss 中，不在 reward 中)
    use_kl_loss: true          # GRPO 必须 true
    kl_loss_coef: 0.001        # KL 系数
    kl_loss_type: low_var_kl   # k3, 推荐

    # Entropy（可选）
    entropy_coeff: 0           # 通常设为 0

    # [可选] Loss 聚合方式
    loss_agg_mode: "token-mean"  # 标准 GRPO
    # loss_agg_mode: "seq-mean-token-sum-norm"  # DrGRPO

  # 参考策略（用于 KL loss）
  ref:
    # GRPO 通常需要 ref policy
    # 可以 offload 到 CPU 省显存
    fsdp_config:
      param_offload: true

# === Critic 配置 ===
critic:
  # [关键] GRPO 不需要 Critic
  # verl 会自动根据 adv_estimator 判断

# === 训练器配置 ===
trainer:
  total_epochs: 15
  save_freq: 20
  test_freq: 5
  critic_warmup: 0             # GRPO 不需要

  # 硬件
  nnodes: 1
  n_gpus_per_node: 8
```

### 7.2 从 PPO 迁移到 GRPO

如果你已经有 PPO 配置，迁移到 GRPO 很简单：

```yaml
# PPO → GRPO 迁移清单

# 1. 修改 advantage 估计器
algorithm.adv_estimator: gae → grpo  ✓

# 2. 增加组采样
actor_rollout_ref.rollout.n: 1 → 5   ✓

# 3. 启用 KL loss
actor_rollout_ref.actor.use_kl_loss: false → true  ✓
actor_rollout_ref.actor.kl_loss_coef: 0.001        ✓
actor_rollout_ref.actor.kl_loss_type: low_var_kl   ✓

# 4. 关闭 reward 中的 KL
algorithm.use_kl_in_reward: true → false  ✓

# 5. 减少更新轮数（可选）
actor_rollout_ref.actor.ppo_epochs: 4 → 1  ✓

# 6. 完成！Critic 会自动禁用
```

### 7.3 启动训练

**单节点多卡 (8xGPU)**:

```bash
cd /path/to/verl

# 启动 Ray
ray start --head --port=6379

# 运行 GRPO 训练
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.use_kl_in_reward=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=15
```

**使用现成脚本**:

```bash
# verl 提供了很多 GRPO 示例脚本
bash examples/grpo_trainer/run_qwen3-8b.sh
```

### 7.4 监控指标

GRPO 的关键监控指标：

```python
# Advantage 相关
"adv/mean"                   # 应该接近 0
"adv/std"                    # 组内差异
"adv/max"                    # 最大 advantage
"adv/min"                    # 最小 advantage

# Actor 相关
"actor/pg_loss"              # Policy gradient loss
"actor/clipfrac"             # 裁剪比例
"actor/kl_loss"              # [GRPO 特色] KL loss
"actor/kl_coef"              # KL 系数

# Reward 相关
"reward/mean"                # 平均奖励
"reward/std"                 # 奖励标准差
"reward/group_std_mean"      # [GRPO 特色] 组内标准差均值

# 验证相关
"val-core/acc"               # 验证准确率
```

**健康的 GRPO 训练**：

```yaml
reward/mean: 递增             # 奖励在提升
reward/group_std_mean: > 0.1  # 组内有足够差异
actor/kl_loss: < 0.1         # KL 不要太大
actor/clipfrac: 0.1 - 0.3    # 适度裁剪
adv/mean: ~0.0               # Advantage 均值接近 0
```

**异常信号**：

```yaml
reward/group_std_mean: ~0    # 组内无差异，学不到东西！
                             # → 增加 rollout.n 或检查数据

actor/kl_loss: > 0.5         # KL 太大，策略偏离太多
                             # → 增加 kl_loss_coef

reward/mean: 不增反降         # 训练崩溃
                             # → 降低学习率或检查配置
```

### 7.5 显存和速度优化

**显存优化**：

```yaml
# 1. RefPolicy offload 到 CPU
actor_rollout_ref.ref.fsdp_config.param_offload: true

# 2. 使用 gradient checkpointing
actor_rollout_ref.model.enable_gradient_checkpointing: true

# 3. 使用 LoRA（参数高效微调）
actor_rollout_ref.actor.ppo_lora_rank: 16  # 启用 LoRA

# 4. 减少 rollout.n
actor_rollout_ref.rollout.n: 5 → 3  # 牺牲一些效果换显存
```

**速度优化**：

```yaml
# 1. 使用向量化 GRPO
algorithm.adv_estimator: grpo → grpo_vectorized  # 快 10-100x

# 2. 使用 vLLM/SGLang rollout
actor_rollout_ref.rollout.name: vllm  # 生成加速

# 3. 启用 remove_padding
actor_rollout_ref.model.use_remove_padding: true

# 4. 调整 tensor parallel
actor_rollout_ref.rollout.tensor_model_parallel_size: 2  # 推理并行
```

**不同配置的显存占用** (7B 模型):

| 配置 | Actor | Ref | Critic | 总显存 |
|------|-------|-----|--------|--------|
| PPO | 14GB | 7GB | 14GB | **35GB** |
| GRPO (Ref on GPU) | 14GB | 7GB | - | **21GB** ✓ |
| GRPO (Ref offload) | 14GB | 2GB | - | **16GB** ✓✓ |
| GRPO + LoRA | 4GB | 2GB | - | **6GB** ✓✓✓ |

---

## 8. 总结

### 8.1 GRPO 的核心优势

1. **省显存**: 不需要 Critic 网络，省 ~40% 显存
2. **训练快**: 不更新 Critic，省 ~30% 时间
3. **效果好**: 在数学推理任务上接近甚至超过 PPO
4. **易实现**: 算法简单，代码少
5. **易调参**: 超参数少，不需要调 Critic 相关参数

### 8.2 GRPO vs PPO vs DAPO

| 维度 | PPO | GRPO | DAPO |
|------|-----|------|------|
| **Critic** | 需要 | 不需要 ✓ | 不需要 ✓ |
| **组采样** | 可选 | 必须 | 必须 |
| **样本过滤** | 无 | 无 | 有 (动态采样) |
| **裁剪** | 对称 | 对称 | 非对称 |
| **显存** | 高 | 中 ✓ | 中 ✓ |
| **速度** | 慢 | 快 ✓ | 中 |
| **适用** | 通用 | 推理任务 | 高难度推理 |

**选择建议**：
- **有充足资源**: PPO（最稳定）
- **显存受限**: GRPO（省显存）
- **数学推理**: GRPO 或 DAPO（效果好）
- **对话任务**: PPO（GRPO 可能不够稳定）

### 8.3 GRPO 的局限性

1. **必须组采样**: 每个 prompt 生成多个回答，增加生成成本
2. **组内依赖**: 如果组内所有回答质量相近，advantage 接近 0，学不到东西
3. **不适合某些任务**: 对话任务中，同一个问题的多个回答可能都"合理"
4. **样本效率**: 不能多轮复用（通常），样本效率低于 PPO

### 8.4 GRPO 的变体

| 变体 | 特点 | 适用场景 |
|------|------|---------|
| **标准 GRPO** | 归一化标准差 | 通用推理任务 |
| **DrGRPO** | 不归一化标准差 | 避免长度偏差 |
| **GRPO_VECTORIZED** | 向量化实现 | 大规模训练（快 100x） |
| **GRPO_PASSK** | 只奖励最好的 | Pass@k 评估 |

### 8.5 实现要点总结

对于 infra 初学者：

1. **GRPO = PPO - Critic + 组采样**
   - 不需要 Critic 网络
   - 必须每个 prompt 生成多个回答

2. **组内归一化是核心**
   - `advantage = (score - group_mean) / group_std`
   - 用组内平均替代 Critic 的估计

3. **KL 在 Loss 中，不在 Reward 中**
   - `use_kl_loss=True` + `use_kl_in_reward=False`
   - 更直接地约束策略

4. **UID 用于分组**
   - 每个 prompt 一个唯一 ID
   - Repeat 时 UID 保持不变

5. **通常只更新 1 轮**
   - `ppo_epochs=1`
   - 省时间，效果也不差

### 8.6 进一步学习

想要深入理解，建议：

1. **源码**:
   - `verl/trainer/ppo/core_algos.py:compute_grpo_outcome_advantage` - GRPO 核心
   - `verl/trainer/ppo/ray_trainer.py:fit` - 训练流程
   - `examples/grpo_trainer/run_qwen3-8b.sh` - 示例脚本

2. **文档**:
   - `docs/algo/grpo.md` - GRPO 使用指南
   - `docs/algo/baseline.md` - 性能对比

3. **论文**:
   - [DeepSeekMath](https://arxiv.org/pdf/2402.03300) - GRPO 原论文
   - [DrGRPO](https://arxiv.org/abs/2503.20783) - 解决长度偏差
   - [Pass@k GRPO](https://arxiv.org/abs/2503.19595) - Pass@k 变体

---

## 附录

### A. 常见问题

**Q1: GRPO 一定比 PPO 快吗？**

通常是，但要看具体配置：
- GRPO 生成更多样本（n=5 vs n=1）→ 生成时间更长
- GRPO 不训练 Critic → 更新时间更短
- **总体**: 7B 模型上 GRPO 通常快 20-40%

**Q2: `rollout.n` 设多少合适？**

经验值：
- **小模型 (< 7B)**: n=4-5
- **大模型 (7B-30B)**: n=5-8
- **超大模型 (> 30B)**: n=3-4（显存限制）

太小学不到对比，太大浪费计算。

**Q3: 为什么我的组内标准差很小？**

可能原因：
1. **数据太简单**: 模型已经学会了，所有回答都对
   - 解决: 换更难的数据
2. **rollout.n 太小**: 只有 2-3 个回答，差异小
   - 解决: 增加 n 到 5-8
3. **temperature 太低**: 生成太确定性
   - 解决: 增加 temperature 到 0.8-1.0

**Q4: GRPO 能用于对话任务吗？**

可以，但效果可能不如 PPO：
- 对话任务中，同一个问题可能有多个"正确"答案
- 组内对比的意义不大
- PPO 的 Critic 能更好地估计"平均质量"

推荐：简单对话用 GRPO，复杂对话用 PPO

**Q5: DrGRPO 什么时候用？**

当你发现：
- 模型生成的回答越来越长
- 错误回答比正确回答更长
- 长度和准确率负相关

这时候用 DrGRPO 解决长度偏差。

**Q6: `kl_loss_type` 用哪个？**

推荐 `low_var_kl`:
- 比 `kl` 更稳定（低方差）
- 比 `mse` 更准确（低偏差）
- DeepSeek 论文也用这个

**Q7: 能不能不用 KL loss？**

可以，但不推荐：
- 没有 KL 约束，策略可能偏离太多
- 可能出现训练崩溃
- 至少设一个小的 `kl_loss_coef=0.0001`

### B. 术语对照表

| 英文 | 中文 | 解释 |
|------|------|------|
| Group | 组 | 同一个 prompt 的多个回答 |
| Outcome Supervision | 结果监督 | 只有最后结果有奖励 |
| Group Normalization | 组内归一化 | 在组内计算均值和标准差 |
| Critic-free | 无 Critic | 不需要价值网络 |
| DrGRPO | Dr.GRPO | 解决长度偏差的变体 |
| Pass@k | Pass@k | 生成 k 个回答，至少一个对 |
| UID | 唯一标识符 | 标记同一组的回答 |

### C. 配置速查表

**快速启动 GRPO**:

```yaml
# 最小配置
algorithm.adv_estimator: grpo
actor_rollout_ref.rollout.n: 5
actor_rollout_ref.actor.use_kl_loss: true
actor_rollout_ref.actor.kl_loss_coef: 0.001
algorithm.use_kl_in_reward: false
```

**省显存配置**:

```yaml
actor_rollout_ref.ref.fsdp_config.param_offload: true
actor_rollout_ref.model.enable_gradient_checkpointing: true
actor_rollout_ref.rollout.n: 3  # 减少采样
```

**快速训练配置**:

```yaml
algorithm.adv_estimator: grpo_vectorized  # 向量化
actor_rollout_ref.actor.ppo_epochs: 1
actor_rollout_ref.rollout.name: vllm
```

**DrGRPO 配置**:

```yaml
actor_rollout_ref.actor.loss_agg_mode: seq-mean-token-sum-norm
algorithm.norm_adv_by_std_in_grpo: false
actor_rollout_ref.actor.use_kl_loss: false
```

---

**文档版本**: v1.0
**贡献者**: Claude (AI Assistant)
**反馈**: 欢迎在 GitHub Issues 中提出问题和建议
