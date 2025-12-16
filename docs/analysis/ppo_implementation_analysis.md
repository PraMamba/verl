# PPO 训练框架实现详解

**面向对象**: 基础设施（Infrastructure）初学者
**最后更新**: 2025-11-27

---

## 目录

1. [什么是 PPO？](#1-什么是-ppo)
2. [PPO 的核心思想](#2-ppo-的核心思想)
3. [整体架构设计](#3-整体架构设计)
4. [核心组件实现](#4-核心组件实现)
5. [训练流程详解](#5-训练流程详解)
6. [关键代码解析](#6-关键代码解析)
7. [配置与使用](#7-配置与使用)
8. [总结](#8-总结)

---

## 1. 什么是 PPO？

### 1.1 PPO 简介

**PPO** (Proximal Policy Optimization，近端策略优化) 是 OpenAI 在 2017 年提出的一种强化学习算法，目前是 RLHF (人类反馈强化学习) 中**最主流**的算法之一。

**为什么叫"近端"（Proximal）？**

想象你在爬山（优化策略）：
- **太激进**：一步迈太大，可能摔下悬崖（策略崩溃）
- **太保守**：每步都很小，爬得太慢（训练效率低）
- **PPO 的做法**：每次更新都保持在"近端"（附近），既稳又快

### 1.2 PPO 解决的问题

传统策略梯度算法（如 REINFORCE）存在的问题：

| 问题 | 表现 | PPO 的解决方案 |
|------|------|---------------|
| **高方差** | 训练不稳定，抖动大 | 使用 GAE 降低方差 |
| **步长难调** | 学习率太大崩溃，太小训练慢 | 自适应裁剪，自动控制步长 |
| **样本效率低** | 每个样本只能用一次 | 可以重复使用样本（多个 epochs） |
| **难以收敛** | 容易陷入局部最优 | 平滑的目标函数 |

### 1.3 与其他算法的对比

| 算法 | Critic 网络 | Advantage 计算 | 样本复用 | 适用场景 |
|------|------------|---------------|---------|----------|
| **REINFORCE** | 不需要 | 简单差分 | 1次 | 简单任务 |
| **A2C/A3C** | 需要 | TD-error | 1次 | 中等任务 |
| **PPO** | 需要 | GAE | 多次 | 复杂任务，大模型 |
| **GRPO** | 不需要 | 组内归一化 | 1次 | 数学推理 |
| **DAPO** | 可选 | GRPO + 动态过滤 | 1次 | 高难度推理 |

**关键区别**：
- PPO 是最"经典"的 Actor-Critic 方法
- 需要额外训练一个 Critic 网络来估计 Value
- 可以多次重复使用同一批样本（提高样本效率）

---

## 2. PPO 的核心思想

### 2.1 裁剪策略比率（Clipped Surrogate Objective）

这是 PPO 最核心的创新！

**什么是策略比率？**

```python
# 策略比率
ratio = π_new(a|s) / π_old(a|s)

# 用 log 概率表示（数值更稳定）
ratio = exp(log_prob_new - log_prob_old)
```

**含义**：
- `ratio = 1.0`: 新旧策略对这个动作的偏好程度一样
- `ratio > 1.0`: 新策略更喜欢这个动作（概率增大）
- `ratio < 1.0`: 新策略不太喜欢这个动作（概率减小）

**PPO 的裁剪策略**：

```python
# 传统 Policy Gradient
loss = -advantage * ratio

# PPO 裁剪
clip_range = 0.2  # ε，通常设为 0.2
ratio_clipped = clamp(ratio, 1-clip_range, 1+clip_range)  # [0.8, 1.2]
loss_clipped = -advantage * ratio_clipped

# 取两者中更"悲观"的（更大的loss）
loss = max(loss, loss_clipped)
```

**可视化理解**：

```
Advantage > 0 (好的行为):
  ratio = 0.5  → clipped to 0.8  → 不会过度削弱
  ratio = 1.5  → clipped to 1.2  → 不会过度增强 ✓
  ratio = 1.1  → no clip       → 正常更新

Advantage < 0 (坏的行为):
  ratio = 0.5  → no clip       → 正常惩罚
  ratio = 1.5  → clipped to 1.2  → 不会过度惩罚 ✓
  ratio = 0.9  → no clip       → 正常更新
```

**直观例子**：

```
假设模型在回答问题:
  旧策略: "2+2=4" 的概率是 0.5
  新策略: "2+2=4" 的概率是 0.9

  ratio = 0.9 / 0.5 = 1.8
  clipped_ratio = min(1.8, 1.2) = 1.2

  如果 advantage > 0 (答对了):
    loss = -advantage * 1.2  (限制增强幅度)

  作用: 防止一次更新就把概率从 0.5 拉到 0.9，
       而是逐步提升，更稳定
```

### 2.2 广义优势估计（GAE）

**什么是 Advantage？**

Advantage 告诉我们：**这个动作比平均水平好多少**

```python
Advantage = Q(s,a) - V(s)
          = "做这个动作的价值" - "平均价值"
```

**为什么需要 GAE？**

直接计算 Advantage 有两种方法，但都有问题：

1. **蒙特卡洛方法**（MC）：
   ```python
   A = G_t - V(s)  # G_t 是实际累积回报
   ```
   - 优点：无偏估计（准确）
   - 缺点：高方差（不稳定）

2. **时间差分方法**（TD）：
   ```python
   A = r + γ*V(s') - V(s)  # 只看下一步
   ```
   - 优点：低方差（稳定）
   - 缺点：有偏估计（不够准确）

**GAE 的解决方案**：**两者加权平均**

```python
# GAE 是多步 TD 的加权组合
A_t^GAE = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...

其中:
  δ_t = r_t + γ*V_{t+1} - V_t  # 单步 TD-error
  λ ∈ [0, 1]  # 权衡参数
```

**λ 的作用**：
- `λ = 0`: 纯 TD，低方差但有偏
- `λ = 1`: 纯 MC，无偏但高方差
- `λ = 0.95`: **常用值**，平衡两者

**直观理解**：

```
假设玩游戏，要评估"向右走"这个动作:

单步 TD (λ=0):
  只看一步 → "向右走后立即得 +1 分" → 短视

蒙特卡洛 (λ=1):
  看到游戏结束 → "这局最后赢了 +100 分" → 但中间很多随机因素

GAE (λ=0.95):
  主要看近期几步，远期指数衰减
  → 平衡短期反馈和长期回报
```

### 2.3 Actor-Critic 架构

PPO 使用 **Actor-Critic** 架构，需要两个网络：

```
┌─────────────────────────────────────────┐
│          Environment                     │
│  (状态 s, 奖励 r)                        │
└─────────────────────────────────────────┘
         ↓                ↑
    状态 s              动作 a
         ↓                ↑
┌─────────────┐    ┌─────────────┐
│   Critic    │    │    Actor    │
│   (价值网络) │    │   (策略网络) │
│             │    │             │
│  输入: s     │    │  输入: s     │
│  输出: V(s)  │    │  输出: π(a|s)│
└─────────────┘    └─────────────┘
      ↓                  ↓
   Value            Log Prob
      ↓                  ↓
      └─────→ GAE ←──────┘
                ↓
            Advantage
                ↓
         Update Both
```

**两个网络的分工**：

| 网络 | 作用 | 输出 | 训练目标 |
|------|------|------|---------|
| **Actor** | 决策（做什么） | 动作概率分布 | 最大化 Advantage |
| **Critic** | 评估（有多好） | 状态价值 V(s) | 准确预测回报 |

**为什么需要 Critic？**

没有 Critic，我们不知道 Advantage：
```python
# 没有 Critic
A = G - ？？？  # 不知道平均有多好

# 有了 Critic
A = G - V(s)  # V(s) 告诉我们平均值
```

### 2.4 多轮更新（Multiple Epochs）

PPO 的一个重要特性：**同一批数据可以用多次**

```python
# 传统 Policy Gradient: 一批数据只用一次
for batch in dataloader:
    update_policy(batch)  # 用完就扔

# PPO: 一批数据用多次
for batch in dataloader:
    for epoch in range(ppo_epochs):  # 通常 3-5 轮
        for mini_batch in split(batch):
            update_policy(mini_batch)
            update_critic(mini_batch)
```

**为什么可以复用？**

因为有**裁剪**保护：
- 如果某个 mini-batch 更新太激进（ratio 太大）
- 裁剪会限制更新幅度
- 所以可以安全地多次更新

**提高样本效率**：
```
假设生成 1000 个样本:
  传统 PG: 更新 1 次  → 样本效率 1x
  PPO:      更新 5 次  → 样本效率 5x ✓
```

但不能无限复用，因为：
- 太多次后，新旧策略差距太大
- 裁剪失效，训练不稳定

---

## 3. 整体架构设计

### 3.1 系统架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                    Ray Cluster (分布式环境)                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │              TaskRunner (主控节点)                          │   │
│  │  - 初始化配置                                               │   │
│  │  - 创建 Worker Groups                                      │   │
│  │  - 创建 RayPPOTrainer                                       │   │
│  └───────────────────────────────────────────────────────────┘   │
│                           │                                       │
│                           ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │           RayPPOTrainer (训练协调器)                        │   │
│  │                                                             │   │
│  │  核心方法:                                                  │   │
│  │  • fit()                    - 主训练循环                    │   │
│  │  • _validate()              - 验证                         │   │
│  │  • _save_checkpoint()       - 保存检查点                   │   │
│  │  • _load_checkpoint()       - 加载检查点                   │   │
│  │  • _balance_batch()         - 序列长度均衡                 │   │
│  └───────────────────────────────────────────────────────────┘   │
│      │              │                │              │             │
│      ▼              ▼                ▼              ▼             │
│  ┌───────────┐ ┌───────────┐ ┌──────────────┐ ┌───────────┐    │
│  │  Actor    │ │  Rollout  │ │  RefPolicy   │ │  Critic   │    │
│  │WorkerGroup│ │WorkerGroup│ │ WorkerGroup  │ │WorkerGroup│    │
│  │           │ │           │ │              │ │           │    │
│  │• 更新策略  │ │• 生成回答  │ │• 计算参考    │ │• 估计Value│    │
│  │• 计算logp │ │• 记录logp │ │  log_prob    │ │• 更新Critic│    │
│  └───────────┘ └───────────┘ └──────────────┘ └───────────┘    │
│                                                                    │
│  ┌──────────────────────────┐         ┌──────────────────────┐   │
│  │  Reward Model (可选)      │         │  Reward Function     │   │
│  │  - 神经网络评分           │         │  - 规则评分           │   │
│  └──────────────────────────┘         └──────────────────────┘   │
│                                                                    │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │              Driver Process (Controller)                   │   │
│  │                                                             │   │
│  │  • compute_advantage()  - 计算 GAE Advantage               │   │
│  │  • apply_kl_penalty()   - 应用 KL 惩罚                     │   │
│  │  • _balance_batch()     - 负载均衡                         │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 数据流

```
输入数据 (Prompts)
    │
    ▼
┌─────────────────┐
│  1. Rollout     │  生成 N 个候选回答
│  (生成阶段)      │  → responses, rollout_log_probs
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 2. Compute      │  重新计算当前策略的 log_probs
│    Old LogProb  │  → old_log_probs (用于 PPO 裁剪)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 3. Compute      │  如果使用 KL 约束
│    Ref LogProb  │  → ref_log_probs
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 4. Reward       │  评估每个回答的质量
│  (奖励计算)      │  → token_level_rewards
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 5. Compute      │  Critic 估计每个 token 的价值
│    Values       │  → values (用于 GAE)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 6. GAE          │  [PPO 核心] 计算优势函数
│  (优势计算)      │  → advantages, returns
└─────────────────┘
    │
    ├──────────────────┬──────────────────┐
    ▼                  ▼                  ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Epoch 1     │  │ Epoch 2     │  │ Epoch N     │
│ Mini-batch  │  │ Mini-batch  │  │ Mini-batch  │
└─────────────┘  └─────────────┘  └─────────────┘
    │                  │                  │
    ▼                  ▼                  ▼
┌─────────────────────────────────────────┐
│  7. Update Actor & Critic               │
│  (策略更新 - 多轮)                       │
│                                          │
│  for epoch in range(ppo_epochs):        │
│    for mini_batch in split(batch):      │
│      • Update Critic (value loss)       │
│      • Update Actor (PPO loss)          │
└─────────────────────────────────────────┘
```

**关键特点**：

1. **两阶段**: 生成（Rollout）和更新（Update）分离
2. **多轮更新**: 同一批数据更新 3-5 轮
3. **Mini-batch**: 大 batch 拆成小 mini-batch 更新

### 3.3 Worker 角色分工

| Worker | 主要功能 | 输入 | 输出 | 是否必需 |
|--------|---------|------|------|---------|
| **ActorRollout** | 生成回答 + 更新策略 | prompts | responses, log_probs | ✓ 必需 |
| **Critic** | 估计价值 + 更新 Value | sequences | values | ✓ PPO 必需 |
| **RefPolicy** | 计算参考概率 | sequences | ref_log_probs | △ KL 约束时需要 |
| **RewardModel** | 神经网络打分 | sequences | rm_scores | △ 可选 |

**Hybrid Engine 模式**：

在 verl 中，**ActorRollout 通常是混合引擎**：
```python
ActorRollout = {
    "Rollout": vLLM/SGLang (高效推理),
    "Actor":   FSDP/Megatron (训练)
}
```

好处：
- Rollout 用推理优化的引擎（快）
- Actor 用训练优化的引擎（支持梯度）

---

## 4. 核心组件实现

### 4.1 RayPPOTrainer

**位置**: `verl/trainer/ppo/ray_trainer.py`

**核心职责**: 协调整个 PPO 训练流程

**关键属性**:

```python
class RayPPOTrainer:
    def __init__(
        self,
        config,                      # 配置
        tokenizer,                   # 分词器
        role_worker_mapping,         # 角色到 Worker 类的映射
        resource_pool_manager,       # 资源池管理器
        reward_fn=None,              # 奖励函数
        val_reward_fn=None,          # 验证奖励函数
        ...
    ):
        self.config = config
        self.tokenizer = tokenizer

        # 判断是否需要各个组件
        self.use_critic = need_critic(config)
        self.use_reference_policy = need_reference_policy(config)
        self.use_rm = need_reward_model(config)

        # KL 控制器 (如果启用 KL 惩罚)
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.make_kl_controller(
                config.algorithm.kl_ctrl
            )

        # Worker Groups (稍后初始化)
        self.actor_rollout_wg = None
        self.critic_wg = None
        self.ref_policy_wg = None
        self.rm_wg = None
```

**关键方法**: `fit()` - 主训练循环

```python
def fit(self):
    """PPO 的主训练循环"""
    # 1. 初始化
    self._load_checkpoint()  # 加载检查点（如果有）

    # 2. 训练前验证
    if self.val_reward_fn is not None:
        val_metrics = self._validate()
        print(f"Initial validation metrics: {val_metrics}")

    # 3. 主训练循环
    for epoch in range(total_epochs):
        for batch_dict in self.train_dataloader:
            # 3.1 构建 batch
            batch = DataProto.from_single_dict(batch_dict)

            # 3.2 生成回答
            gen_batch_output = self.actor_rollout_wg.generate_sequences(batch)
            batch = batch.union(gen_batch_output)

            # 3.3 重新计算 old_log_probs (用于 PPO 裁剪)
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            batch = batch.union(old_log_prob)

            # 3.4 计算参考 log_prob (如果使用 KL 约束)
            if self.use_reference_policy:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

            # 3.5 计算 values (Critic 估计)
            if self.use_critic:
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

            # 3.6 计算奖励
            if self.use_rm:
                rm_scores = self.rm_wg.compute_rm_score(batch)
                batch = batch.union(rm_scores)
            reward_tensor = compute_reward(batch, self.reward_fn)
            batch.batch["token_level_scores"] = reward_tensor

            # 3.7 应用 KL 惩罚 (如果启用)
            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(
                    batch, self.kl_ctrl_in_reward
                )

            # 3.8 计算 Advantage (GAE)
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
            )

            # 3.9 更新 Critic
            if self.use_critic:
                critic_output = self.critic_wg.update_critic(batch)

            # 3.10 更新 Actor
            actor_output = self.actor_rollout_wg.update_actor(batch)

            # 3.11 验证和保存
            if should_validate:
                val_metrics = self._validate()
            if should_save:
                self._save_checkpoint()
```

### 4.2 GAE Advantage 计算

**位置**: `verl/trainer/ppo/core_algos.py:compute_gae_advantage_return`

**算法实现**:

```python
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,  # [bs, seq_len]
    values: torch.Tensor,                # [bs, seq_len]
    response_mask: torch.Tensor,         # [bs, seq_len]
    gamma: float,                        # 折扣因子, 通常 1.0
    lam: float,                          # GAE λ, 通常 0.95
):
    """
    计算 GAE (Generalized Advantage Estimation)

    GAE 公式:
    A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...

    其中 δ_t = r_t + γ*V_{t+1} - V_t
    """
    with torch.no_grad():  # 不需要梯度
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        # 从后往前计算 (因为需要 V_{t+1})
        for t in reversed(range(gen_len)):
            # 1. 计算 TD-error: δ_t = r_t + γ*V_{t+1} - V_t
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]

            # 2. 计算 GAE: A_t = δ_t + (γλ)*A_{t+1}
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # 3. 处理 mask (对于 padding 的位置)
            # 如果当前位置是 padding (mask=0), 则保持之前的值
            nextvalues = values[:, t] * response_mask[:, t] + \
                        (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + \
                        (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)

        # 4. 反转列表 (因为是从后往前算的)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        # 5. 计算 returns: R_t = A_t + V_t
        returns = advantages + values

        # 6. 归一化 advantages (减均值除标准差)
        advantages = masked_whiten(advantages, response_mask)

    return advantages, returns
```

**详细例子**:

```python
假设一个简单的序列 (4 个 token):
  rewards = [0, 0, 0, 1]    # 只有最后一个 token 有奖励
  values  = [0.2, 0.4, 0.6, 0.8]  # Critic 的估计
  mask    = [1, 1, 1, 1]    # 都是有效 token
  γ = 1.0, λ = 0.95

从后往前计算:

t=3 (最后一个 token):
  δ_3 = r_3 + γ*V_4 - V_3
      = 1 + 1.0*0 - 0.8  # V_4=0 (序列结束)
      = 0.2
  A_3 = δ_3 + γλ*A_4
      = 0.2 + 1.0*0.95*0  # A_4=0 (序列结束)
      = 0.2

t=2:
  δ_2 = r_2 + γ*V_3 - V_2
      = 0 + 1.0*0.8 - 0.6
      = 0.2
  A_2 = δ_2 + γλ*A_3
      = 0.2 + 1.0*0.95*0.2
      = 0.2 + 0.19
      = 0.39

t=1:
  δ_1 = 0 + 1.0*0.6 - 0.4 = 0.2
  A_1 = 0.2 + 0.95*0.39 = 0.57

t=0:
  δ_0 = 0 + 1.0*0.4 - 0.2 = 0.2
  A_0 = 0.2 + 0.95*0.57 = 0.74

最终 advantages = [0.74, 0.57, 0.39, 0.2]
```

**为什么从后往前算？**
- 因为 `A_t` 需要 `A_{t+1}`
- 从后往前可以一次遍历完成

**为什么归一化？**
- 保证不同 batch 的 advantage 尺度一致
- 训练更稳定

### 4.3 Actor Worker

**位置**: `verl/workers/roles/actor.py`

**核心方法**: `update_actor`

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
def update_actor(self, data: DataProto):
    """
    更新 Actor (策略网络)

    执行多轮 mini-batch 更新
    """
    # 1. 配置 micro-batch 大小
    if self.config.use_dynamic_bsz:
        data.meta_info["max_token_len_per_gpu"] = self.config.ppo_max_token_len_per_gpu
    else:
        data.meta_info["micro_batch_size_per_gpu"] = self.config.ppo_micro_batch_size_per_gpu

    metrics_list = []

    # 2. 多轮更新 (ppo_epochs)
    for _ in range(self.config.ppo_epochs):
        # 3. 拆分成 mini-batches
        for mini_batch in self._split_mini_batches(data):
            # 4. 转换数据格式
            mini_batch = mini_batch.to_tensordict()
            mini_batch = left_right_2_no_padding(mini_batch)

            # 5. 前向传播 + 反向传播
            with self.engine.train_mode():
                output = self.engine.train_batch(
                    data=mini_batch,
                    loss_fn=self.loss_fn,  # ppo_loss
                )

            # 6. 收集 metrics
            if self.engine.is_mp_src_rank_with_outputs():
                metrics_list.append(output["metrics"])

    # 7. 聚合 metrics
    if self.engine.is_mp_src_rank_with_outputs():
        aggregated_metrics = aggregate_metrics(metrics_list)
        return DataProto(meta_info={"metrics": aggregated_metrics})

    return None
```

**PPO Loss 函数** (`verl/workers/roles/utils/losses.py:ppo_loss`):

```python
def ppo_loss(config, model_output, data, dp_group=None):
    """
    计算 PPO 损失

    Loss = Policy Gradient Loss + Entropy Loss + KL Loss
    """
    # 1. 提取数据
    log_prob = model_output["log_probs"]       # 新策略的 log_prob
    old_log_prob = data["old_log_probs"]       # 旧策略的 log_prob
    advantages = data["advantages"]            # GAE 计算的 advantage
    response_mask = data["response_mask"]      # 有效 token 的 mask

    # 2. 计算策略比率
    ratio = torch.exp(log_prob - old_log_prob)

    # 3. PPO 裁剪
    clip_range = config.clip_ratio  # 通常 0.2

    # 未裁剪的 loss
    pg_losses1 = -advantages * ratio

    # 裁剪后的 loss
    ratio_clipped = torch.clamp(
        ratio,
        1 - clip_range,  # 下界 0.8
        1 + clip_range   # 上界 1.2
    )
    pg_losses2 = -advantages * ratio_clipped

    # 取两者中较大的 (更悲观的估计)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # 4. 聚合 loss (跨所有 token)
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode  # "token-mean" 或 "seq-mean"
    )

    policy_loss = pg_loss
    metrics = {"actor/pg_loss": pg_loss.detach().item()}

    # 5. 添加 Entropy Loss (鼓励探索)
    if config.entropy_coeff > 0:
        entropy = model_output["entropy"]
        entropy_loss = agg_loss(entropy, response_mask, config.loss_agg_mode)
        policy_loss -= config.entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = entropy_loss.item()

    # 6. 添加 KL Loss (如果启用)
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        kld = kl_penalty(log_prob, ref_log_prob, config.kl_loss_type)
        kl_loss = agg_loss(kld, response_mask, config.loss_agg_mode)
        policy_loss += config.kl_loss_coef * kl_loss
        metrics["actor/kl_loss"] = kl_loss.item()

    # 7. 计算裁剪率 (监控训练稳定性)
    clipfrac = torch.mean(
        ((ratio < 1 - clip_range) | (ratio > 1 + clip_range)).float()
    )
    metrics["actor/clipfrac"] = clipfrac.item()

    return policy_loss, metrics
```

**关键点**:

1. **多轮更新**: 同一批数据更新 `ppo_epochs` 次（通常 3-5）
2. **Mini-batch**: 大 batch 拆成小的 mini-batch
3. **裁剪保护**: `ratio` 被限制在 `[1-ε, 1+ε]`
4. **三部分 Loss**: PG + Entropy + KL

### 4.4 Critic Worker

**位置**: `verl/workers/roles/critic.py`

**核心方法**: `update_critic`

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
def update_critic(self, data: DataProto):
    """
    更新 Critic (价值网络)
    """
    metrics_list = []

    # 1. 多轮更新
    for _ in range(self.config.ppo_epochs):
        # 2. Mini-batch 更新
        for mini_batch in self._split_mini_batches(data):
            mini_batch = mini_batch.to_tensordict()
            mini_batch = left_right_2_no_padding(mini_batch)

            # 3. 训练
            with self.engine.train_mode():
                output = self.engine.train_batch(
                    data=mini_batch,
                    loss_fn=self.loss_fn,  # value_loss
                )

            if self.engine.is_mp_src_rank_with_outputs():
                metrics_list.append(output["metrics"])

    # 4. 聚合 metrics
    if self.engine.is_mp_src_rank_with_outputs():
        aggregated_metrics = aggregate_metrics(metrics_list)
        return DataProto(meta_info={"metrics": aggregated_metrics})

    return None
```

**Value Loss 函数** (`verl/workers/roles/utils/losses.py:value_loss`):

```python
def value_loss(config, model_output, data, dp_group=None):
    """
    计算 Value 损失

    使用 Clipped Value Loss (类似 PPO 的裁剪思想)
    """
    # 1. 提取数据
    vpreds = model_output["values"]       # 新的 value 预测
    values = data["values"]               # 旧的 value (计算 GAE 时的)
    returns = data["returns"]             # GAE 计算的 returns
    response_mask = data["response_mask"]

    # 2. 计算未裁剪的 value loss
    vf_losses1 = (vpreds - returns) ** 2

    # 3. 裁剪后的 value loss
    cliprange_value = config.cliprange_value  # 通常 0.2
    vpredclipped = values + torch.clamp(
        vpreds - values,
        -cliprange_value,
        cliprange_value
    )
    vf_losses2 = (vpredclipped - returns) ** 2

    # 4. 取两者中较大的
    vf_losses = torch.maximum(vf_losses1, vf_losses2)

    # 5. 聚合
    vf_loss = agg_loss(
        loss_mat=vf_losses,
        loss_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode
    )

    # 6. 计算裁剪率
    vf_clipfrac = torch.mean(
        (vf_losses2 > vf_losses1).float()
    ).item()

    metrics = {
        "critic/vf_loss": vf_loss.item(),
        "critic/vf_clipfrac": vf_clipfrac,
        "critic/vpred_mean": masked_mean(vpreds, response_mask).item(),
    }

    return vf_loss, metrics
```

**为什么 Value 也要裁剪？**

- 防止 Value 估计变化太快
- 和 Policy 裁剪保持一致
- 训练更稳定

---

## 5. 训练流程详解

### 5.1 完整训练步骤

```
Step 0: 初始化
├─ 加载预训练模型
├─ 创建 Actor, Critic, RefPolicy Worker Groups
├─ 加载训练数据
└─ 设置配置参数

Step 1: 数据加载
├─ 从 dataloader 获取一个 batch
├─ Batch 包含: {prompts, ground_truth, ...}
└─ 例如: "What is 2+2?"

Step 2: Rollout (生成阶段)
├─ 使用 vLLM/SGLang 高效生成
├─ 对每个 prompt 生成 N 个回答
├─ 记录 rollout_log_probs
└─ 输出: {responses, rollout_log_probs}

Step 3: Recompute Old Log Probs
├─ 使用当前 Actor 重新计算 log_probs
├─ 这个会作为 PPO 的"旧策略"
├─ 为什么重新算？因为 Rollout 和 Actor 可能不同引擎
└─ 输出: {old_log_probs}

Step 4: Compute Ref Log Probs (可选)
├─ 如果使用 KL 约束
├─ 计算参考策略的 log_probs
└─ 输出: {ref_log_probs}

Step 5: Compute Values
├─ Critic 估计每个 token 的价值
├─ V(s_t) 表示"从这个状态开始的期望回报"
└─ 输出: {values}

Step 6: Compute Rewards
├─ 使用 Reward Model 或 Reward Function
├─ 评估生成的回答质量
└─ 输出: {token_level_rewards}

Step 7: Apply KL Penalty (可选)
├─ 如果启用 use_kl_in_reward
├─ rewards = rewards - β * KL(π_new || π_ref)
└─ 输出: {token_level_rewards (调整后)}

Step 8: Compute GAE Advantages
├─ [PPO 核心] 使用 GAE 算法
├─ 从后往前计算 advantage
├─ 归一化
└─ 输出: {advantages, returns}

Step 9: Multi-Epoch Updates
├─ for epoch in range(ppo_epochs):  # 通常 3-5
│   ├─ Split into mini-batches
│   │
│   ├─ Update Critic
│   │   ├─ 前向: 计算 new_values
│   │   ├─ 计算 value_loss (裁剪)
│   │   └─ 反向: 更新 Critic 参数
│   │
│   └─ Update Actor
│       ├─ 前向: 计算 new_log_probs
│       ├─ 计算 ratio = exp(new - old)
│       ├─ 裁剪 ratio
│       ├─ 计算 ppo_loss
│       └─ 反向: 更新 Actor 参数
│
└─ 输出: 更新后的 Actor 和 Critic

Step 10: Validation & Checkpointing
├─ 定期在验证集上测试
├─ 保存 checkpoint
└─ 记录 metrics

Step 11: 重复
└─ 回到 Step 1，直到达到总训练步数
```

### 5.2 时间线视图

```
时间轴 (以 7B 模型, batch_size=256 为例):
0s    2s        3s    4s      6s    8s       10s
│     │         │     │       │     │        │
├─────┤         │     │       │     │        │
│Gen  │         │     │       │     │        │
│     │         │     │       │     │        │
│     ├─────────┤     │       │     │        │
│     │Old LogP │     │       │     │        │
│     │         │     │       │     │        │
│     │         ├─────┤       │     │        │
│     │         │ Ref │       │     │        │
│     │         │LogP │       │     │        │
│     │         │     │       │     │        │
│     │         │     ├───────┤     │        │
│     │         │     │Reward │     │        │
│     │         │     │+Values│     │        │
│     │         │     │       │     │        │
│     │         │     │       ├─────┤        │
│     │         │     │       │ GAE │        │
│     │         │     │       │     │        │
│     │         │     │       │     ├────────┤
│     │         │     │       │     │ Update │
│     │         │     │       │     │Actor & │
│     │         │     │       │     │Critic  │
│     │         │     │       │     │(3-5轮) │
└─────┴─────────┴─────┴───────┴─────┴────────┘
```

**各阶段耗时** (7B 模型):
- Generation: ~2s (GPU 推理)
- Recompute LogProbs: ~1s
- Ref LogProbs: ~1s
- Reward + Values: ~2s (可能并行)
- GAE: ~0.5s (CPU 轻量计算)
- Update: ~2s (主要是 3-5 轮 mini-batch 更新)

**总时间**: ~8-10s/step

### 5.3 Mini-Batch 切分

PPO 使用 mini-batch 更新以提高效率：

```python
# 假设配置
train_batch_size = 256       # 总 batch size
rollout.n = 4                # 每个 prompt 生成 4 个回答
total_trajectories = 256 * 4 = 1024

ppo_mini_batch_size = 128    # mini-batch size
ppo_epochs = 4               # 更新 4 轮

# 更新过程
for epoch in range(4):  # 4 轮
    # 将 1024 个轨迹拆成 8 个 mini-batch (1024/128=8)
    for mini_batch in split(batch, mini_batch_size=128):
        update_critic(mini_batch)
        update_actor(mini_batch)

# 总更新次数
total_updates = 4 * 8 = 32 次
```

**为什么用 Mini-Batch？**

1. **内存限制**: 1024 个序列一次性训练可能 OOM
2. **梯度噪声**: 小 batch 有一定噪声，有助于泛化
3. **更新频率**: 更频繁的参数更新

### 5.4 张量形状变化

跟踪数据的形状：

```python
# 输入 batch
prompts: [batch_size, prompt_length]
# [256, 512]

# 生成阶段 (每个 prompt 生成 N 个)
responses: [batch_size * N, response_length]
# [256 * 4 = 1024, 1024]

rollout_log_probs: [1024, 1024]

# 重新计算 old_log_probs
old_log_probs: [1024, 1024]

# Critic 计算 values
values: [1024, 1024]

# Rewards
token_level_rewards: [1024, 1024]
# 通常只有最后一个 token 有非零 reward

# GAE 计算
advantages: [1024, 1024]
returns: [1024, 1024]

# Mini-batch (拆分)
mini_batch_size = 128
mini_batch.advantages: [128, 1024]
mini_batch.old_log_probs: [128, 1024]

# 更新时的 loss
pg_loss: scalar  # 聚合后的单个值
vf_loss: scalar
```

---

## 6. 关键代码解析

### 6.1 KL 惩罚的应用

**位置**: `verl/trainer/ppo/ray_trainer.py:apply_kl_penalty`

```python
def apply_kl_penalty(data, kl_ctrl, kl_penalty="kl"):
    """
    在奖励中应用 KL 惩罚

    reward = original_reward - β * KL(π_new || π_ref)

    目的: 防止策略偏离参考策略太远
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]

    # 1. 计算 KL 散度
    kld = kl_penalty(
        data.batch["old_log_probs"],
        data.batch["ref_log_prob"],
        kl_penalty="kl"  # 使用标准 KL
    )
    # kld: [bs, seq_len]

    # 2. 只计算有效 token 的 KL
    kld = kld * response_mask

    # 3. 获取 KL 系数 β
    beta = kl_ctrl.value  # 从控制器获取

    # 4. 调整奖励
    token_level_rewards = token_level_scores - beta * kld

    # 5. 计算平均 KL (用于监控)
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()

    # 6. 自适应调整 β (如果使用 adaptive controller)
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)

    # 7. 更新数据
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {
        "actor/reward_kl_penalty": current_kl,
        "actor/reward_kl_penalty_coeff": beta
    }

    return data, metrics
```

**KL 散度的计算** (`verl/trainer/ppo/core_algos.py:kl_penalty`):

```python
def kl_penalty(logprob, ref_logprob, kl_penalty="kl"):
    """
    计算 KL 散度: KL(π || π_ref)

    支持多种近似方法
    """
    if kl_penalty == "kl":  # 标准 KL
        # KL(p||q) = p * log(p/q) = p * (log(p) - log(q))
        # 在 log 空间: exp(logp) * (logp - logq)
        kl = torch.exp(logprob) * (logprob - ref_logprob)

    elif kl_penalty == "abs":  # 绝对差
        kl = torch.abs(torch.exp(logprob) - torch.exp(ref_logprob))

    elif kl_penalty == "mse":  # 均方误差
        kl = 0.5 * (logprob - ref_logprob) ** 2

    elif kl_penalty == "full":  # 完整 KL
        # KL(p||q) = sum_a p(a) * log(p(a)/q(a))
        # 需要所有动作的概率，计算量大
        ...

    return kl
```

**自适应 KL 控制器**:

```python
class AdaptiveKLController:
    """
    自适应调整 KL 系数 β

    目标: 保持 KL 在目标值 target_kl 附近
    """
    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef    # 当前 β
        self.target = target_kl      # 目标 KL
        self.horizon = horizon       # 调整周期

    def update(self, current_kl, n_steps):
        """根据当前 KL 调整 β"""
        proportional_error = np.clip(
            current_kl / self.target - 1,
            -0.2, 0.2
        )
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult
```

**为什么需要 KL 约束？**

想象训练一个聊天机器人：
```
初始模型 (π_ref): 有礼貌但回答简单
  "你好！" → 90%
  "嗨"    → 10%

训练后没有 KL 约束:
  "你好！" → 1%
  "嗨"    → 99%  ✗ 变得不礼貌了

训练后有 KL 约束:
  "你好！" → 70%  ✓ 保持礼貌
  "嗨，很高兴认识你" → 30%  ✓ 更有趣
```

### 6.2 Dual-Clip PPO

verl 还支持 **Dual-Clip PPO**，进一步优化裁剪策略：

```python
def dual_clip_ppo_loss(old_log_prob, log_prob, advantages, clip_ratio, clip_ratio_c):
    """
    Dual-Clip PPO

    标准 PPO 的问题:
    - advantage < 0 时，如果 ratio 很大，惩罚被限制

    Dual-Clip 的改进:
    - 当 advantage < 0 且 ratio > clip_ratio_c 时，额外裁剪
    """
    ratio = torch.exp(log_prob - old_log_prob)

    # 标准 PPO 裁剪
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Dual-Clip: 对 advantage < 0 的额外裁剪
    # 如果 ratio 太大 (比如 3.5)，进一步限制
    clip_ratio_c = 3.0  # 通常设为 3
    pg_losses3 = -advantages * torch.clamp(ratio, 1-clip_ratio, clip_ratio_c)

    # 选择: advantage < 0 时考虑第三项
    pg_losses = torch.where(
        advantages < 0,
        torch.maximum(pg_losses, pg_losses3),
        pg_losses
    )

    return pg_losses.mean()
```

**Dual-Clip 的作用**：

```
Scenario: 模型突然输出了一个非常坏的回答
  advantage = -0.8  (很差)
  ratio = 4.0       (新策略认为这个回答很好，旧策略不这么认为)

标准 PPO:
  clipped_ratio = clamp(4.0, 0.8, 1.2) = 1.2
  loss = -(-0.8) * 1.2 = 0.96

Dual-Clip PPO:
  clipped_ratio = clamp(4.0, 0.8, 3.0) = 3.0
  loss = -(-0.8) * 3.0 = 2.4  ✓ 更大的惩罚

作用: 对明显的坏行为给予更强的惩罚
```

### 6.3 序列长度均衡

**位置**: `verl/trainer/ppo/ray_trainer.py:_balance_batch`

```python
def _balance_batch(self, batch, metrics):
    """
    序列长度均衡

    问题: 不同 GPU 拿到的序列长度差异很大
    - GPU 0: 全是长序列 (2048 tokens/seq)
    - GPU 1: 全是短序列 (512 tokens/seq)

    结果: GPU 0 慢，GPU 1 快，总体效率低

    解决: 重新排列数据，让每个 GPU 的总 token 数相近
    """
    attention_mask = batch.batch["attention_mask"]
    batch_size = attention_mask.shape[0]

    # 1. 计算每个序列的长度
    seqlen_lst = attention_mask.sum(dim=-1)  # [batch_size]

    # 2. 计算"工作量" (考虑了计算复杂度)
    workload_lst = calculate_workload(seqlen_lst)

    # 3. 划分到各个 GPU (贪心算法)
    world_size = self.actor_rollout_wg.world_size
    partition_lst = get_seqlen_balanced_partitions(
        workload_lst,
        k_partitions=world_size,
        equal_size=True  # 每个 partition 样本数相同
    )

    # 4. 重新排列数据
    global_idx = torch.tensor([
        j for partition in partition_lst for j in partition
    ])
    batch.reorder(global_idx)

    # 5. 记录统计信息
    balance_stats = log_seqlen_unbalance(
        seqlen_list=seqlen_lst,
        partitions=partition_lst
    )
    metrics.update(balance_stats)
```

**示例**：

```
假设有 8 个序列，2 个 GPU:
  序列长度: [2048, 1024, 512, 2048, 1024, 512, 1024, 512]

不均衡分配:
  GPU 0: [2048, 1024, 512, 2048]  → 总 token 5632
  GPU 1: [1024, 512, 1024, 512]   → 总 token 3072
  不平衡度: 5632 / 3072 = 1.83x

均衡分配:
  GPU 0: [2048, 1024, 512]        → 总 token 3584
  GPU 1: [2048, 1024, 1024, 512]  → 总 token 4608
  不平衡度: 4608 / 3584 = 1.29x  ✓ 更好

最优分配:
  GPU 0: [2048, 1024, 512, 512]   → 总 token 4096
  GPU 1: [2048, 1024, 1024, 512]  → 总 token 4608
  不平衡度: 4608 / 4096 = 1.13x  ✓ 最佳
```

---

## 7. 配置与使用

### 7.1 核心配置项

**完整 PPO 配置示例**:

```yaml
# verl/trainer/config/ppo_trainer.yaml

# === 数据配置 ===
data:
  train_batch_size: 256        # prompt 数量
  max_prompt_length: 1024      # 最大 prompt 长度
  max_response_length: 1024    # 最大 response 长度

# === 算法配置 ===
algorithm:
  # Advantage 估计方法
  adv_estimator: gae           # gae / grpo / reinforce++

  # GAE 参数
  gamma: 1.0                   # 折扣因子
  lam: 0.95                    # GAE lambda

  # KL 约束 (在 reward 中)
  use_kl_in_reward: false      # 是否启用
  kl_penalty: kl               # kl / abs / mse
  kl_ctrl:
    type: fixed                # fixed / adaptive
    kl_coef: 0.001            # KL 系数
    target_kl: 0.1            # 目标 KL (adaptive 模式)
    horizon: 10000            # 调整周期

# === Actor 配置 ===
actor_rollout_ref:
  rollout:
    n: 4                       # 每个 prompt 生成几个回答
    temperature: 1.0
    top_k: 50
    top_p: 0.95

  actor:
    # PPO 参数
    ppo_epochs: 4              # 多轮更新
    ppo_mini_batch_size: 64    # mini-batch size
    ppo_micro_batch_size: 2    # micro-batch (控制显存)

    # 裁剪参数
    clip_ratio: 0.2            # PPO 裁剪范围 ε
    clip_ratio_c: 3.0          # Dual-Clip 的下界

    # Entropy 正则化
    entropy_coeff: 0.0         # Entropy 系数

    # KL Loss (在 loss 中，与 kl_in_reward 二选一)
    use_kl_loss: false
    kl_loss_coef: 0.001
    kl_loss_type: kl

    # Loss 聚合方式
    loss_agg_mode: "seq-mean-token-sum"  # token-mean / seq-mean-token-sum

# === Critic 配置 ===
critic:
  ppo_epochs: 4                # 与 actor 相同或独立设置
  ppo_mini_batch_size: 64
  ppo_micro_batch_size: 2
  cliprange_value: 0.2         # Value 裁剪范围
  loss_agg_mode: "seq-mean-token-sum"

# === 训练器配置 ===
trainer:
  total_epochs: 30
  total_training_steps: null   # 如果设置，会覆盖 epochs

  # 验证和保存
  test_freq: 100               # 每 100 步验证
  save_freq: 500               # 每 500 步保存
  val_before_train: true       # 训练前验证

  # Critic 预热
  critic_warmup: 0             # Critic 预热步数

  # 负载均衡
  balance_batch: true          # 启用序列长度均衡

  # 硬件配置
  nnodes: 2                    # 节点数
  n_gpus_per_node: 8           # 每节点 GPU 数

  # 日志
  logger: ["console", "wandb"]
  project_name: verl_ppo
  experiment_name: gsm8k
```

### 7.2 不同配置的影响

**1. `adv_estimator` 的选择**

```yaml
# GAE (标准 PPO)
adv_estimator: gae
gamma: 1.0
lam: 0.95
critic: {enable: true}  # 需要 Critic
→ 适合: 通用任务，样本效率要求高

# GRPO
adv_estimator: grpo
critic: {enable: false}  # 不需要 Critic
→ 适合: 数学推理，想省显存

# REINFORCE++
adv_estimator: reinforce_plus_plus
→ 适合: 简单任务，快速迭代
```

**2. `ppo_epochs` 的影响**

```yaml
# 少轮更新
ppo_epochs: 1
→ 训练快，但样本效率低
→ 需要更多数据

# 多轮更新
ppo_epochs: 5
→ 训练慢，但样本效率高
→ 可能过拟合单个 batch
```

**3. `clip_ratio` 的调整**

```yaml
# 保守
clip_ratio: 0.1
→ 更新步长小，稳定但慢

# 标准
clip_ratio: 0.2  # 推荐
→ 平衡

# 激进
clip_ratio: 0.3
→ 更新步长大，快但可能不稳定
```

**4. KL 约束的选择**

```yaml
# 方案 1: KL in Reward
algorithm:
  use_kl_in_reward: true
  kl_ctrl:
    type: adaptive
    kl_coef: 0.001
    target_kl: 0.1
actor:
  use_kl_loss: false
→ KL 影响 reward，间接影响训练

# 方案 2: KL Loss
algorithm:
  use_kl_in_reward: false
actor:
  use_kl_loss: true
  kl_loss_coef: 0.001
→ KL 直接加到 loss，更直接

# 方案 3: 不用 KL
→ 适合: SFT 模型质量高，不怕偏离
```

### 7.3 启动训练

**单节点多卡**:

```bash
cd /path/to/verl

# 启动 Ray
ray start --head --port=6379

# 运行 PPO 训练
python examples/ppo_trainer/run_ppo.py \
    data.train_batch_size=256 \
    actor_rollout_ref.actor.ppo_epochs=4 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    algorithm.adv_estimator=gae \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    trainer.logger=wandb
```

**多节点**:

```bash
# 节点 1 (head)
ray start --head --node-ip-address=192.168.1.10 --port=6379

# 节点 2 (worker)
ray start --address='192.168.1.10:6379'

# 从任意节点提交任务
python examples/ppo_trainer/run_ppo.py \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    data.train_batch_size=512  # 可以用更大的 batch
```

**使用 Slurm**:

```bash
#!/bin/bash
#SBATCH --job-name=ppo_train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00

# 启动 Ray 集群
srun --nodes=2 --ntasks=2 ray-start-cluster.sh

# 运行训练
python examples/ppo_trainer/run_ppo.py \
    trainer.nnodes=$SLURM_NNODES \
    trainer.n_gpus_per_node=$SLURM_GPUS_PER_NODE
```

### 7.4 监控指标

训练过程中的关键指标：

```python
# Actor 相关
"actor/pg_loss"              # Policy Gradient loss
"actor/clipfrac"             # 被裁剪的比例 (监控更新幅度)
"actor/entropy"              # 策略熵 (探索程度)
"actor/ratio_mean"           # 平均策略比率
"actor/ratio_std"            # 策略比率标准差
"actor/approx_kl"            # 近似 KL 散度

# Critic 相关
"critic/vf_loss"             # Value function loss
"critic/vf_clipfrac"         # Value 被裁剪的比例
"critic/vpred_mean"          # 平均 value 预测

# Reward 相关
"reward/mean"                # 平均奖励
"reward/std"                 # 奖励标准差
"reward/max"                 # 最大奖励
"reward/min"                 # 最小奖励

# Advantage 相关
"adv/mean"                   # 平均 advantage (应该接近 0)
"adv/std"                    # Advantage 标准差
"adv/max"                    # 最大 advantage
"adv/min"                    # 最小 advantage

# KL 相关 (如果启用)
"actor/kl_loss"              # KL loss
"actor/kl_coef"              # KL 系数
"actor/reward_kl_penalty"    # Reward 中的 KL 惩罚

# 性能相关
"timing/gen"                 # 生成耗时
"timing/update_actor"        # Actor 更新耗时
"timing/update_critic"       # Critic 更新耗时
"throughput/tokens_per_sec"  # 吞吐量

# 验证相关
"val-core/acc"               # 验证准确率
```

**健康的训练指标**：

```yaml
actor/clipfrac: 0.1 - 0.3    # 10-30% 被裁剪，说明步长合适
actor/entropy: > 2.0         # 足够的探索
actor/ratio_mean: 0.95-1.05  # 策略变化不大
actor/approx_kl: < 0.05      # KL 不要太大

critic/vf_loss: 递减         # Critic 在学习
adv/mean: ~0.0               # Advantage 均值接近 0

reward/mean: 递增             # 奖励在提升
```

**异常信号**：

```yaml
actor/clipfrac: > 0.5        # 裁剪太多，步长太大，降低 lr
actor/approx_kl: > 0.1       # KL 太大，策略变化太快
critic/vf_loss: 不降反升      # Critic 崩溃了
reward/mean: 震荡剧烈         # 不稳定，检查数据或降低 lr
```

---

## 8. 总结

### 8.1 PPO 的核心优势

1. **稳定性强**: 裁剪机制保证更新不会太激进
2. **样本效率高**: 可以多轮复用同一批数据
3. **易于调参**: 超参数对结果影响相对稳定
4. **通用性好**: 适用于各种 RL 任务

### 8.2 PPO vs 其他算法

| 对比维度 | PPO | GRPO | DAPO | REINFORCE |
|---------|-----|------|------|-----------|
| **Critic 网络** | 需要 | 不需要 | 可选 | 不需要 |
| **样本复用** | 多次 (3-5) | 1次 | 1次 | 1次 |
| **计算开销** | 中等 | 较低 | 中等 | 最低 |
| **样本效率** | 高 | 中等 | 中等 | 低 |
| **训练稳定性** | 高 | 中等 | 高 | 低 |
| **显存占用** | 高 (2个网络) | 低 | 中等 | 最低 |
| **适用场景** | 通用 | 推理任务 | 高难度推理 | 简单任务 |

### 8.3 何时选择 PPO？

**推荐使用 PPO**:
- 对话任务 (ChatGPT 风格)
- 需要高样本效率
- 有足够的计算资源 (能跑 2 个大模型)
- 任务复杂，需要稳定训练

**考虑其他算法**:
- 显存不足 → GRPO (不需要 Critic)
- 数学推理 → DAPO (动态采样)
- 简单任务 → REINFORCE (最简单)
- 需要极致速度 → GRPO (少一个网络)

### 8.4 verl PPO 实现的特点

1. **混合引擎**: Rollout 用 vLLM/SGLang，训练用 FSDP/Megatron
2. **灵活架构**: 支持 FSDP, Megatron, vLLM, SGLang 等多种后端
3. **高度优化**: 序列长度均衡、动态 batch size、profiling
4. **完整功能**: KL 约束、Dual-Clip、多种 Advantage 估计器
5. **分布式友好**: 基于 Ray，支持多节点多卡

### 8.5 实现要点总结

对于 infra 初学者：

1. **PPO = Actor-Critic + Clipping**
   - 需要两个网络: Actor (策略) + Critic (价值)
   - 核心是裁剪策略比率 `ratio`

2. **GAE 是降方差的关键**
   - 从后往前计算 advantage
   - `λ` 权衡偏差和方差

3. **Multi-epoch 提高样本效率**
   - 同一批数据更新 3-5 次
   - 裁剪保证安全性

4. **KL 约束防止崩溃**
   - 可以在 reward 里加
   - 也可以在 loss 里加

5. **Mini-batch 提高效率**
   - 大 batch 拆成小 mini-batch
   - 更频繁的参数更新

### 8.6 进一步学习

想要深入理解，建议：

1. **源码**:
   - `verl/trainer/ppo/ray_trainer.py` - 训练器
   - `verl/trainer/ppo/core_algos.py` - 核心算法
   - `verl/workers/roles/actor.py` - Actor Worker
   - `verl/workers/roles/critic.py` - Critic Worker

2. **文档**:
   - `docs/algo/ppo.md` - PPO 使用指南
   - `docs/workers/fsdp_workers.rst` - FSDP Worker 文档
   - `docs/advance/fully_async.md` - 异步训练

3. **论文和教程**:
   - [PPO 原论文](https://arxiv.org/abs/1707.06347)
   - [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
   - [GAE 论文](https://arxiv.org/abs/1506.02438)

---

## 附录

### A. 常见问题

**Q1: PPO 的 Critic 一定要和 Actor 一样大吗？**

不一定。Critic 可以：
- 和 Actor 一样大 (共享大部分参数，省显存)
- 比 Actor 小 (独立训练，省计算)
- 使用不同架构 (如 Critic 用 LSTM)

verl 默认使用相同大小，但可以配置。

**Q2: `ppo_epochs` 设多少合适？**

经验值:
- **小模型 (< 7B)**: 3-4 轮
- **大模型 (7B-70B)**: 4-5 轮
- **超大模型 (> 70B)**: 1-2 轮 (显存限制)

太多容易过拟合，太少样本效率低。

**Q3: GAE 的 `lam` 怎么调？**

- `lam = 0.95`: 默认值，通常不需要调
- `lam = 0.9`: 更重视近期回报，适合短任务
- `lam = 0.99`: 更重视长期回报，适合长任务
- `lam = 1.0`: 纯 MC，高方差但无偏

**Q4: KL 约束必须用吗？**

不一定：
- **SFT 质量高**: 可以不用，训练更快
- **怕模型崩溃**: 建议用，更稳定
- **对话任务**: 建议用，保持风格
- **数学推理**: 可以不用，让模型充分探索

**Q5: 为什么我的 `clipfrac` 很低？**

可能原因：
- 学习率太小 → 策略更新幅度小
- 数据太简单 → 策略已经很好了
- `clip_ratio` 太大 → 很少触发裁剪

不一定是坏事，看 reward 是否在增长。

**Q6: Critic 不收敛怎么办？**

尝试：
1. 增加 `critic_warmup` (先训练 Critic)
2. 降低 Critic 学习率
3. 增加 Critic 的 `ppo_epochs`
4. 检查 value 裁剪范围 `cliprange_value`

**Q7: 单卡能跑 PPO 吗？**

可以，但需要：
- 小模型 (如 Qwen2.5-0.5B)
- 减小 batch size
- 使用 LoRA (只训练少量参数)
- Critic 共享 Actor 参数

### B. 术语对照表

| 英文 | 中文 | 解释 |
|------|------|------|
| Actor | 执行者 / 策略网络 | 决策网络，输出动作 |
| Critic | 评论家 / 价值网络 | 评估网络，估计价值 |
| Advantage | 优势函数 | 当前动作比平均好多少 |
| GAE | 广义优势估计 | 降方差的 Advantage 计算方法 |
| Clipping | 裁剪 | 限制更新幅度 |
| Policy Ratio | 策略比率 | 新旧策略的概率比 |
| TD-error | 时间差分误差 | 单步预测误差 |
| Value Function | 价值函数 | 状态的期望回报 |
| Return | 回报 | 累积奖励 |
| Rollout | 推演 / 生成 | 使用策略生成轨迹 |

---

**文档版本**: v1.0
**贡献者**: Claude (AI Assistant)
**反馈**: 欢迎在 GitHub Issues 中提出问题和建议
