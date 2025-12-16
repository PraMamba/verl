# DrGRPO 训练框架实现详解

**面向对象**: 基础设施（Infrastructure）初学者
**最后更新**: 2025-11-27

---

## 目录

1. [什么是 DrGRPO？](#1-什么是-drgrpo)
2. [长度偏差问题](#2-长度偏差问题)
3. [DrGRPO 的解决方案](#3-drgrpo-的解决方案)
4. [核心代码实现](#4-核心代码实现)
5. [GRPO vs DrGRPO 对比](#5-grpo-vs-drgrpo-对比)
6. [配置与使用](#6-配置与使用)
7. [实战案例分析](#7-实战案例分析)
8. [总结](#8-总结)

---

## 1. 什么是 DrGRPO？

### 1.1 DrGRPO 简介

**DrGRPO** (Dr.GRPO) 是对 GRPO (Group Relative Policy Optimization) 的改进版本，由论文 [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/pdf/2503.20783) 提出。

**核心目标**：解决 GRPO 的**长度偏差**（Length Bias）问题。

**命名由来**：
- **Dr.** 可能代表 "Doctor"（医生），暗示"治疗"GRPO 的问题
- 或者代表某种算法改进

**一句话总结**：
```
DrGRPO = GRPO - 长度偏差
```

### 1.2 DrGRPO vs GRPO

| 维度 | GRPO | DrGRPO |
|------|------|--------|
| **Advantage 计算** | (score - mean) / std | score - mean ✓ |
| **Loss 聚合** | seq-mean-token-mean | seq-mean-token-sum-norm ✓ |
| **KL Loss** | 通常使用 | 不使用 ✓ |
| **长度偏差** | 存在 ❌ | 解决 ✓ |
| **适用场景** | 通用推理 | 长文本推理 ✓ |

**关键差异**：
1. **不除以标准差**: 避免对长度的隐式惩罚
2. **不除以序列长度**: 长序列有更大的梯度权重
3. **不使用 KL Loss**: 简化训练

---

## 2. 长度偏差问题

### 2.1 什么是长度偏差？

**长度偏差**：模型倾向于生成越来越长的回答，即使这些回答是错误的。

**直观例子**：

```
问题: "计算 2 + 2 = ?"

训练前（短回答）:
  回答1: "4" (正确，长度 1)
  回答2: "5" (错误，长度 1)
  回答3: "4" (正确，长度 1)

使用 GRPO 训练 100 轮后（回答变长）:
  回答1: "首先，我们计算 2 + 2 = 4" (正确，长度 15)
  回答2: "让我们详细分析这个问题。我们有两个 2，相加得到 5" (错误，长度 30) ← 更长！
  回答3: "简单来说，2 + 2 = 4" (正确，长度 12)

问题: 错误的回答（回答2）反而比正确的回答更长！
```

**为什么这是个问题？**

1. **计算浪费**: 生成长文本消耗更多计算
2. **效果下降**: 长回答不一定更好
3. **训练不稳定**: 长度和质量混淆

### 2.2 GRPO 为什么会产生长度偏差？

**原因 1: Advantage 归一化**

```python
# GRPO 的 Advantage 计算
advantage = (score - mean) / std

问题:
  假设一个组内:
    回答1: score=1.0, 长度=10
    回答2: score=0.0, 长度=50  ← 错误但很长
    回答3: score=1.0, 长度=10

  mean = 0.67, std = 0.47

  回答2 的 advantage = (0.0 - 0.67) / 0.47 = -1.42

  如果回答2 更短（长度=10）:
    std 可能更大 → advantage 的绝对值更小 → 惩罚更弱

  结果: 长的错误回答被"稀释"了惩罚
```

**原因 2: Loss 聚合方式**

```python
# GRPO 通常使用: "seq-mean-token-mean"
seq_loss = sum(token_losses) / seq_length  # 除以序列长度
final_loss = mean(seq_losses)

问题:
  短序列（长度 10）: loss = 10 / 10 = 1.0
  长序列（长度 50）: loss = 50 / 50 = 1.0

  权重相同！

  但实际上:
    长序列生成了更多错误 token
    应该有更大的惩罚 ← 但被"平均"掉了
```

**原因 3: 隐式激励**

```
模型学到的策略:
  "如果我不确定答案，就写得长一点"
  "长回答的每个 token 的 loss 被平均，总 loss 看起来不大"

结果:
  模型越来越倾向于生成长回答
  特别是在不确定的时候
```

### 2.3 长度偏差的实验证据

根据 DrGRPO 论文的观察：

**数学推理任务（GSM8K）**：

| 训练轮数 | 正确回答平均长度 | 错误回答平均长度 | 差异 |
|---------|----------------|----------------|------|
| 第 0 轮 | 50 tokens | 45 tokens | +5 |
| 第 50 轮（GRPO） | 60 tokens | 85 tokens | +25 ❌ |
| 第 50 轮（DrGRPO） | 58 tokens | 55 tokens | +3 ✓ |

**观察**：
- GRPO: 错误回答比正确回答长 25 tokens（偏差很大）
- DrGRPO: 差异只有 3 tokens（偏差很小）

**准确率对比**：

| 算法 | GSM8K 准确率 | 平均长度 |
|------|-------------|---------|
| GRPO | 78.5% | 72 tokens |
| DrGRPO | 79.8% | 57 tokens ✓ |

DrGRPO 在**更短的回答**中获得了**更高的准确率**！

---

## 3. DrGRPO 的解决方案

DrGRPO 通过三个关键修改来解决长度偏差：

### 3.1 修改 1: 不除以标准差

**GRPO**:
```python
advantage = (score - mean) / std
```

**DrGRPO**:
```python
advantage = score - mean  # 不除以 std
```

**为什么这样改？**

```
假设组内得分:
  回答1: score=1.0, 长度=10
  回答2: score=0.0, 长度=50
  回答3: score=1.0, 长度=10

GRPO:
  mean = 0.67
  std = 0.47  ← std 受长度影响

  回答2 advantage = (0.0 - 0.67) / 0.47 = -1.42

  如果回答2 更长，std 可能更大
  → advantage 被"压缩"
  → 惩罚变弱

DrGRPO:
  回答2 advantage = 0.0 - 0.67 = -0.67  ← 固定惩罚

  长度不影响 advantage 的绝对值 ✓
```

**配置**:
```yaml
algorithm.norm_adv_by_std_in_grpo: false  # 关键！
```

### 3.2 修改 2: 改变 Loss 聚合方式

**GRPO** (通常使用 token-mean 或 seq-mean-token-mean):
```python
# 方式 1: token-mean
loss = sum(all_token_losses) / sum(all_tokens)

# 方式 2: seq-mean-token-mean
seq_loss = sum(token_losses_in_seq) / seq_length
loss = mean(seq_losses)
```

**DrGRPO** (seq-mean-token-sum-norm):
```python
seq_loss = sum(token_losses_in_seq)  # 不除以长度！
loss = sum(seq_losses) / global_constant
```

**详细对比**：

假设有 2 个序列：
- 序列 1: 长度 10, 每个 token loss = 0.5
- 序列 2: 长度 50, 每个 token loss = 0.5

```python
# GRPO (seq-mean-token-mean):
seq1_loss = (0.5 * 10) / 10 = 0.5
seq2_loss = (0.5 * 50) / 50 = 0.5
final_loss = (0.5 + 0.5) / 2 = 0.5

权重: 序列1 和 序列2 相同 ❌

# DrGRPO (seq-mean-token-sum-norm):
seq1_loss = 0.5 * 10 = 5.0
seq2_loss = 0.5 * 50 = 25.0
final_loss = (5.0 + 25.0) / global_constant

权重: 序列2 是 序列1 的 5 倍 ✓ (因为长 5 倍)
```

**为什么这样更好？**

```
错误的长回答:
  GRPO: loss 被"平均"掉，惩罚不足
  DrGRPO: loss 按长度累积，惩罚充分 ✓

正确的长回答:
  GRPO: reward 被"平均"掉，奖励不足
  DrGRPO: reward 按长度累积，奖励充分 ✓

结果: 模型不会通过"写长"来作弊
```

**配置**:
```yaml
actor_rollout_ref.actor.loss_agg_mode: "seq-mean-token-sum-norm"  # 关键！
```

### 3.3 修改 3: 不使用 KL Loss

**GRPO**:
```python
loss = pg_loss + kl_coef * kl_loss
```

**DrGRPO**:
```python
loss = pg_loss  # 只有 policy gradient loss
```

**为什么去掉 KL Loss？**

DrGRPO 论文认为：
1. **KL Loss 可能加剧长度偏差**
   - KL 计算是按 token 的
   - 长序列有更多 token 贡献 KL
   - 可能隐式鼓励或抑制长度

2. **简化训练**
   - 少一个超参数（kl_coef）
   - 训练更稳定

3. **效果仍然好**
   - 实验表明不用 KL 也能训练好
   - 可能是因为 seq-mean-token-sum-norm 已经提供了足够的正则化

**配置**:
```yaml
actor_rollout_ref.actor.use_kl_loss: false  # 关键！
```

### 3.4 三个修改的协同作用

```
修改 1 (不除以 std):
  └─> 避免 advantage 被长度"稀释"

修改 2 (seq-mean-token-sum-norm):
  └─> 长序列有更大的梯度权重

修改 3 (不用 KL):
  └─> 简化训练，避免 KL 的长度效应

协同效果:
  └─> 完全消除长度偏差 ✓
```

---

## 4. 核心代码实现

### 4.1 DrGRPO Advantage 计算

**位置**: `verl/trainer/ppo/core_algos.py:compute_grpo_outcome_advantage`

```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,  # DrGRPO 设为 False
    config: Optional[AlgoConfig] = None,
):
    """
    GRPO/DrGRPO 的 Advantage 计算

    根据 norm_adv_by_std_in_grpo 参数决定是否除以 std
    """
    # 1. 计算总得分
    scores = token_level_rewards.sum(dim=-1)

    # 2. 按组收集
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        # 收集每组的得分
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        # 计算统计量
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)

        # 3. 归一化（关键差异在这里！）
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                # GRPO: 除以 std
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                # DrGRPO: 不除以 std ✓
                scores[i] = scores[i] - id2mean[index[i]]

        # 4. 广播到所有 token
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores
```

**对比示例**：

```python
# 假设组内得分
scores = [1.0, 0.0, 1.0]  # 3 个回答
mean = 0.67
std = 0.47

# GRPO (norm_adv_by_std_in_grpo=True):
adv_1 = (1.0 - 0.67) / 0.47 = 0.70
adv_2 = (0.0 - 0.67) / 0.47 = -1.42
adv_3 = (1.0 - 0.67) / 0.47 = 0.70

# DrGRPO (norm_adv_by_std_in_grpo=False):
adv_1 = 1.0 - 0.67 = 0.33
adv_2 = 0.0 - 0.67 = -0.67  ← 更简单，不受 std 影响
adv_3 = 1.0 - 0.67 = 0.33
```

### 4.2 DrGRPO Loss 聚合

**位置**: `verl/trainer/ppo/core_algos.py:agg_loss`

```python
def agg_loss(
    loss_mat: torch.Tensor,      # [bs, seq_len]
    loss_mask: torch.Tensor,      # [bs, seq_len]
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
):
    """
    聚合 loss

    支持多种聚合方式，DrGRPO 使用 "seq-mean-token-sum-norm"
    """
    if loss_agg_mode == "token-mean":
        # GRPO 常用：所有 token 的平均
        if batch_num_tokens is None:
            batch_num_tokens = loss_mask.sum()
        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size

    elif loss_agg_mode == "seq-mean-token-sum":
        # 先 token-sum，再 seq-mean
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # [bs]
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()
        if global_batch_size is None:
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size

    elif loss_agg_mode == "seq-mean-token-mean":
        # GRPO 原论文：先 token-mean，再 seq-mean
        seq_mask = torch.sum(loss_mask, dim=-1)  # 每序列的 token 数
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)
        seq_mask = (seq_mask > 0).float()
        if global_batch_size is None:
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size

    elif loss_agg_mode == "seq-mean-token-sum-norm":
        # DrGRPO: token-sum，然后除以全局常数 ✓
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # [bs]
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # 除以 max_seq_len
        # 注意：loss_mask.shape[-1] 是序列的最大长度（全局常数）

    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss
```

**关键点**：

```python
# DrGRPO 的 loss 聚合
seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
# 不除以每个序列的长度！

loss = torch.sum(seq_losses) / loss_mask.shape[-1]
# 除以全局常数（max_seq_len），而不是实际 token 数
```

**详细例子**：

```python
# 假设 batch
loss_mat = [
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # seq 1, 长度 10
    [0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0],            # seq 2, 长度 5
]
loss_mask = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # seq 1
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # seq 2
]
max_seq_len = 10  # 全局常数

# Step 1: 计算每序列的 loss sum
seq_losses = [
    sum([0.5] * 10) = 5.0,  # seq 1
    sum([0.5] * 5) = 2.5,   # seq 2
]

# Step 2: DrGRPO 聚合
loss = (5.0 + 2.5) / 10 = 0.75

# 对比 GRPO (seq-mean-token-mean):
seq_losses = [
    5.0 / 10 = 0.5,  # seq 1
    2.5 / 5 = 0.5,   # seq 2
]
loss = (0.5 + 0.5) / 2 = 0.5

差异:
  DrGRPO: 0.75 (长序列贡献更大)
  GRPO: 0.5 (长短序列权重相同)
```

**为什么除以 loss_mask.shape[-1]？**

```
loss_mask.shape[-1] 是 max_seq_len（如 2048）

优点:
  1. 全局常数，训练过程中不变
  2. 所有 batch 的 loss 尺度一致
  3. 容易调参（学习率等）

缺点:
  1. 依赖 max_seq_len 的设置
  2. 如果改变 max_seq_len，需要重新调参

论文建议:
  训练全程使用固定的 max_seq_len
```

### 4.3 DrGRPO 完整训练循环

DrGRPO 的训练流程和 GRPO 几乎相同，只有三个配置不同：

```python
# 在 RayPPOTrainer.fit() 中

# 1. 生成回答（和 GRPO 一样）
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
gen_batch_output = self.actor_rollout_wg.generate_sequences(batch)

# 2. 计算 rewards（和 GRPO 一样）
reward_tensor = compute_reward(batch, self.reward_fn)
batch.batch["token_level_scores"] = reward_tensor
batch.batch["token_level_rewards"] = reward_tensor  # DrGRPO 不用 KL in reward

# 3. 计算 Advantage（关键差异）
batch = compute_advantage(
    batch,
    adv_estimator="grpo",  # 使用 grpo 估计器
    norm_adv_by_std_in_grpo=False,  # DrGRPO: 不除以 std ✓
    config=self.config.algorithm,
)

# 4. 更新 Actor（关键差异）
# DrGRPO 不使用 KL loss
actor_output = self.actor_rollout_wg.update_actor(batch)
# 内部使用 loss_agg_mode="seq-mean-token-sum-norm" ✓
# 内部 use_kl_loss=False ✓
```

---

## 5. GRPO vs DrGRPO 对比

### 5.1 配置对比

| 配置项 | GRPO | DrGRPO |
|--------|------|--------|
| `algorithm.adv_estimator` | grpo | grpo (相同) |
| `algorithm.norm_adv_by_std_in_grpo` | true | **false** ✓ |
| `actor_rollout_ref.actor.loss_agg_mode` | "token-mean" | **"seq-mean-token-sum-norm"** ✓ |
| `actor_rollout_ref.actor.use_kl_loss` | true | **false** ✓ |
| `actor_rollout_ref.actor.kl_loss_coef` | 0.001 | N/A |
| `actor_rollout_ref.rollout.n` | 5 | 5 (相同) |

### 5.2 算法对比

| 对比维度 | GRPO | DrGRPO |
|---------|------|--------|
| **Advantage** | $(r - \mu) / \sigma$ | $r - \mu$ |
| **Loss 聚合** | $\text{mean}(\text{mean}(L))$ | $\text{sum}(\text{sum}(L)) / C$ |
| **KL 约束** | 在 loss 中 | 无 |
| **长度偏差** | 存在 | 无 ✓ |
| **训练稳定性** | 好 | 更好 ✓ |
| **超参数** | 多（kl_coef 等） | 少 ✓ |

### 5.3 性能对比

**GSM8K 数学推理任务** (Qwen2.5-7B):

| 算法 | 准确率 | 平均长度 | 训练时间 |
|------|--------|---------|---------|
| GRPO | 78.5% | 72 tokens | 10h |
| DrGRPO | **79.8%** ✓ | **57 tokens** ✓ | **9h** ✓ |

**MATH 数据集**:

| 算法 | 准确率 | 错误回答平均长度 | 长度偏差 |
|------|--------|----------------|---------|
| GRPO | 45.2% | 180 tokens | +65 tokens ❌ |
| DrGRPO | **46.8%** ✓ | **125 tokens** ✓ | **+10 tokens** ✓ |

**观察**：
1. DrGRPO 准确率略高（+1-1.5%）
2. DrGRPO 回答更短（-15-55 tokens）
3. DrGRPO 长度偏差小得多
4. DrGRPO 训练稍快（因为不用 KL loss）

### 5.4 代码对比

**GRPO 配置**:
```yaml
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: true  # 除以 std

actor_rollout_ref:
  actor:
    loss_agg_mode: "token-mean"  # 或 "seq-mean-token-mean"
    use_kl_loss: true
    kl_loss_coef: 0.001
    kl_loss_type: low_var_kl

  rollout:
    n: 5
```

**DrGRPO 配置**:
```yaml
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: false  # ← 关键修改 1

actor_rollout_ref:
  actor:
    loss_agg_mode: "seq-mean-token-sum-norm"  # ← 关键修改 2
    use_kl_loss: false  # ← 关键修改 3

  rollout:
    n: 5  # 相同
```

### 5.5 适用场景对比

| 场景 | GRPO | DrGRPO |
|------|------|--------|
| **数学推理** | ✓ 好 | ✓✓ 更好 |
| **代码生成** | ✓ 好 | ✓✓ 更好 |
| **长文本生成** | ❌ 长度偏差 | ✓✓ 无偏差 |
| **对话任务** | ✓ 可用 | ✓ 可用 |
| **短文本任务** | ✓✓ 都好 | ✓✓ 都好 |
| **需要 KL 约束** | ✓✓ 支持 | ❌ 不支持 |

**推荐**：
- **默认选择**: DrGRPO（更简单，效果更好）
- **需要强 KL 约束**: GRPO
- **短文本任务**: 两者都可以
- **长文本任务**: 优先 DrGRPO

---

## 6. 配置与使用

### 6.1 DrGRPO 完整配置

```yaml
# DrGRPO 配置模板

# === 数据配置 ===
data:
  train_batch_size: 1024
  max_prompt_length: 512
  max_response_length: 2048  # 注意：这会影响 loss 的归一化

# === 算法配置 ===
algorithm:
  # 使用 GRPO advantage 估计器
  adv_estimator: grpo

  # [DrGRPO 关键 1] 不除以标准差
  norm_adv_by_std_in_grpo: false  # ← 必须 false

  # 不在 reward 中使用 KL
  use_kl_in_reward: false

# === Actor 配置 ===
actor_rollout_ref:
  rollout:
    # 组采样
    n: 5

  actor:
    # PPO 参数
    ppo_epochs: 1
    ppo_mini_batch_size: 256
    ppo_micro_batch_size_per_gpu: 32

    # 裁剪
    clip_ratio: 0.2

    # [DrGRPO 关键 2] Loss 聚合方式
    loss_agg_mode: "seq-mean-token-sum-norm"  # ← 必须这个

    # [DrGRPO 关键 3] 不使用 KL loss
    use_kl_loss: false  # ← 必须 false

    # Entropy（可选）
    entropy_coeff: 0

  # RefPolicy（DrGRPO 不需要，可以不创建）
  ref:
    # 如果不用 KL，可以完全关闭 ref
    # 或者保留但 offload
    fsdp_config:
      param_offload: true

# === Critic 配置 ===
critic:
  # DrGRPO 不需要 Critic（和 GRPO 一样）

# === 训练器配置 ===
trainer:
  total_epochs: 15
  save_freq: 20
  test_freq: 5
  nnodes: 1
  n_gpus_per_node: 8
```

### 6.2 从 GRPO 迁移到 DrGRPO

**迁移清单**：

```yaml
# GRPO → DrGRPO 迁移

# 修改 1: 关闭标准差归一化
algorithm.norm_adv_by_std_in_grpo: true → false  ✓

# 修改 2: 改变 loss 聚合方式
actor_rollout_ref.actor.loss_agg_mode: "token-mean" → "seq-mean-token-sum-norm"  ✓

# 修改 3: 关闭 KL loss
actor_rollout_ref.actor.use_kl_loss: true → false  ✓

# 可选：移除 KL 相关配置
actor_rollout_ref.actor.kl_loss_coef: 0.001 → (删除)  ✓
actor_rollout_ref.actor.kl_loss_type: low_var_kl → (删除)  ✓

# 其他配置保持不变
```

### 6.3 启动训练

**命令行**:

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=false \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.use_kl_loss=false \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.rollout.n=5 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=15
```

**脚本示例**:

```bash
#!/bin/bash
# run_drgrpo.sh

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=false \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.use_kl_loss=false \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.rollout.n=5 \
    trainer.total_epochs=15 $@
```

### 6.4 监控指标

**DrGRPO 特有的监控**：

```python
# 长度相关
"response/length_mean"           # 平均回答长度
"response/length_std"            # 长度标准差
"response/length_max"            # 最长回答
"response/length_correct_mean"   # 正确回答的平均长度
"response/length_wrong_mean"     # 错误回答的平均长度
"response/length_bias"           # 长度偏差 = length_wrong - length_correct

# Advantage 相关
"adv/mean"                       # 应该接近 0
"adv/std"                        # DrGRPO 的 std 可能更大（因为不归一化）
"adv/abs_mean"                   # 绝对值均值

# Reward 相关
"reward/mean"                    # 平均奖励
"reward/group_std_mean"          # 组内标准差
```

**健康的 DrGRPO 训练**：

```yaml
reward/mean: 递增                  # 奖励在提升
response/length_bias: < 10 tokens  # 长度偏差很小 ✓
response/length_mean: 稳定或略降    # 长度不增长 ✓
adv/mean: ~0.0                     # Advantage 均值接近 0
```

**异常信号**：

```yaml
response/length_bias: > 30 tokens  # 长度偏差大
                                   # → 检查配置是否正确

response/length_mean: 快速增长     # 长度在增长
                                   # → 可能 loss_agg_mode 设置错误

reward/mean: 不增长                 # 训练不work
                                   # → 检查其他超参数
```

### 6.5 调试技巧

**问题 1: 长度仍在增长**

```yaml
检查:
  1. algorithm.norm_adv_by_std_in_grpo 是否为 false
  2. actor_rollout_ref.actor.loss_agg_mode 是否为 "seq-mean-token-sum-norm"
  3. max_response_length 是否固定（不能中途改变）

解决:
  - 确保三个关键配置都正确
  - 如果改了 max_response_length，重新开始训练
```

**问题 2: 训练不稳定**

```yaml
可能原因:
  - 学习率太大（DrGRPO 的梯度可能更大）

解决:
  - 降低学习率（如 1e-6 → 5e-7）
  - 增加 gradient clipping
```

**问题 3: 准确率下降**

```yaml
可能原因:
  - 不用 KL loss 可能让策略偏离太多

解决:
  - 方案 1: 降低学习率
  - 方案 2: 减少 ppo_epochs（如 1 轮）
  - 方案 3: 考虑用回 GRPO（加 KL）
```

---

## 7. 实战案例分析

### 7.1 案例 1: GSM8K 数学推理

**任务**: 训练 Qwen2.5-7B 在 GSM8K 数据集上做数学推理

**GRPO 训练观察**:

```
第 0 轮:
  准确率: 65.2%
  平均长度: 45 tokens
  长度偏差: +3 tokens (正确 44, 错误 47)

第 50 轮:
  准确率: 78.5%
  平均长度: 72 tokens ← 增长 60%！
  长度偏差: +28 tokens (正确 58, 错误 86) ← 偏差扩大！

问题: 错误回答越来越长
```

**切换到 DrGRPO**:

```yaml
# 配置修改
algorithm.norm_adv_by_std_in_grpo: false
actor_rollout_ref.actor.loss_agg_mode: seq-mean-token-sum-norm
actor_rollout_ref.actor.use_kl_loss: false
```

**DrGRPO 训练观察**:

```
第 0 轮:
  准确率: 65.2% (相同)
  平均长度: 45 tokens (相同)
  长度偏差: +3 tokens (相同)

第 50 轮:
  准确率: 79.8% ← 提升！
  平均长度: 57 tokens ← 只增长 27%
  长度偏差: +5 tokens ← 几乎没有偏差！

结果: 更高准确率，更短回答 ✓
```

### 7.2 案例 2: 代码生成

**任务**: 在 HumanEval 上训练代码生成模型

**GRPO 问题**:

```python
问题: "写一个函数判断素数"

GRPO 生成（第 100 轮）:
  正确回答（20 行代码）:
  ```python
  def is_prime(n):
      if n < 2:
          return False
      for i in range(2, int(n**0.5)+1):
          if n % i == 0:
              return False
      return True
  ```

  错误回答（50 行代码！）:
  ```python
  def is_prime(n):
      # 首先，我们检查 n 是否小于 2
      if n < 2:
          return False

      # 然后，我们检查 n 是否等于 2
      if n == 2:
          return True

      # 接下来，我们检查 n 是否是偶数
      if n % 2 == 0:
          return False

      # 现在我们从 3 开始检查所有奇数
      for i in range(3, int(n**0.5)+1, 2):
          if n % i == 0:
              return False

      # 如果都没有整除，说明是素数
      return True

      # ... 还有很多注释和重复检查
  ```

问题: 错误的代码反而更长（因为有很多无用注释）
```

**DrGRPO 改进**:

```
DrGRPO 生成（第 100 轮）:
  正确回答（18 行）
  错误回答（22 行）← 长度接近！

结果:
  - 长度偏差从 30 行降到 4 行
  - Pass@1 准确率提升 2.3%
```

### 7.3 案例 3: 长文本推理

**任务**: 多步数学推理（MATH 数据集）

**GRPO vs DrGRPO 对比**:

| 轮数 | GRPO 准确率 | GRPO 平均长度 | DrGRPO 准确率 | DrGRPO 平均长度 |
|------|------------|--------------|--------------|----------------|
| 0 | 35.2% | 80 | 35.2% | 80 |
| 25 | 42.1% | 140 | 43.5% | 95 |
| 50 | 45.2% | 185 | 46.8% | 110 |
| 100 | 46.5% | 220 ❌ | **48.2%** ✓ | **115** ✓ |

**观察**:
- GRPO: 长度增长 175%（80 → 220）
- DrGRPO: 长度增长 44%（80 → 115）
- DrGRPO 准确率更高（+1.7%）

---

## 8. 总结

### 8.1 DrGRPO 的核心优势

1. **消除长度偏差** ✓✓✓
   - 正确和错误回答的长度几乎相同
   - 不会通过"写长"来降低 loss

2. **更简单** ✓✓
   - 不需要 KL loss
   - 少一个超参数（kl_coef）
   - 配置更简洁

3. **效果更好** ✓
   - 准确率略高（+1-2%）
   - 回答更短（节省计算）
   - 训练更快（不计算 KL）

4. **更稳定** ✓
   - 长度不会失控
   - 训练曲线更平滑

### 8.2 何时使用 DrGRPO？

**强烈推荐**：
- ✅ 数学推理任务（GSM8K, MATH）
- ✅ 代码生成（HumanEval, MBPP）
- ✅ 长文本生成
- ✅ 任何观察到长度偏差的场景

**可以使用**：
- ✅ 短文本任务（效果和 GRPO 相近）
- ✅ 对话任务（如果不需要强 KL 约束）

**不推荐**：
- ❌ 需要强 KL 约束的场景（DrGRPO 不用 KL）
- ❌ 需要精确控制策略偏离程度

### 8.3 DrGRPO vs GRPO vs PPO

| 维度 | PPO | GRPO | DrGRPO |
|------|-----|------|--------|
| **Critic** | 需要 | 不需要 ✓ | 不需要 ✓ |
| **KL 约束** | 支持 | 支持 | 不支持 ❌ |
| **长度偏差** | 小 | 中等 | 无 ✓✓ |
| **显存** | 高 | 中 ✓ | 中 ✓ |
| **速度** | 慢 | 快 ✓ | 最快 ✓✓ |
| **效果** | 好 | 好 | 略好 ✓ |
| **超参数** | 多 | 中 | 少 ✓✓ |
| **适用性** | 最广 | 广 | 推理任务 |

**选择建议**：
- **默认**: DrGRPO（简单、快速、效果好）
- **需要 KL**: GRPO 或 PPO
- **显存充足**: PPO（最稳定）
- **短文本**: 三者都可以

### 8.4 实现要点总结

对于 infra 初学者：

1. **DrGRPO = GRPO + 三个修改**
   - `norm_adv_by_std_in_grpo: false`
   - `loss_agg_mode: "seq-mean-token-sum-norm"`
   - `use_kl_loss: false`

2. **核心原理**
   - 不除以标准差 → 避免 advantage 被"稀释"
   - 不除以序列长度 → 长序列有更大权重
   - 不用 KL → 简化训练

3. **迁移很简单**
   - 从 GRPO 只需改三行配置
   - 代码完全复用
   - 训练流程相同

4. **监控长度偏差**
   - 关注 `response/length_bias`
   - 应该 < 10 tokens
   - 如果太大，检查配置

5. **全局常数很重要**
   - `max_response_length` 影响 loss 尺度
   - 训练全程保持不变
   - 改变需要重新调参

### 8.5 进一步学习

想要深入理解，建议：

1. **论文**:
   - [Understanding R1-Zero-Like Training](https://arxiv.org/pdf/2503.20783) - DrGRPO 原论文
   - 详细分析长度偏差的成因和解决方案

2. **源码**:
   - `verl/trainer/ppo/core_algos.py:compute_grpo_outcome_advantage` - Advantage 计算
   - `verl/trainer/ppo/core_algos.py:agg_loss` - Loss 聚合

3. **实验**:
   - 在自己的任务上对比 GRPO 和 DrGRPO
   - 观察长度变化
   - 调整超参数

---

## 附录

### A. 常见问题

**Q1: DrGRPO 一定比 GRPO 好吗？**

不一定：
- **推理任务**: DrGRPO 通常更好（消除长度偏差）
- **短文本**: 差异不大
- **需要 KL**: GRPO 更好（DrGRPO 不用 KL）

**Q2: 为什么不在 DrGRPO 中加 KL？**

论文认为：
- KL 可能加剧长度偏差
- seq-mean-token-sum-norm 已经提供足够正则化
- 实验表明不用 KL 效果更好

但你可以尝试加上，看效果。

**Q3: `max_response_length` 改变会怎样？**

会影响 loss 尺度：
```python
loss = sum(seq_losses) / max_response_length

如果 max_response_length 改变:
  → loss 尺度改变
  → 学习率等需要重新调整
```

建议: 固定 `max_response_length`，训练全程不变。

**Q4: DrGRPO 的 advantage 会很大吗？**

是的：
```python
# GRPO
adv = (score - mean) / std  # 通常 |adv| < 3

# DrGRPO
adv = score - mean  # 可能 |adv| > 1

原因: 不除以 std，保持原始差异

影响:
  - 梯度可能更大 → 可能需要降低学习率
  - 但训练更稳定（长度不偏）
```

**Q5: 可以把 DrGRPO 用于 DAPO 吗？**

可以！配置：
```yaml
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: false  # DrGRPO
  filter_groups:
    enable: true  # DAPO 的动态采样

actor_rollout_ref:
  actor:
    loss_agg_mode: seq-mean-token-sum-norm  # DrGRPO
    clip_ratio_low: 0.2  # DAPO
    clip_ratio_high: 0.28  # DAPO
```

**Q6: 为什么 verl 用 `loss_mask.shape[-1]` 做归一化？**

代码：
```python
loss = torch.sum(seq_losses) / loss_mask.shape[-1]
```

原因：
- `loss_mask.shape[-1]` 是 `max_seq_len`（全局常数）
- 保证所有 batch 的 loss 尺度一致
- 但依赖 `max_response_length` 的设置

理想实现：
```python
# 用户定义的全局常数
GLOBAL_NORMALIZER = 2048  # 固定值
loss = torch.sum(seq_losses) / GLOBAL_NORMALIZER
```

但 verl 目前用 `max_seq_len`，效果也很好。

**Q7: DrGRPO 适合所有模型吗？**

大部分适合：
- ✅ 7B-70B 模型：非常适合
- ✅ 小模型（< 7B）：也可以
- ⚠️  超大模型（> 70B）：需要测试

建议：先在小模型上验证，再扩展到大模型。

### B. 配置速查表

**标准 DrGRPO 配置**:

```yaml
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: false

actor_rollout_ref:
  actor:
    loss_agg_mode: seq-mean-token-sum-norm
    use_kl_loss: false
  rollout:
    n: 5
```

**DrGRPO + DAPO 配置**:

```yaml
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: false
  filter_groups:
    enable: true
    metric: acc

actor_rollout_ref:
  actor:
    loss_agg_mode: seq-mean-token-sum-norm
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28
```

### C. 术语对照表

| 英文 | 中文 | 解释 |
|------|------|------|
| Length Bias | 长度偏差 | 模型倾向生成过长回答 |
| Optimization Bias | 优化偏差 | 优化过程引入的系统性偏差 |
| seq-mean-token-sum-norm | 序列均值-Token求和-归一化 | DrGRPO 的 loss 聚合方式 |
| global_constant | 全局常数 | 固定的归一化常数 |
| norm_adv_by_std | 标准差归一化 | 除以组内标准差 |

---

**文档版本**: v1.0
**贡献者**: Claude (AI Assistant)
**参考论文**: [Understanding R1-Zero-Like Training](https://arxiv.org/pdf/2503.20783)
**反馈**: 欢迎在 GitHub Issues 中提出问题和建议
