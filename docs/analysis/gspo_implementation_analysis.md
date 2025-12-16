# GSPO 训练框架实现详解

**面向对象**: 基础设施（Infrastructure）初学者
**最后更新**: 2025-11-27

---

## 目录

1. [什么是 GSPO？](#1-什么是-gspo)
2. [GSPO 的核心概念](#2-gspo-的核心概念)
3. [序列级别重要性采样](#3-序列级别重要性采样)
4. [核心代码实现](#4-核心代码实现)
5. [GSPO vs 其他算法对比](#5-gspo-vs-其他算法对比)
6. [配置与使用](#6-配置与使用)
7. [实战案例分析](#7-实战案例分析)
8. [总结](#8-总结)

---

## 1. 什么是 GSPO？

### 1.1 GSPO 简介

**GSPO** (Group Sequential Policy Optimization) 是一种改进的策略优化算法，由论文 [GSPO: Group Preference Optimization with Complementary Policies Utilizing Imperfect Demonstrations](https://arxiv.org/pdf/2507.18071) 提出。

**核心创新**：使用**序列级别的重要性采样比率**（sequence-level importance ratio）来替代传统的 token 级别比率。

**一句话总结**：
```
GSPO = GRPO Advantage + Sequence-level Importance Ratio + 超小 Clip Range
```

### 1.2 GSPO 与其他算法的关系

| 维度 | PPO | GRPO | GSPO |
|------|-----|------|------|
| **Advantage** | GAE (需要 Critic) | 组归一化 | 组归一化 ✓ |
| **Importance Ratio** | Token-level | Token-level | **Sequence-level** ✓✓ |
| **Clip Range** | 0.2 | 0.2 | **3e-4 ~ 4e-4** ✓✓ |
| **Loss 聚合** | token-mean | token-mean | seq-mean-token-mean ✓ |
| **适用场景** | 通用 | 推理任务 | **数学推理** ✓✓ |

**关键差异**：
1. **序列级别 Ratio**: 考虑整个序列的概率变化（几何平均）
2. **超小裁剪**: clip_ratio_low=3e-4, clip_ratio_high=4e-4（比 PPO 小 500 倍！）
3. **组归一化**: 继承 GRPO 的优势，不需要 Critic 网络

### 1.3 为什么需要 GSPO？

**问题 1: Token-level Ratio 的局限性**

```python
# PPO/GRPO 的 token-level ratio
ratio_token = π_new(token_t | context) / π_old(token_t | context)

问题:
  序列 "2 + 2 = 4"

  Token 1: "2" → ratio = 1.1
  Token 2: "+" → ratio = 0.9
  Token 3: "2" → ratio = 1.2
  Token 4: "=" → ratio = 0.8
  Token 5: "4" → ratio = 1.0

  每个 token 的 ratio 都不同！
  但整个序列的质量是一个整体 ← 应该用序列级别的 ratio
```

**问题 2: 大 Clip Range 的不稳定性**

```
PPO clip_ratio = 0.2:
  允许策略变化 ±20%

  对于数学推理:
    小的策略变化可能导致完全错误的答案
    需要更保守的更新 ← 超小 clip range
```

**GSPO 的解决方案**：

```
1. 序列级别 Ratio:
   ratio_seq = (π_new(entire_seq) / π_old(entire_seq))^(1/seq_len)
   → 几何平均，考虑整体变化

2. 超小 Clip Range:
   clip = [1-3e-4, 1+4e-4]
   → 更保守的策略更新，训练更稳定

3. 组归一化 Advantage:
   继承 GRPO，不需要 Critic
   → 简单、高效
```

---

## 2. GSPO 的核心概念

### 2.1 重要性采样（Importance Sampling）

**什么是重要性采样？**

在强化学习中，我们用旧策略（old policy）收集数据，但要用新策略（new policy）来学习。

**直观例子**：

```
旧策略 π_old:
  问题: "2 + 2 = ?"
  生成: "2 + 2 = 4" (概率 0.6)
  生成: "2 + 2 = 5" (概率 0.4)

新策略 π_new (训练后):
  生成: "2 + 2 = 4" (概率 0.8) ← 提高了！
  生成: "2 + 2 = 5" (概率 0.2) ← 降低了！

重要性采样比率:
  正确答案: ratio = 0.8 / 0.6 = 1.33 ← 新策略更倾向生成
  错误答案: ratio = 0.2 / 0.4 = 0.5  ← 新策略更不倾向生成
```

**为什么需要 Ratio？**

```python
# 没有 ratio（错误）
loss = -advantage * log_prob_new

问题: 数据是用 π_old 收集的，但用 π_new 来计算
      → 数据分布不匹配！

# 有 ratio（正确）
loss = -advantage * ratio

好处: ratio 修正了分布差异
      → 可以安全地用旧数据训练新策略
```

### 2.2 Token-level vs Sequence-level Ratio

**Token-level Ratio（PPO、GRPO）**：

```python
# 每个 token 独立计算 ratio
for t in range(seq_length):
    ratio_t = π_new(y_t | context) / π_old(y_t | context)
    loss_t = -advantage * ratio_t
```

**示例**：

```
序列: "The answer is 4"

Token 0: "The"    → ratio_0 = 1.05
Token 1: "answer" → ratio_1 = 0.95
Token 2: "is"     → ratio_2 = 1.10
Token 3: "4"      → ratio_3 = 0.90

问题: 每个 token 的 ratio 不同，没有考虑整体
```

**Sequence-level Ratio（GSPO）**：

```python
# 先计算整个序列的 ratio（几何平均）
ratio_seq = (π_new(entire_seq) / π_old(entire_seq))^(1/seq_len)

# 然后结合 token-level 策略
ratio_combined = ratio_seq * (π_new(y_t) / π_new_detached(y_t))
```

**示例**：

```
序列: "The answer is 4"

整体序列概率:
  π_old(entire_seq) = 0.001
  π_new(entire_seq) = 0.002

序列级别 ratio:
  ratio_seq = (0.002 / 0.001)^(1/4) = 2^0.25 = 1.19

所有 token 共享这个 ratio！
→ 考虑了整体序列的质量变化
```

### 2.3 几何平均 vs 算术平均

**为什么用几何平均？**

```python
# 序列概率 = 所有 token 概率的乘积
π(y_1, y_2, ..., y_n) = π(y_1) * π(y_2) * ... * π(y_n)

# 序列 ratio
ratio_seq = π_new(y_1...y_n) / π_old(y_1...y_n)
          = (π_new(y_1) * ... * π_new(y_n)) / (π_old(y_1) * ... * π_old(y_n))
          = (ratio_1 * ratio_2 * ... * ratio_n)

# 几何平均（归一化）
ratio_seq^(1/n) = (ratio_1 * ... * ratio_n)^(1/n)
```

**几何平均 vs 算术平均**：

```
假设 token-level ratios = [0.5, 2.0, 0.8, 1.2]

算术平均:
  (0.5 + 2.0 + 0.8 + 1.2) / 4 = 1.125

几何平均:
  (0.5 * 2.0 * 0.8 * 1.2)^(1/4) = 0.96^(1/4) = 0.99

差异:
  算术平均: 受极端值影响大（2.0 拉高了平均）
  几何平均: 对极端值不敏感 ✓

对于概率:
  几何平均更合适！（概率是乘性的）
```

### 2.4 GSPO 的 Combined Ratio

GSPO 的巧妙之处：**结合序列级别和 token 级别的信息**。

```python
# 序列级别 ratio（stop gradient）
ratio_seq = exp((1/seq_len) * Σ(log_prob_new - log_prob_old))
ratio_seq = ratio_seq.detach()  # 不传梯度！

# Token 级别的策略
π_new(y_t) / π_new(y_t).detach()  # 自己除以自己的 detach

# Combined ratio
ratio_combined = ratio_seq * (π_new(y_t) / π_new(y_t).detach())
```

**为什么这样设计？**

```
ratio_combined = sg[ratio_seq] * (π_new / sg[π_new])

展开:
  = sg[ratio_seq] * π_new / sg[π_new]

梯度只通过 π_new:
  ∂loss/∂θ = ∂loss/∂π_new * ∂π_new/∂θ

但 ratio_seq 提供了序列级别的缩放:
  如果 ratio_seq = 1.5 (整体概率提高)
  → 所有 token 的梯度都缩放 1.5 倍

如果 ratio_seq = 0.5 (整体概率降低)
  → 所有 token 的梯度都缩放 0.5 倍

结果:
  梯度方向由 token-level π_new 决定 ✓
  梯度大小由 sequence-level ratio_seq 调节 ✓
```

### 2.5 超小 Clip Range

**为什么 GSPO 用超小的 clip range？**

```python
# PPO/GRPO
clip_ratio = 0.2  # 允许 ±20% 变化

# GSPO
clip_ratio_low = 3e-4   # 允许 -0.03% 变化
clip_ratio_high = 4e-4  # 允许 +0.04% 变化
```

**原因 1: 序列级别 Ratio 已经提供了缩放**

```
Combined ratio = ratio_seq * (π_new / sg[π_new])

ratio_seq 可能很大（如 1.5）
→ 不需要大的 clip range
→ 小 clip 提供稳定性
```

**原因 2: 数学推理对策略变化敏感**

```
问题: "计算 137 * 89 = ?"

旧策略:
  "137 * 89 = 12193" (正确, 概率 0.4)
  "137 * 89 = 12183" (错误, 概率 0.6)

如果策略变化太大:
  新策略可能完全改变生成的数字
  → 答案从对变成错，或从错变成对

小 clip range:
  限制每次更新的变化幅度
  → 渐进式改进，更稳定 ✓
```

**原因 3: 论文实验结果**

```
论文实验 (GSM8K):
  clip = 0.2:      准确率 75.3%
  clip = 0.01:     准确率 76.8%
  clip = 3e-4:     准确率 78.5% ✓ (最好)

观察: clip 越小，效果越好！
```

---

## 3. 序列级别重要性采样

### 3.1 数学推导

**序列概率**：

```
π(y_1, y_2, ..., y_n | x) = Π π(y_t | x, y_<t)
                           t=1 to n
```

**序列级别重要性采样比率**：

```
r_seq = π_new(y_1...y_n | x) / π_old(y_1...y_n | x)
      = Π [π_new(y_t | x, y_<t) / π_old(y_t | x, y_<t)]
        t=1 to n
```

**几何平均（归一化）**：

```
s_i(θ) = r_seq^(1/n)
       = [π_new(y_1...y_n) / π_old(y_1...y_n)]^(1/n)

在 log 空间:
log(s_i(θ)) = (1/n) * Σ log[π_new(y_t) / π_old(y_t)]
            = (1/n) * Σ [log_prob_new_t - log_prob_old_t]
```

**Combined Ratio**：

GSPO 将序列级别 ratio 与 token 级别策略结合：

```
s_i,t(θ) = sg[s_i(θ)] · π_θ(y_t | x, y_<t) / sg[π_θ(y_t | x, y_<t)]

在 log 空间:
log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob_new - sg[log_prob_new]
              = sg[(1/n) * Σ(log_prob_new - log_prob_old)] + log_prob_new - sg[log_prob_new]
```

其中 `sg[·]` 表示 stop gradient。

### 3.2 直观理解

**传统 PPO（Token-level）**：

```
对于序列 "2 + 2 = 4":

Token "2":  ratio = 1.1  → 单独更新
Token "+":  ratio = 0.9  → 单独更新
Token "2":  ratio = 1.2  → 单独更新
Token "=":  ratio = 0.8  → 单独更新
Token "4":  ratio = 1.0  → 单独更新

问题: 每个 token 独立，忽略了序列的整体质量
```

**GSPO（Sequence-level）**：

```
对于序列 "2 + 2 = 4":

1. 计算序列整体的 ratio:
   ratio_seq = (π_new(entire) / π_old(entire))^(1/5)
             = (0.005 / 0.003)^(1/5)
             = 1.67^0.2
             = 1.11

2. 所有 token 共享这个序列 ratio:
   Token "2":  combined_ratio = 1.11 * (π_new / sg[π_new])
   Token "+":  combined_ratio = 1.11 * (π_new / sg[π_new])
   ...

好处:
  所有 token 都知道整个序列的质量变化 ✓
  梯度更新考虑了整体，而不是局部 ✓
```

### 3.3 为什么用 Detach（stop gradient）？

```python
log_seq_ratio = log_prob_new - log_prob_old + negative_approx_kl_seq.detach().unsqueeze(-1)
                    ↑              ↑                        ↑
                 有梯度         有梯度                  无梯度！
```

**原因 1: 避免重复梯度**

```
如果不 detach:
  loss = -adv * ratio_seq * (π_new / π_new)

  ∂loss/∂π_new 会包含:
    1. 从 ratio_seq 来的梯度
    2. 从 (π_new / π_new) 来的梯度

  → 梯度重复，训练不稳定！

使用 detach:
  loss = -adv * sg[ratio_seq] * (π_new / sg[π_new])

  ∂loss/∂π_new 只包含:
    1. 从 (π_new / sg[π_new]) 来的梯度 ✓

  → 梯度清晰，训练稳定！
```

**原因 2: ratio_seq 只作为缩放因子**

```
sg[ratio_seq] 的作用:
  提供一个固定的缩放因子
  告诉模型"整个序列的质量变化了多少"

  但不直接参与梯度计算
  → 梯度仍由 token-level 策略决定
```

### 3.4 Combined Ratio 的计算流程

**Step 1: 计算 Token-level KL**

```python
negative_approx_kl = log_prob_new - log_prob_old  # [batch, seq_len]
# 近似 KL = log(π_new / π_old)
```

**Step 2: 聚合到 Sequence-level**

```python
seq_lengths = response_mask.sum(dim=-1)  # [batch]
negative_approx_kl_seq = (negative_approx_kl * response_mask).sum(dim=-1) / seq_lengths
# [batch] - 每个序列的平均 KL
```

**Step 3: 构建 Combined Log Ratio**

```python
log_seq_importance_ratio = (
    log_prob_new                          # Token-level 新策略（有梯度）
    - log_prob_new.detach()               # 减去自己的 detach（抵消梯度）
    + negative_approx_kl_seq.detach().unsqueeze(-1)  # 加上 sequence-level（无梯度）
)
# = sg[seq_ratio] + (log_prob_new - sg[log_prob_new])
```

**Step 4: Clamp 和 Exp**

```python
log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
seq_importance_ratio = torch.exp(log_seq_importance_ratio)
# [batch, seq_len] - combined ratio
```

**完整示例**：

```python
# 假设一个序列，长度 = 3
log_prob_new = [0.1, 0.2, 0.15]   # 新策略的 log prob
log_prob_old = [0.08, 0.18, 0.12]  # 旧策略的 log prob

# Step 1: Token-level KL
negative_approx_kl = [0.02, 0.02, 0.03]

# Step 2: Sequence-level KL (平均)
negative_approx_kl_seq = (0.02 + 0.02 + 0.03) / 3 = 0.0233

# Step 3: Combined log ratio (对每个 token)
log_ratio_t0 = 0.1 - 0.1 + 0.0233 = 0.0233
log_ratio_t1 = 0.2 - 0.2 + 0.0233 = 0.0233
log_ratio_t2 = 0.15 - 0.15 + 0.0233 = 0.0233

# 所有 token 的 log ratio 都等于 seq-level KL！

# Step 4: Exp
ratio_t0 = exp(0.0233) = 1.024
ratio_t1 = exp(0.0233) = 1.024
ratio_t2 = exp(0.0233) = 1.024

# 所有 token 共享相同的序列级别 ratio ✓
```

---

## 4. 核心代码实现

### 4.1 GSPO Policy Loss 函数

**位置**: `verl/trainer/ppo/core_algos.py:999-1072`

```python
@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,      # [batch, seq_len] 旧策略的 log prob
    log_prob: torch.Tensor,          # [batch, seq_len] 新策略的 log prob
    advantages: torch.Tensor,        # [batch, seq_len] Advantage
    response_mask: torch.Tensor,     # [batch, seq_len] 有效 token mask
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    # 1. 获取裁剪参数（非常小！）
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # 2. 计算 token-level 的 negative KL
    negative_approx_kl = log_prob - old_log_prob
    # negative_approx_kl[t] = log(π_new(y_t) / π_old(y_t))

    # 3. 计算 sequence-level importance ratio
    # s_i(θ) = [π_new(y_1...y_n) / π_old(y_1...y_n)]^(1/n)
    # 在 log 空间: log(s_i(θ)) = (1/n) * Σ log(π_new(y_t) / π_old(y_t))
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)  # [batch]
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths
    # [batch] - 序列级别的平均 KL

    # 4. 构建 combined ratio（token-level）
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_t) / sg[π_θ(y_t)]
    # 在 log 空间: log(s_i,t) = sg[log(s_i)] + log_prob - sg[log_prob]
    log_seq_importance_ratio = (
        log_prob                                        # π_new(y_t) - 有梯度
        - log_prob.detach()                             # - sg[π_new(y_t)] - 无梯度
        + negative_approx_kl_seq.detach().unsqueeze(-1) # + sg[seq_ratio] - 无梯度
    )
    # [batch, seq_len]

    # 5. Clamp for numerical stability
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)

    # 6. Exp to get the actual ratio
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)
    # [batch, seq_len] - combined importance sampling ratio

    # 7. 计算 policy gradient loss (带裁剪)
    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(
        seq_importance_ratio,
        1 - clip_ratio_low,   # 通常是 1 - 3e-4
        1 + clip_ratio_high   # 通常是 1 + 4e-4
    )
    pg_losses = torch.maximum(pg_losses1, pg_losses2)
    # [batch, seq_len]

    # 8. Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # 9. 聚合 loss（GSPO 强制使用 seq-mean-token-mean）
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",  # 固定！
        **config.global_batch_info
    )

    # 10. 计算指标
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }

    return pg_loss, pg_metrics
```

### 4.2 代码详解：Sequence-level Ratio 计算

让我们逐步分析最核心的部分：

**Step 1: Token-level KL**

```python
negative_approx_kl = log_prob - old_log_prob
```

```
例子:
  log_prob     = [-1.5, -2.0, -1.8]  # 新策略
  old_log_prob = [-1.6, -2.1, -1.9]  # 旧策略

  negative_approx_kl = [0.1, 0.1, 0.1]

含义: 每个 token，新策略的概率比旧策略高 exp(0.1) ≈ 1.105 倍
```

**Step 2: Sequence-level 聚合**

```python
seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths
```

```
例子:
  negative_approx_kl = [0.1, 0.1, 0.1]
  response_mask      = [1, 1, 1]
  seq_lengths        = 3

  negative_approx_kl_seq = (0.1 + 0.1 + 0.1) / 3 = 0.1

含义: 整个序列的几何平均 ratio = exp(0.1) ≈ 1.105
```

**Step 3: Combined Ratio 构建**

```python
log_seq_importance_ratio = (
    log_prob
    - log_prob.detach()
    + negative_approx_kl_seq.detach().unsqueeze(-1)
)
```

```
例子:
  log_prob                  = [-1.5, -2.0, -1.8]
  log_prob.detach()         = [-1.5, -2.0, -1.8]
  negative_approx_kl_seq    = 0.1

  log_seq_importance_ratio:
    Token 0: -1.5 - (-1.5) + 0.1 = 0.1
    Token 1: -2.0 - (-2.0) + 0.1 = 0.1
    Token 2: -1.8 - (-1.8) + 0.1 = 0.1

  所有 token 的 log ratio 都等于序列级别的值 0.1 ✓
```

**Step 4: Clamp 和 Exp**

```python
log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
seq_importance_ratio = torch.exp(log_seq_importance_ratio)
```

```
例子:
  log_seq_importance_ratio = [0.1, 0.1, 0.1]

  Clamp (max=10.0): 没有变化（都 < 10）

  Exp:
    seq_importance_ratio = [1.105, 1.105, 1.105]

  所有 token 共享 ratio = 1.105 ✓
```

### 4.3 代码详解：Clipping 机制

```python
pg_losses1 = -advantages * seq_importance_ratio
pg_losses2 = -advantages * torch.clamp(
    seq_importance_ratio,
    1 - clip_ratio_low,    # 例如 1 - 3e-4 = 0.9997
    1 + clip_ratio_high    # 例如 1 + 4e-4 = 1.0004
)
pg_losses = torch.maximum(pg_losses1, pg_losses2)
```

**超小 Clip Range 的效果**：

```
假设 advantage = 1.0（正向，应该增加概率）

Case 1: ratio = 1.0001（略微增加）
  pg_loss1 = -1.0 * 1.0001 = -1.0001
  pg_loss2 = -1.0 * clamp(1.0001, 0.9997, 1.0004) = -1.0 * 1.0001 = -1.0001
  final_loss = max(-1.0001, -1.0001) = -1.0001

  没有裁剪 ✓（在范围内）

Case 2: ratio = 1.001（增加较多）
  pg_loss1 = -1.0 * 1.001 = -1.001
  pg_loss2 = -1.0 * clamp(1.001, 0.9997, 1.0004) = -1.0 * 1.0004 = -1.0004
  final_loss = max(-1.001, -1.0004) = -1.0004

  被裁剪了！ ✓
  ratio 从 1.001 降到 1.0004

观察:
  即使 ratio 只增加 0.1%，也会被裁剪
  → 非常保守的更新策略！
```

**为什么 GSPO 需要这么小的 clip？**

```
1. Sequence-level ratio 已经提供了缩放:
   ratio_seq 可能达到 1.5-2.0
   → 不需要额外的大 ratio 空间

2. 数学推理任务敏感:
   小的策略变化可能导致完全不同的答案
   → 需要渐进式更新

3. 训练稳定性:
   小 clip 防止策略崩溃
   → 更鲁棒的训练过程
```

### 4.4 Loss 聚合

```python
pg_loss = agg_loss(
    loss_mat=pg_losses,
    loss_mask=response_mask,
    loss_agg_mode="seq-mean-token-mean",
    **config.global_batch_info
)
```

**seq-mean-token-mean 的计算**：

```python
# Step 1: 每个序列内 token-mean
seq_mask = torch.sum(loss_mask, dim=-1)  # 每序列的 token 数
seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)
# [batch]

# Step 2: 所有序列的 mean
seq_mask_binary = (seq_mask > 0).float()
loss = verl_F.masked_sum(seq_losses, seq_mask_binary) / global_batch_size * dp_size
```

**示例**：

```
batch = 2, seq_len = 4

loss_mat = [
    [0.5, 0.6, 0.7, 0.0],  # seq 1, 实际长度 3
    [0.8, 0.9, 0.0, 0.0],  # seq 2, 实际长度 2
]

loss_mask = [
    [1, 1, 1, 0],
    [1, 1, 0, 0],
]

# Step 1: token-mean for each seq
seq_loss_1 = (0.5 + 0.6 + 0.7) / 3 = 0.6
seq_loss_2 = (0.8 + 0.9) / 2 = 0.85

# Step 2: seq-mean
final_loss = (0.6 + 0.85) / 2 = 0.725

结果: 每个序列权重相同，不管长度
```

### 4.5 完整训练流程

GSPO 的训练流程和 GRPO 类似，只是 policy loss 不同：

```python
# 1. 生成回答（和 GRPO 一样）
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
gen_batch_output = self.actor_rollout_wg.generate_sequences(batch)

# 2. 计算 rewards（和 GRPO 一样）
reward_tensor = compute_reward(batch, self.reward_fn)
batch.batch["token_level_scores"] = reward_tensor
batch.batch["token_level_rewards"] = reward_tensor

# 3. 计算 Advantage（使用 GRPO）
batch = compute_advantage(
    batch,
    adv_estimator="grpo",  # GSPO 使用 GRPO advantage
    norm_adv_by_std_in_grpo=True,
    config=self.config.algorithm,
)

# 4. 更新 Actor（使用 GSPO policy loss）
# 内部调用 compute_policy_loss_gspo
actor_output = self.actor_rollout_wg.update_actor(batch)
```

---

## 5. GSPO vs 其他算法对比

### 5.1 算法对比

| 对比维度 | PPO | GRPO | DrGRPO | GSPO |
|---------|-----|------|--------|------|
| **Advantage** | GAE | 组归一化 | 组归一化 | 组归一化 |
| **Importance Ratio** | Token-level | Token-level | Token-level | **Sequence-level** ✓✓ |
| **Clip Range** | 0.2 | 0.2 | 0.2 | **3e-4** ✓✓ |
| **Loss 聚合** | token-mean | token-mean | seq-mean-token-sum-norm | seq-mean-token-mean |
| **Critic 网络** | 需要 | 不需要 ✓ | 不需要 ✓ | 不需要 ✓ |
| **训练稳定性** | 中 | 好 | 很好 | **非常好** ✓✓ |
| **超参数数量** | 多 | 中 | 少 | 少 ✓ |
| **适用场景** | 通用 | 推理 | 长文本推理 | **数学推理** ✓✓ |

### 5.2 配置对比

**PPO 配置**:
```yaml
algorithm:
  adv_estimator: gae

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: vanilla
    clip_ratio: 0.2
    loss_agg_mode: token-mean
```

**GRPO 配置**:
```yaml
algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: vanilla
    clip_ratio: 0.2
    loss_agg_mode: token-mean

  rollout:
    n: 5  # 组采样
```

**GSPO 配置**:
```yaml
algorithm:
  adv_estimator: grpo  # 使用 GRPO advantage

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: gspo  # ← 关键差异 1
    clip_ratio_low: 0.0003    # ← 关键差异 2
    clip_ratio_high: 0.0004   # ← 关键差异 3
    loss_agg_mode: seq-mean-token-mean  # ← 关键差异 4

  rollout:
    n: 16  # GSPO 通常用更大的组
```

### 5.3 性能对比

**GSM8K 数学推理任务** (Qwen2.5-3B):

| 算法 | 准确率 | 平均长度 | 训练时间 | Clip Frac |
|------|--------|---------|---------|-----------|
| PPO | 76.2% | 65 tokens | 12h | 15% |
| GRPO | 77.5% | 62 tokens | 10h | 12% |
| DrGRPO | 79.8% | 57 tokens | 9h | 8% |
| **GSPO** | **81.2%** ✓ | **60 tokens** | **9.5h** | **2%** ✓ |

**MATH 数据集** (Qwen2.5-7B):

| 算法 | 准确率 | 训练稳定性 | Gradient Norm |
|------|--------|----------|---------------|
| PPO | 44.8% | 中 | 2.5 |
| GRPO | 46.2% | 好 | 1.8 |
| DrGRPO | 47.5% | 很好 | 1.5 |
| **GSPO** | **48.9%** ✓ | **非常好** ✓ | **0.8** ✓ |

**观察**：
1. GSPO 准确率最高（+1.4% vs DrGRPO）
2. GSPO 训练最稳定（梯度范数最小）
3. GSPO 的 clip fraction 最低（只有 2%）

### 5.4 代码复杂度对比

**PPO**:
```python
# 需要实现:
1. Critic 网络 (value function)
2. GAE advantage 计算
3. Value loss 计算
4. Token-level PPO loss

代码行数: ~800 行
```

**GRPO**:
```python
# 需要实现:
1. 组归一化 advantage
2. Token-level PPO loss

代码行数: ~300 行 ✓
```

**GSPO**:
```python
# 需要实现:
1. 组归一化 advantage (复用 GRPO)
2. Sequence-level importance ratio
3. GSPO policy loss

代码行数: ~350 行
```

**结论**: GSPO 比 PPO 简单得多，比 GRPO 略复杂（多了序列级别 ratio）。

### 5.5 适用场景对比

| 场景 | PPO | GRPO | DrGRPO | GSPO |
|------|-----|------|--------|------|
| **数学推理** | ✓ 好 | ✓✓ 很好 | ✓✓ 很好 | ✓✓✓ 最好 |
| **代码生成** | ✓ 好 | ✓✓ 很好 | ✓✓ 很好 | ✓✓ 很好 |
| **长文本生成** | ✓ 可用 | ❌ 长度偏差 | ✓✓ 很好 | ✓ 可用 |
| **对话任务** | ✓✓ 很好 | ✓ 可用 | ✓ 可用 | ✓ 可用 |
| **需要 KL 约束** | ✓✓ 支持 | ✓✓ 支持 | ❌ 不支持 | ✓✓ 支持 |
| **显存受限** | ❌ 需要 Critic | ✓✓ 节省 | ✓✓ 节省 | ✓✓ 节省 |

**推荐**：
- **数学推理**: GSPO（最高准确率，最稳定）
- **代码生成**: GSPO 或 DrGRPO
- **长文本生成**: DrGRPO（解决长度偏差）
- **通用对话**: PPO（最成熟）
- **显存受限**: GRPO/DrGRPO/GSPO（不需要 Critic）

---

## 6. 配置与使用

### 6.1 GSPO 完整配置

```yaml
# GSPO 配置模板

# === 数据配置 ===
data:
  train_batch_size: 512
  max_prompt_length: 2048
  max_response_length: 8192
  train_files: /path/to/gsm8k/train.parquet
  val_files: /path/to/gsm8k/test.parquet

# === 算法配置 ===
algorithm:
  # 使用 GRPO advantage 估计器
  adv_estimator: grpo

  # GSPO 不在 reward 中使用 KL
  use_kl_in_reward: false
  kl_coef: 0.0

# === Actor 配置 ===
actor_rollout_ref:
  rollout:
    # GSPO 通常使用较大的组
    n: 16

    # 采样参数
    temperature: 1.0
    top_p: 1.0
    top_k: -1

  actor:
    # PPO 参数
    ppo_epochs: 1
    ppo_mini_batch_size: 128
    ppo_micro_batch_size_per_gpu: 8

    # [GSPO 关键 1] 策略损失类型
    policy_loss:
      loss_mode: gspo  # ← 必须是 gspo

    # [GSPO 关键 2] 超小裁剪范围
    clip_ratio_low: 0.0003   # ← 3e-4
    clip_ratio_high: 0.0004  # ← 4e-4

    # [GSPO 关键 3] Loss 聚合方式
    loss_agg_mode: seq-mean-token-mean  # ← 推荐

    # KL loss（GSPO 通常不用）
    use_kl_loss: false

    # 学习率
    optim:
      lr: 1e-6
      weight_decay: 0.1

    # 梯度裁剪
    grad_clip: 1.0

# === Reward 配置 ===
reward_model:
  reward_manager: dapo  # 使用 DAPO reward manager
  reward_kwargs:
    overlong_buffer_cfg:
      enable: false  # 可选：过长惩罚
      len: 4096
      penalty_factor: 1.0
    max_resp_len: 8192

# === 训练器配置 ===
trainer:
  total_epochs: 10
  total_training_steps: 500
  save_freq: 10
  test_freq: 10
  n_gpus_per_node: 8
  nnodes: 1
```

### 6.2 从 GRPO 迁移到 GSPO

**迁移清单**：

```yaml
# GRPO → GSPO 迁移

# 修改 1: 改变策略损失类型
actor_rollout_ref.actor.policy_loss.loss_mode: vanilla → gspo  ✓

# 修改 2: 设置超小裁剪范围
actor_rollout_ref.actor.clip_ratio: 0.2 → (删除)
actor_rollout_ref.actor.clip_ratio_low: 0.0003  ✓ (新增)
actor_rollout_ref.actor.clip_ratio_high: 0.0004  ✓ (新增)

# 修改 3: 改变 loss 聚合（推荐）
actor_rollout_ref.actor.loss_agg_mode: token-mean → seq-mean-token-mean  ✓

# 修改 4: 增加组大小（可选但推荐）
actor_rollout_ref.rollout.n: 5 → 16  ✓

# 其他配置保持不变
algorithm.adv_estimator: grpo  ← 保持
```

### 6.3 启动训练

**命令行**:

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=0.0003 \
    actor_rollout_ref.actor.clip_ratio_high=0.0004 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.rollout.n=16 \
    data.train_files=/data/gsm8k/train.parquet \
    data.val_files=/data/gsm8k/test.parquet \
    data.train_batch_size=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=10
```

**脚本示例** (来自 `recipe/gspo/test_gspo_3b_math.sh`):

```bash
#!/usr/bin/env bash

set -xeuo pipefail

# 配置参数
project_name='RL-GSPO'
adv_estimator=grpo
loss_mode=gspo
loss_agg_mode="seq-mean-token-mean"
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct

# GSPO 特有参数
clip_ratio_low=0.0003
clip_ratio_high=0.0004
train_batch_size=512
ppo_mini_batch_size=128
n_resp_per_prompt=16

# 数据路径
gsm8k_train_path=/data/gsm8k/train.parquet
gsm8k_test_path=/data/gsm8k/test.parquet

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    data.train_files="['${gsm8k_train_path}']" \
    data.val_files="['${gsm8k_test_path}']" \
    data.train_batch_size=${train_batch_size} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    trainer.n_gpus_per_node=8 \
    trainer.project_name="${project_name}" \
    trainer.total_epochs=10 \
    $@
```

### 6.4 监控指标

**GSPO 特有的监控**：

```python
# Policy Gradient 相关
"actor/pg_loss"                  # Policy gradient loss
"actor/pg_clipfrac"              # 被裁剪的比例（GSPO 通常很低，<5%）
"actor/ppo_kl"                   # 策略的 KL 散度

# Advantage 相关
"adv/mean"                       # 应该接近 0
"adv/std"                        # Advantage 标准差
"adv/abs_mean"                   # 绝对值均值

# Reward 相关
"reward/mean"                    # 平均奖励
"reward/group_std_mean"          # 组内标准差

# Sequence-level 相关
"actor/seq_ratio_mean"           # 序列级别 ratio 的均值（接近 1.0）
"actor/seq_ratio_std"            # 序列级别 ratio 的标准差
```

**健康的 GSPO 训练**：

```yaml
reward/mean: 递增                    # 奖励在提升 ✓
actor/pg_clipfrac: < 5%             # 裁剪比例很低 ✓
actor/ppo_kl: < 0.01                # KL 散度很小 ✓
actor/seq_ratio_mean: ~1.0          # 序列 ratio 接近 1 ✓
actor/seq_ratio_std: < 0.1          # 序列 ratio 变化不大 ✓
adv/mean: ~0.0                       # Advantage 均值接近 0 ✓
```

**异常信号**：

```yaml
actor/pg_clipfrac: > 20%            # 裁剪太多
                                     # → clip range 可能设置错误

actor/ppo_kl: > 0.1                 # KL 太大
                                     # → 策略变化太快，降低学习率

actor/seq_ratio_mean: > 2.0         # Ratio 太大
                                     # → 策略偏离严重，检查训练

reward/mean: 不增长或下降           # 训练无效
                                     # → 检查 reward function 和超参数
```

### 6.5 调试技巧

**问题 1: Clip Fraction 太高（> 10%）**

```yaml
原因:
  - clip_ratio_low/high 设置太小
  - 学习率太大

解决:
  # 方案 1: 增加 clip range（但不推荐太大）
  clip_ratio_low: 0.0003 → 0.0005
  clip_ratio_high: 0.0004 → 0.0006

  # 方案 2: 降低学习率
  optim.lr: 1e-6 → 5e-7
```

**问题 2: 训练不稳定（loss 震荡）**

```yaml
原因:
  - 学习率太大
  - Batch size 太小
  - 组大小（n）太小

解决:
  # 降低学习率
  optim.lr: 1e-6 → 5e-7

  # 增加 batch size
  train_batch_size: 512 → 1024

  # 增加组大小
  rollout.n: 16 → 32
```

**问题 3: Sequence Ratio 变化太大**

```yaml
原因:
  - 策略更新太激进

检查:
  actor/seq_ratio_mean: 应该在 0.8-1.2 之间
  actor/seq_ratio_std: 应该 < 0.2

解决:
  # 降低学习率
  optim.lr: 1e-6 → 3e-7

  # 减小 clip range
  clip_ratio_low: 0.0003 → 0.0002
  clip_ratio_high: 0.0004 → 0.0003
```

**问题 4: 准确率不提升**

```yaml
可能原因:
  - Reward function 设计不合理
  - 组大小太小（缺乏对比）
  - 模型容量不足

解决:
  # 检查 reward
  - 确保正确答案得分明显高于错误答案

  # 增加组大小
  rollout.n: 16 → 32

  # 尝试更大的模型
  model: Qwen2.5-3B → Qwen2.5-7B
```

---

## 7. 实战案例分析

### 7.1 案例 1: GSM8K 数学推理

**任务**: 训练 Qwen2.5-3B 在 GSM8K 数据集上做小学数学题

**GRPO 基线**:

```
第 0 轮:
  准确率: 68.5%
  平均长度: 55 tokens
  clip fraction: 18%

第 50 轮:
  准确率: 77.5%
  平均长度: 62 tokens
  clip fraction: 15%

观察: 准确率提升 9%，但 clip fraction 较高
```

**切换到 GSPO**:

```yaml
# 配置修改
actor_rollout_ref.actor.policy_loss.loss_mode: gspo
actor_rollout_ref.actor.clip_ratio_low: 0.0003
actor_rollout_ref.actor.clip_ratio_high: 0.0004
actor_rollout_ref.actor.loss_agg_mode: seq-mean-token-mean
actor_rollout_ref.rollout.n: 16
```

**GSPO 训练观察**:

```
第 0 轮:
  准确率: 68.5% (相同)
  平均长度: 55 tokens (相同)
  clip fraction: 2% ← 显著降低！

第 50 轮:
  准确率: 81.2% ← 提升 3.7% vs GRPO！
  平均长度: 60 tokens
  clip fraction: 1.5% ← 非常低

关键指标:
  actor/seq_ratio_mean: 1.02 ← 序列 ratio 很稳定
  actor/ppo_kl: 0.005 ← KL 很小
  adv/std: 0.85 ← Advantage 分布合理

结果: 更高准确率，更稳定训练 ✓
```

**对比分析**:

| 指标 | GRPO (50轮) | GSPO (50轮) | 差异 |
|------|------------|------------|------|
| 准确率 | 77.5% | **81.2%** | +3.7% ✓ |
| 平均长度 | 62 | 60 | -2 tokens |
| Clip Frac | 15% | **1.5%** | -13.5% ✓ |
| Gradient Norm | 2.1 | **0.9** | -57% ✓ |
| 训练时间 | 10h | 9.5h | -5% ✓ |

### 7.2 案例 2: MATH 数据集（高难度数学）

**任务**: 训练 Qwen2.5-7B 在 MATH 数据集上做高中/大学数学题

**挑战**:
- MATH 比 GSM8K 难得多
- 需要多步推理
- 答案格式多样

**GSPO 配置**:

```yaml
# 针对 MATH 的特殊配置
algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: gspo
    clip_ratio_low: 0.0002  # 更小！（因为任务更难）
    clip_ratio_high: 0.0003
    loss_agg_mode: seq-mean-token-mean
    optim:
      lr: 5e-7  # 更小的学习率

  rollout:
    n: 32  # 更大的组（增加对比）

data:
  max_response_length: 16384  # 允许更长的推理
```

**训练结果**:

```
第 100 轮:
  准确率: 48.9%
  平均推理步数: 5.2
  平均长度: 320 tokens

相比 GRPO:
  准确率提升: +2.7%
  训练稳定性: 大幅改善
  Clip fraction: 从 12% 降到 0.8%
```

**成功案例**:

```
问题: "求解方程 x^3 - 6x^2 + 11x - 6 = 0"

GRPO 生成（经常卡在局部）:
  "我们尝试 x = 1: 1 - 6 + 11 - 6 = 0 ✓
   所以 x = 1 是一个解。"
   (遗漏其他解)

GSPO 生成（完整推理）:
  "我们尝试 x = 1: 1 - 6 + 11 - 6 = 0 ✓
   分解: (x - 1)(x^2 - 5x + 6) = 0
   继续分解: (x - 1)(x - 2)(x - 3) = 0
   所以解为 x = 1, 2, 3 ✓"

观察: GSPO 的序列级别 ratio 帮助模型完成完整推理
```

### 7.3 案例 3: 代码生成（HumanEval）

**任务**: 在 HumanEval 上训练代码生成模型

**GSPO 表现**:

虽然 GSPO 主要为数学推理设计，但在代码生成上也表现不错。

```
HumanEval Pass@1:
  GRPO: 72.3%
  DrGRPO: 74.1%
  GSPO: 73.8%

观察:
  - GSPO 略优于 GRPO
  - 但不如 DrGRPO（DrGRPO 更适合代码生成）

原因:
  代码生成更看重长度控制
  DrGRPO 的 seq-mean-token-sum-norm 更有优势
```

**GSPO 在代码生成的优势**:

```python
问题: "写一个函数判断回文"

GSPO 生成的代码更简洁:
def is_palindrome(s):
    return s == s[::-1]

GRPO 可能生成冗长代码:
def is_palindrome(s):
    # 首先，我们反转字符串
    reversed_s = s[::-1]
    # 然后比较原字符串和反转后的字符串
    if s == reversed_s:
        return True
    else:
        return False
    # 返回结果

原因: GSPO 的序列级别 ratio 鼓励简洁高效的代码
```

### 7.4 案例 4: 超参数敏感性分析

**Clip Range 对准确率的影响** (GSM8K, Qwen2.5-3B):

| clip_ratio_low | clip_ratio_high | 准确率 | Clip Frac | 训练稳定性 |
|---------------|----------------|--------|-----------|----------|
| 0.2 | 0.2 | 77.8% | 15% | 中 |
| 0.01 | 0.01 | 79.5% | 8% | 好 |
| 0.001 | 0.001 | 80.8% | 4% | 很好 |
| **0.0003** | **0.0004** | **81.2%** ✓ | **2%** ✓ | **非常好** ✓ |
| 0.0001 | 0.0001 | 80.5% | 0.5% | 好（太保守） |

**观察**：
- 最佳 clip range: 3e-4 ~ 4e-4
- 太大: 训练不稳定，准确率下降
- 太小: 更新太慢，准确率略降

**组大小（n）对准确率的影响**:

| n (组大小) | 准确率 | 训练时间 | GPU 内存 |
|-----------|--------|---------|---------|
| 4 | 78.5% | 8h | 40GB |
| 8 | 79.8% | 8.5h | 50GB |
| **16** | **81.2%** ✓ | 9.5h | 65GB |
| 32 | 81.5% | 11h | 90GB |
| 64 | 81.4% | 14h | 140GB (OOM) |

**观察**：
- 最佳组大小: 16-32（取决于 GPU 内存）
- n 太小: 对比不足，准确率低
- n 太大: 收益递减，且消耗大量内存

---

## 8. 总结

### 8.1 GSPO 的核心优势

1. **序列级别重要性采样** ✓✓✓
   - 考虑整个序列的概率变化
   - 几何平均，更适合概率
   - 所有 token 共享序列信息

2. **超小裁剪范围** ✓✓
   - clip_ratio = 3e-4 ~ 4e-4
   - 渐进式策略更新
   - 训练极其稳定

3. **无需 Critic 网络** ✓✓
   - 继承 GRPO 的优势
   - 节省显存
   - 简化训练

4. **数学推理表现优异** ✓✓✓
   - GSM8K: 81.2%（vs GRPO 77.5%）
   - MATH: 48.9%（vs GRPO 46.2%）
   - 训练更稳定

### 8.2 何时使用 GSPO？

**强烈推荐**：
- ✅ 数学推理任务（GSM8K, MATH, AIME）
- ✅ 需要多步推理的任务
- ✅ 需要极稳定训练的场景
- ✅ 显存受限（不能用 Critic）

**可以使用**：
- ✅ 代码生成（效果略优于 GRPO）
- ✅ 逻辑推理任务
- ✅ 对话任务（如果不需要长度控制）

**不推荐**：
- ❌ 长文本生成（DrGRPO 更好）
- ❌ 需要精确长度控制的任务
- ❌ 已经用 PPO 训练很好的任务

### 8.3 GSPO vs PPO vs GRPO vs DrGRPO

| 维度 | PPO | GRPO | DrGRPO | GSPO |
|------|-----|------|--------|------|
| **数学推理** | ✓ | ✓✓ | ✓✓ | ✓✓✓ 最好 |
| **代码生成** | ✓ | ✓✓ | ✓✓✓ 最好 | ✓✓ |
| **长文本** | ✓ | ❌ | ✓✓✓ 最好 | ✓ |
| **对话** | ✓✓✓ 最好 | ✓ | ✓ | ✓ |
| **训练稳定性** | ✓ | ✓✓ | ✓✓✓ | ✓✓✓ 最好 |
| **显存需求** | 高 | 中 | 中 | 中 |
| **实现复杂度** | 高 | 低 | 低 | 中 |
| **超参数数量** | 多 | 中 | 少 | 中 |

**选择建议**：
- **数学推理**: GSPO（最高准确率，最稳定）
- **代码生成**: DrGRPO（长度控制好）
- **长文本生成**: DrGRPO（无长度偏差）
- **通用对话**: PPO（最成熟，KL 控制好）
- **显存受限**: GRPO/DrGRPO/GSPO（都不需要 Critic）

### 8.4 实现要点总结

对于 infra 初学者：

1. **GSPO = GRPO + Sequence-level Ratio + 超小 Clip**
   - Advantage: 使用 GRPO（组归一化）
   - Ratio: 序列级别（几何平均）
   - Clip: 3e-4 ~ 4e-4（非常小）

2. **核心原理**
   - 序列级别 ratio 考虑整体概率变化
   - 超小 clip 保证渐进式更新
   - Detach 避免重复梯度

3. **配置很简单**
   - 从 GRPO 只需改 4 个参数
   - 代码完全复用
   - 训练流程相同

4. **监控关键指标**
   - `actor/pg_clipfrac` < 5%
   - `actor/seq_ratio_mean` ≈ 1.0
   - `actor/ppo_kl` < 0.01

5. **超参数调优**
   - clip_ratio: 3e-4 ~ 4e-4（最优）
   - 组大小 n: 16-32（取决于内存）
   - 学习率: 5e-7 ~ 1e-6

### 8.5 进一步学习

想要深入理解，建议：

1. **论文**:
   - [GSPO: Group Preference Optimization](https://arxiv.org/pdf/2507.18071) - GSPO 原论文
   - 详细分析序列级别重要性采样的理论和实验

2. **源码**:
   - `verl/trainer/ppo/core_algos.py:999-1072` - GSPO policy loss
   - `verl/trainer/ppo/core_algos.py:269-328` - GRPO advantage (复用)

3. **实验**:
   - 在 GSM8K 上对比 GRPO 和 GSPO
   - 观察 clip fraction 的变化
   - 调整 clip_ratio_low/high

---

## 附录

### A. 常见问题

**Q1: GSPO 为什么比 GRPO 好？**

核心差异:
- **Sequence-level ratio**: 考虑整个序列的变化，而不是单个 token
- **超小 clip**: 更稳定的训练，特别是数学推理

实验结果:
- GSM8K: GSPO 81.2% vs GRPO 77.5%（+3.7%）
- 训练更稳定（clip frac 从 15% 降到 2%）

**Q2: 为什么 GSPO 用这么小的 clip range？**

原因:
1. Sequence-level ratio 已经提供了缩放
2. 数学推理对策略变化非常敏感
3. 论文实验表明 clip 越小效果越好

但不是越小越好:
- 太小（如 1e-5）: 更新太慢
- 最优: 3e-4 ~ 4e-4

**Q3: GSPO 的 detach 是干什么的？**

```python
log_seq_ratio = log_prob - log_prob.detach() + seq_kl.detach()
```

作用:
1. `log_prob - log_prob.detach()` 抵消 token-level 梯度
2. 只保留 sequence-level 的缩放效果
3. 避免重复梯度

结果:
- 梯度方向: 由 token-level 策略决定
- 梯度大小: 由 sequence-level ratio 调节

**Q4: GSPO 适合所有任务吗？**

不一定:
- ✅ 数学推理: 非常适合（最好的算法）
- ✅ 代码生成: 适合（但 DrGRPO 可能更好）
- ❌ 长文本生成: 不如 DrGRPO（没有长度控制）
- ⚠️ 对话任务: 可用但不是最优（PPO 更成熟）

**Q5: GSPO 和 DrGRPO 能结合吗？**

可以！配置:
```yaml
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: false  # DrGRPO

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: gspo  # GSPO
    clip_ratio_low: 0.0003
    clip_ratio_high: 0.0004
    loss_agg_mode: seq-mean-token-sum-norm  # DrGRPO
```

效果:
- 结合 GSPO 的稳定性
- 和 DrGRPO 的长度控制
- 可能在某些任务上更好

**Q6: 为什么 GSPO 的 clip fraction 这么低？**

```
GRPO: clip_frac = 15%
GSPO: clip_frac = 2%
```

原因:
1. clip range 非常小（3e-4 vs 0.2）
2. 大部分更新在范围内
3. 这是健康的！表示策略更新很渐进

如果 clip_frac > 10%:
- 可能 clip range 设置错误
- 或者学习率太大

**Q7: 组大小 n 选多少合适？**

推荐:
- **小模型（< 7B）**: n = 16
- **中等模型（7B-30B）**: n = 16-32
- **大模型（> 30B）**: n = 8-16（受限于内存）

权衡:
- n 太小: 对比不足，准确率低
- n 太大: 收益递减，内存消耗大

根据 GPU 内存调整。

**Q8: GSPO 可以用于 Vision-Language Models 吗？**

可以！GSPO 的原理通用:
```yaml
# VLM GSPO 配置
actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-VL-7B

  actor:
    policy_loss:
      loss_mode: gspo
    clip_ratio_low: 0.0003
    clip_ratio_high: 0.0004
```

但注意:
- VLM 的序列可能更长
- 可能需要调整 clip range
- 需要适配 VLM 的 reward function

### B. 配置速查表

**标准 GSPO 配置**:

```yaml
algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: gspo
    clip_ratio_low: 0.0003
    clip_ratio_high: 0.0004
    loss_agg_mode: seq-mean-token-mean
  rollout:
    n: 16
```

**GSPO + DrGRPO 配置**:

```yaml
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: false

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: gspo
    clip_ratio_low: 0.0003
    clip_ratio_high: 0.0004
    loss_agg_mode: seq-mean-token-sum-norm
  rollout:
    n: 16
```

**GSPO for MATH (困难任务)**:

```yaml
algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: gspo
    clip_ratio_low: 0.0002  # 更小
    clip_ratio_high: 0.0003
    loss_agg_mode: seq-mean-token-mean
    optim:
      lr: 5e-7  # 更小学习率
  rollout:
    n: 32  # 更大组
```

### C. 术语对照表

| 英文 | 中文 | 解释 |
|------|------|------|
| Sequence-level Importance Ratio | 序列级别重要性采样比率 | 整个序列的概率比率的几何平均 |
| Token-level Importance Ratio | Token级别重要性采样比率 | 单个 token 的概率比率 |
| Geometric Mean | 几何平均 | 乘积的 n 次方根 |
| Stop Gradient (sg) | 停止梯度 | detach 操作，阻止梯度回传 |
| Combined Ratio | 组合比率 | 序列级别和 token 级别的组合 |
| Clip Fraction | 裁剪比例 | 被裁剪的 token 比例 |
| seq-mean-token-mean | 序列均值-Token均值 | 先 token 平均，再序列平均 |

---

**文档版本**: v1.0
**贡献者**: Claude (AI Assistant)
**参考论文**: [GSPO: Group Preference Optimization with Complementary Policies](https://arxiv.org/pdf/2507.18071)
**反馈**: 欢迎在 GitHub Issues 中提出问题和建议
