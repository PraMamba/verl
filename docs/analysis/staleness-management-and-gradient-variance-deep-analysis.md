# verl Staleness 管理深度源码分析：长 CoT 下的梯度方差控制

> **分析范围**: verl 中 TIS 裁剪、OPSM、拒绝采样、多种 RL 损失函数在长序列（32K token）off-policy 场景下的梯度方差控制能力
>
> **分析日期**: 2026-03-20
>
> **核心源码路径**:
> - `verl/trainer/ppo/core_algos.py` — 所有 RL 损失函数与优势估计器
> - `verl/trainer/ppo/rollout_corr_helper.py` — Rollout Correction 框架（TIS + 拒绝采样 + OPSM 指标）
> - `verl/workers/actor/dp_actor.py` — Actor 训练循环、梯度裁剪
> - `verl/trainer/config/algorithm.py` — 算法与 Rollout Correction 配置
> - `verl/experimental/fully_async_policy/` — 全异步架构的 staleness 处理

---

## 第一部分：问题建模 — 长 CoT 下的 Policy Lag 困境

### 1.1 核心矛盾

在基于长 CoT（如 32K token）的数学推理任务中：

```
时间轴:
t=0 ─────────────── t=T_gen (生成耗时长) ──── t=T_train
│                        │                        │
Policy π_rollout         │                 Policy π_θ (已更新多步)
生成 rollout             │                 使用 rollout 数据训练
                         │
                   Policy 已漂移 Δ ∝ T_gen
```

**问题本质**：生成耗时越长 → 训练策略与生成策略的漂移越大 → 重要性采样比率（IS ratio）的方差随序列长度指数增长。

**数学推导**：对于长度 T 的序列，序列级 IS 比率为：

```
ρ_seq = ∏_{t=1}^{T} π_θ(a_t|s_t) / π_old(a_t|s_t) = exp(∑_{t=1}^{T} log_ratio_t)
```

假设每个 token 的 log ratio 方差为 σ²，则：
- `Var[log(ρ_seq)] = T · σ²` — 方差随序列长度 **线性增长**
- `Var[ρ_seq]` — 由于指数变换，方差随 T **指数增长**

对于 T=32K，即使每个 token 的 log ratio 标准差仅为 0.01，`Σlog_ratio` 的标准差已达 `0.01 × √32000 ≈ 1.79`，`exp(1.79) ≈ 6.0`。

---

## 第二部分：verl 的多层防御体系

verl 针对 off-policy staleness 和梯度方差提供了 **五层防御**，从数学保证到工程兜底逐层递进。

### 2.1 第一层：Per-Token IS 比率 + PPO Clip（核心防线）

#### 2.1.1 标准 PPO 损失（Vanilla）

**`core_algos.py:1165-1256`** — `compute_policy_loss_vanilla`

```python
# 1. 计算 per-token 重要性比率
negative_approx_kl = log_prob - old_log_prob               # (batch, seq_len)
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
ratio = torch.exp(negative_approx_kl)                      # per-token!

# 2. PPO 双重裁剪
pg_losses1 = -advantages * ratio                            # 未裁剪
pg_losses2 = -advantages * torch.clamp(                     # 裁剪到 [1-ε_low, 1+ε_high]
    ratio, 1 - cliprange_low, 1 + cliprange_high)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)     # 标准 PPO clip

# 3. Dual-clip 下界（当 advantage < 0 时）
pg_losses3 = -advantages * clip_ratio_c                     # clip_ratio_c 默认 3.0
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

# 4. 乘以 Rollout Correction IS 权重（如果有）
if rollout_is_weights is not None:
    pg_losses = pg_losses * rollout_is_weights
```

**关键设计决策**：verl 默认在 **per-token 粒度** 计算 IS ratio 并裁剪。

**对 32K 长序列的影响**：

| 维度 | 分析 |
|------|------|
| 单 token 方差 | **有界**：`ratio ∈ [1-ε, 1+ε]`（默认 ε=0.2），单 token 方差 ≤ 0.04 |
| 序列级效应 | 32K token 各自独立裁剪，不存在"乘积爆炸"问题 |
| 梯度贡献 | 单序列贡献 = `Σ_{t=1}^{T} clip(ratio_t) × A_t`，**线性于 T** |
| 潜在问题 | 使用 `token-mean` 聚合时，32K 序列对梯度的贡献是 200 token 序列的 **160 倍** |

**结论**：Per-token PPO clip 是处理长序列 off-policy 问题的 **最佳默认选择**。它将每个 token 的方差严格限制在裁剪范围内，避免了序列级乘积方差爆炸。

#### 2.1.2 非对称裁剪（DAPO 风格）

`clip_ratio_low` 和 `clip_ratio_high` 可独立配置（`core_algos.py:1201-1202`），支持 DAPO 论文中的非对称信赖域：

```python
clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
```

### 2.2 第二层：Rollout Correction 框架（TIS + 拒绝采样）

#### 2.2.1 TIS（截断重要性采样）权重

**`rollout_corr_helper.py:481-598`** — `compute_rollout_correction_weights`

处理 **两种粒度** 的 IS 权重：

**Token-level TIS**（`rollout_is="token"`，`rollout_corr_helper.py:530-534`）:

```python
log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)  # SAFETY_BOUND=20.0
rollout_is_weights = torch.exp(log_ratio_safe)     # per-token 权重
```

**Sequence-level TIS**（`rollout_is="sequence"`，`rollout_corr_helper.py:536-544`）:

```python
log_ratio_sum = verl_F.masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(-1)  # Σ log_ratio
log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)
```

随后截断（`line 562`）和 detach（`line 566`）：

```python
rollout_is_weights = rollout_is_weights.clamp(max=rollout_is_threshold)  # 默认 2.0
rollout_is_weights = rollout_is_weights.detach()    # 阻止梯度流 — IS 改变测度，不改变目标
```

**32K 长序列下的 Sequence-level TIS 退化分析**：

```
假设: 每 token log_ratio ~ N(0, 0.01²)
T = 32000 tokens

Σ log_ratio ~ N(0, 32000 × 0.01²) = N(0, 3.2)
std(Σ log_ratio) = √3.2 ≈ 1.79

P(|Σ log_ratio| > 0.693) ≈ P(|Z| > 0.387) ≈ 70%
（0.693 = ln(2), 即 exp > 2.0 的截断阈值）
```

**结论**：对于 32K 序列，sequence-level IS 权重在 **~70% 的情况下被截断到阈值 2.0**，实质上退化为二元变量（2.0 或接近 1.0）。对长序列而言，sequence-level TIS 提供的信息量极为有限。

**推荐**：长 CoT 场景应使用 `rollout_is="token"` 而非 `"sequence"`。

#### 2.2.2 拒绝采样（Rejection Sampling）— 硬信赖域

**`rollout_corr_helper.py:156-372`** — `compute_rollout_rejection_mask`

支持三族散度估计器（K1/K2/K3），可在 token 级或序列级应用：

| 模式 | 散度公式 | 特性 |
|------|---------|------|
| `token_k1` | `-log(r)` | 双向比率界，需上下限阈值 |
| `token_k2` | `0.5 × (log r)²` | 单侧上界，对中等偏离更稳定 |
| `token_k3` | `exp(log r) - 1 - log r` | χ² 散度代理，捕获二阶行为 |
| `seq_sum_k*` | `Σ_t k*(token_t)` | 序列级总偏离 |
| `seq_mean_k*` | `(1/T) Σ_t k*(token_t)` | 序列级平均偏离 |
| `seq_max_k*` | `max_t k*(token_t)` | 序列中最大偏离 token |

**对 32K 长序列的关键区别**：

- `seq_sum_k2`：偏离量随 T 线性增长 → 长序列更容易被拒绝 → **有效但可能过度保守**
- `seq_mean_k2`：对 T 归一化 → **与序列长度无关**，更公平
- `seq_max_k2`：只看最偏离的 token → **对长序列更宽容**（32K token 中最大偏离 ≈ 几个标准差）

**推荐**：长 CoT 场景使用 `seq_mean_k2` 或 `seq_mean_k3` 作为拒绝标准，避免 `seq_sum` 对长序列的系统性偏见。

### 2.3 第三层：长度感知的损失聚合

**`core_algos.py:1025-1086`** — `agg_loss`

这是控制 **单条序列对梯度贡献权重** 的关键函数：

```python
if loss_agg_mode == "token-mean":
    # 所有 token 平等 → 32K 序列贡献 = 160 × 200-token 序列
    loss = masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size

elif loss_agg_mode == "seq-mean-token-mean":
    # 先序列内 token 平均，再序列间平均 → 每条序列贡献相等
    seq_losses = sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
    loss = masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size

elif loss_agg_mode == "seq-mean-token-sum-norm":
    # 序列内 token 求和后除以归一化因子 → 可配置的长度归一化
    seq_losses = sum(loss_mat * loss_mask, dim=-1)
    loss = masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size
    loss /= loss_scale_factor   # 默认 = response_length
```

**32K 场景影响**：

| 模式 | 32K 序列 vs 200 序列的梯度贡献比 | 推荐场景 |
|------|------|------|
| `token-mean`（默认） | **160:1** | 短序列、序列长度均匀 |
| `seq-mean-token-mean` | **1:1** | **长 CoT 首选**，序列间公平 |
| `seq-mean-token-sum-norm` | 取决于 `loss_scale_factor` | 可调节的折中 |

### 2.4 第四层：专用长序列 Loss 函数

#### 2.4.1 GSPO — 几何均值序列级比率

**`core_algos.py:1425-1498`**，论文 arXiv:2507.18071

GSPO 的核心创新：**除以序列长度来归一化 IS 比率**，从而阻止方差随 T 的增长：

```python
# 序列级重要性比率 s_i(θ) = (π_θ/π_old)^(1/|y_i|)
seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths  # ÷ T ！

# 组合 token-level + sequence-level：
# s_{i,t}(θ) = sg[s_i(θ)] · π_θ(y_{i,t}) / sg[π_θ(y_{i,t})]
log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
seq_importance_ratio = torch.exp(log_seq_importance_ratio)
```

**方差分析**：

```
standard PPO:   Var[ratio_t] ∈ [0, ε²]     for each token — OK
seq-level IS:   Var[ρ_seq] ∝ exp(T·σ²) - 1  — 指数爆炸 ❌
GSPO:           Var[s_i] ∝ exp(σ²) - 1       — 与 T 无关 ✓
```

GSPO 通过几何均值将序列级比率的方差从 O(exp(T)) 降至 O(1)。

**注意**：GSPO 默认使用 `loss_agg_mode="seq-mean-token-mean"`（`line 1431`），进一步确保序列间公平。

#### 2.4.2 GMPO — 先 clip 再几何均值

**`core_algos.py:1808-1890`**，论文 arXiv:2507.20673

GMPO 采用不同的策略：**先在 token 级 clip，再取几何均值**：

```python
# Step 1: Token-level clipping in log space
sgn_advantage = torch.sign(advantages)
negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
negative_approx_kl_min = torch.min(
    sgn_advantage * negative_approx_kl,
    sgn_advantage * negative_approx_kl_clamp
)
negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

# Step 2: Geometric mean of clipped token ratios
response_mask_sum = response_mask.sum(dim=-1)
ratio = torch.exp(
    (negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
)

# Step 3: Sequence-level loss
advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
pg_losses = -advantage * ratio
```

**方差控制**：先 clip 确保每个 `negative_approx_kl_min ∈ [-ε_low, ε_high]`，然后几何均值（除以 T）将累积效应归一化。双重保险。

#### 2.4.3 对比三种方案在 32K 下的方差特性

```
            ┌────────────────────────────────────────────┐
            │   Var[每个 token 对梯度的贡献]  vs  T       │
            │                                            │
  Var ↑     │   seq-level IS                             │
            │   (product ratio)     /                    │
            │                     /   ← 指数增长          │
            │                   /                        │
            │                 /                          │
            │               /                            │
            │  ──────────────── token-level PPO clip     │
            │  ============== GSPO / GMPO               │
            │                                            │
            └──────────────── T (seq_len) ──────────────→│
```

### 2.5 第五层：Optimal Token Baseline — 位置感知方差缩减

**`core_algos.py:758-873`** — `compute_optimal_token_baseline_advantage`

这是 verl 中 **最精密的方差缩减机制**，为每个时间步计算独立的最优 baseline：

```python
# 理论: B_t* = E[G_t × W_t] / E[W_t]
# 其中 W_t = Σ_{j=1}^t ||s_j||² 是累积路径方差代理

# Step 1: 计算每个 timestep 的方差代理
pi_t = torch.exp(old_log_probs)
w_per_timestep = 1 - 2 * pi_t + sum_pi_squared     # ||s_j||² 与词汇表分布相关

# Step 2: 如果有 IS 权重，用 ρ̄² 缩放
if rollout_is_weights is not None:
    w_per_timestep = w_per_timestep * (rollout_is_weights**2)  # ← 感知 off-policy 程度

# Step 3: 累积路径方差
w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

# Step 4: 分组计算 per-timestep baseline
# B_t* = Σ[G_t × W_t] / Σ[W_t]  在同 prompt 组内
```

**为什么 OTB 对 32K 长序列特别有价值**：

1. **位置感知**：序列后端 token 获得不同于前端的 baseline — 后端累积方差更大，baseline 更精确
2. **IS-aware**：当提供 rollout IS 权重时，baseline 会感知 off-policy 程度并调整
3. **组内自适应**：同一 prompt 的不同响应互相对比，消除共有的方差成分
4. **对比**：GRPO/RLOO 为整条序列广播一个标量 advantage → 32K token 全部接收相同信号，无位置区分

**代价**：需要额外计算 `sum_pi_squared`（词汇表概率平方和），需设置 `calculate_sum_pi_squared=True`。

---

## 第三部分：全异步路径的 Staleness 控制

### 3.1 全异步架构的版本追踪

`verl/experimental/fully_async_policy/` 中的全异步架构通过 **param_version** 追踪 staleness：

```python
# MessageQueue（message_queue.py:67-96）
async def put_sample(self, sample, param_version):
    # 样本携带其生成时的策略版本
    self.queue.append(sample)

# FullyAsyncTrainer（fully_async_trainer.py:615-632）
stale_count = sum(1 for v in samples_param_versions
                  if self.current_param_version - v >= 1)
self.stale_samples_processed += stale_count
```

### 3.2 流量控制而非 IS 校正

**关键设计选择**：全异步路径 **不使用** 基于版本年龄的 IS 权重校正。它依赖：

1. **流量节流**（`fully_async_rollouter.py:727-749`）：

```python
async def _should_pause_generation(self):
    # 条件 1: 队列满 → 暂停
    if queue_size >= self.max_queue_size:
        return True
    # 条件 2: 陈旧样本过多 → 暂停
    if self.staleness_samples >= self.max_required_samples:
        return True
```

2. **Bypass 模式 PPO-clip**（默认配置 `bypass_mode: True`）：

在 bypass 模式下，`old_log_prob = rollout_log_prob`，PPO 比率变为 `π_θ / π_rollout`。PPO clip 本身就充当 IS 校正 — 将策略更新限制在信赖域内，无需额外 IS 权重。

3. **Partial Rollout**：权重同步时中断进行中的生成，用新策略恢复，减少 staleness 积累。

### 3.3 Staleness 阈值的数学含义

`staleness_threshold` 控制 rollouter 可以"跑在前面"多远（`fully_async_rollouter.py:205-208`）：

```python
self.max_required_samples = int(
    self.required_samples * (self.staleness_threshold + 1) * self.trigger_parameter_sync_step
)
```

当 `staleness_threshold = 0`：系统退化为同步 → 无 staleness 问题
当 `staleness_threshold > 0`：允许 rollouter 提前生成，staleness 由 PPO clip 吸收

---

## 第四部分：梯度裁剪 — 最后的工程兜底

### 4.1 全局梯度范数裁剪

**`dp_actor.py:391-422`**：

```python
def _optimizer_step(self):
    # FSDP2 路径
    grad_norm = fsdp2_clip_grad_norm_(
        self.actor_module.parameters(),
        max_norm=self.config.grad_clip    # 默认 1.0
    )

    if not torch.isfinite(grad_norm):
        # 梯度不有限 → 跳过整步更新
        self.actor_optimizer.zero_grad()
    else:
        self.actor_optimizer.step()
```

**默认 `grad_clip=1.0`** 提供了对梯度方差爆炸的 **最后兜底**。但它是粗粒度的：

- 不区分"合理的大梯度信号"和"off-policy 方差导致的异常梯度"
- 当 32K 序列频繁触发裁剪时，**有效学习率被全局降低**

### 4.2 PPO Epochs 的内步 Off-Policy 漂移

**`dp_actor.py:550-599`**：

```python
on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
# ...
if on_policy:
    old_log_prob = log_prob.detach()   # ratio ≡ 1.0，完全 on-policy
```

当 `ppo_epochs > 1` 时，同一批数据重复使用，每个 epoch 后 `log_prob` 偏离 `old_log_prob`。对于 32K 序列：
- 第 1 个 epoch 后，32K token 各自轻微偏移
- 第 2 个 epoch，偏移累积
- 序列级效应 = 32K 个独立偏移的乘积 → 可能触发大量 clipping

**建议**：长 CoT 训练考虑 `ppo_epochs=1`。

---

## 第五部分：综合评估 — TIS + OPSM 能否压制梯度方差爆炸？

### 5.1 各层防御的有效性矩阵

| 防御层 | 对 32K 长序列的效果 | 是否充分？ |
|--------|-------------------|---------|
| **Per-token PPO clip** | 单 token 方差严格有界，无乘积爆炸 | **是，核心防线** |
| **Token-level TIS** | 额外的 per-token 权重校正 | 是，作为补充 |
| **Sequence-level TIS** | 对 32K 退化为二元变量 | **否，信息量不足** |
| **拒绝采样（seq_mean_k*）** | 长度归一化后与 T 无关 | 是，但丢弃样本 |
| **拒绝采样（seq_sum_k*）** | 随 T 增大系统性拒绝长序列 | 否，过度保守 |
| **GSPO 几何均值** | 方差与 T 无关 | **是，专为此设计** |
| **GMPO clip+几何均值** | 双重保险 | **是，最稳健** |
| **Optimal Token Baseline** | 位置感知的方差缩减 | 是，最精密 |
| **`seq-mean-token-mean` 聚合** | 消除长序列梯度权重偏差 | **关键配套** |
| **梯度裁剪（grad_clip=1.0）** | 粗粒度兜底 | 必要但非充分 |

### 5.2 核心结论

**回答原问题**：*verl 仅依靠 TIS 裁剪和 OPSM，能否在不丢弃样本的情况下真正压制住梯度方差的爆炸？*

**分层回答**：

#### ✅ 能够压制的情况：

1. **使用 per-token PPO clip（默认）**：梯度方差严格有界。每个 token 的 IS ratio 被独立裁剪到 `[1-ε, 1+ε]`，不存在乘积爆炸。这是 verl 默认行为，**无需丢弃样本即可控制方差**。

2. **使用 GSPO 或 GMPO 损失函数**：序列级比率通过几何均值（÷T）归一化，方差与序列长度无关。搭配 `seq-mean-token-mean` 聚合，是长 CoT 的最佳选择。

3. **使用 Optimal Token Baseline**：为每个时间步提供自适应 baseline，最大程度缩减方差。

#### ⚠️ 不够充分的情况：

1. **Sequence-level TIS 单独使用**：对 32K 序列退化为近似二元值，无法提供有效的校正。必须搭配 per-token PPO clip。

2. **默认 `token-mean` 聚合 + 长短混合 batch**：长序列对梯度的贡献与其长度成正比，可能导致训练不稳定。**必须切换到 `seq-mean-token-mean`**。

3. **多 PPO epochs（`ppo_epochs > 1`）**：内步漂移在 32K 序列上累积显著，建议降至 1。

4. **全异步路径**：仅依赖流量控制 + PPO clip，**无** 基于版本年龄的 IS 校正。当 staleness_threshold 较大时，PPO clip 可能不足以消化策略漂移。

### 5.3 推荐的长 CoT 最优配置

```yaml
# 损失函数选择（三选一）
actor:
  policy_loss: "gspo"          # 或 "geo_mean"
  loss_agg_mode: "seq-mean-token-mean"
  clip_ratio: 0.2
  clip_ratio_low: 0.2
  clip_ratio_high: 0.28        # DAPO 风格非对称
  ppo_epochs: 1                # 避免内步漂移

# Rollout Correction（如需 off-policy 校正）
algorithm:
  rollout_correction:
    rollout_is: "token"         # 不要用 "sequence"
    rollout_is_threshold: 2.0
    rollout_rs: "seq_mean_k2"   # 长度归一化的拒绝标准
    rollout_rs_threshold: 0.5

# KL 控制
  kl_ctrl:
    type: "adaptive"
    kl_coef: 0.001
    target_kl: 0.05             # 对长序列更保守的 KL 目标

# 梯度裁剪
  grad_clip: 1.0

# 优势估计（如资源允许）
  adv_estimator: "optimal_token_baseline"
  calculate_sum_pi_squared: true
```

---

## 附录

### A. 完整 Loss 函数注册表

| 算法 | 注册名 | 行号 | 长序列特性 |
|------|--------|------|-----------|
| Vanilla PPO | `"vanilla"` | 1165-1256 | Per-token clip，标准选择 |
| DPPO-TV | `"dppo_tv"` | 1259-1340 | Total Variation 约束 |
| DPPO-KL | `"dppo_kl"` | 1341-1422 | KL 散度约束 |
| GSPO | `"gspo"` | 1425-1498 | **几何均值，长序列友好** |
| SAPO | `"sapo"` | 1501-1586 | Sigmoid 门控 |
| GPG | `"gpg"` | 1587-1621 | 几何均值策略梯度 |
| Clip-CoV | `"clip_cov"` | 1623-1726 | 协方差感知裁剪 |
| KL-CoV | `"kl_cov"` | 1728-1806 | KL-协方差裁剪 |
| GMPO | `"geo_mean"` | 1808-1890 | **先 clip 再几何均值，最稳健** |
| CISPO | `"cispo"` | 1893-1951 | Stop-gradient 裁剪 IS |
| REINFORCE | `"reinforce"` | 2156-2233 | 策略梯度 + 可选 IS |
| Bypass | `"bypass_mode"` | 2236-2372 | 统一接口：REINFORCE/PPO-clip |

### B. 优势估计器注册表

| 估计器 | 枚举值 | 特性 | 长序列建议 |
|--------|--------|------|-----------|
| GAE | `"gae"` | 需要 Critic，TD 残差 + λ 衰减 | 标准，需 Critic |
| GRPO | `"grpo"` | 组内归一化奖励，广播标量 | 信号稀释 |
| RLOO | `"rloo"` | 留一法 baseline | 信号稀释 |
| REINFORCE++ | `"reinforce_plus_plus"` | 折扣回报 + 全局白化 | γ<1 有位置区分 |
| OTB | `"optimal_token_baseline"` | **每时间步最优 baseline** | **最佳** |

### C. Rollout Correction 配置预设

**`algorithm.py:181-256`** — `RolloutCorrectionConfig` 工厂方法：

| 预设 | 方法名 | IS 模式 | RS 模式 | 适用场景 |
|------|--------|---------|---------|---------|
| Token IS（解耦） | `decoupled_token_is()` | token | 无 | **长 CoT 推荐** |
| Seq IS（解耦） | `decoupled_seq_is()` | sequence | 无 | 短序列 |
| Seq IS + RS | `decoupled_seq_is_rs()` | sequence | seq_sum_k2 | 短序列 + 硬边界 |
| Bypass PPO-clip | `bypass_ppo_clip()` | 无（PPO clip 自处理） | 无 | 简单 off-policy |
| Bypass PG + IS | `bypass_pg_is()` | sequence | 无 | REINFORCE 变体 |

### D. 核心文件索引

| 文件 | 行数 | 核心内容 |
|------|------|---------|
| `verl/trainer/ppo/core_algos.py` | ~2370 | 所有 RL 损失函数、优势估计器、KL 惩罚 |
| `verl/trainer/ppo/rollout_corr_helper.py` | ~1074 | TIS 权重、拒绝采样、off-policy 指标 |
| `verl/workers/actor/dp_actor.py` | ~670 | Actor 训练循环、梯度裁剪、PPO epochs |
| `verl/trainer/config/algorithm.py` | ~568 | 算法配置、RolloutCorrection 配置 |
| `verl/trainer/ppo/ray_trainer.py` | ~1600 | 训练编排、KL 惩罚应用 |
| `verl/utils/torch_functional.py` | ~300 | masked_whiten, masked_mean, logprobs_from_logits |
