# 08 - Core Algorithms: RL 核心算法库

## 1. 模块概览

| 属性 | 值 |
|------|------|
| 文件路径 | `verl/trainer/ppo/core_algos.py` |
| 总行数 | 2,508 |
| 优势估计器数量 | 14 种 |
| 策略损失函数数量 | 11 种 |
| 注册表 | `ADV_ESTIMATOR_REGISTRY`, `POLICY_LOSS_REGISTRY` |
| KL 控制器 | `AdaptiveKLController`, `FixedKLController` |
| 最后更新 | 2026-08-02 |
| 基准源码 | 上游 `e3573545` |

`core_algos.py` 是 verl 的算法核心文件，集中实现了所有 RL 后训练算法的数学计算——优势估计、策略梯度损失、价值损失、KL 惩罚和损失聚合。多数 advantage/policy-loss 入口采用函数式接口 + 注册表组织，并与分布式训练后端（FSDP/Megatron）解耦；但 `AdaptiveKLController` 会更新内部状态（`core_algos.py:153-174`），部分路径还使用随机采样（如 `torch.multinomial`，`core_algos.py:2257-2264`），不能概括为全部无状态纯函数。

## 2. 注册表架构

### 2.1 双注册表模式

文件定义了两个并行的注册表：

**`ADV_ESTIMATOR_REGISTRY`**（第 113 行）：优势估计器注册表
```python
ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}
```

通过 `@register_adv_est(name_or_enum)` 装饰器注册（第 116-134 行）。注册时检查重复注册并报错：
```python
if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
    raise ValueError(f"Adv estimator {name} has already been registered: ...")
```

**`POLICY_LOSS_REGISTRY`**（第 50 行）：策略损失注册表
```python
POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}
```

通过 `@register_policy_loss(name)` 装饰器注册（第 53-67 行）。`PolicyLossFn` 的类型签名（第 37-48 行）：

```python
PolicyLossFn = Callable[
    [torch.Tensor,  # old_log_prob
     torch.Tensor,  # log_prob
     torch.Tensor,  # advantages
     torch.Tensor,  # response_mask
     str,           # loss_agg_mode
     Optional[DictConfig | ActorConfig],  # config
     torch.Tensor | None],  # rollout_log_probs
    tuple[torch.Tensor, dict[str, Any]],
]
```

### 2.2 AdvantageEstimator 枚举 (第 88-111 行)

```python
class AdvantageEstimator(str, Enum):
    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
    OPTIMAL_TOKEN_BASELINE = "optimal_token_baseline"
    TIR_OPTIMAL_TOKEN_BASELINE = "tir_optimal_token_baseline"
    GDPO = "gdpo"
```

枚举继承自 `str`，可直接用字符串值索引注册表。代码注释（第 93-95 行）指出此枚举创建后不可变，用户可通过 `register_adv_est` 用字符串名注册自定义估计器而无需修改枚举。

### 2.3 查询接口

- `get_adv_estimator_fn(name_or_enum)`（第 137-150 行）：按名称或枚举获取优势估计函数
- `get_policy_loss_fn(name)`（第 70-85 行）：按名称获取策略损失函数

## 3. 优势估计器详解

### 3.1 GAE -- 广义优势估计 (第 216-263 行)

注册名：`"gae"`

经典的 GAE(lambda) 算法（Schulman 2016, https://arxiv.org/abs/1506.02438），**唯一需要 Critic 价值函数的估计器**。

数学原理：
```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_l
```

实现要点：
- 反向遍历时间步 `for t in reversed(range(gen_len))`
- 对 `response_mask` 为 0 的位置（EOS 后的 token），保持 `nextvalues` 和 `lastgaelam` 不变（第 255-256 行）
- 返回值 `returns = advantages + values`
- 优势做 masked whiten 归一化：`verl_F.masked_whiten(advantages, response_mask)`

### 3.2 GRPO -- 组内相对策略优化 (第 268-331 行)

注册名：`"grpo"`

GRPO 将同一 prompt 的多个采样分组，计算组内归一化优势（DeepSeek-R1 采用的核心算法）。

数学原理：
```
score_i = sum(token_level_rewards_i)  # 每个 response 的总分
A_i = (score_i - mean_group) / (std_group + epsilon)  # 组内 z-score
```

关键参数：
- `norm_adv_by_std_in_grpo`（默认 `True`）：若为 `False` 则不除以标准差，即 Dr.GRPO 变体（https://arxiv.org/abs/2503.20783）
- `epsilon`：默认 `1e-6`，避免除零
- 单样本组（`len == 1`）：mean 设为 0，std 设为 1（第 316-317 行）

### 3.3 GRPO Vectorized (第 335-358 行)

注册名：`"grpo_vectorized"`

GRPO 的向量化实现，使用 `group_mean_std` 工具函数避免 Python 循环。核心差异在于用 `as_torch_index` 和 `group_mean_std` 替代手动字典收集。

### 3.4 GDPO -- 组级解耦归一化 (第 362-468 行)

注册名：`"gdpo"`（https://arxiv.org/abs/2601.05242）

GDPO 解决了多维奖励信号中强势维度淹没弱势维度的问题：

```
Step 1: 对每个奖励维度 k 独立做 GRPO 归一化: A_k = GRPO(r_k)
Step 2: 加权聚合: A_sum = sum(w_k * A_k)
Step 3: 批次级白化: A_final = masked_whiten(A_sum)
```

配置项：
- `gdpo_reward_keys`：各维度奖励的键名列表（如 `['format_reward', 'accuracy_reward']`）
- `gdpo_reward_weights`：各维度权重（默认全 1）

### 3.5 GRPO Pass@k (第 472-530 行)

注册名：`"grpo_passk"`（https://arxiv.org/abs/2503.19595）

Pass@k 优化：**组内只有最佳 response 获得非零优势**。

```
A_best = r_max - r_second_max
A_others = 0
```

要求每组至少 2 个样本（第 517-519 行），否则 raise `ValueError`。

### 3.6 REINFORCE++ (第 694-729 行)

注册名：`"reinforce_plus_plus"`（https://arxiv.org/abs/2501.03262）

基于折扣累积回报的 token 级优势估计：

```
returns[t] = r_t + gamma * returns[t+1]  # 反向累积
running_return = running_return * response_mask[:, t]  # EOS 后重置
advantages = masked_whiten(returns)
```

与 GAE 不同，REINFORCE++ 不需要 Critic。

### 3.7 REINFORCE++ Baseline (第 536-584 行)

注册名：`"reinforce_plus_plus_baseline"`

在 REINFORCE++ 基础上增加组内均值 baseline：

```
A_i = score_i - mean_group(scores)
advantages = masked_whiten(A) * response_mask
```

### 3.8 RLOO -- Leave-One-Out (第 588-636 行)

注册名：`"rloo"`（https://arxiv.org/abs/2402.14740）

RLOO 的核心思想是用"去掉自己之后的组均值"作为 baseline：

```
A_i = score_i * n/(n-1) - mean_group * n/(n-1)
```

其中 `n = response_num`。当 `n == 1` 时 mean 设为 0（第 622-623 行）。

### 3.9 RLOO Vectorized (第 832-866 行)

注册名：`"rloo_vectorized"`

RLOO 的全向量化版本，核心一行计算（第 862 行）：

```python
adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)
```

其中 `c` 是每组的样本计数，`inv` 是 unique 逆索引。

### 3.10 OPO -- 带长度加权的优势估计 (第 640-690 行)

注册名：`"opo"`（https://arxiv.org/pdf/2505.23585）

OPO 的 baseline 使用响应长度加权：

```
bsl_g = sum(len_i * score_i) / sum(len_i)  # 组内长度加权均值
A_i = score_i - bsl_g
```

### 3.11 ReMax (第 733-765 行)

注册名：`"remax"`（https://arxiv.org/abs/2310.10505）

使用外部提供的 `reward_baselines`（贪心解码的奖励）作为 baseline：

```
returns = cumsum_reverse(token_level_rewards * response_mask)
advantages = returns - reward_baselines.unsqueeze(-1) * response_mask
```

### 3.12 GPG -- 组策略梯度 (第 769-828 行)

注册名：`"gpg"`

GPG 使用动态 alpha 和固定 f_norm：

```
alpha = bsz / count_nonzero(scores)  # 非零奖励的比例倒数
A_i = alpha * (score_i - mean_group) / f_norm
```

### 3.13 Optimal Token Baseline (第 870-985 行)

注册名：`"optimal_token_baseline"`

基于路径方差的 **token 级** baseline，为每个时间步计算独立 baseline：

```
w_j = 1 - 2*pi_j + sum(pi^2)          # 每步方差代理
W_t = cumsum(w_j)                       # 累积路径方差
B_t* = sum_group(G_t * W_t) / sum_group(W_t)  # 最优 baseline
A_t = G_t - B_t*
```

关键特性：
- `handle_zero_tail`（默认 `True`）：当组内最长轨迹超出第二长轨迹的部分，baseline 置零（第 970-980 行）
- 支持 rollout IS 权重校正：`w_per_timestep *= rollout_is_weights**2`（第 931 行）
- 单轨迹组（`N==1`）不设 baseline

### 3.14 TIR Optimal Token Baseline (第 989-1119 行)

注册名：`"tir_optimal_token_baseline"`

多轮（Tool-Integrated Reasoning）场景下的 OTB。与单轮版本的差异：
- 将 `response_mask` 中的有效 token 提取到紧凑的 `all_w_values` 和 `all_returns` 中（第 1061-1067 行）
- 支持非连续的 response token（多轮对话中的间隔 mask）

## 4. 策略损失函数详解

### 4.1 vanilla -- 标准 PPO 裁剪 (第 1279-1369 行)

注册名：`"vanilla"`

经典 PPO-Clip 实现，核心公式：

```
ratio = exp(log_prob - old_log_prob)            # 重要性采样比
pg_losses1 = -advantages * ratio                # 无裁剪损失
pg_losses2 = -advantages * clamp(ratio, 1-eps_low, 1+eps_high)  # 裁剪损失
clip_pg_losses1 = max(pg_losses1, pg_losses2)   # 悲观估计
```

**双裁剪 PPO**（Dual-Clip, https://arxiv.org/pdf/1912.09729）：
```
pg_losses3 = -advantages * clip_ratio_c         # 下界裁剪（默认 clip_ratio_c=3.0）
clip_pg_losses2 = min(pg_losses3, clip_pg_losses1)
pg_losses = where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

数值稳定性：`negative_approx_kl = clamp(log_prob - old_log_prob, min=-20.0, max=20.0)`（第 1331 行）。

配置项来自 `ActorConfig`：
- `clip_ratio`：标准裁剪范围 epsilon
- `clip_ratio_low` / `clip_ratio_high`：非对称裁剪
- `clip_ratio_c`：双裁剪下界（默认 3.0，第 1317 行）

返回指标：`pg_clipfrac`（被裁剪的比例）、`ppo_kl`（近似 KL 散度）、`pg_clipfrac_lower`（下界裁剪比例）。

### 4.2 dppo_tv -- DPPO Total Variation (第 1373-1450 行)

注册名：`"dppo_tv"`（https://arxiv.org/pdf/2602.04879）

使用 Total Variation 散度约束而非 ratio 裁剪：

```
valid_positive_mask = (prob - old_prob) <= clip_divergence_high
valid_negative_mask = (prob - old_prob) >= -clip_divergence_low
valid_mask = where(advantages > 0, valid_positive_mask, valid_negative_mask)
```

特点：
- 使用截断重要性采样（TIS）：`truncated_ratio = clamp(ratio, max=clip_ratio_c)`，默认 `clip_ratio_c=20.0`（第 1420 行）
- 损失公式：`-advantages * truncated_ratio * log_prob * valid_mask`
- 梯度只通过 `log_prob` 流动（`truncated_ratio` 已 detach）

### 4.3 dppo_kl -- DPPO Binary KL (第 1454-1535 行)

注册名：`"dppo_kl"`（https://arxiv.org/pdf/2602.04879）

使用 Binary KL 散度约束：

```python
binary_kl = old_prob * (old_log_prob - log_prob) + (1 - old_prob) * log((1 - old_prob + 1e-8) / (1 - prob + 1e-8))
valid_positive_mask = (binary_kl <= clip_divergence_high) | (prob <= old_prob)
valid_negative_mask = (binary_kl <= clip_divergence_low) | (prob >= old_prob)
```

同样使用 TIS，默认 `clip_ratio_c=20.0`（第 1501 行）。

### 4.4 gspo -- 序列级重要性采样 (第 1539-1611 行)

注册名：`"gspo"`（https://arxiv.org/pdf/2507.18071）

GSPO 的关键创新是**序列级重要性比**：

```
s_i(theta) = exp(mean_token(log(pi/pi_old)))     # 几何均值
s_i,t(theta) = sg[s_i] * pi_t / sg[pi_t]         # 组合比
pg_losses = max(-A * s_i,t, -A * clamp(s_i,t, 1-eps, 1+eps))
```

强制使用 `"seq-mean-token-mean"` 聚合模式（第 1598 行）。

### 4.5 sapo -- 平滑近端策略优化 (第 1615-1696 行)

注册名：`"sapo"`（https://arxiv.org/pdf/2511.20347）

SAPO 使用 sigmoid 门函数替代 PPO 的硬裁剪：

```python
def gate_function(x, tau):
    return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)
```

根据优势正负使用不同温度：
```
tau = tau_pos if A > 0 else tau_neg
gates = gate_function(ratio, tau)
pg_losses = -gates * advantages
```

配置项：`config.tau_pos`、`config.tau_neg`。

### 4.6 gpg -- GPG 策略梯度损失 (第 1700-1732 行)

注册名：`"gpg"`

最简单的策略梯度：`pg_losses = -log_prob * advantages`，不使用重要性采样比。

### 4.7 clip_cov -- 协方差裁剪 (第 1736-1837 行)

注册名：`"clip_cov"`（https://github.com/PRIME-RL/Entropy-Mechanism-of-RL）

在标准 PPO 裁剪基础上，额外识别高协方差 token 并裁剪：

```
cov_all = (A - mean(A)) * (log_prob - mean(log_prob))
# 找出 cov 在 [clip_cov_lb, clip_cov_ub] 范围内的 token
# 随机选取 clip_cov_ratio 比例进行裁剪
corr[selected] = 0  # 将选中 token 的损失归零
```

默认参数（第 1780-1785 行）：
- `clip_cov_ratio = 0.0002`
- `clip_cov_ub = 5.0`
- `clip_cov_lb = 1.0`

### 4.8 kl_cov -- KL 协方差惩罚 (第 1841-1917 行)

注册名：`"kl_cov"`

对高协方差 token 添加 KL 惩罚而非裁剪：

```
pg_losses_kl = -A * ratio + ppo_kl_coef * |log_prob - old_log_prob|
# 选出 top-k 高协方差 token，替换为 pg_losses_kl
```

默认参数（第 1876-1877 行）：
- `kl_cov_ratio = 0.0002`
- `ppo_kl_coef = 1.0`

### 4.9 geo_mean -- 几何均值策略优化 (第 1921-2003 行)

注册名：`"geo_mean"`（GMPO, https://arxiv.org/abs/2507.20673）

将 token 级裁剪的 log-ratio 通过序列级几何均值聚合：

```
# Token 级裁剪
neg_kl_clamp = clamp(log_prob - old_log_prob, -eps_low, eps_high)
neg_kl_min = min(sgn(A) * neg_kl, sgn(A) * neg_kl_clamp) * sgn(A)

# 序列级几何均值
ratio = exp(sum(neg_kl_min * mask) / sum(mask))
advantage = sum(A * mask) / sum(mask)
pg_losses = -advantage * ratio
```

最终损失用简单的 `torch.mean` 聚合（第 1992 行）。

### 4.10 cispo -- 裁剪重要性采样策略优化 (第 2007-2064 行)

注册名：`"cispo"`（https://arxiv.org/pdf/2506.13585）

关键区别：**对裁剪后的 ratio 使用 stop-gradient**，梯度只通过 `log_prob` 流动：

```
clipped_ratio = clamp(ratio, 1-eps, 1+eps)
clipped_ratio_sg = clipped_ratio.detach()  # stop gradient
pg_losses = -clipped_ratio_sg * advantages * log_prob
```

### 4.11 bypass_mode -- 旁路模式 (第 2373-2508 行, `def compute_policy_loss_bypass_mode`)

注册名：`"bypass_mode"`

`bypass_mode` 是一个分派入口，当 `old_log_prob = rollout_log_prob` 时使用：

- **`loss_type="ppo_clip"`**（默认）：调用 `compute_policy_loss_vanilla`，不额外应用 IS 权重（PPO 的 ratio 已隐含 IS 修正）
- **`loss_type="reinforce"`**：调用 `compute_policy_loss_reinforce`，显式应用 IS 权重 `w = pi_current / pi_rollout`

使用 `compute_rollout_correction_and_rejection_mask` 计算 IS 权重和拒绝采样掩码。配置项（第 2443-2448 行）：
- `rollout_is`：IS 聚合级别（`"token"` / `"sequence"` / `None`）
- `rollout_is_threshold`：截断阈值（默认 `2.0`）
- `rollout_rs`：拒绝采样模式
- `rollout_is_batch_normalize`：是否对 IS 权重做批次归一化

### 4.12 compute_policy_loss_reinforce (第 2292-2369 行)

未注册到 POLICY_LOSS_REGISTRY 的辅助函数，被 bypass_mode 调用：

```
# 标准 REINFORCE: L = -E[log pi * A]
# 带 IS 修正: L = -E[w * log pi * A]
```

## 5. 损失聚合函数 agg_loss (第 1138-1199 行)

`agg_loss` 是保证**损失对分布式并行不变性**的关键函数，支持 4 种模式：

| 模式 | 公式 | 适用场景 |
|------|------|---------|
| `token-mean` | `sum(loss * mask) / batch_num_tokens * dp_size` | 标准 PPO（默认） |
| `seq-mean-token-sum` | `sum_seq(sum_token(loss * mask)) / global_batch_size * dp_size` | GSPO, SAPO |
| `seq-mean-token-sum-norm` | 同上再除以 `loss_scale_factor`（默认为 `loss_mask.shape[-1]`） | 需要长度归一化 |
| `seq-mean-token-mean` | `sum_seq(mean_token(loss * mask)) / global_batch_size * dp_size` | GSPO, SAPO |

注意事项：
- `dp_size > 1` 时必须提供 `batch_num_tokens` 或 `global_batch_size`，否则 raise `ValueError`（第 1170-1171, 1178-1179 行）
- 完全被 mask 掉的序列被 `seq_mask = (sum(mask, dim=-1) > 0).float()` 排除

## 6. 辅助组件

### 6.1 KL 控制器 (第 153-212 行)

**`AdaptiveKLController`**（第 153-174 行）：
```python
proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
mult = 1 + proportional_error * n_steps / self.horizon
self.value *= mult
```

来自论文 https://arxiv.org/pdf/1909.08593.pdf，比例误差裁剪到 `[-0.2, 0.2]`。

**`FixedKLController`**（第 177-190 行）：`update` 方法为空操作。

**工厂函数 `get_kl_controller`**（第 193-212 行）：根据 `kl_ctrl.type` 创建对应控制器，adaptive 模式要求 `horizon > 0`。

### 6.2 KL 惩罚 (第 2147-2210 行)

`kl_penalty` 函数（第 2147 行, `def kl_penalty`）支持多种 KL 散度估计器：

| 名称 | 公式 | 说明 |
|------|------|------|
| `"kl"` / `"k1"` | `logprob - ref_logprob` | 简单差异 |
| `"abs"` | `\|logprob - ref_logprob\|` | 绝对值 |
| `"mse"` / `"k2"` | `0.5 * (logprob - ref_logprob)^2` | 均方差 |
| `"low_var_kl"` / `"k3"` | `exp(ref-log) - (ref-log) - 1`，裁剪到 `[-10, 10]` | 低方差估计器 |

带 `"+"` 后缀（如 `"k3+"`）时使用 **straight-through trick**（第 2165-2172 行）：前向用指定估计器，反向用 k2 估计器的梯度，确保无偏梯度估计：

```python
backward_score = 0.5 * (logprob - ref_logprob).square()
return backward_score - backward_score.detach() + forward_score.detach()
```

### 6.3 价值损失 (第 2084-2145 行)

`compute_value_loss` 实现 PPO 的裁剪价值损失：

```
vpredclipped = clip(vpreds, values - cliprange_value, values + cliprange_value)
vf_losses1 = (vpreds - returns)^2
vf_losses2 = (vpredclipped - returns)^2
vf_loss = 0.5 * agg_loss(max(vf_losses1, vf_losses2))
```

**行为变更**：`compute_value_loss` 新增 4 个参数 `dp_size` / `batch_num_tokens` / `global_batch_size` / `loss_scale_factor`，并全部转发给 `agg_loss`（第 2138-2141 行），因此价值损失现已支持 DP / global-batch 归一化，与策略损失保持一致的分布式不变性。

### 6.4 compute_rewards (第 1122-1135 行)

计算带 KL 惩罚的 token 级奖励：
```python
kl = old_log_prob - ref_log_prob
return token_level_scores - kl * kl_ratio
```

### 6.5 PF-PPO 重加权 (第 2213-2289 行)

`compute_pf_ppo_reweight_data` 实现基于奖励的数据重采样：

| 方法 | 权重公式 |
|------|---------|
| `"pow"` | `\|score\|^weight_pow`（默认 `weight_pow=2.0`） |
| `"max_min"` | max/min 得 1.0，其余得 0.0 |
| `"max_random"` | max 得 0.4，其余得 0.1 |

通过 `torch.multinomial` 有放回采样。

## 7. 算法对比总结

### 7.1 优势估计器对比

| 估计器 | 需要 Critic | 粒度 | 需要分组 | 核心特点 |
|--------|-----------|------|---------|---------|
| GAE | 是 | Token | 否 | 经典 TD(lambda) |
| GRPO | 否 | Outcome | 是 | 组内 z-score |
| GRPO_VECTORIZED | 否 | Outcome | 是 | GRPO 向量化 |
| GDPO | 否 | Outcome | 是 | 多维奖励解耦归一化 |
| GRPO_PASSK | 否 | Outcome | 是 | 只有最佳 response 获奖 |
| REINFORCE++ | 否 | Token | 否 | 折扣累积回报 + whiten |
| REINFORCE++_BASELINE | 否 | Outcome | 是 | 组均值 baseline + whiten |
| RLOO | 否 | Outcome | 是 | Leave-one-out baseline |
| RLOO_VECTORIZED | 否 | Outcome | 是 | RLOO 向量化 |
| OPO | 否 | Outcome | 是 | 长度加权 baseline |
| ReMax | 否 | Token | 否 | 外部贪心 baseline |
| GPG | 否 | Outcome | 是 | 动态 alpha，固定 f_norm |
| OPTIMAL_TOKEN_BASELINE | 否 | Token | 是 | 路径方差 token 级 baseline |
| TIR_OPTIMAL_TOKEN_BASELINE | 否 | Token | 是 | 多轮 OTB |

### 7.2 策略损失对比

| 损失函数 | 注册名 | 使用 ratio | 使用 log_prob | 裁剪方式 | 默认聚合 |
|---------|--------|-----------|-------------|---------|---------|
| PPO-Clip | `vanilla` | 是 | 否 | ratio 裁剪 | token-mean |
| DPPO-TV | `dppo_tv` | 是(截断) | 是 | TV 散度掩码 | token-mean |
| DPPO-KL | `dppo_kl` | 是(截断) | 是 | Binary KL 掩码 | token-mean |
| GSPO | `gspo` | 序列级 | 否 | 序列级 ratio 裁剪 | seq-mean-token-mean |
| SAPO | `sapo` | 是 | 否 | sigmoid 门函数 | seq-mean-token-mean |
| GPG | `gpg` | 否 | 是 | 无裁剪 | token-mean |
| Clip-Cov | `clip_cov` | 是 | 否 | PPO + 协方差裁剪 | token-mean |
| KL-Cov | `kl_cov` | 是 | 否 | KL 惩罚替代 | token-mean |
| GMPO | `geo_mean` | 几何均值 | 否 | 序列级几何均值 | mean |
| CISPO | `cispo` | 是(detach) | 是 | sg(clip(ratio)) * log_prob | token-mean |
| Bypass | `bypass_mode` | 分派 | 分派 | 分派 | token-mean |

## 8. 设计要点与扩展指南

### 8.1 函数式接口 + 注册表

注册的多数算法实现采用**函数式接口**（输入张量，输出张量和指标字典），通过注册表装饰器自动发现；状态控制器与 RNG 路径是例外。用户扩展只需：

```python
from verl.trainer.ppo.core_algos import register_adv_est, register_policy_loss

@register_adv_est("my_estimator")
def compute_my_advantage(token_level_rewards, response_mask, **kwargs):
    ...
    return advantages, returns

@register_policy_loss("my_loss")
def compute_my_loss(old_log_prob, log_prob, advantages, response_mask,
                     loss_agg_mode, config, rollout_is_weights):
    ...
    return loss, metrics_dict
```

### 8.2 与 Trainer 的接口

Trainer 负责根据 `algorithm.adv_estimator` 选择/计算 advantage；policy loss 名称则由 Worker 侧的 loss adapter（`verl/workers/utils/losses.py:101-104`）解析 `get_policy_loss_fn`。算法切换仍主要通过配置文件中的 `algorithm.adv_estimator` 和 `actor.policy_loss` 字段完成。

### 8.3 agg_loss 的分布式不变性

`agg_loss` 通过显式传入 `dp_size`、`batch_num_tokens`、`global_batch_size` 参数，确保无论数据如何分片到不同 DP rank，最终的梯度聚合结果一致。FSDP 的损失直接 backward；Megatron 需要额外乘以 `num_microbatches` 和 `cp_size`。

### 8.4 Rollout IS 修正

多个损失函数支持 `rollout_is_weights` 参数，用于修正 rollout 策略（如 vLLM 的 BF16 推理）与训练策略之间的精度差异。`bypass_mode` 进一步整合了 IS 权重计算和拒绝采样到统一入口。
