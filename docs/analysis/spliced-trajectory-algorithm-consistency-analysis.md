# 拼接轨迹的算法一致性分析：IS 比率、Advantage 估计与 Staleness 评估的数学缝合线

**Author:** Claude Opus 4.6 (基于 verl 源码深度分析，综合 4 个专家 Agent 的独立分析结果)
**Date:** 2026-03-20
**Scope:** verl `main` branch，聚焦 Fully Async Partial Rollout 场景下拼接轨迹的算法正确性

**分析贡献者：**
- **Algorithm Expert Agent #1**：遍历所有 11+ policy loss 函数，追踪 IS 比率的版本盲区
- **Explore Agent（Staleness 追踪）**：深度分析 MessageQueue、Staleness 评估、多轮工具调用变体
- **Algorithm Expert Agent #2（Advantage 估计）**：逐估计器分析混合版本数据的偏差源
- **Architect Reviewer Agent**：完整 8 阶段数据流追踪、架构缺口评估、分级修复建议

---

## 目录

1. [问题定义：拼接轨迹的数学本质](#1-问题定义)
2. [完整数据流追踪：8 阶段生命周期](#2-数据流追踪)
3. [Logprob 的真实性：π_old 到底算哪个？](#3-logprob-真实性)
4. [两条恢复路径的 Logprob 语义差异与前缀覆写问题](#4-两条路径)
5. [IS 比率在拼接轨迹上的数学推导：全部 11+ Loss 函数遍历](#5-is-比率)
6. [Advantage 估计的 7 种偏差源：逐估计器分析](#6-advantage)
7. [Staleness 评估机制：有界队列与版本过滤的缺失](#7-staleness)
8. [Reward 信号的阶跃问题与 PRM 假设分析](#8-reward-阶跃)
9. [架构缺口的严重度评估与分级修复建议](#9-修复建议)
10. [关键源码索引](#10-源码索引)

---

## 1. 问题定义：拼接轨迹的数学本质 {#1-问题定义}

假设一条完整序列 $x_1, x_2, ..., x_N$：

- **前缀** $x_{1...k}$：由模型版本 $v_n$（权重 $\theta_n$）生成，对应 action logprob 为 $\log \pi_{\theta_n}(x_t | x_{<t})$
- **后缀** $x_{k+1...N}$：由模型版本 $v_{n+1}$（权重 $\theta_{n+1}$）生成，logprob 为 $\log \pi_{\theta_{n+1}}(x_t | x_{<t})$

**Architect Reviewer 指出的关键细微之处**：后缀 token $x_{k+1...N}$ 是由 $\theta_{n+1}$ 生成的，但其条件上下文 $x_{<t}$ 包含了由 $\theta_n$ 产生的前缀。这构成了一种**跨策略条件化（Cross-Policy Conditioning）**——$\theta_{n+1}$ 在一个它未曾见过的上下文分布上生成 token。这在标准 RL 理论中没有直接对应物。

当这条轨迹进入 Trainer 的 PPO/GRPO Loss 计算时，算法需要：
1. 一个一致的 $\pi_{old}$ 作为 IS 比率的分母
2. 正确的 Advantage 估计
3. 对数据"过时度"的评估

### 1.1 前提澄清：拼接在何时发生？

**Advantage Expert 的重要澄清**：Token 级别的拼接**仅在** `partial_rollout=True`（Mode d）时通过 `PartialSingleTurnAgentLoop` 发生。在其他模式下：

| 模式 | 是否拼接 | 说明 |
|------|---------|------|
| 同步 Hybrid Engine | **否** | 生成完成后才 sleep，不存在中途中断 |
| Async `partial_rollout=False` | **否** | 等待正在进行的生成完成后才同步 |
| Async `partial_rollout=True` | **是** | 中断→保存→同步→恢复→拼接 |

但即使在非拼接模式下，**Batch 级别的混合版本**仍然存在——一个 training batch 可能包含来自不同参数版本的完整轨迹。

---

## 2. 完整数据流追踪：8 阶段生命周期（Architect Reviewer Agent 贡献）{#2-数据流追踪}

以下追踪一条 partial rollout 样本从生成到训练的完整生命周期，标注每个阶段**保留了什么**和**丢失了什么**版本元数据。

### 阶段 1：V_n 下开始生成

```
FullyAsyncRollouter._process_single_sample_streaming()
  → full_batch.non_tensor_batch["param_version"] = [current_param_version]  # V_n
  → FullyAsyncAgentLoopManager.generate_single_sample_async()
    → FullyAsyncAgentLoopWorker.generate_sequences_no_post()
      → _partial_run_agent_loop()
        → PartialSingleTurnAgentLoop.run()
          → param_version_start = V_n, param_version_end = V_n   # 初始化
          → generate_for_partial(prompt_ids, ...)                  # 开始生成
```

**保留**：`param_version_start=V_n`
**丢失**：无（刚开始）

### 阶段 2：在 token k 处被 abort

```
vLLMHttpServerForPartial.generate_for_partial()
  → 竞速：generation_handle vs cancel_event.wait()
  → cancel 胜出 → is_cancel = True
  → 返回 (token_ids[0..k], log_probs[0..k], is_cancel=True)
```

**保留**：部分 token IDs、部分 logprobs（来自 V_n）
**丢失**：★ **没有版本标签附加到这些 logprobs 上**

### 阶段 3：保存到 cancel_queue

```
PartialSingleTurnAgentLoop.run() 返回:
  AgentLoopOutput(
    response_ids=[x_1...x_k],
    response_logprobs=[ℓ_1...ℓ_k],    # 来自 V_n，无版本标记
    extra_fields={
      "is_cancel": True,
      "param_version_start": V_n,       # ✓ 保留
      "param_version_end": V_n,         # ✓ 保留
    }
  )

_process_single_sample_streaming():
  rollout_sample.agent_loop_output_list = [output]  # 保存部分结果
  await self.cancel_queue.put(rollout_sample)
```

**保留**：`param_version_start=V_n`, `param_version_end=V_n`, 部分 logprobs
**丢失**：★ **拼接点 k 的位置（`len(output.response_ids)`）可用但未被显式存储**

### 阶段 4：权重同步到 V_{n+1}

```
ParameterSynchronizer.sync_weights()
  → Rollouter.pause(): cancel 所有生成，清空 KV Cache
  → checkpoint_manager.update_weights(): NCCL 传输权重
  → Rollouter.update_param_version(V_{n+1})
  → Rollouter.resume(): 恢复推理引擎
```

**保留**：cancel_queue 中的部分样本不受影响
**丢失**：★ KV Cache 被清空

### 阶段 5：从 cancel_queue 恢复

```
_processor_worker():
  rollout_sample = await self.cancel_queue.get()  # 优先取消队列
  full_batch.non_tensor_batch["param_version"] = [V_{n+1}]  # ★ 更新为新版本
  → generate_single_sample_async(full_batch, agent_loop_output_list)
```

**保留**：`agent_loop_output_list` 中的部分结果（V_n logprobs）
**丢失**：★ 样本级 `param_version` 被覆写为 V_{n+1}

### 阶段 6：PartialSingleTurnAgentLoop 第二次运行

```python
# partial_single_turn_agent_loop.py:86-114
output.extra_fields.get("is_cancel") == True  # 进入恢复分支
prompt_ids = output.prompt_ids + output.response_ids  # 拼接为新 prompt
param_version_start = V_n  # ✓ 从 extra_fields 恢复

# 用 V_{n+1} 权重生成（Re-Prefill 整个前缀 + 继续生成）
response_ids_new, logprobs_new, is_cancel = generate_for_partial(prompt_ids, ...)

# ★★★ 核心拼接点 ★★★
response_logprobs = output.response_logprobs + logprobs_new  # V_n 的 ℓ_{1..k} + V_{n+1} 的 ℓ_{k+1..N}
response_ids = output.response_ids + response_ids_new

return AgentLoopOutput(
    extra_fields={
        "param_version_start": V_n,      # ✓ 保留
        "param_version_end": V_{n+1},    # ✓ 更新
    }
)
```

**保留**：`param_version_start=V_n`, `param_version_end=V_{n+1}`
**丢失**：★★★ **拼接点位置 k 被丢弃**（可从 `len(output.response_ids)` 获取但未存储）
**丢失**：★★★ **前缀 logprobs 未被 V_{n+1} 重新计算**（Re-Prefill 只返回新 token 的 logprob）

### 阶段 7：组装 Batch 进入 MessageQueue

```
assemble_batch_from_rollout_samples()  # detach_utils.py:167-187
  → param_version_diff = |V_{n+1} - V_n| = 1     # 计算版本跨度
  → partial_stats["partial_ratio"] = 统计         # ✓ 监控指标
  → meta_info["rollout_param_versions"] = [V_{n+1}]  # 样本完成时的版本
  → meta_info["trajectory_param_versions"] = [V_{n+1}]
```

**保留**：`partial_ratio`, `max_partial_span` 统计指标
**丢失**：★ 拼接点 k 的位置、per-token 版本信息

### 阶段 8：Trainer 消费并训练

```
FullyAsyncTrainer.fit_step()
  → _collect_metrics_from_samples(): 统计 stale_count → 仅写入 metrics dict
  → _fit_compute_log_prob():
      _compute_old_log_prob(): 用锚定版本的参数重新计算 old_log_probs（一致的）
  → _fit_compute_advantage(): 调用 compute_advantage()，无版本感知
  → _fit_update_actor(): 调用 policy_loss_fn()，无版本感知
```

**保留**：WandB 监控指标
**丢失**：★★★ **所有版本信息在进入 Loss 计算前完全丢弃**

---

## 3. Logprob 的真实性：π_old 到底算哪个？ {#3-logprob-真实性}

### 3.1 三策略架构

verl 区分三个策略（`verl/trainer/ppo/ray_trainer.py:1379-1432`）：

| 符号 | 含义 | 数据字段 | 来源 |
|------|------|---------|------|
| $\pi_{rollout}$ | 推理引擎生成数据时的策略 | `rollout_log_probs` | 生成时记录（可能跨版本） |
| $\pi_{old}$ | 训练 batch 开始时的当前模型 | `old_log_probs` | 训练引擎重新计算（一致的） |
| $\pi_\theta$ | mini-batch 更新中演化的模型 | `log_prob` | 训练前向传播 |

### 3.2 Bypass Mode：π_old = π_rollout（混合版本）

```python
# verl/trainer/ppo/rollout_corr_helper.py → apply_bypass_mode()
old_log_probs = rollout_log_probs  # π_old := π_rollout（拼接的）
```

$$\pi_{old}(x_t | x_{<t}) = \begin{cases} \pi_{\theta_n}(x_t | x_{<t}) & \text{if } t \leq k \\ \pi_{\theta_{n+1}}(x_t | x_{<t}) & \text{if } t > k \end{cases}$$

**Algorithm Expert 确认**：在 `dp_actor.py:594-601`，当 `use_rollout_log_probs=True`（Fully Async 默认设置）时：

```python
if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
    old_log_prob = model_inputs["old_log_probs"]  # = rollout_log_probs（混合版本）
```

### 3.3 Decoupled Mode：π_old 一致（FullyAsyncTrainer 的 MIS 机制）

FullyAsyncTrainer 覆写了 `_compute_old_log_prob`（`fully_async_trainer.py:428-448`）：

```python
def _compute_old_log_prob(self, batch):
    if self.local_trigger_step == 1:
        self.actor_rollout_wg.save_model_to_cpu(1)  # 保存"锚定版本"到 CPU
        old_log_prob = super()._compute_old_log_prob(batch)
    else:
        self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
        self.actor_rollout_wg.restore_model_from_cpu(1)  # ★ 恢复到锚定版本
        old_log_prob = super()._compute_old_log_prob(batch)
        self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
```

**这实现了 AReaL 的 "Decoupled PPO"**：$\pi_{old}$ 始终是同步周期开始时的模型，对整条序列做完整前向传播。**不管序列如何拼接，$\pi_{old}$ 都是一致的。**

---

## 4. 两条恢复路径的 Logprob 语义差异与前缀覆写问题 {#4-两条路径}

### 4.1 路径 A：PartialSingleTurnAgentLoop（Token 拼接）

`partial_single_turn_agent_loop.py:86-114`

恢复时将已生成 token 追加到 prompt，用新权重继续。**前缀 logprobs 保留旧值，后缀 logprobs 来自新权重。**

### 4.2 路径 B：FullyAsyncLLMServerManager（Abort→Re-prefill 循环）

`agent_loop.py:86-128`

Re-Prefill 循环将 `prompt_ids + final_output.token_ids` 作为新 prompt 重新提交。

### 4.3 前缀 Logprob 覆写问题（Architect Reviewer 关键发现）

**两条路径都存在同一问题**：Re-Prefill 时，推理引擎用 V_{n+1} 对包含 V_n 生成 token 的整个序列做了完整的前向传播（重建了 KV Cache），但 **`generate()` / `generate_for_partial()` 只返回新生成 token 的 logprob**，不返回 prompt（含前缀）的 logprob。

```python
# agent_loop.py:97-99
final_output.token_ids.extend(output.token_ids)    # 只追加新 token
final_output.log_probs.extend(output.log_probs)    # ★ extend 而非 replace
```

**前缀 logprobs 未被覆写。** 这意味着：

| 属性 | 前缀 token (1..k) | 后缀 token (k+1..N) |
|------|-------------------|---------------------|
| Token 采样策略 | θ_n | θ_{n+1} |
| KV Cache 计算权重 | θ_n（前缀原始 KV） | θ_{n+1}（Re-Prefill 后） |
| Logprob 计算权重 | θ_n | θ_{n+1} |
| Logprob 是否可被覆写 | **理论可行但未实现** | 已是新权重 |

**理论上**，Re-Prefill 时推理引擎已经用 V_{n+1} 计算了前缀每个位置的 logprob（因为前向传播必须经过所有 token），但 vLLM/SGLang 的 `generate()` API 默认不返回 prompt logprobs。**这是一个可行的改进点。**

---

## 5. IS 比率在拼接轨迹上的数学推导：全部 11+ Loss 函数遍历（Algorithm Expert Agent 贡献）{#5-is-比率}

### 5.1 核心发现：所有 Policy Loss 函数都是版本盲的

**Algorithm Expert 遍历了 `core_algos.py` 中所有已注册的 policy loss 函数**，确认它们全部使用相同的 IS 比率计算模式：

```python
negative_approx_kl = log_prob - old_log_prob   # 无版本感知
ratio = torch.exp(negative_approx_kl)           # 逐 token 元素运算
```

| Loss 函数 | 行号 | 对拼接敏感度 | 原因 |
|-----------|------|-----------|------|
| `vanilla` (PPO) | 1216-1219 | 中 | Token-mean 聚合混合不同版本的 ratio |
| `dppo_tv` | 1298-1301 | 中 | 同 vanilla |
| `dppo_kl` | 1379-1382 | 中 | 同 vanilla |
| **`gspo`** | **1458-1464** | **高** | **序列级几何平均** $\exp(\frac{1}{L}\sum_t \log r_t)$，混合不同分布 |
| `sapo` | 1543 | 中 | sigmoid gate，但仍基于 ratio |
| **`geo_mean`** | **1863** | **高** | **全序列 ratio 之积**，跨版本乘积无数学意义 |
| `clip_cov` | 1676 | 中 | 同 vanilla |
| `kl_cov` | 1768 | 中 | 同 vanilla |
| **`cispo`** | **1915-1925** | **低** | Stop-gradient clipped ratio，梯度仅流经 log_prob |
| `reinforce` | 2212 | 低 | 使用 rollout_is_weights 而非 ratio |
| `bypass_mode` | 2315 | 取决于下游 | 设置 old=rollout 后委托给其他函数 |

**特别危险的函数**：

**GSPO**（Geometric Sequence Policy Optimization）对拼接尤为敏感：

```python
negative_approx_kl_seq = sum(negative_approx_kl * mask) / seq_lengths  # 序列级平均
```

这将前缀（ratio 相对于 θ_n）和后缀（ratio 相对于 θ_{n+1}）的 log-ratio 混合求平均，得到一个**对应于不存在的策略**的几何平均 ratio。

**geo_mean** 同理：`ratio = exp(sum(log_ratio_min * mask) / mask_sum)` 是全序列 ratio 之积的几何平均。

### 5.2 Bypass Mode 下的完整偏差推导

当 `old_log_prob = rollout_log_probs` 且训练时 $\theta \approx \theta_{n+1}$：

| Token 位置 | IS 比率 $r_t$ | 数值范围 | PPO Clip 行为 |
|-----------|--------------|---------|-------------|
| $t \leq k$（前缀） | $\frac{\pi_\theta}{\pi_{\theta_n}} \approx \frac{\pi_{\theta_{n+1}}}{\pi_{\theta_n}}$ | 可能显著偏离 1 | **频繁触发** clip |
| $t > k$（后缀） | $\frac{\pi_\theta}{\pi_{\theta_{n+1}}} \approx 1$ | 接近 1 | 几乎不触发 clip |

**结果**：PPO 的梯度信号在序列中**非均匀分布**——前缀产生大梯度（被 clip），后缀产生小梯度。这种非均匀性来自版本切换，而非数据本身。

### 5.3 Decoupled Mode 下的矫正机制

PPO 核心 ratio 不受拼接影响（$\pi_{old}$ 一致）。但 Rollout Correction 的 IS 权重受影响：

$$w_t = \frac{\pi_{old}(x_t|x_{<t})}{\pi_{rollout}(x_t|x_{<t})} = \begin{cases} \frac{\pi_{old}}{\pi_{\theta_n}} & t \leq k \text{（可能 ≠ 1）} \\ \frac{\pi_{old}}{\pi_{\theta_{n+1}}} & t > k \text{（≈ 1 if } \pi_{old} \approx \pi_{\theta_{n+1}}\text{）} \end{cases}$$

IS 权重应用到 Loss（`core_algos.py:1244-1245`）：

```python
if rollout_is_weights is not None:
    pg_losses = pg_losses * rollout_is_weights  # 逐 token 乘以 IS 权重
```

**Token 级 IS**（`rollout_is="token"`）对拼接轨迹最安全：每个 token 独立截断。
**序列级 IS**（`rollout_is="sequence"`）混合了不同版本的 log-ratio，产生一个对应于"弗兰肯斯坦策略"的权重。

---

## 6. Advantage 估计的 7 种偏差源：逐估计器分析（Advantage Expert Agent 贡献）{#6-advantage}

### 6.1 偏差源总览

| 偏差源 | 机制 | verl 是否矫正？ | 数学形式 |
|--------|------|---------------|---------|
| ① 精度不匹配（BF16 vs FP32） | $\pi_{rollout} \neq \pi_{old}$ | ✓ IS 权重 | $\rho_t = \pi_{old}/\pi_{rollout}$ |
| ② Action 分布偏移（过时策略） | 旧策略采样的动作分布 | 部分（TIS + RS） | 截断的 $\rho_{seq}$ |
| ③ **State 分布偏移** | 旧策略访问的状态分布 | **✗ 未矫正** | $d^{\pi_{old}}(s) / d^{\pi_\theta}(s)$（不可追踪） |
| ④ **GRPO/RLOO 组基线污染** | 混合版本组的均值基线偏移 | **✗ 未矫正** | $\mu_g^{mixed} \neq E_{\pi_\theta}[R]$ |
| ⑤ **REINFORCE++ Batch 归一化污染** | 混合版本 Batch 的统计量偏移 | **✗ 未矫正** | $\text{mean}(G)^{mixed} \neq E_{\pi_\theta}[G]$ |
| ⑥ **KL Reward 污染（Bypass 模式）** | KL 惩罚使用 rollout logprob | **✗ 未矫正** | $\text{KL}(\pi_{V_k} \| \pi_{ref}) \neq \text{KL}(\pi_\theta \| \pi_{ref})$ |
| ⑦ **OTB 基线方差膨胀** | IS 权重 $\rho^2$ 放大基线方差 | 部分（IS 截断） | $\text{Var}(B_t^*) \sim O(\max(\rho^4))$ |

### 6.2 GAE 的状态分布偏差

`compute_gae_advantage_return`（`core_algos.py:214-262`）：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$$A_t^{GAE} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

GAE 不直接依赖 rollout logprob，但存在**状态分布偏差**：

$$\underbrace{E_{s \sim d^{\pi_{old}}}[A^{\pi_\theta}(s,a)]}_{\text{实际计算}} \neq \underbrace{E_{s \sim d^{\pi_\theta}}[A^{\pi_\theta}(s,a)]}_{\text{理论正确}}$$

IS 权重矫正的是 action 分布（给定状态下选择不同动作的概率），但**不矫正 state 分布**（哪些状态被访问到）。矫正 state 分布需要轨迹级别的 IS 比率——所有之前 token 的 IS 比率之积——方差呈指数增长，不可行。

**对于拼接轨迹**：Critic $V(s_t)$ 由当前模型评估，对整条序列一致。但如果前缀和后缀的质量因策略切换而突变，Critic 对这种非平稳轨迹的 Value 估计可能不准确。TD-error $\delta_t$ 在拼接点 k 附近可能出现异常峰值。

### 6.3 GRPO 的组基线污染

`compute_grpo_outcome_advantage`（`core_algos.py:266-330`）：

$$A_i = \frac{R_i - \mu_g}{\sigma_g + \epsilon}$$

**Advantage Expert 的关键发现**：在 Fully Async 模式下，同一 prompt 的 `n` 条 completion 是在 `_process_single_sample_streaming` 中一起生成的（通过 `rollout.n` 的 repeat）。因此在同一次生成调用中，所有 completion 由同一版本生成。**组内混版本在单样本级别不会发生。**

但在 **Batch 级别**，不同 prompt 的 completion group 可能来自不同版本。如果使用了 `use_prefix_grouper`（跨 prompt 共享前缀），不同组的版本差异可能通过 Advantage 归一化间接影响彼此。

### 6.4 RLOO 的基线偏差

`compute_rloo_outcome_advantage`（`core_algos.py:476-525`）：

$$A_i = \frac{N}{N-1}\left(R_i - \frac{1}{N-1}\sum_{j \neq i} R_j\right)$$

与 GRPO 类似，Leave-One-Out 基线在组内是正确的（同版本），但跨 Batch 的归一化可能受混版本影响。偏差为：

$$\text{Bias}_i = \frac{N}{N-1}\left(E_{\pi_{mixed}}[R_{-i}] - E_{\pi_\theta}[R_{-i}]\right)$$

### 6.5 REINFORCE++ 的 Batch 归一化污染

`compute_reinforce_plus_plus_outcome_advantage`（`core_algos.py:582-618`）：

$$A_t = \frac{G_t - \text{mean}(G)}{\sqrt{\text{var}(G) + \epsilon}}$$

REINFORCE++ **没有组结构**——在整个 Batch 上做 whiten。如果 Batch 包含来自不同版本的轨迹：
- 旧版本轨迹的 return 可能系统性偏低
- Batch 均值被拉低，导致新版本轨迹获得过高的优势
- 这种偏差方向性取决于训练方向（模型是否在改进）

### 6.6 OTB 的 IS 权重方差膨胀

`compute_optimal_token_baseline_advantage`（`core_algos.py:758-873`）：

$$B_t^* = \frac{\sum_i G_t^i \cdot \rho_i^2 \cdot W_t^i}{\sum_i \rho_i^2 \cdot W_t^i}$$

OTB 是**唯一显式使用 IS 权重的 Advantage 估计器**。$\rho_i^2$ 的非均匀性（前缀和后缀段的 IS 权重量级不同）破坏了基线的 MSE 最优性。

**偏差形式**：

$$\text{Bias}_t = B_t^{mixed} - B_t^{on-policy} = \frac{\sum_i G_t^i \cdot \rho_i^2 \cdot W_t^i}{\sum_i \rho_i^2 \cdot W_t^i} - E_{\pi_\theta}[G_t]$$

这是一个 ratio estimator，偏差为 $O(1/N)$，但方差可能很大（当 $\rho_i$ 值差异大时）。TIS 截断将 $\rho$ 上界限制到 2.0-10.0，但 $\rho^2$ 仍可达 4.0-100.0。

### 6.7 KL Penalty 的 Reward 污染（Bypass 模式）

在 Bypass 模式下，KL 惩罚嵌入了 rollout 策略的 logprob：

$$r_t^{KL} = r_t^{task} - \beta \cdot \left(\log \pi_{rollout}(x_t|x_{<t}) - \log \pi_{ref}(x_t|x_{<t})\right)$$

对于拼接轨迹：

$$r_t^{KL} = \begin{cases} r_t^{task} - \beta \cdot (\log \pi_{\theta_n} - \log \pi_{ref}) & t \leq k \\ r_t^{task} - \beta \cdot (\log \pi_{\theta_{n+1}} - \log \pi_{ref}) & t > k \end{cases}$$

KL 惩罚在 token k 处发生**阶跃**。这不是 IS 权重能矫正的——IS 权重矫正 action 分布，而 KL reward 嵌入在 reward signal 中直接影响 Advantage。

**在 Decoupled 模式下不存在此问题**：`old_log_prob` 由训练引擎一致计算。

---

## 7. Staleness 评估机制：有界队列与版本过滤的缺失（Explore Agent 贡献）{#7-staleness}

### 7.1 版本追踪的三层数据流

| 层级 | 数据结构 | 字段 | 来源 |
|------|---------|------|------|
| Token 级 | `rollout_log_probs` 张量 | 无版本标记 | 生成时拼接 |
| 轨迹级 | `AgentLoopOutput.extra_fields` | `param_version_start/end` | PartialSingleTurnAgentLoop |
| 样本级 | `RolloutSample` dataclass | `param_version`, `param_version_start/end` 列表 | FullyAsyncRollouter |
| Batch 级 | `meta_info` dict | `rollout_param_versions`, `trajectory_param_versions` | `assemble_batch_from_rollout_samples` |
| 模型权重级 | `TokenOutput.extra_info` | `global_steps`, `min/max_global_steps` | vLLMHttpServer |

### 7.2 MessageQueue：无版本感知的 FIFO

**Explore Agent 确认**：`MessageQueue`（`message_queue.py:67-96`）的 `put_sample` 方法**不检查 `param_version`**：

```python
async def put_sample(self, sample: Any, param_version: int) -> bool:
    async with self._lock:
        if len(self.queue) >= self.max_queue_size:
            self.queue.popleft()  # 满了就丢弃最旧的（不检查版本）
        self.queue.append(sample)  # 无条件接受
```

`param_version` 参数被传入但**未被使用**。Queue 是纯 FIFO，没有版本过滤、版本优先级或版本淘汰机制。

### 7.3 Rollouter 端的流量控制（非质量控制）

`_should_pause_generation`（`fully_async_rollouter.py:727-749`）有两个停止条件：

```python
# 条件 1：Queue 满
if queue_size >= self.max_queue_size:
    return True

# 条件 2：过时样本数超限
if self.staleness_samples >= self.max_required_samples:
    return True
```

**这是流量控制**（控制生产速度以避免过度生产），**不是质量控制**（拒绝低质量/过时数据）。所有生成的样本无论版本都会进入队列。

### 7.4 Trainer 端的 Staleness 统计：仅监控

```python
# fully_async_trainer.py:620-631
stale_count = sum(1 for v in samples_param_versions if self.current_param_version - v >= 1)
self.stale_samples_processed += stale_count
metrics["fully_async/count/stale_samples_processed"] = self.stale_samples_processed
```

`stale_count` 被计算并写入 WandB metrics，但**不用于**：
- 修改 loss 计算
- 调整 IS 权重
- 过滤样本
- 降低学习率

### 7.5 多轮工具调用变体

**Explore Agent 发现**：`AsyncPartialToolAgentLoop`（`experimental/fully_async_policy/agent_loop/partial_tool_agent_loop.py`）继承自 `ToolAgentLoop`，支持多轮工具调用场景的 partial rollout。它通过 `cancellation_event` 在 `GENERATING` 阶段中断，保存当前 turn 的状态，同步参数后恢复。版本追踪逻辑与 `PartialSingleTurnAgentLoop` 相同——有 `param_version_start/end`，但不进入 Loss 计算。

---

## 8. Reward 信号的阶跃问题与 PRM 假设分析 {#8-reward-阶跃}

### 8.1 Outcome Reward：不受影响

verl 的主要 Reward 机制（数学正确率、规则匹配）只看最终结果，不关心生成过程。拼接轨迹的 Outcome Reward 与完整生成的无异。

### 8.2 KL Penalty 的阶跃

在 Bypass 模式下，KL 惩罚在 token k 处出现不连续（见 §6.7）。在 Decoupled 模式下不存在此问题。

### 8.3 PRM 假设分析（Advantage Expert 贡献）

**verl 核心代码不包含 PRM**。但 `recipe/prime/` 目录实现了一个基于 DPO 的 token-level reward model（PRIME），可产生 `rm_scores` 作为逐 token reward。

**假设 PRM 被集成**，对拼接轨迹的影响：

1. **PRM 评估推理链质量**：前缀（$\theta_n$ 风格）和后缀（$\theta_{n+1}$ 风格）的推理风格可能不同
2. **Token k 处的 TD-error 峰值**：
   - PRM reward $r_k$ 可能因风格突变而异常
   - Critic $V(s_k)$ 是平滑的（看到完整上下文）
   - $\delta_k = r_k + \gamma V(s_{k+1}) - V(s_k)$ 在拼接点产生**局部化的 TD-error 尖峰**
3. **GAE 传播**：TD-error 的尖峰通过 $(\gamma\lambda)^l$ 衰减传播，影响拼接点附近的 Advantage 估计

---

## 9. 架构缺口的严重度评估与分级修复建议（Architect Reviewer Agent 贡献）{#9-修复建议}

### 9.1 按模式分级的严重度评估

| 运行模式 | 严重度 | 说明 |
|---------|--------|------|
| Decoupled（无 rollout_correction） | **低** | $\pi_{old}$ 一致，`rollout_log_probs` 仅用于诊断。策略梯度正确。 |
| Decoupled（有 rollout_correction） | **中** | IS 权重从混合版本 `rollout_log_probs` 计算，导致矫正权重失真。 |
| **Bypass Mode** | **高** | `old_log_probs = rollout_log_probs`（混合版本），PPO ratio 本身被污染。 |

### 9.2 已丢失但应保留的元数据

**Architect Reviewer 识别的 5 项缺失元数据**：

| 元数据 | 当前状态 | 影响 |
|--------|---------|------|
| **Per-token 版本向量** `version[t]` | ✗ 不存在 | 无法做 per-token IS 矫正 |
| **拼接点索引 k** | 可用（`len(output.response_ids)`）但**未存储** | 无法区分前缀/后缀 |
| **前缀在新权重下的 logprob** | Re-Prefill 时可计算但**未获取** | 无法统一版本 |
| **多次中断的中间边界** | 仅保留首尾版本，中间边界丢失 | 多次中断的情况无法追踪 |
| **跨策略条件化标记** | ✗ 不存在 | 无法标记"V_{n+1} 在 V_n 上下文上生成"的特殊性 |

### 9.3 四级修复建议（按优先级排序）

#### 级别 1：即时修复（低工作量）

**在 Bypass 模式下保护拼接轨迹**：

```python
# 在 PartialSingleTurnAgentLoop.run() 中，存储拼接点
split_point = len(output.response_ids)  # 已可用，只需存储

# 选项 A：mask 掉前缀
response_mask[:split_point] = 0  # 不从前缀学习

# 选项 B：直接拒绝拼接轨迹
if param_version_start != param_version_end:
    response_mask[:] = 0  # 整条拒绝
```

**实现位置**：`partial_single_turn_agent_loop.py:114-115`，在计算 `response_mask` 时插入判断。

#### 级别 2：短期改进（中等工作量）

**存储拼接点并在 Rollout Correction 中使用**：

1. 在 `AgentLoopOutput.extra_fields` 中添加 `split_point_k: int`
2. 传播到 `non_tensor_batch` 中
3. 在 `rollout_corr_helper.py` 的 `compute_rollout_correction_weights()` 中，对拼接轨迹按段分别计算 IS 权重：

```python
if split_points is not None:
    for i, k in enumerate(split_points):
        if k > 0:  # 是拼接轨迹
            # 前缀段：IS = π_old / π_{V_n}
            is_weights[i, :k] = compute_segment_is(old_lp[i,:k], rollout_lp[i,:k])
            # 后缀段：IS = π_old / π_{V_{n+1}}
            is_weights[i, k:] = compute_segment_is(old_lp[i,k:], rollout_lp[i,k:])
```

#### 级别 3：中期改进（较高工作量）

**Re-Prefill 时获取前缀 logprob，统一版本**：

在 `vLLMHttpServerForPartial._generate_step()` 中，启用 vLLM 的 `prompt_logprobs` 参数，获取 Re-Prefill 时前缀每个 token 在新权重下的 logprob，覆盖旧值：

```python
# vLLM 支持 prompt_logprobs 参数
sampling_params = SamplingParams(
    max_tokens=...,
    logprobs=1,
    prompt_logprobs=1,  # ★ 同时返回 prompt 的 logprob
)
```

这使得整条序列的 `rollout_log_probs` 统一为 V_{n+1} 版本。**代价几乎为零**——Re-Prefill 时前向传播已经做了，logprob 只需额外的 softmax + log 运算。

#### 级别 4：长期架构（高工作量）

**将"生成版本"与"logprob 计算版本"解耦为一等公民概念**：

1. `TokenOutput` 添加 per-token 版本注解
2. `DataProto` 支持 `token_level_version_mask` 张量
3. Policy loss 函数接受版本掩码，按版本段分组计算
4. Advantage 估计器支持版本感知的基线计算

### 9.4 为什么当前设计在实践中有效？

尽管存在理论缺口，verl 的实验结果（128 卡 2.35x 加速，准确率 max 0.3521 vs 同步 0.3573，差距仅 1.5%）表明当前设计足够好。核心原因：

1. **Decoupled Mode 的 $\pi_{old}$ 一致性**是最关键的数学保障——PPO 核心 IS 比率正确
2. **单步更新量小**：$\theta_n$ 和 $\theta_{n+1}$ 差异有限，拼接点不连续性受控
3. **TIS 截断**（默认 2.0）限制任何单个 token 的 IS 权重影响不超过 2x
4. **Rejection Sampling** 拒绝偏离过大的样本（divergence-based 硬信任域）
5. **GRPO/RLOO/REINFORCE++** 等主流 Advantage 估计器不依赖 logprob
6. **Partial Rollout 占比可控**：`partial_ratio` 通常远小于 100%，大部分样本是完整的
7. **Outcome Reward** 对生成过程的非平稳性不敏感

---

## 10. 关键源码索引 {#10-源码索引}

### Logprob 拼接与恢复

| 功能 | 文件 | 行号 | Agent 来源 |
|------|------|------|-----------|
| Token 拼接（路径 A） | `experimental/fully_async_policy/agent_loop/partial_single_turn_agent_loop.py` | 86-114 | Architect |
| Abort→Re-prefill 循环（路径 B） | `experimental/fully_async_policy/agent_loop/agent_loop.py` | 52-128 | Architect |
| vLLM Partial 生成 | `experimental/fully_async_policy/vllm_rollout/vllm_async_server.py` | 95-135 | Architect |
| Cancel 传播链 | `experimental/fully_async_policy/agent_loop/agent_loop.py` | 376-384 | Architect |
| `use_rollout_log_probs` 路径 | `workers/actor/dp_actor.py` | 594-601 | Algorithm #1 |

### IS 比率计算（全部 Loss 函数）

| Loss 函数 | 文件 | 行号 | 敏感度 | Agent 来源 |
|-----------|------|------|--------|-----------|
| `vanilla` (PPO) | `trainer/ppo/core_algos.py` | 1166-1256 | 中 | Algorithm #1 |
| `gspo` | `trainer/ppo/core_algos.py` | 1425-1498 | **高** | Algorithm #1 |
| `geo_mean` | `trainer/ppo/core_algos.py` | 1807-1890 | **高** | Algorithm #1 |
| `cispo` | `trainer/ppo/core_algos.py` | 1893-1951 | 低 | Algorithm #1 |
| Rollout IS 权重 | `trainer/ppo/rollout_corr_helper.py` | 530-598 | - | Algorithm #1 |
| Rejection Sampling | `trainer/ppo/rollout_corr_helper.py` | 156-372 | - | Algorithm #1 |

### Advantage 估计

| 估计器 | 文件 | 行号 | 对拼接敏感度 | Agent 来源 |
|--------|------|------|------------|-----------|
| GAE | `trainer/ppo/core_algos.py` | 214-262 | 中（State 分布） | Advantage |
| GRPO | `trainer/ppo/core_algos.py` | 266-330 | 低（组内同版本） | Advantage |
| RLOO | `trainer/ppo/core_algos.py` | 476-525 | 低（组内同版本） | Advantage |
| REINFORCE++ | `trainer/ppo/core_algos.py` | 582-618 | 中（Batch 归一化） | Advantage |
| OTB | `trainer/ppo/core_algos.py` | 758-873 | 高（IS² 方差） | Advantage |

### Staleness 追踪

| 功能 | 文件 | 行号 | Agent 来源 |
|------|------|------|-----------|
| RolloutSample 版本字段 | `experimental/fully_async_policy/detach_utils.py` | 44-46 | Explore |
| Batch 组装版本统计 | `experimental/fully_async_policy/detach_utils.py` | 167-187 | Explore |
| Trainer 端 Staleness 计数 | `experimental/fully_async_policy/fully_async_trainer.py` | 620-632 | Explore |
| Rollouter 流量控制 | `experimental/fully_async_policy/fully_async_rollouter.py` | 727-749 | Explore |
| MessageQueue（无版本过滤） | `experimental/fully_async_policy/message_queue.py` | 67-96 | Explore |
| 多轮工具调用变体 | `experimental/fully_async_policy/agent_loop/partial_tool_agent_loop.py` | 全文件 | Explore |

### FullyAsyncTrainer 核心路径

| 功能 | 文件 | 行号 | Agent 来源 |
|------|------|------|-----------|
| `_compute_old_log_prob`（MIS 锚定版本） | `experimental/fully_async_policy/fully_async_trainer.py` | 428-448 | Architect |
| `fit_step` 训练循环 | `experimental/fully_async_policy/fully_async_trainer.py` | 374-415 | Explore |
| `_get_samples_from_queue` | `experimental/fully_async_policy/fully_async_trainer.py` | 191-248 | Explore |
| Bypass/Decoupled 模式选择 | `trainer/ppo/ray_trainer.py` | 1379-1419 | Algorithm #1 |
| 参数同步流程 | `experimental/fully_async_policy/param_sync.py` | 103-136 | Architect |

---

## 总结

本文档综合了 4 个专家 Agent 的独立分析结果，从 8 个维度深度解剖了 verl 在 Fully Async Partial Rollout 场景下处理拼接轨迹的算法一致性。

**核心发现**：

1. **Algorithm Expert #1 的遍历结论**：所有 11+ policy loss 函数都是**版本盲**的。`param_version` 元数据在进入 Loss 计算前被完全丢弃。`gspo` 和 `geo_mean` 对拼接尤为敏感（序列级聚合）。

2. **Explore Agent 的流量分析**：MessageQueue 无版本过滤；`staleness_threshold` 是流量控制而非质量控制；版本统计仅用于 WandB 监控。

3. **Advantage Expert 的偏差分类**：识别了 7 种偏差源，其中 State 分布偏移（③）、GRPO 组基线污染（④）、REINFORCE++ Batch 归一化污染（⑤）是算法固有的，IS 权重无法矫正。

4. **Architect Reviewer 的架构评估**：拼接点 k 可用但未存储是最关键的缺口；4 级修复建议从"即时 mask 前缀"到"长期 per-token 版本解耦"提供了清晰的演进路径。

**最终评价**：verl 在 Decoupled Mode + Token-level IS + GRPO/RLOO 的推荐组合下，对拼接轨迹的算法影响是**有限且可控的**。Bypass Mode 下存在数学上的不一致性，但在实践中（单步更新量小、TIS 截断、Rejection Sampling）被有效抑制。实验结果（2.35x 加速，准确率损失 <2%）验证了这种务实策略的有效性。
