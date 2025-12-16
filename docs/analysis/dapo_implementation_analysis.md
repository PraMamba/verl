# DAPO 训练框架实现详解

**面向对象**: 基础设施（Infrastructure）初学者
**最后更新**: 2025-11-27

---

## 目录

1. [什么是 DAPO？](#1-什么是-dapo)
2. [DAPO 的核心创新](#2-dapo-的核心创新)
3. [整体架构设计](#3-整体架构设计)
4. [核心组件实现](#4-核心组件实现)
5. [训练流程详解](#5-训练流程详解)
6. [关键代码解析](#6-关键代码解析)
7. [配置与使用](#7-配置与使用)
8. [总结](#8-总结)

---

## 1. 什么是 DAPO？

### 1.1 DAPO 简介

**DAPO** (Decoupled Clip and Dynamic Sampling Policy Optimization) 是一种改进的强化学习算法，专门用于训练大语言模型。它在传统 PPO 算法的基础上做了三个重要改进：

- **解耦的裁剪范围** (Decoupled Clip)：使用不同的上下界来裁剪策略比率
- **动态采样** (Dynamic Sampling)：智能过滤无效训练样本
- **Token 级别的损失聚合** (Token-level Loss)：更细粒度的优化目标

### 1.2 与其他算法的对比

让我们用一个简单的表格来对比 DAPO 和其他常见算法：

| 算法 | Advantage 计算 | 裁剪策略 | 样本过滤 | Critic 网络 |
|------|--------------|---------|---------|------------|
| **PPO** | GAE | 对称裁剪 (ε=0.2) | 无 | 需要 |
| **GRPO** | 组内归一化 | 无裁剪 | 无 | 不需要 |
| **DAPO** | GRPO 方式 | 非对称裁剪 | 动态过滤 | 可选 |

**关键区别**：
- DAPO 基于 GRPO 的优势函数（不需要 Critic），但加入了**非对称裁剪**
- DAPO 使用**动态采样**来过滤"太简单"或"太难"的样本

---

## 2. DAPO 的核心创新

### 2.1 非对称裁剪 (Clip-Higher)

**传统 PPO 的裁剪**：
```python
# 对称裁剪：ratio 被限制在 [1-ε, 1+ε] 范围内
ratio = π_new / π_old
clipped_ratio = clamp(ratio, 1-0.2, 1+0.2)  # ε=0.2
```

**DAPO 的非对称裁剪**：
```python
# 非对称裁剪：上下界不同
ratio = π_new / π_old
clipped_ratio = clamp(ratio, 1-ε_low, 1+ε_high)  # 例如 ε_low=0.2, ε_high=0.28
```

**为什么这样做？**

想象你在训练一个数学题求解模型：
- 当模型输出**正确答案**时（advantage > 0），我们希望**更激进地增强**这个行为 → `ε_high` 更大
- 当模型输出**错误答案**时（advantage < 0），我们要**谨慎地抑制**，避免过度修正 → `ε_low` 较小

这就像教小孩：做对了要大力表扬，做错了要温和纠正。

### 2.2 动态采样 (Dynamic Sampling with Group Filtering)

**核心思想**：过滤掉"学不到东西"的样本组。

**什么是样本组？**
- 对同一个问题（prompt），模型生成 N 个不同的回答（responses）
- 这 N 个回答组成一个"组"

**过滤规则**：
```python
# 计算组内所有回答的得分标准差
std_score = std([score_1, score_2, ..., score_N])

# 如果标准差为 0，说明所有回答都一样好（或一样差）
if std_score == 0:
    过滤掉这个组  # 学不到对比信息
else:
    保留这个组  # 可以学习哪个回答更好
```

**实际例子**：

```
问题: 1+1=?
回答组 A:
  - 回答1: "2" (正确) ✓
  - 回答2: "2" (正确) ✓
  - 回答3: "2" (正确) ✓
  - 回答4: "2" (正确) ✓
  得分: [1, 1, 1, 1] → std=0 → 过滤掉（太简单，学不到东西）

回答组 B:
  - 回答1: "2" (正确) ✓
  - 回答2: "3" (错误) ✗
  - 回答3: "4" (错误) ✗
  - 回答4: "2" (正确) ✓
  得分: [1, 0, 0, 1] → std>0 → 保留（可以学习区分好坏）
```

### 2.3 Token 级别的损失聚合

**传统方式 (Sequence-level)**：
```python
# 先对每个序列求和，再求平均
seq_loss = sum(token_losses_in_sequence)  # 每个序列一个 loss
final_loss = mean(seq_loss)  # 对所有序列求平均
```

**DAPO 方式 (Token-level)**：
```python
# 直接对所有 token 求平均
final_loss = mean(all_token_losses)  # 所有 token 一视同仁
```

**区别**：
- Sequence-level：长序列和短序列的权重相同
- Token-level：每个 token 权重相同，长序列的总权重更大

**为什么 Token-level 更好？**
- 数学推理任务中，长回答通常包含更多推理步骤
- Token-level 让模型更关注这些复杂的长推理过程

### 2.4 过长惩罚 (Overlong Penalty)

避免模型生成过长但无意义的回答。

**实现机制**：
```python
# 配置
max_response_length = 20480  # 最大允许长度
buffer_len = 4096            # 缓冲区长度
penalty_factor = 1.0         # 惩罚系数

# 惩罚计算
expected_len = max_response_length - buffer_len  # 16384
if response_length > expected_len:
    exceed_len = response_length - expected_len
    # 线性增长的惩罚
    penalty = -min(exceed_len / buffer_len * penalty_factor, penalty_factor)
    final_reward = original_reward + penalty
```

**示意图**：
```
Reward
  ^
  |     _______________  (无惩罚区)
  |    /               \
  |   /                 \___  (惩罚区：线性下降)
  |  /                      \
  +-----|---------|---------|----> Response Length
      0    16384    20480  (max)
           ^         ^
       期望长度   缓冲区结束
```

---

## 3. 整体架构设计

### 3.1 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Ray Cluster (分布式环境)                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              TaskRunner (主控节点)                     │   │
│  │  - 初始化配置                                          │   │
│  │  - 创建 Worker Groups                                 │   │
│  │  │  创建 RayDAPOTrainer                               │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           RayDAPOTrainer (训练协调器)                  │   │
│  │                                                        │   │
│  │  核心方法:                                             │   │
│  │  • fit()              - 主训练循环                     │   │
│  │  • compute_kl_related_metrics()  - 计算 KL 指标       │   │
│  │  • _validate()        - 验证                          │   │
│  │  • _save_checkpoint() - 保存检查点                     │   │
│  └──────────────────────────────────────────────────────┘   │
│           │              │              │              │      │
│           ▼              ▼              ▼              ▼      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────┐│
│  │ ActorRollout │ │   Critic     │ │  RefPolicy   │ │ RM  ││
│  │ WorkerGroup  │ │ WorkerGroup  │ │ WorkerGroup  │ │ WG  ││
│  │              │ │              │ │              │ │     ││
│  │ • 生成回答    │ │ • 计算 Value  │ │ • 计算参考   │ │奖励 ││
│  │ • 计算 logp  │ │ • 更新 Critic │ │   log_prob   │ │打分 ││
│  │ • 更新 Actor │ │              │ │              │ │     ││
│  └──────────────┘ └──────────────┘ └──────────────┘ └─────┘│
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            DAPORewardManager (奖励管理器)              │   │
│  │                                                        │   │
│  │  • 解码生成的文本                                       │   │
│  │  • 调用 compute_score 评估答案                         │   │
│  │  • 应用过长惩罚                                         │   │
│  │  • 返回 reward tensor                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 数据流

```
输入数据 (Prompt)
    │
    ▼
┌────────────────┐
│  1. Rollout    │  生成 N 个候选回答
│  (生成阶段)     │  → responses, log_probs
└────────────────┘
    │
    ▼
┌────────────────┐
│  2. Reward     │  评估每个回答的质量
│  (奖励计算)     │  → reward_scores
└────────────────┘
    │
    ▼
┌────────────────┐
│  3. Filter     │  [DAPO 特色] 动态过滤
│  (动态采样)     │  → filtered_batch
└────────────────┘
    │
    ▼
┌────────────────┐
│  4. Advantage  │  计算优势函数 (GRPO 方式)
│  (优势计算)     │  → advantages
└────────────────┘
    │
    ▼
┌────────────────┐
│  5. Update     │  [DAPO 特色] 非对称裁剪
│  (策略更新)     │  → updated_policy
└────────────────┘
```

---

## 4. 核心组件实现

### 4.1 DAPORewardManager

**位置**: `verl/workers/reward_manager/dapo.py`

**核心职责**: 计算每个生成回答的奖励分数

**关键代码解析**:

```python
@register("dapo")
class DAPORewardManager(AbstractRewardManager):
    def __init__(
        self,
        tokenizer,
        num_examine,           # 打印多少个样本用于调试
        compute_score=None,    # 自定义评分函数
        reward_fn_key="data_source",  # 数据来源标识
        max_resp_len=None,     # 最大回答长度
        overlong_buffer_cfg=None,  # 过长惩罚配置
    ):
        # 如果没有提供自定义评分函数，使用默认的
        self.compute_score = compute_score or default_compute_score
        # ... 其他初始化
```

**核心方法 `__call__`**:

```python
def __call__(self, data: DataProto, return_dict: bool = False):
    # 1. 初始化 reward tensor (全零)
    reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

    # 2. 遍历每个样本
    for i in range(len(data)):
        data_item = data[i]

        # 3. 解码 prompt 和 response
        prompt_ids = data_item.batch["prompts"]
        response_ids = data_item.batch["responses"]

        # 提取有效部分 (去除 padding)
        valid_response_ids = response_ids[:valid_response_length]

        # 解码为文本
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        # 4. 获取 ground truth (正确答案)
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

        # 5. 计算得分
        result = self.compute_score(
            data_source=data_source,      # 数据集类型 (如 "math_dapo")
            solution_str=response_str,    # 模型的回答
            ground_truth=ground_truth,    # 正确答案
            extra_info=extra_info,        # 额外信息
        )

        # 6. 提取分数
        if isinstance(result, dict):
            score = result["score"]  # 可能包含额外信息
        else:
            score = result

        reward = score  # 基础奖励

        # 7. [DAPO 特色] 应用过长惩罚
        if self.overlong_buffer_cfg.enable:
            overlong_buffer_len = self.overlong_buffer_cfg.len  # 4096
            expected_len = self.max_resp_len - overlong_buffer_len  # 16384
            exceed_len = valid_response_length - expected_len

            if exceed_len > 0:
                # 线性惩罚
                penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * penalty_factor, 0)
                reward += overlong_reward  # 减少奖励

        # 8. 将奖励放在序列的最后一个有效 token 位置
        reward_tensor[i, valid_response_length - 1] = reward

    return reward_tensor
```

**关键点解释**:

1. **奖励是稀疏的**: 只在序列的最后一个 token 有非零奖励
   ```python
   reward_tensor[i, valid_response_length - 1] = reward
   # 其他位置都是 0
   ```

2. **为什么放在最后一个 token？**
   - 这是"结果监督" (Outcome Supervision) 的做法
   - 只有完整的回答才能判断对错
   - 中间的 token 没有明确的对错标准

3. **过长惩罚的设计**:
   ```
   response_length = 18000
   expected_len = 16384
   exceed_len = 18000 - 16384 = 1616

   penalty = -1616 / 4096 * 1.0 = -0.395

   final_reward = 1.0 + (-0.395) = 0.605
   ```

### 4.2 RayDAPOTrainer

**位置**: `recipe/dapo/dapo_ray_trainer.py`

**继承关系**: `RayDAPOTrainer` → `RayPPOTrainer`

**DAPO 特有的修改**:

#### 修改 1: `compute_kl_related_metrics`

这个方法重写了父类的实现，去掉了一些不必要的计算：

```python
def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
    # 1. 计算 response mask (标记哪些位置是有效 token)
    batch.batch["response_mask"] = compute_response_mask(batch)

    # 2. 重新计算 old_log_prob (当前策略的 log 概率)
    with marked_timer("old_log_prob", timing_raw, "blue"):
        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)

        # 计算 entropy (用于监控)
        entropys = old_log_prob.batch["entropys"]
        response_masks = batch.batch["response_mask"]
        entropy_agg = agg_loss(
            loss_mat=entropys,
            loss_mask=response_masks,
            loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode
        )
        metrics.update({"actor/entropy": entropy_agg.detach().item()})

        # 合并到 batch
        old_log_prob.batch.pop("entropys")
        batch = batch.union(old_log_prob)

    # 3. 如果使用参考策略，计算参考 log_prob
    if self.use_reference_policy:
        with marked_timer("ref", timing_raw, "olive"):
            if not self.ref_in_actor:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
            else:
                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)

    return batch
```

**为什么要重写这个方法？**
- DAPO 基于 GRPO，不需要 Critic 网络
- 但仍然可以选择性使用 KL 散度作为正则化
- 这个方法确保必要的概率计算都完成了

#### 修改 2: `fit` 方法中的动态采样

**核心逻辑** (位于 `fit()` 方法的训练循环中):

```python
# 在主训练循环中
for batch_dict in self.train_dataloader:
    # ... 生成 responses ...

    # ========== DAPO 特色：动态采样 ==========
    if not self.config.algorithm.filter_groups.enable:
        # 不启用过滤，直接使用
        batch = new_batch
    else:
        # 启用动态过滤
        metric_name = self.config.algorithm.filter_groups.metric  # 例如 "acc"

        # 1. 计算每个回答的指标值
        if metric_name == "seq_final_reward":
            new_batch.non_tensor_batch["seq_final_reward"] = (
                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
            )
        elif metric_name == "seq_reward":
            new_batch.non_tensor_batch["seq_reward"] = (
                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
            )

        # 2. 按 prompt 分组，收集每组的指标值
        prompt_uid2metric_vals = defaultdict(list)
        for uid, metric_val in zip(
            new_batch.non_tensor_batch["uid"],
            new_batch.non_tensor_batch[metric_name]
        ):
            prompt_uid2metric_vals[uid].append(metric_val)

        # 3. 计算每组的标准差
        prompt_uid2metric_std = {}
        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

        # 4. 过滤：只保留标准差 > 0 的组
        kept_prompt_uids = [
            uid for uid, std in prompt_uid2metric_std.items()
            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
        ]
        num_prompt_in_batch += len(kept_prompt_uids)

        # 5. 过滤样本
        kept_traj_idxs = []
        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
            if traj_from_prompt_uid in kept_prompt_uids:
                kept_traj_idxs.append(idx)

        new_batch = new_batch[kept_traj_idxs]
        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

        # 6. 如果过滤后样本不足，继续生成
        prompt_bsz = self.config.data.train_batch_size  # 期望的 batch size
        if num_prompt_in_batch < prompt_bsz:
            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches

            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                print(f"{num_gen_batches=}. Keep generating...")
                self.gen_steps += 1
                continue  # 跳到下一个生成循环
            else:
                raise ValueError("Generated too many batches. Data might be too difficult.")
        else:
            # 样本足够，对齐 batch size
            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
            batch = batch[:traj_bsz]

    # ... 继续训练更新 ...
```

**动态采样流程图**:

```
开始
  │
  ▼
生成 gen_batch_size 个样本
  │
  ▼
计算每个样本的 metric (如 acc)
  │
  ▼
按 prompt 分组
  │
  ▼
计算每组的标准差
  │
  ▼
过滤掉 std=0 的组
  │
  ▼
样本数够吗？
  │
  ├─ 是 ──> 继续训练
  │
  └─ 否 ──> 继续生成 ──> (循环)
                 │
                 ▼
            超过最大次数？
                 │
                 ├─ 是 ──> 报错
                 └─ 否 ──> 继续生成
```

**为什么这样设计？**

假设 `train_batch_size=512`, `gen_batch_size=1536`:
- 每次生成 1536 个样本
- 过滤后可能只剩 400 个有效样本
- 不够 512，继续生成
- 累积到 512 后才开始训练

这样确保每个训练 batch 都是"高质量"的样本。

### 4.3 非对称裁剪的实现

虽然 `RayDAPOTrainer` 没有重写更新逻辑，但非对称裁剪是在配置和底层实现中体现的。

**配置文件** (`recipe/dapo/config/dapo_trainer.yaml`):
```yaml
actor_rollout_ref:
  actor:
    clip_ratio_low: 0.2      # ε_low
    clip_ratio_high: 0.28    # ε_high
```

**实际使用** (在 Actor Worker 的更新代码中):
```python
# 这部分在 verl/workers/fsdp_workers.py 或 megatron_workers.py 中
def update_actor(self, data: DataProto):
    # ... 前向传播 ...

    # 计算策略比率
    ratio = torch.exp(new_log_probs - old_log_probs)

    # 计算优势
    advantages = data.batch["advantages"]

    # [DAPO] 非对称裁剪
    cliprange_low = self.config.clip_ratio_low    # 0.2
    cliprange_high = self.config.clip_ratio_high  # 0.28

    # 两种损失
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio,
        1 - cliprange_low,   # 下界: 0.8
        1 + cliprange_high   # 上界: 1.28
    )

    # 取最大值 (最悲观的估计)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # [DAPO] Token-level 聚合
    if loss_agg_mode == "token-mean":
        loss = masked_mean(pg_losses, response_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(pg_losses * response_mask, dim=-1)
        loss = torch.mean(seq_losses)

    # ... 反向传播和优化 ...
```

**裁剪效果对比**:

```
Advantage > 0 (好的行为):
  传统 PPO: ratio ∈ [0.8, 1.2]  → 增强幅度最多 20%
  DAPO:     ratio ∈ [0.8, 1.28] → 增强幅度最多 28% ✓ 更激进

Advantage < 0 (坏的行为):
  传统 PPO: ratio ∈ [0.8, 1.2]  → 抑制幅度最多 20%
  DAPO:     ratio ∈ [0.8, 1.28] → 抑制幅度最多 20% ✓ 同样谨慎
```

---

## 5. 训练流程详解

### 5.1 完整训练循环

让我们跟踪一个完整的训练步骤：

```
Step 0: 初始化
├─ 加载预训练模型 (如 Qwen2.5-7B)
├─ 创建 Worker Groups (Actor, Critic, RefPolicy, RM)
├─ 加载训练数据 (DAPO-Math-17k)
└─ 设置配置参数

Step 1: 数据加载
├─ 从 dataloader 获取一个 batch
├─ Batch 包含: {prompts, ground_truth, data_source, ...}
└─ 例如: "Solve: x^2 + 2x + 1 = 0"

Step 2: Rollout (生成阶段)
├─ 使用 ActorRollout Worker Group
├─ 对每个 prompt 生成 N 个回答 (如 N=4)
├─ 生成时记录 log_probs
└─ 输出: {responses, rollout_log_probs}

Step 3: Reward 计算
├─ 调用 DAPORewardManager
├─ 解码每个 response
├─ 与 ground_truth 比对
├─ 计算 accuracy 或其他指标
├─ 应用过长惩罚
└─ 输出: {token_level_scores}

Step 4: [DAPO] 动态过滤
├─ 按 prompt 分组
├─ 计算每组的 metric 标准差
├─ 过滤 std=0 的组
├─ 如果样本不足:
│   ├─ 重复 Step 1-4
│   └─ 累积样本直到足够
└─ 输出: filtered_batch

Step 5: KL 相关计算
├─ 重新计算当前策略的 log_probs (old_log_probs)
├─ 如果使用参考策略，计算 ref_log_probs
├─ 计算 KL 散度 (可选)
└─ 输出: {old_log_probs, ref_log_probs}

Step 6: Advantage 计算
├─ 使用 GRPO 方式
├─ 按 prompt 分组
├─ 组内: advantage = score - mean(scores_in_group)
├─ 可选归一化: advantage /= std(scores_in_group)
└─ 输出: {advantages}

Step 7: Critic 更新 (可选)
├─ 如果使用 Critic 网络
├─ 计算 value loss
└─ 更新 Critic 参数

Step 8: Actor 更新
├─ 计算策略比率: ratio = exp(new_log_probs - old_log_probs)
├─ [DAPO] 非对称裁剪
├─ 计算 policy gradient loss
├─ [DAPO] Token-level 聚合
├─ 反向传播
└─ 更新 Actor 参数

Step 9: 验证和保存
├─ 定期在验证集上测试
├─ 保存 checkpoint
└─ 记录 metrics

Step 10: 重复
└─ 回到 Step 1，直到达到总训练步数
```

### 5.2 时间线视图

```
时间轴：
0s     2s         10s        11s    13s        14s    15s
│      │          │          │      │          │      │
├──────┤          │          │      │          │      │
│ Gen  │          │          │      │          │      │
│      │          │          │      │          │      │
│      ├──────────┤          │      │          │      │
│      │  Reward  │          │      │          │      │
│      │          │          │      │          │      │
│      │          ├──────────┤      │          │      │
│      │          │ Filter & │      │          │      │
│      │          │  Adv     │      │          │      │
│      │          │          │      │          │      │
│      │          │          ├──────┤          │      │
│      │          │          │Update│          │      │
│      │          │          │Critic│          │      │
│      │          │          │      │          │      │
│      │          │          │      ├──────────┤      │
│      │          │          │      │  Update  │      │
│      │          │          │      │  Actor   │      │
│      │          │          │      │          │      │
└──────┴──────────┴──────────┴──────┴──────────┴──────┘
```

**各阶段耗时** (7B 模型, batch_size=512):
- Generation: ~2s (最耗时，GPU 推理)
- Reward: ~8s (CPU 评估，可能调用外部评分器)
- Filter & Advantage: ~1s (CPU 计算)
- Update Critic: ~2s (小网络，快)
- Update Actor: ~1s (主要是前向传播)

### 5.3 关键张量的形状变化

跟踪数据的形状变化：

```python
# 输入 batch
prompts: [batch_size, prompt_length]
# 例如: [512, 1024]

# 生成阶段 (重复 N 次)
responses: [batch_size * N, response_length]
# 例如: [512*4=2048, 2048]

# Reward 计算
token_level_scores: [batch_size * N, response_length]
# 例如: [2048, 2048]
# 大部分位置是 0，只有最后一个 valid token 有分数

# 过滤后 (假设过滤掉 25%)
filtered_batch_size = batch_size * N * 0.75
# 例如: [1536, 2048]

# Advantage 计算
advantages: [filtered_batch_size, response_length]
# 例如: [1536, 2048]
# 广播到所有 token 位置

# Loss 计算
pg_loss: scalar
# Token-level mean: sum(all_losses) / sum(all_valid_tokens)
```

---

## 6. 关键代码解析

### 6.1 GRPO Advantage 计算

**位置**: `verl/trainer/ppo/core_algos.py:compute_grpo_outcome_advantage`

```python
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,  # [bs*N, seq_len]
    response_mask: torch.Tensor,         # [bs*N, seq_len]
    index: np.ndarray,                   # [bs*N], 标记哪些样本属于同一个 prompt
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
):
    # 1. 计算每个序列的总分数
    scores = token_level_rewards.sum(dim=-1)  # [bs*N]

    # 2. 按 prompt 分组
    id2score = defaultdict(list)
    for i in range(len(scores)):
        id2score[index[i]].append(scores[i])

    # 3. 计算每组的均值和标准差
    id2mean = {}
    id2std = {}
    for idx in id2score:
        if len(id2score[idx]) == 1:
            # 只有一个样本，无法计算 std
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        else:
            scores_tensor = torch.stack(id2score[idx])
            id2mean[idx] = torch.mean(scores_tensor)
            id2std[idx] = torch.std(scores_tensor)

    # 4. 归一化：(score - mean) / std
    for i in range(len(scores)):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            # Dr.GRPO 风格：不除以 std
            scores[i] = scores[i] - id2mean[index[i]]

    # 5. 广播到所有 token
    scores = scores.unsqueeze(-1) * response_mask  # [bs*N, seq_len]

    return scores, scores  # advantages, returns
```

**举例说明**:

```python
# 假设一个 prompt 有 3 个回答
scores = [0.8, 0.2, 1.0]  # 3 个回答的得分

# 计算均值和标准差
mean = (0.8 + 0.2 + 1.0) / 3 = 0.667
std = sqrt(((0.8-0.667)^2 + (0.2-0.667)^2 + (1.0-0.667)^2) / 3) = 0.327

# 归一化
adv_1 = (0.8 - 0.667) / 0.327 = 0.407   # 略好
adv_2 = (0.2 - 0.667) / 0.327 = -1.428  # 明显差
adv_3 = (1.0 - 0.667) / 0.327 = 1.019   # 明显好
```

**为什么这样计算？**
- 组内比较：只和同一个 prompt 的其他回答比
- 相对优势：不是绝对的好坏，而是"比平均水平好多少"
- 归一化：确保不同难度的问题有相似的梯度尺度

### 6.2 动态采样的实现细节

```python
# 1. 为每个 prompt 生成唯一 ID
import uuid
new_batch.non_tensor_batch["uid"] = np.array(
    [str(uuid.uuid4()) for _ in range(len(new_batch.batch))],
    dtype=object
)

# 2. 重复 UID 以匹配 rollout 的重复次数
# 如果 rollout.n=4，每个 prompt 生成 4 个回答
# 那么同一个 prompt 的 4 个回答会有相同的 uid
new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

# 例如:
# 原始: [prompt_1, prompt_2]
# uid:  [uid_1,    uid_2]
#
# repeat(n=4, interleave=True):
# [prompt_1, prompt_1, prompt_1, prompt_1, prompt_2, prompt_2, prompt_2, prompt_2]
# [uid_1,    uid_1,    uid_1,    uid_1,    uid_2,    uid_2,    uid_2,    uid_2]

# 3. 计算每个回答的指标
metric_vals = new_batch.non_tensor_batch["acc"]  # 例如 [1, 0, 1, 1, 0, 0, 0, 0]

# 4. 分组统计
prompt_uid2metric_vals = defaultdict(list)
for uid, metric_val in zip(uids, metric_vals):
    prompt_uid2metric_vals[uid].append(metric_val)

# 结果:
# uid_1 -> [1, 0, 1, 1]  → std = 0.5 > 0 ✓ 保留
# uid_2 -> [0, 0, 0, 0]  → std = 0   ✗ 过滤

# 5. 过滤样本
kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0]
# kept_prompt_uids = [uid_1]

kept_traj_idxs = []
for idx, traj_uid in enumerate(uids):
    if traj_uid in kept_prompt_uids:
        kept_traj_idxs.append(idx)
# kept_traj_idxs = [0, 1, 2, 3]  (uid_1 的 4 个样本)

new_batch = new_batch[kept_traj_idxs]
# 过滤后只保留 uid_1 的 4 个样本
```

### 6.3 Token-level Loss 聚合

**位置**: `verl/trainer/ppo/core_algos.py:agg_loss`

```python
def agg_loss(loss_mat, loss_mask, loss_agg_mode):
    """
    Args:
        loss_mat: [bs, seq_len] - 每个 token 的 loss
        loss_mask: [bs, seq_len] - 哪些 token 是有效的
        loss_agg_mode: str - 聚合方式
    """
    if loss_agg_mode == "token-mean":
        # Token-level: 所有有效 token 的平均
        loss = masked_mean(loss_mat, loss_mask)
        # = sum(loss_mat * loss_mask) / sum(loss_mask)

    elif loss_agg_mode == "seq-mean-token-sum":
        # 先对每个序列求和，再求平均
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # [bs]
        loss = torch.mean(seq_losses)

    elif loss_agg_mode == "seq-mean-token-mean":
        # 每个序列内求平均，再对序列求平均
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        loss = torch.mean(seq_losses)

    return loss
```

**对比三种方式**:

假设有 2 个序列：
- 序列 1: 长度 100, loss=[0.5, 0.5, ..., 0.5] (100个)
- 序列 2: 长度 10,  loss=[1.0, 1.0, ..., 1.0] (10个)

```python
# 方式 1: token-mean
total_tokens = 100 + 10 = 110
total_loss = 100*0.5 + 10*1.0 = 60
final_loss = 60 / 110 = 0.545

# 方式 2: seq-mean-token-sum
seq1_loss = 100 * 0.5 = 50
seq2_loss = 10 * 1.0 = 10
final_loss = (50 + 10) / 2 = 30

# 方式 3: seq-mean-token-mean
seq1_loss = 50 / 100 = 0.5
seq2_loss = 10 / 10 = 1.0
final_loss = (0.5 + 1.0) / 2 = 0.75
```

**DAPO 为什么选择 token-mean？**
- 长序列通常对应更复杂的推理
- Token-level 让长序列有更大的权重
- 更符合数学推理任务的特点

---

## 7. 配置与使用

### 7.1 核心配置项

**DAPO 完整配置示例**:

```yaml
# recipe/dapo/config/dapo_trainer.yaml

# === 数据配置 ===
data:
  train_batch_size: 512        # 每个训练步骤的 prompt 数量
  gen_batch_size: 1536         # 每次生成的 prompt 数量 (用于动态采样)
  max_prompt_length: 1024      # 最大 prompt 长度
  max_response_length: 20480   # 最大 response 长度 (16384 + 4096 buffer)

# === 算法配置 ===
algorithm:
  adv_estimator: grpo          # 使用 GRPO 方式计算 advantage
  norm_adv_by_std_in_grpo: true

  # [DAPO 特色 1] 动态采样
  filter_groups:
    enable: true               # 启用过滤
    metric: acc                # 过滤依据: acc / score / seq_reward / seq_final_reward
    max_num_gen_batches: 10    # 最多生成多少次 (0=无限制)

  use_kl_in_reward: false      # 是否在 reward 中加入 KL 惩罚

# === Actor 配置 ===
actor_rollout_ref:
  rollout:
    n: 4                       # 每个 prompt 生成多少个回答
    temperature: 1.0
    top_k: 50
    top_p: 0.95

  actor:
    # [DAPO 特色 2] 非对称裁剪
    clip_ratio_low: 0.2        # ε_low
    clip_ratio_high: 0.28      # ε_high

    # [DAPO 特色 3] Token-level 聚合
    loss_agg_mode: "token-mean"  # / "seq-mean-token-sum" / "seq-mean-token-mean"

    entropy_coeff: 0.0         # Entropy 正则化系数

# === Reward 配置 ===
reward_model:
  enable: false                # DAPO 通常不用 RM，使用规则评分
  reward_manager: dapo         # 使用 DAPORewardManager

  # [DAPO 特色 4] 过长惩罚
  overlong_buffer:
    enable: true
    len: 4096                  # 缓冲区长度
    penalty_factor: 1.0        # 惩罚系数
    log: true                  # 是否记录 log

# === Critic 配置 ===
critic:
  enable: false                # DAPO 基于 GRPO，不需要 Critic
  strategy: fsdp               # 如果启用: fsdp / megatron

# === 训练配置 ===
trainer:
  total_epochs: 1
  total_training_steps: 1000
  save_freq: 100               # 每 100 步保存一次
  test_freq: 50                # 每 50 步验证一次
  logger: wandb                # wandb / tensorboard / console
```

### 7.2 启动训练

**本地单机启动**:

```bash
cd /path/to/verl

# 启动 Ray 集群
ray start --head --port=6379

# 运行训练
python recipe/dapo/main_dapo.py \
    --config-name dapo_trainer \
    data.train_batch_size=512 \
    actor_rollout_ref.actor.clip_ratio_high=0.28
```

**Ray 集群启动**:

```bash
# 1. 在 Ray 集群上准备数据
bash recipe/dapo/prepare_dapo_data.sh

# 2. 从本地机器提交任务
export RAY_ADDRESS="http://your-ray-cluster:8265"
export WORKING_DIR="${PWD}"
export RUNTIME_ENV="./recipe/dapo/runtime_env.yaml"

bash recipe/dapo/run_dapo_qwen2.5_32b.sh
```

### 7.3 关键配置的影响

**1. `filter_groups.enable`**

```yaml
# 不启用过滤
filter_groups.enable: false
→ 训练速度快，但可能包含无效样本
→ 适合: 数据质量高、难度适中的任务

# 启用过滤
filter_groups.enable: true
→ 训练速度慢 (可能需要多次生成)，但样本质量高
→ 适合: 数学推理等高难度任务
```

**2. `clip_ratio_high`**

```yaml
# 保守设置
clip_ratio_high: 0.2
→ 更新步长小，训练稳定但慢

# 激进设置
clip_ratio_high: 0.3
→ 更新步长大，训练快但可能不稳定
```

**3. `loss_agg_mode`**

```yaml
# Token-level
loss_agg_mode: "token-mean"
→ 长序列权重大，适合推理任务

# Sequence-level
loss_agg_mode: "seq-mean-token-sum"
→ 所有序列权重相同，适合对话任务
```

### 7.4 监控指标

训练过程中需要关注的指标：

```python
# Reward 相关
"reward/mean"              # 平均奖励
"reward/max"               # 最大奖励
"reward/min"               # 最小奖励
"reward/acc"               # 准确率 (如果是分类任务)

# 动态采样相关
"train/num_gen_batches"    # 每个训练步骤生成了多少次
"filter/kept_ratio"        # 保留的样本比例

# Policy 相关
"actor/loss"               # Actor loss
"actor/entropy"            # 输出的熵 (多样性指标)
"actor/ratio_mean"         # 策略比率的均值
"actor/ratio_clipped"      # 被裁剪的比率

# Advantage 相关
"adv/mean"                 # Advantage 均值 (应该接近 0)
"adv/std"                  # Advantage 标准差
"adv/max"                  # 最大 advantage
"adv/min"                  # 最小 advantage

# 过长惩罚相关
"overlong/ratio"           # 过长样本的比例
"overlong/penalty_mean"    # 平均惩罚值
```

---

## 8. 总结

### 8.1 DAPO 的核心优势

1. **更高效的学习**: 动态采样过滤无效样本，每个训练步骤都在"有价值"的数据上学习

2. **更激进的优化**: 非对称裁剪允许对正确行为更大的强化

3. **更好的长文本处理**: Token-level 聚合让模型更关注复杂的长推理链

4. **防止过长**: 过长惩罚避免模型通过"水字数"来获得高分

### 8.2 适用场景

DAPO 特别适合：
- 数学推理 (如 AIME)
- 代码生成
- 复杂问答
- 需要长推理链的任务

DAPO 不太适合：
- 简单的对话任务 (动态采样可能过滤掉太多样本)
- 需要多样性的生成任务 (过滤会降低多样性)
- 数据规模很小的情况 (过滤后样本更少)

### 8.3 与其他算法的组合

DAPO 的技巧可以和其他方法组合：

```yaml
# DAPO + Rollout Correction
algorithm:
  rollout_correction:
    enable: true              # 处理 rollout 和 training 的分布偏移
  filter_groups:
    enable: true              # DAPO 的动态采样

# DAPO + Entropy Regularization
actor_rollout_ref:
  actor:
    entropy_coeff: 0.01       # 增加探索

# DAPO + KL Constraint
algorithm:
  use_kl_in_reward: true      # 防止过度偏离原模型
```

### 8.4 实现要点总结

对于 infra 初学者，记住这些要点：

1. **DAPO ≈ GRPO + 三个改进**
   - 基础还是 GRPO (组内归一化 advantage)
   - 加上非对称裁剪、动态采样、Token-level 聚合

2. **继承而非重写**
   - `RayDAPOTrainer` 继承自 `RayPPOTrainer`
   - 只重写了少数关键方法
   - 大部分逻辑复用

3. **配置驱动**
   - 很多特性是通过配置开关控制的
   - 不需要改代码就能尝试不同设置

4. **模块化设计**
   - RewardManager 负责评分
   - Trainer 负责训练循环
   - Core Algos 负责 advantage 计算
   - 各司其职，易于扩展

5. **分布式友好**
   - 基于 Ray 的分布式架构
   - 支持多节点训练
   - Worker Group 抽象层

### 8.5 进一步学习

想要深入理解，建议阅读：

1. **源码**:
   - `verl/workers/reward_manager/dapo.py` - Reward 计算
   - `recipe/dapo/dapo_ray_trainer.py` - 训练器
   - `verl/trainer/ppo/core_algos.py` - Advantage 计算

2. **文档**:
   - `docs/algo/dapo.md` - 算法使用指南
   - `docs/algo/grpo.md` - GRPO 基础
   - `docs/algo/rollout_corr.md` - Rollout Correction

3. **论文**:
   - DAPO 论文: https://arxiv.org/abs/2503.14476
   - GRPO: Group Relative Policy Optimization
   - PPO: Proximal Policy Optimization

---

## 附录

### A. 常见问题

**Q1: DAPO 一定比 GRPO 好吗？**

不一定。DAPO 在数学推理等高难度任务上表现更好，但在简单任务上可能因为过滤太多样本而效率下降。

**Q2: 动态采样会不会过滤掉太多样本？**

可能会。可以通过调整 `max_num_gen_batches` 来控制。设置为 0 表示无限制生成，直到凑够样本。

**Q3: 为什么不用 Critic 网络？**

DAPO 基于 GRPO，使用组内归一化来估计 advantage，不需要显式的 value function。这简化了训练，减少了超参数。

**Q4: 如何调参？**

建议的调参顺序：
1. 先用默认配置跑通
2. 调整 `clip_ratio_high` (0.25-0.3)
3. 调整 `filter_groups.metric` 和 `max_num_gen_batches`
4. 最后调整 `loss_agg_mode`

**Q5: 能在单卡上跑吗？**

可以，但需要：
- 小模型 (如 Qwen2.5-7B)
- 减小 `train_batch_size` 和 `max_response_length`
- 使用 LoRA 或其他参数高效方法

### B. 术语对照表

| 英文 | 中文 | 解释 |
|------|------|------|
| Advantage | 优势函数 | 衡量某个动作比平均水平好多少 |
| Clip | 裁剪 | 限制策略更新的幅度 |
| Dynamic Sampling | 动态采样 | 智能选择训练样本 |
| Token-level | Token 级别 | 以 token 为单位计算 |
| Sequence-level | 序列级别 | 以完整序列为单位计算 |
| Rollout | 推演/生成 | 使用策略生成轨迹 |
| Policy Gradient | 策略梯度 | RL 的一类算法 |
| GRPO | 组相对策略优化 | 组内归一化的 PG 方法 |

---

**文档版本**: v1.0
**贡献者**: Claude (AI Assistant)
**反馈**: 欢迎在 GitHub Issues 中提出问题和建议
