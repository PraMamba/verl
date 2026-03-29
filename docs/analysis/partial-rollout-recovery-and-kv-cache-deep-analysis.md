# 长文本异步 RL 中的部分 Rollout 恢复机制与 KV Cache 困境：verl 源码深度分析

**Author:** Claude Opus 4.6 (基于 verl 源码分析)
**Date:** 2026-03-20
**Scope:** verl `main` branch，覆盖同步 Hybrid Engine、分离式异步架构、Fully Async Streaming 三种模式

---

## 目录

1. [问题域：异步 RL 的"电车难题"](#1-问题域)
2. [verl 的三层架构与权重同步生命周期](#2-三层架构)
3. [KV Cache 的"弗兰肯斯坦"困境：verl 的明确抉择](#3-kv-cache-困境)
4. [部分 Rollout 恢复机制的四种模式](#4-四种模式)
5. [Fully Async Partial Rollout：源码级完整链路追踪](#5-fully-async-partial-rollout)
6. [算法层面的离策略矫正：Rollout Correction 框架](#6-rollout-correction)
7. [万卡规模下的 Re-Prefill 风暴分析](#7-re-prefill-风暴)
8. [关键源码索引表](#8-源码索引)

---

## 1. 问题域：异步 RL 的"电车难题" {#1-问题域}

在同步 RL 中，训练和推理严格交替执行，不存在权重版本冲突。但在异步 RL 中，当一条长推理序列（如 64K token 的 Chain-of-Thought）正在生成时，训练端完成了新一轮梯度更新，系统面临两个选择：

- **不中断继续生成**：序列前半段由 π_old 生成，后半段由 π_new 生成，破坏轨迹一致性
- **直接中止（Abort）**：丢弃已生成的 token，浪费大量 GPU 算力

verl 对这个问题给出了**多层次、可配置**的解决方案。

---

## 2. verl 的三层架构与权重同步生命周期 {#2-三层架构}

### 2.1 三种部署模式

verl 通过 `RolloutMode` 枚举定义了三种部署模式（`verl/workers/rollout/replica.py:54-67`）：

```python
class RolloutMode(Enum):
    HYBRID = "hybrid"      # 训练引擎与推理引擎在同一进程，GPU 时分复用
    COLOCATED = "colocated" # 同一 Placement Group 不同进程，共享 GPU
    STANDALONE = "standalone" # 独立 GPU 资源，完全分离
```

### 2.2 Hybrid Engine 的 Sleep/Wake 时分复用

在默认的 Hybrid 模式中，训练循环严格遵循以下 10 步周期（`verl/trainer/ppo/ray_trainer.py:1221-1524`）：

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: generate_sequences()    ← GPU 用于推理（vLLM/SGLang） │
│  Step 2: sleep_replicas()        ← 释放推理引擎 GPU 内存       │
│  Step 3-6: compute rewards/values/ref_log_prob                  │
│  Step 7: compute_advantage()     ← 在 CPU Driver 上计算        │
│  Step 8: update_critic()         ← GPU 用于训练                 │
│  Step 9: update_actor()          ← GPU 用于训练                 │
│  Step 10: update_weights()       ← 同步权重到推理引擎          │
└─────────────────────────────────────────────────────────────────┘
```

**关键特性：在 Hybrid 模式中，推理完成后才 sleep，不存在生成中途被中断的情况。** 这是最简单但也最浪费的模式——长尾样本会阻塞整个集群。

### 2.3 权重同步的两种后端

由 `CheckpointEngineManager`（`verl/checkpoint_engine/base.py:308-437`）管理：

| 后端 | 适用模式 | 权重传输方式 | 是否支持 Partial Rollout |
|------|---------|------------|------------------------|
| `naive` | HYBRID（同进程） | CUDA IPC / ZMQ / 共享内存 | **否**（生成完成后才同步） |
| `nccl` / `nixl` | STANDALONE（分离式） | NCCL Collective / RDMA | **是**（abort → sync → resume） |

---

## 3. KV Cache 的"弗兰肯斯坦"困境：verl 的明确抉择 {#3-kv-cache-困境}

### 3.1 核心结论：verl 选择了"清空并重新 Prefill"

**verl 在所有模式下都选择了算法正确的路径：完全清空 KV Cache，绝不复用旧权重的 KV Cache。**

源码证据链：

**vLLM 路径：**

```python
# verl/workers/rollout/vllm_rollout/vllm_rollout.py:177-179
# 每次 update_weights 后，显式清空 prefix cache
if self.rollout_rank == 0:
    await self.server_handle.clear_kv_cache.remote()
```

```python
# verl/workers/rollout/vllm_rollout/vllm_async_server.py:665-667
async def clear_kv_cache(self):
    if self.node_rank == 0:
        await self.engine.reset_prefix_cache()
```

**SGLang 路径：**

```python
# verl/workers/rollout/sglang_rollout/sglang_rollout.py:222
await self._engine.flush_cache()  # 调用 /flush_cache HTTP 端点
```

**abort_all_requests 路径（分离式）：**

```python
# verl/workers/rollout/vllm_rollout/vllm_async_server.py:700-704
await self.engine.pause_generation(
    wait_for_inflight_requests=False,
    clear_cache=reset_prefix_cache,  # 默认 True，清空缓存
)
```

### 3.2 为什么不能复用旧 KV Cache？

这不仅是 verl 的选择，而是数学必然。在 Transformer 的 Multi-Head Attention 中：

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

- **Q（Query）** 由当前 token 的隐藏状态经 **当前权重** W_Q 变换得到
- **K（Key）** 由历史 token 的隐藏状态经 **生成时权重** W_K 变换得到
- **V（Value）** 由历史 token 的隐藏状态经 **生成时权重** W_V 变换得到

如果权重更新后复用旧 KV Cache：

- 新权重的 Q 矩阵与旧权重的 K 矩阵做点积 → **注意力分布偏移**
- 新权重的 Q 矩阵选出旧权重的 V 矩阵 → **激活值分布偏移**

这会导致模型输出严重退化。verl 的代码注释和设计清楚地表明，他们完全理解这一点。

### 3.3 Sleep Level 的精细控制

vLLM 的 sleep 机制提供两个级别（`verl/third_party/vllm/__init__.py:33-49`）：

| Level | 释放内容 | 适用场景 | 源码位置 |
|-------|---------|---------|---------|
| Level 1 | 仅释放 KV Cache | LoRA 微调（基础权重不变） | `vllm_async_server.py:636-637` |
| Level 2 | KV Cache + 模型权重 | 全量权重更新（默认） | `vllm_async_server.py:639` |

```python
# verl/workers/rollout/vllm_rollout/vllm_async_server.py:633-640
if self.lora_as_adapter:
    sleep_level = 1  # LoRA: 只释放 KV Cache，保留基础权重
else:
    sleep_level = 2  # 全量更新：释放 KV Cache + 模型权重
await self.engine.collective_rpc("sleep", kwargs={"level": sleep_level})
```

### 3.4 两阶段 Resume：权重优先于 KV Cache

权重恢复采用严格的两阶段策略（`verl/workers/fsdp_workers.py:822-843`）：

```python
# Phase 1: 先恢复权重内存
await self.rollout.resume(tags=["weights"])

# Phase 2: 传输权重（CUDA IPC / ZMQ）
await self.rollout.update_weights(per_tensor_param, ...)

# Phase 3: 再恢复 KV Cache 内存
await self.rollout.resume(tags=["kv_cache"])
```

**设计理由**：必须先加载权重，才能确定 GPU 剩余多少内存可用于 KV Cache block 分配。这是 vLLM 的 PagedAttention 内存管理的要求。

---

## 4. 部分 Rollout 恢复机制的四种模式 {#4-四种模式}

verl 的 `fully_async_policy` 实验性模块（`verl/experimental/fully_async_policy/`）实现了四种递进的异步模式，文档见 `verl/experimental/fully_async_policy/README.md`：

### 模式 a：On-Policy Pipeline（同步流水线）

```
参数：trigger_parameter_sync_step=1, staleness_threshold=0
```

Rollouter 一次性生成 `require_batches × ppo_mini_batch_size` 个样本，Trainer 消费后立即同步参数。长尾样本导致资源空闲。

### 模式 b：Stream Off-Policy Pipeline（流式同步）

```
参数：trigger_parameter_sync_step>1, staleness_threshold=0
```

Rollouter 一次性生成更多样本，Trainer 分批消费并多次训练后再同步。减少空闲但仍有同步等待。

### 模式 c：Async Stream + Stale Samples（异步 + 过时样本）

```
参数：trigger_parameter_sync_step>=1, staleness_threshold>0, partial_rollout=False
```

允许使用旧参数版本生成的样本。当触发参数同步时，如果 Rollouter 有正在进行的任务，**等待其完成**后再同步。

### 模式 d：Async Stream + Partial Rollout（异步 + 部分回滚）★

```
参数：trigger_parameter_sync_step>=1, staleness_threshold>0, partial_rollout=True
```

**这是 verl 对"电车难题"的终极回答。** 当触发参数同步时，Rollouter **中断**正在进行的生成，保存部分输出，同步参数后**继续**生成。

实验数据（128 卡 H20，Qwen2.5-Math-7B，DAPO 算法）：

| 模式 | 400 步总时间 | 加速比 | 准确率 |
|------|------------|-------|--------|
| Colocate Sync | 1d 16h 48m | 1.00x | max: 0.3573 |
| Stream Off-Policy | 1d 1h 53m | 1.60x | max: 0.2844 |
| **Async + Partial Rollout** | **17h 22m** | **2.35x** | **max: 0.3521** |

---

## 5. Fully Async Partial Rollout：源码级完整链路追踪 {#5-fully-async-partial-rollout}

### 5.1 Pause 阶段：中断正在进行的生成

当 Trainer 完成一轮训练并触发参数同步时，`FullyAsyncRollouter.pause()` 被调用：

```python
# verl/experimental/fully_async_policy/fully_async_rollouter.py:751-768
async def pause(self):
    async with self.lock:
        self.paused = True
        # 关键：如果启用 partial_rollout，取消所有正在进行的生成
        if self.config.async_training.partial_rollout:
            await self.async_rollout_manager.cancel()
        # 等待所有活跃任务完成/取消
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
            self.active_tasks.clear()
        # 始终清空 KV Cache 以释放 GPU 内存
        await self.async_rollout_manager.clear_kv_cache()
```

### 5.2 Cancel 传播链路

```
FullyAsyncRollouter.pause()
  → FullyAsyncAgentLoopManager.cancel()                    # agent_loop.py:376-379
    → FullyAsyncAgentLoopWorker.cancel_agent_loops()        # agent_loop.py:305-307
      → self.cancellation_event.set()                       # asyncio.Event 广播
    → FullyAsyncvLLMReplica.cancel()                        # vllm_async_server.py:160-162
      → vLLMHttpServerForPartial.cancel()                   # vllm_async_server.py:137-141
        → self.paused = True
        → 对所有正在进行的请求设置 cancel_event
```

### 5.3 vLLM 层面的 Partial 生成捕获

`vLLMHttpServerForPartial.generate_for_partial()` 是关键方法（`verl/experimental/fully_async_policy/vllm_rollout/vllm_async_server.py:95-135`）：

```python
async def generate_for_partial(self, prompt_ids, sampling_params, request_id, ...):
    async with self.lock:
        if self.paused:
            return [], [], True  # 已暂停，直接返回空结果 + is_cancel=True

        # 创建取消事件和生成任务
        self.cancel_event[request_id] = asyncio.Event()
        cancel_handle = asyncio.create_task(self.cancel_event[request_id].wait())
        generation_handle = asyncio.create_task(
            self._generate_step(prompt_ids, sampling_params, request_id, ...)
        )

    # 关键：等待"生成完成"或"取消信号"，哪个先到用哪个
    done, pend = await asyncio.wait(
        [generation_handle, cancel_handle],
        return_when=asyncio.FIRST_COMPLETED
    )

    async with self.lock:
        if self.req_output[request_id] is None:
            return [], [], True  # 没有任何输出，完全取消

        # 收集已生成的部分 token 和 logprobs
        token_ids = self.req_output[request_id].outputs[0].token_ids
        log_probs = [x[token_id].logprob for i, x in enumerate(...)]
        is_cancel = generation_handle not in done
        return token_ids, log_probs, is_cancel
```

**核心机制**：`_generate_step` 在每个 token 生成后更新 `self.req_output[request_id]`。当 cancel 事件触发时，`asyncio.wait` 返回，此时 `req_output` 中已经有了部分生成的 token（如 64K 序列已生成 32K 个 token）。

### 5.4 部分输出的保存与恢复

被取消的样本通过 `cancel_queue` 保存，等待恢复：

```python
# verl/experimental/fully_async_policy/fully_async_rollouter.py:578-606
async def _process_single_sample_streaming(self, rollout_sample):
    ret, is_cancel = await self.async_rollout_manager.generate_single_sample_async(
        rollout_sample.full_batch, rollout_sample.agent_loop_output_list
    )
    if not is_cancel:
        # 正常完成：发送到 MessageQueue
        rollout_sample.full_batch = ret
        await self.message_queue_client.put_sample(...)
    else:
        # 被取消：保存部分输出到 cancel_queue，等待恢复
        rollout_sample.agent_loop_output_list = ret  # 保存部分 AgentLoopOutput
        await self.cancel_queue.put(rollout_sample)
```

### 5.5 Resume 阶段：用新权重继续生成

参数同步完成后，`FullyAsyncRollouter.resume()` 被调用：

```python
# verl/experimental/fully_async_policy/fully_async_rollouter.py:770-779
async def resume(self, dependency_ref=None):
    if dependency_ref is not None:
        ray.get(dependency_ref)  # 等待权重同步完成
    async with self.lock:
        if self.config.async_training.partial_rollout:
            await self.async_rollout_manager.resume()  # 恢复推理引擎
        self.paused = False
        self.condition.notify_all()  # 唤醒 _processor_worker
```

`_processor_worker` 恢复后优先从 `cancel_queue` 取样本（`fully_async_rollouter.py:529-532`）：

```python
if not self.cancel_queue.empty():
    rollout_sample = await self.cancel_queue.get()  # 优先恢复被取消的样本
else:
    rollout_sample = await self.pending_queue.get()  # 否则取新样本
```

### 5.6 PartialSingleTurnAgentLoop：Token 拼接逻辑

被恢复的样本在 `PartialSingleTurnAgentLoop.run()` 中继续（`verl/experimental/fully_async_policy/agent_loop/partial_single_turn_agent_loop.py:38-134`）：

```python
async def run(self, sampling_params, **kwargs):
    output: Optional[AgentLoopOutput] = kwargs.get("output", None)

    if not output:
        # 全新样本：正常 tokenize prompt
        prompt_ids = await self.apply_chat_template(messages, ...)
    else:
        if output.extra_fields.get("is_cancel", False):
            # ★ 恢复被取消的样本：将已生成的 token 追加到 prompt 后面
            prompt_ids = output.prompt_ids + output.response_ids
            param_version_start = output.extra_fields.get("param_version_start", ...)
        else:
            return output  # 同批次中未被取消的样本直接返回

    # 用新权重继续生成（prompt_ids 现在包含原始 prompt + 已生成的部分 response）
    response_ids, response_logprobs, is_cancel = await self.server_manager.generate_for_partial(
        request_id=request_id,
        prompt_ids=prompt_ids,  # ★ 关键：包含已生成的部分作为新的"prompt"
        sampling_params=sampling_params, ...
    )

    if output:
        # 拼接新旧输出
        prompt_ids = output.prompt_ids  # 恢复原始 prompt
        response_logprobs = output.response_logprobs + response_logprobs  # 拼接 logprobs
        response_ids = output.response_ids + response_ids  # 拼接 token IDs
```

### 5.7 FullyAsyncLLMServerManager：透明的 Abort → Re-prefill 循环

`FullyAsyncLLMServerManager.generate()` 方法实现了另一种恢复策略（`verl/experimental/fully_async_policy/agent_loop/agent_loop.py:52-128`）：

```python
async def generate(self, request_id, *, prompt_ids, sampling_params, ...):
    final_output = TokenOutput(token_ids=[], log_probs=[], num_preempted=0)

    while True:
        # 1. 发起生成请求（prompt = 原始 prompt + 已生成的 token）
        output = await super().generate(
            request_id=request_id,
            prompt_ids=prompt_ids + final_output.token_ids,  # ★ Re-prefill
            sampling_params=sampling_params, ...
        )

        # 2. 合并输出
        final_output.token_ids.extend(output.token_ids)
        final_output.log_probs.extend(output.log_probs)

        # 3. 更新剩余 max_tokens
        if original_max_tokens is not None:
            sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)

        # 4. 检查是否被 abort（触发 re-prefill 循环）
        if output.stop_reason not in ("aborted", "abort") or \
           not self.config.async_training.partial_rollout_resume:
            break  # 正常完成或不需要恢复
        # 否则：循环回到 step 1，用 "原始 prompt + 已生成 token" 作为新 prompt
        # ★ 这就是 Re-Prefill！新权重会重新计算整个序列的 KV Cache

    final_output.extra_info["min_global_steps"] = min_global_steps  # 记录跨越的权重版本
    final_output.extra_info["max_global_steps"] = max_global_steps
    return final_output
```

**这就是 verl 对 KV Cache 困境的回答**：当生成被 abort 后，已生成的 token IDs 被保存。恢复时，这些 token IDs 被拼接到原始 prompt 后面，作为新的 "prompt" 输入给推理引擎。推理引擎用**新权重**对整个序列执行 **Re-Prefill**，重新计算所有位置的 KV Cache，然后从中断点继续自回归生成。

---

## 6. 算法层面的离策略矫正：Rollout Correction 框架 {#6-rollout-correction}

### 6.1 三策略架构

verl 的核心算法设计区分三个策略（`verl/trainer/ppo/ray_trainer.py:1379-1432`）：

| 策略 | 符号 | 含义 | 数据来源 |
|------|------|------|---------|
| Rollout Policy | π_rollout | 推理引擎（vLLM/SGLang）生成数据时的策略 | `rollout_log_probs` |
| Old Policy | π_old | 训练引擎在当前 batch 开始时的策略 | `old_log_probs`（重新计算） |
| Current Policy | π_θ | 训练引擎在 mini-batch 更新中演化的策略 | `log_prob`（前向传播） |

在 Partial Rollout 场景下，由于序列的前半段和后半段可能由不同权重版本生成，`rollout_log_probs` 实际上是混合策略的产物。

### 6.2 Importance Sampling 权重计算

Rollout Correction 计算 IS 权重（`verl/trainer/ppo/rollout_corr_helper.py:530-598`）：

```python
# 核心 IS 比率：π_old / π_rollout
log_ratio = old_log_prob - rollout_log_prob

# Token 级别 IS
if rollout_is == "token":
    is_weights = torch.exp(torch.clamp(log_ratio, -20, 20))

# 序列级别 IS
elif rollout_is == "sequence":
    seq_log_ratio = torch.sum(log_ratio * response_mask, dim=-1)
    is_weights = torch.exp(torch.clamp(seq_log_ratio, -20, 20))

# 截断 IS（TIS）
is_weights = torch.clamp(is_weights, max=rollout_is_threshold)  # 默认 2.0
is_weights = is_weights.detach()  # 不参与梯度计算
```

### 6.3 Rejection Sampling 作为硬信任域

除了软性 IS 权重，还有硬性拒绝采样（`rollout_corr_helper.py:156-372`）：

```python
# K1（比率界限）：|log(π_old/π_rollout)| > threshold → 拒绝
# K2（二次散度）：0.5 × log(π_old/π_rollout)² > threshold → 拒绝
# K3（χ² 散度）：exp(log_ratio) - 1 - log_ratio > threshold → 拒绝
```

超过阈值的 token/序列被直接 mask 掉（`response_mask` 置零），不参与训练。

### 6.4 Bypass Mode vs Decoupled Mode

两种运行模式决定了如何处理离策略数据：

**Bypass Mode**（`ray_trainer.py:1385-1392`）：
```python
# 直接使用 rollout log_probs 作为 old_log_probs
# 2 策略系统：π_rollout = π_old, π_θ
old_log_probs = rollout_log_probs
```

**Decoupled Mode**（`ray_trainer.py:1393-1419`）：
```python
# 用当前训练模型重新计算 old_log_probs
# 3 策略系统：π_rollout, π_old, π_θ
old_log_prob = self._compute_old_log_prob(batch)
# 然后计算 IS 权重：π_old / π_rollout
```

### 6.5 Partial Rollout 对 IS 的影响

在 Partial Rollout 场景下，`rollout_log_probs` 的前半段来自权重版本 V_n，后半段来自权重版本 V_{n+1}。`PartialSingleTurnAgentLoop` 记录了版本信息：

```python
# verl/experimental/fully_async_policy/agent_loop/partial_single_turn_agent_loop.py:125-128
extra_fields={
    "is_cancel": is_cancel,
    "param_version_start": param_version_start,  # 起始权重版本
    "param_version_end": param_version_end,       # 结束权重版本
}
```

`FullyAsyncLLMServerManager.generate()` 也跟踪：
```python
# agent_loop.py:126-127
final_output.extra_info["min_global_steps"] = min_global_steps
final_output.extra_info["max_global_steps"] = max_global_steps
```

这使得训练端可以知道每条序列跨越了多少个权重版本，从而做出更精确的 IS 矫正。

---

## 7. 万卡规模下的 Re-Prefill 风暴分析 {#7-re-prefill-风暴}

### 7.1 Re-Prefill 的计算成本

当权重同步导致 N 条长序列被 abort 后，每条序列需要对 `原始 prompt + 已生成 token` 执行 Re-Prefill。

**Prefill 的计算复杂度**：对于序列长度 L、模型参数 P：
- Prefill 是 **Compute-bound**：FLOPs ∝ 2 × P × L（矩阵乘法主导）
- Decode 是 **Memory-bound**：FLOPs ∝ 2 × P × 1（每次只处理 1 个 token）

对于 64K token 的 Re-Prefill：
```
假设：70B 模型，80 层，128 head_dim，bf16
单序列 Prefill FLOPs ≈ 2 × 70B × 64K ≈ 9 × 10^15 FLOPs
在 H100（990 TFLOPS bf16）上 ≈ 9.1 秒
在 H20（~148 TFLOPS bf16）上 ≈ 60.8 秒
```

### 7.2 万卡场景下的影响

**同步 Re-Prefill 风暴**（最坏情况）：

假设 10,000 张 GPU 同时进行权重同步，每张 GPU 有 1 条 64K 的中断序列需要 Re-Prefill：
- 所有 GPU 在权重同步后**同时**开始 Prefill
- 这造成一个 ~60 秒（H20）的 **功耗尖峰**（Prefill 消耗的能量远高于 Decode）
- 网络也可能出现压力（NCCL 权重同步本身需要时间）

**实际影响缓解因素**：

1. **分布式异步架构**：在 Fully Async 模式下，不同 Rollouter 的权重同步时间点是**交错**的，不会所有 GPU 同时触发
2. **序列长度分布**：不是所有序列都是 64K，大多数可能更短
3. **Chunked Prefill**：vLLM 支持 `enable_chunked_prefill=True`（`vllm_async_server.py:313`），将长 Prefill 分块处理，降低单次峰值
4. **Pipeline 效果**：由于 Re-Prefill 是 Compute-bound 而 Decode 是 Memory-bound，它们可以在不同硬件层面并行

### 7.3 verl 的实测数据

从 `fully_async_policy/README.md` 的实验数据：

| 模型 | Trainer 卡数 | Rollout 卡数 | 使用 checkpoint-engine | 单步同步时间 |
|------|------------|------------|---------------------|------------|
| Qwen2.5-7B | 4 | 4 | 是 | **0.02s** |
| Qwen3-30B | 16 | 16 | 是 | **4.38s** |
| Qwen3-235B | 64 | 64 | 是 | **23.70s** |

这些时间**不包括** Re-Prefill，仅是权重传输时间。Re-Prefill 成本取决于被中断序列的长度。

### 7.4 未来优化方向

verl 代码中已经预留了优化接口：

**1. `CheckpointEngineWithCache`（`checkpoint_engine/base.py:180-194`）：**
```python
class CheckpointEngineWithCache(CheckpointEngine):
    """允许将权重同步到本地缓存（SHM/磁盘），不中断推理。
    推理请求完成后再从缓存加载新权重。

    参考：Laminar (https://arxiv.org/abs/2510.12633)
    """
    async def get_weights(self):
        """从本地缓存获取权重"""
        raise NotImplementedError
```

这个抽象类暗示了一种更优的策略：**异步权重预加载**——在生成继续的同时，新权重被传输到本地缓存。当生成完成后，直接从本地缓存加载新权重，避免同步等待。

**2. LoRA 的 KV Cache 部分保留：**
对于 LoRA 微调（`sleep_level=1`），基础权重不变，因此基础层的 KV Cache 理论上可以保留。verl 目前仍然清空全部 KV Cache，但这是一个潜在的优化点。

**3. 层级式权重分发：**
当前的 NCCL 权重广播是扁平结构（所有 rank 参与同一 process group）。对于万卡场景，可以采用树形广播：Trainer → 节点本地副本 → 每 GPU 推理引擎，将关键路径从 O(N) 降低到 O(log N)。

---

## 8. 关键源码索引表 {#8-源码索引}

### 核心训练循环

| 功能 | 文件 | 行号 |
|------|------|------|
| 主训练循环 `fit()` | `verl/trainer/ppo/ray_trainer.py` | 1221-1609 |
| 生成序列 | `verl/trainer/ppo/ray_trainer.py` | 1312 |
| Sleep replicas | `verl/trainer/ppo/ray_trainer.py` | 1313 |
| 更新权重到 rollout | `verl/trainer/ppo/ray_trainer.py` | 1524 |

### 权重同步引擎

| 功能 | 文件 | 行号 |
|------|------|------|
| CheckpointEngineManager | `verl/checkpoint_engine/base.py` | 308-437 |
| sleep_replicas() | `verl/checkpoint_engine/base.py` | 394-399 |
| update_weights()（abort→sync→resume） | `verl/checkpoint_engine/base.py` | 402-437 |
| CheckpointEngineWithCache（抽象） | `verl/checkpoint_engine/base.py` | 180-194 |

### vLLM 推理引擎

| 功能 | 文件 | 行号 |
|------|------|------|
| ServerAdapter.update_weights() | `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | 154-184 |
| sleep()/wake_up() | `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 615-647 |
| clear_kv_cache() | `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 665-667 |
| abort_all_requests() | `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 676-737 |
| resume_generation() | `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 739-748 |
| BucketedWeightSender | `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` | 73-197 |

### Fully Async Partial Rollout

| 功能 | 文件 | 行号 |
|------|------|------|
| FullyAsyncRollouter.pause() | `verl/experimental/fully_async_policy/fully_async_rollouter.py` | 751-768 |
| FullyAsyncRollouter.resume() | `verl/experimental/fully_async_policy/fully_async_rollouter.py` | 770-779 |
| cancel_queue 保存/恢复 | `verl/experimental/fully_async_policy/fully_async_rollouter.py` | 529-606 |
| vLLMHttpServerForPartial.generate_for_partial() | `verl/experimental/fully_async_policy/vllm_rollout/vllm_async_server.py` | 95-135 |
| vLLMHttpServerForPartial.cancel() | `verl/experimental/fully_async_policy/vllm_rollout/vllm_async_server.py` | 137-141 |
| PartialSingleTurnAgentLoop.run() | `verl/experimental/fully_async_policy/agent_loop/partial_single_turn_agent_loop.py` | 38-134 |
| FullyAsyncLLMServerManager.generate()（Re-Prefill 循环） | `verl/experimental/fully_async_policy/agent_loop/agent_loop.py` | 52-128 |
| FullyAsyncAgentLoopManager.cancel()/resume() | `verl/experimental/fully_async_policy/agent_loop/agent_loop.py` | 376-384 |

### 算法与 IS 矫正

| 功能 | 文件 | 行号 |
|------|------|------|
| IS 权重计算 | `verl/trainer/ppo/rollout_corr_helper.py` | 530-598 |
| Rejection Sampling | `verl/trainer/ppo/rollout_corr_helper.py` | 156-372 |
| Bypass Mode | `verl/trainer/ppo/ray_trainer.py` | 1385-1392 |
| Decoupled Mode | `verl/trainer/ppo/ray_trainer.py` | 1393-1419 |
| Policy Loss + IS 权重应用 | `verl/trainer/ppo/core_algos.py` | 1244-1245 |
| Optimal Token Baseline + IS | `verl/trainer/ppo/core_algos.py` | 758-873 |

### Rollout Replica 基础设施

| 功能 | 文件 | 行号 |
|------|------|------|
| RolloutMode 枚举 | `verl/workers/rollout/replica.py` | 54-67 |
| RolloutReplica（sleep/wake/abort/resume） | `verl/workers/rollout/replica.py` | 248-266 |
| Sleep Level 定义 | `verl/third_party/vllm/__init__.py` | 33-49 |
| rollout_mode() 上下文切换 | `verl/workers/fsdp_workers.py` | 742-847 |

---

## 总结

verl 对 KV Cache "弗兰肯斯坦"困境的回答是清晰而坚定的：

1. **绝不复用旧权重的 KV Cache** —— 数学上不正确，工程上不可接受
2. **选择 Re-Prefill** —— 将已生成的部分 token IDs 保存，用新权重重新 prefill 整个序列
3. **通过 Partial Rollout 减少浪费** —— 不是丢弃整条序列，而是保存部分输出，Re-Prefill 后从中断点继续
4. **通过 Rollout Correction 保证算法正确性** —— IS 权重和 Rejection Sampling 处理跨版本数据
5. **通过 Fully Async Streaming 最大化利用率** —— 2.35-2.67x 的吞吐提升证明了这一架构的有效性

对于万卡规模的 Re-Prefill 风暴，verl 的缓解策略包括：异步交错同步时间点、Chunked Prefill 降低峰值、以及未来通过 `CheckpointEngineWithCache`（Laminar 论文）实现的异步权重预加载。
