# 系统级调度分析：显存碎片、OOM 窗口与调度优先级饥饿

**Author:** Claude Opus 4.6 (基于 verl 源码深度分析，综合 5 个专家 Agent 的独立分析结果)
**Date:** 2026-03-20
**Scope:** verl `main` branch，聚焦 Partial Rollout 恢复场景下的 GPU 显存管理与调度公平性

**分析贡献者：**
- **vLLM/SGLang Expert #1**：vLLM PagedAttention 块生命周期、abort/resume 中的块回收
- **vLLM/SGLang Expert #2**：SGLang RadixAttention 内存管理、`abort_request` 的调度器实现
- **FSDP Engine Expert**：Hybrid Engine 完整显存时间线、`expandable_segments` 切换逻辑
- **Explore Agent**：调度优先级机制、cancel_queue vs pending_queue、长前缀饥饿分析
- **Architect Reviewer**：OOM 窗口定量分析、"大前缀"问题、NCCL 双缓冲区内存竞争

---

## 目录

1. [GPU 显存在 Abort/Sync/Resume 三阶段的完整生命周期](#1-显存生命周期)
2. [Hybrid Engine 模式：显存时间线](#2-hybrid-模式)
3. [分离式（Disaggregated）模式：NCCL 权重传输的内存开销](#3-分离式模式)
4. [OOM 风险窗口：定量分析](#4-oom-窗口)
5. [显存碎片化：重复 Sleep/Wake 循环的长期效应](#5-显存碎片化)
6. [调度优先级与饥饿问题](#6-调度饥饿)
7. ["大前缀"问题：64K Re-Prefill 的显存需求](#7-大前缀问题)
8. [verl 的显存保护机制总览](#8-保护机制)
9. [架构缺口与改进建议](#9-改进建议)
10. [关键源码索引](#10-源码索引)

---

## 1. GPU 显存在 Abort/Sync/Resume 三阶段的完整生命周期 {#1-显存生命周期}

### 1.1 显存释放策略：verl 选择**完全释放**

当权重同步信号下达时，verl 对所有正在运行的 Request 采取**完全释放**策略——KV Cache 块被释放，不存在"锁定显存等待恢复"的路径。

**vLLM 路径**（`vllm_async_server.py:676-704`）：

```python
# vLLM >= 0.12.0
await self.engine.pause_generation(
    wait_for_inflight_requests=False,  # 不等待正在进行的请求完成
    clear_cache=reset_prefix_cache,    # 默认 True：清空 prefix cache
)
```

`pause_generation` 的内部行为（vLLM V1 AsyncLLM）：
1. 设置引擎为 paused 状态（阻止新请求）
2. **Abort 所有 in-flight 请求**——vLLM 的 EngineCore 释放这些请求的 KV Cache blocks
3. 等待所有请求 drain
4. 如果 `clear_cache=True`，清空 prefix cache（RadixAttention 树结构被重置）

**SGLang 路径**（vLLM/SGLang Expert #2 从 SGLang 源码分析）：

```python
# async_sglang_server.py → tokenizer_manager.pause_generation()
await self.tokenizer_manager.pause_generation(PauseGenerationReqInput(mode="abort"))
```

SGLang 的 `abort_request`（SGLang `scheduler.py:2328-2360`）内部行为：
1. 从 `waiting_queue` 中直接 pop 请求（**立即释放**，未开始的请求不占 KV Cache）
2. 从 `grammar_queue` 中标记 `set_finish_with_abort`（请求仍会运行一次 prefill forward pass，但 `input_ids` 被替换为仅 1 个 token，**成本极低**）
3. 正在运行的 batch 中的请求通过设置 `finished` 标志在下一个调度循环结束

**关键结论**：在两种引擎中，abort 后 KV Cache blocks 都被**释放回 block pool**。没有任何"锁定"机制保留中断请求的显存。

### 1.2 三阶段显存状态

| 阶段 | GPU 上的内容 | 状态 |
|------|-------------|------|
| **Abort 前**（推理中） | 模型权重 + KV Cache blocks + 活跃 batch 张量 | 推理引擎占满 `gpu_memory_utilization` |
| **Abort 后** | 模型权重 + 空闲 KV Cache pool（blocks 已归还） | KV Cache 内存回到 pool 但**未归还 CUDA** |
| **Sleep 后** | Level 2：全部释放；Level 1：仅 KV Cache 释放 | `aggressive_empty_cache()` 归还到 CUDA |
| **权重传输中** | 新权重（通过 CUDA IPC / NCCL 接收） | 临时需要 bucket 缓冲区 |
| **Resume(weights)** | 新模型权重 | 推理引擎重新分配权重内存 |
| **Resume(kv_cache)** | 新模型权重 + 新 KV Cache pool | 重新分配可用显存给 KV blocks |

---

## 2. Hybrid Engine 模式：显存时间线（FSDP Expert Agent 贡献）{#2-hybrid-模式}

在 Hybrid 模式（训练与推理共享同一 GPU）中，`rollout_mode()`（`fsdp_workers.py:742-847`）精心编排了显存交接。

### 2.1 完整显存时间线

```
时间 ─────────────────────────────────────────────────────────────────────►

Phase 1: 推理（Rollout Generation）
┌──────────────────────────────────────────────────────────────────────┐
│ GPU Memory: [vLLM 模型权重] [KV Cache Blocks] [Activation Buffers]  │
│ 总占用: gpu_memory_utilization (默认 50%) 的 GPU 显存               │
└──────────────────────────────────────────────────────────────────────┘
                    │
                    ▼ checkpoint_manager.sleep_replicas()
Phase 2: Sleep（释放推理引擎）
┌──────────────────────────────────────────────────────────────────────┐
│ sleep(level=2): 释放 KV Cache + 模型权重                            │
│ aggressive_empty_cache(): gc.collect() + torch.cuda.empty_cache()   │
│ GPU Memory: [PyTorch Reserved Pool（大部分空闲）]                    │
└──────────────────────────────────────────────────────────────────────┘
                    │
                    ▼ FSDP 训练阶段
Phase 3: 训练（FSDP Forward/Backward/Optimizer）
┌──────────────────────────────────────────────────────────────────────┐
│ GPU Memory: [FSDP 分片参数] [梯度] [优化器状态] [激活值]            │
│ 占用接近 100% GPU 显存（通过 PyTorch Caching Allocator）            │
└──────────────────────────────────────────────────────────────────────┘
                    │
                    ▼ checkpoint_manager.update_weights()
Phase 4: 权重传输（rollout_mode() 内部）
┌──────────────────────────────────────────────────────────────────────┐
│ Step 1: aggressive_empty_cache()                  # 清理训练残留    │
│ Step 2: load_fsdp_model_to_gpu() (如果 CPU offload)                 │
│ Step 3: 提取 state_dict → per_tensor_param 生成器                   │
│ Step 4: offload_fsdp_model_to_cpu() (如果支持)                      │
│ Step 5: set_expandable_segments(False)            # ★ 关闭扩展段    │
│ Step 6: rollout.resume(tags=["weights"])          # 分配推理权重内存│
│ Step 7: rollout.update_weights(per_tensor_param)  # CUDA IPC 传输   │
│ Step 8: aggressive_empty_cache()                  # 清理传输残留    │
│ Step 9: rollout.resume(tags=["kv_cache"])         # 分配 KV Cache   │
│ Step 10: set_expandable_segments(True)            # ★ 重新打开扩展段│
└──────────────────────────────────────────────────────────────────────┘
                    │
                    ▼ 回到 Phase 1
```

### 2.2 `expandable_segments` 切换的深层原因

`set_expandable_segments(False)`（`fsdp_workers.py:794`）在权重传输**之前**被调用，`True`（line 847）在**之后**恢复。

**原因**（FSDP Expert 分析）：PyTorch 的 CUDA Caching Allocator 有两种内存分配模式：
- **Expandable Segments = True**：分配器可以扩展现有内存段而非分配新段，减少碎片化。但这与 vLLM 的内存管理冲突——vLLM 需要精确控制 GPU 显存的分配边界。
- **Expandable Segments = False**：分配器使用固定段，vLLM 可以准确计算可用内存。

**NCCL Checkpoint Engine 的兼容性**：`nccl_checkpoint_engine.py:129-130` 中，master 进程使用 CuPy（而非 PyTorch）分配 buffer：

```python
if self.is_master:
    self.send_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)  # CuPy，绕过 PyTorch
    self.recv_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
```

注释说明：使用 CuPy "to avoid memory register error when `expandable_segments:True`"。

### 2.3 Peak Memory 分析

权重传输阶段（Phase 4）的峰值显存出现在 **Step 6-7** 之间：
- FSDP 模型参数（如果未 CPU offload）：模型大小
- 推理引擎权重（resume 后重新分配）：模型大小
- CUDA IPC 传输缓冲区：`update_weights_bucket_megabytes`（默认 2048 MB = 2 GB）

**最坏情况**（无 CPU offload）：短暂持有**两份模型权重** + 2 GB 缓冲区。
**最佳情况**（有 CPU offload）：FSDP 模型在 CPU，仅持有推理引擎权重 + 2 GB 缓冲区。

---

## 3. 分离式（Disaggregated）模式：NCCL 权重传输的内存开销（Architect Reviewer 贡献）{#3-分离式模式}

在分离式部署（Trainer 和 Rollouter 使用不同 GPU）中，权重传输通过 NCCL 进行。

### 3.1 NCCL 双缓冲区机制

`NCCLCheckpointEngine`（`nccl_checkpoint_engine.py:96-136`）使用**双缓冲区**实现流水线传输：

```python
def prepare(self):
    # Master (Trainer rank 0): 使用 CuPy 分配
    self.send_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)   # bucket_size bytes
    self.recv_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)   # bucket_size bytes
    # Non-master (Rollout ranks): 使用 PyTorch 分配
    self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")
    self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")
```

**显存开销**：每个参与 NCCL 通信的 GPU 需要 **2 × bucket_size** 的显存。

默认 `update_weights_bucket_megabytes = 2048`（2 GB），所以 NCCL 传输需要 **4 GB** 显存。

### 3.2 传输期间 Rollout GPU 的内存布局

```
Rollout GPU（分离式，权重同步期间）:
┌──────────────────────────────────────────────────────────────────┐
│ [旧模型权重 - 已 sleep 释放?]                                    │
│ [NCCL send_buf: 2 GB]  [NCCL recv_buf: 2 GB]                   │
│ [新权重逐 bucket 写入推理引擎]                                   │
│ [KV Cache - 已释放]                                              │
└──────────────────────────────────────────────────────────────────┘
```

**关键问题**：在分离式模式中，`CheckpointEngineManager.update_weights()`（`base.py:402-437`）的流程是：

1. `abort_all_requests()` — KV Cache blocks 归还到 pool
2. 但**没有显式调用 `sleep()`**！Rollout 引擎的模型权重仍在 GPU 上
3. `build_process_group()` + `update_weights()` — NCCL 双缓冲区分配
4. 新权重通过 `receive_weights()` → `server_adapter.update_weights()` 写入

**Architect Reviewer 发现**：在分离式模式下，旧权重和 NCCL 缓冲区**同时在 GPU 上**。对于大模型：

| 模型 | 权重大小 (bf16) | NCCL 双缓冲区 | 合计额外开销 |
|------|----------------|--------------|------------|
| 7B | 14 GB | 4 GB | 18 GB |
| 30B-A3B (MoE) | ~6 GB active | 4 GB | 10 GB |
| 70B | 140 GB (跨 TP) | 4 GB | 每卡 ~22 GB |
| 235B-A22B (MoE) | ~44 GB active | 4 GB | 每卡 ~11 GB |

**但注意**：vLLM/SGLang 的 `update_weights()` 是**逐 bucket 原地覆写**的（不分配新内存），所以不需要同时持有新旧两份完整权重。NCCL buffer 是滑动窗口。

### 3.3 与 Hybrid 模式的对比

| 维度 | Hybrid (naive) | Disaggregated (NCCL) |
|------|---------------|---------------------|
| 传输前是否 sleep | ✓（Level 2，释放所有） | ✗（abort 释放 KV 但不释放权重） |
| 权重传输方式 | CUDA IPC（零拷贝，同 GPU） | NCCL broadcast（跨 GPU/节点） |
| 额外缓冲区 | 2 GB (ZMQ bucket) | 4 GB (NCCL 双缓冲) |
| OOM 风险 | 低（sleep 释放了大量内存） | **中**（旧权重 + NCCL buffer） |
| 传输后清理 | `finalize()` 释放 buffer + `empty_cache()` | 同左（`nccl_checkpoint_engine.py:148-151`） |

---

## 4. OOM 风险窗口：定量分析（Architect Reviewer 贡献）{#4-oom-窗口}

### 4.1 Hybrid 模式 OOM 分析

**80 GB GPU，70B 模型（TP=8，每卡约 17.5 GB 权重）**：

| 阶段 | 占用 | 可用 |
|------|------|------|
| 推理中 | 权重 17.5 GB + KV Cache ~22 GB | ~40 GB |
| Sleep(level=2) 后 | ~0 GB（全部释放） | ~80 GB |
| FSDP 训练中 | 权重 17.5 GB + 梯度 17.5 GB + 优化器 35 GB + 激活 ~10 GB | ~0 GB |
| 训练→推理过渡 | FSDP offload CPU 后 ~0 GB → resume 权重 17.5 GB + buffer 2 GB | 60.5 GB ✓ |
| KV Cache 分配 | 权重 17.5 GB + KV ~(80×0.5-17.5) = 22.5 GB | 正常 ✓ |

**OOM 风险**：**低**。Sleep(level=2) + CPU offload 释放了足够空间。

### 4.2 分离式模式 OOM 分析

**80 GB Rollout GPU，70B 模型（TP=8，每卡约 17.5 GB 权重）**：

| 阶段 | 占用 | 可用 |
|------|------|------|
| 推理中 | 权重 17.5 GB + KV Cache ~22 GB | ~40 GB |
| Abort 后（KV 释放，权重保留） | 权重 17.5 GB | ~62.5 GB |
| NCCL buffer 分配 | 权重 17.5 GB + buffer 4 GB | ~58.5 GB ✓ |
| 权重逐 bucket 覆写中 | 权重 17.5 GB + buffer 4 GB | ~58.5 GB ✓ |
| Resume KV Cache | 权重 17.5 GB + KV ~22 GB | ~40 GB ✓ |

**OOM 风险**：**低**（对于常规 TP 配置）。

### 4.3 高风险场景：大模型 + 低 TP

**80 GB GPU，70B 模型（TP=2，每卡约 70 GB 权重）**——这种极端配置下：

| 阶段 | 占用 | 可用 |
|------|------|------|
| 推理中 | 权重 70 GB + KV 少量 | ~0 GB |
| Abort 后 | 权重 70 GB | ~10 GB |
| NCCL buffer 分配 4 GB | 权重 70 GB + 4 GB = 74 GB | ~6 GB |
| ★ **如果某个 bucket 覆写失败** | 可能 OOM | ✗ |

**此场景下 OOM 风险高**。但实际使用中 70B 模型不太可能在 2 卡 TP 上运行（显存不够加载完整权重）。

### 4.4 真正的 OOM 隐患：`update_weights_from_ipc` 中的 clone

`bucketed_weight_transfer.py` 的 **BucketedWeightReceiver**（FSDP Expert 确认）中：

```python
# 对于 CUDA IPC 模式（非共享内存）：
tensor = tensor.clone()  # ★ clone 释放 IPC 引用，但暂时持有两份
```

每个 bucket（2 GB）的权重在 `clone()` 时，短暂需要 **双倍 bucket 内存**（4 GB）。但由于 `load_weights()` 是逐 bucket 调用的，旧 bucket 在新 bucket 分配前就已被消费和释放。

---

## 5. 显存碎片化：重复 Sleep/Wake 循环的长期效应 {#5-显存碎片化}

### 5.1 碎片化的来源

每个训练步骤涉及两次重大的显存分配/释放转换：
1. 推理→训练：释放 KV Cache + 模型权重 → 分配 FSDP 参数 + 梯度 + 优化器
2. 训练→推理：释放 FSDP 状态 → 分配模型权重 + KV Cache

PyTorch 的 CUDA Caching Allocator 维护一个 free list。如果释放和分配的 block 大小不匹配，free list 会产生碎片——大量小块空闲内存无法合并成大块。

### 5.2 verl 的反碎片化机制

**① `aggressive_empty_cache()`**（`memory_utils.py:31-75`）：

```python
def aggressive_empty_cache(force_sync=True, max_retries=3):
    for attempt in range(max_retries):
        gc.collect()                  # Python GC，释放循环引用的张量
        device.empty_cache()          # 归还 PyTorch 缓存到 CUDA
        if force_sync:
            device.synchronize()      # 确保所有异步操作完成
        if reserved_freed < 1GB:
            break                     # 释放量不足 1GB 就停止重试
```

**② `set_expandable_segments()` 切换**（`device.py:132-146`）：

```python
torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")
```

- **False**（权重传输期间）：禁止分配器扩展段，确保 vLLM 能精确计算可用内存
- **True**（常规运行时）：允许扩展段，减少碎片化

**③ vLLM 的 `enable_sleep_mode`**（默认 `True`）：

当 `enable_sleep_mode=True` 时，vLLM 使用其内部的 sleep/wake API 管理显存，绕过 PyTorch 缓存分配器。这避免了 PyTorch allocator 的碎片化问题。

当 `enable_sleep_mode=False` 时（`vllm_async_server.py:236-238`）：

```python
if not self.config.enable_sleep_mode:
    from verl.utils.device import set_expandable_segments
    set_expandable_segments(True)  # 依赖 expandable segments 管理碎片
```

### 5.3 长期碎片化风险评估

| 模式 | 碎片化风险 | 原因 |
|------|----------|------|
| Hybrid + sleep_mode=True | **低** | vLLM 的 sleep/wake 完全释放/重分配，不依赖 PyTorch allocator |
| Hybrid + sleep_mode=False | **中** | 依赖 PyTorch allocator + expandable_segments |
| 分离式 | **低** | Rollout 引擎不需要 sleep/wake 循环（权重原地覆写） |

---

## 6. 调度优先级与饥饿问题（Explore Agent + vLLM/SGLang Expert #2 贡献）{#6-调度饥饿}

### 6.1 恢复后的请求竞争

当权重同步完成后，`FullyAsyncRollouter._processor_worker` 恢复运行。它**优先从 cancel_queue 取样本**：

```python
# fully_async_rollouter.py:529-534
if not self.cancel_queue.empty():
    rollout_sample = await self.cancel_queue.get()  # ★ 优先恢复被取消的
else:
    rollout_sample = await self.pending_queue.get()  # 否则取新样本
```

但这只是在**提交给推理引擎之前**的优先级。当恢复的长前缀请求和新的短请求**同时到达推理引擎**时，调度由推理引擎的 scheduler 决定。

### 6.2 vLLM 的调度策略

verl 配置 vLLM 使用 `scheduling_policy: "fcfs"`（`rollout.py:173`）——**First Come First Serve**。

vLLM V1 的 scheduler 在每个 step 中：
1. 从 `waiting_queue` 按 FCFS 顺序取请求
2. 对每个请求，检查是否有足够的 KV Cache blocks
3. 如果不够，停止（不会跳过大请求取小请求）

**饥饿风险**：如果一个 64K token 的恢复请求排在队列前面，但 KV Cache 不够分配 64K blocks，那么：
- **该请求被阻塞**
- **其后所有请求都被阻塞**（FCFS 不允许跳过）
- 直到其他运行中的请求完成并释放 blocks

vLLM 支持 `priority` 参数（`vllm_async_server.py:513, 574`）：

```python
generator = self.engine.generate(
    prompt=prompt,
    sampling_params=sampling_params,
    request_id=request_id,
    priority=priority,  # 默认 0
)
```

但 verl **未设置不同优先级**——所有请求的 priority 都是 0。

### 6.3 SGLang 的调度策略

SGLang 支持更丰富的调度策略（vLLM/SGLang Expert #2 从 SGLang 源码确认）：

```python
# sglang/srt/managers/schedule_policy.py
class CacheAwarePolicy(Enum):
    LPM = "lpm"          # Longest Prefix Match：优先调度前缀命中率高的
    DFS_WEIGHT = "dfs-weight"  # 深度优先搜索加权

class CacheAgnosticPolicy(Enum):
    FCFS = "fcfs"        # 先来先服务
    LOF = "lof"          # 最长输出优先
    RANDOM = "random"
```

verl 配置 SGLang 使用 `scheduling_policy: "fcfs"`（默认）。如果切换为 `"lpm"`（Longest Prefix Match），在 prefix cache 被清空后**等效于 FCFS**（没有前缀可以 match）。

### 6.4 SGLang 的 Abort 机制（vLLM/SGLang Expert #2 关键发现）

SGLang 的 `abort_request`（scheduler.py:2328-2360）处理**不同队列中的请求**：

1. **waiting_queue 中的请求**：直接 `pop`，**KV Cache 未分配**，无需释放
2. **grammar_queue 中的请求**：`set_finish_with_abort` + `input_ids` 替换为 1 个 token，**下次 forward pass 成本极低**
3. **running batch 中的请求**：通过 `finished` 标志在下次调度循环结束，**KV Cache 在 batch 完成后释放**

**注意**：running batch 中的请求**不会立即释放** KV Cache——它们会在当前 forward pass 完成后才被清理。这意味着 abort 后有一个短暂的窗口，KV Cache 仍被占用。

### 6.5 饥饿场景分析

**场景**：权重同步后，cancel_queue 有 10 条 64K 恢复请求，pending_queue 有 100 条新的 2K 请求。

```
Rollouter 提交顺序（cancel_queue 优先）：
  [64K恢复#1] [64K恢复#2] ... [64K恢复#10] [2K新#1] [2K新#2] ...

推理引擎 waiting_queue（FCFS）：
  [64K恢复#1] [64K恢复#2] ... [64K恢复#10] [2K新#1] [2K新#2] ...
```

**推理引擎调度**：
1. 64K 恢复#1 需要 ~500 blocks（64K / 128 tokens per block），如果 pool 有 2000 blocks，可以调度
2. 64K 恢复#2 需要另外 500 blocks → 1000/2000 used
3. 64K 恢复#3 → 1500/2000 used
4. 64K 恢复#4 → 2000/2000 used → **后续所有请求被阻塞**
5. 2K 新请求虽然只需 ~16 blocks，但在 FCFS 下**排在 64K 恢复请求后面**
6. 必须等到某个 64K 完成后才能调度

**这不是"饥饿"（长期无法被服务），而是"延迟"**——短请求被长请求阻塞，但最终会被服务。

**真正的饥饿场景**：如果 KV Cache pool **不够分配一个** 64K 请求（例如 pool 只有 400 blocks < 500 needed），那么：
- 64K 请求被**永久阻塞**在 waiting_queue 头部
- 所有后续请求（包括短的）也被阻塞
- 只有当 `max_model_len` 允许且 KV Cache pool 足够时才能调度

### 6.6 Chunked Prefill 的缓解作用

verl 默认启用 `enable_chunked_prefill: True`（`rollout.py:220`），并配置 `max_num_batched_tokens: 8192`（`rollout.py:171`）。

Chunked Prefill 将大 prefill 分块处理：
- 64K prefill 被分成 8192/次 = 8 个 chunks
- 每个 chunk 可以与正在运行的 decode 请求**共享 GPU 时间**
- 这减少了 prefill 的 latency spike，但**不减少 KV Cache 的总需求**

**对饥饿的影响**：Chunked Prefill 不解决 KV Cache 分配的问题——即使分块 prefill，最终还是需要 500 blocks 才能完成整个 64K 的 prefill。

### 6.7 Preemption（vLLM/SGLang 内置机制）

vLLM 和 SGLang 都支持**请求 preemption**——当内存不足时，正在运行的低优先级请求可以被 swap 到 CPU 或 recompute，为高优先级请求腾出空间。

verl 的 `TokenOutput` 中有 `num_preempted` 字段（`replica.py:48-49`），并在 metrics 中追踪。说明 preemption 在实际运行中确实发生。

但 preemption 的成本是高昂的——被 preempt 的请求需要重新 prefill（swap 回 KV Cache 或从头计算），增加端到端延迟。

---

## 7. "大前缀"问题：64K Re-Prefill 的显存需求（Architect Reviewer 贡献）{#7-大前缀问题}

### 7.1 KV Cache 需求估算

对于一条 64K token 的恢复序列（原始 prompt + 已生成的部分 response 作为新 "prompt"）：

**每个 KV block 的内存**（以 Llama-70B 为例）：
```
每 block = page_size × num_layers × 2 (K+V) × num_kv_heads × head_dim × dtype_size
= 128 tokens × 80 layers × 2 × 8 heads × 128 dim × 2 bytes (bf16)
= 128 × 80 × 2 × 8 × 128 × 2 = 41.9 MB per block
```

**64K tokens 需要的 blocks**：
```
64K / 128 = 512 blocks
512 × 41.9 MB = 21.5 GB per sequence
```

### 7.2 单 GPU 能容纳多少 64K 序列？

**80 GB GPU，gpu_memory_utilization=0.5**：
```
总 KV Cache 预算 = 80 × 0.5 - 模型权重(~17.5 GB per shard) ≈ 22.5 GB
每条 64K 序列 ≈ 21.5 GB
→ ★ 只能容纳 ~1 条 64K 序列！
```

**80 GB GPU，gpu_memory_utilization=0.9**（分离式，无训练）：
```
总 KV Cache 预算 = 80 × 0.9 - 17.5 ≈ 54.5 GB
→ 可容纳 ~2 条 64K 序列
```

### 7.3 保护机制

verl 通过以下配置参数限制：

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `max_model_len` | `None`（使用模型 config） | 限制单条序列最大长度 |
| `max_num_seqs` | 1024 | 限制并发序列数 |
| `max_num_batched_tokens` | 8192 | 限制单次 prefill 的 token 数 |
| `gpu_memory_utilization` | 0.5 | KV Cache 占 GPU 内存比例 |
| `enable_chunked_prefill` | True | 分块 prefill 减少峰值 |

**但没有针对"恢复序列"的特殊限制**——恢复的 64K prompt 与普通 64K prompt 被同等对待。

### 7.4 万卡规模下的 Re-Prefill 冲击

**场景**：10,000 GPU，每 GPU 有 1 条 64K 中断序列，权重同步后同时恢复。

**计算资源冲击**（Prefill 是 Compute-bound）：
```
单条 64K Prefill FLOPs ≈ 2 × 70B × 64K ≈ 9 × 10^15
H100 (990 TFLOPS bf16) → 9.1 秒
H20 (~148 TFLOPS bf16) → 60.8 秒
```

**所有 10,000 GPU 同时做 64K Prefill**：
- 功耗从 Decode 的 ~30% utilization 跳到 ~100% utilization
- 网络压力来自同时的 NCCL weight broadcast
- 如果电源管理有功率限制，可能触发降频

**缓解**：
1. 分离式模式中，不同 Rollouter 的同步时间自然交错（不是所有 GPU 同时同步）
2. Chunked Prefill 将 64K prefill 分成 ~8 个 step，降低瞬时计算峰值
3. `max_concurrent_samples`（`fully_async_rollouter.py:215`）限制并发提交量

---

## 8. verl 的显存保护机制总览 {#8-保护机制}

| 机制 | 源码位置 | 作用 |
|------|---------|------|
| `aggressive_empty_cache()` | `memory_utils.py:31` | 多次重试的 gc + empty_cache + sync |
| `set_expandable_segments()` | `device.py:132` | 切换 CUDA 分配器模式，兼容 vLLM |
| `sleep(level=2)` | `vllm_async_server.py:639` | 完全释放推理引擎 GPU 内存 |
| `gpu_memory_utilization=0.5` | `rollout.py:161` | Hybrid 模式下限制推理引擎内存 |
| `free_cache_engine=True` | `rollout.py:165` | 允许 sleep/wake 内存管理 |
| `enable_sleep_mode=True` | `rollout.py:242` | 启用 vLLM 的 sleep API |
| `enable_chunked_prefill=True` | `rollout.py:220` | 分块 prefill 降低峰值 |
| `max_num_batched_tokens=8192` | `rollout.py:171` | 限制单次 prefill token 数 |
| `max_num_seqs=1024` | `rollout.py:181` | 限制并发序列数 |
| CPU offload（`_is_offload_param`） | `fsdp_workers.py:790-791` | FSDP 参数 offload 到 CPU |
| NCCL 双缓冲（CuPy 绕过 PyTorch） | `nccl_checkpoint_engine.py:131-132` | 避免 expandable_segments 冲突 |
| `finalize()` 释放 buffer | `nccl_checkpoint_engine.py:148-151` | 传输后释放 NCCL 缓冲区 |

---

## 9. 架构缺口与改进建议 {#9-改进建议}

### 9.1 缺口 1：分离式模式下 abort 不释放模型权重

**当前**：`CheckpointEngineManager.update_weights()` 对 non-naive 后端只调用 `abort_all_requests()`（释放 KV Cache），不调用 `sleep()`（不释放权重）。

**影响**：权重覆写期间，旧权重和 NCCL buffer 同时在 GPU 上。

**建议**：对于显存紧张的场景，添加可选的 sleep 步骤：

```python
# checkpoint_engine/base.py update_weights():
if self.config.sleep_before_sync:
    await asyncio.gather(*[r.sleep() for r in self.replicas])  # 释放权重
```

### 9.2 缺口 2：恢复序列没有调度优先级控制

**当前**：所有请求 priority=0，恢复的长前缀请求可能阻塞后续短请求。

**建议**：
- 对恢复序列设置**较低优先级**（让新序列先跑起来，避免资源争夺）
- 或者反过来设置**较高优先级**（尽快完成半成品，释放流水线）

```python
# vllm_async_server.py generate():
priority = -1 if is_restored_sequence else 0  # 恢复序列低优先级
```

### 9.3 缺口 3：缺少 KV Cache 需求预检

**当前**：恢复序列被直接提交给推理引擎，如果 KV Cache 不足，会被调度器阻塞。

**建议**：在提交前检查 KV Cache 容量，如果不够容纳恢复序列的需求，**放弃恢复**（将其当作新请求从头生成）：

```python
available_blocks = engine.get_available_blocks()
required_blocks = len(restored_prompt_ids) // page_size
if required_blocks > available_blocks * 0.8:  # 留 20% 余量
    # 放弃恢复，当作新请求
    prompt_ids = original_prompt_ids  # 不拼接已生成的 token
```

### 9.4 缺口 4：无显存碎片化监控

**当前**：`memory_utils.py` 提供了 `get_memory_info()` 和 `log_memory_usage()`，但没有碎片化指标。

**建议**：添加碎片化比率监控：

```python
fragmentation = 1 - (largest_free_block / total_free_memory)
```

PyTorch 的 `torch.cuda.memory_stats()` 提供了 `inactive_split` 等指标，可用于碎片化估计。

---

## 10. 关键源码索引 {#10-源码索引}

### 显存管理核心

| 功能 | 文件 | 行号 | Agent 来源 |
|------|------|------|-----------|
| `aggressive_empty_cache()` | `verl/utils/memory_utils.py` | 31-75 | FSDP Expert |
| `set_expandable_segments()` | `verl/utils/device.py` | 132-146 | FSDP Expert |
| `rollout_mode()` 显存编排 | `verl/workers/fsdp_workers.py` | 742-847 | FSDP Expert |
| Sleep Level 定义 | `verl/third_party/vllm/__init__.py` | 33-49 | FSDP Expert |

### Abort/Resume 机制

| 功能 | 文件 | 行号 | Agent 来源 |
|------|------|------|-----------|
| vLLM `abort_all_requests()` | `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 676-737 | vLLM Expert #1 |
| vLLM `pause_generation()` | `vllm_async_server.py` | 700-704 | vLLM Expert #1 |
| SGLang `abort_request()` | SGLang `scheduler.py` (外部包) | 2328-2360 | SGLang Expert #2 |
| SGLang `pause_generation()` | `async_sglang_server.py` | 424-426 | SGLang Expert #2 |
| Partial 模式 `cancel()` | `fully_async_policy/vllm_rollout/vllm_async_server.py` | 137-141 | vLLM Expert #1 |

### 权重传输与内存

| 功能 | 文件 | 行号 | Agent 来源 |
|------|------|------|-----------|
| `BucketedWeightSender` | `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` | 73-197 | vLLM Expert #1 |
| `BucketedWeightReceiver` (clone) | 同上 | 200-301 | vLLM Expert #1 |
| NCCL 双缓冲分配 | `verl/checkpoint_engine/nccl_checkpoint_engine.py` | 128-136 | Architect |
| NCCL send/receive | 同上 | 223-362 | Architect |
| `CheckpointEngineManager.update_weights()` | `verl/checkpoint_engine/base.py` | 402-437 | Architect |

### 调度与优先级

| 功能 | 文件 | 行号 | Agent 来源 |
|------|------|------|-----------|
| `cancel_queue` 优先级 | `verl/experimental/fully_async_policy/fully_async_rollouter.py` | 529-534 | Explore |
| `max_concurrent_samples` | 同上 | 215-216 | Explore |
| vLLM `priority` 参数 | `vllm_async_server.py` | 513, 574 | Explore |
| `scheduling_policy: "fcfs"` | `verl/workers/config/rollout.py` | 173 | Explore |
| SGLang `CacheAwarePolicy` | SGLang `schedule_policy.py` (外部) | 62-74 | SGLang Expert #2 |
| SGLang `calc_priority()` | 同上 | 98-108 | SGLang Expert #2 |

### 配置参数

| 参数 | 文件 | 行号 | 默认值 |
|------|------|------|--------|
| `gpu_memory_utilization` | `verl/workers/config/rollout.py` | 161 | 0.5 |
| `free_cache_engine` | 同上 | 165 | True |
| `enable_sleep_mode` | 同上 | 242 | True |
| `enable_chunked_prefill` | 同上 | 220 | True |
| `enable_prefix_caching` | 同上 | 222 | True |
| `max_num_seqs` | 同上 | 181 | 1024 |
| `max_num_batched_tokens` | 同上 | 171 | 8192 |
| `update_weights_bucket_megabytes` | 同上 | 132 | 2048 |

---

## 总结

verl 对"显存碎片隐形杀手"问题的回答是**完全释放、重新分配**的策略。具体而言：

1. **显存滞留问题**：verl 选择**释放**而非锁定。Abort 后 KV Cache blocks 回到 pool，sleep(level=2) 后权重也释放。不存在"锁定等待"导致的 OOM。

2. **碎片化问题**：通过 `aggressive_empty_cache()` + `expandable_segments` 切换 + vLLM 的原生 sleep/wake 管理，碎片化风险可控。

3. **调度饥饿问题**：这是**实际存在但尚未解决**的问题。恢复的 64K 长前缀请求在 FCFS 策略下会阻塞后续短请求。verl 已具备 `priority` 参数但未使用，SGLang 有更丰富的调度策略但 verl 默认使用 FCFS。

4. **OOM 风险**：在 Hybrid 模式下风险低（sleep 释放了大量内存）；在分离式模式下中等（旧权重 + NCCL 缓冲区共存）；在极端配置下（低 TP + 大模型）有风险。

5. **"大前缀"是真正的瓶颈**：单条 64K 序列可能消耗 ~21.5 GB KV Cache（70B 模型），在 `gpu_memory_utilization=0.5` 的 80 GB GPU 上几乎填满全部 KV Cache 预算。这是 long-context RL 的根本性物理限制。
