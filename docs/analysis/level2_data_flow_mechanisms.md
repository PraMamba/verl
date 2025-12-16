# [Level 2] 数据流转机制

**面向对象**: 基础设施（Infrastructure）初学者  
**学习时长**: 建议 4-5 天（32-40 小时）  
**前置知识**: Level 0 (分布式基础) + Level 1 (并行策略)  
**最后更新**: 2025-12-15

---

## 📋 目录

1. [学习目标](#学习目标)
2. [RL训练三阶段数据流概览](#rl训练三阶段数据流概览)
3. [问题 1: Stage1→Stage2 的 TP→DP 转换](#问题1-stage1stage2-的-tpdp-转换)
4. [问题 2: 不同并行度间的数据重分布](#问题2-不同并行度间的数据重分布)
5. [问题 3: Ray ObjectRef 的异步数据流](#问题3-ray-objectref-的异步数据流)
6. [问题 4: Batch 切分与聚合的数学原理](#问题4-batch-切分与聚合的数学原理)
7. [问题 5: 数据流中的同步点识别](#问题5-数据流中的同步点识别)
8. [概念验证实验](#概念验证实验)
9. [源码阅读指南](#源码阅读指南)
10. [自我检测清单](#自我检测清单)

---

## 学习目标

完成本 Level 后，你将能够：

- [ ] ✅ **理解RL三阶段**: 能画出 stage1/2/3 的完整数据流图
- [ ] ✅ **掌握DP↔TP转换**: 能解释为何需要 all-gather 和 chunk 操作
- [ ] ✅ **理解异步传递**: 能说明 Ray ObjectRef 如何避免阻塞
- [ ] ✅ **计算数据形状**: 给定并行配置，能推导每个阶段的 tensor shape
- [ ] 🔍 **识别同步点**: 能在代码中标注所有 barrier 和通信操作
- [ ] 🔍 **优化数据流**: 能提出减少通信开销的改进方案

---

## RL训练三阶段数据流概览

### 伪代码流程

```python
for prompts in dataloader:
    # Stage 1: 生成响应 (Response Generation)
    batch = actor.generate_sequences(prompts)
    
    # Stage 2: 准备训练数据 (Data Preparation)
    batch = critic.compute_values(batch)
    batch = reference.compute_log_prob(batch)
    batch = reward.compute_reward(batch)
    batch = compute_advantages(batch)
    
    # Stage 3: 模型训练 (Model Training)
    critic_metrics = critic.update_critic(batch)
    actor_metrics = actor.update_actor(batch)
```

### 数据流可视化

```
Dataloader
    ↓ [prompts: B×L_prompt]
┌─────────────────────────────┐
│  Stage 1: Generate          │
│  Parallel: DP=8, rollout TP=4│
│  Input: B×L_prompt          │
│  Output: B×(L_prompt+L_resp)│
└─────────────────────────────┘
    ↓ [完整序列: B×L_total]
┌─────────────────────────────┐
│  Stage 2: Prepare           │
│  Parallel: DP=8             │
│  - Critic: values           │
│  - Ref: old_logprobs        │
│  - Reward: scores           │
└─────────────────────────────┘
    ↓ [batch with labels]
┌─────────────────────────────┐
│  Stage 3: Train             │
│  Parallel: DP=8             │
│  - Actor update (PPO)       │
│  - Critic update            │
└─────────────────────────────┘
```

### 关键挑战

| 阶段 | 并行策略 | 数据分布 | 挑战 |
|------|---------|---------|------|
| Stage 1 | rollout TP=4 | 各 TP rank 输入相同 | TP→DP 转换 |
| Stage 2 | actor/critic DP=8 | 各 DP rank 输入不同 | 重计算 logprobs |
| Stage 3 | actor/critic DP=8 | 各 DP rank 输入不同 | 梯度同步 |

💡 **核心问题**: Stage1 使用 TP（为了显存），Stage2/3 使用 DP（为了效率），如何转换？

---

## 问题1: Stage1→Stage2 的 TP→DP 转换

### 提问目标

理解为什么 rollout 用 TP、训练用 DP，以及如何在两者之间转换数据。

### 深挖细节

#### 细节问题 1.1: 为什么 rollout 要用 TP？

**答案**: 推理引擎（vLLM/SGLang）针对 TP 做了深度优化，性能远超 DP。

```
DP=4 的 generate (transformers):
   每个 GPU: 完整模型 (7B×2 bytes = 14GB)
   KV cache: 独立分配
   总显存: 4×(14GB + KV) ≈ 60GB+

TP=4 的 generate (vLLM):
   每个 GPU: 1/4 模型 (7B/4×2 bytes = 3.5GB)
   KV cache: 共享管理 (PagedAttention)
   总显存: 4×(3.5GB + KV/4) ≈ 20GB

性能提升: 2-3× (通过 kernel fusion)
```

#### 细节问题 1.2: TP→DP 的数据转换流程是什么？

**具体流程** (以 world_size=8, rollout TP=4 为例):

```
Rollout 阶段 (TP=4):
   GPU 0-3: TP group 0
      输入: prompts [B, L_prompt] (相同)
      输出: responses [B, L_total] (相同)
   
   GPU 4-7: TP group 1
      输入: prompts [B, L_prompt] (相同)
      输出: responses [B, L_total] (相同)

↓ 转换 (TP→DP)

Training 阶段 (DP=8):
   GPU 0: batch[0] [B/8, L_total]
   GPU 1: batch[1] [B/8, L_total]
   ...
   GPU 7: batch[7] [B/8, L_total]
```

**代码示意**:

```python
# verl/workers/sharding_manager/fsdp_vllm.py (简化)

class FSDPVLLMShardingManager:
    def __enter__(self):
        # Stage 1: TP generate
        # 此时各 TP rank 的输出相同
        
        # 转换: 在 TP group 内 chunk
        tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        tp_rank = vllm_ps.get_tensor_model_parallel_rank()
        
        # 把 TP group 内的数据聚合
        data.batch = allgather_dict_tensors(
            data.batch.contiguous(),
            size=tp_size,
            group=vllm_ps.get_tensor_model_parallel_group(),
            dim=0  # batch 维度
        )
```

#### 细节问题 1.3: 为什么不能直接用 DP rollout？

**对比**:

| 方案 | 优点 | 缺点 |
|------|------|------|
| **DP rollout** | 数据流简单 | 显存占用大，无法用 vLLM 优化 |
| **TP rollout** | 显存小，速度快 | 需要数据转换 |

**实际选择**: TP rollout + 数据转换（转换开销 << 性能收益）

### 代码路径

**关键文件**:
- `verl/workers/sharding_manager/fsdp_vllm.py:80-150`
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py:120-180`

**关键函数**:

```python
# verl/workers/sharding_manager/fsdp_vllm.py:108

def preprocess_data(data):
    """DP→TP: chunk 数据给各 TP rank"""
    tp_size = vllm_ps.get_tensor_model_parallel_world_size()
    
    # 把 DP 的数据切分给 TP
    for key in data.batch.keys():
        tensor = data.batch[key]
        # 沿 batch 维度切分
        data.batch[key] = allgather_dict_tensors(
            tensor.contiguous(),
            size=tp_size,
            group=vllm_ps.get_tensor_model_parallel_group(),
            dim=0
        )
    return data
```

### 实践任务

#### 任务 1.1: 模拟 TP→DP 转换

创建 `test_tp_dp_conversion.py`:

```python
import torch
import torch.distributed as dist

def test_tp_to_dp_conversion():
    """
    模拟: TP=2, DP=4 的数据转换
    """
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()  # 4
    
    # 假设 TP=2, DP=2
    # Rank 0,1 是 TP group 0
    # Rank 2,3 是 TP group 1
    
    tp_size = 2
    dp_size = 2
    tp_rank = rank % tp_size
    dp_rank = rank // tp_size
    
    # 创建 TP 和 DP 进程组
    tp_groups = []
    for i in range(dp_size):
        tp_group = dist.new_group([i*tp_size, i*tp_size+1])
        tp_groups.append(tp_group)
    
    my_tp_group = tp_groups[dp_rank]
    
    # Rollout 阶段: TP generate
    # 模拟输出 (各 TP rank 相同)
    B, L = 8, 10
    if dp_rank == 0:
        # TP group 0 生成 batch 0-3
        output = torch.arange(B//dp_size * L, device='cuda').reshape(B//dp_size, L)
    else:
        # TP group 1 生成 batch 4-7
        output = torch.arange(B//dp_size * L, device='cuda').reshape(B//dp_size, L) + 1000
    
    # 确保 TP group 内相同
    dist.broadcast(output, src=dp_rank*tp_size, group=my_tp_group)
    
    print(f"[Rank {rank}] After TP generate: {output.shape}, first elem: {output[0,0].item()}")
    
    # TODO: 转换到 DP
    # 提示: 需要在 TP group 内做什么操作？
    
    dist.destroy_process_group()

if __name__ == "__main__":
    test_tp_to_dp_conversion()
```

运行:
```bash
torchrun --nproc_per_node=4 test_tp_dp_conversion.py
```

**思考**:
- TP group 内的输出相同，如何避免重复存储？
- 转换到 DP 后，每个 rank 应该拿哪部分数据？

---

## 问题2: 不同并行度间的数据重分布

### 提问目标

掌握通用的数据重分布方法，能处理任意 DP/TP 组合的转换。

### 深挖细节

#### 细节问题 2.1: DP=8 如何转换到 TP=4？

**场景**: Actor (DP=8) → Rollout (TP=4)

```
Before (DP=8):
   Rank 0: batch[0] [B/8, L]
   Rank 1: batch[1] [B/8, L]
   ...
   Rank 7: batch[7] [B/8, L]

After (TP=4, 2 个 TP groups):
   TP Group 0 (Rank 0-3):
      输入: [B/2, L] (gather from Rank 0-3)
   
   TP Group 1 (Rank 4-7):
      输入: [B/2, L] (gather from Rank 4-7)
```

**操作**: 在各 TP group 内 all-gather

```python
# 伪代码
tp_group_rank = rank % tp_size
dp_group_size = world_size // tp_size

# 在 TP group 内 gather
gathered = all_gather_in_group(my_data, tp_group)  # [dp_group_size×B/8, L]
```

#### 细节问题 2.2: 通用的数据重分布公式是什么？

**定义**:
- `N`: 总进程数 (world_size)
- `DP_old`: 原 DP 度
- `TP_old`: 原 TP 度
- `DP_new`: 新 DP 度
- `TP_new`: 新 TP 度

**约束**: `DP_old × TP_old = DP_new × TP_new = N`

**转换策略**:

| 转换类型 | 操作 | 通信原语 |
|---------|------|---------|
| DP 增大 (TP 减小) | chunk | scatter 或 取 slice |
| DP 减小 (TP 增大) | gather | all-gather |
| DP 不变, TP 交换 | 复杂重排 | all-to-all |

#### 细节问题 2.3: verl 如何优化 TP 切换？

**verl 的 micro_dp_group 设计**:

```
传统 Megatron (TP 优先):
   TP=2, DP=4 → Rank 排列:
   [Rank 0, Rank 1] TP group 0
   [Rank 2, Rank 3] TP group 1
   [Rank 4, Rank 5] TP group 2
   [Rank 6, Rank 7] TP group 3

verl 优化 (DP 优先):
   TP=2, DP=4 → Rank 排列:
   [Rank 0, Rank 1] TP group 0
   [Rank 2, Rank 3] TP group 0 (重复)
   [Rank 4, Rank 5] TP group 1
   [Rank 6, Rank 7] TP group 1 (重复)
```

**好处**: TP=4→TP=2 时，只需在相邻 rank 间通信！

### 代码路径

**关键文件**:
- `verl/workers/sharding_manager/megatron_vllm.py:150-250` (Megatron 的重排)
- `verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py:80-120` (进程组管理)

**关键代码**:

```python
# verl/workers/sharding_manager/megatron_vllm.py:180

def merge_tensor_parallel_group(tensor, from_tp, to_tp):
    """
    从 TP=from_tp 转到 TP=to_tp
    """
    assert from_tp % to_tp == 0, "必须整除"
    
    merge_factor = from_tp // to_tp
    
    # 创建 micro_dp_group (相邻 merge_factor 个 rank)
    micro_dp_groups = []
    for i in range(0, world_size, merge_factor):
        group = dist.new_group(list(range(i, i+merge_factor)))
        micro_dp_groups.append(group)
    
    # 在 micro_dp_group 内 all-gather 并 concat
    merged = all_gather_and_concat(tensor, micro_dp_groups[my_group_idx])
    
    return merged
```

### 实践任务

#### 任务 2.1: 实现通用数据重分布

完成框架代码:

```python
def redistribute_data(tensor, from_dp, from_tp, to_dp, to_tp):
    """
    通用数据重分布
    
    Args:
        tensor: 当前 rank 的数据 [B_local, ...]
        from_dp, from_tp: 当前并行度
        to_dp, to_tp: 目标并行度
    
    Returns:
        redistributed: 重分布后的数据
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    assert from_dp * from_tp == to_dp * to_tp == world_size
    
    # TODO: 实现重分布逻辑
    # 提示: 
    # 1. 计算当前在哪个 from_group
    # 2. 计算目标在哪个 to_group
    # 3. 如果 to_tp > from_tp: gather
    #    如果 to_tp < from_tp: scatter
    
    pass
```

---

## 问题3: Ray ObjectRef 的异步数据流

### 提问目标

理解 Ray 如何管理数据流，避免阻塞式传输，提高训练效率。

### 深挖细节

#### 细节问题 3.1: 什么是 ObjectRef？

**定义**: ObjectRef 是 Ray 中的异步数据引用（类似 Future/Promise）。

```python
import ray

# 同步方式（阻塞）
result = slow_function()  # 等待完成
use_result(result)

# 异步方式（非阻塞）
ref = slow_function.remote()  # 立即返回 ObjectRef
# 可以做其他事情...
result = ray.get(ref)  # 需要时才等待
use_result(result)
```

**优势**: 隐藏延迟，提高并发度。

#### 细节问题 3.2: verl 如何使用 ObjectRef 优化流程？

**传统 SPMD 方式**:

```python
# Stage 1
responses = rollout_model.generate(prompts)  # 阻塞

# Stage 2 (必须等 Stage 1 完成)
values = critic_model.compute_values(responses)  # 阻塞
```

**Ray 方式**:

```python
# Stage 1 (返回 ObjectRef)
responses_ref = rollout_worker.generate.remote(prompts)  # 非阻塞！

# 可以立即启动 Stage 2 (传递 ref)
values_ref = critic_worker.compute_values.remote(responses_ref)  # 非阻塞！

# Ray 自动处理依赖: critic 会等 rollout 完成
values = ray.get(values_ref)  # 最终获取结果
```

**好处**: Driver 不阻塞，可以同时调度多个任务。

#### 细节问题 3.3: ObjectRef 的数据何时物化？

**答案**: 在 `ray.get()` 或 worker 实际使用时。

```python
@ray.remote
class Worker:
    def process(self, data_ref):
        # 这里 data_ref 自动被 ray.get()
        data = data_ref  # Ray 内部物化
        return data * 2

# Driver
data_ref = expensive_compute.remote()
result_ref = worker.process.remote(data_ref)  # 传递 ref

# data 仅在 Worker.process 内部物化，Driver 无需等待
```

### 代码路径

**关键文件**:
- `verl/single_controller/ray/base.py:200-300` (RayWorkerGroup)
- `verl/single_controller/base/decorator.py:50-120` (@register 装饰器)

**关键代码**:

```python
# verl/single_controller/base/decorator.py:80

def _materialize_futures(*args, **kwargs):
    """自动物化 ObjectRef"""
    def _materialize(obj):
        if isinstance(obj, ray.ObjectRef):
            return ray.get(obj)
        elif isinstance(obj, list):
            return [_materialize(x) for x in obj]
        return obj
    
    new_args = [_materialize(arg) for arg in args]
    new_kwargs = {k: _materialize(v) for k, v in kwargs.items()}
    
    return new_args, new_kwargs
```

### 实践任务

#### 任务 3.1: 对比同步vs异步数据流

创建 `test_async_dataflow.py`:

```python
import ray
import time
import torch

@ray.remote
class SlowWorker:
    def slow_compute(self, data, sleep_time=1.0):
        time.sleep(sleep_time)  # 模拟耗时计算
        return data * 2

def sync_pipeline():
    """同步流水线"""
    worker = SlowWorker.remote()
    
    start = time.time()
    
    # 顺序执行
    data1 = torch.tensor([1.0])
    result1 = ray.get(worker.slow_compute.remote(data1, 1.0))
    
    data2 = torch.tensor([2.0])
    result2 = ray.get(worker.slow_compute.remote(data2, 1.0))
    
    elapsed = time.time() - start
    print(f"同步执行: {elapsed:.2f}s")
    return elapsed

def async_pipeline():
    """异步流水线"""
    worker = SlowWorker.remote()
    
    start = time.time()
    
    # 并发执行
    data1 = torch.tensor([1.0])
    ref1 = worker.slow_compute.remote(data1, 1.0)  # 不等待
    
    data2 = torch.tensor([2.0])
    ref2 = worker.slow_compute.remote(data2, 1.0)  # 不等待
    
    # 一起等待
    result1, result2 = ray.get([ref1, ref2])
    
    elapsed = time.time() - start
    print(f"异步执行: {elapsed:.2f}s")
    return elapsed

if __name__ == "__main__":
    ray.init()
    
    t_sync = sync_pipeline()
    t_async = async_pipeline()
    
    print(f"加速比: {t_sync / t_async:.2f}x")
    
    ray.shutdown()
```

**预期输出**:
```
同步执行: 2.00s
异步执行: 1.00s
加速比: 2.00x
```

---

## 问题4: Batch 切分与聚合的数学原理

### 提问目标

理解 batch 在不同 rank 间的切分规则，能计算任意配置下的 local batch size。

### 深挖细节

#### 细节问题 4.1: DP 的 batch 切分公式？

**定义**:
- `B_global`: 全局 batch size
- `N_dp`: DP 并行度
- `B_local`: 每个 DP rank 的 local batch size

**公式**:
```
B_local = B_global / N_dp
```

**示例**:
```
B_global = 128, N_dp = 8
B_local = 128 / 8 = 16

Rank 0: batch[0:16]
Rank 1: batch[16:32]
...
Rank 7: batch[112:128]
```

#### 细节问题 4.2: TP 的 batch 为何不切分？

**答案**: TP 切分的是**模型参数**，不是数据！

```
TP=4 的 forward:
   输入 X: [B, D] (所有 rank 相同)
   
   Rank 0: Y_0 = X W_0  (W_0: [D, D/4])
   Rank 1: Y_1 = X W_1  (W_1: [D, D/4])
   Rank 2: Y_2 = X W_2  (W_2: [D, D/4])
   Rank 3: Y_3 = X W_3  (W_3: [D, D/4])
   
   all-gather → Y = [Y_0, Y_1, Y_2, Y_3]
```

💡 **关键**: TP 要求输入在所有 rank 上**相同**！

#### 细节问题 4.3: 混合并行的 batch 计算？

**场景**: DP=4, TP=2 (world_size=8)

```
全局: B_global = 128

DP 维度切分:
   DP group 0 (Rank 0-1): B_dp = 128/4 = 32
   DP group 1 (Rank 2-3): B_dp = 32
   DP group 2 (Rank 4-5): B_dp = 32
   DP group 3 (Rank 6-7): B_dp = 32

TP 维度不切分:
   Rank 0: B_local = 32 (与 Rank 1 相同)
   Rank 1: B_local = 32
   ...
```

**通用公式**:
```python
dp_size = world_size // tp_size
local_batch_size = global_batch_size // dp_size
```

### 代码路径

**关键函数**:
```python
# verl/single_controller/base/worker_group.py:150

def _chunk_data_by_dp(data, dp_size, dp_rank):
    """按 DP 切分数据"""
    if isinstance(data, torch.Tensor):
        chunk_size = data.size(0) // dp_size
        start = dp_rank * chunk_size
        end = start + chunk_size
        return data[start:end]
    
    elif isinstance(data, dict):
        return {k: _chunk_data_by_dp(v, dp_size, dp_rank) 
                for k, v in data.items()}
    
    return data
```

### 实践任务

#### 任务 4.1: 计算各种配置的 local batch

给定配置，计算每个 rank 的 batch size:

```python
def calculate_local_batch(global_batch, world_size, dp, tp):
    """
    Args:
        global_batch: 全局 batch size
        world_size: 总进程数
        dp: Data Parallel 度
        tp: Tensor Parallel 度
    
    Returns:
        local_batch: 每个 rank 的 local batch size
    """
    assert dp * tp == world_size
    assert global_batch % dp == 0, "global_batch 必须被 dp 整除"
    
    return global_batch // dp

# 测试
configs = [
    (128, 8, 8, 1),   # 纯 DP
    (128, 8, 4, 2),   # DP=4, TP=2
    (128, 8, 2, 4),   # DP=2, TP=4
    (128, 8, 1, 8),   # 纯 TP
]

for global_batch, world_size, dp, tp in configs:
    local = calculate_local_batch(global_batch, world_size, dp, tp)
    print(f"B={global_batch}, DP={dp}, TP={tp} → local_batch={local}")
```

**预期输出**:
```
B=128, DP=8, TP=1 → local_batch=16
B=128, DP=4, TP=2 → local_batch=32
B=128, DP=2, TP=4 → local_batch=64
B=128, DP=1, TP=8 → local_batch=128
```

---

## 问题5: 数据流中的同步点识别

### 提问目标

能够在复杂的训练流程中识别所有同步点（barrier），理解同步的必要性。

### 深挖细节

#### 细节问题 5.1: 哪些操作是同步点？

**同步操作**:
1. **通信原语**: all-reduce, all-gather, broadcast, barrier
2. **进程组创建**: init_process_group()
3. **Ray 的 ray.get()**: 等待 ObjectRef 完成
4. **Context manager**: __enter__() 和 __exit__()

**示例标注**:

```python
# 1. 初始化（同步）
dist.init_process_group(backend="nccl")  # ← 同步点

# 2. Forward（可能异步）
output = model(input)  # 无同步（FSDP 内部可能有）

# 3. Backward（同步）
loss.backward()  # ← 同步点（梯度 all-reduce）

# 4. Update（同步）
optimizer.step()  # ← 同步点（参数更新）
```

#### 细节问题 5.2: verl 的 ShardingManager 中有哪些同步点？

**分析**:

```python
# verl/workers/sharding_manager/fsdp_vllm.py

class FSDPVLLMShardingManager:
    def __enter__(self):
        # 1. 获取 state_dict（同步 - FSDP unshard）
        params = self.module.state_dict()  # ← 同步点
        
        # 2. 同步参数到 vLLM（同步 - copy）
        self.inference_engine.sync_model_weights(params)  # ← 同步点
        
        # 3. 清理显存
        del params
        torch.cuda.empty_cache()  # 异步（GPU 内部队列）
        
        return self
    
    def __exit__(self, *args):
        # 4. 卸载模型（同步 - GPU 操作）
        self.inference_engine.offload_model_weights()  # ← 同步点
```

#### 细节问题 5.3: 如何减少同步开销？

**策略**:

1. **合并通信**: Bucket 机制
2. **重叠计算与通信**: Gradient accumulation
3. **异步 offload**: CPU offload with streams

**示例**:

```python
# ❌ 低效：多次小通信
for param in model.parameters():
    dist.all_reduce(param.grad)  # N 次通信

# ✅ 高效：一次大通信
all_grads = torch.cat([p.grad.flatten() for p in model.parameters()])
dist.all_reduce(all_grads)
```

### 代码路径

**关键文件**:
- `verl/workers/sharding_manager/fsdp_vllm.py` (ShardingManager 同步点)
- `verl/single_controller/ray/base.py` (Ray 调度同步点)

### 实践任务

#### 任务 5.1: 标注同步点

阅读以下代码并标注所有同步点:

```python
def training_step(model, data, optimizer):
    # TODO: 标注同步点
    
    # 1
    output = model(data)
    
    # 2
    loss = criterion(output, target)
    
    # 3
    loss.backward()
    
    # 4
    dist.all_reduce(loss)  # 收集 loss 用于 logging
    
    # 5
    optimizer.step()
    
    # 6
    optimizer.zero_grad()
    
    return loss.item()
```

**答案**: 标注哪几行是同步点？为什么？

---

## 概念验证实验

### 实验 1: 测量数据转换开销

测量 TP→DP 转换的通信时间:

```python
import torch
import torch.distributed as dist
import time

def measure_conversion_overhead(B, L, tp_size):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    
    # 模拟 TP generate 的输出
    data = torch.randn(B, L, device='cuda')
    
    # Warmup
    for _ in range(10):
        gathered = [torch.zeros_like(data) for _ in range(tp_size)]
        dist.all_gather(gathered, data)
    
    # 测量
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        gathered = [torch.zeros_like(data) for _ in range(tp_size)]
        dist.all_gather(gathered, data)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # 计算带宽
    data_size = B * L * 4 * 100  # FP32
    bandwidth = data_size / elapsed / 1e9
    
    if rank == 0:
        print(f"TP={tp_size}, B={B}, L={L}:")
        print(f"  时间: {elapsed:.3f}s")
        print(f"  带宽: {bandwidth:.2f} GB/s")
    
    dist.destroy_process_group()

# 测试
for B in [32, 64, 128]:
    measure_conversion_overhead(B, 2048, tp_size=4)
```

---

## 源码阅读指南

### 阅读任务 1: FSDPVLLMShardingManager

**文件**: `verl/workers/sharding_manager/fsdp_vllm.py:80-180`

**目标**: 理解 FSDP → vLLM 的数据流

**关键代码**:

```python
def __enter__(self):
    # 1. Unshard FSDP 参数
    params = self.module.state_dict()
    
    for name, param in params.items():
        if hasattr(param, '_full_param_padded'):
            params[name] = param._full_param_padded  # ← 关键
    
    # 2. 加载到 vLLM
    self.inference_engine.sync_model_weights(params, load_format='dtensor')
    
    # 3. 清理
    del params
    torch.cuda.empty_cache()
```

**阅读提示**:
- 关注 `_full_param_padded` 的来源
- 理解 `load_format='dtensor'` 的作用
- 追踪 `sync_model_weights` 的实现

### 阅读任务 2: Ray Decorator

**文件**: `verl/single_controller/base/decorator.py:50-150`

**目标**: 理解 `@register` 如何处理数据分发

**关键代码**:

```python
def register(dispatch_mode=Dispatch.ALL_TO_ALL, ...):
    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            # 自动物化 ObjectRef
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return func(*args, **kwargs)
        
        # 附加元数据
        setattr(inner, MAGIC_ATTR, {
            'dispatch_mode': dispatch_mode,
            'execute_mode': execute_mode,
            'blocking': blocking,
        })
        return inner
    return decorator
```

**阅读提示**:
- 理解 `dispatch_mode` 的不同模式
- 关注 `_materialize_futures` 的实现
- 追踪 `MAGIC_ATTR` 如何被 WorkerGroup 使用

---

## 自我检测清单

### 基础理解

- [ ] 能画出 RL 三阶段的数据流图
- [ ] 能解释为何 rollout 用 TP，training 用 DP
- [ ] 能说出 TP→DP 转换的具体步骤
- [ ] 能区分同步操作和异步操作
- [ ] 能计算混合并行的 local batch size

### 进阶理解

- [ ] 能设计通用的 DP/TP 转换算法
- [ ] 能解释 Ray ObjectRef 的优势
- [ ] 能识别代码中的所有同步点
- [ ] 能提出减少通信开销的方案
- [ ] 能对比不同数据流方案的优劣

### 代码能力

- [ ] 能实现 TP→DP 的数据转换
- [ ] 能测量数据转换的开销
- [ ] 能用 Ray 编写异步数据流
- [ ] 能阅读 ShardingManager 的源码
- [ ] 能调试数据形状不匹配的问题

---

## 下一步

完成 Level 2 后，进入 [Level 3: 内存管理策略](./level3_memory_optimization.md)，学习显存优化的各种技术。

---

**总结**: 本 Level 深入讲解了 verl 中的数据流转机制，重点是 DP/TP 转换和 Ray 异步数据流。掌握这些知识后，你将能够理解和优化 RL 训练的数据流效率。
