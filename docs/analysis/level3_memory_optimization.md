# [Level 3] 内存管理策略

**面向对象**: 基础设施（Infrastructure）初学者  
**学习时长**: 建议 4-5 天（32-40 小时）  
**前置知识**: Level 0-2 (分布式基础 + 并行策略 + 数据流)  
**最后更新**: 2025-12-15

---

## 📋 目录

1. [学习目标](#学习目标)
2. [显存分析框架](#显存分析框架)
3. [问题 1: vLLM 的 gpu_memory_utilization 计算](#问题1-vllm-的-gpu_memory_utilization-计算)
4. [问题 2: KV Cache 的动态管理](#问题2-kv-cache-的动态管理)
5. [问题 3: CPU Offload 的权衡](#问题3-cpu-offload-的权衡)
6. [问题 4: remove_padding 的优化原理](#问题4-remove_padding-的优化原理)
7. [问题 5: 显存碎片化与内存池](#问��5-显存碎片化与内存池)
8. [概念验证实验](#概念验证实验)
9. [源码阅读指南](#源码阅读指南)
10. [自我检测清单](#自我检测清单)

---

## 学习目标

完成本 Level 后，你将能够：

- [ ] ✅ **显存占用分析**: 能拆解训练过程的显存组成
- [ ] ✅ **理解 KV cache**: 能解释 PagedAttention 的原理
- [ ] ✅ **掌握 offload**: 能判断何时使用 CPU/Disk offload
- [ ] ✅ **优化 padding**: 能计算 remove_padding 的收益
- [ ] 🔍 **调优参数**: 能通过配置突破显存瓶颈
- [ ] 🔍 **诊断 OOM**: 能快速定位显存泄漏原因

---

## 显存分析框架

### 显存占用的五大组成部分

```
Total GPU Memory = 
    (1) Model Parameters        # 模型权重
  + (2) Optimizer States        # Adam: momentum + variance
  + (3) Gradients               # 反向传播梯度
  + (4) Activations             # 中间激活值
  + (5) KV Cache (推理)         # 注意力缓存
```

### 显存占用计算表

| 模型大小 | FP16 参数 | Adam 状态 | 梯度 | 总计 (训练) | KV Cache (推理) |
|---------|-----------|-----------|------|-------------|----------------|
| 1B | 2 GB | 8 GB | 2 GB | 12 GB | ~2 GB (B=32, L=2k) |
| 7B | 14 GB | 56 GB | 14 GB | 84 GB | ~14 GB |
| 70B | 140 GB | 560 GB | 140 GB | 840 GB | ~140 GB |

💡 **关键洞察**: 优化器状态占大头（70%+）！

### 优化策略对比

| 策略 | 节省显存 | 性能影响 | 适用场景 |
|------|---------|---------|---------|
| **Zero1** | 优化器 /N | 无 | 大模型基础优化 |
| **Zero2** | (优化器+梯度) /N | 极小 | 进一步节省 |
| **Zero3** | 全部 /N | 中等 (通信+50%) | 超大模型 |
| **CPU Offload** | 50-80% | 大 (PCIe 瓶颈) | 显存紧张时 |
| **Gradient Checkpoint** | 激活值 -70% | 中等 (重计算) | 长序列 |
| **remove_padding** | 0-30% | 无 (反而加速) | 有 padding 的数据 |

---

## 问题1: vLLM 的 gpu_memory_utilization 计算

### 提问目标

理解 vLLM 如何分配显存，掌握 verl 对这个参数的魔改。

### 深挖细节

#### 细节问题 1.1: vLLM 原生的显存分配逻辑？

**vLLM 的分配流程**:

```python
# vLLM 原生逻辑 (v0.6)

# Step 1: 加载模型
model = load_model(model_path)  # 假设占用 M GB

# Step 2: 运行 dummy input 测量峰值显存
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    model(dummy_input)
peak_memory = torch.cuda.max_memory_allocated()  # 假设 P GB

# Step 3: 计算可用于 KV cache 的显存
total_gpu_memory = torch.cuda.get_device_properties(0).total_memory  # T GB
kv_cache_memory = total_gpu_memory * gpu_memory_utilization - peak_memory

# 示例:
# T = 80 GB
# gpu_memory_utilization = 0.9
# peak_memory = 20 GB
# kv_cache = 80 * 0.9 - 20 = 52 GB
```

**问题**: 如果 RL 训练中还有 actor/critic/ref 模型，总占用可能超过 80GB！

#### 细节问题 1.2: verl 如何魔改 gpu_memory_utilization？

**verl 的改进**:

```python
# verl 魔改逻辑

# Step 1: 先加载所有训练模型
actor_model = load_model(...)   # 20 GB
critic_model = load_model(...)  # 20 GB
ref_model = load_model(...)     # 20 GB

current_used = torch.cuda.memory_allocated()  # 60 GB

# Step 2: 计算剩余显存
total_gpu_memory = 80 GB
free_memory = total_gpu_memory - current_used  # 80 - 60 = 20 GB

# Step 3: 按比例分配给 vLLM
kv_cache_memory = free_memory * gpu_memory_utilization
# 0.5 → 20 * 0.5 = 10 GB
```

**对比**:

| 场景 | vLLM 原生 | verl 魔改 |
|------|-----------|-----------|
| 解释 | 总显存的比例 | **剩余显存**的比例 |
| 示例 (80GB, util=0.9) | 72 GB (可能 OOM) | 18 GB (安全) |

**代码路径**:
- `verl/third_party/vllm/vllm_v_0_6_3/engine/llm_engine.py:150-200`

```python
# verl 魔改部分 (简化)

class LLMEngine:
    def __init__(self, ...):
        # 获取当前显存占用
        current_allocated = torch.cuda.memory_allocated()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        # verl 的改动: 基于剩余显存
        free_memory = total_memory - current_allocated
        target_kv_cache = free_memory * self.gpu_memory_utilization
```

#### 细节问题 1.3: 如何估算需要的 KV cache 大小？

**公式**:

```
KV Cache Size = 
    2 (K + V)
  × num_layers
  × num_heads
  × head_dim
  × batch_size
  × seq_length
  × bytes_per_element
```

**示例** (7B Llama2, FP16):

```python
num_layers = 32
num_heads = 32
head_dim = 128
batch_size = 32
seq_length = 2048
bytes_per_elem = 2  # FP16

kv_cache = 2 * 32 * 32 * 128 * 32 * 2048 * 2
         = 17,179,869,184 bytes
         ≈ 16 GB
```

💡 **技巧**: KV cache 与 seq_length 成正比，长序列需要更多显存！

### 实践任务

#### 任务 1.1: 计算不同配置的显存需求

编写工具计算显存:

```python
def estimate_vllm_memory(
    model_size_gb,
    num_layers,
    num_heads,
    head_dim,
    batch_size,
    seq_length,
    gpu_total_gb=80,
    other_models_gb=0,
    gpu_memory_util=0.9
):
    """
    估算 vLLM 的显存需求
    
    Returns:
        dict: {
            'model': 模型参数显存,
            'kv_cache': KV cache 显存,
            'total': 总显存,
            'fits': 是否能放下
        }
    """
    # TODO: 实现估算逻辑
    
    # 1. 模型参数
    model_memory = model_size_gb
    
    # 2. KV cache
    kv_cache_memory = (
        2 * num_layers * num_heads * head_dim 
        * batch_size * seq_length * 2 / 1024**3
    )
    
    # 3. verl 逻辑: 从剩余显存分配
    free_memory = gpu_total_gb - other_models_gb - model_memory
    allocated_kv = free_memory * gpu_memory_util
    
    # 4. 检查是否够用
    total_needed = model_memory + kv_cache_memory
    fits = (allocated_kv >= kv_cache_memory) and (total_needed + other_models_gb <= gpu_total_gb)
    
    return {
        'model_gb': model_memory,
        'kv_cache_needed_gb': kv_cache_memory,
        'kv_cache_allocated_gb': allocated_kv,
        'total_gb': total_needed + other_models_gb,
        'fits': fits
    }

# 测试
configs = [
    # (model_gb, layers, heads, head_dim, bs, seq_len, other_models_gb)
    (14, 32, 32, 128, 32, 2048, 40),  # 7B, 有其他模型
    (14, 32, 32, 128, 64, 2048, 40),  # 增大 batch
    (14, 32, 32, 128, 32, 4096, 40),  # 增大序列长度
]

for cfg in configs:
    result = estimate_vllm_memory(*cfg)
    print(f"Config {cfg}:")
    print(f"  Model: {result['model_gb']:.1f} GB")
    print(f"  KV cache needed: {result['kv_cache_needed_gb']:.1f} GB")
    print(f"  KV cache allocated: {result['kv_cache_allocated_gb']:.1f} GB")
    print(f"  Fits: {'✅' if result['fits'] else '❌'}")
```

---

## 问题2: KV Cache 的动态管理

### 提问目标

理解 vLLM 的 KV cache 管理机制，掌握 init/free cache 的时机。

### 深挖细节

#### 细节问题 2.1: 为什么需要 free_cache_engine？

**背景**: RL 训练分三个阶段，每个阶段显存需求不同。

```
Stage 1 (rollout):
   需要: model + KV cache (16 GB + 16 GB = 32 GB)

Stage 2 (data prep):
   需要: actor/critic/ref forward (40 GB)
   不需要: KV cache

Stage 3 (training):
   需要: actor/critic backward + optimizer (60 GB)
   不需要: KV cache
```

**问题**: 如果 KV cache 一直占用，Stage 2/3 会 OOM！

**解决**: 在 Stage 1 结束后释放 KV cache。

#### 细节问题 2.2: free_cache_engine 做了什么？

**代码路径**:
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py:180-200`

**关键操作**:

```python
# verl/workers/rollout/vllm_rollout/vllm_rollout.py

def free_cache_engine(self):
    """释放 KV cache 显存"""
    if self.inference_engine.cache_engine is not None:
        # 1. 释放 GPU cache
        del self.inference_engine.cache_engine.gpu_cache
        
        # 2. 释放 cache_engine 对象
        del self.inference_engine.cache_engine
        
        # 3. 强制回收
        torch.cuda.empty_cache()
        
        self.inference_engine.cache_engine = None

def init_cache_engine(self):
    """重新初始化 KV cache"""
    # 1. 计算可用显存
    free_memory = get_free_gpu_memory()
    
    # 2. 创建 cache_engine
    self.inference_engine.cache_engine = CacheEngine(
        cache_config=self.cache_config,
        model_config=self.model_config,
        parallel_config=self.parallel_config,
    )
```

**时间线**:

```
__enter__() → init_cache_engine()
    ↓
generate_sequences()  # 使用 KV cache
    ↓
__exit__() → free_cache_engine()  # 释放
    ↓
[Stage 2/3 有更多显存可用]
    ↓
[下一个 iter]
__enter__() → init_cache_engine()  # 重新初始化
```

#### 细节问题 2.3: init/free 的开销是多少？

**测量**:

```python
import time

# 测量 init_cache_engine
start = time.time()
vllm_rollout.init_cache_engine()
init_time = time.time() - start

# 测量 free_cache_engine
start = time.time()
vllm_rollout.free_cache_engine()
free_time = time.time() - start

print(f"Init: {init_time:.3f}s")
print(f"Free: {free_time:.3f}s")
```

**典型值**:
```
Init: 0.1-0.5s
Free: 0.01-0.05s
```

💡 **结论**: 开销很小，相比节省的显存（10-20GB）完全值得！

### 实践任务

#### 任务 2.1: 对比 free vs 不 free 的显存

创建 `test_kv_cache_management.py`:

```python
import torch
from verl.workers.rollout.vllm_rollout import vLLMRollout

def test_with_free():
    # 初始化 rollout
    rollout = vLLMRollout(...)
    
    # Stage 1
    with rollout.sharding_manager:
        output = rollout.generate_sequences(prompts)
    # __exit__ 自动调用 free_cache_engine
    
    # 测量 Stage 2 的可用显存
    free_mem_stage2 = torch.cuda.memory_available()
    
    return free_mem_stage2

def test_without_free():
    rollout = vLLMRollout(...)
    
    # Stage 1
    rollout.init_cache_engine()
    output = rollout.generate_sequences(prompts)
    # 不释放！
    
    # 测量 Stage 2 的可用显存
    free_mem_stage2 = torch.cuda.memory_available()
    
    return free_mem_stage2

# 对比
mem_with_free = test_with_free()
mem_without_free = test_without_free()

print(f"With free: {mem_with_free / 1024**3:.2f} GB")
print(f"Without free: {mem_without_free / 1024**3:.2f} GB")
print(f"节省: {(mem_with_free - mem_without_free) / 1024**3:.2f} GB")
```

---

## 问题3: CPU Offload 的权衡

### 提问目标

理解 CPU offload 的原理，能判断何时使用，如何配置。

### 深挖细节

#### 细节问题 3.1: CPU offload 的三种层次？

**PyTorch FSDP/DeepSpeed 的 offload 选项**:

| 层次 | Offload 内容 | 显存节省 | 性能损失 |
|------|-------------|---------|---------|
| **None** | 无 | 0% | 0% |
| **Optimizer** | 优化器状态 | ~50% | 10-20% |
| **Parameter** | 参数 + 优化器 | ~80% | 40-60% |
| **Gradient** | 梯度 + 参数 + 优化器 | ~90% | 60-80% |

**示例** (7B 模型):

```
无 offload:
   GPU: 84 GB (参数14 + 梯度14 + 优化器56)

Optimizer offload:
   GPU: 28 GB (参数14 + 梯度14)
   CPU: 56 GB (优化器)

Parameter offload:
   GPU: 14 GB (梯度)
   CPU: 70 GB (参数14 + 优化器56)
```

#### 细节问题 3.2: Offload 的性能瓶颈在哪？

**PCIe 带宽限制**:

```
PCIe 4.0 x16: ~32 GB/s (理论)
实际: ~25 GB/s

7B 模型参数传输时间:
14 GB / 25 GB/s = 0.56 s

每个 step:
   forward 前: CPU→GPU (0.56s)
   backward 后: GPU→CPU (0.56s)
   总计: 1.12s

如果 step 时间 < 1.12s → 瓶颈！
```

💡 **经验法则**: 仅在显存紧张且 step 时间 > 2s 时使用 parameter offload。

#### 细节问题 3.3: verl 如何使用 offload？

**配置示例**:

```yaml
# verl/trainer/config/actor/dp_actor.yaml

actor:
  fsdp_config:
    # Optimizer offload (推荐)
    cpu_offload: true  # 仅 offload 优化器
    param_offload: false  # 不 offload 参数
    
    # 或全部 offload (显存极度紧张时)
    cpu_offload: true
    param_offload: true
```

**代码实现**:

```python
# verl/workers/fsdp_workers.py:230

from torch.distributed.fsdp import CPUOffload

model = FSDP(
    model,
    cpu_offload=CPUOffload(offload_params=param_offload),
    # ...
)
```

### 实践任务

#### 任务 3.1: 对比 offload 配置的性能

创建 `test_offload.py`:

```python
import torch
from torch.distributed.fsdp import FSDP, CPUOffload
import time

def benchmark_offload(offload_params):
    model = create_model()  # 7B
    model = FSDP(
        model,
        cpu_offload=CPUOffload(offload_params=offload_params)
    )
    
    optimizer = torch.optim.Adam(model.parameters())
    
    # Warmup
    for _ in range(5):
        train_step(model, optimizer)
    
    # 测量
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(20):
        train_step(model, optimizer)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # 显存
    max_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    return {
        'time': elapsed / 20,
        'memory_gb': max_memory
    }

# 测试
configs = [
    ('No offload', False),
    ('Param offload', True),
]

for name, offload_params in configs:
    result = benchmark_offload(offload_params)
    print(f"{name}:")
    print(f"  Time: {result['time']:.3f}s/step")
    print(f"  Memory: {result['memory_gb']:.1f} GB")
```

---

## 问题4: remove_padding 的优化原理

### 提问目标

理解 remove_padding 如何提升效率，掌握实现方法。

### 深挖细节

#### 细节问题 4.1: Padding 的浪费在哪？

**示例数据**:

```
Batch (4 条样本):
   [I love AI <PAD> <PAD> <PAD>]  # 实际长度 3, padding 到 6
   [Hello world <PAD> <PAD> <PAD> <PAD>]  # 实际长度 2, padding 到 6
   [Deep learning is amazing !]   # 实际长度 6
   [RL <PAD> <PAD> <PAD> <PAD> <PAD>]  # 实际长度 1, padding 到 6

Shape: [4, 6]
总 tokens: 24
实际有效 tokens: 3 + 2 + 6 + 1 = 12
浪费: 50%!
```

**计算浪费**:

```
Attention (with padding):
   计算量: B × L × L × H
   = 4 × 6 × 6 × H = 144H

Attention (without padding):
   计算量: sum(L_i × L_i × H)
   = (3×3 + 2×2 + 6×6 + 1×1) × H
   = (9 + 4 + 36 + 1) × H = 50H

节省: (144 - 50) / 144 = 65%!
```

#### 细节问题 4.2: remove_padding 如何实现？

**原理**: 将 `[B, L]` 打平为 `[total_tokens]`，去除 PAD。

**转换**:

```
Before (with padding):
   input_ids: [
      [1, 2, 3, 0, 0, 0],
      [4, 5, 0, 0, 0, 0],
      [6, 7, 8, 9, 10, 11],
      [12, 0, 0, 0, 0, 0]
   ]  # Shape: [4, 6]
   
   attention_mask: [
      [1, 1, 1, 0, 0, 0],
      [1, 1, 0, 0, 0, 0],
      [1, 1, 1, 1, 1, 1],
      [1, 0, 0, 0, 0, 0]
   ]

After (remove padding):
   input_ids_unpad: [1,2,3, 4,5, 6,7,8,9,10,11, 12]  # Shape: [12]
   cu_seqlens: [0, 3, 5, 11, 12]  # 累积长度
```

**Attention 调用**:

```python
# Before (padded)
output = flash_attn_func(q, k, v, causal=True)  # [B, L, H]

# After (unpadded)
output = flash_attn_varlen_func(
    q_unpad,  # [total_tokens, H]
    k_unpad,
    v_unpad,
    cu_seqlens_q,  # [B+1]
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal=True
)  # [total_tokens, H]
```

#### 细节问题 4.3: 哪些层需要适配 unpadded？

**需要修改的层**:

| 层 | Padded 实现 | Unpadded 实现 |
|---|-------------|---------------|
| **Embedding** | `embed(input_ids)` | `embed(input_ids_unpad)` |
| **Attention** | `flash_attn_func` | `flash_attn_varlen_func` |
| **RoPE** | `apply_rope(q, k, pos_ids)` | `apply_rope(q_unpad, k_unpad, pos_ids_unpad)` |
| **MLP/LN** | 无需改动 (element-wise) | 无需改动 |
| **LM Head** | `lm_head(hidden)` | `lm_head(hidden_unpad)` |

### 代码路径

**关键文件**:
- `verl/models/transformers/modeling_llama.py` (LLaMA 的 unpad 实现)
- `verl/utils/dataset/dataset_utils.py` (数据预处理)

**关键函数**:

```python
# verl/utils/dataset/dataset_utils.py:50

def prepare_unpadded_data(input_ids, attention_mask):
    """
    转换为 unpadded 格式
    """
    # 1. 计算每个样本的实际长度
    seqlens = attention_mask.sum(dim=1)  # [B]
    
    # 2. 去除 padding
    input_ids_unpad = input_ids[attention_mask.bool()]  # [total_tokens]
    
    # 3. 计算累积长度
    cu_seqlens = torch.cat([
        torch.tensor([0], device=seqlens.device),
        seqlens.cumsum(dim=0)
    ])  # [B+1]
    
    return input_ids_unpad, cu_seqlens, seqlens.max().item()
```

### 实践任务

#### 任务 4.1: 对比 padded vs unpadded

创建 `test_remove_padding.py`:

```python
import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func
import time

def benchmark_attention(use_unpad, batch_size, max_len, hidden_size):
    # 生成数据 (模拟不同长度)
    seqlens = torch.randint(max_len // 2, max_len, (batch_size,))
    
    if not use_unpad:
        # Padded
        q = torch.randn(batch_size, max_len, 32, hidden_size, device='cuda', dtype=torch.float16)
        k = q.clone()
        v = q.clone()
        
        # Warmup
        for _ in range(10):
            output = flash_attn_func(q, k, v, causal=True)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(100):
            output = flash_attn_func(q, k, v, causal=True)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
    
    else:
        # Unpadded
        total_tokens = seqlens.sum().item()
        q_unpad = torch.randn(total_tokens, 32, hidden_size, device='cuda', dtype=torch.float16)
        k_unpad = q_unpad.clone()
        v_unpad = q_unpad.clone()
        
        cu_seqlens = torch.cat([torch.tensor([0]), seqlens.cumsum(dim=0)]).int().cuda()
        
        # Warmup
        for _ in range(10):
            output = flash_attn_varlen_func(
                q_unpad, k_unpad, v_unpad,
                cu_seqlens, cu_seqlens,
                max_len, max_len,
                causal=True
            )
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(100):
            output = flash_attn_varlen_func(
                q_unpad, k_unpad, v_unpad,
                cu_seqlens, cu_seqlens,
                max_len, max_len,
                causal=True
            )
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
    
    return elapsed / 100

# 测试
configs = [
    (8, 2048, 128),
    (16, 2048, 128),
    (32, 2048, 128),
]

for batch_size, max_len, hidden_size in configs:
    t_padded = benchmark_attention(False, batch_size, max_len, hidden_size)
    t_unpadded = benchmark_attention(True, batch_size, max_len, hidden_size)
    
    print(f"B={batch_size}, L={max_len}:")
    print(f"  Padded: {t_padded*1000:.2f} ms")
    print(f"  Unpadded: {t_unpadded*1000:.2f} ms")
    print(f"  Speedup: {t_padded/t_unpadded:.2f}x")
```

---

## 问题5: 显存碎片化与内存池

### 提问目标

理解显存碎片化的成因，掌握缓解方法。

### 深挖细节

#### 细节问题 5.1: 什么是显存碎片化？

**示意**:

```
Initial state:
   [Used: 10GB] [Free: 70GB]

After allocate & free:
   [Used: 10GB] [Free: 5GB] [Used: 20GB] [Free: 5GB] [Used: 10GB] [Free: 30GB]
   
Problem: 最大连续空间仅 30GB，但总空闲 40GB！
```

**检测方法**:

```python
import torch

allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3

fragmentation = (reserved - allocated) / reserved * 100

print(f"Allocated: {allocated:.2f} GB")
print(f"Reserved: {reserved:.2f} GB")
print(f"Fragmentation: {fragmentation:.1f}%")
```

#### 细节问题 5.2: 如何缓解碎片化？

**方法**:

1. **empty_cache()**: 释放未使用的cached块

```python
torch.cuda.empty_cache()
```

2. **内存池**: 预分配固定大小的块

```python
# vLLM 的做法
self.gpu_cache = [
    torch.empty(cache_block_size, device='cuda')
    for _ in range(num_blocks)
]
```

3. **Checkpoint**: 定期清空显存重启

```python
if step % checkpoint_freq == 0:
    save_checkpoint(model, optimizer)
    torch.cuda.empty_cache()
```

### 实践任务

#### 任务 5.1: 测量碎片化

```python
import torch

def test_fragmentation():
    # 分配和释放多次
    tensors = []
    for _ in range(100):
        t = torch.randn(100, 1000, 1000, device='cuda')
        tensors.append(t)
    
    # 随机释放一半
    import random
    for _ in range(50):
        idx = random.randint(0, len(tensors)-1)
        del tensors[idx]
    
    # 测量
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    
    print(f"Before empty_cache:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")
    
    # 清理
    torch.cuda.empty_cache()
    
    reserved_after = torch.cuda.memory_reserved() / 1024**3
    print(f"\nAfter empty_cache:")
    print(f"  Reserved: {reserved_after:.2f} GB")
    print(f"  Freed: {reserved - reserved_after:.2f} GB")
```

---

## 概念验证实验

### 实验 1: 显存占用拆解

分析一个完整训练 step 的显存组成:

```python
import torch
import torch.nn as nn

def analyze_memory_breakdown():
    torch.cuda.reset_peak_memory_stats()
    
    # 1. 模型
    model = create_model()  # 假设 7B
    mem_model = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Model: {mem_model:.2f} GB")
    
    # 2. 优化器
    optimizer = torch.optim.Adam(model.parameters())
    mem_opt = torch.cuda.max_memory_allocated() / 1024**3 - mem_model
    print(f"Optimizer: {mem_opt:.2f} GB")
    
    # 3. Forward (激活值)
    data = torch.randn(32, 2048, 4096, device='cuda')
    output = model(data)
    mem_fwd = torch.cuda.max_memory_allocated() / 1024**3 - mem_model - mem_opt
    print(f"Activations: {mem_fwd:.2f} GB")
    
    # 4. Backward (梯度)
    loss = output.mean()
    loss.backward()
    mem_bwd = torch.cuda.max_memory_allocated() / 1024**3 - mem_model - mem_opt - mem_fwd
    print(f"Gradients: {mem_bwd:.2f} GB")
    
    total = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nTotal: {total:.2f} GB")
```

---

## 源码阅读指南

### 阅读任务 1: vLLM 的 CacheEngine

**文件**: `verl/third_party/vllm/vllm_v_0_6_3/engine/cache_engine.py`

**目标**: 理解 KV cache 的分配和管理

**关键代码**:

```python
class CacheEngine:
    def __init__(self, cache_config, model_config, parallel_config):
        # 计算 cache block 数量
        self.num_blocks = self._calculate_num_blocks(
            cache_config.cache_memory_budget
        )
        
        # 预分配 cache blocks
        self.gpu_cache = self._allocate_cache_blocks()
    
    def _allocate_cache_blocks(self):
        # 每个 block: [num_layers, 2, num_heads, block_size, head_dim]
        cache_blocks = []
        for _ in range(self.num_blocks):
            block = torch.empty(
                self.num_layers, 2, self.num_heads,
                self.block_size, self.head_dim,
                device='cuda', dtype=self.dtype
            )
            cache_blocks.append(block)
        
        return cache_blocks
```

**阅读提示**:
- 关注 `block_size` 的选择
- 理解 PagedAttention 如何使用这些 blocks
- 追踪 `_calculate_num_blocks` 的逻辑

---

## 自我检测清单

### 基础理解

- [ ] 能拆解训练显存的五大组成
- [ ] 能解释 vLLM 和 verl 的 gpu_memory_utilization 差异
- [ ] 能说出 free_cache_engine 的作用
- [ ] 能计算 KV cache 的大小
- [ ] 能解释 remove_padding 的原理

### 进阶理解

- [ ] 能估算不同配置的显存需求
- [ ] 能选择合适的 offload 策略
- [ ] 能实现 unpadded attention
- [ ] 能诊断显存碎片化问题
- [ ] 能优化显存占用

### 代码能力

- [ ] 能配置 FSDP offload
- [ ] 能对比 offload 的性能
- [ ] 能实现 remove_padding
- [ ] 能测量显存占用
- [ ] 能调试 OOM 问题

---

## 下一步

完成 Level 3 后，进入 [Level 4: 并行通信模式](./level4_communication_patterns.md)，深入学习通信优化。

---

**总结**: 本 Level 详细讲解了 verl 的内存管理策略，重点是 KV cache 管理、CPU offload 和 remove_padding。掌握这些技术后，你将能够突破显存瓶颈，训练更大的模型。
