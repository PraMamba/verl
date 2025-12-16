# [Level 1] 并行策略基础

**面向对象**: 基础设施（Infrastructure）初学者  
**学习时长**: 建议 3-4 天（24-32 小时）  
**前置知识**: Level 0 - Python 分布式与 PyTorch 基础  
**最后更新**: 2025-12-15

---

## 📋 目录

1. [学习目标](#学习目标)
2. [并行策略全景图](#并行策略全景图)
3. [问题 1: DP vs TP - 切分维度的本质区别](#问题1-dp-vs-tp---切分维度的本质区别)
4. [问题 2: Zero1/2/3 的演进逻辑与通信分析](#问题2-zero123-的演进逻辑与通信分析)
5. [问题 3: SPMD 编程范式的"无形之手"](#问题3-spmd-编程范式的无形之手)
6. [问题 4: Tensor Parallel 的矩阵切分](#问题4-tensor-parallel-的矩阵切分)
7. [问题 5: Pipeline Parallel 的 Bubble 问题](#问题5-pipeline-parallel-的-bubble-问题)
8. [概念验证实验](#概念验证实验)
9. [源码阅读指南](#源码阅读指南)
10. [自我检测清单](#自我检测清单)

---

## 学习目标

完成本 Level 后，你将能够：

- [ ] ✅ **区分 DP 和 TP**: 能用矩阵乘法图示解释切分维度的差异
- [ ] ✅ **理解 Zero1/2/3**: 能计算每种模式的显存占用和通信量
- [ ] ✅ **掌握 SPMD 范式**: 能识别代码中的 rank 条件分支，理解同步点
- [ ] ✅ **实现 TP 算子**: 能手写 ColumnParallelLinear 和 RowParallelLinear
- [ ] 🔍 **分析通信开销**: 能计算不同并行度下的带宽需求
- [ ] 🔍 **选择并行策略**: 能根据模型大小和硬件配置推荐并行方案

---

## 并行策略全景图

### 核心矛盾

```
训练大模型的三大瓶颈:
1. 显存不足 → 模型并行 (TP/PP/Zero)
2. 训练太慢 → 数据并行 (DP)
3. 序列太长 → 序列并行 (SP)
```

### 策略对比表

| 策略 | 切分对象 | 通信时机 | 通信量 | 典型场景 |
|------|---------|---------|--------|---------|
| **Data Parallel (DP)** | 输入数据 | backward 后 | 梯度 (M) | 小模型多卡 |
| **Tensor Parallel (TP)** | 权重矩阵 | forward/backward 中 | 激活值 (B×S×H) | 大模型单层 |
| **Pipeline Parallel (PP)** | 模型层 | 层边界 | 边界激活 | 超大模型 |
| **Zero1** | 优化器状态 | backward 后 | 梯度 (M) | 节省显存 |
| **Zero2** | 梯度 | backward 中 | 梯度 (M) | Zero1 加强 |
| **Zero3** | 参数 | forward/backward 中 | 参数+梯度 (1.5M) | 最大节省 |

**M**: 模型参数量, **B**: batch size, **S**: 序列长度, **H**: 隐藏层维度

---

## 问题1: DP vs TP - 切分维度的本质区别

### 提问目标

深入理解 DP 和 TP 的计算模式差异，能够根据计算图判断应该使用哪种并行策略。

### 深挖细节

#### 细节问题 1.1: 为什么 DP 切分输入，TP 切分权重？

**理论分析**:

对于矩阵乘法 `Y = XW`（X: [B×S, D_in], W: [D_in, D_out], Y: [B×S, D_out]）

**Data Parallel (DP)**:
```
切分 X 的 batch 维度:
   GPU 0: Y_0 = X_0 W  (X_0: [B/2×S, D_in])
   GPU 1: Y_1 = X_1 W  (X_1: [B/2×S, D_in])

特点:
- 每个 GPU 持有完整的 W
- 无需通信即可计算（独立）
- 梯度需要 all-reduce: ∂L/∂W = X_0^T ∂L/∂Y_0 + X_1^T ∂L/∂Y_1
```

**Tensor Parallel (TP, Column 切分)**:
```
切分 W 的 column 维度:
   GPU 0: Y_0 = X W_0  (W_0: [D_in, D_out/2])
   GPU 1: Y_1 = X W_1  (W_1: [D_in, D_out/2])
   
输出聚合:
   Y = [Y_0, Y_1]  (all-gather on column)

特点:
- 每个 GPU 仅持有 W 的一部分
- 需要通信才能得到完整输出
- 梯度无需 all-reduce（各 GPU 独立）
```

**可视化**:

```
Data Parallel (切输入):
   X_0 [████]         [████████] W
   X_1 [████]    ×    [████████] W
   
   = Y_0 [████████]
     Y_1 [████████]

Tensor Parallel (切权重):
   X [████████]       [████    ] W_0
   X [████████]   ×   [    ████] W_1
   
   = Y_0 [████    ]
     Y_1 [    ████]  → Concat → Y [████████]
```

**代码示例**:

```python
# Data Parallel
class DPLinear(nn.Module):
    def forward(self, x_local):  # x_local: [B/N, D_in]
        y_local = self.weight @ x_local.T  # [D_out, B/N]
        return y_local  # 各 GPU 输出不同

# Tensor Parallel (Column)
class ColumnTPLinear(nn.Module):
    def forward(self, x_full):  # x_full: [B, D_in] (各 GPU 相同)
        y_local = self.weight_column @ x_full.T  # [D_out/N, B]
        # all-gather 得到完整 Y
        y_full = all_gather(y_local, dim=0)  # [D_out, B]
        return y_full  # 各 GPU 输出相同
```

#### 细节问题 1.2: DP 和 TP 何时发生通信？

**DP 通信时机**:
```
forward:  无通信（各 GPU 独立计算）
backward: 无通信（各 GPU 独立计算梯度）
update:   all-reduce 梯度
```

**TP 通信时机**:
```
forward:  all-gather/reduce-scatter 激活值
backward: all-gather/reduce-scatter 梯度
update:   无通信（梯度已分片）
```

**通信量对比** (模型参数 M, batch size B, 序列长度 S, 隐藏层 H):

| 阶段 | DP 通信量 | TP 通信量 |
|------|-----------|-----------|
| Forward | 0 | B×S×H (激活值) |
| Backward | 0 | B×S×H (梯度) |
| Update | M (参数梯度) | 0 |
| **总计** | M | 2×B×S×H |

💡 **关键洞察**: 
- DP 通信量与数据量**无关**，仅与模型大小有关
- TP 通信量与数据量**相关**，batch size 越大通信越多

#### 细节问题 1.3: 为什么 TP 需要两种切分方式（Column 和 Row）？

**答案**: MLP 的两层 Linear 需要配合！

```
MLP 结构:
   h = Linear1(x)  # [D, 4D]
   h = ReLU(h)
   y = Linear2(h)  # [4D, D]
```

**错误的 TP 切分** (都用 Column):
```
GPU 0: h_0 = Linear1_col0(x) → [B, 2D]
GPU 1: h_1 = Linear1_col1(x) → [B, 2D]

需要 all-gather → h = [h_0, h_1]  # [B, 4D]

GPU 0: y_0 = Linear2_col0(h) → [B, D/2]
GPU 1: y_1 = Linear2_col1(h) → [B, D/2]

又需要 all-gather → y = [y_0, y_1]  # [B, D]

总通信: 2 次 all-gather ❌
```

**正确的 TP 切分** (Column + Row):
```
GPU 0: h_0 = Linear1_col0(x) → [B, 2D]  (x 相同)
GPU 1: h_1 = Linear1_col1(x) → [B, 2D]

GPU 0: y_0 = Linear2_row0(h_0) → [B, D]  (h_0 不同)
GPU 1: y_1 = Linear2_row1(h_1) → [B, D]

reduce-scatter → y = y_0 + y_1  # [B, D]

总通信: 1 次 reduce-scatter ✅
```

**图示**:

```
Column-Row 组合:
   
   x [████] (相同)
      ↓
   Linear1 (Column TP)
   [col0 | col1]
      ↓
   h_0 [██]  h_1 [██] (各持一半)
      ↓
   Linear2 (Row TP)
   [row0]    [row1]
      ↓
   y_0 [████]  y_1 [████]
      ↓
   reduce-scatter
      ↓
   y [████] (相同)
```

### 代码路径

**verl 中的 TP 实现**:

- **FSDP TP**: 不直接支持（通过 vLLM rollout 的 TP）
- **Megatron TP**: `verl/models/mcore/` (使用 Megatron-LM 的 TP 实现)
- **vLLM TP**: `verl/third_party/vllm/` (用于 rollout 阶段)

**关键函数**:
```python
# verl/third_party/vllm/vllm_v_0_6_3/model_executor/layers/linear.py

class ColumnParallelLinear:
    def forward(self, input_):
        # 输入广播（或已经相同）
        output_parallel = F.linear(input_, self.weight)
        # all-gather 沿 column 维度
        output = all_gather(output_parallel)
        return output

class RowParallelLinear:
    def forward(self, input_):
        # 输入已分片
        output_parallel = F.linear(input_, self.weight)
        # reduce-scatter 聚合
        output = reduce_scatter(output_parallel)
        return output
```

### 实践任务

#### 任务 1.1: 手写 TP Linear 层

完成 `test_tp.py`:

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # TODO: 初始化 weight_column (仅持有 1/N 的 columns)
        self.weight_column = nn.Parameter(
            torch.randn(out_features // self.world_size, in_features)
        )
    
    def forward(self, x):
        # TODO: 
        # 1. 本地计算 y_local
        # 2. all-gather 得到完整 y
        pass

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # TODO: 初始化 weight_row
        pass
    
    def forward(self, x):
        # TODO:
        # 1. 本地计算 y_local
        # 2. reduce-scatter 聚合
        pass

# 测试
def test_tp_mlp():
    # 初始化
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    
    # 创建 MLP
    linear1 = ColumnParallelLinear(128, 512).cuda()
    linear2 = RowParallelLinear(512, 128).cuda()
    
    # 输入
    x = torch.randn(32, 128, device='cuda')
    dist.broadcast(x, src=0)  # 确保输入相同
    
    # Forward
    h = linear1(x)
    y = linear2(h)
    
    # 验证：y 应该在所有 rank 上相同
    y_list = [torch.zeros_like(y) for _ in range(dist.get_world_size())]
    dist.all_gather(y_list, y)
    
    for i, y_rank in enumerate(y_list):
        torch.testing.assert_close(y, y_rank, rtol=1e-3, atol=1e-3)
    
    print(f"[Rank {rank}] TP MLP 测试通过！")
```

运行:
```bash
torchrun --nproc_per_node=2 test_tp.py
```

---

## 问题2: Zero1/2/3 的演进逻辑与通信分析

### 提问目标

理解 DeepSpeed Zero 如何逐步优化显存，能够计算每种模式的显存占用和通信开销。

### 深挖细节

#### 细节问题 2.1: 训练时显存都去哪了？

**显存占用分解** (以 Adam 优化器为例):

```
模型参数 (FP16):     M × 2 bytes
梯度 (FP16):         M × 2 bytes
优化器状态 (FP32):
   - 参数副本:       M × 4 bytes
   - 一阶动量:       M × 4 bytes
   - 二阶动量:       M × 4 bytes
中间激活值:          B × S × H × L × 2 bytes

总计:
   Model States:     M × 16 bytes
   Activations:      B × S × H × L × 2 bytes
```

**示例** (7B 模型, FP16, Adam):
```
M = 7B
Model States = 7B × 16 = 112 GB
Activations (B=8, S=2048, H=4096, L=32) ≈ 32 GB
总计 ≈ 144 GB

单卡 A100 (80GB): 放不下 ❌
```

#### 细节问题 2.2: Zero1/2/3 分别优化了什么？

**Zero1: 切分优化器状态**

```
Baseline DP:
   每个 GPU:
      参数:       M × 2
      梯度:       M × 2
      优化器:     M × 12  (3 个状态 × FP32)
   总计: M × 16

Zero1:
   每个 GPU:
      参数:       M × 2
      梯度:       M × 2
      优化器:     M × 12 / N  (切分！)
   总计: M × (4 + 12/N)

N=8: M × (4 + 1.5) = M × 5.5  (节省 65% !)
```

**通信变化**: 无（梯度仍然 all-reduce）

**Zero2: 再切分梯度**

```
Zero2:
   每个 GPU:
      参数:       M × 2
      梯度:       M × 2 / N  (切分！)
      优化器:     M × 12 / N
   总计: M × (2 + 14/N)

N=8: M × (2 + 1.75) = M × 3.75  (节省 77% !)
```

**通信变化**: 梯度改为 reduce-scatter（通信量不变）

**Zero3: 再切分参数**

```
Zero3:
   每个 GPU:
      参数:       M × 2 / N  (切分！)
      梯度:       M × 2 / N
      优化器:     M × 12 / N
   总计: M × 16 / N

N=8: M × 2  (节省 87.5% !)
```

**通信变化**: 
- Forward 前: all-gather 参数
- Backward 前: all-gather 参数
- 通信量增加 50%！

#### 细节问题 2.3: 为什么 Zero3 更慢？

**理论分析**:

```
通信量对比:

Baseline DP:
   all-reduce 梯度: M × 2 bytes

Zero3:
   all-gather 参数 (forward): M × 2 bytes
   all-gather 参数 (backward): M × 2 bytes
   reduce-scatter 梯度: M × 2 bytes
   总计: M × 6 bytes  (3 倍！)
```

**实际影响**:

| 模型大小 | 通信时间占比 (Zero1) | 通信时间占比 (Zero3) |
|---------|---------------------|---------------------|
| 1B | 5% | 15% |
| 7B | 15% | 40% |
| 70B | 40% | 70% |

💡 **权衡**: Zero3 牺牲速度换显存！

**优化方法**:
1. **CPU Offload**: 把优化器状态放 CPU
2. **Gradient Checkpointing**: 减少激活值显存
3. **Parameter Offload**: 仅在计算时加载参数

### 代码路径

**verl 中的 FSDP (类似 Zero)**:

- **配置**: `verl/trainer/config/actor/dp_actor.yaml`
- **FSDP 初始化**: `verl/workers/fsdp_workers.py:200-250`

**关键配置**:

```yaml
# verl/trainer/config/actor/dp_actor.yaml

fsdp_config:
  # 类似 Zero3
  sharding_strategy: FULL_SHARD  
  
  # CPU Offload (优化器状态)
  cpu_offload: false
  
  # 参数 offload
  param_offload: false
```

**代码示例**:

```python
# verl/workers/fsdp_workers.py:220

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # Zero3
    cpu_offload=CPUOffload(offload_params=False),
    use_orig_params=True,  # 保留原始参数名
)
```

### 实践任务

#### 任务 2.1: 对比 Zero1/2/3 的显存和速度

创建 `test_zero.py`:

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
import time

def train_step(model, data, target):
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    return loss.item()

def benchmark_zero(strategy_name, sharding_strategy):
    # 初始化
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    
    # 模型 (1B 参数)
    model = nn.Sequential(
        nn.Linear(4096, 16384),
        nn.ReLU(),
        nn.Linear(16384, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1000)
    ).cuda()
    
    model = FSDP(model, sharding_strategy=sharding_strategy)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 数据
    data = torch.randn(32, 4096, device='cuda')
    target = torch.randint(0, 1000, (32,), device='cuda')
    
    # Warmup
    for _ in range(10):
        train_step(model, data, target)
        optimizer.step()
        optimizer.zero_grad()
    
    # 测速
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        train_step(model, data, target)
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # 显存
    max_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    if rank == 0:
        print(f"{strategy_name}:")
        print(f"  时间: {elapsed:.2f} s")
        print(f"  显存: {max_memory:.2f} GB")
        print(f"  吞吐: {100/elapsed:.2f} step/s")
    
    dist.destroy_process_group()

# 测试
if __name__ == "__main__":
    import sys
    strategy = sys.argv[1] if len(sys.argv) > 1 else "FULL_SHARD"
    
    strategies = {
        "NO_SHARD": ShardingStrategy.NO_SHARD,      # Baseline DP
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,  # Zero2
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,  # Zero3
    }
    
    benchmark_zero(strategy, strategies[strategy])
```

运行:
```bash
# 4 卡
torchrun --nproc_per_node=4 test_zero.py NO_SHARD
torchrun --nproc_per_node=4 test_zero.py SHARD_GRAD_OP
torchrun --nproc_per_node=4 test_zero.py FULL_SHARD
```

**记录结果**并分析：
- 显存节省符合理论吗？
- 速度下降多少？
- 什么情况下应该用 Zero3？

---

## 问题3: SPMD 编程范式的"无形之手"

### 提问目标

理解 SPMD (Single Program Multiple Data) 的编程模式，能够编写无需中心调度器的分布式代码。

### 深挖细节

#### 细节问题 3.1: SPMD 与中心化调度的区别？

**中心化调度 (Master-Worker)**:

```python
# Master 进程
if rank == 0:
    data_chunks = split_data(data, world_size)
    for i in range(1, world_size):
        send(data_chunks[i], dst=i)  # 发送数据给 Worker
    
    results = []
    for i in range(1, world_size):
        results.append(recv(src=i))  # 收集结果

# Worker 进程
else:
    my_data = recv(src=0)  # 接收数据
    result = process(my_data)
    send(result, dst=0)  # 发送结果
```

**SPMD (无中心)**:

```python
# 所有进程运行相同代码
rank = dist.get_rank()
world_size = dist.get_world_size()

# 各自计算数据分片
chunk_size = len(data) // world_size
my_data = data[rank * chunk_size : (rank + 1) * chunk_size]

# 各自处理
my_result = process(my_data)

# 聚合（无需 Master）
all_results = [torch.zeros_like(my_result) for _ in range(world_size)]
dist.all_gather(all_results, my_result)
```

💡 **关键区别**:
- 中心化: Master 主动分配任务
- SPMD: Worker 自己决定做什么（根据 rank）

#### 细节问题 3.2: SPMD 如何避免死锁？

**常见死锁场景**:

```python
# ❌ 错误：条件分支导致不同步
if rank == 0:
    dist.broadcast(tensor, src=0)  # Rank 0 调用
# Rank 1-3 没有调用 → 死锁！
```

**正确写法**:

```python
# ✅ 正确：所有 Rank 都调用
dist.broadcast(tensor, src=0)  # 全部调用，src 参数指定源
```

**更复杂的例子**:

```python
# ❌ 错误：随机数导致分支不一致
if random.random() > 0.5:  # 各 Rank 的随机数不同！
    dist.all_reduce(tensor)
```

**正确写法**:

```python
# ✅ 正确：同步随机种子
torch.manual_seed(42)  # 所有 Rank 相同
if random.random() > 0.5:  # 现在一致了
    dist.all_reduce(tensor)
```

#### 细节问题 3.3: SPMD 如何实现 DP 参数对齐？

**魔法在于**: 初始化后从不显式同步参数！

```python
# 初始化时
torch.manual_seed(42)  # 所有 Rank 相同种子
model = MyModel()  # 参数初始化完全一致

# 训练时
for data, target in dataloader:
    # 各 Rank 处理不同数据
    my_data = data[rank::world_size]  # 切片
    
    output = model(my_data)  # 参数相同 → 输出确定性
    loss = criterion(output, target)
    loss.backward()
    
    # 梯度自动 all-reduce (DDP 的 hook)
    # 因为梯度相同 → optimizer.step() 后参数仍相同！
    optimizer.step()
```

💡 **不变量**: 只要初始参数相同 + 梯度相同 → 参数永远相同！

### 代码路径

**verl 中的 SPMD 示例**:

- **TP 示例**: `verl/third_party/vllm/vllm_v_0_6_3/spmd_gpu_executor.py`
- **FSDP Worker**: `verl/workers/fsdp_workers.py`

**关键代码**:

```python
# verl/workers/fsdp_workers.py:150

def init_torch_distributed():
    if not torch.distributed.is_initialized():
        # 所有 Rank 运行相同代码
        torch.distributed.init_process_group(backend="nccl")
    
    # 各 Rank 根据环境变量决定行为
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
```

### 实践任务

#### 任务 3.1: 调试 SPMD 死锁

故意制造死锁并修复：

```python
import torch
import torch.distributed as dist

def buggy_spmd():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    
    tensor = torch.tensor([rank], device='cuda')
    
    # Bug 1: 条件分支不一致
    if rank % 2 == 0:
        dist.all_reduce(tensor)  # Rank 0, 2 调用
    # Rank 1, 3 没调用 → 死锁！
    
    print(f"[Rank {rank}] Result: {tensor.item()}")

# 修复方法：？
```

**思考**:
1. 如何检测这种死锁？
2. 如何使用 `timeout` 避免无限等待？
3. 如何用 `NCCL_DEBUG=INFO` 诊断？

---

## 问题4: Tensor Parallel 的矩阵切分

### 提问目标

掌握 TP 的数学原理和工程实现，能够为任意 Linear 层实现 TP 版本。

### 深挖细节

#### 细节问题 4.1: Attention 的 TP 如何切分？

**Multi-Head Attention 的 TP**:

```
QKV 投影 (Column TP):
   Q = XW_Q  (W_Q: [D, D])
   K = XW_K
   V = XW_V

切分 attention heads:
   GPU 0: heads 0-3
   GPU 1: heads 4-7

每个 GPU 独立计算:
   Attn_local = Softmax(Q_local K_local^T / √d) V_local

输出投影 (Row TP):
   O = Attn W_O  (W_O: [D, D])
   reduce-scatter 聚合
```

**代码示例** (简化):

```python
class TPAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        self.n_heads_per_partition = n_heads // dist.get_world_size()
        
        # Column TP: 切分 heads
        self.qkv = ColumnParallelLinear(d_model, 3 * d_model)
        
        # Row TP: 聚合输出
        self.out_proj = RowParallelLinear(d_model, d_model)
    
    def forward(self, x):
        # x: [B, S, D] (所有 GPU 相同)
        qkv = self.qkv(x)  # [B, S, 3*D_local]
        
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 各 GPU 计算自己的 heads
        attn = self.scaled_dot_product_attention(q, k, v)
        
        # Row TP: reduce-scatter
        out = self.out_proj(attn)
        
        return out  # [B, S, D] (所有 GPU 相同)
```

#### 细节问题 4.2: TP 的通信量如何计算？

**前向传播**:

```
Column TP (QKV 投影):
   输入: X [B, S, D] (broadcast, 或已经相同)
   输出: QKV [B, S, 3D_local]
   通信: all-gather → 3×B×S×D bytes

Row TP (输出投影):
   输入: Attn [B, S, D_local] (各 GPU 不同)
   输出: O [B, S, D]
   通信: reduce-scatter → B×S×D bytes

单层总通信: 4×B×S×D bytes
```

**反向传播**: 类似，总计 `8×B×S×D bytes`

**示例** (B=8, S=2048, D=4096, FP16):
```
单层通信 = 8 × 8 × 2048 × 4096 × 2 = 1 GB
32 层 Transformer = 32 GB
```

💡 **优化方法**:
- 减少 TP 度（如 TP=4 → TP=2）
- 使用 NVLINK（带宽更大）
- 序列并行（分摊通信）

### 实践任务

#### 任务 4.1: 实现 TP Attention

完成骨架代码：

```python
class TPMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.n_heads_local = n_heads // self.world_size
        self.d_head = d_model // n_heads
        
        # TODO: 初始化 QKV 和 output 投影
        self.qkv_proj = None  # ColumnParallelLinear
        self.out_proj = None  # RowParallelLinear
    
    def forward(self, x):
        B, S, D = x.shape
        
        # TODO: 
        # 1. QKV 投影
        # 2. 重塑为 heads 形状
        # 3. 计算 attention
        # 4. 输出投影
        pass
```

---

## 问题5: Pipeline Parallel 的 Bubble 问题

### 提问目标

理解 PP 的 Bubble 开销，能够计算不同 micro-batch 策略的效率。

### 深挖细节

#### 细节问题 5.1: 什么是 Pipeline Bubble？

**示意图** (4 层, 4 micro-batches):

```
时间 →
GPU 0 (Layer 0-7):   [F0][F1][F2][F3]              [B3][B2][B1][B0]
GPU 1 (Layer 8-15):      [F0][F1][F2][F3]      [B3][B2][B1][B0]
GPU 2 (Layer 16-23):         [F0][F1][F2][F3][B3][B2][B1][B0]
GPU 3 (Layer 24-31):             [F0][F1][F2][F3][B3][B2][B1][B0]

Bubble (灰色) = 空闲时间
```

**Bubble 比例**:
```
Bubble Ratio = (P - 1) / (P - 1 + M)

P: Pipeline stages (4)
M: Micro-batches (4)

Bubble = (4-1) / (4-1+4) = 3/7 ≈ 43%
```

#### 细节问题 5.2: 如何减少 Bubble？

**方法 1: 增加 Micro-batch 数量**

```
M = 8 (翻倍):
Bubble = (4-1) / (4-1+8) = 3/11 ≈ 27%  (减半!)
```

⚠️ **代价**: 每个 micro-batch 更小 → 计算效率下降

**方法 2: 使用 1F1B Schedule**

```
传统 (GPipe):
   Forward 全部 → Backward 全部

1F1B (Interleaved):
   F0 → F1 → B0 → F2 → B1 → ...

Bubble 降低到: (P-1) / M
```

**方法 3: Virtual Pipeline Parallel**

```
将模型切分为 2P 段，交替放置:
   GPU 0: [Stage 0] [Stage 4]
   GPU 1: [Stage 1] [Stage 5]
   GPU 2: [Stage 2] [Stage 6]
   GPU 3: [Stage 3] [Stage 7]

Bubble 进一步减少！
```

### 代码路径

**verl 中的 PP 实现**:

- **Megatron PP**: `verl/workers/megatron_workers.py` (使用 Megatron-LM)
- **AllGatherPPModel**: `verl/workers/sharding_manager/megatron_vllm.py:50`

**关键概念**:

```python
# verl/workers/sharding_manager/megatron_vllm.py:65

class AllGatherPPModel:
    def __init__(self):
        # 每个 PP rank 构建所有 PP stage 的模型
        self.models_for_all_pp = []
        for pp_rank in range(pp_world_size):
            model = build_model_for_rank(pp_rank)
            self.models_for_all_pp.append(model)
        
        # 仅当前 rank 的模型用于训练
        self.model = self.models_for_all_pp[pp_rank]
```

### 实践任务

#### 任务 5.1: 计算 PP Bubble

给定配置，计算 Bubble：

```python
def calculate_bubble(P, M):
    """
    P: Pipeline stages
    M: Micro-batches
    """
    bubble_ratio = (P - 1) / (P - 1 + M)
    return bubble_ratio

# 测试
configs = [
    (4, 4),
    (4, 8),
    (8, 16),
    (16, 32),
]

for P, M in configs:
    bubble = calculate_bubble(P, M)
    print(f"P={P}, M={M}: Bubble={bubble:.2%}")
```

**思考**:
- PP=16 时，需要多少 micro-batches 才能让 Bubble < 20%？
- Micro-batch 太多有什么副作用？

---

## 概念验证实验

### 实验 1: TP 通信量测量

测量不同 TP 度下的通信开销：

```python
import torch
import torch.distributed as dist
import time

def measure_tp_comm(tp_size):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    
    # 模拟 TP forward 的 all-gather
    B, S, H = 8, 2048, 4096
    tensor = torch.randn(B, S, H // tp_size, device='cuda')
    
    # Warmup
    for _ in range(10):
        output_tensors = [torch.zeros_like(tensor) for _ in range(tp_size)]
        dist.all_gather(output_tensors, tensor)
    
    # 测量
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        output_tensors = [torch.zeros_like(tensor) for _ in range(tp_size)]
        dist.all_gather(output_tensors, tensor)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # 计算带宽
    data_size = B * S * H * 2 * 100  # FP16
    bandwidth = data_size / elapsed / 1e9  # GB/s
    
    if rank == 0:
        print(f"TP={tp_size}: {bandwidth:.2f} GB/s")
    
    dist.destroy_process_group()

# 运行
for tp in [2, 4, 8]:
    measure_tp_comm(tp)
```

---

## 源码阅读指南

### 阅读任务 1: FSDP 的 all-gather 时机

**文件**: `verl/workers/sharding_manager/fsdp_vllm.py:80-120`

**关键代码**:

```python
def __enter__(self):
    # 获取完整参数
    params = self.module.state_dict()
    
    # 对于 FSDP，需要 unshard
    for name, param in params.items():
        if hasattr(param, '_full_param_padded'):
            # 这里触发 all-gather!
            params[name] = param._full_param_padded
```

**阅读提示**:
- 关注 `_full_param_padded` 的实现
- 理解为何在 context manager 中 unshard

### 阅读任务 2: Megatron TP 的实现

**文件**: (Megatron-LM 外部库)

**建议阅读**:
- `megatron/core/tensor_parallel/layers.py` (ColumnParallelLinear)
- `megatron/core/tensor_parallel/mappings.py` (通信原语)

---

## 自我检测清单

### 核心理解

- [ ] 能用矩阵乘法图解释 DP vs TP 的切分差异
- [ ] 能计算 Zero1/2/3 的显存占用
- [ ] 能解释 Zero3 为何比 Zero2 慢
- [ ] 能识别 SPMD 代码中的死锁风险
- [ ] 能手写 ColumnParallelLinear

### 进阶理解

- [ ] 能设计 Attention 的 TP 切分方案
- [ ] 能计算 TP 的通信量
- [ ] 能解释 PP 的 Bubble 来源
- [ ] 能对比不同并行策略的适用场景

### 代码能力

- [ ] 能运行 FSDP 训练脚本
- [ ] 能对比不同 sharding strategy 的性能
- [ ] 能实现简单的 TP 算子
- [ ] 能调试 SPMD 死锁问题

---

## 下一步

完成 Level 1 后，进入 [Level 2: 数据流转机制](./level2_data_flow_mechanisms.md)，学习 RL 三阶段的数据流动规律。
