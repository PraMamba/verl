# [Level 0] Python 分布式与 PyTorch 基础

**面向对象**: 基础设施（Infrastructure）初学者
**学习时长**: 建议 2-3 天（16-24 小时）
**前置知识**: Python 基础、PyTorch 基础
**最后更新**: 2025-12-15

---

## 📋 目录

1. [学习目标](#学习目标)
2. [为什么需要分布式训练](#为什么需要分布式训练)
3. [核心概念：进程、进程组、后端](#核心概念)
4. [问题 1: torch.distributed 的初始化流程](#问题1-torchdistributed-的初始化流程)
5. [问题 2: NCCL 通信原语](#问题2-nccl-通信原语)
6. [问题 3: DDP 的工作原理](#问题3-ddp-的工作原理)
7. [概念验证实验](#概念验证实验)
8. [源码阅读指南](#源码阅读指南)
9. [自我检测清单](#自我检测清单)
10. [进阶挑战](#进阶挑战)

---

## 学习目标

完成本 Level 后，你将能够：

- [ ] ✅ **理解分布式训练的必要性**：能解释为什么需要多卡训练，单卡的瓶颈在哪
- [ ] ✅ **掌握 torch.distributed API**：能独立编写 2-GPU 的分布式训练脚本
- [ ] ✅ **理解通信原语**：能区分 all-reduce、all-gather、broadcast 的应用场景
- [ ] ✅ **理解 DDP 机制**：能画出 DDP 的 forward/backward/update 流程图
- [ ] 🔍 **理解环境变量**：能解释 RANK、WORLD_SIZE、MASTER_ADDR 的作用
- [ ] 🔍 **理解进程组**：能创建自定义进程组并进行局部通信

---

## 为什么需要分布式训练

### 场景 1: 模型太大，单卡放不下

**案例**: 训练一个 70B 参数的 LLaMA 模型

```
模型参数占用 (FP16):
70B * 2 bytes = 140 GB

优化器状态 (Adam, FP32):
70B * 12 bytes = 840 GB

总计: ~980 GB
```

**单卡 A100 (80GB)**: 无法容纳 ❌  
**8 卡 A100 (640GB)**: 可以通过模型并行 ✅

### 场景 2: 数据太多，训练太慢

**案例**: 在 1M 样本上训练 1 epoch

```
单卡 A100:
- Batch size: 8
- 吞吐量: 100 samples/s
- 时间: 1M / 100 = 10,000 s = 2.78 h

8 卡 A100 (理想):
- 总 batch size: 64
- 吞吐量: 800 samples/s
- 时间: 1M / 800 = 1,250 s = 0.35 h
```

**加速比**: ~8x (接近线性加速)

### 并行策略对比

| 并行方式 | 解决问题 | 通信开销 | 实现难度 |
|---------|---------|---------|---------|
| **Data Parallel (DP)** | 加速训练 | 低（仅梯度） | 简单 ⭐ |
| **Tensor Parallel (TP)** | 大模型显存 | 高（激活值） | 复杂 ⭐⭐⭐ |
| **Pipeline Parallel (PP)** | 大模型显存 | 中（边界激活） | 中等 ⭐⭐ |
| **ZeRO (DP 增强)** | 大模型显存 | 低-中 | 简单 ⭐ |

💡 **本 Level 专注于 Data Parallel**，为后续学习 TP/PP 打基础。

---

## 核心概念

### 1. 进程 (Process)

在分布式训练中，**每块 GPU 对应一个独立的 Python 进程**。

```
机器 1 (4× GPU):
├─ GPU 0 → Process 0 (Rank 0)
├─ GPU 1 → Process 1 (Rank 1)
├─ GPU 2 → Process 2 (Rank 2)
└─ GPU 3 → Process 3 (Rank 3)

机器 2 (4× GPU):
├─ GPU 0 → Process 4 (Rank 4)
├─ GPU 1 → Process 5 (Rank 5)
├─ GPU 2 → Process 6 (Rank 6)
└─ GPU 3 → Process 7 (Rank 7)
```

**关键变量**:
- `RANK`: 全局进程编号 (0 ~ WORLD_SIZE-1)
- `LOCAL_RANK`: 本机 GPU 编号 (0 ~ N_GPUS_PER_NODE-1)
- `WORLD_SIZE`: 总进程数 (8)

### 2. 进程组 (Process Group)

进程组定义了**哪些进程可以互相通信**。

```python
import torch.distributed as dist

# 全局进程组 (默认)
dist.init_process_group(backend="nccl")

# 自定义子组
dp_group = dist.new_group([0, 1, 2, 3])  # 数据并行组
tp_group = dist.new_group([0, 4])        # 张量并行组
```

### 3. 后端 (Backend)

后端决定了底层通信库。

| 后端 | 设备 | 性能 | 使用场景 |
|------|------|------|---------|
| **nccl** | GPU | 极快 ⚡ | GPU 训练（推荐） |
| **gloo** | CPU/GPU | 中等 | CPU 训练或调试 |
| **mpi** | CPU/GPU | 快 | HPC 环境 |

**verl 使用**: `nccl` (GPU 专用，性能最佳)

---

## 问题1: torch.distributed 的初始化流程

### 提问目标

掌握分布式训练的启动流程，理解环境变量的作用，能够排查初始化失败问题。

### 深挖细节

#### 细节问题 1.1: 为什么需要 `MASTER_ADDR` 和 `MASTER_PORT`？

**答案**: 

分布式训练中，所有进程需要一个**集合点 (Rendezvous)** 来交换信息（如IP地址、端口）。

```
初始化流程:
1. Rank 0 (Master) 在 MASTER_ADDR:MASTER_PORT 启动 TCP 服务器
2. 其他 Rank 连接到 Master
3. Master 收集所有 Rank 的网络信息
4. Master 广播完整的 Rank→IP 映射表
5. 所有 Rank 建立点对点连接（NCCL）
```

**实验验证**:

```python
import socket
import os

# 获取本机 IP
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip

# 找一个可用端口
def get_free_port():
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]

print(f"MASTER_ADDR={get_ip()}")
print(f"MASTER_PORT={get_free_port()}")
```

#### 细节问题 1.2: 为什么 `init_process_group()` 会阻塞？

**答案**:

`init_process_group()` 是一个**同步操作**（barrier），确保所有进程都完成初始化后才继续。

```python
import torch.distributed as dist
import time

rank = int(os.environ['RANK'])

print(f"[Rank {rank}] 开始初始化...")

if rank == 0:
    time.sleep(5)  # Rank 0 延迟 5 秒

dist.init_process_group(backend="nccl")  # 所有进程都会在这里等待

print(f"[Rank {rank}] 初始化完成！")
```

**输出**（所有 Rank 同时打印）:
```
[Rank 0] 初始化完成！
[Rank 1] 初始化完成！
[Rank 2] 初始化完成！
[Rank 3] 初始化完成！
```

⚠️ **常见错误**: 如果 Rank 0 卡住，其他 Rank 会无限等待（hang）！

#### 细节问题 1.3: `torch.cuda.set_device()` 的作用是什么？

**答案**:

将当前进程绑定到特定 GPU，确保后续操作在正确的设备上执行。

```python
import torch
import os

# ❌ 错误写法（所有进程都用 GPU 0）
device = torch.device("cuda:0")

# ✅ 正确写法
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# 或者更简洁
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
device = torch.device("cuda")  # 自动使用当前设备
```

### 代码路径

在 verl 中，初始化逻辑位于：

- **文件**: `verl/workers/fsdp_workers.py:150-180`
- **关键代码**:
```python
def init_torch_distributed():
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
```

### 实践任务

#### 任务 1.1: 手写初始化脚本

创建 `test_init.py`:

```python
import torch
import torch.distributed as dist
import os

def main():
    # 1. 读取环境变量
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # 2. 初始化分布式环境
    dist.init_process_group(backend="nccl")
    
    # 3. 绑定 GPU
    torch.cuda.set_device(local_rank)
    
    # 4. 验证
    print(f"[Rank {rank}/{world_size}] 在 GPU {local_rank} 上运行")
    print(f"[Rank {rank}] Device: {torch.cuda.current_device()}")
    
    # 5. 清理
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

运行:
```bash
# 单机 4 卡
torchrun --nproc_per_node=4 test_init.py

# 2 机 8 卡 (在 node0 上运行)
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    test_init.py

# 在 node1 上运行
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    test_init.py
```

---

## 问题2: NCCL 通信原语

### 提问目标

理解分布式训练中的核心通信操作，能够根据场景选择合适的通信原语。

### 深挖细节

#### 细节问题 2.1: all-reduce 和 reduce 有什么区别？

**区别**:

| 操作 | 输入 | 输出 | 使用场景 |
|------|------|------|---------|
| **reduce** | 各 Rank 不同 | 仅 Rank 0 有结果 | 收集指标（如 loss） |
| **all-reduce** | 各 Rank 不同 | 所有 Rank 有相同结果 | 梯度同步 |

**图示**:

```
all-reduce (SUM):
   Rank 0: [1]      Rank 0: [10]
   Rank 1: [2]  =>  Rank 1: [10]
   Rank 2: [3]      Rank 2: [10]
   Rank 3: [4]      Rank 3: [10]

reduce (SUM):
   Rank 0: [1]      Rank 0: [10]
   Rank 1: [2]  =>  Rank 1: [2]  (未变)
   Rank 2: [3]      Rank 2: [3]  (未变)
   Rank 3: [4]      Rank 3: [4]  (未变)
```

**代码示例**:

```python
import torch
import torch.distributed as dist

rank = dist.get_rank()
tensor = torch.tensor([rank + 1], device='cuda')

# all-reduce
tensor_all = tensor.clone()
dist.all_reduce(tensor_all, op=dist.ReduceOp.SUM)
print(f"[Rank {rank}] all_reduce: {tensor_all.item()}")  # 都是 10

# reduce (仅 Rank 0 有结果)
tensor_reduce = tensor.clone()
dist.reduce(tensor_reduce, dst=0, op=dist.ReduceOp.SUM)
if rank == 0:
    print(f"[Rank {rank}] reduce: {tensor_reduce.item()}")  # 10
```

#### 细节问题 2.2: all-gather 的输出形状是怎样的？

**答案**:

all-gather 在**第 0 维**拼接所有 Rank 的 tensor。

```python
import torch
import torch.distributed as dist

rank = dist.get_rank()
world_size = dist.get_world_size()

# 输入: 各 Rank 的 tensor 形状相同
input_tensor = torch.tensor([[rank, rank * 2]], device='cuda')  # [1, 2]

# 输出: [world_size, ...]
output_tensors = [torch.zeros_like(input_tensor) for _ in range(world_size)]
dist.all_gather(output_tensors, input_tensor)

print(f"[Rank {rank}] all_gather output:")
for i, t in enumerate(output_tensors):
    print(f"  from Rank {i}: {t.tolist()}")
```

**输出**（每个 Rank 都打印）:
```
[Rank 0] all_gather output:
  from Rank 0: [[0, 0]]
  from Rank 1: [[1, 2]]
  from Rank 2: [[2, 4]]
  from Rank 3: [[3, 6]]
```

💡 **关键**: 输出是一个**列表**，每个元素是一个 Rank 的 tensor！

#### 细节问题 2.3: broadcast 的通信量是多少？

**答案**:

Broadcast 使用**树形拓扑**，通信量为 O(log N)。

```
4 个 Rank 的 broadcast (从 Rank 0):

Step 1: Rank 0 → Rank 2
   0 [data]   0 [data]
   1 [ ]      1 [ ]
   2 [ ]  =>  2 [data]  (接收)
   3 [ ]      3 [ ]

Step 2: Rank 0 → Rank 1, Rank 2 → Rank 3
   0 [data]   0 [data]
   1 [ ]      1 [data]  (接收)
   2 [data]   2 [data]
   3 [ ]  =>  3 [data]  (接收)

总通信次数: log₂(4) = 2
```

**代码示例**:

```python
import torch
import torch.distributed as dist
import time

rank = dist.get_rank()

if rank == 0:
    tensor = torch.tensor([42.0], device='cuda')
else:
    tensor = torch.zeros(1, device='cuda')

print(f"[Rank {rank}] Before broadcast: {tensor.item()}")

# 从 Rank 0 广播
dist.broadcast(tensor, src=0)

print(f"[Rank {rank}] After broadcast: {tensor.item()}")
```

### 通信原语对比表

| 操作 | 输入 | 输出 | 通信量 | DDP 阶段 |
|------|------|------|--------|---------|
| **all-reduce** | 各不同 | 全相同 | O(N×M) | backward (梯度) |
| **all-gather** | 各不同 | 全相同（拼接） | O(N×M) | forward (TP) |
| **reduce-scatter** | 各不同 | 各不同（切分） | O(N×M) | - |
| **broadcast** | Rank 0 | 全相同 | O(log N) | 参数初始化 |
| **reduce** | 各不同 | Rank 0 | O(M) | 指标收集 |

**N**: world_size, **M**: tensor size

### 代码路径

在 verl 中，通信原语的使用示例：

- **all-reduce**: `verl/workers/fsdp_workers.py:450` (梯度同步)
- **all-gather**: `verl/workers/sharding_manager/fsdp_vllm.py:85` (参数聚合)
- **broadcast**: `verl/workers/sharding_manager/megatron_vllm.py:120` (参数分发)

### 实践任务

#### 任务 2.1: 实现自定义 all-reduce

不使用 `dist.all_reduce()`，用基础通信原语实现：

```python
import torch
import torch.distributed as dist

def my_all_reduce(tensor):
    """
    使用 reduce + broadcast 实现 all-reduce
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Step 1: reduce 到 Rank 0
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    
    # Step 2: 从 Rank 0 broadcast 到所有 Rank
    dist.broadcast(tensor, src=0)
    
    return tensor

# 测试
rank = dist.get_rank()
tensor = torch.tensor([rank + 1.0], device='cuda')

result = my_all_reduce(tensor.clone())
print(f"[Rank {rank}] my_all_reduce: {result.item()}")

# 对比官方实现
tensor_official = tensor.clone()
dist.all_reduce(tensor_official, op=dist.ReduceOp.SUM)
print(f"[Rank {rank}] official: {tensor_official.item()}")
```

**思考**: 
- 这个实现的通信量是多少？
- 为什么官方实现更快？（提示：Ring All-Reduce）

---

## 问题3: DDP 的工作原理

### 提问目标

深入理解 DDP 的梯度同步机制，能够调试 DDP 训练中的性能问题。

### 深挖细节

#### 细节问题 3.1: DDP 什么时候同步梯度？

**答案**: 在 **backward 期间**，每个参数的梯度计算完成后立即同步！

```
传统理解（错误）:
   forward → backward → all-reduce 梯度 → optimizer.step()

DDP 实际流程:
   forward → backward (逐层 all-reduce 梯度) → optimizer.step()
```

**为什么这样设计？**

1. **隐藏通信延迟**: 梯度计算和通信可以重叠
2. **减少显存**: 不需要存储完整的梯度 tensor

**图示**:

```
Layer 4 (输出层):
   计算梯度 ✓ → all-reduce ✓
Layer 3:
   计算梯度 ✓ → all-reduce ✓ (与 Layer 2 计算重叠)
Layer 2:
   计算梯度 ✓ → all-reduce ✓ (与 Layer 1 计算重叠)
Layer 1 (输入层):
   计算梯度 ✓ → all-reduce ✓
```

**代码验证**:

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 初始化
dist.init_process_group(backend="nccl")
model = SimpleModel().cuda()
ddp_model = DDP(model)

# Hook 来监控梯度同步
def grad_hook(name):
    def hook(grad):
        print(f"[{name}] 梯度计算完成，shape: {grad.shape}")
    return hook

for name, param in ddp_model.named_parameters():
    param.register_hook(grad_hook(name))

# 训练
x = torch.randn(2, 10, device='cuda')
y = torch.randn(2, 1, device='cuda')

output = ddp_model(x)
loss = ((output - y) ** 2).mean()
loss.backward()  # 观察打印顺序
```

**输出**（注意顺序）:
```
[fc2.bias] 梯度计算完成, shape: torch.Size([1])
[fc2.weight] 梯度计算完成, shape: torch.Size([1, 10])
[fc1.bias] 梯度计算完成, shape: torch.Size([10])
[fc1.weight] 梯度计算完成, shape: torch.Size([10, 10])
```

💡 **关键**: 梯度同步顺序与计算顺序**相反**（后向传播）！

#### 细节问题 3.2: DDP 如何处理不同 Rank 的 batch size？

**答案**: DDP 不强制要求 batch size 相同，但梯度会**平均**！

```python
# Rank 0: batch_size = 4
# Rank 1: batch_size = 2

# Rank 0 的梯度会自动除以 world_size (2)
# Rank 1 的梯度也会除以 2
# 导致 Rank 0 的样本权重被稀释！
```

⚠️ **最佳实践**: 始终保持各 Rank 的 batch size 相同！

**代码示例**:

```python
from torch.utils.data import DistributedSampler

# ✅ 正确写法
dataset = MyDataset()
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    drop_last=True  # 确保 batch size 一致
)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

#### 细节问题 3.3: DDP 的 bucket 机制是什么？

**答案**: DDP 将参数分组为"bucket"，每个 bucket 的梯度一起 all-reduce。

```
默认 bucket_size = 25 MB

参数分组:
   Bucket 0: layer1.weight (10MB) + layer1.bias (1MB) + layer2.weight (14MB) = 25MB
   Bucket 1: layer2.bias (1MB) + layer3.weight (20MB) = 21MB
   ...
```

**为什么需要 bucket？**

1. **减少通信次数**: 一次通信多个参数
2. **提高带宽利用率**: 大 tensor 更高效

**调整 bucket_size**:

```python
ddp_model = DDP(
    model,
    bucket_cap_mb=50,  # 增大 bucket (default: 25)
    gradient_as_bucket_view=True  # 减少内存拷贝
)
```

### 代码路径

在 verl 中，DDP 相关代码：

- **FSDP 初始化**: `verl/workers/fsdp_workers.py:200-250`
- **梯度同步**: 由 PyTorch FSDP 自动处理
- **Bucket 配置**: `verl/trainer/config/actor/dp_actor.yaml:15`

### 实践任务

#### 任务 3.1: 对比 DDP 和单卡训练

创建 `test_ddp.py`:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

def train_step(model, optimizer, data, target, use_ddp=False):
    optimizer.zero_grad()
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    # 初始化
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    
    # 模型
    model = ToyModel().cuda()
    ddp_model = DDP(model)
    
    # 优化器
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    # 数据
    data = torch.randn(32, 1024, device='cuda')
    target = torch.randint(0, 10, (32,), device='cuda')
    
    # Warmup
    for _ in range(10):
        train_step(ddp_model, optimizer, data, target, use_ddp=True)
    
    # 计时
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        train_step(ddp_model, optimizer, data, target, use_ddp=True)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    if rank == 0:
        print(f"DDP训练 100 step: {elapsed:.3f} s")
        print(f"吞吐量: {100 / elapsed:.2f} step/s")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

运行:
```bash
# 单卡
python test_ddp.py  # 基准

# 2 卡
torchrun --nproc_per_node=2 test_ddp.py

# 4 卡
torchrun --nproc_per_node=4 test_ddp.py
```

**思考**:
- 加速比能达到线性吗？
- 哪些因素限制了加速比？（提示：通信 vs 计算比）

---

## 概念验证实验

### 实验 1: 手写 Data Parallel

**目标**: 理解 DP 的本质：切分数据 + 同步梯度

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ManualDP:
    def __init__(self, model):
        self.model = model.cuda()
        dist.init_process_group(backend="nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # 同步初始参数
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
    
    def forward(self, data):
        # 各 Rank 处理不同数据
        return self.model(data)
    
    def backward(self, loss):
        loss.backward()
        
        # 手动同步梯度
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size  # 平均
    
    def step(self, optimizer):
        optimizer.step()
        optimizer.zero_grad()

# 使用
model = nn.Linear(10, 1)
manual_dp = ManualDP(model)

data = torch.randn(4, 10, device='cuda')  # 各 Rank 不同
target = torch.randn(4, 1, device='cuda')

output = manual_dp.forward(data)
loss = ((output - target) ** 2).mean()
manual_dp.backward(loss)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
manual_dp.step(optimizer)
```

**验证**: 对比 ManualDP 和 DDP 的参数是否一致。

---

## 源码阅读指南

### 阅读任务 1: torch.distributed.init_process_group

**目标**: 理解初始化流程中的关键步骤

**文件**: `torch/distributed/distributed_c10d.py`

**关键代码**（约 500 行，建议阅读 50-150 行）:

```python
# torch/distributed/distributed_c10d.py:50-150

def init_process_group(
    backend,
    init_method=None,
    timeout=default_pg_timeout,
    world_size=-1,
    rank=-1,
    store=None,
    group_name="",
    pg_options=None,
):
    # 1. 解析环境变量
    if rank == -1:
        rank = int(os.environ.get('RANK', 0))
    
    if world_size == -1:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # 2. 创建 Store (用于 Rendezvous)
    if store is None:
        if init_method is None:
            init_method = "env://"  # 从环境变量读取
        
        store = _create_store_from_uri(init_method, rank, world_size, timeout)
    
    # 3. 创建 ProcessGroup
    backend_obj = Backend(backend)
    pg = ProcessGroupNCCL(store, rank, world_size, pg_options)
    
    # 4. 设置为全局默认 ProcessGroup
    _world.pg_group_ranks[GroupMember.WORLD] = pg
    
    return pg
```

**阅读提示**:
- 注意 `_create_store_from_uri` 如何解析 `env://`
- 理解 `ProcessGroupNCCL` 的初始化参数
- 关注 `timeout` 的作用（避免 hang）

### 阅读任务 2: DistributedDataParallel.forward

**目标**: 理解 DDP 如何注册梯度钩子

**文件**: `torch/nn/parallel/distributed.py`

**关键代码**（约 1000 行，建议阅读 200-300 行）:

```python
# torch/nn/parallel/distributed.py:200-300

class DistributedDataParallel(Module):
    def __init__(self, module, ...):
        super().__init__()
        self.module = module
        
        # 注册梯度 hook
        self._register_comm_hook()
        
        # 分配参数到 bucket
        self._build_buckets()
    
    def _register_comm_hook(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self._make_hook(param))
    
    def _make_hook(self, param):
        def hook(grad):
            # 将梯度加入 bucket
            bucket = self._get_bucket_for_param(param)
            bucket.append(grad)
            
            # 如果 bucket 满了，触发 all-reduce
            if bucket.is_full():
                self._all_reduce_bucket(bucket)
        
        return hook
```

**阅读提示**:
- 注意 `_build_buckets()` 的分组逻辑
- 理解 `_all_reduce_bucket()` 的调用时机
- 关注 `gradient_as_bucket_view` 优化

---

## 自我检测清单

### 基础理解（必须全部掌握）

- [ ] 能解释为什么需要多卡训练（显存 + 速度）
- [ ] 能说出 RANK、WORLD_SIZE、LOCAL_RANK 的区别
- [ ] 能解释 `init_process_group()` 的作用
- [ ] 能区分 all-reduce、all-gather、broadcast
- [ ] 能画出 DDP 的梯度同步流程

### 进阶理解（建议掌握）

- [ ] 能解释 MASTER_ADDR 和 MASTER_PORT 的作用
- [ ] 能说出 nccl 和 gloo 的区别
- [ ] 能解释 DDP 的 bucket 机制
- [ ] 能调试 "NCCL hang" 问题
- [ ] 能优化 DDP 的通信效率

### 代码能力

- [ ] 能手写 2-GPU 的分布式训练脚本
- [ ] 能用 torchrun 启动多卡训练
- [ ] 能用 DistributedSampler 切分数据
- [ ] 能监控 GPU 通信带宽（nvidia-smi dmon）
- [ ] 能对比 DDP 和单卡的吞吐量

---

## 进阶挑战

### 挑战 1: 实现自定义通信原语

实现一个 `reduce_scatter`（结合 reduce 和 scatter）:

```python
def my_reduce_scatter(input_list, op=dist.ReduceOp.SUM):
    """
    输入: input_list = [tensor0, tensor1, ..., tensorN]
    输出: 当前 Rank 对应的 reduce 结果
    
    示例 (4 个 Rank):
       Rank 0: [t0, t1, t2, t3] → reduce(t0)
       Rank 1: [t0, t1, t2, t3] → reduce(t1)
       Rank 2: [t0, t1, t2, t3] → reduce(t2)
       Rank 3: [t0, t1, t2, t3] → reduce(t3)
    """
    # TODO: 你的实现
    pass
```

**提示**: 先 all-reduce，再取自己的 slice。

### 挑战 2: 优化 DDP 训练速度

给定一个慢速 DDP 训练脚本，优化至少 20% 吞吐量。

可能的优化方向:
- 调整 bucket_size
- 启用 gradient_as_bucket_view
- 使用混合精度训练
- 优化数据加载（num_workers、pin_memory）

### 挑战 3: 调试 NCCL Hang

模拟一个 NCCL hang 场景并解决：

```python
# 故意制造 hang
if rank == 0:
    dist.all_reduce(tensor)  # Rank 0 调用
# Rank 1-3 没有调用 → hang！

# 如何检测和修复？
```

**工具**:
- `export NCCL_DEBUG=INFO` (查看通信日志)
- `timeout` 命令 (避免无限等待)
- Stack trace (找到卡住的代码行)

---

## 总结

恭喜完成 Level 0！你已经掌握了：

✅ **分布式训练的基础**: RANK、ProcessGroup、Backend  
✅ **通信原语**: all-reduce、all-gather、broadcast  
✅ **DDP 原理**: 梯度同步、bucket 机制

**下一步**: 进入 [Level 1: 并行策略基础](./level1_parallelism_fundamentals.md)，学习 TP/PP/Zero 的原理和实现。

---

**反馈**: 如有问题或建议，请在 [GitHub Issues](https://github.com/volcengine/verl/issues) 中提出。
