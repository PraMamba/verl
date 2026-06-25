# 07 - Checkpoint Engine: 训练-推理权重传输子系统

## 1. 模块概览

| 属性 | 值 |
|------|------|
| 模块路径 | `verl/checkpoint_engine/` |
| 文件数量 | 7 |
| 总行数 | 2,611 |
| 核心抽象 | `CheckpointEngine` ABC |
| 注册后端 | `naive`, `nccl`, `nixl`, `kimi_ckpt_engine`, `mooncake` |
| 协调层 | `CheckpointEngineManager` |
| 上游依赖 | Ray, ZeroMQ, cupy, NIXL, Mooncake TransferEngine |

Checkpoint Engine 是 verl 混合引擎（Hybrid Engine）架构中**训练器与推理引擎之间权重同步**的核心传输层。在 RL 训练循环中，每个 step 结束后需要将更新后的模型权重从训练进程（FSDP/Megatron）传输到推理进程（vLLM/SGLang），Checkpoint Engine 正是负责这一高性能传输过程的抽象。

## 2. 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `__init__.py` | 66 | 条件导入与公开 API (`__all__`)，所有后端均用 `try/except ImportError` 防护 |
| `base.py` | 585 | 核心抽象层：`TensorMeta`、`CheckpointEngineRegistry`、`CheckpointEngine` ABC、`CheckpointEngineWithCache`、`ColocatedCheckpointEngine`、`CheckpointEngineWorker`、`CheckpointEngineManager`、以及 `split_weight_chunks` / `merge_weight_chunks` 辅助函数 |
| `nccl_checkpoint_engine.py` | 375 | `NCCLCheckpointEngine` -- 基于 NCCL 集合通信的广播传输 |
| `hccl_checkpoint_engine.py` | 375 | `HCCLCheckpointEngine` -- 华为 Ascend NPU 的 HCCL 集合通信 |
| `kimi_checkpoint_engine.py` | 391 | `KIMICheckpointEngine` -- 月之暗面定制的 ParameterServer P2P 传输 |
| `mooncake_checkpoint_engine.py` | 287 | `MooncakeCheckpointEngine` -- 基于 Mooncake TransferEngine 的 RDMA P2P 传输 |
| `nixl_checkpoint_engine.py` | 532 | `NIXLCheckpointEngine` -- NVIDIA NIXL 链式 P2P 传输 |

## 3. 核心数据结构

### 3.1 TensorMeta (base.py:34-46)

```python
@dataclass
class TensorMeta:
    name: str           # 权重张量名称
    shape: torch.Size   # 张量形状
    dtype: torch.dtype   # 张量数据类型
    chunk_offset: int   # 块偏移量（字节）
    chunk_size: int     # 块大小（字节）
    offset: int         # 在 bucket 中的偏移量
```

`TensorMeta` 是传输过程中的元数据单元。当一个大张量需要被分割为多个 chunk 装入 bucket 时，`chunk_offset` 和 `chunk_size` 记录该 chunk 在原始张量中的位置，`offset` 记录该 chunk 在传输 bucket 中的位置。

### 3.2 CheckpointEngineRegistry (base.py:49-93)

基于字典的后端注册表，采用装饰器模式：

```python
class CheckpointEngineRegistry:
    _registry: dict[str, type["CheckpointEngine"]] = {}

    def register(backend: str):        # 装饰器，注册后端
    def get(cls, backend: str):        # 获取后端类
    def new(cls, backend: str, *args, **kwargs):  # 工厂方法
```

已注册的后端名称：
- `"naive"` -- `ColocatedCheckpointEngine`（base.py:220）
- `"nccl"` -- `NCCLCheckpointEngine`（nccl_checkpoint_engine.py:102）或 `HCCLCheckpointEngine`（hccl_checkpoint_engine.py:96，覆盖同名注册）
- `"nixl"` -- `NIXLCheckpointEngine`（nixl_checkpoint_engine.py:238）
- `"kimi_ckpt_engine"` -- `KIMICheckpointEngine`（kimi_checkpoint_engine.py:222）
- `"mooncake"` -- `MooncakeCheckpointEngine`（mooncake_checkpoint_engine.py:34）

注意：`HCCLCheckpointEngine` 也注册为 `"nccl"`，在华为 Ascend 环境下会覆盖 NCCL 后端。

## 4. 类继承与接口体系

### 4.1 CheckpointEngine ABC (base.py:96-201)

这是所有传输后端的抽象基类，定义了标准的生命周期协议：

```
prepare()           -> dict[str, Any]           # 分配 bucket、注册 RDMA 内存、返回元数据
build_topology()    -> (trainer_kwargs, rollout_kwargs)  # 类方法，构建通信拓扑
init_process_group(**kwargs)                     # 初始化进程组
send_weights(weights, global_steps)              # async，发送权重
receive_weights(global_steps)                    # async，接收权重
finalize()                                       # 释放 bucket、销毁进程组
```

核心设计思想是**每次权重更新都经历 prepare -> init_process_group -> send/receive -> finalize 的完整生命周期**。`build_topology` 是类方法，由 Manager 在控制平面调用，根据 trainer 和 rollout 的 world_size 构建通信拓扑。

### 4.2 CheckpointEngineWithCache (base.py:203-217)

继承自 `CheckpointEngine`，增加了 `get_weights()` 方法用于从本地缓存（共享内存、磁盘等）获取权重。这是为 **Partial Rollout** 场景设计的：权重同步可以在推理请求未全部完成时异步进行，请求耗尽后再从本地缓存加载。文档注释引用了 Laminar 论文（https://arxiv.org/abs/2510.12633）。

### 4.3 ColocatedCheckpointEngine (base.py:221-275)

注册名为 `"naive"` 的最简实现，用于训练和推理共存于同一 GPU 的场景。`send_weights` 直接将 generator 赋值给 `self.weights`，`receive_weights` 通过 `yield from` 返回，**零拷贝**：

```python
def send_weights(self, weights, global_steps=None):
    self.weights = weights  # 直接保存引用

def receive_weights(self, global_steps=None):
    yield from self.weights
    self.weights = None
```

### 4.4 类层次总结

```
CheckpointEngine (ABC)
  |-- CheckpointEngineWithCache (带本地缓存)
  |-- ColocatedCheckpointEngine [naive] (同 GPU 零拷贝)
  |-- NCCLCheckpointEngine [nccl] (NCCL 广播)
  |-- HCCLCheckpointEngine [nccl] (HCCL 广播, Ascend NPU)
  |-- KIMICheckpointEngine [kimi_ckpt_engine] (ParameterServer P2P)
  |-- MooncakeCheckpointEngine [mooncake] (TransferEngine RDMA)
  |-- NIXLCheckpointEngine [nixl] (NIXL 链式 P2P)
```

## 5. CheckpointEngineWorker 与 CheckpointEngineManager

### 5.1 CheckpointEngineWorker (base.py:278-340)

`CheckpointEngineWorker` 继承自 `Worker`（verl 的 Ray Worker 基类），与推理引擎的 `WorkerProc` 共处同一 GPU。关键逻辑：

- **构造函数**（base.py:287-320）：从 `rollout_config.checkpoint_engine` 读取配置，通过 `update_weights_bucket_megabytes << 20` 将 MB 转换为字节（base.py:301），调用 `CheckpointEngineRegistry.new()` 创建后端实例。同时支持自定义后端模块的动态导入（`import_external_libs`）。
- **`update_weights`**（base.py:322-325）：`@register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)` 注册的异步方法，从 checkpoint engine 接收权重后调用 `server_adapter.update_weights()` 更新推理引擎。
- **`execute_checkpoint_engine`**（base.py:327-329）：`@register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)` 注册的通用方法分派器，通过 `getattr(self.checkpoint_engine, method)` 转发任意方法调用。

### 5.2 CheckpointEngineManager (base.py:345-514)

Manager 是整个权重同步流程的编排器，由 Ray 控制平面的 Trainer 持有。其构造函数接收 `CheckpointEngineConfig`、trainer `RayWorkerGroup` 和 rollout `RolloutReplica` 列表。

核心流程 `update_weights`（base.py:470-514）分 8 步：

```
步骤 0: 若 backend == "naive"，直接调用 trainer.update_weights 并返回（同 GPU 场景）
步骤 1: abort_replicas() -- 中止所有未完成的推理请求（partial rollout）
步骤 2: 将所有 replica 的 worker 合并为临时 RayWorkerGroup
步骤 3: release_kv_cache_replicas() -- 释放 KV cache（保留模型权重）
步骤 4: build_process_group() -- 构建通信拓扑并初始化进程组
步骤 5: 并行调用 trainer.update_weights + rollout.update_weights
步骤 6: finalize -- 清理进程组和 buffer
步骤 7: resume_kv_cache_replicas() -- 恢复 KV cache
步骤 8: resume_generation_replicas() -- 恢复推理请求处理
```

`build_process_group`（base.py:387-412）的三阶段：
1. 所有 worker 调用 `prepare`，收集元数据
2. 调用 `backend_cls.build_topology()` 计算拓扑
3. 所有 worker 调用 `init_process_group` 建立连接

弹性伸缩支持：`add_replicas`（base.py:414-420）和 `remove_replicas`（base.py:422-429）允许动态增减 rollout replica。

### 5.3 Bucket 分块辅助函数

**`split_weight_chunks`**（base.py:517-543）：将权重 generator 按 `bucket_size` 分块。每个权重先 `view(-1).view(torch.uint8)` 展平为字节流，然后按 bucket 容量切割，生成 `(TensorMeta, chunk)` 元组。

**`merge_weight_chunks`**（base.py:546-585）：反向操作。若单个权重小于 bucket，直接 `view(dtype).view(shape)` 还原；若跨多个 chunk，则分配临时 buffer 逐块拼接。通过 `chunk_offset` 和 `chunk_size` 的断言保证顺序正确。

## 6. 传输后端详解

### 6.1 NCCLCheckpointEngine (nccl_checkpoint_engine.py:103-375)

**传输机制**：1-to-N 广播模式。Trainer rank 0 通过 NCCL `collective.broadcast` 将权重广播给所有 rollout worker。

**拓扑结构**（nccl_checkpoint_engine.py:160-171）：
- Trainer rank 0 的 rank=0，其余 trainer worker 的 rank=-1（不参与通信）
- Rollout worker 的 rank 从 1 到 rollout_world_size
- 总 world_size = rollout_world_size + 1

**元数据传输**：使用 ZeroMQ PUB/SUB 模式。Master（rank 0）启动 ZMQ PUB server（nccl_checkpoint_engine.py:173-185），其他 rank 连接并订阅 `"bucket_metadata"` topic。每次广播前先通过 ZMQ 发送 `bucket_meta` 字典，再通过 NCCL 广播 bucket 数据。

**双缓冲**：使用 `send_buf` 和 `recv_buf` 两个 buffer（各 bucket_size 字节），通过交换实现流水线化——当前 bucket 广播的同时可以填充下一个 bucket。Master 进程使用 cupy 分配 buffer 以避免 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 下的内存注册错误（nccl_checkpoint_engine.py:137-139）。

**`BroadcastOperation`**（nccl_checkpoint_engine.py:49-99）：将 NCCL broadcast 放在线程池执行器中异步运行。先通过 ZMQ 发送/接收元数据，再调用 `collective.broadcast`。

**send_weights 流程**（nccl_checkpoint_engine.py:229-299）：
1. 遍历 `split_weight_chunks` 分块
2. 当 bucket 满时，等待前一个 `BroadcastOperation` 完成，启动新的广播，交换双缓冲
3. 最后一个 bucket 标记 `is_last: True`

### 6.2 HCCLCheckpointEngine (hccl_checkpoint_engine.py:97-375)

华为 Ascend NPU 的对应实现，与 NCCL 版本结构几乎一致，关键差异：
- 使用 `torch.npu` 替代 `torch.cuda`（hccl_checkpoint_engine.py:122）
- 使用 `StatelessProcessGroup` 进行进程组初始化而非 `ray.util.collective`（hccl_checkpoint_engine.py:209-211）
- `BroadcastOperation._run` 直接执行而非在线程池中运行（hccl_checkpoint_engine.py:73-85）
- `MasterMetadata` 增加 `dist_ip` 和 `dist_port` 字段（hccl_checkpoint_engine.py:38-42）
- 注册名也是 `"nccl"`，在 NPU 环境中会覆盖 CUDA 版本的注册

### 6.3 KIMICheckpointEngine (kimi_checkpoint_engine.py:223-391)

月之暗面定制的传输引擎，关键特点：
- **All-to-All 拓扑**：所有 trainer 和 rollout worker 都参与进程组，rank 从 0 到 trainer_world_size + rollout_world_size - 1（kimi_checkpoint_engine.py:268-283）
- **外部 ParameterServer**：依赖 `checkpoint_engine.ps.ParameterServer` 外部包
- **CPU 卸载**：发送端先用 `ThreadPoolExecutor(max_workers=32)` 将权重并行卸载到 CPU（kimi_checkpoint_engine.py:340-351），然后通过 `register_checkpoint` 注册到 ParameterServer
- **多 trainer 分片**：通过 `ckpt_get_named_tensor_buckets` 按 `tensor_idx % world_size == rank_id` 将权重分片到不同 trainer rank（kimi_checkpoint_engine.py:37-63）
- 接收端使用 monkey-patched 的 `receive_tensor` 方法（kimi_checkpoint_engine.py:66-172, 311），支持 RDMA 设备的 H2D buffer 优化

### 6.4 MooncakeCheckpointEngine (mooncake_checkpoint_engine.py:35-287)

基于 Mooncake TransferEngine 的 RDMA 传输，特点：
- **P2P 链式传输**：rank 0 -> rank 1 -> ... -> rank N-1 的链式传输拓扑
- **TransferEngine 初始化**（mooncake_checkpoint_engine.py:65-73）：使用 `"P2PHANDSHAKE"` 模式，根据设备类型选择 `"ascend_direct"` 或 `"rdma"` 传输后端
- **双 bucket 分区**：`self.buf = torch.empty(2 * self.bucket_size, ...)` 分为两段交替使用（mooncake_checkpoint_engine.py:79）
- **Magic 同步**：使用 4 字节 magic 值 `[0xAB, 0xDC, 0xEF, 0x88]` 作为完成信号（mooncake_checkpoint_engine.py:143, 236）。发送端通过轮询 buffer 头部的 magic 值来判断接收端是否已读取完毕
- **进程组**：使用 `StatelessProcessGroup` 进行 `all_gather_obj` 和 `send_obj/recv_obj` 元数据交换
- **`transfer_sync_read/write`**：接收端通过同步读取远端 buffer，完成后向发送端写入 magic 信号

### 6.5 NIXLCheckpointEngine (nixl_checkpoint_engine.py:239-532)

NVIDIA NIXL 提供的高性能 P2P 传输库，支持 UCX、UCCL、Mooncake 等多种底层传输：

**NixlAgent 封装**（nixl_checkpoint_engine.py:55-141）：
- 每个 worker 拥有一个 `NixlAgent`，使用 UUID 作为 agent_name
- ZMQ PULL/PUSH 替代 nixl 原生的 `send_notif` 进行元数据通信
- 维护 `notifications` 和 `messages` 两个字典队列

**两种 Operation 抽象**：
- `ReadableOperation`（nixl_checkpoint_engine.py:143-174）：发送端创建，将本地 buffer 暴露给远端读取，然后等待远端读取完成的 notification
- `ReadOperation`（nixl_checkpoint_engine.py:177-235）：接收端创建，先 `read_metadata` 获取远端 buffer 描述符，再 `begin_read` 启动 RDMA 读取，最后 `wait_for_complete` 轮询直到状态变为 `"DONE"`

**链式传输拓扑**（nixl_checkpoint_engine.py:288-305）：
- 每个 worker 只与前驱和后继 agent 通信（`prev_agent` / `next_agent`）
- rank 0（trainer）只有 next_agent
- 中间 rollout worker 同时有 prev_agent 和 next_agent
- 最后一个 rollout worker 只有 prev_agent

**接收端流水线**（nixl_checkpoint_engine.py:452-532）：
1. 从 prev_agent 读取 bucket
2. 同时向 next_agent 暴露当前 bucket 供其读取
3. yield 当前 buffer 中的张量
4. 等待双向操作完成，交换双缓冲

## 7. 架构图

```
┌──────────────────────────────────────────────────────────────┐
│                    CheckpointEngineManager                    │
│                  (Ray 控制平面编排器)                          │
│                                                              │
│  update_weights 8-step 流程:                                 │
│  abort -> merge workers -> release KV -> build PG ->         │
│  send/recv weights -> finalize -> resume KV -> resume gen    │
└──────┬─────────────────────────────────────────────┬─────────┘
       │                                             │
       v                                             v
┌──────────────┐                          ┌──────────────────┐
│ Trainer Side │                          │   Rollout Side   │
│              │                          │                  │
│ ModelEngine  │                          │ CheckpointEngine │
│   (FSDP/     │                          │    Worker (Ray)  │
│  Megatron)   │                          │                  │
│      │       │                          │    ┌──────────┐  │
│      v       │                          │    │ Server   │  │
│ CheckpointEng│   ===== 传输层 =====     │    │ Adapter  │  │
│   send_wts   │──────────────────────────│──> │(vLLM/    │  │
│              │  NCCL/NIXL/Mooncake/KIMI │    │ SGLang)  │  │
└──────────────┘                          └──────────────────┘
```

## 8. 设计要点与扩展指南

### 8.1 统一的生命周期协议

所有后端都遵循 `prepare -> build_topology -> init_process_group -> send/receive -> finalize` 的标准生命周期。这使得 Manager 可以完全不感知具体传输实现。新增后端只需：

1. 继承 `CheckpointEngine`
2. 用 `@CheckpointEngineRegistry.register("backend_name")` 注册
3. 实现 6 个抽象方法
4. 或者通过 `custom_backend_module` 配置动态加载

### 8.2 双缓冲流水线

NCCL、HCCL、NIXL、Mooncake 四种后端都实现了**双缓冲**策略：维护 `send_buf` 和 `recv_buf` 两个等大的 buffer，通过交换引用实现通信与数据填充的流水线化。这意味着设备内存开销为 `2 * bucket_size`。

### 8.3 元数据与数据分离

所有基于集合通信的后端都使用**带外元数据传输**：
- NCCL/HCCL：ZeroMQ PUB/SUB
- KIMI：ParameterServer 的 `gather_metas`
- Mooncake：`StatelessProcessGroup.send_obj/recv_obj`
- NIXL：ZMQ PUSH/PULL

这样数据通道（NCCL/RDMA）只传输原始字节，元数据（张量名、形状、dtype、偏移）走轻量级的 TCP 通道。

### 8.4 Partial Rollout 支持

Manager 的 `update_weights` 流程中内置了 partial rollout 支持（base.py:483, 514）：
1. 先 `abort_replicas()` 中止正在处理的推理请求
2. 权重同步完成后 `resume_generation_replicas()` 恢复请求处理
3. 配合 `release_kv_cache_replicas()` / `resume_kv_cache_replicas()` 单独管理 KV cache 内存

### 8.5 广播 vs P2P 拓扑

两种根本不同的传输拓扑：
- **广播**（NCCL/HCCL）：1-to-N，trainer rank 0 广播给所有 rollout worker，world_size = rollout_world_size + 1
- **链式 P2P**（NIXL/Mooncake）：rank 0 -> rank 1 -> ... -> rank N-1，每个节点只与相邻节点通信，延迟更可预测但总传输时间线性增长
- **分布式 P2P**（KIMI）：所有 trainer 并行分担权重分片，通过 ParameterServer 协调
