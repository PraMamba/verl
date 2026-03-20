# verl 编排原语与 Rollout Buffer 深度源码分析

> **分析范围**: verl 的 Ray 单控制器编排架构、DataProto 序列化传输、异步调度与背压控制机制
>
> **分析日期**: 2026-03-20
>
> **核心源码路径**:
> - `verl/single_controller/ray/base.py` — RayWorkerGroup, 协同定位 Worker
> - `verl/single_controller/base/decorator.py` — Dispatch 模式与执行模式
> - `verl/protocol.py` — DataProto 数据协议与序列化
> - `verl/utils/ray_utils.py` — parallel_put 并行序列化
> - `verl/trainer/ppo/ray_trainer.py` — 主训练循环
> - `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` — 分桶权重传输
> - `verl/experimental/fully_async_policy/` — 实验性全异步架构

---

## 第一部分：编排原语 — Ray 单控制器架构的系统开销与扩展性

### 1.1 架构概览：单控制器模式

verl 采用经典的 **Ray 单控制器（Single-Controller）** 模式：一个 CPU 驱动进程（Controller）编排所有分布式 GPU Worker，所有 Worker 间通信必须经由 Controller 中转。

```
┌─────────────────────────────────────────────────┐
│              CPU Controller (Driver)             │
│                                                  │
│  RayPPOTrainer.fit()                             │
│    ├── dispatch_fn: DataProto.chunk() → split    │
│    ├── execute_fn: for w in workers: w.remote()  │
│    ├── ray.get(outputs)  ← 阻塞等待全部完成       │
│    └── collect_fn: DataProto.concat() → merge    │
└───────┬──────────┬──────────┬──────────┬─────────┘
        │          │          │          │
   ┌────▼──┐ ┌────▼──┐ ┌────▼──┐ ┌────▼──┐
   │GPU W0 │ │GPU W1 │ │GPU W2 │ │ ...   │
   │(FSDP) │ │(FSDP) │ │(FSDP) │ │       │
   └───────┘ └───────┘ └───────┘ └───────┘
```

**核心调度流程** (`verl/single_controller/ray/base.py:48-66`):

```python
def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    class Functor:
        def __call__(this, *args, **kwargs):
            args, kwargs = dispatch_fn(self, *args, **kwargs)       # ① 分片
            padding_count = kwargs.pop(_padding_size_key, 0)
            output = execute_fn(method_name, *args, **kwargs)       # ② 远程执行
            if blocking:
                output = ray.get(output)                            # ③ 阻塞收集
            output = collect_fn(self, output)                       # ④ 聚合
            return output
```

每一次 Worker 方法调用都经历四步：**分片 → 远程执行 → 阻塞收集 → 聚合**。

### 1.2 万卡规模下的调度器瓶颈

#### 1.2.1 Ray Actor 数量估算

verl 通过 `create_colocated_worker_cls`（`ray/base.py:981-1022`）将 Actor/Critic/Rollout 融合到单个 Ray Actor 中：

| 配置模式 | 10K GPU 下 Actor 数 | 说明 |
|---------|-------|------|
| 全融合（Actor+Critic+Rollout 共置） | ~10K | 每 GPU 一个 FusedWorker |
| 分池（Actor/Rollout 分离资源池） | ~15K-20K | 两个 RayWorkerGroup |

每个 `RayWorkerGroup` 会维护一个 `self._workers` 列表，长度等于该组的 Worker 数。

#### 1.2.2 `execute_all_async`：顺序 `.remote()` 扇出

**关键瓶颈** 在 `ray/base.py:859-887`：

```python
def execute_all_async(self, method_name: str, *args, **kwargs):
    length = len(self._workers)
    # ...
    result = []
    for i in range(length):                                    # ← 顺序 Python 循环
        sliced_args = tuple(arg[i] for arg in args)
        sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
        result.append(
            self._execute_remote_single_worker(               # ← 每次 gRPC 调用
                self._workers[i], method_name, *sliced_args, **sliced_kwargs
            )
        )
    return result
```

这是一个 **顺序 Python for 循环**，逐一向 N 个 Worker 发送 `.remote()` 调用。每次 `.remote()` 涉及：

1. 参数序列化（如果不是 ObjectRef）
2. 通过 gRPC 向 Ray Scheduler 提交任务
3. Ray Scheduler 将任务路由到目标节点

**开销估算**（10K Worker）：

| 操作 | 单次耗时 | 万卡规模总耗时 |
|------|---------|-------------|
| Python 循环 + 参数切片 | ~10μs | ~100ms |
| `.remote()` gRPC 提交 | ~100-500μs | **1-5s** |
| Ray GCS 任务追踪 | O(N) 元数据流量 | GCS 内存/CPU 显著增长 |

#### 1.2.3 `ray.get()` 的收集瓶颈

`func_generator:55` 中 `output = ray.get(output)` 接收 N 个 ObjectRef 的结果：

- **GCS 轮询开销**：Driver 对每个 Object 轮询 GCS 获取完成状态，O(N) 元数据流量
- **反序列化瓶颈**：所有结果在 Driver 单个 CPU 上通过 Python GIL 串行反序列化
- **内存压力**：N 个 DataProto 结果须同时物化到 Driver 节点内存

**然后** `collect_fn` 通常调用 `DataProto.concat()`（对 `DP_COMPUTE_PROTO` 模式），在 Driver CPU 上执行所有分片的合并 — 又一个串行 CPU 操作。

#### 1.2.4 与 Meta Monarch/TorchForge 的架构对比

| 维度 | verl（Ray 单控制器） | Monarch/TorchForge（原生 PyTorch P2P） |
|------|---------------------|--------------------------------------|
| **编排模式** | 单 CPU 控制器集中调度 | Worker 间 NCCL 集合通信自协调 |
| **调度延迟** | O(N) `.remote()` 调用（万卡 1-5s） | O(1) broadcast/scatter via NCCL（~10ms） |
| **数据传输** | Serialize → Plasma → Deserialize（≥3 次拷贝） | NCCL 直接 GPU-GPU（零拷贝） |
| **收集延迟** | O(N) 对象拉取到 Driver | O(log N) Tree reduce |
| **带宽利用率** | 受限于 Driver NIC + Plasma 吞吐 | 充分利用 RDMA/NVLink（200-900 GB/s） |
| **故障域** | Driver 故障 = 全局失败 | 单 Worker 故障可恢复 |
| **编程复杂度** | **极低**：Driver 上顺序 Python | **高**：SPMD 编程模型 |

**带宽差距**的直觉量化：对于 1GB DataProto 分片，Ray Object Store 路径需要 ~200-500ms（序列化+网络+反序列化），而 NCCL over RDMA 仅需 ~5ms。

### 1.3 DataProto 序列化与零拷贝分析

#### 1.3.1 默认序列化路径（`protocol.py:376-401`）

```python
def __getstate__(self):
    batch = self.batch.contiguous().consolidate()   # ① 内存整理（原地）
    # 默认路径:
    buffer = io.BytesIO()
    torch.save(batch, buffer)                       # ② torch.save → BytesIO（拷贝1）
    buffer_bytes = buffer.getvalue()                # ③ 提取 bytes（拷贝2）
    return buffer_bytes, self.non_tensor_batch, self.meta_info
    # ④ Ray pickle → Plasma Object Store（拷贝3）
```

**最少 3 次内存拷贝**：consolidate → torch.save 到 BytesIO → getvalue 提取 bytes → Ray pickle 到 Plasma。

#### 1.3.2 Numpy 优化路径

通过 `VERL_DATAPROTO_SERIALIZATION_METHOD=numpy` 启用（`protocol.py:386-394`）：

```python
def serialize_single_tensor(obj: torch.Tensor):     # protocol.py:240-243
    data = obj.flatten().contiguous().view(torch.uint8).numpy()  # ← memoryview 零拷贝
    dtype = str(obj.dtype).removeprefix("torch.")
    return dtype, obj.shape, data
```

`.numpy()` 返回底层存储的 memoryview（**零拷贝**），但当 pickle 序列化 numpy 数组到 Ray 时仍会创建拷贝。反序列化侧（`protocol.py:261-271`）通过 `bytearray(data)` 又创建一次拷贝：

```python
def deserialize_single_tensor(arr):
    dtype, shape, data = arr
    buffer = bytearray(data)              # ← 拷贝
    arr = torch.frombuffer(buffer, dtype=torch.uint8)
    return arr.view(torch_dtype).view(shape)
```

**Numpy 路径总结**：中间有一步零拷贝，但整体管线仍非零拷贝。

#### 1.3.3 唯一真正的零拷贝：CUDA IPC 分桶权重传输

`bucketed_weight_transfer.py:73-197` 实现了 verl 中 **唯一的 GPU 零拷贝传输**：

```python
class BucketedWeightSender:
    def _init_buffer(self):
        # 创建 CUDA buffer（默认 512MB）
        buffer = torch.empty(self.bucket_size, dtype=torch.uint8,
                           device=f"{get_device_name()}:{get_device_id()}")
        handle = reduce_tensor(buffer)          # ← 获取 CUDA IPC handle
        self.socket.send_pyobj(handle)          # ← 通过 ZMQ IPC 发送 handle

    async def async_send_weights(self, weights):
        async for name, weight in ensure_async_iterator(weights):
            # 填充 bucket buffer
            self.buffer[offset:offset+weight.nbytes].copy_(
                weight.view(-1).view(torch.uint8), non_blocking=True  # GPU → GPU 拷贝
            )
            offset += weight.nbytes

            if offset + weight.nbytes > self.bucket_size:
                get_torch_device().synchronize()
                self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
```

**传输链路**：
1. Sender 创建 CUDA buffer → 获取 IPC handle
2. 通过 ZMQ Unix Domain Socket (`ipc:///tmp/...`) 发送 handle
3. Receiver 通过 `rebuild_ipc(handle, device_id)` 映射同一 GPU 内存 — **零拷贝**
4. 权重分桶打包（默认 512MB/桶），减少 IPC 握手次数

**限制**：仅适用于 **同节点内** 协同定位的 FSDP Trainer 与 vLLM Rollout Worker（同一 GPU）。跨节点传输不适用。

有 shared memory 回退路径（`use_shm=True`，面向 NPU），走 CPU 共享内存 + `.to(device)` 拷贝。

#### 1.3.4 `parallel_put`：并行序列化优化

`verl/utils/ray_utils.py:51-84`：

```python
def parallel_put(data_list, max_workers=None):
    if max_workers is None:
        max_workers = min(len(data_list), 16)      # ← 硬编码上限 16
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        data_list_f = [executor.submit(put_data, i, data)
                       for i, data in enumerate(data_list)]
```

16 线程并行调用 `ray.put()`，利用 GIL 释放（memcpy 阶段）实现并行。但在万卡场景（dp_size=128+）下，16 线程上限成为瓶颈。

#### 1.3.5 Arrow/Plasma 优化 — 无

verl **未利用** Ray Object Store 内置的零拷贝能力：

- 未使用 Arrow 零拷贝反序列化
- 未使用 Plasma 共享内存的节点内对象共享
- `torch.save/BytesIO` 默认路径将整个 DataProto 作为不透明 bytes blob pickle，**主动绕过** 了 Ray 对 numpy array 的零拷贝支持

### 1.4 RLHF 典型负载下的序列化开销估算

对于 batch_size=1024, seq_len=32768 的典型配置：

| 字段 | Shape | dtype | 单个大小 |
|------|-------|-------|---------|
| input_ids | (1024, 32768) | int64 | 256 GB |
| attention_mask | (1024, 32768) | int64 | 256 GB |
| old_log_probs | (1024, 32768) | float32 | 128 GB |
| ref_log_prob | (1024, 32768) | float32 | 128 GB |
| values | (1024, 32768) | float32 | 128 GB |
| advantages | (1024, 32768) | float32 | 128 GB |
| **全量 batch 总计** | | | **~1 TB** |

实际上 Controller 会先按 dp_size 分片。以 dp_size=128（10K GPU / 80 TP）为例，每个分片约 8GB。每次分片传输的开销：

| 步骤 | 操作 | 估算耗时 |
|------|------|---------|
| consolidate | 内存整理 | ~10ms |
| torch.save → BytesIO | 序列化 + 拷贝1 | ~50ms |
| getvalue() | 拷贝2 | ~20ms |
| ray.put() → Plasma | pickle + 拷贝3 | ~50ms |
| **单分片总计** | | **~130ms** |
| **128 分片（16线程并行）** | | **~1s** |

加上 `ray.get()` 反序列化和 `concat()` 收集：**单步数据传输总开销约 2-5 秒**。

---

## 第二部分：Rollout Buffer 与背压控制

### 2.1 生产路径：严格同步，无 Buffer

**`RayPPOTrainer.fit()`**（`ray_trainer.py:1277-1350`）实现的是 **严格同步的 10 步流水线**：

```python
for epoch in range(current_epoch, self.config.trainer.total_epochs):
    for batch_dict in self.train_dataloader:
        # Step 0: 异步 checkpoint 完成（唯一的异步操作）
        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)

        batch = DataProto.from_single_dict(batch_dict)               # Step 1: 加载

        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
        self.checkpoint_manager.sleep_replicas()                      # Step 2: 生成 + 休眠

        batch = batch.union(gen_batch_output)                         # Step 3: 合并
        batch_reward = self._compute_reward_colocate(batch)           # Step 4: 奖励
        old_log_prob = self._compute_old_log_prob(batch)              # Step 5: 旧 log prob
        ref_log_prob = self._compute_ref_log_prob(batch)              # Step 6: 参考 log prob
        values = self._compute_values(batch)                          # Step 7: 价值估计
        batch = compute_advantage(batch, ...)                         # Step 8: 优势计算
        self._update_critic(batch)                                    # Step 9: 更新 Critic
        self._update_actor(batch)                                     # Step 10: 更新 Actor
```

**关键特征**：

- **无流水线重叠**：Rollout 必须完成，Critic/Actor 更新才能开始
- **无双缓冲**：不会在训练当前 batch 时预取下一个 batch
- **无异步队列**：无 Rollout Buffer
- **无背压机制**：同步意味着吞吐量由最慢阶段决定

**唯一的异步操作**：Megatron 异步 checkpoint 保存的 finalize（`ray_trainer.py:1279-1280`）。

### 2.2 混合引擎：GPU 时分复用与 Sleep/Wake

当推理引擎（vLLM/SGLang）和训练引擎（FSDP/Megatron）共享同一 GPU 时，verl 通过 **Sleep/Wake 机制** 进行时分复用：

```
┌────────────────────────────────────────┐
│         同一 GPU（协同定位）              │
│                                        │
│  ┌──────────┐  sleep()   ┌──────────┐  │
│  │ vLLM     │ ────────→  │ 释放     │  │
│  │ Rollout  │            │ KV Cache │  │
│  │ (推理)   │            │ + 权重   │  │
│  └──────────┘            └──────────┘  │
│        ↑                      ↓        │
│   wake_up()              训练使用      │
│   + 权重同步             全部 GPU 内存   │
│        ↑                      ↓        │
│  ┌──────────┐            ┌──────────┐  │
│  │ vLLM     │            │ FSDP     │  │
│  │ 恢复     │ ←────────  │ Training │  │
│  └──────────┘  完成训练   └──────────┘  │
└────────────────────────────────────────┘
```

**`CheckpointEngineManager.sleep_replicas()`**（`checkpoint_engine/base.py:394-399`）：

```python
async def sleep_replicas(self):
    """Sleep all rollout replicas: free weight and kv_cache device memory."""
    if self.backend != "naive":
        return
    await asyncio.gather(*[r.sleep() for r in self.replicas])
```

**隐式背压**：生产路径的"背压"本质上是 **GPU 内存互斥** — 推理和训练不能同时占用 GPU，因此不存在吞吐量不匹配问题：它们是串行的。

### 2.3 实验性全异步架构：真正的 Producer-Consumer 模式

`verl/experimental/fully_async_policy/` 目录实现了一个完整的异步替代架构。

#### 2.3.1 MessageQueue：有界异步队列

`message_queue.py:26-96` — 一个 Ray Actor 封装的有界队列：

```python
@ray.remote(num_cpus=2, max_concurrency=20)
class MessageQueue:
    def __init__(self, config, max_queue_size=1000):
        self.queue = deque(maxlen=self.max_queue_size)     # 有界 deque
        self.current_param_version = 0
        self.staleness_threshold = 3

        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)

    async def put_sample(self, sample, param_version):
        async with self._lock:
            if len(self.queue) >= self.max_queue_size:
                self.queue.popleft()                        # ← 队列满：丢弃最老样本
                self.dropped_samples += 1
            self.queue.append(sample)
            self._consumer_condition.notify_all()           # ← 唤醒消费者

    async def get_sample(self):
        async with self._lock:
            while len(self.queue) == 0 and self.running:
                await self._consumer_condition.wait()       # ← 阻塞等待
            data = self.queue.popleft()
            return data, len(self.queue)
```

**背压策略**：**Drop-Oldest**（丢弃最老样本）。队列满时静默丢弃并记录 `dropped_samples`。

#### 2.3.2 FullyAsyncRollouter：三层背压控制

`fully_async_rollouter.py:727-768` 实现了三层背压判断：

```python
async def _should_pause_generation(self) -> bool:
    queue_stats = self.message_queue_client.get_statistics_sync()
    queue_size = queue_stats["queue_size"]

    # 背压条件 1：队列溢出
    if queue_size >= self.max_queue_size:
        return True                     # → 暂停生成

    # 背压条件 2：陈旧度超限
    if self.staleness_samples >= self.max_required_samples:
        return True                     # → 暂停生成

    return False
```

**三层控制**：

| 层级 | 机制 | 阈值 | 效果 |
|------|------|------|------|
| L1: 队列容量 | `queue_size >= max_queue_size` | 可配置（默认 1000） | 暂停生成 |
| L2: 样本陈旧度 | `staleness_samples >= max_required_samples` | `required_samples × (staleness_threshold + 1)` | 暂停生成 |
| L3: 并发限制 | `len(active_tasks) >= max_concurrent_samples` | `num_servers × 16` | 等待完成 |

**暂停/恢复协议**（`fully_async_rollouter.py:751-768`）：

```python
async def pause(self):
    async with self.lock:
        self.paused = True
        if self.config.async_training.partial_rollout:
            await self.async_rollout_manager.cancel()       # 取消未完成请求
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
            self.active_tasks.clear()
        await self.async_rollout_manager.clear_kv_cache()   # 释放 GPU 内存
        self.monitor_loop_trigger = False
```

#### 2.3.3 FullyAsyncTrainer：解耦消费

`fully_async_trainer.py` 的 Trainer 独立于 Rollouter 运行，通过 MessageQueue 消费样本：

```python
def _get_samples_from_queue(self):
    queue_samples = []
    while len(queue_samples) < self.required_samples:
        sample, queue_len = self.message_queue_client.get_sample_sync()
        if sample is None:
            raise TrainingStopException()
        queue_samples.append(sample)
    batch = assemble_batch_from_rollout_samples(queue_samples, ...)
    return 0, batch
```

#### 2.3.4 异步 vs 同步架构对比

```
同步架构（生产路径）:
┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
│Rollout│──→│Reward│──→│OldLP │──→│Adv   │──→│Train │
│ (阻塞) │   │ (阻塞) │   │ (阻塞) │   │ (阻塞) │   │ (阻塞) │
└──────┘   └──────┘   └──────┘   └──────┘   └──────┘
           ← 每步串行，GPU 利用率由最慢阶段决定 →

异步架构（实验路径）:
┌──────────────────┐         ┌────────────────┐
│ FullyAsyncRollouter │       │ FullyAsyncTrainer │
│ （持续生成样本）    │       │ （持续消费训练）    │
│   ↓ put_sample    │       │   ↑ get_sample   │
│   ↓               │       │   ↑               │
│ ┌───────────────────────────────────────┐     │
│ │        MessageQueue (有界 deque)        │     │
│ │  [S1] [S2] [S3] ... [Sn]              │     │
│ │  ← max_queue_size 限制                  │     │
│ └───────────────────────────────────────┘     │
└──────────────────┘         └────────────────┘
```

| 维度 | 同步 RayPPOTrainer | FullyAsync Policy |
|------|-------------------|-------------------|
| **调度模型** | 同步阻塞，锁步执行 | 解耦异步 Producer-Consumer |
| **训练循环** | 等 Rollout 完成才训练 | 队列有样本即可训练 |
| **Buffer** | 无 | 有界 deque（`max_queue_size`） |
| **背压** | GPU 内存互斥（隐式） | 队列容量 + 陈旧度 + 并发限制 |
| **吞吐失配处理** | GPU 时分复用（sleep/wake） | 丢弃旧样本 + pause/resume |
| **延迟** | 较低（直接流水线） | 较高（队列缓冲） |
| **吞吐量** | 受限于最慢阶段 | 更高（推理/训练并行） |
| **成熟度** | 生产可用 | 实验性（`experimental/`） |

---

## 第三部分：万卡扩展性的核心瓶颈与优化建议

### 3.1 瓶颈总结

| 瓶颈 | 当前影响（万卡） | 根因 | 严重度 |
|------|---------------|------|--------|
| 顺序 `.remote()` 扇出 | 调度延迟 1-5s/步 | `execute_all_async` 串行循环 | **高** |
| `ray.get()` 万级 ObjectRef | 收集延迟 5-30s/步 | GCS 轮询 + 串行反序列化 | **高** |
| DataProto 3 次拷贝序列化 | ~130ms/8GB 分片 | torch.save + BytesIO + pickle | **中** |
| Driver CPU 优势计算 | ~100ms（可接受） | 串行 CPU 计算 | **低** |
| `parallel_put` 16 线程上限 | dp_size=128 时仍需 8 轮 | 硬编码常量 | **低** |
| 单 Driver 单点故障 | 训练全局失败 | 无 HA 机制 | **中** |

### 3.2 优化建议

#### 短期（代码级优化）

1. **提升 `parallel_put` 并发**（`ray_utils.py:70`）：
   - `max_workers = min(len(data_list), os.cpu_count())` 替代硬编码 16

2. **启用 Numpy 序列化路径**：
   - 设置 `VERL_DATAPROTO_SERIALIZATION_METHOD=numpy`
   - 减少一次 torch.save 拷贝

3. **利用 `DataProtoFuture` 延迟物化**（`protocol.py:1173-1227`）：
   - 当前几乎未使用，所有 `@register` 装饰器默认 `blocking=True`
   - 可改为 `blocking=False` + 显式 `.get()` 实现流水线

#### 中期（架构级优化）

4. **Worker 直连数据通道**：
   - 使用 NCCL 集合通信替代 Controller 中转数据
   - Controller 仅发送轻量级命令（方法名 + 元数据），不触碰数据
   - 优势计算下推到 Worker 端（每样本独立可并行）

5. **分层控制器**：
   ```
   Root Controller
     ├── Node Controller 0 (管理 8 GPU)
     ├── Node Controller 1 (管理 8 GPU)
     └── ...
     └── Node Controller 1250 (管理 8 GPU)
   ```
   - `.remote()` 扇出从 10K 降至 ~1250
   - 节点内使用 NCCL/CUDA IPC

6. **扩展 CUDA IPC 到数据传输**：
   - 当前 CUDA IPC 仅用于权重同步（`bucketed_weight_transfer.py`）
   - 可扩展到所有节点内 DataProto 传输

#### 长期（架构重构）

7. **混合架构：控制平面/数据平面分离**：
   - **控制平面**：保留 Ray 单控制器的编程简洁性
   - **数据平面**：使用 NCCL/RDMA 直接 GPU-GPU 传输
   - 类似 Monarch 的思路，但保留 verl 的灵活性

8. **Arrow 零拷贝序列化**：
   - 为 DataProto 实现自定义 Ray 序列化器
   - Tensor 数据以 Arrow buffer 存入 Plasma（零拷贝读取）
   - 仅 pickle 小型元数据（shapes, dtypes, keys）

---

## 附录

### A. 核心文件索引

| 文件路径 | 核心类/函数 | 功能 |
|---------|-----------|------|
| `verl/single_controller/ray/base.py` | `RayWorkerGroup`, `func_generator`, `execute_all_async`, `create_colocated_worker_cls` | Ray Worker 管理与调度 |
| `verl/single_controller/base/decorator.py` | `Dispatch`, `dispatch_dp_compute_data_proto`, `collect_dp_compute_data_proto` | 数据分发与收集策略 |
| `verl/protocol.py` | `DataProto`, `DataProtoFuture`, `serialize_tensordict`, `deserialize_tensordict` | 数据协议与序列化 |
| `verl/utils/ray_utils.py` | `parallel_put` | 并行 Object Store 写入 |
| `verl/trainer/ppo/ray_trainer.py` | `RayPPOTrainer.fit()` | 主训练循环（同步） |
| `verl/checkpoint_engine/base.py` | `CheckpointEngineManager`, `sleep_replicas`, `update_weights` | 权重同步与 GPU 时分复用 |
| `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` | `BucketedWeightSender`, `BucketedWeightReceiver` | CUDA IPC 零拷贝权重传输 |
| `verl/workers/rollout/replica.py` | `RolloutReplica`, `sleep`, `wake_up`, `abort_all_requests` | 推理引擎生命周期管理 |
| `verl/experimental/fully_async_policy/message_queue.py` | `MessageQueue` | 有界异步消息队列 |
| `verl/experimental/fully_async_policy/fully_async_rollouter.py` | `FullyAsyncRollouter`, `_should_pause_generation`, `pause`, `resume` | 异步 Rollout 生成与背压 |
| `verl/experimental/fully_async_policy/fully_async_trainer.py` | `FullyAsyncTrainer`, `_get_samples_from_queue` | 异步训练消费者 |

### B. DataProtoFuture — 未充分利用的并发钩子

`protocol.py:1173-1227` 提供了延迟物化机制，但在生产训练循环中未生效：

```python
@dataclass
class DataProtoFuture:
    collect_fn: Callable
    futures: list[ray.ObjectRef]

    def get(self):                  # ← 显式触发 ray.get()
        data = self._data
        return self.collect_fn(data)
```

`decorator.py:382-394` 中的 `materialize_futures=True` 默认值会自动解析所有 DataProtoFuture：

```python
def _materialize_futures(*args, **kwargs):
    for arg in args:
        if isinstance(arg, DataProtoFuture):
            arg = arg.get()             # ← 自动阻塞解析
```

这是一个为未来流水线设计的架构钩子，当前未产生实际并发收益。

### C. 关键配置参数（全异步模式）

| 参数 | 路径 | 默认值 | 含义 |
|------|------|--------|------|
| `staleness_threshold` | `config.async_training.staleness_threshold` | 3 | 样本最大陈旧版本数 |
| `require_batches` | `config.async_training.require_batches` | - | 每步训练所需 batch 数 |
| `max_queue_size` | `MessageQueue.__init__` | 1000 | 队列最大容量 |
| `partial_rollout` | `config.async_training.partial_rollout` | - | 权重同步时是否取消进行中的请求 |
| `trigger_parameter_sync_step` | `config.async_training` | - | 触发权重同步的 mini-batch 间隔 |
