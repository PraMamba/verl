# verl 权重同步与部分 Rollout 处理深度源码分析

> **分析范围**: 权重同步多后端架构（NCCL/NIXL/HCCL/Mooncake/IPC）、Megatron 3D 并行权重拼装、部分 Rollout 的 Save/Restore 机制
>
> **分析日期**: 2026-03-20
>
> **核心源码路径**:
> - `verl/checkpoint_engine/` — 全部权重同步后端
> - `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` — CUDA IPC 分桶传输
> - `verl/utils/megatron_utils.py` — Megatron 3D 并行权重采集（`per_tensor_generator`）
> - `verl/models/mcore/weight_converter.py` — Megatron → HuggingFace 权重名映射
> - `verl/workers/rollout/vllm_rollout/vllm_async_server.py` — vLLM Sleep/Wake/Abort/Resume
> - `verl/experimental/fully_async_policy/` — 全异步部分 Rollout 实现

---

## 第一部分：权重同步架构总览

### 1.1 多后端 Checkpoint Engine 注册表

verl 通过 `CheckpointEngineRegistry`（`base.py:37-82`）支持多种权重同步后端：

| 后端 | 注册名 | 拓扑 | 传输层 | 适用场景 |
|------|--------|------|--------|---------|
| 协同定位 IPC | `"naive"` | 点对点 | ZMQ + CUDA IPC | 同 GPU 共置 |
| NCCL 广播 | `"nccl"` | 星形（1→N） | NCCL broadcast | NVIDIA GPU 分离部署 |
| HCCL 广播 | `"hccl"` | 星形（1→N） | HCCL broadcast | 华为 Ascend NPU |
| NIXL P2P | `"nixl"` | 环形链 | NIXL (UCX/UCCL/Mooncake) | RDMA 优化 |
| Mooncake P2P | `"mooncake"` | 环形链 | Mooncake TransferEngine | 月光 RDMA |
| Kimi 参数服务器 | `"kimi"` | 分片分发 | checkpoint_engine lib | Kimi 定制 |

### 1.2 端到端权重同步流程

```
┌─────────────────────────────────────────────────────────────────┐
│                  CheckpointEngineManager.update_weights()        │
│                         (base.py:401-437)                        │
├─────────────────────────────────────────────────────────────────┤
│ Step 1: abort_all_requests() → 中断进行中的推理请求               │
│ Step 2: 创建临时 RayWorkerGroup 聚合所有 rollout workers          │
│ Step 3: build_process_group() → 建立 trainer↔rollout 通信拓扑     │
│ Step 4: trainer.send_weights() ∥ rollout.receive_weights()       │
│ Step 5: finalize() → 销毁进程组                                  │
│ Step 6: resume_generation() → 恢复推理                           │
└─────────────────────────────────────────────────────────────────┘
```

源码（`base.py:401-437`）：

```python
async def update_weights(self, global_steps=None):
    # naive 路径：直接同步
    if self.backend == "naive":
        ray.get(self.trainer.update_weights(global_steps=global_steps))
        return

    # 1. 中断未完成请求（部分 rollout 支持）
    await asyncio.gather(*[r.abort_all_requests() for r in self.replicas])

    # 2. 创建临时 worker group
    workers = []
    for replica in self.replicas:
        workers.extend(replica.workers)
    rollout = RayWorkerGroup(worker_handles=workers, ...)

    # 3. 建立进程组
    self.build_process_group(rollout)

    # 4. 并行发送+接收
    ray.get(trainer.update_weights(...) + rollout.update_weights(...))

    # 5. 清理
    ray.get(trainer.execute_checkpoint_engine(["finalize"] * ...) +
            rollout.execute_checkpoint_engine(["finalize"] * ...))

    # 6. 恢复推理
    await asyncio.gather(*[r.resume_generation() for r in self.replicas])
```

---

## 第二部分：部分 Rollout 的 Save/Restore 机制

### 2.1 问题：长序列生成中的权重同步中断

当 100K+ token 的超长序列正在生成时，训练端完成一步更新需要同步权重。此时推理引擎面临选择：

| 策略 | verl 实现 | 开销 |
|------|----------|------|
| ~~等待生成完成~~ | 不采用 | 延迟过大 |
| ~~丢弃未完成序列~~ | 不采用 | 浪费计算 |
| **中断 + 保存 + 恢复** | ✅ 采用 | 显存 I/O 开销 |

### 2.2 vLLM Abort/Resume 实现

**`vllm_async_server.py:676-748`**：

```python
async def abort_all_requests(self, reset_prefix_cache=True):
    # vLLM >= 0.12.0: 使用原生 pause_generation API
    await self.engine.pause_generation(
        wait_for_inflight_requests=False,   # 不等待完成
        clear_cache=reset_prefix_cache,      # 清理 prefix cache
    )
    # 返回中断的请求 ID 和已生成的 token 数

async def resume_generation(self):
    await self.engine.resume_generation()
```

### 2.3 部分 Rollout 的"显式保存/恢复"策略

**关键发现**：verl **不是** 将 KV cache 换出到 CPU/磁盘再换回，而是采用了一种 **更聪明的设计 — Prompt 拼接续生成**。

#### 2.3.1 保存阶段：只保存 Token IDs 和 LogProbs

当 `abort_all_requests()` 被调用时，已生成的 token 会被捕获到 `TokenOutput` 结构中（`replica.py:39-51`）：

```python
class TokenOutput(BaseModel):
    token_ids: list[int]                      # 已生成的 token IDs（CPU 列表）
    log_probs: Optional[list[float]] = None   # 对应的 log probs（CPU 列表）
    routed_experts: Optional[Any] = None      # MoE 路由信息
    stop_reason: Optional[str] = None         # "aborted" | "completed" | "length"
```

**关键点**：
- Token IDs 和 log probs 是 **CPU 侧的 Python 列表**，不是 GPU 张量
- KV cache **直接释放**（`clear_cache=True`），**不保存到任何 buffer**
- 显存 I/O 开销 ≈ **零**（只有轻量级 CPU 元数据）

#### 2.3.2 恢复阶段：Prompt 拼接续生成

权重同步完成后，恢复机制 **不重建 KV cache**，而是将已生成 token 拼接到原始 prompt 后重新推理：

**`fully_async_policy/agent_loop/agent_loop.py:47-128`** — `FullyAsyncLLMServerManager.generate()`：

```python
async def generate(self, request_id, *, prompt_ids, sampling_params, ...):
    final_output = TokenOutput(token_ids=[], log_probs=[], num_preempted=0)

    while True:
        # ★ 核心：将之前生成的 token 拼接到 prompt 后续生成
        output = await super().generate(
            request_id=request_id,
            prompt_ids=prompt_ids + final_output.token_ids,  # ← 拼接！
            sampling_params=sampling_params, ...
        )

        # 累积输出
        final_output.token_ids.extend(output.token_ids)
        if output.log_probs:
            final_output.log_probs.extend(output.log_probs)

        # 更新剩余 token 预算
        if original_max_tokens:
            sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)

        # 如果不是 abort 触发的终止，退出循环
        if output.stop_reason not in ("aborted", "abort") or not partial_rollout_resume:
            break

    return final_output
```

**`partial_single_turn_agent_loop.py:28-134`** — 更详细的恢复逻辑：

```python
async def run(self, sampling_params, **kwargs):
    output = kwargs.get("output", None)

    if output and output.extra_fields.get("is_cancel", False):
        # ★ 恢复模式：用原始 prompt + 已生成 token 重新推理
        prompt_ids = output.prompt_ids + output.response_ids   # 拼接
        param_version_start = output.extra_fields["param_version_start"]
    else:
        prompt_ids = await self.apply_chat_template(messages, ...)

    # 用新权重续生成
    response_ids, response_logprobs, is_cancel = await self.server_manager.generate_for_partial(
        request_id=request_id,
        prompt_ids=prompt_ids,   # 拼接后的 prompt
        sampling_params=sampling_params, ...
    )

    # 合并历史和新生成的 token
    if output:
        response_logprobs = output.response_logprobs + response_logprobs
        response_ids = output.response_ids + response_ids
```

### 2.4 100K 超长上下文下的 I/O 开销分析

#### 2.4.1 保存阶段开销

| 保存内容 | 100K token 的大小 | 存储位置 | GPU 显存 I/O |
|---------|-----------------|---------|-------------|
| Token IDs | ~400KB（100K × 4B int） | CPU Python list | **0** |
| Log Probs | ~400KB（100K × 4B float） | CPU Python list | **0** |
| MoE 路由 | ~数 MB（如有） | CPU Python list | **0** |
| KV Cache | **直接释放** | 不保存 | **0** |
| **总 GPU I/O** | | | **≈ 0** |

**结论**：保存阶段 **没有任何 GPU 显存 I/O 开销**。Token IDs 和 logprobs 已经作为推理输出存在于 CPU 侧。

#### 2.4.2 恢复阶段开销：Prefill 重计算

恢复时的主要开销是 **用新权重对拼接 prompt 做 Prefill**（KV cache 重建）：

| 开销项 | 100K token 估算 | 说明 |
|--------|---------------|------|
| Prefill 计算 | ~数秒（A100/H100） | 取决于模型大小和 TP |
| KV Cache 重建 | ~数 GB | 自动由 vLLM/SGLang engine 管理 |
| 剩余 token 生成 | 正常生成速度 | 续生成剩余部分 |

**Prefill 开销 vs 从头重新生成**：

```
从头重新生成 100K:  Prefill(prompt) + Decode(100K)
部分恢复（50K处中断）: Prefill(prompt + 50K) + Decode(剩余 50K)
                     ↓
节省: ~50% 的 Decode 时间（50K tokens × ~10ms/token ≈ 500s）
额外: Prefill 50K tokens（~2-5s，因为 Prefill 是并行的）
```

**净收益巨大**：Prefill 50K token 仅需数秒（GPU 并行），而 Decode 50K token 需数百秒（自回归串行）。恢复策略的时间收益远超 Prefill 开销。

#### 2.4.3 Prefix Cache 的作用

如果启用 `enable_prefix_caching`，vLLM 可以缓存 prompt 部分的 KV cache，进一步减少恢复开销：

```python
# vllm_async_server.py:314
args["enable_prefix_caching"] = self.config.enable_prefix_caching
```

但在权重同步后，KV cache 必须用新权重重新计算，因此 **权重更新后 prefix cache 无效**：

```python
# vllm_async_server.py:625
await self.engine.reset_prefix_cache()  # 新权重 → 清理旧 prefix cache
```

### 2.5 时间收益保证分析

```
同步训练（无部分 rollout）:
┌────────────────┐ ┌──────┐ ┌────────────────┐
│ Generate 100K  │→│ Sync │→│ Generate 100K  │  每步 2×生成时间
│   (耗时 T_gen) │ │  Tw  │ │   (耗时 T_gen) │
└────────────────┘ └──────┘ └────────────────┘

异步 + 部分 rollout:
┌──────────┐              ┌──────┐    ┌──────────┐
│ Gen 50K  │─→ abort ──→  │ Sync │ →  │ Prefill  │ + Gen 50K
│          │   保存 IDs    │  Tw  │    │ 50K      │   续生成
└──────────┘              └──────┘    └──────────┘

节省时间 ≈ 50K × T_decode_per_token - T_prefill_50K
         ≈ 500s - 3s = ~497s（对于 100K 序列在 50K 处中断的情况）
```

---

## 第三部分：NCCL 分桶广播与 3D 并行的耦合

### 3.1 核心挑战

当训练后端是 Megatron-Core（TP×PP×VPP×CP×EP）时：

```
Megatron-Core 训练集群 (TP=8, PP=4, CP=2, EP=4):
┌──────────────────────────────────────────────────┐
│  PP Stage 0        PP Stage 1      ... Stage 3   │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐ │
│  │ TP0 TP1  │     │ TP0 TP1  │     │ TP0 TP1  │ │
│  │ TP2 TP3  │     │ TP2 TP3  │     │ TP2 TP3  │ │
│  │ TP4 TP5  │     │ TP4 TP5  │     │ TP4 TP5  │ │
│  │ TP6 TP7  │     │ TP6 TP7  │     │ TP6 TP7  │ │
│  └──────────┘     └──────────┘     └──────────┘ │
│   Layers 0-7       Layers 8-15     Layers 24-31  │
│   (每个 TP shard                                  │
│    持有 1/8 参数)                                  │
└──────────────────────────────────────────────────┘
                    ↓ 需要转换为 ↓
vLLM 推理集群 (TP=4):
┌──────────────────┐
│  TP0  TP1  TP2  TP3  │  ← 每个持有全部 32 层的 1/4 参数
└──────────────────┘
```

**挑战**：
1. 训练 TP=8 ≠ 推理 TP=4
2. PP=4 意味着每个 stage 只有 1/4 的层
3. VPP 交错进一步打散层分布
4. EP=4 意味着专家分片分布在 4 个 rank 上
5. Megatron 参数名（如 `decoder.layers.0.self_attention.linear_qkv.weight`）≠ HuggingFace/vLLM 参数名（如 `model.layers.0.self_attn.q_proj.weight`）

### 3.2 verl 的解决方案：两阶段分离设计

**核心设计原则**：Checkpoint Engine 对 3D 并行完全无感。所有并行维度的处理在 **权重提取阶段** 完成，Checkpoint Engine 只传输 **完整的、未分片的、HuggingFace 命名的** 张量。

```
┌────────────────────────────────────────────────────┐
│  阶段 1: 权重提取 (per_tensor_generator)             │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐      │
│  │ PP 广播   │→│ TP 聚合   │→│ EP 聚合   │        │
│  │(跨 stage) │  │(跨 rank) │  │(跨 expert)│        │
│  └───────────┘  └───────────┘  └───────────┘      │
│       ↓              ↓              ↓               │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐      │
│  │ VPP 层号  │  │ QKV/GateUp│  │ 专家编号  │      │
│  │ 归一化    │  │ 拆分重组   │  │ 重排     │       │
│  └───────────┘  └───────────┘  └───────────┘      │
│                      ↓                              │
│         weight_converter.convert_param()            │
│         Megatron 名 → HuggingFace 名               │
│                      ↓                              │
│  Generator[tuple[hf_name, full_tensor]]  ← 完整张量 │
└────────────────────────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────┐
│  阶段 2: 权重传输 (Checkpoint Engine)                │
│  对 3D 并行完全无感，只看到 (name, tensor) 流        │
│  ┌─────────────────────────────────────────────┐   │
│  │ Bucketed Broadcast / P2P / IPC              │   │
│  │ (NCCL / NIXL / HCCL / Mooncake / ZMQ)      │   │
│  └─────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────┐
│  阶段 3: 权重加载 (vLLM/SGLang)                     │
│  model.load_weights(weights_generator)              │
│  vLLM 内部按自己的 TP 拓扑自动分片                    │
└────────────────────────────────────────────────────┘
```

### 3.3 阶段 1 详解：`per_tensor_generator`

**`megatron_utils.py:956-1098`** — 从 Megatron 3D 并行模型中提取完整参数的核心函数。

#### 3.3.1 Step 1: PP 元信息全聚合

```python
# megatron_utils.py:990-1005
meta_info = []
for scan_vpp_idx in range(vpp_size):
    model = unwrap_model(actor_module[scan_vpp_idx])
    for idx, (name, _) in enumerate(model.named_parameters()):
        meta_info.append((pp_rank, scan_vpp_idx, idx, name))

# 所有 PP rank 交换参数目录
obj_spec_output = [None] * mpu.get_pipeline_model_parallel_world_size()
torch.distributed.all_gather_object(
    object_list=obj_spec_output, obj=meta_info,
    group=mpu.get_pipeline_model_parallel_group()
)
```

结果：每个 rank 获得 **全部 PP stage × VPP chunk** 的参数名清单。

#### 3.3.2 Step 2: PP 广播

对于每个参数，只有拥有它的 PP rank 持有实际张量：

```python
# megatron_utils.py:1030-1031
cur_name = broadcast_str_from_megatron_pp(cur_name)           # 广播参数名
broad_pp_tensor = broadcast_from_megatron_pp(cur_tensor)       # 广播参数张量
```

`broadcast_from_megatron_pp`（`megatron_utils.py:811-845`）：
1. 先 `all_gather_object` 收集张量 spec（shape, dtype, TP 属性）
2. 找到持有张量的唯一 rank
3. 其他 rank 分配空张量
4. 通过 PP 进程组 `torch.distributed.broadcast()` 广播

#### 3.3.3 Step 3: VPP 层号归一化

```python
# megatron_utils.py:1025
cur_name = normalize_model_name(name, cur_pp_rank, scan_vpp_idx, transformer_config)
```

调用 `get_transformer_layer_offset`（`megatron_utils.py:1101-1243`）将 `(pp_rank, vpp_stage)` 映射到全局层号。处理：
- 标准 PP 分割
- VPP 交错调度
- 不均匀 PP（首/末 stage 层数不同）
- Pipeline layout 配置

#### 3.3.4 Step 4: TP 全聚合 + 重组

```python
# megatron_utils.py:1074-1090
if tp_utils.is_tensor_parallel_param(broad_pp_tensor):
    infer_params = [torch.empty_like(broad_pp_tensor) for _ in range(all_gather_group_size)]
    torch.distributed.all_gather(infer_params, broad_pp_tensor,
                                 group=mpu.get_tensor_model_parallel_group())

    # 模型特定的 TP 重组（QKV 拆分、Gate-Up 拆分等）
    infer_params = default_tp_concat_fn(
        layer_name_mapping, cur_name, broad_pp_tensor,
        infer_params, model_config, hf_config, ...)
```

`default_tp_concat_fn`（`megatron_utils.py:874-951`）处理 Megatron 融合参数的拆分重组：

**QKV 融合权重**（`megatron_utils.py:895-928`）：

```
Megatron: linear_qkv.weight (TP=8)
  TP rank 0: [Q_shard0 | K_shard0 | V_shard0]
  TP rank 1: [Q_shard1 | K_shard1 | V_shard1]
  ...

           ↓ all_gather ↓

  All TP shards: [[Q0|K0|V0], [Q1|K1|V1], ..., [Q7|K7|V7]]

           ↓ default_tp_concat_fn ↓

  q_proj.weight: [Q0, Q1, ..., Q7] 拼接
  k_proj.weight: [K0, K1, ..., K7] 拼接（考虑 GQA group）
  v_proj.weight: [V0, V1, ..., V7] 拼接
```

**Gate-Up 融合权重**（`megatron_utils.py:930-944`）：

```
Megatron: linear_fc1.weight (TP=8)
  TP rank 0: [Gate_shard0 | Up_shard0]
  ...

           ↓ all_gather + split ↓

  gate_proj.weight: [Gate0, Gate1, ..., Gate7] 拼接
  up_proj.weight:   [Up0, Up1, ..., Up7] 拼接
```

#### 3.3.5 Step 5: EP 全聚合（MoE 模型）

```python
# megatron_utils.py:1038-1072
if ep_size > 1 and "experts" in cur_name:
    # 跨 EP 组收集所有专家
    all_params = [torch.empty_like(broad_pp_tensor) for _ in range(ep_size)]
    torch.distributed.all_gather(all_params, broad_pp_tensor, group=ep_group)

    if etp_size > 1:
        # 进一步跨 ETP 组收集
        etp_params = [torch.empty_like(tensor) for _ in range(etp_size)]
        torch.distributed.all_gather(etp_params, tensor, group=etp_group)
```

#### 3.3.6 Step 6: 参数名转换

```python
# megatron_utils.py:1096
converted_names, converted_params = weight_converter.convert_param(cur_name, infer_params)
yield from zip(converted_names, [param.detach() for param in converted_params])
```

`weight_converter`（`models/mcore/weight_converter.py`）完成最终的名称映射：

| Megatron 名称 | HuggingFace/vLLM 名称 |
|--------------|---------------------|
| `embedding.word_embeddings.weight` | `model.embed_tokens.weight` |
| `decoder.layers.N.self_attention.linear_qkv.weight` | → `q_proj`, `k_proj`, `v_proj`（已拆分） |
| `decoder.layers.N.self_attention.linear_proj.weight` | `model.layers.N.self_attn.o_proj.weight` |
| `decoder.layers.N.mlp.linear_fc1.weight` | → `gate_proj`, `up_proj`（已拆分） |
| `decoder.layers.N.mlp.linear_fc2.weight` | `model.layers.N.mlp.down_proj.weight` |
| `decoder.final_layernorm.weight` | `model.norm.weight` |
| `output_layer.weight` | `lm_head.weight` |

支持的模型转换器：Dense (Llama/Qwen), Mixtral, Qwen2-MoE, DeepSeek-V3, Qwen2.5-VL, Qwen3-MoE。

### 3.4 阶段 2 详解：NCCL 分桶广播

#### 3.4.1 拓扑建立

**`nccl_checkpoint_engine.py:153-165`**：

```python
# 星形拓扑: trainer rank 0 → 所有 rollout ranks
trainer_kwargs = {
    "rank": [0] + [-1] * (trainer_world_size - 1),  # 只有 rank 0 发送
    "world_size": [rollout_world_size + 1] * trainer_world_size,
    "master_metadata": [metadata[0]] * trainer_world_size,
}
rollout_kwargs = {
    "rank": list(range(1, rollout_world_size + 1)),  # ranks 1..N 接收
    "world_size": [rollout_world_size + 1] * rollout_world_size,
    "master_metadata": [metadata[0]] * rollout_world_size,
}
```

**关键设计**：只有 **trainer rank 0** 参与 NCCL 广播（其他 trainer rank 设为 -1，只消费 generator 不发送）。这是因为 `per_tensor_generator` 在所有 Megatron rank 上都运行（因为需要 PP/TP/EP 集合通信），但只有 rank 0 需要将结果发送给推理集群。

#### 3.4.2 双缓冲流水线

**`nccl_checkpoint_engine.py:224-294`**：

```python
async def send_weights(self, weights):
    send_buf, recv_buf = self.send_buf, self.recv_buf
    broadcast_op = None

    for name, weight in weights:
        if offset + weight.nbytes > self.bucket_size:
            torch.cuda.synchronize()

            # 等待上一个 bucket 广播完成
            if broadcast_op is not None:
                await broadcast_op.wait_for_complete()

            # 启动当前 bucket 的广播（在线程池中异步执行）
            broadcast_op = BroadcastOperation(
                rank=self.rank, group_name=self.group_name,
                bucket=send_buf, metadata={...}, ...
            )

            # ★ 交换 send 和 recv buffer — 实现流水线
            send_buf, recv_buf = recv_buf, send_buf
            offset = 0

        # 填充当前 buffer
        send_buf[offset:offset+weight.nbytes] = cp.asarray(
            weight.view(-1).view(torch.uint8)
        )
        offset += weight.nbytes
```

**流水线时序**：

```
时间 →
Buffer A: [填充 Bucket 1] [广播 Bucket 1] [填充 Bucket 3] [广播 Bucket 3]
Buffer B:                 [填充 Bucket 2] [广播 Bucket 2] [填充 Bucket 4]
                          ↑
                          overlap: 填充和广播并行
```

内存开销 = **2 × bucket_size**（默认 2 × 512MB = 1GB）。

#### 3.4.3 NCCL 广播 vs ZMQ 元数据

```python
# BroadcastOperation._run() — nccl_checkpoint_engine.py:74-84
def _run(self):
    if self.rank == 0:
        # 元数据: ZMQ PUB/SUB (轻量级)
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj(self.metadata)     # {name: shape/dtype/offset}
    else:
        self.socket.recv_string()
        self.metadata = self.socket.recv_pyobj()

    # 张量数据: NCCL broadcast (高带宽)
    collective.broadcast(self.bucket, src_rank=0, group_name=self.group_name)
```

**带宽分析**：对于 7B 模型（~14GB bf16 权重），bucket_size=512MB：
- 需要 ~28 个 bucket
- 每个 bucket 的 NCCL broadcast 延迟 ≈ 512MB / RDMA带宽(200Gbps) ≈ ~20ms
- 双缓冲流水线将填充和传输重叠
- **总延迟 ≈ 28 × 20ms = ~560ms**（不含 Megatron 内部聚合）

### 3.5 死锁预防机制

#### 3.5.1 进程组完全隔离

```
Megatron 内部进程组:
  ├── TP group (torch.distributed)
  ├── PP group (torch.distributed)
  ├── CP group (torch.distributed)
  └── EP group (torch.distributed)

Checkpoint Engine 进程组:
  └── NCCL group (ray.util.collective) ← 完全独立！
```

Checkpoint Engine 使用 `ray.util.collective` 创建 NCCL 组，与 Megatron 使用的 `torch.distributed` 进程组 **完全隔离**。

#### 3.5.2 顺序执行保证

`per_tensor_generator` 是一个 **Python 生成器**（惰性求值），它在被 Checkpoint Engine 消费时逐个产出参数。执行顺序是：

```
对于每个参数:
  1. PP broadcast (使用 Megatron PP group) → 完成
  2. TP all_gather (使用 Megatron TP group) → 完成
  3. EP all_gather (使用 Megatron EP group) → 完成
  4. yield (name, tensor) → Generator 暂停
  5. Checkpoint Engine 将 tensor 填入 bucket
  6. 如果 bucket 满 → NCCL broadcast (使用 Checkpoint Engine group)
  7. 回到 1, 处理下一个参数
```

**Megatron 集合通信和 Checkpoint Engine 集合通信永远不会同时进行** — 生成器的惰性特性保证了顺序执行。

#### 3.5.3 单发送者设计

只有 trainer rank 0 参与 Checkpoint Engine 的 NCCL 广播：

```python
# nccl_checkpoint_engine.py:230-236
assert self.rank <= 0, "Trainer workers other than rank 0 should not send weights."
if self.rank < 0:
    for name, weight in weights:
        pass  # 其他 rank 只消费 generator（触发 Megatron 集合通信）
    return
```

其他 trainer rank（rank < 0）仅运行 `per_tensor_generator`（参与 Megatron 内部的 PP/TP 聚合），但 **不参与 Checkpoint Engine 的 NCCL 通信**。

### 3.6 FSDP 路径对比

**`fsdp_workers.py:742-847`**：FSDP 路径远比 Megatron 简单。

```python
# FSDP 权重提取
params = self.actor_module_fsdp.state_dict()          # HF 名称，DTensor
per_tensor_param = (
    (name, param.full_tensor().to(torch.bfloat16)     # DTensor → 全张量
     if isinstance(param, DTensor) else param)
    for name, param in params.items()
)
```

| 维度 | Megatron 路径 | FSDP 路径 |
|------|-------------|---------|
| PP 处理 | `broadcast_from_megatron_pp` | 无 PP |
| TP 处理 | `all_gather` + `default_tp_concat_fn` | `.full_tensor()`（DTensor 透明处理）|
| EP 处理 | `all_gather` across EP group | N/A |
| 名称转换 | `weight_converter.convert_param` | 已是 HF 名称 |
| 融合参数 | 手动 QKV/GateUp 拆分 | 无融合参数 |
| 复杂度 | ~150 行核心逻辑 | ~20 行 |

### 3.7 其他 Checkpoint Engine 后端

#### 3.7.1 NIXL（环形 P2P）

**`nixl_checkpoint_engine.py`** — 环形拓扑，适合 RDMA 优化：

```
Trainer Rank 0 → Rollout 1 → Rollout 2 → ... → Rollout N
     (P2P RDMA)    (forward)    (forward)
```

- 每个 rank 有 `prev_agent`（源）和 `next_agent`（目标）
- 支持 UCX, UCCL, Mooncake 等多种后端
- 适合弹性扩缩容（无需全局进程组重建）

#### 3.7.2 Mooncake（TransferEngine P2P）

**`mooncake_checkpoint_engine.py`** — 基于 Mooncake TransferEngine 的 RDMA P2P：

- 显式内存注册（`engine.register_memory`）
- 双缓冲 + magic ACK 机制
- `transfer_sync_read()` / `transfer_sync_write()` 用于 P2P 传输

#### 3.7.3 Kimi（参数服务器 + 分片分发）

**`kimi_checkpoint_engine.py`** — 分片分发模式：

```python
# 按 rank 交错分片
for tensor_idx, (name, tensor) in enumerate(iterable):
    if tensor_idx % world_size == rank_id:  # 每个 rank 获取 1/N 的参数
        yield bucket
```

---

## 第四部分：性能特征与优化建议

### 4.1 端到端延迟拆解（7B 模型，NCCL 路径）

| 阶段 | 操作 | 估算延迟 |
|------|------|---------|
| Megatron PP broadcast | 全模型参数逐层广播 | ~100ms |
| Megatron TP all_gather | TP=8 → 全张量 | ~50ms |
| 名称转换 + QKV 拆分 | CPU 计算 | ~10ms |
| NCCL 分桶广播（28 buckets × 512MB） | 双缓冲流水线 | ~560ms |
| vLLM load_weights | GPU 权重覆写 | ~100ms |
| **总延迟** | | **~820ms** |

对比 naive (CUDA IPC) 路径：
- 同 GPU 直接 IPC：~200ms（包括 ZMQ 元数据 + IPC 传输）
- 因为跳过了 Megatron 聚合（协同定位时权重已在同一 GPU 上）

### 4.2 显存开销

| 后端 | 额外显存 | 说明 |
|------|---------|------|
| Naive (IPC) | 512MB | 单 bucket buffer |
| NCCL | 1GB | 双 buffer 流水线 |
| NIXL | 1GB | 双 buffer + RDMA 注册 |
| Mooncake | 1GB | 双 buffer + TransferEngine 注册 |

### 4.3 配置参数

| 参数 | 路径 | 默认值 | 影响 |
|------|------|--------|------|
| `checkpoint_engine.backend` | rollout config | `"naive"` | 选择同步后端 |
| `update_weights_bucket_megabytes` | rollout config | 512 | Bucket 大小(MB) |
| `partial_rollout` | async_training | `True` | 启用部分 rollout |
| `partial_rollout_resume` | async_training | `True` | 自动恢复中断的生成 |
| `free_cache_engine` | rollout config | `False` | 权重同步时是否释放 KV cache engine |
| `enable_prefix_caching` | vLLM config | - | 启用 prefix cache |
| `rebuild_group` | NCCL engine | `False` | 每次同步是否重建进程组 |

---

## 附录

### A. 核心文件索引

| 文件 | 核心类/函数 | 功能 |
|------|-----------|------|
| `verl/checkpoint_engine/base.py` | `CheckpointEngineManager`, `CheckpointEngine` | 权重同步编排基类 |
| `verl/checkpoint_engine/nccl_checkpoint_engine.py` | `NCCLCheckpointEngine`, `BroadcastOperation` | NCCL 分桶广播 |
| `verl/checkpoint_engine/nixl_checkpoint_engine.py` | `NIXLCheckpointEngine`, `NixlAgent` | NIXL 环形 P2P |
| `verl/checkpoint_engine/hccl_checkpoint_engine.py` | `HCCLCheckpointEngine` | 华为 HCCL 广播 |
| `verl/checkpoint_engine/mooncake_checkpoint_engine.py` | `MooncakeCheckpointEngine` | Mooncake P2P |
| `verl/checkpoint_engine/kimi_checkpoint_engine.py` | `KimiCheckpointEngine` | Kimi 参数服务器 |
| `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` | `BucketedWeightSender/Receiver` | CUDA IPC 分桶传输 |
| `verl/utils/megatron_utils.py` | `per_tensor_generator`, `broadcast_from_megatron_pp`, `default_tp_concat_fn`, `get_transformer_layer_offset` | Megatron 3D 并行权重采集 |
| `verl/models/mcore/weight_converter.py` | `McoreToHFWeightConverter*` | Megatron→HF 名称映射 |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | `sleep`, `wake_up`, `abort_all_requests`, `resume_generation` | vLLM 生命周期管理 |
| `verl/experimental/fully_async_policy/agent_loop/agent_loop.py` | `FullyAsyncLLMServerManager.generate` | Prompt 拼接续生成 |
| `verl/experimental/fully_async_policy/agent_loop/partial_single_turn_agent_loop.py` | `PartialSingleTurnAgentLoop.run` | 部分 rollout 恢复逻辑 |

### B. 完整 Abort→Resume 时序

```
1. CheckpointEngineManager.update_weights() 触发
2. ∀ replica: abort_all_requests()
   ├── vLLM: engine.pause_generation(clear_cache=True)
   │   ├── 阻止新请求
   │   ├── 中断进行中请求 → stop_reason="aborted"
   │   ├── 返回 partial TokenOutput (token_ids, log_probs)
   │   └── 清理 prefix cache
   └── 部分输出存入 cancel_queue（全异步模式）
3. build_process_group() — 创建 Checkpoint Engine NCCL 组
4. trainer.send_weights() ∥ rollout.receive_weights()
   ├── Megatron: per_tensor_generator → PP 广播 → TP 聚合 → 名称转换
   ├── NCCL: 双缓冲分桶广播
   └── vLLM: model.load_weights(received_weights)
5. finalize() — 销毁进程组
6. ∀ replica: resume_generation()
   ├── vLLM: engine.resume_generation()
   └── 全异步模式: 从 cancel_queue 取出部分输出 → prompt 拼接 → 续生成
```
