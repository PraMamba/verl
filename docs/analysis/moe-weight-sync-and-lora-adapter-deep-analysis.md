# verl MoE 权重同步与 LoRA Adapter 精准同步深度源码分析

> **分析范围**: MoE 模型 EP 权重聚合通信瓶颈、Checkpoint Engine 计算-通信掩盖、MoE LoRA Adapter 抽取与热加载
>
> **分析日期**: 2026-03-20
>
> **核心源码路径**:
> - `verl/utils/megatron_utils.py` — `per_tensor_generator` EP/ETP 专家权重聚合
> - `verl/models/mcore/weight_converter.py` — MoE 权重名映射（Mixtral/Qwen/DeepSeek-V3）
> - `verl/checkpoint_engine/nccl_checkpoint_engine.py` — NCCL 双缓冲分桶广播
> - `verl/workers/engine_workers.py` — 两阶段权重同步编排
> - `verl/workers/rollout/vllm_rollout/utils.py` — vLLM LoRA 热加载
> - `verl/utils/megatron_peft_utils.py` — Megatron→HF LoRA 名称转换
> - `verl/utils/megatron/router_replay_utils.py` — MoE 路由回放

---

## 第一部分：MoE 模型的 EP 权重聚合通信瓶颈

### 1.1 问题量化：200B+ 稀疏 MoE 的通信规模

以 **DeepSeek-V3**（671B 参数，256 experts，EP=8）为例：

| 参数 | 值 | 说明 |
|------|---|------|
| 总专家数 | 256 | 配置值 |
| EP 大小 | 8 | 每 rank 持有 32 experts |
| 每专家参数量 | ~30.3M | linear_fc1 [2816, 7168] + linear_fc2 [7168, 1408] |
| 每专家 BF16 大小 | ~60.6MB | 30.3M × 2 bytes |
| 每 EP rank 本地专家总量 | ~1.94GB | 32 × 60.6MB |
| MoE 层数 | 60 | Layer 1-60（Layer 0 为 Dense） |
| **EP AllGather 总通信量** | **~756GB / rank** | 60 layers × 32 experts × 7 × 60.6MB |
| Checkpoint Engine 广播总量 | ~400GB（全模型 BF16） | 包含 Dense + MoE + 非专家参数 |

### 1.2 EP 权重聚合的源码实现

**`megatron_utils.py:1038-1072`**：

```python
# EP 聚合：对 ".mlp.experts.linear_fc" 命名的参数触发
if ".mlp.experts.linear_fc" in cur_name and ep_size > 1:
    num_experts = weight_converter.mcore_config.num_moe_experts       # 256
    num_experts_per_rank = num_experts // ep_size                      # 32

    # Step 1: EP AllGather — 每个 EP rank 的本地专家 → 所有 rank
    infer_params = [torch.empty_like(broad_pp_tensor) for _ in range(ep_size)]
    torch.distributed.all_gather(infer_params, broad_pp_tensor, group=ep_group)  # ← 同步阻塞

    # Step 2: 本地→全局专家 ID 映射
    name_prefix, local_expert_id = cur_name.split(".weight")
    local_expert_id = int(local_expert_id)
    global_expert_ids = [num_experts_per_rank * ep_rank + local_expert_id
                         for ep_rank in range(ep_size)]

    # Step 3: 逐专家处理（ETP 聚合 + TP 拼接 + 名称转换）
    for name, param in zip(global_expert_names, infer_params):
        if etp_size > 1:
            # 嵌套 ETP AllGather
            etp_params = [torch.empty_like(param) for _ in range(etp_size)]
            torch.distributed.all_gather(etp_params, param, group=etp_group)  # ← 同步阻塞
            params = etp_params
        else:
            params = [param]

        # TP 拼接（gate/up 拆分等）
        merge_params = default_tp_concat_fn(...)
        # Megatron→HF 名称转换
        converted_names, converted_params = weight_converter.convert_param(name, merge_params)
        yield from zip(converted_names, [param.detach() for param in converted_params])
```

### 1.3 通信与计算的 Overlap 分析

**核心发现：EP AllGather 与 Checkpoint Engine NCCL Broadcast 之间 没有 任何 Overlap。**

执行时序图：

```
per_tensor_generator (Python Generator, 惰性求值):
  ┌─────────────────────────────────────────────────────┐
  │ for each expert parameter:                           │
  │   ① PP broadcast (Megatron PP group)    ← 同步阻塞   │
  │   ② EP all_gather (Megatron EP group)   ← 同步阻塞   │
  │   ③ [ETP all_gather (Megatron ETP group)] ← 同步阻塞 │
  │   ④ TP concat + name convert            ← CPU 计算   │
  │   ⑤ yield (hf_name, full_tensor)        ← Generator 暂停
  │                                                      │
  └───────────────────────────────────────┐              │
                                          ↓              │
  Checkpoint Engine (消费 Generator):                     │
  ┌─────────────────────────────────────────────────────┐│
  │   ⑥ 填入 bucket buffer                              ││
  │   ⑦ if bucket 满 → NCCL broadcast (async, 线程池)   ││
  │   ⑧ 交换双缓冲 → 回到 ①                             ││
  └─────────────────────────────────────────────────────┘│
```

**时间线**（无 Overlap）：

```
Time →
EP AllGather: [AG_1][AG_2]...[AG_32]     [AG_1]...[AG_32]     ...
                                ↓                     ↓
Fill Buffer:              [fill]              [fill]
                              ↓                   ↓
NCCL Broadcast:          [BC_1]              [BC_2]
                         ↑                   ↑
              只有这里有 overlap: fill 和 broadcast 通过双缓冲重叠
```

**双缓冲只在 "填充 bucket" 和 "NCCL 广播" 之间实现了 Overlap**（`nccl_checkpoint_engine.py:238-293`），但 **EP AllGather 和 NCCL Broadcast 之间是严格串行的**，因为它们通过 Python Generator 的 `yield` 同步点连接。

### 1.4 为什么没有 Overlap？

```python
# Generator 的惰性特性 = 隐式同步点
for name, weight in weights:          # ← 触发 generator 的 next()
    send_buf[offset:...] = weight     #    next() 会执行到下一个 yield
                                      #    yield 之前的 EP AllGather 必须完成
    if bucket_full:
        broadcast_op = BroadcastOperation(...)  # ← 异步广播
        await broadcast_op.wait_for_complete()  # ← 等待上一个广播
```

Generator 的 `yield` 是一个 **隐式同步屏障**：
- 在 `yield` 之前，Megatron 的 EP/TP/PP 集合通信 **必须完成**
- 在 `yield` 之后，Checkpoint Engine 才能消费数据
- 两者不可能同时执行

### 1.5 通信延迟估算

| 阶段 | 每次操作延迟 | 次数（DeepSeek-V3, EP=8） | 总延迟 |
|------|-----------|-----------|---------|
| PP Broadcast | ~1ms/参数 | ~3000（非专家参数） | ~3s |
| EP AllGather (per expert param) | ~2ms | 60 layers × 32 experts × 2 (fc1+fc2) = 3840 | ~7.7s |
| ETP AllGather (if ETP=2) | ~1ms | 同上 | ~3.8s |
| TP AllGather | ~1ms | 3840 | ~3.8s |
| NCCL Broadcast (双缓冲, ~782 buckets) | ~20ms/bucket | ~782 | ~8s（overlap 后） |
| **总端到端延迟** | | | **~22-26s** |

### 1.6 潜在优化方向

#### 1.6.1 异步 EP AllGather + NCCL Broadcast 流水线

当前架构无法实现的原因：Generator 是单线程惰性执行。改为以下架构可实现真正的 Overlap：

```
Producer Thread (Megatron collectives):
  EP AllGather → yield to Queue

Consumer Thread (Checkpoint Engine):
  从 Queue 取出 → 填 bucket → NCCL broadcast

两者通过 bounded Queue 解耦，实现并行
```

#### 1.6.2 增量专家同步

对于 LoRA 训练，仅同步修改过的专家参数（adapter），而非所有 256 experts 的全量参数。

#### 1.6.3 EP-aware Checkpoint Engine

当前 Checkpoint Engine 对 EP 完全不感知。如果让 Rollout 端的 vLLM 也使用 EP，则不需要在 Trainer 端 all_gather 所有专家 — 每个 EP rank 只需发送自己持有的专家给对应的 vLLM EP rank。

---

## 第二部分：MoE 权重名映射

### 2.1 支持的 MoE 架构

**`weight_converter.py`** 提供了完整的 Megatron→HF 名称转换：

| 模型 | 转换器类 | 行号 | 特殊处理 |
|------|---------|------|---------|
| Mixtral | `McoreToHFWeightConverterMixtral` | 422-443 | `block_sparse_moe.experts.{id}.w1/w2/w3` |
| Qwen2-MoE | `McoreToHFWeightConverterQwen2Moe` | 103-147 | Shared experts + gate_weight |
| Qwen3-MoE | `McoreToHFWeightConverterQwen3Moe` | 446-479 | 无 shared experts |
| DeepSeek-V3 | `McoreToHFWeightConverterDpskv3` | 269-419 | MLA 注意力 + 256 experts + MTP |

### 2.2 映射示例（DeepSeek-V3）

```
Megatron-Core 名称                                    → HuggingFace/vLLM 名称
─────────────────────────────────────────────────────────────────────────────
decoder.layers.1.mlp.router.weight                    → model.layers.1.mlp.gate.weight
decoder.layers.1.mlp.router.expert_bias               → model.layers.1.mlp.gate.e_score_correction_bias
decoder.layers.1.mlp.experts.linear_fc1.weight0       → model.layers.1.mlp.experts.0.gate_proj.weight
                                                        model.layers.1.mlp.experts.0.up_proj.weight (拆分)
decoder.layers.1.mlp.experts.linear_fc2.weight0       → model.layers.1.mlp.experts.0.down_proj.weight
decoder.layers.1.mlp.shared_experts.linear_fc1.weight → model.layers.1.mlp.shared_experts.gate_proj.weight
                                                        model.layers.1.mlp.shared_experts.up_proj.weight
```

### 2.3 Routing Replay — MoE 路由回放

**`router_replay_utils.py`** 实现了推理时记录专家路由决策、训练时回放的机制：

```
推理阶段 (SGLang):
  Token → Router → TopK experts → 记录 routed_experts tensor
  routed_experts shape: [seq_len, num_moe_layers, top_k]

训练阶段 (Megatron):
  设置 RouterReplay.target_topk_idx → 跳过 router 计算 → 直接使用记录的路由
```

三种模式：
- `disabled`：不使用路由回放（默认）
- `R2`：记录模式
- `R3`：完整回放（前向 + 反向）

---

## 第三部分：MoE LoRA 的精准抽取与热加载

### 3.1 LoRA 两阶段同步架构

verl 实现了 **Base-Once, Adapter-Always** 的两阶段同步模式：

```
┌─────────────────────────────────────────────────────────────┐
│ 第一步（仅首次 / sleep_level=2 后）:                         │
│   sync_base: 全量权重 → vLLM                                │
│   (含 .base_layer 后缀的参数名，vLLM 识别为 LoRA 基座)        │
│                                                             │
│ 后续每步:                                                   │
│   sync_adapter: 仅 LoRA A/B 矩阵 → vLLM add_lora()         │
│   (通过 TensorLoRARequest 内存热加载，不触碰 Base 权重)        │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Adapter 抽取路径

#### 3.2.1 FSDP 路径

**`fsdp/transformer_impl.py:727-800`**：

```python
def get_per_tensor_param(self, layered_summon=False, base_sync_done=False):
    if peft_config and not merge_lora:
        if base_sync_done:
            # ★ Adapter-only: 仅抽取 LoRA 参数
            params = collect_lora_params(
                self.actor_module_fsdp,
                layered_summon=layered_summon,
                base_sync_done=True
            )
        else:
            # Base + LoRA 结构（首次同步）
            params = collect_lora_params(..., base_sync_done=False)
            # 添加 .base_layer 后缀让 vLLM 识别
```

**`layered_summon` 内存优化**（`fsdp_utils.py:571-610`）：逐层 summon FSDP 参数，避免一次性物化全部参数到 GPU：

```python
def layered_summon_lora_params(fsdp_module):
    for name, submodule in __prefix_submodules(fsdp_module, prefix):
        with FSDP.summon_full_params(submodule):
            adapter_params = get_peft_model_state_dict(submodule)
            for k, v in adapter_params.items():
                lora_params[prefix + k] = v.cpu().clone()
        get_torch_device().empty_cache()  # 释放每层后的 GPU 内存
```

#### 3.2.2 Megatron-Bridge 路径

**`megatron/transformer_impl.py:612-630`**：

```python
def get_per_tensor_param(self, base_sync_done=False, **kwargs):
    adapter_only = base_sync_done and non_merge_lora_sync

    if adapter_only:
        # ★ 仅加载 adapter 参数到 GPU（跳过 frozen base）
        load_megatron_model_to_gpu(self.module, load_grad=False,
                                   load_frozen_params=not adapter_only)  # False!
        peft_config = build_peft_config_for_vllm(self.model_config.lora)
        per_tensor_param = self.bridge.export_adapter_weights(self.module)
    else:
        per_tensor_param = self.bridge.export_hf_weights(self.module)
```

**关键优化**：`load_frozen_params=False` 跳过了 Base 模型的 CPU→GPU 传输，仅加载 Adapter 参数。对 200B+ MoE 模型，这节省了数十秒的加载时间。

### 3.3 vLLM 热加载机制

**`rollout/vllm_rollout/utils.py:208-231`**：

```python
def _update_weights(self, weights, peft_config, base_sync_done):
    if peft_config and base_sync_done:
        # ★ LoRA 热加载：构建 TensorLoRARequest，直接内存加载
        weights = dict(weights)
        lora_request = TensorLoRARequest(
            lora_name=VLLM_LORA_NAME,          # "123"
            lora_int_id=VLLM_LORA_INT_ID,      # 123
            lora_path=VLLM_LORA_PATH,          # 假路径
            peft_config=peft_config,            # LoRA 配置
            lora_tensors=weights,               # 内存中的张量
        )
        self.add_lora(lora_request)  # vLLM 原生 API
    else:
        self.model_runner.model.load_weights(weights)  # 标准权重加载
```

**VLLMHijack**（`vllm/utils.py:36-123`）：Monkey-patch vLLM 的 `_load_adapter` 方法，使其支持从 `TensorLoRARequest.lora_tensors` 直接加载（绕过文件系统）：

```python
# 原始 vLLM: LoRAModel.from_local_checkpoint(path)  ← 需要文件
# verl 劫持: LoRAModel.from_lora_tensors(tensors)    ← 直接从内存
```

### 3.4 MoE LoRA 的具体支持

#### 3.4.1 LoRA 目标模块

**`megatron_peft.py:50-61`** — 默认 LoRA 目标包含专家层：

```python
LoRA(
    target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
    #                                               ↑ gate+up     ↑ down
    #                                               专家层 MLP 的两个线性层
    dim=rank, alpha=alpha, dropout=dropout, ...
)
```

由于 `linear_fc1` 和 `linear_fc2` 在 MoE 模型中是 **每个专家独立的模块**，Megatron-Bridge 的 LoRA 自然是 **per-expert** 的 — 每个专家拥有自己的 LoRA A/B 矩阵。

**名称映射**（`megatron_peft_utils.py:23-44`）：

```python
MEGATRON_TO_HF_MODULES = {
    "linear_fc1": ["gate_proj", "up_proj"],   # 专家 MLP 上投影
    "linear_fc2": ["down_proj"],               # 专家 MLP 下投影
    "router": ["gate"],                        # MoE 路由器
    "linear_qkv": ["q_proj", "k_proj", "v_proj"],
    "linear_proj": ["o_proj"],
}
```

#### 3.4.2 CanonicalLoRA — 更细粒度的专家 LoRA

**`megatron_peft.py:78-99`** 提供了 CanonicalLoRA 变体：

```python
CanonicalLoRA(
    target_modules=["linear_q", "linear_k", "linear_v",
                     "linear_fc1_up", "linear_fc1_gate", "linear_fc2"],
    #                ↑ 拆分 QKV     ↑ 拆分 gate 和 up 投影
)
```

允许对每个专家的 gate 和 up 投影 **分别** 设置 LoRA，而不是对融合的 `linear_fc1` 统一设置。

### 3.5 MoE LoRA 数据量估算

| 配置 | 值 |
|------|---|
| 模型 | Qwen3-MoE-30B-A3B（128 experts, 60 MoE layers） |
| LoRA rank | 16 |
| 每专家 LoRA 参数量 (fc1+fc2) | 2 × 2 × (hidden × rank) ≈ 2 × 2 × (4096 × 16) = 262K |
| 每专家 LoRA BF16 大小 | ~524KB |
| 所有专家 LoRA (128 experts × 60 layers) | ~3.9GB |
| **对比全量权重同步** | ~60GB |
| **Adapter-only 加速比** | **~15.4×** |

| 配置 | 值 |
|------|---|
| 模型 | DeepSeek-V3（256 experts, 60 MoE layers） |
| LoRA rank | 16 |
| 每专家 LoRA BF16 大小 | ~1.2MB (较大 hidden dim) |
| 所有专家 LoRA (256 × 60) | ~18.4GB |
| **对比全量权重同步** | ~400GB |
| **Adapter-only 加速比** | **~21.7×** |

### 3.6 能否实现"亚毫秒级"精准抽取？

**直接回答：不能。但已足够快。**

| 阶段 | 耗时估算（DeepSeek-V3, LoRA rank=16） | 瓶颈 |
|------|------|------|
| Megatron Adapter 抽取 | ~1-2s | EP AllGather 仍需执行（adapter 也是分布在 EP rank 上的） |
| ZMQ IPC 传输 (18.4GB) | ~2-4s | 受限于 PCIe/NVLink 带宽 |
| vLLM add_lora() | ~0.5-1s | 内存加载 + LoRA 合并 |
| **总延迟** | **~3.5-7s** | |

虽然无法达到"亚毫秒级"，但相比全量同步（22-26s+），Adapter-only 模式 **节省 60-80% 的同步时间**。

关键瓶颈在于：即使是 Adapter-only，EP AllGather 仍需要执行（因为 per-expert LoRA 参数分布在不同 EP rank 上）。但通信量从 ~756GB 降至 ~18GB，降低了约 42 倍。

---

## 第四部分：权重同步编排的代码细节

### 4.1 `engine_workers.py` 两阶段编排

**`engine_workers.py:621-683`**：

```python
async def update_weights(self, global_steps=None):
    # Phase 1: Adapter 同步（每步执行）
    per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
        layered_summon=self.layered_summon,
        base_sync_done=True            # ← 请求 adapter-only
    )
    await self.rollout.update_weights(
        per_tensor_param, peft_config=peft_config,
        base_sync_done=True, ...
    )

    # Phase 2: Base 同步（仅首次 或 sleep_level=2 后）
    do_lora_base_sync = False
    if not self.peft_merge and peft_config is not None:
        self.rollout.sleep_level = 1   # ← 强制 sleep_level=1（保留 base 权重）
        do_lora_base_sync = (not self.base_sync_done) or (
            self.rollout.sleep_level != 1 and ...  # ← 永远为 False（刚设为 1）
        )

    if do_lora_base_sync:
        per_tensor_base_params, _ = self.actor.engine.get_per_tensor_param(
            base_sync_done=False       # ← 请求 base 权重
        )
        await self.rollout.update_weights(
            per_tensor_base_params, peft_config=peft_config,
            base_sync_done=False
        )
```

### 4.2 Sleep Level 与 LoRA 的交互

| Sleep Level | vLLM 行为 | LoRA 影响 |
|------------|----------|----------|
| Level 1 | 释放 KV cache，**保留 base 权重** | 后续只需同步 adapter |
| Level 2 | 释放 KV cache + base 权重 | 后续需要同步 base + adapter |

verl 在检测到 LoRA 时强制使用 `sleep_level=1`（`engine_workers.py:659`），避免了每步全量同步的开销。代价是 GPU 显存中始终保留 base 模型权重。

### 4.3 vLLM 侧的 LoRA 加载/卸载

```python
# utils.py:164-166 — 加载新 adapter 前先移除旧的
if peft_config and base_sync_done:
    self.remove_lora(VLLM_LORA_INT_ID)     # 移除旧 adapter

# utils.py:208-218 — 创建 TensorLoRARequest 并加载
lora_request = TensorLoRARequest(
    lora_name="123", lora_int_id=123,
    lora_path="simon_lora_path",           # 假路径（内存加载不需要真实路径）
    peft_config=peft_config,
    lora_tensors=weights,                  # 内存中的 LoRA 张量
)
self.add_lora(lora_request)                # vLLM 原生 LoRA API
```

### 4.4 MoE 模型权重加载器 Patch

**`utils.py:152-153`**：

```python
# patch weight loader to support MoE model
patch_vllm_moe_model_weight_loader(self.model_runner.model)
```

在每次异步 IPC 权重同步后重新 patch MoE 权重加载器，确保 vLLM 能正确处理 MoE 专家权重的分片和加载。

---

## 第五部分：综合评估

### 5.1 深度提问 6 回答：EP AllGather 的通信掩盖

**verl 的 Checkpoint Engine 对 EP AllGather 和 NCCL Broadcast 之间 没有 实现 Overlap。**

- **有 Overlap 的部分**：NCCL 双缓冲广播（填充 bucket N 时同时广播 bucket N-1）
- **没有 Overlap 的部分**：Megatron EP/TP/PP AllGather 与 Checkpoint Engine NCCL Broadcast
- **根因**：Python Generator 的 `yield` 是隐式同步点，生产者（Megatron 集合通信）和消费者（Checkpoint Engine）串行执行
- **影响**：对 DeepSeek-V3 规模的模型，端到端权重同步需 ~22-26 秒
- **优化空间**：引入 bounded Queue 解耦生产者/消费者，实现真正的流水线；或实现 EP-aware Checkpoint Engine 避免 Trainer 端全量聚合

### 5.2 深度提问 7 回答：MoE LoRA 的精准同步

**verl 能够做到将 per-expert LoRA Adapter 精准抽取、传输，并在 vLLM 端动态热加载，无需触碰 Base 模型。**

| 能力 | 支持情况 | 说明 |
|------|---------|------|
| Per-expert LoRA | ✅ 支持 | Megatron-Bridge 对每个 expert 的 fc1/fc2 独立应用 LoRA |
| Adapter-only 抽取 | ✅ 支持 | `export_adapter_weights` + `load_frozen_params=False` |
| 跳过 Base 模型 | ✅ 支持 | `sleep_level=1` 保留 base + `base_sync_done=True` 只同步 adapter |
| vLLM 热加载 | ✅ 支持 | `TensorLoRARequest` + `VLLMHijack` 直接从内存加载 |
| 亚毫秒级速度 | ❌ 不能 | EP AllGather + IPC 传输仍需 3-7 秒 |
| 实际加速比 | **15-22×** | 相比全量同步（全部 base + experts） |

**限制**：
1. EP AllGather 仍是必需的（per-expert adapter 分布在不同 EP rank 上）
2. 当前实现使用单一 LoRA adapter ID（`VLLM_LORA_INT_ID=123`），不支持多 adapter 并发
3. `bridge.export_adapter_weights` 是外部依赖（megatron-bridge），verl 内部未定义具体实现

---

## 附录

### A. 核心文件索引

| 文件 | 核心内容 | 行号 |
|------|---------|------|
| `verl/utils/megatron_utils.py` | EP/ETP AllGather, per_tensor_generator | 956-1098 |
| `verl/models/mcore/weight_converter.py` | MoE 权重名映射 (Mixtral/Qwen/DeepSeek) | 103-479 |
| `verl/checkpoint_engine/nccl_checkpoint_engine.py` | 双缓冲 NCCL 广播 | 43-363 |
| `verl/workers/engine_workers.py` | 两阶段同步编排 | 621-683 |
| `verl/workers/rollout/vllm_rollout/utils.py` | vLLM LoRA 热加载 | 155-231 |
| `verl/utils/megatron_peft_utils.py` | Megatron→HF LoRA 名称转换 | 23-360 |
| `verl/utils/fsdp_utils.py` | FSDP LoRA 抽取 + layered_summon | 571-893 |
| `verl/utils/vllm/utils.py` | TensorLoRARequest + VLLMHijack | 31-123 |
| `verl/workers/config/megatron_peft.py` | LoRA/DoRA/CanonicalLoRA 配置 | 17-120 |
| `verl/utils/megatron/router_replay_utils.py` | MoE 路由回放 | 175-509 |
| `verl/utils/megatron/router_replay_patch.py` | Router 猴子补丁 | 44-442 |

### B. MoE LoRA 训练示例脚本

```bash
# Qwen3-MoE-30B LoRA 训练（来自 examples/）
examples/grpo_trainer/run_qwen3moe-30b_megatron_lora.sh:
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP
  actor_rollout_ref.rollout.expert_parallel_size=${gen_ep}
```

### C. 通信量对比总结

| 同步模式 | DeepSeek-V3 数据量 | 估算延迟 |
|---------|-------------------|---------|
| 全量同步（Base + 全部 Expert） | ~400GB | 22-26s |
| Base 同步 + Adapter 同步（首次） | ~400GB + ~18GB | 22-30s |
| Adapter-only 同步（后续每步） | ~18GB | 3-7s |
| Dense-only LoRA（无 MoE） | ~200MB | <1s |
