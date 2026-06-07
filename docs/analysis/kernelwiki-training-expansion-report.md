# KernelWiki Training Library Expansion Report: verl

**Library**: volcengine/verl  
**GitHub URL**: https://github.com/volcengine/verl  
**Analysis Path**: `/root/verl/.worktrees/source_code_analysis`  
**Date**: 2026-05-28  
**Library Type**: **Training Orchestration Framework** (zero CUDA kernel files; orchestrates kernels from upstream libraries; owns a small set of Triton kernels for training-critical fused operations)

---

## Dimension 1: Compute Kernels

### Kernel File Census

| Category | Count | Directories |
|----------|-------|-------------|
| CUDA/C++ (`.cu`/`.cuh`/`.ptx`) | **0** | N/A |
| Triton (`@triton.jit`) | **3 files, 11 kernels** | `verl/utils/kernel/`, `verl/utils/qat/` |
| C++ extension entry points | **0** | N/A |
| `torch.autograd.Function` subclasses | **11** | `verl/utils/`, `verl/models/`, `verl/trainer/` |

verl contains **zero CUDA/C++ source files**. All native GPU kernels are written in Triton. The vast majority of GPU compute is delegated to external libraries.

### Training-Specific Kernels

#### Native Triton Kernels

| Kernel | File | Proposed Tag | Description |
|--------|------|--------------|-------------|
| `efficient_entropy_kernel_general_mainloop` | `verl/utils/kernel/kernels.py:L187` | `fused-linear-cross-entropy` | Forward: GEMM + online softmax + log-prob extraction + entropy accumulation in one pass. Supports TMA (sm90+). |
| `efficient_entropy_triton_kernel_epilogue` | `verl/utils/kernel/kernels.py:L350` | `fused-linear-cross-entropy` | Forward epilogue: reduces partial per-split values to global. |
| `efficient_entropy_triton_kernel_epilogue_tp` | `verl/utils/kernel/kernels.py:L442` | `fused-linear-cross-entropy` | TP-aware forward epilogue with cross-rank max reduction. |
| `efficient_entropy_triton_epilogue_tp_update` | `verl/utils/kernel/kernels.py:L518` | `fused-linear-cross-entropy` | TP final update after cross-rank accumulation. |
| `efficient_entropy_backward_kernel_general_mainloop_MN` | `verl/utils/kernel/kernels.py:L770` | `fused-linear-cross-entropy-backward` | Fully fused backward: recomputes logits, computes d_hidden and d_weight without materializing d_logits. |
| `efficient_entropy_backward_kernel_d_hidden` | `verl/utils/kernel/kernels.py:L979` | `fused-linear-cross-entropy-backward` | Separate d_hidden computation, iterates over vocab dimension. |
| `efficient_entropy_backward_kernel_d_weight` | `verl/utils/kernel/kernels.py:L1106` | `fused-linear-cross-entropy-backward` | Separate d_weight computation, iterates over token dimension. |
| `efficient_entropy_backward_kernel_general_d_logits` | `verl/utils/kernel/kernels.py:L1226` | `fused-linear-cross-entropy-backward` | Materializes full d_logits then delegates to PyTorch matmul. |
| `efficient_entropy_backward_kernel_general_d_logits_split_N` | `verl/utils/kernel/kernels.py:L1387` | `fused-linear-cross-entropy-backward` | Split-N variant: iterates in vocab chunks to bound memory (default backward strategy). |
| `_blockwise_cast_to_fp8_kernel` | `verl/utils/kernel/fp8_kernel.py:L53` | `fp8-blockwise-quantize` | Single-pass: compute per-block absmax scale, clamp to FP8 range, cast. Block size `[128,128]`. |
| `_fp4_fake_quant_kernel` | `verl/utils/qat/linear.py:L42` | `fp4-fake-quantize` | NVFP4 (E2M1) fake-quantization with FP8 global scale for QAT forward/backward (STE). |

#### Training-Only Autograd Functions

| Function | File | Description |
|----------|------|-------------|
| `LinearCrossEntropy` | `verl/utils/kernel/linear_cross_entropy.py:L38` | Wraps efficient_entropy Triton suite. Avoids `[T × V]` logit materialization. |
| `FusedLinearForPPOFunction` | `verl/utils/experimental/torch_functional.py:L87` | Chunked fused linear+softmax+CE for PPO. Optionally uses `flash_attn.ops.triton.cross_entropy`. |
| `TiledMLP` | `verl/models/transformers/tiled_mlp.py:L86` | Tiles MLP forward/backward over sequence dimension to reduce peak activation memory. |
| `_VocabParallelEntropy` | `verl/utils/megatron/tensor_parallel.py:L109` | TP-sharded entropy computation with `@torch.compile(dynamic=True)`. |
| `_VocabParallelKLDivergence` | `verl/trainer/distillation/megatron/losses.py:L58` | KL divergence for distillation under Megatron TP. |
| `NPUGmmFunction` | `verl/models/transformers/npu_patch.py:L87` | Ascend NPU grouped-matrix-multiply for MoE expert dispatch. |
| `GroupCommitFunction` | `verl/utils/activation_offload.py:L111` | Synchronization fence for async activation CPU offloading prefetch. |
| `SeqAllToAll` / `Gather` | `verl/utils/ulysses.py:L169,L198` | Ulysses SP all-to-all with correct backward. |
| `IndexFirstAxis` / `IndexPutFirstAxis` | `verl/utils/npu_flash_attn_utils.py:L22,L56` | NPU attention unpad/pad (replaces flash_attn.bert_padding). |

### Inference-Shared Kernels (with Training Behavior Differences)

| Kernel | File Path | Training Behavior Difference |
|--------|-----------|------------------------------|
| `efficient_entropy_forward()` | `verl/utils/kernel/kernels.py` | Saves `maximum`, `accumulate`, `entropy_b` for backward via `ctx.save_for_backward` |
| `_blockwise_cast_to_fp8_kernel` | `verl/utils/kernel/fp8_kernel.py` | Used identically for weight sync to rollout engines; same kernel in both modes |
| `logprobs_from_logits` | `verl/utils/torch_functional.py` | Default `inplace_backward=True` optimizes backward; no backward triggered in inference |
| `flash_attn_func/varlen_func` | HF model patches in `verl/models/transformers/` | In training, gradient tape active enabling FlashAttn backward; in inference (HF rollout), no gradients |

### Fused Kernels

| Kernel | Operations Fused | Memory Savings |
|--------|-----------------|----------------|
| `efficient_entropy_kernel_general_mainloop` | GEMM + online max/softmax + log-prob + entropy | Eliminates `[T × V]` logit tensor (~largest intermediate) |
| `efficient_entropy_backward_kernel_general_mainloop_MN` | Forward recompute + d_logits + d_hidden GEMM + d_weight GEMM | No d_logits materialization |
| `_blockwise_cast_to_fp8_kernel` | Per-block absmax + scale computation + FP8 cast | Single-pass, no intermediate scale tensor |
| `_fp4_fake_quant_kernel` | Per-block scale + FP4 quantization simulation | Single-pass |
| `FusedLinearForPPOFunction` | Chunked linear + softmax + CE + entropy | Python-level chunking reduces peak memory |
| `_VocabParallelEntropy` | `(softmax * logits).sum()` fused via `torch.compile` | Compiler-generated fusion |

### Kernel Dependency Graph

verl is primarily a **kernel orchestration layer**. The dependency graph:

| Provider Library | Kernel Types Provided | Import Evidence |
|-----------------|----------------------|-----------------|
| **FlashAttention** (`flash_attn`) | fused SDPA (forward+backward), cross_entropy (Triton), bert_padding | `verl/models/transformers/llama.py`, `qwen2.py`, `verl/utils/torch_functional.py:L34`, `verl/utils/attention_utils.py:L30` |
| **Megatron-Core** (`megatron.core`) | TP linear layers (ColumnParallel/RowParallel), fused attention, pipeline schedule, activation recompute | `verl/workers/engine/megatron/transformer_impl.py:L23-25`, `verl/models/mcore/config_converter.py:L312` |
| **TransformerEngine** (`transformer_engine`) | FusedAdam optimizer, FP8 training infra, fused RMSNorm, fused attention | `verl/utils/megatron/dist_checkpointing.py:L65`, `verl/utils/megatron_utils.py:L711`, `verl/workers/config/engine.py:L486-490` |
| **vLLM** (`vllm`) | PagedAttention, fused RoPE, fused RMSNorm, CUTLASS GEMM, Marlin FP8 MoE, `scaled_fp8_quant` | `verl/workers/rollout/vllm_rollout/`, `verl/utils/vllm/vllm_fp8_utils.py` |
| **SGLang** (`sglang`) | Full inference kernel stack, weight sync utils | `verl/workers/rollout/sglang_rollout/sglang_rollout.py:L25-37` |
| **TensorRT-LLM** (`tensorrt_llm`) | TRT-optimized inference kernels, FP8 block quant | `verl/workers/rollout/trtllm_rollout/` |
| **Ascend NPU** (`torch_npu`) | `npu_rms_norm`, `npu_cross_entropy_loss`, `npu_gmm`, NPU flash attention | `verl/models/transformers/npu_patch.py`, `verl/utils/npu_flash_attn_utils.py` |
| **PyTorch** (`torch`) | Standard autograd, `torch.compile`, AdamW, AMP | Used throughout |
| **TorchAO** (`torchao`) | BF16 stochastic-rounding AdamW, Float8Linear | `verl/workers/config/optimizer.py:L236-238` |
| **BitsAndBytes** (`bitsandbytes`) | AdamW8bit optimizer | `verl/workers/config/optimizer.py:L240-242` |

### Proposed New kernel_types

| Tag | Representative File | Description |
|-----|---------------------|-------------|
| `fused-linear-cross-entropy` | `verl/utils/kernel/kernels.py` | GEMM + softmax + cross-entropy + entropy fused in single Triton kernel |
| `fused-linear-cross-entropy-backward` | `verl/utils/kernel/kernels.py` | Backward for fused linear+CE with 4 strategies (fully fused, split-N, etc.) |
| `fp8-blockwise-quantize` | `verl/utils/kernel/fp8_kernel.py` | Single-pass blockwise BF16→FP8 E4M3 quantization |
| `fp4-fake-quantize` | `verl/utils/qat/linear.py` | NVFP4 (E2M1) fake-quantization with STE for QAT |
| `tiled-mlp` | `verl/models/transformers/tiled_mlp.py` | Sequence-tiled MLP reducing activation memory by 1/num_shards |
| `vocab-parallel-entropy` | `verl/utils/megatron/tensor_parallel.py` | TP-sharded entropy via torch.compile |

---

## Dimension 2: Communication Kernels and Strategies

### Overview

verl implements a **two-layer communication architecture**:

1. **Upper layer (cross-worker)**: Ray object store + DataProto as the universal inter-worker data protocol
2. **Lower layer (intra-worker)**: NCCL/Gloo-based `torch.distributed` collectives for training parallelism

There are **no custom CUDA communication kernels**, no zero-SM Copy Engine collectives, no NVSHMEM/MSCCL++ usage. All GPU collective operations flow through PyTorch/NCCL.

### Collective Operations

| Operation | Algorithm(s) | SM Usage | File Path |
|-----------|-------------|----------|-----------|
| AllReduce | NCCL (Ring/Tree/NVLS via PyTorch) | Full SM | `verl/utils/megatron/tensor_parallel.py:L117-178`, `verl/utils/kernel/kernels.py:L705-737` |
| AllGather | NCCL via FSDP2 `fully_shard` | Full SM | `verl/utils/fsdp_utils.py:L559-599` (implicit), `verl/utils/ulysses.py:L165` |
| ReduceScatter | NCCL via FSDP2 backward | Full SM | `verl/utils/fsdp_utils.py` (implicit via FSDP2 runtime) |
| AllToAll | NCCL via `dist.all_to_all` | Full SM | `verl/utils/ulysses.py:L148` (Ulysses SP) |
| Broadcast | NCCL via `dist.broadcast` | Full SM | `verl/models/mcore/saver.py`, `verl/utils/fsdp_utils.py:L372-375`, `verl/checkpoint_engine/nccl_checkpoint_engine.py:L90` |
| Send/Recv | NCCL P2P via Megatron PP | Full SM | `verl/models/mcore/config_converter.py:L51`, `verl/utils/megatron_utils.py:L405` |
| Barrier | NCCL/Gloo | N/A | Widespread, critical transition points |

### Communication-Compute Overlap Patterns

| Pattern | Mechanism | Evidence |
|---------|-----------|----------|
| FSDP2 forward prefetch | AllGather of module `i+1` overlapped with forward of module `i` | `verl/utils/fsdp_utils.py:L588-599` (`set_modules_to_forward_prefetch`) |
| Megatron P2P pipeline overlap | P2P send/recv overlapped with micro-batch computation in 1F1B schedule | `verl/models/mcore/config_converter.py:L51` (`overlap_p2p_comm=True`) |
| Megatron 1F1B model chunk overlap | Fine-grained `TransformerModelChunkSchedulePlan` pipelining | `verl/models/mcore/model_forward_1f1b_overlap.py:L33-60` |
| Megatron gradient reduce overlap | Bucket AllReduce overlapped with backward (when `overlap_grad_reduce=True`) | `verl/utils/megatron_utils.py:L1356-1364` (default: disabled) |
| NCCL CheckpointEngine double-buffering | Ping-pong `send_buf`/`recv_buf` swap: one NCCL broadcast in flight while next bucket fills | `verl/checkpoint_engine/nccl_checkpoint_engine.py:L244-295` |
| FSDP2 `reshard_after_forward=False` | Avoids redundant AllGather during generation by keeping parameters unsharded | `verl/utils/fsdp_utils.py:L759` |

### Advanced Communication Features Checklist

- [ ] Symmetric memory support (NCCL 2.27+): **No** — no references found
- [ ] Device API support (NCCL 2.28+): LSA **No**, Multimem **No**, GIN **No**
- [ ] Copy Engine zero-SM collectives (NCCL 2.28+): **No** — no references found
- [ ] NCCL Inspector integration: **No** — only `NCCL_DEBUG=WARN` level set
- [ ] PyTorch SymmetricMemory: **No** — no references found
- [ ] Alternative backend support (MSCCL++, NVSHMEM): **No** — no references found

### Ray-Based Cross-Worker Communication

verl's single-controller architecture uses Ray as the cross-worker bus:

| Mechanism | File | Description |
|-----------|------|-------------|
| `ray.put` / `ray.get` | `verl/single_controller/ray/base.py:L810,862` | DataProto transfer via Ray object store |
| `DataProtoFuture` | `verl/protocol.py:L1174-1228` | Lazy dispatch: holds `list[ray.ObjectRef]`, defers `ray.get()` until data needed |
| `DataProto.chunk()` | `verl/protocol.py` | Splits batch across DP workers |
| `DataProto.concat()` | `verl/protocol.py` | Reassembles results from all DP workers |
| `all_gather_data_proto` | `verl/protocol.py:L1334-1345` | Intra-worker AllGather of DataProto across SP group |

### Weight Synchronization Transport Backends (Trainer → Rollout)

| Engine | File | Transport |
|--------|------|-----------|
| Colocated (naive) | `base.py:ColocatedCheckpointEngine` | In-process Python reference, no copy |
| NCCL | `nccl_checkpoint_engine.py` | `ray.util.collective.broadcast`, ZMQ PUB/SUB metadata |
| CUDA IPC + ZMQ | `bucketed_weight_transfer.py` | `torch.multiprocessing.reduce_tensor` for IPC handles; ZMQ REQ/REP; POSIX SHM fallback for NPU |
| NIXL | `nixl_checkpoint_engine.py` | NVIDIA NIXL (NVLink/RDMA) via `nixl._api.nixl_agent` |
| Mooncake | `mooncake_checkpoint_engine.py` | RDMA P2P via `mooncake.engine.TransferEngine` |

### Proposed New Communication kernel_types

| Tag | Representative File | Description |
|-----|---------------------|-------------|
| `ray-dataproto-dispatch` | `verl/single_controller/base/decorator.py` | Ray-based DataProto scatter/gather for cross-worker RL training |
| `ulysses-alltoall` | `verl/utils/ulysses.py` | DeepSpeed Ulysses sequence-parallel AllToAll with autograd backward |
| `cuda-ipc-weight-sync` | `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` | Zero-copy GPU→GPU weight transfer via CUDA IPC handles |
| `nixl-weight-sync` | `verl/checkpoint_engine/nixl_checkpoint_engine.py` | NVLink/RDMA weight transfer via NVIDIA NIXL |

### Proposed New Communication techniques

| Tag | Evidence | Description |
|-----|----------|-------------|
| `compute-comm-overlap` | `verl/utils/fsdp_utils.py:L588-599` | FSDP2 forward prefetch: AllGather of next module overlapped with current forward |
| `double-buffer-weight-sync` | `verl/checkpoint_engine/nccl_checkpoint_engine.py:L244-295` | Ping-pong buffer swap for overlapped NCCL broadcast of weight buckets |
| `hybrid-engine-sleep-wake` | `verl/workers/engine_workers.py:L667-738` | Time-shared GPU between training (FSDP) and inference (vLLM/SGLang) via sleep/resume |
| `dataproto-lazy-dispatch` | `verl/protocol.py:L1174-1228` | DataProtoFuture defers ray.get() enabling async pipelining between rollout and training |

---

## Dimension 3: Parallelism Strategies

### Supported Parallelism Dimensions

| Dimension | FSDP Engine | Megatron Engine | TorchTitan Engine | Automodel Engine | VeOmni Engine | Rollout (Inference) |
|-----------|-------------|-----------------|-------------------|------------------|---------------|---------------------|
| Data Parallel (FSDP/ZeRO) | ZeRO-2/3, HSDP | ZeRO-1 (distributed optimizer) + optional `megatron_fsdp` | ZeRO-3 + outer DDP | FSDP2/DDP | FSDP2 | DP copies |
| Tensor Parallel | No | Megatron ColumnParallel/RowParallel | PyTorch native TP | nemo_automodel TP | No | vLLM/SGLang/TRT-LLM TP |
| Pipeline Parallel | No | 1F1B, Interleaved (VPP) | Declared, NotImplementedError | Not supported (asserts) | No | vLLM/SGLang/TRT-LLM PP |
| Sequence Parallel (Ulysses) | Ulysses AllToAll | No (uses Megatron-style SP in TP group) | No | No | Ulysses AllToAll | No |
| Sequence Parallel (Megatron) | No | ReduceScatter/AllGather within TP | No | Megatron-style SP | No | No |
| Context Parallel | No | Static CP, Dynamic CP | TorchTitan CP | nemo_automodel CP | VeOmni CP | vLLM DCP |
| Expert Parallel | No | `expert_model_parallel_size` + `expert_tensor_parallel_size` | TorchTitan EP | `ep_size` + DeepEP | `expert_parallel_size` | vLLM/SGLang/TRT-LLM EP |

### DeviceMesh Topology

| Backend | Mesh Dim Names | Construction |
|---------|----------------|--------------|
| FSDP (pure) | `["fsdp"]` | `init_device_mesh(mesh_shape=(world_size,))` |
| FSDP (HSDP) | `["ddp", "fsdp"]` | `init_device_mesh(mesh_shape=(world_size//fsdp_size, fsdp_size))` |
| FSDP + Ulysses | `["dp", "sp"]` (separate mesh) | Additional 2D mesh for SP |
| Megatron | Via `mpu.parallel_state` (internal) | `mpu.initialize_model_parallel(tp, pp, cp, ep, ...)` |
| TorchTitan | `["dp_replicate", "fsdp", "cp", "tp", "pp", "ep"]` | `ParallelDims.build_mesh()` |
| Automodel | `["dp", "dp_replicate", "tp", "cp", "ep"]` + `moe_mesh` | `nemo_automodel.create_device_mesh` |
| Rollout | `["dp", "infer_tp", "infer_pp"]` | `engine_workers.py:L603-604` |

### Pipeline Scheduling Strategies

| Schedule | Description | Bubble Rate | Evidence |
|----------|-------------|-------------|----------|
| 1F1B | One forward one backward per micro-batch (Megatron default) | ~P/M | `verl/workers/engine/megatron/transformer_impl.py:L641-675` via `get_forward_backward_func()` |
| Interleaved (VPP) | Multiple virtual stages per device | Reduced vs 1F1B | `virtual_pipeline_model_parallel_size` config in `McoreEngineConfig:L161` |
| 1F1B with overlap | `TransformerModelChunkSchedulePlan` fine-grained pipelining | Further reduced | `verl/models/mcore/model_forward_1f1b_overlap.py:L33-60` |

### Hybrid Engine (Colocated Workers)

The central architectural innovation: `create_colocated_worker_cls` merges multiple worker classes (actor, rollout, reference) into a single Ray actor sharing GPUs.

| Component | File |
|-----------|------|
| Worker fusion | `verl/single_controller/ray/base.py:L986-1027` |
| Sleep/wake protocol | `verl/workers/engine_workers.py:L667-738` |
| Bucketed weight transfer (CUDA IPC) | `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` |
| `FusedWorker` variant | `verl/single_controller/ray/base.py:L1105` |

Sleep/wake lifecycle:
1. `rollout.sleep()` → frees KV cache + weights from GPU
2. Training runs (FSDP forward/backward/optimizer step)
3. `rollout.resume(tags=["weights"])` → reload weights into inference engine
4. Weight sync via CUDA IPC / NCCL / NIXL
5. `rollout.resume(tags=["kv_cache"])` → re-allocate KV cache
6. Rollout (inference) runs

### Worker Dispatch Modes

| Mode | Description | Usage |
|------|-------------|-------|
| `RANK_ZERO` | Only rank 0 executes | Config/metadata ops |
| `ONE_TO_ALL` | Same input to all workers | Checkpoint load, weight sync trigger |
| `DP_COMPUTE_PROTO` | DataProto split across DP, auto-padded | Standard training steps |
| `DP_COMPUTE_PROTO_WITH_FUNC` | DP split + arbitrary function dispatch | Custom functions |
| `DP_COMPUTE_METRIC` | DP split, metric aggregation | Loss/metric collection |
| `nd_compute_dataproto` | N-D mesh-aware dispatch | Multi-role workers (actor/ref with different DP sizes) |

### Proposed New Parallelism techniques

| Tag | Evidence | Description |
|-----|----------|-------------|
| `hybrid-engine` | `verl/single_controller/ray/base.py:L986` | Time-shared GPU between training and inference via colocated workers |
| `sleep-wake-memory-sharing` | `verl/workers/engine_workers.py:L667-738` | Sleep/resume protocol for GPU memory time-sharing |
| `ray-single-controller` | `verl/single_controller/` | CPU controller orchestrates distributed GPU workers without touching tensors |
| `dataproto-dispatch` | `verl/single_controller/base/decorator.py` | DataProto-based automatic data scatter/gather across DP dimensions |
| `ulysses-sequence-parallel` | `verl/utils/ulysses.py` | DeepSpeed Ulysses AllToAll-based sequence parallelism |
| `nd-dispatch` | `verl/single_controller/base/decorator.py:L300` | N-dimensional mesh-aware dispatch for multi-role workers |

---

## Dimension 4: Memory Management

### Memory Component Analysis

| Component | Storage Format | Sharding Strategy | Communication Kernel Triggered |
|-----------|---------------|-------------------|---------------------------------|
| Parameters | BF16 (default) | FSDP2 ZeRO-3 (`FULL_SHARD`) | AllGather before each layer forward |
| Gradients | BF16 (autocast) | FSDP2 ReduceScatter | ReduceScatter after each layer backward (reduced in FP32) |
| Optimizer States | FP32 (AdamW default), configurable to BF16/8-bit | FSDP2 ZeRO-3 | None (local optimizer step after ReduceScatter) |
| Activations | BF16 | Selective recomputation / CPU offload | None (stored, recomputed, or offloaded locally) |
| KV Cache (rollout) | FP16/BF16/FP8 | Per inference engine | None (local to inference engine) |

### Activation Checkpointing Strategies

| Strategy | Engine | Memory-Compute Tradeoff | Evidence |
|----------|--------|------------------------|----------|
| Full recompute (HF-style) | FSDP/HF | All layer activations recomputed, ~33% compute overhead | `verl/workers/engine/fsdp/transformer_impl.py:L303-304` |
| Uniform recompute | Megatron | Evenly spaced N layers recomputed | `verl/models/mcore/mtp_patch.py:L446-452` |
| Selective (MLA qkv_up_proj only) | Megatron (DeepSeek) | Only attention projection recomputed | `verl/models/mcore/patch.py:L221-227` |
| VeOmni combined | VeOmni | Checkpoint + activation offload simultaneous | `verl/workers/engine/veomni/transformer_impl.py:L323-327` |
| No recompute | All | Maximum speed, maximum memory | Default when `enable_gradient_checkpointing=False` |

### Activation CPU Offloading (Async Double-Buffer)

`verl/utils/activation_offload.py` — the most sophisticated memory optimization:

- `AsyncDoubleBufferGroupOffloadHandler` uses two CUDA streams (`d2h_stream`, `h2d_stream`) and a sliding window to overlap D2H/H2D with computation
- `GroupCommitFunction` acts as a synchronization fence: forward triggers bulk offload; backward triggers prefetch
- At most 2 activation groups reside in GPU memory at once
- Compatible with FSDP2 via `FSDPParameterFilter` (excludes FSDP-managed params from offloading)
- When combined with gradient checkpointing, replaces HF's `use_reentrant=False` with custom `use_reentrant=True` checkpoint wrapper

### CPU Offload Support Summary

| Target | Config Flag | Default | Notes |
|--------|-------------|---------|-------|
| Parameters | `param_offload: bool` | False | Moved to CPU after weight sync to rollout |
| Gradients | `grad_offload: bool` | False | Alongside param offload |
| Optimizer states | `optimizer_offload: bool` | False | Adam m/v on CPU |
| Activations | `enable_activation_offload: bool` | False | Async double-buffer pipeline |
| FSDP2 policy | `offload_policy: bool` or auto for ref | Auto for ref | `CPUOffloadPolicy(pin_memory=True)` |
| Reference model | Auto (FSDP1) | `CPUOffload(offload_params=True)` | Always on for reference model in FSDP1 |

### TiledMLP

`verl/models/transformers/tiled_mlp.py` — sequence-dimension tiling for MLP:
- Splits input along sequence dim into `num_shards` chunks (default 4)
- Each chunk processed independently; peak activation memory ∝ `1/num_shards`
- Backward re-runs forward per shard (activation recomputation) with `float32` gradient accumulation
- Supported for: Llama, Qwen2, Qwen2.5, Qwen3

### Gradient Accumulation

- Mini-batch split into micro-batches via `ppo_micro_batch_size_per_gpu`
- Megatron: `no_sync_func` defers AllReduce to final micro-batch (eliminates N-1 redundant all-reduce calls)
- Automodel: `prepare_for_grad_accumulation()` / `prepare_for_final_backward()` bracket micro-batch loop

### Hybrid Engine GPU Memory Management

The colocated setup requires careful memory budgeting:

1. `gpu_memory_utilization: float = 0.5` — fraction reserved for inference KV cache
2. `free_cache_engine: bool = True` — release KV cache between rollout rounds
3. `enable_sleep_mode: bool = True` — vLLM sleep/wake for memory release
4. `multi_stage_wake_up: bool = False` — staged weight/KV restoration
5. `aggressive_empty_cache()` — multi-retry gc + empty_cache + synchronize at transition points

---

## Dimension 5: Precision Management

### FP8 Scaling Strategies Found

| Strategy | Implementation | Granularity | Data Format | Scale Type | GPU Support | Evidence |
|----------|---------------|-------------|-------------|------------|-------------|----------|
| Software Block Scaling (Triton) | `scaled_fp8_blockwise_triton` | `[128, 128]` blocks | E4M3 | FP32 descale per block | Hopper + Blackwell | `verl/utils/kernel/fp8_kernel.py:L173` |
| Software Block Scaling (PyTorch) | `_scaled_fp8_blockwise_pytorch` | `[128, 128]` blocks | E4M3 | FP32 descale per block | All CUDA | `verl/utils/kernel/fp8_kernel.py:L227` |
| NVFP4 Fake Quantization | `STEFP4QuantTriton` / `_fp4_fake_quant_kernel` | 16-element groups | E2M1 + E4M3 scale | FP8 E4M3 global scale | Blackwell (native inference) | `verl/utils/qat/linear.py:L42` |
| Ascend MXFP8 | `torch_npu.npu_dynamic_mx_quant` | Hardware-defined | E4M3 | Hardware MX | Ascend NPU | `verl/utils/vllm/vllm_fp8_utils.py:L189-193` |

**Note**: verl does **not** use TransformerEngine's `DelayedScaling`, `Float8CurrentScaling`, or `Float8BlockScaling` APIs for training. FP8 quantization in verl is purely a **weight-sync-time operation** — weights are trained in BF16 and quantized to FP8 only when transferred to the inference engine.

### Precision per Training Component

| Component | Forward Pass | Backward Pass | Optimizer Step | Evidence |
|-----------|-------------|---------------|----------------|----------|
| Linear GEMM inputs | BF16 (autocast) | BF16 | N/A | `verl/workers/engine/fsdp/transformer_impl.py:L346-366` |
| Weights | BF16 (`param_dtype`) | N/A | BF16 params, FP32 master (optional) | Same file |
| Gradients | N/A | BF16 (autocast range) | N/A | Same file |
| Gradient AllReduce | N/A | **FP32** (`reduce_dtype=fp32`) | N/A | Same file |
| Adam momentum (m) | N/A | N/A | FP32 (default), configurable BF16 | `verl/workers/config/optimizer.py:L206-210` |
| Adam variance (v) | N/A | N/A | FP32 (default), configurable BF16 | Same file |
| FP16 loss scaler | `ShardedGradScaler` | `scaler.scale(loss)` | `scaler.unscale_() + step() + update()` | `transformer_impl.py:L362-364` |

### FP8 Communication Integration

- [ ] FP8 AllGather in FSDP2: **No** — weights communicated in BF16, cast to FP8 only at weight-sync boundary
- [ ] FP8 ReduceScatter: **No** — gradients reduced in FP32
- [ ] NVLink-SHARP FP8 in-switch reduction: **No**
- [ ] Estimated communication volume reduction vs BF16: **0%** (no FP8 communication in training path)

### FP8 Weight Sync Pipeline (Training → Inference)

1. `get_per_tensor_param()` gathers FSDP shards → BF16 full tensors (`transformer_impl.py:L838-846`)
2. `is_fp8_model()` detects if rollout engine uses FP8 quantization (`vllm_fp8_utils.py:L49`)
3. `scaled_fp8_blockwise()` quantizes BF16→FP8 E4M3 on-the-fly with `[128,128]` blocks
4. Sends `(fp8_tensor, descale_tensor)` pairs to inference engine
5. For Blackwell MoE: `requant_weight_ue8m0_inplace()` re-quantizes scales to UE8M0 format

### QAT (Quantization-Aware Training) — NVFP4

| Mode | Weight Quantization | Activation Quantization | Scale Management |
|------|--------------------|-----------------------|------------------|
| W4A16 | NVFP4 fake-quant (STE) | None (BF16) | Per-group (16 elements) global amax |
| W4A4 | NVFP4 fake-quant (STE) | NVFP4 fake-quant (STE) | Per-group scale + running EMA/static amax per `QATLinear` |

Three activation observer strategies for W4A4:
- `memoryless_minmax`: per-step amax, no history
- `static_minmax` (default): running maximum of historical amax
- `minmax`: EMA with α=0.01

Scale fusion: Q/K/V and Gate/Up projections use shared minimum global amax via `enable_qat_fuse()`.

### Proposed New Precision techniques

| Tag | Evidence | Description |
|-----|----------|-------------|
| `fp8-blockwise-weight-sync` | `verl/utils/kernel/fp8_kernel.py` | On-the-fly BF16→FP8 block-scaled quantization at trainer-to-rollout weight transfer |
| `nvfp4-qat` | `verl/utils/qat/linear.py` | NVFP4 quantization-aware training with Triton STE kernel |
| `nvfp4-scale-fusion` | `verl/utils/qat/core.py:L177` | QKV and GateUp scale fusion via minimum global amax |

---

## Dimension 6: Profiling and Observability

### Built-in Profiling Capabilities

| Feature | Supported | Integration Method | Evidence |
|---------|-----------|-------------------|----------|
| NVTX annotations for nsys | **Yes** | `nvtx.start_range/end_range` + decorator + context manager | `verl/utils/profiler/nvtx_profile.py` |
| Nsight Systems profiler | **Yes** | `torch.cuda.profiler.start/stop` + Ray `runtime_env={"nsight": opts}` | `nvtx_profile.py`, `main_ppo.py:L96-99` |
| PyTorch profiler (Chrome trace) | **Yes** | `torch.profiler.profile` + `export_chrome_trace` (.json.gz) | `verl/utils/profiler/torch_profile.py` |
| Memory snapshots | **Yes** | `torch.cuda.memory._record_memory_history` + `_dump_snapshot` (.pkl) | `verl/utils/profiler/profile.py:L210-287`, `verl/utils/memory_utils.py:L253-305` |
| Flight Recorder | **No** | — | Not implemented |
| Structured logging | **Yes** | JSONL via `FileLogger`, console via `LocalLogger` | `verl/utils/tracking.py` |
| MFU metrics | **Yes** | `FlopsCounter` with 10+ architecture-specific estimators, per-role | `verl/utils/flops_counter.py`, `engine_workers.py:L226` |
| Peak memory metrics | **Yes** | Per-step `max_memory_allocated/reserved` + CPU via psutil | `engine_workers.py:L210-212` |
| Timing breakdown | **Yes** | `marked_timer` / `simple_timer` with `reduce_timing` across ranks | `verl/utils/profiler/performance.py` |
| Gradient norm tracking | **Yes** | Per-step `actor/grad_norm`, `critic/grad_norm` | `ray_trainer.py`, `metric_utils.py` |
| Ascend NPU profiler | **Yes** | MSTX ranges + `torch_npu.profiler.tensorboard_trace_handler` | `verl/utils/profiler/mstx_profile.py` |
| Ascend precision debugger | **Yes** | `msprobe.pytorch.PrecisionDebugger` per stage | `verl/utils/profiler/precision_debugger_profile.py` |
| Rollout tracing (agentic) | **Yes** | Weave, MLflow, Trackio backends via `@rollout_trace_op` | `verl/utils/rollout_trace.py` |
| Ray timeline trace | **Yes** | `ray.timeline()` → Chrome JSON | `main_ppo.py:L104-108` |
| Prometheus (rollout servers) | **Yes** | vLLM/SGLang stats collection (port 9090 default) | `verl/workers/config/rollout.py:L127-138` |

### Experiment Tracking Backends (9 supported)

| Backend | Adapter | Notes |
|---------|---------|-------|
| WandB | Direct | `wandb.init(project, name, entity, config)` |
| TensorBoard | `_TensorboardAdapter` | `SummaryWriter`; dir from `TENSORBOARD_DIR` env var |
| MLflow | `_MlflowLoggingAdapter` | 3-retry logic; supports Azure ML / Databricks |
| SwanLab | Direct | Open-source WandB alternative |
| ClearML | `ClearMLLogger` | Task integration; DataFrame tables |
| Trackio | `_TrackioLoggingAdapter` | Trace logging support |
| vemlp_wandb | Direct | ByteDance VolcEngine ML Platform |
| Console | `LocalLogger` | Print to stdout |
| File | `FileLogger` | JSONL append |

### Performance Metrics Collected

| Metric | Formula | Reported As |
|--------|---------|-------------|
| MFU (actor) | estimated_tflops / promised_tflops / world_size | `perf/mfu/actor` |
| MFU (actor inference) | Same, for log-prob computation | `perf/mfu/actor_infer` |
| MFU (critic) | Same, for critic update | `perf/mfu/critic` |
| Throughput | total_tokens / (time × n_gpus) | `perf/throughput` (tokens/s/GPU) |
| Step time | Wall-clock per full PPO step | `perf/time_per_step` |
| Phase timing | Per-phase wall-clock (gen, ref, values, adv, update_actor, update_critic) | `timing_s/{phase}` |
| Per-token timing | Phase time / token count × 1000 | `timing_per_token_ms/{phase}` |
| Peak GPU memory | `max_memory_allocated()` | `perf/max_memory_allocated_gb` |
| Peak GPU reserved | `max_memory_reserved()` | `perf/max_memory_reserved_gb` |
| CPU memory | `psutil.virtual_memory().used` | `perf/cpu_memory_used_gb` |
| Gradient norm | Per-step L2 norm | `actor/grad_norm`, `critic/grad_norm` |
| Trainer idle ratio | gen_time / step_time (async mode) | `fully_async/trainer/idle_ratio` |
| Weight sync time | Wall-clock for param_sync | `timing_s/param_sync` |

### Profiling Configuration (Hydra)

```yaml
global_profiler:
  tool: "nsys"        # nsys | torch | torch_memory | npu | precision_debugger
  enable: true
  steps: [5, 10]      # which steps to profile
  save_path: "./profiler_output/"
  all_ranks: false
  ranks: [0]
  global_tool_config:
    nsys:
      controller_nsight_options: "--trace=cuda,nvtx,osrt"
      worker_nsight_options: "--trace=cuda,nvtx"
```

---

## Synthesis: Expansion Decision Summary

### S.1 Library Classification

| Property | Value |
|----------|-------|
| Library | verl |
| GitHub URL | volcengine/verl |
| Type | **training-orchestration** (with small native Triton kernel set for training-critical fused ops) |
| Contains CUDA Kernels | No (zero `.cu`/`.cuh` files) |
| Contains Triton Kernels | Yes (11 kernels in 3 files) |
| Primary Knowledge Dimensions | Dim 3 (Parallelism — 5 engines, 6D composable), Dim 4 (Memory — hybrid engine, activation offload), Dim 2 (Communication — two-layer Ray+NCCL) |
| Recommended KernelWiki Priority | **P1** (important orchestration framework, not a core kernel provider; high value for parallelism/memory/architecture patterns) |

### S.2 Proposed Tags (for controlled vocabulary YAML)

```yaml
kernel_types:
  # New from verl
  - fused-linear-cross-entropy       # Triton GEMM+softmax+CE+entropy forward fused kernel
  - fused-linear-cross-entropy-backward  # Backward with 4 strategies (fully fused, split-N, etc.)
  - fp8-blockwise-quantize           # Single-pass blockwise BF16→FP8 E4M3 quantization
  - fp4-fake-quantize                # NVFP4 (E2M1) STE fake-quantization for QAT
  - tiled-mlp                        # Sequence-tiled MLP with per-shard recomputation
  - vocab-parallel-entropy           # TP-sharded entropy via torch.compile
  - ulysses-alltoall                 # Ulysses SP AllToAll with autograd backward

techniques:
  # New from verl
  - hybrid-engine                    # Time-shared GPU between training (FSDP) and inference (vLLM/SGLang)
  - sleep-wake-memory-sharing        # Sleep/resume protocol for GPU memory time-sharing between engines
  - ray-single-controller            # CPU orchestrator dispatches work to GPU workers without touching tensors
  - dataproto-dispatch               # DataProto scatter/gather across DP dimensions with auto-padding
  - dataproto-lazy-dispatch          # DataProtoFuture defers ray.get() for async pipelining
  - async-activation-offload         # Double-buffer async CPU offload with d2h/h2d stream overlap
  - fp8-blockwise-weight-sync        # On-the-fly BF16→FP8 quantization at trainer→rollout boundary
  - nvfp4-qat                        # NVFP4 quantization-aware training with Triton STE kernel
  - nvfp4-scale-fusion               # QKV/GateUp scale fusion via minimum global amax
  - cuda-ipc-weight-transfer         # Zero-copy GPU→GPU weight sync via CUDA IPC handles
  - nixl-weight-transfer             # NVLink/RDMA weight transfer via NVIDIA NIXL
  - double-buffer-weight-sync        # Ping-pong buffer for overlapped NCCL broadcast
  - tiled-mlp-recompute              # Sequence-tiled MLP with per-shard activation recomputation
  - ulysses-sequence-parallel        # DeepSpeed Ulysses AllToAll-based sequence parallelism
  - nd-dispatch                      # N-dimensional mesh-aware dispatch for multi-role workers

hardware_features:
  # New from verl (hardware features the library explicitly supports or targets)
  - tma                              # TMA (Tensor Memory Accelerator) path in efficient_entropy kernel (sm90+)
  - ascend-npu                       # Ascend NPU support via torch_npu (RMSNorm, GMM, flash attention, MXFP8)

source_categories:
  # New (if needed)
  - rl-training-framework            # Reinforcement learning post-training framework for LLMs
```

### S.3 Wiki Page Topics

| # | Wiki Subdirectory | Proposed Page ID | Title | Source Evidence | Related Existing KernelWiki Pages |
|---|-------------------|------------------|-------|----------------|-----------------------------------|
| 1 | training/ | `training-fused-linear-cross-entropy` | Fused Linear Cross-Entropy: Eliminating the [T×V] Logit Bottleneck | `verl/utils/kernel/kernels.py`, `linear_cross_entropy.py` | `kernels/flash-attention`, `techniques/kernel-fusion` |
| 2 | training/ | `training-fp8-weight-sync` | FP8 Weight Synchronization in Hybrid Training-Inference Engines | `verl/utils/kernel/fp8_kernel.py`, `verl/utils/vllm/vllm_fp8_utils.py` | `techniques/quantization`, `hardware/fp8` |
| 3 | training/ | `training-nvfp4-qat` | NVFP4 Quantization-Aware Training for Blackwell Inference | `verl/utils/qat/linear.py`, `quantizer.py`, `core.py` | `techniques/quantization`, `hardware/nvfp4` |
| 4 | training/ | `training-activation-offload` | Async Double-Buffer Activation CPU Offloading | `verl/utils/activation_offload.py` | `techniques/memory-optimization` |
| 5 | training/ | `training-tiled-mlp` | Sequence-Tiled MLP for Memory-Efficient Training | `verl/models/transformers/tiled_mlp.py` | `techniques/activation-checkpointing` |
| 6 | parallelism/ | `parallel-hybrid-engine` | Hybrid Engine: Time-Sharing GPUs Between Training and Inference | `verl/single_controller/ray/base.py`, `verl/workers/engine_workers.py` | `patterns/gpu-sharing` |
| 7 | parallelism/ | `parallel-ray-single-controller` | Ray Single-Controller Architecture for RL Training | `verl/single_controller/`, `verl/protocol.py` | `patterns/distributed-orchestration` |
| 8 | parallelism/ | `parallel-dataproto-dispatch` | DataProto: Universal Data Interchange for Distributed RL | `verl/protocol.py`, `verl/single_controller/base/decorator.py` | `patterns/data-protocol` |
| 9 | communication/ | `comm-weight-sync-backends` | Weight Synchronization Backends: CUDA IPC, NCCL, NIXL, Mooncake | `verl/checkpoint_engine/`, `bucketed_weight_transfer.py` | `communication/nccl`, `communication/rdma` |
| 10 | communication/ | `comm-ulysses-sp` | DeepSpeed Ulysses Sequence Parallelism Implementation | `verl/utils/ulysses.py` | `parallelism/sequence-parallel` |

### S.4 Repository Mappings (slug -> org/repo)

```python
# For the PR candidate search script
"verl": "volcengine/verl",

# For the PR page generation script
"verl": "volcengine/verl",
```

### S.5 Keyword-to-Tag Mappings (for automated PR tagger)

```python
# keyword -> kernel_type tag
"linear_cross_entropy": "fused-linear-cross-entropy",
"LinearCrossEntropy": "fused-linear-cross-entropy",
"efficient_entropy": "fused-linear-cross-entropy",
"fused_linear_ppo": "fused-linear-cross-entropy",
"blockwise_cast_to_fp8": "fp8-blockwise-quantize",
"scaled_fp8_blockwise": "fp8-blockwise-quantize",
"fp4_fake_quant": "fp4-fake-quantize",
"STEFP4Quant": "fp4-fake-quantize",
"QATLinear": "fp4-fake-quantize",
"tiled_mlp": "tiled-mlp",
"TiledMLP": "tiled-mlp",
"VocabParallelEntropy": "vocab-parallel-entropy",
"SeqAllToAll": "ulysses-alltoall",

# keyword -> technique tag
"hybrid_engine": "hybrid-engine",
"colocated_worker": "hybrid-engine",
"create_colocated_worker_cls": "hybrid-engine",
"sleep_mode": "sleep-wake-memory-sharing",
"wake_up": "sleep-wake-memory-sharing",
"single_controller": "ray-single-controller",
"DataProto": "dataproto-dispatch",
"DataProtoFuture": "dataproto-lazy-dispatch",
"activation_offload": "async-activation-offload",
"AsyncDoubleBuffer": "async-activation-offload",
"GroupCommitFunction": "async-activation-offload",
"fp8_weight_sync": "fp8-blockwise-weight-sync",
"quant_weights": "fp8-blockwise-weight-sync",
"QAT": "nvfp4-qat",
"apply_qat": "nvfp4-qat",
"qat_fuse": "nvfp4-scale-fusion",
"BucketedWeightSender": "cuda-ipc-weight-transfer",
"BucketedWeightReceiver": "cuda-ipc-weight-transfer",
"NIXL": "nixl-weight-transfer",
"nixl_agent": "nixl-weight-transfer",
"ulysses": "ulysses-sequence-parallel",
"nd_compute": "nd-dispatch",

# keyword -> hardware_feature tag
"tl.make_tensor_descriptor": "tma",
"torch_npu": "ascend-npu",
"npu_rms_norm": "ascend-npu",
"npu_gmm": "ascend-npu",
"npu_flash_attn": "ascend-npu",
"mindspeed": "ascend-npu",
```

### S.6 PR Search Keywords (for candidate ledger)

```yaml
keywords_used:
  - LinearCrossEntropy
  - efficient_entropy
  - fused_linear
  - fp8_blockwise
  - scaled_fp8
  - QAT
  - QATLinear
  - fp4_fake_quant
  - tiled_mlp
  - TiledMLP
  - activation_offload
  - GroupCommitFunction
  - AsyncDoubleBuffer
  - hybrid_engine
  - colocated_worker
  - sleep_mode
  - wake_up
  - DataProto
  - DataProtoFuture
  - single_controller
  - BucketedWeight
  - NIXL
  - nixl
  - Mooncake
  - ulysses
  - SeqAllToAll
  - checkpoint_engine
  - nccl_checkpoint
  - fsdp2
  - fully_shard
  - nd_compute
  - VocabParallelEntropy
  - fp8_weight_sync
  - quant_weights
  - mxfp8
  - nvfp4
  - qat_fuse
  - FlopsCounter
  - mfu
  - rollout_trace
```

### S.7 Inclusion Policy Lane

```yaml
training-orchestration:
  description: |
    Captures PRs from verl that touch RL training orchestration,
    hybrid engine design, parallelism strategies, memory management,
    precision management, and native Triton kernels. Skips pure
    documentation, CI, and example-only changes.
  capture_criteria:
    - changed_paths_match:
        - "verl/utils/kernel/**"
        - "verl/utils/qat/**"
        - "verl/utils/fp8_utils.py"
        - "verl/utils/activation_offload.py"
        - "verl/utils/ulysses.py"
        - "verl/utils/fsdp_utils.py"
        - "verl/utils/megatron_utils.py"
        - "verl/utils/megatron/**"
        - "verl/utils/vllm/**"
        - "verl/utils/sglang/**"
        - "verl/utils/trtllm/**"
        - "verl/utils/modelopt/**"
        - "verl/workers/engine/**"
        - "verl/workers/rollout/**"
        - "verl/workers/sharding_manager/**"
        - "verl/workers/engine_workers.py"
        - "verl/workers/fsdp_workers.py"
        - "verl/workers/megatron_workers.py"
        - "verl/single_controller/**"
        - "verl/protocol.py"
        - "verl/checkpoint_engine/**"
        - "verl/trainer/ppo/core_algos.py"
        - "verl/models/mcore/**"
        - "verl/models/transformers/**"
    - title_contains_any:
        - kernel
        - fused
        - fp8
        - fp4
        - qat
        - quantiz
        - fsdp
        - megatron
        - parallelism
        - hybrid
        - colocated
        - sleep
        - wake
        - rollout
        - checkpoint_engine
        - weight_sync
        - ulysses
        - activation_offload
        - tiled_mlp
        - DataProto
        - mfu
        - profiler
        - nixl
        - mooncake
  skip_criteria:
    - changed_paths_match_only:
        - "docs/**"
        - "examples/**"
        - "tests/**"
        - "*.md"
        - ".github/**"
        - "docker/**"
    - pure_config_only: true
```

### S.8 Schema Extensions (if any)

New optional frontmatter fields for Wiki pages from this library:

```yaml
# Library type classification
scope: training-orchestration   # training-compute | communication-library | training-orchestration | full-stack

# For parallelism pages
parallelism_dimensions:
  - dp
  - tp
  - pp
  - cp
  - ep
  - sp

# For hybrid engine pages
memory_sharing_protocol: sleep-wake   # sleep-wake | static-partition

# For weight sync pages
transport_backend: cuda-ipc   # cuda-ipc | nccl | nixl | mooncake | colocated

# For kernel pages
kernel_language: triton   # cuda | triton | python
backward_strategy: fused   # fused | split-n | d-logits | pytorch-fallback
```

### S.9 Hardware Features Relevant to This Library's Training Workloads

| Hardware Feature | Inference Relevance | Training Relevance | Specific Impact on verl |
|-----------------|--------------------|--------------------|------------------------|
| NVLink 5 (1.8 TB/s) | Partial | Core | Doubles FSDP2 AllGather/ReduceScatter bandwidth for gradient sync |
| NVSwitch 4 (NVL72, 130 TB/s) | Partial | Core | Enables larger FSDP sharding groups with low-latency collectives |
| NVLink-SHARP FP8 | No | Core | Not yet utilized — potential 4x reduction in gradient AllReduce bandwidth |
| Symmetric Memory (9x latency reduction) | Partial | Core | Not yet utilized — potential for small-message AllReduce optimization |
| Copy Engine (zero-SM transfer) | Partial | Core | Not yet utilized — potential for FSDP2 AllGather overlap without SM consumption |
| TMA (sm90+) | Yes | Yes | Already utilized in `efficient_entropy` kernel forward path for accelerated tensor loads |
| MXFP8 hardware (Blackwell) | Yes | Partial | Ascend NPU MXFP8 supported; NVIDIA Blackwell MXFP8 training not yet implemented |
| NVFP4 (Blackwell) | Yes | Yes (QAT) | QAT W4A4/W4A16 training with inference export to Marlin/compressed-tensors on Blackwell |
| HBM3e (192 GB @ 8 TB/s) | Core | Core | Enables larger models per GPU; faster activation recomputation |
| GPU Direct RDMA | Important | Important | NIXL and Mooncake weight sync backends use RDMA for disaggregated setups |
| Ascend NPU (910B, 950DT) | Yes | Yes | Full training + inference support via torch_npu + MindSpeed engine |

### S.10 Upstream/Downstream Dependencies to Also Track

| Slug | GitHub URL | Relationship | Justification |
|------|-----------|-------------|---------------|
| `flash-attn` | Dao-AILab/flash-attention | kernel-provider | FlashAttention SDPA + Triton cross-entropy used in training forward/backward |
| `megatron-lm` | NVIDIA/Megatron-LM | kernel-provider + parallelism | Megatron-Core provides TP/PP/CP parallelism and TransformerLayer kernels |
| `transformer-engine` | NVIDIA/TransformerEngine | kernel-provider | FusedAdam, FP8 infra, fused attention/RMSNorm via Megatron backend |
| `vllm` | vllm-project/vllm | runtime-dependency | Primary inference engine for rollout generation |
| `sglang` | sgl-project/sglang | runtime-dependency | Alternative inference engine for rollout |
| `torchao` | pytorch/ao | runtime-dependency | BF16 stochastic-rounding optimizer, Float8Linear integration |
| `nccl` | NVIDIA/nccl | communication-backend | All GPU collectives flow through NCCL via PyTorch |
| `nixl` | ai-hypercomputer/nixl | communication-backend | NVLink/RDMA weight transfer for disaggregated training |
| `torchtitan` | pytorch/torchtitan | runtime-dependency | TorchTitan engine backend for composable parallelism |
| `nemo-automodel` | NVIDIA/NeMo | runtime-dependency | Automodel engine backend for FSDP2+TP+CP+EP |

---

## Dimension Emphasis Rationale

verl is classified as a **training orchestration framework**: it contains zero CUDA kernel files and a small set of Triton kernels focused on training-critical fused operations. Accordingly:

- **Dimension 1 (Compute Kernels)**: Light treatment for native kernels (only 11 Triton kernels), heavy emphasis on dependency graph showing which upstream libraries provide GPU compute.
- **Dimension 2 (Communication)**: Full treatment of the two-layer (Ray + NCCL) architecture, weight sync backends, and DataProto protocol. No custom communication kernels exist.
- **Dimension 3 (Parallelism)**: **Deep analysis** — this is verl's strongest dimension. Five pluggable training backends with composable up-to-6D parallelism, plus the hybrid engine innovation.
- **Dimension 4 (Memory)**: **Deep analysis** — the hybrid engine's sleep/wake protocol, async activation offloading, TiledMLP, and multi-backend CPU offload are core differentiators.
- **Dimension 5 (Precision)**: Moderate treatment. FP8 is used only at the weight-sync boundary (not in training compute). QAT (NVFP4) is the most novel precision feature.
- **Dimension 6 (Profiling)**: Full treatment. Comprehensive profiling infra with 9 experiment tracking backends, NVTX/nsys/torch.profiler integration, memory snapshots, MFU tracking, and rollout tracing.
