# PR Review: Change Type Detection Reference

This file contains the change type detection tables for PR review. Referenced by:
`.claude/commands/review-pr.md`

______________________________________________________________________

## CRITICAL Level (Must use Opus)

| Change Type         | File Path Pattern                                              | Code Pattern                                              |
| ------------------- | -------------------------------------------------------------- | --------------------------------------------------------- |
| **FSDP_CORE**       | `verl/workers/fsdp_workers.py`, `verl/utils/fsdp_utils/`      | `FSDP`, `FullyShardedDataParallel`, `fully_shard`         |
| **MEGATRON_CORE**   | `verl/models/mcore/`, `verl/utils/megatron_utils/`             | `MegatronEngine`, `megatron`                              |
| **DCP_CHECKPOINT**  | `verl/checkpoint_engine/`                                      | `DCP`, `DistributedCheckpoint`, `dcp.save`, `dcp.load`    |
| **RAY_CONTROLLER**  | `verl/single_controller/`, `verl/trainer/ppo/ray_trainer.py`   | `RayPPOTrainer`, `@register`, `Dispatch`                  |

## HIGH Level (Recommend Opus)

| Change Type           | File Path Pattern                              | Code Pattern                                                                     |
| --------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------- |
| **DISTRIBUTED_COMM**  | -                                              | `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `dist.`              |
| **DTENSOR**           | -                                              | `DTensor`, `DeviceMesh`, `Shard(`, `Replicate(`, `Partial(`, `distribute_tensor` |
| **TENSOR_PARALLEL**   | -                                              | `ColwiseParallel`, `RowwiseParallel`, `parallelize_module`                       |
| **SEQUENCE_PARALLEL** | -                                              | `SequenceParallel`, `context_parallel`, `cp_size`                                |
| **ASYNC_CONCURRENT**  | -                                              | `async def`, `await`, `asyncio`, `threading.Lock`                                |
| **TRAINER_CORE**      | `verl/trainer/ppo/`, `verl/trainer/grpo/`      | `RayPPOTrainer`, `train_step`, `compute_ppo_loss`                                |
| **WORKER_BASE**       | `verl/single_controller/base/`                 | `Worker`, `@register`, `Dispatch`, `dispatch_mode`                               |
| **DATAPROTO**         | `verl/protocol.py`                             | `DataProto`, `DataProtoConfig`, `TensorDict`                                     |

## MEDIUM Level (Use Sonnet)

| Change Type             | File Path Pattern                                                                   | Code Pattern                                                             |
| ----------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **TENSOR_OPS**          | -                                                                                   | `.view(`, `.reshape(`, `dtype=`, `.detach()`, `no_grad`, `.contiguous()` |
| **NUMERICAL**           | -                                                                                   | `log(`, `softmax`, `cross_entropy`, `eps=`, `.clamp(`, `nan`, `inf`      |
| **ROLLOUT_VLLM**        | `verl/workers/rollout/vllm*.py`, `verl/utils/vllm_utils*`                           | `vLLM`, `SamplingParams`, `PagedAttention`                               |
| **ROLLOUT_SGLANG**      | `verl/workers/rollout/sglang*.py`, `verl/utils/sglang_utils*`                       | `sglang`, `sgl.gen`, `RadixAttention`                                    |
| **API_CONFIG**          | `verl/trainer/config/`, `verl/workers/config/`                                      | `@dataclass`, `__post_init__`, `field(`, `OmegaConf`                     |
| **COMPILE**             | -                                                                                   | `torch.compile`, `_dynamo`, `mark_dynamic`, `fullgraph`                  |
| **ACTIVATION_CKPT**     | -                                                                                   | `activation_checkpoint`, `checkpoint_wrapper`, `selective_checkpoint`    |
| **CHECKPOINT_RECOVERY** | `verl/checkpoint_engine/`, `verl/utils/checkpoint/`                                 | `state_dict`, `load_state_dict`, `checkpoint`                            |
| **REWARD**              | `verl/reward/`                                                                      | `reward_fn`, `compute_reward`                                            |
| **DATASET**             | `verl/data/`, `verl/utils/dataset/`                                                 | `DataLoader`, `IterableDataset`, `get_*_dataset`                         |
| **ATTENTION**           | -                                                                                   | `flash_attn`, `sdpa`, `varlen`, `causal_mask`                            |
| **ALGORITHM**           | `verl/trainer/ppo/core_algos.py`, `verl/trainer/ppo/actor.py`                       | `compute_advantage`, `ppo_loss`, `grpo`, `kl_penalty`                    |

## LOW Level (Use Haiku)

| Change Type     | File Path Pattern            | Code Pattern |
| --------------- | ---------------------------- | ------------ |
| **TESTS**       | `tests/`, `*_test.py`        | -            |
| **DOCS**        | `docs/`, `*.md`              | -            |
| **CONFIG_ONLY** | `*.yaml`, `*.json`, `*.toml` | -            |
| **EXAMPLES**    | `examples/`                  | -            |

______________________________________________________________________

## Framework-Specific Risk Identification

### FSDP Risks (When FSDP\_\* types detected)

- **Shard/reshard timing error**: premature or delayed sharding operations
- **Gradient divide factor calculation**: incorrect relationship with world size
- **State dict save/load inconsistency**: mixing sharded vs full modes
- **Optimizer state handling**: aggregation and distribution of sharded state
- **DCP compatibility**: ensure DCP save/load works with FSDP2
- **Device mesh mismatch**: wrong mesh dimensions for DP/TP

### Megatron Risks (When MEGATRON\_\* types detected)

- **Pipeline stage splitting error**: unbalanced layer distribution
- **Micro-batch scheduling issues**: pipeline bubble handling
- **Weight sharding and sync**: tied weights handling
- **AC interaction**: checkpointing under pipeline parallelism
- **Checkpoint conversion**: HF ↔ Megatron format key mapping errors

### Ray/Controller Risks (When RAY\_\* types detected)

- **Dispatch mode mismatch**: using wrong dispatch mode for data distribution
- **Object store memory**: large objects not cleaned up with `ray.internal.free()`
- **Worker resource allocation**: GPU/CPU resource mismatch
- **Blocking operations**: synchronous ops in async contexts
- **Worker lifecycle**: initialization order dependencies

### vLLM/SGLang Risks (When ROLLOUT\_\* types detected)

- **Weight update protocol**: incorrect weight sync between training and rollout workers
- **GPU memory utilization**: OOM during inference
- **Sampling parameter mismatch**: inconsistent generation configs
- **Multi-turn state management**: context leaks between conversations
- **LoRA adapter loading**: incorrect adapter paths or rank mismatch

### DCP/Checkpoint Risks (When DCP_CHECKPOINT or CHECKPOINT_RECOVERY detected)

- **Distributed checkpoint consistency**: all ranks must participate in save/load
- **State dict key mismatch**: keys must match between save and load
- **Optimizer state compatibility**: ensure optimizer state is correctly sharded/gathered
- **Version compatibility**: old checkpoints should load in new code
- **Storage backend compatibility**: filesystem, S3, etc.

______________________________________________________________________

## Risk Linkage Rules

| Detected Change             | Auto-Linked Review                                        |
| --------------------------- | --------------------------------------------------------- |
| FSDP changes                | DCP checkpoint check, device mesh check                   |
| Megatron changes            | Pipeline + AC check, checkpoint conversion check          |
| Ray controller changes      | Worker lifecycle + dispatch mode check                    |
| Distributed comm changes    | Process group + sync check                                |
| SEQUENCE_PARALLEL changes   | TP combination + Attention mask check                     |
| CHECKPOINT_RECOVERY changes | FSDP state dict check, DCP compatibility check            |
| DCP_CHECKPOINT changes      | FSDP2 integration check, distributed consistency check    |
| COMPILE changes             | Performance regression + FSDP/TP interaction check        |
| REWARD changes              | DataProto format check, reward function signature check   |
| ROLLOUT changes             | Weight update protocol check, DataProto integration check |
| TRAINER_CORE changes        | Worker lifecycle + algorithm integration check            |
| DATAPROTO changes           | Worker compatibility check, serialization check           |

______________________________________________________________________

## Core Framework Paths (Must Use Opus)

**FSDP Core**:

- `verl/workers/fsdp_workers.py`
- `verl/utils/fsdp_utils/`

**Megatron Core**:

- `verl/models/mcore/`
- `verl/utils/megatron_utils/`

**Ray Controller Core**:

- `verl/single_controller/base/`
- `verl/trainer/ppo/ray_trainer.py`

**Trainer Core**:

- `verl/trainer/ppo/`

**DataProto**:

- `verl/protocol.py`
