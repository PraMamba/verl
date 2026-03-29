---
inclusion: always
---

# verl Code Style Guide

## Core Design Principles

### 1. Composition Over Inheritance
- Keep inheritance hierarchies shallow (≤2 levels)
- Prefer explicit delegation over deep class hierarchies
- Use mixins sparingly and document their purpose

### 2. Ray Single-Controller Pattern
- Use `@ray.remote` decorators for distributed workers
- Workers communicate via Ray object store and DataProto
- Controller orchestrates workers without direct inter-worker communication
- Always use `ray.get()` for blocking operations, `ray.wait()` for async coordination

### 3. DataProto Protocol
- Use `DataProto` for all data transfer between workers
- Wrap tensors in TensorDict for structured data
- Use `DataProto.concat()` for batching, `DataProto.split()` for distribution
- Enable auto-padding with `DataProtoConfig.auto_padding = True` when needed

## Naming Conventions

### Classes
- Worker classes: `XxxWorker` (e.g., `ActorRolloutWorker`, `FSDPCriticWorker`)
- Config dataclasses: `XxxConfig` (e.g., `FSDPEngineConfig`, `RolloutConfig`)
- Trainer classes: `XxxTrainer` (e.g., `RayPPOTrainer`)
- Sharding managers: `XxxShardingManager`

### Functions
- Rollout functions: `xxx_rollout` (e.g., `vllm_rollout`, `sglang_rollout`)
- Reward functions: `xxx_reward_fn`
- Utility functions: lowercase with underscores

### Files
- Worker implementations: `xxx_workers.py`
- Configs: `config/xxx.yaml` (Hydra configs)

## Logging

Use Python's standard logging with appropriate levels:

```python
import logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
```

**Log levels:**
- `DEBUG`: Detailed diagnostic info
- `INFO`: General informational messages
- `WARN`: Warning messages (default level)
- `ERROR`: Error messages

**Color scheme (if using colored logging):**
- Blue: Infrastructure/Ray operations
- White: Orchestration/controller
- Purple: RL algorithm logic
- Green: Data loading/processing
- Cyan: Compute/model operations

## Import Organization

Group imports in this order:
1. Standard library
2. Third-party (torch, ray, hydra, etc.)
3. verl imports (absolute imports from verl.*)

```python
import os
import logging

import ray
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.single_controller.base import Worker
from verl.utils.fsdp_utils import apply_fsdp2
```

## Performance Guidelines

### Avoid GPU-CPU Synchronization
- Minimize `.item()`, `.cpu()`, `.numpy()` calls in training loops
- Use `torch.cuda.synchronize()` only when necessary for profiling
- Batch operations to reduce kernel launch overhead

### Memory Management
- Use `aggressive_empty_cache()` after large memory operations
- Enable activation offloading for large models: `enable_activation_offloading(model)`
- Use FSDP2 CPU offload for memory-constrained scenarios

### Ray Object Store
- Avoid passing large objects through Ray repeatedly
- Use `ray.put()` once and pass object refs
- Clean up with `ray.internal.free()` when done

## Distributed Patterns

### Process Group Management
- Never create global process groups
- Always pass `process_group` explicitly to collectives
- Use `init_device_mesh()` for FSDP2 device mesh creation

### Worker Initialization
- Initialize workers with proper resource allocation: `@ray.remote(num_gpus=X)`
- Use `get_event_loop()` for async operations in workers
- Handle worker failures gracefully with try-except

## Configuration with Hydra

### Config Structure
- Use `@hydra.main(config_path="config", config_name="xxx")` decorator
- Validate configs with `validate_config(config)`
- Use `OmegaConf.to_container()` for Ray serialization
- Merge configs with `OmegaConf.merge()`

### Config Naming
- Main configs: `ppo_trainer.yaml`, `grpo_trainer.yaml`
- Component configs: `actor/`, `critic/`, `rollout/`, `reward/`

## Error Handling

### Graceful Degradation
- Check for CUDA availability: `is_cuda_available()`
- Auto-detect device: `auto_set_device(config)`
- Fallback to CPU when GPU unavailable

### Informative Errors
- Raise `ValueError` with clear messages for config validation
- Use `assert` with descriptive messages for invariants
- Log errors before raising for debugging

## Testing Patterns

See `testing.md` for detailed testing guidelines.

## Maintainer Notes

**When to update this file:**
- New design patterns emerge in verl core
- Performance best practices change
- Ray or PyTorch APIs evolve
- Common anti-patterns are identified

**Related files:**
- `distributed.md`: Distributed training specifics
- `testing.md`: Testing conventions
- `api-config.md`: Hydra configuration patterns
