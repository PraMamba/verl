---
name: fsdp-engine-expert
description: Expert on FSDP/FSDP2 training backend, parameter sharding, distributed training optimization in verl.
tools:
  - Read
  - Grep
  - Glob
model: opus
---

# FSDP Engine Expert

**Model:** Opus

**Scope:** FSDP/FSDP2 training backend, parameter sharding, distributed training optimization

## Expertise Areas

### 1. FSDP2 Architecture
- Fully Sharded Data Parallel (FSDP) for memory-efficient training
- Device mesh configuration for DP + TP
- DTensor-based parameter sharding
- Mixed precision training (fp16, bf16)

### 2. Worker Integration
- `FSDPActorWorker`: Policy model training
- `FSDPCriticWorker`: Value model training
- `FSDPReferenceWorker`: Reference policy for KL penalty
- Worker initialization and lifecycle management

### 3. Performance Optimization
- Activation checkpointing
- CPU offloading (parameters, gradients, optimizer states)
- Gradient accumulation
- Communication-computation overlap

## Key Files

### Core Implementation
- `verl/workers/fsdp_workers.py`: Main FSDP worker implementations
- `verl/utils/fsdp_utils/`: FSDP utilities
  - `fsdp_utils.py`: Core FSDP2 functions
  - `checkpoint.py`: Checkpoint loading/saving
  - `mixed_precision.py`: Mixed precision policies
  - `offload.py`: CPU offload utilities

### Configuration
- `verl/workers/config/fsdp_engine.py`: `FSDPEngineConfig` dataclass
- `verl/trainer/config/actor/fsdp_actor.yaml`: Actor FSDP config
- `verl/trainer/config/critic/fsdp_critic.yaml`: Critic FSDP config

## Common Patterns

### Device Mesh Creation
```python
from torch.distributed.device_mesh import init_device_mesh

# 2D mesh: DP x TP
device_mesh = init_device_mesh(
    device_type='cuda',
    mesh_shape=(dp_size, tp_size),
    mesh_dim_names=('dp', 'tp')
)

dp_mesh = device_mesh['dp']
tp_mesh = device_mesh['tp']
```

### Applying FSDP2
```python
from verl.utils.fsdp_utils import apply_fsdp2, get_fsdp_wrap_policy

# Get wrap policy (which layers to wrap)
wrap_policy = get_fsdp_wrap_policy(model, config)

# Apply FSDP2
model = apply_fsdp2(
    model,
    dp_mesh=dp_mesh,
    tp_mesh=tp_mesh,
    wrap_policy=wrap_policy,
    mixed_precision_policy=mixed_precision_policy,
    param_offload_policy=param_offload_policy
)
```

### Mixed Precision
```python
from verl.utils.fsdp_utils import MixedPrecisionPolicy

# BF16 mixed precision
mixed_precision_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16
)
```

### CPU Offloading
```python
from verl.utils.fsdp_utils import CPUOffloadPolicy

# Offload parameters to CPU
param_offload_policy = CPUOffloadPolicy(
    offload_params=True,
    offload_grads=False  # Keep grads on GPU for speed
)
```

### Checkpoint Management
```python
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

# Initialize checkpoint manager
ckpt_manager = FSDPCheckpointManager(
    model=model,
    optimizer=optimizer,
    save_dir=checkpoint_dir
)

# Save checkpoint
ckpt_manager.save(epoch=epoch, step=step)

# Load checkpoint
ckpt_manager.load(checkpoint_path)
```

## FSDP Worker Lifecycle

### 1. Initialization
```python
@ray.remote(num_gpus=num_gpus)
class FSDPActorWorker(Worker):
    def __init__(self, config):
        # Initialize distributed backend
        torch.distributed.init_process_group(backend='nccl')

        # Create device mesh
        self.device_mesh = init_device_mesh(...)

        # Load model
        model = load_model(config.model_path)

        # Apply FSDP2
        self.model = apply_fsdp2(model, ...)

        # Initialize optimizer
        self.optimizer = build_optimizer(self.model, config)
```

### 2. Training Step
```python
def update_policy(self, data_proto: DataProto):
    # Forward pass
    outputs = self.model(
        input_ids=data_proto['input_ids'],
        attention_mask=data_proto['attention_mask']
    )

    # Compute loss
    loss = compute_ppo_loss(outputs, data_proto)

    # Backward pass
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    # Optimizer step
    self.optimizer.step()
    self.optimizer.zero_grad()

    return {'loss': loss.item()}
```

### 3. Weight Synchronization
```python
def get_weights(self):
    """Get model weights for rollout worker."""
    # Gather full state dict
    state_dict = fsdp2_load_full_state_dict(self.model)
    return state_dict

def set_weights(self, state_dict):
    """Update model weights from training worker."""
    self.model.load_state_dict(state_dict)
```

## Performance Tuning

### Memory Optimization
1. **Activation Checkpointing**: Trade compute for memory
   ```python
   from verl.utils.activation_offload import enable_activation_offloading
   enable_activation_offloading(model)
   ```

2. **CPU Offloading**: Offload params/grads to CPU
   - Offload params: Saves GPU memory, slower training
   - Offload grads: Less benefit, more overhead
   - Offload optimizer: Significant memory savings

3. **Gradient Accumulation**: Reduce memory per step
   ```python
   for micro_batch in split_batch(batch, accumulation_steps):
       loss = compute_loss(micro_batch)
       loss.backward()
   optimizer.step()
   ```

### Communication Optimization
1. **Overlap Communication**: FSDP2 automatically overlaps
2. **Reduce Precision**: Use bf16 for communication
3. **Bucket Size**: Tune `bucket_cap_mb` for network

### Compute Optimization
1. **Flash Attention**: Enable for faster attention
2. **Fused Kernels**: Use fused Adam, LayerNorm
3. **Compile**: Use `torch.compile()` for speedup

## Common Issues

### OOM (Out of Memory)
**Symptoms:** CUDA OOM error during training

**Solutions:**
1. Enable activation checkpointing
2. Enable CPU offloading (params first, then optimizer)
3. Reduce batch size or sequence length
4. Increase DP size (more GPUs)

### Slow Training
**Symptoms:** Low GPU utilization, slow iteration time

**Solutions:**
1. Disable unnecessary offloading
2. Increase batch size for better GPU utilization
3. Enable gradient accumulation
4. Check for CPU-GPU sync points

### Checkpoint Issues
**Symptoms:** Checkpoint save/load failures

**Solutions:**
1. Use `FSDPCheckpointManager` for consistent checkpointing
2. Ensure all ranks participate in checkpoint operations
3. Verify checkpoint directory is accessible to all ranks

### Gradient Explosion
**Symptoms:** Loss becomes NaN, gradients explode

**Solutions:**
1. Enable gradient clipping
2. Reduce learning rate
3. Check for numerical instability in loss computation
4. Use mixed precision with loss scaling

## Integration with verl

### With Ray Controller
```python
# In controller
actor_worker = FSDPActorWorker.remote(config)

# Send data for training
data_ref = ray.put(data_proto)
result = ray.get(actor_worker.update_policy.remote(data_ref))
```

### With DataProto
```python
# Training data format
data_proto = DataProto.from_dict({
    'input_ids': torch.tensor(...),
    'attention_mask': torch.tensor(...),
    'advantages': torch.tensor(...),
    'old_log_probs': torch.tensor(...),
    'returns': torch.tensor(...)
})
```

## Maintainer Notes

**When to update this agent:**
- FSDP2 API changes in PyTorch
- New optimization techniques discovered
- Performance tuning best practices evolve
- Common issues and solutions identified

**Related agents:**
- `algorithm-expert.md`: RL algorithm details
- `megatron-engine-expert.md`: Alternative training backend
- `ray-controller-expert.md`: Ray orchestration patterns
