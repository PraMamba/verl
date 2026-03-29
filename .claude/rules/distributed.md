---
inclusion: fileMatch
fileMatchPattern: "verl/workers/**|verl/models/**|verl/utils/fsdp_utils/**|verl/utils/megatron_utils/**"
---

# Distributed Training Guidelines for verl

## Core Principles

### 1. Never Create Global Process Groups
**Wrong:**
```python
dist.init_process_group(backend='nccl')  # Global group
dist.all_reduce(tensor)  # Uses global group
```

**Right:**
```python
device_mesh = init_device_mesh('cuda', mesh_shape=(dp_size, tp_size))
dp_group = device_mesh.get_group(mesh_dim=0)
dist.all_reduce(tensor, group=dp_group)
```

### 2. Always Pass Process Groups Explicitly
Every collective operation must specify which process group to use:
- `dist.all_reduce(tensor, group=pg)`
- `dist.all_gather(output, tensor, group=pg)`
- `dist.broadcast(tensor, src=0, group=pg)`

### 3. Worker-Based Architecture
verl uses separate workers for different roles:
- **Actor Worker**: Policy model for generation
- **Critic Worker**: Value model for advantage estimation
- **Rollout Worker**: Inference engine (vLLM/SGLang) for generation
- **Reference Worker**: Reference policy for KL penalty

Each worker has its own process group and device mesh.

## FSDP2 Patterns

### Device Mesh Creation
```python
from torch.distributed.device_mesh import init_device_mesh

# Create 2D mesh for DP + TP
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

wrap_policy = get_fsdp_wrap_policy(model, config)
model = apply_fsdp2(
    model,
    dp_mesh=dp_mesh,
    tp_mesh=tp_mesh,
    wrap_policy=wrap_policy,
    mixed_precision_policy=mixed_precision_policy
)
```

### DTensor Placement
- `Shard(dim)`: Shard tensor along dimension
- `Replicate()`: Replicate tensor across ranks
- `Partial()`: Partial tensor (needs reduction)

**Example:**
```python
from torch.distributed.tensor import DTensor, Shard, Replicate

# Shard along batch dimension for DP
dtensor = DTensor.from_local(
    local_tensor,
    device_mesh=dp_mesh,
    placements=[Shard(0)]
)
```

## Megatron-LM Integration

### Tensor Parallel
- Use `verl.utils.megatron_utils` for Megatron integration
- Initialize Megatron with proper TP/PP/CP configuration
- Handle pipeline parallel schedules carefully

### Common Patterns
```python
from verl.utils.megatron_utils import init_megatron, get_megatron_args

# Initialize Megatron
init_megatron(config)
args = get_megatron_args()

# Use Megatron's parallel state
from megatron.core import parallel_state
tp_group = parallel_state.get_tensor_model_parallel_group()
```

## Ray + Distributed Training

### Worker Resource Allocation
```python
@ray.remote(num_gpus=num_gpus_per_worker)
class ActorWorker(Worker):
    def __init__(self, config):
        # Initialize distributed backend within worker
        torch.distributed.init_process_group(backend='nccl')
        self.device_mesh = init_device_mesh(...)
```

### Data Transfer Between Workers
Use DataProto for efficient data transfer:
```python
# In controller
data_ref = ray.put(data_proto)
result_refs = [worker.process.remote(data_ref) for worker in workers]
results = ray.get(result_refs)
```

## Common Pitfalls

### 1. Mismatched Collectives
**Problem:** Different ranks call different collectives → deadlock

**Solution:** Ensure all ranks in a process group call the same collective:
```python
# All ranks must participate
if condition:
    result = dist.all_reduce(tensor, group=pg)
else:
    result = dist.all_reduce(torch.zeros_like(tensor), group=pg)
```

### 2. Wrong Process Group
**Problem:** Using wrong process group → incorrect results or hang

**Solution:** Verify process group matches the intended parallelism:
```python
# For DP reduction, use DP group
dist.all_reduce(grad, group=dp_group)  # Not tp_group!
```

### 3. Tensor Shape Mismatches
**Problem:** Tensors have different shapes across ranks → error

**Solution:** Verify shapes before collectives:
```python
# Gather shapes first
local_shape = torch.tensor(tensor.shape, device='cuda')
shapes = [torch.zeros_like(local_shape) for _ in range(world_size)]
dist.all_gather(shapes, local_shape, group=pg)
# Verify all shapes match
```

### 4. DTensor Placement Errors
**Problem:** Incorrect placement specification → wrong sharding

**Solution:** Match placement to intended parallelism:
```python
# For DP: shard along batch (dim 0)
placements = [Shard(0)]

# For TP: shard along hidden dimension
placements = [Shard(1)]  # or Shard(2) depending on tensor
```

### 5. NCCL Errors
**Problem:** NCCL timeout or communication errors

**Debug:**
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

**Common causes:**
- Network issues between nodes
- Mismatched NCCL versions
- Incorrect NCCL_SOCKET_IFNAME
- Firewall blocking NCCL ports

## Debugging Distributed Code

### 1. Rank-Conditional Logging
```python
if dist.get_rank() == 0:
    logger.info(f"Step {step}: loss={loss.item()}")
```

### 2. Synchronization Points
Add barriers to isolate issues:
```python
dist.barrier(group=pg)  # All ranks wait here
logger.info(f"Rank {dist.get_rank()} passed barrier")
```

### 3. Tensor Inspection
```python
# Check tensor placement
if isinstance(tensor, DTensor):
    logger.info(f"DTensor placements: {tensor.placements}")
    logger.info(f"Local shape: {tensor.to_local().shape}")
```

### 4. Process Group Verification
```python
# Verify process group membership
rank = dist.get_rank(group=pg)
world_size = dist.get_world_size(group=pg)
logger.info(f"PG rank: {rank}/{world_size}")
```

## Performance Optimization

### 1. Overlap Communication and Computation
- Use FSDP2's built-in overlap
- Enable gradient accumulation for better overlap
- Use `torch.cuda.Stream` for manual overlap

### 2. Reduce Communication Volume
- Use mixed precision (fp16/bf16) for communication
- Enable gradient compression if available
- Minimize all-gather operations

### 3. Optimize Collective Operations
- Use `dist.all_reduce` instead of `reduce` + `broadcast`
- Batch small tensors into single collective
- Use in-place operations when possible

## Maintainer Notes

**When to update this file:**
- New parallelism strategies added (e.g., context parallel, expert parallel)
- FSDP2 or Megatron APIs change
- New distributed debugging techniques discovered
- Common distributed bugs identified

**Related files:**
- `code-style.md`: General coding patterns
- `verl/utils/fsdp_utils/`: FSDP2 utilities
- `verl/utils/megatron_utils/`: Megatron utilities
