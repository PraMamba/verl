---
name: debug-distributed
description: Debug distributed training issues in verl (hangs, wrong results, OOM, NCCL errors)
---

# Debug Distributed Training

This skill guides you through debugging common distributed training issues in verl.

## Issue Categories

### 1. Training Hangs / Deadlocks
### 2. Wrong Results / Numerical Issues
### 3. Out of Memory (OOM)
### 4. NCCL Communication Errors

---

## 1. Training Hangs / Deadlocks

### Symptoms
- Training stops progressing
- No error messages
- GPU utilization drops to 0%
- Process doesn't respond to Ctrl+C

### Common Causes

**A. Mismatched Collectives**
Different ranks call different collective operations.

**Debug:**
```python
# Add rank-conditional logging before collectives
rank = dist.get_rank()
print(f"Rank {rank}: About to call all_reduce")
dist.all_reduce(tensor, group=pg)
print(f"Rank {rank}: Finished all_reduce")
```

**Fix:**
Ensure all ranks in process group call same collective:
```python
# Wrong: Only some ranks participate
if condition:
    dist.all_reduce(tensor, group=pg)  # Deadlock!

# Right: All ranks participate
if condition:
    result = dist.all_reduce(tensor, group=pg)
else:
    result = dist.all_reduce(torch.zeros_like(tensor), group=pg)
```

**B. Wrong Process Group**
Using wrong process group for collective.

**Debug:**
```python
# Verify process group membership
rank = dist.get_rank(group=pg)
world_size = dist.get_world_size(group=pg)
print(f"PG rank: {rank}/{world_size}")
```

**C. Barrier Mismatch**
Not all ranks reach barrier.

**Debug:**
```python
# Add barriers to isolate issue
print(f"Rank {rank}: Before section A")
dist.barrier(group=pg)
print(f"Rank {rank}: After section A")
```

### Debugging Tools

**1. Enable Distributed Debug Mode**
```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
```

**2. Use py-spy for Call Stacks**
```bash
# In another terminal
py-spy dump --pid <process_id>
```

**3. Timeout Detection**
```python
import torch.distributed as dist
dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=30))
```

---

## 2. Wrong Results / Numerical Issues

### Symptoms
- Loss becomes NaN or Inf
- Gradients explode
- Model outputs incorrect results
- Results differ across runs

### Common Causes

**A. Incorrect Reduction**
Wrong reduction operation or process group.

**Debug:**
```python
# Check tensor values before/after reduction
print(f"Rank {rank}: Before reduce: {tensor}")
dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=pg)
print(f"Rank {rank}: After reduce: {tensor}")
```

**Fix:**
```python
# Use correct reduction op
dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=dp_group)  # Not SUM!
```

**B. DTensor Placement Errors**
Incorrect sharding specification.

**Debug:**
```python
if isinstance(tensor, DTensor):
    print(f"DTensor placements: {tensor.placements}")
    print(f"Local shape: {tensor.to_local().shape}")
    print(f"Global shape: {tensor.shape}")
```

**C. Mixed Precision Issues**
Numerical instability in fp16/bf16.

**Fix:**
```python
# Use gradient scaling for fp16
scaler = torch.cuda.amp.GradScaler()
loss = compute_loss()
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Or use bf16 (more stable)
mixed_precision_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16
)
```

---

## 3. Out of Memory (OOM)

### Symptoms
- `RuntimeError: CUDA out of memory`
- Training crashes during forward/backward pass
- Inconsistent OOM (sometimes works, sometimes fails)

### Solutions

**A. Enable Activation Checkpointing**
```python
from verl.utils.activation_offload import enable_activation_offloading
enable_activation_offloading(model)
```

**B. Enable CPU Offloading**
```python
from verl.utils.fsdp_utils import CPUOffloadPolicy

# Offload parameters
param_offload = CPUOffloadPolicy(offload_params=True)

# Offload optimizer states
optimizer_offload = CPUOffloadPolicy(offload_optimizer=True)
```

**C. Reduce Batch Size**
```python
# Reduce batch size
config.data.train_batch_size = 512  # Was 1024

# Or use gradient accumulation
config.training.gradient_accumulation_steps = 4
```

**D. Increase Parallelism**
```python
# Increase DP size (more GPUs)
config.parallel.dp_size = 8  # Was 4

# Or increase TP size
config.parallel.tp_size = 4  # Was 2
```

**E. Monitor Memory Usage**
```python
import torch

torch.cuda.reset_peak_memory_stats()
# ... training step
peak_memory = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory: {peak_memory:.2f} GB")
```

---

## 4. NCCL Communication Errors

### Symptoms
- `NCCL error: unhandled system error`
- `NCCL timeout`
- Communication hangs

### Debug

**Enable NCCL Debugging**
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=eth0  # Your network interface
```

### Common Causes

**A. Network Issues**
- Firewall blocking NCCL ports
- Network interface misconfigured
- Slow network connection

**Fix:**
```bash
# Check network interface
ifconfig

# Set correct interface
export NCCL_SOCKET_IFNAME=eth0

# Disable IB if not available
export NCCL_IB_DISABLE=1
```

**B. NCCL Version Mismatch**
Different NCCL versions across nodes.

**Fix:**
```bash
# Check NCCL version
python -c "import torch; print(torch.cuda.nccl.version())"

# Ensure same version on all nodes
```

**C. Timeout Too Short**
Communication takes longer than timeout.

**Fix:**
```python
# Increase timeout
dist.init_process_group(
    backend='nccl',
    timeout=datetime.timedelta(minutes=10)  # Was 30 seconds
)
```

---

## General Debugging Workflow

### Step 1: Isolate the Issue
```python
# Add barriers to narrow down location
dist.barrier()  # Before suspected code
# ... code section
dist.barrier()  # After suspected code
```

### Step 2: Add Logging
```python
# Rank-conditional logging
if dist.get_rank() == 0:
    print(f"Step {step}: loss={loss.item()}")

# All ranks logging
print(f"Rank {dist.get_rank()}: tensor shape={tensor.shape}")
```

### Step 3: Verify Assumptions
```python
# Check tensor shapes match across ranks
local_shape = torch.tensor(tensor.shape, device='cuda')
shapes = [torch.zeros_like(local_shape) for _ in range(world_size)]
dist.all_gather(shapes, local_shape)
print(f"Rank {rank}: All shapes: {shapes}")
```

### Step 4: Simplify
- Reduce to single GPU if possible
- Disable optimizations (activation checkpointing, offloading)
- Use smaller model/batch size
- Test with synthetic data

---

## Verification

After fixing:
```bash
# Run with debug mode
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m verl.trainer.main_ppo

# Monitor GPU memory
watch -n 1 nvidia-smi

# Check for hangs with timeout
timeout 300 python -m verl.trainer.main_ppo
```

## Related Files
- `verl/utils/fsdp_utils/`: FSDP utilities
- `verl/workers/fsdp_workers.py`: FSDP worker implementation
- `.claude/rules/distributed.md`: Distributed training guidelines
