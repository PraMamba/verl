# PR Review: Task Templates Reference

This file contains the review task templates for PR review. Referenced by:
`.claude/commands/review-pr.md`

______________________________________________________________________

## Framework-Specific Review Task Templates

### FSDP Tasks \[Opus/Sonnet\]

**Task: FSDP Core Correctness \[Opus\]**

```
Checklist:
- Shard/reshard operation timing and correctness
- ShardedTensor and DTensor conversion
- Mixed precision (param_dtype vs reduce_dtype)
- Device mesh dimension names and structure
```

**Task: FSDP Interaction with Other Parallel Strategies \[Opus\]**

```
Checklist:
- FSDP must be applied after TP/CP
- Gradient divide factor relationship with world size
- CPU offload interaction with gradient accumulation
```

**Task: FSDP State Management \[Sonnet\]**

```
Checklist:
- state_dict save/load sharded vs full mode
- Optimizer state sharding and aggregation
- Checkpoint compatibility
```

### Megatron Tasks \[Opus\]

**Task: Pipeline Parallelism Correctness**

```
Checklist:
- Stage splitting correctness and balance
- Micro-batch scheduling
- Pipeline flush and bubble handling
```

**Task: Megatron Model Sharding**

```
Checklist:
- Weight sharding and synchronization
- Tied weights handling
- Embedding/output layer parallel strategy
- Checkpoint format conversion (HF ↔ Megatron)
```

### Ray Controller Tasks \[Opus\]

**Task: Ray Controller Correctness**

```
Checklist:
- Worker initialization order and dependencies
- Dispatch mode selection for each worker method
- DataProto format consistency between controller and workers
- Resource allocation (num_gpus, num_cpus) correctness
- Object store memory management (ray.put/ray.internal.free)
```

**Task: Worker Lifecycle Management**

```
Checklist:
- Worker creation and initialization
- Weight synchronization between training and rollout workers
- Graceful shutdown and cleanup
- Error handling and recovery
```

### DCP/Checkpoint Tasks \[Opus\]

**Task: Distributed Checkpoint Correctness**

```
Checklist:
- All ranks participate in DCP save/load operations
- State dict keys match between save and load
- No tensor shape/dtype mismatches
- Storage backend compatibility (filesystem, S3)
- Checkpoint versioning and migration
```

**Task: FSDP2 + DCP Integration**

```
Checklist:
- FSDP2 state dict options (full vs sharded)
- Optimizer state handling with DCP
- Async checkpointing correctness
- Checkpoint resumption logic
```

### Rollout/Inference Tasks \[Sonnet\]

**Task: vLLM Rollout Correctness**

```
Checklist:
- vLLM engine initialization (model path, TP size, memory utilization)
- Sampling parameter configuration
- Weight update protocol with training workers
- DataProto format for input/output
- LoRA adapter management
```

**Task: SGLang Rollout Correctness**

```
Checklist:
- SGLang runtime initialization
- Multi-turn conversation state management
- Tool calling implementation
- Constrained generation patterns
- RadixAttention prefix caching
```

### Trainer Tasks \[Opus\]

**Task: Trainer Core Logic**

```
Checklist:
- RayPPOTrainer initialization correctness
- Worker coordination and data flow
- Algorithm integration (PPO, GRPO, RLOO)
- Distributed training coordination
- Metric logging and checkpointing
```

______________________________________________________________________

## General Review Task Templates

### Logic and Boundary Conditions \[Opus\]

```
Applicable: Any non-doc/config changes
Checklist:
- Conditional logic errors (if/else inversion, boundary condition omission, short-circuit issues)
- Loop errors (off-by-one, infinite loops, early exit, iterator invalidation)
- Missing null/None/empty list handling
- Type mismatch or implicit type conversion issues
- Improper exception handling (swallowing exceptions, wrong exception type, return in finally)
- Return value errors (wrong type, missing return, inconsistent multi-path returns)
- Boolean expression errors (De Morgan's law violation, precedence errors)
```

### Concurrency and Async \[Opus\]

```
Applicable: ASYNC_CONCURRENT type detected
Checklist:
- Race conditions
- Deadlock risks (inconsistent lock ordering, nested locks)
- Non-thread-safe access to shared state
- Missing await in async code
- Blocking calls in async functions (should use executor)
- Resource leaks (file handles, network connections, GPU memory not released)
- State inconsistency (dirty state after partial update failure)
- Improper context manager usage
- Signal handling and graceful shutdown issues
```

### Tensor Shape and Data Type \[Opus\]

```
Applicable: TENSOR_OPS type detected with complex tensor operations
Checklist:
- Tensor shape mismatch (dimension errors, broadcast errors)
- Batch dimension handling errors (missing batch dim, wrong dimension order)
- Sequence length and padding handling (missing mask, padding token in computation)
- Index out of bounds risk (dynamic indexing, negative indexing)
- dtype mismatch (fp16/fp32/bf16 mixing, integer overflow)
- Device placement errors (tensor on wrong device, CPU/GPU mixed operations)
- Gradient-related issues (missing detach, missing no_grad context, gradient accumulation errors)
- view/reshape contiguity requirements
- In-place operation effects on gradient computation
```

### Numerical Stability \[Sonnet\]

```
Applicable: NUMERICAL type detected
Checklist:
- Numerical precision issues (floating point precision loss, accumulated errors)
- Numerical stability (log(0), division by zero, exp overflow, softmax stability)
- Numerical issues in loss function computation
- Gradient vanishing/exploding risks
- Scaling issues in mixed precision training
```

### Tensor Parallel (TP) Correctness \[Opus\]

```
Applicable: TENSOR_PARALLEL or DISTRIBUTED_COMM type detected
Checklist:
- Missing or misplaced all-reduce
- Missing or misplaced all-gather
- Reduce handling after weight sharding (column/row sharding)
- Input Replicate / output Partial DTensor semantics
- scatter/gather correctness in Sequence Parallel (SP)
- TP group communication correctness
```

### Communication and Synchronization \[Sonnet\]

```
Applicable: DISTRIBUTED_COMM type detected
Checklist:
- Process group usage errors (using default instead of explicit group)
- Device mesh configuration errors
- Improper barrier placement
- Unnecessary synchronization operations (GPU-CPU sync)
- Collective communication order dependencies
```

### API Compatibility \[Sonnet\]

```
Applicable: API_CONFIG type detected
Checklist:
- Function signature changes (parameter add/delete/rename/reorder)
- Return type changes
- Default value changes causing behavior changes
- Breaking changes to public APIs
- Deprecated API usage
- Class/module rename or move
- Hydra config compatibility
```

### Configuration and Parameter Validation \[Sonnet\]

```
Applicable: API_CONFIG type detected with dataclass
Checklist:
- New config items missing validation (__post_init__ validation)
- Unreasonable config default values
- Missing parameter range checks
- Unhandled dependencies between config items
- Hydra/CLI compatibility issues
- Backward compatibility of env vars/config files
- Incorrect dataclass field types
- Auto-generated YAML config consistency
```

### DataProto Integration \[Sonnet\]

```
Applicable: DATAPROTO type detected or worker changes
Checklist:
- DataProto field names match between producer and consumer
- Tensor shapes match expected dimensions
- DataProto.concat() and DataProto.split() usage correctness
- Auto-padding configuration when needed
- Serialization/deserialization compatibility
```

### Ray Worker Patterns \[Sonnet\]

```
Applicable: WORKER_BASE or RAY_CONTROLLER type detected
Checklist:
- @register decorator with correct dispatch_mode
- Worker __init__ resource allocation (num_gpus, num_cpus)
- ray.get() vs ray.wait() usage
- ray.put() for large objects
- Worker method return types compatible with Ray serialization
```

### Activation Checkpointing (AC) \[Sonnet\]

```
Applicable: ACTIVATION_CKPT type detected
Checklist:
- AC application order (must after TP/CP, before FSDP)
- Selective AC op registration correctness
- AC config validation logic
- Compatibility with torch.compile
```

### Performance Regression Risk \[Sonnet\]

```
Applicable: Any non-doc changes, especially TENSOR_OPS, DISTRIBUTED_COMM
Checklist:
- Unnecessary GPU-CPU sync (.item(), .tolist(), printing tensors)
- Memory allocation pattern changes (potential OOM)
- Communication volume increase
- Computational complexity changes
- torch.compile compatibility breakage
- Unnecessary tensor copies
```

### Context-Aware Review \[Sonnet\]

```
Applicable: Any code changes
Checklist:
- Read git blame and history of modified code
- Check for accidental rollback of previous fixes
- Check for breaking previously established patterns or conventions
- Check if changes violate code comments
- Check for violations of TODO/FIXME constraints
- Check for ignored NOTE/WARNING comments
```

### Sequence Parallel (SP/CP) Correctness \[Opus\]

```
Applicable: sequence_parallel, context_parallel, SP, CP
Checklist:
- scatter/gather operation correctness
- Attention mask handling under SP
- Position encoding sharding
- KV cache handling under CP
- Combination correctness with TP
```

### Checkpoint and Recovery \[Sonnet\]

```
Applicable: verl/checkpoint_engine/, verl/utils/checkpoint/, state_dict, checkpoint
Checklist:
- Checkpoint save/load completeness
- Distributed checkpoint consistency
- Version compatibility (can old checkpoints load)
- Recovery logic correctness
- Optimizer state handling
```

### Reward Function Correctness \[Sonnet\]

```
Applicable: verl/reward/ directory
Checklist:
- Reward function signature matches (prompt, completions, prompt_ids, completion_ids, **kwargs)
- Deterministic computation (same input produces same output)
- Numerical range reasonableness
- Edge case handling (empty input, abnormal answers)
- Return tensor shape [batch_size]
```

### Dataset Loader Correctness \[Sonnet\]

```
Applicable: verl/data/ or verl/utils/dataset/ directory
Checklist:
- Data format validation (messages, answer fields)
- Tokenizer compatibility
- max_length truncation logic
- Distributed sampling correctness
- Memory efficiency (avoid loading all data at once)
```

### torch.compile Compatibility \[Sonnet\]

```
Applicable: COMPILE type detected or hot path code modified
Checklist:
- Dynamic shape mark_dynamic marking
- Graph break risks (Python control flow, data-dependent branches)
- Unsupported operations (some in-place ops)
- fullgraph=True compatibility
- Interaction with FSDP/TP
```

### Documentation Format Check \[Haiku\]

```
Applicable: DOCS type detected
Checklist:
- Markdown format correctness
- Internal link validity
- Code example correctness
```

### Test Coverage Check \[Haiku\]

```
Applicable: TESTS type detected
Checklist:
- Test cases cover main paths
- Boundary condition tests
- Error handling tests
```

### Logging and Metrics \[Haiku\]

```
Applicable: logging, wandb, tensorboard
Checklist:
- Use logging.getLogger(__file__) not print
- Structured metrics for wandb/tensorboard
- Reasonable log levels (no DEBUG on hot paths)
- Sensitive info not logged
- VERL_LOGGING_LEVEL respected
```

### Import and Dependencies \[Haiku\]

```
Applicable: Any Python file changes
Checklist:
- Avoid wildcard imports (from x import *)
- Correct third-party vs internal import grouping
- Heavy optional deps inside functions
- Circular import risks
```

### Security and Sensitive Information \[Haiku\]

```
Applicable: Config files, environment variables, API calls
Checklist:
- No hardcoded keys/tokens/passwords
- Sensitive info not committed to repo
- API endpoints configurable
- Error messages don't leak sensitive details
```
