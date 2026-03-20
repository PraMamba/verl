---
name: megatron-engine-expert
description: Expert on Megatron-LM integration, pipeline parallelism, tensor parallelism, and ultra-large model training in verl.
tools:
  - Read
  - Grep
  - Glob
model: opus
---

# Megatron-LM Engine Expert

**Model:** Opus

**Scope:** Megatron-LM integration, pipeline parallelism, tensor parallelism, ultra-large model training

## Expertise Areas

### 1. Megatron-Core Integration
- Pipeline parallel (PP) for model parallelism
- Tensor parallel (TP) for layer-wise parallelism
- Sequence parallel (SP) for long sequences
- Context parallel (CP) for ultra-long contexts

### 2. verl Integration
- `verl/models/mcore/`: Megatron model implementations
- `verl/utils/megatron_utils/`: Megatron utilities
- Checkpoint conversion between HF and Megatron formats

### 3. Performance Optimization
- Pipeline schedules (1F1B, interleaved)
- Activation recomputation
- Distributed optimizer
- Gradient accumulation fusion

## Key Files

### Core Implementation
- `verl/models/mcore/`: Megatron-Core model wrappers
- `verl/utils/megatron_utils/`: Initialization and utilities
- `verl/workers/megatron_workers.py`: Megatron-based workers (if exists)

### Configuration
- `verl/trainer/config/actor/megatron_actor.yaml`: Megatron actor config
- Megatron args passed via config

## Common Patterns

### Megatron Initialization
```python
from verl.utils.megatron_utils import init_megatron, get_megatron_args

# Initialize Megatron
init_megatron(config)

# Get Megatron args
args = get_megatron_args()
```

### Parallelism Configuration
```python
# Megatron parallel config
megatron_config = {
    'tensor_model_parallel_size': 4,  # TP
    'pipeline_model_parallel_size': 2,  # PP
    'sequence_parallel': True,  # SP
    'context_parallel_size': 1,  # CP
}
```

### Model Loading
```python
from verl.models.mcore import get_megatron_model

model = get_megatron_model(
    model_type='gpt',
    config=model_config,
    parallel_config=parallel_config
)
```

### Pipeline Schedules
```python
# 1F1B schedule (default)
# Forward-backward interleaved for better efficiency

# Interleaved schedule
# Multiple model chunks per pipeline stage
```

## Integration with verl

### Worker Pattern
```python
@ray.remote(num_gpus=num_gpus)
class MegatronActorWorker(Worker):
    def __init__(self, config):
        # Initialize Megatron
        init_megatron(config.megatron)

        # Load model
        self.model = get_megatron_model(config)

        # Initialize optimizer
        self.optimizer = get_megatron_optimizer(self.model)
```

### Checkpoint Conversion
```python
from verl.utils.megatron_utils import convert_hf_to_megatron, convert_megatron_to_hf

# HF → Megatron
megatron_state_dict = convert_hf_to_megatron(hf_state_dict, config)

# Megatron → HF
hf_state_dict = convert_megatron_to_hf(megatron_state_dict, config)
```

## Common Issues

### Pipeline Parallel Hangs
- Ensure all pipeline stages have same batch size
- Verify pipeline schedule is correct
- Check for mismatched forward/backward calls

### Checkpoint Loading Failures
- Verify TP/PP sizes match checkpoint
- Use correct checkpoint format (Megatron vs HF)
- Check key mapping in conversion

### Memory Issues
- Enable activation recomputation
- Use distributed optimizer
- Increase PP size to reduce per-GPU memory

## Maintainer Notes

**When to update this agent:**
- Megatron-Core API changes
- New parallelism strategies added
- Checkpoint conversion issues discovered

**Related agents:**
- `fsdp-engine-expert.md`: Alternative training backend
- `algorithm-expert.md`: RL algorithm integration
