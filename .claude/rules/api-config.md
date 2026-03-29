---
inclusion: fileMatch
fileMatchPattern: "verl/trainer/config/**|verl/workers/config/**"
---

# Hydra Configuration Guidelines for verl

## Configuration Structure

verl uses Hydra for hierarchical configuration management. Configs are organized by component:

```
verl/trainer/config/
├── ppo_trainer.yaml          # Main PPO config
├── grpo_trainer.yaml         # Main GRPO config
├── actor/                    # Actor worker configs
├── critic/                   # Critic worker configs
├── rollout/                  # Rollout worker configs
├── reward/                   # Reward function configs
└── algorithm/                # Algorithm-specific configs
```

## Config File Patterns

### Main Trainer Config
```yaml
# @package _global_

defaults:
  - actor: fsdp_actor
  - critic: fsdp_critic
  - rollout: vllm_rollout
  - reward: default_reward
  - algorithm: ppo

trainer:
  total_epochs: 10
  save_freq: 1
  project_name: verl_experiment
  experiment_name: ppo_llama

data:
  train_files: ???  # Required field
  val_files: null
  train_batch_size: 1024

model:
  path: ???  # Required field

ray_kwargs:
  ray_init:
    num_cpus: 32
```

### Component Config
```yaml
# actor/fsdp_actor.yaml

actor_rollout_ref:
  model:
    path: ${model.path}

  rollout:
    name: vllm
    gpu_memory_utilization: 0.4

  fsdp_config:
    param_offload: false
    grad_offload: false
```

## Config Validation

### Required Fields
Use `???` to mark required fields:
```yaml
model:
  path: ???  # Must be provided by user
  hidden_size: 4096  # Has default
```

### Validation in Code
```python
from verl.utils.config import validate_config

@hydra.main(config_path="config", config_name="ppo_trainer")
def main(config):
    validate_config(config)  # Raises error if required fields missing
    # ... rest of code
```

## Config Composition

### Overriding Defaults
```bash
# Override actor config
python -m verl.trainer.main_ppo actor=megatron_actor

# Override multiple components
python -m verl.trainer.main_ppo \
  actor=fsdp_actor \
  rollout=sglang_rollout \
  reward=custom_reward
```

### Command-Line Overrides
```bash
# Override nested values
python -m verl.trainer.main_ppo \
  trainer.total_epochs=20 \
  model.path=/path/to/model \
  data.train_batch_size=2048
```

### Config Groups
```yaml
# Use config groups for variants
defaults:
  - actor: fsdp_actor
  - _self_  # This config takes precedence

# Override specific actor settings
actor_rollout_ref:
  fsdp_config:
    param_offload: true  # Override default
```

## Dataclass Integration

### Converting OmegaConf to Dataclass
```python
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import FSDPEngineConfig

# Convert Hydra config to typed dataclass
engine_config = omega_conf_to_dataclass(
    config.actor.fsdp_config,
    FSDPEngineConfig
)
```

### Dataclass Definition
```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class FSDPEngineConfig:
    """FSDP engine configuration."""

    # Required fields (no default)
    model_path: str

    # Common optional fields
    param_offload: bool = False
    grad_offload: bool = False

    # Advanced optional fields
    mixed_precision: str = "bf16"
    activation_checkpointing: bool = False

    # Internal fields (prefix with _)
    _device_mesh: Optional[object] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.mixed_precision not in ["fp32", "fp16", "bf16"]:
            raise ValueError(
                f"Invalid mixed_precision: {self.mixed_precision}. "
                f"Must be one of: fp32, fp16, bf16"
            )

        if self.param_offload and self.grad_offload:
            logger.warning(
                "Both param_offload and grad_offload enabled. "
                "This may cause high CPU memory usage."
            )
```

## Config Best Practices

### 1. Use Interpolation
```yaml
model:
  path: /models/llama-7b

actor:
  model_path: ${model.path}  # Reference other config values

critic:
  model_path: ${model.path}  # DRY principle
```

### 2. Environment Variables
```yaml
data:
  train_files: ${oc.env:TRAIN_DATA_PATH}  # Read from env var

wandb:
  api_key: ${oc.env:WANDB_API_KEY,null}  # Default to null if not set
```

### 3. Conditional Config
```yaml
# Use resolvers for conditional logic
actor:
  use_lora: true
  lora_rank: ${oc.select:actor.use_lora,16,null}  # 16 if use_lora else null
```

### 4. Config Inheritance
```yaml
# base_actor.yaml
model:
  hidden_size: 4096
  num_layers: 32

# llama_actor.yaml
defaults:
  - base_actor

model:
  hidden_size: 5120  # Override base value
```

## Auto-Generated Configs

verl uses scripts to auto-generate configs from code:

```bash
# Regenerate trainer configs
bash scripts/generate_trainer_config.sh
```

**When to regenerate:**
- New config fields added to dataclasses
- Default values changed
- New config groups added

## Backward Compatibility

### Deprecating Fields
```python
def __post_init__(self):
    # Handle deprecated field
    if hasattr(self, 'old_field_name'):
        warnings.warn(
            "old_field_name is deprecated. Use new_field_name instead.",
            DeprecationWarning
        )
        self.new_field_name = self.old_field_name
```

### Adding New Fields
Always provide defaults for new fields:
```python
@dataclass
class Config:
    existing_field: int
    new_field: bool = False  # Default for backward compatibility
```

## Common Patterns

### Multi-Stage Training
```yaml
# Stage 1: SFT
defaults:
  - base_config

trainer:
  stage: sft
  total_epochs: 3

# Stage 2: RL
defaults:
  - base_config

trainer:
  stage: rl
  total_epochs: 10
  load_checkpoint: ${stage1_checkpoint_path}
```

### Experiment Sweeps
```yaml
# Use Hydra multirun for sweeps
# python -m verl.trainer.main_ppo -m \
#   algorithm.ppo.kl_coef=0.01,0.05,0.1 \
#   algorithm.ppo.clip_ratio=0.1,0.2,0.3

algorithm:
  ppo:
    kl_coef: 0.05
    clip_ratio: 0.2
```

## Maintainer Notes

**When to update this file:**
- New Hydra features adopted
- Config structure changes
- New validation patterns emerge
- Backward compatibility issues arise

**Related files:**
- `verl/utils/config.py`: Config utilities
- `scripts/generate_trainer_config.sh`: Config generation
- `verl/workers/config/`: Worker config dataclasses
