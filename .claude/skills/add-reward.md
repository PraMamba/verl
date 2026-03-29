---
name: add-reward
description: Add a new reward function to verl
---

# Add Reward Function

This skill guides you through adding a new reward function to verl.

## Steps

### 1. Create Reward Function File (5 min)

**Location:** `verl/reward/your_reward_name.py`

**Template:**
```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

import torch
from typing import List, Optional

def your_reward_fn(
    prompt: List[str],
    completions: List[str],
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    """
    Compute rewards for completions.

    Args:
        prompt: List of prompt strings [batch_size]
        completions: List of completion strings [batch_size]
        prompt_ids: Tokenized prompts [batch_size, prompt_len]
        completion_ids: Tokenized completions [batch_size, completion_len]
        **kwargs: Additional context (e.g., ground_truth, metadata)

    Returns:
        rewards: Tensor of shape [batch_size] with reward values
    """
    batch_size = len(completions)
    rewards = torch.zeros(batch_size)

    for i, completion in enumerate(completions):
        # Compute reward for this completion
        reward = compute_single_reward(completion)
        rewards[i] = reward

    return rewards
```

**Key Points:**
- Return tensor of shape `[batch_size]`
- Handle empty completions gracefully
- Normalize rewards to reasonable range (e.g., [0, 1] or [-1, 1])
- Add docstring explaining reward semantics

### 2. Register Reward Function (2 min)

**File:** `verl/reward/__init__.py`

Add import and export:
```python
from verl.reward.your_reward_name import your_reward_fn

__all__ = [
    # ... existing rewards
    'your_reward_fn',
]
```

### 3. Create Reward Config (Optional, 3 min)

**Location:** `verl/trainer/config/reward/your_reward.yaml`

```yaml
# @package _global_

reward_fn:
  name: your_reward_fn
  kwargs:
    # Reward-specific parameters
    threshold: 0.5
    weight: 1.0
```

### 4. Add Unit Tests (10 min)

**Location:** `tests/test_your_reward.py`

```python
import pytest
import torch
from verl.reward.your_reward_name import your_reward_fn

def test_your_reward_basic():
    """Test basic reward computation."""
    prompts = ["What is AI?", "Explain ML"]
    completions = ["AI is...", "ML is..."]
    prompt_ids = torch.randint(0, 1000, (2, 10))
    completion_ids = torch.randint(0, 1000, (2, 20))

    rewards = your_reward_fn(prompts, completions, prompt_ids, completion_ids)

    assert rewards.shape == (2,)
    assert torch.all(rewards >= 0) and torch.all(rewards <= 1)

def test_your_reward_empty():
    """Test with empty completions."""
    prompts = ["Test"]
    completions = [""]
    prompt_ids = torch.randint(0, 1000, (1, 10))
    completion_ids = torch.zeros(1, 0, dtype=torch.long)

    rewards = your_reward_fn(prompts, completions, prompt_ids, completion_ids)

    assert rewards.shape == (1,)
    # Define expected behavior for empty completions

@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_your_reward_batch_sizes(batch_size):
    """Test with different batch sizes."""
    prompts = ["Test"] * batch_size
    completions = ["Response"] * batch_size
    prompt_ids = torch.randint(0, 1000, (batch_size, 10))
    completion_ids = torch.randint(0, 1000, (batch_size, 20))

    rewards = your_reward_fn(prompts, completions, prompt_ids, completion_ids)

    assert rewards.shape == (batch_size,)
```

### 5. Integration Test (Optional, 5 min)

Test reward in PPO trainer:
```python
def test_reward_in_trainer():
    """Test reward function in training loop."""
    config = load_test_config('ppo_trainer.yaml')
    config.reward_fn.name = 'your_reward_fn'

    # Run one training step
    trainer = RayPPOTrainer(config)
    trainer.init_workers()
    trainer.train_step(test_batch)
```

## Common Patterns

### Multi-Component Rewards
```python
def multi_component_reward_fn(prompt, completions, prompt_ids, completion_ids, **kwargs):
    # Task reward
    task_rewards = compute_task_reward(completions)

    # Style reward
    style_rewards = compute_style_reward(completions)

    # Safety reward
    safety_rewards = compute_safety_reward(completions)

    # Combine with weights
    total_rewards = (
        0.6 * task_rewards +
        0.3 * style_rewards +
        0.1 * safety_rewards
    )

    return total_rewards
```

### Using External Models
```python
def model_based_reward_fn(prompt, completions, prompt_ids, completion_ids, **kwargs):
    # Load reward model (do this once in __init__ if using class)
    reward_model = load_reward_model()

    # Compute rewards
    with torch.no_grad():
        rewards = reward_model(prompt_ids, completion_ids)

    return rewards
```

### Async Rewards (for blocking operations)
If reward computation involves blocking I/O (API calls, file access):
```python
import asyncio

async def async_reward_fn(prompt, completions, prompt_ids, completion_ids, **kwargs):
    tasks = [compute_reward_async(c) for c in completions]
    rewards = await asyncio.gather(*tasks)
    return torch.tensor(rewards)
```

## Verification

Run tests:
```bash
pytest tests/test_your_reward.py -v
```

Test in trainer:
```bash
python -m verl.trainer.main_ppo \
  reward_fn.name=your_reward_fn \
  trainer.total_epochs=1
```

## Troubleshooting

**Reward values too large/small:**
- Normalize to [0, 1] or [-1, 1] range
- Use reward scaling in config

**Slow reward computation:**
- Vectorize operations with torch
- Move computation to GPU if possible
- Consider async rewards for I/O

**NaN rewards:**
- Check for division by zero
- Handle edge cases (empty completions)
- Add numerical stability (eps=1e-8)

## Related Files
- `verl/reward/`: Existing reward implementations
- `verl/trainer/config/reward/`: Reward configs
- `examples/ppo_trainer/`: Example usage
