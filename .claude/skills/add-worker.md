---
name: add-worker
description: Add a new Ray worker to verl
---

# Add Ray Worker

This skill guides you through adding a new Ray worker to verl.

## Steps

### 1. Create Worker File (15 min)

**Location:** `verl/workers/your_worker_name.py`

**Template:**
```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

import ray
import torch
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch

@ray.remote(num_gpus=1)
class YourWorker(Worker):
    """
    Worker for your specific task.

    Args:
        config: Worker configuration
    """

    def __init__(self, config):
        self.config = config
        # Initialize resources (model, etc.)
        self._init_model()

    def _init_model(self):
        """Initialize model and other resources."""
        # Load model, optimizer, etc.
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def update_config(self, config):
        """Update worker configuration."""
        self.config = config

    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    def process_batch(self, data_proto: DataProto) -> DataProto:
        """
        Process a batch of data.

        Args:
            data_proto: Input data

        Returns:
            result_proto: Processed results
        """
        # Process data
        results = self._process(data_proto)

        # Return as DataProto
        result_proto = DataProto.from_dict(results)
        return result_proto

    def _process(self, data_proto: DataProto):
        """Internal processing logic."""
        # Implement your logic here
        pass
```

### 2. Register Worker (2 min)

**File:** `verl/workers/__init__.py`

```python
from verl.workers.your_worker_name import YourWorker

__all__ = [
    # ... existing workers
    'YourWorker',
]
```

### 3. Add Worker Config (5 min)

**Location:** `verl/workers/config/your_worker.py`

```python
from dataclasses import dataclass

@dataclass
class YourWorkerConfig:
    """Configuration for YourWorker."""

    # Required fields
    model_path: str

    # Optional fields
    num_gpus: int = 1
    batch_size: int = 32

    def __post_init__(self):
        """Validate configuration."""
        if self.num_gpus < 1:
            raise ValueError("num_gpus must be >= 1")
```

### 4. Integrate with Controller (10 min)

**File:** `verl/trainer/ppo/ray_trainer.py` (or your trainer)

```python
def init_workers(self):
    # ... existing workers

    # Initialize your worker
    self.your_workers = [
        YourWorker.remote(self.config.your_worker)
        for _ in range(self.config.num_your_workers)
    ]

def train_step(self, batch):
    # ... existing logic

    # Use your worker
    data_ref = ray.put(data_proto)
    result_refs = [
        worker.process_batch.remote(data_ref)
        for worker in self.your_workers
    ]
    results = ray.get(result_refs)
```

### 5. Add Tests (10 min)

**Location:** `tests/test_your_worker.py`

```python
import pytest
import ray
from verl import DataProto
from verl.workers.your_worker_name import YourWorker

@pytest.fixture(scope="module")
def ray_cluster():
    if not ray.is_initialized():
        ray.init(num_cpus=4)
    yield
    ray.shutdown()

def test_worker_init(ray_cluster):
    """Test worker initialization."""
    config = create_test_config()
    worker = YourWorker.remote(config)

    # Verify worker is alive
    assert ray.get(worker.__ray_ready__.remote())

def test_worker_process(ray_cluster):
    """Test worker processing."""
    config = create_test_config()
    worker = YourWorker.remote(config)

    # Create test data
    data_proto = DataProto.from_dict({
        'input': torch.randn(4, 128)
    })

    # Process
    result = ray.get(worker.process_batch.remote(data_proto))

    assert isinstance(result, DataProto)
    assert len(result) == len(data_proto)
```

## Dispatch Modes

### ONE_TO_ALL
Send same data to all worker replicas:
```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def broadcast_config(self, config):
    self.config = config
```

### DP_COMPUTE
Split data across workers (data parallel):
```python
@register(dispatch_mode=Dispatch.DP_COMPUTE)
def compute_batch(self, data_proto: DataProto):
    # Each worker gets a shard
    return results
```

### MEGATRON_COMPUTE
Pipeline parallel execution:
```python
@register(dispatch_mode=Dispatch.MEGATRON_COMPUTE)
def forward_pass(self, data_proto: DataProto):
    # Pipeline parallel
    return outputs
```

## Verification

Run tests:
```bash
pytest tests/test_your_worker.py -v
```

Test in trainer:
```bash
python -m verl.trainer.main_ppo \
  your_worker.num_gpus=1 \
  trainer.total_epochs=1
```

## Related Files
- `verl/workers/`: Existing worker implementations
- `verl/single_controller/base/`: Worker base classes
- `verl/trainer/ppo/ray_trainer.py`: Controller integration
