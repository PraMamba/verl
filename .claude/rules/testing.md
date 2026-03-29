---
inclusion: fileMatch
fileMatchPattern: "**/tests/**|*_test.py|test_*.py"
---

# Testing Guidelines for verl

## Test Organization

### Directory Structure
```
tests/
├── unit/              # Fast unit tests
├── integration/       # Integration tests
├── special_e2e/       # End-to-end tests
└── special_sanity/    # Sanity checks
```

### Test File Naming
- Unit tests: `test_<module>.py`
- Integration tests: `test_<feature>_integration.py`
- E2E tests: `test_<algorithm>_e2e.py`

## Pytest Markers

### Standard Markers
```python
import pytest

@pytest.mark.slow  # Long-running tests (>10s)
@pytest.mark.asyncio  # Async tests
@pytest.mark.parametrize("param", [val1, val2])  # Parameterized tests
```

### Custom Markers
```python
@pytest.mark.gpu  # Requires GPU
@pytest.mark.multi_gpu  # Requires multiple GPUs
@pytest.mark.ray  # Requires Ray cluster
@pytest.mark.megatron  # Requires Megatron-LM
@pytest.mark.vllm  # Requires vLLM
@pytest.mark.sglang  # Requires SGLang
```

### Skip Conditions
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not ray.is_initialized(), reason="Ray not initialized")
```

## Test Structure (Arrange-Act-Assert)

```python
def test_dataproto_concat():
    # Arrange: Set up test data
    proto1 = DataProto.from_dict({'x': torch.tensor([1, 2])})
    proto2 = DataProto.from_dict({'x': torch.tensor([3, 4])})

    # Act: Perform the operation
    result = DataProto.concat([proto1, proto2])

    # Assert: Verify the result
    expected = torch.tensor([1, 2, 3, 4])
    torch.testing.assert_close(result['x'], expected)
```

## GPU Tests

### Graceful Skipping
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fsdp_training():
    device = torch.device('cuda')
    model = MyModel().to(device)
    # ... test code
```

### Multi-GPU Tests
```python
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need 2+ GPUs")
def test_distributed_training():
    # Use torch.multiprocessing or Ray for multi-GPU tests
    pass
```

### Cleanup
```python
def test_gpu_operation():
    try:
        # Test code
        pass
    finally:
        torch.cuda.empty_cache()
```

## Ray Tests

### Ray Initialization
```python
@pytest.fixture(scope="module")
def ray_cluster():
    if not ray.is_initialized():
        ray.init(num_cpus=4, num_gpus=torch.cuda.device_count())
    yield
    ray.shutdown()

def test_ray_worker(ray_cluster):
    @ray.remote
    class Worker:
        def compute(self):
            return 42

    worker = Worker.remote()
    result = ray.get(worker.compute.remote())
    assert result == 42
```

## Async Tests

```python
@pytest.mark.asyncio
async def test_async_rollout():
    rollout_worker = AsyncRolloutWorker()
    result = await rollout_worker.generate(prompts)
    assert len(result) == len(prompts)
```

## Assertions

### Tensor Comparisons
```python
# Use torch.testing.assert_close with explicit tolerances
torch.testing.assert_close(
    actual,
    expected,
    rtol=1e-5,  # Relative tolerance
    atol=1e-8   # Absolute tolerance
)

# For exact equality (integers, booleans)
assert torch.equal(actual, expected)
```

### DataProto Comparisons
```python
def assert_dataproto_equal(proto1, proto2):
    assert len(proto1) == len(proto2)
    for key in proto1.keys():
        torch.testing.assert_close(proto1[key], proto2[key])
```

### Shape Assertions
```python
assert tensor.shape == (batch_size, seq_len, hidden_dim)
assert tensor.dim() == 3
```

## Fixtures

### Common Fixtures
```python
@pytest.fixture
def sample_config():
    return OmegaConf.create({
        'model': {'hidden_size': 768},
        'training': {'batch_size': 32}
    })

@pytest.fixture
def mock_model():
    return torch.nn.Linear(10, 10)

@pytest.fixture
def sample_dataproto():
    return DataProto.from_dict({
        'input_ids': torch.randint(0, 1000, (4, 128)),
        'attention_mask': torch.ones(4, 128)
    })
```

## Mocking

### Mock External Dependencies
```python
from unittest.mock import Mock, patch

def test_with_mock_vllm():
    with patch('verl.workers.rollout.vllm_rollout') as mock_rollout:
        mock_rollout.return_value = DataProto(...)
        # Test code using mocked vllm
```

### Mock Ray Workers
```python
@ray.remote
class MockWorker:
    def process(self, data):
        return data  # Simplified behavior

def test_with_mock_worker():
    worker = MockWorker.remote()
    result = ray.get(worker.process.remote(test_data))
    assert result == test_data
```

## Performance Tests

### Timing
```python
import time

def test_performance():
    start = time.time()
    # Operation to test
    elapsed = time.time() - start
    assert elapsed < 1.0, f"Too slow: {elapsed}s"
```

### Memory
```python
def test_memory_usage():
    torch.cuda.reset_peak_memory_stats()
    # Operation to test
    peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
    assert peak_memory < 10.0, f"Too much memory: {peak_memory}GB"
```

## Integration Tests

### End-to-End Training
```python
@pytest.mark.slow
@pytest.mark.gpu
def test_ppo_training_e2e():
    config = load_test_config('ppo_trainer.yaml')
    config.trainer.total_epochs = 1  # Quick test

    trainer = RayPPOTrainer(config)
    trainer.init_workers()
    trainer.fit()

    # Verify training completed
    assert trainer.current_epoch == 1
```

## CI/CD Considerations

### Fast Tests for Pre-commit
- Run CPU unit tests only
- Skip slow/GPU tests
- Use `pytest -m "not slow and not gpu"`

### Full Test Suite for CI
- Run all tests including GPU
- Use multiple workers: `pytest -n auto`
- Generate coverage report

## Common Patterns

### Temporary Files
```python
import tempfile

def test_checkpoint_save():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
        save_checkpoint(model, checkpoint_path)
        assert os.path.exists(checkpoint_path)
```

### Random Seeds
```python
def test_reproducibility():
    torch.manual_seed(42)
    result1 = random_operation()

    torch.manual_seed(42)
    result2 = random_operation()

    torch.testing.assert_close(result1, result2)
```

## Maintainer Notes

**When to update this file:**
- New test markers added
- Testing infrastructure changes
- New testing patterns emerge
- CI/CD pipeline updates

**Related files:**
- `.github/workflows/`: CI/CD workflows
- `pytest.ini` or `pyproject.toml`: Pytest configuration
