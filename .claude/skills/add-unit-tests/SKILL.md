---
name: add-unit-tests
description: Guide for adding unit tests to verl. Use when user wants to add tests for new functionality or increase test coverage.
---

# Add Unit Tests

Add unit tests to verl following the project's testing conventions.

## When to Use

This skill is triggered when:

- User asks "how do I add tests?"
- User wants to increase test coverage
- User needs to write tests for new functionality
- User wants to understand verl testing patterns

## Step-by-Step Guide

### Step 1: Understand Test Types

verl has organized test categories:

| Test Type             | Purpose                            | Location                           | How It Runs                      |
| --------------------- | ---------------------------------- | ---------------------------------- | -------------------------------- |
| **Unit Tests**        | Test individual functions/modules  | `tests/unit/`                      | Directly via pytest              |
| **Sanity Checks**     | Quick validation of basic behavior | `tests/special_sanity/`            | Via pytest                       |
| **E2E Tests**         | End-to-end training tests          | `tests/special_e2e/`              | Via pytest (requires GPU)        |
| **Distributed Tests** | Test distributed/parallel behavior | `tests/special_distributed/`       | Via torchrun / pytest subprocess |
| **Worker Tests**      | Test Ray worker implementations    | `tests/workers/`                   | Via pytest (may require GPU)     |

### Step 2: Create Test File Structure

Create test file with naming convention: `test_<module>_<feature>.py` (or `test_<module>_on_cpu.py` for CPU tests)

```python
import pytest
import torch

# Import the module to test
from verl.protocol import DataProto
from verl.workers.config.fsdp_engine import FSDPEngineConfig
```

### Step 3: Write Test Functions

Follow Arrange-Act-Assert pattern:

```python
def test_function_under_condition_returns_expected():
    """Test that function returns expected value under condition."""
    # Arrange
    input_data = torch.tensor([1.0, 2.0, 3.0])
    expected_output = torch.tensor([2.0, 4.0, 6.0])

    # Act
    result = function_under_test(input_data)

    # Assert
    torch.testing.assert_close(result, expected_output, rtol=1e-5, atol=1e-8)
```

### Step 4: Add Pytest Markers

Use appropriate pytest markers:

| Marker                                  | When to Use                                   |
| --------------------------------------- | --------------------------------------------- |
| `@pytest.mark.slow`                     | Test takes > 10 seconds                       |
| `@pytest.mark.asyncio`                  | Async test functions                          |
| `@pytest.mark.skipif(cond, reason=...)` | Conditional skip                              |
| `@pytest.mark.parametrize(...)`         | Parameterized tests                           |
| `@pytest.mark.gpu`                      | Requires GPU                                  |
| `@pytest.mark.multi_gpu`               | Requires multiple GPUs                        |
| `@pytest.mark.ray`                      | Requires Ray cluster                          |
| `@pytest.mark.vllm`                    | Requires vLLM                                 |
| `@pytest.mark.sglang`                  | Requires SGLang                               |

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_feature():
    tensor = torch.tensor([1, 2, 3], device="cuda")
    # ... assertions

@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_with_parameters(batch_size):
    # Parameterized test
    pass
```

### Step 5: Mock Distributed Environment

For unit tests that need distributed mocks:

```python
import torch.distributed as dist

def test_distributed_function(monkeypatch):
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    result = distributed_function()
    assert result == expected
```

### Step 6: Handle GPU Dependencies

Always skip gracefully when GPU unavailable:

```python
CUDA_AVAILABLE = torch.cuda.is_available()

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_gpu_function():
    tensor = torch.tensor([1, 2, 3], device="cuda")
    # ... assertions
    torch.cuda.empty_cache()  # Clean up
```

### Step 7: Ray Worker Tests

For testing Ray workers:

```python
import ray

@pytest.fixture(scope="module")
def ray_cluster():
    if not ray.is_initialized():
        ray.init(num_cpus=4, num_gpus=torch.cuda.device_count())
    yield
    ray.shutdown()

def test_worker_function(ray_cluster):
    from verl.single_controller.base import Worker

    @ray.remote
    class MockWorker(Worker):
        def compute(self):
            return 42

    worker = MockWorker.remote()
    result = ray.get(worker.compute.remote())
    assert result == 42
```

## Key Requirements (Based on testing.md)

### Mocking Distributed

- Use `torch.distributed.fake_pg` for unit tests
- Mock `dist.get_rank()` and `dist.get_world_size()` explicitly
- Don't mock internals of FSDP/DTensor

### GPU Test Constraints

- **Always skip gracefully** when GPU unavailable
- Clean up GPU memory: `torch.cuda.empty_cache()` in fixtures
- Use smallest possible model/batch for unit tests

### Assertions

- Use `torch.testing.assert_close()` for tensor comparison
- Specify `rtol`/`atol` explicitly for numerical tests
- Avoid bare `assert tensor.equal()` - no useful error message

### DataProto Tests

```python
def assert_dataproto_equal(proto1, proto2):
    assert len(proto1) == len(proto2)
    for key in proto1.keys():
        torch.testing.assert_close(proto1[key], proto2[key])
```

## Reference Implementations

| Test File                              | Description                | Key Patterns                |
| -------------------------------------- | -------------------------- | --------------------------- |
| `tests/unit/`                          | Unit tests                 | Fixtures, parametrized      |
| `tests/special_sanity/`               | Sanity checks              | Quick validation            |
| `tests/special_e2e/`                  | E2E training tests         | Full pipeline tests         |
| `tests/workers/`                       | Worker tests               | Ray fixtures, DataProto     |

## Common Mistakes

- ❌ **Missing test file registration**: Ensure file follows `test_*.py` naming
- ❌ **GPU dependency without skip**: Always use `@pytest.mark.skipif` for GPU tests
- ❌ **Incorrect tensor comparisons**: Use `torch.testing.assert_close()` not
  `assert tensor.equal()`
- ❌ **Memory leaks in GPU tests**: Clean up with `torch.cuda.empty_cache()`
- ❌ **Mocking too much**: Don't mock FSDP/DTensor internals
- ❌ **Unclear test names**: Follow `test_<what>_<condition>_<expected>` pattern
- ❌ **No docstrings**: Add descriptive docstrings to test functions
- ❌ **Global process groups**: Never use default process group in tests

## Integration with Other Skills

This skill complements other verl development skills:

- **After `/add-dataset`**: Add tests for new dataset loaders
- **After `/add-worker`**: Add tests for new workers
- **After `/add-reward`**: Add tests for new reward functions
- **With `planner` agent**: Reference this skill when planning test implementation

## Running Tests

```bash
# First check GPU availability (many tests require GPU)
python -c "import torch; print('GPU available:', torch.cuda.is_available())"

# Run unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_<name>.py -v

# Run with timeout
pytest tests/unit/ -v --timeout=60

# Run sanity checks
pytest tests/special_sanity/ -v

# Run distributed tests (requires torchrun and multi-GPU)
torchrun --nproc_per_node=2 tests/special_distributed/run_<test>.py
```

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: .claude/skills/add-unit-tests/SKILL.md
Invocation: /add-unit-tests

## Purpose

Step-by-step guide for adding unit tests to verl.

## How to Update

### When Testing Conventions Change
1. Update "Key Requirements" section based on `testing.md`
2. Update test examples to match new patterns
3. Update reference implementations

### When Test Types Need Update
1. Update "Understand Test Types" table
2. Add new examples if needed
3. Update common mistakes

### Integration with Other Skills
Ensure references to other skills (`/add-dataset`, `/add-worker`, `/add-reward`) remain accurate.

================================================================================
-->
