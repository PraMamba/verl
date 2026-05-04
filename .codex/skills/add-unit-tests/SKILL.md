---
name: add-unit-tests
description: Use when adding tests for new functionality or increasing coverage.
---

# Add Unit Tests

## Purpose
Use when adding tests for new functionality or increasing coverage.

## Use When
- The user asks for this workflow in natural language.
- The task touches the corresponding verl development area.

## Steps
1. Choose the smallest test layer that proves behavior: CPU unit, sanity, distributed, GPU, or e2e.
2. Follow Arrange-Act-Assert and clear `test_<module>_<behavior>` naming.
3. Add parametrization for important shapes, batch sizes, and edge cases.
4. Use skip markers for CUDA/Ray/Megatron/vLLM/SGLang availability rather than silently weakening tests.
5. Keep tests deterministic and small enough for CI category expectations.

## verl Constraints
- CPU-only tests should use `test_*_on_cpu.py` where project conventions require.
- Distributed tests must not assume a global process group.

## Verification
Run the new tests directly, then run the nearest existing related test subset.

## Notes
This is a Codex-native skill. It intentionally does not define Claude slash commands or hooks.
