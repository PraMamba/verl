---
name: add-worker
description: Use when adding or changing a Ray worker in verl.
---

# Add Ray Worker

## Purpose
Use when adding or changing a Ray worker in verl.

## Use When
- The user asks for this workflow in natural language.
- The task touches the corresponding verl development area.

## Steps
1. Confirm the controller owns orchestration; workers must not communicate directly with each other.
2. Model the worker API around `DataProto` inputs/outputs where data crosses worker boundaries.
3. Use existing `Worker` and `@register(dispatch_mode=...)` patterns.
4. Configure resources deliberately (`num_gpus`, CPU needs, placement assumptions).
5. Add focused tests for dispatch, DataProto shape, and controller integration boundaries.

## verl Constraints
- Follow `XxxWorker` naming.
- Avoid global process groups and hidden distributed side effects.
- Large shared objects should go through Ray object-store references where appropriate.

## Verification
Run targeted worker/controller tests; GPU or Ray-cluster tests may require environment-specific skips.

## Notes
This is a Codex-native skill. It intentionally does not define Claude slash commands or hooks.
