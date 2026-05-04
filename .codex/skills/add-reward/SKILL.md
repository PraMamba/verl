---
name: add-reward
description: Use when adding or modifying a verl reward function.
---

# Add Reward Function

## Purpose
Use when adding or modifying a verl reward function.

## Use When
- The user asks for this workflow in natural language.
- The task touches the corresponding verl development area.

## Steps
1. Locate the integration point: custom reward function path/config or existing reward manager behavior.
2. Implement a batch-safe reward function with clear semantics and empty-response handling.
3. Keep return shape `[batch_size]`; normalize values to a documented range such as `[0, 1]` or `[-1, 1]`.
4. Register or configure the reward using existing verl config patterns; do not introduce new framework dependencies.
5. Add tests for basic scoring, empty completions, batch sizes, and edge cases.

## verl Constraints
- Preserve DataProto boundaries when rewards interact with trainer/worker data.
- Avoid slow Python or external I/O in hot reward paths unless explicitly required.
- Document expected kwargs such as ground truth or metadata.

## Verification
Run targeted reward tests and any relevant trainer/config parse checks.

## Notes
This is a Codex-native skill. It intentionally does not define Claude slash commands or hooks.
