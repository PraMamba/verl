---
name: add-dataset
description: Use when adding a dataset loader or adapting a data format for verl.
---

# Add Dataset Loader

## Purpose
Use when adding a dataset loader or adapting a data format for verl.

## Use When
- The user asks for this workflow in natural language.
- The task touches the corresponding verl development area.

## Steps
1. Identify the existing data loading pattern closest to the new format.
2. Create or adapt the dataset class using `torch.utils.data.Dataset` conventions.
3. Return fields required by the training mode, usually `messages` plus optional answer/metadata for RL rewards.
4. Handle JSONL/Parquet/HuggingFace/multiturn parsing explicitly and fail with clear errors.
5. Add unit tests for loading, `__len__`, `__getitem__`, malformed rows, and DataLoader compatibility.

## verl Constraints
- Keep tokenization and max-length behavior consistent with existing data utilities.
- Do not add new data dependencies unless explicitly approved.

## Verification
Run targeted dataset tests and a small DataLoader smoke test when practical.

## Notes
This is a Codex-native skill. It intentionally does not define Claude slash commands or hooks.
