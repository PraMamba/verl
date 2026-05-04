---
name: debug-distributed
description: Use when diagnosing hangs, NCCL errors, OOM, wrong results, or distributed flakiness.
---

# Debug Distributed Training

## Purpose
Use when diagnosing hangs, NCCL errors, OOM, wrong results, or distributed flakiness.

## Use When
- The user asks for this workflow in natural language.
- The task touches the corresponding verl development area.

## Steps
1. Classify the symptom: hang/deadlock, numeric wrongness, OOM, NCCL/process-group error, or Ray scheduling issue.
2. Reproduce with the smallest rank/world-size/config that still fails.
3. Check collective ordering and ensure every collective uses the intended explicit process group.
4. Check device placement, mesh dimensions, DataProto batch splits, and Ray worker resources.
5. Add temporary rank-aware diagnostics only as local debugging; remove or convert to proper logging before completion.

## verl Constraints
- Never create or rely on implicit global process groups for new code.
- Prefer `logging.getLogger(__file__)` over `print` in committed diagnostics.

## Verification
Rerun the reproducer, then the closest distributed/sanity test. Document environment gaps if multi-GPU verification is unavailable.

## Notes
This is a Codex-native skill. It intentionally does not define Claude slash commands or hooks.
