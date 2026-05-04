---
name: review-pr
description: Use when reviewing a PR or branch with dynamic focus on verl risk areas.
---

# Review Pull Request

## Purpose
Use when reviewing a PR or branch with dynamic focus on verl risk areas.

## Use When
- The user asks for this workflow in natural language.
- The task touches the corresponding verl development area.

## Steps
1. Identify the PR or compare current branch against the base branch.
2. Classify changed files by risk: distributed/FSDP/Megatron/trainer core as high risk; docs/tests/config as lower risk unless they affect runtime behavior.
3. Map each risk to the right review lens: correctness, distributed safety, API/config compatibility, test adequacy, performance, and documentation.
4. Spawn or ask for specialized Codex agents only when explicitly requested or clearly useful; summarize each finding with evidence.
5. Report findings by severity with file references, impact, and concrete fix suggestions.

## verl Review Checklist
- DataProto for worker transfer.
- Ray single-controller ownership.
- Explicit process groups for collectives.
- Hydra config compatibility and generated config drift.
- vLLM/SGLang rollout compatibility.
- Tests cover CPU/GPU/distributed boundaries as applicable.

## Verification
State reviewed diff range, commands used, and residual unverified areas.

## Notes
This is a Codex-native skill. It intentionally does not define Claude slash commands or hooks.
