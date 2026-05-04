# verl Codex Configuration

This directory contains Codex-native assistance artifacts for the verl repository. It is derived from the useful knowledge in `.claude/`, but it is **not** a 1:1 Claude Code migration.

## What Codex loads here

```text
.codex/
├── config.toml          # Project-scoped Codex config and registrations
├── agents/              # Project custom agents as standalone TOML files
├── skills/              # Codex skills, each with SKILL.md
└── README.md            # This guide
```

## Important behavior difference from Claude Code

Codex custom agents/subagents do **not** run automatically just because files changed. Ask Codex explicitly when you want a project agent used, for example:

- "Use `verl_planner` to plan this multi-file change."
- "Have `verl_code_reviewer` review this branch for distributed/Ray/DataProto issues."
- "Ask `verl_fsdp_expert` to check this FSDP2 design."

Skills can be invoked by natural language, for example:

- "Use the add-reward skill to guide this reward function change."
- "Use review-pr to review this branch."
- "Use prepare-pr to draft a PR."

## Agents

| Agent | Purpose |
| --- | --- |
| `verl_planner` | Read-only planning for multi-file features and architecture decisions. |
| `verl_code_reviewer` | Read-only code review for verl-specific correctness risks. |
| `verl_code_verifier` | Verification planning/runs, depending on parent session permissions. |
| `verl_algorithm_expert` | PPO/GRPO/RLOO/reward/advantage/loss expertise. |
| `verl_fsdp_expert` | FSDP/FSDP2/DTensor/device-mesh expertise. |
| `verl_megatron_expert` | Megatron-Core and model-parallel training expertise. |
| `verl_rollout_expert` | vLLM/SGLang rollout and generation expertise. |
| `verl_ray_controller_expert` | Ray single-controller, worker, and dispatch expertise. |

## Skills

| Skill | Purpose |
| --- | --- |
| `add-reward` | Add or modify reward functions. |
| `add-dataset` | Add dataset loaders or data-format adapters. |
| `add-worker` | Add Ray workers following verl conventions. |
| `add-unit-tests` | Add targeted tests and choose the right test category. |
| `debug-distributed` | Diagnose distributed hangs, NCCL errors, OOM, and wrong results. |
| `prepare-pr` | Prepare PR title/body/workflow safely. |
| `generate-commit-message` | Generate a commit message from staged changes. |
| `review-pr` | Review PRs/branches with verl-specific risk classification. |

## Intentionally omitted

- Hooks and hook configuration.
- Claude slash-command files or slash-command semantics.
- Wholesale copies of `.claude/docs`.
- New external dependencies or helper scripts.

## Maintenance

Update these artifacts when verl architecture changes, Codex custom-agent schema changes, or repeated review/implementation patterns reveal better guidance. Keep the content concise and prefer Codex-native behavior over Claude compatibility shims.
