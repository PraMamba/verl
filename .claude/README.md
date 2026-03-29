# verl Claude Code Configuration

This directory contains AI assistant configuration for the verl codebase, adapted from AReaL's configuration with targeted modifications for verl's architecture.

## Structure

```
.claude/
├── agents/           # Expert agents for specialized domains
│   ├── planner.md              # [Opus, Proactive] Implementation planning
│   ├── algorithm-expert.md     # [Opus] RL algorithms
│   ├── fsdp-engine-expert.md   # [Opus] FSDP2 training backend
│   ├── megatron-engine-expert.md # [Opus] Megatron-LM integration
│   ├── vllm-sglang-expert.md   # [Opus] Inference engines
│   ├── ray-controller-expert.md # [Opus] Ray orchestration
│   ├── simple-code-reviewer.md # [Sonnet, Proactive] Quick code review
│   └── code-verifier.md        # [Haiku, Proactive] Linting/testing
├── commands/         # Custom CLI commands
│   ├── create-pr.md            # /create-pr - Full PR workflow
│   ├── review-pr.md            # /review-pr - Dynamic PR review
│   └── gen-commit-msg.md       # /gen-commit-msg - Commit messages
├── data/             # Reference data for commands
│   ├── review-pr-change-types.md # Change type detection tables
│   └── review-pr-templates.md    # Review task templates
├── hooks/            # Post-tool-use hooks
│   └── check-expert-update.sh  # Expert doc update reminder
├── rules/            # Code style & standards (steering files)
│   ├── code-style.md           # Design principles, naming, logging
│   ├── distributed.md          # Distributed training patterns
│   ├── testing.md              # Test organization and patterns
│   └── api-config.md           # Hydra configuration guidelines
├── skills/           # Step-by-step guides
│   ├── add-reward.md           # Add new reward functions
│   ├── add-dataset.md          # Add new dataset loaders
│   ├── add-worker.md           # Add new Ray workers
│   ├── add-unit-tests/SKILL.md # Add unit tests
│   └── debug-distributed.md    # Debug distributed issues
├── settings.json     # Hook configuration
└── README.md         # This file
```

## Expert Agents

Located in `agents/`, these provide specialized expertise:

| Agent | Model | Activation | Purpose |
|-------|-------|-----------|---------|
| **planner** | Opus | Proactive | Implementation planning for complex tasks |
| **algorithm-expert** | Opus | Manual | RL algorithms (PPO, GRPO, RLOO, etc.) |
| **fsdp-engine-expert** | Opus | Manual | FSDP2 training backend |
| **megatron-engine-expert** | Opus | Manual | Megatron-LM integration |
| **vllm-sglang-expert** | Opus | Manual | vLLM & SGLang inference engines |
| **ray-controller-expert** | Opus | Manual | Ray single-controller orchestration |
| **simple-code-reviewer** | Sonnet | Proactive | Quick code review after changes |
| **code-verifier** | Haiku | Proactive | Formatting, linting, testing checks |

## Commands

Located in `commands/`, these are invoked with `/command-name`:

- **`/create-pr`**: Full PR workflow - rebase, squash, push, create/update PR with verl's PR template
- **`/review-pr`**: Dynamic PR review with intelligent agent allocation based on change types
- **`/gen-commit-msg`**: Generate conventional commit messages from staged changes

## Rules (Steering Files)

Located in `rules/`, these define coding standards and are automatically included:

- **code-style.md**: Design principles (composition over inheritance, Ray single-controller, DataProto), naming conventions, logging, performance
- **distributed.md**: Distributed training patterns (FSDP2, process groups, device mesh, DTensor)
- **testing.md**: Test organization (unit/e2e/sanity/distributed), markers, assertions
- **api-config.md**: Hydra configuration patterns, dataclass integration

## Skills

Located in `skills/`, these provide step-by-step guides:

- **add-reward.md**: Add new reward functions
- **add-dataset.md**: Add new dataset loaders
- **add-worker.md**: Add new Ray workers
- **add-unit-tests/SKILL.md**: Add unit tests with verl conventions
- **debug-distributed.md**: Debug distributed training issues

## Data Files

Located in `data/`, reference data for commands:

- **review-pr-change-types.md**: Change type detection tables (CRITICAL/HIGH/MEDIUM/LOW) with framework-specific risks
- **review-pr-templates.md**: Review task templates for different change types

## Hooks

Located in `hooks/`:

- **check-expert-update.sh**: Reminds to update expert docs when related code changes

Configured in `settings.json` to run after Write/Edit operations.

## Adaptation from AReaL

This configuration was adapted from AReaL's `.claude` setup with the following changes:

### Removed (AReaL-specific)
- **archon-engine-expert**: AReaL's MoE engine (not in verl)
- **launcher-scheduler-expert**: AReaL's infra layer (verl uses Ray)
- **add-archon-model skill**: AReaL-specific
- **add-workflow skill**: AReaL's RolloutWorkflow pattern

### Added (verl-specific)
- **vllm-sglang-expert**: verl's inference engines
- **ray-controller-expert**: verl's Ray orchestration pattern
- **add-worker skill**: verl's worker-based architecture
- **add-unit-tests skill**: verl's test organization

### Migrated from AReaL with verl adaptations
- **simple-code-reviewer**: Adapted for verl patterns (Ray, DataProto, workers)
- **code-verifier**: Adapted for verl tooling (pytest, Ruff, mypy)
- **create-pr command**: Adapted for verl's PR template and module naming
- **review-pr command**: Adapted for verl's architecture (Ray workers, FSDP, vLLM/SGLang)
- **gen-commit-msg command**: Adapted scopes for verl directory structure
- **review-pr data files**: Change types and templates for verl's codebase

### Key Differences from AReaL
- verl uses **Ray single-controller** vs AReaL's direct distributed approach
- verl uses **DataProto protocol** for all inter-worker data transfer
- verl has **worker-based architecture** (actor/critic/rollout separation)
- verl uses **Hydra** for configuration vs AReaL's dataclass-first approach
- verl uses **pip/setuptools** vs AReaL's uv build system
- verl uses **mypy** for type checking (AReaL does not)
- All agents have **YAML frontmatter** (name, description, tools, model)

## Maintenance

### When to Update Agents
- verl architecture evolves (new worker types, patterns)
- Common issues and solutions identified
- Framework APIs evolve (Ray, PyTorch, vLLM, SGLang)

### When to Update Rules
- Coding standards change
- New best practices discovered
- Framework APIs evolve

### When to Update Skills
- New common tasks identified
- Existing workflows improve
- User feedback on clarity

### When to Update Commands
- PR template changes
- New module scopes added
- Review change types evolve
