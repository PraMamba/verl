---
name: prepare-pr
description: Use when preparing a branch for PR without preserving Claude slash-command shape.
---

# Prepare Pull Request

## Purpose
Use when preparing a branch for PR without preserving Claude slash-command shape.

## Use When
- The user asks for this workflow in natural language.
- The task touches the corresponding verl development area.

## Safety
Rebase, squash, force-push, and PR mutation are destructive or history-rewriting actions. Ask for explicit user confirmation before performing them.

## Steps
1. Inspect branch, status, existing PR, and target base.
2. Ensure no unrelated uncommitted changes will be included.
3. Fetch/rebase only after confirmation when history rewriting is involved.
4. Generate a PR title using verl module scopes such as `worker`, `rollout`, `trainer`, `ray`, `cfg`, `reward`, `data`, `model`, or `docs`.
5. Draft a PR body with summary, tests, risks, and compatibility notes.
6. Create or update the PR with `gh` only when credentials and confirmation are available.

## Verification
Show branch status, final commits, PR URL if created, and exact tests run.

## Notes
This is a Codex-native skill. It intentionally does not define Claude slash commands or hooks.
