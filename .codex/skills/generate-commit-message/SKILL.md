---
name: generate-commit-message
description: Use when generating a commit message from staged changes.
---

# Generate Commit Message

## Purpose
Use when generating a commit message from staged changes.

## Use When
- The user asks for this workflow in natural language.
- The task touches the corresponding verl development area.

## Steps
1. Inspect staged files with `git diff --cached --name-only` and staged content with `git diff --cached`.
2. Infer conventional type: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, or `perf`.
3. Infer scope from paths, for example `workers`, `trainer`, `rollout`, `reward`, `data`, `models`, `utils`, `ray`, `cfg`, `docs`, `tests`, or `ci`.
4. Produce a concise subject and body listing key changes and verification.
5. Do not commit unless the user explicitly asks.

## Output Format
```text
<type>(<scope>): <subject>

<body>

Tested: <commands or not run>
```

## Verification
Confirm staged files match the generated message.

## Notes
This is a Codex-native skill. It intentionally does not define Claude slash commands or hooks.
