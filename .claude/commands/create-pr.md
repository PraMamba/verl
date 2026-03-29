---
name: create-pr
description: Rebase from the latest `origin/main`, squash the commits from it, and then create a PR on github with intelligent commit messages based on staged changes. Invoke with /create-pr.
---

# Create Pull Request

Rebase from the latest `origin/main`, squash commits, and create a PR on GitHub with an
intelligent title and description.

## Usage

```
/create-pr [--draft] [--base <branch>]
```

**Arguments:**

- `--draft`: Create as draft PR
- `--base <branch>`: Target branch (default: `main`)

## Workflow

### Step 1: Verify Prerequisites

```bash
# Check current branch
git branch --show-current

# Check if on main/master (should NOT be)
if [[ $(git branch --show-current) == "main" || $(git branch --show-current) == "master" ]]; then
  echo "ERROR: Cannot create PR from main/master branch"
  exit 1
fi

# Check for uncommitted changes
git status --short

# Ensure gh CLI is available
gh --version
```

**Action:** If there are uncommitted changes, stop, and then ask user to commit or stash
them first.

### Step 2: Check for Existing PR

```bash
# Check if PR already exists for current branch
gh pr view --json number,title,url 2>/dev/null || echo "No existing PR"
```

**Handle Existing PR:**

- If PR exists, inform user and ask permission to force-update it
- Warn that this will rewrite the commit history and PR description
- If user declines, abort the process

### Step 3: Fetch and Rebase

```bash
# Fetch latest from origin
git fetch origin main

# Check divergence
git log --oneline HEAD ^origin/main

# Non-interactive rebase onto origin/main
git rebase origin/main
```

**Handle Conflicts:** If rebase fails due to conflicts, abort and let user handle rebase
manually:

```bash
# On rebase failure, abort automatically
git rebase --abort

# Inform user to resolve conflicts manually
echo "Rebase failed due to conflicts. Please resolve manually and retry /create-pr"
exit 1
```

### Step 4: Squash Commits into Single Commit

After successful rebase, squash all commits since `origin/main` into a single commit:

```bash
# Count commits to squash
git rev-list --count origin/main..HEAD

# Soft reset to origin/main (keeps changes staged)
git reset --soft origin/main

# Generate commit message using /gen-commit-msg logic
# See .claude/commands/gen-commit-msg.md for message generation rules
```

**Generate Commit Message** (following `/gen-commit-msg` format):

1. Analyze staged changes:

   ```bash
   git diff --cached --name-only
   git diff --cached
   ```

1. Categorize changes (feat/fix/docs/refactor/test/chore/perf)

1. Determine scope from changed files (workers/trainer/reward/data/models/utils/etc.)

1. Generate message in format:

   ```
   <type>(<scope>): <subject>

   <body>

   [Optional sections:]
   Key changes:
   - change 1
   - change 2

   Refs: #123, #456
   ```

1. Commit with generated message:

   ```bash
   git commit -m "$(cat <<'EOF'
   <generated commit message>
   EOF
   )"
   ```

### Step 5: Analyze Combined Changes

After squashing into a single commit:

```bash
# Get all changes since origin/main
git diff origin/main...HEAD --name-only

# Get full diff content
git diff origin/main...HEAD

# Check commit history
git log --oneline origin/main..HEAD
```

**Categorize Changes:**

Follow same categorization as `/gen-commit-msg`:

| Type       | When to Use                     |
| ---------- | ------------------------------- |
| `feat`     | New feature or capability       |
| `fix`      | Bug fix                         |
| `docs`     | Documentation only              |
| `refactor` | Code change without feature/fix |
| `test`     | Adding or fixing tests          |
| `chore`    | Build, deps, config changes     |
| `perf`     | Performance improvement         |

**Determine Scope (verl modules):**

Infer from changed files:

- `verl/workers/` → `worker`
- `verl/workers/rollout/` → `rollout`
- `verl/trainer/` → `trainer`
- `verl/reward/` → `reward`
- `verl/data/` or `verl/utils/dataset/` → `data`
- `verl/models/` → `model`
- `verl/utils/fsdp_utils/` → `fsdp`
- `verl/models/mcore/` or `verl/utils/megatron_utils/` → `megatron`
- `verl/single_controller/` → `ray`
- `verl/checkpoint_engine/` → `ckpt`
- `verl/trainer/config/` → `cfg`
- `docs/` → `doc`
- `examples/` → `examples`
- Multiple areas → combine with comma, e.g., `[fsdp, megatron]`

### Step 6: Generate PR Title and Description

**PR Title Format** (per verl convention):

```
[{modules}] {type}: {brief description}
```

**Rules:**

- Keep under 70 characters
- Use imperative mood
- No period at end
- Modules in brackets, comma-separated
- If breaking API: prefix with `[BREAKING]`

**PR Description Format:**

MUST strictly follow the [GitHub PR template](../../.github/PULL_REQUEST_TEMPLATE.md):

```markdown
### What does this PR do?

> <2-4 sentences explaining what this PR does and why>

### Checklist Before Starting

- [x] Search for similar PRs. Paste at least one query link here: ...
- [x] Format the PR title as `[{modules}] {type}: {description}`

### Test

> <Validation approach and results>

### API and Usage Example

> <Demonstrate API changes if any>

```python
# Add code snippet if applicable
```

### Design & Code Changes

> <High-level design and specific changes list>

### Checklist Before Submitting

- [x] Read the [Contribute Guide](https://github.com/volcengine/verl/blob/main/CONTRIBUTING.md).
- [x] Apply [pre-commit checks](https://github.com/volcengine/verl/blob/main/CONTRIBUTING.md#code-linting-and-formatting)
- [ ] Add / Update documentation
- [ ] Add unit or end-to-end tests to CI workflow
- [x] My branch is up to date with main
- [ ] If your PR is related to the `recipe` submodule, update the reference
```

### Step 7: Push and Create/Update PR

Show preview to user, then **confirm with user** before executing:

```bash
# Force push branch to remote (required after squash)
git push -f -u origin $(git branch --show-current)

# Create or edit PR using gh CLI
if gh pr view &>/dev/null; then
  # Update existing PR
  gh pr edit \
    --title "[workers] feat: add CPU offload support" \
    --body "$(cat <<'EOF'
[PR description here]
EOF
)"
else
  # Create new PR
  gh pr create \
    --base main \
    --title "[workers] feat: add CPU offload support" \
    --body "$(cat <<'EOF'
### What does this PR do?

[description]

### Checklist Before Starting

- [x] Search for similar PRs.
- [x] Format the PR title as `[{modules}] {type}: {description}`

### Test

[test details]

### Design & Code Changes

[design details]

### Checklist Before Submitting

- [x] Read the Contribute Guide
- [x] Apply pre-commit checks
- [ ] Add / Update documentation
- [ ] Add unit or end-to-end tests
- [x] My branch is up to date with main
EOF
)"
fi
```

Add `--draft` flag if requested.

**Capture PR URL** and display to user:

```
✓ PR created/updated successfully!
https://github.com/volcengine/verl/pull/123
```

## Error Handling

### Rebase Conflicts

If rebase fails:

1. Show conflict files
1. Provide resolution instructions
1. Wait for user to resolve
1. After resolution, continue with squashing step
1. Offer to abort rebase if needed: `git rebase --abort`

### Push Failures

If force push fails:

1. Verify remote branch exists
1. Check GitHub authentication: `gh auth status`
1. Confirm branch protection rules allow force push
1. Provide manual push instructions if needed

### PR Creation/Update Failures

If `gh pr create` or `gh pr edit` fails:

1. Check if PR already exists: `gh pr view`
1. Verify GitHub authentication: `gh auth status`
1. Check for branch protection rules
1. Provide manual PR creation/update link

## Safety Checks

**Before Starting:**

- Confirm no uncommitted changes
- Confirm not on main/master branch
- Check for existing PR and get user permission to overwrite if exists
- Backup branch: `git branch backup/$(git branch --show-current)-$(date +%s)`

**Before Rebase:**

- Fetch latest from origin
- Show divergence summary

**Before Squash:**

- Show commits that will be squashed
- Confirm user wants to proceed

**Before Force Push:**

- **CRITICAL**: Warn user that force push will rewrite history
- Show current commit that will replace remote history
- Confirm branch name
- If PR exists, emphasize that PR history will be rewritten

**Before PR Creation/Update:**

- Show full preview of title/description
- Confirm target branch
- If updating existing PR, show what will change

______________________________________________________________________

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: .claude/commands/create-pr.md
Invocation: /create-pr

## Design Philosophy

- Automates full PR creation workflow: fetch, rebase, **squash to single commit**, push, create/update PR
- **Always squashes all commits** since `origin/main` into a single commit
- **Handles existing PRs** by detecting them and force-updating after user permission
- Follows verl's PR title convention: `[{modules}] {type}: {description}`
- Requires user confirmation at critical steps
- Uses force push (`-f`) by design, as squashing requires rewriting history

## How to Update

### Adding New Scopes/Modules
Update "Determine Scope" section with new file path mappings.

### Changing PR Template
Update "PR Description Format" to match any changes to .github/PULL_REQUEST_TEMPLATE.md.

### Modifying Workflow Steps
Update relevant "Step N" sections with new git commands or logic.

================================================================================
-->
