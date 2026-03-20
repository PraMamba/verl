## WHAT: Project Overview

verl is a flexible, high-performance RL training framework for LLM post-training
(RLHF/RLAIF), built on a Ray single-controller architecture with pluggable training
backends and inference engines.

**Tech Stack**: Python 3.10+ | PyTorch | Ray | FSDP2/Megatron-LM | vLLM/SGLang/TRT-LLM | Hydra

**Core Directories**:

- `verl/` - Core package
  - `protocol.py` - `DataProto` dataclass: the universal data interchange protocol
  - `base_config.py` - `BaseConfig` typed configuration base class
  - `single_controller/` - Ray single-controller pattern implementation
    - `base/` - `Worker` base class, `@register` decorator, `Dispatch` modes
    - `ray/` - `RayWorkerGroup`, `create_colocated_worker_cls` for hybrid engine
  - `workers/` - Distributed worker implementations
    - `fsdp_workers.py` - FSDP2-based Actor, Critic, Reference workers
    - `megatron_workers.py` - Megatron-LM-based workers
    - `engine_workers.py` - Generic engine-based workers (TorchTitan, etc.)
    - `rollout/` - Inference engine adapters (vLLM, SGLang, TRT-LLM, HF)
    - `sharding_manager/` - Hybrid engine memory management (FSDP/Megatron)
    - `reward_manager/` - Reward computation orchestration
  - `trainer/` - Training orchestration
    - `ppo/ray_trainer.py` - `RayPPOTrainer`: main training loop (10-step cycle)
    - `ppo/core_algos.py` - Core RL algorithms (PPO, GRPO, RLOO, REINFORCE++, etc.)
    - `config/` - Hydra YAML configs (actor/, critic/, rollout/, reward/, algorithm/)
    - `main_ppo.py` - Hydra entry point via `TaskRunner.run()`
  - `models/` - Model utilities
    - `mcore/` - Megatron-Core model wrappers and checkpoint conversion
  - `reward/` - Reward function implementations
  - `data/` - Dataset loaders
  - `utils/` - Cross-cutting utilities
    - `fsdp_utils.py` - FSDP2 sharding, checkpoint, mixed precision
    - `megatron_utils/` - Megatron initialization and utilities
    - `config.py` - `omega_conf_to_dataclass()`, `validate_config()`
- `examples/` - Training scripts, configs, and tutorials
- `docs/` - Sphinx documentation source
- `tests/` - Test suites (unit, sanity, e2e, distributed, workers, npu)

## WHY: Purpose

- Enable flexible RL post-training for LLMs with multiple algorithm choices
  (PPO, GRPO, RLOO, REINFORCE++, DAPO, VAPO, and more)
- Ray single-controller architecture: one CPU controller orchestrates distributed GPU
  workers without touching GPU tensors, enabling clean separation of concerns
- Hybrid engine design: time-share GPUs between training (FSDP/Megatron) and inference
  (vLLM/SGLang) via sleep/resume weight synchronization
- Pluggable backends: training engines, inference engines, reward functions, and datasets
  are independently extensible

## HOW: Core Commands

```bash
# Check environment
python --version                    # Requires 3.10+

# Install verl (choose one inference backend)
pip install -e .[test,vllm]         # With vLLM
pip install -e .[test,sglang]       # With SGLang

# Pre-commit hooks
pip install pre-commit              # Install pre-commit (once)
pre-commit install                  # Set up hooks (once)
pre-commit run --all-files          # Format and lint (ruff, mypy, etc.)

# Run tests
# First check GPU availability (many tests require GPU)
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
pytest tests/unit/ -v --timeout=60                      # CPU unit tests
pytest tests/special_sanity/ -v                         # Sanity checks
pytest tests/special_e2e/ -v                            # E2E (requires GPU)
torchrun --nproc_per_node=2 tests/special_distributed/  # Distributed (multi-GPU)

# Run training
python -m verl.trainer.main_ppo \
  trainer.total_epochs=10 \
  model.path=/path/to/model \
  data.train_files=/path/to/data

# Build docs
cd docs && pip install -r requirements-docs.txt && make html
```

## Boundaries

### Constraints

- Designed for distributed GPU clusters; supports NVIDIA, AMD, and Ascend NPU
- Many tests require GPU hardware; skip gracefully with `@pytest.mark.skipif`
- Ray must be initialized for worker-based tests
- Secrets, model paths, and cluster configs are managed outside the repo

### Always Do

- Read relevant files before modifying code
- Run `pre-commit run --all-files` before committing
- Follow existing code patterns in the same module
- Use `DataProto` for all data transfer between workers
- Use explicit process groups for all collective operations (`group=pg`)
- Use `logging.getLogger(__file__)` with `VERL_LOGGING_LEVEL` env var (never `print`)
- Add tests for new functionality (Arrange-Act-Assert pattern)
- Use `torch.testing.assert_close()` for tensor comparisons
- Clean up GPU memory with `torch.cuda.empty_cache()` in tests

### Ask First

- Modifying `DataProto` protocol in `verl/protocol.py`
- Changing the Ray single-controller architecture in `verl/single_controller/`
- Modifying Hydra config structures in `verl/trainer/config/`
- Adding new dependencies to `pyproject.toml`
- Changing worker dispatch modes or `@register` decorators
- Deleting or renaming public APIs
- Running GPU/distributed tests (check GPU first:
  `python -c "import torch; print('GPU available:', torch.cuda.is_available())"`)

### Never Do

- Create global process groups (`dist.init_process_group` without explicit group)
- Use collective operations without `group=` parameter
- Allow workers to communicate directly (all communication through controller)
- Use `.item()`, `.cpu()`, `.numpy()` in training loops (GPU-CPU sync overhead)
- Hardcode secrets, model paths, or cluster endpoints
- Skip pre-commit hooks
- Use wildcard imports (`from x import *`)
- Use `print()` for logging (use `logger.info/warn/error`)
- Build inheritance hierarchies deeper than 2 levels

## Progressive Disclosure: Detailed Guides

| Task                    | Reference                                                             |
| ----------------------- | --------------------------------------------------------------------- |
| Add Reward Function     | `verl/reward/`, `.claude/skills/add-reward.md`                        |
| Add Dataset Loader      | `verl/data/`, `.claude/skills/add-dataset.md`                         |
| Add Ray Worker          | `verl/workers/`, `.claude/skills/add-worker.md`                       |
| Add Unit Tests          | `tests/`, `.claude/skills/add-unit-tests/SKILL.md`                    |
| Debug Distributed       | `.claude/skills/debug-distributed.md`                                 |
| Algorithm Details       | `verl/trainer/ppo/core_algos.py`, `docs/algo/`                        |
| Architecture Overview   | `docs/`, `verl/single_controller/`                                    |
| Hydra Configuration     | `verl/trainer/config/`, `.claude/rules/api-config.md`                 |
| FSDP2 Training          | `verl/utils/fsdp_utils.py`, `verl/workers/fsdp_workers.py`           |
| Megatron Integration    | `verl/models/mcore/`, `verl/utils/megatron_utils/`                    |
| Inference Engines       | `verl/workers/rollout/vllm_rollout/`, `verl/workers/rollout/sglang_rollout/` |
| Hybrid Engine           | `verl/single_controller/ray/base.py` (`create_colocated_worker_cls`) |
| Quickstart              | `docs/start/install.rst`, `examples/`                                 |

## Git Workflow

- **Commits**: `[component] type: description` format (e.g., `[worker] feat: add reward
  model worker (#456)`), imperative voice, reasoning in body
- **Component tags**: `worker`, `rollout`, `trainer`, `reward`, `data`, `model`, `fsdp`,
  `megatron`, `ray`, `ckpt`, `cfg`
- **Squash**: Squash WIP commits before opening PR
- **PR requirements**: Run pre-commit, document test coverage, note hardware limitations
- **Line length**: 120 characters (enforced by Ruff)
- **Import order**: stdlib -> third-party -> verl (absolute imports, enforced by isort)

## Extended Configuration

See `.claude/agents/`, `.claude/skills/`, `.claude/commands/`, and `.claude/rules/` for
specialized instructions.

### Agents

| Agent                    | Purpose                               | Activation Trigger                                                  |
| ------------------------ | ------------------------------------- | ------------------------------------------------------------------- |
| `planner`                | Implementation planning               | Before multi-file changes, new features, or architectural decisions |
| `simple-code-reviewer`   | Quick code quality checks             | After code changes, before committing                               |
| `code-verifier`          | Formatting/linting/tests              | After code changes, before committing                               |
| `algorithm-expert`       | RL algorithms (PPO/GRPO/RLOO/etc.)    | Algorithm code changes or questions                                 |
| `fsdp-engine-expert`     | FSDP2 training backend                | FSDP worker/utility code changes or questions                       |
| `megatron-engine-expert` | Megatron-LM integration               | Megatron worker/model code changes or questions                     |
| `vllm-sglang-expert`     | Inference engines (vLLM/SGLang)       | Rollout worker code changes or questions                            |
| `ray-controller-expert`  | Ray single-controller orchestration   | Ray controller/worker coordination questions                        |

**Stage-by-Stage Agent Guidance**:

1. **Planning Stage** (Before coding): Use `planner` for architecture design and
   implementation planning
1. **Code Formatting & Linting** (After coding): Use `code-verifier` to automatically
   run formatting, linting, and tests, catching syntax errors and style issues quickly
1. **Code Quality Check** (After formatting): Use `simple-code-reviewer` for quick code
   quality checks, focusing on verl-specific patterns and distributed code correctness

### Skills (Guided Development Workflows)

Skills provide step-by-step guides for common development tasks:

- `/add-reward` - Reward function creation guide (5 steps, ~20 min)
- `/add-dataset` - Dataset loader creation guide (5 steps, ~30 min)
- `/add-worker` - Ray Worker creation guide (5 steps, ~40 min)
- `/add-unit-tests` - Test development guide (7 steps, ~20 min)
- `/debug-distributed` - Distributed training debugging guide

### Commands (User-invoked Actions)

Commands perform specific actions when invoked:

- `/create-pr` - Rebase, squash commits, and create/update PR with intelligent messages
- `/gen-commit-msg` - Generate commit messages from staged changes
- `/review-pr` - Intelligent PR code review with dynamic agent allocation

### Rules (Code Quality Standards)

Project-wide standards enforced across all code changes:

- `code-style.md` - Design principles (composition over inheritance, Ray single-controller,
  DataProto protocol), naming conventions, logging, performance guidelines (**always loaded**)
- `distributed.md` - Process group management, FSDP2/Megatron patterns, distributed
  debugging (loaded when editing `verl/workers/**`, `verl/models/**`, `verl/utils/fsdp_utils/**`)
- `testing.md` - Test organization, pytest markers, assertion patterns, GPU test guidelines
  (loaded when editing `**/tests/**`, `test_*.py`)
- `api-config.md` - Hydra config structure, dataclass integration, backward compatibility
  (loaded when editing `verl/trainer/config/**`, `verl/workers/config/**`)

## Code Intelligence & Navigation

When navigating and understanding code:

1. **ALWAYS prefer LSP tools over text search for code relationships**:
   - Use `goToDefinition` to jump to symbol definitions
   - Use `findReferences` to find all usages across the codebase
   - Use `goToImplementation` for interfaces/abstract methods
   - Use `workspaceSymbol` to search symbols across entire project
   - Use `getDiagnostics` to check for type errors when relevant

2. **Use Grep/Glob/Read ONLY for**:
   - Text/pattern searches in comments or strings
   - Searching configuration files (JSON, YAML, etc.)
   - Exploratory "fuzzy" searches when unsure what you're looking for
   - Finding files by name patterns

3. **Workflow**:
   - First: Use LSP to understand code structure and relationships
   - Second: Use text tools only when LSP cannot help (non-code content)
   - NEVER read entire large files to find references; use LSP instead

Remember: LSP provides semantic understanding (types, inheritance, references), while grep only provides text matching.


## ExecPlans

When writing complex features or significant refactors, use an ExecPlan (as described in ./PLANS.md) from design to implementation.


## Superpowers Expert

When handling complex tasks, you must strictly follow this four-phase methodology:

## Phase 1: Design & Planning
- **brainstorming**: Use Socratic dialogue to clarify ambiguous requirements; decompose business concepts into actionable technical tasks
- **writing-plans**: Apply time-boxed decomposition; break tasks into 2-5 minute executable units with clear checkpoints

## Phase 2: Execution & Development
Select based on context:
- **test-driven-development**: Production-grade code must follow Red-Green-Refactor; write tests BEFORE implementation
- **executing-plans**: Batch execution with mandatory pause-and-confirm at every checkpoint
- **subagent-driven-development**: Critical modules require isolated sub-agents to ensure task isolation and quality control

## Phase 3: Debugging & Verification
- **systematic-debugging**: Hypothesis → Test → Validate/Refute; iterate to root cause; blind trial-and-error is prohibited
- **verification-before-completion**: Multi-dimensional verification (tests, linting, integration) required before marking any task complete

## Phase 4: Collaboration & Review
- **requesting-code-review**: Mandatory pre-merge review; treat critical issues as blockers requiring immediate resolution
- **receiving-code-review**: Triage feedback by Critical/High/Medium/Low priority; batch fixes; re-verify

## Iron Rules
1. **No skipping steps**, even if the task appears simple
2. When uncertain which skill applies, default to **brainstorming** for clarification


## Recommended Skill Bundles

### `verl` / RL post-training in this repository

- 改 RL 算法、trainer、reward、advantage、rollout 数据流：`verl`
- 改 rollout engine：
  - `vllm`
  - `sglang`
- 改训练并行/显存策略：
  - `pytorch-fsdp2`
  - `megatron-core`
  - `deepspeed`
  - `ray-train`
- 改吞吐/显存热点：
  - `flash-attention`
  - `bitsandbytes`
  - `awq` / `gptq` / `hqq`
- 做训练观测和实验对比：
  - `tensorboard`
  - `weights-and-biases`
  - `mlflow`

### Serve a fine-tuned model in production

- 主 serving skill：`vllm` / `sglang` / `tensorrt-llm` / `llama-cpp`
- 若要降显存或量化：`bitsandbytes` / `awq` / `gptq` / `hqq` / `gguf`
- 若要加速 attention：`flash-attention`
- 若要追踪线上效果：`langsmith` / `phoenix`

### Train a new model at scale

- 架构主 Skill：`litgpt` / `nanogpt` / `mamba` / `rwkv` / `torchtitan`
- 分布式：`megatron-core` / `pytorch-fsdp2` / `deepspeed` / `ray-train`
- 数据：`ray-data` / `nemo-curator`
- tokenizer：`huggingface-tokenizers` / `sentencepiece`
- tracking：`weights-and-biases` / `mlflow` / `tensorboard`

## Full Category Map

### Tokenization

- `huggingface-tokenizers`: fast Rust tokenizer training, BPE/WordPiece/Unigram pipelines
- `sentencepiece`: language-agnostic subword tokenization and tokenizer research workflows

### Fine-Tuning

- `axolotl`: config-driven SFT and instruction tuning across many open models
- `llama-factory`: UI-first or low-code fine-tuning workflows
- `peft`: LoRA, QLoRA, DoRA, adapters, and broader parameter-efficient tuning
- `unsloth`: faster, lower-memory QLoRA fine-tuning

### Data Processing

- `nemo-curator`: large-scale data curation, deduplication, and filtering
- `ray-data`: distributed ETL, data loading, and training-data preprocessing

### Post-Training

- `verl`: HybridFlow RL post-training, actor-rollout-ref pipelines, GRPO/PPO style systems

### Distributed Training

- `accelerate`: lightweight distributed training abstraction around PyTorch/Hugging Face
- `deepspeed`: ZeRO-based memory scaling and distributed optimizer strategies
- `megatron-core`: tensor/pipeline/data/expert parallel large-scale training
- `pytorch-fsdp2`: PyTorch native full sharding workflows
- `ray-train`: cluster orchestration, multi-node execution, and tuning loops

### Inference Serving

- `sglang`: structured generation and high-throughput agentic rollout serving
- `vllm`: PagedAttention-based production LLM serving

### MLOps

- `tensorboard`: training visualization, scalars, embeddings, and profiling
- `weights-and-biases`: experiment tracking, comparisons, sweeps, and artifacts

### Emerging Techniques

- `knowledge-distillation`: teacher-student compression and transfer
- `long-context`: context extension strategies and positional encoding choices
- `model-merging`: model fusion with mergekit-style methods
- `model-pruning`: sparsification and parameter removal
- `moe-training`: mixture-of-experts training workflows
- `speculative-decoding`: draft-target decoding for faster inference


## Sub-Agent Execution Policy

### Core Principles
- **Mandatory Waiting**: Once a sub-agent task is initiated, you MUST wait for its complete response regardless of time consumption.
- **Zero Polling**: Strictly prohibited from rushing or requesting "final review results" during sub-agent execution. No requests for expedited responses allowed.
- **Quality Over Speed**: Acknowledge the principle that "quality work takes time" (慢工出细活). Do not impose artificial time pressure. Allow sub-agents sufficient time for thorough analysis.

### Execution Rules
1. **Post-Initiation Lock**: After invoking a sub-agent, the current process enters a waiting state until receiving the sub-agent's termination signal (e.g., `status: completed`).
2. **Asynchronous Tolerance**: If the sub-agent requires "some time" (time delay), you MUST wait silently. Under no circumstances shall you:
   - Poll for status updates repeatedly
   - Send "what's the progress?" type messages  
   - Set short timeouts (&lt;5 minutes) to force termination
3. **Result Integrity Verification**: Only proceed to the next workflow step upon receiving explicit final result markers from the sub-agent (e.g., `final_review` field or task completion identifiers).

### Constraints
- **FORBIDDEN**: `Urging sub-agents to return results quickly`
- **FORBIDDEN**: `Assuming sub-agent failure and initiating new requests before completion`
- **REQUIRED**: `Accept and respect reasonable processing time for sub-agents`
