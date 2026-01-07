# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

verl (Volcano Engine Reinforcement Learning) is a flexible, efficient, and production-ready RL training library for large language models. It is the open-source version of the HybridFlow framework presented at EuroSys 2025.

Key design principles:
- **Hybrid-controller programming model**: Enables flexible representation and efficient execution of complex post-training dataflows
- **Modular backend integration**: Decouples computation and data dependencies, allowing seamless integration with existing LLM frameworks (FSDP, Megatron-LM, vLLM, SGLang)
- **Flexible device mapping**: Supports various placement of models onto different GPU sets for efficient resource utilization

## Common Development Commands

### Installation

**Python-only development** (recommended for quick iteration):
```bash
pip install -e .[test,vllm]  # For vLLM backend
pip install -e .[test,sglang]  # For SGLang backend
```

For full installation details, see the [installation documentation](https://verl.readthedocs.io/en/latest/start/install.html).

### Linting and Formatting

```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run on staged changes
pre-commit run

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run --all-files --show-diff-on-failure --color=always ruff
```

### Testing

**Run specific test categories:**
```bash
# CPU unit tests
pytest tests/**/test_*_on_cpu.py

# GPU unit tests
pytest tests/ --ignore=tests/special_npu --ignore=tests/**/test_*_on_cpu.py

# Specific test suites
pytest tests/trainer/  # Test trainer components
pytest tests/models/   # Test model implementations
pytest tests/special_distributed/  # Multi-GPU tests
pytest tests/special_e2e/  # End-to-end tests
```

See `.github/workflows/` for comprehensive CI test configurations.

### Documentation

```bash
cd docs
pip install -r requirements-docs.txt
make clean
make html

# Preview locally
python -m http.server -d _build/html/
# Open http://localhost:8000
```

### Running Training Examples

**PPO training:**
```bash
# Basic PPO example
bash examples/ppo_trainer/run_gemma.sh

# With custom parameters
python3 -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    actor_rollout_ref.model.path=google/gemma-2-2b-it \
    trainer.n_gpus_per_node=8
```

**GRPO training:**
```bash
# Basic GRPO example
bash examples/grpo_trainer/run_qwen3-8b.sh

# GRPO requires rollout.n > 1 for group sampling
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.actor.use_kl_loss=True
```

**Multi-turn training with SGLang:**
```bash
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
```

## Architecture and Code Structure

### Core Components

**1. Hybrid-Controller Architecture (`verl/single_controller/`)**

The hybrid-controller is the core orchestration layer that coordinates training and generation phases:
- `verl/single_controller/base/`: Base controller abstractions
- `verl/single_controller/ray/`: Ray-based distributed controller implementation

**2. Workers (`verl/workers/`)**

Workers execute specific tasks in the RL training loop:
- `actor/`: Policy network (actor model) training
- `critic/`: Value network (critic model) training (PPO only)
- `rollout/`: Generation/rollout workers (vLLM, SGLang, HF)
  - `rollout/vllm_rollout/`: vLLM backend integration
  - `rollout/sglang_rollout/`: SGLang backend integration
  - `rollout/hf_rollout.py`: HuggingFace Transformers backend
- `reward_model/`: Reward model workers
- `reward_manager/`: Rule-based reward function management
- `engine/`: Backend-specific engine implementations
- `sharding_manager/`: Model sharding coordination

**3. Training Backends**

- `fsdp_workers.py`: PyTorch FSDP/FSDP2 backend implementation
- `megatron_workers.py`: Megatron-LM backend implementation
- `engine_workers.py`: Generic engine worker interface

**4. Trainers (`verl/trainer/`)**

Main training loops and configuration:
- `verl/trainer/main_ppo.py`: Main PPO/GRPO/RLOO training entry point
- `verl/trainer/ppo/`: PPO-specific algorithms and utilities
- `verl/trainer/config/`: Hydra configuration files
  - `ppo_trainer.yaml`: Default PPO configuration
  - `ppo_megatron_trainer.yaml`: Megatron backend configuration
  - `sft_trainer.yaml`: Supervised fine-tuning configuration

**5. Models (`verl/models/`)**

Model implementations and adapters:
- `transformers/`: HuggingFace Transformers integration
- `mcore/`: Megatron-Core model implementations
- `llama/`, `qwen2/`: Model-specific implementations

**6. Third-party Integrations (`verl/third_party/`)**

Patches and utilities for third-party libraries:
- `vllm/`: vLLM integration utilities
- `sglang/`: SGLang integration utilities
- `torch/`: PyTorch utilities

### Key Algorithms

All algorithms share a similar training loop structure:

- **PPO (Proximal Policy Optimization)**: Actor-critic algorithm with clipped surrogate objective
  - Requires both actor and critic models
  - Uses GAE (Generalized Advantage Estimation)
  - Config: `algorithm.adv_estimator=gae`

- **GRPO (Group Relative Policy Optimization)**: Critic-free algorithm using group sampling
  - No critic model needed
  - Requires `rollout.n > 1` for group sampling
  - Config: `algorithm.adv_estimator=grpo`, `actor.use_kl_loss=True`

- **RLOO (REINFORCE Leave-One-Out)**: Another critic-free algorithm
  - Config: `algorithm.adv_estimator=rloo`

- **REINFORCE++**: Enhanced REINFORCE with baseline
  - Config: `algorithm.adv_estimator=reinforce_plus_plus`

See `examples/*/` and `recipe/*/` for additional algorithms (DAPO, SPPO, GSPO, etc.).

### Configuration System

verl uses Hydra for configuration management:
- All configs are in `verl/trainer/config/`
- Override configs via command line: `key=value` or `key.subkey=value`
- Common config patterns:
  - `data.*`: Dataset configuration
  - `actor_rollout_ref.*`: Actor, rollout, and reference model config
  - `critic.*`: Critic model config (PPO only)
  - `algorithm.*`: Algorithm hyperparameters
  - `trainer.*`: Training loop settings

**Important config distinctions:**
- `*_batch_size`: Global batch size across all workers
- `*_micro_batch_size_per_gpu`: Per-GPU batch size to avoid OOM (does not affect convergence)
- `ppo_mini_batch_size`: Batch size for PPO updates (affects convergence)

### Backend Selection

**Training backends:**
- FSDP/FSDP2 (default): Set `actor.strategy=fsdp` or `actor.strategy=fsdp2`
- Megatron-LM: Use `ppo_megatron_trainer.yaml` config

**Inference backends for rollout:**
- vLLM: `rollout.name=vllm` (recommended, supports vLLM >= 0.8.2)
- SGLang: `rollout.name=sglang` (best for multi-turn and VLM)
- HuggingFace: `rollout.name=hf` (fallback option)

**Backend feature matrix:**
- FSDP: Supports vLLM, SGLang, HF rollout; AMD ROCm support
- Megatron: Supports vLLM, SGLang; scales to 671B models with expert parallelism
- Sequence parallelism: DeepSpeed Ulysses via `actor.sequence_parallel_config`

### Extending verl

**Adding a new model:**
1. For FSDP backend: Add model to `verl/models/transformers/`
2. For Megatron backend: Add model to `verl/models/mcore/`
3. Register in appropriate `__init__.py`
4. See documentation: [FSDP extension](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html), [Megatron extension](https://verl.readthedocs.io/en/latest/advance/megatron_extension.html)

**Adding a custom reward function:**
```python
# In your custom file (e.g., my_reward.py)
def compute_score(data):
    # data is a DataProto object with prompts and responses
    # Return reward scores
    pass

# In config
custom_reward_function.path=path/to/my_reward.py
custom_reward_function.name=compute_score
```

**Adding a new RL algorithm:**
- Implement advantage estimator in `verl/trainer/ppo/core_algos.py`
- Add config option to `algorithm.adv_estimator`
- See examples in `recipe/` directory

## Testing Strategy

Tests are organized by component namespace:
- `tests/trainer/`: Trainer-related tests
- `tests/models/`: Model implementation tests
- `tests/special_distributed/`: Multi-GPU unit tests
- `tests/special_e2e/`: End-to-end training tests
- `tests/special_npu/`: NPU-specific tests
- `tests/special_standalone/`: Tests requiring dedicated environments

**Test naming conventions:**
- `test_*_on_cpu.py`: CPU-only tests
- All other tests assume GPU availability (except `special_npu/`)

**When adding new tests:**
1. Find the relevant workflow in `.github/workflows/`
2. Add path patterns if needed
3. Minimize workload (use small models/datasets)
4. Exclude from `cpu_unit_tests.yml` or `gpu_unit_tests.yml` if using a dedicated workflow

## Important Notes

- **FSDP2 is recommended**: Better performance and memory usage. Enable with `actor.strategy=fsdp2`
- **Avoid vLLM 0.7.x**: Contains bugs causing OOMs. Use vLLM >= 0.8.2
- **SGLang for multi-turn**: SGLang is the recommended backend for multi-turn and VLM RLHF
- **Sequence packing**: Enable with `data.pack_sequence=True` for better throughput
- **Memory optimization**: Use `actor.fsdp_config.offload_policy=True` for CPU offloading with FSDP2

## Performance Tuning

See the detailed [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) for:
- Throughput optimization
- Memory reduction techniques
- Profiling with Nsight Systems
- Distributed training best practices

## Project Structure

```
verl/
├── verl/                          # Main package
│   ├── single_controller/         # Hybrid-controller orchestration
│   ├── workers/                   # Worker implementations
│   │   ├── actor/                 # Actor model training
│   │   ├── critic/                # Critic model training
│   │   ├── rollout/               # Rollout/generation workers
│   │   ├── reward_model/          # Reward model workers
│   │   └── engine/                # Backend engines
│   ├── trainer/                   # Training loops and configs
│   │   ├── ppo/                   # PPO algorithms
│   │   └── config/                # Hydra configs
│   ├── models/                    # Model implementations
│   ├── utils/                     # Utilities
│   └── third_party/               # Third-party integrations
├── examples/                      # Training examples
│   ├── ppo_trainer/               # PPO examples
│   ├── grpo_trainer/              # GRPO examples
│   ├── sglang_multiturn/          # Multi-turn examples
│   └── sft/                       # SFT examples
├── recipe/                        # Algorithm recipes
│   ├── dapo/                      # DAPO implementation
│   ├── sppo/                      # SPPO implementation
│   └── ...                        # Other algorithms
├── tests/                         # Test suites
└── docs/                          # Documentation
```
