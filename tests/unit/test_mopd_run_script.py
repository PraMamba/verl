from pathlib import Path


def test_full_run_script_uses_conservative_memory_defaults():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "recipe" / "mopd" / "run_mopd_qwen3_4b.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert 'MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}' in script_text
    assert 'MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}' in script_text
    assert 'MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}' in script_text
    assert 'TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-32}' in script_text
    assert 'N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-4}' in script_text
    assert 'TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-8}' in script_text
    assert 'ACTOR_PPO_MAX_TOKEN_LEN=${ACTOR_PPO_MAX_TOKEN_LEN:-${MAX_MODEL_LEN}}' in script_text
    assert 'INFER_PPO_MAX_TOKEN_LEN=${INFER_PPO_MAX_TOKEN_LEN:-${MAX_MODEL_LEN}}' in script_text
    assert 'ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-false}' in script_text
    assert 'TEACHER_LOG_PROB_MICRO_BATCH_SIZE=${TEACHER_LOG_PROB_MICRO_BATCH_SIZE:-1}' in script_text
    assert 'ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}' in script_text
    assert 'REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-true}' in script_text
    assert 'GEN_TP=${GEN_TP:-2}' in script_text
    assert 'FSDP_SIZE=${FSDP_SIZE:-4}' in script_text
    assert 'ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}' in script_text
    assert "algorithm.mopd.teachers=[" in script_text
    assert "log_prob_micro_batch_size:" in script_text
    assert "tokenizer_compat_group:" in script_text
    assert "${TEACHER_LOG_PROB_MICRO_BATCH_SIZE}" in script_text
    assert "data.max_prompt_length=${MAX_PROMPT_LENGTH}" in script_text
    assert "data.max_response_length=${MAX_RESPONSE_LENGTH}" in script_text
    assert "data.train_batch_size=${TRAIN_PROMPT_BSZ}" in script_text
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_PROMPT_MINI_BSZ}" in script_text
    assert "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN}" in script_text
    assert "actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}" in script_text
    assert "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_PARAM_OFFLOAD}" in script_text
    assert "actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT}" in script_text
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}" in script_text
    assert "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}" in script_text
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" in script_text
    assert "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN}" in script_text
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}" in script_text
    assert "actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE}" in script_text
    assert "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN}" in script_text
    assert "actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD}" in script_text
    assert "[preset] MAX_MODEL_LEN=" in script_text
    assert "[preset] ACTOR_PPO_MAX_TOKEN_LEN=" in script_text
    assert "[preset] INFER_PPO_MAX_TOKEN_LEN=" in script_text
    assert "[preset] ACTOR_PARAM_OFFLOAD=" in script_text
    assert "[preset] REF_PARAM_OFFLOAD=" in script_text
    assert "[preset] ROLLOUT_MAX_NUM_BATCHED_TOKENS=" in script_text


def test_run_scripts_bind_python_imports_to_current_worktree():
    repo_root = Path(__file__).resolve().parents[2]
    train_script = (repo_root / "recipe" / "mopd" / "run_mopd_qwen3_4b.sh").read_text(encoding="utf-8")
    preflight_script = (repo_root / "recipe" / "mopd" / "run_mopd_qwen3_4b_preflight.sh").read_text(encoding="utf-8")

    assert 'REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)' in train_script
    assert 'cd "${REPO_ROOT}"' in train_script
    assert 'export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"' in train_script

    assert 'ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)' not in preflight_script
    assert 'SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)' in preflight_script
    assert 'ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)' in preflight_script
    assert 'cd "${ROOT}"' in preflight_script
    assert 'export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"' in preflight_script


def test_run_scripts_log_verl_import_self_check():
    repo_root = Path(__file__).resolve().parents[2]
    train_script = (repo_root / "recipe" / "mopd" / "run_mopd_qwen3_4b.sh").read_text(encoding="utf-8")
    preflight_script = (repo_root / "recipe" / "mopd" / "run_mopd_qwen3_4b_preflight.sh").read_text(encoding="utf-8")

    assert 'VERL_IMPORT_FILE=$(python3 -c' in train_script
    assert '[self-check] repo_root=' in train_script
    assert '[self-check] verl.__file__=' in train_script
    assert 'tee -a "${LOG_FILE}"' in train_script

    assert 'VERL_IMPORT_FILE=$(python3 -c' in preflight_script
    assert '[self-check] repo_root=' in preflight_script
    assert '[self-check] verl.__file__=' in preflight_script
    assert 'tee -a "${PREFLIGHT_LOG_FILE}"' in preflight_script
