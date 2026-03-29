import json
import os
import subprocess
from pathlib import Path


def _capture_scripted_mini_val_invocation(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "recipe" / "mopd"
    dst_dir = tmp_path / "recipe" / "mopd"
    dst_dir.mkdir(parents=True)

    main_script = (src_dir / "run_mopd_qwen3_4b.sh").read_text(encoding="utf-8")
    main_script = main_script.replace(
        """# === Environment Setup ===
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
unset http_proxy https_proxy all_proxy
conda activate veRL_126

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
""",
        """# === Environment Setup ===
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
""",
    )
    main_script = main_script.replace(
        'python3 -m verl.trainer.main_ppo \\',
        'python3 "${SCRIPT_DIR}/capture_invocation.py" \\',
        1,
    )
    (dst_dir / "run_mopd_qwen3_4b.sh").write_text(main_script, encoding="utf-8")

    for name in ["run_mopd_qwen3_4b_mini.sh", "run_mopd_qwen3_4b_mini_val.sh", "mopd_longrun_topology.sh"]:
        (dst_dir / name).write_text((src_dir / name).read_text(encoding="utf-8"), encoding="utf-8")

    capture_script = """import json
import os
import sys
from pathlib import Path

capture_file = Path(os.environ["CAPTURE_FILE"])
capture = {
    "argv": sys.argv[1:],
    "arg_map": {
        arg.split("=", 1)[0]: arg.split("=", 1)[1]
        for arg in sys.argv[1:]
        if "=" in arg
    },
    "env": {
        key: os.environ.get(key, "")
        for key in [
            "CKPTS_ROOT",
            "RUN_ID",
            "RUNTIME_ROOT",
            "RUNTIME_CACHE_ROOT",
            "RUNTIME_CACHE_TAG",
            "TMP_ROOT_BASE",
            "TMPDIR",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "VLLM_CONFIG_ROOT",
            "VLLM_CACHE_ROOT",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_HOME",
            "TORCH_EXTENSIONS_DIR",
            "FLASHINFER_WORKSPACE_BASE",
            "DG_JIT_CACHE_DIR",
            "CUDA_CACHE_PATH",
            "RAY_TMPDIR",
        ]
    },
}
capture_file.write_text(json.dumps(capture, sort_keys=True), encoding="utf-8")
"""
    (dst_dir / "capture_invocation.py").write_text(capture_script, encoding="utf-8")

    reward_root = tmp_path / "rvq" / "custom_rewards"
    reward_root.mkdir(parents=True)
    (reward_root / "deepseek_singlecell_reward.py").write_text("# stub\n", encoding="utf-8")

    verl_pkg = tmp_path / "verl"
    verl_pkg.mkdir()
    (verl_pkg / "__init__.py").write_text("__version__ = 'test'\n", encoding="utf-8")

    training_output_root = tmp_path / "training_output"
    local_runtime_cache_root = tmp_path / "local_cache"
    capture_file = tmp_path / "capture.json"

    env = os.environ.copy()
    env.update(
        {
            "CAPTURE_FILE": str(capture_file),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "TRAINING_OUTPUT_ROOT": str(training_output_root),
            "LOCAL_RUNTIME_CACHE_ROOT": str(local_runtime_cache_root),
            "RUN_ID": "mini_val_integration",
            "RUNTIME_CACHE_TAG": "0102030404",
            "TMPDIR_TAG": "0102030405",
            "SAVE_FREQ": "2",
            "TEST_FREQ": "2",
            "RAY_TMPDIR": str(tmp_path / "ray_tmp"),
            "RVQ_REWARD_ROOT": str(tmp_path / "rvq"),
            "SWANLAB_MODE": "disabled",
        }
    )

    subprocess.run(
        ["bash", str(dst_dir / "run_mopd_qwen3_4b_mini_val.sh")],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    return json.loads(capture_file.read_text(encoding="utf-8"))


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
    assert 'FSDP_SIZE=${FSDP_SIZE:-${NGPUS_PER_NODE}}' in script_text
    assert 'ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}' in script_text
    assert 'source "${SCRIPT_DIR}/mopd_longrun_topology.sh"' in script_text
    assert '"${MOPD_TEACHERS_OVERRIDE}"' in script_text
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


def test_full_run_script_allows_external_cuda_visible_devices_override():
    repo_root = Path(__file__).resolve().parents[2]
    train_script = (repo_root / "recipe" / "mopd" / "run_mopd_qwen3_4b.sh").read_text(encoding="utf-8")

    assert 'NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}' in train_script
    assert 'if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then' in train_script
    assert 'CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPUS_PER_NODE - 1)))' in train_script
    assert 'export CUDA_VISIBLE_DEVICES' in train_script
    assert 'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"' not in train_script


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


def test_preflight_script_uses_rvq_v1_reward_contract():
    repo_root = Path(__file__).resolve().parents[2]
    preflight_script = (repo_root / "recipe" / "mopd" / "run_mopd_qwen3_4b_preflight.sh").read_text(encoding="utf-8")

    assert 'PREFLIGHT_RVQ_REWARD_ROOT="${PREFLIGHT_RVQ_REWARD_ROOT:-/home/scbjtfy/RVQ-Alpha/rlvr}"' in preflight_script
    assert 'PREFLIGHT_REWARD_PROVIDER="${PREFLIGHT_REWARD_PROVIDER:-sglang}"' in preflight_script
    assert 'PREFLIGHT_REWARD_API_BASE="${PREFLIGHT_REWARD_API_BASE:-http://127.0.0.1:30000/v1}"' in preflight_script
    assert 'PREFLIGHT_REWARD_MODEL="${PREFLIGHT_REWARD_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"' in preflight_script
    assert 'PREFLIGHT_REWARD_API_KEY="${PREFLIGHT_REWARD_API_KEY:-EMPTY}"' in preflight_script
    assert 'PREFLIGHT_REWARD_NUM_WORKERS="${PREFLIGHT_REWARD_NUM_WORKERS:-1}"' in preflight_script
    assert 'PREFLIGHT_REWARD_MAX_CONCURRENT="${PREFLIGHT_REWARD_MAX_CONCURRENT:-1}"' in preflight_script
    assert 'PREFLIGHT_REWARD_MAX_RPM_PER_WORKER="${PREFLIGHT_REWARD_MAX_RPM_PER_WORKER:-60}"' in preflight_script
    assert 'PREFLIGHT_REWARD_MAX_TPM_PER_WORKER="${PREFLIGHT_REWARD_MAX_TPM_PER_WORKER:-60000}"' in preflight_script
    assert 'PREFLIGHT_REWARD_EST_TOKENS_PER_REQUEST="${PREFLIGHT_REWARD_EST_TOKENS_PER_REQUEST:-18000}"' in preflight_script
    assert 'PREFLIGHT_REWARD_TIMEOUT="${PREFLIGHT_REWARD_TIMEOUT:-900.0}"' in preflight_script
    assert '--rvq-reward-root "${PREFLIGHT_RVQ_REWARD_ROOT}" \\' in preflight_script
    assert '--reward-provider "${PREFLIGHT_REWARD_PROVIDER}" \\' in preflight_script
    assert '--reward-api-base "${PREFLIGHT_REWARD_API_BASE}" \\' in preflight_script
    assert '--reward-model "${PREFLIGHT_REWARD_MODEL}" \\' in preflight_script
    assert '--reward-api-key "${PREFLIGHT_REWARD_API_KEY}" \\' in preflight_script
    assert '--reward-num-workers "${PREFLIGHT_REWARD_NUM_WORKERS}" \\' in preflight_script
    assert '--reward-max-concurrent "${PREFLIGHT_REWARD_MAX_CONCURRENT}" \\' in preflight_script
    assert '--reward-max-rpm-per-worker "${PREFLIGHT_REWARD_MAX_RPM_PER_WORKER}" \\' in preflight_script
    assert '--reward-max-tpm-per-worker "${PREFLIGHT_REWARD_MAX_TPM_PER_WORKER}" \\' in preflight_script
    assert '--reward-est-tokens-per-request "${PREFLIGHT_REWARD_EST_TOKENS_PER_REQUEST}" \\' in preflight_script
    assert '--reward-timeout "${PREFLIGHT_REWARD_TIMEOUT}" \\' in preflight_script


def test_preflight_script_defaults_to_smoke_safe_gpu_binding_and_batch_size():
    repo_root = Path(__file__).resolve().parents[2]
    preflight_script = (repo_root / "recipe" / "mopd" / "run_mopd_qwen3_4b_preflight.sh").read_text(encoding="utf-8")

    assert 'NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"' in preflight_script
    assert 'if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then' in preflight_script
    assert 'CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPUS_PER_NODE - 1)))' in preflight_script
    assert 'export CUDA_VISIBLE_DEVICES' in preflight_script
    assert 'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"' not in preflight_script
    assert 'PREFLIGHT_TRAIN_BATCH_SIZE="${PREFLIGHT_TRAIN_BATCH_SIZE:-2}"' in preflight_script


def test_preflight_script_fail_fast_guard_matches_agent_loop_floor():
    repo_root = Path(__file__).resolve().parents[2]
    preflight_script = (repo_root / "recipe" / "mopd" / "run_mopd_qwen3_4b_preflight.sh").read_text(encoding="utf-8")

    assert 'PREFLIGHT_MIN_SAMPLES=$((PREFLIGHT_TRAIN_BATCH_SIZE * PREFLIGHT_ROLLOUT_N))' in preflight_script
    assert 'if [ "${PREFLIGHT_MIN_SAMPLES}" -lt 8 ]; then' in preflight_script
    assert 'train_batch_size * rollout_n must be >= 8' in preflight_script


def test_full_run_script_uses_rvq_v1_validation_reward_and_deterministic_validation():
    repo_root = Path(__file__).resolve().parents[2]
    train_script = (repo_root / "recipe" / "mopd" / "run_mopd_qwen3_4b.sh").read_text(encoding="utf-8")

    assert 'reward.custom_reward_function.path="${SCRIPT_DIR}/rvq_v1_reward.py"' in train_script
    assert 'reward.custom_reward_function.name=compute_score' in train_script
    assert 'reward.reward_manager.name=rate_limited' in train_script
    assert 'RVQ_REWARD_ROOT=${RVQ_REWARD_ROOT:-"/home/scbjtfy/RVQ-Alpha/rlvr"}' in train_script
    assert 'REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-' in train_script
    assert 'REWARD_MAX_CONCURRENT=${REWARD_MAX_CONCURRENT:-' in train_script
    assert 'REWARD_MAX_RPM_PER_WORKER=${REWARD_MAX_RPM_PER_WORKER:-' in train_script
    assert 'REWARD_MAX_TPM_PER_WORKER=${REWARD_MAX_TPM_PER_WORKER:-' in train_script
    assert 'REWARD_TIMEOUT=${REWARD_TIMEOUT:-' in train_script
    assert 'MOPD_ORM_WEIGHT=${MOPD_ORM_WEIGHT:-0.0}' in train_script
    assert "+trainer.val_metric_group_key=teacher_id" in train_script
    assert "actor_rollout_ref.rollout.val_kwargs.do_sample=False" in train_script
    assert "actor_rollout_ref.rollout.val_kwargs.n=1" in train_script


def test_full_run_script_externalizes_teacher_topology():
    repo_root = Path(__file__).resolve().parents[2]
    train_script = (repo_root / "recipe" / "mopd" / "run_mopd_qwen3_4b.sh").read_text(encoding="utf-8")
    topology_script = (repo_root / "recipe" / "mopd" / "mopd_longrun_topology.sh").read_text(encoding="utf-8")

    assert 'source "${SCRIPT_DIR}/mopd_longrun_topology.sh"' in train_script
    assert '"${MOPD_RESOURCE_POOLS_OVERRIDE[@]}"' in train_script
    assert '"${MOPD_TEACHERS_OVERRIDE}"' in train_script
    assert "'algorithm.mopd.teachers=[" not in train_script

    assert "MOPD_RESOURCE_POOLS_OVERRIDE=(" in topology_script
    assert "MOPD_TEACHERS_OVERRIDE=" in topology_script
    assert "cell_type_teacher" in topology_script
    assert "disease_state_teacher" in topology_script


def test_full_run_script_defaults_to_gpfs_output_and_runtime_roots():
    repo_root = Path(__file__).resolve().parents[2]
    train_script = (repo_root / "recipe" / "mopd" / "run_mopd_qwen3_4b.sh").read_text(encoding="utf-8")

    assert 'TRAINING_OUTPUT_ROOT=${TRAINING_OUTPUT_ROOT:-"/gpfs/Mamba/Project/Single_Cell/Training"}' in train_script
    assert 'CKPTS_ROOT=${CKPTS_ROOT:-"${TRAINING_OUTPUT_ROOT}/Qwen3-4B-Thinking-2507_Manual_Resize_Block-8_Size-32_Num-256_ExplicitTokens_Continued-Pretraining-ALL_Augmented-V1_SFT-ALL_DFT-MOPD"}' in train_script
    assert 'RUN_ID=${RUN_ID:-$(date +"%Y%m%d_%H%M%S")}' in train_script
    assert 'CKPTS_DIR=${CKPTS_DIR:-"${CKPTS_ROOT}/${RUN_ID}"}' in train_script
    assert 'RESUME_MODE=${RESUME_MODE:-disable}' in train_script
    assert 'RUNTIME_ROOT=${RUNTIME_ROOT:-"${CKPTS_DIR}/runtime"}' in train_script
    assert 'trainer.default_local_dir="${CKPTS_DIR}"' in train_script
    assert 'trainer.resume_mode=${RESUME_MODE}' in train_script
    assert 'hydra.run.dir="${CKPTS_DIR}/hydra"' in train_script
    assert "hydra.output_subdir=.hydra" in train_script


def test_full_run_script_externalizes_runtime_temp_and_cache_roots():
    repo_root = Path(__file__).resolve().parents[2]
    train_script = (repo_root / "recipe" / "mopd" / "run_mopd_qwen3_4b.sh").read_text(encoding="utf-8")

    assert 'RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT:-}' in train_script
    assert 'RUNTIME_CACHE_TAG=${RUNTIME_CACHE_TAG:-$(date +"%m%d%H%M%S")}' in train_script
    assert 'if [ -n "${RUNTIME_CACHE_ROOT}" ]; then' in train_script
    assert 'TMP_ROOT_BASE=${TMP_ROOT_BASE:-"${RUNTIME_CACHE_ROOT}/t"}' in train_script
    assert 'TMP_ROOT_BASE=${TMP_ROOT_BASE:-"${TRAINING_OUTPUT_ROOT}/t"}' in train_script
    assert 'TMPDIR_TAG=${TMPDIR_TAG:-$(date +"%m%d%H%M%S")}' in train_script
    assert 'TMPDIR=${TMPDIR:-"${TMP_ROOT_BASE}/${TMPDIR_TAG}"}' in train_script
    assert 'TEMP=${TEMP:-"${TMPDIR}"}' in train_script
    assert 'TMP=${TMP:-"${TMPDIR}"}' in train_script
    assert 'XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${RUNTIME_CACHE_ROOT}/c/${RUNTIME_CACHE_TAG}"}' in train_script
    assert 'XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-"${RUNTIME_CACHE_ROOT}/f/${RUNTIME_CACHE_TAG}"}' in train_script
    assert 'XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${RUNTIME_ROOT}/xdg_cache"}' in train_script
    assert 'XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-"${RUNTIME_ROOT}/xdg_config"}' in train_script
    assert 'VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT:-"${XDG_CONFIG_HOME}/vllm"}' in train_script
    assert 'VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-"${XDG_CACHE_HOME}/vllm"}' in train_script
    assert 'VLLM_RPC_BASE_PATH=${VLLM_RPC_BASE_PATH:-"${TMPDIR}/vllm_rpc"}' in train_script
    assert 'VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}' in train_script
    assert 'TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-"${RUNTIME_CACHE_ROOT}/i/${RUNTIME_CACHE_TAG}"}' in train_script
    assert 'TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-"${RUNTIME_ROOT}/torchinductor_cache"}' in train_script
    assert 'TRITON_CACHE_ROOT="${TORCHINDUCTOR_CACHE_DIR}/triton"' in train_script
    assert 'TRITON_HOME=${TRITON_HOME:-"${RUNTIME_CACHE_ROOT}/h/${RUNTIME_CACHE_TAG}"}' in train_script
    assert 'TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"${RUNTIME_CACHE_ROOT}/x/${RUNTIME_CACHE_TAG}"}' in train_script
    assert 'FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-"${RUNTIME_CACHE_ROOT}/fi/${RUNTIME_CACHE_TAG}"}' in train_script
    assert 'DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-"${RUNTIME_CACHE_ROOT}/dg/${RUNTIME_CACHE_TAG}"}' in train_script
    assert 'CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-"${RUNTIME_CACHE_ROOT}/cu/${RUNTIME_CACHE_TAG}"}' in train_script
    assert 'RAY_TMPDIR=${RAY_TMPDIR:-"${RUNTIME_CACHE_ROOT}/r_${RUNTIME_CACHE_TAG}"}' in train_script
    assert 'TRITON_HOME=${TRITON_HOME:-"${RUNTIME_ROOT}/triton_home"}' in train_script
    assert 'TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"${RUNTIME_ROOT}/torch_extensions"}' in train_script
    assert 'FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-"${RUNTIME_ROOT}/flashinfer"}' in train_script
    assert 'DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-"${RUNTIME_ROOT}/deep_gemm"}' in train_script
    assert 'CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-"${RUNTIME_ROOT}/cuda_cache"}' in train_script
    assert 'RAY_TMPDIR=${RAY_TMPDIR:-"${RUNTIME_ROOT}/ray_tmp"}' in train_script
    assert 'export VLLM_NO_USAGE_STATS' in train_script
    assert '[self-check] runtime_root=' in train_script
    assert '[self-check] runtime_cache_root=' in train_script
    assert '[self-check] runtime_cache_tag=' in train_script
    assert '[self-check] tmp_root_base=' in train_script
    assert '[self-check] tmpdir_tag=' in train_script
    assert '[self-check] tmpdir=' in train_script
    assert '[self-check] xdg_cache_home=' in train_script
    assert '[self-check] vllm_config_root=' in train_script
    assert '[self-check] torchinductor_cache_dir=' in train_script
    assert '[self-check] triton_cache_root=' in train_script


def test_full_run_script_respects_external_gpu_visibility_and_smoke_overrides():
    repo_root = Path(__file__).resolve().parents[2]
    train_script = (repo_root / "recipe" / "mopd" / "run_mopd_qwen3_4b.sh").read_text(encoding="utf-8")

    assert 'CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPUS_PER_NODE - 1)))' in train_script
    assert 'ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-false}' in train_script
    assert 'TEST_FREQ=${TEST_FREQ:-5}' in train_script
    assert 'SAVE_FREQ=${SAVE_FREQ:-5}' in train_script
    assert 'TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}' in train_script
    assert 'TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}' in train_script
    assert 'VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}' in train_script
    assert 'VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}' in train_script
    assert 'LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-100}' in train_script
    assert 'trainer.test_freq=${TEST_FREQ}' in train_script
    assert 'trainer.save_freq=${SAVE_FREQ}' in train_script
    assert 'trainer.total_epochs=${TOTAL_EPOCHS}' in train_script
    assert 'data.train_max_samples=${TRAIN_MAX_SAMPLES}' in train_script
    assert 'data.val_max_samples=${VAL_MAX_SAMPLES}' in train_script
    assert 'trainer.val_before_train=${VAL_BEFORE_TRAIN}' in train_script
    assert 'trainer.log_val_generations=${LOG_VAL_GENERATIONS}' in train_script
    assert 'actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}' in train_script
    assert '[preset] ROLLOUT_ENFORCE_EAGER=' in train_script


def test_main_launcher_mini_wrapper_applies_bounded_defaults_under_gpfs():
    repo_root = Path(__file__).resolve().parents[2]
    mini_script = (repo_root / "recipe" / "mopd" / "run_mopd_qwen3_4b_mini.sh").read_text(encoding="utf-8")

    assert 'TRAINING_OUTPUT_ROOT=${TRAINING_OUTPUT_ROOT:-"/gpfs/Mamba/Project/Single_Cell/Training"}' in mini_script
    assert 'NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}' in mini_script
    assert 'if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then' in mini_script
    assert 'CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPUS_PER_NODE - 1)))' in mini_script
    assert 'export CUDA_VISIBLE_DEVICES' in mini_script
    assert 'TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-2}' in mini_script
    assert 'TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-2}' in mini_script
    assert 'N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-4}' in mini_script
    assert 'MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}' in mini_script
    assert 'TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-4}' in mini_script
    assert 'VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-4}' in mini_script
    assert 'VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}' in mini_script
    assert 'TEST_FREQ=${TEST_FREQ:--1}' in mini_script
    assert 'SAVE_FREQ=${SAVE_FREQ:--1}' in mini_script
    assert 'TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}' in mini_script
    assert 'LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}' in mini_script
    assert 'ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-true}' in mini_script
    assert 'VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}' in mini_script
    assert 'REWARD_API_BASE=${REWARD_API_BASE:-http://127.0.0.1:30005/v1}' in mini_script
    assert 'CKPTS_ROOT=${CKPTS_ROOT:-"${TRAINING_OUTPUT_ROOT}/mopd_qwen3_4b_mini"}' in mini_script
    assert 'RUNTIME_ROOT=${RUNTIME_ROOT:-"${TRAINING_OUTPUT_ROOT}/mopd_qwen3_4b_mini_runtime/${RUN_ID}"}' in mini_script
    assert 'LOCAL_RUNTIME_CACHE_ROOT=${LOCAL_RUNTIME_CACHE_ROOT:-/dev/shm/v}' in mini_script
    assert 'RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT:-"${LOCAL_RUNTIME_CACHE_ROOT}"}' in mini_script
    assert 'RUNTIME_CACHE_TAG=${RUNTIME_CACHE_TAG:-$(date +"%m%d%H%M%S")}' in mini_script
    assert 'export RUNTIME_ROOT' in mini_script
    assert 'export LOCAL_RUNTIME_CACHE_ROOT' in mini_script
    assert 'export RUNTIME_CACHE_ROOT' in mini_script
    assert 'export RUNTIME_CACHE_TAG' in mini_script
    assert 'export ROLLOUT_ENFORCE_EAGER' in mini_script
    assert 'export VLLM_ATTENTION_BACKEND' in mini_script
    assert '[mini-run] RUNTIME_ROOT=' in mini_script
    assert '[mini-run] LOCAL_RUNTIME_CACHE_ROOT=' in mini_script
    assert '[mini-run] RUNTIME_CACHE_ROOT=' in mini_script
    assert '[mini-run] RUNTIME_CACHE_TAG=' in mini_script
    assert '[mini-run] ROLLOUT_ENFORCE_EAGER=' in mini_script
    assert '[mini-run] VLLM_ATTENTION_BACKEND=' in mini_script
    assert 'bash "${SCRIPT_DIR}/run_mopd_qwen3_4b.sh"' in mini_script


def test_main_launcher_mini_val_wrapper_applies_validation_first_defaults_under_gpfs():
    repo_root = Path(__file__).resolve().parents[2]
    mini_val_script = (repo_root / "recipe" / "mopd" / "run_mopd_qwen3_4b_mini_val.sh").read_text(encoding="utf-8")

    assert 'TRAINING_OUTPUT_ROOT=${TRAINING_OUTPUT_ROOT:-"/gpfs/Mamba/Project/Single_Cell/Training"}' in mini_val_script
    assert 'TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-16}' in mini_val_script
    assert 'VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-16}' in mini_val_script
    assert 'VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}' in mini_val_script
    assert 'ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-true}' in mini_val_script
    assert 'VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}' in mini_val_script
    assert 'REWARD_API_BASE=${REWARD_API_BASE:-http://127.0.0.1:30005/v1}' in mini_val_script
    assert 'CKPTS_ROOT=${CKPTS_ROOT:-"${TRAINING_OUTPUT_ROOT}/mopd_qwen3_4b_mini_val"}' in mini_val_script
    assert 'RUNTIME_ROOT=${RUNTIME_ROOT:-"${TRAINING_OUTPUT_ROOT}/mopd_qwen3_4b_mini_val_runtime/${RUN_ID}"}' in mini_val_script
    assert 'LOCAL_RUNTIME_CACHE_ROOT=${LOCAL_RUNTIME_CACHE_ROOT:-/dev/shm/v}' in mini_val_script
    assert 'RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT:-"${LOCAL_RUNTIME_CACHE_ROOT}"}' in mini_val_script
    assert 'RUNTIME_CACHE_TAG=${RUNTIME_CACHE_TAG:-$(date +"%m%d%H%M%S")}' in mini_val_script
    assert 'export RUNTIME_ROOT' in mini_val_script
    assert 'export LOCAL_RUNTIME_CACHE_ROOT' in mini_val_script
    assert 'export RUNTIME_CACHE_ROOT' in mini_val_script
    assert 'export RUNTIME_CACHE_TAG' in mini_val_script
    assert 'export ROLLOUT_ENFORCE_EAGER' in mini_val_script
    assert 'export VLLM_ATTENTION_BACKEND' in mini_val_script
    assert '[mini-val] RUNTIME_ROOT=' in mini_val_script
    assert '[mini-val] LOCAL_RUNTIME_CACHE_ROOT=' in mini_val_script
    assert '[mini-val] RUNTIME_CACHE_ROOT=' in mini_val_script
    assert '[mini-val] RUNTIME_CACHE_TAG=' in mini_val_script
    assert '[mini-val] ROLLOUT_ENFORCE_EAGER=' in mini_val_script
    assert '[mini-val] VLLM_ATTENTION_BACKEND=' in mini_val_script
    assert 'RUN_ID=${RUN_ID:-mini_val_' in mini_val_script
    assert 'bash "${SCRIPT_DIR}/run_mopd_qwen3_4b_mini.sh"' in mini_val_script


def test_mini_val_wrapper_exec_resolves_runtime_cache_and_output_roots(tmp_path):
    capture = _capture_scripted_mini_val_invocation(tmp_path)
    training_output_root = tmp_path / "training_output"
    run_id = "mini_val_integration"
    local_runtime_cache_root = tmp_path / "local_cache"

    assert capture["env"]["CKPTS_ROOT"] == str(training_output_root / "mopd_qwen3_4b_mini_val")
    assert capture["env"]["RUNTIME_ROOT"] == str(
        training_output_root / "mopd_qwen3_4b_mini_val_runtime" / run_id
    )
    assert capture["env"]["RUNTIME_CACHE_ROOT"] == str(local_runtime_cache_root)
    assert capture["env"]["TMPDIR"] == str(local_runtime_cache_root / "t" / "0102030405")
    assert capture["env"]["XDG_CACHE_HOME"] == str(local_runtime_cache_root / "c" / "0102030404")
    assert capture["env"]["XDG_CONFIG_HOME"] == str(local_runtime_cache_root / "f" / "0102030404")
    assert capture["env"]["VLLM_CONFIG_ROOT"] == str(local_runtime_cache_root / "f" / "0102030404" / "vllm")
    assert capture["env"]["VLLM_CACHE_ROOT"] == str(local_runtime_cache_root / "c" / "0102030404" / "vllm")
    assert capture["env"]["TORCHINDUCTOR_CACHE_DIR"] == str(local_runtime_cache_root / "i" / "0102030404")
    assert capture["env"]["TRITON_HOME"] == str(local_runtime_cache_root / "h" / "0102030404")
    assert capture["env"]["TORCH_EXTENSIONS_DIR"] == str(local_runtime_cache_root / "x" / "0102030404")
    assert capture["env"]["FLASHINFER_WORKSPACE_BASE"] == str(local_runtime_cache_root / "fi" / "0102030404")
    assert capture["env"]["DG_JIT_CACHE_DIR"] == str(local_runtime_cache_root / "dg" / "0102030404")
    assert capture["env"]["CUDA_CACHE_PATH"] == str(local_runtime_cache_root / "cu" / "0102030404")
    assert capture["env"]["RAY_TMPDIR"] == str(tmp_path / "ray_tmp")
    assert capture["arg_map"]["trainer.default_local_dir"] == str(
        training_output_root / "mopd_qwen3_4b_mini_val" / run_id
    )
    assert capture["arg_map"]["hydra.run.dir"] == str(
        training_output_root / "mopd_qwen3_4b_mini_val" / run_id / "hydra"
    )
    assert capture["arg_map"]["trainer.resume_mode"] == "disable"
    assert capture["arg_map"]["trainer.save_freq"] == "2"
    assert capture["arg_map"]["trainer.test_freq"] == "2"
