from pathlib import Path


def test_full_run_script_uses_conservative_memory_defaults():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "recipe" / "mopd" / "run_mopd_qwen3_4b.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert 'TEACHER_LOG_PROB_MICRO_BATCH_SIZE=${TEACHER_LOG_PROB_MICRO_BATCH_SIZE:-2}' in script_text
    assert 'ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}' in script_text
    assert 'REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-true}' in script_text
    assert "algorithm.mopd.teachers=[" in script_text
    assert "log_prob_micro_batch_size:" in script_text
    assert "${TEACHER_LOG_PROB_MICRO_BATCH_SIZE}" in script_text
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}" in script_text
    assert "actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD}" in script_text
