# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the MOPD first-batch preflight helper."""

from importlib import util
from pathlib import Path
import sys

import pytest


def _load_preflight_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "recipe" / "mopd" / "check_mopd_first_batch.py"
    spec = util.spec_from_file_location("check_mopd_first_batch", module_path)
    module = util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_training_command_uses_first_batch_overrides():
    module = _load_preflight_module()

    config = module.PreflightConfig(
        student_model_path="/models/student",
        cell_type_teacher_path="/teachers/cell",
        disease_state_teacher_path="/teachers/disease",
        train_file="/data/mopd_train.parquet",
        val_file="/data/mopd_test.parquet",
        ckpt_dir="/tmp/mopd-preflight",
        project_name="RVQ-Alpha_MOPD",
        experiment_name="mopd-preflight",
        train_batch_size=8,
        max_prompt_length=4096,
        max_response_length=128,
        rollout_n=4,
        n_gpus_per_node=4,
        nnodes=1,
        teacher_log_prob_micro_batch_size=4,
        nccl_timeout_seconds=180,
        rvq_reward_root="/rvq/root",
        reward_provider="sglang",
        reward_api_base="http://127.0.0.1:30000/v1",
        reward_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        reward_api_key="EMPTY",
        reward_num_workers=2,
        reward_max_concurrent=3,
        reward_max_rpm_per_worker=120,
        reward_max_tpm_per_worker=120000,
        reward_est_tokens_per_request=18000,
        reward_timeout=900.0,
    )

    command = module.build_training_command(config)

    assert command[:3] == [module.sys.executable, "-m", "verl.trainer.main_ppo"]
    assert "data.train_batch_size=8" in command
    assert "data.max_response_length=128" in command
    assert "trainer.val_before_train=False" in command
    assert "trainer.save_freq=-1" in command
    assert "trainer.test_freq=-1" in command
    assert "trainer.logger=[console]" in command
    assert "actor_rollout_ref.nccl_timeout=180" in command
    assert "actor_rollout_ref.rollout.n=4" in command
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.45" in command
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=8" in command
    assert "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4224" in command
    assert "actor_rollout_ref.rollout.max_model_len=4224" in command
    assert "reward.num_workers=2" in command
    assert "reward.reward_model.enable=False" in command
    assert "reward.reward_manager.name=rate_limited" in command
    assert "+reward.max_concurrent=3" in command
    assert "+reward.max_rpm=120" in command
    assert "+reward.max_tpm=120000" in command
    assert "+reward.estimated_tokens_per_request=18000" in command
    assert "+reward.timeout=900.0" in command
    assert f"reward.custom_reward_function.path={module.SCRIPT_DIR / 'rvq_v1_reward.py'}" in command
    assert "reward.custom_reward_function.name=compute_score" in command
    assert "+reward.custom_reward_function.reward_kwargs.rvq_reward_root=/rvq/root" in command
    assert "+reward.custom_reward_function.reward_kwargs.reward_version=v1" in command
    assert "+reward.custom_reward_function.reward_kwargs.provider=sglang" in command
    assert "+reward.custom_reward_function.reward_kwargs.api_base=http://127.0.0.1:30000/v1" in command
    assert "+reward.custom_reward_function.reward_kwargs.model=Qwen/Qwen3-30B-A3B-Instruct-2507" in command
    assert "+reward.custom_reward_function.reward_kwargs.api_key=EMPTY" in command
    assert "+reward.custom_reward_function.reward_kwargs.timeout=900.0" in command
    assert "+reward.custom_reward_function.reward_kwargs.max_response_length=128" in command

    teacher_override = next(item for item in command if item.startswith("algorithm.mopd.teachers=["))
    assert "name: cell_type_teacher" in teacher_override
    assert "model_path: '/teachers/cell'" in teacher_override
    assert "name: disease_state_teacher" in teacher_override
    assert "model_path: '/teachers/disease'" in teacher_override
    assert "log_prob_micro_batch_size: 4" in teacher_override
    assert "tokenizer_compat_group: 'qwen3-shared'" in teacher_override


def test_build_training_command_allows_rollout_n_override():
    module = _load_preflight_module()

    config = module.PreflightConfig(
        student_model_path="/models/student",
        cell_type_teacher_path="/teachers/cell",
        disease_state_teacher_path="/teachers/disease",
        train_file="/data/mopd_train.parquet",
        val_file="/data/mopd_test.parquet",
        ckpt_dir="/tmp/mopd-preflight",
        project_name="RVQ-Alpha_MOPD",
        experiment_name="mopd-preflight",
        rollout_n=8,
    )

    command = module.build_training_command(config)

    assert "actor_rollout_ref.rollout.n=8" in command


def test_build_training_command_defaults_to_rvq_v1_reward_root():
    module = _load_preflight_module()

    config = module.PreflightConfig(
        student_model_path="/models/student",
        cell_type_teacher_path="/teachers/cell",
        disease_state_teacher_path="/teachers/disease",
        train_file="/data/mopd_train.parquet",
        val_file="/data/mopd_test.parquet",
        ckpt_dir="/tmp/mopd-preflight",
        project_name="RVQ-Alpha_MOPD",
        experiment_name="mopd-preflight",
    )

    assert config.rvq_reward_root == "/home/scbjtfy/RVQ-Alpha/rlvr"
    assert config.reward_function_path.endswith("recipe/mopd/rvq_v1_reward.py")
    assert config.train_batch_size == 2


def test_validate_preflight_config_accepts_smoke_safe_minimum():
    module = _load_preflight_module()

    config = module.PreflightConfig(
        student_model_path="/models/student",
        cell_type_teacher_path="/teachers/cell",
        disease_state_teacher_path="/teachers/disease",
        train_file="/data/mopd_train.parquet",
        val_file="/data/mopd_test.parquet",
        ckpt_dir="/tmp/mopd-preflight",
        project_name="RVQ-Alpha_MOPD",
        experiment_name="mopd-preflight",
        train_batch_size=2,
        rollout_n=4,
    )

    module.validate_preflight_config(config)


def test_validate_preflight_config_rejects_undersized_smoke_batch():
    module = _load_preflight_module()

    config = module.PreflightConfig(
        student_model_path="/models/student",
        cell_type_teacher_path="/teachers/cell",
        disease_state_teacher_path="/teachers/disease",
        train_file="/data/mopd_train.parquet",
        val_file="/data/mopd_test.parquet",
        ckpt_dir="/tmp/mopd-preflight",
        project_name="RVQ-Alpha_MOPD",
        experiment_name="mopd-preflight",
        train_batch_size=1,
        rollout_n=4,
    )

    with pytest.raises(ValueError, match="train_batch_size \\* rollout_n"):
        module.validate_preflight_config(config)


def test_detect_terminal_event_ignores_validation_only_logs():
    module = _load_preflight_module()

    event = module.detect_terminal_event(
        "step:0 - val-core/population_cell_type_homogeneous_test/acc/mean@1:0.0"
    )

    assert event is None


def test_detect_terminal_event_ignores_hydra_banner_without_root_cause():
    module = _load_preflight_module()

    event = module.detect_terminal_event("Error executing job with overrides: ['foo=bar']")

    assert event is None


def test_detect_terminal_event_recognizes_first_actor_update_success():
    module = _load_preflight_module()

    event = module.detect_terminal_event(
        "step:1 - training/global_step:1 - training/epoch:0 - actor/grad_norm:0.42 - perf/time_per_step:12.3"
    )

    assert event is not None
    assert event.status == module.PreflightStatus.SUCCESS
    assert event.reason == "first_actor_update"


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        (
            "[rank0]:[E312 14:28:21] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(...)",
            "nccl_timeout",
        ),
        (
            "ray.exceptions.ActorDiedError: The actor died unexpectedly before finishing this task.",
            "actor_died",
        ),
        (
            "Fatal Python error: Aborted",
            "fatal_python",
        ),
        (
            "!!!!!!! Segfault encountered !!!!!!!",
            "segfault",
        ),
    ],
)
def test_detect_terminal_event_recognizes_failure_markers(line, reason):
    module = _load_preflight_module()

    event = module.detect_terminal_event(line)

    assert event is not None
    assert event.status == module.PreflightStatus.FAILURE
    assert event.reason == reason
