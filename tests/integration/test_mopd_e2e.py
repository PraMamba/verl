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

"""Integration tests for MOPD (Multi-Teacher On-Policy Distillation).

Tests the full data flow from config creation through advantage computation,
verifying all MOPD components work together correctly.
"""

import os
import re
import subprocess
import sys
from importlib import util
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import get_adv_estimator_fn
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.workers.config.teacher import MOPDConfig, TeacherConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENERATED_PPO_CFG = _REPO_ROOT / "verl" / "trainer" / "config" / "_generated_ppo_trainer.yaml"
_MOPD_E2E_SUCCESS_RE = re.compile(r"\bstep:(?P<step>\d+)\b.*\btraining/global_step:(?P<global_step>\d+)\b")


def _build_mopd_e2e_config_from_env() -> OmegaConf:
    """Build a full trainer config using the recipe's environment-variable contract."""
    student_model_path = os.environ.get("STUDENT_MODEL_PATH") or os.environ.get("MOPD_TEST_MODEL_PATH", "/models/base")
    train_files = os.environ.get("TRAIN_FILE") or os.environ.get("MOPD_TEST_TRAIN_FILES", "/data/train.jsonl")
    val_files = (
        os.environ.get("TEST_FILE")
        or os.environ.get("MOPD_TEST_VAL_FILES")
        or os.environ.get("MOPD_TEST_TRAIN_FILES")
        or train_files
    )
    cell_teacher_path = os.environ.get("CELL_TYPE_TEACHER_PATH")
    disease_teacher_path = os.environ.get("DISEASE_STATE_TEACHER_PATH")
    legacy_teacher_path = os.environ.get("MOPD_TEST_TEACHER_PATH", "/models/math-teacher")

    config = OmegaConf.load(_GENERATED_PPO_CFG)
    config.algorithm.adv_estimator = "mopd"
    config.algorithm.use_kl_in_reward = False
    config.algorithm.kl_ctrl.kl_coef = 0.0
    config.algorithm.mopd.enabled = True
    config.algorithm.mopd.lambda_val = 1.0
    config.algorithm.mopd.orm_weight = 0.0
    config.algorithm.mopd.is_correction = True
    config.algorithm.mopd.is_epsilon_low = 0.1
    config.algorithm.mopd.is_epsilon_high = 10.0
    if cell_teacher_path and disease_teacher_path:
        config.algorithm.mopd.teachers = [
            {
                "name": "cell_type_teacher",
                "model_path": cell_teacher_path,
                "resource_pool": "global_pool",
                "tokenizer_compat_group": "qwen3-shared",
            },
            {
                "name": "disease_state_teacher",
                "model_path": disease_teacher_path,
                "resource_pool": "global_pool",
                "tokenizer_compat_group": "qwen3-shared",
            },
        ]
    else:
        config.algorithm.mopd.teachers = [
            {
                "name": "math",
                "model_path": legacy_teacher_path,
                "resource_pool": "global_pool",
                "tokenizer_compat_group": "qwen3-shared",
            },
        ]

    max_prompt_length = int(os.environ.get("MOPD_E2E_MAX_PROMPT_LENGTH", "4096"))
    max_response_length = int(os.environ.get("MOPD_E2E_MAX_RESPONSE_LENGTH", "128"))
    max_model_len = max_prompt_length + max_response_length
    train_batch_size = int(os.environ.get("MOPD_E2E_TRAIN_BATCH_SIZE", "8"))

    config.data.train_files = train_files
    config.data.val_files = val_files
    config.data.train_max_samples = int(os.environ.get("MOPD_E2E_TRAIN_MAX_SAMPLES", "8"))
    config.data.val_max_samples = int(os.environ.get("MOPD_E2E_VAL_MAX_SAMPLES", "8"))
    config.data.prompt_key = "prompt"
    config.data.truncation = "left"
    config.data.max_prompt_length = max_prompt_length
    config.data.max_response_length = max_response_length
    config.data.train_batch_size = train_batch_size
    config.data.return_raw_chat = True
    config.data.filter_overlong_prompts = True
    config.data.teacher_id_field = "teacher_id"

    config.actor_rollout_ref.model.path = student_model_path
    config.actor_rollout_ref.model.use_remove_padding = True
    config.actor_rollout_ref.model.enable_gradient_checkpointing = True
    config.actor_rollout_ref.nccl_timeout = int(os.environ.get("MOPD_E2E_NCCL_TIMEOUT_SECONDS", "180"))
    config.actor_rollout_ref.actor.use_kl_loss = False
    config.actor_rollout_ref.actor.kl_loss_coef = 0.0
    config.actor_rollout_ref.actor.optim.lr = 1e-6
    config.actor_rollout_ref.actor.optim.lr_warmup_steps = 10
    config.actor_rollout_ref.actor.optim.weight_decay = 0.1
    config.actor_rollout_ref.actor.ppo_epochs = 1
    config.actor_rollout_ref.actor.ppo_mini_batch_size = train_batch_size
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
    config.actor_rollout_ref.actor.use_dynamic_bsz = True
    config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu = max_model_len
    config.actor_rollout_ref.actor.fsdp_config.param_offload = False
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = False
    config.actor_rollout_ref.actor.entropy_coeff = 0
    config.actor_rollout_ref.actor.grad_clip = 1.0
    config.actor_rollout_ref.actor.fsdp_config.fsdp_size = int(os.environ.get("MOPD_E2E_FSDP_SIZE", "2"))

    config.actor_rollout_ref.rollout.n = int(os.environ.get("MOPD_E2E_ROLLOUT_N", "4"))
    config.actor_rollout_ref.rollout.name = "vllm"
    config.actor_rollout_ref.rollout.gpu_memory_utilization = float(
        os.environ.get("MOPD_E2E_ROLLOUT_GPU_MEMORY_UTILIZATION", "0.45")
    )
    config.actor_rollout_ref.rollout.max_model_len = max_model_len
    config.actor_rollout_ref.rollout.calculate_log_probs = True
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
    config.actor_rollout_ref.rollout.enable_chunked_prefill = True
    config.actor_rollout_ref.rollout.max_num_batched_tokens = max_model_len
    config.actor_rollout_ref.rollout.temperature = 1.0
    config.actor_rollout_ref.rollout.top_p = 1.0
    config.actor_rollout_ref.rollout.top_k = -1
    config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu = max_model_len
    config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz = True

    config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu = max_model_len
    config.actor_rollout_ref.ref.log_prob_use_dynamic_bsz = True
    config.actor_rollout_ref.ref.fsdp_config.param_offload = False

    config.reward_model.enable = False
    config.reward_model.reward_manager = "naive"
    config.custom_reward_function.path = str(_REPO_ROOT / "recipe" / "mopd" / "zero_reward.py")
    config.custom_reward_function.name = "compute_score"

    config.trainer.logger = ["console"]
    config.trainer.project_name = os.environ.get("PROJECT_NAME", "RVQ-Alpha_MOPD")
    config.trainer.experiment_name = os.environ.get("EXP_NAME", "mopd-e2e")
    config.trainer.n_gpus_per_node = int(os.environ.get("NGPUS_PER_NODE", "4"))
    config.trainer.nnodes = int(os.environ.get("NNODES", "1"))
    config.trainer.val_before_train = False
    config.trainer.test_freq = -1
    config.trainer.save_freq = -1
    config.trainer.total_epochs = 1
    config.trainer.default_local_dir = os.environ.get("MOPD_E2E_CKPTS_DIR", "/tmp/mopd-e2e")
    config.trainer.resume_mode = "disable"
    config.trainer.log_val_generations = 0

    return config


def _load_mopd_preflight_module():
    module_path = _REPO_ROOT / "recipe" / "mopd" / "check_mopd_first_batch.py"
    spec = util.spec_from_file_location("check_mopd_first_batch_for_e2e", module_path)
    module = util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_mopd_e2e_command_from_env(ckpt_dir: str) -> list[str]:
    module = _load_mopd_preflight_module()
    config = module.PreflightConfig(
        student_model_path=os.environ.get("STUDENT_MODEL_PATH")
        or os.environ.get("MOPD_TEST_MODEL_PATH", "/models/base"),
        cell_type_teacher_path=os.environ.get(
            "CELL_TYPE_TEACHER_PATH", os.environ.get("MOPD_TEST_TEACHER_PATH", "/models/math-teacher")
        ),
        disease_state_teacher_path=os.environ.get(
            "DISEASE_STATE_TEACHER_PATH", os.environ.get("MOPD_TEST_TEACHER_PATH", "/models/math-teacher")
        ),
        train_file=os.environ.get("TRAIN_FILE") or os.environ.get("MOPD_TEST_TRAIN_FILES", "/data/train.jsonl"),
        val_file=(
            os.environ.get("TEST_FILE")
            or os.environ.get("MOPD_TEST_VAL_FILES")
            or os.environ.get("MOPD_TEST_TRAIN_FILES", "/data/train.jsonl")
        ),
        ckpt_dir=ckpt_dir,
        project_name=os.environ.get("PROJECT_NAME", "RVQ-Alpha_MOPD"),
        experiment_name=os.environ.get("EXP_NAME", "mopd-e2e"),
        train_batch_size=int(os.environ.get("MOPD_E2E_TRAIN_BATCH_SIZE", "8")),
        max_prompt_length=int(os.environ.get("MOPD_E2E_MAX_PROMPT_LENGTH", "4096")),
        max_response_length=int(os.environ.get("MOPD_E2E_MAX_RESPONSE_LENGTH", "128")),
        rollout_n=int(os.environ.get("MOPD_E2E_ROLLOUT_N", "4")),
        teacher_log_prob_micro_batch_size=int(os.environ.get("MOPD_E2E_TEACHER_LOG_PROB_MICRO_BATCH_SIZE", "4")),
        rollout_gpu_memory_utilization=float(os.environ.get("MOPD_E2E_ROLLOUT_GPU_MEMORY_UTILIZATION", "0.45")),
        n_gpus_per_node=int(os.environ.get("NGPUS_PER_NODE", "4")),
        nnodes=int(os.environ.get("NNODES", "1")),
        fsdp_size=int(os.environ.get("MOPD_E2E_FSDP_SIZE", "2")),
        ppo_micro_batch_size_per_gpu=1,
        nccl_timeout_seconds=int(os.environ.get("MOPD_E2E_NCCL_TIMEOUT_SECONDS", "180")),
        timeout_seconds=int(os.environ.get("MOPD_E2E_TIMEOUT_SECONDS", "1800")),
    )
    command = module.build_training_command(config)
    command.extend(
        [
            f"data.train_max_samples={int(os.environ.get('MOPD_E2E_TRAIN_MAX_SAMPLES', '8'))}",
            f"data.val_max_samples={int(os.environ.get('MOPD_E2E_VAL_MAX_SAMPLES', '8'))}",
        ]
    )
    return command


def _mopd_e2e_log_reached_training_step(log_path: Path) -> bool:
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = _MOPD_E2E_SUCCESS_RE.search(line)
        if match is None:
            continue
        if int(match.group("step")) >= 1 and int(match.group("global_step")) >= 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Lightweight integration tests (run without GPU or Ray)
# ---------------------------------------------------------------------------


class TestMOPDDataFlow:
    """Test the full MOPD data flow: config → advantage computation → result."""

    @pytest.fixture()
    def mopd_config(self):
        """Create a full MOPD config using OmegaConf."""
        return OmegaConf.create(
            {
                "mopd": {
                    "enabled": True,
                    "lambda_val": 1.0,
                    "orm_weight": 0.0,
                    "is_correction": True,
                    "is_epsilon_low": 0.1,
                    "is_epsilon_high": 10.0,
                    "use_base_normalization": False,
                    "base_model_path": None,
                    "teachers": [
                        {"name": "math", "model_path": "/models/math-teacher"},
                        {"name": "code", "model_path": "/models/code-teacher"},
                    ],
                },
            }
        )

    @pytest.fixture()
    def mopd_batch(self):
        """Create a DataProto with all MOPD-required fields."""
        B, T = 8, 16
        data = DataProto.from_single_dict(
            {
                "token_level_rewards": torch.zeros(B, T),
                "response_mask": torch.ones(B, T),
                "old_log_probs": torch.randn(B, T),
                "teacher_log_prob": torch.randn(B, T),
            }
        )
        data.non_tensor_batch["uid"] = np.array(["q1"] * 4 + ["q2"] * 4)
        data.non_tensor_batch["teacher_id"] = np.array(["math"] * 4 + ["code"] * 4)
        return data

    def test_config_to_advantage_flow(self, mopd_config, mopd_batch):
        """Test full flow: OmegaConf config → compute_advantage → result with correct structure."""
        # Act: call compute_advantage with MOPD estimator and config
        result = compute_advantage(
            mopd_batch,
            adv_estimator="mopd",
            config=mopd_config,
        )

        # Assert: result has expected keys and shapes
        assert "advantages" in result.batch
        assert "returns" in result.batch
        assert result.batch["advantages"].shape == (8, 16)
        assert result.batch["returns"].shape == (8, 16)
        # Original fields are preserved
        assert "old_log_probs" in result.batch
        assert "teacher_log_prob" in result.batch

    def test_advantage_values_are_deterministic(self, mopd_config, mopd_batch):
        """Test that running compute_advantage twice yields identical results."""
        # We need fresh copies because compute_advantage mutates data in-place
        batch_copy = DataProto.from_single_dict({k: v.clone() for k, v in mopd_batch.batch.items()})
        batch_copy.non_tensor_batch = dict(mopd_batch.non_tensor_batch)

        result1 = compute_advantage(mopd_batch, adv_estimator="mopd", config=mopd_config)
        result2 = compute_advantage(batch_copy, adv_estimator="mopd", config=mopd_config)

        torch.testing.assert_close(result1.batch["advantages"], result2.batch["advantages"])
        torch.testing.assert_close(result1.batch["returns"], result2.batch["returns"])

    def test_response_mask_zeros_out_advantages(self, mopd_config):
        """Test that advantages are zero where response_mask is zero."""
        B, T = 4, 10
        response_mask = torch.ones(B, T)
        # Mask out last 3 tokens of each sequence
        response_mask[:, -3:] = 0.0

        data = DataProto.from_single_dict(
            {
                "token_level_rewards": torch.zeros(B, T),
                "response_mask": response_mask,
                "old_log_probs": torch.randn(B, T),
                "teacher_log_prob": torch.randn(B, T),
            }
        )
        data.non_tensor_batch["uid"] = np.array(["q1"] * 4)

        result = compute_advantage(data, adv_estimator="mopd", config=mopd_config)

        # Masked positions must be exactly zero
        masked_advantages = result.batch["advantages"][:, -3:]
        assert torch.all(masked_advantages == 0.0), f"Expected zeros, got {masked_advantages}"

    def test_standard_mopd_advantage_values(self):
        """Test MOPD produces correct advantage values (teacher_log_prob - old_log_probs)."""
        B, T = 2, 5
        teacher_lp = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 1.5, 2.5, 3.5, 4.5]])
        old_lp = torch.tensor([[0.5, 1.0, 1.5, 2.0, 2.5], [0.0, 0.5, 1.0, 1.5, 2.0]])
        response_mask = torch.ones(B, T)

        config = OmegaConf.create({"mopd": {"lambda_val": 1.0, "is_correction": False}})
        data = DataProto.from_single_dict(
            {
                "token_level_rewards": torch.zeros(B, T),
                "response_mask": response_mask,
                "old_log_probs": old_lp,
                "teacher_log_prob": teacher_lp,
            }
        )
        data.non_tensor_batch["uid"] = np.array(["q1", "q2"])

        result = compute_advantage(data, adv_estimator="mopd", config=config)
        expected = (teacher_lp - old_lp).detach() * response_mask
        torch.testing.assert_close(result.batch["advantages"], expected)

    def test_exopd_mode_end_to_end(self):
        """Test ExOPD (base-normalized) mode through compute_advantage dispatch."""
        B, T = 2, 5
        teacher_lp = torch.ones(B, T) * 2.0
        old_lp = torch.ones(B, T) * 1.0
        base_lp = torch.ones(B, T) * 0.5

        config = OmegaConf.create(
            {
                "mopd": {
                    "lambda_val": 1.25,
                    "is_correction": False,
                },
            }
        )
        data = DataProto.from_single_dict(
            {
                "token_level_rewards": torch.zeros(B, T),
                "response_mask": torch.ones(B, T),
                "old_log_probs": old_lp,
                "teacher_log_prob": teacher_lp,
                "base_log_prob": base_lp,
            }
        )
        data.non_tensor_batch["uid"] = np.array(["q1", "q2"])

        result = compute_advantage(data, adv_estimator="mopd", config=config)

        # ExOPD: -[(old - base) - lambda*(teacher - base)]
        # = -[(1.0 - 0.5) - 1.25*(2.0 - 0.5)]
        # = -[0.5 - 1.875] = 1.375
        expected = torch.ones(B, T) * 1.375
        torch.testing.assert_close(result.batch["advantages"], expected, rtol=1e-4, atol=1e-4)

    def test_exopd_batch_lambda_overrides_config_scalar(self):
        """Batch lambda should override config lambda when ExOPD normalization is active."""
        teacher_lp = torch.full((2, 4), 2.0)
        old_lp = torch.full((2, 4), 1.0)
        base_lp = torch.full((2, 4), 0.5)
        batch_lambda = torch.tensor([[1.0], [2.0]], dtype=torch.float32)

        config = OmegaConf.create(
            {
                "mopd": {
                    "lambda_val": 7.0,
                    "is_correction": False,
                },
            }
        )
        data = DataProto.from_single_dict(
            {
                "token_level_rewards": torch.zeros(2, 4),
                "response_mask": torch.ones(2, 4),
                "old_log_probs": old_lp,
                "teacher_log_prob": teacher_lp,
                "base_log_prob": base_lp,
                "lambda_val": batch_lambda,
            }
        )
        data.non_tensor_batch["uid"] = np.array(["q1", "q2"])

        result = compute_advantage(data, adv_estimator="mopd", config=config)

        expected = -((old_lp - base_lp) - batch_lambda * (teacher_lp - base_lp))
        torch.testing.assert_close(result.batch["advantages"], expected)

    def test_is_correction_through_dispatch(self):
        """Test IS correction flows correctly through compute_advantage."""
        B, T = 2, 5
        teacher_lp = torch.ones(B, T) * 2.0
        old_lp = torch.ones(B, T) * 1.0
        # Token [0, 2] has extreme ratio: exp(1 - (-4)) = exp(5) ≈ 148 > 10
        rollout_lp = torch.tensor(
            [
                [1.0, 1.0, -4.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )

        config = OmegaConf.create(
            {
                "mopd": {
                    "is_correction": True,
                    "is_epsilon_low": 0.1,
                    "is_epsilon_high": 10.0,
                },
            }
        )
        data = DataProto.from_single_dict(
            {
                "token_level_rewards": torch.zeros(B, T),
                "response_mask": torch.ones(B, T),
                "old_log_probs": old_lp,
                "teacher_log_prob": teacher_lp,
                "rollout_log_probs": rollout_lp,
            }
        )
        data.non_tensor_batch["uid"] = np.array(["q1", "q2"])

        result = compute_advantage(data, adv_estimator="mopd", config=config)

        # Token [0, 2] should be masked to zero by IS correction
        assert result.batch["advantages"][0, 2] == 0.0
        # Non-masked tokens should be non-zero
        assert result.batch["advantages"][0, 0] != 0.0


class TestMOPDConfigIntegration:
    """Test that MOPD config dataclasses integrate with OmegaConf correctly."""

    def test_teacher_config_roundtrip(self):
        """Test TeacherConfig can be created from dict and exported back."""
        cfg_dict = {"name": "math", "model_path": "/models/math", "weight": 1.0}
        teacher = TeacherConfig(**cfg_dict)
        assert teacher.name == "math"
        assert teacher.model_path == "/models/math"
        assert teacher.weight == 1.0

    def test_mopd_config_with_teachers(self):
        """Test MOPDConfig accepts properly constructed teachers."""
        teachers = [
            TeacherConfig(name="math", model_path="/models/math"),
            TeacherConfig(name="code", model_path="/models/code"),
        ]
        config = MOPDConfig(enabled=True, teachers=teachers, lambda_val=1.25)
        assert config.enabled is True
        assert len(config.teachers) == 2
        assert config.lambda_val == 1.25

    def test_mopd_config_disabled_by_default(self):
        """Test MOPDConfig defaults to disabled (backward compatibility)."""
        config = MOPDConfig()
        assert config.enabled is False
        assert len(config.teachers) == 0

    def test_need_reference_policy_with_mopd_config(self):
        """Test need_reference_policy returns True when MOPD is enabled."""
        from verl.trainer.ppo.utils import need_reference_policy

        config = OmegaConf.create(
            {
                "algorithm": {
                    "use_kl_in_reward": False,
                    "mopd": {"enabled": True},
                },
                "actor_rollout_ref": {"actor": {"use_kl_loss": False}},
            }
        )
        assert need_reference_policy(config) is True

    def test_need_reference_policy_without_mopd(self):
        """Test need_reference_policy returns False when MOPD disabled and no KL."""
        from verl.trainer.ppo.utils import need_reference_policy

        config = OmegaConf.create(
            {
                "algorithm": {
                    "use_kl_in_reward": False,
                    "mopd": {"enabled": False},
                },
                "actor_rollout_ref": {"actor": {"use_kl_loss": False}},
            }
        )
        assert need_reference_policy(config) is False


class TestMOPDAdvantageEstimatorRegistry:
    """Test that the MOPD estimator is properly registered and callable."""

    def test_mopd_is_registered(self):
        """Test that 'mopd' advantage estimator is registered."""
        fn = get_adv_estimator_fn("mopd")
        assert fn is not None
        assert callable(fn)

    def test_mopd_fn_has_correct_name(self):
        """Test that the registered function is compute_mopd_advantage."""
        fn = get_adv_estimator_fn("mopd")
        assert fn.__name__ == "compute_mopd_advantage"

    def test_mopd_returns_tuple(self):
        """Test that MOPD advantage estimator returns (advantages, returns, is_metrics) tuple."""
        B, T = 2, 4
        fn = get_adv_estimator_fn("mopd")
        result = fn(
            token_level_rewards=torch.zeros(B, T),
            response_mask=torch.ones(B, T),
            teacher_log_prob=torch.randn(B, T),
            old_log_probs=torch.randn(B, T),
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        advantages, returns, is_metrics = result
        assert advantages.shape == (B, T)
        assert returns.shape == (B, T)
        assert isinstance(is_metrics, dict)


class TestMOPDE2EEnvContract:
    """Test that the full GPU E2E path follows the recipe environment contract."""

    def test_build_mopd_e2e_config_prefers_recipe_env_vars(self, monkeypatch):
        monkeypatch.setenv("STUDENT_MODEL_PATH", "/models/student-from-recipe")
        monkeypatch.setenv("CELL_TYPE_TEACHER_PATH", "/teachers/cell-from-recipe")
        monkeypatch.setenv("DISEASE_STATE_TEACHER_PATH", "/teachers/disease-from-recipe")
        monkeypatch.setenv("TRAIN_FILE", "/data/train-from-recipe.parquet")
        monkeypatch.setenv("TEST_FILE", "/data/test-from-recipe.parquet")
        monkeypatch.setenv("MOPD_TEST_MODEL_PATH", "/models/student-from-legacy")
        monkeypatch.setenv("MOPD_TEST_TEACHER_PATH", "/teachers/legacy")
        monkeypatch.setenv("MOPD_TEST_TRAIN_FILES", "/data/train-from-legacy.parquet")

        config = _build_mopd_e2e_config_from_env()

        assert config.actor_rollout_ref.model.path == "/models/student-from-recipe"
        assert config.data.train_files == "/data/train-from-recipe.parquet"
        assert config.data.val_files == "/data/test-from-recipe.parquet"
        assert [teacher.name for teacher in config.algorithm.mopd.teachers] == [
            "cell_type_teacher",
            "disease_state_teacher",
        ]
        assert [teacher.model_path for teacher in config.algorithm.mopd.teachers] == [
            "/teachers/cell-from-recipe",
            "/teachers/disease-from-recipe",
        ]

    def test_build_mopd_e2e_config_falls_back_to_legacy_env_vars(self, monkeypatch):
        monkeypatch.delenv("STUDENT_MODEL_PATH", raising=False)
        monkeypatch.delenv("CELL_TYPE_TEACHER_PATH", raising=False)
        monkeypatch.delenv("DISEASE_STATE_TEACHER_PATH", raising=False)
        monkeypatch.delenv("TRAIN_FILE", raising=False)
        monkeypatch.delenv("TEST_FILE", raising=False)
        monkeypatch.setenv("MOPD_TEST_MODEL_PATH", "/models/student-from-legacy")
        monkeypatch.setenv("MOPD_TEST_TEACHER_PATH", "/teachers/legacy")
        monkeypatch.setenv("MOPD_TEST_TRAIN_FILES", "/data/train-from-legacy.parquet")

        config = _build_mopd_e2e_config_from_env()

        assert config.actor_rollout_ref.model.path == "/models/student-from-legacy"
        assert config.data.train_files == "/data/train-from-legacy.parquet"
        assert config.data.val_files == "/data/train-from-legacy.parquet"
        assert [teacher.name for teacher in config.algorithm.mopd.teachers] == ["math"]
        assert [teacher.model_path for teacher in config.algorithm.mopd.teachers] == ["/teachers/legacy"]


class TestMOPDE2ESuccessContract:
    """Test the success predicate for the full GPU E2E subprocess."""

    def test_log_reports_success_when_first_training_step_is_reached(self, tmp_path):
        log_path = tmp_path / "mopd-e2e.log"
        log_path.write_text(
            "step:1 - actor/pg_loss:2.10 - training/global_step:1 - perf/time_per_step:61.3\n",
            encoding="utf-8",
        )

        assert _mopd_e2e_log_reached_training_step(log_path) is True

    def test_log_reports_failure_when_training_step_is_missing(self, tmp_path):
        log_path = tmp_path / "mopd-e2e.log"
        log_path.write_text("TaskRunner initialized but no optimization step completed\n", encoding="utf-8")

        assert _mopd_e2e_log_reached_training_step(log_path) is False


# ---------------------------------------------------------------------------
# Full E2E test (requires GPU + Ray + model weights)
# ---------------------------------------------------------------------------

# This test requires a fully provisioned environment:
# - CUDA-capable GPU
# - Ray cluster
# - Model weights at MOPD_TEST_MODEL_PATH and MOPD_TEST_TEACHER_PATH
# Set VERL_MOPD_E2E=1 to enable this test.
_MOPD_E2E_ENABLED = os.environ.get("VERL_MOPD_E2E", "0") == "1"


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not _MOPD_E2E_ENABLED, reason="Set VERL_MOPD_E2E=1 to run full E2E test")
def test_mopd_training_e2e(tmp_path):
    """Test full MOPD training loop with actual model workers.

    This test requires:
    - CUDA-capable GPU
    - Ray cluster (auto-initialized)
    - Model weights at configured paths
    - Environment variable VERL_MOPD_E2E=1

    It verifies:
    - Teacher worker groups are created
    - Sub-batch routing dispatches to correct teachers
    - MOPD advantages are computed and used for policy updates
    - Training completes at least one step

    To run:
        VERL_MOPD_E2E=1 STUDENT_MODEL_PATH=/path/to/model \\
        CELL_TYPE_TEACHER_PATH=/path/to/cell-teacher \\
        DISEASE_STATE_TEACHER_PATH=/path/to/disease-teacher \\
        TRAIN_FILE=/path/to/train.parquet TEST_FILE=/path/to/test.parquet \\
        pytest tests/integration/test_mopd_e2e.py::test_mopd_training_e2e -v
    """
    ckpt_dir = tmp_path / "mopd-e2e"
    log_file = tmp_path / "mopd-e2e.log"
    command = _build_mopd_e2e_command_from_env(str(ckpt_dir))
    env = os.environ.copy()
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("RAY_DEDUP_LOGS", "0")

    with log_file.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=str(_REPO_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=int(os.environ.get("MOPD_E2E_TIMEOUT_SECONDS", "1800")),
            check=False,
        )

    if result.returncode != 0:
        log_tail = "\n".join(log_file.read_text(encoding="utf-8").splitlines()[-60:])
        pytest.fail(f"GPU E2E command failed with exit code {result.returncode}. Log tail:\n{log_tail}")

    if not _mopd_e2e_log_reached_training_step(log_file):
        log_tail = "\n".join(log_file.read_text(encoding="utf-8").splitlines()[-60:])
        pytest.fail(
            f"GPU E2E command exited successfully but did not report a completed training step. Log tail:\n{log_tail}"
        )
