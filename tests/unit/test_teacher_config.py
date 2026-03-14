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

import pytest

from verl.workers.config.teacher import MOPDConfig, TeacherConfig, TeacherResourcePoolConfig


def test_teacher_config_requires_name():
    with pytest.raises(ValueError, match="name must be non-empty"):
        TeacherConfig(name="", model_path="/models/test")


def test_teacher_config_requires_model_path():
    with pytest.raises(ValueError, match="model_path must be non-empty"):
        TeacherConfig(name="test", model_path="")


def test_mopd_config_rejects_duplicate_teacher_names():
    teachers = [
        TeacherConfig(name="math", model_path="/models/math"),
        TeacherConfig(name="math", model_path="/models/math2"),
    ]
    with pytest.raises(ValueError, match="Duplicate teacher names"):
        MOPDConfig(enabled=True, teachers=teachers)


def test_mopd_config_validates_lambda():
    with pytest.raises(ValueError, match="lambda_val must be positive"):
        MOPDConfig(enabled=False, lambda_val=0.0)


def test_teacher_config_validates_optional_lambda():
    with pytest.raises(ValueError, match="lambda_val must be positive"):
        TeacherConfig(name="math", model_path="/models/math", lambda_val=0.0)


def test_teacher_config_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend must be one of"):
        TeacherConfig(name="math", model_path="/models/math", backend="fp8_ref")


def test_teacher_config_rejects_unknown_tokenizer_policy():
    with pytest.raises(ValueError, match="tokenizer_policy must be one of"):
        TeacherConfig(name="math", model_path="/models/math", tokenizer_policy="orm_bridge")


def test_teacher_config_rejects_non_positive_seq_reward_weight():
    with pytest.raises(ValueError, match="seq_reward_weight must be positive"):
        TeacherConfig(name="math", model_path="/models/math", seq_reward_weight=0.0)


def test_teacher_config_exposes_backend_and_sequence_policy_fields():
    teacher_cfg = TeacherConfig(
        name="math",
        model_path="/models/math",
        backend="hf_int8",
        tokenizer_policy="sequence_reward",
        seq_reward_weight=0.25,
    )

    assert teacher_cfg.backend == "hf_int8"
    assert teacher_cfg.tokenizer_policy == "sequence_reward"
    assert teacher_cfg.seq_reward_weight == pytest.approx(0.25)


def test_mopd_config_validates_epsilon_bounds():
    with pytest.raises(ValueError, match=r"is_epsilon_low .* must be < is_epsilon_high"):
        MOPDConfig(enabled=False, is_epsilon_low=10.0, is_epsilon_high=1.0)


def test_mopd_config_rejects_empty_teachers_when_enabled():
    with pytest.raises(ValueError, match="requires at least one teacher"):
        MOPDConfig(enabled=True, teachers=[])


def test_teacher_resource_pool_config_validates_positive_dimensions():
    with pytest.raises(ValueError, match="nnodes must be positive"):
        TeacherResourcePoolConfig(nnodes=0, n_gpus_per_node=1)

    with pytest.raises(ValueError, match="n_gpus_per_node must be positive"):
        TeacherResourcePoolConfig(nnodes=1, n_gpus_per_node=0)

    with pytest.raises(ValueError, match="max_colocate_count must be positive"):
        TeacherResourcePoolConfig(nnodes=1, n_gpus_per_node=1, max_colocate_count=0)
