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

"""Minimal tests for MOPD teacher worker config parsing logic.

These tests verify that the MOPD teacher configuration structure is
properly accessible from OmegaConf and can be parsed as expected by
the trainer's init_workers() method. Full worker initialization tests
require a Ray cluster and GPU, so they belong in integration tests.
"""

from importlib import import_module, util

import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.workers.teacher_workers import HFQuantizedTeacherWorker


class _FakeSeqTokenizer:
    pad_token_id = 0
    padding_side = "left"

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        token_ids = []
        for message in messages:
            base_token = 1 if message["role"] == "user" else 4
            content_tokens = [base_token + idx + 1 for idx, _ in enumerate(message["content"].split())]
            token_ids.extend([base_token, *content_tokens])
        return token_ids


class _RecordingModel:
    def __init__(self, vocab_size=16):
        self.vocab_size = vocab_size
        self.batch_sizes = []

    def __call__(self, *, input_ids, attention_mask):
        del attention_mask
        self.batch_sizes.append(int(input_ids.shape[0]))
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, dtype=torch.float32)
        return type("Outputs", (), {"logits": logits})()


def test_teacher_config_accessible_from_algorithm():
    """Verify MOPD teacher config is properly structured and accessible."""
    config = OmegaConf.create(
        {
            "algorithm": {
                "mopd": {
                    "enabled": True,
                    "teachers": [
                        {"name": "math", "model_path": "/models/math", "resource_pool": "global_pool"},
                        {"name": "code", "model_path": "/models/code", "resource_pool": "global_pool"},
                    ],
                }
            }
        }
    )

    assert config.algorithm.mopd.enabled is True
    assert len(config.algorithm.mopd.teachers) == 2
    assert config.algorithm.mopd.teachers[0].name == "math"
    assert config.algorithm.mopd.teachers[0].model_path == "/models/math"
    assert config.algorithm.mopd.teachers[1].name == "code"
    assert config.algorithm.mopd.teachers[1].model_path == "/models/code"


def test_teacher_resource_pools_accessible_from_algorithm():
    """Verify optional MOPD teacher resource pools are accessible from config."""
    config = OmegaConf.create(
        {
            "algorithm": {
                "mopd": {
                    "enabled": True,
                    "resource_pools": {
                        "code_pool": {
                            "nnodes": 1,
                            "n_gpus_per_node": 4,
                            "max_colocate_count": 1,
                        }
                    },
                    "teachers": [
                        {"name": "code", "model_path": "/models/code", "resource_pool": "code_pool"},
                    ],
                }
            }
        }
    )

    assert config.algorithm.mopd.resource_pools.code_pool.nnodes == 1
    assert config.algorithm.mopd.resource_pools.code_pool.n_gpus_per_node == 4
    assert config.algorithm.mopd.resource_pools.code_pool.max_colocate_count == 1


def test_mopd_disabled_by_default():
    """Verify MOPD defaults to disabled when not configured."""
    config = OmegaConf.create({"algorithm": {}})

    # The get() pattern used in init_workers() should return False when mopd is absent
    assert config.algorithm.get("mopd", {}).get("enabled", False) is False


def test_mopd_config_with_all_teacher_fields():
    """Verify all teacher config fields are accessible."""
    config = OmegaConf.create(
        {
            "algorithm": {
                "mopd": {
                    "enabled": True,
                    "teachers": [
                        {
                            "name": "math",
                            "model_path": "/models/math-teacher",
                            "weight": 1.0,
                            "resource_pool": "global_pool",
                            "log_prob_micro_batch_size": 8,
                            "base_model_path": None,
                        },
                    ],
                }
            }
        }
    )

    teacher = config.algorithm.mopd.teachers[0]
    assert teacher.name == "math"
    assert teacher.model_path == "/models/math-teacher"
    assert teacher.weight == 1.0
    assert teacher.resource_pool == "global_pool"
    assert teacher.log_prob_micro_batch_size == 8
    assert teacher.base_model_path is None


def test_mopd_config_iterates_over_teachers():
    """Verify teachers list can be iterated (as done in init_workers)."""
    config = OmegaConf.create(
        {
            "algorithm": {
                "mopd": {
                    "enabled": True,
                    "teachers": [
                        {"name": "math", "model_path": "/models/math", "resource_pool": "global_pool"},
                        {"name": "code", "model_path": "/models/code", "resource_pool": "code_pool"},
                        {"name": "reasoning", "model_path": "/models/reasoning", "resource_pool": "global_pool"},
                    ],
                }
            }
        }
    )

    teacher_names = []
    for teacher_cfg in config.algorithm.mopd.teachers:
        teacher_names.append(teacher_cfg.name)

    assert teacher_names == ["math", "code", "reasoning"]


def test_teacher_worker_config_uses_fixed_ref_micro_batching():
    """Teacher worker configs should disable dynamic ref batching and use per-teacher micro batches."""
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"path": "/models/student"},
                "ref": {
                    "log_prob_use_dynamic_bsz": True,
                    "log_prob_micro_batch_size_per_gpu": None,
                    "log_prob_max_token_len_per_gpu": 20480,
                },
            }
        }
    )
    teacher_cfg = OmegaConf.create(
        {
            "name": "math",
            "model_path": "/models/math-teacher",
            "resource_pool": "global_pool",
            "log_prob_micro_batch_size": 4,
        }
    )

    teacher_worker_config = trainer._build_teacher_worker_config(teacher_cfg)

    assert teacher_worker_config.model.path == "/models/math-teacher"
    assert teacher_worker_config.ref.log_prob_use_dynamic_bsz is False
    assert teacher_worker_config.ref.log_prob_micro_batch_size_per_gpu == 4


def test_teacher_worker_config_propagates_teacher_tokenizer_path():
    """Teacher worker config should use the teacher tokenizer override when provided."""
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"path": "/models/student", "tokenizer_path": "/tokenizers/student"},
                "ref": {
                    "log_prob_use_dynamic_bsz": True,
                    "log_prob_micro_batch_size_per_gpu": None,
                    "log_prob_max_token_len_per_gpu": 20480,
                },
            }
        }
    )
    teacher_cfg = OmegaConf.create(
        {
            "name": "math",
            "model_path": "/models/math-teacher",
            "tokenizer_path": "/tokenizers/math-teacher",
            "resource_pool": "global_pool",
            "log_prob_micro_batch_size": 4,
        }
    )

    teacher_worker_config = trainer._build_teacher_worker_config(teacher_cfg)

    assert teacher_worker_config.model.path == "/models/math-teacher"
    assert teacher_worker_config.model.tokenizer_path == "/tokenizers/math-teacher"


def test_quantized_teacher_backend_has_dedicated_worker_module():
    """P2 quantized teachers should live in a dedicated worker module, not the legacy ref worker path."""
    spec = util.find_spec("verl.workers.teacher_workers")

    assert spec is not None

    teacher_workers = import_module("verl.workers.teacher_workers")
    assert hasattr(teacher_workers, "HFQuantizedTeacherWorker")


def test_quantized_teacher_sequence_scores_respect_micro_batch_size(monkeypatch):
    monkeypatch.setattr(
        "verl.workers.teacher_workers.logprobs_from_logits",
        lambda logits, labels: torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1),
    )

    worker = HFQuantizedTeacherWorker.__new__(HFQuantizedTeacherWorker)
    worker.config = OmegaConf.create(
        {
            "teacher": {
                "backend": "hf_int8",
                "log_prob_micro_batch_size": 2,
            }
        }
    )
    worker.device = torch.device("cpu")
    worker.model = _RecordingModel()
    worker.tokenizer = _FakeSeqTokenizer()

    batch = DataProto.from_single_dict({})
    batch.non_tensor_batch["raw_prompt"] = [
        [{"role": "user", "content": "solve now"}],
        [{"role": "user", "content": "write code"}],
        [{"role": "user", "content": "plan carefully"}],
    ]
    batch.non_tensor_batch["response_text"] = [
        "answer clearly",
        "optimize later",
        "explain steps",
    ]

    output = worker._compute_seq_scores_impl(batch)

    assert worker.model.batch_sizes == [2, 1]
    assert output.batch["seq_scores"].shape == (3,)


def test_teacher_worker_config_rejects_quantized_backend_on_legacy_ref_path():
    """Quantized teachers must not silently reuse the legacy ref-worker config builder."""
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"path": "/models/student", "tokenizer_path": "/tokenizers/student"},
                "ref": {
                    "log_prob_use_dynamic_bsz": True,
                    "log_prob_micro_batch_size": None,
                    "log_prob_micro_batch_size_per_gpu": None,
                    "log_prob_max_token_len_per_gpu": 20480,
                },
            }
        }
    )
    teacher_cfg = OmegaConf.create(
        {
            "name": "judge",
            "model_path": "/models/judge",
            "tokenizer_path": "/tokenizers/judge",
            "backend": "hf_int8",
            "resource_pool": "global_pool",
            "log_prob_micro_batch_size": 2,
        }
    )

    with pytest.raises(ValueError, match="dedicated quantized teacher worker"):
        trainer._build_teacher_worker_config(teacher_cfg)


def test_base_worker_config_uses_shared_base_model_path():
    """Base worker config should point to the shared ExOPD base model path."""
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"path": "/models/student", "tokenizer_path": "/tokenizers/student"},
                "ref": {
                    "log_prob_use_dynamic_bsz": True,
                    "log_prob_micro_batch_size_per_gpu": None,
                    "log_prob_max_token_len_per_gpu": 20480,
                },
            },
            "algorithm": {
                "mopd": {
                    "enabled": True,
                    "use_base_normalization": True,
                    "base_model_path": "/models/shared-base",
                    "teachers": [
                        {"name": "math", "model_path": "/models/math-teacher"},
                    ],
                }
            },
        }
    )

    base_worker_config = trainer._build_base_worker_config()

    assert base_worker_config.model.path == "/models/shared-base"
    assert base_worker_config.model.tokenizer_path == "/tokenizers/student"
