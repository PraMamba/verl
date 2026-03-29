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

"""Unit tests for MOPD trainer runtime helpers."""

import logging
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import CHECKPOINT_COMPLETE_FILENAME, RayPPOTrainer


class _ListDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class _MockLogProbWG:
    def __init__(self, fill_value: float):
        self.fill_value = fill_value

    def compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        batch_size, response_len = batch.batch["responses"].shape
        return DataProto.from_single_dict(
            {
                "ref_log_prob": torch.full((batch_size, response_len), self.fill_value, dtype=torch.float32),
            }
        )


class _MockSeqScoreWG:
    def __init__(self, seq_scores):
        self.seq_scores = torch.tensor(seq_scores, dtype=torch.float32)
        self.calls = []

    def compute_seq_scores(self, batch: DataProto) -> DataProto:
        self.calls.append(batch)
        batch_size = len(batch.non_tensor_batch["raw_prompt"])
        return DataProto.from_single_dict(
            {
                "seq_scores": self.seq_scores[:batch_size].clone(),
            }
        )


class _ExplodingLogProbWG:
    def compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        raise AssertionError("sequence_reward teachers must not be routed through compute_ref_log_prob")


class _FakeTokenizer:
    def __init__(
        self,
        *,
        vocab_size=32000,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        padding_side="left",
        special_tokens_map=None,
        added_vocab=None,
        vocab=None,
        decode_map=None,
    ):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.padding_side = padding_side
        self.special_tokens_map = special_tokens_map or {
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "bos_token": "<s>",
        }
        self._added_vocab = added_vocab or {}
        self._vocab = vocab or {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
        }
        self._decode_map = decode_map or {}

    def get_added_vocab(self):
        return self._added_vocab

    def get_vocab(self):
        return self._vocab

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        special_ids = {self.pad_token_id, self.eos_token_id, self.bos_token_id}
        tokens = []
        for token_id in ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id in special_ids:
                continue
            tokens.append(self._decode_map.get(token_id, f"tok{token_id}"))
        return " ".join(tokens)

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [self.decode(sequence, skip_special_tokens=skip_special_tokens) for sequence in sequences]


def _make_config():
    return OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "mopd",
                "mopd": {
                    "enabled": True,
                    "lambda_val": 1.0,
                    "orm_weight": 0.0,
                    "is_correction": True,
                    "is_epsilon_low": 0.1,
                    "is_epsilon_high": 10.0,
                    "use_base_normalization": True,
                    "base_model_path": "/models/base",
                    "teachers": [
                        {
                            "name": "math",
                            "model_path": "/models/math",
                            "resource_pool": "global_pool",
                            "log_prob_micro_batch_size": 4,
                        },
                        {
                            "name": "code",
                            "model_path": "/models/code",
                            "resource_pool": "code_pool",
                            "log_prob_micro_batch_size": 8,
                        },
                    ],
                },
            },
            "actor_rollout_ref": {
                "model": {
                    "path": "/models/student",
                    "tokenizer_path": "/tokenizers/student",
                    "trust_remote_code": False,
                },
                "ref": {
                    "log_prob_use_dynamic_bsz": True,
                    "log_prob_micro_batch_size": None,
                    "log_prob_micro_batch_size_per_gpu": None,
                    "log_prob_max_token_len_per_gpu": 20480,
                },
            },
            "trainer": {
                "default_local_dir": "/tmp/mopd-runtime-tests",
                "default_hdfs_dir": None,
                "del_local_ckpt_after_load": False,
            },
        }
    )


def _make_trainer(config=None, train_dataset=None):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = config or _make_config()
    trainer.train_dataset = train_dataset or _ListDataset([])
    trainer.use_legacy_worker_impl = "enable"
    trainer.base_policy_wg = None
    trainer.teacher_wgs = {}
    return trainer


class _RecordingCheckpointWG:
    def __init__(self):
        self.calls = []

    def load_checkpoint(self, local_path, del_local_after_load=False):
        self.calls.append((local_path, del_local_after_load))


class _RecordingDataloader:
    def __init__(self):
        self.loaded_state = None

    def load_state_dict(self, state_dict):
        self.loaded_state = state_dict


class _CheckpointStateDataloader(_RecordingDataloader):
    def __init__(self, state_dict):
        super().__init__()
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict


class _SavingCheckpointWG:
    def __init__(self):
        self.calls = []

    def save_checkpoint(self, local_path, remote_path, global_steps, max_ckpt_to_keep=None):
        self.calls.append((local_path, remote_path, global_steps, max_ckpt_to_keep))
        os.makedirs(local_path, exist_ok=True)
        with open(os.path.join(local_path, "payload.pt"), "w", encoding="utf-8") as f:
            f.write(str(global_steps))


def _write_mock_async_checkpoint_payload(role_dir):
    role_dir.mkdir(parents=True, exist_ok=True)
    dist_ckpt_dir = role_dir / "dist_ckpt"
    huggingface_dir = role_dir / "huggingface"
    dist_ckpt_dir.mkdir()
    huggingface_dir.mkdir()
    (dist_ckpt_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (huggingface_dir / "config.json").write_text("{}", encoding="utf-8")
    (role_dir / "transformer_config.json").write_text("{}", encoding="utf-8")


def test_compute_base_log_prob_uses_base_worker():
    trainer = _make_trainer()
    trainer.base_policy_wg = _MockLogProbWG(fill_value=1.25)

    batch = DataProto.from_single_dict(
        {
            "responses": torch.randint(0, 100, (3, 5)),
        }
    )

    result = trainer._compute_base_log_prob(batch)

    assert "base_log_prob" in result.batch
    torch.testing.assert_close(result.batch["base_log_prob"], torch.full((3, 5), 1.25))


def test_cleanup_teacher_workers_releases_base_policy_group_and_is_idempotent():
    trainer = _make_trainer()
    trainer.teacher_wgs = {"math": object()}
    trainer.base_policy_wg = object()

    trainer.cleanup_teacher_workers()

    assert trainer.teacher_wgs == {}
    assert trainer.base_policy_wg is None


def test_load_checkpoint_auto_uses_complete_checkpoint_when_tracker_missing(tmp_path):
    config = _make_config()
    config.trainer.default_local_dir = str(tmp_path)
    config.trainer.resume_mode = "auto"
    config.trainer.del_local_ckpt_after_load = False
    trainer = _make_trainer(config=config)
    trainer.use_critic = False
    trainer.global_steps = 0
    trainer.actor_rollout_wg = _RecordingCheckpointWG()
    trainer.train_dataloader = _RecordingDataloader()

    complete_dir = tmp_path / "global_step_3"
    actor_dir = complete_dir / "actor"
    actor_dir.mkdir(parents=True)
    (actor_dir / "model.pt").write_text("ok", encoding="utf-8")
    torch.save({"position": 7}, complete_dir / "data.pt")
    (complete_dir / CHECKPOINT_COMPLETE_FILENAME).write_text("3", encoding="utf-8")

    trainer._load_checkpoint()

    assert trainer.global_steps == 3
    assert trainer.actor_rollout_wg.calls == [(os.fspath(actor_dir), False)]
    assert trainer.train_dataloader.loaded_state == {"position": 7}


def test_load_checkpoint_auto_skips_incomplete_tracked_checkpoint(tmp_path):
    config = _make_config()
    config.trainer.default_local_dir = str(tmp_path)
    config.trainer.resume_mode = "auto"
    config.trainer.del_local_ckpt_after_load = False
    trainer = _make_trainer(config=config)
    trainer.use_critic = False
    trainer.global_steps = 0
    trainer.actor_rollout_wg = _RecordingCheckpointWG()
    trainer.train_dataloader = _RecordingDataloader()

    fallback_dir = tmp_path / "global_step_1"
    fallback_actor_dir = fallback_dir / "actor"
    fallback_actor_dir.mkdir(parents=True)
    (fallback_actor_dir / "model.pt").write_text("ok", encoding="utf-8")
    torch.save({"position": 3}, fallback_dir / "data.pt")
    (fallback_dir / CHECKPOINT_COMPLETE_FILENAME).write_text("1", encoding="utf-8")

    incomplete_dir = tmp_path / "global_step_2"
    (incomplete_dir / "actor").mkdir(parents=True)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("2", encoding="utf-8")

    trainer._load_checkpoint()

    assert trainer.global_steps == 1
    assert trainer.actor_rollout_wg.calls == [(os.fspath(fallback_actor_dir), False)]
    assert trainer.train_dataloader.loaded_state == {"position": 3}


def test_load_checkpoint_auto_ignores_async_payload_without_marker_when_tracker_missing(tmp_path):
    config = _make_config()
    config.trainer.default_local_dir = str(tmp_path)
    config.trainer.resume_mode = "auto"
    config.trainer.del_local_ckpt_after_load = False
    config.actor_rollout_ref.actor = OmegaConf.create({"checkpoint": {"async_save": True}})

    trainer = _make_trainer(config=config)
    trainer.use_critic = False
    trainer.global_steps = 0
    trainer.actor_rollout_wg = _RecordingCheckpointWG()
    trainer.train_dataloader = _RecordingDataloader()

    incomplete_dir = tmp_path / "global_step_5"
    actor_dir = incomplete_dir / "actor"
    actor_dir.mkdir(parents=True)
    (actor_dir / "partial.bin").write_text("partial", encoding="utf-8")
    torch.save({"position": 9}, incomplete_dir / "data.pt")

    trainer._load_checkpoint()

    assert trainer.global_steps == 0
    assert trainer.actor_rollout_wg.calls == []
    assert trainer.train_dataloader.loaded_state is None


def test_load_checkpoint_auto_uses_tracked_single_async_checkpoint_without_marker(tmp_path):
    config = _make_config()
    config.trainer.default_local_dir = str(tmp_path)
    config.trainer.resume_mode = "auto"
    config.trainer.del_local_ckpt_after_load = False
    config.actor_rollout_ref.actor = OmegaConf.create({"checkpoint": {"async_save": True}})

    trainer = _make_trainer(config=config)
    trainer.use_critic = False
    trainer.global_steps = 0
    trainer.actor_rollout_wg = _RecordingCheckpointWG()
    trainer.train_dataloader = _RecordingDataloader()

    complete_dir = tmp_path / "global_step_6"
    actor_dir = complete_dir / "actor"
    _write_mock_async_checkpoint_payload(actor_dir)
    torch.save({"position": 12}, complete_dir / "data.pt")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("6", encoding="utf-8")

    trainer._load_checkpoint()

    assert trainer.global_steps == 6
    assert trainer.actor_rollout_wg.calls == [(os.fspath(actor_dir), False)]
    assert trainer.train_dataloader.loaded_state == {"position": 12}


def test_load_checkpoint_auto_rejects_tracked_multi_async_checkpoint_without_marker(tmp_path):
    config = _make_config()
    config.trainer.default_local_dir = str(tmp_path)
    config.trainer.resume_mode = "auto"
    config.trainer.del_local_ckpt_after_load = False
    config.actor_rollout_ref.actor = OmegaConf.create({"checkpoint": {"async_save": True}})
    config.critic = OmegaConf.create({"checkpoint": {"async_save": True}})

    trainer = _make_trainer(config=config)
    trainer.use_critic = True
    trainer.global_steps = 0
    trainer.actor_rollout_wg = _RecordingCheckpointWG()
    trainer.critic_wg = _RecordingCheckpointWG()
    trainer.train_dataloader = _RecordingDataloader()

    checkpoint_dir = tmp_path / "global_step_8"
    actor_dir = checkpoint_dir / "actor"
    critic_dir = checkpoint_dir / "critic"
    _write_mock_async_checkpoint_payload(actor_dir)
    _write_mock_async_checkpoint_payload(critic_dir)
    torch.save({"position": 15}, checkpoint_dir / "data.pt")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("8", encoding="utf-8")

    trainer._load_checkpoint()

    assert trainer.global_steps == 0
    assert trainer.actor_rollout_wg.calls == []
    assert trainer.critic_wg.calls == []
    assert trainer.train_dataloader.loaded_state is None


def test_load_checkpoint_resume_path_rejects_older_partial_single_async_checkpoint_when_tracker_is_newer(tmp_path):
    config = _make_config()
    config.trainer.default_local_dir = str(tmp_path)
    config.trainer.resume_mode = "resume_path"
    config.trainer.resume_from_path = str(tmp_path / "global_step_6")
    config.trainer.del_local_ckpt_after_load = False
    config.actor_rollout_ref.actor = OmegaConf.create({"checkpoint": {"async_save": True}})

    trainer = _make_trainer(config=config)
    trainer.use_critic = False
    trainer.actor_rollout_wg = _RecordingCheckpointWG()
    trainer.train_dataloader = _RecordingDataloader()

    checkpoint_dir = tmp_path / "global_step_6"
    actor_dir = checkpoint_dir / "actor"
    actor_dir.mkdir(parents=True)
    (actor_dir / "partial.bin").write_text("partial", encoding="utf-8")
    torch.save({"position": 21}, checkpoint_dir / "data.pt")

    newer_checkpoint_dir = tmp_path / "global_step_8"
    newer_actor_dir = newer_checkpoint_dir / "actor"
    _write_mock_async_checkpoint_payload(newer_actor_dir)
    torch.save({"position": 34}, newer_checkpoint_dir / "data.pt")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("8", encoding="utf-8")

    with pytest.raises(ValueError, match="Requested resume checkpoint is incomplete"):
        trainer._load_checkpoint()


def test_save_checkpoint_writes_complete_marker_and_tracker(tmp_path):
    config = _make_config()
    config.trainer.default_local_dir = str(tmp_path)
    config.actor_rollout_ref.actor = OmegaConf.create({"checkpoint": {}})
    trainer = _make_trainer(config=config)
    trainer.use_critic = False
    trainer.global_steps = 4
    trainer.actor_rollout_wg = _SavingCheckpointWG()
    trainer.train_dataloader = _CheckpointStateDataloader({"cursor": 11})

    trainer._save_checkpoint()

    global_step_dir = tmp_path / "global_step_4"
    assert (global_step_dir / "actor" / "payload.pt").exists()
    assert (global_step_dir / "data.pt").exists()
    assert (global_step_dir / CHECKPOINT_COMPLETE_FILENAME).read_text(encoding="utf-8") == "4"
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text(encoding="utf-8") == "4"

    trainer.cleanup_teacher_workers()

    assert trainer.teacher_wgs == {}
    assert trainer.base_policy_wg is None


def test_save_checkpoint_skips_complete_marker_and_tracker_when_critic_async(tmp_path):
    config = _make_config()
    config.trainer.default_local_dir = str(tmp_path)
    config.actor_rollout_ref.actor = OmegaConf.create({"checkpoint": {}})
    config.critic = OmegaConf.create({"checkpoint": {"async_save": True}})

    trainer = _make_trainer(config=config)
    trainer.use_critic = True
    trainer.global_steps = 7
    trainer.actor_rollout_wg = _SavingCheckpointWG()
    trainer.critic_wg = _SavingCheckpointWG()
    trainer.train_dataloader = _CheckpointStateDataloader({"cursor": 13})

    trainer._save_checkpoint()

    global_step_dir = tmp_path / "global_step_7"
    assert (global_step_dir / "actor" / "payload.pt").exists()
    assert (global_step_dir / "critic" / "payload.pt").exists()
    assert (global_step_dir / "data.pt").exists()
    assert not (global_step_dir / CHECKPOINT_COMPLETE_FILENAME).exists()
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()


def test_fit_finalizes_resources_when_validation_raises(monkeypatch):
    tracker_instances = []

    class _FakeTracking:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.finished = False
            tracker_instances.append(self)

        def log(self, *args, **kwargs):
            raise AssertionError("log should not be called when validation raises")

        def finish(self):
            self.finished = True

    monkeypatch.setattr("verl.utils.tracking.Tracking", _FakeTracking)

    config = _make_config()
    config.trainer.project_name = "mopd-tests"
    config.trainer.experiment_name = "fit-finalize"
    config.trainer.logger = ["console"]
    config.trainer.val_before_train = True
    config.trainer.val_only = False
    config.trainer.total_epochs = 1

    trainer = _make_trainer(config=config)
    trainer.train_dataloader = [object()]
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda *_args, **_kwargs: None)
    trainer._load_checkpoint = lambda: None

    finalize_calls = []

    def _finalize_fit_resources(*, tracking_logger=None, progress_bar=None):
        finalize_calls.append((tracking_logger, progress_bar))

    trainer._finalize_fit_resources = _finalize_fit_resources
    trainer._validate = lambda: (_ for _ in ()).throw(RuntimeError("validation boom"))

    with pytest.raises(RuntimeError, match="validation boom"):
        trainer.fit()

    assert len(tracker_instances) == 1
    assert finalize_calls == [(tracker_instances[0], None)]


def test_validation_metric_group_key_prefers_teacher_id_when_configured():
    config = _make_config()
    config.trainer.val_metric_group_key = "teacher_id"
    trainer = _make_trainer(config=config)

    batch = DataProto.from_dict(
        tensors={"prompts": torch.ones((2, 2), dtype=torch.long)},
        non_tensors={
            "teacher_id": np.array(["math", "code"], dtype=object),
            "data_source": np.array(["cell_type", "disease_state"], dtype=object),
        },
    )

    groups = trainer._get_validation_metric_groups(batch, batch_size=2)

    assert groups.tolist() == ["math", "code"]


def test_validation_metric_group_key_falls_back_to_data_source():
    trainer = _make_trainer()

    batch = DataProto.from_dict(
        tensors={"prompts": torch.ones((2, 2), dtype=torch.long)},
        non_tensors={
            "data_source": np.array(["cell_type", "disease_state"], dtype=object),
        },
    )

    groups = trainer._get_validation_metric_groups(batch, batch_size=2)

    assert groups.tolist() == ["cell_type", "disease_state"]


def test_build_mopd_lambda_tensor_uses_teacher_overrides():
    trainer = _make_trainer()
    trainer.config.algorithm.mopd.teachers[1].lambda_val = 2.5

    batch = DataProto.from_single_dict(
        {
            "responses": torch.randint(0, 100, (4, 6)),
        }
    )
    batch.non_tensor_batch["teacher_id"] = ["math", "code", "math", "code"]

    lambda_tensor = trainer._build_mopd_lambda_tensor(batch)

    expected = torch.tensor([[1.0], [2.5], [1.0], [2.5]], dtype=torch.float32)
    torch.testing.assert_close(lambda_tensor.cpu(), expected)


def test_run_mopd_preflight_rejects_unknown_teacher_ids():
    config = _make_config()
    config.algorithm.mopd.teachers[0].tokenizer_path = "/tokenizers/student"
    config.algorithm.mopd.teachers[1].tokenizer_path = "/tokenizers/student"
    trainer = _make_trainer(
        config=config,
        train_dataset=_ListDataset(
            [
                {"teacher_id": "math"},
                {"teacher_id": "biology"},
            ]
        ),
    )

    with pytest.raises(ValueError, match="unknown teacher_ids"):
        trainer._run_mopd_preflight_checks()


def test_run_mopd_preflight_rejects_missing_configured_teachers():
    config = _make_config()
    config.algorithm.mopd.teachers[0].tokenizer_path = "/tokenizers/student"
    config.algorithm.mopd.teachers[1].tokenizer_path = "/tokenizers/student"
    trainer = _make_trainer(
        config=config,
        train_dataset=_ListDataset(
            [
                {"teacher_id": "math"},
                {"teacher_id": "math"},
            ]
        ),
    )

    with pytest.raises(ValueError, match="missing configured teachers"):
        trainer._run_mopd_preflight_checks()


def test_run_mopd_preflight_supports_single_teacher_reverse_kl_runtime_without_enabled_flag():
    config = _make_config()
    config.algorithm.adv_estimator = "single_teacher_reverse_kl"
    config.algorithm.mopd.enabled = False
    config.algorithm.mopd.teachers = [config.algorithm.mopd.teachers[0]]
    config.algorithm.mopd.teachers[0].tokenizer_path = "/tokenizers/student"

    trainer = _make_trainer(
        config=config,
        train_dataset=_ListDataset(
            [
                {
                    "teacher_id": "unknown",
                    "raw_prompt": [{"role": "user", "content": "Route this"}],
                }
            ]
        ),
    )

    with pytest.raises(ValueError, match="unknown teacher_ids"):
        trainer._run_mopd_preflight_checks()


def test_run_mopd_preflight_skips_zero_teacher_orm_only_runtime_without_enabled_flag():
    config = _make_config()
    config.algorithm.adv_estimator = "mopd_zero_teacher_orm_only"
    config.algorithm.mopd.enabled = False

    trainer = _make_trainer(
        config=config,
        train_dataset=_ListDataset(
            [
                {
                    "teacher_id": "unknown",
                    "raw_prompt": [{"role": "user", "content": "Route this"}],
                }
            ]
        ),
    )

    trainer._run_mopd_preflight_checks()


def test_validate_tokenizer_compatibility_rejects_mismatched_paths_without_override(monkeypatch):
    trainer = _make_trainer()
    trainer.config.algorithm.mopd.teachers[0].tokenizer_path = "/tokenizers/math"

    fake_tokenizers = {
        "/tokenizers/student": _FakeTokenizer(),
        "/tokenizers/math": _FakeTokenizer(vocab_size=16000),
    }
    monkeypatch.setattr(
        "verl.trainer.ppo.ray_trainer.hf_tokenizer",
        lambda path, **_: fake_tokenizers[path],
    )

    with pytest.raises(ValueError, match="tokenizer compatibility"):
        trainer._validate_mopd_tokenizer_compatibility()


def test_validate_tokenizer_compatibility_accepts_matching_metadata_with_compat_group(monkeypatch):
    trainer = _make_trainer()
    trainer.config.algorithm.mopd.teachers[0].tokenizer_path = "/tokenizers/math"
    trainer.config.algorithm.mopd.teachers[0].tokenizer_compat_group = "shared-qwen"
    trainer.config.algorithm.mopd.teachers[1].tokenizer_path = "/tokenizers/student"

    shared = _FakeTokenizer(added_vocab={"<tool>": 32001})
    fake_tokenizers = {
        "/tokenizers/student": shared,
        "/tokenizers/math": _FakeTokenizer(added_vocab={"<tool>": 32001}),
        "/models/base": _FakeTokenizer(added_vocab={"<tool>": 32001}),
    }
    monkeypatch.setattr(
        "verl.trainer.ppo.ray_trainer.hf_tokenizer",
        lambda path, **_: fake_tokenizers[path],
    )

    trainer._validate_mopd_tokenizer_compatibility()


def test_validate_tokenizer_compatibility_rejects_metadata_mismatch_with_compat_group(monkeypatch):
    trainer = _make_trainer()
    trainer.config.algorithm.mopd.teachers[0].tokenizer_path = "/tokenizers/math"
    trainer.config.algorithm.mopd.teachers[0].tokenizer_compat_group = "shared-qwen"
    trainer.config.algorithm.mopd.teachers[1].tokenizer_path = "/tokenizers/student"

    fake_tokenizers = {
        "/tokenizers/student": _FakeTokenizer(pad_token_id=0),
        "/tokenizers/math": _FakeTokenizer(pad_token_id=99),
    }
    monkeypatch.setattr(
        "verl.trainer.ppo.ray_trainer.hf_tokenizer",
        lambda path, **_: fake_tokenizers[path],
    )

    with pytest.raises(ValueError, match="tokenizer metadata"):
        trainer._validate_mopd_tokenizer_compatibility()


def test_validate_tokenizer_compatibility_rejects_vocab_mismatch_with_compat_group(monkeypatch):
    trainer = _make_trainer()
    trainer.config.algorithm.mopd.teachers[0].tokenizer_path = "/tokenizers/math"
    trainer.config.algorithm.mopd.teachers[0].tokenizer_compat_group = "shared-qwen"
    trainer.config.algorithm.mopd.teachers[1].tokenizer_path = "/tokenizers/student"

    fake_tokenizers = {
        "/tokenizers/student": _FakeTokenizer(vocab={"a": 0, "b": 1, "c": 2}),
        "/tokenizers/math": _FakeTokenizer(vocab={"x": 0, "y": 1, "z": 2}),
    }
    monkeypatch.setattr(
        "verl.trainer.ppo.ray_trainer.hf_tokenizer",
        lambda path, **_: fake_tokenizers[path],
    )

    with pytest.raises(ValueError, match="tokenizer metadata"):
        trainer._validate_mopd_tokenizer_compatibility()


def test_validate_tokenizer_compatibility_wraps_load_failures(monkeypatch):
    trainer = _make_trainer()
    trainer.config.algorithm.mopd.teachers[0].tokenizer_path = "/tokenizers/math"
    trainer.config.algorithm.mopd.teachers[0].tokenizer_compat_group = "shared-qwen"
    trainer.config.algorithm.mopd.teachers[1].tokenizer_path = "/tokenizers/student"

    def _raise(path, **_kwargs):
        raise RuntimeError(f"cannot load {path}")

    monkeypatch.setattr("verl.trainer.ppo.ray_trainer.hf_tokenizer", _raise)

    with pytest.raises(ValueError, match="failed while loading student tokenizer"):
        trainer._validate_mopd_tokenizer_compatibility()


def test_validate_tokenizer_compatibility_rejects_base_model_vocab_mismatch(monkeypatch):
    trainer = _make_trainer()
    trainer.config.algorithm.mopd.teachers[0].tokenizer_path = "/models/student"
    trainer.config.algorithm.mopd.teachers[1].tokenizer_path = "/models/student"
    trainer.config.actor_rollout_ref.model.tokenizer_path = None

    fake_tokenizers = {
        "/models/student": _FakeTokenizer(vocab={"a": 0, "b": 1, "c": 2}),
        "/models/base": _FakeTokenizer(vocab={"x": 0, "y": 1, "z": 2}),
    }
    monkeypatch.setattr(
        "verl.trainer.ppo.ray_trainer.hf_tokenizer",
        lambda path, **_: fake_tokenizers[path],
    )

    with pytest.raises(ValueError, match="base model tokenizer"):
        trainer._validate_mopd_tokenizer_compatibility()


def test_validate_tokenizer_compatibility_checks_base_model_even_with_student_tokenizer_path(monkeypatch):
    trainer = _make_trainer()
    trainer.config.algorithm.mopd.teachers[0].tokenizer_path = "/tokenizers/student"
    trainer.config.algorithm.mopd.teachers[1].tokenizer_path = "/tokenizers/student"

    fake_tokenizers = {
        "/tokenizers/student": _FakeTokenizer(),
        "/models/base": _FakeTokenizer(vocab={"x": 0, "y": 1, "z": 2, "w": 3}),
    }
    monkeypatch.setattr(
        "verl.trainer.ppo.ray_trainer.hf_tokenizer",
        lambda path, **_: fake_tokenizers[path],
    )

    with pytest.raises(ValueError, match="base model tokenizer"):
        trainer._validate_mopd_tokenizer_compatibility()


def test_get_gen_batch_preserves_raw_prompt_and_tensor_payload_for_sequence_teachers():
    trainer = _make_trainer()

    batch = DataProto.from_single_dict(
        {
            "prompts": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
            "responses": torch.tensor([[7, 8, 1], [9, 1, 0]], dtype=torch.long),
            "attention_mask": torch.ones(2, 6, dtype=torch.long),
        }
    )
    batch.non_tensor_batch["uid"] = np.array(["q1", "q2"])
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "code"])
    batch.non_tensor_batch["raw_prompt"] = np.array(
        [
            [{"role": "user", "content": "Solve 2+2"}],
            [{"role": "user", "content": "Write a sorter"}],
        ],
        dtype=object,
    )

    gen_batch = trainer._get_gen_batch(batch)

    assert gen_batch.batch is not None
    assert set(gen_batch.batch.keys()) >= {"prompts", "responses", "attention_mask"}
    assert "raw_prompt" in gen_batch.non_tensor_batch
    assert gen_batch.non_tensor_batch["raw_prompt"][1][0]["content"] == "Write a sorter"


def test_decode_mopd_response_texts_uses_student_tokenizer():
    trainer = _make_trainer()
    trainer.tokenizer = _FakeTokenizer(
        decode_map={
            11: "solve",
            12: "now",
            21: "write",
            22: "code",
        }
    )

    batch = DataProto.from_single_dict(
        {
            "responses": torch.tensor(
                [
                    [11, 12, 1],
                    [21, 22, 0],
                ],
                dtype=torch.long,
            ),
        }
    )

    assert hasattr(trainer, "_decode_mopd_response_texts")

    response_texts = trainer._decode_mopd_response_texts(batch)

    assert list(response_texts) == ["solve now", "write code"]


def test_build_mopd_sequence_teacher_jobs_uses_raw_prompt_and_response_text():
    config = _make_config()
    config.algorithm.mopd.teachers[0].backend = "legacy_ref"
    config.algorithm.mopd.teachers[0].tokenizer_policy = "compatible"
    config.algorithm.mopd.teachers[1].backend = "hf_int8"
    config.algorithm.mopd.teachers[1].tokenizer_policy = "sequence_reward"
    config.algorithm.mopd.teachers[1].seq_reward_weight = 0.5

    trainer = _make_trainer(config=config)
    trainer.teacher_wgs = {"math": object(), "code": object()}
    trainer.tokenizer = _FakeTokenizer(
        decode_map={
            31: "write",
            32: "code",
            41: "solve",
            42: "carefully",
        }
    )

    batch = DataProto.from_single_dict(
        {
            "responses": torch.tensor(
                [
                    [41, 42, 1],
                    [31, 32, 1],
                ],
                dtype=torch.long,
            ),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "code"])
    batch.non_tensor_batch["raw_prompt"] = np.array(
        [
            [{"role": "user", "content": "Solve 2+2"}],
            [{"role": "user", "content": "Write a quicksort"}],
        ],
        dtype=object,
    )

    assert hasattr(trainer, "_build_mopd_sequence_teacher_jobs")

    jobs, device = trainer._build_mopd_sequence_teacher_jobs(batch)

    assert device == torch.device("cpu")
    assert len(jobs) == 1
    assert jobs[0]["teacher_name"] == "code"
    assert jobs[0]["response_texts"] == ["write code"]
    assert jobs[0]["raw_prompts"][0][0]["content"] == "Write a quicksort"


def test_build_mopd_sequence_teacher_jobs_pads_to_teacher_dp_size_without_balancing_small_subset():
    config = _make_config()
    config.algorithm.mopd.teachers[0].backend = "legacy_ref"
    config.algorithm.mopd.teachers[0].tokenizer_policy = "compatible"
    config.algorithm.mopd.teachers[1].backend = "hf_int8"
    config.algorithm.mopd.teachers[1].tokenizer_policy = "sequence_reward"

    trainer = _make_trainer(config=config)
    trainer.teacher_wgs = {"math": object(), "code": object()}
    trainer.tokenizer = _FakeTokenizer(decode_map={31: "write", 32: "code", 41: "solve"})

    captured = {}
    trainer._get_dp_size = lambda *_args, **_kwargs: 2

    def _capture_balance_batch(batch, metrics, logging_prefix="global_seqlen", keep_minibatch=False, dp_size=None):
        captured["dp_size"] = dp_size

    trainer._balance_batch = _capture_balance_batch

    batch = DataProto.from_single_dict(
        {
            "responses": torch.tensor(
                [
                    [41, 1, 0],
                    [31, 32, 1],
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "code"])
    batch.non_tensor_batch["raw_prompt"] = np.array(
        [
            [{"role": "user", "content": "Solve 2+2"}],
            [{"role": "user", "content": "Write a quicksort"}],
        ],
        dtype=object,
    )

    jobs, _device = trainer._build_mopd_sequence_teacher_jobs(batch)

    assert captured == {}
    assert len(jobs) == 1
    assert jobs[0]["pad_size"] == 1


def test_build_mopd_sequence_teacher_jobs_balances_when_subset_is_divisible_by_teacher_dp_size():
    config = _make_config()
    config.algorithm.mopd.teachers[0].backend = "legacy_ref"
    config.algorithm.mopd.teachers[0].tokenizer_policy = "compatible"
    config.algorithm.mopd.teachers[1].backend = "hf_int8"
    config.algorithm.mopd.teachers[1].tokenizer_policy = "sequence_reward"

    trainer = _make_trainer(config=config)
    trainer.teacher_wgs = {"math": object(), "code": object()}
    trainer.tokenizer = _FakeTokenizer(decode_map={31: "write", 32: "code", 33: "tests"})

    captured = {}
    trainer._get_dp_size = lambda *_args, **_kwargs: 2

    def _capture_balance_batch(batch, metrics, logging_prefix="global_seqlen", keep_minibatch=False, dp_size=None):
        captured["dp_size"] = dp_size

    trainer._balance_batch = _capture_balance_batch

    batch = DataProto.from_single_dict(
        {
            "responses": torch.tensor(
                [
                    [31, 32, 1],
                    [31, 33, 1],
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["code", "code"])
    batch.non_tensor_batch["raw_prompt"] = np.array(
        [
            [{"role": "user", "content": "Write a quicksort"}],
            [{"role": "user", "content": "Write tests"}],
        ],
        dtype=object,
    )

    jobs, _device = trainer._build_mopd_sequence_teacher_jobs(batch)

    assert captured["dp_size"] == 2
    assert len(jobs) == 1
    assert jobs[0]["pad_size"] == 0


def test_compute_teacher_sequence_rewards_builds_reward_tensor_and_teacher_token_mask():
    config = _make_config()
    config.algorithm.mopd.teachers[0].backend = "legacy_ref"
    config.algorithm.mopd.teachers[0].tokenizer_policy = "compatible"
    config.algorithm.mopd.teachers[1].backend = "hf_4bit"
    config.algorithm.mopd.teachers[1].tokenizer_policy = "sequence_reward"
    config.algorithm.mopd.teachers[1].seq_reward_weight = 0.5

    trainer = _make_trainer(config=config)
    trainer.teacher_wgs = {"math": object(), "code": _MockSeqScoreWG([0.75])}
    trainer.tokenizer = _FakeTokenizer(decode_map={31: "write", 32: "code", 41: "solve"})

    batch = DataProto.from_single_dict(
        {
            "responses": torch.tensor(
                [
                    [41, 1, 0],
                    [31, 32, 1],
                ],
                dtype=torch.long,
            ),
            "response_mask": torch.tensor(
                [
                    [1, 0, 0],
                    [1, 1, 1],
                ],
                dtype=torch.float32,
            ),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "code"])
    batch.non_tensor_batch["raw_prompt"] = np.array(
        [
            [{"role": "user", "content": "Solve 2+2"}],
            [{"role": "user", "content": "Write a quicksort"}],
        ],
        dtype=object,
    )

    assert hasattr(trainer, "_compute_teacher_sequence_rewards")

    seq_reward_batch = trainer._compute_teacher_sequence_rewards(batch)

    assert set(seq_reward_batch.batch.keys()) >= {"teacher_seq_reward", "teacher_token_mask"}
    torch.testing.assert_close(
        seq_reward_batch.batch["teacher_seq_reward"],
        torch.tensor([0.0, 0.75], dtype=torch.float32),
    )
    torch.testing.assert_close(
        seq_reward_batch.batch["teacher_token_mask"],
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_compute_teacher_log_probs_skips_sequence_reward_only_teachers():
    config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "mopd",
                "mopd": {
                    "enabled": True,
                    "teachers": [
                        {
                            "name": "judge",
                            "model_path": "/models/judge",
                            "resource_pool": "global_pool",
                            "backend": "hf_int8",
                            "tokenizer_policy": "sequence_reward",
                        }
                    ],
                },
            },
            "trainer": {
                "default_local_dir": "/tmp/mopd-runtime-tests",
            },
        }
    )

    trainer = _make_trainer(config=config)
    trainer.teacher_wgs = {"judge": _ExplodingLogProbWG()}

    batch = DataProto.from_single_dict(
        {
            "responses": torch.tensor([[31, 32, 1]], dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["judge"])

    teacher_log_prob = trainer._compute_teacher_log_probs(batch)

    torch.testing.assert_close(teacher_log_prob, torch.zeros(1, 3, dtype=torch.float32))


def test_run_mopd_preflight_allows_sequence_reward_teacher_with_distinct_tokenizer_path():
    config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "mopd",
                "mopd": {
                    "enabled": True,
                    "teachers": [
                        {
                            "name": "judge",
                            "model_path": "/models/judge",
                            "tokenizer_path": "/tokenizers/judge",
                            "tokenizer_policy": "sequence_reward",
                            "backend": "hf_4bit",
                            "resource_pool": "global_pool",
                            "log_prob_micro_batch_size": 2,
                        }
                    ],
                },
            },
            "actor_rollout_ref": {
                "model": {
                    "path": "/models/student",
                    "tokenizer_path": "/tokenizers/student",
                    "trust_remote_code": False,
                },
                "ref": {
                    "log_prob_use_dynamic_bsz": True,
                    "log_prob_micro_batch_size": None,
                    "log_prob_micro_batch_size_per_gpu": None,
                    "log_prob_max_token_len_per_gpu": 20480,
                },
            },
            "trainer": {
                "default_local_dir": "/tmp/mopd-runtime-tests",
            },
        }
    )
    trainer = _make_trainer(
        config=config,
        train_dataset=_ListDataset(
            [
                {
                    "teacher_id": "judge",
                    "raw_prompt": [{"role": "user", "content": "Summarize the result"}],
                }
            ]
        ),
    )

    trainer._run_mopd_preflight_checks()


def test_record_mopd_teacher_metrics_adds_teacher_breakdown():
    trainer = _make_trainer()
    trainer.config.algorithm.mopd.is_correction = True

    batch = DataProto.from_single_dict(
        {
            "responses": torch.randint(0, 10, (3, 2)),
            "response_mask": torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.float32),
            "teacher_log_prob": torch.tensor([[2.0, 2.0], [1.0, 0.0], [4.0, 4.0]], dtype=torch.float32),
            "old_log_probs": torch.tensor([[1.0, 1.0], [0.0, 0.0], [3.0, 1.0]], dtype=torch.float32),
            "rollout_log_probs": torch.tensor([[1.0, 1.0], [-5.0, 0.0], [3.0, 1.0]], dtype=torch.float32),
            "advantages": torch.tensor([[0.5, 1.5], [4.0, 0.0], [2.0, 4.0]], dtype=torch.float32),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "code", "math"])

    metrics = {}

    trainer._record_mopd_teacher_metrics(batch, metrics)

    assert metrics["mopd/math/sample_fraction"] == pytest.approx(2 / 3)
    assert metrics["mopd/code/sample_fraction"] == pytest.approx(1 / 3)
    assert metrics["mopd/math/adv_mean"] == pytest.approx(2.0)
    assert metrics["mopd/code/adv_mean"] == pytest.approx(4.0)
    assert metrics["mopd/math/adv_std"] == pytest.approx((1.625) ** 0.5)
    assert metrics["mopd/code/adv_std"] == pytest.approx(0.0)
    assert metrics["mopd/math/reverse_kl_mean"] == pytest.approx(1.5)
    assert metrics["mopd/code/reverse_kl_mean"] == pytest.approx(1.0)
    assert metrics["mopd/math/is_valid_fraction"] == pytest.approx(1.0)
    assert metrics["mopd/code/is_valid_fraction"] == pytest.approx(0.0)


def test_resolve_teacher_log_prob_output_does_not_materialize_tensordict():
    output = TensorDict({"ref_log_prob": torch.ones(2, 3)}, batch_size=[2])

    resolved = RayPPOTrainer._resolve_teacher_log_prob_output(output)

    assert resolved is output


def test_validate_loaded_mopd_manifest_rejects_semantic_drift():
    trainer = _make_trainer()
    manifest = trainer._build_mopd_manifest()
    manifest["semantic"]["base_model_path"] = "/models/other-base"

    with pytest.raises(ValueError, match="semantic drift"):
        trainer._validate_loaded_mopd_manifest(manifest)


def test_validate_loaded_mopd_manifest_warns_on_deployment_drift(caplog):
    trainer = _make_trainer()
    manifest = trainer._build_mopd_manifest()
    manifest["deployment"]["teachers"][0]["resource_pool"] = "other_pool"

    with caplog.at_level(logging.WARNING):
        trainer._validate_loaded_mopd_manifest(manifest)

    assert "deployment-only drift" in caplog.text


def test_build_mopd_manifest_records_teacher_backend_and_tokenizer_policy():
    config = _make_config()
    config.algorithm.mopd.teachers[0].backend = "legacy_ref"
    config.algorithm.mopd.teachers[0].tokenizer_policy = "compatible"
    config.algorithm.mopd.teachers[1].backend = "hf_int8"
    config.algorithm.mopd.teachers[1].tokenizer_policy = "sequence_reward"
    config.algorithm.mopd.teachers[1].seq_reward_weight = 0.5

    trainer = _make_trainer(config=config)

    manifest = trainer._build_mopd_manifest()
    code_teacher = next(teacher for teacher in manifest["semantic"]["teachers"] if teacher["name"] == "code")

    assert code_teacher.get("backend") == "hf_int8"
    assert code_teacher.get("tokenizer_policy") == "sequence_reward"
    assert code_teacher.get("seq_reward_weight") == pytest.approx(0.5)
