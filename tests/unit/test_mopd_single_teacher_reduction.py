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

"""Unit tests for the single-teacher MOPD reduction harness."""

from importlib import util
from pathlib import Path
import re
import sys

import pandas as pd


def _load_reduction_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "recipe" / "mopd" / "run_single_teacher_reduction.py"
    spec = util.spec_from_file_location("run_single_teacher_reduction", module_path)
    module = util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_reduction_commands_share_core_training_settings():
    module = _load_reduction_module()

    config = module.ReductionConfig(
        student_model_path="/models/student",
        teacher_model_path="/models/teacher",
        train_file="/data/train.parquet",
        val_file="/data/val.parquet",
        output_root="/tmp/mopd-reduction",
        project_name="RVQ-Alpha_MOPD",
        seed=43,
        rollout_n=2,
        teacher_log_prob_micro_batch_size=4,
    )

    commands = module.build_reduction_commands(config)
    mopd_command = commands["mopd"]
    baseline_command = commands["single_teacher_reverse_kl"]

    assert mopd_command[:3] == [module.sys.executable, "-m", "verl.trainer.main_ppo"]
    assert baseline_command[:3] == [module.sys.executable, "-m", "verl.trainer.main_ppo"]
    assert "data.train_files=/data/train.parquet" in mopd_command
    assert "data.train_files=/data/train.parquet" in baseline_command
    assert "data.val_files=/data/val.parquet" in mopd_command
    assert "data.val_files=/data/val.parquet" in baseline_command
    assert "actor_rollout_ref.rollout.n=2" in mopd_command
    assert "actor_rollout_ref.rollout.n=2" in baseline_command
    assert "algorithm.mopd.enabled=True" in mopd_command
    assert "algorithm.mopd.enabled=True" in baseline_command
    assert "algorithm.mopd.orm_weight=0.0" in mopd_command
    assert "algorithm.mopd.orm_weight=0.0" in baseline_command
    assert "algorithm.mopd.is_correction=False" in mopd_command
    assert "algorithm.mopd.is_correction=False" in baseline_command
    assert "actor_rollout_ref.actor.ppo_epochs=1" in mopd_command
    assert "actor_rollout_ref.actor.ppo_epochs=1" in baseline_command
    assert "trainer.save_freq=-1" in mopd_command
    assert "trainer.save_freq=-1" in baseline_command
    assert "trainer.test_freq=-1" in mopd_command
    assert "trainer.test_freq=-1" in baseline_command
    assert "data.seed=43" in mopd_command
    assert "data.seed=43" in baseline_command
    assert "algorithm.adv_estimator=mopd" in mopd_command
    assert "algorithm.adv_estimator=single_teacher_reverse_kl" in baseline_command

    teacher_override = next(item for item in mopd_command if item.startswith("algorithm.mopd.teachers=["))
    assert "name: single_teacher" in teacher_override
    assert "model_path: '/models/teacher'" in teacher_override
    assert "tokenizer_policy: compatible" in teacher_override
    assert "log_prob_micro_batch_size: 4" in teacher_override


def test_extract_step_metrics_parses_console_metric_lines():
    module = _load_reduction_module()

    line = (
        "step:3 - training/global_step:3 - mopd/single_teacher/reverse_kl_mean:-0.81 "
        "- critic/advantages/mean:-0.76 - val-core/test/reward/mean@1:0.42"
    )

    step, metrics = module.extract_step_metrics(line)

    assert step == 3
    assert metrics["training/global_step"] == 3.0
    assert metrics["mopd/single_teacher/reverse_kl_mean"] == -0.81
    assert metrics["critic/advantages/mean"] == -0.76
    assert metrics["val-core/test/reward/mean@1"] == 0.42


def test_extract_step_metrics_parses_prefixed_console_metric_lines():
    module = _load_reduction_module()

    line = (
        "(TaskRunner pid=2031990) step:12 - training/global_step:12 "
        "- mopd/single_teacher/reverse_kl_mean:-2.44 - critic/advantages/mean:-2.44"
    )

    step, metrics = module.extract_step_metrics(line)

    assert step == 12
    assert metrics["step"] == 12.0
    assert metrics["training/global_step"] == 12.0
    assert metrics["mopd/single_teacher/reverse_kl_mean"] == -2.44
    assert metrics["critic/advantages/mean"] == -2.44


def test_prepare_single_teacher_datasets_rewrites_teacher_id_column(tmp_path):
    module = _load_reduction_module()

    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    pd.DataFrame(
        {
            "prompt": ["a", "b"],
            "teacher_id": ["cell_type_teacher", "disease_state_teacher"],
            "value": [1, 2],
        }
    ).to_parquet(train_path)
    pd.DataFrame(
        {
            "prompt": ["c"],
            "teacher_id": ["cell_type_teacher"],
            "value": [3],
        }
    ).to_parquet(val_path)

    output_dir = tmp_path / "prepared"
    prepared_train, prepared_val = module.prepare_single_teacher_datasets(
        train_file=str(train_path),
        val_file=str(val_path),
        output_dir=output_dir,
        teacher_id="single_teacher",
    )

    prepared_train_df = pd.read_parquet(prepared_train)
    prepared_val_df = pd.read_parquet(prepared_val)

    assert set(prepared_train_df["teacher_id"]) == {"single_teacher"}
    assert set(prepared_val_df["teacher_id"]) == {"single_teacher"}
    assert prepared_train_df["value"].tolist() == [1, 2]
    assert prepared_val_df["value"].tolist() == [3]


def test_default_student_model_matches_production_explicit_token_recipe():
    module = _load_reduction_module()
    repo_root = Path(__file__).resolve().parents[2]
    train_script = (repo_root / "recipe" / "mopd" / "run_mopd_qwen3_4b.sh").read_text(encoding="utf-8")
    match = re.search(r'STUDENT_MODEL_PATH=\$\{STUDENT_MODEL_PATH:-"([^"]+)"\}', train_script)

    assert match is not None
    assert match.group(1) == module.DEFAULT_STUDENT_MODEL
    assert "ExplicitTokens" in module.DEFAULT_STUDENT_MODEL


def test_main_dry_run_does_not_prepare_datasets(monkeypatch, tmp_path):
    module = _load_reduction_module()
    output_root = tmp_path / "dry-run-output"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_single_teacher_reduction.py",
            "--dry-run",
            "--train-file",
            str(tmp_path / "train.parquet"),
            "--val-file",
            str(tmp_path / "val.parquet"),
            "--output-root",
            str(output_root),
        ],
    )

    def fail_prepare(**_kwargs):
        raise AssertionError("dry-run should not prepare datasets")

    monkeypatch.setattr(module, "prepare_single_teacher_datasets", fail_prepare)

    exit_code = module.main()

    assert exit_code == 0
    assert not (output_root / "prepared_data").exists()
