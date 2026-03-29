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

"""Unit tests for the zero-teacher ORM-only MOPD reduction harness."""

from importlib import util
from pathlib import Path
import sys


def _load_reduction_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "recipe" / "mopd" / "run_zero_teacher_orm_reduction.py"
    spec = util.spec_from_file_location("run_zero_teacher_orm_reduction", module_path)
    module = util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_zero_teacher_reduction_commands_share_core_training_settings():
    module = _load_reduction_module()

    config = module.ReductionConfig(
        student_model_path="/models/student",
        train_file="/data/train.parquet",
        val_file="/data/val.parquet",
        output_root="/tmp/mopd-zero-teacher-reduction",
        project_name="RVQ-Alpha_MOPD",
        seed=7,
        rollout_n=2,
    )

    commands = module.build_reduction_commands(config)
    reduced_command = commands["mopd_zero_teacher_orm_only"]
    baseline_command = commands["grpo"]

    assert reduced_command[:3] == [module.sys.executable, "-m", "verl.trainer.main_ppo"]
    assert baseline_command[:3] == [module.sys.executable, "-m", "verl.trainer.main_ppo"]
    assert "data.train_files=/data/train.parquet" in reduced_command
    assert "data.train_files=/data/train.parquet" in baseline_command
    assert "data.val_files=/data/val.parquet" in reduced_command
    assert "data.val_files=/data/val.parquet" in baseline_command
    assert "actor_rollout_ref.rollout.n=2" in reduced_command
    assert "actor_rollout_ref.rollout.n=2" in baseline_command
    assert "trainer.save_freq=-1" in reduced_command
    assert "trainer.save_freq=-1" in baseline_command
    assert "trainer.test_freq=-1" in reduced_command
    assert "trainer.test_freq=-1" in baseline_command
    assert "data.seed=7" in reduced_command
    assert "data.seed=7" in baseline_command
    assert "algorithm.adv_estimator=mopd_zero_teacher_orm_only" in reduced_command
    assert "algorithm.adv_estimator=grpo" in baseline_command
    assert "algorithm.mopd.enabled=False" in reduced_command
    assert "algorithm.mopd.enabled=False" in baseline_command


def test_extract_step_metrics_parses_console_metric_lines():
    module = _load_reduction_module()

    line = "step:4 - training/global_step:4 - critic/advantages/mean:0.75 - critic/score/mean:0.20"

    step, metrics = module.extract_step_metrics(line)

    assert step == 4
    assert metrics["training/global_step"] == 4.0
    assert metrics["critic/advantages/mean"] == 0.75
    assert metrics["critic/score/mean"] == 0.20
