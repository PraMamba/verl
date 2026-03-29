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

"""Unit tests for the long-run MOPD baseline harness."""

from importlib import util
from pathlib import Path
import sys


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "recipe" / "mopd" / "run_mopd_longrun_baselines.py"
    spec = util.spec_from_file_location("run_mopd_longrun_baselines", module_path)
    module = util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_longrun_baseline_commands_cover_expected_matrix():
    module = _load_module()

    config = module.LongRunBaselineConfig(
        student_model_path="/models/student",
        cell_type_teacher_path="/models/cell",
        disease_state_teacher_path="/models/disease",
        merged_train_file="/data/mopd_train.parquet",
        merged_val_file="/data/mopd_val.parquet",
        cell_type_train_file="/data/cell_train.parquet",
        cell_type_val_file="/data/cell_val.parquet",
        disease_state_train_file="/data/disease_train.parquet",
        disease_state_val_file="/data/disease_val.parquet",
        output_root="/tmp/mopd-longrun",
        project_name="RVQ-Alpha_MOPD",
    )

    commands = module.build_baseline_commands(config)

    assert set(commands) == {
        "student_only_grpo",
        "cell_type_single_teacher",
        "disease_state_single_teacher",
        "dual_teacher_mopd",
    }

    student_only = commands["student_only_grpo"]
    assert student_only[:3] == [module.sys.executable, "-m", "verl.trainer.main_ppo"]
    assert "algorithm.adv_estimator=grpo" in student_only
    assert "algorithm.mopd.enabled=False" in student_only
    assert "custom_reward_function.path=recipe/mopd/real_eval_reward.py" in student_only
    assert "+trainer.val_metric_group_key=data_source" in student_only

    cell_teacher = commands["cell_type_single_teacher"]
    assert "algorithm.adv_estimator=mopd" in cell_teacher
    assert "algorithm.mopd.enabled=True" in cell_teacher
    assert "data.train_files=/data/cell_train.parquet" in cell_teacher
    assert "data.val_files=/data/cell_val.parquet" in cell_teacher
    assert "+data.teacher_id_field=data_source" in cell_teacher
    assert "+trainer.val_metric_group_key=data_source" in cell_teacher
    teacher_override = next(item for item in cell_teacher if item.startswith("algorithm.mopd.teachers=["))
    assert "name: population_cell_type_homogeneous_train" in teacher_override
    assert "model_path: '/models/cell'" in teacher_override

    dual_teacher = commands["dual_teacher_mopd"]
    assert dual_teacher[0] == "bash"
    assert dual_teacher[1].endswith("recipe/mopd/run_mopd_qwen3_4b.sh")
    assert "RUN_ID=dual_teacher_mopd" in dual_teacher
    assert "RESUME_MODE=disable" in dual_teacher


def test_main_dry_run_does_not_execute_commands(monkeypatch, capsys, tmp_path):
    module = _load_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mopd_longrun_baselines.py",
            "--dry-run",
            "--output-root",
            str(tmp_path / "outputs"),
        ],
    )

    def fail_run(*_args, **_kwargs):
        raise AssertionError("dry-run should not execute baseline commands")

    monkeypatch.setattr(module, "run_command", fail_run)

    exit_code = module.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "[student_only_grpo]" in captured.out
    assert "[cell_type_single_teacher]" in captured.out
    assert "[disease_state_single_teacher]" in captured.out
    assert "[dual_teacher_mopd]" in captured.out


def test_main_applies_smoke_cli_overrides_to_all_runs(monkeypatch, tmp_path):
    module = _load_module()
    captured_calls = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mopd_longrun_baselines.py",
            "--output-root",
            str(tmp_path / "outputs"),
            "--n-gpus-per-node",
            "8",
            "--train-batch-size",
            "8",
            "--max-response-length",
            "128",
            "--rollout-n",
            "2",
            "--teacher-log-prob-micro-batch-size",
            "2",
            "--total-epochs",
            "1",
            "--test-freq",
            "-1",
            "--save-freq",
            "-1",
        ],
    )

    def fake_run(label, command, env_overrides=None):
        captured_calls.append((label, command, env_overrides))
        return 0

    monkeypatch.setattr(module, "run_command", fake_run)

    exit_code = module.main()

    assert exit_code == 0
    assert [label for label, _, _ in captured_calls] == [
        "student_only_grpo",
        "cell_type_single_teacher",
        "disease_state_single_teacher",
        "dual_teacher_mopd",
    ]

    student_command = captured_calls[0][1]
    assert "data.train_batch_size=8" in student_command
    assert "data.max_response_length=128" in student_command
    assert "actor_rollout_ref.rollout.n=2" in student_command
    assert "actor_rollout_ref.actor.fsdp_config.fsdp_size=8" in student_command
    assert "trainer.n_gpus_per_node=8" in student_command
    assert "trainer.total_epochs=1" in student_command
    assert "trainer.test_freq=-1" in student_command
    assert "trainer.save_freq=-1" in student_command

    dual_env = captured_calls[-1][2]
    assert dual_env is not None
    assert dual_env["NGPUS_PER_NODE"] == "8"
    assert dual_env["FSDP_SIZE"] == "8"
    assert dual_env["TRAIN_PROMPT_BSZ"] == "8"
    assert dual_env["MAX_RESPONSE_LENGTH"] == "128"
    assert dual_env["N_RESP_PER_PROMPT"] == "2"
    assert dual_env["TEACHER_LOG_PROB_MICRO_BATCH_SIZE"] == "2"
    assert dual_env["TRAINER_TOTAL_EPOCHS"] == "1"
    assert dual_env["TRAINER_TEST_FREQ"] == "-1"
    assert dual_env["TRAINER_SAVE_FREQ"] == "-1"


def test_build_longrun_baseline_commands_propagate_smoke_overrides():
    module = _load_module()

    config = module.LongRunBaselineConfig(
        student_model_path="/models/student",
        cell_type_teacher_path="/models/cell",
        disease_state_teacher_path="/models/disease",
        merged_train_file="/data/mopd_train.parquet",
        merged_val_file="/data/mopd_val.parquet",
        cell_type_train_file="/data/cell_train.parquet",
        cell_type_val_file="/data/cell_val.parquet",
        disease_state_train_file="/data/disease_train.parquet",
        disease_state_val_file="/data/disease_val.parquet",
        output_root="/tmp/mopd-longrun",
        project_name="RVQ-Alpha_MOPD",
        train_batch_size=8,
        max_prompt_length=2048,
        max_response_length=256,
        rollout_n=2,
        ppo_mini_batch_size=4,
        teacher_log_prob_micro_batch_size=2,
        rollout_gpu_memory_utilization=0.35,
        n_gpus_per_node=8,
        total_epochs=1,
        test_freq=1,
        save_freq=-1,
    )

    commands = module.build_baseline_commands(config)

    student_only = commands["student_only_grpo"]
    assert "data.train_batch_size=8" in student_only
    assert "data.max_prompt_length=2048" in student_only
    assert "data.max_response_length=256" in student_only
    assert "actor_rollout_ref.rollout.n=2" in student_only
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=4" in student_only
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.35" in student_only
    assert "trainer.n_gpus_per_node=8" in student_only
    assert "trainer.total_epochs=1" in student_only
    assert "trainer.test_freq=1" in student_only
    assert "trainer.save_freq=-1" in student_only
    assert "hydra.run.dir=/tmp/mopd-longrun/student_only_grpo/hydra" in student_only
    assert "hydra.output_subdir=.hydra" in student_only

    cell_teacher = commands["cell_type_single_teacher"]
    assert "+data.teacher_id_field=data_source" in cell_teacher
    assert "+trainer.val_metric_group_key=data_source" in cell_teacher
    teacher_override = next(item for item in cell_teacher if item.startswith("algorithm.mopd.teachers=["))
    assert "log_prob_micro_batch_size: 2" in teacher_override
    assert "name: population_cell_type_homogeneous_train" in teacher_override

    dual_teacher = commands["dual_teacher_mopd"]
    assert "NGPUS_PER_NODE=8" in dual_teacher
    assert "TRAIN_PROMPT_BSZ=8" in dual_teacher
    assert "MAX_PROMPT_LENGTH=2048" in dual_teacher
    assert "MAX_RESPONSE_LENGTH=256" in dual_teacher
    assert "N_RESP_PER_PROMPT=2" in dual_teacher
    assert "TRAIN_PROMPT_MINI_BSZ=4" in dual_teacher
    assert "TEACHER_LOG_PROB_MICRO_BATCH_SIZE=2" in dual_teacher
    assert "ROLLOUT_GPU_MEMORY_UTILIZATION=0.35" in dual_teacher
    assert "FSDP_SIZE=8" in dual_teacher
    assert "TOTAL_EPOCHS=1" in dual_teacher
    assert "TEST_FREQ=1" in dual_teacher
    assert "SAVE_FREQ=-1" in dual_teacher


def test_build_longrun_baseline_commands_bind_hydra_outputs_to_baseline_dirs():
    module = _load_module()

    config = module.LongRunBaselineConfig(
        student_model_path="/models/student",
        cell_type_teacher_path="/models/cell",
        disease_state_teacher_path="/models/disease",
        merged_train_file="/data/mopd_train.parquet",
        merged_val_file="/data/mopd_val.parquet",
        cell_type_train_file="/data/cell_train.parquet",
        cell_type_val_file="/data/cell_val.parquet",
        disease_state_train_file="/data/disease_train.parquet",
        disease_state_val_file="/data/disease_val.parquet",
        output_root="/tmp/mopd-longrun",
        project_name="RVQ-Alpha_MOPD",
    )

    commands = module.build_baseline_commands(config)

    for label in ("student_only_grpo", "cell_type_single_teacher", "disease_state_single_teacher"):
        command = commands[label]
        assert f"hydra.run.dir=/tmp/mopd-longrun/{label}/hydra" in command
        assert "hydra.output_subdir=.hydra" in command


def test_main_dry_run_can_select_subset_runs(monkeypatch, capsys, tmp_path):
    module = _load_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mopd_longrun_baselines.py",
            "--dry-run",
            "--output-root",
            str(tmp_path / "outputs"),
            "--run",
            "student_only_grpo",
            "--run",
            "dual_teacher_mopd",
        ],
    )

    exit_code = module.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "[student_only_grpo]" in captured.out
    assert "[dual_teacher_mopd]" in captured.out
    assert "[cell_type_single_teacher]" not in captured.out
    assert "[disease_state_single_teacher]" not in captured.out
