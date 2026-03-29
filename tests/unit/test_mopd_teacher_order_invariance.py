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

"""Unit tests for the MOPD teacher-order invariance experiment harness."""

from importlib import util
from pathlib import Path
import sys


def _load_order_invariance_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "recipe" / "mopd" / "run_teacher_order_invariance.py"
    spec = util.spec_from_file_location("run_teacher_order_invariance", module_path)
    module = util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_command(command: list[str]) -> list[str]:
    normalized = []
    for item in command:
        if item.startswith("algorithm.mopd.teachers=["):
            normalized.append("<teacher-order-override>")
        elif item.startswith("trainer.experiment_name="):
            normalized.append("<experiment-name>")
        elif item.startswith("trainer.default_local_dir="):
            normalized.append("<output-dir>")
        else:
            normalized.append(item)
    return normalized


def test_build_order_invariance_commands_share_core_settings_except_teacher_order():
    module = _load_order_invariance_module()

    config = module.OrderInvarianceConfig(
        student_model_path="/models/student",
        cell_type_teacher_model_path="/models/cell",
        disease_state_teacher_model_path="/models/disease",
        train_file="/data/train.parquet",
        val_file="/data/val.parquet",
        output_root="/tmp/mopd-order-invariance",
        project_name="RVQ-Alpha_MOPD",
        seed=17,
        train_batch_size=8,
        rollout_n=1,
        teacher_log_prob_micro_batch_size=4,
    )

    commands = module.build_order_invariance_commands(config)
    declared = commands["declared_order"]
    reversed_declared = commands["reversed_declared_order"]

    assert declared[:3] == [module.sys.executable, "-m", "verl.trainer.main_ppo"]
    assert reversed_declared[:3] == [module.sys.executable, "-m", "verl.trainer.main_ppo"]
    assert _normalize_command(declared) == _normalize_command(reversed_declared)
    assert "data.shuffle=False" in declared
    assert "data.shuffle=False" in reversed_declared

    declared_teacher_override = next(item for item in declared if item.startswith("algorithm.mopd.teachers=["))
    reversed_teacher_override = next(item for item in reversed_declared if item.startswith("algorithm.mopd.teachers=["))

    assert "name: cell_type_teacher" in declared_teacher_override
    assert "name: disease_state_teacher" in declared_teacher_override
    assert "model_path: '/models/cell'" in declared_teacher_override
    assert "model_path: '/models/disease'" in declared_teacher_override
    assert "log_prob_micro_batch_size: 4" in declared_teacher_override
    assert declared_teacher_override.index("name: cell_type_teacher") < declared_teacher_override.index(
        "name: disease_state_teacher"
    )
    assert reversed_teacher_override.index("name: disease_state_teacher") < reversed_teacher_override.index(
        "name: cell_type_teacher"
    )


def test_extract_step_metrics_parses_teacher_specific_metrics():
    module = _load_order_invariance_module()

    line = (
        "step:4 - training/global_step:4 - critic/advantages/mean:-0.88 "
        "- mopd/is_ratio_mean:1.00001 - mopd/is_valid_fraction:1.0 "
        "- mopd/cell_type_teacher/reverse_kl_mean:-0.83 "
        "- mopd/disease_state_teacher/reverse_kl_mean:-0.94"
    )

    step, metrics = module.extract_step_metrics(line)

    assert step == 4
    assert metrics["training/global_step"] == 4.0
    assert metrics["critic/advantages/mean"] == -0.88
    assert metrics["mopd/is_ratio_mean"] == 1.00001
    assert metrics["mopd/is_valid_fraction"] == 1.0
    assert metrics["mopd/cell_type_teacher/reverse_kl_mean"] == -0.83
    assert metrics["mopd/disease_state_teacher/reverse_kl_mean"] == -0.94


def test_summarize_log_collects_final_teacher_metrics(tmp_path):
    module = _load_order_invariance_module()
    log_path = tmp_path / "console.log"
    log_path.write_text(
        "\n".join(
            [
                "step:1 - training/global_step:1 - critic/advantages/mean:-0.95 "
                "- mopd/cell_type_teacher/reverse_kl_mean:-0.91",
                "step:2 - training/global_step:2 - critic/advantages/mean:-0.89 "
                "- mopd/cell_type_teacher/reverse_kl_mean:-0.85 "
                "- mopd/disease_state_teacher/reverse_kl_mean:-0.97 "
                "- mopd/is_zeroed_fraction:0.0",
            ]
        ),
        encoding="utf-8",
    )

    summary = module.summarize_log(log_path)

    assert summary["last_step"] == 2.0
    assert summary["training/global_step"] == 2.0
    assert summary["critic/advantages/mean"] == -0.89
    assert summary["mopd/cell_type_teacher/reverse_kl_mean"] == -0.85
    assert summary["mopd/disease_state_teacher/reverse_kl_mean"] == -0.97
    assert summary["mopd/is_zeroed_fraction"] == 0.0


def test_prepare_order_invariance_datasets_balances_teacher_batches(tmp_path):
    module = _load_order_invariance_module()

    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    train_rows = []
    val_rows = []
    for index in range(5):
        train_rows.append({"prompt": f"cell-{index}", "teacher_id": "cell_type_teacher", "value": index})
        train_rows.append({"prompt": f"disease-{index}", "teacher_id": "disease_state_teacher", "value": index})
        val_rows.append({"prompt": f"val-cell-{index}", "teacher_id": "cell_type_teacher", "value": index})
        val_rows.append({"prompt": f"val-disease-{index}", "teacher_id": "disease_state_teacher", "value": index})

    import pandas as pd

    pd.DataFrame(train_rows).to_parquet(train_path)
    pd.DataFrame(val_rows).to_parquet(val_path)

    prepared_train, prepared_val = module.prepare_order_invariance_datasets(
        train_file=str(train_path),
        val_file=str(val_path),
        output_dir=tmp_path / "prepared",
        teacher_names=("cell_type_teacher", "disease_state_teacher"),
        batch_size=8,
        dp_size=4,
        max_train_batches=1,
        max_val_batches=1,
    )

    prepared_train_df = pd.read_parquet(prepared_train)
    prepared_val_df = pd.read_parquet(prepared_val)

    assert len(prepared_train_df) == 8
    assert len(prepared_val_df) == 8
    assert prepared_train_df["teacher_id"].tolist() == [
        "cell_type_teacher",
        "cell_type_teacher",
        "cell_type_teacher",
        "cell_type_teacher",
        "disease_state_teacher",
        "disease_state_teacher",
        "disease_state_teacher",
        "disease_state_teacher",
    ]
    assert prepared_val_df["teacher_id"].tolist() == [
        "cell_type_teacher",
        "cell_type_teacher",
        "cell_type_teacher",
        "cell_type_teacher",
        "disease_state_teacher",
        "disease_state_teacher",
        "disease_state_teacher",
        "disease_state_teacher",
    ]
