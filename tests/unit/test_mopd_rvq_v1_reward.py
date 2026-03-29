import importlib.util
from pathlib import Path

import pytest


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_rvq_v1_wrapper_fails_fast_when_reward_root_missing(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    wrapper_path = repo_root / "recipe" / "mopd" / "rvq_v1_reward.py"
    wrapper = _load_module(wrapper_path, "mopd_rvq_v1_reward_missing")

    missing_root = tmp_path / "missing_rlvr_root"
    with pytest.raises(FileNotFoundError, match="RVQ reward entrypoint not found"):
        await wrapper.compute_score(
            data_source="population_cell_type_homogeneous_train",
            solution_str="Cell 1: Oligodendrocyte",
            ground_truth=["Oligodendrocyte"],
            extra_info={"task_type": "cell_type_homogeneous"},
            rvq_reward_root=str(missing_root),
        )


@pytest.mark.asyncio
async def test_rvq_v1_wrapper_delegates_to_external_stable_entrypoint(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    wrapper_path = repo_root / "recipe" / "mopd" / "rvq_v1_reward.py"
    wrapper = _load_module(wrapper_path, "mopd_rvq_v1_reward_delegate")

    reward_root = tmp_path / "rlvr"
    reward_dir = reward_root / "custom_rewards"
    reward_dir.mkdir(parents=True)
    entrypoint = reward_dir / "deepseek_singlecell_reward.py"
    entrypoint.write_text(
        """
async def compute_deepseek_singlecell_reward(data_source, solution_str, ground_truth, extra_info, **kwargs):
    return {
        "score": 0.8,
        "llm_judge_accuracy": 0.7,
        "answer_valid": 1.0,
        "echo_data_source": data_source,
        "echo_ground_truth_len": float(len(ground_truth)),
        "echo_task_type_present": 1.0 if extra_info.get("task_type") else 0.0,
        "echo_reward_version_v1": 1.0 if kwargs.get("reward_version") == "v1" else 0.0,
    }
""",
        encoding="utf-8",
    )

    result = await wrapper.compute_score(
        data_source="population_cell_type_homogeneous_train",
        solution_str="  In summary, the answer is: Cell 1: Oligodendrocyte  ",
        ground_truth=["Oligodendrocyte"],
        extra_info={"task_type": "cell_type_homogeneous"},
        rvq_reward_root=str(reward_root),
    )

    assert result["score"] == pytest.approx(0.8)
    assert result["acc"] == pytest.approx(0.7)
    assert result["echo_data_source"] == "population_cell_type_homogeneous_train"
    assert result["echo_ground_truth_len"] == pytest.approx(1.0)
    assert result["echo_task_type_present"] == pytest.approx(1.0)
    assert result["echo_reward_version_v1"] == pytest.approx(1.0)
    assert result["pred"] == "In summary, the answer is: Cell 1: Oligodendrocyte"

