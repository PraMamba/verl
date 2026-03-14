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

"""Unit tests for MOPD teacher resource-pool configuration."""

import pytest
from omegaconf import OmegaConf

from verl.single_controller.ray import ResourcePoolManager
from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.ray_trainer import Role


def _make_config():
    return OmegaConf.create(
        {
            "trainer": {
                "nnodes": 1,
                "n_gpus_per_node": 8,
            },
            "reward": {
                "reward_model": {
                    "enable_resource_pool": False,
                    "nnodes": 0,
                    "n_gpus_per_node": 0,
                }
            },
            "algorithm": {
                "mopd": {
                    "enabled": True,
                    "use_base_normalization": True,
                    "base_model_path": "/models/base",
                    "resource_pools": {
                        "code_pool": {
                            "nnodes": 1,
                            "n_gpus_per_node": 4,
                            "max_colocate_count": 1,
                        }
                    },
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
                            "log_prob_micro_batch_size": 4,
                        },
                    ],
                }
            },
        }
    )


def test_init_resource_pool_mgr_adds_teacher_pools_and_dynamic_colocate_capacity():
    runner = TaskRunner()
    runner.mapping = {
        Role.ActorRollout: "global_pool",
        Role.Critic: "global_pool",
        Role.RefPolicy: "global_pool",
    }
    config = _make_config()

    resource_pool_manager = runner.init_resource_pool_mgr(config)

    assert resource_pool_manager.resource_pool_spec["global_pool"]["process_on_nodes"] == [8]
    assert resource_pool_manager.resource_pool_spec["global_pool"]["max_colocate_count"] == 5
    assert resource_pool_manager.resource_pool_spec["code_pool"]["process_on_nodes"] == [4]
    assert resource_pool_manager.resource_pool_spec["code_pool"]["max_colocate_count"] == 1


def test_resource_pool_manager_parses_rich_resource_pool_spec(monkeypatch):
    created_pools = {}

    class FakeRayResourcePool:
        def __init__(self, process_on_nodes, use_gpu, max_colocate_count, name_prefix, accelerator_type=None):
            created_pools[name_prefix] = {
                "process_on_nodes": process_on_nodes,
                "use_gpu": use_gpu,
                "max_colocate_count": max_colocate_count,
                "accelerator_type": accelerator_type,
            }

    monkeypatch.setattr("verl.single_controller.ray.base.RayResourcePool", FakeRayResourcePool)
    monkeypatch.setattr(ResourcePoolManager, "_check_resource_available", lambda self: None)

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={
            "global_pool": {"process_on_nodes": [8], "max_colocate_count": 5},
            "code_pool": {"process_on_nodes": [4], "max_colocate_count": 1},
        },
        mapping={},
    )

    resource_pool_manager.create_resource_pool()

    assert created_pools["global_pool"]["process_on_nodes"] == [8]
    assert created_pools["global_pool"]["max_colocate_count"] == 5
    assert created_pools["code_pool"]["process_on_nodes"] == [4]
    assert created_pools["code_pool"]["max_colocate_count"] == 1


def test_resource_pool_manager_keeps_legacy_list_specs(monkeypatch):
    created_pools = {}

    class FakeRayResourcePool:
        def __init__(self, process_on_nodes, use_gpu, max_colocate_count, name_prefix, accelerator_type=None):
            created_pools[name_prefix] = {
                "process_on_nodes": process_on_nodes,
                "max_colocate_count": max_colocate_count,
            }

    monkeypatch.setattr("verl.single_controller.ray.base.RayResourcePool", FakeRayResourcePool)
    monkeypatch.setattr(ResourcePoolManager, "_check_resource_available", lambda self: None)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec={"global_pool": [8]}, mapping={})

    resource_pool_manager.create_resource_pool()

    assert created_pools["global_pool"]["process_on_nodes"] == [8]
    assert created_pools["global_pool"]["max_colocate_count"] == 3


def test_init_resource_pool_mgr_rejects_reserved_teacher_pool_names():
    runner = TaskRunner()
    runner.mapping = {
        Role.ActorRollout: "global_pool",
        Role.Critic: "global_pool",
        Role.RefPolicy: "global_pool",
    }
    config = _make_config()
    config.algorithm.mopd.resource_pools = {
        "global_pool": {
            "nnodes": 1,
            "n_gpus_per_node": 4,
            "max_colocate_count": 1,
        }
    }

    with pytest.raises(ValueError, match="cannot redefine reserved pool"):
        runner.init_resource_pool_mgr(config)
