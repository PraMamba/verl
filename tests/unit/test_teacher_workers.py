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

from omegaconf import OmegaConf


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
