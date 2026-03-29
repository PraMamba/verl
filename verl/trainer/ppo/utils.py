# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import warnings
from enum import Enum

from omegaconf import DictConfig

from verl.single_controller.base import Worker
from verl.trainer.ppo.core_algos import AdvantageEstimator

WorkerType = type[Worker]
MOPD_TEACHER_RUNTIME_ADV_ESTIMATORS = {"mopd", "single_teacher_reverse_kl"}


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    Env = 7

    def __str__(self):
        return self._get_role_string()

    def _get_role_string(self):
        role_mapping = {
            Role.Actor: "actor",
            Role.Rollout: "rollout",
            Role.ActorRollout: "actor_rollout",
            Role.Critic: "critic",
            Role.RefPolicy: "ref",
            Role.RewardModel: "rm",
            Role.ActorRolloutRef: "actor_rollout_ref",
        }
        return role_mapping.get(self, self.name.lower())

    @classmethod
    def from_string(cls, name: str):
        string_mapping = {
            "actor": cls.Actor,
            "rollout": cls.Rollout,
            "actor_rollout": cls.ActorRollout,
            "critic": cls.Critic,
            "ref": cls.RefPolicy,
            "rm": cls.RewardModel,
            "actor_rollout_ref": cls.ActorRolloutRef,
        }
        role = string_mapping.get(name.lower())
        if role is None:
            raise ValueError(f"No Role found for string: {name}")
        return role


def need_reference_policy(
    config: DictConfig,
) -> bool:
    """Given the config, do we need ref policy (for KL penalty or MOPD)."""
    return (
        config.algorithm.use_kl_in_reward
        or config.actor_rollout_ref.actor.use_kl_loss
        or need_mopd_teacher_runtime(config.algorithm)
    )


def need_mopd_teacher_runtime(algorithm_config: DictConfig) -> bool:
    """Whether the current algorithm config requires MOPD teacher orchestration.

    This is true for the native MOPD estimator and for any other estimator that
    reuses the same trainer-side teacher routing/runtime surface, such as the
    single-teacher reverse-KL reduction baseline.
    """

    adv_estimator = algorithm_config.get("adv_estimator", "")
    if hasattr(adv_estimator, "value"):
        adv_estimator = adv_estimator.value
    return algorithm_config.get("mopd", {}).get("enabled", False) or adv_estimator in MOPD_TEACHER_RUNTIME_ADV_ESTIMATORS


def need_reward_model(
    config: DictConfig,
) -> bool:
    """Given the config, do we need reward model."""
    return config.reward.reward_model.enable


def need_critic(config: DictConfig) -> bool:
    """Given a config, do we need critic."""
    if config.critic.enable is not None:
        return bool(config.critic.enable)
    elif config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return True
    else:
        warnings.warn(
            "Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True",
            stacklevel=2,
        )
        return False
