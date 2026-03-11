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

import torch

from verl.trainer.ppo.core_algos import get_adv_estimator_fn


def test_mopd_advantage_basic():
    """Test basic MOPD advantage computation (lambda=1.0)."""
    B, T = 4, 10
    teacher_log_prob = torch.randn(B, T)
    old_log_probs = torch.randn(B, T)
    response_mask = torch.ones(B, T)
    token_level_rewards = torch.randn(B, T)

    mopd_fn = get_adv_estimator_fn("mopd")
    advantages, _returns = mopd_fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        teacher_log_prob=teacher_log_prob,
        old_log_probs=old_log_probs,
        lambda_val=1.0,
    )

    # Advantage should be teacher_log_prob - old_log_probs (detached)
    expected = (teacher_log_prob - old_log_probs).detach() * response_mask
    torch.testing.assert_close(advantages, expected)


def test_mopd_advantage_with_is_correction():
    """Test IS correction masks tokens outside epsilon bounds."""
    B, T = 2, 5
    teacher_log_prob = torch.ones(B, T) * 2.0  # Non-zero to verify masking
    old_log_probs = torch.ones(B, T) * 1.0  # Non-zero advantage
    rollout_log_probs = torch.tensor(
        [
            [1.0, 1.0, -4.0, 1.0, 1.0],  # token 2: ratio = exp(1-(-4)) = 148 > 10
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    response_mask = torch.ones(B, T)
    token_level_rewards = torch.zeros(B, T)

    mopd_fn = get_adv_estimator_fn("mopd")
    advantages, _ = mopd_fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        teacher_log_prob=teacher_log_prob,
        old_log_probs=old_log_probs,
        rollout_log_probs=rollout_log_probs,
        is_correction=True,
        is_epsilon_low=0.1,
        is_epsilon_high=10.0,
    )

    # Token [0, 2] should be masked to 0 (ratio = exp(1-(-4)) = 148 > 10)
    assert advantages[0, 2] == 0.0
    # Non-masked tokens should have non-zero advantage (teacher - old = 2-1 = 1)
    assert advantages[0, 0] != 0.0


def test_mopd_advantage_exopd_mode():
    """Test ExOPD mode with base model normalization."""
    B, T = 2, 5
    teacher_log_prob = torch.ones(B, T) * 2.0
    old_log_probs = torch.ones(B, T) * 1.0
    base_log_prob = torch.ones(B, T) * 0.5
    response_mask = torch.ones(B, T)
    token_level_rewards = torch.zeros(B, T)

    mopd_fn = get_adv_estimator_fn("mopd")
    advantages, _ = mopd_fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        teacher_log_prob=teacher_log_prob,
        old_log_probs=old_log_probs,
        base_log_prob=base_log_prob,
        lambda_val=1.25,
        is_correction=False,
    )

    # ExOPD: -[(old - base) - lambda*(teacher - base)]
    # = -[(1.0 - 0.5) - 1.25*(2.0 - 0.5)]
    # = -[0.5 - 1.875] = 1.375
    expected = torch.ones(B, T) * 1.375
    torch.testing.assert_close(advantages, expected, rtol=1e-4, atol=1e-4)


def test_mopd_kwargs_received_via_dispatch():
    """Test that compute_mopd_advantage receives correct kwargs from dispatch."""
    import numpy as np

    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage

    B, T = 4, 10
    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.randn(B, T),
            "response_mask": torch.ones(B, T),
            "old_log_probs": torch.randn(B, T),
            "teacher_log_prob": torch.randn(B, T),
        }
    )
    data.non_tensor_batch["uid"] = np.array(["a", "a", "b", "b"])

    # Should not crash — verifies kwargs are passed through
    result = compute_advantage(
        data,
        adv_estimator="mopd",
        config=None,
    )
    assert "advantages" in result.batch
    assert "returns" in result.batch


def test_need_reference_policy_with_mopd():
    """Test that need_reference_policy returns True when MOPD is enabled."""
    from omegaconf import OmegaConf

    from verl.trainer.ppo.utils import need_reference_policy

    config = OmegaConf.create(
        {
            "algorithm": {
                "use_kl_in_reward": False,
                "mopd": {"enabled": True},
            },
            "actor_rollout_ref": {"actor": {"use_kl_loss": False}},
        }
    )
    assert need_reference_policy(config) is True


def test_mopd_is_correction_overflow_protection():
    """Test that IS correction handles extreme log prob differences without overflow."""
    B, T = 2, 5
    teacher_log_prob = torch.ones(B, T) * 2.0
    old_log_probs = torch.ones(B, T) * 1.0
    # Extreme log prob difference that would overflow without clamping
    rollout_log_probs = torch.tensor(
        [
            [1.0, 1.0, -50.0, 1.0, 1.0],  # token 2: diff = 51, would overflow without clamp
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    response_mask = torch.ones(B, T)
    token_level_rewards = torch.zeros(B, T)

    mopd_fn = get_adv_estimator_fn("mopd")
    advantages, _ = mopd_fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        teacher_log_prob=teacher_log_prob,
        old_log_probs=old_log_probs,
        rollout_log_probs=rollout_log_probs,
        is_correction=True,
        is_epsilon_low=0.1,
        is_epsilon_high=10.0,
    )

    # Should not contain inf or nan
    assert not torch.isinf(advantages).any(), "Advantages contain inf values"
    assert not torch.isnan(advantages).any(), "Advantages contain nan values"
    # Token with extreme ratio should be masked to 0
    assert advantages[0, 2].item() == 0.0, "Extreme ratio token should be masked"


def test_mopd_degenerate_fallback_2d_indexing():
    """Test that degenerate fallback uses correct 2D indexing."""
    B, T = 3, 5
    teacher_log_prob = torch.ones(B, T) * 2.0
    old_log_probs = torch.ones(B, T) * 1.0
    # All tokens masked for first sample, partial for second, none for third
    rollout_log_probs = torch.tensor(
        [
            [-50.0, -50.0, -50.0, -50.0, -50.0],  # All masked (ratio > 10)
            [-50.0, 1.0, 1.0, 1.0, 1.0],  # Partially masked
            [1.0, 1.0, 1.0, 1.0, 1.0],  # None masked
        ]
    )
    response_mask = torch.ones(B, T)
    token_level_rewards = torch.zeros(B, T)

    mopd_fn = get_adv_estimator_fn("mopd")
    advantages, _ = mopd_fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        teacher_log_prob=teacher_log_prob,
        old_log_probs=old_log_probs,
        rollout_log_probs=rollout_log_probs,
        is_correction=True,
        is_epsilon_low=0.1,
        is_epsilon_high=10.0,
    )

    # First sample: all tokens should have fallback (unweighted) advantages
    # Advantage = teacher - old = 2.0 - 1.0 = 1.0
    assert torch.allclose(advantages[0], torch.ones(T)), "Degenerate fallback failed for all-masked sample"
    # Third sample: no masking, should have weighted advantages
    assert advantages[2, 0].item() != 0.0, "Non-masked sample should have non-zero advantages"

