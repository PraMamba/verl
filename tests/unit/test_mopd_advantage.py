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

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import get_adv_estimator_fn
from verl.trainer.ppo.ray_trainer import compute_advantage


def test_mopd_advantage_basic():
    """Test basic MOPD advantage computation (lambda=1.0)."""
    B, T = 4, 10
    teacher_log_prob = torch.randn(B, T)
    old_log_probs = torch.randn(B, T)
    response_mask = torch.ones(B, T)
    token_level_rewards = torch.randn(B, T)

    mopd_fn = get_adv_estimator_fn("mopd")
    advantages, _returns, _is_metrics = mopd_fn(
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
    advantages, _, _ = mopd_fn(
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
    advantages, _, _ = mopd_fn(
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


def test_batch_lambda_overrides_config_scalar_for_exopd_dispatch():
    """Batch-provided lambda tensor should override the global config scalar in ExOPD mode."""
    teacher_log_prob = torch.full((2, 3), 2.0)
    old_log_probs = torch.full((2, 3), 1.0)
    base_log_prob = torch.full((2, 3), 0.5)
    lambda_val = torch.tensor([[1.0], [2.0]], dtype=torch.float32)

    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.zeros(2, 3),
            "response_mask": torch.ones(2, 3),
            "old_log_probs": old_log_probs,
            "teacher_log_prob": teacher_log_prob,
            "base_log_prob": base_log_prob,
            "lambda_val": lambda_val,
        }
    )
    data.non_tensor_batch["uid"] = np.array(["q1", "q2"])

    config = OmegaConf.create({"mopd": {"lambda_val": 9.0, "is_correction": False}})

    result = compute_advantage(data, adv_estimator="mopd", config=config)

    expected = -((old_log_probs - base_log_prob) - lambda_val * (teacher_log_prob - base_log_prob))
    torch.testing.assert_close(result.batch["advantages"], expected)


def test_batch_lambda_does_not_change_standard_mopd_without_base_log_prob():
    """Standard MOPD should ignore lambda even if batch-level overrides are present."""
    teacher_log_prob = torch.tensor([[2.0, 2.5], [3.0, 3.5]])
    old_log_probs = torch.tensor([[1.0, 1.5], [2.0, 2.5]])

    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.zeros(2, 2),
            "response_mask": torch.ones(2, 2),
            "old_log_probs": old_log_probs,
            "teacher_log_prob": teacher_log_prob,
            "lambda_val": torch.tensor([[1.0], [8.0]], dtype=torch.float32),
        }
    )
    data.non_tensor_batch["uid"] = np.array(["q1", "q2"])

    config = OmegaConf.create({"mopd": {"lambda_val": 4.0, "is_correction": False}})

    result = compute_advantage(data, adv_estimator="mopd", config=config)

    expected = (teacher_log_prob - old_log_probs).detach()
    torch.testing.assert_close(result.batch["advantages"], expected)


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
    advantages, _, _ = mopd_fn(
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
    advantages, _, _ = mopd_fn(
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


def test_mopd_orm_mixing_formula():
    """Test ORM mixing: A_final = weights * (A_mopd + orm_weight * A_orm)."""
    import numpy as np

    B, T = 4, 5
    teacher_log_prob = torch.ones(B, T) * 2.0
    old_log_probs = torch.ones(B, T) * 1.0
    response_mask = torch.ones(B, T)
    # Constant reward: sum=5.0 per sample. With unique uids (group size 1),
    # GRPO normalizes: mean=0, std=1, so A_orm = (5.0 - 0) / (1 + 1e-6) ≈ 5.0
    token_level_rewards = torch.ones(B, T) * 1.0
    uids = np.array(["a", "b", "c", "d"])  # Each sample in own group

    mopd_fn = get_adv_estimator_fn("mopd")
    orm_weight = 0.5
    advantages, returns, is_metrics = mopd_fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        teacher_log_prob=teacher_log_prob,
        old_log_probs=old_log_probs,
        is_correction=False,
        orm_weight=orm_weight,
        index=uids,
    )

    # A_mopd = teacher - old = 2.0 - 1.0 = 1.0
    # A_orm �� 5.0 (score=5.0, group-of-1 normalization)
    # weights = 1.0 (no IS correction)
    # A_final = 1.0 * (1.0 + 0.5 * 5.0) = 3.5
    expected_per_token = 1.0 + orm_weight * (5.0 / (1.0 + 1e-6))
    assert advantages.shape == (B, T)
    torch.testing.assert_close(
        advantages[0, 0],
        torch.tensor(expected_per_token, dtype=torch.float32),
        rtol=1e-4,
        atol=1e-4,
    )


def test_mopd_orm_without_index_raises():
    """Test that orm_weight > 0 without index raises ValueError."""
    import pytest

    B, T = 2, 5
    teacher_log_prob = torch.ones(B, T)
    old_log_probs = torch.ones(B, T)
    response_mask = torch.ones(B, T)
    token_level_rewards = torch.ones(B, T)

    mopd_fn = get_adv_estimator_fn("mopd")
    with pytest.raises(ValueError, match="requires 'index'"):
        mopd_fn(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            teacher_log_prob=teacher_log_prob,
            old_log_probs=old_log_probs,
            is_correction=False,
            orm_weight=0.5,
            # No index provided
        )


def test_mopd_advantage_sequence_teacher_signal_changes_result_when_orm_disabled():
    """Sequence-level teacher rewards should affect the final advantage even with ORM disabled."""
    teacher_log_prob = torch.zeros(2, 3)
    old_log_probs = torch.zeros(2, 3)
    response_mask = torch.ones(2, 3)
    token_level_rewards = torch.zeros(2, 3)
    teacher_seq_reward = torch.tensor([0.0, 4.0], dtype=torch.float32)
    uids = np.array(["q1", "q2"])

    mopd_fn = get_adv_estimator_fn("mopd")
    base_advantages, _, _ = mopd_fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        teacher_log_prob=teacher_log_prob,
        old_log_probs=old_log_probs,
        is_correction=False,
        orm_weight=0.0,
        index=uids,
    )
    seq_advantages, returns, _ = mopd_fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        teacher_log_prob=teacher_log_prob,
        old_log_probs=old_log_probs,
        is_correction=False,
        orm_weight=0.0,
        teacher_seq_reward=teacher_seq_reward,
        teacher_seq_weight=0.5,
        index=uids,
    )

    torch.testing.assert_close(base_advantages, torch.zeros_like(base_advantages))
    assert not torch.allclose(seq_advantages, base_advantages)
    torch.testing.assert_close(returns, token_level_rewards)


def test_mopd_advantage_sequence_teacher_signal_adds_on_top_of_orm():
    """Sequence-level teacher rewards should remain a distinct additive path instead of being aliased to ORM."""
    teacher_log_prob = torch.zeros(2, 3)
    old_log_probs = torch.zeros(2, 3)
    response_mask = torch.ones(2, 3)
    token_level_rewards = torch.ones(2, 3)
    teacher_seq_reward = torch.tensor([1.0, 3.0], dtype=torch.float32)
    uids = np.array(["q1", "q2"])

    mopd_fn = get_adv_estimator_fn("mopd")
    orm_only_advantages, _, _ = mopd_fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        teacher_log_prob=teacher_log_prob,
        old_log_probs=old_log_probs,
        is_correction=False,
        orm_weight=0.25,
        index=uids,
    )
    seq_plus_orm_advantages, _, _ = mopd_fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        teacher_log_prob=teacher_log_prob,
        old_log_probs=old_log_probs,
        is_correction=False,
        orm_weight=0.25,
        teacher_seq_reward=teacher_seq_reward,
        teacher_seq_weight=0.5,
        index=uids,
    )

    assert not torch.allclose(seq_plus_orm_advantages, orm_only_advantages)


def test_compute_advantage_dispatch_uses_teacher_sequence_reward_and_mask_fields():
    """Batch plumbing should forward explicit sequence-teacher fields into the MOPD estimator."""
    data = DataProto.from_single_dict(
        {
            "token_level_rewards": torch.zeros(2, 3),
            "response_mask": torch.tensor(
                [
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                ]
            ),
            "old_log_probs": torch.zeros(2, 3),
            "teacher_log_prob": torch.zeros(2, 3),
            "teacher_seq_reward": torch.tensor([0.0, 2.0], dtype=torch.float32),
            "teacher_seq_weight": torch.tensor([[0.0], [0.5]], dtype=torch.float32),
            "teacher_token_mask": torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        }
    )
    data.non_tensor_batch["uid"] = np.array(["q1", "q2"])

    config = OmegaConf.create({"mopd": {"orm_weight": 0.0, "is_correction": False}})

    result = compute_advantage(data, adv_estimator="mopd", config=config)

    assert not torch.allclose(result.batch["advantages"], torch.zeros_like(result.batch["advantages"]))


def test_mopd_is_metrics_values():
    """Test that IS diagnostic metrics have correct values."""
    B, T = 2, 5
    teacher_log_prob = torch.ones(B, T) * 2.0
    old_log_probs = torch.ones(B, T) * 1.0
    # Sample 0, token 2: ratio = exp(1-(-4)) = exp(5) ≈ 148.4 > 10 → invalid
    # All other tokens: ratio = exp(1-1) = 1.0 → valid
    rollout_log_probs = torch.tensor(
        [
            [1.0, 1.0, -4.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    response_mask = torch.ones(B, T)
    token_level_rewards = torch.zeros(B, T)

    mopd_fn = get_adv_estimator_fn("mopd")
    _advantages, _returns, is_metrics = mopd_fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        teacher_log_prob=teacher_log_prob,
        old_log_probs=old_log_probs,
        rollout_log_probs=rollout_log_probs,
        is_correction=True,
        is_epsilon_low=0.1,
        is_epsilon_high=10.0,
    )

    # Verify metrics are present and are Python scalars
    assert "mopd/is_ratio_mean" in is_metrics
    assert "mopd/is_valid_fraction" in is_metrics
    assert "mopd/is_zeroed_fraction" in is_metrics
    assert isinstance(is_metrics["mopd/is_ratio_mean"], float)

    # 9 of 10 tokens have ratio=1.0, 1 token has ratio=exp(5)≈148.4
    # valid_fraction = 9/10 = 0.9 (the one with ratio>10 is invalid)
    assert abs(is_metrics["mopd/is_valid_fraction"] - 0.9) < 1e-5
    # zeroed_fraction = 1/10 = 0.1
    assert abs(is_metrics["mopd/is_zeroed_fraction"] - 0.1) < 1e-5
