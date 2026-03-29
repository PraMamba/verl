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

"""Unit tests for MOPD sub-batch teacher routing logic.

Tests the routing logic in isolation without requiring Ray or GPU resources.
The core function `compute_teacher_log_probs_standalone` replicates the
routing algorithm from `RayPPOTrainer._compute_teacher_log_probs`.
"""

import numpy as np
import torch

from verl import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class MockTeacherWG:
    """Mock teacher worker group that returns deterministic ref_log_prob.

    Each mock teacher produces a unique constant value so tests can verify
    that results are correctly scattered back to the right batch indices.
    """

    def __init__(self, fill_value: float = 0.0):
        self.fill_value = fill_value
        self.call_count = 0
        self.last_batch_size = 0

    def compute_ref_log_prob(self, sub_batch: DataProto) -> DataProto:
        """Return a DataProto with ref_log_prob filled with self.fill_value."""
        batch_size = sub_batch.batch["responses"].shape[0]
        response_len = sub_batch.batch["responses"].shape[1]
        self.call_count += 1
        self.last_batch_size = batch_size
        result = DataProto.from_single_dict(
            {
                "ref_log_prob": torch.full(
                    (batch_size, response_len),
                    self.fill_value,
                    dtype=torch.float32,
                ),
            }
        )
        return result


class CapturingTeacherWG(MockTeacherWG):
    """Mock teacher that records the batch it receives for routing assertions."""

    def __init__(self, fill_value: float = 0.0):
        super().__init__(fill_value=fill_value)
        self.last_sub_batch = None

    def compute_ref_log_prob(self, sub_batch: DataProto) -> DataProto:
        self.last_sub_batch = sub_batch
        return super().compute_ref_log_prob(sub_batch)


class SentinelTeacherWG(MockTeacherWG):
    """Mock teacher that echoes a per-sample marker tensor for order-preservation checks."""

    def compute_ref_log_prob(self, sub_batch: DataProto) -> DataProto:
        markers = sub_batch.batch["sample_marker"].float()
        response_len = sub_batch.batch["responses"].shape[1]
        return DataProto.from_single_dict(
            {
                "ref_log_prob": markers.expand(markers.shape[0], response_len).clone(),
            }
        )


class CapturingSentinelTeacherWG(SentinelTeacherWG):
    """Sentinel teacher that also records the padded sub-batch it receives."""

    def __init__(self):
        super().__init__(fill_value=0.0)
        self.last_sub_batch = None

    def compute_ref_log_prob(self, sub_batch: DataProto) -> DataProto:
        self.last_sub_batch = sub_batch
        return super().compute_ref_log_prob(sub_batch)


class _FakeTeacherFuture:
    def __init__(self, event_log: list[str], teacher_name: str, output: DataProto):
        self.event_log = event_log
        self.teacher_name = teacher_name
        self.output = output

    def get(self) -> DataProto:
        self.event_log.append(f"get:{self.teacher_name}")
        return self.output


class AsyncTeacherWG(MockTeacherWG):
    """Mock teacher worker group exposing the planned non-blocking API."""

    def __init__(self, teacher_name: str, event_log: list[str], fill_value: float = 0.0):
        super().__init__(fill_value=fill_value)
        self.teacher_name = teacher_name
        self.event_log = event_log
        self.sync_call_count = 0
        self.async_call_count = 0

    def compute_ref_log_prob(self, sub_batch: DataProto) -> DataProto:
        self.sync_call_count += 1
        return super().compute_ref_log_prob(sub_batch)

    def compute_ref_log_prob_async(self, sub_batch: DataProto) -> _FakeTeacherFuture:
        self.async_call_count += 1
        self.event_log.append(f"submit:{self.teacher_name}")
        return _FakeTeacherFuture(self.event_log, self.teacher_name, super().compute_ref_log_prob(sub_batch))


class EngineMeshTeacherClass:
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"), blocking=False)
    def compute_ref_log_prob_async(self, data):
        return data


class EngineMeshTeacherWG(AsyncTeacherWG):
    def __init__(self, teacher_name: str, event_log: list[str], fill_value: float = 0.0):
        super().__init__(teacher_name=teacher_name, event_log=event_log, fill_value=fill_value)
        self.ray_cls_with_init = type("RayClsWithInit", (), {"cls": EngineMeshTeacherClass})()


def compute_teacher_log_probs_standalone(
    batch: DataProto,
    teacher_wgs: dict,
) -> torch.Tensor:
    """Standalone implementation of sub-batch teacher routing.

    This replicates the logic of RayPPOTrainer._compute_teacher_log_probs
    without requiring a trainer instance, for unit testing purposes.
    Kept in sync with the production implementation.

    NOTE: This test function intentionally omits pad_dataproto_to_divisor /
    unpad_dataproto (DP padding). The production code pads sub-batches to be
    divisible by the teacher worker DP size before forwarding. That padding
    logic is exercised by the integration tests; this function focuses on
    routing correctness only.
    """
    teacher_ids = batch.non_tensor_batch["teacher_id"]
    batch_size = len(teacher_ids)
    response_len = batch.batch["responses"].shape[1]

    # Validate all teacher_ids are known (fail fast before processing)
    known_teachers = set(teacher_wgs.keys())
    unique_ids = set(teacher_ids)
    unknown_ids = unique_ids - known_teachers
    if unknown_ids:
        raise ValueError(
            f"Samples have unknown teacher_ids not matching any teacher worker: "
            f"{unknown_ids}. Available teachers: {sorted(known_teachers)}"
        )

    # Initialize output tensor for collecting teacher log probs
    device = batch.batch["responses"].device
    teacher_log_probs = torch.zeros(
        batch_size,
        response_len,
        dtype=torch.float32,
        device=device,
    )

    # Group by teacher_id and forward sub-batches
    for teacher_name, teacher_wg in teacher_wgs.items():
        # Get indices for this teacher (ensure same device as output tensor)
        mask = teacher_ids == teacher_name
        indices = torch.tensor(np.where(mask)[0], dtype=torch.long, device=device)

        if len(indices) == 0:
            continue

        # Select sub-batch using integer index tensor
        sub_batch = batch.select_idxs(indices)

        # Forward to teacher
        teacher_output = teacher_wg.compute_ref_log_prob(sub_batch)
        sub_log_probs = teacher_output.batch["ref_log_prob"]

        # Validate shape matches expected response length
        if sub_log_probs.shape[1] != response_len:
            raise ValueError(
                f"Teacher '{teacher_name}' returned shape {sub_log_probs.shape}, "
                f"expected (*, {response_len})"
            )

        # Scatter back to full batch (ensure dtype/device match after Ray serialization)
        teacher_log_probs[indices] = sub_log_probs.to(dtype=torch.float32, device=device)

    return teacher_log_probs


def test_teacher_log_prob_basic_shape():
    """Test that teacher log prob computation returns correct shape."""
    # Arrange
    batch = DataProto.from_single_dict(
        {
            "input_ids": torch.randint(0, 1000, (4, 128)),
            "responses": torch.randint(0, 1000, (4, 64)),
            "attention_mask": torch.ones(4, 192),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "math", "code", "code"])
    teacher_wgs = {"math": MockTeacherWG(), "code": MockTeacherWG()}

    # Act
    teacher_log_prob = compute_teacher_log_probs_standalone(batch, teacher_wgs)

    # Assert: correct shape and dtype
    assert teacher_log_prob.shape == (4, 64)
    assert teacher_log_prob.dtype == torch.float32


def test_teacher_log_prob_correct_routing():
    """Test that sub-batches are correctly split by teacher_id and results scattered back."""
    # Arrange
    batch = DataProto.from_single_dict(
        {
            "input_ids": torch.randint(0, 1000, (6, 32)),
            "responses": torch.randint(0, 1000, (6, 16)),
            "attention_mask": torch.ones(6, 48),
        }
    )
    # Indices 0, 2, 4 -> math; indices 1, 3, 5 -> code
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "code", "math", "code", "math", "code"])

    math_wg = MockTeacherWG(fill_value=1.0)
    code_wg = MockTeacherWG(fill_value=2.0)
    teacher_wgs = {"math": math_wg, "code": code_wg}

    # Act
    teacher_log_prob = compute_teacher_log_probs_standalone(batch, teacher_wgs)

    # Assert: math samples (indices 0, 2, 4) should have value 1.0
    torch.testing.assert_close(teacher_log_prob[0], torch.ones(16) * 1.0)
    torch.testing.assert_close(teacher_log_prob[2], torch.ones(16) * 1.0)
    torch.testing.assert_close(teacher_log_prob[4], torch.ones(16) * 1.0)

    # Assert: code samples (indices 1, 3, 5) should have value 2.0
    torch.testing.assert_close(teacher_log_prob[1], torch.ones(16) * 2.0)
    torch.testing.assert_close(teacher_log_prob[3], torch.ones(16) * 2.0)
    torch.testing.assert_close(teacher_log_prob[5], torch.ones(16) * 2.0)


def test_teacher_log_prob_sub_batch_sizes():
    """Test that each teacher receives the correct sub-batch size."""
    # Arrange
    batch = DataProto.from_single_dict(
        {
            "input_ids": torch.randint(0, 1000, (5, 32)),
            "responses": torch.randint(0, 1000, (5, 16)),
            "attention_mask": torch.ones(5, 48),
        }
    )
    # 3 math samples, 2 code samples
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "math", "code", "math", "code"])

    math_wg = MockTeacherWG(fill_value=1.0)
    code_wg = MockTeacherWG(fill_value=2.0)
    teacher_wgs = {"math": math_wg, "code": code_wg}

    # Act
    compute_teacher_log_probs_standalone(batch, teacher_wgs)

    # Assert: each teacher was called once with correct sub-batch size
    assert math_wg.call_count == 1
    assert math_wg.last_batch_size == 3
    assert code_wg.call_count == 1
    assert code_wg.last_batch_size == 2


def test_teacher_log_prob_single_teacher():
    """Test routing when all samples go to a single teacher."""
    # Arrange
    batch = DataProto.from_single_dict(
        {
            "input_ids": torch.randint(0, 1000, (3, 32)),
            "responses": torch.randint(0, 1000, (3, 16)),
            "attention_mask": torch.ones(3, 48),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "math", "math"])

    math_wg = MockTeacherWG(fill_value=3.0)
    code_wg = MockTeacherWG(fill_value=4.0)
    teacher_wgs = {"math": math_wg, "code": code_wg}

    # Act
    teacher_log_prob = compute_teacher_log_probs_standalone(batch, teacher_wgs)

    # Assert: all samples routed to math teacher
    expected = torch.ones(3, 16) * 3.0
    torch.testing.assert_close(teacher_log_prob, expected)
    assert math_wg.call_count == 1
    assert math_wg.last_batch_size == 3
    # code teacher should not be called (skipped due to empty indices)
    assert code_wg.call_count == 0


def test_teacher_log_prob_empty_teacher():
    """Test that teachers with no matching samples are skipped gracefully."""
    # Arrange
    batch = DataProto.from_single_dict(
        {
            "input_ids": torch.randint(0, 1000, (2, 32)),
            "responses": torch.randint(0, 1000, (2, 16)),
            "attention_mask": torch.ones(2, 48),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "math"])

    math_wg = MockTeacherWG(fill_value=5.0)
    code_wg = MockTeacherWG(fill_value=6.0)
    unused_wg = MockTeacherWG(fill_value=7.0)
    teacher_wgs = {"math": math_wg, "code": code_wg, "unused": unused_wg}

    # Act
    teacher_log_prob = compute_teacher_log_probs_standalone(batch, teacher_wgs)

    # Assert: only math teacher was called
    assert math_wg.call_count == 1
    assert code_wg.call_count == 0
    assert unused_wg.call_count == 0
    expected = torch.ones(2, 16) * 5.0
    torch.testing.assert_close(teacher_log_prob, expected)


def test_teacher_log_prob_unknown_teacher_id_raises():
    """Test that unknown teacher_id values raise a clear error."""
    import pytest

    # Arrange
    batch = DataProto.from_single_dict(
        {
            "input_ids": torch.randint(0, 1000, (3, 32)),
            "responses": torch.randint(0, 1000, (3, 16)),
            "attention_mask": torch.ones(3, 48),
        }
    )
    # "biology" is not in teacher_wgs
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "biology", "math"])

    math_wg = MockTeacherWG(fill_value=1.0)
    teacher_wgs = {"math": math_wg}

    # Act & Assert: should raise ValueError with unknown teacher_id
    with pytest.raises(ValueError, match="unknown teacher_ids"):
        compute_teacher_log_probs_standalone(batch, teacher_wgs)


def test_teacher_log_prob_balances_teacher_sub_batches_before_forward():
    """Teacher-routed sub-batches should be balanced across DP chunks before forward."""
    batch = DataProto.from_single_dict(
        {
            "responses": torch.randint(0, 1000, (4, 8)),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=torch.long,
            ),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "math", "math", "math"])

    math_wg = CapturingTeacherWG(fill_value=1.0)
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.teacher_wgs = {"math": math_wg}
    trainer.use_prefix_grouper = False
    trainer._get_dp_size = lambda worker_group, role: 2

    teacher_log_prob = RayPPOTrainer._compute_teacher_log_probs(trainer, batch)

    assert teacher_log_prob.shape == (4, 8)
    assert math_wg.last_sub_batch is not None

    dp_chunks = math_wg.last_sub_batch.chunk(2)
    token_totals = [chunk.batch["attention_mask"].sum().item() for chunk in dp_chunks]

    assert max(token_totals) - min(token_totals) <= 1


def test_teacher_log_prob_preserves_sample_alignment_after_balancing():
    """Balanced teacher routing must scatter results back to the original sample order."""
    batch = DataProto.from_single_dict(
        {
            "responses": torch.randint(0, 1000, (4, 4)),
            "sample_marker": torch.tensor([[10.0], [20.0], [30.0], [40.0]], dtype=torch.float32),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                ],
                dtype=torch.long,
            ),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "math", "math", "math"])

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.teacher_wgs = {"math": SentinelTeacherWG()}
    trainer.use_prefix_grouper = False
    trainer._get_dp_size = lambda worker_group, role: 2

    teacher_log_prob = RayPPOTrainer._compute_teacher_log_probs(trainer, batch)

    expected = torch.tensor(
        [
            [10.0, 10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0, 20.0],
            [30.0, 30.0, 30.0, 30.0],
            [40.0, 40.0, 40.0, 40.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(teacher_log_prob, expected)


def test_teacher_log_prob_pads_small_teacher_sub_batch_before_forward():
    """Teacher routing should pad a small sub-batch up to dp_size without crashing."""
    batch = DataProto.from_single_dict(
        {
            "responses": torch.randint(0, 1000, (4, 4)),
            "sample_marker": torch.tensor([[10.0], [20.0], [30.0], [40.0]], dtype=torch.float32),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                ],
                dtype=torch.long,
            ),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "math", "math", "math"])

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.teacher_wgs = {"math": CapturingSentinelTeacherWG()}
    trainer.use_prefix_grouper = False
    trainer._get_dp_size = lambda worker_group, role: 8

    teacher_log_prob = RayPPOTrainer._compute_teacher_log_probs(trainer, batch)

    expected = torch.tensor(
        [
            [10.0, 10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0, 20.0],
            [30.0, 30.0, 30.0, 30.0],
            [40.0, 40.0, 40.0, 40.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(teacher_log_prob, expected)
    assert trainer.teacher_wgs["math"].last_sub_batch is not None
    assert len(trainer.teacher_wgs["math"].last_sub_batch) == 8


def test_teacher_log_prob_non_divisible_teacher_sub_batch_survives_dp8():
    """Teacher routing should tolerate teacher-local sizes that are not divisible by dp_size."""
    batch = DataProto.from_single_dict(
        {
            "responses": torch.randint(0, 1000, (12, 4)),
            "sample_marker": torch.arange(1, 13, dtype=torch.float32).unsqueeze(1),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 1, 1, 1],
                    [1, 1, 1, 0],
                    [1, 1, 0, 0],
                    [1, 0, 0, 0],
                ],
                dtype=torch.long,
            ),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["math"] * 12)

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.teacher_wgs = {"math": CapturingSentinelTeacherWG()}
    trainer.use_prefix_grouper = False
    trainer._get_dp_size = lambda worker_group, role: 8

    teacher_log_prob = RayPPOTrainer._compute_teacher_log_probs(trainer, batch)

    expected = torch.arange(1, 13, dtype=torch.float32).unsqueeze(1).expand(12, 4)
    torch.testing.assert_close(teacher_log_prob, expected)
    assert trainer.teacher_wgs["math"].last_sub_batch is not None
    assert len(trainer.teacher_wgs["math"].last_sub_batch) == 16


def test_teacher_log_prob_async_overlaps_different_pools_but_serializes_same_pool():
    """Teacher forwards should overlap across pools while preserving same-pool sequencing."""
    batch = DataProto.from_single_dict(
        {
            "responses": torch.randint(0, 1000, (3, 4)),
            "attention_mask": torch.ones(3, 4, dtype=torch.long),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "reasoning", "code"])

    event_log: list[str] = []
    math_wg = AsyncTeacherWG("math", event_log, fill_value=1.0)
    reasoning_wg = AsyncTeacherWG("reasoning", event_log, fill_value=2.0)
    code_wg = AsyncTeacherWG("code", event_log, fill_value=3.0)

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.teacher_wgs = {"math": math_wg, "reasoning": reasoning_wg, "code": code_wg}
    trainer.use_prefix_grouper = False
    trainer._get_dp_size = lambda worker_group, role: 1
    trainer.config = type(
        "Cfg",
        (),
        {
            "algorithm": type(
                "AlgoCfg",
                (),
                {
                    "mopd": type(
                        "MOPDCfg",
                        (),
                        {
                            "teachers": [
                                type("Teacher", (), {"name": "math", "resource_pool": "global_pool"})(),
                                type("Teacher", (), {"name": "reasoning", "resource_pool": "global_pool"})(),
                                type("Teacher", (), {"name": "code", "resource_pool": "code_pool"})(),
                            ]
                        },
                    )()
                },
            )()
        },
    )()

    teacher_log_prob = RayPPOTrainer._compute_teacher_log_probs(trainer, batch)

    expected = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(teacher_log_prob, expected)
    assert event_log[:2] == ["submit:math", "submit:code"]
    assert event_log.index("submit:reasoning") > event_log.index("get:math")
    assert math_wg.sync_call_count == 0
    assert reasoning_wg.sync_call_count == 0
    assert code_wg.sync_call_count == 0


def test_teacher_log_prob_uses_teacher_dispatch_mesh_for_dp_size():
    """Teacher DP sizing should follow the worker method's dispatch mesh metadata."""
    batch = DataProto.from_single_dict(
        {
            "responses": torch.randint(0, 1000, (1, 4)),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["engine"])

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.teacher_wgs = {"engine": EngineMeshTeacherWG("engine", [], fill_value=1.0)}
    trainer.use_prefix_grouper = False
    requested_meshes = []
    trainer._get_dp_size = lambda worker_group, role: requested_meshes.append(role) or 1

    teacher_log_prob = RayPPOTrainer._compute_teacher_log_probs(trainer, batch)

    torch.testing.assert_close(teacher_log_prob, torch.ones(1, 4, dtype=torch.float32))
    assert requested_meshes == ["ref"]


def test_teacher_log_prob_pads_non_divisible_teacher_subset_before_forward():
    """Teacher subsets that are not divisible by DP size should still route correctly."""
    batch = DataProto.from_single_dict(
        {
            "responses": torch.randint(0, 1000, (16, 4)),
            "attention_mask": torch.ones(16, 4, dtype=torch.long),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["math"] * 12 + ["code"] * 4)

    math_wg = CapturingTeacherWG(fill_value=1.0)
    code_wg = CapturingTeacherWG(fill_value=2.0)

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.teacher_wgs = {"math": math_wg, "code": code_wg}
    trainer.use_prefix_grouper = False
    trainer._get_dp_size = lambda worker_group, role: 8

    teacher_log_prob = RayPPOTrainer._compute_teacher_log_probs(trainer, batch)

    torch.testing.assert_close(teacher_log_prob[:12], torch.ones(12, 4, dtype=torch.float32))
    torch.testing.assert_close(teacher_log_prob[12:], torch.full((4, 4), 2.0, dtype=torch.float32))
    assert math_wg.last_batch_size == 16
    assert code_wg.last_batch_size == 8


def test_teacher_log_prob_pads_teacher_subset_smaller_than_dp_size():
    """Teacher subsets smaller than DP size should pad instead of asserting in balancing."""
    batch = DataProto.from_single_dict(
        {
            "responses": torch.randint(0, 1000, (8, 4)),
            "attention_mask": torch.ones(8, 4, dtype=torch.long),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["math"] * 4 + ["code"] * 4)

    math_wg = CapturingTeacherWG(fill_value=1.0)
    code_wg = CapturingTeacherWG(fill_value=2.0)

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.teacher_wgs = {"math": math_wg, "code": code_wg}
    trainer.use_prefix_grouper = False
    trainer._get_dp_size = lambda worker_group, role: 8

    teacher_log_prob = RayPPOTrainer._compute_teacher_log_probs(trainer, batch)

    torch.testing.assert_close(teacher_log_prob[:4], torch.ones(4, 4, dtype=torch.float32))
    torch.testing.assert_close(teacher_log_prob[4:], torch.full((4, 4), 2.0, dtype=torch.float32))
    assert math_wg.last_batch_size == 8
    assert code_wg.last_batch_size == 8
