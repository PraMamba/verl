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

from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig

__all__ = ["TeacherConfig", "TeacherResourcePoolConfig", "MOPDConfig"]


@dataclass
class TeacherConfig(BaseConfig):
    """Configuration for a single teacher model in MOPD.

    Args:
        name: Unique teacher identifier (e.g., "math", "code")
        model_path: HuggingFace model path or local checkpoint
        backend: Teacher backend selection (legacy ref worker or dedicated HF quantized worker)
        weight: Deprecated placeholder. Only the default value 1.0 is accepted because
            the current runtime routes each sample to exactly one teacher instead of
            doing weighted multi-teacher fusion.
        resource_pool: Ray resource pool name (default: "global_pool")
        log_prob_micro_batch_size: Micro-batch size for teacher forward pass
        lambda_val: Optional per-teacher ExOPD lambda override
        tokenizer_path: Optional tokenizer path for compatibility checks
        tokenizer_compat_group: Optional explicit compatibility group label
        tokenizer_policy: Whether this teacher participates through token-level log prob or sequence reward
        seq_reward_weight: Weight for sequence-level teacher reward mixing in the estimator
        base_model_path: Deprecated per-teacher base model knob. The current ExOPD runtime
            only supports the shared algorithm.mopd.base_model_path setting.
    """

    name: str = ""
    model_path: str = ""
    backend: str = "legacy_ref"
    weight: float = 1.0
    resource_pool: str = "global_pool"
    log_prob_micro_batch_size: int = 8
    lambda_val: Optional[float] = None
    tokenizer_path: Optional[str] = None
    tokenizer_compat_group: Optional[str] = None
    tokenizer_policy: str = "compatible"
    seq_reward_weight: float = 1.0
    base_model_path: Optional[str] = None

    def __post_init__(self):
        valid_backends = {"legacy_ref", "hf_int8", "hf_4bit"}
        valid_tokenizer_policies = {"compatible", "sequence_reward"}

        if not self.name:
            raise ValueError("TeacherConfig.name must be non-empty")
        if not self.model_path:
            raise ValueError("TeacherConfig.model_path must be non-empty")
        if self.backend not in valid_backends:
            raise ValueError(f"TeacherConfig.backend must be one of {sorted(valid_backends)}. Got: {self.backend}")
        if self.weight <= 0:
            raise ValueError(f"TeacherConfig.weight must be positive: {self.weight}")
        if self.weight != 1.0:
            raise ValueError(
                "TeacherConfig.weight is not supported by the current MOPD runtime. "
                "The implementation routes each sample to exactly one teacher rather than doing weighted fusion."
            )
        if self.lambda_val is not None and self.lambda_val <= 0:
            raise ValueError(f"TeacherConfig.lambda_val must be positive: {self.lambda_val}")
        if self.tokenizer_compat_group is not None and not self.tokenizer_compat_group:
            raise ValueError("TeacherConfig.tokenizer_compat_group must be non-empty when provided")
        if self.tokenizer_policy not in valid_tokenizer_policies:
            raise ValueError(
                "TeacherConfig.tokenizer_policy must be one of "
                f"{sorted(valid_tokenizer_policies)}. Got: {self.tokenizer_policy}"
            )
        if self.seq_reward_weight <= 0:
            raise ValueError(f"TeacherConfig.seq_reward_weight must be positive: {self.seq_reward_weight}")
        if self.base_model_path is not None:
            raise ValueError(
                "Per-teacher base_model_path is not supported by the current ExOPD runtime. "
                "Use the shared algorithm.mopd.base_model_path instead."
            )


@dataclass
class TeacherResourcePoolConfig(BaseConfig):
    """Configuration for an explicit MOPD teacher resource pool.

    Args:
        nnodes: Number of nodes assigned to this pool
        n_gpus_per_node: Number of GPUs per node in this pool
        max_colocate_count: Optional explicit colocate budget override
    """

    nnodes: int = 1
    n_gpus_per_node: int = 1
    max_colocate_count: Optional[int] = None

    def __post_init__(self):
        if self.nnodes <= 0:
            raise ValueError(f"TeacherResourcePoolConfig.nnodes must be positive: {self.nnodes}")
        if self.n_gpus_per_node <= 0:
            raise ValueError(f"TeacherResourcePoolConfig.n_gpus_per_node must be positive: {self.n_gpus_per_node}")
        if self.max_colocate_count is not None and self.max_colocate_count <= 0:
            raise ValueError(
                "TeacherResourcePoolConfig.max_colocate_count must be positive when provided: "
                f"{self.max_colocate_count}"
            )


@dataclass
class MOPDConfig(BaseConfig):
    """Configuration for Multi-Teacher On-Policy Distillation.

    Implements MiMo paper (arXiv:2601.02780) Eq. 7-9 + G-OPD ExOPD mode.

    Args:
        enabled: Enable MOPD training (default: False for backward compat)
        teachers: List of teacher configurations
        lambda_val: G-OPD scaling coefficient (1.0=standard MOPD, >1.0=extrapolation)
        orm_weight: Weight for outcome reward mixing (α in A_final = A_mopd + α·A_orm)
        is_correction: Enable importance sampling correction for train/inference mismatch
        is_epsilon_low: Lower bound for IS ratio acceptance
        is_epsilon_high: Upper bound for IS ratio acceptance
        use_base_normalization: Enable ExOPD base model normalization
        base_model_path: Path to base model for ExOPD (shared across teachers)
    """

    enabled: bool = False
    teachers: list[TeacherConfig] = field(default_factory=list)
    lambda_val: float = 1.0
    orm_weight: float = 0.0
    is_correction: bool = True
    is_epsilon_low: float = 0.1
    is_epsilon_high: float = 10.0
    use_base_normalization: bool = False
    base_model_path: Optional[str] = None
    resource_pools: dict[str, TeacherResourcePoolConfig] = field(default_factory=dict)

    def __post_init__(self):
        # Validate non-empty teachers when enabled
        if self.enabled and len(self.teachers) == 0:
            raise ValueError(
                "MOPD enabled=True requires at least one teacher. Add teachers to algorithm.mopd.teachers."
            )

        # Validate unique teacher names
        names = [t.name for t in self.teachers]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate teacher names: {set(duplicates)}")

        # Validate lambda_val
        if self.lambda_val <= 0:
            raise ValueError(f"lambda_val must be positive: {self.lambda_val}")

        # Validate IS epsilon bounds
        if self.is_epsilon_low >= self.is_epsilon_high:
            raise ValueError(
                f"is_epsilon_low ({self.is_epsilon_low}) must be < is_epsilon_high ({self.is_epsilon_high})"
            )

        # Validate base normalization config
        if self.use_base_normalization and not self.base_model_path:
            raise ValueError("use_base_normalization=True requires base_model_path to be set")
