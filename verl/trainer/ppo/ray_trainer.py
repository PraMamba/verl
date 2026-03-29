# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import hashlib
import json
import logging
import os
import uuid
from collections import Counter, defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from tensordict import TensorDict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base.decorator import MAGIC_ATTR
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    need_critic,
    need_mopd_teacher_runtime,
    need_reference_policy,
    need_reward_model,
)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.tokenizer import hf_tokenizer
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

logger = logging.getLogger(__file__)
MOPD_MANIFEST_FILENAME = "mopd_manifest.json"
CHECKPOINT_COMPLETE_FILENAME = "checkpoint.complete"


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # MOPD-specific kwargs (pass from batch if available)
        if "teacher_log_prob" in data.batch:
            adv_kwargs["teacher_log_prob"] = data.batch["teacher_log_prob"]
        if "old_log_probs" in data.batch:
            adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]
        if "rollout_log_probs" in data.batch:
            adv_kwargs["rollout_log_probs"] = data.batch["rollout_log_probs"]
        if "base_log_prob" in data.batch:
            adv_kwargs["base_log_prob"] = data.batch["base_log_prob"]
        if "lambda_val" in data.batch:
            adv_kwargs["lambda_val"] = data.batch["lambda_val"]
        if "teacher_seq_reward" in data.batch:
            adv_kwargs["teacher_seq_reward"] = data.batch["teacher_seq_reward"]
        if "teacher_seq_weight" in data.batch:
            adv_kwargs["teacher_seq_weight"] = data.batch["teacher_seq_weight"]
        if "teacher_token_mask" in data.batch:
            adv_kwargs["teacher_token_mask"] = data.batch["teacher_token_mask"]
        # Pass MOPD config values if available
        mopd_cfg = getattr(config, "mopd", None) if config else None
        if mopd_cfg is not None:
            adv_kwargs.setdefault("lambda_val", getattr(mopd_cfg, "lambda_val", 1.0))
            adv_kwargs["orm_weight"] = getattr(mopd_cfg, "orm_weight", 0.0)
            adv_kwargs["is_correction"] = getattr(mopd_cfg, "is_correction", True)
            adv_kwargs["is_epsilon_low"] = getattr(mopd_cfg, "is_epsilon_low", 0.1)
            adv_kwargs["is_epsilon_high"] = getattr(mopd_cfg, "is_epsilon_high", 10.0)
        if adv_estimator in ("mopd", AdvantageEstimator.MOPD_ZERO_TEACHER_ORM_ONLY):
            adv_kwargs["norm_adv_by_std_in_grpo"] = norm_adv_by_std_in_grpo
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        result = adv_estimator_fn(**adv_kwargs)
        # MOPD returns (advantages, returns, is_metrics); others return (advantages, returns)
        if len(result) == 3:
            advantages, returns, mopd_is_metrics = result
            # IS metrics are already Python scalars (converted in compute_mopd_advantage)
            for k, v in mopd_is_metrics.items():
                data.meta_info.setdefault("metrics", {})[k] = v
        else:
            advantages, returns = result
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.checkpoint_manager = None
        self.teacher_wgs: dict[str, RayWorkerGroup] = {}  # MOPD teacher worker groups
        self.base_policy_wg: Optional[RayWorkerGroup] = None

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = (
            set({"data_source", "reward_model", "extra_info", "uid", "teacher_id"}) & batch.non_tensor_batch.keys()
        )

        # pop those keys for generation
        batch_keys_to_pop = list(batch.batch.keys())
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _get_validation_metric_groups(self, batch: DataProto, batch_size: int) -> np.ndarray:
        group_key = self.config.trainer.get("val_metric_group_key", None)
        if group_key:
            group_values = batch.non_tensor_batch.get(group_key, None)
            if group_values is not None:
                return np.asarray(group_values, dtype=object)

        return np.asarray(batch.non_tensor_batch.get("data_source", ["unknown"] * batch_size), dtype=object)

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(self._get_validation_metric_groups(test_batch, batch_size=reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _build_teacher_worker_config(self, teacher_cfg):
        """Build a teacher worker config from actor/ref defaults and teacher overrides."""
        backend = getattr(teacher_cfg, "backend", "legacy_ref")
        if backend != "legacy_ref":
            raise ValueError(
                "Quantized teacher backends must use a dedicated quantized teacher worker instead of "
                "the legacy ref-worker config builder."
            )
        teacher_worker_config = OmegaConf.create(OmegaConf.to_container(self.config.actor_rollout_ref, resolve=True))
        with open_dict(teacher_worker_config):
            teacher_worker_config.model.path = teacher_cfg.model_path
            teacher_worker_config.model.tokenizer_path = (
                getattr(teacher_cfg, "tokenizer_path", None) or teacher_cfg.model_path
            )
            # Teacher forwards run on routed sub-batches. Use fixed per-teacher
            # micro batches rather than inheriting the global dynamic ref path.
            teacher_worker_config.ref.log_prob_use_dynamic_bsz = False
            teacher_worker_config.ref.log_prob_micro_batch_size = None
            teacher_worker_config.ref.log_prob_micro_batch_size_per_gpu = teacher_cfg.log_prob_micro_batch_size
        return teacher_worker_config

    def _build_quantized_teacher_worker_config(self, teacher_cfg):
        """Build config for the dedicated inference-only quantized teacher worker."""
        teacher_worker_config = OmegaConf.create(
            {
                "model": OmegaConf.to_container(self.config.actor_rollout_ref.model, resolve=True),
                "teacher": OmegaConf.to_container(teacher_cfg, resolve=True),
            }
        )
        with open_dict(teacher_worker_config):
            teacher_worker_config.model.path = teacher_cfg.model_path
            teacher_worker_config.model.tokenizer_path = (
                getattr(teacher_cfg, "tokenizer_path", None) or teacher_cfg.model_path
            )
        return teacher_worker_config

    def _build_base_worker_config(self):
        """Build a shared base worker config for ExOPD normalization."""
        base_model_path = OmegaConf.select(self.config, "algorithm.mopd.base_model_path")
        if not base_model_path:
            raise ValueError("algorithm.mopd.base_model_path must be set when base normalization is enabled")

        base_worker_config = OmegaConf.create(OmegaConf.to_container(self.config.actor_rollout_ref, resolve=True))
        with open_dict(base_worker_config):
            base_worker_config.model.path = base_model_path
        return base_worker_config

    def _get_batch_device(self, batch: DataProto) -> torch.device:
        if len(batch.batch.keys()) == 0:
            raise ValueError("Cannot infer device from an empty batch")
        first_tensor = next(iter(batch.batch.values()))
        return first_tensor.device

    def _build_mopd_lambda_tensor(self, batch: DataProto) -> torch.Tensor:
        """Build a per-sample lambda tensor from teacher routing metadata."""
        teacher_ids = batch.non_tensor_batch["teacher_id"]
        teacher_cfg_by_name = self._get_mopd_teacher_config_by_name()
        unknown_ids = sorted(set(teacher_ids) - set(teacher_cfg_by_name))
        if unknown_ids:
            raise ValueError(
                f"Samples have unknown teacher_ids not matching configured teachers: {unknown_ids}. "
                f"Configured teachers: {sorted(teacher_cfg_by_name)}"
            )

        default_lambda = float(getattr(self.config.algorithm.mopd, "lambda_val", 1.0))
        lambda_values = []
        for teacher_id in teacher_ids:
            teacher_cfg = teacher_cfg_by_name[teacher_id]
            lambda_values.append(
                float(teacher_cfg.lambda_val)
                if getattr(teacher_cfg, "lambda_val", None) is not None
                else default_lambda
            )

        device = self._get_batch_device(batch)
        return torch.tensor(lambda_values, dtype=torch.float32, device=device).unsqueeze(-1)

    def _compute_base_log_prob(self, batch: DataProto) -> DataProto:
        if self.base_policy_wg is None:
            raise ValueError("Base normalization is enabled but base_policy_wg has not been initialized")

        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False)
            output = self.base_policy_wg.compute_ref_log_prob(batch_td)
            log_probs = tu.get(output, "log_probs")
            log_probs = no_padding_2_padding(log_probs, batch_td)
            base_log_prob = tu.get_tensordict({"base_log_prob": log_probs.float()})
            return DataProto.from_tensordict(base_log_prob)

        base_log_prob = self.base_policy_wg.compute_ref_log_prob(batch)
        base_log_prob.batch["base_log_prob"] = base_log_prob.batch.pop("ref_log_prob")
        return base_log_prob

    def _get_mopd_teacher_config_by_name(self) -> dict[str, Any]:
        algorithm_cfg = getattr(getattr(self, "config", None), "algorithm", None)
        mopd_cfg = getattr(algorithm_cfg, "mopd", None)
        if mopd_cfg is None:
            return {}
        return {teacher.name: teacher for teacher in mopd_cfg.teachers}

    def _get_mopd_teacher_pool_by_name(self) -> dict[str, str]:
        teacher_cfg_by_name = self._get_mopd_teacher_config_by_name()
        if not teacher_cfg_by_name:
            return {teacher_name: "global_pool" for teacher_name in self.teacher_wgs}
        return {
            teacher_name: getattr(teacher_cfg, "resource_pool", "global_pool")
            for teacher_name, teacher_cfg in teacher_cfg_by_name.items()
        }

    def _get_mopd_teacher_backend_by_name(self) -> dict[str, str]:
        return {
            teacher_name: getattr(teacher_cfg, "backend", "legacy_ref")
            for teacher_name, teacher_cfg in self._get_mopd_teacher_config_by_name().items()
        }

    def _get_mopd_teacher_tokenizer_policy_by_name(self) -> dict[str, str]:
        return {
            teacher_name: getattr(teacher_cfg, "tokenizer_policy", "compatible")
            for teacher_name, teacher_cfg in self._get_mopd_teacher_config_by_name().items()
        }

    def _get_mopd_teacher_names_for_policy(self, tokenizer_policy: str) -> set[str]:
        teacher_cfg_by_name = self._get_mopd_teacher_config_by_name()
        if not teacher_cfg_by_name:
            return set(self.teacher_wgs.keys())
        return {
            teacher_name
            for teacher_name, policy in self._get_mopd_teacher_tokenizer_policy_by_name().items()
            if policy == tokenizer_policy
        }

    def _submit_teacher_log_prob(self, teacher_wg, sub_batch: DataProto):
        async_method = getattr(teacher_wg, "compute_ref_log_prob_async", None)
        if callable(async_method):
            return async_method(sub_batch)
        return teacher_wg.compute_ref_log_prob(sub_batch)

    def _submit_teacher_seq_scores(self, teacher_wg, sub_batch: DataProto):
        async_method = getattr(teacher_wg, "compute_seq_scores_async", None)
        if callable(async_method):
            return async_method(sub_batch)
        return teacher_wg.compute_seq_scores(sub_batch)

    @staticmethod
    def _get_worker_method_mesh_name(worker_group, method_name: str) -> str | None:
        ray_cls_with_init = getattr(worker_group, "ray_cls_with_init", None)
        worker_cls = getattr(ray_cls_with_init, "cls", None)
        if worker_cls is None:
            return None

        method = getattr(worker_cls, method_name, None)
        if method is None:
            return None

        method_attrs = getattr(method, MAGIC_ATTR, None)
        if not isinstance(method_attrs, dict):
            return None

        dispatch_mode = method_attrs.get("dispatch_mode")
        if isinstance(dispatch_mode, dict):
            return dispatch_mode.get("mesh_name")
        return None

    def _get_teacher_dp_mesh_name(self, teacher_wg) -> str:
        for method_name in (
            "compute_ref_log_prob_async",
            "compute_ref_log_prob",
            "compute_seq_scores_async",
            "compute_seq_scores",
        ):
            mesh_name = self._get_worker_method_mesh_name(teacher_wg, method_name)
            if mesh_name is not None:
                return mesh_name
        return "actor"

    def _get_teacher_job_dp_size(self, teacher_wg) -> int:
        try:
            return self._get_dp_size(teacher_wg, self._get_teacher_dp_mesh_name(teacher_wg))
        except AttributeError:
            return 1

    @staticmethod
    def _resolve_teacher_log_prob_output(teacher_output):
        if isinstance(teacher_output, DataProto | TensorDict):
            return teacher_output
        get_fn = getattr(teacher_output, "get", None)
        if callable(get_fn):
            return teacher_output.get()
        return teacher_output

    def _build_mopd_teacher_jobs(
        self,
        batch: DataProto,
        teacher_names: Optional[set[str]] = None,
    ) -> tuple[list[dict[str, Any]], torch.device, int]:
        teacher_ids = batch.non_tensor_batch["teacher_id"]
        response_len = batch.batch["responses"].shape[1]

        known_teachers = set(self.teacher_wgs.keys())
        unique_ids = set(teacher_ids)
        unknown_ids = unique_ids - known_teachers
        if unknown_ids:
            raise ValueError(
                f"Samples have unknown teacher_ids not matching any teacher worker: "
                f"{unknown_ids}. Available teachers: {sorted(known_teachers)}"
            )

        device = batch.batch["responses"].device
        teacher_pool_by_name = self._get_mopd_teacher_pool_by_name()
        selected_teachers = known_teachers if teacher_names is None else known_teachers & set(teacher_names)
        jobs = []
        for teacher_name, teacher_wg in self.teacher_wgs.items():
            if teacher_name not in selected_teachers:
                continue
            mask = teacher_ids == teacher_name
            indices = torch.tensor(np.where(mask)[0], dtype=torch.long, device=device)

            if len(indices) == 0:
                continue

            sub_batch = batch.select_idxs(indices)
            sub_batch.non_tensor_batch["_mopd_scatter_index"] = np.asarray(indices.tolist(), dtype=np.int64)

            dp_size = self._get_teacher_job_dp_size(teacher_wg)
            if self._should_balance_teacher_sub_batch(sub_batch, dp_size):
                self._balance_batch(
                    sub_batch,
                    metrics={},
                    logging_prefix=f"teacher/{teacher_name}/seqlen",
                    dp_size=dp_size,
                )
            scatter_indices = torch.as_tensor(
                sub_batch.non_tensor_batch.pop("_mopd_scatter_index"),
                dtype=torch.long,
                device=device,
            )
            sub_batch_padded, pad_size = pad_dataproto_to_divisor(sub_batch, dp_size)

            jobs.append(
                {
                    "teacher_name": teacher_name,
                    "teacher_wg": teacher_wg,
                    "pool_name": teacher_pool_by_name.get(teacher_name, "global_pool"),
                    "teacher_cfg": self._get_mopd_teacher_config_by_name().get(teacher_name),
                    "sub_batch": sub_batch_padded,
                    "pad_size": pad_size,
                    "scatter_indices": scatter_indices,
                }
            )

        return jobs, device, response_len

    def _decode_mopd_response_texts(self, batch: DataProto) -> np.ndarray:
        return np.asarray(
            self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True),
            dtype=object,
        )

    def _build_mopd_sequence_teacher_jobs(self, batch: DataProto) -> tuple[list[dict[str, Any]], torch.device]:
        if "raw_prompt" not in batch.non_tensor_batch:
            raise ValueError("Sequence teacher jobs require raw_prompt in batch.non_tensor_batch")

        teacher_ids = batch.non_tensor_batch["teacher_id"]
        known_teachers = set(self.teacher_wgs.keys())
        unknown_ids = set(teacher_ids) - known_teachers
        if unknown_ids:
            raise ValueError(
                f"Samples have unknown teacher_ids not matching any teacher worker: "
                f"{unknown_ids}. Available teachers: {sorted(known_teachers)}"
            )

        response_texts = self._decode_mopd_response_texts(batch)
        device = batch.batch["responses"].device
        teacher_pool_by_name = self._get_mopd_teacher_pool_by_name()
        sequence_teacher_names = self._get_mopd_teacher_names_for_policy("sequence_reward")
        jobs = []

        for teacher_name, teacher_wg in self.teacher_wgs.items():
            if teacher_name not in sequence_teacher_names:
                continue

            mask = teacher_ids == teacher_name
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            torch_indices = torch.as_tensor(indices, dtype=torch.long, device=device)
            sub_batch = batch.select_idxs(torch_indices)
            sub_batch.non_tensor_batch["response_text"] = np.asarray(response_texts[indices], dtype=object)
            sub_batch.non_tensor_batch["_mopd_scatter_index"] = np.asarray(indices, dtype=np.int64)

            dp_size = self._get_teacher_job_dp_size(teacher_wg)
            if "attention_mask" in sub_batch.batch and self._should_balance_teacher_sub_batch(sub_batch, dp_size):
                self._balance_batch(
                    sub_batch,
                    metrics={},
                    logging_prefix=f"teacher/{teacher_name}/seqlen",
                    dp_size=dp_size,
                )
            scatter_indices = torch.as_tensor(
                sub_batch.non_tensor_batch.pop("_mopd_scatter_index"),
                dtype=torch.long,
                device=device,
            )
            sub_batch_padded, pad_size = pad_dataproto_to_divisor(sub_batch, dp_size)

            jobs.append(
                {
                    "teacher_name": teacher_name,
                    "teacher_wg": teacher_wg,
                    "pool_name": teacher_pool_by_name.get(teacher_name, "global_pool"),
                    "teacher_cfg": self._get_mopd_teacher_config_by_name().get(teacher_name),
                    "sub_batch": sub_batch_padded,
                    "pad_size": pad_size,
                    "scatter_indices": scatter_indices,
                    "response_texts": list(sub_batch.non_tensor_batch["response_text"]),
                    "raw_prompts": list(sub_batch.non_tensor_batch["raw_prompt"]),
                }
            )

        return jobs, device

    def _consume_teacher_job_result(
        self,
        job: dict[str, Any],
        teacher_output,
        teacher_log_probs: torch.Tensor,
        response_len: int,
        device: torch.device,
    ) -> None:
        teacher_output = self._resolve_teacher_log_prob_output(teacher_output)
        teacher_output = unpad_dataproto(teacher_output, pad_size=job["pad_size"])
        sub_log_probs = teacher_output.batch["ref_log_prob"]

        if sub_log_probs.shape[1] != response_len:
            raise ValueError(
                f"Teacher '{job['teacher_name']}' returned shape {sub_log_probs.shape}, expected (*, {response_len})"
            )

        teacher_log_probs[job["scatter_indices"]] = sub_log_probs.to(dtype=torch.float32, device=device)

    def _compute_mopd_is_valid_mask(self, old_log_probs: torch.Tensor, rollout_log_probs: torch.Tensor) -> torch.Tensor:
        log_ratio = (old_log_probs - rollout_log_probs).clamp(-20, 20)
        ratio = torch.exp(log_ratio)
        is_epsilon_low = float(getattr(self.config.algorithm.mopd, "is_epsilon_low", 0.1))
        is_epsilon_high = float(getattr(self.config.algorithm.mopd, "is_epsilon_high", 10.0))
        return (ratio >= is_epsilon_low) & (ratio <= is_epsilon_high)

    @staticmethod
    def _should_balance_teacher_sub_batch(sub_batch: DataProto, dp_size: int) -> bool:
        if dp_size <= 1:
            return True
        batch_size = len(sub_batch)
        return batch_size >= dp_size and batch_size % dp_size == 0

    def _record_mopd_teacher_metrics(self, batch: DataProto, metrics: dict[str, float]) -> None:
        if "teacher_id" not in batch.non_tensor_batch:
            return
        required_keys = {"response_mask", "advantages", "teacher_log_prob", "old_log_probs"}
        if not required_keys.issubset(batch.batch.keys()):
            return

        teacher_ids = np.asarray(batch.non_tensor_batch["teacher_id"])
        batch_size = len(teacher_ids)
        if batch_size == 0:
            return

        response_mask = batch.batch["response_mask"] > 0
        advantages = batch.batch["advantages"]
        reverse_kl = batch.batch["teacher_log_prob"] - batch.batch["old_log_probs"]
        seq_teacher_token_mask = batch.batch.get("teacher_token_mask", None)
        token_teacher_mask = None
        if seq_teacher_token_mask is not None:
            token_teacher_mask = seq_teacher_token_mask <= 0
        is_valid_mask = None
        if getattr(self.config.algorithm.mopd, "is_correction", False) and "rollout_log_probs" in batch.batch:
            is_valid_mask = self._compute_mopd_is_valid_mask(
                batch.batch["old_log_probs"], batch.batch["rollout_log_probs"]
            )

        for teacher_name in self._get_mopd_teacher_config_by_name():
            sample_mask = teacher_ids == teacher_name
            if not np.any(sample_mask):
                continue

            sample_indices = torch.as_tensor(np.where(sample_mask)[0], dtype=torch.long, device=advantages.device)
            teacher_response_mask = response_mask[sample_indices]
            valid_tokens = teacher_response_mask > 0
            teacher_advantages = advantages[sample_indices][valid_tokens]
            token_count = valid_tokens.sum().item()

            token_level_positions = valid_tokens
            if token_teacher_mask is not None:
                token_level_positions = valid_tokens & token_teacher_mask[sample_indices]
            teacher_reverse_kl = reverse_kl[sample_indices][token_level_positions]

            prefix = f"mopd/{teacher_name}"
            metrics[f"{prefix}/sample_fraction"] = float(sample_mask.sum()) / float(batch_size)
            metrics[f"{prefix}/adv_mean"] = teacher_advantages.float().mean().item() if token_count > 0 else 0.0
            metrics[f"{prefix}/adv_std"] = (
                teacher_advantages.float().std(unbiased=False).item() if token_count > 0 else 0.0
            )
            metrics[f"{prefix}/reverse_kl_mean"] = (
                teacher_reverse_kl.float().mean().item() if teacher_reverse_kl.numel() > 0 else 0.0
            )

            if "teacher_seq_reward" in batch.batch:
                teacher_seq_reward = batch.batch["teacher_seq_reward"][sample_indices]
                metrics[f"{prefix}/seq_reward_mean"] = teacher_seq_reward.float().mean().item()

            if is_valid_mask is not None:
                teacher_is_valid = is_valid_mask[sample_indices] & token_level_positions
                denom = token_level_positions.sum().clamp(min=1)
                metrics[f"{prefix}/is_valid_fraction"] = (teacher_is_valid.sum().float() / denom).item()

    def _get_mopd_teacher_id_counts(self) -> Counter:
        """Collect routed teacher_id counts from the train dataset."""
        dataset = self.train_dataset
        if hasattr(dataset, "teacher_id_field") and dataset.teacher_id_field:
            teacher_id_field = dataset.teacher_id_field
            dataframe = getattr(dataset, "dataframe", None)
            column_names = getattr(dataframe, "column_names", None)
            if dataframe is not None and column_names and teacher_id_field in column_names:
                return Counter((teacher_id or "default") for teacher_id in dataframe[teacher_id_field])

        counts = Counter()
        for idx in range(len(dataset)):
            sample = dataset[idx]
            teacher_id = sample.get("teacher_id", "default")
            counts[teacher_id] += 1
        return counts

    @staticmethod
    def _get_tokenizer_signature(tokenizer) -> dict[str, Any]:
        added_vocab = tokenizer.get_added_vocab() if hasattr(tokenizer, "get_added_vocab") else {}
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
        vocab_hash = None
        if vocab is not None:
            vocab_items = sorted(vocab.items())
            vocab_hash = hashlib.sha256(
                json.dumps(vocab_items, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()

        return {
            "tokenizer_class": tokenizer.__class__.__name__,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "vocab_hash": vocab_hash,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "padding_side": getattr(tokenizer, "padding_side", None),
            "special_tokens_map": deepcopy(getattr(tokenizer, "special_tokens_map", {})),
            "added_vocab_keys": sorted(added_vocab.keys()),
        }

    @staticmethod
    def _load_tokenizer_or_raise(path: str, trust_remote_code: bool, label: str):
        try:
            return hf_tokenizer(path, trust_remote_code=trust_remote_code)
        except Exception as exc:
            raise ValueError(f"MOPD tokenizer compatibility check failed while loading {label} tokenizer.") from exc

    def _validate_mopd_tokenizer_compatibility(self) -> None:
        if not need_mopd_teacher_runtime(self.config.algorithm):
            return

        student_tokenizer_path = OmegaConf.select(
            self.config, "actor_rollout_ref.model.tokenizer_path"
        ) or OmegaConf.select(self.config, "actor_rollout_ref.model.path")
        student_compat_group = OmegaConf.select(self.config, "actor_rollout_ref.model.tokenizer_compat_group")
        trust_remote_code = self.config.actor_rollout_ref.model.get("trust_remote_code", False)
        student_tokenizer = None

        for teacher_cfg in self.config.algorithm.mopd.teachers:
            if getattr(teacher_cfg, "tokenizer_policy", "compatible") == "sequence_reward":
                continue

            teacher_tokenizer_path = getattr(teacher_cfg, "tokenizer_path", None) or teacher_cfg.model_path
            if teacher_tokenizer_path == student_tokenizer_path:
                continue

            compat_group = getattr(teacher_cfg, "tokenizer_compat_group", None)
            if compat_group is None:
                raise ValueError(
                    "MOPD tokenizer compatibility check failed: "
                    f"teacher '{teacher_cfg.name}' uses tokenizer '{teacher_tokenizer_path}' while the student "
                    f"uses '{student_tokenizer_path}'. Token-level teacher log-prob routing requires a matching "
                    "tokenizer contract; sequence-level heterogeneous-tokenizer fallback is not implemented. "
                    "Set tokenizer_compat_group only for tokenizers with matching metadata."
                )

            if student_compat_group is not None and compat_group != student_compat_group:
                raise ValueError(
                    "MOPD tokenizer compatibility check failed: "
                    f"teacher '{teacher_cfg.name}' declared tokenizer_compat_group='{compat_group}', "
                    f"but the student requires '{student_compat_group}'."
                )

            if student_tokenizer is None:
                student_tokenizer = self._load_tokenizer_or_raise(student_tokenizer_path, trust_remote_code, "student")
            teacher_tokenizer = self._load_tokenizer_or_raise(
                teacher_tokenizer_path, trust_remote_code, f"teacher '{teacher_cfg.name}'"
            )

            student_signature = self._get_tokenizer_signature(student_tokenizer)
            teacher_signature = self._get_tokenizer_signature(teacher_tokenizer)
            if teacher_signature != student_signature:
                raise ValueError(
                    "MOPD tokenizer metadata mismatch: "
                    f"teacher '{teacher_cfg.name}' declared tokenizer_compat_group='{compat_group}' but the "
                    "loaded tokenizer metadata does not match the student tokenizer."
                )

        if getattr(self.config.algorithm.mopd, "use_base_normalization", False):
            base_tokenizer_path = getattr(self.config.algorithm.mopd, "base_model_path", None)
            if base_tokenizer_path is None:
                raise ValueError("Base normalization requires a resolvable tokenizer path for the base model")
            if base_tokenizer_path != student_tokenizer_path:
                if student_tokenizer is None:
                    student_tokenizer = self._load_tokenizer_or_raise(
                        student_tokenizer_path, trust_remote_code, "student"
                    )
                base_tokenizer = self._load_tokenizer_or_raise(base_tokenizer_path, trust_remote_code, "base model")
                if self._get_tokenizer_signature(base_tokenizer) != self._get_tokenizer_signature(student_tokenizer):
                    raise ValueError(
                        "MOPD tokenizer metadata mismatch: base model tokenizer does not match the student tokenizer."
                    )

    def _run_mopd_preflight_checks(self) -> None:
        if not need_mopd_teacher_runtime(self.config.algorithm):
            return

        teacher_counts = self._get_mopd_teacher_id_counts()
        logger.info("MOPD teacher_id distribution: %s", dict(sorted(teacher_counts.items())))

        configured_teachers = {teacher.name for teacher in self.config.algorithm.mopd.teachers}
        if not configured_teachers:
            raise ValueError("MOPD teacher runtime requires at least one configured teacher in algorithm.mopd.teachers")
        observed_teachers = set(teacher_counts)
        unknown_ids = sorted(observed_teachers - configured_teachers)
        if unknown_ids:
            raise ValueError(
                f"MOPD preflight found unknown teacher_ids in the training dataset: {unknown_ids}. "
                f"Configured teachers: {sorted(configured_teachers)}"
            )

        if len(configured_teachers) > 1 and (not teacher_counts or observed_teachers == {"default"}):
            raise ValueError(
                "MOPD preflight found no routed teacher_ids in the training dataset. "
                "Check data.teacher_id_field and dataset teacher mappings."
            )

        missing_teachers = sorted(configured_teachers - observed_teachers)
        if missing_teachers:
            raise ValueError(
                f"MOPD preflight found missing configured teachers in the training dataset: {missing_teachers}"
            )

        self._validate_mopd_tokenizer_compatibility()

    def _build_mopd_manifest(self) -> dict[str, Any]:
        if not need_mopd_teacher_runtime(self.config.algorithm):
            return {}

        adv_estimator = self.config.algorithm.adv_estimator
        if hasattr(adv_estimator, "value"):
            adv_estimator = adv_estimator.value

        teachers = sorted(self.config.algorithm.mopd.teachers, key=lambda teacher: teacher.name)
        return {
            "semantic": {
                "adv_estimator": adv_estimator,
                "lambda_val": float(getattr(self.config.algorithm.mopd, "lambda_val", 1.0)),
                "orm_weight": float(getattr(self.config.algorithm.mopd, "orm_weight", 0.0)),
                "is_correction": bool(getattr(self.config.algorithm.mopd, "is_correction", True)),
                "is_epsilon_low": float(getattr(self.config.algorithm.mopd, "is_epsilon_low", 0.1)),
                "is_epsilon_high": float(getattr(self.config.algorithm.mopd, "is_epsilon_high", 10.0)),
                "use_base_normalization": bool(getattr(self.config.algorithm.mopd, "use_base_normalization", False)),
                "base_model_path": getattr(self.config.algorithm.mopd, "base_model_path", None),
                "teachers": [
                    {
                        "name": teacher.name,
                        "model_path": teacher.model_path,
                        "lambda_val": getattr(teacher, "lambda_val", None),
                        "backend": getattr(teacher, "backend", "legacy_ref"),
                        "tokenizer_path": getattr(teacher, "tokenizer_path", None),
                        "tokenizer_compat_group": getattr(teacher, "tokenizer_compat_group", None),
                        "tokenizer_policy": getattr(teacher, "tokenizer_policy", "compatible"),
                        "seq_reward_weight": float(getattr(teacher, "seq_reward_weight", 1.0)),
                    }
                    for teacher in teachers
                ],
            },
            "deployment": {
                "resource_pools": {
                    pool_name: {
                        "nnodes": pool_cfg.nnodes,
                        "n_gpus_per_node": pool_cfg.n_gpus_per_node,
                        "max_colocate_count": getattr(pool_cfg, "max_colocate_count", None),
                    }
                    for pool_name, pool_cfg in getattr(self.config.algorithm.mopd, "resource_pools", {}).items()
                },
                "teachers": [
                    {
                        "name": teacher.name,
                        "resource_pool": teacher.resource_pool,
                        "log_prob_micro_batch_size": teacher.log_prob_micro_batch_size,
                    }
                    for teacher in teachers
                ],
            },
        }

    def _validate_loaded_mopd_manifest(self, loaded_manifest: dict[str, Any]) -> None:
        current_manifest = self._build_mopd_manifest()
        if loaded_manifest.get("semantic") != current_manifest.get("semantic"):
            raise ValueError("MOPD checkpoint semantic drift detected between saved manifest and current config")

        if loaded_manifest.get("deployment") != current_manifest.get("deployment"):
            logger.warning("MOPD checkpoint has deployment-only drift: %s", loaded_manifest.get("deployment"))

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._run_mopd_preflight_checks()
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # Create teacher workers for MOPD
        if need_mopd_teacher_runtime(self.config.algorithm):
            if Role.RefPolicy not in self.role_worker_mapping:
                raise ValueError(
                    "MOPD requires Role.RefPolicy in role_worker_mapping. "
                    "MOPD is not compatible with LoRA ref-in-actor mode. "
                    "Please ensure RefPolicy is mapped to a separate worker."
                )
            if getattr(self.config.algorithm.mopd, "use_base_normalization", False):
                base_worker_config = self._build_base_worker_config()
                base_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RefPolicy],
                    config=base_worker_config,
                    role=str(Role.RefPolicy),
                )
                base_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
                self.base_policy_wg = self.ray_worker_group_cls(
                    resource_pool=base_resource_pool,
                    ray_cls_with_init=base_cls,
                    **wg_kwargs,
                )
                self.base_policy_wg.init_model()
                logger.info(
                    "Initialized shared base worker group for ExOPD (model: %s)",
                    self.config.algorithm.mopd.base_model_path,
                )

            import ray

            from verl.workers.teacher_workers import HFQuantizedTeacherWorker

            for teacher_cfg in self.config.algorithm.mopd.teachers:
                backend = getattr(teacher_cfg, "backend", "legacy_ref")
                tokenizer_policy = getattr(teacher_cfg, "tokenizer_policy", "compatible")
                if backend == "legacy_ref":
                    if tokenizer_policy == "sequence_reward":
                        raise ValueError(
                            "tokenizer_policy=sequence_reward requires a dedicated quantized teacher worker "
                            f"backend for teacher '{teacher_cfg.name}'."
                        )
                    teacher_worker_config = self._build_teacher_worker_config(teacher_cfg)
                    teacher_cls = RayClassWithInitArgs(
                        self.role_worker_mapping[Role.RefPolicy],
                        config=teacher_worker_config,
                        role=str(Role.RefPolicy),
                    )
                elif backend in {"hf_int8", "hf_4bit"}:
                    teacher_worker_config = self._build_quantized_teacher_worker_config(teacher_cfg)
                    teacher_cls = RayClassWithInitArgs(
                        ray.remote(HFQuantizedTeacherWorker),
                        config=teacher_worker_config,
                        role="teacher",
                    )
                else:
                    raise ValueError(f"Unsupported MOPD teacher backend '{backend}' for teacher '{teacher_cfg.name}'.")

                # Validate resource pool exists before using it
                if teacher_cfg.resource_pool not in self.resource_pool_manager.resource_pool_dict:
                    raise ValueError(
                        f"Teacher '{teacher_cfg.name}' specifies unknown resource_pool "
                        f"'{teacher_cfg.resource_pool}'. Available pools: "
                        f"{list(self.resource_pool_manager.resource_pool_dict.keys())}"
                    )
                resource_pool = self.resource_pool_manager.resource_pool_dict[teacher_cfg.resource_pool]

                teacher_wg = self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=teacher_cls,
                    **wg_kwargs,
                )
                teacher_wg.init_model()
                self.teacher_wgs[teacher_cfg.name] = teacher_wg
                logger.info(
                    "Initialized teacher worker group: %s (model: %s, backend: %s, tokenizer_policy: %s)",
                    teacher_cfg.name,
                    teacher_cfg.model_path,
                    backend,
                    tokenizer_policy,
                )

            # Validate MOPD + ppo_epochs constraint early (before training starts)
            actor_ppo_epochs = getattr(self.config.actor_rollout_ref.actor, "ppo_epochs", 1)
            critic_ppo_epochs = getattr(self.config.critic, "ppo_epochs", 1) if self.use_critic else 1
            if actor_ppo_epochs > 1 or critic_ppo_epochs > 1:
                raise ValueError(
                    f"MOPD requires ppo_epochs=1 because teacher advantages are computed "
                    f"once per rollout and become stale across PPO epochs. "
                    f"Got actor.ppo_epochs={actor_ppo_epochs}, critic.ppo_epochs={critic_ppo_epochs}. "
                    f"Set both to 1."
                )

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        mopd_manifest = self._build_mopd_manifest()
        if mopd_manifest:
            manifest_path = os.path.join(local_global_step_folder, MOPD_MANIFEST_FILENAME)
            with open(manifest_path, "w", encoding="utf-8") as manifest_file:
                json.dump(mopd_manifest, manifest_file, indent=2)

        # latest checkpointed iteration tracker (for atomic usage)
        if self._checkpoint_async_save_enabled():
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return

        self._write_checkpoint_complete_marker(local_global_step_folder)

        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        local_latest_checkpointed_iteration_tmp = f"{local_latest_checkpointed_iteration}.tmp"
        with open(local_latest_checkpointed_iteration_tmp, "w", encoding="utf-8") as f:
            f.write(str(self.global_steps))
        os.replace(local_latest_checkpointed_iteration_tmp, local_latest_checkpointed_iteration)

    @staticmethod
    def _role_async_save_enabled(role_config) -> bool:
        if role_config is None:
            return False

        checkpoint_config = role_config.get("checkpoint", None) if hasattr(role_config, "get") else None
        if checkpoint_config is None and hasattr(role_config, "checkpoint"):
            checkpoint_config = role_config.checkpoint
        if checkpoint_config is None:
            return False

        if hasattr(checkpoint_config, "get"):
            return bool(checkpoint_config.get("async_save", False))
        if hasattr(checkpoint_config, "async_save"):
            return bool(checkpoint_config.async_save)
        try:
            return bool(checkpoint_config["async_save"])
        except (KeyError, TypeError):
            return False

    def _get_async_checkpoint_roles(self) -> tuple[str, ...]:
        async_roles: list[str] = []

        actor_config = self.config.actor_rollout_ref.get("actor", None)
        if self._role_async_save_enabled(actor_config):
            async_roles.append("actor")

        critic_config = self.config.get("critic", None) if hasattr(self.config, "get") else None
        if self.use_critic and self._role_async_save_enabled(critic_config):
            async_roles.append(str(Role.Critic))

        return tuple(async_roles)

    def _checkpoint_async_save_enabled(self) -> bool:
        return bool(self._get_async_checkpoint_roles())

    def _get_checkpoint_complete_marker_path(self, global_step_folder: str) -> str:
        return os.path.join(global_step_folder, CHECKPOINT_COMPLETE_FILENAME)

    def _write_checkpoint_complete_marker(self, global_step_folder: str) -> None:
        marker_path = self._get_checkpoint_complete_marker_path(global_step_folder)
        marker_path_tmp = f"{marker_path}.tmp"
        with open(marker_path_tmp, "w", encoding="utf-8") as marker_file:
            marker_file.write(str(self.global_steps))
        os.replace(marker_path_tmp, marker_path)

    @staticmethod
    def _get_checkpoint_step(global_step_folder: str) -> int:
        return int(os.path.basename(global_step_folder).split("global_step_")[-1])

    @staticmethod
    def _checkpoint_dir_has_payload(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        with os.scandir(path) as entries:
            return any(entries)

    def _async_checkpoint_role_has_finalize_artifacts(self, role_path: str) -> bool:
        # async_save is only implemented for Megatron; finalized checkpoints always materialize
        # the distributed shard directory, HuggingFace config/tokenizer directory, and the
        # serialized transformer config under the role checkpoint root.
        dist_ckpt_path = os.path.join(role_path, "dist_ckpt")
        huggingface_path = os.path.join(role_path, "huggingface")
        transformer_config_path = os.path.join(role_path, "transformer_config.json")
        return (
            self._checkpoint_dir_has_payload(dist_ckpt_path)
            and self._checkpoint_dir_has_payload(huggingface_path)
            and os.path.isfile(transformer_config_path)
        )

    def _is_checkpoint_complete(self, global_step_folder: str, tracker_step: int | None = None) -> bool:
        if not os.path.isdir(global_step_folder):
            return False

        if os.path.exists(self._get_checkpoint_complete_marker_path(global_step_folder)):
            return True

        actor_path = os.path.join(global_step_folder, "actor")
        dataloader_path = os.path.join(global_step_folder, "data.pt")
        if not (self._checkpoint_dir_has_payload(actor_path) and os.path.isfile(dataloader_path)):
            return False

        if self.use_critic:
            critic_path = os.path.join(global_step_folder, str(Role.Critic))
            if not self._checkpoint_dir_has_payload(critic_path):
                return False

        async_roles = self._get_async_checkpoint_roles()
        if not async_roles:
            return True

        if "actor" in async_roles and not self._async_checkpoint_role_has_finalize_artifacts(actor_path):
            return False
        if self.use_critic and str(Role.Critic) in async_roles:
            critic_path = os.path.join(global_step_folder, str(Role.Critic))
            if not self._async_checkpoint_role_has_finalize_artifacts(critic_path):
                return False

        checkpoint_step = self._get_checkpoint_step(global_step_folder)
        return tracker_step == checkpoint_step and len(async_roles) == 1

    def _find_latest_complete_checkpoint(self, checkpoint_folder: str) -> str | None:
        if not checkpoint_folder or not os.path.isdir(checkpoint_folder):
            return None

        step_dirs: list[tuple[int, str]] = []
        for entry in os.scandir(checkpoint_folder):
            if not entry.is_dir() or not entry.name.startswith("global_step_"):
                continue
            try:
                step = self._get_checkpoint_step(entry.path)
            except ValueError:
                continue
            step_dirs.append((step, entry.path))

        for _step, path in sorted(step_dirs, reverse=True):
            if self._is_checkpoint_complete(path):
                return path
        return None

    def _resolve_resume_checkpoint(self, checkpoint_folder: str) -> str | None:
        tracked_checkpoint = find_latest_ckpt_path(checkpoint_folder)
        tracked_step = -1
        best_checkpoint = None

        if tracked_checkpoint is not None:
            tracked_step = self._get_checkpoint_step(tracked_checkpoint)
            if self._is_checkpoint_complete(tracked_checkpoint, tracker_step=tracked_step):
                best_checkpoint = tracked_checkpoint
            else:
                logger.warning("Ignoring incomplete tracked checkpoint at %s", tracked_checkpoint)

        scanned_checkpoint = self._find_latest_complete_checkpoint(checkpoint_folder)
        if scanned_checkpoint is None:
            return best_checkpoint

        scanned_step = self._get_checkpoint_step(scanned_checkpoint)
        if best_checkpoint is None or scanned_step > tracked_step:
            if tracked_checkpoint is None:
                logger.warning(
                    "Checkpoint tracker missing under %s; falling back to latest complete checkpoint %s",
                    checkpoint_folder,
                    scanned_checkpoint,
                )
            elif tracked_checkpoint != scanned_checkpoint:
                logger.warning(
                    "Checkpoint tracker selected %s, but newer complete checkpoint %s was found; using the newer one",
                    tracked_checkpoint,
                    scanned_checkpoint,
                )
            best_checkpoint = scanned_checkpoint

        return best_checkpoint

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = self._resolve_resume_checkpoint(checkpoint_folder)

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
                resume_tracker_step = None
                resume_checkpoint_root = os.path.dirname(global_step_folder)
                tracked_resume_checkpoint = find_latest_ckpt_path(resume_checkpoint_root)
                if tracked_resume_checkpoint == global_step_folder:
                    resume_tracker_step = self._get_checkpoint_step(tracked_resume_checkpoint)
                if not self._is_checkpoint_complete(global_step_folder, tracker_step=resume_tracker_step):
                    raise ValueError(f"Requested resume checkpoint is incomplete: {global_step_folder}")
        print(f"Load from checkpoint folder: {global_step_folder}")
        manifest_path = os.path.join(global_step_folder, MOPD_MANIFEST_FILENAME)
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as manifest_file:
                self._validate_loaded_mopd_manifest(json.load(manifest_file))
        elif self._build_mopd_manifest():
            logger.warning(
                "No %s found in %s; skipping MOPD drift validation",
                MOPD_MANIFEST_FILENAME,
                global_step_folder,
            )
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(
        self,
        batch: DataProto,
        metrics,
        logging_prefix="global_seqlen",
        keep_minibatch=False,
        dp_size: Optional[int] = None,
    ):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        if dp_size is None:
            dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_teacher_log_probs(self, batch: DataProto) -> torch.Tensor:
        """Compute teacher log probs with sub-batch routing by teacher_id.

        Groups samples by teacher_id, forwards sub-batches to the corresponding
        teacher worker group, and scatters results back into a full-batch tensor.

        Implements Fixes 2 (stacked storage), 3 (sub-batch forwarding),
        4 (correct broadcasting), 7 (select_idxs with integer tensor).

        Args:
            batch: DataProto containing "teacher_id" in non_tensor_batch
                and "responses" in batch (used for shape).

        Returns:
            torch.Tensor: Teacher log probs of shape [batch_size, response_len].

        Raises:
            ValueError: If any teacher_id in the batch has no corresponding teacher
                worker group, or if a teacher returns output with mismatched
                response length.
        """
        compatible_teacher_names = self._get_mopd_teacher_names_for_policy("compatible")
        jobs, device, response_len = self._build_mopd_teacher_jobs(batch, teacher_names=compatible_teacher_names)
        teacher_log_probs = torch.zeros(
            len(batch.non_tensor_batch["teacher_id"]),
            response_len,
            dtype=torch.float32,
            device=device,
        )
        if not jobs:
            return teacher_log_probs

        jobs_by_pool = defaultdict(list)
        for job in jobs:
            jobs_by_pool[job["pool_name"]].append(job)

        pending_by_pool = {}
        for pool_name, pool_jobs in jobs_by_pool.items():
            first_job = pool_jobs.pop(0)
            pending_by_pool[pool_name] = (
                first_job,
                self._submit_teacher_log_prob(first_job["teacher_wg"], first_job["sub_batch"]),
            )

        for pool_name in list(pending_by_pool.keys()):
            while pool_name in pending_by_pool:
                job, teacher_output = pending_by_pool[pool_name]
                self._consume_teacher_job_result(job, teacher_output, teacher_log_probs, response_len, device)
                if jobs_by_pool[pool_name]:
                    next_job = jobs_by_pool[pool_name].pop(0)
                    pending_by_pool[pool_name] = (
                        next_job,
                        self._submit_teacher_log_prob(next_job["teacher_wg"], next_job["sub_batch"]),
                    )
                else:
                    pending_by_pool.pop(pool_name)

        return teacher_log_probs

    def _compute_teacher_sequence_rewards(self, batch: DataProto) -> DataProto:
        jobs, device = self._build_mopd_sequence_teacher_jobs(batch)
        batch_size, response_len = batch.batch["responses"].shape
        teacher_seq_reward = torch.zeros(batch_size, dtype=torch.float32, device=device)
        teacher_seq_weight = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
        teacher_token_mask = torch.zeros(batch_size, response_len, dtype=torch.float32, device=device)

        if not jobs:
            return DataProto.from_single_dict(
                {
                    "teacher_seq_reward": teacher_seq_reward,
                    "teacher_seq_weight": teacher_seq_weight,
                    "teacher_token_mask": teacher_token_mask,
                }
            )

        jobs_by_pool = defaultdict(list)
        for job in jobs:
            jobs_by_pool[job["pool_name"]].append(job)

        pending_by_pool = {}
        for pool_name, pool_jobs in jobs_by_pool.items():
            first_job = pool_jobs.pop(0)
            pending_by_pool[pool_name] = (
                first_job,
                self._submit_teacher_seq_scores(first_job["teacher_wg"], first_job["sub_batch"]),
            )

        for pool_name in list(pending_by_pool.keys()):
            while pool_name in pending_by_pool:
                job, teacher_output = pending_by_pool[pool_name]
                teacher_output = self._resolve_teacher_log_prob_output(teacher_output)
                teacher_output = unpad_dataproto(teacher_output, pad_size=job["pad_size"])
                seq_scores = teacher_output.batch.get("seq_scores", None)
                if seq_scores is None:
                    seq_scores = teacher_output.batch.get("teacher_seq_reward", None)
                if seq_scores is None:
                    raise ValueError(
                        f"Teacher '{job['teacher_name']}' sequence scorer must return 'seq_scores' or "
                        "'teacher_seq_reward'."
                    )

                teacher_seq_reward[job["scatter_indices"]] = seq_scores.to(dtype=torch.float32, device=device)
                teacher_seq_weight[job["scatter_indices"]] = float(
                    getattr(job["teacher_cfg"], "seq_reward_weight", 1.0)
                )
                teacher_token_mask[job["scatter_indices"]] = batch.batch["response_mask"][job["scatter_indices"]].to(
                    dtype=torch.float32, device=device
                )

                if jobs_by_pool[pool_name]:
                    next_job = jobs_by_pool[pool_name].pop(0)
                    pending_by_pool[pool_name] = (
                        next_job,
                        self._submit_teacher_seq_scores(next_job["teacher_wg"], next_job["sub_batch"]),
                    )
                else:
                    pending_by_pool.pop(pool_name)

        return DataProto.from_single_dict(
            {
                "teacher_seq_reward": teacher_seq_reward,
                "teacher_seq_weight": teacher_seq_weight,
                "teacher_token_mask": teacher_token_mask,
            }
        )

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def _finalize_fit_resources(self, tracking_logger=None, progress_bar=None):
        if progress_bar is not None:
            progress_bar.close()

        self._shutdown_dataloader_workers()

        async_rollout_manager = getattr(self, "async_rollout_manager", None)
        if async_rollout_manager is not None:
            try:
                async_rollout_manager.shutdown()
            except Exception:
                logger.exception("Async rollout manager shutdown failed during trainer finalization")

        try:
            self.cleanup_teacher_workers()
        except Exception:
            logger.exception("Teacher worker cleanup failed during trainer finalization")

        if tracking_logger is not None:
            try:
                tracking_logger.finish()
            except Exception:
                logger.exception("Tracking finalization failed during trainer finalization")

    def _shutdown_dataloader_workers(self):
        for loader_name in ("train_dataloader", "val_dataloader"):
            dataloader = getattr(self, loader_name, None)
            if dataloader is None:
                continue

            iterator = getattr(dataloader, "_iterator", None)
            if iterator is None:
                continue

            shutdown_workers = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown_workers):
                try:
                    shutdown_workers()
                except Exception:
                    logger.exception("Failed to shutdown %s workers during trainer finalization", loader_name)
            dataloader._iterator = None

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = None
        progress_bar = None
        resources_finalized = False
        try:
            logger = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

            self.global_steps = 0

            # load checkpoint and update weights before doing anything
            self._load_checkpoint()
            self.checkpoint_manager.update_weights(self.global_steps)

            current_epoch = self.global_steps // len(self.train_dataloader)

            # perform validation before training
            # currently, we only support validation using the reward_function.
            if self.config.trainer.get("val_before_train", True):
                val_metrics = self._validate()
                assert val_metrics, f"{val_metrics=}"
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
                if self.config.trainer.get("val_only", False):
                    self._finalize_fit_resources(tracking_logger=logger, progress_bar=progress_bar)
                    resources_finalized = True
                    return

            if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
                rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
                rollout_skip.wrap_generate_sequences()

            # add tqdm
            progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

            # we start from step 1
            self.global_steps += 1
            last_val_metrics = None
            self.max_steps_duration = 0

            prev_step_profile = False
            curr_step_profile = (
                self.global_steps in self.config.global_profiler.steps
                if self.config.global_profiler.steps is not None
                else False
            )
            next_step_profile = False

            for epoch in range(current_epoch, self.config.trainer.total_epochs):
                for batch_dict in self.train_dataloader:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                    metrics = {}
                    timing_raw = {}

                    with marked_timer("start_profile", timing_raw):
                        self._start_profiling(
                            not prev_step_profile and curr_step_profile
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                    # add uid to batch
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )

                    gen_batch = self._get_gen_batch(batch)

                    # pass global_steps to trace
                    gen_batch.meta_info["global_steps"] = self.global_steps
                    gen_batch_output = gen_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )

                    is_last_step = self.global_steps >= self.total_training_steps
                    with marked_timer("step", timing_raw):
                        # generate a batch
                        with marked_timer("gen", timing_raw, color="red"):
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()

                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            with marked_timer("gen_max", timing_raw, color="purple"):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                if curr_step_profile:
                                    self.async_rollout_manager.start_profile()
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                self.checkpoint_manager.sleep_replicas()
                                if curr_step_profile:
                                    self.async_rollout_manager.stop_profile()
                                batch = batch.union(gen_baseline_output)
                                # compute reward model score on batch
                                rm_scores = None
                                if self.use_rm and "rm_scores" not in batch.batch.keys():
                                    batch_reward = self._compute_reward_colocate(batch)
                                    batch = batch.union(batch_reward)

                                # Compute or extract reward for REMAX baseline
                                reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                                keys_to_pop = set(gen_baseline_output.batch.keys())
                                if rm_scores is not None:
                                    keys_to_pop.update(rm_scores.batch.keys())
                                batch.pop(batch_keys=list(keys_to_pop))

                                batch.batch["reward_baselines"] = reward_baseline_tensor

                                del rm_scores, gen_baseline_batch, gen_baseline_output
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                        if "response_mask" not in batch.batch.keys():
                            batch.batch["response_mask"] = compute_response_mask(batch)
                        # Balance the number of valid tokens across DP ranks.
                        # NOTE: This usually changes the order of data in the `batch`,
                        # which won't affect the advantage calculation (since it's based on uid),
                        # but might affect the loss calculation (due to the change of mini-batching).
                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        # compute global_valid tokens
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                        # get images_seqlens
                        images_seqlens_all = []
                        for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                            if "image_grid_thw" not in multi_modal_input.keys():
                                continue
                            images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                        batch.meta_info["images_seqlens"] = images_seqlens_all
                        with marked_timer("reward", timing_raw, color="yellow"):
                            # compute reward model score
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # extract reward_tensor and reward_extra_infos_dict for training
                            reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                        # Operating Mode Selection:
                        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                        # - Decoupled mode: Recomputes old_log_probs as proximal anchor
                        #   (3 policies: π_rollout, π_old, π_θ)
                        #   Note: π_old computed once per data batch and serves as a stable
                        #   reference during mini-batch updates.
                        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get(
                            "bypass_mode", False
                        )
                        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                            from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                            apply_bypass_mode(
                                batch=batch,
                                rollout_corr_config=rollout_corr_config,
                                policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                            )
                        else:  # Recompute old_log_probs
                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                actor_config = self.config.actor_rollout_ref.actor
                                entropy_agg = agg_loss(
                                    loss_mat=entropys,
                                    loss_mask=response_masks,
                                    loss_agg_mode=actor_config.loss_agg_mode,
                                    loss_scale_factor=actor_config.loss_scale_factor,
                                )
                                old_log_prob_metrics = {
                                    "actor/entropy": entropy_agg.detach().item(),
                                    "perf/mfu/actor_infer": old_log_prob_mfu,
                                }
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                                if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                    router_mode = getattr(
                                        self.config.actor_rollout_ref.actor.router_replay, "mode", "disabled"
                                    )
                                    if router_mode == "R2":
                                        batch.batch.pop("routed_experts")
                                    else:
                                        old_log_prob.batch.pop("routed_experts")
                                batch = batch.union(old_log_prob)
                                if "rollout_log_probs" in batch.batch.keys():
                                    # TODO: we may want to add diff of probs too.
                                    from verl.utils.debug.metrics import calculate_debug_metrics

                                    metrics.update(calculate_debug_metrics(batch))

                        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                if self.teacher_wgs:
                                    # MOPD: compute teacher log probs via sub-batch routing
                                    teacher_log_prob = self._compute_teacher_log_probs(batch)
                                    batch.batch["teacher_log_prob"] = teacher_log_prob
                                    batch = batch.union(self._compute_teacher_sequence_rewards(batch))
                                    batch.batch["lambda_val"] = self._build_mopd_lambda_tensor(batch)
                                    if getattr(self.config.algorithm.mopd, "use_base_normalization", False):
                                        base_log_prob = self._compute_base_log_prob(batch)
                                        batch = batch.union(base_log_prob)
                                    # Also compute ref_log_prob if KL loss or KL-in-reward is used alongside MOPD
                                    if (
                                        self.config.actor_rollout_ref.actor.use_kl_loss
                                        or self.config.algorithm.use_kl_in_reward
                                    ):
                                        ref_log_prob = self._compute_ref_log_prob(batch)
                                        batch = batch.union(ref_log_prob)
                                else:
                                    # Standard: compute ref log prob
                                    ref_log_prob = self._compute_ref_log_prob(batch)
                                    batch = batch.union(ref_log_prob)

                        # compute values
                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                values = self._compute_values(batch)
                                batch = batch.union(values)

                        with marked_timer("adv", timing_raw, color="brown"):
                            # we combine with rule-based rm
                            reward_extra_infos_dict: dict[str, list]
                            batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                )

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            # Compute rollout correction: IS weights, rejection sampling, and metrics
                            # Only runs in decoupled mode (computes once per batch using stable π_old)
                            # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                            if (
                                rollout_corr_config is not None
                                and "rollout_log_probs" in batch.batch
                                and not bypass_recomputing_logprobs  # Only in decoupled mode
                            ):
                                from verl.trainer.ppo.rollout_corr_helper import (
                                    compute_rollout_correction_and_add_to_batch,
                                )

                                # Compute IS weights, apply rejection sampling, compute metrics
                                batch, is_metrics = compute_rollout_correction_and_add_to_batch(
                                    batch, rollout_corr_config
                                )
                                # IS and off-policy metrics already have rollout_corr/ prefix
                                metrics.update(is_metrics)

                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                            # Extract MOPD IS metrics if present (set by compute_mopd_advantage)
                            if batch.meta_info.get("metrics"):
                                metrics.update(batch.meta_info.pop("metrics"))
                            if self.teacher_wgs:
                                self._record_mopd_teacher_metrics(batch, metrics)

                        # update critic
                        if self.use_critic:
                            with marked_timer("update_critic", timing_raw, color="pink"):
                                critic_output = self._update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                        # implement critic warmup
                        if self.config.trainer.critic_warmup <= self.global_steps:
                            # update actor
                            with marked_timer("update_actor", timing_raw, color="red"):
                                actor_output = self._update_actor(batch)

                            # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                            esi_close_to_expiration = should_save_ckpt_esi(
                                max_steps_duration=self.max_steps_duration,
                                redundant_time=self.config.trainer.esi_redundant_time,
                            )
                            # Check if the conditions for saving a checkpoint are met.
                            # The conditions include a mandatory condition (1) and
                            # one of the following optional conditions (2/3/4):
                            # 1. The save frequency is set to a positive value.
                            # 2. It's the last training step.
                            # 3. The current step number is a multiple of the save frequency.
                            # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                            if self.config.trainer.save_freq > 0 and (
                                is_last_step
                                or self.global_steps % self.config.trainer.save_freq == 0
                                or esi_close_to_expiration
                            ):
                                if esi_close_to_expiration:
                                    print("Force saving checkpoint: ESI instance expiration approaching.")
                                with marked_timer("save_checkpoint", timing_raw, color="green"):
                                    self._save_checkpoint()

                            # update weights from trainer to rollout
                            with marked_timer("update_weights", timing_raw, color="red"):
                                self.checkpoint_manager.update_weights(self.global_steps)

                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                        # Log rollout generations if enabled
                        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                        if rollout_data_dir:
                            self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                    # validate
                    if self.config.trainer.test_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    with marked_timer("stop_profile", timing_raw):
                        next_step_profile = (
                            self.global_steps + 1 in self.config.global_profiler.steps
                            if self.config.global_profiler.steps is not None
                            else False
                        )
                        self._stop_profiling(
                            curr_step_profile and not next_step_profile
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )
                        prev_step_profile = curr_step_profile
                        curr_step_profile = next_step_profile

                    steps_duration = timing_raw["step"]
                    self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                    # training metrics
                    metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                        }
                    )
                    # collect metrics
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    # TODO: implement actual tflpo and theoretical tflpo
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                    # compute variance proxy metrics
                    gradient_norm = metrics.get("actor/grad_norm", None)
                    metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                    # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                    # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                    if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                        self.train_dataloader.sampler.update(batch=batch)

                    # TODO: make a canonical logger that supports various backend
                    logger.log(data=metrics, step=self.global_steps)

                    progress_bar.update(1)
                    self.global_steps += 1

                    if (
                        hasattr(self.config.actor_rollout_ref.actor, "profiler")
                        and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                    ):
                        self.actor_rollout_wg.dump_memory_snapshot(
                            tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                        )

                    if is_last_step:
                        if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                            self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        self._finalize_fit_resources(tracking_logger=logger, progress_bar=progress_bar)
                        resources_finalized = True
                        return

                    # this is experimental and may be changed/removed in the future
                    # in favor of a general-purpose data buffer pool
                    if hasattr(self.train_dataset, "on_batch_end"):
                        # The dataset may be changed after each training batch
                        self.train_dataset.on_batch_end(batch=batch)

            self._finalize_fit_resources(tracking_logger=logger, progress_bar=progress_bar)
            resources_finalized = True
        finally:
            if not resources_finalized:
                self._finalize_fit_resources(tracking_logger=logger, progress_bar=progress_bar)

    def cleanup_teacher_workers(self):
        """Clean up teacher worker groups for MOPD.

        Releases references to teacher RayWorkerGroups so Ray can garbage-collect
        the underlying actors and free GPU memory. Called automatically at the end
        of fit() on all exit paths, but can also be called manually.
        """
        if self.teacher_wgs:
            logger.info("Cleaning up %d teacher worker groups", len(self.teacher_wgs))
            for teacher_name in list(self.teacher_wgs.keys()):
                logger.debug("Releasing teacher worker group: %s", teacher_name)
            # Clear all references; Ray GC will reclaim actor resources
            self.teacher_wgs.clear()
            logger.info("Teacher worker cleanup complete")
        if self.base_policy_wg is not None:
            logger.info("Releasing shared base worker group for ExOPD cleanup")
            self.base_policy_wg = None
