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

"""Dedicated teacher-only inference workers for MOPD P2 backends."""

import logging
import os
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_device_name
from verl.utils.tokenizer import hf_tokenizer, normalize_token_ids
from verl.utils.torch_functional import log_probs_from_logits_response, logprobs_from_logits

logger = logging.getLogger(__file__)


class HFQuantizedTeacherWorker(Worker):
    """Inference-only teacher worker for rank-local HF int8/4bit models."""

    def __init__(self, config, role: str = "teacher", **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.role = role
        self.model = None
        self.tokenizer = None
        self.device_name = get_device_name()
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.device_name == "cuda" and torch.cuda.is_available():
            self.device = torch.device(self.device_name, self.local_rank)
        else:
            self.device = torch.device("cpu")
        self._register_dispatch_collect_info("ref", dp_rank=self.rank, is_collect=True)

    def _get_teacher_cfg(self):
        teacher_cfg = getattr(self.config, "teacher", None)
        if teacher_cfg is None:
            raise ValueError("Quantized teacher worker config must include a 'teacher' section")
        return teacher_cfg

    def _get_teacher_backend(self) -> str:
        return getattr(self._get_teacher_cfg(), "backend", "legacy_ref")

    def _get_tokenizer_path(self) -> str:
        return getattr(self.config.model, "tokenizer_path", None) or self.config.model.path

    def _get_trust_remote_code(self) -> bool:
        return bool(getattr(self.config.model, "trust_remote_code", False))

    def _get_micro_batch_size(self) -> int:
        teacher_cfg = self._get_teacher_cfg()
        micro_batch_size = int(getattr(teacher_cfg, "log_prob_micro_batch_size", 0) or 0)
        return micro_batch_size if micro_batch_size > 0 else 1

    def _get_rank_local_device_map(self) -> dict[str, str] | None:
        if self.device.type == "cpu":
            return None
        return {"": f"{self.device.type}:{self.device.index}"}

    def _build_quantization_config(self):
        backend = self._get_teacher_backend()
        if backend not in {"hf_int8", "hf_4bit"}:
            raise ValueError(f"Unsupported quantized teacher backend: {backend}")

        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ValueError(
                "Quantized teacher backend requires transformers BitsAndBytesConfig support. "
                "Install a transformers build with bitsandbytes integration."
            ) from exc

        if backend == "hf_int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        return BitsAndBytesConfig(load_in_4bit=True)

    def _normalize_messages(self, messages: Any) -> list[dict[str, Any]]:
        if isinstance(messages, np.ndarray):
            messages = messages.tolist()
        if not isinstance(messages, list):
            raise TypeError(f"raw_prompt must be list-like, got {type(messages).__name__}")

        normalized_messages = []
        for message in messages:
            if isinstance(message, np.ndarray):
                message = message.tolist()
            if hasattr(message, "items"):
                normalized_messages.append(dict(message))
            else:
                normalized_messages.append(message)
        return normalized_messages

    def _pad_token_sequences(self, token_sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        max_len = max(len(ids) for ids in token_sequences)
        input_ids = torch.full((len(token_sequences), max_len), pad_token_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((len(token_sequences), max_len), dtype=torch.long, device=self.device)

        padding_side = getattr(self.tokenizer, "padding_side", "left")
        for idx, token_ids in enumerate(token_sequences):
            token_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device)
            seq_len = len(token_ids)
            if padding_side == "right":
                input_ids[idx, :seq_len] = token_tensor
                attention_mask[idx, :seq_len] = 1
            else:
                input_ids[idx, -seq_len:] = token_tensor
                attention_mask[idx, -seq_len:] = 1
        return input_ids, attention_mask

    def _prepare_sequence_batch(
        self,
        raw_prompts: list[Any],
        response_texts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int], list[int]]:
        token_sequences: list[list[int]] = []
        sequence_lengths: list[int] = []
        prompt_lengths: list[int] = []
        response_lengths: list[int] = []

        for raw_prompt, response_text in zip(raw_prompts, response_texts, strict=True):
            prompt_messages = self._normalize_messages(raw_prompt)
            full_messages = prompt_messages + [{"role": "assistant", "content": response_text}]

            prompt_token_ids = normalize_token_ids(
                self.tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
            )
            full_token_ids = normalize_token_ids(
                self.tokenizer.apply_chat_template(full_messages, tokenize=True, add_generation_prompt=False)
            )
            response_length = len(full_token_ids) - len(prompt_token_ids)
            if response_length <= 0:
                raise ValueError("Teacher sequence reward requires positive response length after retokenization")

            token_sequences.append(full_token_ids)
            sequence_lengths.append(len(full_token_ids))
            prompt_lengths.append(len(prompt_token_ids))
            response_lengths.append(response_length)

        input_ids, attention_mask = self._pad_token_sequences(token_sequences)
        return input_ids, attention_mask, sequence_lengths, prompt_lengths, response_lengths

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from transformers import AutoModelForCausalLM

        quantization_config = self._build_quantization_config()
        model_kwargs = {
            "trust_remote_code": self._get_trust_remote_code(),
            "quantization_config": quantization_config,
        }
        device_map = self._get_rank_local_device_map()
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(self.config.model.path, **model_kwargs)
        self.model.eval()

        self.tokenizer = hf_tokenizer(
            self._get_tokenizer_path(),
            trust_remote_code=self._get_trust_remote_code(),
        )
        custom_chat_template = getattr(self.config.model, "custom_chat_template", None)
        if custom_chat_template is not None:
            self.tokenizer.chat_template = custom_chat_template

    def _compute_ref_log_prob_impl(self, data: DataProto) -> DataProto:
        if self.model is None:
            raise ValueError("Teacher model is not initialized. Call init_model() before compute_ref_log_prob().")

        batch_size = len(data)
        response_len = data.batch["responses"].shape[1]
        micro_batch_size = self._get_micro_batch_size()
        log_prob_chunks = []

        with torch.no_grad():
            for start in range(0, batch_size, micro_batch_size):
                stop = min(start + micro_batch_size, batch_size)
                input_ids = data.batch["input_ids"][start:stop].to(self.device)
                attention_mask = data.batch["attention_mask"][start:stop].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                log_probs = log_probs_from_logits_response(input_ids, outputs.logits.float(), response_len)
                log_prob_chunks.append(log_probs.float().cpu())

        return DataProto.from_single_dict({"ref_log_prob": torch.cat(log_prob_chunks, dim=0)})

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    def compute_ref_log_prob(self, data: DataProto):
        return self._compute_ref_log_prob_impl(data)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"), blocking=False)
    def compute_ref_log_prob_async(self, data: DataProto):
        return self._compute_ref_log_prob_impl(data)

    def _compute_seq_scores_impl(self, data: DataProto) -> DataProto:
        if self.model is None or self.tokenizer is None:
            raise ValueError(
                "Teacher model/tokenizer is not initialized. "
                "Call init_model() before compute_seq_scores()."
            )

        raw_prompts = list(data.non_tensor_batch["raw_prompt"])
        response_texts = list(data.non_tensor_batch["response_text"])
        input_ids, attention_mask, sequence_lengths, prompt_lengths, response_lengths = self._prepare_sequence_batch(
            raw_prompts=raw_prompts,
            response_texts=response_texts,
        )

        seq_scores = []
        micro_batch_size = self._get_micro_batch_size()
        padding_side = getattr(self.tokenizer, "padding_side", "left")
        batch_size = len(raw_prompts)

        with torch.no_grad():
            for start in range(0, batch_size, micro_batch_size):
                stop = min(start + micro_batch_size, batch_size)
                micro_input_ids = input_ids[start:stop]
                micro_attention_mask = attention_mask[start:stop]

                outputs = self.model(input_ids=micro_input_ids, attention_mask=micro_attention_mask)
                full_log_probs = logprobs_from_logits(outputs.logits[:, :-1].float(), micro_input_ids[:, 1:])

                max_seq_len = micro_input_ids.shape[1]
                for idx, (seq_len, prompt_len, response_len) in enumerate(
                    zip(
                        sequence_lengths[start:stop],
                        prompt_lengths[start:stop],
                        response_lengths[start:stop],
                        strict=True,
                    )
                ):
                    pad_offset = 0 if padding_side == "right" else max_seq_len - seq_len
                    response_start = pad_offset + prompt_len - 1
                    response_end = response_start + response_len
                    seq_scores.append(full_log_probs[idx, response_start:response_end].mean())

        return DataProto.from_single_dict({"seq_scores": torch.stack(seq_scores).float().cpu()})

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    def compute_seq_scores(self, data: DataProto):
        return self._compute_seq_scores_impl(data)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"), blocking=False)
    def compute_seq_scores_async(self, data: DataProto):
        return self._compute_seq_scores_impl(data)
