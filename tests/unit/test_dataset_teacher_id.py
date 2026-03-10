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

"""Tests for teacher_id extraction in RLHFDataset (MOPD Task 4)."""

import json
import os
import tempfile

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn


def _create_test_jsonl(path, records):
    """Write a list of dicts as JSONL."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class _FakeTokenizer:
    """Minimal tokenizer stub that satisfies RLHFDataset construction."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        # Return a short token list so prompts pass the length filter
        return [1, 2, 3]


def _make_dataset(tmp_dir, records, teacher_id_field=None):
    """Helper: write records to JSONL and build an RLHFDataset."""
    jsonl_path = os.path.join(tmp_dir, "data.jsonl")
    _create_test_jsonl(jsonl_path, records)

    cfg = {
        "max_prompt_length": 1024,
        "filter_overlong_prompts": True,
        "cache_dir": tmp_dir,
        "prompt_key": "prompt",
    }
    if teacher_id_field is not None:
        cfg["teacher_id_field"] = teacher_id_field

    config = OmegaConf.create(cfg)
    tokenizer = _FakeTokenizer()
    return RLHFDataset(data_files=[jsonl_path], tokenizer=tokenizer, config=config)


# -- Test data ----------------------------------------------------------------
_RECORDS_WITH_DOMAIN = [
    {
        "prompt": [{"role": "user", "content": "Solve x+1=2"}],
        "data_source": "math_ds",
        "domain": "math",
    },
    {
        "prompt": [{"role": "user", "content": "Write fizzbuzz"}],
        "data_source": "code_ds",
        "domain": "code",
    },
    {
        "prompt": [{"role": "user", "content": "What is gravity?"}],
        "data_source": "science_ds",
        # "domain" intentionally missing → should fallback to "default"
    },
]


# -- Tests --------------------------------------------------------------------


def test_teacher_id_included_when_configured():
    """teacher_id should appear in __getitem__ output when teacher_id_field is set."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ds = _make_dataset(tmp_dir, _RECORDS_WITH_DOMAIN, teacher_id_field="domain")

        item0 = ds[0]
        assert "teacher_id" in item0
        assert item0["teacher_id"] == "math"

        item1 = ds[1]
        assert item1["teacher_id"] == "code"


def test_teacher_id_defaults_when_field_missing():
    """If the record lacks the configured field, teacher_id should be 'default'."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ds = _make_dataset(tmp_dir, _RECORDS_WITH_DOMAIN, teacher_id_field="domain")

        item2 = ds[2]
        assert item2["teacher_id"] == "default"


def test_no_teacher_id_when_not_configured():
    """Backward compatibility: no teacher_id key when teacher_id_field is not set."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ds = _make_dataset(tmp_dir, _RECORDS_WITH_DOMAIN, teacher_id_field=None)

        item0 = ds[0]
        assert "teacher_id" not in item0


def test_collate_fn_puts_teacher_id_in_non_tensors():
    """collate_fn should put string teacher_id into non-tensor batch (numpy array)."""
    data_list = [
        {"dummy_tensor": torch.tensor([0], dtype=torch.uint8), "teacher_id": "math"},
        {"dummy_tensor": torch.tensor([0], dtype=torch.uint8), "teacher_id": "code"},
        {"dummy_tensor": torch.tensor([0], dtype=torch.uint8), "teacher_id": "math"},
    ]
    batch = collate_fn(data_list)

    # teacher_id should be a numpy array of strings
    assert "teacher_id" in batch
    assert isinstance(batch["teacher_id"], np.ndarray)
    assert batch["teacher_id"].tolist() == ["math", "code", "math"]

    # dummy_tensor should be a stacked tensor
    assert isinstance(batch["dummy_tensor"], torch.Tensor)
    assert batch["dummy_tensor"].shape == (3, 1)


def test_teacher_id_end_to_end_with_collate():
    """Full pipeline: dataset __getitem__ → collate_fn → teacher_id in batch."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ds = _make_dataset(tmp_dir, _RECORDS_WITH_DOMAIN[:2], teacher_id_field="domain")

        items = [ds[i] for i in range(len(ds))]
        batch = collate_fn(items)

        assert "teacher_id" in batch
        assert batch["teacher_id"].tolist() == ["math", "code"]
