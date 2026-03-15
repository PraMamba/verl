import logging
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.workers.actor import dp_actor


def _make_minibatch(batch_size: int = 4, seq_len: int = 8) -> DataProto:
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    input_ids = torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len)
    return DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )


def test_log_dynamic_bsz_update_diagnostics_logs_rank0_summary(monkeypatch, caplog):
    mini_batch = _make_minibatch()
    micro_batches = [mini_batch[:2], mini_batch[2:]]

    monkeypatch.setattr(dp_actor.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dp_actor.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(dp_actor.dist, "get_world_size", lambda group=None: 4)

    def _fake_all_gather_object(output, obj, group=None):
        assert obj == {
            "rank": 0,
            "mini_batch_idx": 1,
            "num_micro_batches": 2,
            "local_batch_size": 4,
            "local_token_sum": 32,
        }
        assert group is dp_actor.dist.group.WORLD
        payloads = [
            {"rank": 0, "mini_batch_idx": 1, "num_micro_batches": 2, "local_batch_size": 4, "local_token_sum": 32},
            {"rank": 1, "mini_batch_idx": 1, "num_micro_batches": 2, "local_batch_size": 4, "local_token_sum": 31},
            {"rank": 2, "mini_batch_idx": 1, "num_micro_batches": 3, "local_batch_size": 4, "local_token_sum": 36},
            {"rank": 3, "mini_batch_idx": 1, "num_micro_batches": 2, "local_batch_size": 4, "local_token_sum": 33},
        ]
        for idx, payload in enumerate(payloads):
            output[idx] = payload

    monkeypatch.setattr(dp_actor.dist, "all_gather_object", _fake_all_gather_object)

    with caplog.at_level(logging.WARNING):
        dp_actor._log_dynamic_bsz_update_diagnostics(mini_batch=mini_batch, micro_batches=micro_batches, batch_idx=1)

    assert "dynamic_bsz_update_diagnostics" in caplog.text
    assert "mini_batch_idx=1" in caplog.text
    assert "num_micro_batches_by_rank={0: 2, 1: 2, 2: 3, 3: 2}" in caplog.text
    assert "mismatch=True" in caplog.text


def test_log_dynamic_bsz_update_diagnostics_is_rank0_only(monkeypatch, caplog):
    mini_batch = _make_minibatch()
    micro_batches = [mini_batch]

    monkeypatch.setattr(dp_actor.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dp_actor.dist, "get_rank", lambda group=None: 2)
    monkeypatch.setattr(dp_actor.dist, "get_world_size", lambda group=None: 4)

    def _fake_all_gather_object(output, obj, group=None):
        assert obj == {
            "rank": 2,
            "mini_batch_idx": 0,
            "num_micro_batches": 1,
            "local_batch_size": 4,
            "local_token_sum": 32,
        }
        assert group is dp_actor.dist.group.WORLD
        payloads = [
            {"rank": 0, "mini_batch_idx": 0, "num_micro_batches": 1, "local_batch_size": 4, "local_token_sum": 32},
            {"rank": 1, "mini_batch_idx": 0, "num_micro_batches": 1, "local_batch_size": 4, "local_token_sum": 32},
            {"rank": 2, "mini_batch_idx": 0, "num_micro_batches": 1, "local_batch_size": 4, "local_token_sum": 32},
            {"rank": 3, "mini_batch_idx": 0, "num_micro_batches": 1, "local_batch_size": 4, "local_token_sum": 32},
        ]
        for idx, payload in enumerate(payloads):
            output[idx] = payload

    monkeypatch.setattr(dp_actor.dist, "all_gather_object", _fake_all_gather_object)

    with caplog.at_level(logging.WARNING):
        dp_actor._log_dynamic_bsz_update_diagnostics(mini_batch=mini_batch, micro_batches=micro_batches, batch_idx=0)

    assert caplog.text == ""


def test_log_dynamic_bsz_update_diagnostics_returns_early_without_dist(monkeypatch, caplog):
    mini_batch = _make_minibatch()
    micro_batches = [mini_batch]

    monkeypatch.setattr(dp_actor.dist, "is_initialized", lambda: False)

    with caplog.at_level(logging.WARNING):
        dp_actor._log_dynamic_bsz_update_diagnostics(mini_batch=mini_batch, micro_batches=micro_batches, batch_idx=0)

    assert caplog.text == ""


def test_log_dynamic_bsz_compute_log_prob_diagnostics_logs_rank0_summary(monkeypatch, caplog):
    batch = _make_minibatch()
    micro_batches = [batch[:2], batch[2:]]

    monkeypatch.setattr(dp_actor.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dp_actor.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(dp_actor.dist, "get_world_size", lambda group=None: 4)

    def _fake_all_gather_object(output, obj, group=None):
        assert obj == {
            "rank": 0,
            "num_micro_batches": 2,
            "local_batch_size": 4,
            "local_token_sum": 32,
        }
        assert group is dp_actor.dist.group.WORLD
        payloads = [
            {"rank": 0, "num_micro_batches": 2, "local_batch_size": 4, "local_token_sum": 32},
            {"rank": 1, "num_micro_batches": 2, "local_batch_size": 4, "local_token_sum": 31},
            {"rank": 2, "num_micro_batches": 3, "local_batch_size": 4, "local_token_sum": 36},
            {"rank": 3, "num_micro_batches": 2, "local_batch_size": 4, "local_token_sum": 33},
        ]
        for idx, payload in enumerate(payloads):
            output[idx] = payload

    monkeypatch.setattr(dp_actor.dist, "all_gather_object", _fake_all_gather_object)

    with caplog.at_level(logging.WARNING):
        dp_actor._log_dynamic_bsz_compute_log_prob_diagnostics(batch=batch, micro_batches=micro_batches)

    assert "dynamic_bsz_compute_log_prob_diagnostics" in caplog.text
    assert "num_micro_batches_by_rank={0: 2, 1: 2, 2: 3, 3: 2}" in caplog.text
    assert "mismatch=True" in caplog.text


def test_log_dynamic_bsz_compute_log_prob_diagnostics_is_rank0_only(monkeypatch, caplog):
    batch = _make_minibatch()
    micro_batches = [batch]

    monkeypatch.setattr(dp_actor.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dp_actor.dist, "get_rank", lambda group=None: 2)
    monkeypatch.setattr(dp_actor.dist, "get_world_size", lambda group=None: 4)

    def _fake_all_gather_object(output, obj, group=None):
        assert obj == {
            "rank": 2,
            "num_micro_batches": 1,
            "local_batch_size": 4,
            "local_token_sum": 32,
        }
        assert group is dp_actor.dist.group.WORLD
        payloads = [
            {"rank": 0, "num_micro_batches": 1, "local_batch_size": 4, "local_token_sum": 32},
            {"rank": 1, "num_micro_batches": 1, "local_batch_size": 4, "local_token_sum": 32},
            {"rank": 2, "num_micro_batches": 1, "local_batch_size": 4, "local_token_sum": 32},
            {"rank": 3, "num_micro_batches": 1, "local_batch_size": 4, "local_token_sum": 32},
        ]
        for idx, payload in enumerate(payloads):
            output[idx] = payload

    monkeypatch.setattr(dp_actor.dist, "all_gather_object", _fake_all_gather_object)

    with caplog.at_level(logging.WARNING):
        dp_actor._log_dynamic_bsz_compute_log_prob_diagnostics(batch=batch, micro_batches=micro_batches)

    assert caplog.text == ""


def test_log_dynamic_bsz_update_diagnostics_uses_explicit_sync_group(monkeypatch, caplog):
    mini_batch = _make_minibatch()
    micro_batches = [mini_batch[:2], mini_batch[2:]]
    sync_group = object()

    monkeypatch.setattr(dp_actor.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dp_actor.dist, "get_rank", lambda group=None: 0 if group is sync_group else 7)
    monkeypatch.setattr(dp_actor.dist, "get_world_size", lambda group=None: 2 if group is sync_group else 8)

    def _fake_all_gather_object(output, obj, group=None):
        assert group is sync_group
        payloads = [
            {"rank": 4, "mini_batch_idx": 1, "num_micro_batches": 2, "local_batch_size": 4, "local_token_sum": 32},
            {"rank": 6, "mini_batch_idx": 1, "num_micro_batches": 2, "local_batch_size": 4, "local_token_sum": 31},
        ]
        for idx, payload in enumerate(payloads):
            output[idx] = payload

    monkeypatch.setattr(dp_actor.dist, "all_gather_object", _fake_all_gather_object)

    with caplog.at_level(logging.WARNING):
        dp_actor._log_dynamic_bsz_update_diagnostics(
            mini_batch=mini_batch,
            micro_batches=micro_batches,
            batch_idx=1,
            sync_group=sync_group,
        )

    assert "num_micro_batches_by_rank={4: 2, 6: 2}" in caplog.text
    assert "mismatch=False" in caplog.text


def _make_actor_for_dynamic_bsz_sync_tests(sync_group=None):
    actor = object.__new__(dp_actor.DataParallelPPOActor)
    actor.config = OmegaConf.create(
        {
            "calculate_sum_pi_squared": False,
            "ppo_epochs": 1,
            "ppo_mini_batch_size": 4,
            "ppo_max_token_len_per_gpu": 16,
            "use_dynamic_bsz": True,
            "use_kl_loss": False,
            "entropy_coeff": 0.0,
        }
    )
    actor.actor_module = SimpleNamespace(eval=lambda: None, train=lambda: None)
    actor.dynamic_bsz_sync_group = sync_group
    actor.use_prefix_grouper = False
    actor.ulysses_sequence_parallel_size = 1
    return actor


def _make_actor_init_config():
    return OmegaConf.create(
        {
            "use_remove_padding": False,
            "use_fused_kernels": False,
            "ulysses_sequence_parallel_size": 1,
            "use_dynamic_bsz": True,
            "use_prefix_grouper": False,
            "entropy_from_logits_with_chunking": False,
            "use_torch_compile": False,
            "fsdp_config": {"dtype": "bfloat16"},
            "calculate_sum_pi_squared": False,
        }
    )


def _construct_actor_via_init(monkeypatch, *, sync_group=None, world_group=None):
    config = _make_actor_init_config()
    actor_module = SimpleNamespace()

    monkeypatch.setattr(dp_actor.BasePPOActor, "__init__", lambda self, cfg: setattr(self, "config", cfg))
    monkeypatch.setattr(dp_actor.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(dp_actor.dist, "is_initialized", lambda: True)
    if world_group is not None:
        monkeypatch.setattr(dp_actor.dist, "group", SimpleNamespace(WORLD=world_group))

    return dp_actor.DataParallelPPOActor(
        config=config,
        actor_module=actor_module,
        dynamic_bsz_sync_group=sync_group,
    )


def _make_compute_log_prob_batch(batch_size: int = 4, seq_len: int = 8, response_len: int = 3) -> DataProto:
    input_ids = torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    responses = input_ids[:, -response_len:]
    return DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
        },
        meta_info={
            "micro_batch_size": batch_size,
            "temperature": 1.0,
            "use_dynamic_bsz": True,
            "max_token_len": 16,
        },
    )


def _make_update_policy_batch(batch_size: int = 4, seq_len: int = 8, response_len: int = 3) -> DataProto:
    input_ids = torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    responses = input_ids[:, -response_len:]
    response_mask = torch.ones(batch_size, response_len, dtype=torch.long)
    old_log_probs = torch.zeros(batch_size, response_len, dtype=torch.float32)
    advantages = torch.ones(batch_size, response_len, dtype=torch.float32)
    return DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
        },
        meta_info={"temperature": 1.0},
    )


def test_actor_init_defaults_dynamic_bsz_sync_group_to_world(monkeypatch):
    world_group = object()
    actor = _construct_actor_via_init(monkeypatch, world_group=world_group)

    assert actor.dynamic_bsz_sync_group is world_group


def test_compute_log_prob_dynamic_bsz_syncs_micro_batch_count_on_world_group(monkeypatch):
    world_group = object()
    actor = _make_actor_for_dynamic_bsz_sync_tests(sync_group=world_group)
    batch = _make_compute_log_prob_batch()
    captured = {}

    class _StopAfterCapture(RuntimeError):
        pass

    def _fake_prepare_dynamic_batch(data, **kwargs):
        captured["dp_group"] = kwargs["dp_group"]
        captured["max_token_len"] = kwargs["max_token_len"]
        raise _StopAfterCapture

    monkeypatch.setattr(dp_actor, "prepare_dynamic_batch", _fake_prepare_dynamic_batch)

    try:
        actor.compute_log_prob(batch, calculate_entropy=False)
    except _StopAfterCapture:
        pass
    else:
        raise AssertionError("expected compute_log_prob to stop after prepare_dynamic_batch capture")

    assert captured["dp_group"] is world_group
    assert captured["max_token_len"] == 16


def test_compute_log_prob_dynamic_bsz_prefers_explicit_sync_group(monkeypatch):
    sync_group = object()
    actor = _make_actor_for_dynamic_bsz_sync_tests(sync_group=sync_group)
    batch = _make_compute_log_prob_batch()
    captured = {}

    class _StopAfterCapture(RuntimeError):
        pass

    def _fake_prepare_dynamic_batch(data, **kwargs):
        captured["dp_group"] = kwargs["dp_group"]
        raise _StopAfterCapture

    monkeypatch.setattr(dp_actor, "prepare_dynamic_batch", _fake_prepare_dynamic_batch)

    try:
        actor.compute_log_prob(batch, calculate_entropy=False)
    except _StopAfterCapture:
        pass
    else:
        raise AssertionError("expected compute_log_prob to stop after prepare_dynamic_batch capture")

    assert captured["dp_group"] is sync_group


def test_update_policy_dynamic_bsz_syncs_micro_batch_count_on_world_group(monkeypatch):
    world_group = object()
    actor = _make_actor_for_dynamic_bsz_sync_tests(sync_group=world_group)
    batch = _make_update_policy_batch()
    captured = {}

    class _StopAfterCapture(RuntimeError):
        pass

    def _fake_prepare_dynamic_batch(data, **kwargs):
        captured["dp_group"] = kwargs["dp_group"]
        captured["max_token_len"] = kwargs["max_token_len"]
        raise _StopAfterCapture

    monkeypatch.setattr(dp_actor, "prepare_dynamic_batch", _fake_prepare_dynamic_batch)

    try:
        actor.update_policy(batch)
    except _StopAfterCapture:
        pass
    else:
        raise AssertionError("expected update_policy to stop after prepare_dynamic_batch capture")

    assert captured["dp_group"] is world_group
    assert captured["max_token_len"] == 16


def test_update_policy_dynamic_bsz_prefers_explicit_sync_group(monkeypatch):
    sync_group = object()
    actor = _make_actor_for_dynamic_bsz_sync_tests(sync_group=sync_group)
    batch = _make_update_policy_batch()
    captured = {}

    class _StopAfterCapture(RuntimeError):
        pass

    def _fake_prepare_dynamic_batch(data, **kwargs):
        captured["dp_group"] = kwargs["dp_group"]
        raise _StopAfterCapture

    monkeypatch.setattr(dp_actor, "prepare_dynamic_batch", _fake_prepare_dynamic_batch)

    try:
        actor.update_policy(batch)
    except _StopAfterCapture:
        pass
    else:
        raise AssertionError("expected update_policy to stop after prepare_dynamic_batch capture")

    assert captured["dp_group"] is sync_group
