import asyncio
import os
from types import SimpleNamespace

import pytest

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.tracking import Tracking


class _FinishRecorder:
    def __init__(self):
        self.calls = []

    def finish(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_tracking_finish_is_idempotent():
    tracker = object.__new__(Tracking)
    tracker.logger = {
        "wandb": _FinishRecorder(),
        "swanlab": _FinishRecorder(),
        "tensorboard": _FinishRecorder(),
    }
    tracker._finished = False

    tracker.finish()
    tracker.finish()
    tracker.__del__()

    assert tracker._finished is True
    assert tracker.logger["wandb"].calls == [((), {"exit_code": 0})]
    assert tracker.logger["swanlab"].calls == [((), {})]
    assert tracker.logger["tensorboard"].calls == [((), {})]


def test_install_shared_memory_resource_tracker_bypass_only_skips_shared_memory(monkeypatch):
    from multiprocessing import resource_tracker

    from verl.utils import shared_memory_resource_tracker as shm_tracker

    forwarded_calls = []

    def _fake_register(name, rtype):
        forwarded_calls.append(("register", name, rtype))

    def _fake_unregister(name, rtype):
        forwarded_calls.append(("unregister", name, rtype))

    monkeypatch.setattr(resource_tracker, "register", _fake_register)
    monkeypatch.setattr(resource_tracker, "unregister", _fake_unregister)
    monkeypatch.setattr(shm_tracker, "_PATCHED", False)
    monkeypatch.setattr(shm_tracker, "_ORIGINAL_REGISTER", None)
    monkeypatch.setattr(shm_tracker, "_ORIGINAL_UNREGISTER", None)

    shm_tracker.install_shared_memory_resource_tracker_bypass()

    resource_tracker.register("/psm_test", "shared_memory")
    resource_tracker.unregister("/psm_test", "shared_memory")
    resource_tracker.register("/sem_test", "semaphore")
    resource_tracker.unregister("/sem_test", "semaphore")

    assert forwarded_calls == [
        ("register", "/sem_test", "semaphore"),
        ("unregister", "/sem_test", "semaphore"),
    ]


def test_enable_vllm_shared_memory_resource_tracker_bypass_sets_env_and_installs(monkeypatch):
    from verl.utils import shared_memory_resource_tracker as shm_tracker

    install_calls = []
    monkeypatch.delenv("VERL_VLLM_DISABLE_SHARED_MEMORY_RESOURCE_TRACKER", raising=False)
    monkeypatch.setattr(
        shm_tracker,
        "install_shared_memory_resource_tracker_bypass",
        lambda: install_calls.append("called"),
    )

    shm_tracker.enable_vllm_shared_memory_resource_tracker_bypass()

    assert os.environ["VERL_VLLM_DISABLE_SHARED_MEMORY_RESOURCE_TRACKER"] == "1"
    assert install_calls == ["called"]


@pytest.mark.asyncio
async def test_vllm_http_server_shutdown_stops_uvicorn_and_engine():
    try:
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
    except Exception as exc:
        pytest.skip(f"vllm or its dependencies are not available: {exc}")

    engine = SimpleNamespace(shutdown_calls=0)

    def _shutdown_engine():
        engine.shutdown_calls += 1

    engine.shutdown = _shutdown_engine

    server = SimpleNamespace(should_exit=False, force_exit=False)

    async def _serve_until_exit():
        while not server.should_exit:
            await asyncio.sleep(0)
        # Simulate FastAPI lifespan cleanup that owns AsyncLLM shutdown.
        engine.shutdown()

    http_server = object.__new__(vLLMHttpServer)
    http_server.node_rank = 0
    http_server.engine = engine
    http_server._uvicorn_server = server
    http_server._server_task = asyncio.create_task(_serve_until_exit())

    await vLLMHttpServer.shutdown(http_server)

    assert server.should_exit is True
    assert http_server._server_task.done()
    assert engine.shutdown_calls == 1


@pytest.mark.asyncio
async def test_vllm_replica_shutdown_calls_server_shutdown_and_kills_servers(monkeypatch):
    try:
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica
    except Exception as exc:
        pytest.skip(f"vllm or its dependencies are not available: {exc}")

    class _FakeServer:
        def __init__(self):
            self.shutdown_calls = 0
            self.shutdown = SimpleNamespace(remote=self._shutdown_remote)

        async def _shutdown_remote(self):
            self.shutdown_calls += 1

    server_a = _FakeServer()
    server_b = _FakeServer()

    killed = []
    monkeypatch.setattr("ray.kill", lambda actor, no_restart=True: killed.append((actor, no_restart)))

    replica = object.__new__(vLLMReplica)
    replica.servers = [server_a, server_b]
    replica._server_handle = server_a
    replica._server_address = "127.0.0.1:8000"

    await vLLMReplica.shutdown(replica)

    assert server_a.shutdown_calls == 1
    assert server_b.shutdown_calls == 1
    assert killed == [(server_a, True), (server_b, True)]
    assert replica.servers == []
    assert replica._server_handle is None
    assert replica._server_address is None


def test_agent_loop_manager_shutdown_releases_replicas_and_actors(monkeypatch):
    class _FakeReplica:
        def __init__(self):
            self.shutdown_calls = 0

        async def shutdown(self):
            self.shutdown_calls += 1

    replica_a = _FakeReplica()
    replica_b = _FakeReplica()
    worker_a = object()
    worker_b = object()
    load_balancer = object()

    killed = []
    monkeypatch.setattr("ray.kill", lambda actor, no_restart=True: killed.append((actor, no_restart)))

    manager = object.__new__(AgentLoopManager)
    manager.rollout_replicas = [replica_a, replica_b]
    manager.agent_loop_workers = [worker_a, worker_b]
    manager.global_load_balancer = load_balancer
    manager.server_handles = ["server-a", "server-b"]
    manager.server_addresses = ["addr-a", "addr-b"]

    manager.shutdown()

    assert replica_a.shutdown_calls == 1
    assert replica_b.shutdown_calls == 1
    assert killed == [(worker_a, True), (worker_b, True), (load_balancer, True)]
    assert manager.rollout_replicas == []
    assert manager.agent_loop_workers == []
    assert manager.server_handles == []
    assert manager.server_addresses == []
    assert manager.global_load_balancer is None


def test_trainer_shutdown_dataloader_workers_is_idempotent():
    class _FakeIterator:
        def __init__(self):
            self.shutdown_calls = 0

        def _shutdown_workers(self):
            self.shutdown_calls += 1

    train_iter = _FakeIterator()
    val_iter = _FakeIterator()

    train_loader = SimpleNamespace(_iterator=train_iter)
    val_loader = SimpleNamespace(_iterator=val_iter)

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.train_dataloader = train_loader
    trainer.val_dataloader = val_loader

    trainer._shutdown_dataloader_workers()
    trainer._shutdown_dataloader_workers()

    assert train_iter.shutdown_calls == 1
    assert val_iter.shutdown_calls == 1
    assert train_loader._iterator is None
    assert val_loader._iterator is None
