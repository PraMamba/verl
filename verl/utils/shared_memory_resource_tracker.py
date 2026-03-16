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
import os
from multiprocessing import resource_tracker

_PATCHED = False
_ORIGINAL_REGISTER = None
_ORIGINAL_UNREGISTER = None


def install_shared_memory_resource_tracker_bypass() -> None:
    """Disable multiprocessing resource tracking for POSIX shared memory only.

    vLLM already manages the lifecycle of its shared-memory segments and may
    unlink them manually across spawned workers. Letting Python's
    resource_tracker track the same objects can lead to shutdown-time relaunches
    and KeyError noise when the tracker loses cache state. Keep all other
    resource types untouched.
    """
    global _PATCHED, _ORIGINAL_REGISTER, _ORIGINAL_UNREGISTER

    if _PATCHED:
        return

    _ORIGINAL_REGISTER = resource_tracker.register
    _ORIGINAL_UNREGISTER = resource_tracker.unregister

    def _register(name: str, rtype: str):
        if rtype == "shared_memory":
            return None
        return _ORIGINAL_REGISTER(name, rtype)

    def _unregister(name: str, rtype: str):
        if rtype == "shared_memory":
            return None
        return _ORIGINAL_UNREGISTER(name, rtype)

    resource_tracker.register = _register
    resource_tracker.unregister = _unregister
    _PATCHED = True


def enable_vllm_shared_memory_resource_tracker_bypass() -> None:
    os.environ["VERL_VLLM_DISABLE_SHARED_MEMORY_RESOURCE_TRACKER"] = "1"
    install_shared_memory_resource_tracker_bypass()
