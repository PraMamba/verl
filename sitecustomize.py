import os

if os.environ.get("VERL_VLLM_DISABLE_SHARED_MEMORY_RESOURCE_TRACKER", "0") == "1":
    from verl.utils.shared_memory_resource_tracker import install_shared_memory_resource_tracker_bypass

    install_shared_memory_resource_tracker_bypass()
