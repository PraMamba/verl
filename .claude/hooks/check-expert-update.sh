#!/usr/bin/env bash
# Hook to check if expert agents need updating after code changes

# Get the file that was modified
MODIFIED_FILE="$1"

# Define mappings: file pattern -> expert agent
declare -A EXPERT_MAPPINGS=(
    ["verl/workers/fsdp_workers.py"]="fsdp-engine-expert.md"
    ["verl/utils/fsdp_utils/"]="fsdp-engine-expert.md"
    ["verl/models/mcore/"]="megatron-engine-expert.md"
    ["verl/utils/megatron_utils/"]="megatron-engine-expert.md"
    ["verl/trainer/ppo/"]="algorithm-expert.md"
    ["verl/reward/"]="algorithm-expert.md"
    ["verl/workers/rollout/vllm"]="vllm-sglang-expert.md"
    ["verl/workers/rollout/sglang"]="vllm-sglang-expert.md"
    ["verl/utils/vllm_utils"]="vllm-sglang-expert.md"
    ["verl/utils/sglang_utils"]="vllm-sglang-expert.md"
    ["verl/single_controller/"]="ray-controller-expert.md"
    ["verl/trainer/ppo/ray_trainer.py"]="ray-controller-expert.md"
)

# Check if file matches any pattern
for pattern in "${!EXPERT_MAPPINGS[@]}"; do
    if [[ "$MODIFIED_FILE" == *"$pattern"* ]]; then
        expert="${EXPERT_MAPPINGS[$pattern]}"
        echo "💡 Consider updating .claude/agents/$expert to reflect changes in $MODIFIED_FILE"
        break
    fi
done
