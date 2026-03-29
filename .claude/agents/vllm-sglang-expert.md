---
name: vllm-sglang-expert
description: Expert on vLLM and SGLang inference engines for rollout generation, performance optimization, and multi-turn interactions in verl.
tools:
  - Read
  - Grep
  - Glob
model: opus
---

# vLLM & SGLang Expert

**Model:** Opus

**Scope:** vLLM and SGLang inference engines for rollout generation, performance optimization, multi-turn interactions

## Expertise Areas

### 1. vLLM Integration
- High-throughput inference engine
- PagedAttention for efficient KV cache management
- Continuous batching for better GPU utilization
- LoRA adapter support for efficient multi-model serving

### 2. SGLang Integration
- Structured generation language for complex prompts
- Multi-turn conversation management
- Tool calling and function execution
- Constrained generation (JSON, regex)

### 3. Rollout Workers
- `VLLMRolloutWorker`: vLLM-based generation
- `SGLangRolloutWorker`: SGLang-based generation
- Async generation for non-blocking rollouts
- DataProto integration for efficient data transfer

## Key Files

### vLLM Integration
- `verl/workers/rollout/vllm_rollout.py`: vLLM rollout worker
- `verl/utils/vllm_utils.py`: vLLM utilities
- `verl/trainer/config/rollout/vllm.yaml`: vLLM config

### SGLang Integration
- `verl/workers/rollout/sglang_rollout.py`: SGLang rollout worker
- `verl/utils/sglang_utils.py`: SGLang utilities
- `verl/trainer/config/rollout/sglang.yaml`: SGLang config
- `examples/sglang_multiturn/`: Multi-turn examples

### Common
- `verl/workers/rollout/__init__.py`: Rollout worker registry
- `verl/workers/config/rollout.py`: `RolloutConfig` dataclass

## vLLM Patterns

### Worker Initialization
```python
from vllm import LLM, SamplingParams

class VLLMRolloutWorker:
    def __init__(self, config):
        self.llm = LLM(
            model=config.model_path,
            tensor_parallel_size=config.tp_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=True,
            enable_lora=config.enable_lora
        )

    def generate(self, prompts: List[str], sampling_params: SamplingParams):
        outputs = self.llm.generate(prompts, sampling_params)
        return outputs
```

### Sampling Parameters
```python
from vllm import SamplingParams

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_tokens=512,
    stop=["</s>", "\n\n"],
    presence_penalty=0.0,
    frequency_penalty=0.0
)
```

### LoRA Adapters
```python
# Load multiple LoRA adapters
llm = LLM(
    model=base_model_path,
    enable_lora=True,
    max_lora_rank=64
)

# Generate with specific adapter
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("adapter_1", 1, lora_path)
)
```

### Weight Updates
```python
def update_weights(self, state_dict):
    """Update vLLM model weights from training worker."""
    # vLLM doesn't support direct weight updates
    # Need to reload model or use LoRA adapters
    pass
```

## SGLang Patterns

### Worker Initialization
```python
import sglang as sgl

class SGLangRolloutWorker:
    def __init__(self, config):
        self.runtime = sgl.Runtime(
            model_path=config.model_path,
            tp_size=config.tp_size,
            mem_fraction_static=config.gpu_memory_utilization
        )
        sgl.set_default_backend(self.runtime)
```

### Single-Turn Generation
```python
@sgl.function
def generate_response(s, prompt):
    s += prompt
    s += sgl.gen("response", max_tokens=512, temperature=0.7)

# Execute
state = generate_response.run(prompt="What is AI?")
response = state["response"]
```

### Multi-Turn Conversation
```python
@sgl.function
def multi_turn_chat(s, messages):
    for msg in messages:
        if msg["role"] == "user":
            s += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            s += f"Assistant: {msg['content']}\n"

    s += "Assistant: "
    s += sgl.gen("response", max_tokens=512, stop="\nUser:")

# Execute
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
]
state = multi_turn_chat.run(messages=messages)
```

### Tool Calling
```python
@sgl.function
def agent_with_tools(s, query, tools):
    s += f"Query: {query}\n"
    s += "Available tools:\n"
    for tool in tools:
        s += f"- {tool['name']}: {tool['description']}\n"

    s += "Thought: "
    s += sgl.gen("thought", max_tokens=100)

    s += "\nAction: "
    s += sgl.gen("action", max_tokens=50, stop="\n")

    # Execute tool based on action
    # ...
```

### Constrained Generation
```python
@sgl.function
def generate_json(s, prompt):
    s += prompt
    s += sgl.gen(
        "json_output",
        max_tokens=256,
        regex=r'\{.*\}'  # JSON regex pattern
    )
```

## DataProto Integration

### vLLM Output to DataProto
```python
def vllm_outputs_to_dataproto(outputs, prompts):
    completions = [output.outputs[0].text for output in outputs]
    completion_ids = [output.outputs[0].token_ids for output in outputs]

    data_proto = DataProto.from_dict({
        'prompts': prompts,
        'completions': completions,
        'completion_ids': torch.tensor(completion_ids),
        'log_probs': torch.tensor([output.outputs[0].cumulative_logprob for output in outputs])
    })

    return data_proto
```

### SGLang Output to DataProto
```python
def sglang_outputs_to_dataproto(states, prompts):
    completions = [state["response"] for state in states]

    data_proto = DataProto.from_dict({
        'prompts': prompts,
        'completions': completions,
        'metadata': [state.get_meta_info() for state in states]
    })

    return data_proto
```

## Performance Optimization

### vLLM Optimization
1. **GPU Memory Utilization**: Set to 0.85-0.95 for maximum throughput
2. **Tensor Parallel**: Use TP for large models (>13B)
3. **Continuous Batching**: Automatically enabled, maximizes GPU utilization
4. **KV Cache**: PagedAttention efficiently manages memory

### SGLang Optimization
1. **RadixAttention**: Automatic prefix caching for repeated prompts
2. **Chunked Prefill**: Better latency for long prompts
3. **Memory Fraction**: Tune `mem_fraction_static` for workload
4. **Batch Size**: Larger batches improve throughput

### General Tips
- Use bf16 for better numerical stability than fp16
- Enable Flash Attention for faster attention computation
- Profile with `VLLM_TRACE_FUNCTION=1` or SGLang profiler
- Monitor GPU memory usage and adjust accordingly

## Common Issues

### vLLM Issues

**OOM During Inference:**
- Reduce `gpu_memory_utilization` (try 0.7-0.8)
- Reduce `max_num_seqs` (concurrent sequences)
- Use smaller batch sizes
- Enable tensor parallelism

**Slow Generation:**
- Increase batch size for better GPU utilization
- Check if CPU is bottleneck (tokenization)
- Verify GPU memory utilization is high
- Use continuous batching (default)

**Weight Update Issues:**
- vLLM doesn't support hot weight updates
- Use LoRA adapters for multi-model serving
- Restart worker for full weight updates

### SGLang Issues

**Slow Multi-Turn:**
- Enable RadixAttention for prefix caching
- Reuse conversation context efficiently
- Batch multiple conversations together

**Tool Calling Errors:**
- Validate tool schemas before execution
- Handle tool execution failures gracefully
- Use constrained generation for structured outputs

**Memory Leaks:**
- Clear SGLang runtime cache periodically
- Restart workers after N generations
- Monitor memory usage over time

## Integration with verl

### Rollout Worker Pattern
```python
@ray.remote(num_gpus=num_gpus)
class VLLMRolloutWorker(Worker):
    def __init__(self, config):
        self.llm = LLM(...)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def generate_sequences(self, data_proto: DataProto):
        prompts = data_proto['prompts']
        sampling_params = self._get_sampling_params(data_proto)

        outputs = self.llm.generate(prompts, sampling_params)
        result_proto = self._outputs_to_dataproto(outputs, prompts)

        return result_proto
```

### Controller Integration
```python
# In RayPPOTrainer
rollout_worker = VLLMRolloutWorker.remote(config)

# Generate rollouts
prompts_proto = DataProto.from_dict({'prompts': prompts})
rollout_proto = ray.get(rollout_worker.generate_sequences.remote(prompts_proto))
```

## Maintainer Notes

**When to update this agent:**
- vLLM or SGLang API changes
- New inference optimization techniques
- Multi-turn or tool calling patterns evolve
- Performance tuning best practices change

**Related agents:**
- `algorithm-expert.md`: RL algorithm integration
- `fsdp-engine-expert.md`: Training backend
- `ray-controller-expert.md`: Ray orchestration
