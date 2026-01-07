# verl Parameter Reference

> Based on verl source code analysis, referencing the document structure from [Awesome-ML-SYS-Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial)
>
> Contributors: Ji Li (Ant Group), Zhuoran Yin (CMU), Changyi Yang (CMU), Chengxi Li (CMU), Xinpeng Wei (Amazon), Chenyang Zhao (Amazon)

Due to [Hydra's usage](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme.md#脚本配置), verl parameters are scattered throughout the framework. This document is organized based on YAML configuration files in the `verl/trainer/config/` directory.

---

## Table of Contents

- [Batch Size Parameters](#batch-size-parameters)
- [Dynamic Batch Size](#dynamic-batch-size)
- [Rollout Sampling Parameters](#rollout-sampling-parameters)
- [Rollout Performance and Resource Management](#rollout-performance-and-resource-management)
- [SGLang Configuration](#sglang-configuration)
- [Multi-turn Configuration](#multi-turn-configuration)
- [Validation Configuration](#validation-configuration)
- [Dataset Configuration](#dataset-configuration)
- [Model Configuration](#model-configuration)
- [Actor Configuration](#actor-configuration)
- [Reference Model Configuration](#reference-model-configuration)
- [Critic Configuration](#critic-configuration)
- [Reward Model Configuration](#reward-model-configuration)
- [Custom Reward Function](#custom-reward-function)
- [Algorithm Configuration](#algorithm-configuration)
- [Rollout Correction](#rollout-correction)
- [Trainer Configuration](#trainer-configuration)
- [FSDP Engine Configuration](#fsdp-engine-configuration)
- [Megatron Engine Configuration](#megatron-engine-configuration)
- [Optimizer Configuration](#optimizer-configuration)
- [Profiler Configuration](#profiler-configuration)
- [Ray Configuration](#ray-configuration)

---

## Batch Size Parameters

| Parameter | Description |
| --------- | ----------- |
| `data.train_batch_size` | **Purpose**: Defines the number of samples sent to the Rollout Engine in a single training iteration, i.e., the number of prompts sampled from the training dataset at the beginning of each PPO iteration.<br><br>**Details**: This value is the basic sample count in RL training. For example, setting it to 1024 means in one iteration:<br>1. Randomly sample 1024 prompts from the dataset<br>2. Send these 1024 prompts to the Rollout Engine to get 1024 complete trajectories (prompt, response)<br>3. These 1024 trajectories undergo experience computation (make experience) for subsequent Actor and Critic model updates<br><br>**Default**: 1024 |
| `data.val_batch_size` | **Purpose**: Batch size used during validation phase.<br><br>**Details**: When set to `null`, the entire validation dataset is sent to the inference engine at once, which manages memory internally. Recommended to set to `null`.<br><br>**Default**: `null` |
| `actor_rollout_ref.actor.ppo_mini_batch_size` / `critic.ppo_mini_batch_size` | **Purpose**: Defines the mini-batch size for PPO training updates.<br><br>**Details**: All experience data collected from `data.train_batch_size` will be split into multiple mini-batches, each of size `ppo_mini_batch_size`. The model performs one parameter update after processing each mini-batch.<br><br>For example, if `train_batch_size = 1024`, `ppo_mini_batch_size = 256`, then in one PPO Epoch, the model will perform `1024 / 256 = 4` parameter updates.<br><br>**Trade-offs**: Larger mini-batch means more stable gradients per update, but lower update frequency and fewer updates.<br><br>**Default**: 256 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` / `critic.ppo_micro_batch_size_per_gpu` | **Purpose**: Defines the data size for a single forward/backward pass on each GPU.<br><br>**Details**: This is the core parameter for gradient accumulation. Mini-batches are further split into micro-batches. For example, on a single GPU with `ppo_mini_batch_size = 256`, `ppo_micro_batch_size_per_gpu = 32`, the gradient accumulation steps would be `256 / 32 = 8`. This means the model runs 8 forward passes to get losses, then backward to get gradients. Each pass processes 32 samples until gradients for the entire mini-batch are accumulated. Then the accumulated gradients are used for one parameter update (`optimizer.step()`).<br><br>**Trade-offs**: Larger values reduce gradient accumulation steps, increasing training throughput but also memory usage. This value must be carefully tuned based on GPU memory to prevent OOM.<br><br>**Default**: `null` |
| `actor_rollout_ref.actor.ppo_micro_batch_size` / `critic.ppo_micro_batch_size` | **Deprecated**: Replaced by `*_per_gpu` versions which better adapt to distributed training environments. |

---

## Dynamic Batch Size

When sample lengths vary significantly, batching by sample count can lead to very uneven computation across batches. Controlling batch size by total token count is a solution to balance training time per batch.

| Parameter | Description |
| --------- | ----------- |
| `actor_rollout_ref.actor.use_dynamic_bsz` / `critic.use_dynamic_bsz` / `reward_model.use_dynamic_bsz` | **Purpose**: Whether to enable Dynamic Batch Size.<br><br>**Details**: When `True`, the system ignores sample-count-based `micro_batch_size_per_gpu` parameters and instead uses token-count-based `max_token_len_per_gpu` parameters to construct batches.<br><br>**Default**: `false` |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` / `critic.ppo_max_token_len_per_gpu` | **Purpose**: Defines the maximum total Token count a single GPU can process in one PPO micro batch.<br><br>**Details**: This is an alternative to `ppo_micro_batch_size_per_gpu`, used with `use_dynamic_bsz`. The system automatically packs samples until total tokens (`prompt_len + response_len`) approach this threshold, forming a dynamic micro batch size to stabilize computational efficiency.<br><br>**Recommended**: Typically set to `n * (max_prompt_length + max_response_length)`<br><br>**Default**: Actor 16384, Critic 32768 |
| `critic.forward_max_token_len_per_gpu` / `reward_model.forward_max_token_len_per_gpu` / `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu` | **Purpose**: Maximum token count per micro-batch for models that only do forward computation.<br><br>**Details**: Some models (Reward Model, Critic for values, Reference Model for log probs) only do forward computation during the make experience phase. At this time, the rollout engine is already offloaded and the training engine hasn't started yet, so memory usage is low. Therefore, a larger batch size can be set to speed up computation. |
| `critic.forward_micro_batch_size_per_gpu` / `reward_model.micro_batch_size_per_gpu` / `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | **Purpose**: Sets micro-batch size (by sample count) for forward-only models. |
| `trainer.balance_batch` | **Purpose**: Whether to balance batch sizes across dp ranks in distributed training.<br><br>**Details**: Reorders data on the single controller so each dp rank gets a similar number of tokens.<br><br>**Default**: `True` |

---

## Rollout Sampling Parameters

| Parameter | Description |
| --------- | ----------- |
| `actor_rollout_ref.rollout.name` | Inference engine type: `hf`/`vllm`/`sglang`. Must be set. |
| `actor_rollout_ref.rollout.mode` | Inference mode: `sync` (synchronous LLM) or `async` (asynchronous AsyncLLM).<br>**Default**: `async` |
| `actor_rollout_ref.rollout.temperature` | Higher temperature means smoother probability distribution, more diverse and random generation; lower values mean sharper distribution, generation favors high-probability tokens. `temperature=0` typically equals Greedy Decoding.<br>**Default**: 1.0 |
| `actor_rollout_ref.rollout.top_k` | Only consider the top K tokens by probability for sampling at each generation step. E.g., `top_k=50` means only sample from the top 50 tokens by probability.<br>- Disable: Set to `-1` for vLLM/SGLang, `0` for HF.<br>**Default**: -1 |
| `actor_rollout_ref.rollout.top_p` | Accumulate tokens from highest probability until total probability reaches P, then sample from this nucleus token set. A dynamic approach to selecting sampling range. `top_p=1.0` means no restriction.<br>**Default**: 1.0 |
| `actor_rollout_ref.rollout.n` | Number of responses to generate per prompt, i.e., group size in GRPO.<br>**Default**: 1 |
| `actor_rollout_ref.rollout.ignore_eos` | Whether to ignore EOS (End-of-Sentence) token. If `True`, continue generating until `response_length` even after generating EOS.<br>**Default**: `False` |
| `actor_rollout_ref.rollout.do_sample` | Whether to sample during training rollout. `False` uses greedy sampling.<br>**Default**: `True` |
| `actor_rollout_ref.rollout.calculate_log_probs` | Whether to calculate rollout log probs (for debugging or Truncated Importance Sampling).<br>**Default**: `False` |
| `actor_rollout_ref.rollout.over_sample_rate` | Over-sample rate, controls early termination threshold for training rollouts. System will terminate remaining requests when `(1 - over_sample_rate) * total_requests` completions are reached.<br>**Default**: 0 |

---

## Rollout Performance and Resource Management

| Parameter | Description |
| --------- | ----------- |
| `actor_rollout_ref.rollout.prompt_length` | Maximum prompt length, truncated if exceeded.<br>**Default**: Same as `data.max_prompt_length`, default 512 |
| `actor_rollout_ref.rollout.response_length` | Maximum response length, inference engine returns directly upon reaching max length.<br>**Default**: Same as `data.max_response_length`, default 512 |
| `actor_rollout_ref.rollout.dtype` | Model data type. E.g., `bfloat16`, `float16`, must align with training phase model type.<br>**Default**: `bfloat16` |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | Proportion of GPU memory used for model parameters and KV Cache in vLLM/SGLang. For SGLang >= 0.4.8.post1, can set to 0.85; lower versions should set to around 0.5.<br>**Default**: 0.5 |
| `actor_rollout_ref.rollout.free_cache_engine` | Whether to free engine cache after rollout. In SGLang, enabling this triggers `flush_cache()`: clears KV cache pool, marks all slots as available.<br>**Default**: `True` |
| `actor_rollout_ref.rollout.load_format` | Model weight loading mode: `dummy` (random init for debugging), `hf`, `megatron`, `safetensors` (recommended, safe and efficient).<br>**Default**: `dummy` |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | Tensor parallelism size (TP_SIZE), number of GPUs to jointly run one inference engine. E.g., `TP_SIZE=4` means splitting model weights into 4 parts across 4 GPUs. Not effective for HF rollout.<br>**Default**: 2 |
| `actor_rollout_ref.rollout.data_parallel_size` | Data parallel size (DP_SIZE).<br>**Default**: 1 |
| `actor_rollout_ref.rollout.expert_parallel_size` | Expert parallel size (EP_SIZE) for MoE models.<br>**Default**: 1 |
| `actor_rollout_ref.rollout.pipeline_model_parallel_size` | Pipeline parallel size (PP_SIZE).<br>**Default**: 1 |
| `actor_rollout_ref.rollout.max_model_len` | Maximum total length (prompt + response) the model can process; typically determined by model config if not set.<br>**Default**: `null` |
| `actor_rollout_ref.rollout.max_num_seqs` | Maximum concurrent requests the engine can process, i.e., maximum prompts inferenced simultaneously.<br>**Default**: 1024 |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | Maximum tokens in a batch.<br>**Default**: 8192 |
| `actor_rollout_ref.rollout.enable_chunked_prefill` | Whether to enable Chunked Prefill. For very long prompts, can process in chunks to reduce peak memory, but may reduce throughput.<br>**Default**: `True` |
| `actor_rollout_ref.rollout.enable_prefix_caching` | Whether to enable prefix caching. Prefix caching is a common optimization in LLM inference to avoid redundant prompt computation.<br>**Default**: `True` |
| `actor_rollout_ref.rollout.enforce_eager` | Whether to disable CUDA graph. Default `False` for best performance.<br>**Default**: `False` |
| `actor_rollout_ref.rollout.cudagraph_capture_sizes` | List of CUDA graph batch sizes to capture. Requires `enforce_eager: False`. Since cudagraph in inference engine cannot be offloaded during policy update, smaller batch sizes can save CUDA graph memory.<br>**Supported engines**: vllm<br>**Default**: `null` |
| `actor_rollout_ref.rollout.disable_log_stats` | Whether to disable inference engine statistics logs to reduce console output.<br>**Default**: `True` |
| `actor_rollout_ref.rollout.multi_stage_wake_up` | Whether to enable multi-stage wake up for SGLang inference engine to reduce peak memory during training-rollout transition. Only effective for SGLang rollout.<br>**Default**: `false` |
| `actor_rollout_ref.rollout.update_weights_bucket_megabytes` | Specifies tensor bucket size (in MB) for batch weight updates during rollout operations. Controls maximum payload size for a single weight update request.<br>**Only supported in SGLang rollout**<br>**Default**: 512 |
| `actor_rollout_ref.rollout.skip_rollout` | Whether to skip rollout computation and attempt to load previously generated rollout data from specified directory. Useful for debugging or reusing computation results across runs.<br>**Default**: `False` |
| `actor_rollout_ref.rollout.skip_dump_dir` | Filesystem path to cache rollout data when `skip_rollout` is enabled.<br>**Default**: `/tmp/rollout_dump` |
| `actor_rollout_ref.rollout.skip_tokenizer_init` | Whether to skip tokenizer initialization for rollout engine. When enabled, rollout assumes token in token out for generation.<br>**Default**: `True` |
| `actor_rollout_ref.rollout.enable_rollout_routing_replay` | Whether to enable rollout routing replay for MoE models. When enabled, rollout records routing decisions.<br>**Default**: `False` |

---

## SGLang Configuration

| Parameter | Description |
| --------- | ----------- |
| `actor_rollout_ref.rollout.engine_kwargs.sglang` | Additional configuration parameters for SGLang engine. Can pass any SGLang officially supported parameters. |
| `actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend` | Attention backend used by SGLang. Options: `flashinfer`, `triton`, `flashmla`, `null`, to adapt to different GPUs. |
| `actor_rollout_ref.rollout.engine_kwargs.vllm` | Additional configuration parameters for vLLM engine. Can pass any vLLM officially supported parameters. |

---

## Multi-turn Configuration

These parameters are mainly used for scenarios requiring multi-turn interaction, such as tool calling, continuous dialogue, supported by SGLang Engine.

| Parameter | Description |
| --------- | ----------- |
| `actor_rollout_ref.rollout.multi_turn.enable` | Whether to enable multi-turn dialogue mode. Set `rollout.name` to `sglang`.<br>**Default**: `False` |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` | Maximum number of assistant reply turns. `null` defaults to `max_model_len // 3` to avoid infinite dialogue.<br>**Default**: `null` |
| `actor_rollout_ref.rollout.multi_turn.max_user_turns` | Maximum number of user message turns. `null` means no limit (default `max_model_len // 3`).<br>**Default**: `null` |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | Tool configuration file path, defines external tools the model can call. `null` means no tools.<br>**Default**: `null` |
| `actor_rollout_ref.rollout.multi_turn.interaction_config_path` | Interaction configuration file path. `null` means no interaction.<br>**Default**: `null` |
| `actor_rollout_ref.rollout.multi_turn.max_parallel_calls` | Maximum parallel tool calls in a single turn.<br>**Default**: 1 |
| `actor_rollout_ref.rollout.multi_turn.max_tool_response_length` | Maximum length of tool response.<br>**Default**: 256 |
| `actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side` | Tool response truncation side: `left`, `middle`, `right`.<br>**Default**: `middle` |
| `actor_rollout_ref.rollout.multi_turn.use_inference_chat_template` | Whether to use the model's inference-phase chat template.<br>- `True`: Follow inference-phase template format, typically matches production behavior<br>- `False`: Use pretraining template, may include additional content like reasoning process<br><br>**Important**: Ensure consistent templates between post-training and inference testing phases.<br>**Default**: `False` |
| `actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode` | Tokenization sanity check mode, verifies turn-by-turn tokenization results match tokenizing the entire chat history at once.<br>- `disable`: Disable check<br>- `strict`: Enable strict check (default)<br>- `ignore_strippable`: Ignore strippable tokens<br><br>**Validated models**: Qwen/QwQ-32B, Qwen/Qwen3-xxB<br>**Default**: `strict` |
| `actor_rollout_ref.rollout.multi_turn.format` | Multi-turn interaction format. Options: `hermes`, `llama3_json`, etc.<br>**Default**: `hermes` |
| `actor_rollout_ref.rollout.multi_turn.num_repeat_rollouts` | Number of repeat rollouts per interaction.<br>**Default**: `null` |

---

## Agent Loop Configuration

| Parameter | Description |
| --------- | ----------- |
| `actor_rollout_ref.rollout.agent.num_workers` | Number of Agent loop workers.<br>**Default**: 8 |
| `actor_rollout_ref.rollout.agent.default_agent_loop` | Default agent loop to use if `agent_name` not set in RL dataset.<br>**Default**: `single_turn_agent` |
| `actor_rollout_ref.rollout.agent.agent_loop_config_path` | Custom agent loop config path, should contain list of configs to initialize AgentLoop instances.<br>**Default**: `null` |
| `actor_rollout_ref.rollout.agent.custom_async_server.path` | Path to custom async server implementation.<br>**Default**: `null` |
| `actor_rollout_ref.rollout.agent.custom_async_server.name` | Class name of custom async server class (e.g., `AsyncvLLMServer`).<br>**Default**: `null` |

---

## Validation Configuration

| Parameter | Description |
| --------- | ----------- |
| `actor_rollout_ref.rollout.val_kwargs.*` | Validation phase sampling parameters, allowing different sampling parameters for post-training and validation. E.g., validation typically sets `temperature=0` and `do_sample=False` for greedy decoding for more stable evaluation results. |
| `actor_rollout_ref.rollout.val_kwargs.temperature` | Validation phase temperature.<br>**Default**: 0 |
| `actor_rollout_ref.rollout.val_kwargs.top_k` | Validation phase top-k.<br>**Default**: -1 |
| `actor_rollout_ref.rollout.val_kwargs.top_p` | Validation phase top-p.<br>**Default**: 1.0 |
| `actor_rollout_ref.rollout.val_kwargs.n` | Number of repeats per prompt during validation.<br>**Default**: 1 |
| `actor_rollout_ref.rollout.val_kwargs.do_sample` | Whether to sample during validation.<br>**Default**: `False` |

---

## Rollout Trace Configuration

| Parameter | Description |
| --------- | ----------- |
| `actor_rollout_ref.rollout.trace.backend` | Trace backend, supports `mlflow`, `weave`.<br>**Default**: `null` |
| `actor_rollout_ref.rollout.trace.token2text` | Whether to translate token id to text in output.<br>**Default**: `False` |
| `actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker` | Maximum unique samples to trace per agent worker per training step. If `null`, all samples are traced.<br>**Default**: `null` |

---

## Dataset Configuration

| Parameter | Description |
| --------- | ----------- |
| `data.tokenizer` | Tokenizer class or path. If `null`, will be inferred from the model.<br>**Default**: `null` |
| `data.use_shm` | Whether to use shared memory for data loading.<br>**Default**: `False` |
| `data.train_files` | Training set parquet file(s). Can be list or single file; path can be local or HDFS. Program reads all files into memory, so can't be too large (< 100GB). |
| `data.val_files` | Validation parquet file(s). Can be list or single file. |
| `data.train_max_samples` | Maximum training samples. Set to -1 for full dataset, otherwise randomly select specified number of samples from training dataset.<br>**Default**: -1 |
| `data.val_max_samples` | Maximum validation samples. Set to -1 for full dataset.<br>**Default**: -1 |
| `data.prompt_key` | Field name for prompt in dataset.<br>**Default**: `prompt` |
| `data.reward_fn_key` | Field for selecting reward function (if using different ones per example).<br>**Default**: `data_source` |
| `data.max_prompt_length` | Maximum prompt length. All prompts will be left-padded to this length.<br>**Default**: 512 |
| `data.max_response_length` | Maximum response length. Rollout in RL algorithms (e.g., PPO) generates up to this length.<br>**Default**: 512 |
| `data.tool_config_path` | Tool config path for calculating true prompt length.<br>**Default**: Same as `actor_rollout_ref.rollout.multi_turn.tool_config_path` |
| `data.return_raw_input_ids` | Whether to return original input_ids without adding chat template. Use when reward model's chat template differs from policy model.<br>**Default**: `False` |
| `data.return_raw_chat` | Whether to return original response without applying chat template.<br>**Default**: `True` |
| `data.return_full_prompt` | Whether to return full prompt with chat template.<br>**Default**: `False` |
| `data.return_multi_modal_inputs` | Whether to return multi-modal inputs in dataset. Set to `False` if rollout generates new multi-modal inputs.<br>**Default**: `True` |
| `data.shuffle` | Whether to shuffle data in DataLoader.<br>**Default**: `True` |
| `data.seed` | Random seed for shuffling data.<br>**Default**: `null` |
| `data.dataloader_num_workers` | Number of DataLoader workers.<br>**Default**: 8 |
| `data.validation_shuffle` | Whether to shuffle validation set.<br>**Default**: `False` |
| `data.filter_overlong_prompts` | Whether to filter overlong prompts.<br>**Default**: `False` |
| `data.filter_overlong_prompts_workers` | Number of workers for filtering overlong prompts. Use multiprocessing for large datasets.<br>**Default**: 1 |
| `data.truncation` | Truncate input_ids or prompt if exceeding max length. Options: `error`, `left`, `right`, `middle`.<br>**Default**: `error` |
| `data.image_key` | Field name for images in multi-modal dataset.<br>**Default**: `images` |
| `data.image_patch_size` | Image patch size.<br>**Default**: 14 |
| `data.video_key` | Field name for videos in multi-modal dataset.<br>**Default**: `videos` |
| `data.trust_remote_code` | Whether to trust local huggingface cache. Note: "remote" is relative to huggingface, so this parameter considers "whether to trust local".<br>**Default**: `False` |
| `data.custom_cls.path` | Path to file containing custom dataset class. If not specified, pre-implemented default dataset will be used.<br>**Default**: `null` |
| `data.custom_cls.name` | Dataset class name within specified file.<br>**Default**: `null` |
| `data.apply_chat_template_kwargs` | Additional arguments when calling `tokenizer.apply_chat_template`.<br>**Default**: `{}` |

### Dataset Sampler Configuration

| Parameter | Description |
| --------- | ----------- |
| `data.sampler.class_path` | Path to module containing curriculum class implementing AbstractSampler interface.<br>**Default**: `null` |
| `data.sampler.class_name` | Curriculum class name, e.g., `MySampler`.<br>**Default**: `null` |

### Data Generation Configuration

| Parameter | Description |
| --------- | ----------- |
| `data.datagen.path` | Path to file containing custom data generation class. E.g., `pkg://verl.experimental.dynamic_dataset.dynamicgen_dataset`.<br>**Default**: `null` |
| `data.datagen.name` | Data generation class name within specified file. E.g., `MockDataGenerator`.<br>**Default**: `null` |

---

## Model Configuration

| Parameter | Description |
| --------- | ----------- |
| `actor_rollout_ref.model.path` | Huggingface model path. Can be local or HDFS path. |
| `actor_rollout_ref.model.hf_config_path` | Huggingface config path (if different from model path).<br>**Default**: `null` |
| `actor_rollout_ref.model.tokenizer_path` | Huggingface tokenizer path (if different from model path).<br>**Default**: `null` |
| `actor_rollout_ref.model.use_shm` | Whether to use shared memory (SHM) to accelerate model weight loading.<br>**Default**: `False` |
| `actor_rollout_ref.model.trust_remote_code` | Whether to trust local huggingface cache.<br>**Default**: `False` |
| `actor_rollout_ref.model.custom_chat_template` | Custom chat template for the model.<br>**Default**: `null` |
| `actor_rollout_ref.model.external_lib` | Additional Python packages for registering Huggingface model/tokenizer.<br>**Default**: `null` |
| `actor_rollout_ref.model.override_config` | Override model original config, mainly for dropout, etc.<br>**Default**: `{}` |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | Whether to recompute gradients during actor training, trading time for space. Only valid with HF model definition.<br>**Default**: `True` |
| `actor_rollout_ref.model.enable_activation_offload` | Whether to offload activation to CPU during actor training. Only valid with HF model definition.<br>**Default**: `False` |
| `actor_rollout_ref.model.use_remove_padding` | Whether to remove padding tokens in inputs during training. Only valid with HF model definition.<br>**Default**: `True` |
| `actor_rollout_ref.model.use_liger` | Whether to use Liger kernel for linear layer fusion. Only valid with HF model definition.<br>**Default**: `False` |
| `actor_rollout_ref.model.use_fused_kernels` | Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).<br>**Default**: `False` |
| `actor_rollout_ref.model.fused_kernel_options.impl_backend` | Implementation backend for fused kernels: `torch` or `triton`. Used with `use_fused_kernels`.<br>**Default**: `torch` |

### LoRA Configuration (FSDP)

| Parameter | Description |
| --------- | ----------- |
| `actor_rollout_ref.model.lora_rank` | LoRA rank. Set to positive value to enable LoRA (e.g., 32). Set to 0 to disable.<br>**Default**: 0 |
| `actor_rollout_ref.model.lora_alpha` | LoRA scaling factor.<br>**Default**: 16 |
| `actor_rollout_ref.model.target_modules` | Target modules for LoRA adaptation.<br>**Default**: `all-linear` |
| `actor_rollout_ref.model.exclude_modules` | Modules to exclude from LoRA adaptation.<br>**Default**: `null` |
| `actor_rollout_ref.model.lora_adapter_path` | Path to pre-trained LoRA adapter for continued training.<br>**Default**: `null` |

### LoRA Configuration (Megatron)

| Parameter | Description |
| --------- | ----------- |
| `actor_rollout_ref.model.lora.type` | LoRA type: `lora`, `vlm_lora`, `canonical_lora`, or `dora`.<br>**Default**: `lora` |
| `actor_rollout_ref.model.lora.rank` | LoRA rank (dimension of low-rank projection space). Set to 0 to disable. Typical values: 8, 16, 32, 64.<br>**Default**: 0 |
| `actor_rollout_ref.model.lora.alpha` | Weighting factor for low-rank projection.<br>**Default**: 32 |
| `actor_rollout_ref.model.lora.dropout` | Dropout rate for low-rank projection.<br>**Default**: 0.0 |
| `actor_rollout_ref.model.lora.target_modules` | List of module names to apply LoRA. For fused LoRA, defaults to all linear layers `['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']`. |
| `actor_rollout_ref.model.lora.exclude_modules` | List of module names not to apply LoRA.<br>**Default**: `[]` |
| `actor_rollout_ref.model.lora.dropout_position` | Position for applying dropout: `pre` (before low-rank projection) or `post` (after).<br>**Default**: `pre` |
| `actor_rollout_ref.model.lora.lora_A_init_method` | Initialization method for low-rank matrix A.<br>**Default**: `xavier` |
| `actor_rollout_ref.model.lora.lora_B_init_method` | Initialization method for low-rank matrix B.<br>**Default**: `zero` |
| `actor_rollout_ref.model.lora.a2a_experimental` | Enable experimental All-to-All (A2A) communication strategy.<br>**Default**: `False` |
| `actor_rollout_ref.model.lora.dtype` | Parameter data type for LoRA weights. Default to `null`, uses model's dtype. |
| `actor_rollout_ref.model.lora.adapter_path` | Path to pre-trained LoRA adapter weights (`null` to train from scratch). |
| `actor_rollout_ref.model.lora.freeze_vision_model` | VLMLoRA: Whether to freeze vision model.<br>**Default**: `True` |
| `actor_rollout_ref.model.lora.freeze_vision_projection` | VLMLoRA: Whether to freeze vision projection.<br>**Default**: `True` |
| `actor_rollout_ref.model.lora.freeze_language_model` | VLMLoRA: Whether to freeze language model.<br>**Default**: `True` |

---

## Actor Configuration

| Parameter | Description |
| --------- | ----------- |
| `actor_rollout_ref.hybrid_engine` | Currently only supports hybrid engine, placing actor and rollout models on the same resource group.<br>**Default**: `true` |
| `actor_rollout_ref.nccl_timeout` | Timeout (in seconds) for operations against the process group.<br>**Default**: 600 |
| `actor_rollout_ref.rollout.layered_summon` | For huge models, layered summon can save memory (prevent OOM) but slows it down.<br>**Default**: `False` |
| `actor_rollout_ref.actor.strategy` | Training backend: `fsdp`, `fsdp2`, or `megatron`. Must be set. |
| `actor_rollout_ref.actor.rollout_n` | Number of rollouts per update (mirrors `actor_rollout_ref.rollout.n`). |
| `actor_rollout_ref.actor.grad_clip` | Gradient clipping for actor updates.<br>**Default**: 1.0 |
| `actor_rollout_ref.actor.clip_ratio` | PPO clip ratio.<br>**Default**: 0.2 |
| `actor_rollout_ref.actor.clip_ratio_low` | Lower bound for asymmetric clipping (used in dual-clip PPO).<br>**Default**: 0.2 |
| `actor_rollout_ref.actor.clip_ratio_high` | Upper bound for asymmetric clipping (used in dual-clip PPO).<br>**Default**: 0.2 |
| `actor_rollout_ref.actor.clip_ratio_c` | Constant C in Dual-clip PPO; clips when advantage < 0 and ratio > C.<br>**Default**: 3.0 |
| `actor_rollout_ref.actor.freeze_vision_tower` | Whether to freeze vision model.<br>**Default**: `false` |
| `actor_rollout_ref.actor.loss_agg_mode` | Loss aggregation mode: `token-mean`, `seq-mean-token-sum`, `seq-mean-token-mean`, or `seq-mean-token-sum-norm`.<br>**Default**: `token-mean` |
| `actor_rollout_ref.actor.loss_scale_factor` | Scale factor for `seq-mean-token-sum-norm` loss aggregation mode. If `null`, uses `response_length`. Set to constant for consistent normalization.<br>**Default**: `null` |
| `actor_rollout_ref.actor.entropy_coeff` | Entropy regularization coefficient in PPO loss.<br>**Default**: 0 |
| `actor_rollout_ref.actor.calculate_entropy` | When true, actor forward will request entropy from the model.<br>**Default**: `false` |
| `actor_rollout_ref.actor.use_kl_loss` | Whether to use KL loss instead of KL reward penalty. `True` for GRPO.<br>**Default**: `false` |
| `actor_rollout_ref.actor.use_torch_compile` | Whether to use `torch.compile()`.<br>**Default**: `true` |
| `actor_rollout_ref.actor.kl_loss_coef` | KL loss coefficient when `use_kl_loss` is enabled. For GRPO.<br>**Default**: 0.001 |
| `actor_rollout_ref.actor.kl_loss_type` | Type of KL divergence loss. Options: `kl`(k1), `abs`, `mse`(k2), `low_var_kl`(k3), `full`.<br>**Default**: `low_var_kl` |
| `actor_rollout_ref.actor.ppo_epochs` | Number of PPO epochs per batch.<br>**Default**: 1 |
| `actor_rollout_ref.actor.shuffle` | Shuffle training data across PPO epochs.<br>**Default**: `false` |
| `actor_rollout_ref.actor.data_loader_seed` | Seed used to construct mini-batch.<br>**Default**: 42 |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size` | Ulysses-style sequence parallel size. **Deprecated**: Use `fsdp_config.ulysses_sequence_parallel_size` instead.<br>**Default**: 1 |
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking` | Calculate entropy with chunking to reduce memory peak.<br>**Default**: `False` |
| `actor_rollout_ref.actor.entropy_checkpointing` | Whether to checkpoint entropy.<br>**Default**: `False` |
| `actor_rollout_ref.actor.use_remove_padding` | Whether to remove padding tokens in inputs during training.<br>**Default**: Same as `actor_rollout_ref.model.use_remove_padding` |

### Actor Policy Loss Configuration

| Parameter | Description |
| --------- | ----------- |
| `actor_rollout_ref.actor.policy_loss.loss_mode` | Loss function mode: `vanilla`, `clip-cov`, `kl-cov`, `gpg` (from https://arxiv.org/abs/2505.22617).<br>**Default**: `vanilla` |
| `actor_rollout_ref.actor.policy_loss.clip_cov_ratio` | Ratio of tokens to be clipped for clip-cov loss.<br>**Default**: 0.0002 |
| `actor_rollout_ref.actor.policy_loss.clip_cov_lb` | Lower bound for clip-cov loss.<br>**Default**: 1.0 |
| `actor_rollout_ref.actor.policy_loss.clip_cov_ub` | Upper bound for clip-cov loss.<br>**Default**: 5.0 |
| `actor_rollout_ref.actor.policy_loss.kl_cov_ratio` | Ratio of tokens to apply KL penalty for kl-cov loss.<br>**Default**: 0.0002 |
| `actor_rollout_ref.actor.policy_loss.ppo_kl_coef` | KL divergence penalty coefficient.<br>**Default**: 0.1 |

### Actor Checkpoint Configuration

| Parameter | Description |
| --------- | ----------- |
| `actor_rollout_ref.actor.checkpoint.save_contents` | Contents to include in saved checkpoints. Can be `['model', 'optimizer', 'extra']`. Use `'hf_model'` to save entire model in HF format.<br>**Default**: `['model', 'optimizer', 'extra']` |
| `actor_rollout_ref.actor.checkpoint.load_contents` | Contents to load from checkpoints.<br>**Default**: Same as `save_contents` |
| `actor_rollout_ref.actor.checkpoint.async_save` | Whether to save checkpoints asynchronously. Only effective for Megatron.<br>**Default**: `False` |

### Actor Router Replay Configuration (MoE Models)

| Parameter | Description |
| --------- | ----------- |
| `actor_rollout_ref.actor.router_replay.mode` | Router replay mode: `disabled`, `R2`, `R3`.<br>**Default**: `disabled` |
| `actor_rollout_ref.actor.router_replay.record_file` | File path to save recorded routing decisions. Required when mode is `R2` or `R3`. |
| `actor_rollout_ref.actor.router_replay.replay_file` | File path to load recorded routing decisions for replay. |

### Actor Optimizer Configuration

| Parameter | Description |
| --------- | ----------- |
| `actor_rollout_ref.actor.optim.lr` | Learning rate.<br>**Default**: 1e-6 |
| `actor_rollout_ref.actor.optim.lr_warmup_steps` | Warmup steps; negative value defers to `lr_warmup_steps_ratio`.<br>**Default**: -1 |
| `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio` | Warmup steps ratio (used when `lr_warmup_steps` is negative).<br>**Default**: 0.0 |
| `actor_rollout_ref.actor.optim.total_training_steps` | Total training steps (overridden at runtime).<br>**Default**: -1 |
| `actor_rollout_ref.actor.optim.weight_decay` | Weight decay coefficient.<br>**Default**: 0.01 |

---

## Reference Model Configuration

Reference model is enabled when `actor.use_kl_loss` or `algorithm.use_kl_in_reward` is True.

| Parameter | Description |
| --------- | ----------- |
| `actor_rollout_ref.ref.strategy` | Reference model strategy, same as actor.<br>**Default**: Same as `actor_rollout_ref.actor.strategy` |
| `actor_rollout_ref.ref.rollout_n` | Number of rollouts per update.<br>**Default**: Same as `actor_rollout_ref.rollout.n` |
| `actor_rollout_ref.ref.use_torch_compile` | Whether to enable torch.compile.<br>**Default**: Same as `actor_rollout_ref.actor.use_torch_compile` |
| `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | Batch size for one forward pass in log_prob computation (local per GPU).<br>**Default**: `null` |
| `actor_rollout_ref.ref.log_prob_use_dynamic_bsz` | Whether to enable dynamic batch size (sequence packing) for log_prob computation.<br>**Default**: Same as `actor_rollout_ref.actor.use_dynamic_bsz` |
| `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu` | Max token length per GPU in log_prob computation.<br>**Default**: Same as `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` |

---

## Critic Configuration

| Parameter | Description |
| --------- | ----------- |
| `critic.enable` | Whether to enable critic worker. By default, only enabled if advantage estimator is `gae`. Set to `True` to always enable critic worker.<br>**Default**: `null` (auto) |
| `critic.strategy` | FSDP/FSDP2/Megatron strategy for critic model training. Must be set. |
| `critic.rollout_n` | Number of rollouts per update.<br>**Default**: Same as `actor_rollout_ref.rollout.n` |
| `critic.model.path` | Path to pretrained model weights. |
| `critic.model.tokenizer_path` | Tokenizer path (defaults to actor's model path). |
| `critic.model.override_config` | Hugging Face config override.<br>**Default**: `{}` |
| `critic.model.external_lib` | External model implementation (optional). |
| `critic.model.trust_remote_code` | Whether to trust remote code for Hugging Face models.<br>**Default**: Same as `actor_rollout_ref.model.trust_remote_code` |
| `critic.ppo_mini_batch_size` | PPO mini-batch size per update.<br>**Default**: Same as `actor_rollout_ref.actor.ppo_mini_batch_size` |
| `critic.ppo_micro_batch_size_per_gpu` | Local per-GPU micro batch size. |
| `critic.use_dynamic_bsz` | Whether to automatically adjust batch size at runtime.<br>**Default**: Same as `actor_rollout_ref.actor.use_dynamic_bsz` |
| `critic.ppo_max_token_len_per_gpu` | Max tokens per GPU in one PPO batch.<br>**Default**: 32768 |
| `critic.forward_max_token_len_per_gpu` | Max token length per GPU in forward pass.<br>**Default**: Same as `ppo_max_token_len_per_gpu` |
| `critic.ppo_epochs` | Number of PPO epochs per batch.<br>**Default**: Same as `actor_rollout_ref.actor.ppo_epochs` |
| `critic.shuffle` | Shuffle training data across PPO epochs.<br>**Default**: Same as `actor_rollout_ref.actor.shuffle` |
| `critic.cliprange_value` | PPO value function clipping range.<br>**Default**: 0.5 |
| `critic.loss_agg_mode` | Loss aggregation mode: `token-mean`, `seq-mean-token-sum`, or `seq-mean-token-mean`.<br>**Default**: Same as `actor_rollout_ref.actor.loss_agg_mode` |
| `critic.optim.lr` | Learning rate.<br>**Default**: 1e-5 |
| `critic.optim.lr_warmup_steps_ratio` | Warmup steps ratio.<br>**Default**: 0.0 |
| `critic.optim.weight_decay` | Weight decay.<br>**Default**: 0.01 |

---

## Reward Model Configuration

| Parameter | Description |
| --------- | ----------- |
| `reward_model.enable` | Whether to enable reward model. If `False`, only compute reward with user-defined reward functions. In GSM8K and Math examples, we disable reward model. For RLHF alignment examples using full_hh_rlhf, we use reward model to assess responses.<br>**Default**: `False` |
| `reward_model.enable_resource_pool` | Whether to deploy model to separate resource pool. If true, `n_gpus_per_node` & `nnodes` determine resource node.<br>**Default**: `False` |
| `reward_model.n_gpus_per_node` | GPUs per node in resource pool.<br>**Default**: 0 |
| `reward_model.nnodes` | Number of nodes in resource pool.<br>**Default**: 0 |
| `reward_model.strategy` | FSDP strategy: `fsdp`, `fsdp2`, or `megatron`. Must be set. |
| `reward_model.model.input_tokenizer` | Input tokenizer. If reward model's chat template differs from policy, need to decode to plaintext first, apply RM's chat_template, then score with RM. If templates are consistent, can set to `null`.<br>**Default**: Same as `actor_rollout_ref.model.path` |
| `reward_model.model.path` | RM's HDFS or local path. Note RM only supports `AutoModelForSequenceClassification`. Other model types need to define their own RewardModelWorker and pass from code. |
| `reward_model.model.external_lib` | External model implementation (optional). |
| `reward_model.model.trust_remote_code` | Whether to allow loading remote code model.<br>**Default**: `False` |
| `reward_model.model.override_config` | Override HF config.<br>**Default**: `{}` |
| `reward_model.micro_batch_size_per_gpu` | Local per-GPU micro batch size. |
| `reward_model.max_length` | Maximum sequence length to process for scoring. |
| `reward_model.use_dynamic_bsz` | Whether to dynamically adjust batch size at runtime.<br>**Default**: Same as `critic.use_dynamic_bsz` |
| `reward_model.forward_max_token_len_per_gpu` | Maximum tokens per GPU in one forward pass.<br>**Default**: Same as `critic.forward_max_token_len_per_gpu` |
| `reward_model.reward_manager` | **Deprecated**. Use `reward_manager.name` instead. Defines mechanism for computing rule-based rewards and handling different reward sources.<br>**Default**: `naive` |
| `reward_model.launch_reward_fn_async` | Whether to launch custom reward function asynchronously during log_prob. Custom reward function executed async on CPU.<br>**Default**: `False` |
| `reward_model.sandbox_fusion.url` | Cloud/local function URL for sandbox execution. |
| `reward_model.sandbox_fusion.max_concurrent` | Max concurrent requests allowed to sandbox.<br>**Default**: 64 |
| `reward_model.sandbox_fusion.memory_limit_mb` | Max memory limit for each sandbox process in MB.<br>**Default**: 1024 |

---

## Custom Reward Function

| Parameter | Description |
| --------- | ----------- |
| `custom_reward_function.path` | Path to file containing custom reward function. If not specified, pre-implemented reward functions will be used. |
| `custom_reward_function.name` | Reward function name within specified file.<br>**Default**: `compute_score` |

---

## Reward Manager Configuration

| Parameter | Description |
| --------- | ----------- |
| `reward_manager.source` | Reward manager source: `register` or `module`.<br>**Default**: `register` |
| `reward_manager.name` | Reward manager name.<br>**Default**: Same as `reward_model.reward_manager` (typically `naive`) |
| `reward_manager.module.path` | Module path containing custom reward manager. |
| `reward_manager.module.name` | Custom reward manager class name.<br>**Default**: `custom_reward_manager` |

---

## Algorithm Configuration

| Parameter | Description |
| --------- | ----------- |
| `algorithm.gamma` | Discount factor for future rewards.<br>**Default**: 1.0 |
| `algorithm.lam` | Trade-off between bias and variance in GAE estimator.<br>**Default**: 1.0 |
| `algorithm.adv_estimator` | Advantage estimator type: `gae`, `grpo`, `reinforce_plus_plus`, `rloo`, etc.<br>**Default**: `gae` |
| `algorithm.norm_adv_by_std_in_grpo` | Whether to normalize advantages by std in GRPO.<br>**Default**: `True` |
| `algorithm.use_kl_in_reward` | Whether to enable in-reward KL penalty.<br>**Default**: `False` |
| `algorithm.kl_penalty` | How to estimate KL divergence: `kl`, `abs`, `mse`, `low_var_kl`, or `full`.<br>**Default**: `kl` |
| `algorithm.kl_ctrl.type` | KL control type: `fixed` or `adaptive`.<br>**Default**: `fixed` |
| `algorithm.kl_ctrl.kl_coef` | Initial coefficient for KL penalty.<br>**Default**: 0.001 |
| `algorithm.kl_ctrl.horizon` | Horizon value for adaptive controller (if enabled).<br>**Default**: 10000 |
| `algorithm.kl_ctrl.target_kl` | Target KL divergence (for adaptive controller).<br>**Default**: 0.1 |
| `algorithm.use_pf_ppo` | Whether to enable preference feedback PPO.<br>**Default**: `False` |
| `algorithm.pf_ppo.reweight_method` | Sample reweighting method: `pow`, `max_min`, or `max_random`.<br>**Default**: `pow` |
| `algorithm.pf_ppo.weight_pow` | Power for weight scaling in `pow` method.<br>**Default**: 2.0 |

---

## Rollout Correction

For correcting off-policy distribution shifts. See documentation: `docs/algo/rollout_corr.md`

| Parameter | Description |
| --------- | ----------- |
| `algorithm.rollout_correction.rollout_is` | IS aggregation level: `null` (disabled), `token` (per-token), `sequence` (per-sequence).<br>**Default**: `null` |
| `algorithm.rollout_correction.rollout_is_threshold` | Upper threshold for IS weight truncation (typical: 2.0-5.0).<br>**Default**: 2.0 |
| `algorithm.rollout_correction.rollout_rs` | RS aggregation level: `null` (disabled), `token`, `sequence`, `geometric`.<br>**Default**: `null` |
| `algorithm.rollout_correction.rollout_rs_threshold` | Upper threshold for rejection sampling (`null` = use `rollout_is_threshold`).<br>**Default**: `null` |
| `algorithm.rollout_correction.rollout_rs_threshold_lower` | Lower threshold for rejection sampling (`null` = auto-compute as 1/upper).<br>**Default**: `null` |
| `algorithm.rollout_correction.rollout_token_veto_threshold` | Per-token veto threshold for catastrophic outliers (`null` = disabled).<br>**Default**: `null` |
| `algorithm.rollout_correction.bypass_mode` | Operating mode: `false` = Decoupled (3 policies), `true` = Bypass (2 policies).<br>**Default**: `false` |
| `algorithm.rollout_correction.use_policy_gradient` | Loss function: `false` = PPO with clipping, `true` = Policy gradient (no clipping).<br>**Default**: `false` |
| `algorithm.rollout_correction.rollout_is_batch_normalize` | Batch normalize IS weights: `false` = raw weights, `true` = normalize to mean=1.0.<br>**Default**: `false` |

---

## Trainer Configuration

| Parameter | Description |
| --------- | ----------- |
| `trainer.balance_batch` | Whether to balance batch sizes across distributed workers.<br>**Default**: `True` |
| `trainer.total_epochs` | Total number of training epochs.<br>**Default**: 30 |
| `trainer.total_training_steps` | Total training steps (can be set explicitly or derived from epochs).<br>**Default**: `null` |
| `trainer.project_name` | Project name for experiment tracking (e.g., wandb).<br>**Default**: `verl_examples` |
| `trainer.experiment_name` | Experiment name for run identification in tracking tools.<br>**Default**: `gsm8k` |
| `trainer.logger` | Logging backends to use: `console`, `wandb`, etc.<br>**Default**: `["console", "wandb"]` |
| `trainer.log_val_generations` | Number of generations to log during validation.<br>**Default**: 0 |
| `trainer.rollout_data_dir` | Directory for logging rollout data; no dump if `null`.<br>**Default**: `null` |
| `trainer.validation_data_dir` | Directory for logging validation data; no dump if `null`.<br>**Default**: `null` |
| `trainer.nnodes` | Number of nodes used in training.<br>**Default**: 1 |
| `trainer.n_gpus_per_node` | Number of GPUs per node.<br>**Default**: 8 |
| `trainer.save_freq` | Save frequency (by iteration) for model checkpoints. -1 means no saving.<br>**Default**: -1 |
| `trainer.esi_redundant_time` | ESI (elastic server instance) redundant time. To ensure checkpoint is saved before ESI shuts down, system starts saving in advance. Advance time = Longest historical step duration + Checkpoint save duration + esi_redundant_time.<br>**Default**: 0 |
| `trainer.resume_mode` | Resume mode: `auto` (resume from last checkpoint if available), `disable` (start from scratch), or `resume_path` (resume from user-defined path).<br>**Default**: `auto` |
| `trainer.resume_from_path` | Path to resume training from (only used when `resume_mode` is `resume_path`).<br>**Default**: `null` |
| `trainer.val_before_train` | Whether to run validation before training begins.<br>**Default**: `True` |
| `trainer.val_only` | Whether to run validation only.<br>**Default**: `False` |
| `trainer.test_freq` | Validation frequency (in training iterations). -1 means no validation.<br>**Default**: -1 |
| `trainer.critic_warmup` | Number of iterations to warm up critic before updating policy.<br>**Default**: 0 |
| `trainer.default_hdfs_dir` | Default path to distributed filesystem for saving checkpoints.<br>**Default**: `null` |
| `trainer.del_local_ckpt_after_load` | Whether to delete local checkpoints after loading.<br>**Default**: `False` |
| `trainer.default_local_dir` | Default local directory for saving checkpoints.<br>**Default**: `checkpoints/${trainer.project_name}/${trainer.experiment_name}` |
| `trainer.max_actor_ckpt_to_keep` | Maximum number of actor checkpoints to keep.<br>**Default**: `null` |
| `trainer.max_critic_ckpt_to_keep` | Maximum number of critic checkpoints to keep.<br>**Default**: `null` |
| `trainer.ray_wait_register_center_timeout` | Timeout (in seconds) for Ray worker to wait for registration.<br>**Default**: 300 |
| `trainer.device` | Device to run training on (e.g., `cuda`, `cpu`).<br>**Default**: `cuda` |
| `trainer.use_legacy_worker_impl` | Whether to use legacy worker implementation. Mode: `auto`, `enable`, or `disable`.<br>**Default**: `auto` |

---

## FSDP Engine Configuration

| Parameter | Description |
| --------- | ----------- |
| `*.fsdp_config.wrap_policy.min_num_params` | Minimum number of parameters to trigger wrapping a layer with FSDP.<br>**Default**: 0 |
| `*.fsdp_config.param_offload` | Whether to offload model parameters to CPU (trades speed for memory). Note this differs from FSDP's `offload_policy`.<br>**Default**: `false` |
| `*.fsdp_config.optimizer_offload` | Whether to offload optimizer state to CPU.<br>**Default**: `false` |
| `*.fsdp_config.offload_policy` | **FSDP2 only**: Offload param/grad/optimizer during training.<br>**Default**: `false` |
| `*.fsdp_config.reshard_after_forward` | **FSDP2 only**: Reshard after forward pass to reduce memory footprint.<br>**Default**: `true` |
| `*.fsdp_config.fsdp_size` | Number of GPUs in each FSDP shard group; -1 means auto.<br>**Default**: -1 |
| `*.fsdp_config.forward_prefetch` | **FSDP1 only**: Prefetch next forward-pass all-gather before current forward computation completes.<br>**Default**: `False` |
| `*.fsdp_config.model_dtype` | Model dtype for FSDP.<br>**Default**: `fp32` |
| `*.fsdp_config.use_orig_params` | Whether to use original parameters in FSDP. Only available in FSDP1.<br>**Default**: `false` |
| `*.fsdp_config.seed` | Random seed for reproducibility.<br>**Default**: 42 |
| `*.fsdp_config.full_determinism` | Whether to enable full determinism for distributed training, only for debugging.<br>**Default**: `false` |
| `*.fsdp_config.ulysses_sequence_parallel_size` | Ulysses sequence parallel size.<br>**Default**: 1 |
| `*.fsdp_config.entropy_from_logits_with_chunking` | Whether to use `entropy_from_logits_with_chunking` in FSDP.<br>**Default**: `false` |
| `*.fsdp_config.use_torch_compile` | Whether to use torch compile in FSDP.<br>**Default**: `true` |
| `*.fsdp_config.entropy_checkpointing` | Whether to use entropy checkpointing in FSDP.<br>**Default**: `false` |
| `*.fsdp_config.forward_only` | Whether to use forward only in FSDP.<br>**Default**: `false` |
| `*.fsdp_config.strategy` | Strategy type: `fsdp` or `fsdp2`.<br>**Default**: `fsdp` |
| `*.fsdp_config.dtype` | Mixed precision training param dtype.<br>**Default**: `bfloat16` |

---

## Megatron Engine Configuration

| Parameter | Description |
| --------- | ----------- |
| `*.megatron.param_offload` | Whether to offload model parameters to CPU.<br>**Default**: `False` |
| `*.megatron.grad_offload` | Whether to offload gradients to CPU.<br>**Default**: `False` |
| `*.megatron.optimizer_offload` | Whether to offload optimizer state to CPU.<br>**Default**: `False` |
| `*.megatron.tensor_model_parallel_size` | Tensor model parallel size.<br>**Default**: 1 |
| `*.megatron.expert_model_parallel_size` | Expert model parallel size.<br>**Default**: 1 |
| `*.megatron.expert_tensor_parallel_size` | Expert tensor parallel size (`null` = same as TP).<br>**Default**: `null` |
| `*.megatron.pipeline_model_parallel_size` | Pipeline model parallel size.<br>**Default**: 1 |
| `*.megatron.virtual_pipeline_model_parallel_size` | Virtual pipeline model parallel size.<br>**Default**: `null` |
| `*.megatron.context_parallel_size` | Context parallel size.<br>**Default**: 1 |
| `*.megatron.sequence_parallel` | Whether to enable sequence parallelism.<br>**Default**: `True` |
| `*.megatron.use_distributed_optimizer` | Whether to use distributed optimizer.<br>**Default**: `True` |
| `*.megatron.use_dist_checkpointing` | Whether to use distributed checkpointing.<br>**Default**: `False` |
| `*.megatron.dist_checkpointing_path` | Distributed checkpointing path.<br>**Default**: `null` |
| `*.megatron.dist_checkpointing_prefix` | Distributed checkpointing prefix, e.g., Nemo2 appends prefix 'module.' to state dict keys.<br>**Default**: `''` |
| `*.megatron.seed` | Random seed.<br>**Default**: 42 |
| `*.megatron.override_ddp_config` | Allow overriding Distributed Data Parallel (DDP) config.<br>**Default**: `{}` |
| `*.megatron.override_transformer_config.recompute_granularity` | Recompute granularity, options: `full`, `selective`.<br>**Default**: `null` |
| `*.megatron.override_transformer_config.recompute_modules` | Recompute modules, options: `core_attn`, `moe_act`, `layernorm`, `mla_up_proj`, `mlp`, `moe`.<br>**Default**: `["core_attn"]` |
| `*.megatron.override_transformer_config.recompute_method` | Recompute method: `uniform`, `block`.<br>**Default**: `null` |
| `*.megatron.override_transformer_config.recompute_num_layers` | Number of layers to recompute.<br>**Default**: `null` |
| `*.megatron.override_transformer_config.attention_backend` | Attention backend (`flash`, `fused`, `unfused`, `local`, `auto`).<br>**Default**: `flash` |
| `*.megatron.override_mcore_model_config` | Override MCore model config.<br>**Default**: `{}` |
| `*.megatron.use_mbridge` | Whether to use MBridge.<br>**Default**: `False` |
| `*.megatron.vanilla_mbridge` | Whether to use vanilla MBridge.<br>**Default**: `True` |
| `*.megatron.use_remove_padding` | Whether to use thd format (sequence packing). If not, use bshd format, padding input_ids to longest sequence length.<br>**Default**: `True` |
| `*.megatron.forward_only` | Whether to use forward only.<br>**Default**: `False` |
| `*.megatron.dtype` | Mixed precision training param dtype.<br>**Default**: `bfloat16` |

---

## Optimizer Configuration

### FSDP Optimizer

| Parameter | Description |
| --------- | ----------- |
| `*.optim.optimizer` | Optimizer class name (e.g., `AdamW`, `AdamW8bit`, `_AdamW`, `Adam`).<br>**Default**: `AdamW` |
| `*.optim.optimizer_impl` | Module path to import optimizer. Examples: `torch.optim`, `torchao.optim`, `bitsandbytes.optim`.<br>**Default**: `torch.optim` |
| `*.optim.lr` | Learning rate.<br>**Default**: 1e-3 |
| `*.optim.lr_warmup_steps_ratio` | LR warmup steps ratio.<br>**Default**: 0.0 |
| `*.optim.total_training_steps` | Total training steps.<br>**Default**: -1 |
| `*.optim.weight_decay` | Weight decay.<br>**Default**: 0.01 |
| `*.optim.lr_warmup_steps` | LR warmup steps.<br>**Default**: -1 |
| `*.optim.betas` | Betas for Adam optimizer.<br>**Default**: `[0.9, 0.999]` |
| `*.optim.clip_grad` | Gradient clipping.<br>**Default**: 1.0 |
| `*.optim.min_lr_ratio` | Minimum LR ratio for cosine schedule.<br>**Default**: 0.0 |
| `*.optim.num_cycles` | Number of cosine cycles in LR schedule.<br>**Default**: 0.5 |
| `*.optim.lr_scheduler_type` | LR scheduler type: `constant` or `cosine`.<br>**Default**: `constant` |
| `*.optim.override_optimizer_config` | Additional optimizer-specific keyword arguments. E.g., for torchao bf16 stochastic rounding.<br>**Default**: `null` |

### Megatron Optimizer

| Parameter | Description |
| --------- | ----------- |
| `*.optim.optimizer` | Optimizer type: `adam`.<br>**Default**: `adam` |
| `*.optim.lr` | Learning rate.<br>**Default**: 1e-3 |
| `*.optim.lr_warmup_steps_ratio` | LR warmup steps ratio.<br>**Default**: 0.0 |
| `*.optim.lr_warmup_steps` | LR warmup steps.<br>**Default**: -1 |
| `*.optim.lr_warmup_init` | Initial learning rate for warmup.<br>**Default**: 0.0 |
| `*.optim.lr_decay_steps` | LR decay steps.<br>**Default**: `null` |
| `*.optim.lr_decay_style` | LR decay style: `constant`, `linear`, `cosine`, `inverse_square_root`.<br>**Default**: `constant` |
| `*.optim.min_lr` | Minimum learning rate.<br>**Default**: 0.0 |
| `*.optim.weight_decay` | Weight decay.<br>**Default**: 0.01 |
| `*.optim.weight_decay_incr_style` | Weight decay increase style: `constant`, `linear`, `cosine`.<br>**Default**: `constant` |
| `*.optim.lr_wsd_decay_style` | LR WSD decay style: `constant`, `exponential`, `cosine`.<br>**Default**: `exponential` |
| `*.optim.lr_wsd_decay_steps` | LR WSD decay steps.<br>**Default**: `null` |
| `*.optim.betas` | Betas for Adam optimizer.<br>**Default**: `[0.9, 0.999]` |
| `*.optim.clip_grad` | Gradient clipping.<br>**Default**: 1.0 |
| `*.optim.use_checkpoint_opt_param_scheduler` | Whether to use checkpoint optimizer parameter scheduler.<br>**Default**: `False` |
| `*.optim.override_optimizer_config` | Override optimizer config.<br>**Default**: `{}` |

---

## Profiler Configuration

### Global Profiler

| Parameter | Description |
| --------- | ----------- |
| `global_profiler.tool` | Profiling tool: `nsys`, `npu`, `torch`, `torch_memory`.<br>**Default**: `null` |
| `global_profiler.steps` | List of steps to profile.<br>**Default**: `null` |
| `global_profiler.profile_continuous_steps` | Whether to combine continuous steps into one database. If `True`, `worker.profiler.discrete` must be `False`.<br>**Default**: `False` |
| `global_profiler.save_path` | Path to save profiling contents.<br>**Default**: `outputs/profile` |

### Nsight Systems Configuration

| Parameter | Description |
| --------- | ----------- |
| `global_profiler.global_tool_config.nsys.discrete` | `True` for each task has its own database, `False` for all tasks share one.<br>**Default**: `False` |
| `global_profiler.global_tool_config.nsys.controller_nsight_options.trace` | Select APIs to trace (e.g., cuda, nvtx, cublas, ucx).<br>**Default**: `cuda,nvtx,cublas,ucx` |
| `global_profiler.global_tool_config.nsys.controller_nsight_options.cuda-memory-usage` | Whether to track CUDA memory usage. Must be string `"true"` or `"false"`.<br>**Default**: `"true"` |
| `global_profiler.global_tool_config.nsys.controller_nsight_options.cuda-graph-trace` | CUDA graphs trace mode.<br>**Default**: `"graph"` |
| `global_profiler.global_tool_config.nsys.worker_nsight_options.trace` | Select APIs to trace.<br>**Default**: `cuda,nvtx,cublas,ucx` |
| `global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range` | Profile only in `torch.cuda.profiler.start` and `stop` range. Do not change this config.<br>**Default**: `cudaProfilerApi` |
| `global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range-end` | Specify desired behavior when capture range ends. Valid values: `"repeat-shutdown:n"` or `null`.<br>**Default**: `null` |
| `global_profiler.global_tool_config.nsys.worker_nsight_options.kill` | Send signal to target application's process group.<br>**Default**: `none` |

### Torch Memory Profiler Configuration

| Parameter | Description |
| --------- | ----------- |
| `global_profiler.global_tool_config.torch_memory.trace_alloc_max_entries` | Maximum number of allocation entries to record.<br>**Default**: 100000 |
| `global_profiler.global_tool_config.torch_memory.stack_depth` | Depth of call stack to capture for each allocation.<br>**Default**: 32 |
| `global_profiler.global_tool_config.torch_memory.context` | `alloc`: records only allocation events, `state`: records memory state changes, `all`: records both.<br>**Default**: `all` |
| `global_profiler.global_tool_config.torch_memory.stacks` | `python`: records Python stacks, `cpp`: records C++ stacks, `all`: records both.<br>**Default**: `all` |

### Component-level Profiler Configuration

Actor, Critic, Ref, Reward Model, and Rollout all have their own profiler configs:

| Parameter | Description |
| --------- | ----------- |
| `*.profiler.tool` | Profiling tool, default same as `global_profiler.tool`. |
| `*.profiler.enable` | Whether to enable profiling for this component.<br>**Default**: `False` |
| `*.profiler.all_ranks` | Whether to profile all ranks.<br>**Default**: `False` |
| `*.profiler.ranks` | List of ranks to profile. `[]` or `[0,1,...]`.<br>**Default**: `[]` |
| `*.profiler.save_path` | Path to save profiling results. |

---

## Transfer Queue Configuration

| Parameter | Description |
| --------- | ----------- |
| `transfer_queue.enable` | Whether to enable transfer queue.<br>**Default**: `False` |

---

## Ray Configuration

| Parameter | Description |
| --------- | ----------- |
| `ray_kwargs.ray_init.num_cpus` | Number of CPUs for Ray. Use fixed number instead of `null` when using SLURM. `null` means using all CPUs, which might cause hang in systems like SLURM.<br>**Default**: `null` |
| `ray_kwargs.timeline_json_file` | Path to save Ray timeline JSON for performance profiling.<br>**Default**: `null` |

---

## Prometheus Configuration (Server Mode)

| Parameter | Description |
| --------- | ----------- |
| `actor_rollout_ref.rollout.prometheus.enable` | Whether to enable prometheus on server mode rollout.<br>**Default**: `false` |
| `actor_rollout_ref.rollout.prometheus.port` | Port number that Prometheus listens on.<br>**Default**: 9090 |
| `actor_rollout_ref.rollout.prometheus.file` | Path to Prometheus configuration file.<br>**Default**: `/tmp/ray/session_latest/metrics/prometheus/prometheus.yml` |
| `actor_rollout_ref.rollout.prometheus.served_model_name` | Specify served_model_name to avoid displaying overly long model paths in Grafana.<br>**Default**: Same as `actor_rollout_ref.model.path` |

---

## Appendix: Configuration Best Practices

### Key Parameters for Preventing OOM

1. `ppo_micro_batch_size_per_gpu`: Adjust based on GPU memory
2. `gpu_memory_utilization`: For SGLang >= 0.4.8.post1, can set to 0.85
3. `enable_gradient_checkpointing`: Enable to trade time for space
4. `param_offload` / `optimizer_offload`: Offload parameters/optimizer to CPU

### Required Configuration for GRPO

```yaml
algorithm:
  adv_estimator: grpo
actor_rollout_ref:
  rollout:
    n: 5  # > 1 for group sampling
  actor:
    use_kl_loss: True
```

### Recommended Configuration for Multi-turn

```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    multi_turn:
      enable: True
      tool_config_path: /path/to/tools.json
```

---

## Version Notes

This document is organized based on verl source code analysis and may differ due to framework updates. For questions, please refer to YAML configuration files in the `verl/trainer/config/` directory.
