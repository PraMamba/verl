# verl 参数速览

> 基于 verl 源码分析整理，参考了 [Awesome-ML-SYS-Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial) 的文档结构
>
> 参与者：Ji Li（蚂蚁），Zhuoran Yin（CMU），Changyi Yang（CMU），Chengxi Li（CMU），Xinpeng Wei（Amazon），Chenyang Zhao（Amazon）

由于 [Hydra 的使用](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme.md#脚本配置)，verl 的参数分散在整个框架各地。本文档基于 verl 源码 `verl/trainer/config/` 目录下的 YAML 配置文件进行整理。

---

## 目录

- [Batch Size 相关参数](#batch-size-相关参数)
- [Dynamic Batch Size](#dynamic-batch-size)
- [Rollout 采样参数](#rollout-采样参数)
- [Rollout 性能与资源管理](#rollout-性能与资源管理)
- [SGLang 配置](#sglang-配置)
- [Multi-turn 多轮对话配置](#multi-turn-多轮对话配置)
- [验证阶段配置](#验证阶段配置)
- [Dataset 数据集配置](#dataset-数据集配置)
- [Model 模型配置](#model-模型配置)
- [Actor 配置](#actor-配置)
- [Reference 模型配置](#reference-模型配置)
- [Critic 配置](#critic-配置)
- [Reward Model 配置](#reward-model-配置)
- [Custom Reward Function 自定义奖励函数](#custom-reward-function-自定义奖励函数)
- [Algorithm 算法配置](#algorithm-算法配置)
- [Rollout Correction 离策略校正](#rollout-correction-离策略校正)
- [Trainer 训练器配置](#trainer-训练器配置)
- [FSDP Engine 配置](#fsdp-engine-配置)
- [Megatron Engine 配置](#megatron-engine-配置)
- [Optimizer 优化器配置](#optimizer-优化器配置)
- [Profiler 性能分析配置](#profiler-性能分析配置)
- [Ray 配置](#ray-配置)

---

## Batch Size 相关参数

| 参数名称 | 详细解释 |
| ------- | ------- |
| `data.train_batch_size` | **作用**：定义了单次训练发送给 Rollout Engine 的样本数量，也即这是在每个 PPO 迭代开始时，从训练数据集中采样的提示（Prompt）数量。<br><br>**详细解释**：这个值是 RL 训练中的基本样本数量。例如，设置为 1024 意味着在一次迭代中会：<br>1. 从数据集中随机抽取 1024 个 prompt<br>2. 将这 1024 个 prompt 发送给当前的 Rollout Engine，从而得到 1024 组完整的 trajectories（prompt, response）<br>3. 接下来，这 1024 个 trajectories 进行经验计算（make experience），后续用于 Actor 和 Critic 模型的更新<br><br>**默认值**：1024 |
| `data.val_batch_size` | **作用**：在 Validation 阶段使用的批次大小。<br><br>**详细解释**：设置为 `null` 时，整个 validation dataset 一次性发给推理引擎，由引擎自行进行内存管理。推荐设置为 `null`。<br><br>**默认值**：`null` |
| `actor_rollout_ref.actor.ppo_mini_batch_size` / `critic.ppo_mini_batch_size` | **作用**：定义了 PPO 训练更新中的 mini-batch 大小。<br><br>**详细解释**：`data.train_batch_size` 收集到的全部经验数据将被分割成多个 mini-batch，每块的大小就是 `ppo_mini_batch_size`。模型每处理完一个 mini-batch，才会进行一次参数更新。<br><br>例如，如果 `train_batch_size = 1024`，`ppo_mini_batch_size = 256`，那么在一个 PPO Epoch 中，模型会进行 `1024 / 256 = 4` 次参数更新。<br><br>**影响与权衡**：增大 mini-batch，单次更新的梯度更稳定，但更新频率更低，更新次数减少。<br><br>**默认值**：256 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` / `critic.ppo_micro_batch_size_per_gpu` | **作用**：定义了在单个 GPU 上进行一次 forward/backward 的数据大小。<br><br>**详细解释**：这是实现梯度累积的核心参数。mini-batch 会被再次切分为若干个 micro-batch。例如，在单卡上，`ppo_mini_batch_size = 256`，`ppo_micro_batch_size_per_gpu = 32`，那么梯度累积的步数就是 `256 / 32 = 8`。这意味着模型会运行 8 次 forward 得到 loss，然后 backward 得到 gradient。每次处理 32 个样本，直到累积完整个 mini-batch 计算出的梯度。此时，使用累积的总梯度，对模型参数进行一次更新（`optimizer.step()`）。<br><br>**影响与权衡**：增大此值，减少了梯度累积的次数，可以提高训练的吞吐量，但会增大显存消耗。这个值必须根据显存大小来严格调整，是防止 OOM 的关键。<br><br>**默认值**：`null` |
| `actor_rollout_ref.actor.ppo_micro_batch_size` / `critic.ppo_micro_batch_size` | **已弃用**：被 `*_per_gpu` 版本取代，因为它能更好地适应分布式训练环境。 |

---

## Dynamic Batch Size

当样本长度差异很大时，按样本数量划分批次可能导致不同批次的计算量极不均衡，而基于 token 总数来控制 batch size 是一种平衡每个 batch 训练时间的方案。

| 参数名称 | 详细解释 |
| ------- | ------- |
| `actor_rollout_ref.actor.use_dynamic_bsz` / `critic.use_dynamic_bsz` / `reward_model.use_dynamic_bsz` | **作用**：是否启用 Dynamic Batch Size。<br><br>**详细解释**：当此项为 `True` 时，系统会忽略基于样本数的 `micro_batch_size_per_gpu` 参数，转而使用基于 Token 数的 `max_token_len_per_gpu` 参数来构建 batch。<br><br>**默认值**：`false` |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` / `critic.ppo_max_token_len_per_gpu` | **作用**：定义了一个 PPO micro batch 中，单个 GPU 能处理的最大 Token 总数。<br><br>**详细解释**：这是 `ppo_micro_batch_size_per_gpu` 的替代方案，与 `use_dynamic_bsz` 配合使用。系统会自动打包样本，直到总 Token 量（`prompt_len + response_len`）接近这个阈值，形成一个动态的 micro batch size，从而稳定计算效率。<br><br>**推荐设置**：通常设置为 `n * (max_prompt_length + max_response_length)`<br><br>**默认值**：Actor 为 16384，Critic 为 32768 |
| `critic.forward_max_token_len_per_gpu` / `reward_model.forward_max_token_len_per_gpu` / `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu` | **作用**：只进行 forward 计算的模型的一个 micro-batch 的 token 最大数量。<br><br>**详细解释**：一些模型（Reward Model, Critic 求 value, Reference Model 求 log probs）在 make experience 阶段只有 forward 计算，此时 rollout engine 已经 offload 了，而 training engine 还没启动，显存占用是很少的。因此，可以为它们设置一个更大的 batch size 以加速计算。 |
| `critic.forward_micro_batch_size_per_gpu` / `reward_model.micro_batch_size_per_gpu` / `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | **作用**：同样为只进行 forward 计算的 model 设置 micro-batch size（基于样本数量）。 |
| `trainer.balance_batch` | **作用**：是否在分布式训练的各个 dp rank 间平衡 batch size。<br><br>**详细解释**：在 single controller 上将 data 重新排序使得每个 dp rank 获得相似数目的 token。<br><br>**默认值**：`True` |

---

## Rollout 采样参数

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `actor_rollout_ref.rollout.name` | 推理引擎类型：`hf`/`vllm`/`sglang`。必须设置。 |
| `actor_rollout_ref.rollout.mode` | 推理模式：`sync`（同步 LLM）或 `async`（异步 AsyncLLM）。<br>**默认值**：`async` |
| `actor_rollout_ref.rollout.temperature` | temperature 值越高，概率分布越平滑，生成结果更多样、更随机；值越低，分布越尖锐，生成结果更倾向于高概率词元。`temperature=0` 通常等同于 Greedy Decoding。<br>**默认值**：1.0 |
| `actor_rollout_ref.rollout.top_k` | 在每一步生成时，只考虑概率最高的 K 个 token 进行采样。例如，`top_k=50` 表示只从概率前 50 的 token 中选择。<br>- 禁用时：在 vLLM/SGLang 中设置为 `-1`，在 HF 中设置为 `0`。<br>**默认值**：-1 |
| `actor_rollout_ref.rollout.top_p` | 从概率最高的 token 开始累加，直到它们的总概率达到 P，然后从这个 nucleus token 集合中进行采样。是一种动态选择采样范围的方法。`top_p=1.0` 表示不限制。<br>**默认值**：1.0 |
| `actor_rollout_ref.rollout.n` | 为每个 prompt 生成的 response 数量，也即 GRPO 中的 group size。<br>**默认值**：1 |
| `actor_rollout_ref.rollout.ignore_eos` | 是否忽略 EOS (End-of-Sentence) 标记。如果为 `True`，即使模型生成了 EOS 标记，也会继续生成直到达到 `response_length`。<br>**默认值**：`False` |
| `actor_rollout_ref.rollout.do_sample` | 是否在训练 rollout 期间进行采样。`False` 使用贪心采样。<br>**默认值**：`True` |
| `actor_rollout_ref.rollout.calculate_log_probs` | 是否计算 rollout 的 log probs（用于调试或 Truncated Importance Sampling）。<br>**默认值**：`False` |
| `actor_rollout_ref.rollout.over_sample_rate` | 过采样率，控制训练 rollout 的提前终止阈值。当完成 `(1 - over_sample_rate) * total_requests` 个请求后，系统会终止剩余请求。<br>**默认值**：0 |

---

## Rollout 性能与资源管理

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `actor_rollout_ref.rollout.prompt_length` | 最大的 prompt 长度，过长则被截断。<br>**默认值**：与 `data.max_prompt_length` 相同，默认 512 |
| `actor_rollout_ref.rollout.response_length` | 最大的 response 长度，到达最大长度时推理引擎会直接返回。<br>**默认值**：与 `data.max_response_length` 相同，默认 512 |
| `actor_rollout_ref.rollout.dtype` | 模型数据类型。例如 `bfloat16`, `float16`，需要与训练阶段的模型类型对齐。<br>**默认值**：`bfloat16` |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | vLLM/SGLang 中模型参数和 KV Cache 占显存的比例。如果使用 SGLang 0.4.8.post1 以上版本，可以设置到 0.85；使用以下版本则需要设置到 0.5 左右。<br>**默认值**：0.5 |
| `actor_rollout_ref.rollout.free_cache_engine` | Rollout 后是否释放引擎缓存。SGLang 中启用此选项将触发 `flush_cache()` 操作：清空 KV cache pool，将所有 slots 标记为可用。<br>**默认值**：`True` |
| `actor_rollout_ref.rollout.load_format` | 模型权重加载模式：`dummy`（随机初始化权重，用于快速调试）、`hf`、`megatron`、`safetensors`（推荐，安全且高效）。<br>**默认值**：`dummy` |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | 张量并行大小（TP_SIZE），表示用多少个 GPU 来共同运行一个推理引擎。例如，`TP_SIZE=4` 表示将一个大模型的权重切成 4 份，由 4 个 GPU 协同完成推理。HF rollout 不支持此选项。<br>**默认值**：2 |
| `actor_rollout_ref.rollout.data_parallel_size` | 数据并行大小（DP_SIZE）。<br>**默认值**：1 |
| `actor_rollout_ref.rollout.expert_parallel_size` | 专家并行大小（EP_SIZE），用于 MoE 模型。<br>**默认值**：1 |
| `actor_rollout_ref.rollout.pipeline_model_parallel_size` | 流水线并行大小（PP_SIZE）。<br>**默认值**：1 |
| `actor_rollout_ref.rollout.max_model_len` | 模型能处理的最大总长度（prompt + response）；如果未设置，通常由模型配置决定。<br>**默认值**：`null` |
| `actor_rollout_ref.rollout.max_num_seqs` | 引擎能同时处理的最大请求量，或者说同时推理的最多 prompts 数量。<br>**默认值**：1024 |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | 一个批次中的最大 token 数量。<br>**默认值**：8192 |
| `actor_rollout_ref.rollout.enable_chunked_prefill` | 是否启用 Chunked Prefill。对于非常长的 Prompt，可以将其分块处理，减少显存峰值，但可能降低吞吐量。<br>**默认值**：`True` |
| `actor_rollout_ref.rollout.enable_prefix_caching` | 是否启用前缀缓存。Prefix caching 是 LLM 推理中的常见优化，可以避免重复的 prompt 计算。<br>**默认值**：`True` |
| `actor_rollout_ref.rollout.enforce_eager` | 是否禁用 CUDA graph。默认 `False` 以获得最佳性能。<br>**默认值**：`False` |
| `actor_rollout_ref.rollout.cudagraph_capture_sizes` | 要捕获的 CUDA graph 批次大小列表。需要 `enforce_eager: False`。由于推理引擎中的 cudagraph 在更新策略期间无法卸载，可以使用较小的批次大小来节省 CUDA graph 使用的内存。<br>**支持的引擎**：vllm<br>**默认值**：`null` |
| `actor_rollout_ref.rollout.disable_log_stats` | 是否禁用推理引擎的统计日志，以减少控制台输出。<br>**默认值**：`True` |
| `actor_rollout_ref.rollout.multi_stage_wake_up` | 是否在 SGLang 中启用多阶段唤醒推理引擎，以减少训练-rollout 转换期间的峰值内存。仅对 SGLang rollout 有效。<br>**默认值**：`false` |
| `actor_rollout_ref.rollout.update_weights_bucket_megabytes` | 指定 rollout 操作期间批量权重更新的 tensor bucket 大小（以 MB 为单位）。控制单个权重更新请求的最大负载大小。<br>**仅支持 SGLang rollout**<br>**默认值**：512 |
| `actor_rollout_ref.rollout.skip_rollout` | 是否跳过 rollout 计算，尝试从指定目录加载之前生成的 rollout 数据。用于调试或跨多次运行重用计算结果。<br>**默认值**：`False` |
| `actor_rollout_ref.rollout.skip_dump_dir` | 启用 `skip_rollout` 时缓存 rollout 数据的文件系统路径。<br>**默认值**：`/tmp/rollout_dump` |
| `actor_rollout_ref.rollout.skip_tokenizer_init` | 是否跳过 rollout 引擎的 tokenizer 初始化。启用时，rollout 假设 token in token out 进行生成。<br>**默认值**：`True` |
| `actor_rollout_ref.rollout.enable_rollout_routing_replay` | 是否启用 MoE 模型的 rollout routing replay。启用时，rollout 会记录路由决策。<br>**默认值**：`False` |

---

## SGLang 配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `actor_rollout_ref.rollout.engine_kwargs.sglang` | SGLang 引擎的额外配置参数。可以传入任何 SGLang 官方支持的参数。 |
| `actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend` | SGLang 使用的注意力后端。可以选择 `flashinfer`、`triton`、`flashmla`、`null` 等实现，以适应不同显卡。 |
| `actor_rollout_ref.rollout.engine_kwargs.vllm` | vLLM 引擎的额外配置参数。可以传入任何 vLLM 官方支持的参数。 |

---

## Multi-turn 多轮对话配置

这部分参数主要用于需要多轮交互的场景，如工具调用、连续对话等，由 SGLang Engine 支持。

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `actor_rollout_ref.rollout.multi_turn.enable` | 是否启用多轮对话模式。设置 `rollout.name` 为 `sglang`。<br>**默认值**：`False` |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` | 最多进行 assistant 回复的轮次。`null` 时会默认设置成 `max_model_len // 3` 来避免无限对话。<br>**默认值**：`null` |
| `actor_rollout_ref.rollout.multi_turn.max_user_turns` | 最多进行 user 消息的轮次。`null` 时无限制（默认 `max_model_len // 3`）。<br>**默认值**：`null` |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | 工具配置文件路径，定义模型可以调用的外部工具。`null` 表示无工具。<br>**默认值**：`null` |
| `actor_rollout_ref.rollout.multi_turn.interaction_config_path` | 交互配置文件路径。`null` 表示无交互。<br>**默认值**：`null` |
| `actor_rollout_ref.rollout.multi_turn.max_parallel_calls` | 单轮中工具的最大并行调用数。<br>**默认值**：1 |
| `actor_rollout_ref.rollout.multi_turn.max_tool_response_length` | 工具响应的最大长度。<br>**默认值**：256 |
| `actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side` | 工具响应的截断位置：`left`、`middle`、`right`。<br>**默认值**：`middle` |
| `actor_rollout_ref.rollout.multi_turn.use_inference_chat_template` | 是否使用模型在 inference 阶段的 chat template。<br>- `True`：遵循 inference 阶段的模板格式，通常与生产环境行为一致<br>- `False`：使用预训练中的模板，可能包含额外思考过程的完整 Token 序列<br><br>**重要**：对于任何模型，一定要保证在 post training 和后续 inference 进行测试的阶段采用一致的模板。<br>**默认值**：`False` |
| `actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode` | Tokenization 安全性检查模式，检查逐轮 tokenize 的结果与一次 tokenize 整个 chat history 的结果是否一致。<br>- `disable`：禁用检查<br>- `strict`：启用严格检查（默认）<br>- `ignore_strippable`：忽略可剥离的 token<br><br>**已验证的模型**：Qwen/QwQ-32B, Qwen/Qwen3-xxB<br>**默认值**：`strict` |
| `actor_rollout_ref.rollout.multi_turn.format` | 多轮交互的格式。选项：`hermes`、`llama3_json` 等。<br>**默认值**：`hermes` |
| `actor_rollout_ref.rollout.multi_turn.num_repeat_rollouts` | 每个交互的重复 rollout 次数。<br>**默认值**：`null` |

---

## Agent Loop 配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `actor_rollout_ref.rollout.agent.num_workers` | Agent loop worker 的数量。<br>**默认值**：8 |
| `actor_rollout_ref.rollout.agent.default_agent_loop` | 如果 RL dataset 中未设置 `agent_name`，使用的默认 agent loop。<br>**默认值**：`single_turn_agent` |
| `actor_rollout_ref.rollout.agent.agent_loop_config_path` | 自定义 agent loop 配置文件路径，应包含用于初始化 AgentLoop 实例的配置列表。<br>**默认值**：`null` |
| `actor_rollout_ref.rollout.agent.custom_async_server.path` | 自定义异步服务器实现的路径。<br>**默认值**：`null` |
| `actor_rollout_ref.rollout.agent.custom_async_server.name` | 自定义异步服务器类的类名（例如 `AsyncvLLMServer`）。<br>**默认值**：`null` |

---

## 验证阶段配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `actor_rollout_ref.rollout.val_kwargs.*` | 验证阶段的 sampling parameters，这允许我们在 post training 和 validation 时使用不同的 sampling parameters。例如，验证时通常设置 `temperature=0` 和 `do_sample=False` 来进行贪心解码，以获得更稳定的评估结果。 |
| `actor_rollout_ref.rollout.val_kwargs.temperature` | 验证阶段的温度。<br>**默认值**：0 |
| `actor_rollout_ref.rollout.val_kwargs.top_k` | 验证阶段的 top-k。<br>**默认值**：-1 |
| `actor_rollout_ref.rollout.val_kwargs.top_p` | 验证阶段的 top-p。<br>**默认值**：1.0 |
| `actor_rollout_ref.rollout.val_kwargs.n` | 验证阶段每个 prompt 的重复次数。<br>**默认值**：1 |
| `actor_rollout_ref.rollout.val_kwargs.do_sample` | 验证阶段是否采样。<br>**默认值**：`False` |

---

## Rollout Trace 配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `actor_rollout_ref.rollout.trace.backend` | Trace 后端，支持 `mlflow`、`weave`。<br>**默认值**：`null` |
| `actor_rollout_ref.rollout.trace.token2text` | 是否在输出中将 token id 转换为文本。<br>**默认值**：`False` |
| `actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker` | 每个训练步骤中每个 agent worker 要 trace 的最大唯一样本数。如果为 `null`，则 trace 所有样本。<br>**默认值**：`null` |

---

## Dataset 数据集配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `data.tokenizer` | Tokenizer 的类或路径。如果为 `null`，将从模型中自动推断。<br>**默认值**：`null` |
| `data.use_shm` | 是否使用共享内存（shared memory）来加载数据。<br>**默认值**：`False` |
| `data.train_files` | 训练集 parquet 文件。可以是列表或单个文件；路径可以是本地路径或 HDFS 路径。程序会将所有文件读入内存，因此不能太大（< 100GB）。 |
| `data.val_files` | 验证集 parquet 文件。可以是列表或单个文件。 |
| `data.train_max_samples` | 最大训练样本数。设置为 -1 使用完整数据集，否则从训练数据集中随机选择指定数量的样本。<br>**默认值**：-1 |
| `data.val_max_samples` | 最大验证样本数。设置为 -1 使用完整数据集。<br>**默认值**：-1 |
| `data.prompt_key` | 数据集中 prompt 的字段。<br>**默认值**：`prompt` |
| `data.reward_fn_key` | 用于选择奖励函数（如果每个样本使用不同奖励函数）的字段。<br>**默认值**：`data_source` |
| `data.max_prompt_length` | 最大提示长度。所有提示将向左填充到此长度。<br>**默认值**：512 |
| `data.max_response_length` | 最大响应长度。RL 算法（如 PPO）中的 Rollout 最多生成此长度。<br>**默认值**：512 |
| `data.tool_config_path` | 工具配置路径，用于计算真实的 prompt 长度。<br>**默认值**：与 `actor_rollout_ref.rollout.multi_turn.tool_config_path` 相同 |
| `data.return_raw_input_ids` | 是否返回未添加聊天模板的原始 input_ids。当 reward model 的 chat template 与 policy model 不同时使用。<br>**默认值**：`False` |
| `data.return_raw_chat` | 是否返回未应用聊天模板的原始 response。<br>**默认值**：`True` |
| `data.return_full_prompt` | 是否返回带有聊天模板的完整 prompt。<br>**默认值**：`False` |
| `data.return_multi_modal_inputs` | 是否在数据集中返回多模态输入。如果 rollout 生成新的多模态输入，则设置为 `False`。<br>**默认值**：`True` |
| `data.shuffle` | 是否在 DataLoader 中打乱数据。<br>**默认值**：`True` |
| `data.seed` | 打乱数据时使用的随机种子。<br>**默认值**：`null` |
| `data.dataloader_num_workers` | DataLoader worker 的数量。<br>**默认值**：8 |
| `data.validation_shuffle` | 是否打乱验证集。<br>**默认值**：`False` |
| `data.filter_overlong_prompts` | 是否过滤超长的 prompt。<br>**默认值**：`False` |
| `data.filter_overlong_prompts_workers` | 过滤超长 prompt 的工作进程数。对于大型数据集，使用多进程加速。<br>**默认值**：1 |
| `data.truncation` | 如果 input_ids 或 prompt 超过最大长度，则进行截断。选项：`error`、`left`、`right`、`middle`。<br>**默认值**：`error` |
| `data.image_key` | 多模态数据集中表示图像的字段。<br>**默认值**：`images` |
| `data.image_patch_size` | 图像 patch 大小。<br>**默认值**：14 |
| `data.video_key` | 多模态数据集中表示视频的字段。<br>**默认值**：`videos` |
| `data.trust_remote_code` | 是否信任本地的 huggingface cache。注意，这个 remote 是相对 huggingface 而言的，所以这个参数考虑的是"是否信任本地"。<br>**默认值**：`False` |
| `data.custom_cls.path` | 包含自定义数据集类的文件路径。如果未指定，将使用预实现的默认数据集。<br>**默认值**：`null` |
| `data.custom_cls.name` | 指定文件中的数据集类名。<br>**默认值**：`null` |
| `data.apply_chat_template_kwargs` | 调用 `tokenizer.apply_chat_template` 时的额外参数。<br>**默认值**：`{}` |

### Dataset Sampler 配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `data.sampler.class_path` | 包含实现 AbstractSampler 接口的 curriculum 类的模块路径。<br>**默认值**：`null` |
| `data.sampler.class_name` | curriculum 类的名称，如 `MySampler`。<br>**默认值**：`null` |

### Data Generation 配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `data.datagen.path` | 包含自定义数据生成类的文件路径。例如 `pkg://verl.experimental.dynamic_dataset.dynamicgen_dataset`。<br>**默认值**：`null` |
| `data.datagen.name` | 数据生成类在指定文件中的类名。例如 `MockDataGenerator`。<br>**默认值**：`null` |

---

## Model 模型配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `actor_rollout_ref.model.path` | Huggingface 模型路径。可以是本地路径或 HDFS 路径。 |
| `actor_rollout_ref.model.hf_config_path` | Huggingface 配置路径（如果与模型路径不同）。<br>**默认值**：`null` |
| `actor_rollout_ref.model.tokenizer_path` | Huggingface tokenizer 路径（如果与模型路径不同）。<br>**默认值**：`null` |
| `actor_rollout_ref.model.use_shm` | 是否使用共享内存（SHM）来加速模型权重的加载。<br>**默认值**：`False` |
| `actor_rollout_ref.model.trust_remote_code` | 是否信任本地的 huggingface cache。<br>**默认值**：`False` |
| `actor_rollout_ref.model.custom_chat_template` | 模型的自定义 chat template。<br>**默认值**：`null` |
| `actor_rollout_ref.model.external_lib` | 用于注册 Huggingface 模型/分词器的额外 Python 包。<br>**默认值**：`null` |
| `actor_rollout_ref.model.override_config` | 用于覆盖模型原始配置，主要用于 dropout 等。<br>**默认值**：`{}` |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | actor 训练过程是否重算梯度，以时间换空间。仅在使用 HF 模型定义时有效。<br>**默认值**：`True` |
| `actor_rollout_ref.model.enable_activation_offload` | actor 训练是否将 activation offload 到 CPU。仅在使用 HF 模型定义时有效。<br>**默认值**：`False` |
| `actor_rollout_ref.model.use_remove_padding` | 训练期间是否移除输入中的 padding token。仅在使用 HF 模型定义时有效。<br>**默认值**：`True` |
| `actor_rollout_ref.model.use_liger` | 是否使用 Liger kernel 进行线性层融合。仅在使用 HF 模型定义时有效。<br>**默认值**：`False` |
| `actor_rollout_ref.model.use_fused_kernels` | 是否使用自定义 fused kernel（如 FlashAttention, fused MLP）。<br>**默认值**：`False` |
| `actor_rollout_ref.model.fused_kernel_options.impl_backend` | 融合核的实现后端：`torch` 或 `triton`。需要和 `use_fused_kernels` 配合使用。<br>**默认值**：`torch` |

### LoRA 配置（FSDP）

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `actor_rollout_ref.model.lora_rank` | LoRA 秩。设置为正值以启用 LoRA（例如 32）。设置为 0 禁用。<br>**默认值**：0 |
| `actor_rollout_ref.model.lora_alpha` | LoRA 缩放因子。<br>**默认值**：16 |
| `actor_rollout_ref.model.target_modules` | 用于 LoRA 适配的目标模块。<br>**默认值**：`all-linear` |
| `actor_rollout_ref.model.exclude_modules` | 从 LoRA 适配中排除的模块。<br>**默认值**：`null` |
| `actor_rollout_ref.model.lora_adapter_path` | 预训练 LoRA 适配器的加载路径，用于继续训练。<br>**默认值**：`null` |

### LoRA 配置（Megatron）

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `actor_rollout_ref.model.lora.type` | LoRA 类型：`lora`、`vlm_lora`、`canonical_lora` 或 `dora`。<br>**默认值**：`lora` |
| `actor_rollout_ref.model.lora.rank` | LoRA 秩（低秩投影空间的维度）。设置为 0 禁用 LoRA。典型值：8、16、32、64。<br>**默认值**：0 |
| `actor_rollout_ref.model.lora.alpha` | 低秩投影的加权因子。<br>**默认值**：32 |
| `actor_rollout_ref.model.lora.dropout` | 低秩投影的 dropout 率。<br>**默认值**：0.0 |
| `actor_rollout_ref.model.lora.target_modules` | 应用 LoRA 的模块名称列表。对于 fused LoRA，默认为所有线性层 `['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']`。 |
| `actor_rollout_ref.model.lora.exclude_modules` | 不应用 LoRA 的模块名称列表。<br>**默认值**：`[]` |
| `actor_rollout_ref.model.lora.dropout_position` | 应用 dropout 的位置：`pre`（低秩投影前）或 `post`（之后）。<br>**默认值**：`pre` |
| `actor_rollout_ref.model.lora.lora_A_init_method` | 低秩矩阵 A 的初始化方法。<br>**默认值**：`xavier` |
| `actor_rollout_ref.model.lora.lora_B_init_method` | 低秩矩阵 B 的初始化方法。<br>**默认值**：`zero` |
| `actor_rollout_ref.model.lora.a2a_experimental` | 启用实验性的 All-to-All (A2A) 通信策略。<br>**默认值**：`False` |
| `actor_rollout_ref.model.lora.dtype` | LoRA 权重的参数数据类型。默认为 `null`，将使用模型的 dtype。 |
| `actor_rollout_ref.model.lora.adapter_path` | 预训练 LoRA 适配器权重的路径（`null` 表示从头训练）。 |
| `actor_rollout_ref.model.lora.freeze_vision_model` | VLMLoRA：是否冻结视觉模型。<br>**默认值**：`True` |
| `actor_rollout_ref.model.lora.freeze_vision_projection` | VLMLoRA：是否冻结视觉投影。<br>**默认值**：`True` |
| `actor_rollout_ref.model.lora.freeze_language_model` | VLMLoRA：是否冻结语言模型。<br>**默认值**：`True` |

---

## Actor 配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `actor_rollout_ref.hybrid_engine` | 目前只支持 hybrid engine，将 actor 和 rollout 模型放在同一资源组上。<br>**默认值**：`true` |
| `actor_rollout_ref.nccl_timeout` | 针对进程组执行操作的超时时间（秒）。<br>**默认值**：600 |
| `actor_rollout_ref.rollout.layered_summon` | 对于巨大的模型，layered summon 可以节省内存（防止 OOM）但会使其变慢。<br>**默认值**：`False` |
| `actor_rollout_ref.actor.strategy` | 训练 backend：`fsdp`、`fsdp2` 或 `megatron`。必须设置。 |
| `actor_rollout_ref.actor.rollout_n` | 每次更新的 rollout 数量（与 `actor_rollout_ref.rollout.n` 相同）。 |
| `actor_rollout_ref.actor.grad_clip` | Actor 更新的梯度裁剪。<br>**默认值**：1.0 |
| `actor_rollout_ref.actor.clip_ratio` | PPO 裁剪比率。<br>**默认值**：0.2 |
| `actor_rollout_ref.actor.clip_ratio_low` | 非对称裁剪的下界（用于 dual-clip PPO）。<br>**默认值**：0.2 |
| `actor_rollout_ref.actor.clip_ratio_high` | 非对称裁剪的上界（用于 dual-clip PPO）。<br>**默认值**：0.2 |
| `actor_rollout_ref.actor.clip_ratio_c` | Dual-clip PPO 中的常数 C；当优势 < 0 且 ratio > C 时进行裁剪。<br>**默认值**：3.0 |
| `actor_rollout_ref.actor.freeze_vision_tower` | 是否冻结视觉模型。<br>**默认值**：`false` |
| `actor_rollout_ref.actor.loss_agg_mode` | 损失聚合模式：`token-mean`、`seq-mean-token-sum`、`seq-mean-token-mean` 或 `seq-mean-token-sum-norm`。<br>**默认值**：`token-mean` |
| `actor_rollout_ref.actor.loss_scale_factor` | `seq-mean-token-sum-norm` 损失聚合模式的缩放因子。如果为 `null`，使用 `response_length`。设置为常数以确保一致的归一化。<br>**默认值**：`null` |
| `actor_rollout_ref.actor.entropy_coeff` | PPO 损失中的熵正则化系数。<br>**默认值**：0 |
| `actor_rollout_ref.actor.calculate_entropy` | 当为 true 时，actor forward 会请求模型计算熵。<br>**默认值**：`false` |
| `actor_rollout_ref.actor.use_kl_loss` | 是否使用 KL 损失代替 KL 奖励惩罚。对于 GRPO 为 `True`。<br>**默认值**：`false` |
| `actor_rollout_ref.actor.use_torch_compile` | 是否使用 `torch.compile()`。<br>**默认值**：`true` |
| `actor_rollout_ref.actor.kl_loss_coef` | 启用 `use_kl_loss` 时的 KL 损失系数，用于 GRPO。<br>**默认值**：0.001 |
| `actor_rollout_ref.actor.kl_loss_type` | KL 散度损失的类型。选项：`kl`(k1)、`abs`、`mse`(k2)、`low_var_kl`(k3)、`full`。<br>**默认值**：`low_var_kl` |
| `actor_rollout_ref.actor.ppo_epochs` | PPO 轮数。<br>**默认值**：1 |
| `actor_rollout_ref.actor.shuffle` | 在 PPO epochs 中打乱训练数据。<br>**默认值**：`false` |
| `actor_rollout_ref.actor.data_loader_seed` | 用于构建 mini-batch 的种子。<br>**默认值**：42 |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size` | Ulysses 类的 sequence parallel 大小。**已弃用**：使用 `fsdp_config.ulysses_sequence_parallel_size` 代替。<br>**默认值**：1 |
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking` | 通过分块计算熵以减少显存峰值。<br>**默认值**：`False` |
| `actor_rollout_ref.actor.entropy_checkpointing` | 是否将 entropy 通过 checkpoint 存下来。<br>**默认值**：`False` |
| `actor_rollout_ref.actor.use_remove_padding` | 训练期间是否移除输入中的 padding token。<br>**默认值**：与 `actor_rollout_ref.model.use_remove_padding` 相同 |

### Actor Policy Loss 配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `actor_rollout_ref.actor.policy_loss.loss_mode` | 损失函数模式：`vanilla`、`clip-cov`、`kl-cov`、`gpg`（来自 https://arxiv.org/abs/2505.22617）。<br>**默认值**：`vanilla` |
| `actor_rollout_ref.actor.policy_loss.clip_cov_ratio` | clip-cov 损失中要裁剪的 token 比例。<br>**默认值**：0.0002 |
| `actor_rollout_ref.actor.policy_loss.clip_cov_lb` | clip-cov 损失的下界。<br>**默认值**：1.0 |
| `actor_rollout_ref.actor.policy_loss.clip_cov_ub` | clip-cov 损失的上界。<br>**默认值**：5.0 |
| `actor_rollout_ref.actor.policy_loss.kl_cov_ratio` | kl-cov 损失中应用 KL 惩罚的 token 比例。<br>**默认值**：0.0002 |
| `actor_rollout_ref.actor.policy_loss.ppo_kl_coef` | KL 散度惩罚系数。<br>**默认值**：0.1 |

### Actor Checkpoint 配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `actor_rollout_ref.actor.checkpoint.save_contents` | 保存的检查点中包含的内容。可以是 `['model', 'optimizer', 'extra']`。使用 `'hf_model'` 可以将整个模型保存为 HF 格式。<br>**默认值**：`['model', 'optimizer', 'extra']` |
| `actor_rollout_ref.actor.checkpoint.load_contents` | 从检查点加载时指定的内容。<br>**默认值**：与 `save_contents` 相同 |
| `actor_rollout_ref.actor.checkpoint.async_save` | 是否异步保存检查点。目前仅对 Megatron 有效。<br>**默认值**：`False` |

### Actor Router Replay 配置（MoE 模型）

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `actor_rollout_ref.actor.router_replay.mode` | Router replay 模式：`disabled`、`R2`、`R3`。<br>**默认值**：`disabled` |
| `actor_rollout_ref.actor.router_replay.record_file` | 保存记录的路由决策的文件路径。当模式为 `R2` 或 `R3` 时需要。 |
| `actor_rollout_ref.actor.router_replay.replay_file` | 加载记录的路由决策进行 replay 的文件路径。 |

### Actor Optimizer 配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `actor_rollout_ref.actor.optim.lr` | 学习率。<br>**默认值**：1e-6 |
| `actor_rollout_ref.actor.optim.lr_warmup_steps` | 预热步数；负值则由 `lr_warmup_steps_ratio` 决定。<br>**默认值**：-1 |
| `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio` | 预热步数比例（当 `lr_warmup_steps` 为负时使用）。<br>**默认值**：0.0 |
| `actor_rollout_ref.actor.optim.total_training_steps` | 总训练步数（运行时覆盖）。<br>**默认值**：-1 |
| `actor_rollout_ref.actor.optim.weight_decay` | 权重衰减系数。<br>**默认值**：0.01 |

---

## Reference 模型配置

Reference 模型将在 `actor.use_kl_loss` 或 `algorithm.use_kl_in_reward` 为 True 时启用。

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `actor_rollout_ref.ref.strategy` | Reference 模型的策略，与 actor 相同。<br>**默认值**：与 `actor_rollout_ref.actor.strategy` 相同 |
| `actor_rollout_ref.ref.rollout_n` | 每次更新的 rollout 数量。<br>**默认值**：与 `actor_rollout_ref.rollout.n` 相同 |
| `actor_rollout_ref.ref.use_torch_compile` | 是否启用 torch.compile。<br>**默认值**：与 `actor_rollout_ref.actor.use_torch_compile` 相同 |
| `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | 计算 log_prob 时一次 forward 的 batch size（本地每 GPU）。<br>**默认值**：`null` |
| `actor_rollout_ref.ref.log_prob_use_dynamic_bsz` | 是否为 log_prob 计算启用动态 batch size（序列打包）。<br>**默认值**：与 `actor_rollout_ref.actor.use_dynamic_bsz` 相同 |
| `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu` | log_prob 计算中每 GPU 的最大 token 长度。<br>**默认值**：与 `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` 相同 |

---

## Critic 配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `critic.enable` | 是否启用 critic worker。默认情况下，仅当优势估计器为 `gae` 时启用。手动设置为 `True` 以始终启用 critic worker。<br>**默认值**：`null`（自动） |
| `critic.strategy` | 用于 critic 模型训练的 FSDP/FSDP2/Megatron 策略。必须设置。 |
| `critic.rollout_n` | 每次更新的 rollout 数量。<br>**默认值**：与 `actor_rollout_ref.rollout.n` 相同 |
| `critic.model.path` | 预训练模型权重的路径。 |
| `critic.model.tokenizer_path` | Tokenizer 路径（默认为 actor 的模型路径）。 |
| `critic.model.override_config` | Hugging Face 配置覆盖。<br>**默认值**：`{}` |
| `critic.model.external_lib` | 外部模型实现（可选）。 |
| `critic.model.trust_remote_code` | 是否信任 Hugging Face 模型的远程代码。<br>**默认值**：与 `actor_rollout_ref.model.trust_remote_code` 相同 |
| `critic.ppo_mini_batch_size` | 每次更新的 PPO mini-batch 大小。<br>**默认值**：与 `actor_rollout_ref.actor.ppo_mini_batch_size` 相同 |
| `critic.ppo_micro_batch_size_per_gpu` | 本地每 GPU 的 micro batch 大小。 |
| `critic.use_dynamic_bsz` | 是否在运行时自动调整 batch 大小。<br>**默认值**：与 `actor_rollout_ref.actor.use_dynamic_bsz` 相同 |
| `critic.ppo_max_token_len_per_gpu` | 一个 PPO batch 中每 GPU 的最大 token 数。<br>**默认值**：32768 |
| `critic.forward_max_token_len_per_gpu` | forward pass 中每 GPU 的最大 token 长度。<br>**默认值**：与 `ppo_max_token_len_per_gpu` 相同 |
| `critic.ppo_epochs` | 每 batch 的 PPO 轮数。<br>**默认值**：与 `actor_rollout_ref.actor.ppo_epochs` 相同 |
| `critic.shuffle` | 在 PPO epochs 中打乱训练数据。<br>**默认值**：与 `actor_rollout_ref.actor.shuffle` 相同 |
| `critic.cliprange_value` | PPO 值函数裁剪范围。<br>**默认值**：0.5 |
| `critic.loss_agg_mode` | 损失聚合模式：`token-mean`、`seq-mean-token-sum` 或 `seq-mean-token-mean`。<br>**默认值**：与 `actor_rollout_ref.actor.loss_agg_mode` 相同 |
| `critic.optim.lr` | 学习率。<br>**默认值**：1e-5 |
| `critic.optim.lr_warmup_steps_ratio` | 预热步数比例。<br>**默认值**：0.0 |
| `critic.optim.weight_decay` | 权重衰减。<br>**默认值**：0.01 |

---

## Reward Model 配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `reward_model.enable` | 是否启用奖励模型。如果为 `False`，则仅使用用户定义的奖励函数计算奖励。在 GSM8K 和 Math 示例中，我们禁用奖励模型。对于使用 full_hh_rlhf 的 RLHF 对齐示例，我们使用奖励模型来评估响应。<br>**默认值**：`False` |
| `reward_model.enable_resource_pool` | 是否将模型部署到单独的资源池。如果为 true，将使用 `n_gpus_per_node` 和 `nnodes` 来确定资源节点。<br>**默认值**：`False` |
| `reward_model.n_gpus_per_node` | 资源池中每个节点的 GPU 数量。<br>**默认值**：0 |
| `reward_model.nnodes` | 资源池中的节点数量。<br>**默认值**：0 |
| `reward_model.strategy` | FSDP 策略：`fsdp`、`fsdp2` 或 `megatron`。必须设置。 |
| `reward_model.model.input_tokenizer` | 输入分词器。如果奖励模型的聊天模板与策略不一致，需要先解码为纯文本，然后应用 RM 的 chat_template，再用 RM 评分。如果 chat_template 一致，可以设置为 `null`。<br>**默认值**：与 `actor_rollout_ref.model.path` 相同 |
| `reward_model.model.path` | RM 的 HDFS 路径或本地路径。注意 RM 仅支持 `AutoModelForSequenceClassification`。其他模型类型需要定义自己的 RewardModelWorker 并从代码传入。 |
| `reward_model.model.external_lib` | 外部模型实现（可选）。 |
| `reward_model.model.trust_remote_code` | 是否允许加载远程代码模型。<br>**默认值**：`False` |
| `reward_model.model.override_config` | 覆盖 HF 配置。<br>**默认值**：`{}` |
| `reward_model.micro_batch_size_per_gpu` | 本地每 GPU 的 micro batch 大小。 |
| `reward_model.max_length` | 评分处理的最大序列长度。 |
| `reward_model.use_dynamic_bsz` | 是否在运行时动态调整 batch 大小。<br>**默认值**：与 `critic.use_dynamic_bsz` 相同 |
| `reward_model.forward_max_token_len_per_gpu` | 一次 forward pass 中每 GPU 的最大 token 数。<br>**默认值**：与 `critic.forward_max_token_len_per_gpu` 相同 |
| `reward_model.reward_manager` | **已弃用**。使用 `reward_manager.name` 代替。定义计算基于规则的奖励和处理不同奖励源的机制。<br>**默认值**：`naive` |
| `reward_model.launch_reward_fn_async` | 是否在 log_prob 期间异步启动自定义奖励函数。自定义奖励函数在 CPU 上异步执行。<br>**默认值**：`False` |
| `reward_model.sandbox_fusion.url` | 用于沙箱执行的云/本地函数 URL。 |
| `reward_model.sandbox_fusion.max_concurrent` | 允许到沙箱的最大并发请求数。<br>**默认值**：64 |
| `reward_model.sandbox_fusion.memory_limit_mb` | 每个沙箱进程的最大内存限制（MB）。<br>**默认值**：1024 |

---

## Custom Reward Function 自定义奖励函数

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `custom_reward_function.path` | 包含自定义奖励函数的文件路径。如果未指定，将使用预实现的奖励函数。 |
| `custom_reward_function.name` | 指定文件中的奖励函数名称。<br>**默认值**：`compute_score` |

---

## Reward Manager 配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `reward_manager.source` | 奖励管理器来源：`register` 或 `module`。<br>**默认值**：`register` |
| `reward_manager.name` | 奖励管理器名称。<br>**默认值**：与 `reward_model.reward_manager` 相同（通常为 `naive`） |
| `reward_manager.module.path` | 包含自定义奖励管理器的模块路径。 |
| `reward_manager.module.name` | 自定义奖励管理器类名。<br>**默认值**：`custom_reward_manager` |

---

## Algorithm 算法配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `algorithm.gamma` | 未来奖励的折扣因子。<br>**默认值**：1.0 |
| `algorithm.lam` | GAE 估计器中偏差和方差的权衡。<br>**默认值**：1.0 |
| `algorithm.adv_estimator` | 优势估计器类型：`gae`、`grpo`、`reinforce_plus_plus`、`rloo` 等。<br>**默认值**：`gae` |
| `algorithm.norm_adv_by_std_in_grpo` | 是否在 GRPO 中按标准差归一化优势。<br>**默认值**：`True` |
| `algorithm.use_kl_in_reward` | 是否在奖励中启用 KL 惩罚。<br>**默认值**：`False` |
| `algorithm.kl_penalty` | 如何估计 KL 散度：`kl`、`abs`、`mse`、`low_var_kl` 或 `full`。<br>**默认值**：`kl` |
| `algorithm.kl_ctrl.type` | KL 控制类型：`fixed` 或 `adaptive`。<br>**默认值**：`fixed` |
| `algorithm.kl_ctrl.kl_coef` | KL 惩罚的初始系数。<br>**默认值**：0.001 |
| `algorithm.kl_ctrl.horizon` | 自适应控制器的 horizon 值（如果启用）。<br>**默认值**：10000 |
| `algorithm.kl_ctrl.target_kl` | 目标 KL 散度（用于自适应控制器）。<br>**默认值**：0.1 |
| `algorithm.use_pf_ppo` | 是否启用偏好反馈 PPO。<br>**默认值**：`False` |
| `algorithm.pf_ppo.reweight_method` | 样本重加权方法：`pow`、`max_min` 或 `max_random`。<br>**默认值**：`pow` |
| `algorithm.pf_ppo.weight_pow` | `pow` 方法中用于权重缩放的幂。<br>**默认值**：2.0 |

---

## Rollout Correction 离策略校正

用于校正离策略分布偏移。参见文档：`docs/algo/rollout_corr.md`

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `algorithm.rollout_correction.rollout_is` | IS 聚合级别：`null`（禁用）、`token`（逐 token）、`sequence`（逐序列）。<br>**默认值**：`null` |
| `algorithm.rollout_correction.rollout_is_threshold` | IS 权重截断的上阈值（典型值：2.0-5.0）。<br>**默认值**：2.0 |
| `algorithm.rollout_correction.rollout_rs` | RS 聚合级别：`null`（禁用）、`token`、`sequence`、`geometric`。<br>**默认值**：`null` |
| `algorithm.rollout_correction.rollout_rs_threshold` | 拒绝采样的上阈值（`null` = 使用 `rollout_is_threshold`）。<br>**默认值**：`null` |
| `algorithm.rollout_correction.rollout_rs_threshold_lower` | 拒绝采样的下阈值（`null` = 自动计算为 1/upper）。<br>**默认值**：`null` |
| `algorithm.rollout_correction.rollout_token_veto_threshold` | 用于灾难性异常值的逐 token 否决阈值（`null` = 禁用）。<br>**默认值**：`null` |
| `algorithm.rollout_correction.bypass_mode` | 操作模式：`false` = Decoupled（3 个策略），`true` = Bypass（2 个策略）。<br>**默认值**：`false` |
| `algorithm.rollout_correction.use_policy_gradient` | 损失函数：`false` = 带裁剪的 PPO，`true` = 策略梯度（无裁剪）。<br>**默认值**：`false` |
| `algorithm.rollout_correction.rollout_is_batch_normalize` | 批量归一化 IS 权重：`false` = 原始权重，`true` = 归一化到 mean=1.0。<br>**默认值**：`false` |

---

## Trainer 训练器配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `trainer.balance_batch` | 是否在分布式工作节点间平衡批次大小。<br>**默认值**：`True` |
| `trainer.total_epochs` | 训练的总轮数。<br>**默认值**：30 |
| `trainer.total_training_steps` | 总训练步数（可显式设置或从轮数派生）。<br>**默认值**：`null` |
| `trainer.project_name` | 用于实验跟踪（如 wandb）的项目名称。<br>**默认值**：`verl_examples` |
| `trainer.experiment_name` | 用于在跟踪工具中识别运行的实验名称。<br>**默认值**：`gsm8k` |
| `trainer.logger` | 使用的日志后端：`console`、`wandb` 等。<br>**默认值**：`["console", "wandb"]` |
| `trainer.log_val_generations` | 验证期间要记录的生成数量。<br>**默认值**：0 |
| `trainer.rollout_data_dir` | 用于记录 rollout 数据的目录；如果为 `null` 则不转储。<br>**默认值**：`null` |
| `trainer.validation_data_dir` | 用于记录验证数据的目录；如果为 `null` 则不转储。<br>**默认值**：`null` |
| `trainer.nnodes` | 训练中使用的节点数。<br>**默认值**：1 |
| `trainer.n_gpus_per_node` | 每个节点的 GPU 数量。<br>**默认值**：8 |
| `trainer.save_freq` | 模型检查点的保存频率（按迭代次数）。-1 表示不保存。<br>**默认值**：-1 |
| `trainer.esi_redundant_time` | ESI（弹性服务实例）冗余时间。为确保在 ESI 关闭前保存检查点，系统会提前开始保存。提前时间 = 最长历史步骤持续时间 + 检查点保存持续时间 + esi_redundant_time。<br>**默认值**：0 |
| `trainer.resume_mode` | 恢复模式：`auto`（如果有则从最后一个检查点恢复）、`disable`（从头开始）或 `resume_path`（从用户定义的路径恢复）。<br>**默认值**：`auto` |
| `trainer.resume_from_path` | 从该路径恢复训练（仅当 `resume_mode` 为 `resume_path` 时使用）。<br>**默认值**：`null` |
| `trainer.val_before_train` | 是否在训练开始前运行验证。<br>**默认值**：`True` |
| `trainer.val_only` | 是否只运行验证。<br>**默认值**：`False` |
| `trainer.test_freq` | 验证频率（以训练迭代次数计）。-1 表示不验证。<br>**默认值**：-1 |
| `trainer.critic_warmup` | 在更新策略之前预热 critic 的迭代次数。<br>**默认值**：0 |
| `trainer.default_hdfs_dir` | 用于保存检查点的默认分布式文件系统路径。<br>**默认值**：`null` |
| `trainer.del_local_ckpt_after_load` | 加载后是否删除本地检查点。<br>**默认值**：`False` |
| `trainer.default_local_dir` | 用于保存检查点的默认本地目录。<br>**默认值**：`checkpoints/${trainer.project_name}/${trainer.experiment_name}` |
| `trainer.max_actor_ckpt_to_keep` | 保留的 actor 检查点的最大数量。<br>**默认值**：`null` |
| `trainer.max_critic_ckpt_to_keep` | 保留的 critic 检查点的最大数量。<br>**默认值**：`null` |
| `trainer.ray_wait_register_center_timeout` | Ray worker 等待注册的超时时间（秒）。<br>**默认值**：300 |
| `trainer.device` | 运行训练的设备（如 `cuda`、`cpu`）。<br>**默认值**：`cuda` |
| `trainer.use_legacy_worker_impl` | 是否使用旧版 worker 实现。模式：`auto`、`enable` 或 `disable`。<br>**默认值**：`auto` |

---

## FSDP Engine 配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `*.fsdp_config.wrap_policy.min_num_params` | 触发 FSDP 包装一个层的最小参数数量。<br>**默认值**：0 |
| `*.fsdp_config.param_offload` | 是否将模型参数卸载到 CPU（以速度换内存）。注意这与 FSDP 中的 `offload_policy` 不同。<br>**默认值**：`false` |
| `*.fsdp_config.optimizer_offload` | 是否将优化器状态卸载到 CPU。<br>**默认值**：`false` |
| `*.fsdp_config.offload_policy` | **仅用于 FSDP2**：训练期间卸载参数/梯度/优化器。<br>**默认值**：`false` |
| `*.fsdp_config.reshard_after_forward` | **仅用于 FSDP2**：前向传播后重新分片以减少内存占用。<br>**默认值**：`true` |
| `*.fsdp_config.fsdp_size` | 每个 FSDP 分片组中的 GPU 数量；-1 表示自动。<br>**默认值**：-1 |
| `*.fsdp_config.forward_prefetch` | **仅用于 FSDP1**：在前向计算完成前预取下一次前向传播的 all-gather。<br>**默认值**：`False` |
| `*.fsdp_config.model_dtype` | FSDP 的模型 dtype。<br>**默认值**：`fp32` |
| `*.fsdp_config.use_orig_params` | 是否在 FSDP 中使用原始参数。仅在 FSDP1 中可用。<br>**默认值**：`false` |
| `*.fsdp_config.seed` | 可重复性的随机种子。<br>**默认值**：42 |
| `*.fsdp_config.full_determinism` | 是否为分布式训练启用完全确定性，仅用于调试。<br>**默认值**：`false` |
| `*.fsdp_config.ulysses_sequence_parallel_size` | Ulysses 序列并行大小。<br>**默认值**：1 |
| `*.fsdp_config.entropy_from_logits_with_chunking` | 是否在 FSDP 中使用 `entropy_from_logits_with_chunking`。<br>**默认值**：`false` |
| `*.fsdp_config.use_torch_compile` | 是否在 FSDP 中使用 torch compile。<br>**默认值**：`true` |
| `*.fsdp_config.entropy_checkpointing` | 是否在 FSDP 中使用 entropy checkpointing。<br>**默认值**：`false` |
| `*.fsdp_config.forward_only` | 是否在 FSDP 中只使用 forward。<br>**默认值**：`false` |
| `*.fsdp_config.strategy` | 策略类型：`fsdp` 或 `fsdp2`。<br>**默认值**：`fsdp` |
| `*.fsdp_config.dtype` | 混合精度训练参数 dtype。<br>**默认值**：`bfloat16` |

---

## Megatron Engine 配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `*.megatron.param_offload` | 是否将模型参数卸载到 CPU。<br>**默认值**：`False` |
| `*.megatron.grad_offload` | 是否将梯度卸载到 CPU。<br>**默认值**：`False` |
| `*.megatron.optimizer_offload` | 是否将优化器状态卸载到 CPU。<br>**默认值**：`False` |
| `*.megatron.tensor_model_parallel_size` | 张量模型并行大小。<br>**默认值**：1 |
| `*.megatron.expert_model_parallel_size` | 专家模型并行大小。<br>**默认值**：1 |
| `*.megatron.expert_tensor_parallel_size` | 专家张量并行大小（`null` 表示与 TP 相同）。<br>**默认值**：`null` |
| `*.megatron.pipeline_model_parallel_size` | 流水线模型并行大小。<br>**默认值**：1 |
| `*.megatron.virtual_pipeline_model_parallel_size` | 虚拟流水线模型并行大小。<br>**默认值**：`null` |
| `*.megatron.context_parallel_size` | 上下文并行大小。<br>**默认值**：1 |
| `*.megatron.sequence_parallel` | 是否启用序列并行。<br>**默认值**：`True` |
| `*.megatron.use_distributed_optimizer` | 是否使用分布式优化器。<br>**默认值**：`True` |
| `*.megatron.use_dist_checkpointing` | 是否使用分布式检查点。<br>**默认值**：`False` |
| `*.megatron.dist_checkpointing_path` | 分布式检查点路径。<br>**默认值**：`null` |
| `*.megatron.dist_checkpointing_prefix` | 分布式检查点前缀，例如 Nemo2 会在 state dict 键前添加前缀 'module.'。<br>**默认值**：`''` |
| `*.megatron.seed` | 随机种子。<br>**默认值**：42 |
| `*.megatron.override_ddp_config` | 允许覆盖分布式数据并行（DDP）配置。<br>**默认值**：`{}` |
| `*.megatron.override_transformer_config.recompute_granularity` | 重计算粒度，选项：`full`、`selective`。<br>**默认值**：`null` |
| `*.megatron.override_transformer_config.recompute_modules` | 重计算模块，多选项：`core_attn`、`moe_act`、`layernorm`、`mla_up_proj`、`mlp`、`moe`。<br>**默认值**：`["core_attn"]` |
| `*.megatron.override_transformer_config.recompute_method` | 重计算方法：`uniform`、`block`。<br>**默认值**：`null` |
| `*.megatron.override_transformer_config.recompute_num_layers` | 重计算层数。<br>**默认值**：`null` |
| `*.megatron.override_transformer_config.attention_backend` | 注意力后端（`flash`、`fused`、`unfused`、`local`、`auto`）。<br>**默认值**：`flash` |
| `*.megatron.override_mcore_model_config` | 覆盖 MCore 模型配置。<br>**默认值**：`{}` |
| `*.megatron.use_mbridge` | 是否使用 MBridge。<br>**默认值**：`False` |
| `*.megatron.vanilla_mbridge` | 是否使用 vanilla MBridge。<br>**默认值**：`True` |
| `*.megatron.use_remove_padding` | 是否使用 thd 格式（序列打包）。如果不使用，则使用 bshd 格式，将 input_ids 填充到最长序列长度。<br>**默认值**：`True` |
| `*.megatron.forward_only` | 是否只使用 forward。<br>**默���值**：`False` |
| `*.megatron.dtype` | 混合精度训练参数 dtype。<br>**默认值**：`bfloat16` |

---

## Optimizer 优化器配置

### FSDP Optimizer

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `*.optim.optimizer` | 优化器类名（例如 `AdamW`、`AdamW8bit`、`_AdamW`、`Adam`）。<br>**默认值**：`AdamW` |
| `*.optim.optimizer_impl` | 导入优化器的模块路径。示例：`torch.optim`、`torchao.optim`、`bitsandbytes.optim`。<br>**默认值**：`torch.optim` |
| `*.optim.lr` | 学习率。<br>**默认值**：1e-3 |
| `*.optim.lr_warmup_steps_ratio` | LR warmup 步数比例。<br>**默认值**：0.0 |
| `*.optim.total_training_steps` | 总训练步数。<br>**默认值**：-1 |
| `*.optim.weight_decay` | 权重衰减。<br>**默认值**：0.01 |
| `*.optim.lr_warmup_steps` | LR warmup 步数。<br>**默认值**：-1 |
| `*.optim.betas` | Adam 优化器的 betas。<br>**默认值**：`[0.9, 0.999]` |
| `*.optim.clip_grad` | 梯度裁剪。<br>**默认值**：1.0 |
| `*.optim.min_lr_ratio` | 余弦调度的最小 LR 比例。<br>**默认值**：0.0 |
| `*.optim.num_cycles` | LR 调度中的余弦周期数。<br>**默认值**：0.5 |
| `*.optim.lr_scheduler_type` | LR 调度器类型：`constant` 或 `cosine`。<br>**默认值**：`constant` |
| `*.optim.override_optimizer_config` | 额外的优化器特定关键字参数。例如用于 torchao 的 bf16 随机舍入。<br>**默认值**：`null` |

### Megatron Optimizer

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `*.optim.optimizer` | 优化器类型：`adam`。<br>**默认值**：`adam` |
| `*.optim.lr` | 学习率。<br>**默认值**：1e-3 |
| `*.optim.lr_warmup_steps_ratio` | LR warmup 步数比例。<br>**默认值**：0.0 |
| `*.optim.lr_warmup_steps` | LR warmup 步数。<br>**默认值**：-1 |
| `*.optim.lr_warmup_init` | warmup 的初始学习率。<br>**默认值**：0.0 |
| `*.optim.lr_decay_steps` | LR 衰减步数。<br>**默认值**：`null` |
| `*.optim.lr_decay_style` | LR 衰减风格：`constant`、`linear`、`cosine`、`inverse_square_root`。<br>**默认值**：`constant` |
| `*.optim.min_lr` | 最小学习率。<br>**默认值**：0.0 |
| `*.optim.weight_decay` | 权重衰减。<br>**默认值**：0.01 |
| `*.optim.weight_decay_incr_style` | 权重衰减增加风格：`constant`、`linear`、`cosine`。<br>**默认值**：`constant` |
| `*.optim.lr_wsd_decay_style` | LR WSD 衰减风格：`constant`、`exponential`、`cosine`。<br>**默认值**：`exponential` |
| `*.optim.lr_wsd_decay_steps` | LR WSD 衰减步数。<br>**默认值**：`null` |
| `*.optim.betas` | Adam 优化器的 betas。<br>**默认值**：`[0.9, 0.999]` |
| `*.optim.clip_grad` | 梯度裁剪。<br>**默认值**：1.0 |
| `*.optim.use_checkpoint_opt_param_scheduler` | 是否使用检查点优化器参数调度器。<br>**默认值**：`False` |
| `*.optim.override_optimizer_config` | 覆盖优化器配置。<br>**默认值**：`{}` |

---

## Profiler 性能分析配置

### Global Profiler

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `global_profiler.tool` | 性能分析工具：`nsys`、`npu`、`torch`、`torch_memory`。<br>**默认值**：`null` |
| `global_profiler.steps` | 要分析的步骤列表。<br>**默认值**：`null` |
| `global_profiler.profile_continuous_steps` | 是否将连续步骤合并到一个数据库中。如果为 `True`，`worker.profiler.discrete` 必须为 `False`。<br>**默认值**：`False` |
| `global_profiler.save_path` | 保存性能分析内容的路径。<br>**默认值**：`outputs/profile` |

### Nsight Systems 配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `global_profiler.global_tool_config.nsys.discrete` | `True` 表示每个任务有自己的数据库，`False` 表示所有任务共享一个。<br>**默认值**：`False` |
| `global_profiler.global_tool_config.nsys.controller_nsight_options.trace` | 选择要追踪的 API（如 cuda、nvtx、cublas、ucx 等）。<br>**默认值**：`cuda,nvtx,cublas,ucx` |
| `global_profiler.global_tool_config.nsys.controller_nsight_options.cuda-memory-usage` | 是否追踪 CUDA 内存使用情况。必须是字符串 `"true"` 或 `"false"`。<br>**默认值**：`"true"` |
| `global_profiler.global_tool_config.nsys.controller_nsight_options.cuda-graph-trace` | CUDA graphs 追踪方式。<br>**默认值**：`"graph"` |
| `global_profiler.global_tool_config.nsys.worker_nsight_options.trace` | 选择要追踪的 API。<br>**默认值**：`cuda,nvtx,cublas,ucx` |
| `global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range` | 仅在 `torch.cuda.profiler.start` 和 `stop` 范围内进行分析。不要更改此配置。<br>**默认值**：`cudaProfilerApi` |
| `global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range-end` | 指定捕获范围结束时的期望行为。有效值为 `"repeat-shutdown:n"` 或 `null`。<br>**默认值**：`null` |
| `global_profiler.global_tool_config.nsys.worker_nsight_options.kill` | 向目标应用程序的进程组发送信号。<br>**默认值**：`none` |

### Torch Memory Profiler 配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `global_profiler.global_tool_config.torch_memory.trace_alloc_max_entries` | 要记录的最大分配条目数。<br>**默认值**：100000 |
| `global_profiler.global_tool_config.torch_memory.stack_depth` | 每次分配的调用栈深度。<br>**默认值**：32 |
| `global_profiler.global_tool_config.torch_memory.context` | `alloc`：仅记录分配事件，`state`：记录内存状态变化，`all`：两者都记录。<br>**默认值**：`all` |
| `global_profiler.global_tool_config.torch_memory.stacks` | `python`：记录 Python 栈，`cpp`：记录 C++ 栈，`all`：两者都记录。<br>**默认值**：`all` |

### 组件级 Profiler 配置

Actor、Critic、Ref、Reward Model 和 Rollout 都有自己的 profiler 配置：

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `*.profiler.tool` | 性能分析工具，默认与 `global_profiler.tool` 相同。 |
| `*.profiler.enable` | 是否启用该组件的性能分析。<br>**默认值**：`False` |
| `*.profiler.all_ranks` | 是否对所有 rank 进行性能分析。<br>**默认值**：`False` |
| `*.profiler.ranks` | 将被分析的 rank 列表。`[]` 或 `[0,1,...]`。<br>**默认值**：`[]` |
| `*.profiler.save_path` | 性能分析结果保存路径。 |

---

## Transfer Queue 配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `transfer_queue.enable` | 是否启用 transfer queue。<br>**默认值**：`False` |

---

## Ray 配置

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `ray_kwargs.ray_init.num_cpus` | Ray 使用的 CPU 数量。使用 SLURM 时应使用固定数字而不是 `null`。`null` 表示使用所有 CPU，这可能在 SLURM 等系统中导致挂起。<br>**默认值**：`null` |
| `ray_kwargs.timeline_json_file` | 保存 Ray 时间线 JSON 文件以进行性能分析的路径。<br>**默认值**：`null` |

---

## Prometheus 配置（服务器模式）

| 参数名称 | 作用与解释 |
| ------- | ------- |
| `actor_rollout_ref.rollout.prometheus.enable` | 是否在服务器模式 rollout 上启用 prometheus。<br>**默认值**：`false` |
| `actor_rollout_ref.rollout.prometheus.port` | Prometheus 监听的端口号。<br>**默认值**：9090 |
| `actor_rollout_ref.rollout.prometheus.file` | Prometheus 配置文件路径。<br>**默认值**：`/tmp/ray/session_latest/metrics/prometheus/prometheus.yml` |
| `actor_rollout_ref.rollout.prometheus.served_model_name` | 指定 served_model_name 以避免在 Grafana 中显示过长的模型路径。<br>**默认值**：与 `actor_rollout_ref.model.path` 相同 |

---

## 附录：参数配置最佳实践

### 防止 OOM 的关键参数

1. `ppo_micro_batch_size_per_gpu`：根据显存大小调整
2. `gpu_memory_utilization`：SGLang >= 0.4.8.post1 可设置为 0.85
3. `enable_gradient_checkpointing`：启用以时间换空间
4. `param_offload` / `optimizer_offload`：将参数/优化器卸载到 CPU

### GRPO 必需配置

```yaml
algorithm:
  adv_estimator: grpo
actor_rollout_ref:
  rollout:
    n: 5  # > 1 for group sampling
  actor:
    use_kl_loss: True
```

### 多轮对话推荐配置

```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    multi_turn:
      enable: True
      tool_config_path: /path/to/tools.json
```

---

## 版本说明

本文档基于 verl 源码分析整理，可能因框架更新而存在差异。如有疑问，请参考源码 `verl/trainer/config/` 目录下的 YAML 配置文件。
