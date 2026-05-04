# verl 源码走读：Multi-Teacher OPD 实现解析

在 on-policy distillation（OPD）里，学生模型先用当前策略生成回答，再让教师模型对同一段 prompt+response 给出 token 级 logprob，最后把这份教师信号并入 actor 更新。单教师 OPD 已经能把“强模型的分布信息”接进 RL 训练；Multi-Teacher OPD 要解决的是更实际的问题：一个训练 batch 里可能混着 GSM8K、Geometry3K、视觉问答等不同数据源，它们适合不同教师模型，框架必须在不改训练主循环的前提下，把每条样本送到正确教师。

本文不展开知识蒸馏、PPO、GRPO、FSDP 或 Megatron 的基础原理，而是沿着 verl 当前源码里 Multi-Teacher OPD 的真实执行链路，分析它如何从配置变成 Ray 资源、从样本字段变成教师路由、从 top-k logprob 变成 actor loss，以及它在显存、通信、调度和 checkpoint 上付出的代价。

# 前言

## 业务 / 工程背景

Multi-Teacher OPD 出现在 verl 的 PPO/GRPO 训练链路中。它面向的是“学生模型在线生成 + 教师模型在线打分”的蒸馏训练场景：

- 学生 actor/rollout 仍然按普通 PPO 路径生成 response；
- 教师模型不是 reward model，也不是 ref model，而是额外的推理服务；
- 教师输出的不是标量 reward，而是每个 token 的 sampled-token logprob 或 top-k logprobs；
- 多教师通过样本里的 `data_source` 等字段路由，而不是对多个教师做 ensemble。

示例脚本 `examples/on_policy_distillation_trainer/run_qwen3_mopd_gsm8k_geo3k.sh:78-114` 就展示了典型用法：GSM8K 样本走 `openai/gsm8k` 对应的文本教师，Geometry3K 样本走 `hiyouga/geometry3k` 对应的视觉教师。

## 核心矛盾

这个特性背后的工程冲突可以概括为三句话：

1. **训练 batch 是混合的，但教师模型是异构且昂贵的。** 不能把所有样本都送给一个教师，也不能在每步动态创建教师服务。
2. **教师 logprob 是 token 级信号，但训练主链路已经有 padding、remove-padding、SP/CP/TP 切分。** 教师输出必须跟 actor 前向的 shape 精确对齐。
3. **教师推理可以独立部署，但训练 step 需要它的结果。** 当前实现把教师 RPC 放进 rollout 后处理的关键路径，用实现简单性换取了潜在等待时间。

## 本文主线

本文按机制而不是文件展开：

1. 入口与配置归一化：用户如何开启 Multi-Teacher OPD，配置如何变成可校验的 teacher map。
2. 教师资源池与服务初始化：多个教师如何被静态切到 Ray 子资源池，并变成可调用的推理服务。
3. 样本路由与教师查询：真正的“多教师”发生在哪里，`teacher_key` 如何决定每条样本走哪个教师。
4. Actor 更新与蒸馏损失：`teacher_logprobs` 如何穿过 padding/no-padding、FSDP/Megatron 前向和 loss。
5. 完整主路径串联：一次用户命令从配置加载到 checkpoint 的完整链路。
6. Shape、rank、状态与通信：把数据形状、资源拓扑、状态写入和通信开销摊开看。
7. 显存、性能、配置、测试与局限：哪些地方真省，哪些地方新增瓶颈，哪些风险没有测试兜底。

## 不展开的内容

本文不讲 OPD 理论推导、不讲 PPO/GRPO 数学、不讲 FSDP/Megatron 基础，也不分析 vLLM/SGLang 内部调度。我们只关心 verl 如何把 Multi-Teacher OPD 接进现有训练链路。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `verl/trainer/config/ppo_trainer.yaml` | 默认加载 `distillation` 配置组，使 CLI 能覆盖 `distillation.*`。 |
| `verl/workers/config/distillation.py` | OPD 配置 dataclass、单/多教师归一化、top-k 参数校验。 |
| `verl/trainer/main_ppo.py` | 主 CLI/Ray 入口，创建 `teacher_pool` 并映射 `Role.TeacherModel`。 |
| `verl/trainer/ppo/ray_trainer.py` | PPO 主训练循环，初始化教师 manager，并在 rollout 后进入 actor update。 |
| `verl/experimental/teacher_loop/teacher_model.py` | `MultiTeacherModelManager` / `TeacherModelManager`，负责资源池切分和教师服务启动。 |
| `verl/experimental/teacher_loop/teacher_manager.py` | `AsyncTeacherLLMServerManager`，负责按样本路由并请求教师 logprobs。 |
| `verl/experimental/agent_loop/agent_loop.py` | rollout 后处理、教师 logprobs 写入 `DataProto`、agent worker 调度。 |
| `verl/trainer/distillation/losses.py` | 蒸馏 loss registry、top-k / estimator loss 与 PPO loss 的组合。 |
| `verl/trainer/distillation/{fsdp,megatron}/losses.py` | FSDP / Megatron 下 top-k forward KL 的具体计算。 |
| `verl/workers/utils/padding.py` | `teacher_logprobs` / `teacher_ids` 的 padding 到 nested/no-padding 转换。 |

---

# 一、入口与配置归一化：从“一个 teacher”扩展到“按样本路由”

## 1.1 设计哲学与核心问题

Multi-Teacher OPD 首先不是一个新 trainer，而是普通 PPO trainer 的一个可选能力。用户仍然运行：

```bash
python3 -m verl.trainer.main_ppo --config-name ppo_trainer.yaml \
  distillation.enabled=True \
  +distillation.teacher_models.gsm8k.key="openai/gsm8k" \
  +distillation.teacher_models.geo3k.key="hiyouga/geometry3k"
```

这一层要解决的是**配置问题和资源可计算问题**：

- 用户应该能通过 YAML/CLI 增加教师，而不是改 Python；
- 框架必须知道每个教师占多少 GPU，才能提前划分 Ray resource pool；
- 多教师路由不能依赖 YAML entry name，而要依赖数据样本里的业务 key；
- 单教师旧配置还要兼容，不能让已有 OPD 用户全部改写配置。

如果没有这一层，后面的 manager 不知道要启动几个教师、每个教师占多少卡，也不知道样本中的 `data_source` 应该匹配哪个模型。

## 1.2 源码入口与关键对象

```text
verl/trainer/config/ppo_trainer.yaml
  - defaults 中加载 distillation@distillation: distillation

verl/trainer/config/distillation/distillation.yaml
  - enabled / n_gpus_per_node / nnodes / teacher_models / teacher_key

verl/workers/config/distillation.py
  - DistillationLossConfig：loss mode、topk、是否走 policy gradient
  - DistillationTeacherModelConfig：单个教师的 key、model_path、inference、num_replicas
  - DistillationConfig.__post_init__：enabled 后执行配置归一化和资源校验
  - DistillationConfig._resolve_teacher_models：单教师 / 多教师分支的核心逻辑

verl/trainer/main_ppo.py
  - TaskRunner.init_resource_pool_mgr：根据 distillation.enabled 增加 teacher_pool
  - TaskRunner.add_teacher_model_resource_pool：把 Role.TeacherModel 映射到 teacher_pool
```

## 1.3 主流程拆解

配置入口从 `ppo_trainer.yaml` 开始。`verl/trainer/config/ppo_trainer.yaml:42-43` 默认加载 distillation 配置组：

```yaml
# distillation config
- distillation@distillation: distillation
```

这意味着用户不需要额外指定 config group，只要覆盖 `distillation.*` 即可。默认配置在 `verl/trainer/config/distillation/distillation.yaml:13-14` 关闭：

```yaml
enabled: false
```

真正开启行为的是 `distillation.enabled=True`。`main_ppo.py` 看到它之后会创建教师资源池：

```text
verl/trainer/main_ppo.py:176-185
  distillation_config = config.get("distillation")
  if is_distillation_enabled(distillation_config):
      validate n_gpus_per_node / nnodes
      teacher_pool = [distillation_config.n_gpus_per_node] * distillation_config.nnodes
      resource_pool_spec["teacher_pool"] = teacher_pool
```

随后 `verl/trainer/main_ppo.py:203-210` 把 `Role.TeacherModel` 映射到这个 pool。注意注释说得很直白：这里**不注册普通 teacher worker**，只是注册资源池。后面教师服务由 `MultiTeacherModelManager` 自己启动。

配置归一化发生在 `DistillationConfig.__post_init__`：

```text
verl/workers/config/distillation.py:257-275
  if not enabled: return
  teacher_models = _resolve_teacher_models()
  for teacher in teacher_models:
      teacher.validate_and_prepare_for_distillation(...)
      teacher_world_size_sum += teacher.world_size
  assert teacher_world_size_sum == n_gpus_per_node * nnodes
```

单教师和多教师的分叉在 `_resolve_teacher_models()`：

```text
verl/workers/config/distillation.py:277-308
  assert "teacher_model" in self.teacher_models

  if len(self.teacher_models) == 1:
      # 单教师：默认 teacher_model 占满整个 teacher pool
      teacher_model.num_replicas = pool_size // per_replica_world_size
      teacher_model.key = "default"
  else:
      # 多教师：删除默认占位项 teacher_model
      self.teacher_models.pop("teacher_model")

  # 最终 dict 用 teacher_config.key 重新索引，而不是 YAML entry name
  teacher_models[teacher_config.key] = teacher_config
```

这里的关键点是：**多教师最终的字典 key 是样本路由 key，而不是 YAML 中的 `gsm8k` / `geo3k` 变量名**。示例脚本里：

```bash
+distillation.teacher_models.gsm8k.key="openai/gsm8k"
+distillation.teacher_models.geo3k.key="hiyouga/geometry3k"
```

`gsm8k` / `geo3k` 只是 Hydra entry name，运行时真正匹配的是 `openai/gsm8k` 和 `hiyouga/geometry3k`。

教师模型的 GPU 占用由两个 property 给出：

```text
verl/workers/config/distillation.py:140-150
  per_replica_world_size = TP * DP * PP
  world_size = num_replicas * per_replica_world_size
```

因此一个教师模型不是“占 1 张卡”或“占整个 pool”，而是由 `num_replicas` 和 inference 并行度共同决定。`__post_init__` 要求所有教师 `world_size` 之和严格等于 `distillation.n_gpus_per_node * distillation.nnodes`（`distillation.py:269-275`）。这是一种静态资源模型：启动前算清楚，运行中不动态抢占。

## 1.4 关键细节与误区澄清

**误区一：Multi-Teacher OPD 有单独的开关。**
没有。源码里 `need_teacher_policy()` 只是 `is_distillation_enabled(config.get("distillation"))`，见 `verl/trainer/ppo/utils.py:82-86`。是否多教师取决于 `teacher_models` 里实际配置了几个 teacher，而不是另一个 `multi_teacher=True` 字段。

**误区二：默认的 `teacher_model` 可以作为多教师中的第一个教师。**
不可以。`DistillationConfig._resolve_teacher_models()` 在多教师时无条件 `pop("teacher_model")`（`distillation.py:296-299`）。源码注释和 YAML 注释都提醒了这一点：`verl/workers/config/distillation.py:227-245`、`verl/trainer/config/distillation/distillation.yaml:67-81`。如果用户沿用单教师写法再加第二个 teacher，第一个可能被静默丢弃。

**误区三：`topk` 只是 loss 里的一个数字。**
不是。`DistillationTeacherModelConfig.validate_and_prepare_for_distillation()` 会根据 loss 是否需要 top-k 修改教师 inference 配置：vLLM 路径会设置或校验 `engine_kwargs.vllm.max_logprobs >= topk`（`distillation.py:176-196`）；SGLang 则在请求时翻译 per-request 参数（`distillation.py:197-202`）。因此 `topk` 同时影响教师服务启动能力、RPC 返回体大小和 actor loss。

**误区四：`num_replicas` 默认 0 会自然报错。**
多教师路径中 `check_configured()` 只检查 `num_replicas is None`，不检查 `> 0`（`distillation.py:152-158`）。如果某个教师漏配 `num_replicas`，但其他教师刚好占满 pool，资源总和校验仍可能通过；直到路由到这个教师才会暴露空 server 问题。这是当前源码里的一个边界风险。

## 1.5 本章小结

💡 小结

- Multi-Teacher OPD 是 `distillation.enabled` 下的配置扩展，不是独立 trainer。
- 配置归一化把 YAML entry name 转成样本路由 key，最终以 `teacher_config.key` 建索引。
- 教师资源是静态预算：所有教师 `world_size` 之和必须等于 teacher pool 大小。
- 默认 `teacher_model` 是单教师兼容占位，多教师时要避免把它当真实 teacher 使用。

---

# 二、教师资源池与服务初始化：把多个 teacher 变成可路由推理集群

## 2.1 设计哲学与核心问题

配置解决了“要几个教师”的问题，初始化层要解决的是**资源隔离和服务发现问题**。

verl 的 actor、rollout、critic 通常通过 `RayWorkerGroup` 组织；但教师模型在 Multi-Teacher OPD 中不是训练 worker，不参与反向传播，也不需要 actor checkpoint。它更像一组独立推理服务：每个教师有自己的模型路径、inference backend、TP/DP/PP、replica 数和 load balancer。

如果把教师混进 actor worker 组，会带来两个问题：

- 资源生命周期和 actor/rollout 权重同步耦合，浪费或污染训练流程；
- 多教师之间模型路径不同、并行度不同，用一个 worker class 很难表达。

当前实现选择了另一条路：`Role.TeacherModel` 只占资源池，实际服务由 `MultiTeacherModelManager` 自己按 key 启动。

## 2.2 源码入口与关键对象

```text
verl/trainer/ppo/ray_trainer.py
  - RayPPOTrainer.init_workers：use_teacher_policy 时创建 MultiTeacherModelManager
  - AgentLoopManager.create：把 teacher_model_manager 传给 agent loop

verl/experimental/teacher_loop/teacher_model.py
  - MultiTeacherModelManager：拥有总 teacher pool，按 teacher.world_size 切分
  - TeacherModelManager：启动某一个 teacher 的 rollout replicas 和 load balancer
  - _validate_replica_node_alignment：校验线性切分是否跨节点异常

verl/single_controller/ray/base.py
  - split_resource_pool：按 split_size 切 SubRayResourcePool

verl/workers/rollout/{vllm,sglang,trtllm}_rollout/*async_server.py
  - vLLM / SGLang 支持 teacher server 命名
  - TRT-LLM 当前拒绝 is_teacher_model=True
```

## 2.3 主流程拆解

`RayPPOTrainer.init_workers()` 中，actor/critic/reward loop 初始化之后，才初始化教师：

```text
verl/trainer/ppo/ray_trainer.py:828-840
  if self.use_teacher_policy:
      teacher_resource_pool = resource_pool_manager.get_resource_pool(Role.TeacherModel)
      self.teacher_model_manager = MultiTeacherModelManager(config, teacher_resource_pool)
      self.distillation_config = omega_conf_to_dataclass(self.config.distillation)
```

`MultiTeacherModelManager` 做两件事：解析配置、切资源池。

```text
verl/experimental/teacher_loop/teacher_model.py:168-193
  distillation_config = omega_conf_to_dataclass(config.distillation)
  split_sizes = [teacher.world_size for teacher in teacher_models.values()]
  split_pools = split_resource_pool(resource_pool, split_size=split_sizes)

  for (key, teacher_config), teacher_pool in zip(...):
      manager = TeacherModelManager(..., resource_pool=teacher_pool)
      server_addresses[key] = manager.server_addresses
      server_handles[key] = manager.server_handles
      load_balancer_handle[key] = manager.load_balancer_handle
```

`split_resource_pool()` 的实现是线性切分。`verl/single_controller/ray/base.py:280-311` 先把 `split_size` 变成 list，再根据 cumulative sum 生成 `SubRayResourcePool(start_bundle_index, subgroup_world_size)`。这意味着切分顺序取决于 `teacher_models.values()` 的顺序，也意味着某个 teacher replica 可能跨节点。

所以 `TeacherModelManager` 里专门有一个节点对齐校验：

```text
verl/experimental/teacher_loop/teacher_model.py:102-143
  expected_span = ceil(per_replica_world_size / n_gpus_per_node)
  observed_span = last_node - first_node + 1
  if observed_span != expected_span:
      raise ValueError("Reorder teachers or adjust num_replicas / inference parallelism...")
```

启动某个 teacher replica 的逻辑在 `TeacherModelManager._initialize_llm_servers()`：

```text
verl/experimental/teacher_loop/teacher_model.py:61-100
  expected_pool_size = num_replicas * per_replica_world_size
  rollout_replica_class = get_rollout_replica_class(teacher_model_config.inference.name)
  model_config = HFModelConfig(path=teacher_model_config.model_path)
  name_suffix = teacher_key.replace("/", "_")

  rollout_replica_class(
      replica_rank=..., config=teacher_inference_config,
      model_config=model_config,
      is_teacher_model=True,
      name_suffix=name_suffix,
  )
  server.init_colocated(sub_resource_pool)
  collect server_handles / server_addresses
```

用 ASCII 图表示示例脚本的资源拓扑：

```text
trainer.n_gpus_per_node = 2       # 学生 actor/rollout pool
teacher_pool_world_size = 2       # 教师 pool，独立于学生 pool

Teacher pool bundles:
  bundle0                  bundle1
  GSM8K teacher replica    Geo3K teacher replica
  TP=1,DP=1,PP=1           TP=1,DP=1,PP=1

MultiTeacherModelManager
  ├─ key "openai/gsm8k"
  │    └─ TeacherModelManager -> vLLM/SGLang server(s) -> LB actor
  └─ key "hiyouga/geometry3k"
       └─ TeacherModelManager -> vLLM/SGLang server(s) -> LB actor
```

这不是 torch distributed 里的一个新 process group。教师内部如果用 TP/DP/PP，会由对应 rollout backend 自己管理；从 trainer 的视角看，教师是一组 Ray actor 地址和 load balancer handle。

## 2.4 关键细节与误区澄清

**误区一：`Role.TeacherModel` 会像 actor/critic 一样生成一个 worker group。**
不会。`main_ppo.py:203-210` 注释明确说 teacher model workers 不注册到 role-worker mapping，只注册 resource pool。真正启动服务的是 `MultiTeacherModelManager`。

**误区二：教师会随着 actor checkpoint/update_weights 同步权重。**
不会。教师模型用 `HFModelConfig(path=teacher_model_config.model_path)` 初始化（`teacher_model.py:76-89`），是静态教师服务。actor 到学生 rollout 的权重同步由 `CheckpointEngineManager` 管理，初始化时只传入 `self.async_rollout_manager.rollout_replicas`，见 `ray_trainer.py:874-878`；teacher replicas 不在其中。

**误区三：只要 rollout backend 支持，teacher 就支持。**
不完全。`DistillationTeacherModelConfig._validate_topk_logprobs()` 只显式支持 `vllm` 和 `sglang`（`distillation.py:182-206`）；TRT-LLM replica 在 `is_teacher_model=True` 时直接 `raise NotImplementedError`（`verl/workers/rollout/trtllm_rollout/trtllm_async_server.py:336-349`）。

**误区四：多教师资源切分天然按节点对齐。**
不是。`split_resource_pool()` 只是线性切 bundle（`base.py:288-307`），所以源码才额外做 `_validate_replica_node_alignment()`。当 `per_replica_world_size` 与每节点 GPU 数不整齐，或教师顺序不合适时，会报错要求用户调整顺序/副本/并行度。

## 2.5 本章小结

💡 小结

- 教师不是训练 worker，而是一组由 manager 启动的独立推理服务。
- MultiTeacherModelManager 用静态 `world_size` 切分 teacher pool，换来简单可控的资源隔离。
- 资源切分是线性的，节点对齐不是天然保证，源码专门做了校验。
- 教师生命周期独立于 actor checkpoint 和 rollout 权重同步，这既简化了语义，也带来常驻资源和清理风险。

---

# 三、样本路由与教师查询：真正的 Multi-Teacher 发生在 AgentLoop 后处理

## 3.1 设计哲学与核心问题

“多教师”真正发生的地方不是配置解析，也不是资源池切分，而是每条样本 rollout 完成后的后处理。因为只有在这里，框架同时拿到了：

- 这条样本的 prompt 和 response；
- 样本的非 tensor 元信息，例如 `data_source`；
- 可用的 teacher server 地址和 load balancer；
- 是否是 validation 的上下文。

这一层解决的是**数据路由和教师信号采集问题**。如果没有这一层，actor update 时只会看到学生生成结果和 reward，不会有 `teacher_logprobs` / `teacher_ids`。

## 3.2 源码入口与关键对象

```text
verl/experimental/agent_loop/agent_loop.py
  - AgentLoopManager.generate_sequences：把 batch 切给多个 agent loop worker
  - AgentLoopWorker.generate_sequences：每条样本创建 async task
  - AgentLoopWorker._agent_loop_postprocess：reward 和 teacher logprobs 后处理
  - AgentLoopWorker._compute_teacher_logprobs：从 sample_kwargs 中取 teacher_key 并请求教师
  - AgentLoopWorker._postprocess：把 teacher fields 拼进 DataProto.batch

verl/experimental/teacher_loop/teacher_manager.py
  - AsyncTeacherLLMServerManager.__init__：校验 server keys 与 teacher keys 一致
  - _resolve_teacher_key：单教师忽略 routing key，多教师严格匹配
  - compute_teacher_logprobs_single：调用具体 teacher 的 AsyncLLMServerManager.generate
  - _pad_teacher_outputs：把变长 teacher 输出 pad 回 batch 形状
```

## 3.3 主流程拆解

训练主循环在 `ray_trainer.py:1357-1368` 调用：

```text
gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
```

`AgentLoopManager.generate_sequences()` 会把 batch 切到多个 agent loop worker，并等待所有 worker：

```text
verl/experimental/agent_loop/agent_loop.py:1191-1215
  chunks = prompts.chunk(len(agent_loop_workers))
  outputs = await asyncio.gather(worker.generate_sequences.remote(chunk) ...)
  output = DataProto.concat(outputs)
```

每个 worker 内部又按样本创建任务：

```text
verl/experimental/agent_loop/agent_loop.py:594-608
  for i in range(len(batch)):
      kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
      task = self._run_agent_loop(..., **kwargs)
  outputs = await asyncio.gather(*tasks)
  return self._postprocess(outputs, input_non_tensor_batch=...)
```

这里的 `kwargs` 很关键：`data_source`、`uid`、`raw_prompt`、多模态字段等都在这里按样本切出来。教师路由依赖的 `teacher_key` 就是从这里取的。

`_agent_loop_postprocess()` 先做 padding、multi-modal inputs、position_ids，再计算 reward，然后计算教师 logprobs：

```text
verl/experimental/agent_loop/agent_loop.py:739-756
  multi_modal_inputs = _compute_multi_modal_inputs(...)
  position_ids = _compute_position_ids(...)
  await _compute_score(...)
  await _compute_teacher_logprobs(
      output,
      prompt_ids=output.prompt_ids,
      response_ids=output.response_ids,
      validate=validate,
      sample_kwargs=kwargs,
  )
```

`_compute_teacher_logprobs()` 是路由入口：

```text
verl/experimental/agent_loop/agent_loop.py:893-915
  if self.distillation_enabled and not validate:
      routing_value = sample_kwargs.get(self.teacher_key)
      routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value
      teacher_ids, teacher_logprobs = await teacher_server_manager.compute_teacher_logprobs_single(
          sequence_ids=prompt_ids + response_ids,
          multi_modal_data=output.multi_modal_data,
          routing_key=routing_key,
      )
      output.extra_fields["teacher_ids"] = teacher_ids
      output.extra_fields["teacher_logprobs"] = teacher_logprobs
```

进入 `AsyncTeacherLLMServerManager` 后，路由规则很硬：

```text
verl/experimental/teacher_loop/teacher_manager.py:98-112
  if len(teacher_model_configs) == 1:
      return only_teacher_key
  if routing_key is None:
      raise ValueError
  if routing_key not in teacher_model_configs:
      raise ValueError
  return routing_key
```

因此 Multi-Teacher OPD 不是 ensemble，也不是 fallback，而是**每条样本严格选择一个教师**。

教师请求参数由 `_get_teacher_sampling_params()` 给出：

```text
verl/experimental/teacher_loop/teacher_manager.py:31-44
  if teacher temperature != 1.0: raise NotImplementedError
  num_logprobs = topk if loss_settings.use_topk else 0
  return {"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": num_logprobs}
```

这里看似生成 `max_tokens=1`，但训练真正消费的是 `prompt_logprobs`：teacher 对 `prompt_ids + response_ids` 这个完整输入序列返回每个位置的 logprob / top-k logprob。返回后写入：

```text
verl/experimental/teacher_loop/teacher_manager.py:132-137
  teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
  teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
```

最后 `_pad_teacher_outputs()` 把单样本变长输出 pad 到 `[prompt_width + response_width, K]`，再在 `_postprocess()` 中 batch concat：

```text
verl/experimental/teacher_loop/teacher_manager.py:47-63
  F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0)
  F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0)

verl/experimental/agent_loop/agent_loop.py:936-938
  optional_outputs["teacher_logprobs"] = torch.cat(...)
  optional_outputs["teacher_ids"] = torch.cat(...)
```

## 3.4 关键细节与误区澄清

**误区一：validation 也会请求教师。**
不会。`_compute_teacher_logprobs()` 条件是 `self.distillation_enabled and not validate`（`agent_loop.py:902`）。验证阶段只跑 rollout/reward 逻辑，不产生 teacher logprobs。

**误区二：缺少 `data_source` 会自动走默认教师。**
多教师不会。只有单教师路径会忽略 routing key（`teacher_manager.py:98-101`）。多教师下如果 `sample_kwargs[self.teacher_key]` 不存在或值不在 configured teacher keys 中，会抛 `ValueError`（`teacher_manager.py:102-111`）。这意味着数据集的 `data_source` 字符串必须和配置里的 `teacher.key` 精确一致。

**误区三：教师 logprob 是 batch 请求。**
当前主路径是逐样本调用 `compute_teacher_logprobs_single()`（`agent_loop.py:909-913`），只是多个样本 task 通过 `asyncio.gather` 并发。它没有先按 teacher key 分组做 batch teacher request。长序列、大 batch、多教师不均衡时，这会成为吞吐瓶颈。

**误区四：文档里的 one-step/two-step OPD scheduler 就是当前 MOPD 主路径。**
`docs/advance/async-on-policy-distill.md:26-64` 描述了 one-step/two-step scheduler，`docs/advance/async-on-policy-distill.md:99-114` 还提到 `OnPolicyDistillTrainer`、`TeacherClient` 等对象。但当前 worktree 源码中的 MOPD 主路径是 `MultiTeacherModelManager + AsyncTeacherLLMServerManager + AgentLoopManager`。本文以源码为准；旧文档可作为背景，不应当当作当前主调用链。

## 3.5 本章小结

💡 小结

- 真正的多教师路由发生在每条样本 rollout 后的 AgentLoop 后处理阶段。
- 多教师是严格一对一路由，不是 ensemble，也没有 fallback。
- 教师查询只在训练路径执行，validation 不请求教师。
- 当前实现逐样本请求教师 logprobs，靠 async 并发隐藏部分延迟，但没有按 teacher 分组批量化。

---

# 四、Actor 更新与蒸馏损失：teacher top-k 如何进入前向 / 反向

## 4.1 设计哲学与核心问题

拿到 `teacher_logprobs` 还不等于能训练。actor update 面临的是**shape 对齐、显存控制和分布式 loss 语义**问题：

- rollout 后的 batch 是 padding 格式，actor 前向前会转成 remove-padding / nested tensor；
- FSDP + Ulysses SP 会沿序列维切分，Megatron 还可能沿 vocab 维 TP 切分；
- top-k forward KL 需要 student logits，而普通 PPO 只需要 sampled token logprob；
- 对 Megatron vocab parallel，教师 top-k ids 是全局 vocab id，但每个 TP rank 只持有本地 vocab shard。

这一层的目标是在不重写 actor engine 的情况下，把 distillation loss 变成 actor 的 `loss_fn`，并在需要 top-k 时插入 logits processor。

## 4.2 源码入口与关键对象

```text
verl/trainer/ppo/ray_trainer.py
  - _update_actor：设置 distillation_use_topk，并调用 actor_rollout_wg.update_actor

verl/workers/engine_workers.py
  - ActorRolloutRefWorker.init_model：distillation enabled 时把 distillation_ppo_loss 设为 actor loss_fn
  - update_actor -> TrainingWorker.train_mini_batch -> train_batch

verl/trainer/distillation/losses.py
  - distillation_ppo_loss：既可作为 logits processor，也可作为 final loss
  - distillation_loss：根据 loss_mode 聚合监督式或 policy-gradient 式蒸馏
  - compute_forward_kl_topk：从 model_output 中取 logits processor 结果
  - compute_distillation_loss_reverse_kl_estimator：k1/k2/k3 等 sampled-token estimator

verl/workers/engine/fsdp/transformer_impl.py
  - prepare_model_outputs：FSDP 非 fused logits 路径下调用 logits_processor_func

verl/workers/engine/megatron/transformer_impl.py
  - logits_processor：Megatron 非 fused 路径下调用 logits_processor_func

verl/trainer/distillation/fsdp/losses.py
  - FSDP top-k forward KL：SP 切分 teacher top-k，gather student top-k logprob

verl/trainer/distillation/megatron/losses.py
  - Megatron top-k forward KL：vocab parallel log-softmax + 自定义 autograd
```

## 4.3 主流程拆解

训练主循环先把 rollout 输出并回原 batch：

```text
verl/trainer/ppo/ray_trainer.py:1398-1401
  batch = batch.repeat(repeat_times=rollout.n, interleave=True)
  batch = batch.union(gen_batch_output)
```

此时 `gen_batch_output.batch` 已经包含 `teacher_logprobs` / `teacher_ids`。actor update 前，`_update_actor()` 会把 padding batch 转成 no-padding：

```text
verl/trainer/ppo/ray_trainer.py:1202-1235
  batch_td = batch.to_tensordict()
  batch_td = left_right_2_no_padding(batch_td)
  distillation_use_topk = self.distillation_config.distillation_loss.loss_settings.use_topk
  tu.assign_non_tensor(..., distillation_use_topk=distillation_use_topk, compute_loss=True)
  actor_output = self.actor_rollout_wg.update_actor(batch_td)
```

`left_right_2_no_padding()` 对教师字段有专门处理：

```text
verl/workers/utils/padding.py:83-94
  teacher_logprobs: (bsz, seqlen, topk)
  teacher_ids:      (bsz, seqlen, topk)
  index_first_axis(..., indices)
  nested_tensor_from_jagged(..., offsets=cu_seqlens)
```

也就是说，教师输出先被 pad 成 dense，再在 actor update 前转成 nested/no-padding。shape 可以这样理解：

```text
AgentLoop 后:
  teacher_logprobs: [B, P + R, K]
  teacher_ids:      [B, P + R, K]

left_right_2_no_padding 后:
  teacher_logprobs.values(): [total_nnz, K]
  teacher_ids.values():      [total_nnz, K]
  offsets:                   [B + 1]
```

actor worker 初始化时，distillation 会替换普通 PPO loss：

```text
verl/workers/engine_workers.py:573-580
  if self.distillation_enabled:
      self.loss_fn = partial(distillation_ppo_loss, config=actor_config, distillation_config=distillation_config)
  else:
      self.loss_fn = partial(ppo_loss, config=actor_config)
```

`distillation_ppo_loss()` 有双重身份：

```text
verl/trainer/distillation/losses.py:165-222
  if student_logits is not None:
      # logits processor 路径：计算 top-k token loss
      return compute_topk_loss(...)

  # final loss 路径：聚合 distillation loss，并与 PPO loss 组合
  distill_loss, distill_metrics = distillation_loss(...)
  policy_loss, policy_metrics = ppo_loss(...)
  if not use_task_rewards:
      policy_loss = 0.0
  policy_loss += distill_loss * coef
```

### FSDP 路径：SP 切序列，top-k 在局部序列上算

FSDP engine 在非 fused logits 路径里，如果 `distillation_use_topk=True`，会把 student logits 交给 loss function 当 logits processor：

```text
verl/workers/engine/fsdp/transformer_impl.py:1068-1079
  if distillation_use_topk:
      outputs = logits_processor_func(student_logits=logits_rmpad.unsqueeze(0), data=micro_batch)
      if self.use_ulysses_sp:
          v = gather_outputs_and_unpad(v, ...)
      model_output[k] = nested_tensor_from_jagged(v, cu_seqlens)
```

FSDP top-k KL 里，教师 top-k 先取 nested values，再按 Ulysses SP 切序列：

```text
verl/trainer/distillation/fsdp/losses.py:55-74
  teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)
  teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)
  if sp_world_size > 1:
      teacher_topk_log_probs = slice_input_tensor(..., dim=1)
      teacher_topk_ids = slice_input_tensor(..., dim=1)
  student_log_probs = F.log_softmax(student_logits, dim=-1)
  student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
  distillation_losses = KL(P_teacher_topk || Q_student_on_topk)
```

### Megatron 路径：TP 切 vocab，自定义 autograd 还原梯度

Megatron 的非 fused `logits_processor` 在 `verl/workers/engine/megatron/transformer_impl.py:903-932`，同样只在 `distillation_use_topk=True` 时调用 loss function。

Megatron top-k KL 的复杂点在 vocab parallel：teacher ids 是全局 vocab id，但每个 TP rank 只有 `[vocab_start, vocab_end)`。`_VocabParallelKLDivergence.forward()` 做了几件事：

```text
verl/trainer/distillation/megatron/losses.py:85-175
  1. vocab_parallel_log_softmax:
       all_reduce MAX(logits_max)
       all_reduce SUM(exp_logits)
  2. 判断 teacher top-k id 是否落在本 rank vocab shard
  3. out-of-shard id 映射到 dummy index=0，并把概率 mask 为 0
  4. 计算本 shard per-token KL
  5. all_reduce SUM(per_token_kl_loss)
  6. all_reduce SUM(student top-k mass)
```

反向传播也不是交给 PyTorch 自动处理，而是自定义了梯度：

```text
verl/trainer/distillation/megatron/losses.py:177-218
  dL/dz_j = m_A * p_j - q_j * 1[j in active top-k]
  grad_input = source_probs * target_active_mass
  scatter_add_ subtract teacher probs on active top-k ids
  grad_input *= upstream grad_loss
```

这保证了 vocab shard 上的梯度与 full-vocab reference 对齐。测试 `tests/utils/test_special_megatron_kl_loss_tp.py:145-175` 正是在多 rank 下比较 Megatron vocab-parallel loss/grad 和 FSDP full-vocab reference。

## 4.4 关键细节与误区澄清

**误区一：所有 distillation loss 都需要 top-k logits processor。**
不是。`DistillationLossSettings` 把 loss 分成 `use_topk` 和 `use_estimator` 两类（`losses.py:46-68`）。`forward_kl_topk` 才需要 top-k logits processor；`k1/k2/k3/low_var_kl` 等 estimator 只需要 sampled-token teacher logprob 和 student logprob（`losses.py:335-370`）。

**误区二：`forward_kl_topk` 的 loss 是 final loss 阶段直接从 logits 算的。**
不是。top-k loss 在 engine 的 logits processor 阶段先算成 `model_output["distillation_losses"]`，final loss 阶段 `compute_forward_kl_topk()` 只是把它从 no-padding 转回 response padding 并聚合（`losses.py:294-332`）。

**误区三：`use_policy_gradient=True` 等于直接反传 teacher KL。**
不是。`distillation_loss()` 在 `use_policy_gradient=True` 时，把负的 distillation loss 当 advantage，调用 policy loss（`losses.py:257-279`）；在 `False` 时才通过 `agg_loss()` 直接监督式反传（`losses.py:280-289`）。配置里 `k1 + use_policy_gradient=True` 与 `forward_kl_topk + use_policy_gradient=False` 语义完全不同。

**误区四：fused kernels 与 top-k distillation 可以无脑组合。**
源码没有给这个组合做显式友好报错。FSDP 的 top-k logits processor 在非 fused 分支（`fsdp/transformer_impl.py:1041-1079`），Megatron 也是非 fused 分支才调用（`megatron/transformer_impl.py:878-932`）。示例脚本把 `USE_FUSED_KERNELS=False`（`run_qwen3_mopd_gsm8k_geo3k.sh:14`），这是合理的：top-k forward KL 需要 materialize logits。

## 4.5 本章小结

💡 小结

- Actor 初始化时把普通 PPO loss 替换为 `distillation_ppo_loss`，这是蒸馏进入训练的核心接点。
- `distillation_ppo_loss` 同时承担 logits processor 和 final loss 两种角色。
- FSDP 主要处理序列维 SP 对齐；Megatron 还要处理 vocab TP 下的全局 id 到本地 shard 映射。
- top-k loss 会迫使 student logits materialize，显存收益来自“不取教师 full logits”，不是“不取学生 logits”。

---

# 五、完整主路径串联

## 5.1 完整调用栈

下面把前面拆开的机制串成一次真实用户调用。以示例脚本为入口：

```text
User:
  python3 -m verl.trainer.main_ppo \
    distillation.enabled=True \
    distillation.teacher_key=data_source \
    +distillation.teacher_models.gsm8k.key="openai/gsm8k" \
    +distillation.teacher_models.geo3k.key="hiyouga/geometry3k" \
    distillation.distillation_loss.loss_mode=k1/topk/...

  │
  ├─ Step 1: Hydra 加载配置
  │     └─ ppo_trainer.yaml:42-43 加载 distillation config
  │
  ├─ Step 2: main_ppo 注册资源池
  │     ├─ main_ppo.py:176-185 创建 teacher_pool
  │     └─ main_ppo.py:203-210 映射 Role.TeacherModel -> teacher_pool
  │
  ├─ Step 3: RayPPOTrainer 初始化 worker / manager
  │     ├─ ray_trainer.py:689-807 初始化 actor/critic/rollout
  │     ├─ ray_trainer.py:828-837 创建 MultiTeacherModelManager
  │     └─ ray_trainer.py:859-865 创建 AgentLoopManager，并传入 teacher_model_manager
  │
  ├─ Step 4: 教师服务启动
  │     ├─ teacher_model.py:179-193 按 teacher.world_size 切资源池
  │     ├─ teacher_model.py:61-100 每个 teacher 启动 rollout replica
  │     └─ teacher_model.py:145-150 为每个 teacher 创建 load balancer
  │
  ├─ Step 5: 每个训练 step 的 rollout + teacher query
  │     ├─ ray_trainer.py:1357-1368 调 AgentLoopManager.generate_sequences
  │     ├─ agent_loop.py:594-608 每样本 async agent loop
  │     ├─ agent_loop.py:893-915 按 data_source 请求 teacher logprobs
  │     └─ agent_loop.py:936-938 teacher fields 写入 DataProto.batch
  │
  ├─ Step 6: PPO/GRPO 主训练逻辑
  │     ├─ ray_trainer.py:1398-1401 union rollout output
  │     ├─ ray_trainer.py:1444-1469 计算 old_log_probs
  │     ├─ ray_trainer.py:1490-1535 计算 advantage
  │     └─ ray_trainer.py:1550-1551 update actor
  │
  ├─ Step 7: actor 内部蒸馏 loss
  │     ├─ engine_workers.py:573-576 设置 distillation_ppo_loss
  │     ├─ padding.py:83-94 teacher tensors 转 nested/no-padding
  │     ├─ fsdp/megatron transformer_impl logits processor
  │     └─ distillation/losses.py:207-222 组合 distill loss 和 PPO loss
  │
  └─ Step 8: checkpoint / rollout 权重同步
        ├─ ray_trainer.py:913-934 保存 actor/critic/dataloader
        ├─ ray_trainer.py:1575-1577 同步 actor -> student rollout
        └─ teacher 不保存、不同步，由 model_path + config 下次启动重建
```

## 5.2 每一层做了什么

| 层 | 输入 | 输出 / 状态变化 | 通信 | 显存影响 | 频率 |
|---|---|---|---|---|---|
| 配置归一化 | Hydra `distillation.*` | `teacher_models` 按 routing key 重建，校验 world size | 无 | 无 | 启动一次 |
| 资源池注册 | trainer / distillation pool size | `teacher_pool` 加入 Ray resource spec | Ray placement group | 预留教师 GPU | 启动一次 |
| 教师服务启动 | teacher config + sub pool | server handles / addresses / load balancer | Ray actor 创建，backend 内部通信 | 加载教师模型和 KV cache | 启动一次 |
| rollout | prompt batch | response、response_mask、可选 rollout logprobs | 请求学生 rollout server | 学生 rollout KV cache | 每 step |
| teacher query | prompt_ids + response_ids + routing key | `teacher_logprobs` / `teacher_ids` | Ray RPC 到对应 teacher server | 教师 KV cache、CPU/Ray tensor | 每训练样本 |
| actor update | DataProto + teacher fields | actor loss/grad/metrics | FSDP/Megatron DP/SP/TP 通信 | actor 激活、student logits、teacher top-k tensor | 每 step |
| weight sync | actor params | 更新学生 rollout 权重 | checkpoint engine / collectives | rollout sleep/wake 释放/恢复 cache | 每 actor update 后 |
| checkpoint | actor/critic/dataloader | checkpoint 文件 | 文件系统 / 可选远端 | 不保存 teacher | save_freq |

## 5.3 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在当前主流程 |
|---|---|---|
| `docs/advance/async-on-policy-distill.md` 中的 `OnPolicyDistillTrainer` | 文档描述了 OPD scheduler 和独立 teacher client | 当前源码主路径未发现该 trainer；MOPD 走 `AgentLoopManager`。 |
| `verl/experimental/one_step_off_policy/*` | 名字里有 off-policy，文档也提过 OPD overlap | 它是 one-step off-policy PPO 方向，不是当前 Multi-Teacher OPD 主路径。 |
| `verl/trainer/main_ppo_sync.py` | 也接入了 `MultiTeacherModelManager` | 属于 TransferQueue / sync trainer 路径；示例 MOPD 使用 `main_ppo.py`。且 code review 指出 list output + OPD 有 bug。 |
| vLLM FP8 patch (`vllm_async_server.py:828-832`) | 文件里出现 patch | 这是 FP8 量化相关 patch，不是 Multi-Teacher OPD 的 monkey patch。 |
| CheckpointEngineManager 的 `replicas` | 容易以为包含所有 rollout-like replicas | `ray_trainer.py:874-878` 只传学生 rollout replicas，不包含 teacher replicas。 |
| `teacher_key` 在单教师配置中 | 配置存在 | 单教师 `_resolve_teacher_key()` 忽略 routing key，所有样本走唯一 teacher。 |

## 5.4 本章小结

💡 小结

- 一次 MOPD 训练仍是普通 `main_ppo` 主循环，只是在 rollout 后多了一次 teacher query，并在 actor loss 中换成 distillation loss。
- 教师服务启动在 worker 初始化阶段完成；教师查询每个训练样本都会执行。
- checkpoint 和 actor->rollout 权重同步只覆盖学生侧，不覆盖教师侧。
- 旧 OPD scheduler 文档与当前 MOPD 主路径不一致，阅读时应以源码调用链为准。

---

# 六、关键数据流 / 状态流 / Shape 流程

## 6.1 Tensor shape 变化

以 batch size `B`、prompt 最大长度 `P`、response 最大长度 `R`、top-k `K` 为例。示例脚本中 `P=1024`，`R=2048`，`K=64`。

```text
原始 rollout 后单样本:
  prompt_ids:        [prompt_len]
  response_ids:      [response_len]
  sequence_ids:      [prompt_len + response_len]

Teacher 返回:
  teacher_ids:       [prompt_len + response_len, K]      # top-k token ids
  teacher_logprobs:  [prompt_len + response_len, K]      # top-k logprobs

AgentLoop pad 后:
  teacher_ids:       [1, P + R, K]
  teacher_logprobs:  [1, P + R, K]

Batch concat 后:
  teacher_ids:       [B, P + R, K]
  teacher_logprobs:  [B, P + R, K]

left_right_2_no_padding 后:
  teacher_ids.values():       [total_nnz, K]
  teacher_logprobs.values():  [total_nnz, K]
  offsets:                    [B + 1]

FSDP + SP top-k loss:
  teacher_topk:      [1, total_nnz / sp_size, K]
  student_logits:    [1, total_nnz / sp_size, vocab_size]
  distill_loss:      [1, total_nnz / sp_size]
  gather 后 nested:  [total_nnz]

最终 response loss:
  no_padding_2_padding(...): [B, R]
```

为什么先 pad 再 remove padding？因为 AgentLoop 的后处理仍然围绕 padded `DataProto` 构造，`teacher_manager.py:56` 还有 TODO：“remove padding and use tensordict”。真正进入 actor update 前，`padding.py:83-94` 才把 teacher fields 转成 nested tensor。这是兼容现有 batch 管线的做法，但会产生额外 dense tensor 和复制。

哪一步节省显存？

- 教师侧只返回 sampled-token logprob 或 top-k logprobs，而不是 full vocab logits，避免 `[B, S, V]` 级别的教师输出。
- actor 侧 remove-padding 避免在 padding token 上做前向/loss。
- FSDP + Ulysses SP 或 Megatron CP/TP 会进一步切分序列/vocab 计算。

哪一步显存收益会消失？

- top-k forward KL 仍需要 student logits；FSDP/Megatron fused kernels 路径不会调用 top-k logits processor，因此 top-k loss 通常需要非 fused logits materialization。
- AgentLoop 到 actor update 之间的 `[B, P+R, K]` dense teacher tensor 仍然存在。以 `B=128, P+R=3072, K=64` 估算，单 `teacher_logprobs` 若为 fp32 约 `128*3072*64*4 ≈ 96MB`；`teacher_ids` 也有几十 MB。再考虑 Ray object、TensorDict、pad/unpad 临时副本，内存压力会明显放大。

## 6.2 Rank / Mesh / Process Group 变化

Multi-Teacher OPD 同时有两类“并行”：

1. **Ray 资源池层面的 teacher replica 并行**；
2. **actor engine 内部的 FSDP/Megatron 并行**。

它们不是同一个 group。

示例：teacher pool 有 8 张 GPU，每节点 4 张，两个教师：

```text
n_gpus_per_node = 4
teacher_pool world_size = 8

Teacher A:
  num_replicas = 1
  TP=2, DP=1, PP=1
  per_replica_world_size = 2
  world_size = 2

Teacher B:
  num_replicas = 2
  TP=1, DP=1, PP=1
  per_replica_world_size = 1
  world_size = 2

Teacher C:
  num_replicas = 1
  TP=4, DP=1, PP=1
  per_replica_world_size = 4
  world_size = 4

split_sizes = [2, 2, 4]

Bundle layout:
  node0: [0,1] Teacher A replica0
         [2]   Teacher B replica0
         [3]   Teacher B replica1
  node1: [4,5,6,7] Teacher C replica0
```

`split_resource_pool()` 只知道 bundle 序号，不理解模型语义；`TeacherModelManager._validate_replica_node_alignment()` 才检查某个 replica 是否跨节点异常。

Actor 侧则另有自己的 DP/SP/TP/CP：

- FSDP top-k loss 在 `fsdp/losses.py:59-63` 根据 Ulysses SP world size 切 teacher top-k；
- Megatron top-k loss 在 `megatron/losses.py:37-55` 和 `157-172` 对 TP group 做 all-reduce；
- teacher RPC 和 actor loss 之间没有梯度通信，teacher 是 frozen inference。

## 6.3 状态切换与全局状态

Multi-Teacher OPD 里有几类状态：

```text
配置状态:
  DistillationConfig.__post_init__
    - 重写 teacher_models dict
    - 修改 teacher inference.prompt_length / response_length
    - 写入 loss_settings

运行时服务状态:
  MultiTeacherModelManager
    - server_addresses: dict[key, list[str]]
    - server_handles: dict[key, list[ActorHandle]]
    - load_balancer_handle: dict[key, ActorHandle]

AgentLoop worker 状态:
  AgentLoopWorker.__init__
    - distillation_enabled
    - teacher_key
    - teacher_server_manager

Loss registry 状态:
  register_distillation_loss decorator
    - DISTILLATION_LOSS_REGISTRY
    - DISTILLATION_SETTINGS_REGISTRY
```

这些状态大多是进程内或 Ray actor 内状态，不是跨进程全局变量。需要注意两个点：

- `DistillationLossConfig.__post_init__` 中 `self._mutable_fields.add("loss_settings")`（`distillation.py:88-92`）会修改 set，code review 认为有共享可变状态隐患。
- `register_distillation_loss` 是 import-time registry（`losses.py:71-88`），不是 monkey patch，但它仍依赖模块被正确 import；`verl/trainer/distillation/__init__.py:1-14` 通过 `from .losses import *` 触发注册。

## 6.4 通信流

Multi-Teacher OPD 新增和复用的通信可以分开看：

```text
每个训练样本:
  AgentLoopWorker -> Teacher AsyncLLMServerManager -> teacher rollout server
    通信: Ray RPC / backend HTTP-like generate / load balancer
    返回: prompt_ids + prompt_logprobs

每个 actor update micro-batch:
  FSDP + SP:
    可能 gather_outputs_and_unpad 聚合 SP 切分结果

  Megatron + TP:
    all_reduce MAX(logits_max)
    all_reduce SUM(sum_exp_logits)
    all_reduce SUM(per_token_kl_loss)
    all_reduce SUM(topk mass)

每个 actor update 后:
  CheckpointEngineManager.update_weights
    actor -> student rollout 权重同步
    teacher 不参与
```

新增通信的核心不是 all-gather 参数，而是**每样本教师 RPC + top-k tensor 回传**。如果 teacher 慢或某个 teacher key 数据占比很高，对应 load balancer 会成为 step 等待的尾部延迟来源。

## 6.5 本章小结

💡 小结

- `teacher_logprobs` 的主 shape 是 `[B, P+R, K] -> nested [total_nnz, K] -> response [B, R]`。
- Teacher 资源并行和 actor 训练并行是两套体系：前者是 Ray service，后者是 FSDP/Megatron process group。
- 教师 RPC 不参与反向传播；梯度只在学生 actor 内部流动。
- 当前 dense padding 是显存/CPU/Ray object 压力的主要来源之一。

---

# 七、核心机制深挖

## 7.1 “没有 Monkey Patch”的接入：零侵入靠 manager + registry，而不是替换上游函数

### 设计哲学与核心问题

很多分布式框架把新能力接入现有训练链路时，会用 monkey patch 替换下游库函数。Multi-Teacher OPD 当前主路径没有这么做。它的零侵入主要来自三点：

1. 用 `MultiTeacherModelManager` 管教师服务，不改 actor/critic worker 抽象；
2. 用 `AgentLoopWorker._compute_teacher_logprobs()` 在 rollout 后处理阶段附加字段；
3. 用 `distillation_ppo_loss` 替换 actor loss_fn，并复用 engine 的 logits_processor 扩展点。

### 源码入口与关键对象

```text
verl/trainer/distillation/losses.py
  - register_distillation_loss：loss registry，不是 monkey patch

verl/experimental/teacher_loop/teacher_manager.py
  - AsyncTeacherLLMServerManager：组合多个 AsyncLLMServerManager

verl/workers/rollout/sglang_rollout/async_sglang_server.py
  - 把 vLLM-style prompt_logprobs 翻译成 SGLang per-request logprob 参数
```

### 主流程拆解

loss 注册是 decorator：

```text
verl/trainer/distillation/losses.py:75-88
  register_distillation_loss(loss_settings)
    -> DISTILLATION_LOSS_REGISTRY[name] = func
    -> DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
```

`forward_kl_topk` 和 `k1/k2/k3` 等 loss 通过这个 registry 进入 `DistillationLossConfig.__post_init__` 和 `distillation_loss()`。

SGLang 适配也不是 monkey patch，而是在 server 请求层翻译参数：

```text
verl/workers/rollout/sglang_rollout/async_sglang_server.py:460-466
  prompt_logprobs = sampling_params.pop("prompt_logprobs", None)
  if prompt_logprobs is not None:
      return_logprob = True
```

相对真正 monkey patch，这种接入更容易定位，也更容易测试。但它依赖几个隐式接口：`teacher_output.extra_fields["prompt_ids"]`、`teacher_output.extra_fields["prompt_logprobs"]`、engine 的 logits processor 参数约定。这些接口一旦被 vLLM/SGLang 或 agent loop 输出格式改变，MOPD 也会受影响。

### 关键细节与误区澄清

这里有一个容易误解的点：`vllm_async_server.py:828-832` 确实出现了 `apply_vllm_fp8_patches()`，但那是 FP8 quantization 相关，不是 Multi-Teacher OPD 的 patch。当前 MOPD 主路径未发现针对 teacher routing 或 distillation loss 的 monkey patch。

### 本机制小结

💡 小结

- MOPD 当前不是通过 monkey patch 接入，而是通过 manager、agent loop 后处理和 loss registry 接入。
- 这种方式降低了污染全局命名空间的风险，但依赖 `extra_fields` 和 logits processor 这类软接口。
- SGLang 的 `prompt_logprobs` 适配是请求参数翻译，不是全局替换。

## 7.2 通信原语：前向和反向是否对称？

### 设计哲学与核心问题

教师 logprobs 是常量目标，不需要梯度；但 student logits 在 Megatron TP 下是 sharded 的。如果直接在每个 TP rank 上 gather teacher top-k，会遇到全局 vocab id 与本地 shard 不一致的问题。

### 源码实现

Megatron 的 `_VocabParallelKLDivergence` 前向做四类通信：

- `all_reduce MAX`：得到全局 logits max，稳定 softmax；
- `all_reduce SUM`：得到全局 softmax denominator；
- `all_reduce SUM`：聚合每个 token 的 KL loss；
- `all_reduce SUM`：聚合 student/teacher top-k mass 指标。

源码依据：`verl/trainer/distillation/megatron/losses.py:37-55`、`157-172`。

反向没有再对 teacher 做通信，而是使用 forward 保存的 `target_active_mass` 和本地 `source_probs` 构造本地 vocab shard 梯度（`megatron/losses.py:177-218`）。这不是“前向 all-reduce，反向再 all-gather”的对称结构，而是自定义 autograd 把数学梯度写成本地 shard 可计算的形式。

FSDP 路径更直接：student logits 是 full vocab（至少不是 Megatron vocab shard），只需要处理序列 SP 切分和 gather。源码在 `fsdp/losses.py:59-74` 与 `fsdp/transformer_impl.py:1075-1078`。

### 关键细节与误区澄清

**误区：teacher top-k id 在 Megatron 每个 rank 上都能直接 gather。**
不行。`megatron/losses.py:97-108` 先判断 id 是否在本 rank vocab range 内，不在本 shard 的 id 被映射到 dummy index 0 并 mask。测试还专门构造了 dummy index 0 与真实 token index 0 冲突的 case（`tests/utils/test_special_megatron_kl_loss_tp.py:96-115`），并通过 `scatter_add_` 正确处理重复/冲突（`megatron/losses.py:211-215`）。

### 本机制小结

💡 小结

- 教师 RPC 不带梯度，真正的分布式梯度语义只发生在 student actor 内。
- FSDP 主要解决序列维切分对齐；Megatron 还要解决 vocab shard 的全局 id 映射。
- Megatron top-k KL 用自定义 autograd 明确写出了 local shard 梯度，测试覆盖了 loss/grad 与 reference 的一致性。

## 7.3 配置归一化：用户配置如何变成真实行为？

### 设计哲学与核心问题

OPD 配置不只是传给下游库，还会被 verl 自己消费并改写。例如 teacher inference 的 `prompt_length` / `response_length` 会被重写：教师不是生成长 response，而是对学生的 prompt+response 计算 prompt logprobs。

### 源码实现

`validate_and_prepare_for_distillation()` 做了关键变换：

```text
verl/workers/config/distillation.py:160-174
  required_context_len = student_prompt_length + student_response_length + 1
  if max_model_len is not None and required_context_len > max_model_len:
      raise ValueError
  inference.prompt_length = prompt_length + response_length
  inference.response_length = 1
  validate topk logprobs
```

这解释了为什么示例脚本设置：

```bash
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
... teacher.inference.max_model_len=$MAX_NUM_TOKENS
... teacher.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
```

教师服务需要容纳学生 prompt、完整 response，以及额外一个生成 token 的空间。

### 关键细节与误区澄清

**误区：teacher inference 的 `response_length` 表示教师也要生成完整回答。**
不是。配置归一化后，教师 `response_length` 被改成 1（`distillation.py:172-173`）。教师的主要输出来自 `prompt_logprobs`，不是生成一段新回答。

**误区：`teacher_key` 会在数据加载阶段提前校验。**
当前未在源码中确认有全量数据预检。路由错误在 `AgentLoopWorker._compute_teacher_logprobs()` / `_resolve_teacher_key()` 运行到对应样本时才报错。

### 本机制小结

💡 小结

- 配置归一化会主动改写 teacher inference 的 prompt/response 长度。
- top-k 不只是 loss 参数，还影响 vLLM/SGLang 的 logprob 返回能力。
- 当前实现缺少数据源集合的启动前预检，路由错误可能延迟到训练中暴露。

---

# 八、显存、性能与通信分析

## 8.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 学生参数 | ❌ | OPD 不改变 actor 参数分片方式，仍由 FSDP/Megatron 决定。 |
| 教师参数 | ❌/额外增加 | 教师是额外常驻推理模型，占独立 teacher pool GPU。 |
| optimizer state | ❌ | 教师无 optimizer；学生 optimizer state 不因 MOPD 减少。 |
| 学生激活值 | 部分取决于原配置 | remove-padding、SP/CP、gradient checkpointing 仍有效；MOPD 本身不直接减少激活。 |
| 学生 logits | ❌（top-k 模式） | `forward_kl_topk` 需要 student logits；fused kernels 通常不能走 top-k logits processor。 |
| 教师 full logits | ✅ | 教师只返回 sampled-token 或 top-k logprobs，不返回 `[B,S,V]` full logits。 |
| 输入 batch | ❌ | 额外增加 `teacher_logprobs` / `teacher_ids` 字段。 |
| 中间 buffer | ❌ | 当前先 dense pad，再转 nested，存在额外 CPU/Ray/TensorDict 复制。 |
| rollout KV cache | ❌ | 学生 rollout 和教师 rollout 都需要各自 KV cache。 |

真正的显存大头有两个：

1. **教师模型常驻显存**：每个 teacher replica 是独立推理服务，按 `num_replicas * TP * DP * PP` 占卡。
2. **top-k teacher tensor 和 student logits**：top-k 避免了教师 full logits，但 student 侧 top-k forward KL 仍需要 logits；而 `[B, P+R, K]` teacher tensor 会在 CPU/Ray/GPU 迁移中带来峰值。

## 8.2 通信开销

| 阶段 | 通信类型 | group / 范围 | 频率 | 说明 |
|---|---|---|---|---|
| Teacher query | Ray RPC / backend generate | AgentLoopWorker -> 对应 teacher server | 每训练样本 | 当前逐样本请求，按 async 并发。 |
| Teacher load balance | Ray actor 调度 | 每个 teacher 自己的 load balancer | 每请求 | 多教师 key 之间负载隔离。 |
| FSDP SP top-k | gather / unpad | Ulysses SP group | 每 actor forward | `gather_outputs_and_unpad` 聚合 SP 输出。 |
| Megatron top-k | all_reduce MAX/SUM | tensor model parallel group | 每 actor forward | softmax、KL loss、mass 指标都要通信。 |
| actor -> rollout | checkpoint engine / collectives | 学生 actor 与学生 rollout replicas | 每 actor update 后 | teacher 不参与。 |
| checkpoint save/load | 文件系统 | actor/critic/dataloader | save/load 时 | teacher 不保存。 |

新增通信开销主要是 teacher RPC 与 top-k tensor 回传。Megatron top-k loss 额外 all-reduce 是 actor 内部计算代价；它不是多教师特有，但被 top-k distillation 触发。

## 8.3 性能取舍

Multi-Teacher OPD 的性能取舍可以概括为：

- **用独立 GPU 池换教师异构能力。** 教师不抢学生 actor GPU，但额外占用硬件。
- **用静态切分换调度简单性。** 启动时严格校验 `world_size`，运行时不做复杂资源调度；代价是某个 teacher 热点数据多时，其他 teacher 可能空闲。
- **用 top-k 稀疏信号换通信/显存可控。** 不传 full vocab 教师 logits，但 top-k tensor 仍然随 `B*S*K` 增长。
- **用 step 内 async RPC 换代码复用。** 复用 AgentLoop 的 async task 和 load balancer，但 teacher compute 仍在 `generate_sequences()` 返回前完成，慢 teacher 会拖慢 step。
- **用 loss_fn/logits_processor 扩展点换最小侵入。** 不改 trainer 大结构，但 fused kernels、Megatron TP、SP no-padding 都需要额外适配。

## 8.4 本章小结

💡 小结

- MOPD 真正节省的是教师输出分布的体积，而不是学生训练显存。
- 教师模型本身是新增常驻显存；top-k tensor 是新增 batch 内存。
- 通信瓶颈主要来自每样本教师 RPC、Megatron top-k all-reduce 和 actor->rollout 权重同步。
- 当前实现偏向工程简单和复用现有路径，不是极致 overlap 的调度实现。

---

# 九、配置项、边界条件与坑点

## 9.1 设计哲学与核心问题

配置项不应被当成表格背诵，而要看它改变了哪条源码路径。Multi-Teacher OPD 的配置很多，但真正决定行为的只有几类：是否启用、教师资源池、教师模型列表、路由字段、loss mode、teacher inference backend。

## 9.2 配置如何改变源码路径

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `distillation.enabled=True` | `is_distillation_enabled()`，`main_ppo.py:176-185`，`engine_workers.py:573-576` | 创建 teacher pool，actor loss 换成 distillation loss | 没有单独 multi-teacher 开关。 |
| `distillation.n_gpus_per_node / nnodes` | `main_ppo.py:178-184`，`distillation.py:269-275` | 决定 teacher pool 大小 | 必须等于所有 teacher world_size 总和。 |
| `+distillation.teacher_models.<name>.key` | `distillation.py:300-307`，`teacher_manager.py:98-112` | 运行时按该 key 路由 | 必须与样本 `data_source` 等字段完全一致。 |
| `teacher_models.teacher_model` | `distillation.py:277-299` | 单教师占位；多教师时被 pop | 多教师不要用它当真实 teacher。 |
| `num_replicas` | `DistillationTeacherModelConfig.world_size`，`teacher_model.py:79-89` | 决定每个 teacher 启动几个 replica | 当前未拒绝 0，可能延迟失败。 |
| `inference.tensor_model_parallel_size / data_parallel_size / pipeline_model_parallel_size` | `per_replica_world_size` | 决定单 replica 占 GPU 数 | 切分可能跨节点，触发 alignment error。 |
| `inference.name=vllm/sglang` | `get_rollout_replica_class()`，`distillation.py:182-206` | 决定 teacher backend | TRT-LLM teacher 不支持。 |
| `distillation.teacher_key` | `AgentLoopWorker._compute_teacher_logprobs()` | 从 sample kwargs 取路由字段 | 当前未确认启动前全量校验。 |
| `loss_mode=forward_kl_topk` | `loss_settings.use_topk=True` | teacher 请求 top-k，actor logits processor 计算 KL | 需要 student logits；fused kernels 风险高。 |
| `loss_mode=k1/k2/k3/...` | `loss_settings.use_estimator=True` | teacher 请求 sampled-token logprob | `k1` 不能 `use_policy_gradient=False`，源码会报错。 |
| `use_task_rewards=False` | `distillation_ppo_loss()` | 普通 PPO policy loss 置 0，只训练蒸馏 | reward 仍可能计算，但不作为 policy loss。 |
| `max_model_len` | `validate_and_prepare_for_distillation()` | 必须容纳 prompt+response+1 | 长序列/VL 容易配置不足。 |
| `actor_rollout_ref.model.use_fused_kernels` | FSDP/Megatron transformer_impl | fused 路径不调用 top-k logits processor | top-k distillation 示例关闭 fused。 |

## 9.3 静默失效和不兼容组合

- **`teacher_model` 哨兵被 pop**：源码和注释说明了，但用户仍容易踩。
- **`num_replicas=0`**：默认值就是 0；多教师忘配时不一定立刻失败。
- **`teacher_key` 缺失/拼写不一致**：多教师运行到样本才报错。
- **`forward_kl_topk + fused kernels`**：源码未确认有显式校验，top-k logits processor 在非 fused 分支。
- **`teacher temperature != 1.0`**：`_get_teacher_sampling_params()` 直接 `NotImplementedError`（`teacher_manager.py:36-37`）。
- **TRT-LLM teacher**：`TRTLLMReplica` 不支持 `is_teacher_model=True`。
- **FullyAsyncAgentLoopManager + distillation**：`verl/experimental/fully_async_policy/agent_loop/agent_loop.py:168-169` 明确 `NotImplementedError`。

## 9.4 保存 / 加载 / resume 差异

`RayPPOTrainer._save_checkpoint()` 只保存 actor、critic 和 dataloader（`ray_trainer.py:913-934`）；`_load_checkpoint()` 只恢复 actor、critic、dataloader（`ray_trainer.py:988-1017`）。教师模型不保存 runtime state，resume 时由配置和 `model_path` 重新启动。

这对 frozen teacher 是合理的，但有两个运维含义：

- teacher server 的失败恢复不依赖 checkpoint；
- 如果 teacher model path 或 inference config 在 resume 时变化，训练语义会变化，但 checkpoint 本身不会记录 teacher runtime 快照。

## 9.5 本章小结

💡 小结

- MOPD 配置不是简单传参；它会改变资源池、teacher 服务、路由、loss 和 engine 前向路径。
- 最小多教师配置必须包含 `enabled`、teacher pool size、每个 teacher 的 `key/model_path/num_replicas/inference.name`。
- 目前最大的配置坑是 `teacher_model` 哨兵、`num_replicas=0`、`teacher_key` 延迟报错和 top-k/fused kernels 组合。
- teacher 不随 checkpoint 保存，resume 依赖配置重建。

---

# 十、测试、示例与覆盖缺口

## 10.1 设计哲学与核心问题

测试要回答的不是“有哪些测试文件”，而是“它们证明了主路径的哪一段”。Multi-Teacher OPD 的主路径跨配置、Ray 资源、agent loop、teacher server、padding、actor loss、Megatron/FSDP，因此单测很容易只覆盖局部。

## 10.2 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `examples/on_policy_distillation_trainer/run_qwen3_mopd_gsm8k_geo3k.sh` | 多教师 CLI 配置形态 | 覆盖用户推荐配置，但不是自动测试。 |
| `tests/utils/test_special_megatron_kl_loss_tp.py:145-175` | Megatron vocab-parallel top-k KL loss/grad 对齐 FSDP reference | 覆盖核心数学与 TP 梯度，不覆盖 teacher routing。 |
| `tests/test_protocol_v2_on_cpu.py:888-920` | nested `teacher_logprobs` 在 TensorDict chunk/index_select 中保持 ragged layout | 覆盖协议层 nested tensor 操作，不覆盖 teacher manager。 |
| `tests/experimental/agent_loop/test_agent_loop_extra_fields_schema_on_cpu.py:225-263` | AgentLoop extra fields schema | 显式 `distillation_enabled=False`，不覆盖 OPD postprocess。 |
| 单教师示例 `run_qwen_gsm8k*.sh` / `run_qwen3_vl_geo3k.sh` | 单教师 OPD 配置 | 可作为兼容参考，不证明多教师路由。 |

## 10.3 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---:|---|
| `DistillationConfig._resolve_teacher_models` 单/多教师分支 | 未发现专门单测 | `teacher_model` pop、重复 key、资源总和错误不易回归发现。 |
| `num_replicas=0` | 未发现 | 某 teacher 空 server，运行到对应数据源才失败。 |
| 缺失/未知 `teacher_key` | 未发现 | 训练中途 ValueError，定位成本高。 |
| `AsyncTeacherLLMServerManager` 多 key 路由 | 未发现 fake server 单测 | server key 与 teacher key 不一致、单教师忽略 key 等行为无保护。 |
| AgentLoop list output + OPD | code review 指出无覆盖 | `main_ppo_sync.py:379-386` 访问 `list.prompt_ids` 崩溃。 |
| 多机 teacher pool alignment | 未发现 e2e | 资源跨节点报错只在真实集群暴露。 |
| checkpoint resume + teacher 重建 | 未发现 | resume 后 teacher config drift 难以发现。 |
| top-k/fused kernels 不兼容 | 未发现 | 运行期 KeyError 或缺失 distillation outputs。 |
| 性能 / 显存收益 | 未发现自动测试 | top-k tensor 膨胀、teacher RPC 尾延迟无法被 CI 捕获。 |

## 10.4 本章小结

💡 小结

- 现有测试较好覆盖了 Megatron top-k KL 的数学正确性。
- 多教师配置解析、路由、teacher manager、异常配置和 e2e smoke 覆盖不足。
- AgentLoop/TQ 的兼容路径存在已识别风险，尤其是 list output + distillation。
- 性能和显存风险主要依赖人工 profiling，当前未看到自动保护。

---

# 十一、局限性与已知优化点

## 11.1 硬约束

- `distillation.enabled=True` 时 teacher pool 的 `n_gpus_per_node` 和 `nnodes` 必须大于 0（`main_ppo.py:176-181`）。
- 所有 teacher `world_size` 总和必须等于 teacher pool size（`distillation.py:269-275`）。
- 多教师样本必须携带 `teacher_key` 对应字段，且值必须在 configured teacher keys 中（`teacher_manager.py:102-111`）。
- teacher temperature 必须为 1.0（`teacher_manager.py:36-37`）。
- teacher inference backend 目前配置校验只支持 vLLM/SGLang（`distillation.py:182-206`），TRT-LLM teacher 直接不支持（`trtllm_async_server.py:347-349`）。
- `forward_kl_topk` 需要 top-k logprobs 和 student logits，不适合依赖 fused kernels 避免 logits materialization 的路径。
- Fully async agent loop 明确不支持 distillation（`fully_async_policy/agent_loop/agent_loop.py:168-169`）。

## 11.2 维护成本

- **配置经过多层转换。** Hydra YAML -> OmegaConf -> dataclass -> manager dict -> worker kwargs，路由 key 最终才在 agent loop 样本级读取。
- **`teacher_model` 哨兵是兼容成本。** 它让单教师配置简单，但多教师扩展时容易静默丢弃用户配置。
- **teacher output 依赖 `extra_fields` 约定。** `prompt_ids` / `prompt_logprobs` 字段来自 backend adapter，接口变动会影响 MOPD。
- **loss_fn 双重身份增加理解成本。** `distillation_ppo_loss(student_logits=...)` 是 logits processor；无 `student_logits` 时又是 final loss。
- **teacher 生命周期独立。** 简化 checkpoint，但清理、sleep/wake、故障恢复需要额外运维约束。

## 11.3 性能瓶颈

- **逐样本 teacher RPC。** 当前没有按 teacher key 分组批量请求，teacher 热点会造成尾延迟。
- **dense top-k padding。** `_pad_teacher_outputs()` 先构造 `[B,P+R,K]`，再转 nested。源码 TODO 已指出应移除 padding 并使用 tensordict。
- **top-k actor logits materialization。** forward KL top-k 不能完全享受 fused kernels 的 logits-free 路径。
- **Megatron top-k all-reduce。** 每次 top-k forward KL 都需要 TP group 上的 max/sum/loss/mass 通信。
- **静态资源分配。** teacher pool 不随数据分布动态伸缩，不均衡数据源可能导致某些 teacher 忙、某些 teacher 闲。

## 11.4 已知优化点

源码中最直接的 TODO 是去 padding：

- `verl/experimental/teacher_loop/teacher_manager.py:56`：`TODO(wuxibin): remove padding and use tensordict.`
- `verl/experimental/agent_loop/agent_loop.py:762`：同样提示 teacher outputs 应改为更直接的 tensordict/nested 表达。

基于当前实现，还可以看到几个自然优化方向：

1. **按 teacher key 分组批量请求。** AgentLoop worker 已经能拿到每条样本的 routing key，可以把同一 teacher 的样本合并后请求，减少 RPC 数和提升 backend batching。
2. **启动前校验数据源集合。** 读取训练/验证 parquet 的 `teacher_key` 值，提前检查是否是 `teacher_models.keys()` 子集。
3. **更严格配置校验。** `num_replicas > 0`、多教师下禁止使用已配置的 `teacher_model` 哨兵、top-k 与 fused kernels 组合早报错。
4. **teacher lifecycle 管理。** 把 teacher replicas 纳入显式 sleep/cleanup/profile 接口，至少在文档中说明它们不受 checkpoint manager 管控。
5. **调度 overlap。** 当前 teacher query 在 rollout 返回前完成；可以参考旧 OPD scheduler 思路，把 teacher retrieval 与 actor update/下一步 rollout 进一步 overlap，但需要处理 staleness 和 buffer 管理。

## 11.5 本章小结

💡 小结

- 当前 MOPD 的硬约束主要来自静态资源池、严格路由、teacher backend 和 top-k logits 路径。
- 维护成本集中在配置兼容、backend extra_fields、loss_fn 双重身份和 teacher 独立生命周期。
- 性能瓶颈不是单一通信原语，而是逐样本 RPC、dense top-k tensor 和 actor logits materialization 的组合。
- 最值得优先优化的是 teacher 请求批量化、ragged teacher tensor、配置预检和更完整测试。

---

# 小结与展望

verl 的 Multi-Teacher OPD 实现可以用几个关键词概括。

**关键词一：静态资源隔离。**
教师模型不混入 actor worker，而是通过 `teacher_pool + MultiTeacherModelManager` 独立启动。这样做让不同教师模型路径、replica 数和并行度可以共存，也避免教师参与 actor checkpoint 和 optimizer 语义。代价是资源在启动时静态切死，数据源不均衡时容易出现 teacher 利用率不均。

**关键词二：样本级严格路由。**
真正的多教师不是 ensemble，而是 `sample[teacher_key] -> teacher_config.key` 的精确匹配。这个设计清晰、可解释，也很适合“不同数据集对应不同教师”的场景；但它对数据元信息质量要求高，缺少启动前预检时，错误可能在训练中途才暴露。

**关键词三：稀疏教师信号。**
教师返回 sampled-token logprob 或 top-k logprobs，避免传输 full vocab logits。对于大 vocab 模型，这是必要选择。但 top-k tensor 仍按 `B*S*K` 增长，且当前先 dense padding 再 nested 化，长序列和大 batch 下内存压力明显。

**关键词四：loss_fn / logits_processor 双入口。**
`distillation_ppo_loss` 既是 final actor loss，又能在 top-k 模式下作为 logits processor 进入 FSDP/Megatron engine。这让接入很少改 trainer 主体，但也让 top-k/fused kernels、SP/TP/CP 对齐和 loss 聚合语义更复杂。

**关键词五：同步 step 内的 async RPC。**
AgentLoop 内部使用 async task 并发请求教师，但 trainer step 仍要等 `generate_sequences()` 返回。当前实现更像“同步训练步中嵌入异步教师查询”，而不是完全 overlap 的 one-step/two-step OPD scheduler。

这个实现适合：

- 多数据源、多任务混合训练；
- 每个数据源有明确教师模型；
- 可以接受额外 teacher GPU 池；
- 希望最小侵入接入现有 PPO/GRPO 主链路。

它不太适合：

- teacher 计算极慢且必须与 actor update 深度 overlap 的场景；
- 数据源路由不稳定或需要 fallback/ensemble 的场景；
- GPU 资源紧张、无法常驻多个教师模型的场景；
- 强依赖 fused logits-free 训练路径的 top-k distillation。

与“外部 teacher service + 自定义 scheduler”的替代方案相比，当前实现更贴近 verl 的统一 AgentLoop/Ray 生态，代码路径短、可维护性较好；但调度效率和批量化程度还有提升空间。后续值得继续走读的方向包括：AgentLoop 与 TransferQueue 路径的差异、CheckpointEngineManager 的权重同步细节、Megatron top-k KL 在 pipeline/context parallel 下的扩展，以及如何把 teacher query 从逐样本 RPC 优化为按 key 分组的批量服务。
