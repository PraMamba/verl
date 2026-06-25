# 05 - Engine Workers 子模块架构文档

> verl/workers/engine_workers.py (758行) + verl/workers/utils/ (430行)
> 训练 Worker 层: 连接 Engine 引擎与 Ray 单控制器调度

---

## 1. 模块定位与边界

Engine Workers 是 verl 训练架构中"Worker 层"的核心实现。它位于 Engine (训练引擎) 与 Ray 单控制器之间, 将引擎的底层训练能力封装为可被 Ray 远程调用的 Worker 方法, 并负责数据格式转换、损失函数组装、性能指标收集。

**上游依赖**: `verl/workers/engine/` (BaseEngine, EngineRegistry); `verl/single_controller/` (Worker, Dispatch, register 装饰器)

**下游消费者**: `verl/trainer/ppo/ray_trainer.py` 中的 `RayPPOTrainer` 通过 `RayWorkerGroup` 调用 Worker 方法 (如 `update_actor()`, `compute_log_prob()`)

**不包含**: 引擎内部的前向/反向实现 (属 04-engine-backends); Rollout 推理引擎 (属 06-rollout-engines)

---

## 2. 架构总览

```
RayPPOTrainer (单控制器)
      |
      | Ray RPC (DataProto / TensorDict)
      |
ActorRolloutRefWorker (engine_workers.py:434)
      |
      +-- actor: TrainingWorker  (engine_workers.py:76)
      |       |
      |       +-- engine: BaseEngine (由 EngineRegistry 创建)
      |       +-- loss_fn: ppo_loss / distillation_ppo_loss
      |
      +-- ref: TrainingWorker    (同上, forward_only=True)
      |       |
      |       +-- engine: BaseEngine (forward_only, 无优化器)
      |
      +-- rollout: BaseRollout   (vLLM/SGLang ServerAdapter)
      |
      +-- checkpoint_engine: CheckpointEngine (权重同步通道)
```

数据流向:
```
TensorDict(left-right padding)
  -> left_right_2_no_padding() -> TensorDict(NestedTensor, no-padding)
    -> Engine.forward_backward_batch()
      -> model_output(NestedTensor: log_probs, entropy, values)
        -> no_padding_2_padding() -> Tensor(bsz, max_response_len)
          -> ppo_loss() / value_loss() 计算
```

---

## 3. 核心数据结构

### 3.1 TrainingWorker (engine_workers.py:76-431)

通用训练 Worker, 可独立部署也可作为 ActorRolloutRefWorker 的组件。

**构造参数**: `TrainingWorkerConfig` dataclass, 包含:
- `model_type`: "language_model" 或 "value_model"
- `model_config`: HFModelConfig
- `engine_config`: 后端特定配置 (FSDPEngineConfig / McoreEngineConfig 等)
- `optimizer_config`: 优化器配置
- `checkpoint_config`: 检查点配置
- `auto_select_engine_optim_fn`: 可选的自动后端选择函数

**关键属性**:
- `self.engine`: BaseEngine 实例, 通过 `EngineRegistry.new()` 创建 (第127行)
- `self.loss_fn`: 可替换的损失函数, 通过 `set_loss_fn()` 注入
- `self.flops_counter`: MFU 计算器

**注册的 dispatch 方法**:

| 方法 | dispatch 模式 | 行号 | 职责 |
|------|--------------|------|------|
| `reset()` | ONE_TO_ALL | 160 | 调用 `engine.initialize()` |
| `set_loss_fn()` | ONE_TO_ALL | 161 | 注入损失函数 |
| `train_batch()` | nd_compute | 323 | 单个 mini-batch 训练步 |
| `train_mini_batch()` | nd_compute | 233 | 多 epoch 多 mini-batch 训练 |
| `infer_batch()` | nd_compute | 379 | 推理计算 (compute_log_prob / compute_ref_log_prob) |
| `save_checkpoint()` | ONE_TO_ALL | 425 | 保存检查点 |
| `load_checkpoint()` | ONE_TO_ALL | 429 | 加载检查点 |

### 3.2 ActorRolloutRefWorker (engine_workers.py:434-758)

核心融合 Worker, 在同一进程中集成 Actor 训练、Rollout 推理和 Reference 策略三种角色。

**角色组合** (第452行):
- `"actor"`: 仅 Actor 训练
- `"rollout"`: 仅 Rollout 推理
- `"ref"`: 仅 Reference 策略
- `"actor_rollout"`: Actor + Rollout 融合
- `"actor_rollout_ref"`: 完整三合一融合

**init_model() 初始化流程** (第500行):

```
init_model()
  1. 构建 Reference 模型 (若 role 含 "ref"):
     - 深拷贝 model_config, 关闭 MTP
     - 创建 TrainingWorkerConfig (forward_only=True)
     - 实例化 TrainingWorker -> self.ref
     - 注册 dispatch_collect 信息 (mesh_name="ref")

  2. 构建 Actor 模型 (若 role 含 "actor"):
     - 解析 ActorConfig, 配置 dynamic_bsz / max_token_len 等
     - 构造 ppo_loss (或 distillation_ppo_loss) 作为 loss_fn
     - 实例化 TrainingWorker -> self.actor
     - 注册 dispatch_collect 信息 (mesh_name="actor")

  3. 构建 Rollout 引擎 (若 role 含 "rollout"):
     - 计算 rollout 的 device_mesh: (dp, infer_tp, infer_pp) 三维
     - 通过 get_rollout_class() 获取 ServerAdapter 类
     - 实例化 self.rollout

  4. 构建 CheckpointEngine (若 role 含 "actor"):
     - 通过 CheckpointEngineRegistry 创建权重同步通道
```

**关键方法**:

| 方法 | 行号 | 职责 |
|------|------|------|
| `compute_ref_log_prob()` | 634 | 调用 `self.ref.infer_batch()`, mesh="ref" |
| `compute_log_prob()` | 641 | 调用 `self.actor.infer_batch()`, mesh="actor" |
| `update_actor()` | 649 | 调用 `self.actor.train_mini_batch()`, mesh="actor" |
| `update_weights()` | 667 | 从 actor 引擎导出权重, 同步到 rollout 推理引擎 |
| `save_checkpoint()` | 661 | 委托 actor 保存 |
| `load_checkpoint()` | 657 | 委托 actor 加载 |

### 3.3 Router Replay 装饰器 (engine_workers.py:61-73)

`_with_routing_replay_flag()` 装饰器在 `data` TensorDict 上设置 `enable_routing_replay` 标志:
- `compute_log_prob()` 和 `update_actor()` 设置 `enabled=True`
- `compute_ref_log_prob()` 设置 `enabled=False`

---

## 4. 关键流程

### 4.1 update_weights() 权重同步流程 (engine_workers.py:667-746)

这是混合引擎架构中最关键的流程, 负责将训练后的模型权重从 Actor Engine 同步到 Rollout 推理引擎:

```
update_weights(global_steps, mode="auto")
  0. 若 mode != "naive" (异步离散化部署):
     -> engine.get_per_tensor_param()
     -> checkpoint_engine.send_weights(params)  -- 异步传输
     -> 返回

  (以下为 mode="naive" 同步协同部署)
  1. resume rollout 权重内存 (若 free_cache_engine 已释放)
  2. actor.engine.get_per_tensor_param() -- 导出逐张量参数
  3. LoRA 处理:
     - merge 模式: 权重已合并, peft_config=None
     - adapter 模式: 首次需要 base + adapter 两轮同步
  4. rollout.update_weights(per_tensor_param) -- 写入推理引擎
  5. 若 param_offload, 将 actor 模型卸载到 CPU
  6. resume rollout 的 kv_cache
```

### 4.2 train_mini_batch() 训练流程 (engine_workers.py:233-321)

多 epoch 多 mini-batch 训练循环:

```
train_mini_batch(data)
  1. 计算 mini_batch_size_per_gpu (从全局或每 GPU 维度)
  2. tu.make_iterator() 构建 DataLoader 迭代器
  3. 进入 engine.train_mode() 上下文:
     for batch_idx, mini_batch in enumerate(dataloader):
       a. all_gather global_token_num 用于 MFU 计算
       b. 调用 self.train_batch(mini_batch)
       c. 收集输出 metrics
  4. 聚合所有 mini-batch 的 metrics (仅在 mp_src_rank)
```

### 4.3 _postprocess_output() 指标后处理 (engine_workers.py:172-231)

将引擎输出转换为最终指标:
1. `loss`: `torch.sum()` + `all_reduce(AVG)` 跨 DP 组平均
2. `grad_norm`: 不做 all_reduce (已在 clip_grad 时归约)
3. 其他 metrics: `allgather_dict_into_dict()` 跨 DP 组收集
4. 添加 GPU 内存使用、MFU 等性能指标

---

## 5. 工具模块: verl/workers/utils/ (430行)

### 5.1 losses.py (186行)

三个损失函数实现:

**sft_loss()** (第28行): SFT (监督微调) 损失
- 将 log_prob 和 loss_mask 的 nested tensor 展平
- loss_mask 左移一个 token 对齐
- 按全局 `batch_num_tokens` 归一化, 乘以 `dp_size` 补偿梯度平均

**ppo_loss()** (第57行): PPO/GRPO 策略损失, 核心训练损失函数
- 调用 `no_padding_2_padding()` 将 nested tensor 转为 padded tensor (第59行)
- 从 `core_algos.get_policy_loss_fn()` 获取策略损失实现 (vanilla/clip 等)
- 可选组件: entropy loss (entropy_coeff 加权), KL loss (kl_loss_coef 加权)
- 支持 `loss_agg_mode` 控制归一化方式
- 通过 `Metric` 类追踪 pg_clipfrac、ppo_kl 等指标

**value_loss()** (第147行): Critic 值损失
- 调用 `compute_value_loss()` 计算带 clip 的值函数损失
- 追踪 vf_loss、vf_clipfrac、vpred_mean 指标

### 5.2 padding.py (231行)

数据格式转换工具:

**left_right_2_no_padding()** (第23行): 最关键的格式转换函数
- **输入**: 标准的 left-right padding 格式 TensorDict, 包含 `input_ids (bs, seq_len)`, `attention_mask`, `response_mask`, `position_ids`
- **处理**: 使用 `unpad_input()` + `index_first_axis()` 移除 padding token, 构建 jagged NestedTensor
- **输出**: `input_ids` 和 `position_ids` 变为 nested tensor `(bs, j1)`, 新增 `loss_mask` (等于 `response_mask`), 保存 `indices` 和 `max_seq_len` / `max_response_len` 到 NonTensorData
- **附加处理**: `routed_experts` (若存在) 也转为 nested tensor (第73行); `teacher_logprobs` / `teacher_ids` 同样处理 (第84行)

**no_padding_2_padding()** (第99行): 反向转换, 从 unpad 模型输出中提取 response
- **输入**: nested tensor 或 flat tensor `(total_nnz, *)`, 以及包含 `prompts`/`responses` 的 TensorDict
- **处理**: 按 `prompt_lens + response_lens` 的 cumulative offset 切分, 左移一个 token (log_probs/values 的对齐), 右侧 zero-pad 到 max_response_len
- **输出**: padded tensor `(bsz, max_response_len, *)`

**build_attention_mask_from_nested()** (第146行): 从 nested input_ids 构建 padded attention mask

**response_from_nested()** (第196行): 从 nested tensor 中提取 response 部分, 返回 nested tensor

**response_to_nested()** (第215行): 将 padded response tensor 转回 nested tensor

---

## 6. 数据格式转换全景

verl 中存在两种核心数据格式, Engine Workers 层负责它们之间的转换:

```
[Trainer 层 / DataProto]        [Engine 层]
   padded tensor                  NestedTensor (no-padding)
   (bs, max_seq_len)              (bs, jagged)

        |                              ^
        | left_right_2_no_padding()    |
        v                              |
   NestedTensor input     ->     Engine forward
                                       |
                                  model_output
                                  (NestedTensor)
                                       |
                                  no_padding_2_padding()
                                       |
                                       v
                                  padded response
                                  (bs, max_resp_len)
                                       |
                                  ppo_loss() / value_loss()
```

这种双格式设计的原因: Engine 层使用 no-padding NestedTensor 以获得更高的计算效率 (无需计算 padding token); 而上层的 RL 算法 (core_algos) 使用 padded tensor 以简化 mask 操作。

---

## 7. 设计决策

### 7.1 为何 ActorRolloutRefWorker 融合三种角色?

融合 Worker 是混合引擎 (Hybrid Engine) 的核心: Actor 训练完成后, 通过 `get_per_tensor_param()` 直接导出权重到同进程的 Rollout 引擎, 避免跨进程/跨节点传输。三种角色共享同一 GPU, 通过 sleep/wake_up 机制分时复用显存。

### 7.2 为何 loss_fn 通过 set_loss_fn() 注入而非构造时传入?

两个原因:
1. 损失函数需要 `partial(ppo_loss, config=actor_config)` 绑定配置, 但配置在 `init_model()` 时才完全确定
2. 支持动态替换: SFT 和 RL 阶段使用不同损失函数, 或 distillation 模式替换为 `distillation_ppo_loss`

### 7.3 为何 update_weights() 是 async?

`update_weights()` 需要 await rollout 的 `resume()` 和 `update_weights()` 方法, 这些方法涉及跨进程的 Ray 远程调用 (推理引擎运行在独立进程中)。async 使调用方可以在等待期间调度其他工作。

### 7.4 为何 no_padding_2_padding 左移一个 token?

log_probs 和 values 的语义是"预测下一个 token 的概率/价值"。模型输出的第 i 个位置对应预测第 i+1 个 token 的概率。`response_mask` 标记的是 response token 本身的位置。因此需要将模型输出左移一位, 使 log_probs[i] 对应 response_token[i] 的概率。具体实现在 padding.py 第140行: `values[seq_offset - resp_len - 1 : seq_offset - 1]`。

---

## 8. 关键接口与扩展点

### 8.1 自定义损失函数

```python
def my_custom_loss(config, model_output, data, dp_group=None):
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    # ... 自定义损失计算
    return loss, metrics

# 注入
worker.set_loss_fn(partial(my_custom_loss, config=my_config))
```

损失函数签名要求:
- `model_output`: dict, 包含 `log_probs` (NestedTensor), 可选 `entropy`, `values`
- `data`: TensorDict, 包含 `response_mask`, `old_log_probs`, `advantages` 等
- `dp_group`: 数据并行进程组, 用于跨 rank 归约
- 返回 `(loss_tensor, metrics_dict)`

### 8.2 TrainingWorkerConfig 构建

```python
config = TrainingWorkerConfig(
    model_type="language_model",
    model_config=hf_model_config,
    engine_config=fsdp_engine_config,   # 或 None + auto_select_engine_optim_fn
    optimizer_config=optimizer_config,
    checkpoint_config=checkpoint_config,
)
worker = TrainingWorker(config)
worker.reset()  # 调用 engine.initialize()
```

### 8.3 Dispatch 模式说明

- `Dispatch.ONE_TO_ALL`: 所有 rank 执行相同操作 (如 `reset`, `set_loss_fn`)
- `make_nd_compute_dataproto_dispatch_fn(mesh_name="actor")`: 按 mesh 中的 DP rank 分发数据, 仅在 `is_collect=True` 的 rank 收集输出 (通常是 TP rank 0 + PP last stage)
- `Dispatch.DP_COMPUTE`: 按 DP rank 分发, 用于 `execute_checkpoint_engine`
