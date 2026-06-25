# 04 - Engine Backends 子模块架构文档

> verl/workers/engine/ -- 训练引擎后端抽象与六大实现
> 21 个 Python 文件, 6,897 行 (wc -l 验证)

---

## 1. 模块定位与边界

Engine Backends 子模块是 verl 训练侧的核心抽象层。它将"如何在分布式 GPU 上执行一次前向-反向-优化器步骤"封装为统一的 `BaseEngine` 接口, 使上层 Worker 不必关心底层并行策略是 FSDP、Megatron Pipeline 还是 TorchTitan 混合并行。

**上游依赖**: `verl/workers/config/` 提供的各类 dataclass 配置 (FSDPEngineConfig, McoreEngineConfig 等); `verl/utils/fsdp_utils.py`, `verl/utils/megatron_utils/` 提供的底层分布式工具函数。

**下游消费者**: `verl/workers/engine_workers.py` 中的 `TrainingWorker` 通过 `EngineRegistry.new()` 创建引擎实例, 调用 `train_batch()` / `infer_batch()` 进行训练和推理。

**不包含**: Rollout 推理引擎 (vLLM/SGLang) 属于 `verl/workers/rollout/` 子模块; Reward 计算属于 `verl/workers/reward_manager/`。

---

## 2. 架构总览

```
                      EngineRegistry (base.py:268)
                           |
                      BaseEngine (base.py:30)
                           |
          +----------------+----------------+----------------+
          |                |                |                |
     FSDPEngine     MegatronEngine   TorchTitanEngine   AutomodelEngine
     (fsdp/)        (megatron/)      (torchtitan/)       (automodel/)
          |                |                                    |
     +----+----+     +----+----+                          +----+----+
     | WithLM  |     | WithLM  |    (类似继承结构)         | WithLM  |
     | WithVal |     | WithVal |                          +----------+
     +---------+     +---------+
                         |
                    MindSpeed 变体 (mindspeed/)
                    
     VeOmniEngine (veomni/) -- 继承 FSDPEngine, 重写并行初始化
          |
     +----+----+
     | WithLM  |
     | WithVal |
     +---------+
```

注册表查找键由三元组 `(model_type, backend, device)` 决定, 支持按硬件厂商细分。

---

## 3. 核心数据结构

### 3.1 BaseEngine (base.py:30-228)

抽象基类, 定义了训练引擎的完整生命周期接口:

| 方法 | 行号 | 职责 |
|------|------|------|
| `initialize()` | 38 | 构建模型、优化器、学习率调度器 |
| `train_mode()` / `eval_mode()` | 58 / 68 | 返回上下文管理器, 切换训练/评估模式 |
| `forward_backward_batch()` | 99 | 在一个 batch 上执行前向+反向 |
| `train_batch()` | 113 | 封装 zero_grad + forward_backward + optimizer_step |
| `infer_batch()` | 134 | 封装 torch.no_grad + forward_backward(forward_only=True) |
| `optimizer_step()` | 84 | 执行一步优化器更新 |
| `save_checkpoint()` / `load_checkpoint()` | 183 / 203 | 保存/加载检查点 |
| `get_per_tensor_param()` | 151 | 导出逐张量参数用于权重同步到推理引擎 |
| `to()` | 170 | 在 CPU/GPU 之间移动模型和优化器 |

`train_batch()` 的默认实现 (base.py:113-132) 展示了标准训练步骤:

```python
def train_batch(self, data, loss_function):
    self.optimizer_zero_grad()
    outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
    grad_norm = self.optimizer_step()
    outputs["metrics"]["grad_norm"] = grad_norm
    return outputs
```

### 3.2 BaseEngineCtx (base.py:230-266)

引擎上下文管理器基类, 负责 `__enter__` 时将模型/优化器从 CPU 加载到 GPU, `__exit__` 时卸载回 CPU。通过 `_context_switch()` 方法根据 `is_param_offload_enabled` 和 `is_optimizer_offload_enabled` 属性判断是否需要移动。

### 3.3 EngineRegistry (base.py:268-374)

注册表模式的工厂类:

- `register()` (base.py:279): 类装饰器, 接受 `model_type`, `backend`, `device`, `vendor` 四个维度的注册键。支持 list 形式同时注册多个 backend/device 组合。
- `get_engine_cls()` (base.py:327): 查找逻辑优先匹配 `(device, vendor)` 精确键, 回退到 `device` 仅设备键, 再回退到 `(cuda, nvidia)` 默认键。支持 `VERL_ENGINE_DEVICE` 和 `VERL_ENGINE_VENDOR` 环境变量覆盖。
- `new()` (base.py:361): 查找并实例化引擎。

---

## 4. 关键流程

### 4.1 引擎实例化流程

```
TrainingWorker.__init__() (engine_workers.py:127)
  -> EngineRegistry.new(model_type, backend, ...)
    -> get_engine_cls(model_type, backend)
      -> 按 (device, vendor) 查找 _engines[model_type][backend]
    -> engine_cls(model_config, engine_config, optimizer_config, checkpoint_config)
  -> engine.initialize()  -- 由 TrainingWorker.reset() 调用
```

### 4.2 前向-反向批处理流程 (以 FSDP 为例)

```
FSDPEngine.forward_backward_batch() (fsdp/transformer_impl.py:617)
  1. 计算全局 batch_num_tokens, all_reduce 跨 DP 组
  2. prepare_micro_batches() -- 动态 BSZ 或固定 micro_batch 拆分
  3. 遍历 micro_batches:
     a. forward_step() -- 子类实现, 返回 (loss, meta_info)
     b. loss.backward() (若非 forward_only)
  4. postprocess_batch_func() -- 聚合各 micro-batch 的 model_output、loss、metrics
```

### 4.3 Megatron Pipeline 并行流程

MegatronEngine 的 `forward_backward_batch()` (megatron/transformer_impl.py:604) 与 FSDP 有本质区别:

1. 调用 Megatron 的 `get_forward_backward_func()` 获取 pipeline schedule 函数
2. 通过 `make_batch_generator()` 将 micro_batches 包装为 VPP 感知的迭代器
3. pipeline schedule 自动处理跨 PP stage 的 P2P 通信
4. 仅在最后一个 PP stage 收集输出

---

## 5. 六大引擎实现对比

### 5.1 FSDPEngine (fsdp/transformer_impl.py, 1351行)

**注册键**: `(language_model/value_model, fsdp/fsdp2, cuda/npu)`

**核心特性**:
- 支持 FSDP1 和 FSDP2 两种策略 (base.py:378-430), 通过 `engine_config.strategy` 选择
- FSDP2 使用 `fully_shard()` API + `MixedPrecisionPolicy` + 可选 `CPUOffloadPolicy`
- 内置 LoRA 支持: `_build_lora_module()` (第307行) 通过 PEFT 库集成
- 支持 Ulysses 序列并行: `ulysses_sequence_parallel_size > 1` 时创建 `(dp, sp)` 二维 device mesh
- 混合精度: fp16 时自动启用 `ShardedGradScaler` (第363行)
- QAT (量化感知训练): `_apply_qat()` (第491行) 在 FSDP 包裹前对模型应用量化变换
- Liger Kernel 集成: 可选启用 SwigLU 等融合算子

**子类结构**:
- `FSDPEngineWithLMHead` (第924行): 语言模型头, `prepare_model_inputs()` 处理 remove_padding / Ulysses SP 切片, `prepare_model_outputs()` 计算 log_probs、entropy、sum_pi_squared
- `FSDPEngineWithValueHead` (第1300行): 值模型头, `prepare_model_outputs()` 提取 per-token values

### 5.2 MegatronEngine (megatron/transformer_impl.py, 1046行)

**注册键**: `(language_model/value_model, megatron, cuda)`

**核心特性**:
- 通过 `megatron.core.parallel_state` 初始化 TP/PP/CP/EP 并行维度
- 使用 Megatron-Bridge (`AutoBridge`) 将 HF 权重转换为 Megatron 格式
- Pipeline 并行: 使用 Megatron 的 `forward_backward_func` pipeline schedule
- Router Replay (R2/R3): 为 MoE 模型记录和重放路由决策, 消除浮点非确定性 (第119行起)
- 支持 MTP (Multi-Token Prediction) 训练: `patch_engine_mtp()` (第380行)
- VPP (Virtual Pipeline Parallelism): `virtual_pipeline_model_parallel_size` 支持

**权重导出**: `get_per_tensor_param()` (第720行) 通过 `bridge.export_hf_weights()` 将 Megatron 格式参数转回 HF 格式, 供推理引擎使用。

### 5.3 TorchTitanEngine (torchtitan/transformer_impl.py, 735行)

**注册键**: `(language_model, torchtitan, cuda/npu)`

**核心特性**:
- 基于 TorchTitan 的 `Trainer` 类构建, 使用其原生 `ParallelDims` 管理多维并行
- 支持 DP_Shard + DP_Replicate + TP + PP + CP + EP 六维并行
- 使用 TorchTitan 的 `CheckpointManager` 和 `LRSchedulersContainer`
- Context Parallel: 通过 `prepare_context_parallel_input()` 准备 CP 输入
- 权重导出: `sd_adapter.to_hf()` (第501行) 将 TorchTitan 命名空间转回 HF 格式; EP 模式下使用 `all_gather` 收集跨 EP 组的 expert 参数

### 5.4 VeOmniEngine (veomni/transformer_impl.py, 1063行)

**注册键**: `(language_model/value_model, veomni, cuda/npu)`

**核心特性**:
- 继承 `FSDPEngine` 但完全重写初始化流程, 使用 `veomni.distributed.parallel_state` 管理并行
- 序列并行: 通过 `OmniSequenceShardCollator` (第730行) 在序列维度上分片
- Activation Offloading: `build_activation_offloading_context()` 提供独立的前向/反向上下文
- MoE 监控: `MoERouterMonitor` 可定期汇报 expert 负载均衡指标到 wandb (第333行起)
- Router Replay: 使用 `VeOmniRouterReplay` (第206行) 实现 R2/R3 路由重放, 包含严格的状态机管理
- VLM 支持: `_apply_veomni_input_transforms()` 处理图像/视频 mask 和 SP 切片

### 5.5 AutomodelEngine (automodel/transformer_impl.py, 713行)

**注册键**: `(language_model, automodel, cuda)`

**核心特性**:
- 基于 NVIDIA NeMo Automodel 基础设施
- 使用 `nemo_automodel` 的 `build_optimizer`, `OptimizerParamScheduler`, `Checkpointer`
- MoE 支持: `prepare_for_grad_accumulation()` 和 `prepare_for_final_backward()` 处理 MoE aux loss 缩放
- TE (Transformer Engine) 注意力后端: `attn_implementation == "te"` 时传入 `cu_seqlens` (第527行)
- 梯度裁剪: 通过 `scale_grads_and_clip_grad_norm()` 统一处理, 支持 EP/PP 维度归约

### 5.6 MindSpeed 变体 (mindspeed/transformer_impl.py, 166行)

**注册键**: `(language_model/value_model, megatron, npu)` 和 `(language_model, mindspeed_megatron, npu)`

**核心特性**:
- 继承 `MegatronEngineWithLMHead` / `MegatronEngineWithValueHead`
- 为华为 Ascend NPU 提供 MindSpeed 适配: `_mindspeed_repatch()` (第45行) 在 `initialize_model_parallel` 前重新应用猴子补丁
- FP8 支持: `reset_fp8_reuse_quantized_weight()` 在设备切换时重置 FP8 量化权重
- `MindSpeedMegatronEngineWithLMHead` 使用自定义 `gpt_model_provider` 构建模型

---

## 6. 公共工具函数 (utils.py, 159行)

### 6.1 prepare_micro_batches() (utils.py:57)

将一个 batch 拆分为 micro-batch 列表:
- **动态 BSZ 模式** (`use_dynamic_bsz=True`): 调用 `rearrange_micro_batches()` 按 `max_token_len_per_gpu` 自动打包, 跨 DP 组保持相同 micro-batch 数量
- **固定 BSZ 模式**: 按 `micro_batch_size_per_gpu` 均匀切分

### 6.2 postprocess_batch_func() (utils.py:98)

聚合多个 micro-batch 的输出:
1. 将各 micro-batch 的 `model_output` (nested tensor) 拼接
2. 若使用动态 BSZ, 通过 `restore_dynamic_batch()` 恢复原始 batch 顺序
3. 收集 loss 列表和 metrics 字典

### 6.3 enable_full_determinism() (utils.py:31)

设置完全确定性模式: `CUBLAS_WORKSPACE_CONFIG`, `FLASH_ATTENTION_DETERMINISTIC`, NPU 的 `HCCL_DETERMINISTIC` 等。

---

## 7. 设计决策

### 7.1 为何使用注册表而非工厂方法?

EngineRegistry 采用三维键 `(model_type, backend, device)` + 可选 `vendor`, 使得:
- 新增引擎只需一个 `@EngineRegistry.register(...)` 装饰器, 零侵入式扩展
- 同一 backend 可为不同硬件 (CUDA vs NPU) 提供不同实现 (如 MindSpeed)
- 环境变量覆盖机制 (`VERL_ENGINE_DEVICE`) 支持测试和跨平台调试

### 7.2 为何分离 Engine 和 EngineCtx?

Engine 本身只描述能力 (前向、反向、保存); EngineCtx 管理运行时上下文 (offload/reload, train/eval 模式切换)。这种分离使得:
- `train_mode()` / `eval_mode()` 返回上下文管理器, 自动处理 offload 逻辑
- 不同引擎可以定制上下文行为 (如 FSDPEngine 需要 `reshard()`, VeOmniEngine 需要设置 SP group)

### 7.3 为何 FSDPEngineWithLMHead 和 WithValueHead 分开?

唯一区别在于 `prepare_model_outputs()`: LMHead 提取 log_probs + entropy, ValueHead 提取 per-token values。共享 `prepare_model_inputs()` 和整个前向/反向基础设施。这比在一个类中用 if 分支更清晰。

---

## 8. 关键接口与扩展点

### 8.1 新增引擎后端

```python
@EngineRegistry.register(
    model_type="language_model",
    backend="my_backend",
    device="cuda"
)
class MyEngineWithLMHead(BaseEngine):
    def initialize(self): ...
    def forward_backward_batch(self, data, loss_function, forward_only=False): ...
    def optimizer_step(self): ...
    # ... 实现其余 BaseEngine 接口
```

### 8.2 Engine 与 Worker 的接口契约

Worker 通过以下接口与 Engine 交互:

| Worker 调用 | Engine 方法 | 数据格式 |
|------------|------------|---------|
| `train_batch()` | `engine.train_batch(data, loss_fn)` | data: TensorDict (no-padding NestedTensor); 返回 dict(model_output, loss, metrics) |
| `infer_batch()` | `engine.infer_batch(data, loss_fn)` | 同上, forward_only=True |
| `update_weights()` | `engine.get_per_tensor_param()` | 返回 `(Generator[(name, Tensor)], peft_config)` |
| offload 控制 | `engine.to(device)` | device: "cpu" 或 "cuda" |
| 模式切换 | `engine.train_mode()` / `engine.eval_mode()` | 返回 ContextManager |

### 8.3 loss_function 签名

所有引擎的 `forward_backward_batch()` 接受一个 `loss_function` 回调, 签名为:

```python
def loss_function(model_output: dict, data: TensorDict, dp_group=None) -> tuple[Tensor, dict]:
    """
    Args:
        model_output: {"log_probs": nested_tensor, "entropy": ..., "values": ...}
        data: 当前 micro-batch 的 TensorDict
        dp_group: 数据并行进程组
    Returns:
        (loss, metrics_dict)
    """
```
