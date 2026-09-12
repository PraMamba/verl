# verl Worker 配置子模块架构文档

> 源码路径: `verl/workers/config/`  
> 总计: 12 个源文件, 2,987 行代码
> 最后更新: 2026-08-02 (基准源码: 上游 `e3573545`; 通过 `wc -l` 确认行数, `grep -n` 确认行号)

---

## 1. 概述

`verl/workers/config/` 是 verl 训练系统的 **配置中枢**。它以 `BaseConfig` (冻结 dataclass) 为基类, 定义了从模型加载、引擎选择、优化器参数到推理引擎、奖励模型、蒸馏策略的全部配置结构。

该子模块解决的核心问题:

1. **后端统一**: 通过 `EngineConfig` 继承体系, 将 FSDP/Megatron/TorchTitan/VeOmni/Automodel/MindSpeed 六种训练后端的配置归纳到统一接口
2. **分层组合**: `ActorConfig` / `CriticConfig` 是面向用户的高层配置, 内部组合了 `EngineConfig` + `OptimizerConfig` + `HFModelConfig` + `CheckpointConfig`
3. **冻结安全**: 通过 `BaseConfig` 的冻结机制和 `_mutable_fields` 白名单, 防止运行时意外修改关键配置
4. **Hydra 集成**: 所有配置类可通过 `omega_conf_to_dataclass()` 从 Hydra YAML 自动转换, 提供类型安全的配置校验

配置继承体系:

```
BaseConfig (冻结 dataclass, 类字典接口)
  ├── EngineConfig (训练引擎基类)
  │   ├── FSDPEngineConfig
  │   ├── McoreEngineConfig
  │   │   └── MindSpeedEngineConfig
  │   ├── VeOmniEngineConfig
  │   ├── TorchtitanEngineConfig
  │   └── AutomodelEngineConfig
  ├── ActorConfig -> FSDPActorConfig / McoreActorConfig / VeOmniActorConfig / ...
  ├── CriticConfig -> FSDPCriticConfig / McoreCriticConfig / VeOmniCriticConfig / ...
  ├── OptimizerConfig -> FSDPOptimizerConfig / McoreOptimizerConfig / ...
  ├── HFModelConfig (模型加载)
  ├── RolloutConfig (推理引擎)
  ├── RewardConfig (奖励函数)
  └── DistillationConfig (在线蒸馏)
```

---

## 2. 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `__init__.py` | 38 | 汇聚导出所有配置类 |
| `actor.py` | 420 | Actor 配置: `ActorConfig` 及 5 种后端变体 |
| `checkpoint.py` | 60 | 后端特化 checkpoint 配置: `McoreCheckpointConfig`, `MindSpeedCheckpointConfig` |
| `critic.py` | 340 | Critic 配置: `CriticConfig` 及 5 种后端变体 + `FSDPCriticModelCfg` |
| `disaggregation.py` | 60 | Prefill-Decode (PD) 分离配置 (sglang/vllm): `DisaggregationConfig` |
| `distillation.py` | 320 | 在线蒸馏配置: `DistillationConfig`, 多教师模型支持 |
| `engine.py` | 650 | 训练引擎配置: `EngineConfig` 及 6 种后端变体, `QATEngineConfig`, `TrainingWorkerConfig` |
| `megatron_peft.py` | 40 | Megatron LoRA/PEFT 工厂函数 |
| `model.py` | 261 | HuggingFace 模型配置: `HFModelConfig`, `MtpConfig` |
| `optimizer.py` | 351 | 优化器配置: `OptimizerConfig` 及 5 种后端变体, `build_optimizer()` |
| `reward.py` | 105 | 奖励配置: `RewardConfig`, `RewardModelConfig`, `SandboxFusionConfig` |
| `rollout.py` | 342 | 推理引擎配置: `RolloutConfig` 及 9 个子配置 |

---

## 3. 核心数据结构

> 字段覆盖口径：下列代码块是架构导读中的代表性字段，不是逐字段复制。完整字段以对应 dataclass 源码和本节文件清单为准；为避免把示例误读为 API 全集，特别列出紧凑示例中容易遗漏的字段：`PolicyLossConfig` 还包括 `kl_cov_ratio`、`ppo_kl_coef`、`rollout_correction`（`actor.py:98-100`）；`OptimizerConfig` 还包括 `lr_warmup_steps`（`optimizer.py:51`）；`ActorConfig` 还包括 `ppo_infer_micro_batch_size_per_gpu`、`ppo_infer_max_token_len_per_gpu`、`freeze_vision_tower`、`calculate_entropy`、`calculate_sum_pi_squared`、`use_prefix_grouper`、`profiler`、`global_batch_info`、`qat` 等（`actor.py:154-194`）。

### 3.1 BaseConfig --- 冻结 dataclass 基类 (base_config.py 第 22 行)

所有配置类的根基类, 继承自 `collections.abc.Mapping`:

```python
@dataclass
class BaseConfig(collections.abc.Mapping):
    _mutable_fields = set()    # 白名单: 可修改的字段
    _target_: str = ""         # Hydra _target_ 兼容字段
```

核心机制:
- `__setattr__()` 拦截赋值: 如果字段已存在且不在 `_mutable_fields` 中, 抛出 `FrozenInstanceError`
- 实现 `__getitem__`, `__iter__`, `__len__`, `get()`, 使配置对象可像字典一样使用
- 子类通过 `_mutable_fields = BaseConfig._mutable_fields | {"field_name"}` 扩展可变字段集合

### 3.2 EngineConfig --- 训练引擎基类 (engine.py 第 77 行)

统一所有训练后端的公共字段:

```python
@dataclass
class EngineConfig(BaseConfig):
    param_offload: bool = False           # 参数 CPU offload
    optimizer_offload: bool = False       # 优化器 CPU offload
    grad_offload: bool = False            # 梯度 CPU offload
    forward_only: bool = False            # 仅前向 (如 ref policy)
    strategy: str = None                  # 后端标识符
    dtype: str = "bfloat16"               # 模型精度
    use_dynamic_bsz: bool = True          # 动态 batch size
    max_token_len_per_gpu: int = None     # 训练时每 GPU 最大 token 长度
    micro_batch_size_per_gpu: int = None  # 训练时每 GPU micro batch size
    infer_max_token_len_per_gpu: int = None  # 推理时每 GPU 最大 token 长度
    infer_micro_batch_size_per_gpu: int = None
    use_fused_kernels: bool = False       # 融合核 (FlashAttention 等)
    use_remove_padding: bool = True       # 去除 padding 优化
    seed: int = 42
    full_determinism: bool = False        # 全确定性模式 (慢但可复现)
    router_replay: EngineRouterReplayConfig  # MoE 路由回放
```

`_mutable_fields` 包含 9 个运行时可修改字段 (第 78-88 行): `use_dynamic_bsz`, `max_token_len_per_gpu`, `micro_batch_size_per_gpu`, `infer_max_token_len_per_gpu`, `infer_micro_batch_size_per_gpu`, `use_fused_kernels`, `use_remove_padding`, `forward_only`, `param_offload`。

### 3.3 六种 EngineConfig 变体

> 基准 `e3573545` 新增共享字段 `entropy_from_logits_chunk_size` (默认 2048), 现见于全部 5 个含分块熵计算的变体 (Mcore 第 194 行 / FSDP 第 269 行 / VeOmni 第 364 行 / Torchtitan 第 460 行 / Automodel 第 601 行)。此外 engine.py 还定义了 `QATEngineConfig` (第 129 行, 量化感知训练配置), 通过 `qat` 字段嵌入 `McoreEngineConfig` (第 211 行) 与 `FSDPEngineConfig` (第 273 行)。

#### FSDPEngineConfig (engine.py 第 231 行)

```python
@dataclass
class FSDPEngineConfig(EngineConfig):
    wrap_policy: dict = field(default_factory=dict)
    offload_policy: bool = False
    reshard_after_forward: bool = True      # 前向后重分片
    fsdp_size: int = -1                     # FSDP 组大小, -1 表示全部 GPU
    model_dtype: str = "fp32"               # 模型初始化精度
    use_orig_params: bool = False           # FSDP1 使用原始参数
    ulysses_sequence_parallel_size: int = 1 # Ulysses 序列并行
    strategy: str = "fsdp"                  # 支持 "fsdp" 和 "fsdp2"
```

`__post_init__` 校验: `strategy` 必须为 `"fsdp"` 或 `"fsdp2"` (第 277 行, strategy 断言)。

#### McoreEngineConfig (engine.py 第 150 行)

Megatron-Core 配置, 含丰富的并行参数:

```python
tensor_model_parallel_size: int = 1
expert_model_parallel_size: int = 1
expert_tensor_parallel_size: Optional[int] = None
pipeline_model_parallel_size: int = 1
virtual_pipeline_model_parallel_size: Optional[int] = None
context_parallel_size: int = 1
dynamic_context_parallel: bool = False
sequence_parallel: bool = True
use_distributed_optimizer: bool = True
use_mbridge: bool = True
use_megatron_fsdp: bool = False
```

`__post_init__` 自动修正: 当 `tensor_model_parallel_size == 1` 时, 强制关闭 `sequence_parallel` (第 225-227 行, Mcore TP=1 关闭 SP)。

行为变更 (基准 `e3573545`): `vanilla_mbridge` 默认值由 `True` 改为 `False` (第 208 行); 显式设为 `True` 时 `__post_init__` 发出 `FutureWarning` (第 218-224 行), 提示 legacy mbridge 后端已弃用, 应改用 Megatron-Bridge。另新增 `pad_bshd_to_minibatch_max` (第 198 行, 默认 `True`)。

#### VeOmniEngineConfig (engine.py 第 281 行)

VeOmni 引擎配置, 特色在于丰富的算子实现选择:

```python
attn_implementation: str = "flash_attention_2"
moe_implementation: str = "fused"
cross_entropy_loss_implementation: str = "eager"
rms_norm_implementation: str = "eager"           # "triton" 用于 DeepSeek-V3 对齐
swiglu_mlp_implementation: str = "eager"
rotary_pos_emb_implementation: str = "eager"     # "triton" 用于 bitwise 对齐
```

`__post_init__` 自动替换 attention 实现名 (第 398 行, attn 实现替换): 如 `"flash_attention_2"` -> `"veomni_flash_attention_2_with_sp"`。

字段变更 (基准 `e3573545`): 删除 `wrap_policy` / `offload_policy` / `reshard_after_forward` / `use_orig_params` / `moe_load_balance_monitor_interval`; 新增 Qwen3.5 GatedDeltaNet 算子选择 `rms_norm_gated_implementation` / `causal_conv1d_implementation` / `chunk_gated_delta_rule_implementation` (第 391-393 行, 默认 `"eager"`)。

#### TorchtitanEngineConfig (engine.py 第 414 行)

支持多维并行:

```python
data_parallel_size: int = 1
data_parallel_replicate_size: int = 1    # HSDP 复制度
data_parallel_shard_size: int = 1        # HSDP 分片度
tensor_parallel_size: int = 1
expert_parallel_size: int = 1
pipeline_parallel_size: int = 1
context_parallel_size: int = 1
    attn_type: str = "flex"                  # flex/flex_flash/varlen
```

Torchtitan 字段变更 (基准 `e3573545`): 新增 `spmd_backend` (默认 `"spmd_types"`) 与 `activation_checkpoint` (默认 `"selective"`); `__post_init__` (第 480-489 行) 对 `attn_type` / `spmd_backend` / `activation_checkpoint` 各加一条合法值断言。

#### AutomodelEngineConfig (engine.py 第 493 行)

NeMo Automodel 后端, 支持 FSDP2/MegatronFSDP/DDP 三种分布式策略:

```python
distributed_strategy: str = "fsdp2"     # "fsdp2" / "megatron_fsdp" / "ddp"
backend_config: dict = field(...)        # 传给 BackendConfig 的参数
moe_config: dict = field(...)           # MoE 并行化参数
mp_param_dtype: str = "bf16"            # FSDP2 混合精度策略
```

`__post_init__` 校验: `pp_size` 必须为 1 (第 611 行, pp_size 断言)。

#### MindSpeedEngineConfig (engine.py 第 615 行)

继承 `McoreEngineConfig`, 添加 MindSpeed 专用参数:

```python
class MindSpeedEngineConfig(McoreEngineConfig):
    strategy: str = "mindspeed_megatron"    # "mindspeed_megatron" 或 "mindspeed_fsdp"
    mcore_kwargs: dict = field(...)          # mindspeed_megatron 引擎参数
    fsdp_kwargs: dict = field(...)           # mindspeed_fsdp 引擎参数
```

注意: `__post_init__` 不调用 `super().__post_init__()` (第 629 行), 因为 strategy 值不同于父类的 `"megatron"` 断言。

### 3.4 TrainingWorkerConfig (engine.py 第 639 行)

组合配置: 将模型、引擎、优化器、检查点、Profiler 合为一体:

```python
@dataclass
class TrainingWorkerConfig(BaseConfig):
    model_type: str = None
    model_config: HFModelConfig = None
    engine_config: EngineConfig = None
    optimizer_config: OptimizerConfig = None
    checkpoint_config: CheckpointConfig = None
    profiler_config: ProfilerConfig = None
    auto_select_engine_optim_fn: Callable = None  # 自动选择引擎和优化器
    extra_context: dict = field(default_factory=dict)
```

`auto_select_engine_optim_fn` 是一个高阶函数, 接收 `(HFModelConfig, device_name)`, 返回 `(EngineConfig, OptimizerConfig)`, 用于根据模型类型和设备自动选择最优引擎配置。

### 3.4.1 Checkpoint 与 Router Replay 配置

- `McoreCheckpointConfig`（`checkpoint.py:34`）在通用 `CheckpointConfig` 上增加 Megatron-Core 的 `mbridge_config`；`MindSpeedCheckpointConfig`（`checkpoint.py:53`）继承它，供 MindSpeed 的 checkpoint 目标配置使用。
- `EngineRouterReplayConfig`（`engine.py:48`）是引擎层的 Router Replay 配置；`RouterReplayConfig`（`actor.py:50`）是 Actor 层的对应配置。二者均校验 `disabled`/`R2`/`R3` 模式，字段为 `mode`、`record_file`、`replay_file`；引擎类上的 TODO 表明 legacy 命名尚待统一。

### 3.5 ActorConfig (actor.py 第 104 行)

Actor 模型训练配置, 核心字段:

```python
@dataclass
class ActorConfig(BaseConfig):
    strategy: str = MISSING                     # 必须指定
    ppo_mini_batch_size: int = 256              # PPO mini-batch 大小
    ppo_micro_batch_size: Optional[int] = None  # 已废弃
    ppo_micro_batch_size_per_gpu: Optional[int] = None  # 每 GPU micro-batch
    use_dynamic_bsz: bool = False               # 动态 batch size
    ppo_max_token_len_per_gpu: int = 16384      # 每 GPU 最大 token 长度
    clip_ratio: float = 0.2                     # PPO 裁剪比
    clip_ratio_low: float = 0.2                 # 裁剪下界
    clip_ratio_high: float = 0.2                # 裁剪上界
    loss_agg_mode: str = "token-mean"           # 损失聚合模式
    entropy_coeff: float = 0                    # 熵正则系数
    use_kl_loss: bool = False                   # KL 散度损失
    kl_loss_coef: float = 0.001
    ppo_epochs: int = 1                         # PPO 迭代轮数
    rollout_n: int = MISSING                    # 必须由 sampling config 覆盖
    engine: BaseConfig = field(default_factory=BaseConfig)   # 运行时填充
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)
```

合法的 `loss_agg_mode` 取值 (第 213-218 行): `"token-mean"`, `"seq-mean-token-sum"`, `"seq-mean-token-mean"`, `"seq-mean-token-sum-norm"`。

`__post_init__` 校验逻辑 (第 196-220 行):
- `strategy` 和 `rollout_n` 不能为 `MISSING`
- 非 `use_dynamic_bsz` 模式下, `ppo_micro_batch_size` 和 `ppo_micro_batch_size_per_gpu` 二选一
- `loss_agg_mode` 必须在合法值列表中

`validate()` (第 222 行): 运行时验证, 检查 `train_batch_size >= ppo_mini_batch_size`, micro_batch_size 整除性, 序列并行约束。

5 种后端变体:

| 类 | strategy | 引擎类型 | 行号 |
|----|----------|----------|------|
| `FSDPActorConfig` | `"fsdp"` | `FSDPEngineConfig` | 第 289 行 |
| `McoreActorConfig` | `"megatron"` | `McoreEngineConfig` | 第 261 行 |
| `VeOmniActorConfig` | `"veomni"` | `VeOmniEngineConfig` | 第 342 行 |
| `TorchTitanActorConfig` | `"torchtitan"` | `TorchtitanEngineConfig` | 第 373 行 |
| `MindSpeedActorConfig` | `"mindspeed"` | `MindSpeedEngineConfig` | 第 397 行 |

每种变体的 `__post_init__` 都执行 `self.engine = self.xxx_config`, 将特化引擎配置赋值到通用 `engine` 字段。

### 3.6 PolicyLossConfig (actor.py 第 79 行)

策略损失配置:

```python
loss_mode: str = "vanilla"          # "vanilla" / "clip-cov" / "kl-cov" / "gpg"
clip_cov_ratio: float = 0.0002
clip_cov_lb: float = 1.0
clip_cov_ub: float = 5.0
ppo_kl_coef: float = 0.1
```

### 3.7 CriticConfig (critic.py 第 47 行)

Critic 模型训练配置, 结构与 ActorConfig 类似, 核心区别:

```python
cliprange_value: float = 0.5       # 值函数裁剪范围 (Actor 没有)
forward_max_token_len_per_gpu: int = 32768  # 前向推理最大 token 长度
enable: Optional[bool] = None      # 是否启用 Critic (GRPO 等算法不需要)
loss_scale_factor: Optional[int] = None  # 第 97 行, 'seq-mean-token-sum-norm' 损失缩放 (基准 e3573545 新增)
```

5 个 `CriticConfig` 后端变体, 外加 1 个模型层配置 `FSDPCriticModelCfg` (继承 `BaseModelConfig`, 不属于 `CriticConfig` 家族):

| 类 | strategy | 行号 |
|----|----------|------|
| `FSDPCriticConfig` | `"fsdp"` | 第 190 行 |
| `McoreCriticConfig` | `"megatron"` | 第 162 行 |
| `TorchTitanCriticConfig` | `"torchtitan"` | 第 238 行 |
| `MindSpeedCriticConfig` | `"mindspeed"` | 第 286 行 |
| `VeOmniCriticConfig` | `"veomni"` | 第 309 行 |
| `FSDPCriticModelCfg` | -- | 第 258 行 (继承 `BaseModelConfig`, 非 `CriticConfig`) |

`FSDPCriticModelCfg` (第 258 行) 是特殊的: 它继承 `BaseModelConfig` 而非 `CriticConfig`, 是面向模型本身的配置 (含 LoRA、activation offload 等), 而非训练循环配置。

### 3.8 HFModelConfig (model.py 第 71 行)

HuggingFace 模型加载配置, 包含大量自动初始化逻辑:

```python
@dataclass
class HFModelConfig(BaseConfig):
    path: str = MISSING                   # 模型路径 (必须)
    local_path: Optional[str] = None      # 本地缓存路径 (自动填充)
    hf_config_path: Optional[str] = None  # HF config 路径 (默认等于 path)
    tokenizer_path: Optional[str] = None  # tokenizer 路径 (默认等于 path)
    model_type: str = "language_model"    # "language_model" 或 "value_model"
    use_shm: bool = False                 # 共享内存加载
    trust_remote_code: bool = False       # 信任远程代码
    custom_chat_template: Optional[str] = None
    enable_gradient_checkpointing: bool = True
    enable_activation_offload: bool = False
    use_remove_padding: bool = True
    lora_rank: int = 0                    # FSDP LoRA rank
    lora_alpha: int = 16
    target_modules: Optional[Any] = "all-linear"
    lora: dict = field(default_factory=dict)   # Megatron LoRA 配置
    mtp: MtpConfig = field(default_factory=MtpConfig)  # MTP 推测解码
```

`__post_init__` (第 148 行) 执行的自动初始化:
1. 调用 `import_external_libs()` 导入外部库
2. 将远程路径 `copy_to_local()` 到本地
3. 加载 tokenizer 和 processor
4. 同步 chat_template (processor 没有时从 tokenizer 复制)
5. 加载 `generation_config` 和 `hf_config` (AutoConfig)
6. 应用 `override_config` 覆盖
7. 处理 MTP: 当 `mtp.enable=False` 时, 将 `num_nextn_predict_layers` 等字段清零
8. 验证 `target_modules` 类型

变更 (基准 `e3573545`): `external_lib` 类型由 `Optional[str]` 放宽为 `Any` (第 113 行); `__post_init__` 加载 `hf_config` 时新增 deepseek_v4 回退 (第 186-202 行) —— 当 `AutoConfig.from_pretrained` 因 `KeyError("deepseek_v4")` 失败时, 改用 vLLM 的 `get_config()`。

### 3.9 MtpConfig (model.py 第 30 行)

多 Token 预测 (MTP) / 推测解码配置:

```python
enable: bool = False              # 是否加载 MTP 参数
enable_train: bool = False        # 训练时是否使用 MTP
enable_rollout: bool = False      # 推理时是否使用 MTP
detach_encoder: bool = False      # 训练时是否 detach encoder
mtp_loss_scaling_factor: float = 0.1
# vLLM 参数
method: str = "mtp"
num_speculative_tokens: int = 1
# SGLang 参数
speculative_algorithm: str = "EAGLE"
speculative_num_steps: int = 3
speculative_eagle_topk: int = 1
speculative_num_draft_tokens: int = 4
```

### 3.10 OptimizerConfig 及变体 (optimizer.py)

基类 `OptimizerConfig` (第 34 行):

```python
lr: float = 1e-3
lr_warmup_steps_ratio: float = 0.0
total_training_steps: int = -1       # 运行时覆盖
weight_decay: float = 0.01
betas: tuple[float, float] = (0.9, 0.999)
clip_grad: float = 1.0
grad_clip: Optional[float] = None   # 已废弃, 使用 clip_grad
```

5 种变体:

| 类 | 特色 | 行号 |
|----|------|------|
| `FSDPOptimizerConfig` | 动态导入优化器 (`optimizer_impl` + `optimizer`), 支持 cosine LR | 第 88 行 |
| `McoreOptimizerConfig` | Megatron 风格: `lr_decay_style`, `min_lr`, `weight_decay_incr_style` | 第 128 行 |
| `VeOmniOptimizerConfig` | VeOmni 风格: `lr_scheduler_type`, `lr_min`, `lr_start` | 第 65 行 |
| `TorchtitanOptimizerConfig` | TorchTitan 风格: `decay_type`, `min_lr_factor` | 第 238 行 |
| `AutomodelOptimizerConfig` | Automodel 风格: `init_lr_ratio`, `min_lr_ratio`, FP8 优化器 | 第 255 行 |

`build_optimizer()` (第 298 行): 通用优化器构建函数, 通过 `importlib.import_module(config.optimizer_impl)` 动态导入优化器类。支持:
- `torch.optim.AdamW`
- `torchao.optim._AdamW` (bf16 随机舍入)
- `bitsandbytes.optim.AdamW8bit`

`McoreOptimizerConfig` 新增精度感知 / Muon 字段 (基准 `e3573545`): `use_precision_aware_optimizer` 及 `main_grads_dtype` / `exp_avg_dtype` / `exp_avg_sq_dtype` (第 201-204 行), 以及一组 `muon_*` 与 layer-wise 优化器字段 (第 205-224 行); `__post_init__` (第 227 行) 校验三个 dtype 字段取值合法 (`fp32` / `bf16` 等)。

### 3.11 RolloutConfig (rollout.py 第 145 行)

推理引擎配置, 是配置最复杂的类之一:

```python
name: Optional[str] = MISSING          # 引擎名: "vllm" / "sglang" / "trtllm" / "hf"
mode: str = "async"                    # 仅支持 "async" (sync 已移除)
temperature: float = 1.0
top_k: int = -1
top_p: float = 1.0
prompt_length: int = 512
response_length: int = 512
gpu_memory_utilization: float = 0.5
data_parallel_size: int = 1
tensor_model_parallel_size: int = 2
pipeline_model_parallel_size: int = 1
expert_parallel_size: int = 1
enable_chunked_prefill: bool = True
enable_prefix_caching: bool = True
load_format: str = "dummy"             # 混合引擎模式下用 dummy, 权重从 trainer 同步
enable_sleep_mode: bool = True         # vLLM sleep/wake 机制
sglang_engine_mode: str = "local"      # "local" 或 "server"
```

基准 `e3573545` 新增公开字段: `full_determinism` (第 171 行) / `seed` (第 175 行) / `standalone_gpu_memory_utilization` (第 186 行) / `moe_load_balance_metrics_interval` (第 266 行); `_mutable_fields` 新增 `full_determinism` 与 `max_num_seqs` (第 146-156 行)。

嵌入 9 个子配置:

| 子配置 | 类 | 行号 | 说明 |
|--------|-----|------|------|
| `val_kwargs` | `SamplingConfig` | 第 39 行 | 验证集采样参数 |
| `agent` | `AgentLoopConfig` | 第 72 行 | Agent 循环参数 |
| `trace` | `TraceConfig` | 第 83 行 | 追踪/日志 |
| `multi_turn` | `MultiTurnConfig` | 第 48 行 | 多轮对话 |
| `server` | `ServerConfig` | 第 96 行 | SGLang server 模式参数 |
| `prometheus` | `PrometheusConfig` | 第 109 行 | Prometheus 监控 |
| `checkpoint_engine` | `CheckpointEngineConfig` | 第 125 行 | 权重同步引擎 |
| `mtp` | `MtpConfig` | -- | 推测解码 |
| `disaggregation` | `DisaggregationConfig` | -- | Prefill-Decode 分离 |

> 注: `AgentLoopConfig` 内部还嵌套二级子配置 `CustomAsyncServerConfig` (第 66 行, 经 `custom_async_server` 字段接入)。

`__post_init__` 校验 (第 276 行):
- `sync` 模式已移除, 会抛出 `ValueError`
- 当 `expert_parallel_size > 1` 且非 trtllm 时, `expert_parallel_size` 必须等于 `tensor_model_parallel_size * data_parallel_size`；`expert_parallel_size == 1` 不触发该等式校验
- `pipeline_model_parallel_size > 1` 对 vllm/sglang/trtllm 均未实现
- `disaggregation.enabled=True` 仅支持 sglang 和 vllm (第 339 行, `name not in ("sglang", "vllm")`)

### 3.12 DisaggregationConfig (disaggregation.py 第 26 行)

Prefill-Decode 分离 (PD 分离) 配置 (支持 sglang 和 vllm):

```python
enabled: bool = False
prefill_replicas: int = 1
decode_replicas: int = 1
decode_tensor_model_parallel_size: Optional[int] = None
transfer_backend: str = "nixl"    # 允许值: ("nixl", "mooncake", "ascend", "mori", "fake")
bootstrap_port: Optional[int] = None
ib_device: Optional[str] = None
mooncake_protocol: str = "nvlink" # 允许值: ("nvlink", "local", "rdma", "tcp"), 仅 mooncake 后端校验
```

常量 `_ALLOWED_BACKENDS = ("nixl", "mooncake", "ascend", "mori", "fake")` (第 21 行); `_ALLOWED_MOONCAKE_PROTOCOLS = ("nvlink", "local", "rdma", "tcp")` (第 22 行)。`__post_init__` (第 50 行) 仅在 `transfer_backend == "mooncake"` 时校验 `mooncake_protocol`。

### 3.13 DistillationConfig (distillation.py 第 222 行)

在线蒸馏 (On-Policy Distillation) 配置:

```python
enabled: bool = False
n_gpus_per_node: int = 0           # 教师资源池每节点 GPU 数
nnodes: int = 0                     # 教师资源池节点数
teacher_models: dict[str, DistillationTeacherModelConfig] = field(default_factory=dict)
teacher_key: str = "data_source"   # 路由字段名
distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)
```

`DistillationLossConfig` (第 32 行) 支持多种蒸馏损失:

```python
loss_mode: str = "k3"             # "k1" / "k3" / "forward_kl_topk" 等
topk: Optional[int] = 128
use_task_rewards: bool = True     # 是否混合任务奖励
use_policy_gradient: bool = True  # True: 作为奖励信号; False: 直接反向传播
```

校验约束 (第 100 行, `DistillationLossConfig.__post_init__`):
- `use_policy_gradient=False` 且 `loss_mode="k1"` 时报错 (k1 梯度不依赖教师 log-prob)
- `use_policy_gradient=True` 且 `loss_mode="forward_kl_topk"` 时发出警告

基准 `e3573545` 新增 top-k 分块字段 `use_chunked_topk` (第 77 行, 默认 `False`) 与 `chunked_topk_chunk_size` (第 82 行, 默认 4096), 控制 (B*T) 维度上的分块大小。

`DistillationTeacherModelConfig` (第 128 行) 支持多教师:

```python
key: Optional[str] = None           # 路由键值
model_path: Optional[str] = None    # 教师模型路径
inference: RolloutConfig             # 教师推理配置
num_replicas: Optional[int] = 0     # 推理副本数 (单教师时自动计算)
```

### 3.14 RewardConfig (reward.py 第 94 行)

```python
num_workers: int = 8
reward_manager: RewardManagerConfig
reward_model: RewardModelConfig      # 奖励模型 (可选)
sandbox_fusion: SandboxFusionConfig  # 沙箱执行
```

`RewardManagerConfig` (第 32 行) 支持两种来源: `"register"` (内置注册表) 和 `"importlib"` (动态导入)。

### 3.15 CheckpointEngineConfig (rollout.py 第 125 行)

权重同步引擎配置 (trainer -> rollout):

```python
backend: Optional[str] = "naive"               # "naive" / "nccl" / "nixl" / "hccl"
update_weights_bucket_megabytes: int = 2048     # 批量传输大小 (MB)
custom_backend_module: Optional[str] = None     # 自定义后端模块路径
```

`_mutable_fields = {"backend"}` (第 130 行): 仅 `backend` 允许运行时改写。

---

## 4. 算法详解

### 4.1 配置冻结与可变字段机制

`BaseConfig` 使用 `__setattr__` 拦截赋值。整体逻辑:

```
对象初始化 (__init__):
  dataclass 的 __init__ 逐个设置字段值
  第一次赋值时字段不在 __dict__ 中, 允许通过

后续赋值:
  字段已在 __dict__ 中
  如果字段名在 _mutable_fields 中 -> 允许修改
  否则 -> 抛出 FrozenInstanceError
```

子类可通过集合合并扩展可变字段:
```python
class ActorConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields | {"ppo_mini_batch_size", "engine", ...}
```

特殊绕过: 当引擎初始化需要修改冻结字段时, 使用 `object.__setattr__(self.engine, "strategy", self.strategy)` (如 actor.py 第 322 行)。

### 4.2 ActorConfig/CriticConfig 到 TrainingWorkerConfig 的转换路径

用户在 YAML 中配置 `FSDPActorConfig`, Trainer 将其转换为 `TrainingWorkerConfig`:

```
YAML 配置 (Hydra)
  ↓ omega_conf_to_dataclass()
FSDPActorConfig
  ├── strategy = "fsdp"
  ├── fsdp_config: FSDPEngineConfig
  ├── optim: OptimizerConfig
  ├── checkpoint: CheckpointConfig
  └── (各种 PPO 参数)
  ↓ __post_init__()
  │  self.engine = self.fsdp_config    # 统一到 engine 字段
  ↓ Trainer 构建
TrainingWorkerConfig
  ├── model_config: HFModelConfig
  ├── engine_config: FSDPEngineConfig  # 从 actor.engine 取
  ├── optimizer_config: OptimizerConfig
  ├── checkpoint_config: CheckpointConfig
  └── extra_context: dict
```

### 4.3 ppo_micro_batch_size 与 ppo_micro_batch_size_per_gpu 的互斥

这两个参数是历史遗留问题。旧版本使用全局 `ppo_micro_batch_size`, 新版本改为 `ppo_micro_batch_size_per_gpu`。

`_check_mutually_exclusive()` (actor.py 第 245 行, critic.py 第 134 行) 确保:
- 至少设置一个
- 不能同时设置两个

### 4.4 多教师蒸馏的资源分配

`DistillationConfig._resolve_teacher_models()` (第 289 行):

1. 单教师模式: `teacher_models` 字典只有默认 `"teacher_model"` 键
   - 自动计算 `num_replicas = pool_size / per_replica_world_size`
   - 设置 `key = "default"`

2. 多教师模式: `teacher_models` 字典有额外键
   - 移除默认的 `"teacher_model"` 条目
   - 用 `teacher_config.key` 作为新字典键
   - 验证各教师 `world_size` 总和等于资源池大小

### 4.5 build_optimizer 的动态导入

`build_optimizer()` (optimizer.py 第 298 行) 通过 `importlib.import_module()` 动态加载优化器:

```python
module = importlib.import_module(config.optimizer_impl)  # 如 "torch.optim"
optimizer_cls = getattr(module, config.optimizer)          # 如 "AdamW"
```

对 Adam 类优化器自动传入 `betas` 参数 (第 333 行, betas 注入)。支持 `override_optimizer_config` 字典透传额外参数。

---

## 5. 数据流

### 5.1 配置加载与验证流程

```
1. Hydra YAML 文件 (如 ppo_trainer.yaml)
   ↓ @hydra.main 解析
2. OmegaConf DictConfig
   ↓ omega_conf_to_dataclass()
3. 类型安全的 dataclass 实例
   ├── __post_init__() 执行:
   │   ├── 字段合法性校验 (枚举值, 范围)
   │   ├── 自动填充 (engine = fsdp_config)
   │   ├── 废弃字段兼容 (grad_clip -> clip_grad)
   │   └── HFModelConfig: 加载 tokenizer, hf_config, copy_to_local
   │
   ↓ Trainer 调用 config.validate(n_gpus, batch_size)
4. 运行时验证
   ├── batch_size >= mini_batch_size
   ├── mini_batch_size % micro_batch_size == 0
   └── micro_batch_size * sp_size >= n_gpus
```

### 5.2 配置在组件间的传递

```
TrainerConfig (顶层)
  ├── actor: ActorConfig
  │   ├── .engine -> FSDPEngineConfig -> Worker 初始化时用
  │   ├── .optim -> OptimizerConfig -> 构建优化器
  │   ├── .model_config -> HFModelConfig -> 加载模型权重
  │   └── PPO 参数 -> core_algos.py 算法函数
  │
  ├── critic: CriticConfig
  │   └── (同上)
  │
  ├── rollout: RolloutConfig
  │   ├── .name -> 选择推理引擎 (vllm/sglang/trtllm)
  │   ├── 采样参数 -> 生成时使用
  │   ├── .checkpoint_engine -> 权重同步机制
  │   └── .disaggregation -> PD 分离配置
  │
  ├── reward: RewardConfig
  │   ├── .reward_manager -> 奖励计算逻辑
  │   └── .reward_model -> 奖励模型推理
  │
  └── distillation: DistillationConfig
      ├── .teacher_models -> 教师模型推理配置
      └── .distillation_loss -> 蒸馏损失函数配置
```

---

## 6. 设计决策

### 6.1 为什么使用冻结 dataclass 而不是 OmegaConf DictConfig

- **类型安全**: dataclass 的字段有类型注解, IDE 补全友好, `__post_init__` 可做运行时校验
- **防误改**: 冻结后配置不会被训练循环意外修改, 减少 bug
- **可序列化**: dataclass 天然支持 pickle (Ray 需要), 而 OmegaConf 的 DictConfig 序列化有限制
- **兼容性**: 通过实现 `Mapping` 接口, 保持向后兼容的字典访问方式

### 6.2 为什么 EngineConfig 有这么多变体

六种引擎后端的配置差异巨大:
- FSDP: `wrap_policy`, `fsdp_size`, `reshard_after_forward`
- Megatron: `tensor_model_parallel_size`, `pipeline_model_parallel_size`, `sequence_parallel`
- VeOmni: 各种算子实现选择 (`attn_implementation`, `moe_implementation`)
- TorchTitan: 多维并行 (`data_parallel_replicate_size`, `data_parallel_shard_size`)
- Automodel: `distributed_strategy`, `backend_config`, `moe_config`
- MindSpeed: `mcore_kwargs`, `fsdp_kwargs`

通过继承, 公共字段 (offload, dtype, seed) 只定义一次。

### 6.3 为什么 ActorConfig.engine 是 BaseConfig 而不是 EngineConfig

`engine: BaseConfig = field(default_factory=BaseConfig)` (actor.py 第 184 行) 使用宽松类型, 因为:
1. 它在 `__post_init__` 中被赋值为具体的引擎配置 (如 `self.engine = self.fsdp_config`)
2. 使用 `BaseConfig` 默认值避免了在基类中引入特定引擎依赖
3. `_mutable_fields` 中包含 `"engine"`, 允许运行时覆盖

### 6.4 ppo_micro_batch_size 废弃策略

旧字段 `ppo_micro_batch_size` (全局) 被 `ppo_micro_batch_size_per_gpu` (每 GPU) 取代。保留旧字段是为了向后兼容, 但通过 `_check_mutually_exclusive()` 确保不会同时设置两者。最终将移除旧字段。

### 6.5 HFModelConfig 的重初始化问题

`HFModelConfig.__post_init__` 会执行文件系统操作 (copy_to_local) 和模型加载 (AutoConfig.from_pretrained)。这意味着:
- 每次从 Hydra DictConfig 转换时都会触发 I/O
- 在 Ray actor 中反序列化时可能重复执行
- 通过 `use_shm=True` 可使用共享内存缓解

### 6.6 MindSpeedEngineConfig 跳过父类 __post_init__

`MindSpeedEngineConfig.__post_init__()` (第 629 行) 没有调用 `super().__post_init__()`, 而是复制了父类 `McoreEngineConfig` 的校验逻辑。原因是父类的 `assert self.strategy == "megatron"` 会失败 (MindSpeed 的 strategy 是 `"mindspeed_megatron"` 或 `"mindspeed_fsdp"`)。

---

## 7. 已知问题与局限

### 7.1 wildcard import 的使用

`__init__.py` (第 16-25 行) 使用 `from .actor import *` 等 wildcard import, 虽然每个模块都定义了 `__all__`, 但这与项目 CLAUDE.md 中 "Never Do: Use wildcard imports" 的规则不一致。

### 7.2 配置类之间的潜在耦合风险

`distillation.py` 导入 `RolloutConfig` (第 23 行), `reward.py` 也导入 `RolloutConfig` (第 23 行)。当前是 `distillation/reward -> rollout` 的单向依赖，并未形成已发生的循环；如果未来 `RolloutConfig` 反向引用这些类型，才会形成循环风险。现状仍增加了配置层的耦合复杂度。

### 7.3 DistillationLossConfig 中的 print 使用

distillation.py 第 113-118 行使用了 `print("WARNING: ...")` 而不是 `logger.warning()`, 违反了项目日志规范。

### 7.4 FSDPCriticModelCfg 的命名不一致

`FSDPCriticModelCfg` (critic.py 第 258 行) 继承 `BaseModelConfig` 而非 `CriticConfig`, 但放在 critic.py 文件中且名字以 "Critic" 开头, 容易造成混淆。它实际上是模型层面的配置 (含 LoRA, activation offload), 与训练循环配置 (`CriticConfig`) 是不同层面。

### 7.5 EngineRouterReplayConfig 与 RouterReplayConfig 的重复

`engine.py` 第 48 行的 `EngineRouterReplayConfig` 与 `actor.py` 第 50 行的 `RouterReplayConfig` 字段完全相同 (mode, record_file, replay_file), 代码注释中标注了 `# TODO: rename to RouterReplayConfig after removing the legacy implementation`。

### 7.6 megatron_peft.py 使用 print 而非 logger

`megatron_peft.py` 第 31-34 行使用 `print()` 输出 LoRA 配置信息, 违反了项目日志规范。

### 7.7 RolloutConfig.mode 的废弃状态

`mode` 字段默认值为 `"async"`, `sync` 模式已被移除并抛出 `ValueError` (第 280 行)。其他值会发出 `DeprecationWarning`。该字段应在未来版本中完全移除。

---

## 8. 测试覆盖

测试文件位于 `tests/workers/config/`, 共 5 个文件:

| 测试文件 | 测试焦点 |
|----------|----------|
| `test_actor_config_on_cpu.py` | ActorConfig 各变体的创建、校验、互斥参数检查 |
| `test_critic_config_on_cpu.py` | CriticConfig 各变体的创建、校验 |
| `test_engine_config_on_cpu.py` | EngineConfig 各变体的创建、strategy 校验 |
| `test_model_config_on_cpu.py` | HFModelConfig 的创建、tokenizer 加载、override_config |
| `test_optim_config_on_cpu.py` | OptimizerConfig 各变体、build_optimizer 函数 |

所有测试均可在 CPU 上运行 (文件名含 `_on_cpu` 后缀)。

覆盖不足之处:
- `RolloutConfig` 缺少专门的配置单元测试 (验证 disaggregation、expert_parallel_size 等约束)
- `DistillationConfig` 的多教师资源分配逻辑缺少测试
- `DisaggregationConfig` 的 `effective_decode_tp()` 和边界条件缺少测试
- `RewardConfig` / `RewardManagerConfig` 的 register/importlib 两种模式缺少配置级测试
- 配置的 Hydra YAML -> dataclass 端到端转换路径缺少集成测试
- `TrainingWorkerConfig` 的 `auto_select_engine_optim_fn` 回调机制缺少测试
