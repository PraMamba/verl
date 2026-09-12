# 13 - 外围子模块架构文档（plugin / model_merger / tools / third_party）

> **源码位置**: `verl/plugin/`(7 文件, 922 行) + `verl/model_merger/`(6 文件, 1,460 行) + `verl/tools/`(5 文件, 593 行) + `verl/third_party/`(7 文件, 2,667 行)
>
> **总计**: 25 文件, 5,642 行
>
> **最后更新**: 2026-08-02 | **基准源码**: 上游 `e3573545`

---

## 1. 模块定位

这四个模块位于 verl 核心训练循环的外围，提供以下基础能力：

| 模块 | 职责 | 使用场景 |
|------|------|----------|
| `plugin/platform/` | 硬件平台抽象层 | 所有设备操作的统一接口 |
| `model_merger/` | 分片检查点合并 | 训练后将分布式检查点合并为 HuggingFace 格式 |
| `tools/` | 工具调用框架 | Agent Loop 的工具定义和执行 |
| `third_party/` | 第三方库补丁 | 兼容性修复（PyTorch/vLLM） |

---

## 2. plugin/platform/ -- 设备平台抽象层

### 2.1 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `platform_base.py` | 268 | `PlatformBase`（L20）：抽象基类，定义完整的平台操作接口 |
| `platform_cuda.py` | 186 | `PlatformCUDA`：NVIDIA GPU 实现 |
| `platform_npu.py` | 198 | `PlatformNPU`：华为 Ascend NPU 实现 |
| `platform_rocm.py` | 76 | `PlatformROCm`（L21）：AMD ROCm/HIP GPU 实现，注册键 `amd`，继承 `PlatformCUDA` |
| `platform_manager.py` | 174 | `PlatformRegistry`（L29）：注册表 + 自动检测逻辑；`get_platform()` 单例入口 |
| `__init__.py` | 19 | 导出 `get_platform` |
| `../plugin/__init__.py` | 1 | 空 |

### 2.2 PlatformBase 抽象接口

`PlatformBase`（platform_base.py, L20）定义了硬件平台必须实现的完整接口：

```
核心设备管理:
  device_name      -> str          # "cuda" / "npu" / "cpu"
  vendor_name      -> str          # "nvidia" / "huawei" / "intel"
  device_module    -> ModuleType   # torch.cuda / torch.npu
  is_available()   -> bool
  current_device() -> int
  device_count()   -> int
  set_device(idx)
  synchronize()

随机数生成器:
  manual_seed(seed)
  manual_seed_all(seed)

内存管理:
  set_allocator_settings(settings: str)
  empty_cache()

设备属性:
  get_device_capability(device_id) -> (major, minor)

分布式通信:
  communication_backend_name() -> str    # "nccl" / "hccl" / "gloo"
  visible_devices_envvar()     -> str    # "CUDA_VISIBLE_DEVICES" 等

性能分析:
  nvtx_range(msg)       # 上下文管理器
  profiler_start()
  profiler_stop()

Ray 集成:
  ray_resource_name()    -> str          # "GPU" / "NPU"
  ray_noset_envvars()    -> list[str]
  ray_resource_options() -> dict

IPC 支持:
  is_ipc_supported()     -> bool

Rollout 引擎:
  rollout_env_vars()     -> dict
  
底层运行时:
  cudart()               -> Any
```

还包含通用工具方法：
- `check_smi_command(cmd)`（L36）：检测 nvidia-smi / mx-smi 等系统命令
- `get_device_uuid(device_id)`（L195）：获取设备 UUID
- `apply_model_patches(model_type)`（L203）：应用平台特定模型补丁

### 2.3 PlatformRegistry 自动检测

`PlatformRegistry`（platform_manager.py, L29）使用装饰器注册模式：

```python
@PlatformRegistry.register(platform="nvidia")
class PlatformCUDA(PlatformBase): ...

@PlatformRegistry.register(platform="huawei")
class PlatformNPU(PlatformBase): ...

@PlatformRegistry.register(platform="amd")
class PlatformROCm(PlatformCUDA): ...
```

`get_platform()` 的检测优先级：
1. 环境变量 `VERL_PLATFORM` 显式指定
2. 遍历已注册平台，调用 `is_platform_available()` 自动检测
3. 检测失败时回退到已注册键 `nvidia`（`platform_manager.py:83-117`）；这不是 CPU 平台回退，CPU-only Ray actor 仍由平台实例的 `is_available()` 结果单独处理。

检测结果缓存在模块级全局变量 `_current_platform` 中，整个进程生命周期只解析一次。

### 2.4 与 device.py 的关系

`verl/utils/device.py` 是面向调用者的兼容层（多个导入站点），所有函数内部委托给 `get_platform()`：

```
调用者代码                     device.py                PlatformBase
─────────────────────────────────────────────────────────────────────
get_device_name()         -->  get_platform().device_name
get_torch_device()        -->  get_platform().device_module
get_nccl_backend()        -->  get_platform().communication_backend_name()
is_device_available()     -->  get_platform().is_available()
manual_seed(seed)         -->  get_platform().manual_seed(seed)
```

---

## 3. model_merger/ -- 分片检查点合并

### 3.1 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `base_model_merger.py` | 466 | `ModelMergerConfig`（L84）：合并配置 dataclass；`BaseModelMerger`（L177）：抽象基类 |
| `fsdp_model_merger.py` | 265 | `FSDPModelMerger`：FSDP 分片检查点合并实现 |
| `megatron_model_merger.py` | 549 | `MegatronModelMerger`：Megatron dist_ckpt 格式合并实现 |
| `output_validation.py` | 94 | `validate_hf_model_output`（L34）：结构与文件完整性门禁（config.json 为 JSON object、权重文件/分片索引存在且非空、路径安全）；不验证模型可加载性、tensor key 完整性或 tokenizer |
| `__main__.py` | 73 | CLI 入口：`python -m verl.model_merger merge --backend fsdp ...` |
| `__init__.py` | 13 | 导出 |

### 3.2 BaseModelMerger 架构

```
BaseModelMerger (base_model_merger.py, L177)
  │  抽象方法：merge_and_save(), cleanup()
  │
  │  通用功能：
  │  ├── get_transformers_auto_model_class()
  │  │     根据 model_config.architectures 自动选择 AutoModel 类
  │  │     支持：CausalLM, TokenClassification, Vision2Seq
  │  │
  │  ├── save_lora_adapter(state_dict)
  │  │     从 state_dict 中提取 LoRA 参数
  │  │     推断 rank, alpha, target_modules
  │  │     保存为 PEFT 格式 (adapter_config.json + safetensors)
  │  │
  │  ├── save_hf_model_and_tokenizer(state_dict)
  │  │     创建空模型 -> 提取 LoRA -> 保存模型/处理器/tokenizer
  │  │
  │  └── upload_to_huggingface()
  │        通过 HfApi 上传到 HuggingFace Hub
  │
  ├── FSDPModelMerger
  │     从 FSDP 分片检查点重建完整 state_dict
  │     使用 torch.distributed.checkpoint 加载
  │
  └── MegatronModelMerger
        从 Megatron dist_ckpt 格式转换
        处理 TP/PP 分片合并
        支持 v2 布局（model/huggingface + model/dist_ckpt）
```

### 3.3 CLI 使用

```bash
# 合并 FSDP 检查点
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /path/to/checkpoint \
    --target_dir /path/to/output

# 合并 Megatron 检查点并上传
python -m verl.model_merger merge \
    --backend megatron \
    --local_dir /path/to/checkpoint \
    --target_dir /path/to/output \
    --hf_upload_path user/model-name

# 测试合并结果
python -m verl.model_merger test \
    --backend fsdp \
    --local_dir /path/to/checkpoint \
    --test_hf_dir /path/to/reference_model
```

### 3.4 LoRA 适配器处理

`save_lora_adapter()`（base_model_merger.py, L293）的处理流程：
1. 从 state_dict 中过滤 `lora_` 前缀的参数
2. 从 `lora_train_meta.json` 读取训练元数据（rank, alpha, task_type）
3. 如无元数据，从权重形状推断 rank
4. 生成 PEFT `adapter_config.json` 并保存 safetensors
5. 清理 state_dict 中的 `base_model.model.` / `.base_layer.` 前缀

---

## 4. tools/ -- 工具调用框架

### 4.1 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `base_tool.py` | 93 | `BaseTool`（L24）：有状态工具抽象基类（create/execute/release 生命周期） |
| `function_tool.py` | 258 | `FunctionTool`（L45）：无状态函数工具；`@function_tool` 装饰器 |
| `schemas.py` | 127 | OpenAI 函数调用 schema 定义：`OpenAIFunctionToolSchema`、`OpenAIFunctionParsedSchema`（L59）、`OpenAIFunctionCallSchema`（L66）、`OpenAIFunctionToolCall`（L90）、`ToolResponse`（L98） |
| `tool_registry.py` | 101 | `ToolType`（L36）与 `load_all_tools()`：从配置文件和 Python 模块加载工具 |
| `__init__.py` | 14 | 导出 |

### 4.2 双轨工具模型

verl 支持两种工具定义方式，在 `ToolAgentLoop._call_tool()` 中统一调度：

#### BaseTool（有状态）

```python
class BaseTool:  # base_tool.py, L24
    async def create(instance_id, **kwargs) -> (str, ToolResponse)
    async def execute(instance_id, parameters, **kwargs) -> (ToolResponse, float, dict)
    async def calc_reward(instance_id, **kwargs) -> float
    async def release(instance_id, **kwargs) -> None
```

完整生命周期：create -> execute (多次) -> calc_reward -> release。适用于需要维护状态的工具（如代码沙箱、浏览器会话）。

#### FunctionTool（无状态）

```python
@dataclass
class FunctionTool:  # function_tool.py, L45
    name: str
    fn: Callable[..., Any]
    tool_schema: OpenAIFunctionToolSchema
    is_async: bool = False

    async def call(parameters: dict) -> Any
```

通过 `@function_tool` 装饰器注册，自动从函数签名和 docstring 推断 OpenAI schema（使用 `transformers.utils.get_json_schema`）。无生命周期管理，直接调用。

### 4.3 Schema 体系

```
OpenAIFunctionToolSchema (schemas.py)
  ├── type: str  # "function"
  └── function: OpenAIFunctionSchema
        ├── name: str
        ├── description: str
        ├── parameters: OpenAIFunctionParametersSchema
        │     ├── type: str  # "object"
        │     ├── properties: dict[str, OpenAIFunctionPropertySchema]
        │     └── required: list[str]
        └── strict: bool

OpenAIFunctionParsedSchema (schemas.py:59)
  ├── name: str
  └── arguments: str  # JSON 字符串

OpenAIFunctionCallSchema (schemas.py:66)
  ├── name: str
  └── arguments: dict[str, Any]

OpenAIFunctionToolCall (schemas.py:90)
  ├── id: str
  ├── type: Literal["function"]
  └── function: OpenAIFunctionCallSchema

ToolResponse (schemas.py, L98)
  ├── text: str | None
  ├── image: list[Any] | None    # 多模态图像
  └── video: list[Any] | None    # 多模态视频（当前不支持）
```

`ToolResponse` 使用 Pydantic `@model_validator` 确保 image/video 字段必须是列表格式。

`ToolType`（`tool_registry.py:36`）当前仅包含 `NATIVE` 类型，用于配置文件加载时选择原生 `BaseTool` 路径。

### 4.4 工具加载

`tool_registry.py` 中 `load_all_tools()` 支持两种加载路径：
1. **配置文件**（YAML）：`tool_config_path` 指定 `BaseTool` 子类的配置
2. **Python 模块**：`function_tool_path` 指定包含 `@function_tool` 装饰器的模块路径

---

## 5. third_party/ -- 第三方库补丁

### 5.1 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `torch/distributed/_state_dict_utils.py` | 840 | PyTorch FSDP state_dict 工具函数补丁 |
| `torch/distributed/checkpoint/state_dict.py` | 1,493 | PyTorch 2.7.0 的 `set_model_state_dict` 回移 |
| `torch/distributed/checkpoint/__init__.py` | 87 | 仅版权头注释 |
| `torch/distributed/__init__.py` | 87 | 仅版权头注释 |
| `torch/__init__.py` | 87 | 仅版权头注释 |
| `vllm/__init__.py` | 60 | vLLM 版本探测：`get_version` + `VLLM_SLEEP_LEVEL`，按 vllm/sglang 条件导入 `LLM`/`parallel_state` |
| `__init__.py` | 13 | 仅版权头注释 |

### 5.2 补丁动机

**核心问题**：PyTorch 2.6.0 官方的 `set_model_state_dict` API 在加载 FSDP2 全量 state_dict 时会导致 OOM（Out Of Memory）。

**解决方案**：从 PyTorch 2.7.0 回移（backport）修复后的 `set_model_state_dict` 实现：

```python
# fsdp_utils.py, L486-492
if version.parse(torch.__version__) >= version.parse("2.7.0"):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
else:
    # 官方 torch 2.6.0 的 set_model_state_dict API 导致 OOM
    # 使用从 verl/third_party/torch/distributed/checkpoint 回移的 torch 2.7.0 版本
    from verl.third_party.torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
```

### 5.3 _state_dict_utils.py 内容

`_state_dict_utils.py`（840 行）包含 FSDP 检查点操作的底层工具函数，主要处理：
- DTensor 到本地张量的转换
- 分片 state_dict 的 broadcast
- 混合精度下的类型转换
- FSDP1/FSDP2 兼容的 state_dict 操作

### 5.4 版本适配策略

verl 的 PyTorch 版本兼容策略体现在多处条件导入：

```
PyTorch 版本      兼容处理
──────────────────────────────────
>= 2.7.0         使用官方 API
>= 2.6.0         使用 verl/third_party 回移版本
>= 2.4.0         使用 composable FSDP API
< 2.4.0          FSDP2 不可用，设为 None
```

---

## 6. 设计模式与架构决策

### 6.1 策略模式（Platform）

`PlatformBase` 定义策略接口，`PlatformCUDA`/`PlatformNPU`/`PlatformROCm`（AMD，注册键 `amd`，继承 `PlatformCUDA`）提供具体实现。`PlatformRegistry` 管理策略注册和选择。AMD ROCm 已落地实现；其余新硬件（MetaX、XPU、MLU 等）只需实现 `PlatformBase` 子类并注册即可。

### 6.2 抽象基类与后端特化（ModelMerger）

`BaseModelMerger` 是抽象基类，提供共享的配置/加载/保存辅助方法；`merge_and_save()` 与 `cleanup()` 本身是抽象契约，实际合并流程由各后端实现（FSDP：`fsdp_model_merger.py:203-227`；Megatron：`megatron_model_merger.py:494-512`），CLI 在 `__main__.py:68-69` 顺序调用二者。子类扩展面不只是在 `merge_and_save()` 中加载 checkpoint：FSDP 还覆写 `_validate_state_dict` 与 `cleanup`（`fsdp_model_merger.py:229,262`），Megatron 还覆写 `save_hf_model_and_tokenizer` 与 `cleanup`（`megatron_model_merger.py:423,548`）。

### 6.3 装饰器注册模式（Tools）

`@function_tool` 装饰器利用 `transformers.utils.get_json_schema` 从函数签名自动生成 OpenAI function call schema，实现零配置工具注册：

```python
@function_tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.
    
    Args:
        expression: The math expression to evaluate.
    """
    return str(eval(expression))
```

### 6.4 Vendoring 策略（Third Party）

verl 选择 vendor（内嵌）而非 monkey-patch PyTorch 代码，原因：
- 补丁代码量大（`verl/third_party/` 当前 2,667 行），monkey-patch 容易遗漏
- 版本检测 + 条件导入确保只在需要时使用 vendor 版本
- 当 PyTorch 版本升级后自动切换到官方实现

---

## 7. 跨模块交互

```
                            ┌──────────────────────┐
                            │  verl/utils/device.py │ <── 多个调用站点
                            └──────────┬───────────┘
                                       │ 委托
                                       v
                          ┌────────────────────────────┐
                          │  verl/plugin/platform/     │
                          │  PlatformBase -> CUDA/NPU  │
                          │                 -> ROCm    │
                          └────────────────────────────┘

┌──────────────────────┐        ┌───────────────────────┐
│  verl/utils/         │        │  verl/third_party/    │
│  fsdp_utils.py       │ ────> │  torch/.../state_dict │
│  (PyTorch < 2.7 时)  │        │  (OOM 修复回移)       │
└──────────────────────┘        └───────────────────────┘

┌──────────────────────┐        ┌───────────────────────┐
│  verl/experimental/  │        │  verl/tools/          │
│  agent_loop/         │ ────> │  BaseTool/FunctionTool│
│  ToolAgentLoop       │        │  OpenAI Schema 体系   │
└──────────────────────┘        └───────────────────────┘

┌──────────────────────┐        ┌───────────────────────┐
│  训练完成后           │        │  verl/model_merger/   │
│  python -m verl.     │ ────> │  FSDP/Megatron 合并   │
│  model_merger merge  │        │  -> HuggingFace 格式  │
└──────────────────────┘        └───────────────────────┘
```

---

## 8. 依赖关系

```
verl/plugin/platform/
  主要被依赖：verl/utils/device.py；注册表/平台类也可由插件直接导入扩展
  无外部依赖（纯 Python + torch）

verl/model_merger/
  依赖：verl/utils/ (fsdp_utils, model, hf_tokenizer, transformers_compat)
       torch.distributed.checkpoint
       accelerate, transformers, peft, safetensors
       huggingface_hub (上传功能)
  被依赖：CLI 工具、训练后处理脚本

verl/tools/
  依赖：verl/utils/rollout_trace (追踪装饰器)
       transformers.utils.get_json_schema (schema 推断)
       pydantic (数据模型)
  被依赖：verl/experimental/agent_loop/ (ToolAgentLoop)
          用户自定义工具模块

verl/third_party/
  依赖：torch.distributed (内部 API)
  被依赖：verl/utils/fsdp_utils.py (条件导入，PyTorch < 2.7 时)
```
