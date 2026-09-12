# 协议与配置基础

> 源码位置：`verl/protocol.py`, `verl/base_config.py`, `verl/__init__.py`
> 文件数：3 个 | 总行数：1,555 行
> 最后更新：2026-08-02（基准源码版本：上游 verl `e3573545`）

## 1. 模块职责概述

本模块定义了 verl 框架的两个基础设施：`DataProto`（数据交换协议）和 `BaseConfig`（冻结配置基类）。

`DataProto` 是 legacy 控制器分发路径和许多公开 Worker API 的主要数据协议，但不是所有跨 Worker 数据面的唯一载体。V1 PPO 的 `TransferQueue`/`ReplayBuffer` 路径还使用 `TensorDict`、`KVBatchMeta` 等批元数据类型，训练后端内部则直接使用各自的张量集体通信。`DataProto` 自身封装了 PyTorch TensorDict（用于 tensor 数据）、numpy 字典（用于非 tensor 数据如字符串和对象）、以及 meta_info 字典（用于元信息如配置参数）。

`BaseConfig` 是所有配置 dataclass 的基类，提供 dict-like 接口并默认冻结所有字段，防止训练过程中意外修改配置。

上游依赖：`verl/utils/device.py`（设备抽象）、`verl/utils/py_functional.py`（工具函数）、`verl/utils/torch_functional.py`（tensor 操作）。

下游消费者：框架中几乎所有模块都依赖 `DataProto`。

## 2. 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `verl/protocol.py` | 1345 | DataProto、DataProtoFuture、BatchData 核心数据协议 |
| `verl/base_config.py` | 86 | BaseConfig 冻结配置基类 |
| `verl/__init__.py` | 124 | 包初始化、版本号、插件发现、NPU 兼容补丁 |

## 3. 核心数据结构与接口

### DataProto

- **类型**：dataclass
- **字段数**：3 个
- **关键字段**：
  - `batch`: `TensorDict` — 存储所有 tensor 数据（input_ids, attention_mask, logprobs 等），batch_size 维度必须为 1
  - `non_tensor_batch`: `dict[str, np.ndarray]` — 存储非 tensor 数据（data_source, reward_model 参数等），dtype 必须为 object
  - `meta_info`: `dict` — 存储元信息（eos_token_id, temperature 等配置参数和 metrics）
- **关键方法**：
  - `from_dict(tensors, non_tensors, meta_info)` → `DataProto` — 从字典创建，自动检查 batch_size 一致性
  - `from_single_dict(data)` → `DataProto` — 从单个 dict 自动分离 tensor 和 non_tensor
  - `from_tensordict(tensor_dict)` → `DataProto` — 从 TensorDict 创建，需要 tensordict≥0.10.0
  - `to_tensordict()` → `TensorDict` — 反向转换，将 non_tensor_batch 转为 NonTensorStack
  - `concat(data_list)` → `DataProto` — 沿 dim=0 拼接多个 DataProto，自动合并 metrics
  - `chunk(chunks)` → `list[DataProto]` — 沿 dim=0 均等切分，不支持不等分（除非启用 auto_padding）
  - `split(split_size)` → `list[DataProto]` — 按指定大小切分
  - `select(batch_keys, non_tensor_batch_keys, meta_info_keys)` → `DataProto` — 键级别子集选择
  - `pop(batch_keys, ...)` → `DataProto` — 弹出指定键并返回
  - `union(other)` → `DataProto` — 合并两个 DataProto，要求同键数据相等
  - `repeat(repeat_times, interleave)` → `DataProto` — 重复数据，支持交错和堆叠模式
  - `make_iterator(mini_batch_size, epochs, seed)` → `Iterator` — 构建 DataLoader 迭代器
  - `reorder(indices)` → `None` — 原地重排序（用于序列长度均衡）
  - `__getitem__(item)` — 支持 int（返回 DataProtoItem）、slice/list/tensor（返回 DataProto）

### DataProtoItem

- **类型**：dataclass
- **字段数**：3 个
- **关键字段**：
  - `batch`: `TensorDict` — 单个样本的 tensor 数据
  - `non_tensor_batch`: `dict` — 单个样本的非 tensor 数据
  - `meta_info`: `dict` — 共享的元信息引用

### DataProtoFuture

- **类型**：dataclass
- **字段数**：3 个
- **关键字段**：
  - `collect_fn`: `Callable` — 将多个 future 结果聚合的函数（通常是 `DataProto.concat`）
  - `futures`: `list[ray.ObjectRef]` — Ray 对象引用列表
  - `dispatch_fn`: `Callable` — 可选的后处理函数
- **关键方法**：
  - `get()` → `DataProto` — 阻塞获取所有 future，执行 collect_fn + dispatch_fn
  - `concat(data)` → `DataProtoFuture` — 从 ObjectRef 列表创建
  - `chunk(chunks)` → `list[DataProtoFuture]` — 延迟切分

### BatchData

- **类型**：class（非 dataclass）
- **字段数**：1 个（`_data`）
- **职责**：统一分发包装器，集中所有类型特定逻辑（isinstance 检查）
- **关键方法**：
  - `is_chunkable()` → `bool` — 检查是否支持 chunk（DataProto, DataProtoFuture, TensorDict, KVBatchMeta, BatchMeta）
  - `is_concatable()` → `bool` — 检查列表是否支持 concat
  - `chunk(chunks)` → `tuple` — 分发到具体类型的 chunk 实现
  - `concat()` → 具体类型 — 分发到具体类型的 concat 实现

### DataProtoConfig

- **类型**：class（元类 `_DataProtoConfigMeta`）
- **职责**：全局配置，控制 auto_padding 行为
- **关键属性**：
  - `auto_padding`: `bool` — 通过类属性或环境变量 `VERL_AUTO_PADDING` 控制

### BaseConfig

- **类型**：dataclass，继承 `collections.abc.Mapping`
- **dataclass 字段数**：1 个（`_target_`）
- **关键字段**：
  - `_mutable_fields`: `set` — 可变字段白名单（类级别变量，不是 dataclass 字段）
  - `_target_`: `str` — Hydra 实例化目标类
- **关键方法**：
  - `__setattr__()` — 冻结非 `_mutable_fields` 中的字段
  - `__getitem__()` / `__iter__()` / `__len__()` — 实现 Mapping 协议

## 4. 算法与逻辑详解

### DataProto 序列化路径

1. **入口** (`protocol.py:377`): `__getstate__()` 在 pickle 序列化时被调用
2. **TensorDict 合并** (`protocol.py:381`): 如果 tensordict≥0.5.0，先 `contiguous().consolidate()` 将所有 tensor 合并到连续内存
3. **序列化策略分支** (`protocol.py:387`):
   - 环境变量 `VERL_DATAPROTO_SERIALIZATION_METHOD=numpy`：使用自定义 numpy 序列化（`serialize_tensordict`），将每个 tensor 转为 (dtype, shape, bytes) 三元组
   - 默认：使用 `torch.save()` 到 BytesIO 缓冲区
4. **反序列化** (`protocol.py:404`): `__setstate__()` 对称恢复

### auto_padding 机制

1. **触发条件** (`protocol.py:840`): `is_padding_enabled()` 检查 per-DataProto 标志或全局 `DataProtoConfig.auto_padding`
2. **Dispatch 层使用** (`decorator.py:91`): `_split_args_kwargs_data_proto_with_auto_padding()` 在将数据分发到 Worker 前自动 padding
3. **padding 逻辑** (`protocol.py:849`): 复制首个或末尾样本填充到整除 chunks 的大小
4. **unpadding** (`single_controller/ray/base.py:49 (def func_generator)`): `func_generator` 在收集结果后去除 padding——`kwargs.pop(_padding_size_key)`（`:53`）+ `select_idxs`（`:61`）裁掉补齐样本。（原文误记为 `decorator.py:53`，该处实为 `class Execute` 文档串，`func_generator` 不在 decorator.py 中）

### 关键常量与阈值

| 常量名 | 值 | 位置 | 用途 |
|--------|---|------|------|
| `_padding_size_key` | `"_padding_size_key_x123d"` | `protocol.py:71` | auto_padding 时在 kwargs 中传递 padding 大小的内部键 |
| `auto_padding_key` | `"_verl_auto_padding"` | `protocol.py:54` | meta_info 中标记当前 DataProto 启用 auto_padding |

## 5. 数据流（输入/输出）

```
┌──────────────┐     ┌──────────────────────┐     ┌──────────────┐
│  数据加载     │────▶│  DataProto            │────▶│  Worker 方法  │
│              │     │                      │     │              │
│ from_single_ │     │ chunk/concat/padding  │     │ 接收：分片的   │
│ dict(batch)  │     │ 由 @register 驱动     │     │ DataProto    │
└──────────────┘     └──────────────────────┘     └──────────────┘
```

### 输入契约

- `batch` 中所有 tensor 的第 0 维大小必须相同
- `non_tensor_batch` 中所有 ndarray 的第 0 维大小必须与 `batch` 一致
- `non_tensor_batch` 的值必须是 `np.ndarray` 类型（dtype=object）
- 仅支持 `num_batch_dims=1`（当 non_tensor_batch 非空时）

### 输出保证

- `concat()` 沿 dim=0 拼接，metrics 以 list-of-dict 形式合并
- `chunk()` 保证每个 chunk 大小相等（除非启用 auto_padding）
- `union()` 保证同键 tensor 完全相等（通过 `tensor.equal()` 验证）

## 6. 关键设计决策与不变量

1. **TensorDict + numpy dict 双轨设计** — 原因：TensorDict 对 tensor 操作高效（batch 索引、设备迁移），但不支持非 tensor 数据（字符串、嵌套对象）。使用 numpy object array 作为补充通道，代价是序列化时需要双路径处理。

2. **BaseConfig 默认冻结** — 原因：训练过程中意外修改配置是常见 bug 来源。通过 `__setattr__` 拦截实现运行时不可变性，需要可变的字段必须显式声明在 `_mutable_fields` 中。

3. **DataProtoFuture 延迟求值** — 原因：驱动进程不应等待数据传输完成。`DataProtoFuture` 允许将 "从 WorkerGroup A 获取数据" 和 "传递给 WorkerGroup B" 链式组合，只在真正需要数据时才 `ray.get()`。

4. **BatchData 集中类型分发** — 原因：`decorator.py` 中的 dispatch/collect 逻辑需要处理多种数据类型（DataProto, TensorDict, BatchMeta 等）。将所有 `isinstance` 检查集中在 `BatchData` 中，避免在每个 dispatch 函数中重复分支。

### 不变量

- `DataProto.batch.batch_size` 始终是 1 维（即 `len(batch_size) == 1`）
- `non_tensor_batch` 中的每个值的 `shape[0]` 等于 `batch.batch_size[0]`
- `union()` 操作后，两个 DataProto 的同名键持有完全相同的数据

## 7. 已知问题与限制

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 1 | `print_size()` 使用 `print()` 而非 `logger` | `protocol.py:452` | 不符合项目日志规范 |
| 2 | `num_batch_dims > 1` 在有 `non_tensor_batch` 时不支持 | `protocol.py:511` | 限制了多维 batch 的使用场景 |
| 3 | `chunk()` 不支持不等分（非 auto_padding 时） | `protocol.py:875` | 要求 batch_size 必须能被 dp_size 整除 |

## 8. 相关测试覆盖

### 直接测试

| 测试文件 | 测试函数数 | 测试焦点 |
|---------|----------|---------|
| `tests/single_controller/test_auto_padding_on_cpu.py` | 1 | `test_auto_padding`：CPU 上 chunk 补齐与 `func_generator` 去填充路径 |

### 间接测试

| 测试文件 | 相关测试函数数 | 覆盖方式 |
|---------|-------------|---------|
| `tests/trainer/` | 多个 | 通过训练循环间接测试 DataProto 的 concat/chunk/union |
| `tests/single_controller/` | 多个 | 通过 Worker dispatch 间接测试 DataProto 分发 |
