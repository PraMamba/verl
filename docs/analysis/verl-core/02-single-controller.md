# verl 单控制器子模块架构文档

> 源码路径: `verl/single_controller/`  
> 总计: 6 个源文件, 2,225 行代码  
> 最后验证: 2026-06-13 (通过 `wc -l` 确认行数, `grep -n` 确认行号)

---

## 1. 概述

`single_controller` 是 verl 分布式架构的核心基础设施层。它实现了 **"单控制器"模式**: 一个 CPU 侧的控制器 (Trainer) 通过声明式装饰器驱动多个分布式 GPU Worker, 控制器本身不触碰 GPU 张量, 从而实现关注点分离。

该子模块解决的核心问题:

1. **方法分发**: 控制器调用一个 Worker 方法时, 如何自动将输入数据分片到多个 Worker, 并在完成后收集结果
2. **资源编排**: 如何在 Ray 集群中按节点拓扑分配 GPU, 创建 placement group, 并实例化 Worker actor
3. **混合引擎**: 如何将训练 Worker (FSDP/Megatron) 和推理 Worker (vLLM/SGLang) 合并到同一组 GPU 上, 实现时间分片

架构分为两层:

- **base 层** (`base/`): 定义抽象接口 --- `Worker` 基类, `WorkerGroup` 抽象, `@register` 装饰器, `Dispatch`/`Execute` 枚举
- **ray 层** (`ray/`): Ray 具体实现 --- `RayWorkerGroup`, `RayResourcePool`, `create_colocated_worker_cls` 等

---

## 2. 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `base/__init__.py` | 18 | 导出 `Worker`, `WorkerGroup`, `ClassWithInitArgs`, `ResourcePool` |
| `base/decorator.py` | 444 | `Dispatch`/`Execute` 枚举, `@register` 装饰器, 8 种分发/收集函数 |
| `base/worker.py` | 348 | `Worker` 基类, `DistRankInfo`, `DistGlobalInfo`, `WorkerHelper` |
| `base/worker_group.py` | 255 | `ResourcePool`, `ClassWithInitArgs`, `WorkerGroup` 抽象类 |
| `ray/__init__.py` | 33 | 导出 Ray 实现: `RayWorkerGroup`, `RayResourcePool`, `create_colocated_worker_cls` 等 |
| `ray/base.py` | 1,127 | Ray 具体实现: 资源池管理, Worker 编排, FusedWorker 融合机制 |

---

## 3. 核心数据结构

### 3.1 Dispatch 枚举 (decorator.py 第 26 行)

`Dispatch` 继承自 `DynamicEnum`, 支持运行时动态注册新模式。预定义了 8 种分发模式 (第 38-47 行):

```python
RANK_ZERO         # 值=0  仅 rank 0 执行
ONE_TO_ALL        # 值=1  将同一输入广播到所有 Worker
ALL_TO_ALL        # 值=2  直接透传, 不做分片
DP_COMPUTE        # 值=3  输入已按 world_size 切好的 list
DP_COMPUTE_PROTO  # 值=4  自动将 DataProto 按 world_size 切分 (带 auto-padding)
DP_COMPUTE_PROTO_WITH_FUNC  # 值=5  第一个参数是函数, 其余按 DP_COMPUTE_PROTO 切分
DP_COMPUTE_METRIC # 值=6  分发同 DP_COMPUTE_PROTO, 收集同 DP_COMPUTE (不做 concat)
DIRECT_ROLLOUT_METHOD # 值=7  特殊模式, 禁止直接调用 (用于 vLLM 的 ExternalRayDistributedExecutor)
```

每种模式在 `DISPATCH_MODE_FN_REGISTRY` (第 308 行) 中注册了对应的 `dispatch_fn` 和 `collect_fn`。

### 3.2 Execute 枚举 (decorator.py 第 50 行)

定义执行模式, 预定义 2 种 (第 61-63 行):

```python
ALL         # 值=0  所有 Worker 都执行
RANK_ZERO   # 值=1  仅 rank 0 执行
```

### 3.3 MAGIC_ATTR (decorator.py 第 23 行)

```python
MAGIC_ATTR = "attrs_3141562937"
```

使用一个不太可能冲突的魔数字符串作为属性名, 避免与用户自定义函数属性冲突。`@register` 装饰器将 `dispatch_mode`, `execute_mode`, `blocking` 绑定到该属性上。

### 3.4 DistRankInfo 与 DistGlobalInfo (worker.py 第 35-47 行)

```python
@dataclass
class DistRankInfo:
    tp_rank: int       # 张量并行 rank
    dp_rank: int       # 数据并行 rank
    pp_rank: int       # 流水线并行 rank
    cp_rank: int       # 上下文并行 rank

@dataclass
class DistGlobalInfo:
    tp_size: int
    dp_size: int
    pp_size: int
    cp_size: int
```

这两个 dataclass 为下游 Worker 提供了统一的并行维度描述。

### 3.5 Worker 基类 (worker.py 第 76 行)

`Worker` 继承 `WorkerHelper`, 是所有具体 Worker 类 (ActorWorker, CriticWorker 等) 的基类。核心职责:

- 从环境变量读取分布式信息 (`WORLD_SIZE`, `RANK`, `MASTER_ADDR`, `MASTER_PORT` 等)
- 管理 CUDA 设备可见性 (`_setup_env_cuda_visible_devices`, 第 231 行), 兼容 CUDA/HIP/ROCR 三种环境变量
- 维护 `fused_worker_dict` (第 218 行), 用于混合引擎场景下子 Worker 之间的互相感知
- 维护 `__dispatch_dp_rank` 和 `__collect_dp_rank` 字典 (第 219-220 行), 用于 ND-compute 分发

类属性 `fused_worker_attr_name = "fused_worker_dict"` (第 84 行) 是 FusedWorker 注入共享字典的键名。

### 3.6 ResourcePool (worker_group.py 第 27 行)

管理跨节点的进程和 GPU 分配:

```python
class ResourcePool:
    def __init__(self, process_on_nodes=None, max_colocate_count=10, n_gpus_per_node=8):
        self._store = process_on_nodes  # 每个节点的进程数, 如 [8, 8] 表示两个 8-GPU 节点
        self.max_colocate_count = max_colocate_count  # 同一 GPU 上最多的共置进程数
```

- `world_size` 属性: 返回 `sum(self._store)`
- `local_world_size_list()`: 展开为每个进程的局部 world_size
- `local_rank_list()`: 展开为每个进程的局部 rank

### 3.7 ClassWithInitArgs (worker_group.py 第 76 行)

延迟实例化包装器: 存储类和构造参数, 在 `__call__` 时才真正实例化。用于 Ray 远程 actor 的延迟创建。

### 3.8 WorkerGroup (worker_group.py 第 123 行)

Worker 集合管理基类, 核心属性:

- `_workers: list` --- Worker 实例列表
- `_worker_names: list` --- Worker 名称列表
- `_dispatch_info: dict` --- 缓存的分发信息 (mesh_name -> dp_rank 映射)
- `_collect_info: dict` --- 缓存的收集信息 (mesh_name -> bool 掩码)
- `fused_worker_used: bool` --- 是否使用了 FusedWorker 模式

关键方法: `_bind_worker_method()` (第 185 行) --- 扫描 Worker 类中所有被 `@register` 装饰的方法, 自动绑定到 WorkerGroup 上, 生成 `dispatch -> execute -> collect` 流水线。

### 3.9 RayResourcePool (ray/base.py 第 113 行)

Ray 层的资源池实现, 扩展了 `ResourcePool`:

- `get_placement_groups()`: 创建 Ray placement group, 按 `STRICT_PACK` 策略将进程绑定到同一节点
- `sort_placement_group_by_node_ip()` (第 70 行): 按节点 IP 排序, 确保 RANK 在跨 job 间一致 (用于 FSDP checkpoint resume)
- `max_colocate_count` 控制每个 bundle 的 GPU 分配: 实际分配 `num_gpus = 1 / max_colocate_count`

### 3.10 ResourcePoolManager (ray/base.py 第 185 行)

```python
@dataclass
class ResourcePoolManager:
    resource_pool_spec: dict[str, list[int]]   # 如 {"actor_critic": [8, 8], "rollout": [8, 8]}
    mapping: dict[int, str]                     # role -> resource_pool_name 的映射
    max_colocate_count: int = 3                 # FSDP 默认 3: actor+critic+ref 共置
```

提供 `create_resource_pool()` 和 `_check_resource_available()` 来验证集群是否有足够 GPU。

### 3.11 RayWorkerGroup (ray/base.py 第 418 行)

核心编排类, `WorkerGroup` 的 Ray 实现。初始化路径三选一:

1. `_init_with_resource_pool()` (第 538 行) --- 从完整资源池创建全新 Worker
2. `_init_with_subresource_pool()` (第 583 行) --- 从子资源池创建 Worker
3. `_init_with_detached_workers()` (第 511 行) --- 连接到已有的 detached Worker

执行方法:
- `execute_all_async()` (第 866 行): 并行向所有 Worker 提交远程调用
- `execute_rank_zero_async()` (第 814 行): 仅向 rank 0 提交
- `_execute_remote_single_worker()` (第 782 行): 单 Worker 远程调用, 对 FusedWorker 使用 `_fuw_execute` 代理

---

## 4. 算法详解

### 4.1 @register 装饰器工作原理

`@register` 装饰器 (decorator.py 第 398 行) 接受 4 个参数:

```python
def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, 
             blocking=True, materialize_futures=True):
```

工作流程:

1. 检查 dispatch_mode 和 execute_mode 的合法性
2. 通过 `tqbridge()` 包装函数 (支持 TransferQueue 桥接)
3. 将 `{dispatch_mode, execute_mode, blocking}` 存储到函数的 `MAGIC_ATTR` 属性上
4. 如果原函数是协程, 返回 async 包装; 否则返回同步包装
5. 如果 `materialize_futures=True`, 在执行前将 `DataProtoFuture` 参数调用 `.get()` 物化

### 4.2 方法拦截与分发流水线

当控制器调用 `worker_group.some_method(data)` 时:

1. **方法绑定阶段** (`_bind_worker_method`, worker_group.py 第 185 行): 
   - 扫描 Worker 类的所有方法, 查找带 `MAGIC_ATTR` 的方法
   - 提取 `dispatch_mode`, `execute_mode`, `blocking`
   - 从 `DISPATCH_MODE_FN_REGISTRY` 查找 `dispatch_fn` 和 `collect_fn`
   - 从 `execute_mode` 解析出 `execute_fn` (指向 `execute_all` 或 `execute_rank_zero`)
   - 调用 `func_generator()` 生成绑定函数, 绑定到 WorkerGroup 实例上

2. **运行时调用阶段** (`func_generator`, ray/base.py 第 49 行):
   ```
   调用 dispatch_fn(worker_group, *args, **kwargs)  -> 数据分片
        ↓
   调用 execute_fn(method_name, *split_args, **split_kwargs)  -> Ray 远程执行
        ↓
   如果 blocking=True: ray.get(output)  -> 等待结果
        ↓
   调用 collect_fn(worker_group, output)  -> 结果聚合
        ↓
   如果有 auto-padding: 截断多余的填充数据
   ```

### 4.3 八种分发模式的实现细节

| 模式 | dispatch_fn | collect_fn | 典型用途 |
|------|-------------|------------|----------|
| `ONE_TO_ALL` | 将每个参数复制 `world_size` 份 (第 120 行) | 透传 (第 134 行) | 广播超参, 查询配置 |
| `ALL_TO_ALL` | 透传 (第 130 行) | 透传 (第 134 行) | 每个 Worker 接收所有数据 |
| `DP_COMPUTE` | 验证输入已是 `world_size` 长度的 list (第 148 行) | 原样返回 list (第 159 行) | 用户手动切分好的场景 |
| `DP_COMPUTE_PROTO` | 自动将 DataProto 切分 + auto-padding (第 167 行) | concat 回 DataProto (第 191 行) | 训练/推理的主数据流 |
| `DP_COMPUTE_PROTO_WITH_FUNC` | 第一个参数是函数, 复制到各 Worker; 其余按 DP 切分 (第 180 行) | 同上 | `execute_with_func_generator` |
| `DP_COMPUTE_METRIC` | 同 `DP_COMPUTE_PROTO` (第 326 行) | 同 `DP_COMPUTE` --- 不做 concat (第 326 行) | 收集指标 (每 Worker 返回标量) |
| `DIRECT_ROLLOUT_METHOD` | 抛出 NotImplementedError (第 126 行) | 同上 | vLLM ExternalRayDistributedExecutor 专用 |

Auto-padding 逻辑 (第 91-117 行): 当 DataProto 长度不能被 `world_size` 整除时, 自动填充到可整除, 切分后在 `func_generator` 中截掉多余部分。

### 4.4 ND-Compute: 多维并行分发

对于 TP > 1 的场景, 一个 WorkerGroup 的 world_size 可能大于 DP size。ND-compute (第 202-304 行) 机制处理这种情况:

1. `dispatch_nd_compute()` (第 202 行): 使用 `dp_rank_mapping` 将每个 Worker 的全局 rank 映射到 dp_rank, 按 dp_rank 分发数据
2. `collect_nd_compute()` (第 236 行): 使用 `collect_mask` 只收集每个 DP group 的 rank 0 的输出
3. `dispatch_lazy_compute_data_proto()` (第 266 行): 延迟查询 dp_rank_mapping --- 首次调用时从 Worker 查询并缓存

### 4.5 FusedWorker 机制 (create_colocated_worker_cls_fused)

FusedWorker 将多个独立 Worker 类 (如 ActorWorker + CriticWorker + RefWorker) 融合到同一个 Ray actor 内, 实现 GPU 时间分片。

新版 FusedWorker (第 1035-1127 行) 流程:

1. `create_colocated_worker_raw_cls()` (第 1035 行): 创建 FusedWorker 类
   - 类名格式: `FusedWorker_Actor_Critic` (第 1059 行)
   - 继承 `Worker` 基类
   - `__init__` 中使用 `DISABLE_WORKER_INIT=1` 环境变量绕过子 Worker 的分布式初始化
   - 将每个子 Worker 存入 `self.fused_worker_dict` 并设置为属性
   - 向每个子 Worker 注入 `fused_worker_dict` 引用, 使子 Worker 互相可见

2. `_fuw_execute()` (第 1087 行): 代理方法路由
   - 方法名格式: `{cls_name}_fwmn_{method_name}` (fwmn = "fused worker method name")
   - 解析 cls_name 和 method_name, 分发到对应子 Worker

3. `create_colocated_worker_cls_fused()` (第 1107 行): 包装为 `RayClassWithInitArgs`, 设置 `fused_worker_used=True`

旧版 `create_colocated_worker_cls()` (第 988 行) 使用 monkey-patch 方式将子 Worker 方法绑定到父类, 已标记为 deprecated。

### 4.6 spawn 与 fuse --- WorkerGroup 分裂

`RayWorkerGroup.spawn(prefix_set)` (第 718 行) 从融合 WorkerGroup 分裂出独立的 WorkerGroup:

- **非 FusedWorker 路径**: 通过 `_rebind_actor_methods()` 将带前缀的方法 (如 `actor_update_policy`) 重绑定为无前缀版本 (如 `update_policy`)
- **FusedWorker 路径** (`spawn_fused`, 第 753 行): 对每个 prefix, deepcopy 当前 WorkerGroup, 单独绑定对应子类的方法

`fuse(prefix_set)` (第 770 行) 则是反向操作: 将多个角色的 WorkerGroup 融合为一个, 并绑定所有方法。

---

## 5. 数据流

### 5.1 典型 RL 训练数据流 (DP_COMPUTE_PROTO 模式)

```
Controller (CPU)
    │
    │ 1. worker_group.generate_sequences(data_proto)
    │    data_proto: DataProto (batch_size=1024)
    │
    ▼
dispatch_dp_compute_data_proto()
    │ 2. 检测 auto_padding, 补齐到 world_size 的倍数
    │ 3. data_proto.chunk(chunks=world_size)
    │    -> [DataProto(256), DataProto(256), DataProto(256), DataProto(256)]
    │
    ▼
execute_all_async()
    │ 4. 对每个 Worker: worker.generate_sequences.remote(data_chunk)
    │    -> [ObjectRef, ObjectRef, ObjectRef, ObjectRef]
    │
    ▼
ray.get()  (blocking=True)
    │ 5. 等待所有 Worker 完成
    │    -> [DataProto(256), DataProto(256), DataProto(256), DataProto(256)]
    │
    ▼
collect_dp_compute_data_proto()
    │ 6. DataProto.concat([...])
    │    -> DataProto(1024+padding_size)
    │
    ▼
func_generator 截断
    │ 7. output.select_idxs(indices)[:-padding_count]
    │    -> DataProto(1024)
    │
    ▼
Controller 得到最终结果
```

### 5.2 Worker 初始化流程

```
ResourcePoolManager.create_resource_pool()
    │
    ▼
RayResourcePool.get_placement_groups()
    │ 创建 placement_group, 按 STRICT_PACK 策略
    │ sort_placement_group_by_node_ip() 排序
    │
    ▼
RayWorkerGroup.__init__()
    │
    ├── _init_with_resource_pool()
    │   │ 对每个 placement_group:
    │   │   对每个 local_rank:
    │   │     _create_worker()
    │   │       设置环境变量 (WORLD_SIZE, RANK, MASTER_ADDR...)
    │   │       ray_cls_with_init(pg, bundle_idx, num_gpus=1/max_colocate_count)
    │   │       -> ray actor handle
    │   │
    │   └── 收集所有 worker handles 到 self._workers
    │
    └── _bind_worker_method(cls, func_generator)
        遍历 Worker 类的 @register 方法
        生成 dispatch -> execute -> collect 流水线
        setattr 到 self
```

### 5.3 FusedWorker 数据流

```
Controller
    │
    │ wg_dict = colocate_wg.spawn({"actor", "critic", "ref"})
    │ actor_wg = wg_dict["actor"]
    │ actor_wg.update_policy(data)
    │
    ▼
RayWorkerGroup._execute_remote_single_worker()
    │ fused_worker_used=True, method 不在 self.method_names 中
    │ -> worker._fuw_execute.remote("actor_fwmn_update_policy", data)
    │
    ▼
FusedWorker._fuw_execute("actor_fwmn_update_policy", data)
    │ 解析: cls_name="actor", method_name="update_policy"
    │ -> self.fused_worker_dict["actor"].update_policy(data)
    │
    ▼
ActorWorker.update_policy(data) 执行
```

---

## 6. 设计决策

### 6.1 为什么使用 DynamicEnum 而不是 Python Enum

`Dispatch` 和 `Execute` 继承自 `DynamicEnum` 而非标准 `enum.Enum`, 因为需要支持运行时动态注册新模式。`register_dispatch_mode()` (第 338 行) 和 `update_dispatch_mode()` (第 348 行) 允许下游代码扩展分发模式, 无需修改核心代码。

### 6.2 为什么不让 Worker 之间直接通信

verl 严格禁止 Worker 之间直接通信 (所有通信必须经过 Controller)。原因:

- 简化状态管理: Controller 是唯一的状态持有者
- 避免分布式死锁: 如果允许 Worker 之间通信, 容易出现循环等待
- 便于调试: 所有数据流都可以在 Controller 端观察

### 6.3 MAGIC_ATTR 的设计意图

使用字符串 `"attrs_3141562937"` (圆周率的数字序列) 而非更常见的 `"_dispatch_attrs"`, 是为了避免与用户定义的 Worker 方法属性冲突。这是一种防御性编程模式。

### 6.4 FusedWorker 新旧两代实现

旧版 `create_colocated_worker_cls()` (第 988 行) 使用 monkey-patch, 将子 Worker 的所有 `@register` 方法以 `{prefix}_{method_name}` 的形式绑定到父类上。缺点是方法名空间污染。

新版 `create_colocated_worker_cls_fused()` (第 1107 行) 使用 `_fuw_execute` 代理: FusedWorker 只有一个入口方法 `_fuw_execute`, 通过解析 `"{cls_name}_fwmn_{method_name}"` 格式的字符串进行路由。优点是方法隔离更干净, 且子 Worker 之间可以通过共享的 `fused_worker_dict` 互相访问。

### 6.5 Placement Group 排序的必要性

`sort_placement_group_by_node_ip()` (第 70 行) 确保在多次 Ray job 之间, 只要节点不变, RANK 分配保持一致。这对 FSDP 分布式 checkpoint 至关重要 --- checkpoint 按 rank 分片存储在本地磁盘, 恢复时必须保证相同 rank 在相同节点上。

### 6.6 num_gpus = 1 / max_colocate_count

在 `_create_worker()` (第 623 行) 中, 每个 Worker 分配的 GPU 资源为 `1 / max_colocate_count`。例如 `max_colocate_count=3` 时, 每个 Worker 声明 0.33 个 GPU, 允许 3 个 Worker (actor + critic + ref) 共享同一块 GPU。这是 Ray 的分数资源分配机制。

---

## 7. 已知问题与局限

### 7.1 DIRECT_ROLLOUT_METHOD 的空实现

`DIRECT_ROLLOUT_METHOD` 模式的 `dispatch_fn` 和 `collect_fn` 都指向 `dummy_direct_rollout_call` (第 126 行), 该函数直接抛出 `NotImplementedError`。这意味着该模式不能通过正常的 dispatch 流水线调用, 只能通过 vLLM 的内部机制直接访问 Worker 方法。在 `_bind_workers_method_to_parent` (旧版, 第 955 行) 中, `DIRECT_ROLLOUT_METHOD` 方法会被绑定到父类时不加前缀。

### 7.2 旧版 create_colocated_worker_cls 未移除

旧版函数 (第 988 行) 已标记 `# deprecated, switching to FusedWorker`, 但仍在代码中保留。下游代码可能仍在使用, 需要最终迁移。

### 7.3 WorkerHelper 中的拼写错误兼容

`get_availale_master_addr_port()` (第 64 行) 是一个拼写错误的旧方法名, 通过 `warnings.warn` 提示用户迁移到 `get_available_master_addr_port()`。

### 7.4 _check_resource_available 的局限

`ResourcePoolManager._check_resource_available()` (ray/base.py 第 226 行) 只检查总 GPU 数量是否足够, 不检查每个节点的 GPU 数量是否满足 `process_on_nodes` 的要求。在异构集群中可能导致调度失败。

### 7.5 execute_all_async 的参数分发启发式

`execute_all_async()` (第 866 行) 使用启发式判断: 如果所有 args 和 kwargs 都是 list 且长度等于 worker 数量, 就自动分发到各 Worker。这在某些边界情况下可能导致非预期行为 --- 例如用户确实想将一个长度恰好等于 worker 数量的 list 广播给所有 Worker 时。

---

## 8. 测试覆盖

测试文件位于 `tests/single_controller/`, 共 23 个文件, 覆盖以下场景:

| 测试文件 | 测试焦点 | 需要 GPU |
|----------|----------|----------|
| `test_decorator_on_cpu.py` | `@register` 装饰器, Dispatch/Execute 枚举 | 否 |
| `base/test_decorator.py` | 装饰器基础功能 | 否 |
| `test_auto_padding_on_cpu.py` | auto-padding 逻辑正确性 | 否 |
| `test_worker_group_basics.py` | WorkerGroup 基本操作 | 是 |
| `test_colocated_workers.py` | 旧版 `create_colocated_worker_cls` | 是 |
| `test_colocated_workers_fused.py` | 新版 FusedWorker | 是 |
| `test_fused_workers_on_cpu.py` | FusedWorker CPU 模式 | 否 |
| `test_split_resource_pool.py` | `split_resource_pool` 功能 | 是 |
| `test_data_transfer.py` | 数据传输正确性 | 是 |
| `test_ray_collectives.py` | Ray 集合操作 | 是 |
| `test_nested_worker.py` | 嵌套 Worker 场景 | 是 |
| `test_high_level_scheduling_api.py` | 高层调度 API | 是 |
| `test_device_mesh_register.py` | Device mesh 注册 | 是 |
| `test_driverfunc_to_worker.py` | Controller 函数到 Worker 的传递 | 是 |
| `test_get_set_dispatch_collect_cpu.py` | dispatch/collect 信息查询 | 否 |
| `test_ray_local_envs_on_cpu.py` | Ray 环境变量设置 | 否 |
| `test_ray_utils_on_cpu.py` | Ray 工具函数 | 否 |
| `test_worker_group_torch.py` | WorkerGroup + PyTorch | 是 |
| `test_rvdz.py` | Rendezvous 机制 | 是 |
| `check_worker_alive/main.py` | Worker 存活检查 | 是 |
| `detached_worker/server.py`, `client.py` | Detached Worker 模式 | 是 |

覆盖不足之处:
- ND-compute (`dispatch_nd_compute`, `collect_nd_compute`) 的直接单元测试较少
- `update_dispatch_mode()` 和 `register_dispatch_mode()` 的动态注册路径缺少专门测试
- SubRayResourcePool 的多次嵌套 split 场景未见显式测试
