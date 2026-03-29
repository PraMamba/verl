# verl 编码规范 (Rules) 使用指南

> 本文档详细介绍 `verl/.claude/rules/` 下所有编码规范的内容、激活条件和核心约束。
> Rules 是**自动加载**的——当你编辑匹配的文件时，对应的规范会自动注入 AI 的上下文。

## 目录

- [Rules 工作机制](#rules-工作机制)
- [code-style.md — 代码风格（始终加载）](#code-stylemd--代码风格始终加载)
- [distributed.md — 分布式训练规范](#distributedmd--分布式训练规范)
- [testing.md — 测试规范](#testingmd--测试规范)
- [api-config.md — 配置规范](#api-configmd--配置规范)
- [Rules 交互关系](#rules-交互关系)
- [提示词模板](#提示词模板)

---

## Rules 工作机制

### 自动加载条件

每个 Rules 文件头部都有 YAML frontmatter 定义匹配规则：

```yaml
---
inclusion: always           # 始终加载
---

---
inclusion: fileMatch
fileMatchPattern: "verl/workers/**|verl/models/**"  # 文件路径匹配时加载
---
```

### 当前匹配规则

| 规范文件 | 加载条件 | 匹配模式 |
|---------|---------|---------|
| `code-style.md` | **始终加载** | `inclusion: always` |
| `distributed.md` | 编辑分布式相关文件 | `verl/workers/**\|verl/models/**\|verl/utils/fsdp_utils/**\|verl/utils/megatron_utils/**` |
| `testing.md` | 编辑测试文件 | `**/tests/**\|*_test.py\|test_*.py` |
| `api-config.md` | 编辑配置文件 | `verl/trainer/config/**\|verl/workers/config/**` |

### 你不需要手动操作

Rules 完全自动化——你修改对应路径的文件时，AI 自动获得相关规范的上下文。你需要做的是：**在对话中提到你要改哪些文件**，AI 就会按规范行事。

---

## code-style.md — 代码风格（始终加载）

> **激活条件：** 始终加载，适用于所有开发场景。

### 三大核心设计原则

#### 1. 组合优于继承 (Composition Over Inheritance)

```python
# ❌ 深层继承
class BaseWorker:
    pass
class DistributedWorker(BaseWorker):
    pass
class FSDPWorker(DistributedWorker):
    pass
class FSDPActorWorker(FSDPWorker):  # 4 层！太深
    pass

# ✅ 浅层继承 + 组合
class FSDPActorWorker(Worker):  # ≤2 层
    def __init__(self):
        self.fsdp_engine = FSDPEngine()  # 组合
        self.model = ActorModel()        # 组合
```

**硬性约束：** 继承层级 ≤ 2 层。

#### 2. Ray 单控制器模式 (Single-Controller Pattern)

```python
# ❌ Worker 之间直接通信
worker_a.send_to(worker_b, data)

# ✅ 所有通信通过控制器
data_ref = ray.put(data)
result_a = ray.get(worker_a.process.remote(data_ref))
result_b = ray.get(worker_b.process.remote(data_ref))
```

**硬性约束：** Worker 不直接通信，由 Controller 编排。

#### 3. DataProto 协议

```python
# ❌ 传裸字典或裸 tensor
worker.process({"ids": tensor, "mask": tensor})

# ✅ 用 DataProto
data = DataProto.from_dict({"ids": tensor, "mask": tensor})
worker.process.remote(data)

# 拼接与拆分
batch = DataProto.concat([proto1, proto2])
shards = batch.split(num_workers)
```

**硬性约束：** Worker 间数据传输必须使用 DataProto。

### 命名约定

| 类型 | 模式 | 示例 |
|------|------|------|
| Worker 类 | `XxxWorker` | `ActorRolloutWorker`, `FSDPCriticWorker` |
| 配置 dataclass | `XxxConfig` | `FSDPEngineConfig`, `RolloutConfig` |
| Trainer 类 | `XxxTrainer` | `RayPPOTrainer` |
| Sharding Manager | `XxxShardingManager` | — |
| Rollout 函数 | `xxx_rollout` | `vllm_rollout`, `sglang_rollout` |
| Reward 函数 | `xxx_reward_fn` | `math_reward_fn` |
| Worker 文件 | `xxx_workers.py` | `fsdp_workers.py` |
| 配置文件 | `config/xxx.yaml` | `config/ppo_trainer.yaml` |

### 日志规范

```python
import logging
import os

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ✅ 正确
logger.info("Step %d: loss=%.4f", step, loss)

# ❌ 错误
print(f"Step {step}: loss={loss}")  # 不要用 print！
```

**级别约定：**
- `DEBUG`：详细诊断信息
- `INFO`：一般信息
- `WARN`：警告（默认级别）
- `ERROR`：错误

### 导入顺序

```python
# 1. 标准库
import os
import logging

# 2. 第三方
import ray
import torch
from omegaconf import DictConfig

# 3. verl（绝对导入）
from verl import DataProto
from verl.single_controller.base import Worker
from verl.utils.fsdp_utils import apply_fsdp2
```

### 性能红线

| 红线 | 说明 |
|------|------|
| 禁止训练循环中 `.item()` / `.cpu()` / `.numpy()` | GPU-CPU 同步开销巨大 |
| 大内存操作后调用 `aggressive_empty_cache()` | 避免 GPU 内存碎片 |
| Ray 大对象用 `ray.put()` + ref | 避免重复序列化 |
| 集合操作必须指定 `group=pg` | 禁止使用全局 process group |

---

## distributed.md — 分布式训练规范

> **激活条件：** 编辑 `verl/workers/**`, `verl/models/**`, `verl/utils/fsdp_utils/**`, `verl/utils/megatron_utils/**` 时加载。

### 核心铁律

#### 铁律 1：永远不创建全局 Process Group

```python
# ❌ 绝对禁止
dist.init_process_group(backend='nccl')  # 全局 group
dist.all_reduce(tensor)                   # 使用全局 group

# ✅ 必须使用显式 group
device_mesh = init_device_mesh('cuda', mesh_shape=(dp_size, tp_size))
dp_group = device_mesh.get_group(mesh_dim=0)
dist.all_reduce(tensor, group=dp_group)
```

#### 铁律 2：集合操作必须显式指定 Process Group

```python
# 每个 collective 都必须有 group= 参数
dist.all_reduce(tensor, group=pg)
dist.all_gather(output, tensor, group=pg)
dist.broadcast(tensor, src=0, group=pg)
```

#### 铁律 3：Worker 架构隔离

每种 Worker 有自己独立的 process group 和 device mesh：
- **Actor Worker**：策略模型
- **Critic Worker**：价值模型
- **Rollout Worker**：推理引擎（vLLM/SGLang）
- **Reference Worker**：参考策略（KL 惩罚）

### FSDP2 模式

#### Device Mesh 创建

```python
from torch.distributed.device_mesh import init_device_mesh

# 2D mesh: DP × TP
device_mesh = init_device_mesh(
    device_type='cuda',
    mesh_shape=(dp_size, tp_size),
    mesh_dim_names=('dp', 'tp')
)
dp_mesh = device_mesh['dp']
tp_mesh = device_mesh['tp']
```

#### DTensor Placement

| Placement | 含义 | 使用场景 |
|-----------|------|---------|
| `Shard(0)` | 沿第 0 维分片 | DP：沿 batch 维 |
| `Shard(1)` | 沿第 1 维分片 | TP：沿 hidden 维 |
| `Replicate()` | 跨 rank 复制 | 广播参数 |
| `Partial()` | 部分张量（需规约） | 梯度规约前 |

### 五大常见陷阱

| # | 陷阱 | 症状 | 解决方案 |
|---|------|------|---------|
| 1 | **Collective 不匹配** | 死锁 | 确保所有 rank 调用相同 collective |
| 2 | **Process Group 错误** | 结果不正确或 hang | 验证 group 匹配目标并行方式 |
| 3 | **Tensor 形状不匹配** | 运行时错误 | collective 前验证所有 rank 形状一致 |
| 4 | **DTensor Placement 错误** | 分片不正确 | DP 用 Shard(0)，TP 用 Shard(1) |
| 5 | **NCCL 错误** | 超时/通信失败 | 设 NCCL_DEBUG=INFO 排查 |

### 调试四步法

```python
# 1. Rank 条件日志
if dist.get_rank() == 0:
    logger.info(f"Step {step}: loss={loss.item()}")

# 2. 同步屏障（隔离问题区间）
dist.barrier(group=pg)
logger.info(f"Rank {dist.get_rank()} passed barrier")

# 3. DTensor 检查
if isinstance(tensor, DTensor):
    logger.info(f"placements: {tensor.placements}, local_shape: {tensor.to_local().shape}")

# 4. Process Group 验证
rank = dist.get_rank(group=pg)
world_size = dist.get_world_size(group=pg)
logger.info(f"PG rank: {rank}/{world_size}")
```

---

## testing.md — 测试规范

> **激活条件：** 编辑 `**/tests/**`, `*_test.py`, `test_*.py` 时加载。

### 测试目录结构

```
tests/
├── unit/              ← 快速单元测试（无需 GPU）
├── integration/       ← 集成测试
├── special_e2e/       ← 端到端测试（需 GPU）
└── special_sanity/    ← 健全性检查
```

### 测试命名

| 类型 | 命名模式 | 示例 |
|------|---------|------|
| 单元测试 | `test_<module>.py` | `test_dataproto.py` |
| 集成测试 | `test_<feature>_integration.py` | `test_ppo_integration.py` |
| E2E 测试 | `test_<algorithm>_e2e.py` | `test_ppo_e2e.py` |

### 测试结构：Arrange-Act-Assert

```python
def test_dataproto_concat():
    # Arrange: 准备测试数据
    proto1 = DataProto.from_dict({'x': torch.tensor([1, 2])})
    proto2 = DataProto.from_dict({'x': torch.tensor([3, 4])})

    # Act: 执行操作
    result = DataProto.concat([proto1, proto2])

    # Assert: 验证结果
    expected = torch.tensor([1, 2, 3, 4])
    torch.testing.assert_close(result['x'], expected)
```

### Pytest Marker

| Marker | 含义 |
|--------|------|
| `@pytest.mark.slow` | 长时间运行（>10s） |
| `@pytest.mark.gpu` | 需要 GPU |
| `@pytest.mark.multi_gpu` | 需要多 GPU |
| `@pytest.mark.ray` | 需要 Ray 集群 |
| `@pytest.mark.vllm` | 需要 vLLM |
| `@pytest.mark.sglang` | 需要 SGLang |
| `@pytest.mark.megatron` | 需要 Megatron-LM |

### 必须遵守的断言规范

```python
# ✅ 用 torch.testing.assert_close（带容差）
torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-8)

# ✅ 整数/布尔精确比较用 torch.equal
assert torch.equal(actual, expected)

# ❌ 不要用裸 assert（无有用错误信息）
assert tensor.equal(other)  # 失败时看不到具体差异
```

### GPU 测试必须优雅跳过

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fsdp_training():
    device = torch.device('cuda')
    model = MyModel().to(device)
    try:
        # 测试代码
        pass
    finally:
        torch.cuda.empty_cache()  # 必须清理！
```

### Ray 测试 Fixture

```python
@pytest.fixture(scope="module")
def ray_cluster():
    if not ray.is_initialized():
        ray.init(num_cpus=4, num_gpus=torch.cuda.device_count())
    yield
    ray.shutdown()

def test_ray_worker(ray_cluster):
    worker = Worker.remote()
    result = ray.get(worker.compute.remote())
    assert result == 42
```

### CI/CD 规则

| 环境 | 运行范围 | 命令 |
|------|---------|------|
| Pre-commit | 仅 CPU 单元测试 | `pytest -m "not slow and not gpu"` |
| CI | 全量（含 GPU） | `pytest -n auto` |

---

## api-config.md — 配置规范

> **激活条件：** 编辑 `verl/trainer/config/**`, `verl/workers/config/**` 时加载。

### 配置目录结构

```
verl/trainer/config/
├── ppo_trainer.yaml       ← 主 PPO 配置
├── grpo_trainer.yaml      ← 主 GRPO 配置
├── actor/                 ← Actor worker 配置
├── critic/                ← Critic worker 配置
├── rollout/               ← Rollout worker 配置
├── reward/                ← Reward 函数配置
└── algorithm/             ← 算法特定配置
```

### 必填字段用 `???`

```yaml
model:
  path: ???           # 必须由用户提供
  hidden_size: 4096   # 有默认值
```

### 配置插值（DRY）

```yaml
model:
  path: /models/llama-7b

actor:
  model_path: ${model.path}     # 引用其他配置值

critic:
  model_path: ${model.path}     # DRY 原则
```

### Dataclass 与 OmegaConf 转换

```python
from verl.utils.config import omega_conf_to_dataclass

engine_config = omega_conf_to_dataclass(
    config.actor.fsdp_config,
    FSDPEngineConfig
)
```

### Dataclass 定义规范

```python
@dataclass
class FSDPEngineConfig:
    """FSDP engine configuration."""

    # 必填字段（无默认值）
    model_path: str

    # 常用可选字段
    param_offload: bool = False
    grad_offload: bool = False

    # 高级可选字段
    mixed_precision: str = "bf16"

    # 内部字段（_ 前缀）
    _device_mesh: Optional[object] = field(default=None, repr=False)

    def __post_init__(self):
        """初始化后验证。"""
        if self.mixed_precision not in ["fp32", "fp16", "bf16"]:
            raise ValueError(f"Invalid mixed_precision: {self.mixed_precision}")
```

### 向后兼容性

```python
# 新字段必须有默认值
@dataclass
class Config:
    existing_field: int
    new_field: bool = False  # ← 向后兼容

# 废弃字段用 warning
def __post_init__(self):
    if hasattr(self, 'old_field_name'):
        warnings.warn("old_field_name is deprecated. Use new_field_name.", DeprecationWarning)
        self.new_field_name = self.old_field_name
```

### 命令行覆盖

```bash
python -m verl.trainer.main_ppo \
  trainer.total_epochs=20 \
  model.path=/path/to/model \
  data.train_batch_size=2048 \
  actor=megatron_actor \
  rollout=sglang_rollout
```

---

## Rules 交互关系

```
code-style.md （始终加载）
    │
    ├──→ distributed.md
    │    ├ 补充：分布式模式细节
    │    ├ 引用：code-style 的 process group 规则
    │    └ 关联文件：verl/utils/fsdp_utils/, verl/utils/megatron_utils/
    │
    ├──→ testing.md
    │    ├ 补充：测试具体模式
    │    ├ 引用：code-style 的命名约定（测试文件命名）
    │    └ 关联文件：.github/workflows/, pyproject.toml
    │
    └──→ api-config.md
         ├ 补充：Hydra 配置细节
         ├ 引用：code-style 的配置命名约定
         └ 关联文件：verl/utils/config.py, verl/workers/config/
```

---

## 提示词模板

### 请求按规范重构代码

```
请按照 verl 的 code-style 规范重构以下代码：

文件：{file_path}

关注点：
- 继承层级是否过深？
- 是否正确使用了 DataProto？
- 日志是否使用了 logging（不是 print）？
- 导入顺序是否规范？
```

### 请求检查分布式代码

```
请按照 distributed.md 规范检查以下分布式代码：

文件：{file_path}

特别关注：
- 是否有使用全局 process group 的情况？
- 集合操作是否都指定了 group=？
- DTensor placement 是否正确？
- 是否存在死锁风险（mismatched collectives）？
```

### 请求编写测试

```
请按照 testing.md 规范为以下模块编写测试：

被测模块：{module_path}
测试文件位置：tests/unit/test_{module_name}.py

要求：
- 遵循 Arrange-Act-Assert 模式
- GPU 测试要有 skipif 装饰器
- tensor 比较用 torch.testing.assert_close
- 添加适当的 pytest marker
```

### 请求设计配置

```
请按照 api-config.md 规范设计以下组件的配置：

组件名称：{component_name}
配置用途：{purpose}

要求：
- 创建 Hydra YAML 配置文件
- 创建对应的 dataclass
- 必填字段用 ???
- 新字段必须有默认值
- 包含 __post_init__ 验证
```
