# verl 技能指南 (Skills) 使用指南

> 本文档详细介绍 `verl/.claude/skills/` 下所有技能的用途、步骤和最佳实践。
> Skills 提供步骤化的操作引导，像"配方"一样引导你完成重复性开发任务。

## 目录

- [Skills 工作机制](#skills-工作机制)
- [add-reward — 添加 Reward 函数](#add-reward--添加-reward-函数)
- [add-dataset — 添加 Dataset Loader](#add-dataset--添加-dataset-loader)
- [add-worker — 添加 Ray Worker](#add-worker--添加-ray-worker)
- [add-unit-tests — 添加单元测试](#add-unit-tests--添加单元测试)
- [debug-distributed — 调试分布式训练](#debug-distributed--调试分布式训练)
- [Skills 组合使用](#skills-组合使用)
- [提示词模板](#提示词模板)

---

## Skills 工作机制

### 什么是 Skill

Skill 是经过验证的步骤化操作流程，将常见开发任务编码为可重复的"配方"。当 AI 检测到你的意图匹配某个 Skill 时，会自动参考对应流程来引导你。

### 触发方式

Skills 通过以下方式触发：

1. **自然语言匹配**：你说"我要添加一个 reward 函数"，AI 自动参考 `add-reward` skill
2. **斜杠命令**：直接输入 `/add-unit-tests`
3. **规划阶段引用**：planner agent 在规划中引用相关 skill

### 与其他组件的关系

```
你的意图："添加一个 math accuracy reward"
    ↓
planner (自动激活)
    ↓ 制定计划时引用 add-reward skill
add-reward skill
    ↓ 引导你完成 5 个步骤
    ↓ 步骤 4: 添加测试 → 引用 add-unit-tests skill
    ↓ 代码完成后
code-verifier (自动激活)
    ↓ 跑 lint、test
simple-code-reviewer (自动激活)
    ↓ 检查代码质量
/gen-commit-msg → /create-pr → /review-pr
```

---

## add-reward — 添加 Reward 函数

> **预计时间：** ~20 分钟 | **难度：** ★★☆

### 适用场景

- 需要为 RL 训练添加新的奖励信号
- 实现自定义评估标准（数学正确性、代码质量、安全性等）
- 组合多个 reward 组件

### 5 步流程

#### 步骤 1：创建 Reward 函数文件（~5 min）

**位置：** `verl/reward/your_reward_name.py`

**函数签名（必须严格遵守）：**

```python
def your_reward_fn(
    prompt: List[str],
    completions: List[str],
    prompt_ids: torch.Tensor,       # [batch_size, prompt_len]
    completion_ids: torch.Tensor,   # [batch_size, completion_len]
    **kwargs                        # ground_truth, metadata 等
) -> torch.Tensor:                  # [batch_size]
```

**关键约束：**
- 返回 `[batch_size]` 形状的 tensor
- 优雅处理空 completion（返回 0 而非抛异常）
- Reward 值归一化到合理范围（如 `[0, 1]` 或 `[-1, 1]`）

#### 步骤 2：注册 Reward 函数（~2 min）

**文件：** `verl/reward/__init__.py`

```python
from verl.reward.your_reward_name import your_reward_fn
```

#### 步骤 3：创建 Reward 配置（可选，~3 min）

**位置：** `verl/trainer/config/reward/your_reward.yaml`

```yaml
# @package _global_
reward_fn:
  name: your_reward_fn
  kwargs:
    threshold: 0.5
    weight: 1.0
```

#### 步骤 4：添加单元测试（~10 min）

**位置：** `tests/test_your_reward.py`

**必须覆盖的场景：**
- 基本 reward 计算
- 空 completion 处理
- 不同 batch size（1, 4, 16）

#### 步骤 5：集成测试（可选，~5 min）

```bash
python -m verl.trainer.main_ppo \
  reward_fn.name=your_reward_fn \
  trainer.total_epochs=1
```

### 常见 Reward 模式

| 模式 | 描述 | 使用场景 |
|------|------|---------|
| **单组件 Reward** | 单一评估维度 | 简单任务（如数学正确性） |
| **多组件 Reward** | 加权组合多个维度 | 复杂任务（task + style + safety） |
| **外部模型 Reward** | 使用奖励模型打分 | RLHF 场景 |
| **异步 Reward** | 涉及 I/O 的异步计算 | API 调用、文件访问 |

```python
# 多组件 Reward 示例
def multi_reward_fn(prompt, completions, prompt_ids, completion_ids, **kwargs):
    total = (
        0.6 * compute_task_reward(completions) +
        0.3 * compute_style_reward(completions) +
        0.1 * compute_safety_reward(completions)
    )
    return total
```

### 常见问题排查

| 问题 | 原因 | 解决 |
|------|------|------|
| Reward 值过大/过小 | 未归一化 | 归一化到 [0,1] 或 [-1,1] |
| Reward 计算慢 | 未向量化 | 用 torch 操作替代 Python 循环 |
| NaN reward | 除零 / 空输入 | 加 eps=1e-8，处理边界情况 |

---

## add-dataset — 添加 Dataset Loader

> **预计时间：** ~30 分钟 | **难度：** ★★☆

### 适用场景

- 需要加载新格式的训练数据
- 接入新的数据源（JSONL、Parquet、HuggingFace Datasets）
- 支持新的对话格式（多轮、tool calling）

### 5 步流程

#### 步骤 1：创建 Dataset 文件（~10 min）

**位置：** `verl/data/your_dataset_name.py`

**继承 `torch.utils.data.Dataset`：**

```python
class YourDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        ...

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            'prompt': item['prompt'],
            'messages': [
                {'role': 'user', 'content': item['prompt']},
                {'role': 'assistant', 'content': item['response']}
            ],
            'answer': item.get('answer', None)  # RL 场景需要
        }
```

**必要字段：**
- **SFT 场景：** `messages`（对话格式）
- **RL 场景：** `messages` + 可选 `answer`（用于 reward 计算）

#### 步骤 2：注册 Dataset（~2 min）

**文件：** `verl/data/__init__.py`

#### 步骤 3：创建配置（可选，~3 min）

**位置：** `verl/trainer/config/data/your_dataset.yaml`

#### 步骤 4：添加单元测试（~10 min）

**必须覆盖：** 数据加载、item 格式、DataLoader 兼容性

#### 步骤 5：集成测试（可选，~5 min）

### 数据格式支持

| 格式 | 加载方式 |
|------|---------|
| JSONL | `json.loads(line)` 逐行读取 |
| Parquet | `pd.read_parquet(path)` |
| HuggingFace | `load_dataset('name', split='train')` |
| 多轮对话 | 解析 `conversation` 字段为 `messages` 列表 |

### 性能优化

| 问题 | 解决 |
|------|------|
| 加载慢 | 缓存处理后数据到磁盘 |
| 内存不足 | 用 lazy loading 按需加载 |
| 格式错误 | 在 `_load_data()` 中验证并跳过异常数据 |

---

## add-worker — 添加 Ray Worker

> **预计时间：** ~40 分钟 | **难度：** ★★★

### 适用场景

- 需要添加新的计算角色（如 Reward Model Worker、Verifier Worker）
- 需要封装新的推理/训练逻辑为分布式 Worker
- 扩展 verl 的 Worker 架构

### 5 步流程

#### 步骤 1：创建 Worker 文件（~15 min）

**位置：** `verl/workers/your_worker_name.py`

**核心要素：**

```python
@ray.remote(num_gpus=1)
class YourWorker(Worker):
    def __init__(self, config):
        self.config = config
        self._init_model()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def update_config(self, config):
        """广播配置到所有 worker 副本。"""
        self.config = config

    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    def process_batch(self, data_proto: DataProto) -> DataProto:
        """数据并行处理批次。每个 worker 获得数据的一个分片。"""
        results = self._process(data_proto)
        return DataProto.from_dict(results)
```

**关键约束：**
- 必须继承 `verl.single_controller.base.Worker`
- 必须使用 `@ray.remote(num_gpus=N)` 声明资源需求
- 方法必须用 `@register(dispatch_mode=...)` 标注

#### 步骤 2：注册 Worker（~2 min）

**文件：** `verl/workers/__init__.py`

#### 步骤 3：添加 Worker Config（~5 min）

**位置：** `verl/workers/config/your_worker.py`

```python
@dataclass
class YourWorkerConfig:
    model_path: str
    num_gpus: int = 1
    batch_size: int = 32

    def __post_init__(self):
        if self.num_gpus < 1:
            raise ValueError("num_gpus must be >= 1")
```

#### 步骤 4：集成到 Controller（~10 min）

**文件：** `verl/trainer/ppo/ray_trainer.py`

```python
def init_workers(self):
    self.your_workers = [
        YourWorker.remote(self.config.your_worker)
        for _ in range(self.config.num_your_workers)
    ]

def train_step(self, batch):
    data_ref = ray.put(data_proto)
    result_refs = [w.process_batch.remote(data_ref) for w in self.your_workers]
    results = ray.get(result_refs)
```

#### 步骤 5：添加测试（~10 min）

### Dispatch Mode 选择指南

| Mode | 何时使用 | 数据行为 |
|------|---------|---------|
| `ONE_TO_ALL` | 广播配置、同步权重 | 相同数据发给所有 worker |
| `DP_COMPUTE` | 批量训练、批量推理 | 数据按 worker 分片 |
| `MEGATRON_COMPUTE` | 大模型流水线推理 | Megatron 风格并行 |

### Worker 设计 Checklist

- [ ] 继承了 `Worker` 基类
- [ ] 声明了 `@ray.remote(num_gpus=N)`
- [ ] 所有公共方法有 `@register` 装饰器
- [ ] 输入/输出使用 `DataProto`
- [ ] 有对应的 Config dataclass
- [ ] 在 Controller 中集成
- [ ] 有单元测试

---

## add-unit-tests — 添加单元测试

> **预计时间：** ~20 分钟 | **难度：** ★★☆

### 适用场景

- 为新功能添加测试
- 增加测试覆盖率
- 理解 verl 的测试模式

### 7 步流程

#### 步骤 1：确定测试类型

| 测试类型 | 位置 | 运行方式 | GPU |
|---------|------|---------|-----|
| 单元测试 | `tests/unit/` | pytest | 否 |
| 健全性检查 | `tests/special_sanity/` | pytest | 视情况 |
| E2E 测试 | `tests/special_e2e/` | pytest | 是 |
| 分布式测试 | `tests/special_distributed/` | torchrun | 是（多） |
| Worker 测试 | `tests/workers/` | pytest | 视情况 |

#### 步骤 2：创建测试文件

**命名：** `test_<module>_<feature>.py`（或 `test_<module>_on_cpu.py`）

#### 步骤 3：编写测试（Arrange-Act-Assert）

```python
def test_function_under_condition_returns_expected():
    """Test that function returns expected value under condition."""
    # Arrange
    input_data = torch.tensor([1.0, 2.0, 3.0])
    expected = torch.tensor([2.0, 4.0, 6.0])

    # Act
    result = function_under_test(input_data)

    # Assert
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)
```

#### 步骤 4：添加 Pytest Marker

```python
@pytest.mark.slow                                    # > 10 秒
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="...")  # GPU 优雅跳过
@pytest.mark.parametrize("batch_size", [1, 4, 16])    # 参数化
```

#### 步骤 5：Mock 分布式环境

```python
def test_distributed_fn(monkeypatch):
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    result = distributed_function()
    assert result == expected
```

#### 步骤 6：GPU 测试必须优雅跳过

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_function():
    tensor = torch.tensor([1, 2, 3], device="cuda")
    # ... 测试
    torch.cuda.empty_cache()  # 必须清理！
```

#### 步骤 7：Ray Worker 测试

```python
@pytest.fixture(scope="module")
def ray_cluster():
    if not ray.is_initialized():
        ray.init(num_cpus=4, num_gpus=torch.cuda.device_count())
    yield
    ray.shutdown()

def test_worker(ray_cluster):
    worker = MockWorker.remote()
    result = ray.get(worker.compute.remote())
    assert result == 42
```

### 常见错误清单

| ❌ 错误 | ✅ 正确 |
|--------|--------|
| `assert tensor.equal(other)` | `torch.testing.assert_close(tensor, other)` |
| GPU 测试无 skipif | `@pytest.mark.skipif(not CUDA, ...)` |
| GPU 测试不清理内存 | `torch.cuda.empty_cache()` |
| Mock FSDP/DTensor 内部 | 只 mock 外部接口 |
| 测试名不描述性 | `test_<what>_<condition>_<expected>` |
| 无 docstring | 每个测试函数加描述 |
| 使用全局 process group | 显式传 group |

### 运行测试

```bash
# 单元测试
pytest tests/unit/ -v

# 指定文件
pytest tests/unit/test_dataproto.py -v

# 带超时
pytest tests/unit/ -v --timeout=60

# 分布式测试
torchrun --nproc_per_node=2 tests/special_distributed/run_test.py
```

---

## debug-distributed — 调试分布式训练

> **预计时间：** 视问题而定 | **难度：** ★★★★

### 适用场景

- 训练 hang 住 / 死锁
- 数值结果不正确
- Out of Memory (OOM)
- NCCL 通信错误

### 问题诊断矩阵

| 症状 | 最可能原因 | 首选排查方向 |
|------|-----------|------------|
| 训练停止，无错误，GPU 利用率 0% | Collective 不匹配 | 添加 rank 日志检查哪个 collective 没对齐 |
| Loss 变 NaN/Inf | 数值不稳定 / Reduction 错误 | 检查 reduction op 和 DTensor placement |
| CUDA OOM | 模型太大 / batch 太大 | 开启 activation checkpointing → CPU offload |
| NCCL 超时 | 网络问题 / 版本不一致 | NCCL_DEBUG=INFO 查日志 |

### 四类问题详解

#### 1. 训练 Hang / 死锁

**根因定位三板斧：**

```bash
# 1. 启用调试模式
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

# 2. py-spy 抓调用栈（另一个终端）
py-spy dump --pid <process_id>

# 3. 缩短超时（快速暴露问题）
# dist.init_process_group(backend='nccl', timeout=timedelta(seconds=30))
```

**最常见原因：Collective 不匹配**

```python
# ❌ 不同 rank 走了不同分支 → 死锁
if rank == 0:
    dist.all_reduce(tensor, group=pg)

# ✅ 所有 rank 必须参与
if rank == 0:
    dist.all_reduce(tensor, group=pg)
else:
    dist.all_reduce(torch.zeros_like(tensor), group=pg)
```

#### 2. 数值结果不正确

**排查步骤：**

```python
# 打印 reduction 前后的值
print(f"Rank {rank}: Before: {tensor}")
dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=pg)
print(f"Rank {rank}: After: {tensor}")

# 检查 DTensor placement
if isinstance(tensor, DTensor):
    print(f"placements: {tensor.placements}")
    print(f"local_shape: {tensor.to_local().shape}")
    print(f"global_shape: {tensor.shape}")
```

**常见修复：**
- `ReduceOp.SUM` → `ReduceOp.AVG`（梯度平均）
- 混合精度用 `bf16`（比 `fp16` 数值更稳定）

#### 3. OOM

**逐步缓解策略（按优先级）：**

```
① Activation Checkpointing
   enable_activation_offloading(model)
     ↓ 还 OOM？
② CPU Offload 参数
   CPUOffloadPolicy(offload_params=True)
     ↓ 还 OOM？
③ CPU Offload 优化器
   CPUOffloadPolicy(offload_optimizer=True)
     ↓ 还 OOM？
④ 减小 Batch Size / 增大 DP Size
```

**内存监控：**
```python
torch.cuda.reset_peak_memory_stats()
# ... 训练步骤
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory: {peak:.2f} GB")
```

#### 4. NCCL 通信错误

```bash
# 完整调试信息
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口

# 检查版本
python -c "import torch; print(torch.cuda.nccl.version())"

# 如果无 InfiniBand
export NCCL_IB_DISABLE=1
```

### 通用调试工作流

```
步骤 1: 隔离 → 加 barrier 缩小问题区间
步骤 2: 日志 → 添加 rank 条件日志
步骤 3: 验证 → 检查 tensor 形状、process group 成员
步骤 4: 简化 → 单 GPU、小模型、合成数据
```

---

## Skills 组合使用

### 添加完整功能的组合流程

```
add-reward → add-unit-tests → /gen-commit-msg → /create-pr → /review-pr
add-dataset → add-unit-tests → /gen-commit-msg → /create-pr → /review-pr
add-worker → add-unit-tests → /gen-commit-msg → /create-pr → /review-pr
```

### 问题排查的组合流程

```
debug-distributed
    ↓ 定位问题
修复代码
    ↓
add-unit-tests  ← 为修复添加回归测试
    ↓
/create-pr      ← 提交修复
```

---

## 提示词模板

### 添加新 Reward

```
我需要添加一个新的 reward 函数，用于评估 {评估维度}。

具体要求：
- Reward 范围：{范围，如 [0, 1]}
- 输入：{需要什么额外的 kwargs}
- 边界情况：{空 completion 怎么处理}

请按照 add-reward skill 引导我完成。
```

### 添加新 Dataset

```
我需要添加一个新的 dataset loader，用于加载 {数据格式} 格式的数据。

数据结构：
- 文件格式：{JSONL / Parquet / HF Datasets}
- 字段：{字段描述}
- 使用场景：{SFT / RL}

请按照 add-dataset skill 引导我完成。
```

### 添加新 Worker

```
我需要添加一个 {Worker名称}，用于 {具体功能}。

要求：
- GPU 需求：{每个 worker 几张 GPU}
- Dispatch mode：{ONE_TO_ALL / DP_COMPUTE / MEGATRON_COMPUTE}
- 输入数据：{DataProto 中需要哪些字段}
- 输出数据：{返回哪些字段}

请按照 add-worker skill 引导我完成。
```

### 调试分布式问题

```
分布式训练出现了问题：

症状：{hang / OOM / NaN / NCCL error}
环境：
- GPU 数量：{N}
- 节点数：{M}
- 并行配置：DP={dp_size}, TP={tp_size}

错误信息/日志：
```
{粘贴日志}
```

请按照 debug-distributed skill 帮我诊断。
```

### 添加单元测试

```
我需要为 {模块路径} 添加单元测试。

被测功能：{功能描述}
测试类型：{unit / sanity / e2e / distributed}
GPU 需求：{是/否}
需要 Ray：{是/否}

请按照 add-unit-tests skill 引导我编写测试。
```
