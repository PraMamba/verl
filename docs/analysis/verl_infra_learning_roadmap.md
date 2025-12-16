# verl 基础设施（Infrastructure）学习路线图

**面向对象**: 基础设施（Infrastructure）初学者
**学习周期**: 建议 3-4 周（每周 10-15 小时）
**最后更新**: 2025-12-15

---

## 📋 目录

1. [学习路线总览](#学习路线总览)
2. [如何使用本指南](#如何使用本指南)
3. [前置知识要求](#前置知识要求)
4. [学习路径详解](#学习路径详解)
5. [实践环境准备](#实践环境准备)
6. [常见问题与误区](#常见问题与误区)
7. [进阶资源](#进阶资源)

---

## 学习路线总览

本学习路线图旨在帮助 **Infra 小白** 系统掌握 verl 框架的四大核心技能：

| 核心技能 | 学习目标 | 覆盖的 Level |
|---------|---------|-------------|
| **数据流转** | 理解 RL 训练三阶段的数据传递、DP/TP 转换、Ray ObjectRef 异步机制 | Level 2, 4, 5 |
| **内存管理** | 掌握 CPU offload、KV cache 管理、显存优化技巧 | Level 3, 5 |
| **并行通信** | 深入理解 NCCL 通信原语、FSDP/Megatron 切分策略 | Level 1, 4 |
| **生态兼容** | 学会集成 vLLM/SGLang/Megatron、SPMD 改造技巧 | Level 5 |

### 学习路径可视化

```
Level 0: Python 分布式基础 (入门铺垫, 2-3天)
   ↓
Level 1: 并行策略基础 (理解 DP/TP/PP/Zero, 3-4天)
   ↓
Level 2: 数据流转机制 (掌握三阶段数据流, 4-5天)
   ↓
Level 3: 内存管理优化 (学会显存调优, 4-5天)
   ↓
Level 4: 并行通信模式 (深入通信原语, 5-6天)
   ↓
Level 5: 生态兼容集成 (理解框架集成, 5-6天)
```

---

## 如何使用本指南

### 1. 学习方式

**推荐学习流程**（以 Level 1 为例）：

```
Step 1: 阅读概念 (30min)
   ↓
Step 2: 理解原理 (1h, 带纸笔画图)
   ↓
Step 3: 运行 toy example (30min)
   ↓
Step 4: 阅读 verl 源码 (2h, 标注关键行)
   ↓
Step 5: 完成思考题 (1h)
   ↓
Step 6: 自我检测 (30min, 能否不看文档复述)
```

**总计**: 每个 Level 约 5-8 小时

### 2. 文档结构说明

每个 Level 文档包含以下部分：

| 章节 | 内容 | 建议时间 |
|------|------|---------|
| **学习目标** | 本 Level 结束后应达到的具体能力 | 5min 阅读 |
| **核心问题清单** | 3-5 个关键问题，每个问题包含提问目标和深挖细节 | 核心学习部分 |
| **概念验证实验** | 可独立运行的 minimal example | 30-60min |
| **源码阅读指南** | 精确到文件路径和行号的代码导读 | 1-2h |
| **自我检测清单** | 15-20 个判断/简答题 | 30min |
| **进阶挑战** | 可选的深入话题 | 按需学习 |

### 3. 符号约定

- ✅ **必须掌握**: 核心概念，不理解无法继续
- 🔍 **深入理解**: 需要反复阅读代码和文档
- 💡 **重要洞察**: "啊哈时刻"，理解后会豁然开朗
- ⚠️ **常见陷阱**: 初学者容易犯的错误
- 🚀 **性能关键**: 影响训练吞吐的关键点
- 📖 **扩展阅读**: 相关论文或博客

---

## 前置知识要求

### 最低要求（Level 0 会补充）

- [ ] Python 基础（类、装饰器、上下文管理器）
- [ ] PyTorch 基础（Tensor 操作、autograd、nn.Module）
- [ ] 命令行基础（运行 shell 脚本、环境变量）
- [ ] Git 基础（clone、checkout、查看 diff）

### 推荐基础（可加速学习）

- [ ] 多进程/多线程编程经验
- [ ] 分布式训练经验（至少跑过 DDP）
- [ ] CUDA 编程概念（grid、block、kernel）
- [ ] Linux 系统使用经验（ssh、tmux、nvidia-smi）

### 如何自检

运行以下代码，如果能理解每一行，可以直接跳过 Level 0：

```python
import torch
import torch.distributed as dist
import os

# 能解释为什么需要这样初始化吗？
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# 能说出 all-reduce 和 all-gather 的区别吗？
tensor = torch.tensor([rank], device='cuda')
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# 能画出通信拓扑图吗？
print(f"Rank {rank}/{world_size}: {tensor.item()}")
```

---

## 学习路径详解

### Level 0: Python 分布式与 PyTorch 基础 ⭐

**目标**: 补齐分布式训练的前置知识，能独立运行 DDP 脚本

**学习时长**: 2-3 天

**核心内容**:
- `torch.distributed` 基础 API
- NCCL 后端原理
- 环境变量与进程组
- DDP 的 forward/backward 流程

**输出标准**:
- [ ] 能手写一个 2-GPU 的 DDP 训练脚本
- [ ] 能解释 `RANK`、`WORLD_SIZE`、`MASTER_ADDR` 的作用
- [ ] 能画出 all-reduce 的通信拓扑

**文档链接**: [`level0_distributed_pytorch_basics.md`](./level0_distributed_pytorch_basics.md)

---

### Level 1: 并行策略基础 ⭐⭐

**目标**: 理解 DP/TP/PP/Zero 的本质区别，能识别 verl 代码中的并行策略

**学习时长**: 3-4 天

**核心问题**:
1. 为什么 DP 切分输入，TP 切分权重？
2. Zero1/2/3 的通信量差异从何而来？
3. SPMD 与中心化调度的根本差异？

**关键代码**:
- `verl/workers/fsdp_workers.py` (FSDP 实现)
- `verl/workers/megatron_workers.py` (Megatron 实现)

**输出标准**:
- [ ] 能手写 ColumnParallelLinear 和 RowParallelLinear
- [ ] 能解释 Zero3 为何比 Zero2 慢但省显存
- [ ] 能说出 TP=4 时通信量是多少

**文档链接**: [`level1_parallelism_fundamentals.md`](./level1_parallelism_fundamentals.md)

---

### Level 2: 数据流转机制 ⭐⭐⭐

**目标**: 掌握 RL 三阶段数据流，理解 DP↔TP 转换逻辑

**学习时长**: 4-5 天

**核心问题**:
1. stage1→stage2 数据如何从 rollout 的 TP=4 转回 actor 的 DP=8？
2. all-gather 和 reduce-scatter 的调用时机？
3. Ray ObjectRef 如何避免阻塞式传输？

**关键代码**:
- `verl/workers/sharding_manager/fsdp_vllm.py` (FSDPVLLMShardingManager)
- `verl/single_controller/base/decorator.py` (transfer protocols)

**输出标准**:
- [ ] 能画出 TP=4→DP=8 的数据切分示意图
- [ ] 能解释为何 rollout 需要 all-gather 输入
- [ ] 能追踪一条数据在三阶段的完整旅程

**文档链接**: [`level2_data_flow_mechanisms.md`](./level2_data_flow_mechanisms.md)

---

### Level 3: 内存管理策略 ⭐⭐⭐

**目标**: 掌握显存优化技巧，能诊断 OOM 问题

**学习时长**: 4-5 天

**核心问题**:
1. vLLM 的 `gpu_memory_utilization` 如何计算可用显存？
2. `free_cache_engine` 释放的显存去了哪里？
3. remove_padding 为何能提升吞吐而非降低显存？

**关键代码**:
- `verl/third_party/vllm/` (vLLM 魔改版本)
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py` (rollout worker)

**输出标准**:
- [ ] 能计算 7B 模型在 TP=2 时的显存占用
- [ ] 能解释 offload 的时机选择（为何不是全程 offload）
- [ ] 能调整配置让 30B 模型在 8×80GB 机器上跑起来

**文档链接**: [`level3_memory_optimization.md`](./level3_memory_optimization.md)

---

### Level 4: 并行通信模式 ⭐⭐⭐⭐

**目标**: 深入理解 NCCL 通信原语，能优化通信开销

**学习时长**: 5-6 天

**核心问题**:
1. TP group 和 DP group 的通信边界在哪？
2. FSDP 的 all-gather → compute → release 如何避免显存峰值？
3. Megatron 的 PP→DP→TP 三级转换通信量如何计算？

**关键代码**:
- `verl/workers/sharding_manager/megatron_vllm.py` (AllGatherPPModel)
- `verl/utils/ulysses.py` (Sequence Parallelism)

**输出标准**:
- [ ] 能画出 TP=4, PP=2 的通信拓扑图
- [ ] 能解释为何 micro_dp_group 采用 TP<DP 优先级
- [ ] 能计算 100B 模型在 TP=8 时的带宽需求

**文档链接**: [`level4_communication_patterns.md`](./level4_communication_patterns.md)

---

### Level 5: 生态兼容与集成 ⭐⭐⭐⭐⭐

**目标**: 理解 verl 如何集成 vLLM/SGLang/Megatron/Ray

**学习时长**: 5-6 天

**核心问题**:
1. vLLM 的 SPMD 化改造做了哪三件事？
2. `llama_dtensor_weight_loader` 为何需要 hard-code？
3. Ray WorkerGroup 如何管理多个 SPMD 集群？

**关键代码**:
- `verl/third_party/vllm/vllm_v_0_6_3/spmd_gpu_executor.py`
- `verl/single_controller/ray/base.py` (RayWorkerGroup)

**输出标准**:
- [ ] 能解释 vLLM 原生版本为何无法用于训练
- [ ] 能添加一个新模型的 weight_loader
- [ ] 能用 Ray 实现一个简单的 actor-critic 训练循环

**文档链接**: [`level5_ecosystem_integration.md`](./level5_ecosystem_integration.md)

---

## 实践环境准备

### 硬件要求

| 学习阶段 | 最低配置 | 推荐配置 |
|---------|---------|---------|
| Level 0-1 | 1× GPU (16GB+) | 2× GPU (24GB+) |
| Level 2-3 | 2× GPU (24GB+) | 4× GPU (40GB+) |
| Level 4-5 | 4× GPU (40GB+) | 8× GPU (80GB+) |

### 软件环境

```bash
# 1. 克隆 verl 仓库
git clone https://github.com/volcengine/verl.git
cd verl

# 2. 安装 verl (FSDP + vLLM 后端)
pip install -e .[test,vllm]

# 3. 验证安装
python -c "import verl; print(verl.__version__)"
python -c "import torch; print(torch.cuda.device_count())"
```

### 测试数据准备

```bash
# 下载 GSM8K 数据集（用于实验）
mkdir -p $HOME/data/gsm8k
wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl -O $HOME/data/gsm8k/train.jsonl
```

---

## 常见问题与误区

### ❌ 误区 1: 认为 Zero3 = Tensor Parallel

**错误理解**: "Zero3 把模型参数切分了，不就是 TP 吗？"

**正确理解**:
- Zero3 切分是为了**存储**（节省显存），计算前会 all-gather
- TP 切分是为了**计算**（并行矩阵乘），始终只持有 1/N 参数

**验证方法**: 阅读 `fsdp_workers.py` 中的 `full_tensor()` 调用时机

---

### ❌ 误区 2: 以为 DP 就是简单的数据复制

**错误理解**: "DP 不就是每块卡跑不同数据吗？"

**正确理解**:
- DP 的核心是**梯度聚合** (all-reduce)
- 数据切分只是表象，关键是保证各 rank 的模型参数同步

**验证方法**: 追踪一次 `optimizer.step()` 前的 all-reduce 调用

---

### ❌ 误区 3: 盲目追求低显存配置

**错误理解**: "显存用得越少越好！"

**正确理解**:
- 过度 offload 会严重拖慢训练速度
- **权衡**: 显存 vs 速度，Zero3 vs Zero2，batch size vs 通信开销

**验证方法**: 对比 `offload=True` vs `offload=False` 的吞吐量

---

### ❌ 误区 4: 认为 Ray 只是 RPC 框架

**错误理解**: "Ray 就是让 Python 能远程调用函数吧？"

**正确理解**:
- Ray 在 verl 中是**数据流管理器**
- 核心作用：解耦计算流（SPMD）和数据流（Single-Controller）

**验证方法**: 阅读 `@register(dispatch_mode)` 装饰器的实现

---

## 进阶资源

### 论文阅读清单

| 论文 | 相关 Level | 重点章节 |
|------|-----------|---------|
| [HybridFlow (EuroSys'25)](https://arxiv.org/abs/2409.19256) | Level 2, 5 | §3 架构设计, §6 设备映射 |
| [ZeRO (SC'20)](https://arxiv.org/abs/1910.02054) | Level 1, 3 | §3 内存优化 |
| [Megatron-LM (arXiv'19)](https://arxiv.org/abs/1909.08053) | Level 1, 4 | §2 Tensor Parallelism |
| [DeepSpeed Ulysses (arXiv'23)](https://arxiv.org/abs/2309.14509) | Level 4 | §3 All-to-All 通信 |
| [vLLM (SOSP'23)](https://arxiv.org/abs/2309.06180) | Level 3, 5 | §4 PagedAttention |

### 官方文档

- [verl Documentation](https://verl.readthedocs.io/)
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)

### 社区资源

- [verl GitHub Discussions](https://github.com/volcengine/verl/discussions)
- [HybridFlow 论文笔记](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/rlhf/verl)

---

## 学习时间表参考

### 全职学习（每天 8 小时）

| 周 | 周一-周二 | 周三-周四 | 周五-周六 | 周日 |
|----|----------|----------|----------|------|
| Week 1 | Level 0 | Level 1 | Level 2 (上) | 复习 + 实验 |
| Week 2 | Level 2 (下) | Level 3 | Level 4 (上) | 复习 + 实验 |
| Week 3 | Level 4 (下) | Level 5 (上) | Level 5 (下) | 总复习 |

### 业余学习（每天 2 小时）

| 周 | 学习内容 | 累计小时 |
|----|---------|---------|
| Week 1-2 | Level 0 + Level 1 | 28h |
| Week 3-4 | Level 2 | 28h |
| Week 5-6 | Level 3 | 28h |
| Week 7-8 | Level 4 | 28h |
| Week 9-10 | Level 5 | 28h |

---

## 学习成果自检

完成所有 Level 后，你应该能够：

- [ ] **独立配置**: 在 8×A100 机器上配置 70B 模型的 PPO 训练
- [ ] **性能调优**: 通过调整 TP/PP/DP 提升 20%+ 吞吐量
- [ ] **故障排查**: 快速定位 OOM、NCCL hang、性能瓶颈
- [ ] **代码贡献**: 为 verl 添加新模型或优化现有组件
- [ ] **架构设计**: 设计一个支持 multi-turn 的 RL 训练流程

---

## 反馈与改进

如果你在学习过程中遇到问题或有改进建议，欢迎：

1. 提交 GitHub Issue: [verl/issues](https://github.com/volcengine/verl/issues)
2. 参与 Discussions: [verl/discussions](https://github.com/volcengine/verl/discussions)
3. 加入社区 Slack: [verl-project.slack.com](https://join.slack.com/t/verl-project/shared_invite/...)

---

**祝学习顺利！🚀**

*Tips: 建议在学习过程中维护一个个人笔记本（如 Notion/Obsidian），记录你的"啊哈时刻"和踩过的坑。这些经验在未来的工作中会非常宝贵。*
