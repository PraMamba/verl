# verl Infrastructure 学习指南文档汇总

本目录包含一套完整的 verl 基础设施学习指南，共7个文档：

## 📚 文档列表

### 总纲
- **verl_infra_learning_roadmap.md** - 学习路线图总览，包含6个Level的导航

### Level 0: 入门基础
- **level0_distributed_pytorch_basics.md** (1023行)
  - Python 分布式编程
  - torch.distributed API
  - NCCL 通信原语
  - DDP 工作原理

### Level 1: 并行策略
- **level1_parallelism_fundamentals.md** (1127行)  
  - DP vs TP 的本质区别
  - Zero1/2/3 演进逻辑
  - SPMD 编程范式
  - Tensor Parallel 实现

### Level 2-5: 进阶主题
- **level2_data_flow_mechanisms.md** (1013行) - RL三阶段数据流
  - RL 三阶段数据流分析
  - TP↔DP 转换机制
  - Ray ObjectRef 异步模式
- **level3_memory_optimization.md** (1005行) - 显存优化技术
  - vLLM gpu_memory_utilization 计算
  - KV Cache 动态管理
  - CPU Offload 权衡
  - remove_padding 优化
- **level4_communication_patterns.md** (1007行) - 并行通信模式深入
  - NCCL 集合通信原语
  - Process Group 管理
  - FSDP all-gather 流水线
  - Megatron PP→DP→TP 转换
  - Sequence Parallelism
- **level5_ecosystem_integration.md** (1015行) - 生态集成与系统优化
  - vLLM SPMD 化改造
  - 自定义 Weight Loader
  - Ray WorkerGroup 管理
  - Colocate vs Split 模式
  - 新引擎集成指南

## 🎯 当前状态

**已完成** (全部文档):
1. 总纲文档 - 完整的学习路线和导航
2. Level 0 (1023行) - Python 分布式与 PyTorch 基础
3. Level 1 (1127行) - 并行策略基础 (DP/TP/Zero/SPMD)
4. Level 2 (1013行) - RL 三阶段数据流机制
5. Level 3 (1005行) - 显存优化技术
6. Level 4 (1007行) - 并行通信模式深入
7. Level 5 (1015行) - 生态集成与系统优化

**文档特色**:
- 每个 Level 约 1000 行详细内容
- 理论 + 实现（算法原理、数学推导、代码实现）
- 代码阅读任务 + 概念验证实验
- 精确的文件路径和行号引用
- 完整的自我检测清单

## 📖 使用建议

### 对于初学者
1. 从总纲文档开始，了解整体学习路径
2. 按顺序学习 Level 0 → Level 1 → Level 2 → ...
3. 每个Level完成自我检测后再进入下一Level

### 对于开发者
1. 可以直接查阅特定Level的问题清单
2. 使用代码路径快速定位源码
3. 通过实践任务验证理解
4. 所有文档均包含完整的代码示例和理论推导

## 🔗 相关资源

- [verl GitHub](https://github.com/volcengine/verl)
- [verl 官方文档](https://verl.readthedocs.io/)
- [HybridFlow 论文](https://arxiv.org/abs/2409.19256)

---

**创建时间**: 2025-12-15
**面向对象**: Infrastructure 初学者
**维护状态**: 活跃更新中
