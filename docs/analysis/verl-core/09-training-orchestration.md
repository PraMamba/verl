# 09 - 训练编排子模块 (verl/trainer/)

> 源码位置: `verl/trainer/` | 26 个文件 | 11,942 行
>
> 版本基线: 2025-06 main 分支

---

## 1. 模块定位

`verl/trainer/` 是整个 verl 框架的"大脑"——训练编排层。它不直接接触 GPU 张量,
而是通过 Ray 单控制器模式编排分布式 Worker 的计算流程,完成 RL 后训练的完整数据流:
从 prompt 生成、奖励计算、优势估计到策略与价值网络更新。

核心职责:
- **入口与配置解析**: Hydra 配置加载、数据集创建、训练参数验证
- **Worker 资源编排**: 创建 Ray 资源池、Worker 组、共驻 Worker
- **10 步训练循环**: 在 driver 进程上编排 rollout-reward-advantage-update 的完整管线
- **算法分发**: 根据 `adv_estimator` 配置分发到 GAE/GRPO/RLOO/REINFORCE++ 等 13+ 种优势估计器
- **离策略修正**: IS 权重计算、拒绝采样、bypass/decoupled 两种模式
- **辅助训练器**: SFT 训练器(单机和 Ray 分布式两种)、蒸馏损失函数

---

## 2. 文件清单与行数

| 文件 | 行数 | 职责 |
|------|------|------|
| `main_ppo.py` | 380 | Hydra 入口, `TaskRunner` 远程类, 数据集创建 |
| `main_ppo_sync.py` | 1,866 | 新版同步训练器 `PPOTrainer`(TransferQueue + ReplayBuffer) |
| `ppo/ray_trainer.py` | 1,771 | `RayPPOTrainer` 主训练循环, `compute_advantage()` |
| `ppo/metric_utils.py` | 702 | 指标计算: 数据指标、吞吐指标、时间指标 |
| `ppo/reward.py` | 167 | 奖励函数加载 `load_reward_manager()`, 奖励提取 `extract_reward()` |
| `ppo/rollout_corr_helper.py` | 1,137 | 离策略修正: IS 权重、拒绝采样、off-policy 诊断 |
| `ppo/utils.py` | 107 | `Role` 枚举(9 种角色)、`WorkerType`、`need_*` 判断 |
| `ppo/core_algos.py` | — | `AdvantageEstimator` 枚举(13 种)、核心 RL 算法实现 |
| `config/algorithm.py` | 669 | `AlgoConfig`, `RolloutCorrectionConfig`(20+ 工厂预设) |
| `config/config.py` | 100 | `CheckpointConfig`, `ProfileConfig`, `BaseModelConfig` |
| `constants_ppo.py` | 66 | Ray runtime env(NCCL/VLLM/Tokenizer 环境变量) |
| `distillation/losses.py` | 394 | 蒸馏损失: forward KL, reverse KL, JSD 等 |
| `sft_trainer.py` | 484 | 单进程 SFT 训练器 |
| `sft_trainer_ray.py` | 415 | 基于 Ray 的分布式 SFT 训练器 |

---

## 3. 入口与配置体系

### 3.1 Hydra 入口: main_ppo.py

`main_ppo.py` 是 PPO 训练的 Hydra 入口点(第 38 行 `@hydra.main`),整体流程:

```
main() --> run_ppo() --> TaskRunner.run() --> RayPPOTrainer
  |            |              |
  v            v              v
auto_set_device  ray.init    创建 Worker
migrate_reward   TaskRunner  创建数据集
                 .remote()   trainer.fit()
```

**注意**: `main_ppo.py` 已标记为 `@deprecated`(第 35 行),计划在 v0.8.0 由 `main_ppo_sync.py` 替代。

### 3.2 TaskRunner: 角色-Worker 映射

`TaskRunner`（第 111 行）是一个 Ray remote 类,负责:

1. **注册角色-Worker 映射**: `add_actor_rollout_worker()`（第 126 行）将 `ActorRolloutRefWorker` 注册到 `Role.ActorRollout` 或 `Role.ActorRolloutRef`
2. **注册 Critic Worker**: `add_critic_worker()`（第 148 行）将 `TrainingWorker` 注册到 `Role.Critic`
3. **初始化资源池**: `init_resource_pool_mgr()`（第 158 行）创建 `global_pool` 和可选的 `reward_pool`、`teacher_pool`
4. **配置验证**: 调用 `validate_config()` 验证必填项
5. **启动训练**: 创建 `RayPPOTrainer` 实例并调用 `trainer.fit()`

### 3.3 配置 Dataclass 体系

```
config/algorithm.py (669行)
├── AlgoConfig           — 主算法配置 (gamma, lam, adv_estimator, ...)
│   ├── KLControlConfig  — KL 控制 (type="fixed"/"adaptive", kl_coef, horizon)
│   ├── FilterGroupsConfig — DAPO 过滤组
│   └── RolloutCorrectionConfig — 离策略修正 (20+ 工厂方法)
│
config/config.py (100行)
├── CheckpointConfig     — 检查点 (save_contents, async_save)
├── ProfileConfig        — 性能分析 (step_start/end, save_path)
├── BaseModelConfig      — 基础模型 (path, lora, trust_remote_code)
└── ModuleConfig         — 外部模块 (path, name)
```

所有配置类继承自 `BaseConfig`（提供 OmegaConf 兼容的 `dict` 式访问接口）。`AlgoConfig` 中的 `adv_estimator` 字段直接控制优势估计算法的选择（第 653 行）。

---

## 4. Role 枚举与 Worker 编排

### 4.1 Role 枚举

`Role` 枚举（`ppo/utils.py` 第 27 行）定义了 9 种角色:

| 枚举值 | 值 | 字符串表示 | 用途 |
|--------|------|-----------|------|
| `Actor` | 0 | `"actor"` | 策略模型 |
| `Rollout` | 1 | `"rollout"` | 推理引擎 |
| `ActorRollout` | 2 | `"actor_rollout"` | Actor+Rollout 共驻 |
| `Critic` | 3 | `"critic"` | 价值模型 |
| `RefPolicy` | 4 | `"ref"` | 参考策略(KL 惩罚) |
| `RewardModel` | 5 | `"rm"` | 奖励模型 |
| `ActorRolloutRef` | 6 | `"actor_rollout_ref"` | Actor+Rollout+Ref 共驻 |
| `Env` | 7 | `"env"` | 环境 |
| `TeacherModel` | 8 | `"teacher"` | 蒸馏教师模型 |

### 4.2 需求判断函数

`ppo/utils.py` 提供四个 `need_*` 函数根据配置动态决定是否需要特定组件:

- `need_reference_policy()`（第 75 行）: 当 `use_kl_in_reward=True` 或 `use_kl_loss=True` 时需要
- `need_critic()`（第 96 行）: 当 `critic.enable=True` 或 `adv_estimator=GAE` 时需要
- `need_reward_model()`（第 89 行）: 当 `reward_model.enable=True` 时需要
- `need_teacher_policy()`（第 82 行）: 当蒸馏配置启用时需要

---

## 5. init_workers(): Worker 组创建

`RayPPOTrainer.init_workers()`（第 775 行）是分布式训练的初始化核心,流程如下:

### 阶段 1: 创建资源池
```python
self.resource_pool_manager.create_resource_pool()  # 第 782 行
```
根据 `n_gpus_per_node * nnodes` 创建 GPU 资源池。

### 阶段 2: 注册 Worker 类到资源池
```python
# Actor+Rollout 注册到 global_pool (第 788-796 行)
actor_rollout_cls = RayClassWithInitArgs(
    cls=self.role_worker_mapping[actor_role],
    config=self.config.actor_rollout_ref, ...)

# Critic 注册到 global_pool (第 801-826 行)
critic_cls = RayClassWithInitArgs(
    cls=self.role_worker_mapping[Role.Critic],
    config=critic_cfg)
```

### 阶段 3: 创建共驻 Worker 组
```python
# 第 861-871 行: 将同一资源池内的多个角色合并为共驻 Worker
worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
wg_dict = self.ray_worker_group_cls(
    resource_pool=resource_pool,
    ray_cls_with_init=worker_dict_cls, ...)
spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
```

`create_colocated_worker_cls` 是混合引擎的关键——它让 Actor、Critic、Rollout 共享同一组 GPU,通过 sleep/wake 机制在训练和推理之间分时复用。

### 阶段 4: 初始化管理器
- `RewardLoopManager`（第 907 行）: 管理奖励模型的 sleep/wake
- `AgentLoopManager`（第 951 行）: 管理 Agent 循环（支持自定义类）
- `LLMServerManager`（第 942 行）: 管理 LLM 推理服务器
- `CheckpointEngineManager`（第 965 行）: 管理权重同步和检查点

---

## 6. fit(): 10 步训练循环

`RayPPOTrainer.fit()`（第 1362 行）是训练循环的核心。以下是每步的完整数据流:

### 步骤 1: 加载检查点 & 初始化
```
_load_checkpoint()         # 第 1386 行
checkpoint_manager.update_weights()  # 第 1387 行: 同步权重到推理副本
_validate()                # 第 1396 行: 训练前验证
```

### 步骤 2: 生成 Rollout（标记 `gen`）
```python
# 第 1467-1477 行
combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
self.checkpoint_manager.sleep_replicas()  # 休眠推理副本以释放显存
```
对于 REMAX 算法,会额外生成一个贪心基线 rollout（第 1450-1458 行）。

### 步骤 3: 计算奖励（标记 `reward`）
```python
# 第 1518-1525 行
if self.use_rm and "rm_scores" not in batch.batch.keys():
    batch_reward = self._compute_reward_colocate(batch)
reward_tensor, reward_extra_infos_dict = extract_reward(batch)
```

### 步骤 4: 离策略修正
根据 `rollout_correction` 配置选择模式:

**Bypass 模式**（第 1533-1540 行）: `old_log_probs = rollout_log_probs`（2 个策略: pi_rollout, pi_theta）
```python
apply_bypass_mode(batch=batch, rollout_corr_config=rollout_corr_config, ...)
```

**Decoupled 模式**（第 1541-1567 行）: 重新计算 `old_log_probs`（3 个策略: pi_rollout, pi_old, pi_theta）
```python
old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
```

### 步骤 5: 计算参考策略 log prob
```python
# 第 1576-1580 行
if self.use_reference_policy:
    ref_log_prob = self._compute_ref_log_prob(batch)
    batch = batch.union(ref_log_prob)
```

### 步骤 6: 计算 Value（Critic 推理）
```python
# 第 1583-1586 行
if self.use_critic:
    values = self._compute_values(batch)
    batch = batch.union(values)
```

### 步骤 7: 计算优势 & 回报
```python
# 第 1625-1633 行
batch = compute_advantage(
    batch,
    adv_estimator=self.config.algorithm.adv_estimator,
    gamma=self.config.algorithm.gamma,
    lam=self.config.algorithm.lam, ...)
```

### 步骤 8: 更新 Critic
```python
# 第 1636-1640 行
if self.use_critic:
    critic_output = self._update_critic(batch)
```

### 步骤 9: 更新 Actor
```python
# 第 1648-1649 行
actor_output = self._update_actor(batch)
```

### 步骤 10: 同步权重 & 检查点
```python
# 第 1674-1675 行
self.checkpoint_manager.update_weights(self.global_steps)
# 第 1663-1671 行: 按频率保存检查点
```

---

## 7. compute_advantage(): 优势估计分发

`compute_advantage()`（第 185 行）是 driver 进程上的轻量级计算,根据 `adv_estimator` 枚举值分发到不同算法:

### 分发逻辑

```python
if adv_estimator == AdvantageEstimator.GAE:
    # 第 218-224 行: 通用优势估计 (需要 Critic)
    advantages, returns = core_algos.compute_gae_advantage_return(...)

elif adv_estimator == AdvantageEstimator.GRPO:
    # 第 233-245 行: 组相对策略优化 (不需要 Critic)
    advantages, returns = core_algos.compute_grpo_outcome_advantage(...)

else:
    # 第 247-279 行: 其他所有估计器通过注册表查找
    adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
    advantages, returns = adv_estimator_fn(**adv_kwargs)
```

### AdvantageEstimator 枚举

`AdvantageEstimator`（`core_algos.py` 第 88 行）定义 13 种内置估计器:

| 估计器 | 值 | 是否需要 Critic |
|--------|------|:---:|
| `GAE` | `"gae"` | 是 |
| `GRPO` | `"grpo"` | 否 |
| `REINFORCE_PLUS_PLUS` | `"reinforce_plus_plus"` | 否 |
| `REINFORCE_PLUS_PLUS_BASELINE` | `"reinforce_plus_plus_baseline"` | 否 |
| `REMAX` | `"remax"` | 否 |
| `RLOO` | `"rloo"` | 否 |
| `OPO` | `"opo"` | 否 |
| `GRPO_PASSK` | `"grpo_passk"` | 否 |
| `GPG` | `"gpg"` | 否 |
| `RLOO_VECTORIZED` | `"rloo_vectorized"` | 否 |
| `GRPO_VECTORIZED` | `"grpo_vectorized"` | 否 |
| `OPTIMAL_TOKEN_BASELINE` | `"optimal_token_baseline"` | 否 |
| `TIR_OPTIMAL_TOKEN_BASELINE` | `"tir_optimal_token_baseline"` | 否 |
| `GDPO` | `"gdpo"` | 否 |

自定义估计器可通过 `@register_adv_est("name")` 装饰器注册（`core_algos.py` 第 116 行）。

---

## 8. Rollout Correction: 离策略修正

### 8.1 问题背景

RL 训练中存在三种离策略来源:
1. **策略不匹配**: Rollout（vLLM BF16）与训练（FSDP FP32）的精度差异
2. **模型更新滞后**: 使用旧检查点的 rollout 数据
3. **通用分布偏移**: 任何数据收集与训练之间的分布差异

### 8.2 两种模式

**Bypass 模式**（`bypass_mode=True`）:
- 2 个策略: pi_rollout = pi_old, pi_theta
- 跳过 `old_log_prob` 重计算,直接使用 rollout log probs
- `loss_type` 控制损失函数: `"ppo_clip"` 或 `"reinforce"`
- 定义于 `rollout_corr_helper.py` 第 1102 行 `apply_bypass_mode()`

**Decoupled 模式**（`bypass_mode=False`）:
- 3 个策略: pi_rollout, pi_old, pi_theta
- 重新计算 `old_log_prob` 作为近端锚点
- IS 权重修正 pi_old 和 pi_rollout 之间的差距
- 定义于 `rollout_corr_helper.py` 第 1006 行 `compute_rollout_correction_and_add_to_batch()`

### 8.3 RolloutCorrectionConfig 工厂预设

`RolloutCorrectionConfig`（`config/algorithm.py` 第 59 行）提供 20+ 个命名工厂方法:

| 类别 | 预设方法 | 关键参数 |
|------|---------|---------|
| Decoupled IS | `decoupled_token_is()`, `decoupled_seq_is()` | `rollout_is="token"/"sequence"` |
| Decoupled RS | `decoupled_geo_rs()`, `decoupled_k3_rs()` | `rollout_rs="seq_mean_k1"/"seq_mean_k3"` |
| Decoupled 组合 | `decoupled_geo_rs_seq_tis()`, `decoupled_k3_rs_token_tis()` | IS + RS |
| Bypass PPO | `bypass_ppo_clip()`, `bypass_ppo_clip_geo_rs()` | `loss_type="ppo_clip"` |
| Bypass PG | `bypass_pg_is()`, `bypass_pg_geo_rs()` | `loss_type="reinforce"` |
| IcePop | `decoupled_token_icepop()`, `bypass_pg_token_icepop()` | 双边阈值 |

### 8.4 核心能力

`rollout_corr_helper.py`（1,137 行）的核心功能:

- **IS 权重计算**: 支持 token 级和 sequence 级两种粒度,含截断和 batch 归一化
- **拒绝采样**: 12 种模式（`token_k1/k2/k3`, `seq_sum/mean/max_k1/k2/k3`）
- **安全边界**: 对数空间计算 + `SAFETY_BOUND=20.0`（第 75 行）防止数值溢出
- **诊断指标**: KL 散度、困惑度 PPL、有效样本量 ESS、chi-squared 散度

---

## 9. 新旧训练器对比

### 9.1 RayPPOTrainer（main_ppo.py, 已标记 deprecated）

- 文件: `ppo/ray_trainer.py`（1,771 行）
- 特点: 同步 batch 模式,所有 prompt 统一生成后再训练
- 数据流: `DataProto` 在 driver 进程上流转
- 入口: `TaskRunner.run()` 创建并调用 `trainer.fit()`

### 9.2 PPOTrainer（main_ppo_sync.py, 新版）

- 文件: `main_ppo_sync.py`（1,866 行）
- 类定义: 第 501 行 `class PPOTrainer`
- 核心差异:
  - **TransferQueue**: 零拷贝数据传输,避免 padding 开销
  - **ReplayBuffer**: 后台轮询 TransferQueue 元数据（第 194 行）
  - **AgentLoopWorkerTQ**: 每个 prompt 独立的 "fire-and-forget" 异步 agent 循环（第 297 行）
  - **动态 n**: 每个 prompt 可设置不同的 `rollout.n`
  - **多输出支持**: 每个 agent loop 可返回多个输出

两者共享核心算法代码: `compute_advantage()`, `apply_kl_penalty()`, `compute_spec_decode_metrics()` 等函数被 `main_ppo_sync.py` 直接从 `ray_trainer.py` 导入（第 83 行）。

---

## 10. 辅助训练器

### 10.1 SFT 训练器

**单进程版**（`sft_trainer.py`, 484 行）: 基于 `TrainingWorker` 直接在本地进程中训练,不依赖 Ray 的 Worker 编排。

**Ray 分布式版**（`sft_trainer_ray.py`, 415 行）: 复用 `RayWorkerGroup` 进行分布式 SFT 训练,使用与 PPO 相同的 Worker 基础设施。

### 10.2 蒸馏损失函数

`distillation/losses.py`（394 行）实现多种知识蒸馏损失:
- Forward KL、Reverse KL、JSD
- 支持 Top-K 近似以降低计算开销
- 通过 `TeacherModel` 角色和 `teacher_pool` 资源池启用

---

## 11. 关键数据流路径

```
┌─────────────────────────────────────────────────────┐
│                   driver 进程 (CPU)                   │
│                                                       │
│  DataProto ──► gen_batch ──► rollout_manager ────┐    │
│                              (generate_sequences) │    │
│  ◄─── gen_batch_output ◄─────────────────────────┘    │
│     │                                                  │
│     ├──► extract_reward() ──► reward_tensor            │
│     ├──► _compute_old_log_prob() ──► old_log_probs     │  RPC 调用
│     ├──► _compute_ref_log_prob() ──► ref_log_prob      │  Worker GPU
│     ├──► _compute_values() ──► values                  │
│     │                                                  │
│     ▼                                                  │
│  compute_advantage() ──► advantages (driver 本地计算)   │
│     │                                                  │
│     ├──► _update_critic() ──► critic_output            │  RPC 调用
│     ├──► _update_actor() ──► actor_output              │  Worker GPU
│     │                                                  │
│     └──► checkpoint_manager.update_weights()           │
└─────────────────────────────────────────────────────┘
```

核心设计原则: **driver 进程绝不触碰 GPU 张量**——优势计算是唯一在 driver 上执行的数值操作,且它仅涉及奖励标量和 response_mask,计算量极小。

---

## 12. 设计约束与扩展点

### 设计约束
1. 当前仅支持混合引擎模式（`hybrid_engine=True`, 第 334 行断言）
2. 所有 Worker 通信必须通过 controller,不允许 Worker 间直接通信
3. 配置新增字段必须提供默认值以保证向后兼容
4. 检查点保存的原子性通过 `latest_checkpointed_iteration.txt` 文件保证

### 扩展点
1. **自定义优势估计器**: `@register_adv_est("name")` 装饰器注册
2. **自定义 AgentLoopManager**: 通过 `agent_loop_manager_class` 配置项加载
3. **自定义 CheckpointEngineManager**: 通过 `checkpoint_manager_class` 配置项加载
4. **自定义奖励函数**: 通过 `custom_reward_function.path` 配置项加载外部模块
5. **自定义 TaskRunner**: `run_ppo()` 的 `task_runner_class` 参数允许替换默认的 `TaskRunner`
