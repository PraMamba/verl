# MOPD N-Teacher 扩展挑战复核与整合

## 分析边界

本文将以下两份原始分析合并，并按当前 worktree 源码重新复核：

- `docs/plans/mopd-n-teacher-extension-challenges.md`
  视角：基于 verl 框架，从算法、分布式训练、 Ray 编排、推理引擎四个维度分析 MOPD 挑战。
- `/home/scbjtfy/G-OPD/docs/analysis/n-teacher-extension-challenges.md`
  视角：从 G-OPD 当前 2 教师实现扩展到 N 教师时，代码层面会遇到的困难。

本次结论只以当前 worktree 为准，不复用旧版双教师草案的历史状态。主要证据来源：

- 当前实现综述：`docs/plans/mopd-changes-summary.md`
- 当前验证结果：`docs/plans/mopd-test-results.md`
- 论文笔记：`/home/scbjtfy/Paper_Reading/MiMo-V2-Flash/MOPD.md`
- 论文笔记：`/home/scbjtfy/Paper_Reading/MiMo-V2-Flash/MOPD-1.md`
- 运行时代码：`verl/trainer/ppo/ray_trainer.py`、`verl/trainer/ppo/core_algos.py`
- 配置与 teacher schema：`verl/workers/config/teacher.py`
- teacher worker：`verl/workers/teacher_workers.py`
- ref worker 路径：`verl/workers/fsdp_workers.py`、`verl/workers/megatron_workers.py`、`verl/workers/engine_workers.py`

---

## 一页结论

两份原始分析都抓到了一个真实方向：

**MOPD 的难点从来不只是公式本身，而是 teacher 路由、异构 tokenizer、worker 编排、资源与恢复语义。**

但它们也都带着一个现在已经过时的前提：

**把问题建模成“在旧双教师硬编码实现上继续扩 N”。**

当前分支实际上已经完成了一次架构迁移，真实运行形态是：

**`algorithm.mopd.teachers[]` + 显式 `teacher_id` 数据契约 + trainer-side `teacher_wgs` worker graph**

**+ dual-path teacher signal（`teacher_log_prob` / `teacher_seq_reward`）+ 独立 `mopd` advantage estimator**

因此，很多“必须先做的大改造”在当前分支里已经不是待做项，而是已落地能力。

### 12 个旧挑战在当前分支的状态

| 挑战 | 当前状态 | 结论 |
|---|---|---|
| 1. 模型插槽二值化硬编码 | 已解决 | teacher 已提升为 trainer 维护的独立 worker group，不再塞在 actor/ref worker 槽位里 |
| 2. GPU/CPU 内存线性增长 | 部分缓解 | shared base、quantized backend、micro-batch、resource pool 有帮助，但总体仍随 teacher 数增长 |
| 3. 顺序推理时间瓶颈 | 部分缓解 | 已支持“跨 pool 重叠、同 pool 串行”的调度，但不是 fully parallel |
| 4. 配置语义重载与不可扩展 | 已解决 | 已有显式 `algorithm.mopd` 与 `teachers[]` 配置树 |
| 5. 教师路由硬编码字符串匹配 | 已解决 | 已改为 `teacher_id -> teacher_wg` 的数据驱动路由 |
| 6. Batch Pop/Restore 脆弱模式 | 已解决 | 已改为 sub-batch `select_idxs()` + pad/unpad + scatter |
| 7. 分词器单一假设 | 部分缓解 | token-level 兼容路径 + `sequence_reward` 异构路径已存在，但后者不是 token-level exact KL |
| 8. 全局 lambda 无法适配异构教师 | 已解决 | 已支持 per-teacher `lambda_val`，并构造成 batch 级 tensor |
| 9. 数据管道缺少验证 | 已解决 | 已有 `teacher_id` 提取、preflight 分布检查、unknown/missing teacher fail-fast |
| 10. 优势函数覆盖式计算与浪费 | 已解决 | MOPD 已是独立 estimator，不再在 actor 中覆盖优势 |
| 11. 检查点与恢复缺失 | 部分缓解 | 已有 `mopd_manifest.json`、`checkpoint.complete`、latest-complete resume 选择与 actor-death fallback 证据，但 teacher/base artifacts 仍不自包含 |
| 12. 测试基础设施空白 | 已解决 | 当前已有 config/routing/runtime/manifest/advantage/E2E 多层测试，并已 fresh 跑通单机 4 卡 preflight / GPU E2E / full recipe run / 保守 `18/18` long-run rerun，以及单机 actor-death 恢复演练；但多节点、非 actor-death 故障恢复、async-save restart、重复长程质量收益和 real-weight quantized `sequence_reward` 仍未充分验证 |

### 当前仍然最值得关注的真实挑战

在 2026-03-16 的 single-teacher reduction proof 之后，原先缺的排序变化实验也已经补上：

- reduced single-teacher `mopd` 已经在仓库内补上了 algorithm-level reduction proof，
  并完成了 3-way `data.seed` runtime 对照；后者没有显示稳定单边偏移，但训练轨迹证据仍偏 mixed
- zero-teacher ORM-only reduction 也已经在仓库内补上了 proof surface：
  独立 `mopd_zero_teacher_orm_only` estimator、same-batch exact equivalence tests、
  zero-teacher runtime dependency tests（前提是 `algorithm.mopd.enabled=False`），以及
  2026-03-17 在 GPUs `0-3` 上完成的 fresh paired smoke rerun
- 这轮 zero-teacher fresh smoke rerun 的 mean absolute score delta 为 `0.02734375`，
  final delta 为 `-0.03515625`；结论应写成 supportive smoke evidence，而不是
  sequential stochastic smoke 之外的长程曲线级 closure
- 独立结果记录见 `docs/plans/2026-03-17-mopd-zero-teacher-reduction-results.md`
- `teachers[]` order-permutation paired smoke evidence 也已经在 2026-03-17 补上：
  同一配置只打乱 `algorithm.mopd.teachers[]` 声明顺序，在 GPUs `0-3` 上完成 paired rerun
- 第一轮失败来自非算法性的 teacher 子批平衡约束：原始数据触发 `AssertionError: 5 % 4 != 0`，
  因此 harness 改为准备每个 8-sample batch 都是 `4 cell + 4 disease` 的平衡数据，并关闭 shuffle
- 平衡化 rerun 后，两侧都到达 `training/global_step=4`，并在每一步保持完全一致的
  `0.5 / 0.5` teacher sample fraction、`mopd/is_valid_fraction=1.0`、`mopd/is_zeroed_fraction=0.0`
- 这轮实验最强支持的是结构级顺序不变性：未见 teacher-slot swap / hardcoded slot symptom；
  训练轨迹指标则仍只到 short-horizon sequential stochastic smoke 级别
- 独立结果记录见 `docs/plans/2026-03-17-mopd-teacher-order-invariance-results.md`
- 因此当前最重要的 open challenges 已不再包括
  “单教师退化时算法语义是否已经跑偏”
- 也不再包括“`teachers[]` 顺序是否在隐式决定 teacher 语义，而且仓库里完全没有 paired evidence”
- 剩下真正高优先级的，是多教师 composition、routing、异构 tokenizer bridge、成本和恢复语义

1. teacher 内存与吞吐仍大体随 teacher 数增长，只是增长形态已经 backend-specific。
2. teacher 调度是“per-pool pipeline with limited overlap”，不是全局 fully parallel。
3. 异构 tokenizer 当前只有 sequence-level bridge，没有跨 tokenizer 的 token-level exact reverse KL。
4. checkpoint 仍依赖外部 teacher/base 模型路径；当前虽已有 `checkpoint.complete`、latest-complete resume 选择和 manifest drift detection，但仍不能提供 artifact 级自包含恢复。
5. 生命周期清理主路径已经闭环：`fit()` 现在统一走 `_finalize_fit_resources()`，`cleanup_teacher_workers()`
   也会释放 `base_policy_wg`；历史上的 vLLM 关停噪声已经在 2026-03-16 的最新 full recipe rerun
   `model_training_20260316_10.log` 上消失。若还要补更强证据，重点已变成 post-fix GPU E2E /
   `18/18` rerun 的覆盖面，而不是 teardown 根因未修。
6. 部分 schema 字段仍不是 runtime 能力，例如 `TeacherConfig.weight`、per-teacher `base_model_path`；
   不过它们现在会 fail-fast，而不是静默漂移。

---

## 当前实现的真实架构

### 1. 配置入口已经从“重载旧字段”迁移到 `algorithm.mopd`

当前分支的 MOPD 已经不是通过 `base_model_path` 之类历史字段偷塞 teacher，而是使用：

- `algorithm.mopd.enabled`
- `algorithm.mopd.teachers[]`
- `algorithm.mopd.resource_pools`
- `algorithm.mopd.lambda_val`
- `algorithm.mopd.orm_weight`
- `algorithm.mopd.is_correction`
- `algorithm.mopd.use_base_normalization`
- `algorithm.mopd.base_model_path`

对应文件：

- `verl/trainer/config/ppo_trainer.yaml`
- `verl/trainer/config/algorithm/mopd.yaml`
- `verl/workers/config/teacher.py`

但也要注意两个“schema 已有、能力边界必须写清”的细节：

- `TeacherConfig.weight` 不是当前 runtime 能力；非默认值现在会直接报错，而不是静默忽略
- `TeacherConfig.base_model_path` 不是当前 runtime 能力；ExOPD 仍只消费全局 `algorithm.mopd.base_model_path`

这里还有一个需要修正的旧判断：

- `AlgoConfig.__post_init__` 现在会把 `mopd` 材料化成 `MOPDConfig`
- `validate_config()` 也会先实例化 `AlgoConfig`，因此 runtime typed validation 已经接通
- 当前真正残留的是 `AlgoConfig.mopd` 注解仍为 `Any`，所以 Hydra composition-time strictness 还弱于 actor/critic

### 2. teacher 不再是 actor/ref worker 内部插槽，而是 trainer 侧显式 worker graph

`RayPPOTrainer` 当前直接持有：

- `self.teacher_wgs: dict[str, RayWorkerGroup]`
- `self.base_policy_wg: Optional[RayWorkerGroup]`

也就是说：

- 一个 teacher = 一个独立 worker group
- N 个 teacher = `dict[str, RayWorkerGroup]`
- ExOPD 额外引入一个 shared base worker

这已经和旧的“双教师塞进 `ref_log_prob` / `base_ref_log_prob`”思路不是一套架构。

### 3. 当前实际上存在三类不同信号，必须分清

当前 runtime 里同时存在三类信号：

1. `teacher_log_prob`
   兼容 tokenizer teacher 的 token-level log prob
2. `teacher_seq_reward` + `teacher_seq_weight` + `teacher_token_mask`
   异构 tokenizer teacher 的 sequence-level bridge
3. `rollout_log_probs`
   rollout engine 侧返回的 sampled-token logprobs，用于 train/inference mismatch 的 IS correction

如果把这三类信号混写成“teacher log prob”或“引擎 log prob”，文档就会失真。

### 4. MOPD estimator 现在是三路组合，不只是 `A_mopd + alpha * A_orm`

当前 `compute_mopd_advantage()` 组合的是：

- token-level MOPD / ExOPD 项
- optional `sequence_reward` 项
- optional ORM 项
- optional rollout IS correction

更准确的理解应是：

```text
A_final
= weights * (A_token_mopd + seq_reward_weight * A_seq_teacher + orm_weight * A_orm)
```

其中 sequence teacher 样本会通过 `teacher_token_mask` 把 token-level `A_mopd` 清零，而不是两条信号叠加污染。

### 5. 当前有几条必须写清楚的硬运行时约束

这些不是“建议”，而是当前代码已经强制的约束：

- 需要独立 `Role.RefPolicy`，MOPD 不兼容 LoRA ref-in-actor 路径
- `ppo_epochs == 1`
- `teacher_id` 是路由必需字段
- `uid/index` 不仅 ORM mixing 需要，`sequence_reward` 也需要，因为 sequence score 要做组归一化
- `raw_prompt` 是 sequence teacher 的硬契约，因为 teacher 端需要重建 prompt + response
- `sequence_reward` 当前只在 dedicated quantized teacher backend 上实现
- tokenizer mismatch 不会自动 fallback 到 `sequence_reward`，必须由配置显式声明

---

## 对 G-OPD“12 个挑战”的逐项复核

| 原挑战 | 当前判断 | 复核结论 |
|---|---|---|
| 1. 模型插槽二值化硬编码 | 已解决 | 这是旧架构问题。当前 teacher 已经是 trainer-side worker group，不再需要扩成员变量和专用方法 |
| 2. GPU/CPU 内存线性增长 | 部分缓解 | 增长趋势仍在，但要区分 backend：`legacy_ref` 与 shared base 的内存特征不同于 `hf_int8` / `hf_4bit` |
| 3. 顺序推理计算时间瓶颈 | 部分缓解 | 旧文把它写成“严格串行 O(N)”已经过强。当前调度是跨 pool 异步首发、同 pool 串行 drain |
| 4. 配置系统语义重载 | 已解决 | 当前已经有显式 `teachers[]`、`resource_pools`、`base_model_path` 语义分层 |
| 5. 教师路由硬编码字符串匹配 | 已解决 | 当前已是 `teacher_id` 驱动的 sub-batch 路由，unknown teacher 直接 fail-fast |
| 6. Batch Pop/Restore 脆弱模式 | 已解决 | 当前 teacher/base 路径都改为子批视图与 scatter，不再原地 pop/restore 全局 batch |
| 7. 分词器单一假设 | 部分缓解 | 旧分析说到了核心难点，但今天的真实解法是“compatible token path + sequence_reward bridge” |
| 8. 全局 lambda 无法适配异构教师 | 已解决 | 当前已按 `teacher_id` 生成 per-sample `lambda_val` tensor，ExOPD 广播也已接线 |
| 9. 数据管道缺少验证 | 已解决 | 当前已有 dataset 抽取、preflight teacher 分布统计、unknown/missing teacher 检查 |
| 10. 优势函数覆盖式计算与浪费 | 已解决 | MOPD 已成为独立 advantage estimator，actor 只消费预计算好的 `advantages` |
| 11. 检查点与恢复机制缺失 | 部分缓解 | 当前已有 manifest 与 drift validation，但 teacher/base 权重仍不进入 checkpoint |
| 12. 测试基础设施空白 | 已解决 | 这一条在当前分支已不成立；测试已覆盖 routing、advantage、runtime、manifest、resource pool、E2E |

### 对旧分析里最有价值、也最需要改写的 4 条

#### 挑战 2：内存问题仍然真实，但必须改成 backend-specific 叙述

旧分析把 CPU offload 和 CPU 内存增长写成普遍规律，这对当前分支已经不准确：

- `legacy_ref` teacher 复用的是 `Role.RefPolicy` worker 路径，可能是 FSDP、Megatron 或 engine worker
- `hf_int8` / `hf_4bit` teacher 则是 dedicated `HFQuantizedTeacherWorker`，rank-local 加载完整量化模型，不走 FSDP offload
- ExOPD 当前只有一个 shared base worker，不是每个 teacher 一个 base

因此，真实结论应是：

- 总体成本仍随 teacher 数上升
- 但增长形态取决于 teacher backend
- shared base 减少了“每新增一个 teacher 再新增一个 base”的膨胀

#### 挑战 3：teacher 调度不再是“全串行”，但也远不是“完全并行”

当前 `_compute_teacher_log_probs()` 和 `_compute_teacher_sequence_rewards()` 的真实语义是：

- 先对每个 resource pool 提交一个 async job
- 每个 pool 内只维持一个 in-flight teacher job
- driver 按 pool 顺序消费结果并 scatter 回 batch

所以应写成：

**per-pool pipelining with limited overlap**

而不是：

- 严格全串行
- 所有 teacher fully parallel

#### 挑战 7：异构 tokenizer 已有桥接路径，但不是 token-level exact KL

当前实现已经不是“完全不能支持异构 tokenizer”：

- `compatible` teacher 允许 exact tokenizer path match
- 也允许 `tokenizer_compat_group` + tokenizer metadata signature match
- 真正不兼容的 teacher 可以改走 `sequence_reward`

但 `sequence_reward` 的本质是：

- student 侧 decode response text
- teacher 侧用自己的 tokenizer 和 chat template 重编码 prompt + response
- 计算 response span mean log prob
- 经 group normalization 后广播回 response tokens

所以它是 **sequence-level bridge**，不是跨 tokenizer 的 token-level exact reverse KL。

#### 挑战 11：checkpoint 问题已从“完全缺失”收缩为“非自包含”

当前保存与恢复路径已经包含：

- `mopd_manifest.json`
- semantic drift hard-fail
- deployment drift warning
- 同步 checkpoint 的 `checkpoint.complete`
- `resume_mode=auto` 对 tracker 缺失/不完整时的 latest-complete fallback
- 单机 actor-death 注入后，从 complete checkpoint 恢复到 `training/global_step:2` 的真实日志证据

但仍然没有做到：

- teacher/base 权重随 checkpoint 打包
- path 相同但 artifact 内容改变时的 hash-level 验证

更准确的表述应是：

**checkpoint metadata 语义已经补上，但 artifact self-containment 仍未解决。**

---

## 对 verl“四个维度挑战文档”的复核

### 维度 1：算法层

这份文档方向上最有价值的判断是：

- 算法难点不在 reverse KL 公式本身
- 难点在 teacher signal 如何进入 PPO 主链路
- ORM、IS correction、异构 tokenizer teacher 的组合会带来数据契约复杂度

但需要纠正 6 个点：

1. 当前 estimator 不只是 `A_mopd + orm_weight * A_orm`，而是三路组合：
   token MOPD/ExOPD + optional sequence teacher + optional ORM
2. single-teacher reduction 已不再是“缺少仓库内证明”的开放项：当前已有独立
   `single_teacher_reverse_kl` baseline、paired harness、tensor-level reduction tests，
   以及官方 3-way `data.seed` 结果文档
3. per-teacher `lambda_val` 不是待设计项，已经存在并已参与 ExOPD 广播
4. `uid/index` 不仅 ORM mixing 需要，`teacher_seq_reward` 也需要
5. `teacher_id` 路由与公式选择已经不在 `dp_actor.py`，而是在 trainer 侧完成
6. `ppo_epochs=1` 是当前真实限制，必须写出来

### 维度 2：分布式训练 / FSDP / 内存

这份文档方向上保留下来的核心判断是：

- 多 teacher 的主要成本仍在 worker topology、内存和吞吐
- 内存不会因为“显式 worker graph”而自动消失

但需要纠正 4 个点：

1. `legacy_ref` 不是“标准 FSDP teacher”同义词，它是当前 `Role.RefPolicy` 绑定路径，可能是 FSDP、Megatron 或 engine worker
2. quantized teacher backend 不走 FSDP offload，它是 dedicated HF inference worker
3. ExOPD 当前不是 per-teacher base，而是 shared base
4. 文档里对教师数量上限的具体数字应视为未经当前分支 profiling 验证的经验估计，不能写成确定结论

### 维度 3：Ray 编排 / controller

这份文档最有价值的保留判断是：

- 复杂度确实集中在 trainer/controller 的路由、scatter、worker 生命周期和资源布局上

但要纠正 5 个点：

1. 当前已经有显式 `teacher_wgs`，不需要再发明 `TeacherWorkerGroup`
2. teacher 创建发生在 colocated role 初始化之后，独立于 `Role` 扩展
3. `resource_pool` 不是自动独占 GPU；只有专用 pool + 合适 colocate budget 才会形成真实隔离
4. 当前 manifest/drift validation 已存在，不应再写成“恢复时完全没有 teacher 语义”
5. 生命周期清理主路径已经补齐：
   `fit()` 现在有统一 `try/finally`，`cleanup_teacher_workers()` 也会释放 `base_policy_wg`；而且修复后的
   最新真实 shell rerun `model_training_20260316_10.log` 已不再出现 `resource_tracker` /
   `KeyError` / `Traceback`，所以当前不是主链路清理失效，而是历史噪声已经被新运行覆盖

### 维度 4：推理引擎 / rollout backend

这份文档里保留下来的核心判断是：

- rollout engine 与 teacher scoring 是两套边界
- MOPD 的 teacher 路径并不直接复用 rollout engine 作为 teacher backend

但要纠正 5 个点：

1. `rollout_log_probs` 是 rollout engine 信号，不是 teacher 信号
2. 当前 async rollout 路径下，vLLM、SGLang、TRT-LLM 都可以在 `rollout.calculate_log_probs=true` 时返回 sampled-token logprobs
3. 不应再写“TRT-LLM 不支持 logprob，因此 MOPD IS correction 不可用”这种绝对表述
4. `sequence_reward` 只在 dedicated quantized teacher backend 上实现，不能笼统写成“heterogeneous teacher 已支持”
5. `compatible` 的真实含义是 tokenizer contract 兼容，不是“teacher 必须与 student 共用同一路径 tokenizer”

---

## 合并后的真实挑战清单

下面这些，才是当前分支在 N-teacher 场景下仍然真实存在、而且值得继续投入工程工作的挑战。

### 1. teacher 成本仍然近似随 teacher 数增长，只是增长形态已经 backend-specific

当前已做的缓解包括：

- shared base worker
- quantized teacher backend
- per-teacher micro-batch
- teacher resource pool

但没有做的包括：

- teacher model sharing
- per-pool multi-teacher fused inference
- artifact-aware placement optimization

因此真实说法应是：

**线性膨胀没有被消灭，只是被工程手段局部压低。**

### 2. teacher 调度已不是“全串行”，但可扩展性瓶颈仍在

当前 teacher 调度的瓶颈点是：

- 每个 pool 只维持一个 in-flight job
- driver 仍负责 sub-batch 构造、pad/unpad、scatter
- 默认 recipe 往往仍把 teacher 放在 `global_pool`

因此真实挑战不是“如何从 0 实现并行 teacher worker group”，而是：

**如何把当前的 per-pool limited-overlap 调度继续推进为更强的并发与更低的 driver 开销。**

### 3. 异构 tokenizer 目前只有 sequence-level bridge，没有 token-level exact bridge

当前 `sequence_reward` 路径已经足够让异构 teacher 进入训练，但它有明确边界：

- 只支持 quantized teacher backend
- 需要 `raw_prompt`
- 需要 `uid/index`
- 只能给 sequence-level scalar score，而非 token-level exact reverse KL

因此真正尚未解决的问题是：

**跨 tokenizer 的 token-level dense distillation 仍未实现。**

### 4. ExOPD 当前是 shared-base 模式，不是 per-teacher base 模式

当前 runtime 只认：

- `algorithm.mopd.base_model_path`

而不是：

- `teachers[i].base_model_path`

这意味着：

- 当前实现更适合“多 teacher 相对同一 base anchor”的场景
- 如果未来要支持每个 teacher 各自相对不同 base 做归一化，需要重新设计 worker 生命周期、manifest 语义和公式输入

### 5. checkpoint 仍不是 artifact self-contained

当前分支已经补上的能力：

- manifest 记录 semantic / deployment 配置
- 恢复时检测 drift
- 同步 checkpoint 完整性 marker
- tracker 缺失或失真时回退到 latest complete checkpoint
- 单机 actor-death 故障注入后的恢复闭环证据

当前仍未补上的能力：

- teacher/base 权重随 checkpoint 一起保存
- model path 对应 artifact 的 hash/revision pinning
- 真正的“一键可迁移恢复”

这意味着：

**当前 checkpoint 更像“有自描述元数据的训练状态”，而不是“包含全部 teacher/base 依赖的完整快照”。**

### 6. 生命周期主路径已闭环，默认 full recipe 已验证干净退出

当前最容易被误判的点是：

- `fit()` 现在已经统一走 `_finalize_fit_resources()`，会关闭 progress bar、shutdown dataloader workers、
  async rollout manager、teacher workers，并调用 tracking finish
- `cleanup_teacher_workers()` 已经同时释放 `teacher_wgs` 与 `base_policy_wg`
- `fit()` 末尾也已有统一 `try/finally` 兜底

2026-03-15 的 preflight、GPU E2E、full recipe run 和更保守的 `18/18` long-run rerun 都已成功退出，
这说明 cleanup 主路径已经不是阻塞问题。

在此基础上，2026-03-16 又对修复后的默认 full recipe shell path 做了 fresh rerun，并检查了
`model_training_20260316_10.log`：

- 日志到达 `step:6 ... training/global_step:6`
- 打印 `Final validation metrics`
- 以最终 swanlab footer 收尾
- 全文不再出现 `resource_tracker`、`KeyError`、`process died unexpectedly` 或 `Traceback`

这次长程闭环还补上了之前缺的三条运行侧证据：

- teacher 路由均值 `cell=0.4965`、`disease=0.5035`，与 `100 / 100` 数据分布对齐
- IS mask 保持健康，`mopd/is_valid_fraction` 均值 `0.9999998278`，`mopd/is_zeroed_fraction` 均值 `1.714958e-7`
- 在 `response_length/clip_ratio` 最高到 `1.0` 的高压力下仍完成 `18/18`；至少对这次保守 rollout 配置而言，
  之前看到的 generation OOM / vLLM wake-up OOM 边界没有再次出现

本轮还额外补上了一条故障恢复侧证据：

- 单机 actor-death 注入后，初次 run 以 `ActorDiedError` 退出；强制删除 tracker 后，恢复 run 明确回退到
  `global_step_1` 的 complete checkpoint，并完成到 `training/global_step:2`

因此，当前残留问题更准确地说已不再是“关停噪声仍然存在”，而是：

**默认 full recipe shell path 的 teardown noise 已被真实运行证明消失；若要补更强矩阵证据，剩下的是 post-fix GPU E2E / `18/18` rerun 是否也要重跑。**

### 7. typed config 的 runtime 闭环已补齐，但 composition-time strictness 仍偏弱

当前已经有 `TeacherConfig` / `TeacherResourcePoolConfig` / `MOPDConfig`，而且：

- `AlgoConfig.__post_init__` 会把 `mopd` 材料化成 `MOPDConfig`
- `validate_config()` 也会统一先实例化 `AlgoConfig`

因此 runtime typed validation 已经接通。

当前真正还没做到 actor/critic 同等强度的地方是：

**`AlgoConfig.mopd` 的注解仍为 `Any`，所以 Hydra composition-time strictness 仍然较弱。**

---

## 当前分支已经解决了什么，不能再写成“待实现”

下面这些表述，如果继续出现在文档里，会误导判断：

- “MOPD 还需要从双教师硬编码架构重构成显式多教师 worker graph”
- “还需要引入 per-teacher `lambda_val`”
- “还需要新增 manifest 才能做 resume drift 检查”
- “resume 仍然只能盲信 `latest_checkpointed_iteration.txt`”
- “还没有 teacher routing fail-fast”
- “还没有多教师测试基础设施”
- “异构 tokenizer teacher 完全不能接入”

更准确的写法应该是：

- 多教师 worker graph 已落地
- per-teacher lambda 已落地
- manifest drift detection 已落地
- complete-checkpoint 选择与 tracker-missing fallback 已落地
- teacher preflight fail-fast 已落地
- 测试基础设施已成体系
- 异构 tokenizer teacher 已可通过 `sequence_reward` 接入，但不是 token-level exact KL
- single-teacher reduction proof 已完成，当前不应再把
  “reduced MOPD 是否等于单教师 reverse-KL baseline” 写成待验证项
  但要保留 caveat：当前 runtime 证据并不是“曲线几乎完全重合”的强经验性证明

---

## 最终判断

如果只评价你原来的两份分析是否“完全不合理”，答案是否定的。

它们有两点非常有价值：

1. 都准确抓住了 MOPD 的核心工程复杂度并不在公式推导，而在 teacher 路由、worker 编排、资源与 tokenizer 契约。
2. 都把 N-teacher 的风险集中到了真正关键的几个面：算法信号、分布式资源、Ray controller、推理后端边界。

但如果问它们是否还能直接作为“当前分支的实施难点文档”，答案是否定的。

原因很简单：

**当前分支已经跨过了“从双教师硬编码实现往 N-teacher 迈第一步”的阶段。**

所以整合后的正确结论应是：

- 旧文档中大量挑战已经不是“待解决问题”，而是“已被当前架构吸收的问题”
- 当前真正剩下的挑战，集中在
  - backend-specific 成本控制
  - per-pool teacher 调度扩展性
  - 异构 tokenizer 的 token-level bridge
  - checkpoint self-containment
  - 更广的 post-fix runtime 验证覆盖
  - composition-time typed strictness

这才是对当前 worktree 最贴近事实、也最适合指导后续工作的 MOPD N-teacher 挑战清单。

换句话说，下一阶段若继续做 correctness proof，优先级已经变成：

1. 2-teacher composition / routing correctness
2. heterogeneous tokenizer teacher 的 bridge 质量与稳定性
3. N-teacher 成本、placement 和恢复矩阵

而不是回头继续质疑 single-teacher reduced MOPD 的基本语义。
