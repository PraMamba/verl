# 当前分支 MOPD 实现如何应对 N-Teacher 扩展挑战

## 分析边界

本文只以当前 worktree 的源码、脚本、测试和同仓文档为依据，不复用旧版双教师实现的结论；这里的
“当前”包含 worktree 中尚未提交的跟进改动。

本次交叉核对的主要材料：

- 挑战来源：`/home/scbjtfy/G-OPD/docs/analysis/n-teacher-extension-challenges.md`
- 论文笔记：`/home/scbjtfy/Paper_Reading/MiMo-V2-Flash/MOPD.md`
- 论文笔记：`/home/scbjtfy/Paper_Reading/MiMo-V2-Flash/MOPD-1.md`
- 当前分支源码：`verl/`, `recipe/mopd/`, `tests/`
- 当前分支验证记录：`docs/plans/mopd-test-results.md`

> 说明：文中“trainer 侧多教师 worker graph”是对当前实现的描述性概括，不是代码里的正式命名。

本文要回答三件事：

1. 当前分支为了落地 MOPD，实际上采用了什么新架构。
2. 这套架构如何逐项绕开、缓解或解决旧文档列出的 12 个 N-teacher 挑战。
3. 哪些挑战已经解决，哪些只是部分缓解，哪些还不能算彻底解决。

---

## 结论摘要

当前分支没有沿着旧的“双教师塞进单个 actor/ref worker 内部插槽”路线继续打补丁，而是把教师提升成
trainer/controller 侧的显式 worker graph：

- 配置层：`algorithm.mopd.teachers[]` 成为教师声明入口，`algorithm.mopd` 负责算法级参数，
  见 `verl/trainer/config/ppo_trainer.yaml:41-42`、`verl/trainer/config/algorithm/mopd.yaml:1-51`。
- 数据层：数据集和 recipe 明确传递 `teacher_id`，必要时保留 `raw_prompt` 以支持异构 tokenizer
  教师，见 `verl/utils/dataset/rl_dataset.py:349-375`、`recipe/mopd/prepare_data.py:53-71`。
- 控制层：`RayPPOTrainer` 维护 `self.teacher_wgs: dict[str, RayWorkerGroup]`，并在 ExOPD 模式下额外维护
  `self.base_policy_wg`，见 `verl/trainer/ppo/ray_trainer.py:347-349`、`1462-1535`。
- 执行层：兼容 tokenizer 的教师走 token-level `teacher_log_prob` 路径；异构 tokenizer 的教师走
  `teacher_seq_reward` 路径；两条路径都由 trainer 侧按 `teacher_id` 路由，见
  `verl/trainer/ppo/ray_trainer.py:913-1042`、`1912-2035`。
- 算法层：`compute_mopd_advantage()` 显式组合 token-level MOPD、ExOPD base normalization、
  sequence teacher signal、ORM 和 IS correction，见
  `verl/trainer/ppo/core_algos.py:1012-1193`。

换句话说，当前分支真正落地的是：

**“配置化 teacher 声明 + trainer 侧 teacher worker graph + per-sample teacher 路由 +
dual-path teacher signal（token log prob / sequence reward）+ 独立的 MOPD advantage estimator”**

而不是旧双教师实现的泛化版。

### 12 个挑战的当前状态

| 挑战 | 状态 | 当前结论 |
|---|---|---|
| 1. 模型插槽二值化硬编码 | 已解决 | 教师不再是 actor/ref worker 内部槽位，而是 trainer 维护的独立 worker group |
| 2. GPU/CPU 内存线性增长 | 部分缓解 | 有 shared base、量化 backend、resource pool、micro-batch 缓解，但总体仍随教师数近似线性增长 |
| 3. 顺序推理时间瓶颈 | 部分缓解 | 已支持“跨 pool 重叠、同 pool 串行”的调度；默认 recipe 仍把两个教师放在同一 `global_pool` |
| 4. 配置系统语义重载 | 已解决 | `algorithm.mopd` 成为显式配置树，不再借用 `base_model_path` 之类字段偷塞教师 |
| 5. 教师路由硬编码字符串匹配 | 已解决 | 改为 `teacher_id -> teacher_wg` 的数据驱动路由，未知 teacher fail-fast |
| 6. Batch Pop/Restore 脆弱模式 | 已解决 | 改为 `select_idxs()` 子批 + `pad/unpad` + scatter，不再原地 pop/restore 全局 batch |
| 7. 分词器单一假设 | 部分缓解 | 兼容 tokenizer 可走 token-level；异构 tokenizer 可走 `sequence_reward`，但不是通用 token-level 方案 |
| 8. 全局 lambda 无法适配异构教师 | 已解决 | 支持 per-teacher `lambda_val`，并在 batch 内构造 per-sample lambda tensor |
| 9. 数据管道缺少验证 | 已解决 | 增加 `teacher_id` 提取、preflight 分布统计、unknown/missing teacher fail-fast、recipe 侧 teacher_id 注入 |
| 10. 优势函数覆盖式计算与浪费 | 已解决 | MOPD 成为独立注册 estimator，不再在 actor 侧覆盖已有优势 |
| 11. 检查点与恢复缺失 | 部分缓解 | 有 manifest 持久化和 drift 检测，但 checkpoint 仍不自包含 teacher/base artifacts |
| 12. 测试基础设施空白 | 已解决 | 已有 unit/integration/preflight/GPU E2E 多层覆盖；但长程稳定性、多节点和故障恢复仍未充分验证 |

严格按“彻底解决”的标准看，当前仍不能算完全收口的主要是 **2 / 3 / 7 / 11**。
其余挑战在当前实现里已经不再是旧文档描述的那个问题。

---

## 当前分支的真实架构

### 1. 配置入口已经迁移到 `algorithm.mopd`

MOPD 不是靠重载原有字段拼出来的，而是通过 Hydra defaults 显式挂到 `algorithm.mopd` 下：

- `verl/trainer/config/ppo_trainer.yaml:41-42`
- `verl/trainer/config/algorithm/mopd.yaml:1-51`

`TeacherConfig` / `TeacherResourcePoolConfig` / `MOPDConfig` 定义了教师、教师资源池和 MOPD 算法参数：

- `verl/workers/config/teacher.py:23-77`
- `verl/workers/config/teacher.py:80-105`
- `verl/workers/config/teacher.py:108-162`

当前真正被运行时代码消费的关键字段包括：

- `teachers[].name`
- `teachers[].model_path`
- `teachers[].backend`
- `teachers[].resource_pool`
- `teachers[].log_prob_micro_batch_size`
- `teachers[].lambda_val`
- `teachers[].tokenizer_path`
- `teachers[].tokenizer_compat_group`
- `teachers[].tokenizer_policy`
- `teachers[].seq_reward_weight`
- `algorithm.mopd.lambda_val`
- `algorithm.mopd.orm_weight`
- `algorithm.mopd.is_correction`
- `algorithm.mopd.is_epsilon_low/high`
- `algorithm.mopd.use_base_normalization`
- `algorithm.mopd.base_model_path`
- `algorithm.mopd.resource_pools`

同时也要区分“schema 已声明”和“运行时已真正接线”：

- `TeacherConfig.weight` 目前还只是保留字段，当前主链路没有按 teacher weight 做加权聚合，
  `verl/workers/config/teacher.py:31` 已明确标注 “unused in current impl”。
- `TeacherConfig.base_model_path` 也存在于 schema 中，但当前 ExOPD 实际消费的是全局
  `algorithm.mopd.base_model_path` 这一条 shared base 配置，而不是 per-teacher base。

这里还有一个必须单独点出来的现实细节：

- `TeacherConfig`/`MOPDConfig` 确实存在，并实现了 `__post_init__` 校验，见
  `verl/workers/config/teacher.py:55-77`、`137-162`。
- 但当前 trainer 主路径并没有把 `config.algorithm.mopd` 统一实例化成 `MOPDConfig`；
  `validate_config()` 只显式实例化了 actor/critic，而没有实例化 MOPD 配置，见
  `verl/utils/config.py:149-175`。
- 同时 `AlgoConfig` 自身也没有显式 `mopd` 字段，见 `verl/trainer/config/algorithm.py:567-614`。

这意味着：

- “配置语义重载”已经解决。
- “typed config 全链路统一接线”还没有完全收尾。

### 2. 数据契约是 `teacher_id`，不是旧实现里的隐式教师字符串

当前数据管线里的教师身份信号是显式的 `teacher_id`：

- `RLHFDataset` 读取 `teacher_id_field` 配置，见 `verl/utils/dataset/rl_dataset.py:106-116`
  和 `349-375`。
- `collate_fn()` 会把非 tensor 字段放进 `np.ndarray(dtype=object)`，因此 `teacher_id`
  和 `raw_prompt` 都能进入 `batch.non_tensor_batch`，见 `verl/utils/dataset/rl_dataset.py:40-68`。

recipe 侧也不再假设用户自己手工维护这个字段，而是明确生成它：

- teacher 名称映射：`recipe/mopd/prepare_data.py:53-57`
- 注入 `teacher_id`：`recipe/mopd/prepare_data.py:60-71`
- 打印训练集/测试集 teacher 分布：`recipe/mopd/prepare_data.py:214-229`

当前生产 recipe 会把 `teacher_id_field` 打开：

- `recipe/mopd/run_mopd_qwen3_4b.sh:166`
- `recipe/mopd/check_mopd_first_batch.py:124`

这意味着 trainer 不再需要在热路径里猜“这条样本应该走 math 还是 code”。

### 3. 新的核心不是“多模型插槽”，而是 trainer 侧 teacher worker graph

`RayPPOTrainer` 在初始化时显式持有：

- `self.teacher_wgs: dict[str, RayWorkerGroup]`
- `self.base_policy_wg: Optional[RayWorkerGroup]`

见 `verl/trainer/ppo/ray_trainer.py:347-349`。

教师 worker 的创建不是通过扩展 `Role` 枚举完成的，而是 `init_workers()` 在标准 actor / critic / ref
初始化完之后，再额外循环 `self.config.algorithm.mopd.teachers` 创建：

- shared base worker：`verl/trainer/ppo/ray_trainer.py:1470-1487`
- legacy ref teacher：`verl/trainer/ppo/ray_trainer.py:1493-1507`
- quantized teacher：`verl/trainer/ppo/ray_trainer.py:1508-1514`
- 写入 `self.teacher_wgs[name]`：`verl/trainer/ppo/ray_trainer.py:1529-1542`

这点很关键，因为它说明当前分支的设计不是：

- 在 `Role` 里增加 `Teacher1` / `Teacher2` / `Teacher3`
- 在 `fsdp_workers.py` 里加更多模型成员变量
- 在 actor worker 里加更多 `compute_teacher_k_log_prob()`

而是：

- 一个 teacher = 一个独立 worker group
- 多教师 = `dict[str, RayWorkerGroup]`
- 教师 identity = `teacher.name` + `teacher_id`

### 4. teacher backend 已经分成两条实现路径

当前并不是所有教师都复用一个 ref worker：

#### 4.1 `legacy_ref`

兼容 tokenizer 的教师可以复用现有 ref worker 路径：

- 构造 teacher config：`verl/trainer/ppo/ray_trainer.py:719-740`
- 本质上仍调用 `compute_ref_log_prob`

这个路径的特点是：

- 仍然走 token-level teacher log prob
- 仍要求 student 和 teacher 的 token contract 兼容
- ref worker 使用固定 per-teacher micro-batch，而不是全局 dynamic ref path

#### 4.2 `hf_int8` / `hf_4bit`

量化教师使用专门的 `HFQuantizedTeacherWorker`：

- 类定义：`verl/workers/teacher_workers.py:34-262`
- BitsAndBytes 量化配置：`verl/workers/teacher_workers.py:76-91`
- token-level `compute_ref_log_prob`：`verl/workers/teacher_workers.py:183-210`
- sequence-level `compute_seq_scores`：`verl/workers/teacher_workers.py:212-262`

这个 dedicated worker 的意义是：

- 它不依赖 legacy ref worker 的 FSDP 初始化路径
- 它可以支持量化推理后端
- 它既能算 token-level log prob，也能算 sequence-level score

#### 4.3 shared base worker

ExOPD 不是给每个教师配一个 base，而是用一个 shared base worker：

- config builder：`verl/trainer/ppo/ray_trainer.py:757-766`
- worker 初始化：`verl/trainer/ppo/ray_trainer.py:1470-1487`
- base log prob 计算：`verl/trainer/ppo/ray_trainer.py:798-814`

这降低了旧分析里“每新增一个教师就再多一个 base 模型”的复杂度，但也意味着当前实现假设
**所有教师共享同一个 `algorithm.mopd.base_model_path`**。

### 5. teacher signal 已经分成两条独立数据流

当前分支不是“只有一个 `teacher_log_prob` 张量”的单一路径。

#### 5.1 compatible teacher：token-level log prob

trainer 先按 `teacher_id` 分组构造 sub-batch job：

- `verl/trainer/ppo/ray_trainer.py:913-972`

然后 `_compute_teacher_log_probs()`：

- 只挑 `tokenizer_policy == "compatible"` 的教师，见
  `verl/trainer/ppo/ray_trainer.py:1933-1934`
- 为每个 resource pool 先提交一个异步 job，见
  `verl/trainer/ppo/ray_trainer.py:1944-1954`
- 同一 pool 内串行接续，下一个 job 在上一个 `get()` 之后再发，见
  `verl/trainer/ppo/ray_trainer.py:1956-1967`
- 结果 scatter 回统一的 `[batch, response_len] teacher_log_prob` 张量，见
  `verl/trainer/ppo/ray_trainer.py:1912-1969`

#### 5.2 sequence teacher：raw prompt + decoded response text

异构 tokenizer 教师不走 token alignment，而是走 sequence reward：

- 先用 student tokenizer decode `responses`，见
  `verl/trainer/ppo/ray_trainer.py:974-978`
- 要求 batch 中必须保留 `raw_prompt`，见 `verl/trainer/ppo/ray_trainer.py:980-982`
- 构造 `response_text` + `raw_prompt` 子任务，见 `verl/trainer/ppo/ray_trainer.py:999-1042`
- teacher worker 在本地重新套 chat template、重新 tokenize、计算 response span 的平均 log prob，
  见 `verl/workers/teacher_workers.py:127-157`、`212-254`
- trainer 把结果写回：
  - `teacher_seq_reward`
  - `teacher_seq_weight`
  - `teacher_token_mask`
  见 `verl/trainer/ppo/ray_trainer.py:1971-2035`

这条路径还有两个很重要的边界：

- `sequence_reward` 只能配 dedicated quantized teacher backend；legacy ref backend 会直接报错，
  见 `verl/trainer/ppo/ray_trainer.py:1496-1501`
- 它给的是 **sequence-level scalar signal**，不是跨 tokenizer 的 token-level exact reverse KL

### 6. MOPD 已经是独立算法节点，而不是 actor 内的特判

`compute_advantage()` 会把 MOPD 相关字段统一从 batch 转交给 estimator：

- `teacher_log_prob` / `old_log_probs` / `rollout_log_probs` / `base_log_prob`
- `lambda_val`
- `teacher_seq_reward` / `teacher_seq_weight` / `teacher_token_mask`
- `orm_weight` / `is_correction` / epsilon bounds

见 `verl/trainer/ppo/ray_trainer.py:137-259`。

真正的 MOPD 公式在 `compute_mopd_advantage()` 里：

- sequence teacher advantage：`verl/trainer/ppo/core_algos.py:1012-1039`
- 主 estimator：`verl/trainer/ppo/core_algos.py:1042-1193`

当前 estimator 明确做了四件事：

1. token-level MOPD 或 ExOPD：
   `verl/trainer/ppo/core_algos.py:1100-1107`
2. IS correction + clamp + all-masked fallback：
   `verl/trainer/ppo/core_algos.py:1109-1137`
3. sequence teacher signal：
   `verl/trainer/ppo/core_algos.py:1139-1168`
4. ORM mixing：
   `verl/trainer/ppo/core_algos.py:1170-1185`

sequence teacher 样本不会错误地再吃到 token-level MOPD，因为：

- trainer 对 sequence teacher 样本写入 `teacher_token_mask = response_mask`
- estimator 先构造 `seq_token_mask`
- 再用 `token_teacher_mask = 1 - seq_token_mask` 将 `A_mopd` 清零

见 `verl/trainer/ppo/ray_trainer.py:2013-2019` 和
`verl/trainer/ppo/core_algos.py:1085-1108`。

### 7. 资源池、preflight、manifest 和测试已经成为架构的一部分

#### 7.1 resource pools

`TaskRunner.init_resource_pool_mgr()` 会把 MOPD teacher pools 纳入 resource pool spec，并按
actor / critic / ref / teachers / optional base worker 的实际 colocate 数需求抬高
`max_colocate_count`，见 `verl/trainer/main_ppo.py:220-287`。

#### 7.2 preflight

在真正初始化 worker 前，trainer 会先做：

- `teacher_id` 分布统计：`verl/trainer/ppo/ray_trainer.py:1250-1251`
- unknown teacher 检查：`1253-1260`
- “全是 default”检查：`1262-1266`
- missing teacher 检查：`1268-1272`
- tokenizer 兼容性检查：`1177-1244`

并且 MOPD 会强制 `ppo_epochs == 1`，防止 teacher advantage 跨 PPO epoch 过期，
见 `verl/trainer/ppo/ray_trainer.py:1544-1552`。

#### 7.3 manifest

checkpoint 现在会把 MOPD manifest 一起写出并在恢复时检查 drift：

- 写 manifest：`verl/trainer/ppo/ray_trainer.py:1657-1667`
- 读 manifest：`1685-1718`
- manifest 内容：`1276-1329`

semantic manifest 里已经记录：

- global lambda / orm / IS bounds / base normalization
- 每个 teacher 的 `name/model_path/lambda_val/backend/tokenizer_* / seq_reward_weight`

deployment manifest 里记录：

- teacher resource pools
- per-teacher `resource_pool`
- per-teacher `log_prob_micro_batch_size`

#### 7.4 tests

当前不再是“多教师逻辑没有测试”：

- 路由调度：`tests/unit/test_teacher_routing.py`
- 配置：`tests/unit/test_teacher_config.py`
- trainer 运行时：`tests/unit/test_mopd_trainer_runtime.py`
- advantage：`tests/unit/test_mopd_advantage.py`
- teacher worker：`tests/unit/test_teacher_workers.py`
- preflight：`tests/unit/test_mopd_preflight.py`
- resource pools：`tests/unit/test_mopd_resource_pools.py`
- E2E：`tests/integration/test_mopd_e2e.py`

而且当前 worktree 的最新验证记录已经明确区分了“重新执行”和“仅源码复核”两类边界：

- core CPU-safe MOPD suite：`112 collected / 111 passed / 1 skipped / 1 warning`
- 额外 teardown/cleanup regression：`5 passed / 3 warnings`
- recipe preflight shell、完整 GPU subprocess E2E、本轮没有重新执行

这说明当前测试基础设施已经很强，但它证明的是：

- MOPD 主链路的 CPU-safe runtime / contract / routing / manifest 覆盖
- 一部分工程收尾辅助逻辑

而不是：

- 本轮 fresh preflight / GPU E2E / multi-node / fault recovery 全部重新通过

---

## 逐项挑战分析

## 挑战 1：模型插槽的二值化硬编码架构

### 旧挑战的本质

旧实现把教师身份编码在单个 worker 的命名槽位和专用方法里，扩展第 3 个教师就会继续膨胀
成员变量、方法名和输出键名。

### 当前分支怎么处理

当前分支直接换了抽象层：

- 一个 teacher = 一个独立 worker group
- 多教师 = `self.teacher_wgs[name]`
- trainer 按 `teacher_id` 把样本路由给对应 worker group

核心证据：

- `verl/trainer/ppo/ray_trainer.py:347-349`
- `verl/trainer/ppo/ray_trainer.py:1462-1535`
- `verl/trainer/ppo/ray_trainer.py:913-972`
- `verl/trainer/ppo/ray_trainer.py:1912-1969`

### 为什么这算已解决

新增教师不再需要：

- 改 actor/ref worker 的模型成员变量
- 新增 `compute_teacher_k_log_prob()`
- 新增新的 tensor 键名
- 在 actor 热路径里扩 if/elif

现在新增教师的必要步骤只剩：

- 在 `algorithm.mopd.teachers[]` 里加一个 teacher 配置
- 保证数据里有对应的 `teacher_id`

### 残留边界

当前 ExOPD 仍假设所有教师共享一个全局 base worker，而不是 per-teacher base：

- `verl/trainer/ppo/ray_trainer.py:757-766`
- `verl/trainer/ppo/ray_trainer.py:1470-1487`

这不影响“N-teacher 不再受插槽数量限制”的结论，但说明“多 teacher + 多 base anchor”不是当前实现目标。

### 状态

**已解决**

---

## 挑战 2：GPU/CPU 内存的线性增长

### 旧挑战的本质

旧分析担心每新增一个教师都要新增模型实例、CPU offload 和 GPU unshard 开销，内存随教师数线性增长。

### 当前分支怎么处理

当前分支做了四层缓解：

1. **shared base worker**
   不再给每个教师单独配一个 base，而是共享 `algorithm.mopd.base_model_path`：
   `verl/trainer/ppo/ray_trainer.py:757-766`、`1470-1487`
2. **dedicated quantized teacher backend**
   支持 `hf_int8` / `hf_4bit`：
   `verl/workers/teacher_workers.py:76-91`
3. **per-teacher micro batch**
   token 和 sequence 两条 teacher 路径都使用 per-teacher micro-batch：
   `verl/trainer/ppo/ray_trainer.py:735-739`、
   `verl/workers/teacher_workers.py:66-69`、
   `183-200`、
   `226-254`
4. **resource pool 隔离**
   可把特定教师移到单独 pool：
   `verl/trainer/main_ppo.py:245-286`

### 为什么还不能算彻底解决

当前架构仍然是一教师一 worker group 一模型实例：

- legacy ref teacher：`verl/trainer/ppo/ray_trainer.py:1496-1507`
- quantized teacher：`verl/trainer/ppo/ray_trainer.py:1508-1514`
- quantized worker 仍是 rank-local 加载完整模型：
  `verl/workers/teacher_workers.py:71-74`、`163-173`

这意味着：

- 总体模型实例数仍随 teacher 数增加
- quantized backend 只是把单 teacher 成本压低，不是消掉增长趋势
- resource pool 只是把内存压力隔离到不同 GPU 池，不是做模型共享

当前默认生产 recipe 也没有使用独立 teacher pool，而是两个教师都在 `global_pool`：

- `recipe/mopd/run_mopd_qwen3_4b.sh:171`
- `recipe/mopd/check_mopd_first_batch.py:90-107`

### 状态

**部分缓解**

---

## 挑战 3：顺序推理的计算时间瓶颈

### 旧挑战的本质

旧实现按阶段串行计算多个教师 log prob，N 增大时 wall-clock 近似线性变差。

### 当前分支怎么处理

当前实现至少做了两件关键改进：

1. **只计算“这批样本真正命中的教师”**
   trainer 先按 `teacher_id` 切 sub-batch，再送给对应教师：
   `verl/trainer/ppo/ray_trainer.py:913-972`
2. **引入按 resource pool 的异步提交**
   `_compute_teacher_log_probs()` / `_compute_teacher_sequence_rewards()` 都先为每个 pool 提交一个 job：
   `verl/trainer/ppo/ray_trainer.py:1944-1954`、
   `1987-1997`

单测也明确验证了调度语义：

- 不同 pool 的 teacher 会先重叠提交
- 同一 pool 内保持串行

见 `tests/unit/test_teacher_routing.py:427-485`。

### 为什么还只是部分缓解

当前调度仍不是“所有教师 fully parallel”：

- 同一 pool 内是串行 drain：
  `verl/trainer/ppo/ray_trainer.py:1956-1967`、
  `1999-2028`
- ExOPD 的 `base_policy_wg` 仍是独立额外阶段：
  `verl/trainer/ppo/ray_trainer.py:2345-2348`
- 当前生产 recipe 把两个教师都放在 `global_pool`，因此真实默认路径并不会享受到跨 pool 的重叠：
  `recipe/mopd/run_mopd_qwen3_4b.sh:171`

所以现在的真实情况是：

- 已经消除了“全 batch 走所有教师”的浪费
- 已经具备“跨 pool 重叠”的能力
- 但默认部署形态仍会在共享 pool 上串行跑 teacher forwards

### 状态

**部分缓解**

---

## 挑战 4：配置系统的语义重载与不可扩展性

### 旧挑战的本质

旧双教师实现通过重载已有字段表达 student / base / teacher1 / teacher2，
语义混乱且无法扩展到 teacher3。

### 当前分支怎么处理

当前实现已经把配置语义重新分层：

- `actor_rollout_ref.model.path` 只表示 student
- `algorithm.mopd.base_model_path` 只表示 ExOPD shared base
- `algorithm.mopd.teachers[]` 表示 teacher 集合
- `algorithm.mopd.*` 表示算法参数

证据：

- `verl/trainer/config/ppo_trainer.yaml:41-42`
- `verl/trainer/config/algorithm/mopd.yaml:1-51`
- `verl/workers/config/teacher.py:23-162`
- `recipe/mopd/run_mopd_qwen3_4b.sh:167-176`

### 为什么这算已解决

当前再增加一个 teacher，不需要再滥用别的配置字段，只需要在
`algorithm.mopd.teachers[]` 增加一项。

### 仍需记录的实现细节

当前 `mopd` 的 typed dataclass 校验没有像 actor/critic 那样在
`validate_config()` 中统一实例化：

- `verl/utils/config.py:149-175`
- `verl/trainer/config/algorithm.py:567-614`

所以这项挑战的“语义重载问题”已解决，但“typed validation 全链路统一化”还没完全做完。

### 状态

**已解决**

---

## 挑战 5：教师路由的硬编码字符串匹配

### 旧挑战的本质

旧实现在 actor 热路径里用 `"math"` / `"code"` if/elif/else 做路由和公式选择。

### 当前分支怎么处理

现在的教师路由完全数据驱动：

- 数据里显式携带 `teacher_id`：
  `verl/utils/dataset/rl_dataset.py:371-374`
- trainer 将 `teacher_id` 转成 mask/indices：
  `verl/trainer/ppo/ray_trainer.py:918-969`、
  `999-1040`
- unknown teacher 会 fail-fast：
  `verl/trainer/ppo/ray_trainer.py:921-928`、
  `985-991`

### 为什么这算已解决

当前不存在下面这种扩展方式了：

- 在 actor 里继续加 `elif teacher == "reasoning":`
- 把 teacher 名到张量名的映射藏在控制流里
- unknown teacher 静默回退到默认教师

teacher identity 现在只由两件事决定：

- `teacher_id`
- `self.teacher_wgs[teacher_id]`

### 状态

**已解决**

---

## 挑战 6：Batch 张量的 Pop/Restore 脆弱模式

### 旧挑战的本质

旧实现为了在同一个 batch 上临时切换不同输入视图，会原地 pop/restore 张量，
导致数据流脆弱、易污染。

### 当前分支怎么处理

当前实现的策略是：

- 对 teacher 路由：`batch.select_idxs(indices)` 生成子批，再 `pad/unpad` 后 scatter 回原 batch
  位置，见 `verl/trainer/ppo/ray_trainer.py:943-969`
- 对 sequence teacher：同样在子批上操作，不改原 batch 的主张量结构，
  见 `verl/trainer/ppo/ray_trainer.py:1008-1039`
- 对 base log prob：直接用 dedicated `base_policy_wg.compute_ref_log_prob(batch)`，
  再把输出键从 `ref_log_prob` 重命名为 `base_log_prob`，见
  `verl/trainer/ppo/ray_trainer.py:798-814`

### 为什么这算已解决

当前没有看到旧式“先 pop 掉一组张量，再算完以后 restore 回去”的全局 batch 改写模式。
新方案的核心是：

- 子批视图
- 局部 padding/unpadding
- scatter 回原始索引

### 状态

**已解决**

---

## 挑战 7：分词器的单一假设

### 旧挑战的本质

旧实现默认 student 和 teacher 共用一套 ref tokenization 流，跨 tokenizer 时既无法对齐
输入，也无法对齐 token 级别的 teacher 信号。

### 当前分支怎么处理

当前分支做了两层处理：

#### 7.1 compatible 路径

token-level teacher 仍可用，但 trainer 会先做严格兼容性验证：

- teacher tokenizer path 与 student path 相同可直接通过
- 否则必须显式声明 `tokenizer_compat_group`
- 然后再比较 tokenizer signature，包括 vocab hash、special token、padding side 等

见 `verl/trainer/ppo/ray_trainer.py:1177-1244`。

#### 7.2 sequence_reward 路径

异构 tokenizer teacher 不再强行共用 student token ids，而是：

- 保留 `raw_prompt`
- 用 student tokenizer decode `responses`
- teacher 端重新套自己的 chat template 并 retokenize
- 用 response span 的平均 log prob 生成 `seq_scores`

见：

- `verl/trainer/ppo/ray_trainer.py:974-1042`
- `verl/workers/teacher_workers.py:127-157`
- `verl/workers/teacher_workers.py:212-254`
- `verl/trainer/ppo/core_algos.py:1012-1039`

### 为什么还只是部分缓解

当前分支确实支持了异构 tokenizer teacher，但不是“完全通用的 token-level 方案”：

- token-level dense distillation 仍然要求 tokenizer compatible
- heterogeneous teacher 只能走 `sequence_reward`
- `sequence_reward` 只在 dedicated quantized teacher backend 上支持，
  legacy ref backend 会直接报错，见
  `verl/trainer/ppo/ray_trainer.py:1496-1501`
- `sequence_reward` 生成的是 scalar sequence signal，不是跨 tokenizer 精确对齐的
  token-level reverse KL

### 状态

**部分缓解**

---

## 挑战 8：全局 lambda_vals 无法适配异构教师

### 旧挑战的本质

旧实现只有单个全局 `lambda_vals` 标量，所有教师共享同一强度。

### 当前分支怎么处理

当前实现已经支持 per-teacher lambda：

- 配置入口：`verl/workers/config/teacher.py:47-52`
- 按 `teacher_id` 构造 per-sample `lambda_val` tensor：
  `verl/trainer/ppo/ray_trainer.py:774-796`
- dispatch 到 MOPD estimator：
  `verl/trainer/ppo/ray_trainer.py:219-234`
- ExOPD 公式直接消费这个 batch 级 tensor：
  `verl/trainer/ppo/core_algos.py:1100-1107`

### 为什么这算已解决

当前 teacher A 和 teacher B 可以在同一个 batch 中拥有不同的 lambda。
这正是旧全局标量方案做不到的。

### 边界

当前 sequence teacher 使用的是单独的 `seq_reward_weight`，而不是 `lambda_val`。
这不是缺陷，而是因为 sequence 路径本来就不是 ExOPD token-level 公式。

### 状态

**已解决**

---

## 挑战 9：数据管道缺少验证

### 旧挑战的本质

旧实现会把教师身份字段静默带入 batch，但不做 unknown teacher / distribution / coverage 检查，
容易产生错误训练信号而不报错。

### 当前分支怎么处理

当前实现补了三道防线：

1. **recipe 侧显式注入 teacher_id**
   `recipe/mopd/prepare_data.py:53-71`
2. **dataset 侧显式提取 teacher_id**
   `verl/utils/dataset/rl_dataset.py:371-374`
3. **trainer preflight fail-fast**
   `verl/trainer/ppo/ray_trainer.py:1246-1274`

preflight 现在会：

- 打印 `teacher_id` 分布：`1250-1251`
- 拒绝 unknown teacher：`1253-1260`
- 拒绝多教师场景下全是 `default`：`1262-1266`
- 拒绝配置里声明了但数据中完全缺失的 teacher：`1268-1272`

### 为什么这算已解决

旧挑战真正危险的是“数据错了但训练还能继续跑”。当前这条路径已经被 fail-fast 打断。

另外，recipe 数据准备脚本也会打印 teacher 分布：

- `recipe/mopd/prepare_data.py:214-229`

这已经覆盖了旧文档里提到的“无数据集级教师分布统计”问题。

### 状态

**已解决**

---

## 挑战 10：优势函数的覆盖式计算与浪费

### 旧挑战的本质

旧实现把 teacher 蒸馏逻辑放在 actor 侧覆盖已有优势，导致数据流分裂、计算浪费且难以混合 ORM。

### 当前分支怎么处理

MOPD 现在是独立注册的 advantage estimator：

- 注册点：`verl/trainer/ppo/core_algos.py:1042`
- dispatch 入口：`verl/trainer/ppo/ray_trainer.py:137-259`

并且 sequence teacher 与 ORM 都是 estimator 内的显式组成项：

- sequence teacher：`verl/trainer/ppo/core_algos.py:1139-1168`
- ORM：`verl/trainer/ppo/core_algos.py:1170-1185`

### 为什么这算已解决

现在的训练流是：

1. trainer 收集 teacher/base/rollout 信号
2. `compute_advantage()` 调用 `compute_mopd_advantage()`
3. actor 只消费标准化后的 `advantages`

也就是说：

- teacher 蒸馏不再“覆盖”别的优势
- ORM 不是旁路 hack，而是同一个 estimator 的显式组成部分
- actor 层不再承担 teacher-specific 公式逻辑

### 状态

**已解决**

---

## 挑战 11：检查点与恢复机制的缺失

### 旧挑战的本质

旧实现 checkpoint 只保存 actor/critic，恢复训练依赖外部 CLI 参数重新把 teacher 拼回去，
缺少自描述和 drift 检测。

### 当前分支怎么处理

当前分支已经加入 manifest：

- 生成 manifest：`verl/trainer/ppo/ray_trainer.py:1276-1329`
- 保存 manifest：`verl/trainer/ppo/ray_trainer.py:1663-1667`
- 恢复时校验 manifest：`verl/trainer/ppo/ray_trainer.py:1715-1718`
- semantic drift 直接报错，deployment drift 仅 warning：
  `verl/trainer/ppo/ray_trainer.py:1331-1337`

manifest 里已经记录：

- global MOPD semantic config
- 每个 teacher 的 model path / backend / tokenizer policy / lambda / seq weight
- deployment 侧的 resource pool 和 micro-batch 信息

### 为什么还只是部分缓解

当前 checkpoint 仍然不是完全自包含：

- 实际保存的还是 actor / critic checkpoints，manifest 只是额外 JSON 元数据，
  见 `verl/trainer/ppo/ray_trainer.py:1648-1667`
- teacher/base worker 权重本身不随 checkpoint 打包
- resume 仍然依赖外部模型路径在恢复时可用且内容未漂移
- manifest 记录的是路径和值，不是 artifact hash 或 revision pin

所以它已经从“完全不记 MOPD 上下文”进化成“能检测语义漂移”，但还没有进化成
“拿到 checkpoint 就能完整自举恢复 teacher/base artifacts”。

### 状态

**部分缓解**

---

## 挑战 12：测试基础设施的空白

### 旧挑战的本质

旧文档写这条时，多教师蒸馏几乎没有测试。

### 当前分支怎么处理

当前测试已经覆盖：

- teacher config
- teacher routing
- dataset `teacher_id`
- tokenizer compatibility / sequence reward runtime
- quantized teacher worker
- resource pools
- preflight command and failure detection
- manifest drift
- advantage 公式
- lightweight integration
- real recipe-aligned GPU E2E

代表性证据：

- 路由跨 pool overlap / 同 pool 串行：
  `tests/unit/test_teacher_routing.py:427-485`
- sequence reward runtime：
  `tests/unit/test_mopd_trainer_runtime.py:540-690`
- manifest drift：
  `tests/unit/test_mopd_trainer_runtime.py:733-768`
- sequence teacher advantage 组合：
  `tests/unit/test_mopd_advantage.py:328-428`
- 当前验证结果：
  `docs/plans/mopd-test-results.md:3-61`

### 为什么这算已解决

“测试空白”这个问题本身已经不存在了。当前当然仍然不是无限完整，但已经远远不是零覆盖状态。

需要如实保留的剩余验证边界也已经写在 `mopd-test-results.md` 里：

- 长程稳定性
- 多节点
- OOM recovery / fault recovery

见 `docs/plans/mopd-test-results.md:46-61`。

### 状态

**已解决**

---

## 最终判断

如果只问“当前分支到底采用了什么架构”，答案是：

**trainer/controller 侧多教师 worker graph + dual-path teacher signal**

如果只问“和旧双教师实现相比，哪些挑战真正被打掉了”，答案是：

- **已解决**：1 / 4 / 5 / 6 / 8 / 9 / 10 / 12
- **部分缓解**：2 / 3 / 7 / 11
- **当前没有哪一项还是旧文档描述的原样问题**

如果只问“哪些点在当前分支里还不能说已经彻底解决”，最重要的是四个：

1. **内存仍随教师数增长**
   只是通过 shared base、quantization、resource pools 和 micro-batching 做了工程缓解。
2. **teacher forward 仍不是完全并行**
   当前只做到“跨 pool 重叠、同 pool 串行”，默认 recipe 仍然共享 `global_pool`。
3. **异构 tokenizer 不是通用 token-level 解**
   当前的真实解决方案是 `sequence_reward` fallback，而不是跨 tokenizer 的 token-level exact KL。
4. **checkpoint 还不自包含**
   当前有 manifest 和 drift detection，但 teacher/base artifacts 仍是外部依赖。

如果把“工程收尾”单独拿出来看，当前还有两条不能写成彻底收口：

5. **lifecycle cleanup 只是部分收口**
   `fit()` 已新增统一的 `_finalize_fit_resources()`，但 `base_policy_wg` 仍无对称 cleanup，异常路径也还没有
   全局 `finally` 保障。
6. **typed config 仍未全链路接线**
   `TeacherConfig` / `MOPDConfig` 已存在，但 `AlgoConfig` 和 `validate_config()` 还没把 `algorithm.mopd`
   完整 typed 化。

这也是为什么当前分支已经足够支撑 MOPD 主链路落地，但还没有把所有 N-teacher 工程问题做到
“理论上完全消失”的根本原因。
