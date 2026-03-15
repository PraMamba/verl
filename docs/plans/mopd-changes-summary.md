# MOPD Implementation Changes Summary

**Date**: 2026-03-15
**Scope**: current worktree source, including uncommitted follow-up changes, not historical milestone snapshots
**Branch**: `feature/mopd-implementation`
**HEAD**: `4eb8e245`
**Merge-base with `main`**: `80eb57ea`
**Commits on top of `main`**: 21

---

## Refresh Note

旧版 `mopd-changes-summary.md` 已明显落后于当前分支实现，主要问题有三类：

1. 混入了多轮中间里程碑、设计稿和部署收尾说明，导致“已实现 / 待实现 / 历史状态”被写在一起。
2. 把已经落地的内容继续写成“规划项”或“P2 方向”，尤其是异构 tokenizer、量化 teacher、resource pool、manifest 和 per-teacher metrics。
3. 把一些并未真正接线的字段写成当前能力，例如 `TeacherConfig.weight` 的运行时加权聚合。

这次刷新只以当前 worktree 源码、测试和同仓文档为准，目标是回答：

- 当前 worktree 上 MOPD 到底已经落成了什么架构。
- 哪些文件承载了真实实现。
- 哪些旧说法已经不再成立。
- 哪些限制仍然存在，不能被文档包装成“已解决”。

补充边界说明：

- 本文的实现判断以 **当前 worktree** 为准，而不是只看 `HEAD` 提交。
- 但 commit 数、merge-base 和 `git diff main...HEAD` 统计仍然只反映 `HEAD` 快照，不包含未提交改动。

---

## Executive Summary

当前分支的 MOPD 不是“独立 trainer”或“外挂脚本”，而是已经嵌入标准 verl PPO 主路径：

- 入口仍然是 `python -m verl.trainer.main_ppo`
- 通过 `algorithm.adv_estimator=mopd` 和 `algorithm.mopd.*` 打开
- 训练编排仍由 `RayPPOTrainer` 负责

当前真实落地的 runtime 形态是：

**`algorithm.mopd.teachers[]` + trainer 侧 `teacher_wgs` worker graph + per-sample `teacher_id` 路由 + dual-path teacher signal（token-level `teacher_log_prob` / sequence-level `teacher_seq_reward`）+ 独立 MOPD advantage estimator**

也就是说，这个分支已经不再是早期“双教师硬塞进已有 ref/base 插槽”的实现草案，而是一个显式的多教师 trainer-side orchestration 方案。

当前代码已经支持：

- 标准 MOPD reverse-KL advantage
- ExOPD / G-OPD shared-base normalization
- per-teacher `lambda_val`
- train/inference engine IS correction
- per-sample `teacher_id` 路由
- 显式 `algorithm.mopd.resource_pools`
- 跨 pool overlap、同 pool 串行的 teacher 调度
- `legacy_ref` / `hf_int8` / `hf_4bit` teacher backend
- `compatible` / `sequence_reward` tokenizer policy
- teacher manifest + drift detection
- preflight dataset/tokenizer validation
- per-teacher runtime metrics
- fresh single-node runtime evidence across preflight / GPU E2E / default `6/6` recipe run /
  conservative `18/18` long-run rerun

但当前代码也明确 **还不等于**：

- 按 `TeacherConfig.weight` 做运行时多教师加权聚合
- per-teacher base model
- Hydra composition-time typed `AlgoConfig.mopd` 强约束
- 可重复的长程质量收益、多节点容错、恢复一致性的充分验证

---

## Tracked Diff Summary

以下统计以父仓的 `git diff main...HEAD` 为准：

- **47** 个 tracked files changed
- **+9,579 / -53** lines
- 文件分布：
  - `verl/`: **28** 个 runtime/config 相关文件
  - `tests/`: **14** 个测试文件
  - `docs/`: **5** 个文档文件

需要单独说明：

- 当前 worktree 下的 `recipe/` 目录对 MOPD 运行非常重要，但它在父仓视角是 **submodule / companion tree**，因此其内部脚本不会出现在上面的 tracked diff 统计里。
- 旧版文档把 `recipe/` 内部文件也和父仓 tracked diff 混在一起统计，这是不准确的。

---

## What Actually Changed

### 1. MOPD 已成为正式的算法子配置，而不是临时 override

当前主配置已经把 MOPD 作为正式 defaults 组件接入：

- `verl/trainer/config/algorithm/mopd.yaml`
- `verl/trainer/config/ppo_trainer.yaml`
- `verl/trainer/config/ppo_megatron_trainer.yaml`
- `verl/trainer/config/_generated_ppo_trainer.yaml`
- `verl/trainer/config/_generated_ppo_megatron_trainer.yaml`
- `verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml`
- `verl/trainer/config/_generated_ppo_veomni_trainer.yaml`

这意味着当前分支已经从“靠 shell 覆盖注入若干临时字段”的状态，升级为：

- `algorithm.mopd.enabled`
- `algorithm.mopd.lambda_val`
- `algorithm.mopd.orm_weight`
- `algorithm.mopd.is_correction`
- `algorithm.mopd.is_epsilon_low/high`
- `algorithm.mopd.use_base_normalization`
- `algorithm.mopd.base_model_path`
- `algorithm.mopd.resource_pools`
- `algorithm.mopd.teachers[]`

同时，`verl/workers/config/teacher.py` 新增了三套 dataclass：

- `TeacherConfig`
- `TeacherResourcePoolConfig`
- `MOPDConfig`

这些 dataclass 现在已经承担：

- teacher name 唯一性校验
- backend 校验
- tokenizer policy 校验
- `seq_reward_weight` 校验
- resource pool 维度校验
- ExOPD base config 校验
- IS epsilon 边界校验

### 2. 参考策略需求已经被提升到算法层判断

`verl/trainer/ppo/utils.py` 中的 `need_reference_policy()` 已经不再只为 KL penalty 服务。
当前逻辑会在以下场景返回 `True`：

- `algorithm.use_kl_in_reward=True`
- `actor_rollout_ref.actor.use_kl_loss=True`
- `algorithm.mopd.enabled=True`
- `algorithm.adv_estimator == "mopd"`

这保证了 MOPD 不会再因为“未开启 KL，所以没初始化 ref policy”而走到错误路径。

### 3. 数据契约已经稳定到 `teacher_id`

`verl/utils/dataset/rl_dataset.py` 现在支持 `teacher_id_field`：

- 数据集 `__getitem__()` 会在配置打开时提取 `teacher_id`
- 缺失时回退到 `"default"`
- `raw_prompt`、`teacher_id` 等非 tensor 字段会保留到 `batch.non_tensor_batch`

这让 trainer 可以基于明确的 `teacher_id` 做路由，而不是继续依赖硬编码字符串分支或隐式样本类别推断。

### 4. 主体实现已经集中到 `RayPPOTrainer`

当前 MOPD 的核心编排不在额外 trainer 文件里，而是已经内嵌在 `verl/trainer/ppo/ray_trainer.py`。

最重要的运行时事实如下：

- trainer 维护 `self.teacher_wgs: dict[str, RayWorkerGroup]`
- ExOPD 开启时额外维护 `self.base_policy_wg`
- MOPD preflight 在 `init_workers()` 之前执行
- 真正的 teacher log prob / sequence reward 采集、lambda 构造、base normalization、manifest、metrics 都在这个文件里

当前 trainer 侧已经具备的能力包括：

- `_build_teacher_worker_config()`
- `_build_quantized_teacher_worker_config()`
- `_build_base_worker_config()`
- `_build_mopd_lambda_tensor()`
- `_build_mopd_teacher_jobs()`
- `_build_mopd_sequence_teacher_jobs()`
- `_compute_teacher_log_probs()`
- `_compute_teacher_sequence_rewards()`
- `_compute_base_log_prob()`
- `_record_mopd_teacher_metrics()`
- `_build_mopd_manifest()`
- `_validate_loaded_mopd_manifest()`
- `_finalize_fit_resources()`
- `_shutdown_dataloader_workers()`
- `cleanup_teacher_workers()`

### 5. 老的 ref worker 已经补齐异步接口，供 trainer 做非阻塞 teacher 提交

为了支持 trainer 侧 teacher 调度，三个后端都新增了非阻塞 ref-log-prob API：

- `verl/workers/fsdp_workers.py`
- `verl/workers/megatron_workers.py`
- `verl/workers/engine_workers.py`

当前这些 worker 都把原来的同步实现抽成 `_compute_ref_log_prob_impl()`，然后暴露：

- `compute_ref_log_prob()`
- `compute_ref_log_prob_async()`

配套地：

- `verl/single_controller/base/decorator.py` 给 dispatch metadata 增加了 `mesh_name`
- `verl/single_controller/ray/base.py` 可以解析 richer `resource_pool_spec`

这两处改动共同支撑了 trainer 当前的两件事：

1. 从 worker method metadata 反推出 teacher 的 DP mesh 名称
2. 在不同 resource pool 间重叠 teacher 提交，在同 pool 内保持串行

### 6. teacher backend 已经拆成 legacy_ref 与 dedicated quantized worker 两条路径

当前分支并不是所有 teacher 都复用同一种 worker。

#### 6.1 `legacy_ref`

用于 token-compatible teacher：

- 继续复用现有 ref worker 路径
- teacher forward 固定使用 per-teacher micro batch
- 不再沿用全局 dynamic ref batching

#### 6.2 `hf_int8` / `hf_4bit`

`verl/workers/teacher_workers.py` 新增了 dedicated `HFQuantizedTeacherWorker`，负责：

- rank-local HF model load
- BitsAndBytes int8 / 4bit quantization
- token-level `compute_ref_log_prob`
- sequence-level `compute_seq_scores`
- sequence teacher path 的 micro-batching
- teacher-side retokenization 与 response scoring

这个 worker 的引入意味着：

- 量化 teacher backend 已经不是“设计中的扩展”，而是当前 runtime surface 的一部分
- `sequence_reward` 也已经不是文档设想，而是实装路径

同时当前 runtime 也明确限制：

- `tokenizer_policy=sequence_reward` 不能和 `backend=legacy_ref` 搭配
- sequence teacher 当前需要 dedicated quantized teacher worker 路径

### 7. MOPD advantage estimator 已经扩展到 sequence teacher + ORM + ExOPD

`verl/trainer/ppo/core_algos.py` 里当前真正的 MOPD 实现已经不止最早的 reverse-KL。

现在包括两块新增核心逻辑：

#### 7.1 `compute_mopd_sequence_teacher_advantage()`

负责：

- 仅对 active sequence-teacher 样本做组内标准化
- 使用 `uid/index` 做 group normalization
- 把 sequence-level score broadcast 到 response tokens

#### 7.2 `compute_mopd_advantage()`

当前实际组合关系是：

- token-level MOPD / ExOPD advantage
- optional IS correction
- optional sequence teacher advantage
- optional ORM advantage

并且还处理了：

- `teacher_token_mask`
- `teacher_seq_weight`
- all-masked IS fallback
- overflow-safe log-ratio clamp
- `mopd/is_ratio_mean`
- `mopd/is_valid_fraction`
- `mopd/is_zeroed_fraction`

这意味着旧版文档里把 sequence teacher 继续当作“P2 设计项”已经不成立。

### 8. `compute_advantage()` 的 batch plumbing 已经接通 MOPD 专用字段

`verl/trainer/ppo/ray_trainer.py` 顶部的 `compute_advantage()` wrapper 现在已经会把以下字段从 batch 透传到 estimator：

- `teacher_log_prob`
- `old_log_probs`
- `rollout_log_probs`
- `base_log_prob`
- `lambda_val`
- `teacher_seq_reward`
- `teacher_seq_weight`
- `teacher_token_mask`
- `uid`

这保证了 MOPD 不再需要在 actor 更新前额外拼装另一套私有训练数据结构。

### 9. preflight、manifest 和 teacher metrics 已经进入主链路

当前实现已经不再是“代码能跑，但训练前后缺少约束”的状态。

#### 9.1 preflight

`RayPPOTrainer._run_mopd_preflight_checks()` 当前会检查：

- 训练集里的 `teacher_id` 分布
- unknown teacher ids
- missing configured teachers
- 多 teacher 配置下是否真的做了路由
- student / teacher tokenizer compatibility
- base model tokenizer compatibility

#### 9.2 manifest

checkpoint manifest 已经会记录：

- `adv_estimator`
- `lambda_val`
- `orm_weight`
- `is_correction`
- `is_epsilon_low/high`
- `use_base_normalization`
- `base_model_path`
- per-teacher `backend`
- per-teacher `tokenizer_policy`
- per-teacher `seq_reward_weight`
- teacher resource pool deployment info

并区分：

- semantic drift: 直接报错
- deployment-only drift: 记录 warning

#### 9.3 per-teacher metrics

trainer 已经会按 teacher 输出指标，例如：

- `mopd/<teacher>/sample_fraction`
- `mopd/<teacher>/adv_mean`
- `mopd/<teacher>/adv_std`
- `mopd/<teacher>/reverse_kl_mean`
- `mopd/<teacher>/seq_reward_mean`
- `mopd/<teacher>/is_valid_fraction`

#### 9.4 fit teardown 已有明确收口，并已在真实运行闭环中跑通

当前 worktree 中，`RayPPOTrainer.fit()` 已不再只在训练循环末尾零散地做局部清理，而是新增了统一的
`_finalize_fit_resources()`：

- 关闭 `progress_bar`
- 调用 `_shutdown_dataloader_workers()` 释放 dataloader worker 迭代器
- 尝试关闭 `async_rollout_manager`
- 调用 `cleanup_teacher_workers()`
- 调用 `tracking_logger.finish()`

这说明“工程收尾”已经从文档口号进入了真实 trainer 控制流。

同时，当前代码已经补上了此前最明显的两处闭环缺口：

- `cleanup_teacher_workers()` 现在会同时释放 `teacher_wgs` 与 `base_policy_wg`
- `fit()` 末尾已经有统一的 `try/finally` 兜底，异常路径也会进入 `_finalize_fit_resources()`

2026-03-15 的真实运行闭环也给出了运行侧证据：

- recipe preflight 成功到达首个真实 `training/global_step:1`
- opt-in GPU E2E 外层 pytest 成功通过
- full recipe shell run 退出码为 `0`，完成 `Training Progress: 100%|...| 6/6`
- conservative long-run rerun 退出码也为 `0`，完成 `Training Progress: 100%|...| 18/18`，
  打印 `validation generation end` 与 `Final validation metrics`

当前仍需要如实记录的残余问题不是“cleanup 没接线”，而是：

- GPU E2E、默认 full run 和 `18/18` long-run log 末尾仍可见
  `multiprocessing.resource_tracker` `KeyError`
- 这些噪声都出现在成功完成和 checkpoint 写出之后，更像 vLLM / multiprocessing 的关停噪声，而不是 MOPD 主链路失败

---

## Key Files And Their Current Roles

| File | Current role | Landed changes |
|---|---|---|
| `verl/workers/config/teacher.py` | MOPD schema | `TeacherConfig` / `TeacherResourcePoolConfig` / `MOPDConfig` |
| `verl/trainer/config/algorithm/mopd.yaml` | Hydra sub-config | MOPD defaults and example fields |
| `verl/trainer/config/ppo_trainer.yaml` | PPO entry config | 正式挂载 `algorithm@algorithm.mopd: mopd` |
| `verl/trainer/config/ppo_megatron_trainer.yaml` | Megatron PPO config | 同步挂载 `mopd` defaults |
| `verl/trainer/config/_generated_ppo*.yaml` | Generated snapshots | 反映当前 MOPD config surface |
| `verl/utils/dataset/rl_dataset.py` | Dataset contract | `teacher_id_field` 提取与 batch 透传 |
| `verl/trainer/main_ppo.py` | Resource-pool bootstrap | 将 `algorithm.mopd.resource_pools` lower 到 `resource_pool_spec` 并动态扩充 colocate budget |
| `verl/single_controller/base/decorator.py` | Dispatch metadata | 暴露 `mesh_name` 供 trainer 推导 teacher DP mesh |
| `verl/single_controller/ray/base.py` | Resource pool runtime | 支持 rich spec dict，同时保持 legacy list spec 兼容 |
| `verl/workers/fsdp_workers.py` | FSDP ref path | 新增 `compute_ref_log_prob_async()` |
| `verl/workers/megatron_workers.py` | Megatron ref path | 新增 `compute_ref_log_prob_async()` |
| `verl/workers/engine_workers.py` | Engine ref path | 新增 `compute_ref_log_prob_async()` |
| `verl/workers/teacher_workers.py` | New quantized teacher runtime | `HFQuantizedTeacherWorker` + sequence score path |
| `verl/trainer/ppo/core_algos.py` | MOPD math | ExOPD, IS correction, sequence teacher, ORM mixing |
| `verl/trainer/ppo/ray_trainer.py` | Main MOPD orchestration | teacher worker init, routing, metrics, manifest, fit finalization, cleanup |
| `verl/trainer/ppo/utils.py` | Trainer dependency logic | MOPD 也会触发 `need_reference_policy()` |
| `verl/workers/config/__init__.py` | Config export surface | 暴露 `teacher` config module |

---

## Companion Runtime Assets In `recipe/`

虽然 `recipe/` 不计入上面的父仓 tracked diff，但当前 worktree 里的 MOPD 运行说明和验证辅助实际上依赖这些 companion files：

- `recipe/mopd/prepare_data.py`
- `recipe/mopd/check_mopd_first_batch.py`
- `recipe/mopd/run_mopd_qwen3_4b.sh`
- `recipe/mopd/run_mopd_qwen3_4b_preflight.sh`
- `recipe/mopd/README.md`

这些文件体现的 **当前运行契约** 包括：

- 数据集必须提供 `teacher_id`
- recipe 默认使用标准 `main_ppo` 入口而不是单独 MOPD trainer
- production / preflight 会打印 `verl.__file__` 做 import 自检
- 运行脚本显式绑定当前 worktree `PYTHONPATH`
- preflight 以“第一步真实 actor update 出现”为成功边界
- memory defaults 已针对 teacher log-prob 路径做保守设置

因此，旧版文档把 recipe 写成“实现阶段附带脚本”也已经不准确了。当前它们是部署和验证链路的一部分。

---

## Validation Surface

当前测试面已经明显大于旧版文档描述的早期快照。本次刷新重新执行了 MOPD 相关 CPU-safe suite、
collect-only 命令、一个独立的 teardown/cleanup 回归文件，以及一轮真实运行闭环，结果为：

- `pytest -q ...` -> **115 passed, 1 skipped, 1 warning in 11.31s**
- `pytest --collect-only -q ...` -> **116 tests collected in 8.57s**
- `pytest -q tests/unit/test_teardown_cleanup.py` -> **5 passed, 3 warnings in 11.55s**
- `bash recipe/mopd/run_mopd_qwen3_4b_preflight.sh` -> **exit 0**, success boundary at `training/global_step:1`
- `VERL_MOPD_E2E=1 pytest tests/integration/test_mopd_e2e.py::test_mopd_training_e2e -v` -> **1 passed, 1 warning in 345.91s**
- `bash recipe/mopd/run_mopd_qwen3_4b.sh` -> **exit 0**, `Training Progress: 100%|...| 6/6`, `training/global_step:6`
- conservative long-run rerun -> **exit 0**, `Training Progress: 100%|...| 18/18`,
  `training/global_step:18`, `validation generation end`, `Final validation metrics`
- 这次长程 run 还直接回答了三件事：teacher 路由均值 `cell=0.4965` / `disease=0.5035`，IS mask 均值
  `is_valid_fraction=0.9999998278` / `is_zeroed_fraction=1.714958e-7`，最终 `adv_mean` /
  `reverse_kl_mean` 仍在可解释区间而非发散

当前测试覆盖已经包含：

- config schema validation
- dataset `teacher_id` extraction
- routed teacher log-prob computation
- routed sequence teacher reward computation
- async scheduling across resource pools
- tokenizer compatibility checks
- base normalization worker path
- quantized teacher worker behavior
- manifest drift handling
- preflight command generation
- production run-script self-checks
- trainer finalization / teardown helper behavior
- opt-in GPU E2E contract
- recipe-aligned preflight shell path
- full recipe shell path through final checkpoint write
- conservative long-run closure through final validation at `18/18`

本次刷新确认的 suite inventory 为：

- **116 tests collected**
- **115 passed**
- **1 skipped**（GPU E2E opt-in path）

这已经远超过旧版文档中“只完成少量 smoke/regression”的表述。
并且“只验证 CPU-safe surface、没有 fresh preflight / GPU E2E / full run”这类说法也已经过时。

此外，本次还补了一轮真实权重量化 teacher profiling。方法边界是：

- 直接实例化 `HFQuantizedTeacherWorker`
- 用 recipe 默认的两条真实 teacher checkpoint 路径做 `hf_int8` / `hf_4bit` 加载
- 记录 `init_model()` 与首个 `_compute_ref_log_prob_impl()` 的启动时间、首步延迟和 `torch.cuda.max_memory_reserved()`
- `N teacher` 取单机 4 卡；4-teacher 案例复用两条真实 checkpoint 扩成 4 个 teacher worker

最小 profiling 表如下：

| Teacher Count | Backend | `log_prob_micro_batch_size` | Peak Reserved GiB | First-Step Latency (s) | Startup Time (s) |
|---:|---|---:|---:|---:|---:|
| 1 | `hf_int8` | 2 | `5.05` (`5.05` total) | `1.70` | `17.95` |
| 1 | `hf_int8` | 4 | `5.05` (`5.05` total) | `1.20` | `17.45` |
| 2 | `hf_int8` | 2 | `5.05` (`10.05` total) | `2.44` | `28.20` |
| 4 | `hf_int8` | 2 | `5.05` (`20.10` total) | `3.28` | `49.38` |
| 1 | `hf_4bit` | 2 | `3.81` (`3.81` total) | `1.15` | `15.37` |
| 1 | `hf_4bit` | 4 | `3.81` (`3.81` total) | `0.77` | `15.23` |
| 2 | `hf_4bit` | 2 | `3.81` (`7.63` total) | `1.54` | `27.78` |
| 4 | `hf_4bit` | 2 | `3.81` (`15.25` total) | `2.22` | `52.60` |

这轮 profiling 说明了三件事：

- `hf_int8` / `hf_4bit` 都不只是 mock contract，而是已经在真实 recipe teacher 权重上成功加载并完成首个 log-prob 前向
- `hf_4bit` 相比 `hf_int8`，当前这组 teacher 权重的单卡峰值显存从约 `5.05 GiB` 降到了约 `3.81 GiB`
- teacher 数和启动时间在当前 harness 里基本近似线性增长，这和当前 trainer 顺序初始化 teacher worker 的实现一致

---

## Corrections To The Old Summary

下面这些旧说法在当前分支上已经不成立，刷新时应该明确去掉：

1. **“MOPD 主要还是设计文档，P2/P1 尚未落地”**
   - 不成立。sequence teacher、quantized backend、resource pools、manifest、per-teacher metrics 都已落地。

2. **“MOPD 依赖独立 trainer / recipe.mopd.main_mopd”**
   - 不成立。当前主入口就是标准 `verl.trainer.main_ppo`。

3. **“量化 teacher 只是设想”**
   - 不成立。`HFQuantizedTeacherWorker` 已进入 runtime/test surface。

4. **“异构 tokenizer 仅限未来扩展”**
   - 不成立。当前已有 `tokenizer_policy=sequence_reward` 路径。

5. **“resource pool 需要外部手工预建”**
   - 不成立。`TaskRunner.init_resource_pool_mgr()` 已能从 `algorithm.mopd.resource_pools` 降配创建。

6. **“当前只需记录双教师路径”**
   - 不成立。当前真实架构已是 `teachers[] + teacher_wgs + teacher_id routing`。

7. **“TeacherConfig.weight 已参与 runtime 聚合”**
   - 不成立。该字段当前仍不是 runtime 能力；非默认值会 fail-fast，当前实现也没有按它做 teacher signal 的加权融合。

8. **“每个 teacher 都有自己的 base model path”**
   - 不成立。当前 ExOPD 运行时使用的是 shared `algorithm.mopd.base_model_path`；per-teacher `base_model_path` 会直接 fail-fast。

9. **“真实 shell 链路这次没有重新执行”**
   - 不成立。2026-03-15 已 fresh rerun recipe preflight、opt-in GPU E2E 和 `run_mopd_qwen3_4b.sh`。

10. **“完整闭环面 `18/18` 还没有被一条已完成 run 证明”**
   - 不成立。2026-03-15 的第三条更保守 rerun 已显示 `Training Progress: 100%|...| 18/18`，
     并打印 `validation generation end` 与 `Final validation metrics`。

---

## Remaining Limitations

为了让这份文档只反映真实状态，还需要把以下限制写清楚，而不是模糊成“已完成”：

1. `TeacherConfig.weight` 当前仍不是 runtime 能力；非默认值会 fail-fast。
2. `TeacherConfig.base_model_path` 虽在 schema 中存在，但当前 runtime 只支持全局 shared base，per-teacher 值会 fail-fast。
3. `sequence_reward` 当前不是通用任意 backend 能力，而是绑定到 dedicated quantized teacher worker 路径。
4. `MOPDConfig` 的 runtime typed 校验已经接入 `AlgoConfig.__post_init__` 和 `validate_config()`，但 `AlgoConfig.mopd` 的注解仍是 `Any`，因此 Hydra composition-time strictness 仍弱于 actor/critic。
5. 当前单机 4 卡默认 recipe 路径和更保守的 `18/18` 长程 rerun 都已经 fresh 跑通，但关停阶段仍可见
   `resource_tracker` 噪声；它们目前更像后处理噪声而不是主链路 blocker。
6. 当前验证重点已经扩展到 unit/integration/preflight/GPU E2E/full recipe run/`18/18` long-run closure/
   真实权重量化 teacher profiling；多节点、故障恢复、可重复的长程质量收益，以及 real-weight
   `sequence_reward` 路径仍需单独验证。

---

## Bottom Line

如果只用一句话概括当前分支：

**MOPD 已经从早期“双教师定制实现”演进为标准 PPO 主链路内的可配置多教师蒸馏 runtime，核心能力集中在 `algorithm.mopd` 配置、`RayPPOTrainer` 的 trainer-side teacher orchestration、`HFQuantizedTeacherWorker`、以及扩展后的 `compute_mopd_advantage()`。**

因此，这份 summary 的正确写法不应再按“计划实现什么”组织，而应按“当前 worktree 已真实接线到哪里、哪些字段/路径仍未真正生效”组织。
