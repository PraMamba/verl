# MOPD Implementation Changes Summary

**Date**: 2026-03-17
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
- 独立 `single_teacher_reverse_kl` baseline estimator
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
- paired single-teacher reduction harness
- fresh single-node runtime evidence across preflight / GPU E2E / default `6/6` recipe run /
  conservative `18/18` long-run rerun
- fresh single-teacher reduction proof surface:
  tensor-level exact equivalence tests plus a matched 3-way `data.seed` runtime comparison
  with no consistent one-sided offset, but still mixed training-trajectory evidence
- fresh zero-teacher reduction proof surface:
  tensor-level exact equivalence against `grpo` including
  `norm_adv_by_std_in_grpo=False`, zero-teacher runtime dependency tests under
  `algorithm.mopd.enabled=False`, paired harness execution on GPUs `0-3`, and a
  4-step stochastic smoke rerun with mean absolute score delta `0.02734375`
- fresh teacher-order invariance smoke evidence:
  dedicated paired harness, balanced-batch preparation, and a GPUs `0-3`
  declared-vs-reversed `teachers[]` smoke rerun showing identical step horizon,
  identical `0.5 / 0.5` teacher sample fractions, identical IS mask health, and
  no slot-hardcoding symptom; trajectory-level metrics remain supportive but
  still smoke-scale rather than long-horizon curve-overlap evidence
- checkpoint completeness markers, conservative auto-resume selection, and one fresh actor-death recovery drill

但当前代码也明确 **还不等于**：

- 按 `TeacherConfig.weight` 做运行时多教师加权聚合
- per-teacher base model
- Hydra composition-time typed `AlgoConfig.mopd` 强约束
- 可重复的长程质量收益、多节点容错，以及除 actor-death 以外 failure matrix 的充分验证

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

#### 9.3 checkpoint 完整性与 resume 选择逻辑

本次跟进里，checkpoint / resume 语义不再只停留在 manifest：

- 同步 checkpoint 现在会写 `checkpoint.complete`，并用原子替换方式发布 `latest_checkpointed_iteration.txt`
- `resume_mode=auto` 不再盲信 tracker；如果 tracker 缺失或指向不完整目录，会扫描并回退到最新 complete checkpoint
- 无 marker 的 async checkpoint 只在非常保守的条件下接受：
  - 它必须正好是 tracker 指向的那一步
  - 只能有一个 async role
  - role 目录里必须看到 finalize artifacts（`dist_ckpt/`、`huggingface/`、`transformer_config.json`）
- 一旦 `critic.checkpoint.async_save=True`，当前实现会跳过提前写 marker / tracker，避免发布半成品 checkpoint

这部分把“checkpoint 是否完整”“resume 到底该选哪个目录”的语义真正接进了主链路，而不是继续依赖外部脚本猜测。

#### 9.4 per-teacher metrics

trainer 已经会按 teacher 输出指标，例如：

- `mopd/<teacher>/sample_fraction`
- `mopd/<teacher>/adv_mean`
- `mopd/<teacher>/adv_std`
- `mopd/<teacher>/reverse_kl_mean`
- `mopd/<teacher>/seq_reward_mean`
- `mopd/<teacher>/is_valid_fraction`

#### 9.5 fit teardown 已有明确收口，并已在真实运行闭环中跑通

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

随后又补上了针对 Teardown Noise 的根因修复：vLLM 路径现在对 `shared_memory`
安装定向 `resource_tracker` bypass，并通过 `sitecustomize.py` 与 worker subprocess
入口覆盖 `spawn` 子进程。修复后的最新真实 shell rerun
`model_training_20260316_10.log` 已到达 `step:6 ... training/global_step:6`、`Final validation metrics`
与最终 swanlab footer，而且全文不再出现 `resource_tracker`、`KeyError`、
`process died unexpectedly` 或 `Traceback`。

因此，当前需要如实记录的状态已经变成：

- cleanup 主路径已经闭环
- 默认 full recipe shell path 上的 teardown noise 已被最新真实 rerun 证明消失
- 如果还想补更强证据，剩下的是 post-fix GPU E2E / `18/18` long-run 的覆盖面问题，而不是主链路 cleanup 失效

### 10. 单教师归约证明已经进入仓库，并完成了 3-way `data.seed` 运行对照

当前分支已经不再只有“公式看起来对”的 unit-test 级证据，还新增了一层 repository-local 的
reduction proof：

- 独立 baseline estimator：`single_teacher_reverse_kl`
- paired harness：`recipe/mopd/run_single_teacher_reduction.py`
- reduction results doc：
  `docs/plans/2026-03-16-mopd-single-teacher-reduction-results.md`

这层 proof 的目标是验证：

- 单教师
- `tokenizer_policy=compatible`
- `orm_weight=0.0`
- `is_correction=False`
- `use_base_normalization=False`
- zero-reward harness

时，`mopd` 会退化成独立的 on-policy reverse-KL baseline，而不是掺入别的运行时行为。

当前已经落地的 supporting changes 包括：

- 显式-token student 默认值修正，避免 reduction harness 错用 plain HF-cache tokenizer
- reduction harness 默认关闭不必要的 checkpoint / validation I/O
- 真实 `console.log` 前缀行的 step parser 修复
- 对 production recipe 默认 student path 的回归测试加强

官方结果在
`/gpfs/Mamba/Project/Single_Cell/Training/MOPD-single-teacher-reduction-20260316-run1`
下完成。三组 final `mopd - single_teacher_reverse_kl` deltas 分别为：

- seed 42: `-0.548263`
- seed 43: `+0.206836`
- seed 44: `+0.099413`

聚合结果：

- mean final delta: `-0.080671`
- mean absolute final delta: `0.284837`
- max absolute final delta: `0.548263`
- mean relative absolute final delta vs baseline magnitude: `11.62%`

这部分证据应分成两层理解：

- 算法层：`tests/unit/test_mopd_advantage.py` 已经证明 reduced single-teacher `mopd`
  在同一 batch 张量上会退化成独立 reverse-KL baseline
- 运行层：这轮 matched 3-way `data.seed` 对照没有看到稳定的单边系统性偏移，但训练曲线
  也还没有紧到可以宣称“经验上几乎完全重合”

所以更准确的说法是：
**single-teacher reduced MOPD 的算法级归约已经在仓库内被证明，而训练轨迹级证据目前是
non-systematic but not yet tightly aligned。**

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

- `pytest -q ...` -> **123 passed, 1 skipped, 1 warning in 11.45s**
- `pytest --collect-only -q ...` -> **124 tests collected in 7.22s**
- `pytest -q tests/unit/test_teardown_cleanup.py` -> **7 passed, 3 warnings in 9.53s**
- `bash recipe/mopd/run_mopd_qwen3_4b_preflight.sh` -> **exit 0**, success boundary at `training/global_step:1`
- `VERL_MOPD_E2E=1 pytest tests/integration/test_mopd_e2e.py::test_mopd_training_e2e -v` -> **1 passed, 1 warning in 345.91s**
- `bash recipe/mopd/run_mopd_qwen3_4b.sh` -> **exit 0**, `Training Progress: 100%|...| 6/6`, `training/global_step:6`
- post-fix `model_training_20260316_10.log` review -> 到达 `step:6 ... training/global_step:6`、
  `Final validation metrics` 与最终 swanlab footer，且无 `resource_tracker` / `KeyError` /
  `process died unexpectedly` / `Traceback`
- conservative long-run rerun -> **exit 0**, `Training Progress: 100%|...| 18/18`,
  `training/global_step:18`, `validation generation end`, `Final validation metrics`
- actor-death fault-recovery drill -> 初次 run 因 `ActorDiedError` 退出，删除 tracker 后仍自动回退到
  `ckpts/global_step_1`，恢复 run 到达 `Training Progress: 100%|...| 2/2` 与 `training/global_step:2`
- reduction-focused regression slice ->
  **62 passed, 1 warning in 8.57s**
- official 3-way `data.seed` reduction experiment -> 六条 run 全部到达 `training/global_step:12`，
  final deltas 为 `-0.548263` / `+0.206836` / `+0.099413`，没有看到稳定单边偏移，但轨迹仍偏 mixed
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
- checkpoint completeness / tracker publication / auto-resume fallback
- preflight command generation
- production run-script self-checks
- trainer finalization / teardown helper behavior
- opt-in GPU E2E contract
- recipe-aligned preflight shell path
- full recipe shell path through final checkpoint write
- conservative long-run closure through final validation at `18/18`

本次刷新确认的 suite inventory 为：

- **124 tests collected**
- **123 passed**
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

11. **“fault recovery 这次完全没有 fresh 证据”**
   - 不成立。2026-03-15 已完成一轮单机 actor-death 注入演练：初次 run 因 `ActorDiedError` 退出，
     强制删除 `latest_checkpointed_iteration.txt` 后，仍自动回退到 `global_step_1` 的 complete checkpoint，
     并恢复到 `training/global_step:2`。

12. **“resume 仍然只能盲信 `latest_checkpointed_iteration.txt`”**
   - 不成立。当前同步 checkpoint 会写 `checkpoint.complete`；`resume_mode=auto` 会优先选择最新 complete
     checkpoint，而无 marker 的 async checkpoint 只在受限条件下保守接受。

13. **“当前还没有单教师归约证明，只能从公式和局部测试推断算法是对的”**
    - 不成立。当前仓库已经有独立 `single_teacher_reverse_kl` baseline、paired reduction harness，
      `tests/unit/test_mopd_advantage.py` 中的 tensor-level reduction tests，以及 2026-03-16 的官方
      3-way `data.seed` reduction 结果文档。需要保留的 caveat 是：运行轨迹级证据仍是 mixed，
      不是“曲线完全重合”的强经验性证明。

14. **“把 teacher signal 全关后，当前实现可能还残留 ghost teacher effect，而且仓库里没有归约证据”**
    - 部分不成立。当前仓库已经有独立 `mopd_zero_teacher_orm_only` reduction estimator、
      paired harness、same-batch exact equivalence tests、`need_reference_policy()` /
      trainer preflight 的零 teacher runtime tests（前提是 `algorithm.mopd.enabled=False`），以及
      2026-03-17 在 GPUs `0-3` 上完成的
      fresh 4-step smoke rerun。需要保留的 caveat 是：这轮 runtime 证据仍是 smoke-level，
      而且是 sequential stochastic comparison，不是长程“曲线几乎完全重合”的最终经验性证明。
      独立结果记录见 `docs/plans/2026-03-17-mopd-zero-teacher-reduction-results.md`。

15. **“`teachers[]` 的 YAML 声明顺序是否仍然在隐式决定 teacher 槽位，当前没有仓库内实验证据”**
    - 不成立。2026-03-17 已在 GPUs `0-3` 上完成 paired order-permutation smoke rerun：
      同一配置仅打乱 `algorithm.mopd.teachers[]` 声明顺序，并先修复了一个非算法性的
      teacher sub-batch balancing 形状问题（原始数据里某 teacher 子批出现 `5 % 4 != 0`）。
      平衡化 rerun 后，两条 run 都到达 `training/global_step=4`，并在每一步保持相同的
      `0.5 / 0.5` teacher sample fraction、`mopd/is_valid_fraction=1.0`、`mopd/is_zeroed_fraction=0.0`，
      没有出现 teacher 指标槽位交换症状。需要保留的 caveat 是：这仍是 4-step sequential
      stochastic smoke，不是长程“曲线几乎完全重合”的最终经验性证明。独立结果记录见
      `docs/plans/2026-03-17-mopd-teacher-order-invariance-results.md`。

---

## Remaining Limitations

为了让这份文档只反映真实状态，还需要把以下限制写清楚，而不是模糊成“已完成”：

1. `TeacherConfig.weight` 当前仍不是 runtime 能力；非默认值会 fail-fast。
2. `TeacherConfig.base_model_path` 虽在 schema 中存在，但当前 runtime 只支持全局 shared base，per-teacher 值会 fail-fast。
3. `sequence_reward` 当前不是通用任意 backend 能力，而是绑定到 dedicated quantized teacher worker 路径。
4. `MOPDConfig` 的 runtime typed 校验已经接入 `AlgoConfig.__post_init__` 和 `validate_config()`，但 `AlgoConfig.mopd` 的注解仍是 `Any`，因此 Hydra composition-time strictness 仍弱于 actor/critic。
5. 当前单机 4 卡默认 recipe 路径和更保守的 `18/18` 长程 rerun 都已经 fresh 跑通；而且历史上的
   `resource_tracker` 关停噪声已在 2026-03-16 的最新 full recipe rerun 上消失。若还要补更强证据，
   重点已变成 post-fix GPU E2E / `18/18` rerun 的覆盖面，而不是 teardown 根因未修。
6. 当前验证重点已经扩展到 unit/integration/preflight/GPU E2E/full recipe run/`18/18` long-run closure/
   真实权重量化 teacher profiling，以及单机 actor-death recovery drill；多节点、NCCL timeout/OOM/segfault、
   async-save real resume、可重复的长程质量收益，以及 real-weight `sequence_reward` 路径仍需单独验证。
7. 单教师归约已经补上 repository-local proof；接下来真正剩下的正确性问题，优先级已经从
   “single-teacher 是否跑偏” 转向 “2-teacher / n-teacher composition、routing 与 heterogeneous tokenizer
   场景是否仍保持语义正确”。
8. zero-teacher ORM-only 归约现在也已有 repository-local proof surface，但当前 runtime 证据
   仍只到 4-step smoke rerun；若要写成更强经验性结论，仍需补更长程的 paired comparison。
9. `teachers[]` 顺序不变性现在也已有 repository-local smoke evidence；当前最强结论是
   “结构级不变性已有 paired smoke support、未见 slot-hardcoding 症状”，若要写成更强经验性结论，
   仍需补更长程 paired comparison。

---

## Bottom Line

如果只用一句话概括当前分支：

**MOPD 已经从早期“双教师定制实现”演进为标准 PPO 主链路内的可配置多教师蒸馏 runtime，核心能力集中在 `algorithm.mopd` 配置、`RayPPOTrainer` 的 trainer-side teacher orchestration、`HFQuantizedTeacherWorker`、以及扩展后的 `compute_mopd_advantage()`。**

因此，这份 summary 的正确写法不应再按“计划实现什么”组织，而应按“当前 worktree 已真实接线到哪里、哪些字段/路径仍未真正生效”组织。
