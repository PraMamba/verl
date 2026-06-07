# verl 训练精度对齐体系源码与 Commit History 深度分析

## 0. Executive Summary

| 维度 | 结论 |
|------|------|
| **项目名称** | verl (volcengine/verl) |
| **总体判断** | **较完整** — 具备系统性精度对齐的主干能力，但缺少 operator-level 对齐工具和 PPO loss golden regression |
| **最强能力** | ① 多后端 SFT golden loss CI 回归（FSDP/Megatron/VeOmni 5+ 配置 vs 单 GPU baseline）；② 完整的 RNG 状态保存/恢复与 checkpoint round-trip bitwise 验证；③ 全面的 `full_determinism` 开关（覆盖 CUDA、cuBLAS、FlashAttn、NPU） |
| **最大短板** | ① 无 operator-level / layer-level activation/gradient dump + compare 工具（msprobe 仅限 Ascend）；② PPO 训练无 golden loss 回归 CI（`check_results.py` 存在但未被调用）；③ 无跨并行策略 correctness matrix（如 DP=1 vs DP=4 vs TP=2 的逐层对比） |
| **最值得借鉴的源码模块** | `verl/workers/engine/utils.py:enable_full_determinism`; `tests/special_e2e/sft/compare_sft_engine_results.py`; `tests/special_distributed/test_fsdp_ckpt.py`; `verl/utils/checkpoint/megatron_checkpoint_manager.py` (RNG save/restore); `verl/trainer/ppo/rollout_corr_helper.py` (数值稳定 IS 权重) |
| **最值得研究的 commits/PR** | #4378 (full_determinism), #3363 (SFT accuracy CI), #3694 (rollout IS correction), #2646 (vLLM seed bug), #6068 (SP fused kernel label fix), #5186 (msprobe integration), #1779 (KL clamping), #789 (gradient overflow skip) |
| **是否适合作为训练精度对齐基础设施参考** | **是，特别适合学习"如何在多后端/多并行策略框架中做精度回归"**。但如需建设完整的 operator-level 对齐体系，需要额外补充 tensor dump/compare 工具链 |

---

## 1. 项目训练流程与精度相关架构总览

### 1.1 训练主入口

| 入口 | 文件 | 说明 |
|------|------|------|
| PPO/GRPO/RLOO 训练 | `verl/trainer/main_ppo.py` → `TaskRunner.run()` | Hydra 入口，启动 `RayPPOTrainer` |
| 同步 PPO 训练 | `verl/trainer/main_ppo_sync.py` | 同步模式 PPO |
| SFT 训练 | `verl/trainer/fsdp_sft_trainer.py` | FSDP SFT trainer |
| 配置系统 | `verl/trainer/config/` (Hydra YAML) + `verl/workers/config/` (dataclass) | 分层配置 |

### 1.2 训练 step 精度关键路径

```
Controller (CPU) 
  → DataLoader (seed-controlled shuffle)
    → Actor Worker forward (mixed precision autocast)
      → compute_log_prob (fused kernel / torch)
        → rollout correction IS weights (log-space, SAFETY_BOUND=20)
          → advantage estimation (GAE/GRPO/RLOO)
            → policy_loss + value_loss (clamped KL, overflow skip)
              → backward (micro-batch gradient accumulation)
                → optimizer.step (ShardedGradScaler for fp16)
                  → checkpoint save (model + optimizer + RNG state)
                    → weight sync to rollout (NCCL broadcast, allclose verify)
```

### 1.3 配置系统如何控制精度相关参数

- **seed**: `verl/workers/config/engine.py:117` → `seed: int = 42` (所有 engine 共享)
- **full_determinism**: `verl/workers/config/engine.py:114` → `full_determinism: bool = False`
- **mixed_precision**: `verl/workers/engine/fsdp/transformer_impl.py:342-366` → `MixedPrecision(param_dtype, reduce_dtype, buffer_dtype)`
- **data_loader_seed**: `verl/workers/config/actor.py:178`, `critic.py:91` → `data_loader_seed: int = 42`
- **rollout seed**: `verl/workers/config/rollout.py` → per-replica offset: `replica_rank + seed`

---

## 2. 精度对齐能力矩阵

| 能力项 | 是否具备 | 源码证据 | Commit/PR 证据 | 成熟度 | 备注 |
|--------|---------|---------|----------------|--------|------|
| **配置一致性扫描** | 间接存在 | engine.py 共享 seed/full_determinism 字段 | — | 1 | 无专门 config diff/snapshot 工具 |
| **随机种子/RNG 控制** | ✅ 明确存在 | `engine/utils.py:29-55` (enable_full_determinism), checkpoint_manager (RNG save/restore) | #4378 | 4 | 覆盖 Python/NumPy/Torch/CUDA/cuBLAS/FlashAttn/NPU + checkpoint 恢复 |
| **数据加载顺序确定性** | ✅ 明确存在 | `tensordict_utils.py:559` (generator.manual_seed), `protocol.py:821` (DataProto.make_iterator seed), `dataset/rl_dataset.py:154` (HF shuffle seed) | #4156 (resume epoch fix), #3815 (data.seed config) | 3 | 包含 StatefulDataLoader resume |
| **初始权重一致性** | ✅ 间接存在 | 通过 `model.path` 从 HuggingFace checkpoint 加载，所有 worker 共享同一模型路径 | — | 2 | 依赖 HF from_pretrained 保证一致性 |
| **单步 forward loss 对齐** | ✅ 明确存在 | `tests/special_e2e/sft/compare_sft_engine_results.py:38` (golden loss compare), `tests/models/test_engine.py:195` (HF vs mcore logprob) | #3363 | 3 | SFT 有 CI golden；PPO 无 |
| **activation dump/compare** | 部分存在 | `verl/utils/profiler/precision_debugger_profile.py` (msprobe PrecisionDebugger) | #5186 | 2 | 仅限 Ascend NPU (msprobe 依赖)；无通用 activation dump |
| **gradient dump/compare** | 间接存在 | `compare_sft_engine_results.py:39` (grad_norm compare), `test_transformers_ulysses.py:275-279` (SP gradient compare) | — | 2 | 仅 grad_norm 或特定测试的梯度比较，无通用 gradient dump 工具 |
| **optimizer state 对齐** | ✅ 明确存在 | `fsdp_checkpoint_manager.py:258-285` (ShardedOptimStateDictConfig save/load), `megatron_checkpoint_manager.py:481-492` | — | 3 | 通过 checkpoint round-trip 验证 |
| **scheduler/lr curve 对齐** | ✅ 明确存在 | checkpoint_manager 保存/恢复 lr_scheduler state_dict | — | 2 | 无独立 lr curve 回归测试 |
| **loss curve golden regression** | 部分存在 | SFT: `compare_sft_engine_results.py` (CI 中运行)；PPO: `check_results.py` 存在但**未被 CI 调用** | #3363 | 2 | SFT 成熟(3)；PPO 缺失(0) |
| **mixed precision 对齐** | ✅ 明确存在 | `transformer_impl.py:342-366` (FSDP MP), `megatron/optimizer.py:41` (fp16 loss scaling), `torch_dtypes.py` | #6150 (autocast dtype fix), #3068 (missing MP config), #4036 (FP16 support) | 3 | 覆盖 fp16/bf16/fp32；fp8 rollout 有，fp8 train 有 Megatron |
| **FP16/BF16/FP8 数值稳定性** | ✅ 明确存在 | `rollout_corr_helper.py:75` (SAFETY_BOUND=20), `core_algos.py:1583` (clamp max=10), `torch_functional.py:166` (logprobs_v2 bf16 fallback) | #1779 (KL clamp), #789 (overflow skip), #4223 (FP8 train) | 3 | log-space 计算 + 值域裁剪 + 溢出跳步 |
| **TF32 控制** | 间接存在 | `config_converter.py:372` (disable_bf16_reduced_precision_matmul) | — | 1 | 仅 DeepSeek MoE 模型禁用；无全局 TF32 开关 |
| **NaN/Inf/overflow 检测** | ✅ 明确存在 | `test_rollout_corr_integration.py:93-94` (assert not isnan/isinf), `fsdp_sft_trainer.py:093e9599` (grad overflow skip), `fba09392` (NaN for empty tensors), `21f4e490` (reward NaN assert) | #789, #2377, #5899, #5216 | 3 | 有检测+保护+跳步，但无系统化 NaN watchdog |
| **checkpoint resume 一致性** | ✅ 明确存在 | `test_fsdp_ckpt.py:138,153` (atol=0.0, rtol=0.0 bitwise match) | #4156 (resume epoch fix), #5725 (epoch boundary fix) | 4 | **bitwise 精确验证** + RNG 恢复 + CI |
| **data parallel correctness** | ✅ 明确存在 | loss normalization via all_reduce(SUM) of batch_num_tokens, `agg_loss` 校验 dp_size | — | 3 | 所有 engine 后端统一实现 |
| **tensor parallel correctness** | ✅ 明确存在 | `test_tensor_dict.py` (vocab_parallel_entropy forward+backward vs non-TP), `test_special_megatron_kl_loss_tp.py` (TP KL loss vs FSDP), `test_special_linear_cross_entropy_tp.py` | — | 3 | 有 CI GPU 测试 |
| **pipeline parallel correctness** | 部分存在 | `test_pipeline_parallel.py` (batch generator), MTP loss deadlock fix (#5895) | #5895 | 2 | 有 batch 结构测试；无 PP 数值精度对比测试 |
| **sequence parallel correctness** | ✅ 明确存在 | `test_transformers_ulysses.py` (SP fwd+bwd vs non-SP), `test_fused_kernels_ulysses_sp.py` (regression for #6068) | #6068/#6268 | 3 | 有 CI GPU 测试 + 专门的回归测试 |
| **expert parallel / MoE correctness** | 部分存在 | MoE router replay (R2/R3), `config_converter.py:371` (fp64 router), EP config validation assertions | #6325 (VeOmni router replay), #5452 (hybrid dense/MoE PP fix) | 2 | 有功能实现，但无 EP 数值精度对比测试 |
| **collective communication correctness** | 间接存在 | `test_torch_functional.py` (allgather_dict), `test_tensor_dict.py` (all_gather_data_proto), MTP deadlock fix | #5895, #5591 (dp_group deadlock fix) | 2 | 通信功能正确性测试；无通信数值一致性验证 |
| **CI 精度回归测试** | ✅ 明确存在 | `e2e_sft_llm.yml` (SFT golden compare), `model.yml` (SP/TP/FSDP ckpt tests), `gpu_unit_tests.yml` (linear cross entropy TP) | #3363 | 3 | SFT 体系化；PPO 缺失 |
| **自动化二分定位能力** | 未发现 | — | — | 0 | 无 bisect 脚本或工具 |
| **跨硬件/跨后端对齐能力** | ✅ 明确存在 | SFT golden: FSDP/FSDP2/Megatron/VeOmni 全部 vs 单 GPU golden | #3363 | 3 | **这是 verl 最突出的精度对齐能力** |

---

## 3. 源码证据地图

### 3.1 随机性控制

| 控制层 | 文件:行号 | 函数/类 | 覆盖范围 |
|--------|----------|---------|---------|
| 全局确定性开关 | `verl/workers/engine/utils.py:29-55` | `enable_full_determinism(seed)` | Python hash, cuBLAS, FlashAttn, CUDNN, NPU HCCL |
| 引擎配置 | `verl/workers/config/engine.py:114-117` | `EngineConfig.seed=42`, `full_determinism=False` | 所有引擎后端共享 |
| 各引擎初始化调用 | `fsdp/transformer_impl.py:136`, `automodel/transformer_impl.py:105`, `torchtitan/transformer_impl.py:185`, `veomni/transformer_impl.py:140` | 各 engine `__init__` | 在 `full_determinism=True` 时调用 |
| Megatron 种子 | `megatron/utils.py:19-35` | `set_random_seed(seed)` | 含 `tensor_parallel.model_parallel_cuda_manual_seed` |
| DataLoader 种子 | `tensordict_utils.py:559,593-595` | `make_iterator(seed=...)` | PyTorch DataLoader generator seed |
| vLLM rollout 种子 | `vllm_async_server.py:271` | per-replica seed offset | `replica_rank + seed` 避免所有副本相同 |
| Dataset shuffle 种子 | `dataset/rl_dataset.py:154,184` | HuggingFace dataset.shuffle(seed) | 数据集层确定性 |
| Ulysses SP 种子偏移 | `verl/utils/ulysses.py:371` | `self.seed_offset = 12345` | SP rank 间 RNG 偏移 |
| FlashAttn 确定性 | `models/transformers/qwen2_vl.py:49-61` | `FLASH_ATTENTION_DETERMINISTIC` env var + `deterministic=` flag | 模型级 |

### 3.2 Checkpoint RNG 状态保存/恢复

| 操作 | 文件:行号 | 内容 |
|------|----------|------|
| FSDP RNG save | `checkpoint/checkpoint_manager.py:175-194` | `get_rng_state()`: torch + numpy + random + CUDA RNG |
| FSDP RNG load | `checkpoint/fsdp_checkpoint_manager.py:197` | `load_rng_state(extra_state_dict["rng"])` |
| Megatron RNG save | `checkpoint/megatron_checkpoint_manager.py:226-263` | 额外保存 `rng_tracker_states` (Megatron TP CUDA RNG tracker) + all_gather across DP |
| Megatron RNG load | `checkpoint/megatron_checkpoint_manager.py:528-559` | 恢复含 `tensor_parallel.get_cuda_rng_tracker().set_states()` |
| Activation ckpt RNG | `models/mcore/patch.py:519,535` | `_set_all_rng_states` in activation checkpointing recompute |

### 3.3 Mixed Precision 实现

| 组件 | 文件:行号 | 实现 |
|------|----------|------|
| FSDP MixedPrecision | `fsdp/transformer_impl.py:342-366` | `param_dtype=bf16, reduce_dtype=fp32, buffer_dtype=fp32` (默认) |
| FSDP2 MixedPrecisionPolicy | `fsdp/transformer_impl.py:410-412` | `MixedPrecisionPolicy(param_dtype, reduce_dtype, cast_forward_inputs=True)` |
| fp16 GradScaler | `fsdp/transformer_impl.py:359-366` | `ShardedGradScaler(growth_interval=400)` for fp16 |
| Megatron fp16 | `megatron/optimizer.py:41-43` | `initial_loss_scale=32768, min_loss_scale=1` |
| autocast dtype 修复 | `fsdp/transformer_impl.py` (#6150) | `self._autocast_dtype` from config instead of hardcoded bf16 |
| MoE router fp64 | `mcore/config_converter.py:371` | `moe_router_dtype="fp64"` for DeepSeek 数值稳定性 |
| logprobs bf16 fallback | `torch_functional.py:166-200` | `logprobs_from_logits_v2`: bf16 用 `log_softmax` 而非 `logsumexp` |

### 3.4 数值稳定性保护

| 保护机制 | 文件:行号 | 细节 |
|---------|----------|------|
| IS 权重 SAFETY_BOUND | `rollout_corr_helper.py:75` | `SAFETY_BOUND=20.0` 裁剪 log-ratio 防 exp 溢出 |
| KL divergence clamping | `core_algos.py:1583,2029,2179` | `torch.clamp(log_ratio, max=10.0)` |
| Gradient overflow skip | `093e9599` (v0.x) | `if isinf(grad_norm): skip optimizer.step()` |
| NaN masked mean | `5cbad837` | `torch.where(mask, x, 0)` before mean to handle NaN outside mask |
| Reward NaN assert | `21f4e490` | `assert overlong_cfg.len > 0` 防除零产生 NaN |
| Empty tensor NaN | `fba09392` | 空 tensor 返回 NaN + warning 而非 crash |

### 3.5 SFT 多后端 Golden 对齐系统

**核心文件**: `tests/special_e2e/sft/`
- `test_sft_engine_all.sh`: 运行 7+ 配置（单 GPU golden, FSDP SP=2, FSDP no_rmpad, FSDP2 SP=2, VeOmni SP=2, Megatron TP2/PP2/VPP2/CP2 SPMD/Ray）
- `compare_sft_engine_results.py`: 读 golden.jsonl，与所有后端 .jsonl 比较 `train/loss` (atol=1e-2, rtol=1e-2) 和 `train/grad_norm` (atol=1e-4, rtol=3e-2)
- **CI**: `.github/workflows/e2e_sft_llm.yml` 在每次 PR/push 到 main 时触发

### 3.6 Checkpoint Round-Trip 验证

**核心文件**: `tests/special_distributed/test_fsdp_ckpt.py`
- 流程: step1 训练 → 保存 checkpoint → step2 训练 → 记录 logits → 加载 step1 checkpoint → 重做 step2 → 比较 logits
- 断言: `atol=0.0, rtol=0.0` (**bitwise 精确**)
- 前置: `FLASH_ATTENTION_DETERMINISTIC=1`
- **CI**: `.github/workflows/model.yml` 中通过 `torchrun --nproc_per_node=8` 运行

### 3.7 Precision Debugger (msprobe)

**核心文件**: `verl/utils/profiler/precision_debugger_profile.py`
- 类: `PrecisionDebuggerProfiler`
- 功能: 对每个训练阶段 (actor_update, critic_update, ref_compute_log_prob 等) 捕获 msprobe activation/gradient dump
- 输出: `outputs/profile/step_{N}/{stage}/`
- 局限: **依赖 `msprobe` 库，仅 Ascend NPU 可用**

---

## 4. Commit / PR / Issue 历史演进时间线

### 时间线 1: 2025-03 — 梯度溢出跳步保护

| 项目 | 内容 |
|------|------|
| **时间** | 2025-03-29 |
| **Commit/PR** | `093e9599` / #789 |
| **涉及文件** | `verl/trainer/fsdp_sft_trainer.py`, `verl/workers/actor/dp_actor.py`, `verl/workers/critic/dp_critic.py` |
| **问题背景** | 混合精度训练或脏数据导致 gradient overflow，模型训练崩溃（关联 issues #637, #747, #751） |
| **修改内容** | 在 optimizer.step 前检查 grad_norm，如果 inf 则跳过更新步 |
| **新增测试** | 无 |
| **影响范围** | SFT + Actor + Critic 的 update 路径 |
| **启发** | 最早期的数值稳定性防护，"先保护 → 后排查"的工程策略 |

### 时间线 2: 2025-06 — KL 散度数值稳定化

| 项目 | 内容 |
|------|------|
| **时间** | 2025-06-14 |
| **Commit/PR** | `2c85b432` / #1779 |
| **涉及文件** | `verl/trainer/ppo/core_algos.py` |
| **问题背景** | PPO 训练中极端 log-probability 差异导致 KL 散度爆炸，引发梯度不稳定 |
| **修改内容** | `negative_approx_kl` 和 `low_var_kl` 裁剪到 [-10, 10] |
| **新增测试** | 无直接回归测试 |
| **影响范围** | PPO policy loss 计算 |
| **启发** | 对指数运算输入进行值域裁剪是通用数值稳定性模式 |

### 时间线 3: 2025-07 — NaN 安全 masked mean + vLLM seed 重大修复

| 项目 | 内容 |
|------|------|
| **时间** | 2025-07-07 / 2025-07-21 |
| **Commit/PR** | `5cbad837` / #2377 (NaN), `3f6cd479` / #2646 (seed) |
| **涉及文件** | `core_algos.py` (NaN), `vllm_rollout_spmd.py` (seed) |
| **问题背景** | (1) mask 外有 NaN 值传染到 masked_mean 结果；(2) vLLM SamplingParams 意外继承 engine seed 导致所有 sample 确定性（丧失探索能力） |
| **修改内容** | (1) `torch.where(mask, x, 0)` 在 mean 前清除 mask 外 NaN；(2) 将 seed 仅传给 engine 初始化，不传给 SamplingParams |
| **新增测试** | 无 |
| **影响范围** | (1) 所有 RL 算法的 advantage/reward 计算；(2) vLLM 采样多样性 |
| **启发** | seed 在不同层级的含义不同：engine 初始化 seed ≠ 采样 seed |

### 时间线 4: 2025-08 — NPU tensor 传输精度修复

| 项目 | 内容 |
|------|------|
| **时间** | 2025-08-28 |
| **Commit/PR** | `cc2799b2` / #3222 |
| **涉及文件** | NPU tensordict `.to("cpu")` 操作 |
| **问题背景** | tensordict 的 `.to()` 操作是 non-blocking 的，在 NPU 上数据未完全传输就被 CPU 使用，导致精度异常 |
| **修改内容** | 在 NPU 设备上添加显式同步 |
| **新增测试** | 无 |
| **影响范围** | 所有 NPU 训练的 worker → controller 数据传输 |
| **启发** | 异步设备传输是精度对齐的隐蔽陷阱；tensordict 库自身已修复 CUDA/MPS 但遗漏了 NPU |

### 时间线 5: 2025-09 — SFT 多后端精度对齐 CI 建立

| 项目 | 内容 |
|------|------|
| **时间** | 2025-09-06 |
| **Commit/PR** | `7bc70bbf` / #3363 |
| **涉及文件** | `tests/special_e2e/sft/`, `.github/workflows/e2e_sft_llm.yml` |
| **问题背景** | 多后端 (FSDP, Megatron, VeOmni) SFT 训练需要保证数值一致性 |
| **修改内容** | 建立单 GPU golden baseline → 多后端 compare 的 CI 流程 |
| **新增测试** | `compare_sft_engine_results.py` + `test_sft_engine_all.sh` |
| **影响范围** | SFT 训练全流程 |
| **启发** | **这是 verl 精度对齐体系的里程碑**：首次实现了"单卡 golden → 多后端 CI 回归"的完整闭环 |

### 时间线 6: 2025-10 — Rollout 重要性采样框架

| 项目 | 内容 |
|------|------|
| **时间** | 2025-10-13 |
| **Commit/PR** | `21271aab` / #3694 |
| **涉及文件** | `verl/trainer/ppo/rollout_corr_helper.py`, `core_algos.py` |
| **问题背景** | Rollout 引擎 (vLLM BFloat16) 和 Training 引擎 (FSDP FP32) 之间的策略分布不匹配导致训练不稳定 |
| **修改内容** | 实现 importance sampling (IS) 权重修正，log-space 计算 + SAFETY_BOUND=20 防溢出 |
| **新增测试** | `test_rollout_corr.py`, `test_rollout_corr_integration.py` (含 exact expected values) |
| **影响范围** | PPO/GRPO 所有 RL 算法的 policy loss |
| **启发** | Rollout-Training precision mismatch 是大模型 RL 训练特有的精度问题，IS 修正 + 数值安全计算是系统性解决方案 |

### 时间线 7: 2025-11 — FP16 训练 + Megatron FP8 训练

| 项目 | 内容 |
|------|------|
| **时间** | 2025-11-13 / 2025-11-21 |
| **Commit/PR** | `65f9badf` / #4036 (FP16), `b05f8f64` / #4223 (FP8) |
| **涉及文件** | `fsdp/transformer_impl.py`, `megatron/transformer_impl.py` |
| **问题背景** | 支持 FP16 训练（对齐 Precision-RL 论文）和 Megatron FP8 训练 |
| **修改内容** | FP16: `ShardedGradScaler` + fp16 autocast；FP8: Megatron TransformerEngine fp8 config |
| **新增测试** | FP16: `test_engine.py` 中 GradScaler 断言 |
| **影响范围** | 训练精度策略选项扩展 |
| **启发** | 不同精度策略需要不同的数值保护（fp16 需 loss scaling，fp8 需 delayed scaling） |

### 时间线 8: 2025-12 — Full Determinism 开关

| 项目 | 内容 |
|------|------|
| **时间** | 2025-12-02 |
| **Commit/PR** | `fb860f0b` / #4378 |
| **涉及文件** | `verl/workers/engine/utils.py`, `verl/workers/config/engine.py`, 所有 engine `__init__` |
| **问题背景** | 训练调试需要完全可重复性 |
| **修改内容** | 新增 `enable_full_determinism()` 函数，覆盖 Python/NumPy/Torch/CUDA/cuBLAS/FlashAttn/CUDNN/NPU；新增 `full_determinism` config 开关 |
| **新增测试** | 通过 `test_fsdp_ckpt.py` 使用 `FLASH_ATTENTION_DETERMINISTIC=1` |
| **影响范围** | 所有训练后端 |
| **启发** | Determinism 开关的设计模式：默认关闭（性能优先），调试时开启，覆盖所有 RNG 源 |

### 时间线 9: 2026-02 — vLLM per-replica seed 修复

| 项目 | 内容 |
|------|------|
| **时间** | 2026-02-02 |
| **Commit/PR** | `82cf2ddc` / #5179, 后续 `15371c91` / #5181 |
| **涉及文件** | `vllm_async_server.py`, `vllm_rollout_spmd.py` |
| **问题背景** | 所有 vLLM replica 使用相同 seed，确定性模式下所有副本生成相同序列，丧失探索能力 |
| **修改内容** | `seed = replica_rank + config.seed` 给每个副本不同 seed |
| **新增测试** | 无专门回归测试 |
| **影响范围** | vLLM 多副本 rollout |
| **启发** | 多副本系统中 seed 不是全局唯一值，需要 per-rank offset |

### 时间线 10: 2026-04 — NCCL_DETERMINISTIC 幽灵环境变量修复 + msprobe 集成

| 项目 | 内容 |
|------|------|
| **时间** | 2026-04-09 / 2026-04-03 |
| **Commit/PR** | `23ede534` / #5923 (env vars), `b4c82633` / #5186 (msprobe) |
| **涉及文件** | `engine/utils.py` (env vars), `profiler/precision_debugger_profile.py` (msprobe) |
| **问题背景** | (1) `NCCL_DETERMINISTIC` 是不存在的幽灵环境变量，被 LLM 幻觉传播；(2) 缺少系统化的 activation/gradient dump 工具 |
| **修改内容** | (1) 移除 `NCCL_DETERMINISTIC`，保留 `HCCL_DETERMINISTIC`；(2) 集成 msprobe PrecisionDebugger 到 verl profiler 系统 |
| **新增测试** | 无 |
| **影响范围** | (1) 确定性配置正确性；(2) Ascend NPU 精度调试能力 |
| **启发** | (1) 环境变量需验证实际存在性，避免"cargo cult"配置；(2) 精度调试工具需要与训练 profiler 系统集成 |

### 时间线 11: 2026-05 — SP fused kernel label 修复（经典精度 bug → regression test 闭环）

| 项目 | 内容 |
|------|------|
| **时间** | 2026-05-14 |
| **Commit/PR** | `575d5a8a` / #6268 (fix for #6068) |
| **涉及文件** | `verl/models/transformers/*.py` 多个模型文件 |
| **问题背景** | `use_fused_kernels=True` + `ulysses_sequence_parallel_size > 1` 时，`torch.roll` 在 SP-sliced 的 local shard 边界 wrap，导致每个 SP rank 每个 micro-batch 有一个 label 错误 |
| **修改内容** | 使 fused kernels 尊重 engine 已经做好的 SP-rolled labels，而非重新 roll |
| **新增测试** | `tests/special_distributed/test_fused_kernels_ulysses_sp.py` — SP=2 vs SP=1 log_probs 对比 (atol=1e-3, rtol=1e-3) |
| **影响范围** | Ulysses SP + fused kernels 组合 |
| **启发** | **典型的 "bug → fix → regression test → CI 固化" 闭环案例**。SP shard 边界处的 label shift 是一个极其隐蔽的精度 bug |

### 时间线 12: 2026-04 — MTP loss deadlock with CP

| 项目 | 内容 |
|------|------|
| **时间** | 2026-04-10 |
| **Commit/PR** | `516657fa` / #5895 |
| **涉及文件** | `verl/workers/engine/megatron/transformer_impl.py` |
| **问题背景** | MTP (Multi-Token Prediction) loss 中 `get_megatron_mtp_loss` 内部 `all_reduce` 需要所有 CP rank 参与，但调用被 `cp_rank==0` 门控，非零 CP rank 永远不参与 → 死锁 |
| **修改内容** | 将门控从 `is_mp_src_rank_with_outputs()` 改为 `is_pipeline_last_stage(ignore_virtual=True)`，使所有 CP/TP rank 参与 all_reduce |
| **新增测试** | 无直接回归测试 |
| **影响范围** | Megatron CP + MTP |
| **启发** | 分布式 collective 的参与者范围必须严格对齐——少一个 rank 就死锁，多一个 rank 可能数值错误 |

---

## 5. 典型精度问题案例复盘

### Case 1: Checkpoint Resume 后 epoch 计算错误导致数据不可重复

| 项目 | 内容 |
|------|------|
| **现象** | 恢复训练后，数据顺序与首次训练不一致 |
| **根因** | `StatefulDataLoader` 只保存 epoch 内 step，不保存 epoch 号；恢复时 sampler 默认 epoch=0 |
| **如何定位** | 用户报告 issue #3457，发现恢复后数据 batch 内容与首次运行不同 |
| **如何修复** | PR #4156: 计算正确的 `current_epoch = global_steps // len(dataloader)` 并恢复 sampler 状态 |
| **是否新增测试** | 无专门回归测试 |
| **借鉴** | DataLoader 的确定性不仅需要 seed，还需要 epoch 状态的完整保存/恢复 |

### Case 2: BF16 下 logprobs 计算不稳定 (logsumexp)

| 项目 | 内容 |
|------|------|
| **现象** | bf16 下 logprobs 计算偶发大误差 |
| **根因** | `torch.logsumexp` 在 bf16 精度下数值不稳定 |
| **如何定位** | 源码分析 `torch_functional.py:166-200` |
| **如何修复** | `logprobs_from_logits_v2()` 对 bf16 fallback 到 `log_softmax` 实现 |
| **是否新增测试** | `tests/utils/test_torch_functional.py:144` — "Stable under large-magnitude logits where naive exp would overflow" |
| **借鉴** | 不同 dtype 可能需要不同的数值算法实现 |

### Case 3: Tensor Parallel 下 fused kernel + SP label 错位

| 项目 | 内容 |
|------|------|
| **现象** | SP=2 + fused_kernels 时，每个 SP rank 每个 micro-batch 有一个 label 错误 |
| **根因** | fused kernel 内部对 input_ids 做 `torch.roll(-1)`，但 engine 已经做过 SP-slice，roll 在 local shard 边界 wrap 而非 global 边界 |
| **如何定位** | Issue #6068 报告 SP=2 训练精度下降 |
| **如何修复** | PR #6268: 使 fused kernels 尊重 engine 已计算的 labels 而非重新 roll |
| **是否新增测试** | ✅ `test_fused_kernels_ulysses_sp.py` — SP=2 vs SP=1 log_probs 对比 |
| **借鉴** | **闭环案例**: bug → issue → fix → regression test → CI。SP shard 边界的 label 操作是经典的分布式精度陷阱 |

### Case 4: Pipeline Parallel MTP loss 死锁

| 项目 | 内容 |
|------|------|
| **现象** | 启用 MTP + CP>1 时训练挂死 |
| **根因** | MTP loss 内部 `all_reduce` 需要所有 CP rank 参与，但被 `cp_rank==0` 门控 |
| **如何定位** | 死锁排查，发现 CP rank>0 永远不进入 all_reduce |
| **如何修复** | PR #5895: 改用 `is_pipeline_last_stage()` 门控 |
| **是否新增测试** | 无 |
| **借鉴** | distributed collective 的参与者范围是精度/正确性的基础约束 |

### Case 5: DataLoader sampler 在 epoch 边界恢复时训练静默退出

| 项目 | 内容 |
|------|------|
| **现象** | checkpoint 恰好保存在 epoch 边界（global_steps % steps_per_epoch == 0），恢复后训练 0 步直接退出 |
| **根因** | (1) epoch 计算整除导致跳过一个完整 epoch；(2) StatefulDataLoader 标记为 exhausted |
| **如何定位** | Issue #5725 |
| **如何修复** | PR #5725: 检测 epoch boundary，跳过 StatefulDataLoader state restore |
| **是否新增测试** | 无 |
| **借鉴** | 边界条件是 checkpoint resume 精度对齐中最常见的 bug 来源 |

### Case 6: vLLM seed 被 SamplingParams 意外继承

| 项目 | 内容 |
|------|------|
| **现象** | 设置 `actor_rollout_ref.rollout.seed` 后，vLLM 推理变为完全确定性，丧失探索能力 |
| **根因** | engine init seed 被传递给 SamplingParams.seed，本意是初始化确定性但实际导致采样确定性 |
| **如何定位** | 用户发现 RL 训练 reward 不提升，排查到 vLLM 每次生成相同序列 |
| **如何修复** | PR #2646: 仅将 seed 传给 engine init，不传给 SamplingParams |
| **是否新增测试** | 无 |
| **借鉴** | seed 在不同抽象层级有不同语义，需要明确区分 |

### Case 7: NPU 异步 tensor 传输导致精度异常

| 项目 | 内容 |
|------|------|
| **现象** | NPU 训练中 worker 返回的计算结果偶发不正确 |
| **根因** | tensordict `.to("cpu")` 是 non-blocking 的，NPU 上数据未完全传输就被 CPU 读取 |
| **如何定位** | 参考 tensordict 库 issue #725 (已修复 CUDA/MPS)，发现 NPU 遗漏 |
| **如何修复** | PR #3222: NPU 设备添加显式同步 |
| **是否新增测试** | 无 |
| **借鉴** | 异步设备操作是分布式训练精度的隐蔽杀手 |

### Case 8: Rollout-Training 精度不匹配导致 RL 训练崩溃

| 项目 | 内容 |
|------|------|
| **现象** | 高吞吐异步训练中 loss divergence / training collapse |
| **根因** | Rollout 引擎 (vLLM bf16) 和 Training 引擎 (FSDP fp32) 的策略分布存在数值差异 |
| **如何定位** | 理论分析：不同精度下 log_prob 计算结果不同导致 IS 比率偏差 |
| **如何修复** | PR #3694: 系统性 importance sampling 修正框架 |
| **是否新增测试** | ✅ `test_rollout_corr.py` + `test_rollout_corr_integration.py` (含精确期望值断言) |
| **借鉴** | 训练-推理精度不匹配是 RL 训练独有的挑战，需要理论+工程双重解决 |

### Case 9: NCCL_DETERMINISTIC 幽灵环境变量

| 项目 | 内容 |
|------|------|
| **现象** | 代码中设置了不存在的 `NCCL_DETERMINISTIC` 环境变量 |
| **根因** | LLM 幻觉将 HCCL_DETERMINISTIC (华为) 推广为 NCCL_DETERMINISTIC (NVIDIA)，但 NCCL 不支持该变量 |
| **如何定位** | PR #5923: 穷尽审查 NCCL 源码和官方文档 |
| **如何修复** | 移除 NCCL_DETERMINISTIC，保留真实的 HCCL_DETERMINISTIC |
| **是否新增测试** | 无 |
| **借鉴** | 环境变量配置需要验证文档来源，LLM 生成的配置可能包含幻觉 |

### Case 10: mixed_precision.param_dtype 被硬编码 bf16 覆盖

| 项目 | 内容 |
|------|------|
| **现象** | 用户配置 fp32 训练但实际以 bf16 运行 |
| **根因** | `forward_step` 中 `torch.autocast(dtype=torch.bfloat16)` 硬编码，忽略 config 的 param_dtype |
| **如何定位** | Issue #5932 |
| **如何修复** | PR #6150: 将 autocast dtype 改为从 config 读取 `self._autocast_dtype` |
| **是否新增测试** | 无直接测试 |
| **借鉴** | 硬编码 dtype 是精度对齐中常见的"配置不生效"问题 |

---

## 6. 三阶段精度对齐流程映射

### 阶段一：训练前准备与基础对齐

| 检查项 | verl 状态 | 证据 |
|--------|----------|------|
| 配置一致性 | 间接支持 | 所有 engine 共享 `EngineConfig` 基类；无专门 config diff 工具 |
| 环境一致性 | 间接支持 | `enable_full_determinism` 设置所有已知环境变量 |
| seed/RNG | ✅ 体系化 | 全面覆盖 Python/NumPy/Torch/CUDA/cuBLAS/FlashAttn/NPU |
| 数据顺序 | ✅ 支持 | DataLoader seed + StatefulDataLoader 状态保存/恢复 |
| 模型结构 | ✅ 隐式保证 | 所有 worker 从同一 `model.path` 加载 |
| 初始化权重 | ✅ 隐式保证 | HuggingFace from_pretrained 确定性 |
| dropout/正则 | ✅ 支持 | `full_determinism` 开关控制 |
| deterministic flags | ✅ 体系化 | `torch.use_deterministic_algorithms`, cuBLAS, FlashAttn, CUDNN |

**评价**: 阶段一能力**体系化**，`enable_full_determinism` + checkpoint RNG restore 是亮点。

### 阶段二：单卡/单步对齐

| 检查项 | verl 状态 | 证据 |
|--------|----------|------|
| forward loss | ✅ SFT 有 golden；PPO 无 | `compare_sft_engine_results.py` |
| activation | 部分 (Ascend only) | `PrecisionDebuggerProfiler` (msprobe) |
| backward gradient | 间接支持 | grad_norm compare; SP gradient compare in tests |
| optimizer update | ✅ 通过 checkpoint round-trip 验证 | `test_fsdp_ckpt.py` bitwise match |
| scheduler | ✅ 保存/恢复 state_dict | checkpoint manager |
| loss scaling | ✅ 支持 | ShardedGradScaler (fp16), Megatron loss scaling |
| tensor dump | 部分 (Ascend only) | msprobe |
| operator-level compare | 未发现 | — |

**评价**: 阶段二**部分具备**。SFT golden loss 是核心能力，但缺少通用的 activation/gradient dump 和 operator-level compare。

### 阶段三：多步/分布式/长稳对齐

| 检查项 | verl 状态 | 证据 |
|--------|----------|------|
| loss curve | SFT: ✅; PPO: ❌ | SFT golden CI; PPO `check_results.py` 未被调用 |
| checkpoint resume | ✅ bitwise 验证 | `test_fsdp_ckpt.py` atol=0, rtol=0 |
| DP correctness | ✅ 有 | loss normalization all_reduce; SFT multi-backend golden |
| TP correctness | ✅ 有 | vocab_parallel_entropy test; KL loss TP test; linear cross entropy TP test |
| PP correctness | 部分 | batch generator test; MTP deadlock fix; 无 PP 数值精度对比 |
| SP correctness | ✅ 有 | Ulysses SP forward+backward test; fused kernel regression test |
| EP correctness | 部分 | MoE router replay; EP config validation; 无 EP 数值精度对比 |
| gradient accumulation | ✅ 隐式保证 | `same_micro_num_in_dp` all_reduce(MAX) 同步 micro-batch 数 |
| communication collectives | 间接 | allgather_dict test; MTP deadlock fix |
| mixed precision stability | ✅ 有 | IS SAFETY_BOUND; KL clamp; overflow skip; bf16 logprobs fallback |
| NaN/Inf monitoring | 部分 | 断言式检查（非持续监控） |
| CI regression | SFT: ✅; PPO: ❌ | `e2e_sft_llm.yml` |

**评价**: 阶段三在 SFT 上**体系化**，在 PPO 上有**显著缺口**（无 golden loss 回归）。PP/EP 缺少数值精度对比测试。

---

## 7. 可复用设计模式

### 设计模式 1: 多后端 Golden Loss 回归

- **设计目标**: 确保 FSDP/FSDP2/Megatron/VeOmni 等所有后端在相同输入下产生一致的训练 loss 和 grad_norm
- **源码位置**: `tests/special_e2e/sft/test_sft_engine_all.sh` + `compare_sft_engine_results.py`
- **工作流程**: 
  1. 单 GPU 运行 2 步训练 → 输出 golden.jsonl
  2. 每个后端配置运行相同 2 步 → 输出 {backend}.jsonl
  3. 对比 step 0 的 `train/loss` (atol=1e-2) 和 `train/grad_norm` (atol=1e-4)
- **优点**: 覆盖全链路（数据→模型→optimizer→loss）；多后端矩阵
- **局限**: 仅比较 step 0 的 scalar；无逐层/逐 tensor 对比
- **如何迁移**: 抽象为 `def run_golden_compare(golden_config, test_configs, metrics, tolerances)` 通用框架

### 设计模式 2: Checkpoint Round-Trip Bitwise 验证

- **设计目标**: 确保 checkpoint save → load → 继续训练 与 不中断训练产生 bitwise 相同结果
- **源码位置**: `tests/special_distributed/test_fsdp_ckpt.py`
- **工作流程**:
  1. Step 1 训练 → 保存 checkpoint（含 model + optimizer + lr_scheduler + RNG state）
  2. Step 2 训练 → 记录 logits A
  3. 加载 step 1 checkpoint → 重做 step 2 → 记录 logits B
  4. `assert_close(A, B, atol=0.0, rtol=0.0)`
- **前置条件**: `FLASH_ATTENTION_DETERMINISTIC=1`
- **优点**: bitwise 精确 = 最强正确性保证
- **局限**: 要求完全确定性环境，性能受限
- **如何迁移**: 确保 RNG 状态（Python/NumPy/Torch/CUDA/TP RNG tracker）全部保存/恢复

### 设计模式 3: 全面确定性开关

- **设计目标**: 一键开启完全训练可重复性
- **源码位置**: `verl/workers/engine/utils.py:29-55` (`enable_full_determinism`)
- **工作流程**: 设置所有已知 RNG seed + 确定性环境变量 + 禁用 CUDNN benchmark
- **优点**: 覆盖面广（Python hash, cuBLAS, FlashAttn, CUDNN, NPU HCCL）
- **局限**: 性能影响大（默认关闭）；无法覆盖所有第三方库的非确定性行为
- **如何迁移**: 收集目标系统所有 RNG 源和确定性控制点，封装为单一函数

### 设计模式 4: Log-Space 数值安全计算

- **设计目标**: 在 IS 权重计算、KL 散度等指数运算场景防止 overflow/underflow
- **源码位置**: `verl/trainer/ppo/rollout_corr_helper.py` + `core_algos.py`
- **工作流程**: 
  1. 所有概率比率在 log space 计算
  2. 在 exp() 前裁剪到 `[-SAFETY_BOUND, +SAFETY_BOUND]`
  3. 使用 `logsumexp` 替代 `log(sum(exp()))` 
- **优点**: 理论上保证数值安全
- **局限**: 裁剪引入 bias
- **如何迁移**: 适用于任何涉及 log-prob / IS weight / KL 计算的训练框架

### 设计模式 5: Parallel Strategy Correctness Test Matrix

- **设计目标**: 验证 TP/SP 下的 forward 和 backward 与单卡参考一致
- **源码位置**: `tests/special_distributed/test_tensor_dict.py` (TP entropy), `tests/models/test_transformers_ulysses.py` (SP fwd+bwd), `tests/utils/test_special_linear_cross_entropy_tp.py` (TP cross entropy)
- **工作流程**:
  1. 单卡运行完整计算 → 参考值
  2. TP/SP 多卡分片运行 → 分布式值
  3. 对比 forward output 和 backward gradient
- **优点**: 直接验证并行策略的数值等价性
- **局限**: 未覆盖 PP/EP；仅测试特定算子
- **如何迁移**: 扩展为 `correctness_matrix(parallelisms=[DP, TP, PP, SP, EP], test_fn)` 框架

### 设计模式 6: Rollout-Training Mismatch 修正

- **设计目标**: 解决 rollout 引擎和 training 引擎精度差异导致的 policy mismatch
- **源码位置**: `verl/trainer/ppo/rollout_corr_helper.py`
- **工作流程**:
  1. 用 training 引擎重新计算 log_prob (对同一 sequence)
  2. 与 rollout 时的 log_prob 做差 → IS ratio
  3. 用 IS ratio 修正 policy loss
  4. 可选 rejection sampling（ratio 偏差太大则丢弃 sample）
- **优点**: 理论严谨（importance sampling 框架）；可与多种 RL 算法组合
- **局限**: 额外计算开销；rejection sampling 降低有效 batch size
- **如何迁移**: 任何使用独立推理引擎的 RL 训练系统都面临此问题

---

## 8. 缺口分析与改造建议

### P0: 必须补齐

#### P0-1: PPO/RL 训练 Golden Loss 回归测试

- **问题**: `check_results.py`（含 `best_reward > target` 检查）存在但**未被任何 CI workflow 调用**
- **为什么重要**: PPO 是 verl 的核心场景，缺少 loss/reward golden 回归意味着无法自动检测算法级精度退化
- **当前部分实现**: 脚本已存在，仅需纳入 CI
- **建议设计**: 在 `e2e_ppo_trainer.yml` 中运行短 PPO 训练 (2-5 steps)，记录 loss curve 到 jsonl，与 committed golden baseline 比较
- **涉及模块**: `.github/workflows/e2e_ppo_trainer.yml`, `tests/special_e2e/`
- **预期收益**: PPO 算法变更的自动精度回归检测

#### P0-2: 通用 Activation/Gradient Dump + Compare 工具

- **问题**: 当前 msprobe 仅限 Ascend NPU；NVIDIA/AMD 平台无 tensor dump 工具
- **为什么重要**: 精度问题定位的核心手段——逐层 activation/gradient diff 是排查 mismatch 的"金标准"方法
- **当前部分实现**: msprobe 集成 (`PrecisionDebuggerProfiler`) 提供了架构框架
- **建议设计**: 实现通用 `TensorDumpHook`：forward/backward hook 保存每层输入/输出/梯度到文件 + `tensor_compare.py` 工具逐层比较
- **涉及模块**: `verl/utils/debug/`, 新建 `verl/utils/precision/`
- **预期收益**: 任何平台都能做逐层精度对比

### P1: 强烈建议补齐

#### P1-1: Pipeline Parallel 数值精度对比测试

- **问题**: PP 测试仅覆盖 batch generator 结构和 deadlock 修复，无 PP vs non-PP 数值对比
- **为什么重要**: PP 引入 micro-batch schedule、cross-stage gradient 传播，这些都可能引入数值差异
- **当前部分实现**: `test_pipeline_parallel.py` 仅测试 batch 结构
- **建议设计**: 类比 TP 测试：单卡 full model → PP=2 分布式 → 对比 logits 和 gradients
- **涉及模块**: `tests/special_distributed/`
- **预期收益**: PP 变更的自动数值验证

#### P1-2: Expert Parallel / MoE 数值精度对比测试

- **问题**: EP 有 config validation 和 router replay 功能，但无 EP vs non-EP 数值对比
- **为什么重要**: MoE routing 是非确定性的，EP 分片可能引入额外数值差异
- **当前部分实现**: Router replay 机制实际是为解决此问题而设计
- **建议设计**: EP=1 vs EP=2 的 forward loss 和 gradient 对比（需配合 router replay 固定 routing）
- **涉及模块**: `tests/special_distributed/`, `tests/models/`
- **预期收益**: MoE 训练精度保证

#### P1-3: 系统化 NaN/Inf Watchdog

- **问题**: 当前 NaN 检查是断言式（crash 或 skip），无持续监控
- **为什么重要**: 训练中 NaN 可能是渐进式的（先 softly diverge，再 hard NaN），需要早期预警
- **当前部分实现**: `isnan/isinf` 断言、gradient overflow skip、NaN masked mean
- **建议设计**: 在 `engine_workers.py` 的 `_postprocess_output` 中添加 loss/grad_norm NaN 检测 callback，支持告警/停训
- **涉及模块**: `verl/workers/engine_workers.py`, `verl/utils/debug/`
- **预期收益**: 训练异常的及时发现

#### P1-4: Config Snapshot + Diff 工具

- **问题**: 无法快速比较两次训练的完整配置差异
- **为什么重要**: 精度不一致的首要排查手段是"这两次训练的配置是否完全相同"
- **当前部分实现**: Hydra 自动保存 config 到 output 目录
- **建议设计**: `verl config diff run1/ run2/` 命令，输出配置差异
- **涉及模块**: `verl/utils/config.py`
- **预期收益**: 精度排查效率大幅提升

### P2: 长期优化项

#### P2-1: 自动化二分定位工具 (Bisect)

- **问题**: 当 CI golden regression 失败时，需要手动 git bisect 定位引入问题的 commit
- **建议设计**: `verl bisect --golden=golden.jsonl --test="python train.py --steps=2"` 自动执行

#### P2-2: 跨硬件对齐矩阵 (NVIDIA vs AMD vs NPU)

- **问题**: 当前跨后端对齐仅在同一硬件上（同一 CI runner）
- **建议设计**: 多硬件 CI runner 运行相同 golden test，对比 loss/grad_norm

#### P2-3: TF32 全局控制

- **问题**: 仅 DeepSeek MoE 模型禁用 bf16 reduced precision matmul；无全局 TF32 开关
- **建议设计**: 在 `enable_full_determinism()` 中添加 `torch.backends.cuda.matmul.allow_tf32 = False`

#### P2-4: Optimizer State Equivalence Test

- **问题**: checkpoint round-trip 隐式验证了 optimizer state，但无直接的 optimizer state 对比测试
- **建议设计**: 保存两个相同训练的 optimizer state dict，逐 key allclose 对比

---

## 9. 推荐学习路线

### 第 1 步: 读文档和配置

1. `CLAUDE.md` — 项目全景
2. `verl/workers/config/engine.py` — 理解 seed/full_determinism/mixed_precision 配置
3. `verl/trainer/config/` — 理解 Hydra 配置体系
4. `.github/workflows/e2e_sft_llm.yml` + `model.yml` — 理解 CI 精度回归是如何触发的

### 第 2 步: 跑关键测试

1. `python tests/special_e2e/sft/compare_sft_engine_results.py` — 体验 golden loss 比较
2. `torchrun --nproc_per_node=2 tests/special_distributed/test_fsdp_ckpt.py` — 体验 checkpoint bitwise 验证
3. `torchrun --nproc_per_node=2 tests/special_distributed/test_tensor_dict.py` — 体验 TP correctness test
4. `pytest tests/trainer/ppo/test_rollout_corr.py -v` — 体验 IS weight 精确值验证
5. `pytest tests/utils/test_linear_cross_entropy.py -v` — 体验多实现 forward+backward 对比

### 第 3 步: 读核心源码

1. `verl/workers/engine/utils.py:enable_full_determinism` — 确定性控制全貌
2. `verl/utils/checkpoint/megatron_checkpoint_manager.py:226-263` — RNG 状态保存恢复最复杂的实现
3. `verl/trainer/ppo/rollout_corr_helper.py` — 数值安全 IS 权重计算
4. `verl/trainer/ppo/core_algos.py:agg_loss` — 分布式 loss 聚合正确性
5. `verl/utils/torch_functional.py:logprobs_from_logits_v2` — dtype-aware 数值计算
6. `verl/workers/engine/fsdp/transformer_impl.py:342-412` — Mixed precision 配置全链路
7. `verl/model_merger/fsdp_model_merger.py:258` — 模型合并精度验证

### 第 4 步: 复现关键 Commit/PR 中的问题

1. PR #6068 (SP label bug) — 尝试在 SP=2 下不做 fix 运行，观察 log_probs 差异
2. PR #2646 (vLLM seed bug) — 理解 engine seed vs sampling seed 的区别
3. PR #5932/#6150 (autocast dtype hardcode) — 理解配置不生效的调试方法
4. PR #1779 (KL clamp) — 在不 clamp 的情况下观察 KL 爆炸

### 第 5 步: 抽象设计模式

1. "单卡 Golden → 多后端 Compare → CI 固化" 流程
2. "Checkpoint Save → Load → Retrain → Bitwise Compare" 流程
3. "TP/SP Forward+Backward → Single-GPU Reference → allclose" 矩阵
4. "Log-space + SAFETY_BOUND + clamp" 数值安全计算模式
5. "全面 RNG 控制 + Checkpoint RNG 恢复" 确定性保证模式

### 第 6 步: 迁移到自己的训练系统

1. 实现 `enable_full_determinism()` — 收集所有 RNG 源和确定性控制点
2. 实现 checkpoint RNG save/restore — 确保包括框架特有的 RNG tracker
3. 建立 SFT golden loss 回归 CI — 最小化训练 → scalar 比较 → CI 门控
4. 建立 parallel strategy correctness matrix — DP/TP/SP/PP/EP 各一个对比测试
5. 实现通用 activation/gradient dump hook — forward/backward hook + tensor dump
6. 实现 NaN/Inf watchdog — 训练循环中持续监控

---

## 10. 对我自研分布式训练系统的迁移建议

### 10.1 首先迁移 (Week 1)

| 能力 | verl 参考 | 迁移方法 |
|------|----------|---------|
| 全面确定性开关 | `engine/utils.py:enable_full_determinism` | 直接复制并扩展到你的 RNG 源 |
| Checkpoint RNG 保存 | `checkpoint_manager.py:get_rng_state` | 增加你框架特有的 RNG tracker |
| SFT golden loss 回归 | `compare_sft_engine_results.py` | 适配你的日志格式和 CI 系统 |

### 10.2 然后迁移 (Week 2-3)

| 能力 | verl 参考 | 迁移方法 |
|------|----------|---------|
| Checkpoint round-trip bitwise 验证 | `test_fsdp_ckpt.py` | 适配你的 checkpoint 格式 |
| TP/SP correctness matrix | `test_tensor_dict.py` + `test_transformers_ulysses.py` | 适配你的并行策略 |
| 数值安全 log-space 计算 | `rollout_corr_helper.py` | 应用于你的 loss 计算 |

### 10.3 最后补齐 (Week 4+)

| 能力 | verl 缺口 | 你需要 |
|------|----------|--------|
| 通用 tensor dump | verl 仅有 msprobe (Ascend) | 自建通用 hook + compare 工具 |
| PPO golden regression | verl 有脚本但未纳入 CI | 直接建 CI |
| PP/EP correctness test | verl 部分缺失 | 需要自建 |
| 自动化 bisect | verl 完全缺失 | 结合 git bisect + golden test 脚本 |

---

## Appendix A. 检索关键词与命令记录

### 源码搜索关键词（实际执行）

| 类别 | 关键词 |
|------|--------|
| 确定性 | `seed`, `random`, `rng`, `deterministic`, `determinism`, `dropout`, `FLASH_ATTENTION_DETERMINISTIC`, `CUBLAS_WORKSPACE_CONFIG`, `HCCL_DETERMINISTIC` |
| 数值精度 | `allclose`, `rtol`, `atol`, `tolerance`, `golden`, `baseline`, `expected`, `numerical`, `precision`, `accuracy`, `assert_close` |
| Tensor dump | `dump`, `tensor dump`, `activation dump`, `gradient dump`, `compare`, `debug`, `msprobe`, `PrecisionDebugger` |
| Mixed precision | `fp32`, `tf32`, `fp16`, `bf16`, `fp8`, `amp`, `loss_scaling`, `grad_scaler`, `mixed_precision`, `master_weight`, `ShardedGradScaler`, `MixedPrecision`, `autocast` |
| NaN/Inf | `nan`, `inf`, `overflow`, `underflow`, `isnan`, `isinf`, `SAFETY_BOUND`, `clamp` |
| Checkpoint | `checkpoint`, `resume`, `save_checkpoint`, `load_checkpoint`, `state_dict`, `rng_state` |
| 分布式 | `all_reduce`, `reduce_scatter`, `all_gather`, `broadcast`, `collective`, `process_group`, `device_mesh`, `world_size` |
| 并行策略 | `data_parallel`, `tensor_parallel`, `pipeline_parallel`, `sequence_parallel`, `expert_parallel`, `moe`, `fsdp`, `ulysses` |

### Git 搜索命令（实际执行）

```bash
git log --oneline --all --grep="precision"
git log --oneline --all --grep="accuracy"
git log --oneline --all --grep="determin"
git log --oneline --all --grep="golden"
git log --oneline --all --grep="numerical"
git log --oneline --all --grep="seed"
git log --oneline --all --grep="checkpoint"
git log --oneline --all --grep="gradient"
git log --oneline --all --grep="reproducib"
git log --oneline --all --grep="bf16\|fp16\|fp8\|tf32\|mixed.precision"
git log --oneline --all --grep="loss.*mismatch\|loss.*diverge\|nan\|NaN\|overflow"
git log --oneline --all --grep="all_reduce\|reduce_scatter\|allreduce"
git show --stat <commit_hash>  # 对 20+ 关键 commit 执行
```

### 检查的核心目录

| 目录 | 状态 |
|------|------|
| `verl/workers/engine/` (FSDP, Megatron, Automodel, TorchTitan, VeOmni) | ✅ 已检查 |
| `verl/trainer/ppo/` (core_algos, rollout_corr_helper, ray_trainer) | ✅ 已检查 |
| `verl/utils/checkpoint/` | ✅ 已检查 |
| `verl/utils/profiler/` | ✅ 已检查 |
| `verl/utils/` (torch_functional, fsdp_utils, megatron_utils, ulysses) | ✅ 已检查 |
| `verl/workers/config/` | ✅ 已检查 |
| `verl/models/transformers/` | ✅ 已检查 |
| `verl/model_merger/` | ✅ 已检查 |
| `tests/special_distributed/` | ✅ 已检查 |
| `tests/special_e2e/sft/` | ✅ 已检查 |
| `tests/models/` | ✅ 已检查 |
| `tests/trainer/ppo/` | ✅ 已检查 |
| `tests/utils/` | ✅ 已检查 |
| `tests/checkpoint_engine/` | ✅ 已检查 |
| `.github/workflows/` | ✅ 已检查 |

---

## Appendix B. 关键文件清单

### 精度控制核心文件

| 文件 | 核心功能 |
|------|---------|
| `verl/workers/engine/utils.py` | `enable_full_determinism()` 全局确定性开关 |
| `verl/workers/config/engine.py` | seed, full_determinism, mixed_precision 配置 |
| `verl/utils/checkpoint/checkpoint_manager.py` | RNG 状态保存/恢复基类 |
| `verl/utils/checkpoint/fsdp_checkpoint_manager.py` | FSDP checkpoint (含 RNG) |
| `verl/utils/checkpoint/megatron_checkpoint_manager.py` | Megatron checkpoint (含 TP RNG tracker) |
| `verl/utils/torch_dtypes.py` | `PrecisionType` 精度类型标准化 |
| `verl/utils/torch_functional.py` | `logprobs_from_logits_v2` dtype-aware 计算 |
| `verl/trainer/ppo/core_algos.py` | `agg_loss` 分布式 loss 聚合 + KL clamp |
| `verl/trainer/ppo/rollout_corr_helper.py` | IS 权重修正 + SAFETY_BOUND |
| `verl/workers/engine/fsdp/transformer_impl.py` | FSDP mixed precision + GradScaler |
| `verl/utils/profiler/precision_debugger_profile.py` | msprobe PrecisionDebugger 集成 |
| `verl/utils/megatron/tensor_parallel.py` | Vocab parallel entropy/logprobs |

### 精度测试核心文件

| 文件 | 核心功能 |
|------|---------|
| `tests/special_e2e/sft/compare_sft_engine_results.py` | SFT golden loss 比较 |
| `tests/special_e2e/sft/test_sft_engine_all.sh` | 多后端 SFT golden 测试脚本 |
| `tests/special_distributed/test_fsdp_ckpt.py` | Checkpoint round-trip bitwise 验证 |
| `tests/special_distributed/test_tensor_dict.py` | TP entropy correctness |
| `tests/special_distributed/test_fused_kernels_ulysses_sp.py` | SP fused kernel regression |
| `tests/models/test_transformers_ulysses.py` | Ulysses SP fwd+bwd correctness |
| `tests/models/test_engine.py` | HF vs mcore engine logprob parity |
| `tests/utils/test_linear_cross_entropy.py` | 4 实现 forward+backward 对比 |
| `tests/utils/test_special_linear_cross_entropy_tp.py` | TP linear cross entropy |
| `tests/utils/test_special_megatron_kl_loss_tp.py` | TP KL loss correctness |
| `tests/trainer/ppo/test_rollout_corr.py` | IS weight 精确值验证 |
| `tests/trainer/ppo/test_rollout_corr_integration.py` | PPO loss + IS NaN/Inf guard |
| `tests/checkpoint_engine/test_correctness_on_gpu.py` | NCCL weight transfer allclose |

---

## Appendix C. 关键 Commits / PR / Issues 清单

| Commit | PR | 日期 | 类型 | 摘要 |
|--------|-----|------|------|------|
| `093e9599` | #789 | 2025-03 | fix | 梯度溢出时跳过 optimizer step |
| `2c85b432` | #1779 | 2025-06 | fix | KL 散度值域裁剪 [-10,10] 防爆炸 |
| `5cbad837` | #2377 | 2025-07 | fix | NaN 安全 masked mean/sum |
| `3f6cd479` | #2646 | 2025-07 | fix | vLLM seed 被 SamplingParams 意外继承 |
| `cc2799b2` | #3222 | 2025-08 | fix | NPU tensor 异步传输精度修复 |
| `6bbbff13` | #3068 | 2025-08 | fix | FSDPEngineConfig 缺失 mixed_precision 字段 |
| `7bc70bbf` | #3363 | 2025-09 | feat | **SFT 多后端精度对齐 CI** |
| `21271aab` | #3694 | 2025-10 | feat | **Rollout IS 修正框架** |
| `65f9badf` | #4036 | 2025-11 | feat | FP16 训练 + ShardedGradScaler |
| `425d6bb0` | #4097 | 2025-11 | fix | Megatron-FSDP bf16 对齐 |
| `b05f8f64` | #4223 | 2025-11 | feat | Megatron FP8 训练 |
| `ef44dcc7` | #4156 | 2025-11 | fix | 恢复训练 epoch 计算修复 |
| `fb860f0b` | #4378 | 2025-12 | feat | **enable_full_determinism** |
| `01ef5894` | #4783 | 2026-01 | fix | vLLM-Ascend dp+ep+tp 精度问题 |
| `82cf2ddc` | #5179 | 2026-02 | fix | vLLM per-replica 不同 seed |
| `b4c82633` | #5186 | 2026-04 | feat | msprobe PrecisionDebugger 集成 |
| `21f4e490` | #5216 | 2026-02 | fix | Reward NaN (overlong_cfg.len=0 除零) |
| `516657fa` | #5895 | 2026-04 | fix | MTP loss + CP 死锁 |
| `fba09392` | #5899 | 2026-04 | fix | 空 tensor NaN 返回 |
| `23ede534` | #5923 | 2026-04 | refactor | NCCL_DETERMINISTIC 幽灵变量修复 |
| `9a70e235` | #6150 | 2026-04 | fix | autocast dtype 硬编码 bf16 修复 |
| `575d5a8a` | #6268 | 2026-05 | fix | **SP fused kernel label 修复 + regression test** |
| `694e44e2` | #5725 | 2026-03 | fix | Epoch boundary checkpoint resume 修复 |
