# 10 - 模型层子模块 (verl/models/)

> 源码位置: `verl/models/` | 31 个文件 | 9,478 行
>
> 版本基线: 2025-06 main 分支

---

## 1. 模块定位

`verl/models/` 是 verl 框架的模型适配层,解决一个核心问题: **如何让 HuggingFace 生态的模型无缝运行在 Megatron-Core 分布式训练引擎和 verl 的 RL 训练管线中**。

核心职责:
- **MCore 集成** (`mcore/`): 17 个文件,5,238 行——HF 与 Megatron-Core 之间的双向桥接,包括配置转换、权重转换、模型初始化和前向传播
- **HF Transformers 补丁** (`transformers/`): 12 个文件,3,992 行——通过 Monkey Patch 注入 Ulysses 序列并行、融合 kernel、VLM 适配等能力

两个子目录互不依赖,分别服务于 Megatron 后端和 FSDP 后端的训练场景。

---

## 2. 文件清单与行数

### mcore/ 子目录 (17 文件, 5,238 行)

| 文件 | 行数 | 职责 |
|------|------|------|
| `registry.py` | 299 | `SupportedModel` 枚举(17 种架构), 4 个并行注册表 |
| `config_converter.py` | 422 | HF `PretrainedConfig` --> MCore `TransformerConfig`(6 种转换器) |
| `model_initializer.py` | 203 | 模型初始化(5 种初始化器: Dense/Qwen2MoE/Mixtral/Qwen3MoE/DeepseekV3) |
| `loader.py` | 495 | HF 权重 --> MCore 分片加载 |
| `saver.py` | 497 | MCore 分片 --> HF 格式合并保存 |
| `weight_converter.py` | 479 | 在线权重名称转换(6 种转换器) |
| `model_forward.py` | 449 | 标准前向传播(THD/BSHD 两种数据格式) |
| `model_forward_fused.py` | 317 | 融合 kernel 前向(linear_cross_entropy) |
| `model_forward_1f1b_overlap.py` | 252 | 1F1B 流水线并行重叠前向 |
| `mtp_patch.py` | 467 | Multi-Token Prediction 补丁 |
| `patch.py` | 573 | MCore 兼容性补丁 |
| `util.py` | 785 | 数据格式转换工具(preprocess/postprocess) |

### transformers/ 子目录 (12 文件, 3,992 行)

| 文件 | 行数 | 职责 |
|------|------|------|
| `monkey_patch.py` | 539 | 统一补丁入口 `apply_monkey_patch()` |
| `dense_common.py` | 203 | 通用 Dense 模型前向(Llama/Qwen2 等) |
| `llama.py` | 241 | Llama 模型特定补丁 |
| `qwen2.py` | 243 | Qwen2 模型特定补丁 |
| `qwen2_vl.py` | 563 | Qwen2-VL 多模态补丁 |
| `qwen3_vl.py` | 455 | Qwen3-VL 多模态补丁 |
| `qwen3_5.py` | 296 | Qwen3.5 多模态补丁 |
| `glm4v.py` | 548 | GLM-4V 多模态补丁 |
| `kimi_vl.py` | 192 | Kimi-VL 多模态补丁 |
| `apertus.py` | 118 | Apertus 模型补丁 |
| `tiled_mlp.py` | 236 | MLP 分块优化(降低峰值显存) |
| `npu_patch.py` | 358 | NPU(华为昇腾)优化补丁 |

---

## 3. MCore 三层注册表模式

`registry.py`（299 行）是整个 MCore 集成的入口和索引。

### 3.1 SupportedModel 枚举

`SupportedModel`（第 119 行）枚举 17 种模型架构:

| 枚举值 | 架构名 | 测试状态 |
|--------|--------|---------|
| `LLAMA` | `LlamaForCausalLM` | 已测试 |
| `QWEN2` | `Qwen2ForCausalLM` | 已测试 |
| `QWEN2_MOE` | `Qwen2MoeForCausalLM` | pending |
| `DEEPSEEK_V3` | `DeepseekV3ForCausalLM` | 未测试 |
| `MIXTRAL` | `MixtralForCausalLM` | 已测试 |
| `QWEN2_5_VL` | `Qwen2_5_VLForConditionalGeneration` | 不支持(旧注册表) |
| `LLAMA4` | `Llama4ForConditionalGeneration` | 未测试 |
| `QWEN3` | `Qwen3ForCausalLM` | 已测试 |
| `QWEN3_MOE` | `Qwen3MoeForCausalLM` | 已测试 |
| `QWEN3_5_MOE` | `Qwen3_5MoeForCausalLM` | 已测试 |
| `GLM4_MOE` | `Glm4MoeForCausalLM` | — |
| `QWEN3_TOKEN_CLASSIFICATION` | `Qwen3ForTokenClassification` | — |
| `LLAMA_TOKEN_CLASSIFICATION` | `LlamaForTokenClassification` | — |
| `QWEN3_MOE_VL` | `Qwen3VLMoeForConditionalGeneration` | — |
| `QWEN3_VL` | `Qwen3VLForConditionalGeneration` | — |
| `GPT_OSS` | `GptOssForCausalLM` | — |
| `MIMO` | `MiMoForCausalLM` | — |

### 3.2 四个并行注册表

每种模型在以下 4 个注册表中各有一个条目:

**1. MODEL_CONFIG_CONVERTER_REGISTRY** (第 140 行)
```
SupportedModel --> Callable[[PretrainedConfig, dtype], TransformerConfig]
```
将 HF 配置转换为 MCore TransformerConfig。6 种转换函数:
- `hf_to_mcore_config_dense`: Llama, Qwen2, Qwen3 等 Dense 模型
- `hf_to_mcore_config_qwen2moe`: Qwen2-MoE
- `hf_to_mcore_config_qwen3moe`: Qwen3-MoE, Qwen3.5-MoE
- `hf_to_mcore_config_mixtral`: Mixtral
- `hf_to_mcore_config_dpskv3`: DeepSeekV3 (MLA 注意力)
- `hf_to_mcore_config_qwen2_5_vl`: Qwen2.5-VL

**2. MODEL_INITIALIZER_REGISTRY** (第 156 行)
```
SupportedModel --> type[BaseModelInitializer]
```
5 种初始化器:
- `DenseModel`: Llama, Qwen2, Qwen3, Llama4
- `Qwen2MoEModel`: Qwen2-MoE（默认冻结 router）
- `MixtralModel`: Mixtral（默认不冻结 router）
- `Qwen3MoEModel`: Qwen3-MoE, Qwen3.5-MoE（默认冻结 router）
- `DeepseekV3Model`: DeepSeekV3（支持 MTP）

**3. MODEL_FORWARD_REGISTRY** (第 171 行)
```
SupportedModel --> Callable (前向函数)
```
标准前向传播,由 `model_forward_gen()` 工厂函数生成。VLM 模型传入 `vision_model=True`。

**4. MODEL_WEIGHT_CONVERTER_REGISTRY** (第 211 行)
```
SupportedModel --> type[McoreToHFWeightConverterBase]
```
6 种权重转换器用于 MCore --> HF 方向的在线转换。

### 3.3 新模型接入流程

添加新模型架构需要修改的注册表:

```
1. SupportedModel 枚举: 添加新成员
2. MODEL_CONFIG_CONVERTER_REGISTRY: 选择或新建配置转换函数
3. MODEL_INITIALIZER_REGISTRY: 选择或新建初始化器
4. MODEL_FORWARD_REGISTRY: 通常使用 model_forward_gen()
5. MODEL_WEIGHT_CONVERTER_REGISTRY: 新建权重转换器(如有特殊权重融合)
```

---

## 4. 双向权重转换

### 4.1 核心数学操作

HF 和 MCore 的权重格式差异集中在两个融合操作:

**QKV 融合**:
```
HF:   q_proj.weight, k_proj.weight, v_proj.weight  (3 个独立矩阵)
MCore: linear_qkv.weight                           (1 个融合矩阵, interleaved)
```
MCore 将 Q、K、V 按 `[Q_head0, K_head0, V_head0, Q_head1, ...]` 交错排列（interleaving），
以优化张量并行下的通信效率。

**Gate+Up 融合**:
```
HF:   gate_proj.weight, up_proj.weight  (2 个独立矩阵)
MCore: linear_fc1.weight                (1 个融合矩阵, interleaved)
```
MCore 将 gate 和 up 投影交错融合到单个 `linear_fc1` 中。

### 4.2 权重转换器继承层次

`weight_converter.py`（479 行）定义 6 种转换器:

```
McoreToHFWeightConverterBase (第 25 行)
├── McoreToHFWeightConverterDense (第 34 行)
│   ├── McoreToHFWeightConverterQwen2Moe (第 103 行)
│   ├── McoreToHFWeightConverterQwen2_5_VL (第 150 行)
│   ├── McoreToHFWeightConverterMixtral (第 422 行)
│   └── McoreToHFWeightConverterQwen3Moe (第 446 行)
└── McoreToHFWeightConverterDpskv3 (第 269 行) — DeepSeekV3 MLA
```

每个转换器的 `convert_param(name, params)` 方法返回 `(list[str], list[Tensor])`——将 MCore 参数名映射到 HF 参数名,同时拆分融合矩阵。

### 4.3 Dense 模型转换示例

`McoreToHFWeightConverterDense` 的核心名称映射（第 87 行 `convert_param`）:

| MCore 参数名 | HF 参数名 |
|-------------|----------|
| `embedding.word_embeddings.weight` | `model.embed_tokens.weight` |
| `decoder.layers.N.self_attention.linear_qkv.weight` | `model.layers.N.self_attn.{q,k,v}_proj.weight` (拆为 3) |
| `decoder.layers.N.self_attention.linear_proj.weight` | `model.layers.N.self_attn.o_proj.weight` |
| `decoder.layers.N.mlp.linear_fc1.weight` | `model.layers.N.mlp.{gate,up}_proj.weight` (拆为 2) |
| `decoder.layers.N.mlp.linear_fc2.weight` | `model.layers.N.mlp.down_proj.weight` |
| `decoder.final_layernorm.weight` | `model.norm.weight` |
| `output_layer.weight` | `lm_head.weight` |

### 4.4 MoE 模型特殊处理

MoE 模型（Qwen2-MoE, Mixtral, Qwen3-MoE, DeepSeekV3）额外需要处理:
- **Router 权重**: `mlp.router.weight` --> `mlp.gate.weight`
- **Expert 权重**: `mlp.experts.linear_fc1.weightN` --> `mlp.experts.N.{gate,up}_proj.weight`（按 expert_id 拆分）
- **Shared Expert**: `mlp.shared_experts.linear_fc1.weight` --> `mlp.shared_expert.{gate,up}_proj.weight`
- **Expert Bias**: DeepSeekV3 特有的 `mlp.router.expert_bias` --> `mlp.gate.e_score_correction_bias`

---

## 5. 三种前向传播路径

### 5.1 标准前向 (model_forward.py, 449 行)

`model_forward_gen()`（第 38 行）是一个工厂函数,返回支持两种数据格式的前向函数:

**THD 格式** (Token-Head-Dimension, 去 padding):
```python
# 第 74-75 行
input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(
    input_ids, attention_mask, pre_process=..., use_fp8_padding=...)
```
通过 `packed_seq_params` 传递 `cu_seqlens` 给 FlashAttention,避免 padding 的计算浪费。

**BSHD 格式** (Batch-Sequence-Head-Dimension, 有 padding):
```python
# preprocess_bshd() 在 util.py 中
```
传统的 padded batch 格式,兼容不支持 packed sequences 的后端。

`gptmodel_forward_model_engine()`（第 264 行）是用于 model engine 的特化版本,也支持 THD 和 BSHD。

### 5.2 融合前向 (model_forward_fused.py, 317 行)

`fused_forward_model_gen()`（第 68 行）和 `fused_forward_model_engine()`（第 140 行）将 lm_head 线性层和交叉熵损失融合到单个 kernel 调用:

```python
# 利用 linear_cross_entropy 融合 kernel
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
```

关键优化: 避免在 vocab_size 维度上实例化完整的 logits 张量,显著降低显存峰值。

`patch_fused_forward()`（第 52 行）通过运行时替换 `GPTModel.forward` 方法来启用融合前向。

### 5.3 1F1B 流水线重叠前向 (model_forward_1f1b_overlap.py, 252 行)

用于流水线并行 (PP > 1) 场景,将前向和后向计算与通信重叠:
- 在发送激活值到下一阶段时,异步开始处理下一个 micro-batch
- 需要与 Megatron 的 pipeline schedule 配合使用

---

## 6. 配置转换详解

`config_converter.py`（422 行）将 HF 的 `PretrainedConfig` 转换为 MCore 的 `TransformerConfig`。

### 6.1 基础配置构建

`_get_base_transformer_config()`（第 34 行）提取所有架构共享的配置:

```python
base_config = {
    "num_layers": hf_config.num_hidden_layers,
    "hidden_size": hf_config.hidden_size,
    "num_attention_heads": hf_config.num_attention_heads,
    "num_query_groups": hf_config.num_key_value_heads,  # GQA 支持
    "ffn_hidden_size": hf_config.intermediate_size,
    "activation_func": F.silu,
    "normalization": "RMSNorm",
    "gated_linear_unit": True,
    # 并行配置从 megatron parallel_state 动态获取
    "tensor_model_parallel_size": mpu.get_tensor_model_parallel_world_size(),
    "pipeline_model_parallel_size": mpu.get_pipeline_model_parallel_world_size(),
    ...
}
```

### 6.2 架构特殊配置

| 架构 | 特殊配置 |
|------|---------|
| **Dense** (Llama/Qwen) | `add_qkv_bias` (Qwen 系), `qk_layernorm` (Qwen3 系) |
| **Qwen2-MoE** | `moe_shared_expert_overlap=True`, `moe_router_pre_softmax=True`, 冻结 router |
| **Mixtral** | 无 shared expert, `moe_shared_expert_overlap=False` |
| **Qwen3-MoE** | 无 shared expert, `qk_layernorm=True`, 冻结 router |
| **DeepSeekV3** | `MLATransformerConfig`(MLA 注意力), MTP 支持, `moe_router_dtype="fp64"` |
| **Qwen2.5-VL** | `mrope_section` (多模态 RoPE) |

### 6.3 版本兼容检查

`check_and_construct_configs()`（第 139 行）在构造 `TransformerConfig` 前检查当前 Megatron 版本是否支持所有配置键,不支持的键会被自动移除并发出警告。

---

## 7. Monkey Patch 补丁链

`monkey_patch.py`（539 行）中的 `apply_monkey_patch()`（第 291 行）是 HF Transformers 模型的统一补丁入口。

### 7.1 补丁流程

```
apply_monkey_patch(model, ulysses_sp_size, use_fused_kernels, ...)
│
├── [1] TiledMLP 补丁 (可选, 第 318-322 行)
│     └── apply_tiled_mlp_monkey_patch(): MLP 分块降低峰值显存
│
├── [2] PrefixGrouper 补丁 (可选, 第 324-325 行)
│     └── apply_prefix_grouper_patch(): 包装 ALL_ATTENTION_FUNCTIONS
│
├── [3] VLM 模型补丁 (按 model_type 分发, 第 360-528 行)
│     ├── qwen2_5_vl / qwen2_vl:
│     │   ├── Step 1: 替换 forward (forward_with_normal_backend)
│     │   ├── Step 2: 替换 attention (qwen2_vl_attn_forward)
│     │   └── Step 3: patch_vlm_for_ulysses_input_slicing
│     │
│     ├── qwen3_vl / qwen3_vl_moe:
│     │   ├── Step 1: 替换 forward + fast_pos_embed_interpolate
│     │   ├── Step 1.5: 修复 transformers 4.57.3 bug
│     │   └── Step 2: patch_vlm_for_ulysses_input_slicing
│     │
│     ├── glm4v:
│     │   ├── Step 1: 替换 forward (forward_with_normal_backend)
│     │   ├── Step 2: 替换 attention (glm4v_attn_forward)
│     │   └── Step 3: patch_vlm_for_ulysses_input_slicing
│     │
│     ├── kimi_vl:
│     │   └── 替换 FlashAttention2.forward
│     │
│     └── qwen3_5 / qwen3_5_moe:
│         ├── Step 1: 替换 forward + fast_pos_embed_interpolate
│         └── Step 2: patch_vlm_for_ulysses_input_slicing
│
├── [4] Ulysses SP 全局补丁 (第 529-537 行)
│     └── 替换 _flash_attention_forward --> _ulysses_flash_attention_forward
│
└── [5] 融合 kernel 补丁 (第 539 行)
      └── patch_forward_with_backends(model, use_fused_kernels, fused_kernels_backend)
```

### 7.2 Ulysses 序列并行注入

`_ulysses_flash_attention_forward()`（第 87 行）是 Ulysses SP 的核心:

```
输入: (batch, seqlen/sp_size, nheads, head_dim)
          │
          ▼
   repeat_kv (KV heads 重复以满足 SP 分割)
          │
          ▼
   gather_seq_scatter_heads (AlltoAll)
   输入变为 (batch, seqlen, nheads/sp_size, head_dim)
          │
          ▼
   _flash_attention_forward (标准 FlashAttention)
          │
          ▼
   gather_heads_scatter_seq (AlltoAll)
   输出: (batch, seqlen/sp_size, nheads, head_dim)
```

### 7.3 融合 kernel 选择

`patch_forward_with_backends()`（第 234 行）根据 `model_type` 和 `fused_kernels_backend` 选择前向函数:

| model_type | torch 后端 | triton 后端 |
|-----------|-----------|------------|
| `qwen2_5_vl`, `qwen2_vl` | `qwen2_vl.forward_with_torch_backend` | `qwen2_vl.forward_with_triton_backend` |
| `qwen3_vl`, `qwen3_vl_moe` | `qwen3_vl.forward_with_torch_backend` | `qwen3_vl.forward_with_triton_backend` |
| `glm4v` | `glm4v.forward_with_torch_backend` | `glm4v.forward_with_triton_backend` |
| `qwen3_5`, `qwen3_5_moe` | `qwen3_5.forward_with_torch_backend` | `qwen3_5.forward_with_triton_backend` |
| 其他 Dense 模型 | `dense_common.forward_with_torch_backend` | `dense_common.forward_with_triton_backend` |

融合 kernel 将 lm_head 的 logits 计算与交叉熵损失融合,避免实例化 `[batch, seq, vocab_size]` 形状的完整 logits 张量。

---

## 8. 模型初始化器

`model_initializer.py`（203 行）定义 5 种初始化器,全部继承自 `BaseModelInitializer`（第 27 行）。

### 8.1 基类 BaseModelInitializer

`initialize()` 方法（第 50 行）构建 MCore `GPTModel`:

```python
model = GPTModel(
    config=self.tfconfig,
    transformer_layer_spec=transformer_layer_spec,
    vocab_size=self.hf_config.vocab_size,
    max_sequence_length=self.hf_config.max_position_embeddings,
    position_embedding_type="rope",
    rotary_base=get_hf_rope_theta(self.hf_config),
    ...)
```

当 `value=True` 时,将 `output_layer` 替换为 `LinearForLastLayer(hidden_size, 1)`——用于 Critic（价值模型）。

### 8.2 各初始化器的特殊逻辑

| 初始化器 | 特殊逻辑 |
|---------|---------|
| `DenseModel` (第 99 行) | 纯标准 GPT 层规格,无额外处理 |
| `Qwen2MoEModel` (第 108 行) | Patch shared experts gate=True; 默认冻结 router |
| `MixtralModel` (第 132 行) | 默认不冻结 router (可配置) |
| `Qwen3MoEModel` (第 150 行) | 默认冻结 router |
| `DeepseekV3Model` (第 169 行) | 冻结 router 时关闭 load_balancing; 支持 MTP block spec |

**冻结 Router 的设计动机**: 在 RL 训练中,router 的辅助损失 (aux loss) 会影响策略优化的稳定性,因此 MoE 模型的 router 通常被冻结。

---

## 9. VLM 补丁策略

VLM（视觉语言模型）需要特殊处理,因为 vision encoder 和 language model 有不同的注意力模式。

### 9.1 三步补丁策略

每种 VLM 都遵循相同的三步补丁模式:

**Step 1: 模型前向替换**
```python
# 以 Qwen2.5-VL 为例 (monkey_patch.py 第 384-387 行)
Qwen2_5_VLModel.forward = qwen2_vl_base_forward
Qwen2_5_VLForConditionalGeneration.forward = forward_with_normal_backend
```
替换 conditional generation 模型的 forward 以支持 verl 的训练数据格式。

**Step 2: 注意力替换 (用于 Ulysses SP)**
```python
# 第 405-406 行
Qwen2_5_VLAttention.forward = qwen2_vl_attn_forward
```
注入 Ulysses All-to-All 通信到注意力层。

**Step 3: 输入切片 (用于 Ulysses SP)**
```python
# 第 411 行
patch_vlm_for_ulysses_input_slicing(Qwen2_5_VLTextModel)
```
`patch_vlm_for_ulysses_input_slicing()`（第 158 行）包装 decoder 的 forward,在首次调用时沿序列维度切片 `inputs_embeds` 和 `position_ids`。对于 Qwen3-VL 等模型,还需要切片 `visual_pos_masks` 和 `deepstack_visual_embeds`。

### 9.2 TiledMLP 优化

`tiled_mlp.py`（236 行）将 MLP 的 gate+up+down 投影分块执行:

```
标准 MLP: hidden_states --> [gate_proj, up_proj] --> silu --> down_proj
TiledMLP:  分 num_shards 块计算,每块只处理 1/num_shards 的隐藏维度
```

这种方式用更多的 kernel launch 换取更低的峰值显存,适合大模型训练。

---

## 10. 数据格式工具 (util.py)

`util.py`（785 行）提供前向传播所需的数据格式转换函数:

### 预处理函数族
| 函数 | 用途 |
|------|------|
| `preprocess_packed_seqs()` | Padded --> THD packed sequences (计算 cu_seqlens) |
| `preprocess_bshd()` | Padded --> BSHD (标准 padding 格式) |
| `preprocess_thd_engine()` | Engine 模式的 THD 预处理 |
| `preprocess_bshd_engine()` | Engine 模式的 BSHD 预处理 |

### 后处理函数族
| 函数 | 用途 |
|------|------|
| `postprocess_packed_seqs()` | THD --> Padded (恢复 padding) |
| `postprocess_bshd()` | BSHD 后处理 |
| `postprocess_thd_engine()` | Engine 模式 THD 后处理 |
| `postprocess_bshd_engine()` | Engine 模式 BSHD 后处理 |

### VLM 注意力掩码构建
| 函数 | 用途 |
|------|------|
| `build_vlm_attn_mask_thd()` | 为 VLM 构建 THD 格式注意力掩码 |
| `build_vlm_attn_mask_bshd()` | 为 VLM 构建 BSHD 格式注意力掩码 |

---

## 11. 支持的前向函数分发

`registry.py` 同时提供了两套前向函数分发接口:

### 旧接口 (基于注册表)
```python
# MODEL_FORWARD_REGISTRY / MODEL_FORWARD_FUSED_REGISTRY
# 使用 SupportedModel 枚举作为 key
forward_fn = MODEL_FORWARD_REGISTRY[SupportedModel.LLAMA]
```

### 新接口 (基于 HF config)
```python
# 第 40-81 行的 get_mcore_forward_fn / get_mcore_forward_fused_fn 等
forward_fn = get_mcore_forward_fn(hf_config)  # 自动检测 VLM
engine_fn = get_mcore_engine_forward_fn(hf_config)
fused_fn = get_mcore_forward_fused_fn(hf_config)
fused_engine_fn = get_mcore_forward_fused_model_engine_fn(hf_config)
```

新接口通过 `SupportedVLM` 枚举（第 29 行）检测 VLM 架构,自动传递 `vision_model=True`。

---

## 12. 设计约束与扩展点

### 设计约束
1. `SupportedModel` 的 value 必须与 HF config 中的 `architectures[0]` 完全匹配
2. 权重转换器必须保证 QKV/gate+up 的拆分顺序与 HF 权重格式一致
3. 配置转换器中的并行配置从 `megatron.core.parallel_state` 动态获取,不可硬编码
4. Monkey Patch 必须兼容 transformers 4.52+ 的 API 变更（多个版本兼容分支）

### 扩展点
1. **新模型架构**: 在 `SupportedModel` 枚举和 4 个注册表中添加条目
2. **新前向路径**: 通过 `model_forward_gen()` 工厂或自定义前向函数
3. **新 VLM 补丁**: 在 `monkey_patch.py` 中添加新的 `model_type` 分支
4. **新融合 kernel**: 在 `transformers/` 子目录中添加 `forward_with_{torch,triton}_backend`
5. **新权重转换器**: 继承 `McoreToHFWeightConverterBase` 并实现 `convert_param()`
