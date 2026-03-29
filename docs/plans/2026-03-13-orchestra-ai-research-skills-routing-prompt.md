# Orchestra AI Research Skills Routing Prompt

**Date:** 2026-03-13

**Purpose**

`Orchestra-Research/AI-Research-SKILLs` contains 85 skills. This document provides:

- a copy-paste prompt for routing tasks to the right skill(s)
- a concise scenario-to-skill cheat sheet
- a full category map so the skill library is actually usable

## Copy-Paste Prompt

```text
你是一个 AI Research Engineering Skill Router。你的第一职责不是立刻动手，而是先判断当前任务在 AI 研究/训练/部署生命周期中的位置，并为任务选择最合适的 Orchestra Skill。

在开始执行前，严格按下面流程工作：

1. 先识别任务类型
   - 这是想法探索、论文构思，还是代码实现？
   - 是数据处理、tokenizer、模型结构、微调、RL/post-training、分布式训练、推理部署、评测、可观测、MLOps、RAG、Agent、多模态、安全，还是性能优化？
   - 用户是否明确提到了某个框架、库或系统名？

2. 选择 Skill 的基本规则
   - 如果用户点名了具体框架或系统，优先使用同名 Skill。
   - 如果用户没有点名框架，就按任务所处阶段选择最贴近的类别 Skill。
   - 默认选 1 个主 Skill；只有确实跨阶段时，才补 1 到 2 个辅助 Skill。
   - 不要一次加载过多 Skill。通常上限为 3 个。

3. 输出你的路由判断
   在真正执行前，先明确写出：
   - Primary skill:
   - Supporting skills:
   - Why these skills:
   - What each skill will be used for:

4. 路由优先级
   - 具体框架名 > 生命周期阶段 > 通用辅助能力
   - 算法/训练框架优先于基础设施
   - 主问题优先于配套问题

5. 组合 Skill 的方式
   - 基础训练/后训练算法 + 分布式训练
   - 推理引擎 + 优化/量化
   - RAG/Agent 主框架 + 向量库/embedding
   - 训练/部署主 Skill + 评测/可观测/MLOps
   - 安全需求出现时，加安全类 Skill

6. 特殊规则
   - 如果任务是构思研究方向：优先用 `brainstorming-research-ideas` 或 `creative-thinking-for-research`
   - 如果任务是写论文、整理实验到投稿稿件：优先用 `20-ml-paper-writing`
   - 如果任务是 benchmark、harness、指标对比：优先用评测类 Skill
   - 如果任务是 tracing、监控、debug LLM app：优先用 `langsmith` 或 `phoenix`
   - 如果任务是 agentic/RAG 系统：不要只选 agent 框架，要同时判断是否需要 RAG、结构化输出和 observability

7. 在 `verl` / RL post-training 项目中的默认路由
   - 涉及 GRPO / PPO / RLHF / reward shaping / rollout / actor-ref / trainer / HybridFlow：主 Skill 默认 `verl`
   - 涉及 rollout engine：
     - vLLM 路线用 `vllm`
     - SGLang 路线用 `sglang`
   - 涉及训练分片/显存策略：
     - FSDP 路线用 `pytorch-fsdp2`
     - Megatron 路线用 `megatron-core`
     - 通用并行/多机编排可补 `ray-train`、`deepspeed`、`accelerate`
   - 涉及吞吐、显存、量化或 kernel 优化：
     - `flash-attention`
     - `bitsandbytes`
     - `awq` / `gptq` / `hqq` / `gguf`
   - 涉及实验跟踪、指标、debug：
     - `tensorboard`
     - `weights-and-biases`
     - `mlflow`
     - `phoenix` / `langsmith`

8. 输出风格
   - 先路由，后执行
   - 简明，不要泛泛而谈
   - 如果多个 Skill 都能做，优先推荐最贴合用户栈、最少迁移成本、最贴近当前代码库的那个

现在开始：先根据用户任务做 Skill 路由判断，再继续执行。
```

## Quick Routing Cheat Sheet

### 1. Research ideation and paper writing

- 想找研究方向、生成研究问题、设计 novelty angle：`brainstorming-research-ideas`
- 想做更跳跃、更非线性的创新发散：`creative-thinking-for-research`
- 想把实验整理成论文、补 LaTeX 模板、检查引用：`20-ml-paper-writing`

### 2. Build or modify model architecture

- 想做干净、可读、训练路径清晰的 GPT/LLaMA 风格基线：`litgpt`
- 想做极简教育型 GPT baseline、快速理解核心实现：`nanogpt`
- 想做 SSM / selective state space 路线：`mamba`
- 想做 RWKV 路线：`rwkv`
- 想用 PyTorch 原生、面向大规模训练的 Llama/Titan 风格栈：`torchtitan`

### 3. Tokenizer and text preprocessing

- 想训练或定制高性能 tokenizer、BPE/WordPiece/Unigram：`huggingface-tokenizers`
- 想用语言无关的子词切分、T5/ALBERT 风格 tokenizer：`sentencepiece`

### 4. SFT, LoRA, QLoRA, PEFT

- 想用 YAML recipe 快速做 SFT / instruction tuning：`axolotl`
- 想要 WebUI / no-code 微调体验：`llama-factory`
- 想用更快、更省显存的 QLoRA 训练：`unsloth`
- 想系统做 LoRA / QLoRA / DoRA / 各类参数高效微调：`peft`

### 5. RLHF and post-training

- 想做通用 TRL 流水线：`trl-fine-tuning`
- 想做 GRPO，并且希望按 TRL 的 gold-standard 方式落地：`grpo-rl-training`
- 想做 Ray + vLLM 的完整 RLHF 管线：`openrlhf`
- 想做 SimPO：`simpo`
- 想做 `verl` / HybridFlow / actor-rollout-ref / RL 数据流：`verl`
- 想做基于 Megatron + SGLang 的 post-training：`slime`
- 想做偏企业级、高性能、MoE/speculative RL 的扩展路线：`miles`
- 想做 Meta/PyTorch-native RL 栈：`torchforge`

### 6. Mechanistic interpretability

- 想做 HookPoints、activation cache、circuit analysis：`transformer-lens`
- 想做 SAE 训练与 feature discovery：`saelens`
- 想做 intervention / causal representation experiments：`pyvene`
- 想远程跑大模型 interpretability 实验：`nnsight`

### 7. Data curation and large-scale preprocessing

- 想做分布式数据清洗、流式处理、训练前数据管道：`ray-data`
- 想做 GPU 加速数据去重、过滤、curation：`nemo-curator`

### 8. Distributed training and scaling

- 想用 NVIDIA 大模型训练核心栈：`megatron-core`
- 想要 ZeRO / optimizer-state sharding：`deepspeed`
- 想用 PyTorch 原生 FSDP2：`pytorch-fsdp2`
- 想用 Hugging Face 的轻量分布式封装：`accelerate`
- 想用高层 Trainer 风格训练：`pytorch-lightning`
- 想做多机训练编排和调参：`ray-train`

### 9. Inference and serving

- 想高吞吐在线 serving：`vllm`
- 想做 NVIDIA GPU 上极致推理加速：`tensorrt-llm`
- 想做 CPU / Apple Silicon / GGUF 推理：`llama-cpp`
- 想做结构化生成、agent rollout、RadixAttention 路线：`sglang`

### 10. Optimization and model compression

- 想先解决 attention kernel 瓶颈：`flash-attention`
- 想先降显存、做 4-bit/8-bit 训练或推理：`bitsandbytes`
- 想做 GPTQ 后训练量化：`gptq`
- 想做 AWQ：`awq`
- 想做不依赖 calibration 的 HQQ：`hqq`
- 想转 GGUF 并在 llama.cpp 生态运行：`gguf`

### 11. Evaluation

- 想跑通用 LLM benchmark：`lm-evaluation-harness`
- 想跑 code model benchmark：`bigcode-evaluation-harness`
- 想做企业级、多 harness 统一评测：`nemo-evaluator`

### 12. Agent frameworks

- 想搭最通用、生态最大的 agent framework：`langchain`
- 想以数据连接、RAG 为中心构建 agent：`llamaindex`
- 想做多角色、多 agent 协作：`crewai`
- 想做 autonomous agent / continuous execution：`autogpt`

### 13. RAG systems

- 本地优先、开源、轻量 embedding DB：`chroma`
- 极致向量检索性能、大规模 ANN：`faiss`
- 想先把 embedding 模型选好：`sentence-transformers`
- 托管式向量数据库：`pinecone`
- 混合检索、过滤能力强的向量库：`qdrant`

### 14. Prompt engineering and structured output

- 想做 declarative prompt programming / optimizer-driven prompting：`dspy`
- 想要 Pydantic 结构化输出：`instructor`
- 想做 grammar / regex / constrained generation：`guidance`
- 想做 FSM-based 结构化生成：`outlines`

### 15. MLOps and experiment tracking

- 团队常用实验追踪、sweeps、artifacts：`weights-and-biases`
- 想做 model registry + tracking + deployment：`mlflow`
- 想做训练可视化和 profiler：`tensorboard`

### 16. Observability

- 想看 LLM app traces、在线评估、production monitoring：`langsmith`
- 想做开源 observability、OpenTelemetry tracing、eval：`phoenix`

### 17. Safety and alignment

- 想做 principle-based self-improvement / constitutional alignment：`constitutional-ai`
- 想做输入输出安全分类：`llamaguard`
- 想加对话/工作流 guardrails：`nemo-guardrails`
- 想拦 prompt injection / jailbreak：`prompt-guard`

### 18. Multimodal

- 图文对齐、zero-shot image classification：`clip`
- 语音识别 / ASR：`whisper`
- 图像理解聊天 / VLM assistant：`llava`
- 文生图 / Diffusers / SDXL / ControlNet：`stable-diffusion`
- 图像分割：`segment-anything`
- image captioning / VQA：`blip-2`
- 文生音乐 / 文生音效：`audiocraft`

### 19. Emerging techniques

- 想训练 Mixture-of-Experts：`moe-training`
- 想合并多个模型权重：`model-merging`
- 想把上下文窗口拉长到 32k/128k：`long-context`
- 想用 speculative decoding 提升推理速度：`speculative-decoding`
- 想做 teacher-student 蒸馏：`knowledge-distillation`
- 想做稀疏化 / pruning：`model-pruning`

### 20. Infrastructure

- 想上 serverless GPU：`modal`
- 想做多云 GPU 调度：`skypilot`
- 想租用/管理 Lambda GPU 资源：`lambda-labs`

## Recommended Skill Bundles

### `verl` / RL post-training in this repository

- 改 RL 算法、trainer、reward、advantage、rollout 数据流：`verl`
- 改 rollout engine：
  - `vllm`
  - `sglang`
- 改训练并行/显存策略：
  - `pytorch-fsdp2`
  - `megatron-core`
  - `deepspeed`
  - `ray-train`
- 改吞吐/显存热点：
  - `flash-attention`
  - `bitsandbytes`
  - `awq` / `gptq` / `hqq`
- 做训练观测和实验对比：
  - `tensorboard`
  - `weights-and-biases`
  - `mlflow`

### Build an agentic RAG system

- 主框架：`langchain` or `llamaindex`
- 向量库：`qdrant` / `chroma` / `pinecone` / `faiss`
- embedding：`sentence-transformers`
- 结构化输出：`instructor` / `outlines` / `guidance`
- tracing/eval：`langsmith` / `phoenix`
- 安全：`prompt-guard` / `nemo-guardrails`

### Serve a fine-tuned model in production

- 主 serving skill：`vllm` / `sglang` / `tensorrt-llm` / `llama-cpp`
- 若要降显存或量化：`bitsandbytes` / `awq` / `gptq` / `hqq` / `gguf`
- 若要加速 attention：`flash-attention`
- 若要追踪线上效果：`langsmith` / `phoenix`

### Train a new model at scale

- 架构主 Skill：`litgpt` / `nanogpt` / `mamba` / `rwkv` / `torchtitan`
- 分布式：`megatron-core` / `pytorch-fsdp2` / `deepspeed` / `ray-train`
- 数据：`ray-data` / `nemo-curator`
- tokenizer：`huggingface-tokenizers` / `sentencepiece`
- tracking：`weights-and-biases` / `mlflow` / `tensorboard`

## Full Category Map

### 01 Model Architecture

- `litgpt`: clean GPT/LLaMA-style implementation and production-ready training recipes
- `mamba`: state-space model experiments and long-sequence alternatives to Transformers
- `nanogpt`: educational, minimal GPT codepath for rapid understanding and prototyping
- `rwkv`: RNN-Transformer hybrid experiments and infinite-context style exploration
- `torchtitan`: PyTorch-native large-scale Llama training with strong distributed support

### 02 Tokenization

- `huggingface-tokenizers`: fast Rust tokenizer training, BPE/WordPiece/Unigram pipelines
- `sentencepiece`: language-agnostic subword tokenization and tokenizer research workflows

### 03 Fine-Tuning

- `axolotl`: config-driven SFT and instruction tuning across many open models
- `llama-factory`: UI-first or low-code fine-tuning workflows
- `peft`: LoRA, QLoRA, DoRA, adapters, and broader parameter-efficient tuning
- `unsloth`: faster, lower-memory QLoRA fine-tuning

### 04 Mechanistic Interpretability

- `nnsight`: remote interpretability experiments on large hosted models
- `pyvene`: causal interventions, representation editing, and interpretability experiments
- `saelens`: sparse autoencoder workflows and feature analysis
- `transformer-lens`: transformer internals, hooks, caches, and circuit analysis

### 05 Data Processing

- `nemo-curator`: large-scale data curation, deduplication, and filtering
- `ray-data`: distributed ETL, data loading, and training-data preprocessing

### 06 Post-Training

- `grpo-rl-training`: GRPO-specific implementation patterns using TRL
- `miles`: enterprise-grade RL post-training with aggressive performance features
- `openrlhf`: end-to-end RLHF with Ray and vLLM
- `simpo`: preference optimization without a reference model
- `slime`: Megatron + SGLang RL/post-training stack
- `torchforge`: Meta-style PyTorch-native RL stack
- `trl-fine-tuning`: general TRL usage for RLHF and preference optimization
- `verl`: HybridFlow RL post-training, actor-rollout-ref pipelines, GRPO/PPO style systems

### 07 Safety Alignment

- `constitutional-ai`: principle-driven self-improvement and alignment
- `llamaguard`: safety classification for prompts and generations
- `nemo-guardrails`: programmable interaction guardrails for LLM applications
- `prompt-guard`: prompt injection and jailbreak detection

### 08 Distributed Training

- `accelerate`: lightweight distributed training abstraction around PyTorch/Hugging Face
- `deepspeed`: ZeRO-based memory scaling and distributed optimizer strategies
- `megatron-core`: tensor/pipeline/data/expert parallel large-scale training
- `pytorch-fsdp2`: PyTorch native full sharding workflows
- `pytorch-lightning`: higher-level trainer-based distributed training organization
- `ray-train`: cluster orchestration, multi-node execution, and tuning loops

### 09 Infrastructure

- `lambda-labs`: Lambda GPU provisioning and cloud workflows
- `modal`: serverless GPU jobs and Python-native cloud execution
- `skypilot`: multi-cloud GPU scheduling, spot recovery, and portable infra

### 10 Optimization

- `awq`: activation-aware quantization for efficient serving
- `bitsandbytes`: 4-bit/8-bit quantization and memory-efficient training
- `flash-attention`: memory-efficient fast attention kernels
- `gguf`: GGUF conversion and llama.cpp-oriented deployment
- `gptq`: post-training quantization with GPTQ
- `hqq`: calibration-light or calibration-free quantization workflows

### 11 Evaluation

- `bigcode-evaluation-harness`: code-generation benchmark suites and pass@k evaluation
- `lm-evaluation-harness`: broad LLM benchmarking across standard NLP tasks
- `nemo-evaluator`: large-scale, enterprise-style multi-harness evaluation

### 12 Inference Serving

- `llama-cpp`: local CPU/Metal inference and GGUF deployment
- `sglang`: structured generation and high-throughput agentic rollout serving
- `tensorrt-llm`: NVIDIA-optimized inference for maximum GPU throughput
- `vllm`: PagedAttention-based production LLM serving

### 13 MLOps

- `mlflow`: experiment tracking, registry, deployment, and lineage
- `tensorboard`: training visualization, scalars, embeddings, and profiling
- `weights-and-biases`: experiment tracking, comparisons, sweeps, and artifacts

### 14 Agents

- `autogpt`: autonomous or continuously running agent workflows
- `crewai`: role-based multi-agent orchestration
- `langchain`: general-purpose agent and LLM application framework
- `llamaindex`: data-centric agents and RAG-first application patterns

### 15 RAG

- `chroma`: lightweight, open-source vector store
- `faiss`: large-scale similarity search and ANN indexing
- `pinecone`: managed vector database service
- `qdrant`: filtered, hybrid, production vector search
- `sentence-transformers`: embedding model selection and semantic retrieval baselines

### 16 Prompt Engineering

- `dspy`: optimizer-driven declarative prompt programming
- `guidance`: grammar-constrained generation and prompt control
- `instructor`: schema-validated structured output with Pydantic-style models
- `outlines`: FSM-backed structured generation and constrained decoding

### 17 Observability

- `langsmith`: tracing, evaluation, and monitoring for LLM apps
- `phoenix`: open-source tracing and evaluation for AI systems

### 18 Multimodal

- `audiocraft`: text-to-audio, text-to-music generation
- `blip-2`: image captioning, VQA, and vision-language transfer
- `clip`: image-text alignment and zero-shot vision classification
- `llava`: multimodal chat and image understanding
- `segment-anything`: promptable image segmentation
- `stable-diffusion`: text-to-image generation pipelines
- `whisper`: multilingual ASR and speech transcription

### 19 Emerging Techniques

- `knowledge-distillation`: teacher-student compression and transfer
- `long-context`: context extension strategies and positional encoding choices
- `model-merging`: model fusion with mergekit-style methods
- `model-pruning`: sparsification and parameter removal
- `moe-training`: mixture-of-experts training workflows
- `speculative-decoding`: draft-target decoding for faster inference

### 20 ML Paper Writing

- `20-ml-paper-writing`: academic paper drafting, template selection, experiment presentation, citation hygiene

### 21 Research Ideation

- `brainstorming-research-ideas`: structured generation of promising research directions
- `creative-thinking-for-research`: novelty-seeking ideation via cognitive creativity frameworks

## Short Selection Heuristics

- 用户提到具体框架名：直接选同名 Skill
- 用户说“我想做 RLHF / GRPO / PPO / rollout / reward model / actor-ref”：先选 `verl` 或 `trl-fine-tuning` / `openrlhf`
- 用户说“我想部署”：先判断是 `vllm`、`sglang`、`tensorrt-llm` 还是 `llama-cpp`
- 用户说“我显存不够”：优先考虑 `bitsandbytes`、`flash-attention`、`awq`、`gptq`、`hqq`
- 用户说“我想做 agent + knowledge base”：优先考虑 `langchain` / `llamaindex` + `qdrant` / `chroma` + `sentence-transformers`
- 用户说“我想看 trace / 评估 agent 行为”：优先考虑 `langsmith` 或 `phoenix`
- 用户说“我想做多模态”：先分清是语音、图像理解、文生图、分割还是音频生成
- 用户说“我想写论文”：直接用 `20-ml-paper-writing`

## Recommended Default

如果任务不够明确，先按下面顺序做路由：

1. 是否点名具体框架
2. 如果没有，判断任务位于哪个生命周期阶段
3. 选 1 个主 Skill
4. 如有必要，再补 1 到 2 个辅助 Skill 处理分布式、观测、评测或安全
5. 先说明为什么选这些 Skill，再开始执行
