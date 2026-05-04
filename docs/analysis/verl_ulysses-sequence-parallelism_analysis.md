# VERL 源码走读：Ulysses Sequence Parallelism 实现解析

此前，我们走读了 VERL 系列源码，介绍了 Full Async 特性从架构层次上解决强化学习中长尾问题的思路。但长序列场景的问题远不止于此——即使生成侧不再有长尾阻塞，训练侧的显存压力同样是绕不过去的坎。本文从性能和显存的角度，走读 VERL 社区给出的解决方案：在 FSDP 后端下集成 Ulysses Sequence Parallelism。

往期系列：

1. [共卡共进程方案介绍](https://zhuanlan.zhihu.com/p/2016583362190530384)
2. [AgentLoop 与 vLLM Async Server 方案介绍](https://zhuanlan.zhihu.com/p/2016933521420219614)
3. [Full Async 分离方案介绍](https://zhuanlan.zhihu.com/p/2018651508619616893)
4. [Partial Rollout 工程实现介绍](https://zhuanlan.zhihu.com/p/2019147267240653513)

FSDP源码走读系列：

- [FSDP2 源码解析（一）：初始化与参数分片](https://zhuanlan.zhihu.com/p/2017329018903503356)
- [FSDP 源码解析（一）：初始化与显存分配](https://zhuanlan.zhihu.com/p/2014097067107182088)
- [FSDP 源码解析（二上）：前向与反向——执行链路全解析](https://zhuanlan.zhihu.com/p/2014783611962943132)


---


# 前言

以 GRPO 算法为主的强化学习训练，按阶段划分可以分为推理和训练两个部分：

- **推理阶段**：使用 vLLM / SGLang 后端，基于 prompt 输出 n 条 response
- **训练阶段**：使用 FSDP / Megatron 后端，从计算内容上可以细分为两个子阶段：
- **前向阶段**：计算 `old_logp` 和 `ref_logp`，即模型前向生成 logits，再从 logits 中得到 log prob，无反向传播
- **update 阶段**：计算 `pg_loss`，在前向阶段的基础上进行反向传播，更新权重

在长序列场景，无论哪个阶段，显存都是首要瓶颈。推理阶段可以借助 chunked prefill 或 vLLM 支持的 PCP/DCP 控制峰值显存；训练侧若使用 Megatron 后端，原生支持 TP-SP和Context Parallel（ Ring Attention） 等序列切分方案。而 **FSDP 后端默认不支持 SP**，在长序列场景需要上层框架——也就是 VERL 这一层——做额外的适配工作。

**本文重点介绍 VERL 框架如何在 FSDP 后端下集成 Ulysses SP，集成后能带来哪些收益、有哪些局限——这也是为什么超长序列业务仍会倾向于 Megatron 后端的原因之一。**

行文结构如下：第一章介绍初始化与 dispatch/collect 机制；第二章介绍 `FSDPUlyssesShardingManager` 的职责边界；第三章介绍 all-to-all 的注入方式；第四章串通完整的前向调用栈；第五章分析局限性与已知优化点。

注意，本文并不会介绍Ulysses的原理，感兴趣的同学可直接参考猛猿大佬的文章：[(1 封私信 / 3 条消息) 图解大模型训练系列：序列并行2，DeepSpeed Ulysses - 知乎](https://zhuanlan.zhihu.com/p/4496065391)

**本文涉及的核心文件：**

| 文件                                            | 职责                                     |
| --------------------------------------------- | -------------------------------------- |
| verl/workers/fsdp_workers.py                  | ActorRolloutRefWorker，初始化与调用入口         |
| verl/single_controller/base/worker.py         | Worker 基类，dispatch/collect 注册与查询       |
| verl/single_controller/base/decorator.py      | dispatch/collect 执行逻辑                  |
| verl/workers/sharding_manager/fsdp_ulysses.py | FSDPUlyssesShardingManager，SP group 切换 |
| verl/workers/actor/dp_actor.py                | compute_log_prob / update_policy，前向核心  |
| verl/models/transformers/monkey_patch.py      | apply_monkey_patch，all-to-all 注入       |
| verl/utils/ulysses.py                         | 通信原语：SeqAllToAll / Gather              |


---


# 一、初始化与 Dispatch/Collect 机制

## 1.1 设计哲学与核心问题

初始化阶段要解决两个核心问题：**在 FSDP 的并行体系里把 Ulysses SP 安插进去**，以及**让 Controller 知道如何在有 SP 的情况下正确分发和收集数据**。

理解这两点，是读懂后续所有机制的前提。直觉上，SP 要求同一 SP 组内所有卡拿到相同的输入数据——但 FSDP 的 DP 机制是把不同数据发给不同的卡。这个矛盾必须在架构层面解决，而不是靠运行时通信打补丁。

![](https://pic4.zhimg.com/v2-b2d19dd6a209367273c9365b44a57edf_r.jpg)

## 1.2 Device Mesh 的两层设计

FSDP 本身已有一套 DeviceMesh 管理参数分片，Ulysses SP 需要的是另一套管理序列切分的通信域。VERL 的选择是两套独立建立、互不干涉：

```python
# 第一层：FSDP Mesh，控制参数在卡间如何分片
self.device_mesh = create_device_mesh(
    world_size=world_size,
    fsdp_size=config.actor.fsdp_config.fsdp_size
)

# 第二层：Ulysses Mesh，控制序列切分的通信域，shape=(dp, sp)
self.ulysses_device_mesh = init_device_mesh(
    device_name,
    mesh_shape=(dp, ulysses_sequence_parallel_size),
    mesh_dim_names=["dp", "sp"]
)
```

以 `world_size=8`、`sp_size=4` 为例，Ulysses Mesh 的拓扑如下：

```python
全局 8 卡
├── FSDP Mesh：关于参数分片，与 SP 拓扑无关
└── Ulysses Mesh：shape=(dp=2, sp=4)
      SP 组 0: [rank0, rank1, rank2, rank3]  ← 同组 4 卡共同处理一条序列
      SP 组 1: [rank4, rank5, rank6, rank7]
      DP 组 0: [rank0, rank4]
      DP 组 1: [rank1, rank5]  ...
```

![](https://pic2.zhimg.com/v2-59e296ce2135ccb897bb2ab5e3d81a99_r.jpg)

两层 Mesh 之所以独立而不合并，是因为两者用途完全不同——FSDP Mesh 决定参数如何分片存储，Ulysses Mesh 决定序列如何切分计算。它们的交互通过 dispatch/collect 机制和 `FSDPUlyssesShardingManager` 分别在两个层面协调，不依赖 PyTorch 原生的多维 Mesh 机制。

## 1.3 Dispatch/Collect：三层结构

在第一篇《共卡共进程方案介绍》中，我们介绍了 VERL 通过 `@register` 装饰器注入数据从 Driver 下发到各 Worker 时需要按各自的 mesh 拓扑做切分和路由的逻辑。

引入 SP 后，从数据的角度来说，训练的 mesh 发生了变化，Controller 因此需要知道两件事：数据应该发给哪些 rank（dispatch），以及哪些 rank 的结果需要收集回来（collect）。

![](https://pica.zhimg.com/v2-134e07396a5e53f67f69f68c3fb1b430_r.jpg)

VERL 在 SP 场景添加的代码并不多，本质上还是复用原始的数据分发逻辑，用一套三层结构解决这个问题——不得不说实现得相当优雅。

**注册层（初始化时）**：每个 rank 把自己的 `dp_rank` 和 `is_collect` 注册到 Worker 基类的私有字典里：

```python
if self.ulysses_device_mesh is not None:
    is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
    self._register_dispatch_collect_info(
        "actor",
        dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(),
        is_collect=is_collect   # SP 组内只有 sp_rank==0 的进程参与结果收集
    )
```

**查询层（第一次调用时，惰性初始化）**：Controller 通过 `ONE_TO_ALL` 广播调用 `_query_dispatch_info` 和 `_query_collect_info`，收集所有 rank 的注册信息，构建两张映射表，缓存备用：

```python
dp_rank_mapping = [0, 0, 0, 0, 1, 1, 1, 1]   # world_size=8, sp=4, dp=2
collect_mask    = [T, F, F, F, T, F, F, F]    # 每个 SP 组只有 sp_rank==0 收集
```

**执行层（每次函数调用时）**：

- **dispatch**：`_split_args_kwargs_data_proto(dp_size=2, data)` 把全局 batch `chunk(2)`，再由 `dispatch_nd_compute` 按 `dp_rank_mapping` 分发：

```python
for i in range(world_size):             # i = 0,1,2,3,4,5,6,7
    local_dp_rank = dp_rank_mapping[i]  # 0,0,0,0,1,1,1,1
    transformed_args.append(arg[local_dp_rank])
# rank0~3 都拿 chunk[0]，rank4~7 都拿 chunk[1]
```

同一 SP 组内的所有 rank `dp_rank` 相同，因此在 Controller 侧分发时就**自动拿到同一份数据，无需任何额外通信**。

- **collect**：`collect_nd_compute` 按 `collect_mask` 只收 `sp_rank==0` 的结果，其余 rank 的输出直接丢弃，避免重复收集 `sp_size` 份相同结果。

这个机制有一个重要推论，也是很多人走读这部分代码时容易产生的误解：**`FSDPUlyssesShardingManager.preprocess_data` 在标准训练主流程中永远不会被调用**。SP 组内的数据一致性由 Controller 的 dispatch 机制在分发时就已保证，Worker 侧无需任何额外 AllGather。`preprocess_data` 和 `postprocess_data` 看起来像是主流程的一部分，实则是留给不经过 Controller dispatch 的场景（如单元测试）的备用接口。

## 1.4 Batch Size 归一化

开启 SP 时，用户不需要手动调整 batch size 或 world size，VERL 会自动完成归一化。但要理解归一化的逻辑，需要先厘清几个 batch 相关参数的关系：

- `train_batch_size`：本次迭代从数据集读取的 prompt 数目。数据进入推理引擎后，经过 rollout.n 倍的 repeat，生成 train_batch_size × n 条 response。
- `ppo_mini_batch_size`：一次权重更新所需的 prompt 数目，要求被 train_batch_size 整除。每次迭代共更新 **train_batch_size / ppo_mini_batch_size **次，若该比值为 1 则为 on-policy。数据分发后，**每张卡实际的 mini_batch = ppo_mini_batch_size × n / dp。**
- `ppo_micro_batch_size_per_gpu`：每张卡单次前向执行的数据量，**是真正影响显存的参数。**开启 dynamic batch 后该参数不生效，改由 max_token_len 控制。
- `ppo_micro_batch_size`：全局视角的 micro_batch，与 ppo_micro_batch_size_per_gpu 的关系是 per_gpu = micro / dp。**梯度累积步数 = 实际的 mini_batch / micro_batch_per_gpu = ppo_mini_batch_size × n / dp / ppo_micro_batch_size_per_gpu。**

SP 不切 batch，只切 sequence。引入 SP 后真正变化的是 DP 数（`DP 数 = world_size / sp_size`），全局 batch size 语义不变，需要按新的 DP 数重新均分到每张卡。

`ppo_mini_batch_size` 的归一化分两步：先乘 `rollout.n`，再除以 DP 数。开启 SP 后 DP 数减少，每张卡的 mini_batch 等比增大：

```python
self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
```

`ppo_micro_batch_size` 在用户配置了该字段时，同样因 DP 等比变化：

```python
if self.config.actor.ppo_micro_batch_size is not None:
    self.config.actor.ppo_micro_batch_size //= (
        self.device_mesh.size() // self.ulysses_sequence_parallel_size
    )
    self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
```

mini_batch 和 micro_batch 等比变化，梯度累积步数 `= mini_batch / micro_batch` **保持不变**。

这里有一个容易踩的配置陷阱：`ppo_micro_batch_size` 归一化后会同步写入 `ppo_micro_batch_size_per_gpu`，但如果用户**直接配置 `ppo_micro_batch_size_per_gpu`** 而不配置 `ppo_micro_batch_size`，则 per_gpu 值不会被归一化。此时 SP 扩大后单卡 mini_batch 变大，micro_batch 不变，梯度累积步数被动增加。代码逻辑其实可以写得更清晰一些，例如：

```python
if self.config.actor.ppo_micro_batch_size is not None:
    dp_size = self.device_mesh.size() // self.ulysses_sequence_parallel_size
    self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size // dp_size
```

>  💡 **小结**
>

- SP 组内的数据一致性由 **Controller 侧 dispatch 机制**保证——同 SP 组的 rank `dp_rank` 相同，分发时自动拿到同一份 chunk，Worker 侧无需任何通信。
- dispatch/collect 是三层结构：**注册**（初始化）→ **查询**（ONE_TO_ALL 惰性收集，结果缓存）→ **执行**（每次按缓存的 mapping/mask 分发和过滤）。SP 只在注册层添加了少量代码，其余完全复用原有逻辑。
- `ppo_mini_batch_size` 归一化前先乘 `rollout.n`；配置了 `ppo_micro_batch_size` 时随 DP 数等比变化，梯度累积步数不变；直接配置 `ppo_micro_batch_size_per_gpu` 时不被归一化，梯度累积步数增加。


---


# 二、FSDPUlyssesShardingManager 的职责边界

## 2.1 设计哲学与核心问题

明确了 dispatch 机制保证数据一致性之后，`FSDPUlyssesShardingManager` 的定位就非常清晰了。它要解决的问题是：Actor 和 Ref 各自独立部署，每个模型有自己的 SP group——在多模型交替调用的场景下，如何保证每次前向走的是正确的通信域，而不是互相污染？

答案是用 context manager 做 SP group 的切换和恢复。**`FSDPUlyssesShardingManager` 在训练主流程中的核心职责只有一件事：切换 SP group。**

![](https://pic2.zhimg.com/v2-ac6f6f4c50c2a916f36d8a234c312181_r.jpg)

## 2.2 __enter__ / __exit__：SP group 切换

```python
def __enter__(self):
    if self.device_mesh is not None:
        self.prev_sp_group = get_ulysses_sequence_parallel_group()
        set_ulysses_sequence_parallel_group(self.device_mesh["sp"].get_group())

def __exit__(self, exc_type, exc_value, traceback):
    if self.device_mesh is not None:
        set_ulysses_sequence_parallel_group(self.prev_sp_group)
```

进入 `with` 块时把全局 SP group 切换到当前模型的 SP group，退出时恢复前一个状态。这保证了 `_forward_micro_batch` 内部的 all-to-all 通信走正确的通信域。

切换的底层机制很简单——`ulysses.py` 里维护了一个全局变量 `_ULYSSES_SEQUENCE_PARALLEL_GROUP`，`set/get_ulysses_sequence_parallel_group` 就是对它的读写：

```python
_ULYSSES_SEQUENCE_PARALLEL_GROUP = None

def set_ulysses_sequence_parallel_group(group):
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    _ULYSSES_SEQUENCE_PARALLEL_GROUP = group

def get_ulysses_sequence_parallel_group():
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    return _ULYSSES_SEQUENCE_PARALLEL_GROUP
```

这里可能有同学会问：为什么要做切换，直接用一个固定的 group 不行吗？原因在于 ref 和 actor 由于阶段不同，显存压力也不一样——ref 只有前向，actor 有前向和反向。在资源紧张的场景下，两者可能会配置不同的 `sp_size`，即通信组不一致。用全局变量 + 切换的方式，让两个模型可以共享同一套 `gather_seq_scatter_heads` 等通用函数，而不需要把通信组作为参数层层传递。

## 2.3 preprocess_data / postprocess_data 的真实定位

这两个方法在源码里是存在的，容易让人误以为它们是主流程的一部分：

```python
def preprocess_data(self, data: DataProto) -> DataProto:
    if self.device_mesh is not None:
        group = self.device_mesh["sp"].get_group()
        all_gather_data_proto(data=data, process_group=group)
    return data

def postprocess_data(self, data: DataProto) -> DataProto:
    if self.device_mesh is not None:
        sp_size = self.device_mesh["sp"].size()
        sp_rank = self.device_mesh["sp"].get_local_rank()
        data = data.chunk(chunks=sp_size)[sp_rank]
    return data
```

但在标准训练流程中，它们不会被调用。`preprocess_data` 的 AllGather 和 `postprocess_data` 的 chunk，功能上分别对应 dispatch 层的"复制同一 chunk 给 SP 组"和 collect 层的"只收 `sp_rank==0` 的结果"——Controller 侧已经做了，Worker 侧不需要重复。

这两个方法是留给不经过 Controller dispatch 的场景（如单元测试、Worker 内部自发同步）的备用接口，不是主流程的组成部分。

>  💡 **小结**
>

- `FSDPUlyssesShardingManager` 在训练主流程中的唯一职责是**切换 SP group**，`__enter__`/`__exit__` 不触发任何数据变换。
- ref 和 actor 可以配置不同的 `sp_size`，全局变量 + 切换的设计让两个模型共享通用通信函数，避免了参数层层传递。
- `preprocess_data` / `postprocess_data` 是备用接口，其功能已由 Controller 侧的 dispatch/collect 机制替代，主流程不调用。


---


# 三、All-to-All 的注入：Monkey Patch

## 3.1 设计哲学与核心问题

原始 Transformers 的 attention 实现是一个纯本地的计算模块，没有任何分布式通信。要让 Ulysses SP 在每个 attention layer 前后插入 all-to-all，有两种思路：fork Transformers 修改源码，或者在运行时替换函数。VERL 选择了后者，做到**对 Transformers 源码零侵入**。

![](https://pic4.zhimg.com/v2-eda0d57560bbd0cf85dd852f921badf5_r.jpg)

关键的工程洞察在于：Python 的模块系统中，函数调用查找的是**当前模块命名空间里的名字**，而不是导入时绑定的对象。只要在运行时重新绑定这个名字，所有调用该函数的地方都会自动走新逻辑。

## 3.2 命名空间替换

```python
def apply_monkey_patch(model, ulysses_sp_size, ...):
    module = sys.modules[model.__module__]                              # 拿到模型所在的 Python 模块
    module._flash_attention_forward = _ulysses_flash_attention_forward  # 重新绑定名字
```

替换完成后，该模块里所有调用 `_flash_attention_forward` 的地方——也就是每一个 attention layer——都会自动走新逻辑。整个 Transformer forward 对此完全无感知，不需要修改任何 attention 类的代码。

## 3.3 _ulysses_flash_attention_forward：all-to-all 的注入点

替换后的函数在原始 flash attention 前后各插入 all-to-all：

```python
def _ulysses_flash_attention_forward(query_states, key_states, value_states, ..., position_ids):
    if ulysses_sp_size > 1 and position_ids is not None:
        # attention 前：序列维度 gather，head 维度 scatter
        # (bsz, seq/sp, nheads, dim) → (bsz, seq, nheads/sp, dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states   = gather_seq_scatter_heads(key_states,   seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)
        # position_ids 额外 AllGather：flash attn 内部需要完整序列计算 cu_seqlens
        position_ids = all_gather(position_ids, ...)

    attn_output = _flash_attention_forward(...)   # 原始 flash attn，此时每卡持有完整序列、部分 head

    if ulysses_sp_size > 1 and position_ids is not None:
        # attention 后：head 维度 gather，序列维度 scatter
        # (bsz, seq, nheads/sp, dim) → (bsz, seq/sp, nheads, dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    return attn_output
```

![](https://pic1.zhimg.com/v2-274c01e088fa8ee12cb0f77a5697ac40_r.jpg)

值得注意的是，开启 SP 后每个 attention layer 会引入**5 次通信**：q、k、v 各一次 all-to-all（`gather_seq_scatter_heads`）、`position_ids` 一次 AllGather、`attn_output` 一次 all-to-all（`gather_heads_scatter_seq`）。通信开销并不小，这也是第五章会重点讨论的问题。

`position_ids is not None` 在这里同时扮演两个角色：**SP 的生效条件**，以及 **ViT 模块的豁免条件**——ViT 不传 `position_ids`，all-to-all 不会误注入到视觉编码器里。

## 3.4 GQA 的 KV head 处理

标准 Ulysses 要求 `num_kv_heads % sp_size == 0`。GQA 模型（如 Llama3-8B 只有 8 个 KV head）在 `sp_size` 较大时会违反这个约束。VERL 的处理是在 all-to-all 前先把 KV head 复制到足够整除：

```python
# nheads_k=4, sp=8 → repeats=2，复制后 nheads_k=8，可被 sp=8 整除
repeats = max(ulysses_sp_size // key_states.size(2), 1)
key_states   = repeat_kv(key_states,   repeats)
value_states = repeat_kv(value_states, repeats)
```

实际约束放宽为：`num_kv_heads % sp_size == 0` **或** `sp_size % num_kv_heads == 0`。代价是 `repeats > 1` 时 all-to-all 通信量等比增加。

## 3.5 三条 patch 路径

模型类型不同，patch 的复杂度差异很大：

| 模型类型                      | patch 方式           | 核心差异                     |
| ------------------------- | ------------------ | ------------------------ |
| 普通 LLM（Llama、Qwen2 等）     | 一行命名空间替换           | 所有 attention layer 自动受影响 |
| VLM（Qwen2-VL、GLM4V 等）     | 三步 patch           | 通常在文本和图像特征融合后才开始切分       |
| 特殊模型（KimiVL / DeepseekV3） | 直接替换具体 Attention 类 | 无法复用通用路径                 |

VLM 需要三步 patch 的根本原因：文本 token 数量在输入时确定，可以在进入模型前就均匀切分；而 VLM 存在 Vision 部分，业界通常在文本和图像特征融合后才开始切分——这是 VLM 与纯文本 LLM 在 SP 适配上最本质的差异。

有同学会问：ViT 部分的序列能不能也切？技术上是可以的，部分框架确实做了这件事，但 Vision 部分架构异构性较强，实现复杂、通用性不够，所以 VERL 目前不做这个切分。**如果是自研多模态模型接入 SP，需要参考这三步 patch 逻辑自行适配。**

## 3.6 position_ids AllGather：每层一次的已知开销

flash attention 内部调用 `prepare_fa2_from_position_ids` 计算变长序列的 `cu_seqlens`，这个函数需要完整的 `position_ids`。每个 attention layer 为此额外做一次 AllGather，对 32 层的 7B 模型来说是每次前向额外 32 次通信。代码注释中已标注可以通过直接传入预计算的 `cu_seqlens_q/k` 消除（参见 transformers PR #33932），尚未落地。

>  💡 **小结**
>

- all-to-all 通过 Python 模块命名空间替换注入，对 Transformers 源码零侵入。替换后所有 attention layer 自动获得 SP 能力，Transformer forward 无感知。
- 每个 attention layer 进出形状一致：`[bsz, nnz/sp, nheads]` → all-to-all → flash attn → all-to-all → `[bsz, nnz/sp, nheads]`。每层引入 5 次通信（q/k/v 各一次 all-to-all + position_ids AllGather + attn_output all-to-all）。
- VLM 需要三步 patch，核心在于序列切分发生视觉特征和文本特征融合之后，切分时机后移。


---


# 四、前向计算与通信原语

## 4.1 设计哲学与核心问题

前三章分别解决了"数据怎么分发"、"ShardingManager 的边界在哪里"、"all-to-all 怎么注入"三个问题。这一章把它们串起来，回答最终的问题：**一次完整的 `compute_log_prob`，在 SP 下是怎么流转的？显存在哪里节省、在哪里又回到冗余？反向传播的梯度为什么需要特殊处理？**

## 4.2 完整调用栈

从 `ActorRolloutRefWorker.compute_log_prob` 为入口，SP 相关的操作分布在调用栈的不同层级：

```python
ActorRolloutRefWorker.compute_log_prob(data)
  │  # 设置 meta_info（micro_batch_size, max_token_len, use_dynamic_bsz, temperature）
  └── with self.ulysses_sharding_manager:        # 只切换全局 SP group，无数据变换
        actor.compute_log_prob(data)             # data 原样传入，无预处理
          │  # 按 dynamic_bsz 或 static size 切 micro_batch，循环前向
          └── for micro_batch in micro_batches:
                micro_batch.to(get_device_id())  # 数据在这里上 GPU
                with torch.no_grad():
                    _forward_micro_batch()       # SP 核心流水线（见 4.3）
        DataProto.from_dict(...)                 # ← 在 with 块内
  output.to("cpu")                               # ← 在 with 块外
```

`with self.ulysses_sharding_manager` 块内没有任何 `preprocess_data` / `postprocess_data` 的调用，`with` 块的唯一作用是切换 SP group。SP 的序列切分全部在 `_forward_micro_batch` 内部完成。

## 4.3 _forward_micro_batch：序列切分 5 步流水线

`_forward_micro_batch` 是 SP 序列切分的核心执行单元，所有 shape 变化都发生在这里。流水线分 5 步：

![](https://pic1.zhimg.com/v2-15542ee18b3c4bb21e472c96891dd7f6_r.jpg)

**Step 1 — Remove Padding**

```python
# [batch, seqlen] → [1, total_nnz]
input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
```

SP 强依赖 remove padding，原因在于：Ulysses SP 需要在 token 维度上均匀切分序列，如果保留 padding，各卡分到的有效 token 数量会严重不均衡——**padding 集中在序列末尾，切分后最后一张卡会拿到大量无效 padding token，而前面的卡 token 密度更高，造成负载不均。**Remove padding 后，Ulysses 再重新按 token 数均匀 pad 和切分，才能保证每张卡的有效 token 接近。

**Step 2 — Pad & Slice（SP 切分入口）**

```python
if is_vlm_model:
    # VLM：input_ids 只做 pad，不做 slice（图像 token 数量 embedding 前未知）
    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(...)
else:
    # 纯文本 LLM：pad + slice 一起做
    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(...)

# 注意：label（rolled）无论 VLM 还是 LLM，都在这里完成 slice
input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, ...)
# total_nnz=1000, sp_size=4 → 每卡 250 tokens
```

**Step 3 — 模型前向（all-to-all 在此发生）**

```python
# 输入：[1, total_nnz/sp]，每卡只有 1/sp 的序列片段
output = self.actor_module(input_ids=input_ids_rmpad, position_ids=position_ids_rmpad, ...)
# 输出 logits：[total_nnz/sp, vocab_size]
# attention 层内的 all-to-all 由 monkey patch 透明注入
```

**Step 4 — 在切分的序列上计算 log_probs 和 entropy**

```python
logits_rmpad = output.logits.squeeze(0)  # (total_nnz/sp, vocab_size)
logits_rmpad.div_(temperature)
log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

if calculate_entropy:
    # entropy_checkpointing 控制是否用 gradient checkpoint 节省反向显存
    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
```

**Step 3 和 Step 4 是 SP 显存节省的核心区间**——logits 形状从 `[total_nnz, vocab]` 降至 `[total_nnz/sp, vocab]`，激活值随序列维度线性缩减。

**Step 5 — AllGather + Unpad，还原完整序列**

```python
# log_probs 和 entropy 都需要 AllGather
log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
if calculate_entropy:
    entropy_rmpad = gather_outputs_and_unpad(entropy_rmpad, ...)
# [total_nnz/sp] → AllGather → [total_nnz]，所有 rank 此后持有完整结果

full_log_probs = pad_input(log_probs.unsqueeze(-1), indices, batch_size, seqlen)
log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
```

**Step 5 之后显存节省归零**——每张卡都持有完整的 log_probs 和 entropy。SP 的收益仅体现在 Step 3-4 的计算过程中。

## 4.4 通信原语：两套不同的反向逻辑

整个 SP 适配过程中存在两类通信算子，语义不同。

**all-to-all 路径（SeqAllToAll）**：attention 层前后的 all-to-all 通过自定义 autograd Function 实现，反向时 scatter/gather 维度自动互换，与前向完全对称，训练代码无需额外处理：

```python
class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, local_input, scatter_dim, gather_dim, async_op=False):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group, async_op)

    @staticmethod
    def backward(ctx, *grad_output):
        # scatter/gather 维度互换，与前向对称
        return (None, all_to_all_tensor(grad_output[0], ctx.gather_dim, ctx.scatter_dim, ctx.group, False),
                None, None, None, None)
```

**AllGather 路径（Gather）**：Step 5 的 `gather_outputs_and_unpad` 语义不同，Gather 是跨卡聚合，每张卡都拿到了完整输出并独立计算了梯度，反向时有"多算了一遍"的问题需要补偿。

```python
class Gather(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scaler:
            grad_output = grad_output * ctx.sp_world_size  # 关键：缩放后再 split
        return (None, grad_output.split(ctx.part_size, dim=ctx.gather_dim)[ctx.sp_rank].contiguous(),
                None, None, None, None)
```

`gather_outputs_and_unpad` 的前向把各卡的 `log_probs[total_nnz/sp]` AllGather 成完整的 `log_probs[total_nnz]`，之后每张卡基于完整结果独立计算损失。**AllGather 的输入是各卡的局部片段，属于同一条样本的分布式计算结果，归一化因子应为 dp。**但反向时 FSDP 不感知 SP 的存在，在 AllReduce 时按 world_size = dp × sp 归一化，相当于把 SP 组内的卡也当成了独立样本来平均，多除了 sp 倍。**`Gather.backward` 里乘以 `sp_world_size` 就是撤销这个多余的归一化，再 split 取各卡对应片段，还原正确的局部梯度。**




>  💡 **小结**
>

- `with self.ulysses_sharding_manager` 只切换 SP group；SP 的序列切分全部在 `_forward_micro_batch` 内部完成，`preprocess_data` / `postprocess_data` 在整条调用链中不参与。
- SP 的显存节省集中在 **Step 3-4**（logits 和激活值），Step 5 AllGather 之后每张卡都持有完整结果，节省归零。
- SP 里有两套通信算子：`SeqAllToAll` 反向自动对称；`Gather` 反向需要乘以 `sp_world_size` 补偿梯度量级。


---


# 五、局限性与已知优化点

## 5.1 硬约束

SP 有两条不可绕过的约束：

`num_attention_heads % sp_size == 0` 是硬性限制，all-to-all 要求 head 维度均等切分。GQA 模型的约束被 VERL 部分放宽——允许 `sp_size % num_kv_heads == 0`，此时通过复制 KV head 满足整除要求，代价是 all-to-all 通信量等比增加。

SP 必须和 `use_remove_padding` 一起开启，原因前文已分析，混用配置时不报错但 SP 静默失效。

## 5.2 显存节省的实际范围

| 内容                     | 是否节省   | 原因                                                                   |
| ---------------------- | ------ | -------------------------------------------------------------------- |
| attention 激活值          | ✅ 线性节省 | 每卡只处理 1/sp 的序列片段                                                     |
| FFN 激活值                | ✅ 线性节省 | 序列维度并行                                                               |
| logits（log_prob 阶段）    | ✅ 线性节省 | 形状 [total_nnz/sp, vocab_size]，SP 带来线性节省                              |
| log_probs / entropy 输出 | ❌ 无节省  | AllGather 后每卡持有完整结果，但形状 [total_nnz]，与 logits 相比量级差了 vocab_size 倍，可忽略 |
| 输入数据（input_ids 等）      | ❌ 无节省  | token id 本身显存极低                                                      |
| 模型参数                   | ❌ 无关   | 由 FSDP 控制                                                            |

需要说明的是，"Step 5 AllGather 后显存节省归零"这一结论仅针对 log_probs 和 entropy 输出。logits 不参与 AllGather，SP 对其的显存节省是真实且持久的。

log_prob 阶段的显存大头正是 logits，其形状为 `[total_nnz, vocab_size]`，显存占用与序列长度和词表大小均成正比。以 Qwen3 为例，其词表大小为 151936，一条 8192 token 的序列对应的 logits 在 bf16 下约为 8192 × 151936 × 2B ≈ 2.3 GB。

因此除了 SP 之外，VERL 还提供了两个正交的优化手段，两者均在 `calculate_entropy=True` 的前提下生效：

```python
calculate_entropy=False → 熵不计算，两个优化均不涉及
calculate_entropy=True  → 熵需要计算
    entropy_from_logits_with_chunking=False → 一次性计算全序列熵（默认）
    entropy_from_logits_with_chunking=True  → 分块计算，降低前向峰值显存

    entropy_checkpointing=False → 直接计算（默认）
    entropy_checkpointing=True  → gradient checkpoint，降低反向激活值显存
```

**entropy_from_logits_with_chunking**：在 `calculate_entropy=True` 的前提下生效。按 `chunk_size=2048` 分块计算熵，完整序列一次性计算 softmax 需要保留形状为 `[total_nnz, vocab_size]` 的中间激活，分块后每次只保留 `[chunk_size, vocab_size]`，峰值显存从 O(seq × vocab) 降到 O(chunk × vocab)，适用于前向阶段的峰值显存控制：

```python
def entropy_from_logits_with_chunking(logits, chunk_size=2048):
    for i in range(0, logits.shape[0], chunk_size):
        logits_chunk = logits[i : i + chunk_size].float()
        pd_chunk = torch.softmax(logits_chunk, dim=-1)
        entropy[i : i + chunk_size] = (
            torch.logsumexp(logits_chunk, dim=-1) - torch.sum(pd_chunk * logits_chunk, dim=-1)
        )
```

**entropy_checkpointing**：同样在 `calculate_entropy=True` 的前提下生效。默认 False 时直接计算，设为 True 时改用 gradient checkpoint，以重计算换显存。需要注意的是，`compute_log_prob` 阶段有 `torch.no_grad()`，不存在反向图，`entropy_checkpointing` 在此阶段没有效果；该优化仅在 `update_policy` 的前反向阶段真正生效，用于降低反向传播时保留的激活值显存：

```python
# calculate_entropy=True 时才进入此分支
if not self.config.entropy_checkpointing:
    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
else:
    entropy_rmpad = torch.utils.checkpoint.checkpoint(
        self.compute_entropy_from_logits, logits_rmpad
    )
```

两者侧重不同，可同时开启：chunked entropy 降低前向阶段 softmax 中间激活的峰值显存，entropy_checkpointing 降低 `update_policy` 反向时的激活值显存，两者作用阶段不重叠，组合使用收益叠加。


---


## 5.3 通信开销

每个 Transformer layer 有 **4 次 all-to-all**（q/k/v 各一次 + attn_output 一次）+ **1 次 position_ids AllGather**，对 32 层的 7B 模型来说是每次前向额外 160 次通信。两个已知但尚未落地的优化点：

`async_op` 参数在 `all_to_all_tensor` 中存在但调用时默认 `False`，计算和通信**无法 overlap**。值得一提的是，异步 Ulysses 在 VeOmni 中已经实现，后续有机会走读。

![](https://pic2.zhimg.com/v2-a8f05e51750a63e0617bd336b4119399_r.jpg)

在 FSDP/FSDP2 中集成 Ulysses 时，新增的 all-to-all 和计算保持串行时序，与前反向预取在不同的流，在计算不足以掩盖权重 AllGather 通信时，新增的 all-to-all 其实也会和 AllGather 重叠，有一定隐性收益。当然，需要开启 SP 的场景理论上计算都打满了，大于AllGather的耗时，显式的异步 Ulysses 仍然有额外收益。

`position_ids` AllGather 可以通过直接传入预计算的 `cu_seqlens_q/k` 消除（参见 transformers PR #33932），尚未落地。




## 5.4 VLM 的多路径维护成本

普通 LLM 一行替换，VLM 需要维护三步 patch，每新增一种 VLM 架构（Qwen3-VL、GLM4V、KimiVL 等）都要单独适配。随着支持的模型种类增多，维护成本线性增长。


---


**小结与展望**

VERL 的 Ulysses SP 实现可以用三个关键词概括。

**命名空间替换**：all-to-all 零侵入注入 Transformers attention，对 Transformers 源码完全无修改，新模型适配成本极低——纯文本 LLM 一行替换，VLM 三步 patch。

**dispatch/collect 对称设计**：Controller 侧分发时复制同一 chunk 给 SP 组，收集时只取 sp_rank==0 的结果，Worker 侧无需任何额外通信。FSDPUlyssesShardingManager 的职责因此被精确限定为 SP group 切换——preprocess_data / postprocess_data 是备用接口，主流程不调用。SP 只在注册层添加了少量代码，其余完全复用原有逻辑，整体实现相当优雅。

**梯度补偿**：SP 里存在两套通信算子。SeqAllToAll 的反向自动对称，无需额外处理；Gather 的反向需要乘以 sp_world_size 补偿 FSDP 多除的归一化因子。

总体而言，VERL 为 FSDP 后端较好地集成了 Ulysses 序列并行能力，在长序列场景下能有效缓解训练阶段的显存压力。但 Ulysses 受限于 `num_attention_heads % sp_size == 0` 的硬约束，sp_size 的上限取决于模型的 head 数，在超长序列场景下扩展性有限。Megatron 后端提供的 Context Parallel（Ring Attention）不依赖 head 数，序列维度可以任意切分，在超长序列场景下仍然不可替代，后续有机会会走读这部分实现。

此外，无论是 Context Parallel 还是 Ulysses，解决的都是训练阶段的显存问题。推理阶段的长序列同样带来不小的挑战，当前通常默认开启 chunked prefill 来控制峰值显存，但 vLLM 社区目前也提供了更完整的[解决方案（Context Parallel Deployment）](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)，后续有机会也会一并走读。
