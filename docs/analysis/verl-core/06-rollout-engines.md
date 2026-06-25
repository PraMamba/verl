# 06 - Rollout Engines 子模块架构文档

> verl/workers/rollout/ -- 推理引擎封装与三级服务架构
> 25 个 Python 文件, 8,333 行 (wc -l 验证)

---

## 1. 模块定位与边界

Rollout Engines 子模块封装了 verl 的推理侧能力。它将 vLLM、SGLang、TensorRT-LLM 三大推理引擎统一为 HTTP 服务器形态, 通过三级架构 (Replica -> ServerManager -> AgentLoopManager) 管理生成请求的生命周期。

**上游依赖**: `verl/workers/engine_workers.py` 的 `ActorRolloutRefWorker` 在 `init_model()` 中创建 rollout 实例; `verl/trainer/ppo/ray_trainer.py` 的 `RayPPOTrainer` 通过 `LLMServerManager` 管理推理服务器集群。

**下游消费者**: `verl/trainer/ppo/agent_loop_manager.py` 的 `AgentLoopManager` 通过 `LLMServerClient` 发送生成请求。

**不包含**: 训练引擎 (属 04-engine-backends); 权重同步机制的训练侧 (属 05-engine-workers 的 `update_weights()`)。

---

## 2. 架构总览

```
                    LLMServerManager (llm_server.py:223)
                         |
                    管理 N 个 RolloutReplica
                         |
          +--------------+--------------+
          |              |              |
     vLLMReplica    SGLangReplica   TRTLLMReplica
     (vllm_async_    (async_sglang_  (trtllm_async_
      server.py:958)  server.py:721)  server.py:485)
          |              |              |
     每个 Replica 管理 1..N 个 HttpServer 进程 (每节点一个)
          |              |              |
     vLLMHttpServer SGLangHttpServer TRTLLMHttpServer
     (:85)          (:111)           (:80)
          |
     通过 FastAPI/uvicorn 暴露 HTTP 接口
          |
     底层引擎: AsyncLLM / SGLang Engine / TRT-LLM ExecutorBindingsWorker
```

请求路由层:
```
AgentLoopManager
      |
      | generate(request_id, prompt_ids, sampling_params)
      |
LLMServerClient (llm_server.py:146)
      |
      | acquire_server(request_id)  -- sticky session + least-loaded
      |
GlobalRequestLoadBalancer (llm_server.py:44, Ray Actor)
      |
      | -> (server_id, server_handle)
      |
HttpServer.generate.remote()  -- Ray 远程调用
```

---

## 3. 核心数据结构

### 3.1 BaseRollout (base.py:29-80)

推理引擎的抽象基类, 定义了 `ServerAdapter` 的接口:

| 方法 | 行号 | 职责 |
|------|------|------|
| `resume(tags)` | 45 | 恢复 GPU 权重/kv_cache |
| `update_weights(weights)` | 54 | 更新模型权重 |
| `release()` | 67 | 释放 GPU 内存 |
| `generate_sequences(prompts)` | 71 | 同步批量生成 (可选) |

**注册表** `_ROLLOUT_REGISTRY` (base.py:83):
```python
_ROLLOUT_REGISTRY = {
    ("vllm", "async"): "verl.workers.rollout.vllm_rollout.ServerAdapter",
    ("sglang", "async"): "verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter",
    ("trtllm", "async"): "verl.workers.rollout.trtllm_rollout.trtllm_rollout.ServerAdapter",
}
```

`get_rollout_class()` (base.py:90) 按 `(rollout_name, mode)` 查找并动态导入。

### 3.2 RolloutReplica (replica.py:70-300)

推理服务器副本的抽象基类, 管理单个推理服务器实例的生命周期:

**三种部署模式** (replica.py:54-67 RolloutMode):
- `HYBRID`: 推理引擎与训练引擎融合在同一 Ray Worker 进程, 分时复用 GPU
- `COLOCATED`: 推理引擎与训练引擎在同一 Ray Placement Group 但不同进程, 共享 GPU
- `STANDALONE`: 推理引擎独立部署, 拥有专属 GPU 资源

**初始化方法**:
- `init_hybrid(worker_group)` (第131行): 从 worker_group 切片出本 replica 的 workers, 调用 `launch_servers()`
- `init_colocated(resource_pool)` (第160行): 在 resource_pool 中创建新 worker, 调用 `launch_servers()`
- `init_standalone()` (第189行): 创建新 resource_pool + worker_group, 调用 `launch_servers()`

**生命周期控制**:
- `wake_up()` / `sleep()` (第265/269行): 唤醒/休眠服务器 (释放/恢复 kv_cache 和权重)
- `abort_all_requests()` / `resume_generation()` (第273/278行): 部分生成中断与恢复
- `clear_kv_cache()` / `release_kv_cache()` / `resume_kv_cache()` (第281-291行): KV cache 管理

### 3.3 RolloutReplicaRegistry (replica.py:302-380)

Replica 类的工厂注册表, 内置三种加载器:
- `_load_vllm()` (第321行): 导入 `vLLMReplica`
- `_load_sglang()` (第327行): 先 mock vllm 依赖 (SGLang 编译时需要), 再导入 `SGLangReplica`
- `_load_trtllm()` (第371行): 导入 `TRTLLMReplica`

`get_rollout_replica_class()` (第383行) 还处理 PD 分离模式: `disaggregation_enabled=True` 时加载 `SGLangPDReplica`。

### 3.4 TokenOutput (replica.py:39-52)

生成输出的 Pydantic 数据模型:
- `token_ids: list[int]` -- response token ID 序列
- `log_probs: Optional[list[float]]` -- 每个 token 的 log 概率
- `routed_experts: Optional[Any]` -- MoE 路由信息 (Router Replay 用)
- `stop_reason: Optional[str]` -- 停止原因: "completed" / "aborted"
- `num_preempted: Optional[int]` -- 被抢占次数 (性能指标)
- `extra_fields: dict[str, Any]` -- 动态扩展字段

### 3.5 AsyncRolloutRequest (schemas.py:81-713)

异步 rollout 请求的完整数据模型, 管理多轮对话的状态:

**状态机** (schemas.py:63-69 AsyncRolloutRequestStateEnum):
```
PENDING -> RUNNING -> COMPLETED
                  -> FAILED
                  -> TOOL_CALLING -> RUNNING (工具调用后继续生成)
```

**核心字段**:
- 消息历史: `messages: list[Message]`, 多模态: `multi_modal_data`, `multi_modal_inputs`
- Token 序列: `input_ids`, `prompt_ids`, `response_ids` (均为 Tensor)
- Mask 系列: `attention_mask`, `loss_mask`, `position_ids` (及各自的 prompt_/response_ 前缀变体)
- 生成控制: `max_prompt_len`, `max_response_len`, `max_model_len`
- 工具调用: `tool_schemas`, `tools_kwargs`

**关键方法**:
- `initialize_request()` (第123行, model_validator): 构造时自动 tokenize 消息, 计算 position_ids, 处理多模态输入
- `add_user_message()` (第406行): 追加用户消息, 增量更新 input_ids/attention_mask/position_ids
- `add_assistant_message()` (第428行): 追加助手消息, loss_mask 标记为 True (需训练)
- `add_tool_response_messages()` (第453行): 追加工具响应, 处理多模态工具输出 (图像/视频)
- `finalize()` (第591行): 完成请求, 执行 tokenization sanity check, 截断到 max_model_len
- `get_generation_prompt_ids()` (第374行): 获取带 generation prompt 的 token 序列, 供推理引擎使用

**Tokenization Sanity Check** (schemas.py:73-78):
- `DISABLE`: 关闭检查
- `STRICT`: 严格模式, 任何差异都警告
- `IGNORE_STRIPPABLE`: 忽略可 strip 的空白差异

检查逻辑 (finalize() 中第606行起): 将增量构建的 prompt 与一次性 apply_chat_template 的结果对比, 使用 `difflib.SequenceMatcher` 找出差异, 辅助调试 chat template 不一致问题。

---

## 4. 关键流程

### 4.1 LLMServerManager 初始化流程 (llm_server.py:223-373)

```
LLMServerManager.create(config, worker_group)
  1. _initialize_llm_servers():
     a. 计算 num_replicas = world_size / rollout_world_size
     b. 为每个 replica 创建 RolloutReplica 实例
     c. 根据部署模式调用:
        - init_hybrid(worker_group)  -- 协同部署
        - init_standalone()          -- 独立部署
     d. 收集 server_handles 和 server_addresses

  2. _init_global_load_balancer():
     -> 创建 GlobalRequestLoadBalancer Ray Actor
     -> 传入 {address: handle} 映射
```

### 4.2 GlobalRequestLoadBalancer 请求路由 (llm_server.py:44-143)

LRU Cache + 最小在途请求数负载均衡:

```
acquire_server(request_id):
  1. 查找 sticky session (LRUCache, 默认 10000 条)
     -> 命中且服务器仍在线: 直接返回, inflight++
     -> 命中但服务器已移除: 清除缓存, 重新选择
  2. 未命中: 选择 inflight 最少的服务器
     -> 缓存 request_id -> server_id
     -> inflight++
     -> 返回 (server_id, handle)

release_server(server_id):
  -> inflight-- (fire-and-forget)
```

动态扩缩容: `add_servers()` / `remove_servers()` 支持运行时增减服务器。

### 4.3 vLLM 权重同步流程

vLLM Hybrid 模式下, 权重从训练引擎同步到推理引擎的路径:

```
ActorRolloutRefWorker.update_weights()
  -> engine.get_per_tensor_param()        -- 导出 HF 格式参数
  -> rollout.update_weights(params)       -- ServerAdapter 方法
     -> vLLMHttpServer.update_weights()   -- 通过 Ray remote 调用
        -> BucketedWeightSender.send()    -- 通过 ZMQ IPC 传输
           -> vLLM worker 进程
              -> BucketedWeightReceiver.receive_weights()
                 -> 直接写入模型参数 (CUDA IPC 或 shared memory)
```

### 4.4 vLLM Sleep/Wake 流程

```
sleep():
  -> release kv_cache (cudaFree)
  -> release model weights (cudaFree)
  -> GPU 显存归还给训练引擎

wake_up(tags=["weights"]):
  -> resume model weights (cudaMalloc + weight sync)
  
wake_up(tags=["kv_cache"]):
  -> resume kv_cache (cudaMalloc + 重建 block table)
```

---

## 5. 三大推理引擎实现

### 5.1 vLLM 异步服务器 (vllm_rollout/, 约 2,700 行)

**vLLMHttpServer** (vllm_async_server.py:85): 单节点 vLLM HTTP 服务器
- 基于 vLLM 的 `AsyncLLM` + `build_app()` + uvicorn 构建
- `generate()` (第457行): 创建 `TokensPrompt` + `SamplingParams`, 调用 `engine.generate()`, 收集 `RequestOutput`
- `wake_up()` / `sleep()` (第605/626行): 通过 vLLM 的内存管理 API 释放/恢复 GPU 资源
- 支持 LoRA: 通过 `LoRARequest` 传入 adapter 配置

**vLLMReplica** (vllm_async_server.py:958): vLLM 副本管理器
- `launch_servers()` (第974行): 为每个节点创建一个 `vLLMHttpServer` Ray Actor
- Hybrid 模式: 在已有的 worker 进程中启动 HTTP 服务器 (通过 Ray Actor scheduling)
- Standalone 模式: 创建独立 Ray Actor

**bucketed_weight_transfer.py** (333行): 高性能权重传输
- `BucketedWeightSender` (第74行): 将模型参数打包为固定大小的 bucket, 通过 ZMQ IPC socket 发送
- `BucketedWeightReceiver` (第233行): 接收 bucket, 解包参数写入模型
- 传输介质: 优先使用 CUDA IPC (GPU 直连, 零拷贝), 回退到 POSIX shared memory
- `rebuild_ipc()` (第45行): 从 IPC handle 重建 CUDA tensor, 关键在于替换 `device_id` 以适应不同进程的 `CUDA_VISIBLE_DEVICES` 映射

### 5.2 SGLang 异步服务器 (sglang_rollout/, 约 2,200 行)

**SGLangHttpServer** (async_sglang_server.py:111): 单节点 SGLang HTTP 服务器
- 基于 SGLang 的 `ServerArgs` + `_GlobalState` + `app` (FastAPI) 构建
- `generate()` (第508行): 构造 `GenerateReqInput`, 通过 SGLang tokenizer_manager 提交请求
- `wake_up()` / `sleep()` (第442/466行): 使用 `ResumeMemoryOccupationReqInput` / `ReleaseMemoryOccupationReqInput`
- 支持 prompt logprobs: `_extract_prompt_logprobs_sglang()` (第61行) 对齐 SGLang 和 vLLM 的输出格式

**SGLangReplica** (async_sglang_server.py:721): SGLang 副本管理器
- `launch_servers()` (第737行): 与 vLLM 类似, 每节点一个 HttpServer

**PD 分离模式**: `sglang_pd_replica.py` 提供 `SGLangPDReplica`, 支持 Prefill-Decode 分离部署。

### 5.3 TensorRT-LLM 异步服务器 (trtllm_rollout/, 约 1,300 行)

**TRTLLMHttpServer** (trtllm_async_server.py:80): 单节点 TRT-LLM HTTP 服务器
- 基于 TRT-LLM 的 `ExecutorBindingsWorker` 构建
- `generate()` (第321行): 构造 TRT-LLM 请求, 提交到 executor

**TRTLLMReplica** (trtllm_async_server.py:485): TRT-LLM 副本管理器
- `launch_servers()` (第570行): 需要特殊处理 -- 通过 `init_hybrid_colocated()` 使用 resource_pool 而非直接复用 worker

---

## 6. 三级架构详解

```
第一级: RolloutReplica (管理一个推理服务器实例)
  - 职责: 进程管理、部署模式选择、生命周期控制
  - 粒度: 一个 replica 对应一个 TP 组 (可跨节点)
  - 实例数: world_size / (tp_size * dp_size * pp_size)

第二级: LLMServerManager (管理所有 replica)
  - 职责: replica 编排、负载均衡器初始化、弹性扩缩容
  - 实例数: 1 (全局单例)
  - 暴露接口: get_client(), get_addresses(), get_replicas()

第三级: AgentLoopManager (业务逻辑层, 不在本模块)
  - 职责: 多轮对话管理、工具调用编排、奖励计算
  - 通过 LLMServerClient 与推理服务器通信
```

每一级的关注点完全分离:
- Replica 不知道有多少个 Replica 存在
- ServerManager 不知道请求的具体内容
- AgentLoop 不知道底层用的是 vLLM 还是 SGLang

---

## 7. 设计决策

### 7.1 为何使用 HTTP 服务器而非直接 Ray 调用?

HTTP 服务器架构有三个优势:
1. **与推理引擎原生集成**: vLLM/SGLang 本身就是 HTTP 服务器, 复用其内置的请求调度、KV cache 管理、continuous batching
2. **OpenAI 兼容 API**: 支持外部工具 (如 judge 模型) 直接通过 HTTP 调用
3. **进程隔离**: 推理引擎运行在独立进程, GPU 内存管理不受训练进程影响

### 7.2 为何需要 GlobalRequestLoadBalancer?

多 AgentLoopWorker 并行发送请求时, 需要全局协调:
- **Sticky Session**: 多轮对话路由到同一服务器, 利用 prefix caching (KV cache 复用)
- **最小在途**: 新会话选择负载最低的服务器, 避免热点
- **原子操作**: `acquire_server()` 返回 `(server_id, handle)`, 一次 Ray RPC 完成查找 + 路由, 避免竞态

### 7.3 为何 BucketedWeightTransfer 使用 ZMQ 而非 Ray ObjectStore?

权重同步的数据量通常在 GB 级别, Ray ObjectStore 的序列化/反序列化开销不可接受。BucketedWeightTransfer 使用:
- **ZMQ IPC socket**: 进程间零拷贝通信
- **CUDA IPC**: 训练进程直接将 GPU tensor handle 传给推理进程, 推理进程通过 handle 重建 tensor, 避免 GPU->CPU->GPU 往返
- **Bucketed 传输**: 将参数按固定大小 bucket 打包, 减少通信次数, 同时控制峰值内存

### 7.4 为何 AsyncRolloutRequest 自带 Tokenization Sanity Check?

多轮对话中, 每轮通过增量 `_update_input_ids()` 追加 token, 而 finalize 时用一次性 `apply_chat_template()` 验证。两种方式可能因 chat template 的特殊 token 处理不同而产生差异。Sanity check 在训练前捕获这种不一致, 避免训练数据质量问题。

---

## 8. 关键接口与扩展点

### 8.1 新增推理引擎

需要实现三个层级:

```python
# 1. HttpServer: 封装推理引擎的 HTTP 服务器
class MyHttpServer:
    async def generate(self, request_id, prompt_ids, sampling_params, ...) -> TokenOutput: ...
    async def wake_up(self, tags=None): ...
    async def sleep(self): ...

# 2. Replica: 管理 HttpServer 的部署和生命周期
class MyReplica(RolloutReplica):
    async def launch_servers(self): ...

# 3. ServerAdapter: 实现 BaseRollout 接口
class ServerAdapter(BaseRollout):
    async def resume(self, tags): ...
    async def update_weights(self, weights, **kwargs): ...
    async def release(self): ...

# 注册
_ROLLOUT_REGISTRY[("my_engine", "async")] = "my_module.ServerAdapter"
RolloutReplicaRegistry.register("my_engine", lambda: MyReplica)
```

### 8.2 LLMServerClient 自定义

```python
class MyClient(LLMServerClient):
    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        # 自定义请求前/后处理
        server_id, server = await self._acquire_server(request_id)
        try:
            output = await server.generate.remote(...)
            # 自定义后处理
            return output
        finally:
            self._release_server(server_id)

client = server_manager.get_client(client_cls=MyClient)
```

### 8.3 Replica 部署模式选择

| 场景 | 推荐模式 | 原因 |
|------|---------|------|
| On-policy 训练, GPU 有限 | HYBRID | 训练和推理分时复用 GPU |
| On-policy 训练, GPU 充裕 | STANDALONE | 训练和推理并行执行 |
| Off-policy 训练 | STANDALONE | 推理服务器独立运行 |
| LLM as Judge | COLOCATED | 共享 GPU 但独立进程 |
