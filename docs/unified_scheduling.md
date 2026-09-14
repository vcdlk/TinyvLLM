# Chunked prefill 与 Unified scheduling

## vLLM 实现参考

本次阅读了 vLLM V1 的 [Scheduler.schedule](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)、[SchedulerOutput](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/output.py) 和官方 [Chunked Prefill 调优说明](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill)（2026-09-08）。参考的是调度机制，没有引入 vLLM 运行时作为本项目的执行后端。

V1 用 `num_computed_tokens` 表示已计算的进度，用已知 token 数与它的差值表示待计算工作；调度结果记录每个请求本轮分配的 token 数。先处理 running 请求，再使用剩余预算接纳 waiting 请求；KV 空间不足时可以抢占并重算。这个抽象把 prefill、分块 prefill 和 decode 放到同一个调度流程中。

本项目实现上述基础机制，不包含 speculative decoding、异步调度、分离式 P/D 或跨请求 prefix caching。

执行侧也参考了 V1 的 [FlashAttention backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py)：它分别传递 query 边界、实际 KV 长度与 block table，并使用分页 K/V 调用变长 attention。本项目保留自写 Triton kernel，在其中加入相同所需的历史 KV 寻址和位置偏移。

## 原项目的限制与改动

| 环节 | 原实现 | 当前实现 |
| --- | --- | --- |
| 调度结果 | `(sequences, is_prefill)`，整批只有一种阶段 | `SchedulerOutput.requests`，每个请求记录 `start` 和 `num_tokens` |
| token 预算 | prompt 必须完整装入一轮 | 按预算分块，可与运行中的请求同批 |
| KV 分配 | prefill 时分配完整 prompt | 只分配到当前 chunk 的结尾 |
| 计算进度 | 没有逐轮提交的计算游标 | forward 成功后提交 `num_computed_tokens` |
| prefill attention | 只读取本轮连续 K/V | 从 block table 读取历史及本轮 KV，应用带前缀偏移的 causal mask |
| RoPE | 每次 prefill 从零开始 | 使用请求的绝对 token 位置 |
| 采样 | 每个被调度序列都生成一个 token | 只有追平已知 token 的请求才采样 |
| 抢占恢复 | TP 序列化可能只保留最后一个 token | 保留完整历史，释放 KV 后从头分块重算 |

## 状态与执行约定

对请求 `seq`，本轮分配量为：

```python
pending = len(seq) - seq.num_computed_tokens
num_tokens = min(pending, remaining_token_budget)
start = seq.num_computed_tokens
end = start + num_tokens
```

输入是 `seq.token_ids[start:end]`，位置是 `range(start, end)`。KV 块覆盖 `[0, end)`；query 长度是 `end - start`，key 长度是 `end`。query 的第 `i` 个 token 只能关注 `key_position <= start + i`，因此不能直接使用从零起算的三角 mask。

forward 成功后，`postprocess` 将游标推进到 `end`。若 `end < len(seq)`，该请求只更新 KV；若相等，采样一个新 token。新 token 此时还没有 KV，因此下一轮自然剩下一个 token 要计算。游标包含 prompt 和重算的 completion 历史，不等同于输出 token 数。

请求按 running 队列的先来先服务顺序获得预算，随后接纳 waiting 队列。运行中的 partial prefill 也保留队列位置；本版没有额外的 decode 抢占优先级；可通过 `long_prefill_token_threshold` 设置每请求每轮的 token 上限，默认为 0（不额外限制）。这个上限同样适用于抢占后的重算。它能给后续请求留出预算，但不是 decode 延迟保证。`max_num_sequences` 限制全部驻留请求数。KV 不足时从 running 尾部抢占尚未加入本批的请求；本批引用的块不会被释放。抢占发生的这一轮不重新接纳 waiting 请求。

例如预算为 4，A 的 prompt 长度是 2，最多生成 3 个 token；A 首次输出后加入长度为 9 的 B：

| 轮次 | A 本轮输入 | B 本轮输入 | 本轮采样 |
| --- | --- | --- | --- |
| 1 | prompt 的 2 个 token | 尚未加入 | A |
| 2 | 1 个 decode token | prompt 的前 3 个 token | A |
| 3 | 1 个 decode token | prompt 的第 4–6 个 token | A，然后 A 完成 |
| 4 | 已完成 | prompt 的第 7–9 个 token | B |
| 5 | 已完成 | 1 个 decode token | B |

## 配置与接口

两个特性默认启用，无需额外开关。引擎仍使用原来的 `LLMEngine(config)` / `generate(prompts, sampling_params)` 接口。例如在现有模型配置中设置：

```python
config.update(
    max_num_batched_tokens=256,  # 每轮总输入 token 数，可小于 prompt 长度
    max_num_sequences=16,        # 全部 running 请求数上限
    long_prefill_token_threshold=128,  # 单请求每轮最多处理 128 tokens；0 表示不限
    max_model_length=4096,       # prompt + completion 的长度上限
)
```

模型的 RoPE 位置容量也必须支持这个长度。输入不能为空，prompt 必须小于模型总长度上限，单请求的历史仍必须能放进整个 KV cache。Chunked prefill 降低每轮计算量，不会消除完整 attention 的历史 KV 存储需求；生成过程如果超出 cache 容量会报错，避免无限抢占重试。

旧配置名 `max_num_batch_tokens` 和 `max_num_seqs` 仅在对应规范配置名缺失时作为兼容别名。warmup、scheduler 和 CUDA graph 统一使用规范配置值，避免预算小于模型长度时 warmup 得到空批次。

内部调用改为：

```python
batch = scheduler.schedule()
token_ids = runner.run(batch)  # 仅包含 should_sample 为 True 的请求，按批内顺序排列
scheduler.postprocess(batch, token_ids.cpu().tolist())
```

`LLMEngine.step()` 保留三元组返回形式；其中 token 数现在是本轮实际计算的输入数，第三项为 True 表示含 prefill 的执行路径（可以是混合批），不再表示整批都是 prompt。纯单 token、已有 KV 的批次仍可使用 decode CUDA graph，混合批次使用 eager 的变长 attention。

## 前缀缓存与当前边界

本版使用请求独占的 KV 块。原来的 hash 路径在 KV 计算之前就公布可复用块，不能安全地用于分块与混合批；它已替换为按需分配。跨请求复用需要额外实现“已计算完整块”的发布/失效、连续前缀命中、最后 token 重算以及部分块的 copy-on-write，再单独验证。

TP 的序列化现在保留完整历史和采样参数；共享内存仍为原有的 1 MiB，过大批次会明确报错。多卡吞吐优化和动态 IPC 扩容不在本次范围内。

## 验证

```bash
python3 -m unittest discover -s tests -v
```

CPU 测试无需第三方依赖，覆盖分块、混合预算、非块对齐的 slot/position、采样时机、停止条件、输入校验、抢占恢复、序列化以及随机压力下的请求守恒与块回收。

`tests/test_chunked_attention.py` 在 CUDA + PyTorch + Triton 环境运行：

- 分页变长 attention 对比 dense causal attention：非连续物理块、GQA、多种块大小/头维度、decode 与 prefill 混合。
- 小型随机初始化 Qwen3 / Llama：完整 prefill 与分块混合执行的 logits 和确定性输出对比，无需下载权重。
- 混合批次切回纯 decode CUDA graph 的回归。

开发机器未安装 PyTorch/CUDA/Triton：CPU 测试已运行；GPU 测试明确跳过。真实模型、多 GPU 推理、GPU 数值一致性和性能收益仍需在 NVIDIA 环境实测，不能把 CPU 通过视为这些验证已完成。

## 在另一台 GPU 机器上验证

在项目根目录、已安装项目依赖的 Python 环境中，先运行数值测试：

```bash
python3 -m unittest discover -s tests -v
```

确认四项 GPU 测试实际执行且通过，而不是 skipped，然后运行调度基准：

```bash
python3 benchmark_scheduling.py --model qwen \
  --budgets 64 256 1024 --chunk-caps 0 128 \
  --prompt-lengths 32 2048 --requests 16 --output-tokens 64 \
  --arrival-ms 10 --repeats 3 --output scheduling-results.json
```

默认加载 Qwen3-0.6B。`--model llama` 改用 Llama-3.2-1B-Instruct；如需本地权重，传 `--model-path /path/to/Qwen3-0.6B`（目录名须与引擎支持的模型名一致）。`--cuda-graphs` 启用纯 decode graph，用同样的命令再比较 eager 与 graph。该脚本测试单 GPU；多卡仍需独立验证。

可先在 CPU 机器检查工作负载参数，无需导入 GPU 依赖：

```bash
python3 benchmark_scheduling.py --dry-run
```

基准使用明确 token 长度的合成文本输入，按 `--prompt-lengths` 循环交替长短请求，按固定时间间隔安排到达；`--arrival-ms 0` 表示同时到达。每个请求忽略 EOS，生成固定数量 token。所有参数组合复用同一个模型与 KV 池，初始化/warmup 使用最大预算，因此 cache 容量不会随组合改变。每个组合先运行一轮不计入结果的 warmup，再执行指定重复次数。JSON 记录实际 GPU、PyTorch/CUDA、dtype、KV 块数和每次测量结果；此脚本沿用模型默认 dtype，并不自动改为 FP16。

统计含义：

- `ttft_ms`：从计划到达时间到首次生成完成，包含等待上一轮完成才能入队的延迟。
- `itl_ms`：同一请求相邻输出 token 的间隔，跨请求合并统计；只生成一个 token 时为 null。
- `output_tokens_per_second`：实际生成的 completion token 数除以总测量时长，包含工作负载内的空闲时间。
- `computed_input_tokens`：实际执行的 prompt/decode/recompute token 数，避免把它误算成输出吞吐。
- `steps`：变长输入、纯 decode、混合执行和仅抢占轮次计数。
- `requests`：逐请求延迟和 token 间隔，便于分别分析长、短输入。

这些是引擎内部测量，不包含模型加载、分词或 HTTP 服务开销。GPU 输出复制到 CPU 后再记完成时间；不同参数可能有不同随机生成内容，不用这个性能基准判断数值正确性。大预算结果也是统一调度器，不是旧版整批 prefill 调度器的性能基线。
