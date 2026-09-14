<p align="center">
  <img src="./assets/minivllm.png" alt="图片描述" width="50%" height="50%">
</p>

<p align="center">
| <a href="./README.md"><b>English</b></a> 
| <a href="./README_zh.md"><b>简体中文</b></a> |
</p>

# miniVLLM
自定义实现的vLLM推理引擎，基于Nano-vLLM。添加了注意力机制的基准测试，以及Pageattention、FlashAttention的代码实现。

提供了预填充阶段的FlashAttention以及解码阶段的Pageattention的基准测试。


**第一次接触vLLM?** 阅读 [HowToApproachvLLM_zh.md](HowToApproachvLLM_zh.md) 从零开始实现vLLM！学习vLLM中layers、models、Pageattention、FlashAttention、CUDA graphs以及调度实现。

## Chunked prefill 与 Unified scheduling

默认启用按 token 预算分块的 prefill，可与 decode 同批执行。KV 按 chunk 增长，中间 chunk 不采样；Qwen3 与 Llama 使用连续位置编号及历史分页 KV。调度器先处理 running 请求，再用剩余预算接纳 waiting 请求。

可用 `long_prefill_token_threshold` 限制单请求每轮处理量（默认 0，不额外限制）。新增 `benchmark_scheduling.py` 可测量首 token 延迟、token 间延迟和输出吞吐；`--dry-run` 可在 CPU 上预览测试配置。

设置 `max_num_batched_tokens` 控制每轮输入预算，它可以小于 prompt 长度。纯 decode 保留 CUDA graph 路径。当前使用请求独占 KV 块，不支持跨请求 prefix caching。

实现思路、vLLM 源码参考、配置和验证边界见 [改造说明](docs/unified_scheduling.md)。运行测试：`python3 -m unittest discover -s tests -v`（无 CUDA 环境会跳过 GPU 数值测试）。

## 快速开始

```bash
# 安装 uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖
uv sync

# 运行推理引擎
uv run python main.py

# prefilling 基准测试
uv run python benchmark_prefilling.py

# decoding 基准测试
uv run python benchmark_decoding.py
```

## 每个脚本的作用

```bash
uv run python main.py
```
主推理引擎演示入口

演示了使用自定义引擎实现的完整 LLM 推理流程：
- 基于 Qwen3-0.6B，采用随机初始化
- 创建60个聊天 prompt（2个基础 prompt 各重复30次）
- 通过自定义 LLM 引擎使用批处理处理 prompt
- 使用 Pageattention 和 KV cache 管理来提高推理效率
- 每个 prompt 生成最多256个 tokens，采用温度采样

展示了自定义vLLM实现如何处理带有内存高效注意力的批量文本生成。


```bash
uv run python benchmark_prefilling.py
```

预填充阶段对比

比较了在**预填充阶段**（处理输入提示）期间的三种注意力实现：

1. **PyTorch Standard（O(N²) memory）**：传统的注意力机制，会生成完整的注意力矩阵
2. **Naive Triton（O(N²) memory）**：使用 GPU 内核的注意力机制，也使用 O(N²) 内存，受共享内存限制（≤128 tokens）
3. **FlashAttention（O(N) memory）**：内存高效的在线 softmax 算法，通过块处理注意力


```bash
uv run python benchmark_decoding.py
```

解码阶段对比

比较了在**解码阶段**（一次生成一个输出token）期间的三种实现：

1. **Naive PyTorch**：基于循环的实现，使用分页KV缓存
2. **Optimized PyTorch**：向量化实现，支持批量 gathering 和 mask
3. **Triton Kernel**：自定义GPU内核，用于优化 Pageattention 解码


## 项目结构

```
myvllm/
├── src/
│   └── myvllm/           # 核心vllm实现
│       ├── models/       # 模型实现
│       ├── engine/       # LLM引擎逻辑，包括输入提示的序列定义，KV Cache的块管理，基于迭代的序列调度器，预填充和解码器，以及用于生成API接口的引擎
│       ├── layers/       # 模型组件
│       ├── utils/        # 全局变量
│       └── sampling_parameters.py 
├── main.py              # 推理演示
├── benchmark_prefilling.py   # 预填充对比
└── benchmark_decoding.py     # 解码对比
```

## 运行环境

- Python ≥3.11, < 3.12
- CUDA-capable GPU
- 依赖: `transformers`, `torch`, `xxhash` (使用uv进行管理)


## Star History

[![Star History Chart](https://star-history.dera.page/svg?repos=Wenyueh/MinivLLM&type=date&legend=top-left)](https://star-history.dera.page/?utm_source=chatgpt.com#Wenyueh/MinivLLM&type=date&legend=top-left)