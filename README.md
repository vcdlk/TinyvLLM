# TinyvLLM

![TinyvLLM](assets/tinyvllm.png)

[English](README.md) | [简体中文](README_zh.md)

A lightweight LLM inference engine based on Nano-vLLM, built for learning inference workflows and performance optimization.

## Features

- Supports Qwen3-0.6B, Llama-3.2-1B-Instruct, and MiniMind-3-MoE with pretrained weights for text generation.
- **Unified Scheduling**: Prefill and Decode share a token budget and can run in the same batch.
- **Chunked Prefill**: Long inputs are processed in chunks within the token budget, with KV cache allocated on demand.
- Implements paged KV cache, Triton Flash Attention, and Paged Attention; MiniMind uses PyTorch SDPA for Prefill.
- Supports multi-GPU tensor parallelism and CUDA Graphs, with attention and inference throughput benchmarks.

Unified Scheduling and Chunked Prefill are available on the `unified_scheduler` branch and have not yet been merged into the current branch.

## Getting Started

Requires Linux, an NVIDIA GPU with CUDA, Python 3.11, and uv. Models are downloaded on the first run;

```bash
uv sync

# Qwen3 inference
uv run python main.py

# Llama 3.2 inference
uv run python main_llama32.py

# MiniMind-3-MoE inference
uv run python main_minimind.py
```

Set the execution mode in the corresponding script's `config`, then run the same commands above:

| Configuration           | Execution mode                                                                     |
| ----------------------- | ---------------------------------------------------------------------------------- |
| `world_size = 1`        | Single GPU (default)                                                               |
| `world_size = N`        | Tensor parallelism across N GPUs; the engine starts worker processes automatically |
| `enforce_eager = True`  | Eager execution (default)                                                          |
| `enforce_eager = False` | CUDA Graphs during Decode                                                          |

The `unified_scheduler` branch enables Unified Scheduling and Chunked Prefill by default. `max_num_batched_tokens` sets the total token budget per iteration; `long_prefill_token_threshold` caps tokens per request per iteration (0 means no additional limit).

## Benchmarks

```bash
# Prefill attention comparison
uv run python benchmark_prefilling.py

# Decode attention comparison
uv run python benchmark_decoding.py

# TinyvLLM, vLLM, and Transformers throughput comparison
uv run python benchmark_tps.py
```
