# TinyvLLM

基于 Nano-vLLM 的轻量级大模型推理引擎，用于学习推理流程和性能优化。

## 做了什么

- 支持 Qwen3-0.6B、Llama-3.2-1B-Instruct，加载预训练权重进行文本生成。
- 实现批量调度、分页 KV Cache，以及 Triton Flash Attention（Prefill）和 Paged Attention（Decode）。
- 支持多 GPU 张量并行和 CUDA Graph，提供注意力及推理吞吐量基准测试。

## 运行方式

需要 Linux、NVIDIA GPU（CUDA）、Python 3.11 和 uv。首次运行会下载模型，Llama 模型需具备 Hugging Face 访问权限。

```bash
uv sync

# Qwen3 推理
uv run python main.py

# Llama 3.2 推理
uv run python main_llama32.py
```

在对应脚本的 `config` 中设置运行模式，修改后仍使用上面的命令启动：

| 配置                    | 运行模式                                |
| ----------------------- | --------------------------------------- |
| `world_size = 1`        | 单 GPU（默认）                          |
| `world_size = N`        | N 张 GPU 张量并行，由引擎自动启动子进程 |
| `enforce_eager = True`  | Eager 执行（默认）                      |
| `enforce_eager = False` | Decode 阶段使用 CUDA Graph              |

## 性能测试

```bash
# Prefill 注意力性能对比
uv run python benchmark_prefilling.py

# Decode 注意力性能对比
uv run python benchmark_decoding.py

# TinyvLLM、vLLM、Transformers 吞吐量对比
uv run python benchmark_tps.py
```
