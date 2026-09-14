# TinyvLLM 对 MiniMind-3-MoE 的支持

首版补丁已实现模型、严格权重加载、CPU 数值验证和 Linux CUDA engine 接入。当前支持边界是 **单 GPU、eager、未量化、每层均为 MoE、无 shared expert、标准 RoPE**。Mac 可运行 CPU 对照测试；不能据此宣称 CUDA engine 已验证或已有吞吐提升。

## 检查依据

检查日期：2026-09-14。TinyvLLM 基线提交：`591952dd0153b3d32162d1d8fa2bb74adc82b193`。

- [MiniMind 模型源码](https://github.com/jingyaogong/minimind/blob/e84156818aec24ebf0628010d89c9f5d79ea2af5/model/model_minimind.py)：现场下载的 master 文件与本地上述提交的文件 SHA256 一致，`47f124f5f386a8cd76b488b36ecc3e00fc43e3894cdb3ba602420188d9ba8860`。该提交用来固定已核对的文件内容，不表示已经取得远端 master 最新提交号。
- [官方转换脚本](https://github.com/jingyaogong/minimind/blob/e84156818aec24ebf0628010d89c9f5d79ea2af5/scripts/convert_model.py)：现场下载内容与上述本地提交一致，SHA256 `118df87620719bee7b22b1500017f668737a3037b0caceeb886575fee25fbc51`。
- [官方发布配置](https://huggingface.co/jingyaogong/minimind-3-moe/blob/1dc1e5702b361d0fc024060d03efde6f9dfceaa7/config.json)及同版本 `model.safetensors`，版本 `1dc1e5702b361d0fc024060d03efde6f9dfceaa7`。先读取文件头核对形状，随后下载完整权重执行 CPU 对照。
- 数值参考为 `transformers==4.57.6` 的 `Qwen3MoeForCausalLM`，与公开 config 标记的版本一致。新版 packed 权重按官方转换脚本处理，不代表测试了 Transformers 5.x 全部格式。

## 模型事实与兼容性

| 项目 | 官方发布模型 | TinyvLLM 基线 / 补丁 |
|---|---|---|
| 注册 | `model_type=qwen3_moe`，`architectures=[Qwen3MoeForCausalLM]` | 基线按目录尾名匹配两个模型；新 MoE 路径读取 config，不依赖目录名 |
| 尺寸 | H=768，8 层，vocab=6400 | 从 config 填入 runner 和 KV 分配参数 |
| Attention | 8 Q heads、4 KV heads、head_dim=96，GQA | 不能套用默认 64/128，也不能对缓存复制成 8 个 KV heads |
| Norm | Pre-RMSNorm，Q/K head RMSNorm，eps=1e-6 | 基线 Qwen3 的 eps 未传入各 Norm，计算也未升 FP32；新路径明确使用配置与 FP32 归约 |
| RoPE | theta=1e6，32768，split-half，发布配置 scaling=null | 新路径 FP32 计算角度，旋转后回到输入 dtype；缓存前缀必须偏移 position |
| FFN | SwiGLU，4 experts，I=2432，top-1，无 shared expert | 新增 SparseMoE；不将它当成一个 dense FFN |
| Router | `[4,768]` 无 bias gate，softmax → top-k → 可选归一化 | Qwen 导出用 FP32 softmax；原生 MiniMind 用 logits dtype，与源码区分 |
| 权重绑定 | `tie_word_embeddings=true` | 实际 safetensors 只有 embedding，没有 lm_head；加载器补齐别名并绑定同一 Parameter |

路由计算为 `p=softmax(W_router x)`，取 top-k 后若 `norm_topk_prob=true`，令 `w=p_selected/sum(p_selected)`，输出 `sum(w_e * down_e(silu(gate_e(x))*up_e(x)))`。归一化的 top-1 权重恒为 1；不能再乘一次原始 softmax 概率。top-2 需要累加多个 expert 输出，不能覆盖写回。

补丁将 token 按 expert 聚合，用 `index_add_` 散回原 token 次序；只运行收到 token 的 expert。空 token 输入、无 token 的 expert、2D flatten / 3D 输入均有覆盖。训练用辅助 loss、dropout、router logits 输出不属于本次推理接口。

## 文件级修改与权重映射

| 文件 | 修改内容 |
|---|---|
| `src/myvllm/models/registry.py` | 读取本地/HF JSON；识别 qwen3_moe 或原生 minimind+use_moe；规范尺寸、dtype、缓存参数；在 worker 启动前拒绝 TP>1、YaRN、量化、shared expert、混合 dense 层等未实现配置 |
| `src/myvllm/models/minimind_moe.py` | 独立 eager 模型，RMSNorm、SwiGLU expert、router/top-k、Q/K norm、RoPE、GQA、paged KV 接口、缓存 suffix prefill、LM head |
| `src/myvllm/utils/minimind_loader.py` | split/packed expert 格式、单文件/分片索引、绑定权重；检查全部名称和 shape 后才写参数，避免加载失败后带随机参数推理 |
| `src/myvllm/utils/loader.py` | 将新模型委托给专用严格加载器；已有 dense 加载逻辑保留 |
| `src/myvllm/engine/llm_engine.py` | 创建 worker 前解析、校验模型 config |
| `src/myvllm/engine/model_runner.py` | 构建新模型，设置 FP16/BF16，eval，强制 eager；全 prefix 命中时重算末块以产生末 token logits |
| `src/myvllm/layers/attention.py` | KV store 和 paged decode 增加 `BLOCK_D=next_power_of_2(head_dim)`；对超出真实 head_dim 的 lane 做 load/store mask，支持 96 |
| `main_minimind.py` | 使用真实 tokenizer/chat template 的 Linux GPU engine 示例；Mac 上提前提示 CPU 验证入口 |
| `scripts/check_minimind.py` | 本地真实权重与 HF 模型的 prefill + 逐步 decode 数值对照，CPU/CUDA 可选 |
| `tests/test_minimind_moe.py` | router、top-k、split/packed 权重、分片、绑定、严格失败、HF logits、变长批次、跨页 decode、prefix、低精度、GPU kernel/model 测试 |

本版刻意保留独立 Linear，与 HF 4.x/原生源码同名，**不复用现有 merged dense loader**。基线 loader 只识别 `.mlp.gate_proj`，会跳过 expert 的 up_proj，还把一些 shape 错误降级为部分复制/警告；直接接入 MoE 会留下随机参数。它也没有正确调用 TP layer 的 weight_loader，因此不能据此声称已有多卡支持。

| 源权重名/形状（层前缀省略） | 补丁目标 |
|---|---|
| `self_attn.q_proj.weight [768,768]` | 同名原样；首版不合并 QKV |
| `self_attn.k_proj.weight`、`v_proj.weight [384,768]` | 同名原样 |
| `self_attn.o_proj.weight [768,768]` | 同名原样 |
| `self_attn.q_norm.weight`、`k_norm.weight [96]` | 同名，按每 head 最后一维归一化 |
| `mlp.gate.weight [4,768]` | router；不可误当 SwiGLU gate |
| `mlp.experts.e.gate_proj.weight`、`up_proj.weight [2432,768]` | 第 e 个 expert 同名参数 |
| `mlp.experts.e.down_proj.weight [768,2432]` | 第 e 个 expert 同名参数 |
| `mlp.experts.gate_up_proj [4,4864,768]`（HF 5.x） | 沿 expert 维拆分；中间维前 2432 为 gate，后 2432 为 up |
| `mlp.experts.down_proj [4,768,2432]`（HF 5.x） | 沿 expert 维拆分成独立 down_proj |
| `model.embed_tokens.weight [6400,768]` | embedding 及缺省的共享 lm_head |

没有额外 transpose。packed 名称也接受末尾 `.weight`，但严格要求上述轴顺序，不能靠 shape 猜测转置。混合重复 split/packed tensor、分片缺失/索引错误、绑定权重冲突、额外参数、shape mismatch 均报错。首版仍将 checkpoint 放入 CPU 内存再加载；不是流式大模型加载器。raw `.pth/.bin` 需要先用官方转换工具导出 safetensors；未直接支持训练恢复包、量化权重和任意旧版 MiniMind。

## KV cache、prefill 与执行方式

KV pool 保持 `[2, layers, blocks, block_size, kv_heads, head_dim]`。本模型每层只有一个带 `k_cache/v_cache` 的 attention 模块，runner 的缓存发现/挂接逻辑可以复用。cache 保存旋转后的 K 和未旋转 V；GQA 仅在 attention 计算时扩展 KV。FP16/BF16 每 token KV 占 `2×8×4×96×2=12288` 字节，block_size=256 时每个跨层 cache block 为 3 MiB。

96 不是 2 的幂，原 Triton `tl.arange(0, head_dim)` 不能直接编译。补丁仅给 **store/decode** 增加补齐维与 mask，实际 cache stride 仍为 96，不能将 stride 改成 128。GPU 测试覆盖 64、96、128 和不连续物理页；这些测试在本机跳过，待 GPU 实测。

新模型 prefill 使用 PyTorch SDPA 按变长序列执行，不调用旧 Triton prefill kernel。每个序列的 position 从 `K_length - Q_length` 开始；存在缓存 prefix 时按 block table gather 完整 KV。显式 suffix causal mask 允许 query 看 prefix 及当前 query 之前的位置。直接使用 SDPA 的 `is_causal=True` 处理 Q/K 不等长，会产生不正确的对齐。

基线 block manager 仍禁用已释放 block 的跨请求复用；补丁没有改变缓存生命周期策略。对仍被引用的 prefix、同批共享 prefix、重算时产生的 prefix 命中，新模型路径可消费缓存。全命中会重算末块，防止 0 query token 导致无法取得 logits。这一 runner 分支和真实 scheduler → GPU 的完整链路还需要 GPU 集成验收。

SparseMoE 中动态 `where`/变长索引分配会同步和改变 kernel 调度，不能安全捕获为已有 CUDA graph。首版强制 eager，并限制单卡。Expert 全驻留同一设备；没有 EP dispatch、TP shard、all-to-all、容量限制或 token dropping。Prefill 的逐序列 SDPA 和显式 mask，以及 FP32 Norm/RoPE，优先保证数值清晰；没有承诺 fused MoE 性能。

## 已执行的验证

环境：macOS，Python 3.11.15，torch 2.14.0，transformers 4.57.6，CUDA 不可用。

- 新测试：`18 passed, 4 skipped`。包括 FP16/BF16 的 head_dim=96 prefill、top-1/2/4 与两种归一化模式、空输入、变长批次、2 步跨页 decode、缓存 suffix prefill、split/packed/分片/绑定及错误加载。
- 实际下载的官方 FP16 checkpoint 转 FP32 后与 HF 模型比较：5 个固定 token 的 prefill + 8 步 cache decode；最大 logit 绝对误差 `2.288818e-5`，9 次 argmax 全一致。参考模型通过 HF 自带 from_pretrained 独立加载原 checkpoint；物理页逆序，decode 跨页。该固定 token 输入只用于数值测试，不能评估中文回答质量。
- 公开原生 MiniMind 源码：随机小模型、top-2、CPU FP32，与新模型 prefill 全位置 logits 最大误差 `0.0`（现场独立对照，不将上游源码复制进仓库）。
- `compileall`、`git diff --check` 通过。
- 旧 `tests/test_scheduler.py` 的 8 项测试失败，均因 `Sequence(...)` 缺失必需 `block_size`。已将 **未修改 HEAD** 导出到 `/tmp/tinyvllm-baseline-tests` 重跑，复现相同 8 项失败；不是本补丁引入。未为了让报告全绿而改这些旧测试。
- 尚未执行：Triton 编译/运行、NCCL engine 启动、真实 scheduler 批处理、CUDA BF16、GPU 长上下文及性能测试。GPU 结果不得由 CPU 数值结果推定。

## 复现方式

Mac 不需要安装 vLLM 或 Triton。独立 CPU 环境避免项目默认 CUDA 依赖：

```bash
uv venv /tmp/tinyvllm-moe-venv --python 3.11
uv pip install --python /tmp/tinyvllm-moe-venv/bin/python torch 'transformers==4.57.6' safetensors pytest xxhash
PYTHONPATH=src /tmp/tinyvllm-moe-venv/bin/python -m pytest tests/test_minimind_moe.py -q
PYTHONPATH=src /tmp/tinyvllm-moe-venv/bin/python scripts/check_minimind.py /tmp/minimind-3-moe-reference --device cpu
```

本次参考模型与 config 保存在 `/tmp/minimind-3-moe-reference`，测试环境也在 `/tmp`，未加入项目依赖锁文件。对照脚本的 HF 4.57.6 参考加载器要求官方 split 格式；packed 转换由单元测试覆盖。复现需要完整本地 `config.json` 与 `model.safetensors`；模型可从上面固定版本的 HF 页面取得。报告没有提交 397 MB 权重。

Linux NVIDIA 环境，在项目运行依赖已安装后：

```bash
PYTHONPATH=src python -m pytest tests/test_minimind_moe.py -q
PYTHONPATH=src python scripts/check_minimind.py /path/to/minimind-3-moe --device cuda
python main_minimind.py --model /path/to/minimind-3-moe
```

GPU 验收首先确认 4 项 CUDA 测试不再 skip，再运行真实权重对照和 engine 示例。随后覆盖相同整块 prompt、共享部分 prefix、EOS、不同 prompt 长度、大 batch、接近窗口上限和抢占重算。吞吐/显存应在正确性通过后记录；不可把 CPU reference 的速度当成 TinyvLLM GPU 性能。

## 下一阶段的具体改动

1. `minimind_moe.py`：将 Q/K/V 合并为一个 GEMM，保持 Q/K head norm 顺序；expert gate/up 合并。`minimind_loader.py` 同步按子矩阵映射，并增加拆分加载与参考结果测试。
2. 新增 `layers/moe.py`：引入 grouped/fused GEMM、token 排序与逆置换。先支持 top-1/2，然后讨论固定容量的 graph capture；不要直接删除 eager guard。
3. `layers/attention.py`：为新 MoE 实现 paged varlen prefill，替换逐序列 gather+SDPA；覆盖 96 维、suffix causal mask 和 prefix position。原 dense prefill 不能因为本补丁存在就视作已修好。
4. `registry.py`、loader 与 MoE：实现 TP 时必须分别切 Q/K/V、expert intermediate 轴，router 复制，expert down 后归约；测试后才解除 world_size guard。EP 是另一个任务，不能只复用 dense TP 代码。
5. 在真实 Linux GPU 上扩大数值和调度回归后，再支持 YaRN、混合 dense/MoE、原生训练 checkpoint 转换入口及量化。
