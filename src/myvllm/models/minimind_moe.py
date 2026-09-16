"""Correctness-first MiniMind-3 MoE (single-device, eager inference).

Parameter names match MiniMind / Transformers 4.x; Transformers 5.x packed
experts are expanded by minimind_loader. CPU SDPA is a numerical reference;
CUDA uses the engine's paged store/decode kernels and SDPA for prefill.
"""
import torch
from torch import nn
from torch.nn import functional as F
from myvllm.utils.context import get_context


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
                * self.weight.float()).to(x.dtype)


class Expert(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config['num_experts_per_tok']
        self.normalize = config['norm_topk_prob']
        # Native MiniMind uses input-dtype softmax; the Qwen export uses FP32.
        self.router_fp32 = config['model_type'] == 'qwen3_moe'
        self.gate = nn.Linear(config['hidden_size'], config['num_experts'], bias=False)
        self.experts = nn.ModuleList([
            Expert(config['hidden_size'], config['moe_intermediate_size'])
            for _ in range(config['num_experts'])
        ])

    def route(self, x):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32 if self.router_fp32 else logits.dtype)
        weights, indices = torch.topk(probs, self.top_k, dim=-1, sorted=False)
        if self.normalize:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return weights.to(x.dtype), indices

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        weights, indices = self.route(flat)
        result = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            tokens, slots = torch.where(indices == expert_id)
            if tokens.numel():
                result.index_add_(0, tokens, expert(flat[tokens]) * weights[tokens, slots, None])
        return result.reshape_as(x)


def token_positions(context, device):
    if not context.is_prefill:
        return context.context_lens.to(device=device, dtype=torch.long) - 1
    q_bounds = context.cu_seqlens_q.tolist()
    k_bounds = (context.cu_seqlens_k if context.cu_seqlens_k is not None
                else context.cu_seqlens_q).tolist()
    return torch.cat([
        torch.arange((k_bounds[i+1] - k_bounds[i]) - (q_bounds[i+1] - q_bounds[i]),
                     k_bounds[i+1] - k_bounds[i], device=device)
        for i in range(len(q_bounds) - 1)
    ]).long()


class MiniMindAttention(nn.Module):
    def __init__(self, config, block_size):
        super().__init__()
        hidden = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.num_kv_heads = config['num_key_value_heads']
        self.head_dim = config['head_dim']
        self.base = config['rope_theta']
        self.block_size = block_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config['rms_norm_eps'])
        self.k_norm = RMSNorm(self.head_dim, config['rms_norm_eps'])
        # The engine discovers these attributes and attaches its shared cache pool.
        self.register_buffer('k_cache', torch.empty(0), persistent=False)
        self.register_buffer('v_cache', torch.empty(0), persistent=False)

    def rotate(self, x, positions):
        inv_freq = self.base ** (-torch.arange(0, self.head_dim, 2, device=x.device,
                                             dtype=torch.float32) / self.head_dim)
        angles = positions.float()[:, None] * inv_freq[None, :]
        cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
        a, b = x.float().chunk(2, dim=-1)
        return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(x.dtype)

    def cached_kv(self, table, length):
        pos = torch.arange(length, device=self.k_cache.device)
        blocks = table[pos // self.block_size].long()
        return (self.k_cache[blocks, pos % self.block_size],
                self.v_cache[blocks, pos % self.block_size])

    def sdpa(self, q, k, v):
        # Explicit suffix mask: SDPA is_causal alone is wrong for cached prefill.
        nq, nk = q.shape[0], k.shape[0]
        mask = (torch.arange(nk, device=q.device)[None, :] <=
                torch.arange(nk - nq, nk, device=q.device)[:, None])
        repeats = self.num_heads // self.num_kv_heads
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.repeat_interleave(repeats, dim=1).transpose(0, 1),
            v.repeat_interleave(repeats, dim=1).transpose(0, 1), attn_mask=mask)
        return out.transpose(0, 1)

    def forward(self, x, positions):
        ctx = get_context()
        q = self.rotate(self.q_norm(self.q_proj(x).view(-1, self.num_heads, self.head_dim)), positions)
        k = self.rotate(self.k_norm(self.k_proj(x).view(-1, self.num_kv_heads, self.head_dim)), positions)
        v = self.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        if self.k_cache.numel() and ctx.slot_mapping is not None:
            if x.is_cuda:
                from myvllm.layers.attention import store_kvcache
                store_kvcache(k.contiguous(), v.contiguous(), self.k_cache, self.v_cache,
                              ctx.slot_mapping, self.block_size)
            else:
                valid = ctx.slot_mapping >= 0
                slots = ctx.slot_mapping[valid]
                self.k_cache.view(-1, self.num_kv_heads, self.head_dim)[slots] = k[valid]
                self.v_cache.view(-1, self.num_kv_heads, self.head_dim)[slots] = v[valid]
        if ctx.is_prefill:
            qb = ctx.cu_seqlens_q.tolist()
            kb = (ctx.cu_seqlens_k if ctx.cu_seqlens_k is not None else ctx.cu_seqlens_q).tolist()
            outputs = []
            for i, (start, end) in enumerate(zip(qb, qb[1:])):
                length = kb[i+1] - kb[i]
                if length > end - start:
                    if not self.k_cache.numel() or ctx.block_tables is None:
                        raise ValueError('Cached prefill requires KV cache and block tables')
                    ki, vi = self.cached_kv(ctx.block_tables[i], length)
                else:
                    ki, vi = k[start:end], v[start:end]
                outputs.append(self.sdpa(q[start:end], ki, vi))
            out = torch.cat(outputs)
        elif x.is_cuda:
            from myvllm.layers.attention import paged_attention_decode
            out = paged_attention_decode(q, self.k_cache, self.v_cache, ctx.block_tables,
                                         ctx.context_lens, self.head_dim ** -0.5,
                                         self.num_heads, self.num_kv_heads, self.head_dim,
                                         self.block_size)
        else:
            out = torch.cat([self.sdpa(q[i:i+1], *self.cached_kv(ctx.block_tables[i], length))
                             for i, length in enumerate(ctx.context_lens.tolist())])
        return self.o_proj(out.reshape(x.shape[0], -1))


class MiniMindBlock(nn.Module):
    def __init__(self, config, block_size):
        super().__init__()
        self.input_layernorm = RMSNorm(config['hidden_size'], config['rms_norm_eps'])
        self.post_attention_layernorm = RMSNorm(config['hidden_size'], config['rms_norm_eps'])
        self.self_attn = MiniMindAttention(config, block_size)
        self.mlp = SparseMoE(config)

    def forward(self, x, positions):
        x = x + self.self_attn(self.input_layernorm(x), positions)
        return x + self.mlp(self.post_attention_layernorm(x))


class MiniMindModel(nn.Module):
    def __init__(self, config, block_size):
        super().__init__()
        self.embed_tokens = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.layers = nn.ModuleList([MiniMindBlock(config, block_size)
                                     for _ in range(config['num_hidden_layers'])])
        self.norm = RMSNorm(config['hidden_size'], config['rms_norm_eps'])

    def forward(self, input_ids):
        positions = token_positions(get_context(), input_ids.device)
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, positions)
        return self.norm(x)


class MiniMindMoeForCausalLM(nn.Module):
    def __init__(self, config, block_size=256):
        super().__init__()
        self.config = config
        self.model = MiniMindModel(config, block_size)
        self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
        if config['tie_word_embeddings']:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        return self.model(input_ids)

    def compute_logits(self, hidden_states):
        ctx = get_context()
        if ctx.is_prefill:
            hidden_states = hidden_states[ctx.cu_seqlens_q[1:] - 1]
        return self.lm_head(hidden_states)

    def load_checkpoint(self, path):
        from myvllm.utils.minimind_loader import load_minimind_checkpoint
        return load_minimind_checkpoint(self, path)


if __name__ == "__main__":
    from myvllm.models.registry import validate_minimind_config
    from myvllm.utils.context import set_context

    config = validate_minimind_config(dict(
        model_type='qwen3_moe',
        vocab_size=50257,
        hidden_size=768,
        num_attention_heads=12,
        num_key_value_heads=4,
        head_dim=64,
        intermediate_size=3072,
        moe_intermediate_size=3072,
        num_hidden_layers=2,
        num_experts=4,
        num_experts_per_tok=2,
    ))
    model = MiniMindMoeForCausalLM(config, block_size=256).cuda()
    # varlen prefill: two length-16 sequences packed into a single 1D tensor
    input_ids = torch.randint(0, 50257, (32,)).cuda()
    bounds = torch.tensor([0, 16, 32], dtype=torch.int32).cuda()
    set_context(is_prefill=True, cu_seqlens_q=bounds, cu_seqlens_k=bounds)
    output = model(input_ids)
