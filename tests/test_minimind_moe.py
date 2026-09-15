"""CPU numerical tests; CUDA tests additionally exercise real Triton KV kernels."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from myvllm.models.registry import validate_minimind_config, prepare_model_config
from myvllm.models.minimind_moe import MiniMindMoeForCausalLM, SparseMoE
from myvllm.utils.minimind_loader import load_minimind_state_dict, load_minimind_checkpoint
from myvllm.utils.context import set_context, reset_context


def config(**kwargs):
    return validate_minimind_config(dict(model_type='qwen3_moe', hidden_size=32,
        intermediate_size=48, moe_intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_experts=4, num_experts_per_tok=kwargs.pop('num_experts_per_tok', 1),
        vocab_size=64, max_position_embeddings=128, **kwargs))


@pytest.fixture(autouse=True)
def clean_context():
    torch.manual_seed(17)
    torch.set_num_threads(1)
    yield
    reset_context()


def prefill_context(lengths, device='cpu', **kwargs):
    bounds = torch.tensor([0] + lengths, device=device, dtype=torch.int32).cumsum(0).int()
    set_context(True, cu_seqlens_q=bounds, cu_seqlens_k=bounds, **kwargs)


@pytest.mark.parametrize('top_k', [1, 2, 4])
@pytest.mark.parametrize('normalize', [True, False])
def test_router_experts(top_k, normalize):
    cfg = config(num_experts_per_tok=top_k, norm_topk_prob=normalize)
    moe = SparseMoE(cfg)
    x = torch.randn(3, 5, 32)
    flat = x.flatten(0, 1)
    scores = moe.gate(flat).float().softmax(-1)
    weights, indices = scores.topk(top_k, dim=-1)
    if normalize:
        weights /= weights.sum(-1, keepdim=True)
    reference = torch.zeros_like(flat)
    # Deliberately token-major, independent of the production expert dispatch.
    for token in range(flat.shape[0]):
        for slot in range(top_k):
            reference[token] += weights[token, slot] * moe.experts[indices[token, slot]](flat[token])
    torch.testing.assert_close(moe(x).flatten(0, 1), reference)
    if top_k == 1 and normalize:
        torch.testing.assert_close(moe.route(flat)[0], torch.ones(15, 1))
    assert moe(torch.empty(0, 32)).shape == (0, 32)


def hf_pair(cfg, device='cpu', dtype=torch.float32):
    hf = Qwen3MoeForCausalLM(Qwen3MoeConfig(**cfg)).eval().to(device=device, dtype=dtype)
    tiny = MiniMindMoeForCausalLM(cfg, block_size=4).eval().to(device=device, dtype=dtype)
    load_minimind_state_dict(tiny, hf.state_dict())
    return hf, tiny


@pytest.mark.parametrize('top_k', [1, 2])
@torch.inference_mode()
def test_hf_varlen_prefill_and_decode(top_k):
    hf, tiny = hf_pair(config(num_experts_per_tok=top_k))
    a = torch.tensor([2, 6, 8, 3, 7]); b = torch.tensor([1, 4, 9])
    prefill_context([5, 3])
    actual = tiny.lm_head(tiny(torch.cat([a, b])))
    expected = torch.cat([hf(a[None]).logits[0], hf(b[None]).logits[0]])
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    # Non-contiguous physical pages, unequal context lengths, crossing a block boundary.
    tables = torch.tensor([[3, 1], [2, 0]], dtype=torch.int32)
    slots = torch.tensor([12, 13, 14, 15, 4, 8, 9, 10])
    for layer in tiny.model.layers:
        attn = layer.self_attn
        attn.k_cache = torch.zeros(4, 4, 2, 8)
        attn.v_cache = torch.zeros_like(attn.k_cache)
    prefill_context([5, 3], slot_mapping=slots)
    tiny(torch.cat([a, b]))
    for step in range(2):
        next_ids = torch.tensor([10 + step, 20 + step])
        a = torch.cat([a, next_ids[:1]]); b = torch.cat([b, next_ids[1:]])
        lens = torch.tensor([len(a), len(b)])
        slots = tables[torch.arange(2), (lens - 1) // 4].long() * 4 + (lens - 1) % 4
        set_context(False, context_lens=lens, slot_mapping=slots, block_tables=tables)
        actual = tiny.compute_logits(tiny(next_ids))
        expected = torch.cat([hf(a[None]).logits[:, -1], hf(b[None]).logits[:, -1]])
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@torch.inference_mode()
def test_cached_suffix_prefill():
    hf, tiny = hf_pair(config())
    ids = torch.tensor([1, 3, 5, 7, 9, 11, 13])
    for layer in tiny.model.layers:
        layer.self_attn.k_cache = torch.zeros(3, 4, 2, 8)
        layer.self_attn.v_cache = torch.zeros(3, 4, 2, 8)
    prefill_context([4], slot_mapping=torch.tensor([8, 9, 10, 11]))
    tiny(ids[:4])
    set_context(True, cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
                cu_seqlens_k=torch.tensor([0, 7], dtype=torch.int32),
                slot_mapping=torch.tensor([0, 1, 2]), block_tables=torch.tensor([[2, 0]], dtype=torch.int32))
    actual = tiny.lm_head(tiny(ids[4:]))
    torch.testing.assert_close(actual, hf(ids[None]).logits[0, 4:], atol=2e-6, rtol=2e-5)


def packed_state(state, cfg):
    packed = {k: v.clone() for k, v in state.items() if '.experts.' not in k and k != 'lm_head.weight'}
    for layer in range(cfg['num_hidden_layers']):
        p = f'model.layers.{layer}.mlp.experts'
        packed[f'{p}.gate_up_proj'] = torch.stack([torch.cat([
            state[f'{p}.{e}.gate_proj.weight'], state[f'{p}.{e}.up_proj.weight']])
            for e in range(cfg['num_experts'])])
        packed[f'{p}.down_proj'] = torch.stack([state[f'{p}.{e}.down_proj.weight']
                                              for e in range(cfg['num_experts'])])
    return packed


def test_packed_sharded_checkpoint_and_tying(tmp_path):
    cfg = config(); hf, tiny = hf_pair(cfg)
    packed = packed_state(hf.state_dict(), cfg)
    items = list(packed.items()); split = len(items) // 2
    files = {'a.safetensors': dict(items[:split]), 'b.safetensors': dict(items[split:])}
    for name, tensors in files.items():
        save_file(tensors, tmp_path / name)
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {key: name for name, tensors in files.items() for key in tensors}}))
    load_minimind_checkpoint(tiny, str(tmp_path))
    assert tiny.lm_head.weight is tiny.model.embed_tokens.weight
    for name, value in hf.state_dict().items():
        torch.testing.assert_close(tiny.state_dict()[name], value)


@pytest.mark.parametrize('damage', ['missing', 'shape', 'unexpected', 'duplicate', 'tied'])
def test_strict_loader_rejects_before_mutation(damage):
    cfg = config(); hf, tiny = hf_pair(cfg)
    state = {k: v.clone() for k, v in hf.state_dict().items()}
    key = 'model.layers.0.mlp.experts.0.gate_proj.weight'
    before = tiny.model.layers[0].mlp.experts[0].gate_proj.weight.clone()
    if damage == 'missing': del state[key]
    elif damage == 'shape': state[key] = state[key][:1]
    elif damage == 'unexpected': state['garbage'] = torch.ones(1)
    elif damage == 'tied': state['lm_head.weight'].add_(1)
    else: state.update(packed_state(state, cfg))
    with pytest.raises(ValueError): load_minimind_state_dict(tiny, state)
    torch.testing.assert_close(tiny.model.layers[0].mlp.experts[0].gate_proj.weight, before)


def test_registry_renamed_directory_and_guards(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps(config()))
    result = prepare_model_config({'model_name_or_path': str(tmp_path), 'enforce_eager': False})
    assert result['model_architecture'] == 'MiniMindMoeForCausalLM'
    assert result['enforce_eager'] and result['head_dim'] == 8
    with pytest.raises(ValueError, match='world_size'):
        prepare_model_config({'model_name_or_path': str(tmp_path), 'world_size': 2})
    for changes in ({'num_experts_per_tok': 5}, {'rope_scaling': {'type': 'yarn'}},
                    {'decoder_sparse_step': 2}, {'mlp_only_layers': [0]}):
        with pytest.raises(ValueError): validate_minimind_config({**config(), **changes})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires Linux CUDA and Triton')
@pytest.mark.parametrize('head_dim', [64, 96, 128])
@torch.inference_mode()
def test_cuda_paged_kernels(head_dim):
    from myvllm.layers.attention import store_kvcache, paged_attention_decode
    dev = 'cuda'; h = 2; qh = 4; block = 4
    k = torch.randn(11, h, head_dim, device=dev, dtype=torch.float16)
    v = torch.randn_like(k); q = torch.randn(1, qh, head_dim, device=dev, dtype=k.dtype)
    kc = torch.zeros(4, block, h, head_dim, device=dev, dtype=k.dtype); vc = torch.zeros_like(kc)
    table = torch.tensor([[2, 0, 3]], device=dev, dtype=torch.int32)
    pos = torch.arange(11, device=dev)
    slots = table[0, pos // block].long() * block + pos % block
    store_kvcache(k, v, kc, vc, slots, block)
    torch.testing.assert_close(kc.view(-1, h, head_dim)[slots], k)
    out = paged_attention_decode(q, kc, vc, table, torch.tensor([11], device=dev),
                                head_dim ** -0.5, qh, h, head_dim, block)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1), k.repeat_interleave(2, 1).transpose(0, 1),
        v.repeat_interleave(2, 1).transpose(0, 1)).transpose(0, 1)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_low_precision_prefill(dtype):
    cfg = config(); cfg['head_dim'] = 96
    hf, tiny = hf_pair(cfg, dtype=dtype)
    ids = torch.tensor([2, 5, 7, 11])
    prefill_context([4])
    actual = tiny.lm_head(tiny(ids))
    expected = hf(ids[None]).logits[0]
    tolerance = 3e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires Linux CUDA and Triton')
@torch.inference_mode()
def test_cuda_model_prefill_decode():
    cfg = config(); cfg['head_dim'] = 96
    hf, tiny = hf_pair(cfg, device='cuda', dtype=torch.float16)
    ids = torch.tensor([1, 3, 5, 7, 9], device='cuda')
    for layer in tiny.model.layers:
        layer.self_attn.k_cache = torch.zeros(2, 4, 2, 96, device='cuda', dtype=torch.float16)
        layer.self_attn.v_cache = torch.zeros_like(layer.self_attn.k_cache)
    prefill_context([5], device='cuda', slot_mapping=torch.arange(5, device='cuda'))
    torch.testing.assert_close(tiny.compute_logits(tiny(ids)), hf(ids[None]).logits[:, -1], atol=3e-3, rtol=3e-3)
    ids = torch.cat([ids, torch.tensor([11], device='cuda')])
    set_context(False, context_lens=torch.tensor([6], device='cuda'),
                slot_mapping=torch.tensor([5], device='cuda'),
                block_tables=torch.tensor([[0, 1]], device='cuda', dtype=torch.int32))
    torch.testing.assert_close(tiny.compute_logits(tiny(ids[-1:])), hf(ids[None]).logits[:, -1], atol=3e-3, rtol=3e-3)
