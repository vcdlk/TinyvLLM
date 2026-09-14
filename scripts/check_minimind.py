"""Compare a local HF MiniMind-3-MoE checkpoint on CPU or Linux CUDA.

PYTHONPATH=src python scripts/check_minimind.py /path/to/checkpoint --device cpu
This loads weights only (no remote Python) and does not require a tokenizer.
"""
import argparse
import math
import torch
from transformers import Qwen3MoeForCausalLM
from myvllm.models.registry import read_model_config, validate_minimind_config
from myvllm.models.minimind_moe import MiniMindMoeForCausalLM
from myvllm.utils.minimind_loader import load_minimind_checkpoint
from myvllm.utils.context import set_context, reset_context


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--steps', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = validate_minimind_config(read_model_config(args.checkpoint))
    if cfg['model_type'] != 'qwen3_moe':
        raise ValueError('This reference comparison requires a Qwen3-MoE export')
    dtype = torch.float32 if args.device == 'cpu' else torch.float16
    model = MiniMindMoeForCausalLM(cfg, block_size=4).eval().to(device=args.device, dtype=dtype)
    load_minimind_checkpoint(model, args.checkpoint)
    # Independent loading is intentional: compare the custom mapping with HF's
    # own loader. This checker expects the public Transformers 4.x split format;
    # packed Transformers 5.x conversion has separate unit coverage.
    reference = Qwen3MoeForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=dtype, local_files_only=True
    ).eval().to(args.device)
    ids = torch.tensor([[1, 24, 35, 48, 56]], device=args.device)
    assert cfg['vocab_size'] > ids.max()
    blocks = math.ceil((ids.shape[1] + args.steps) / 4)
    table = torch.arange(blocks - 1, -1, -1, device=args.device, dtype=torch.int32)[None]
    for layer in model.model.layers:
        shape = (blocks, 4, cfg['num_key_value_heads'], cfg['head_dim'])
        layer.self_attn.k_cache = torch.zeros(shape, device=args.device, dtype=dtype)
        layer.self_attn.v_cache = torch.zeros(shape, device=args.device, dtype=dtype)
    for step in range(args.steps + 1):
        if step == 0:
            positions = torch.arange(ids.shape[1], device=args.device)
            bounds = torch.tensor([0, ids.shape[1]], device=args.device, dtype=torch.int32)
            set_context(True, cu_seqlens_q=bounds, cu_seqlens_k=bounds,
                        slot_mapping=table[0, positions // 4].long() * 4 + positions % 4)
            input_ids = ids[0]
        else:
            pos = ids.shape[1] - 1
            set_context(False, context_lens=torch.tensor([pos + 1], device=args.device),
                        block_tables=table, slot_mapping=table[:, pos // 4].long() * 4 + pos % 4)
            input_ids = ids[0, -1:]
        actual = model.compute_logits(model(input_ids)).float()
        expected = reference(ids).logits[:, -1].float()
        error = (actual - expected).abs().max().item()
        matches = torch.equal(actual.argmax(-1), expected.argmax(-1))
        print(f'{"prefill" if step == 0 else "decode " + str(step)}: max_abs={error:.7g}, greedy_match={matches}')
        tolerance = 3e-4 if args.device == 'cpu' else 0.1
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        if not matches:
            raise AssertionError('Greedy tokens differ; inspect logit margins and routing')
        ids = torch.cat([ids, actual.argmax(-1, keepdim=True)], dim=1)
    reset_context()
    print('token_ids:', ids.tolist())


if __name__ == '__main__':
    main()
