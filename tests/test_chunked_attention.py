"""CUDA numerical regressions; skipped explicitly on CPU-only machines."""
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

HAS_CUDA = False
if importlib.util.find_spec('torch') and importlib.util.find_spec('triton'):
    import torch
    HAS_CUDA = torch.cuda.is_available()


@unittest.skipUnless(HAS_CUDA, 'Requires PyTorch, Triton and a CUDA GPU')
class PagedPrefillTests(unittest.TestCase):
    def test_mixed_queries_against_dense_causal_attention(self):
        from myvllm.layers.attention import flash_attention_prefill, store_kvcache
        torch.manual_seed(0)
        for block_size in [1, 4, 16, 64]:
            for head_dim in [32, 64, 128]:
                with self.subTest(block_size=block_size, head_dim=head_dim):
                    heads, kv_heads = 4, 2
                    # Includes a long cached prefix, a decode and a fresh prefill.
                    qlens, klens = [7, 1, 35], [73, 17, 35]
                    width = math.ceil(max(klens) / block_size)
                    physical = torch.randperm(3 * width, device='cuda').reshape(3, width)
                    kcache = torch.full((3 * width, block_size, kv_heads, head_dim),
                                        float('nan'), dtype=torch.float16, device='cuda')
                    vcache = torch.full_like(kcache, float('nan'))
                    queries, expected, new_keys, new_values, new_slots = [], [], [], [], []
                    for i, (qlen, klen) in enumerate(zip(qlens, klens)):
                        q = torch.randn(qlen, heads, head_dim, dtype=torch.float16, device='cuda')
                        k = torch.randn(klen, kv_heads, head_dim, dtype=torch.float16, device='cuda')
                        v = torch.randn_like(k)
                        positions = torch.arange(klen, device='cuda')
                        slots = physical[i, positions // block_size] * block_size + positions % block_size
                        prefix = klen - qlen
                        kcache.view(-1, kv_heads, head_dim)[slots[:prefix]] = k[:prefix]
                        vcache.view(-1, kv_heads, head_dim)[slots[:prefix]] = v[:prefix]
                        new_keys.append(k[prefix:])
                        new_values.append(v[prefix:])
                        new_slots.append(slots[prefix:])
                        queries.append(q)
                        kk = k.repeat_interleave(heads // kv_heads, dim=1).float()
                        vv = v.repeat_interleave(heads // kv_heads, dim=1).float()
                        scores = torch.einsum('qhd,khd->hqk', q.float(), kk) / math.sqrt(head_dim)
                        causal = positions[None, :] <= (prefix + torch.arange(qlen, device='cuda'))[:, None]
                        scores.masked_fill_(~causal[None], -torch.inf)
                        expected.append(torch.einsum('hqk,khd->qhd', scores.softmax(-1), vv))
                    store_kvcache(torch.cat(new_keys), torch.cat(new_values), kcache, vcache,
                                  torch.cat(new_slots), block_size)
                    cu = torch.tensor([0, 7, 8, 43], dtype=torch.int32, device='cuda')
                    actual = flash_attention_prefill(
                        torch.cat(queries), kcache, vcache, cu, 1 / math.sqrt(head_dim),
                        heads, kv_heads, head_dim, block_tables=physical.int(),
                        context_lens=torch.tensor(klens, dtype=torch.int32, device='cuda'),
                        max_seqlen_q=max(qlens))
                    torch.testing.assert_close(actual.float(), torch.cat(expected), atol=3e-3, rtol=3e-3)


@unittest.skipUnless(HAS_CUDA, 'Requires PyTorch, Triton and a CUDA GPU')
class ModelChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch.distributed as dist
        cls.owns_group = not dist.is_initialized()
        if cls.owns_group:
            cls.tempdir = tempfile.TemporaryDirectory()
            dist.init_process_group('nccl', init_method=Path(cls.tempdir.name, 'store').as_uri(),
                                    rank=0, world_size=1)
        torch.cuda.set_device(0)

    @classmethod
    def tearDownClass(cls):
        if cls.owns_group:
            import torch.distributed as dist
            dist.destroy_process_group()
            cls.tempdir.cleanup()

    def make_runner(self, kind):
        from myvllm.engine.model_runner import ModelRunner
        from myvllm.layers.attention import Attention
        from myvllm.models.qwen3 import Qwen3ForCausalLM
        from myvllm.models.llama import LlamaForCausalLM
        torch.manual_seed(123)
        common = dict(vocab_size=64, hidden_size=128, head_dim=32, num_kv_heads=2,
                      intermediate_size=128, num_layers=2, block_size=4, ffn_bias=False)
        if kind == 'qwen':
            model = Qwen3ForCausalLM(**common, num_heads=4, max_position=64)
        else:
            model = LlamaForCausalLM(**common, num_qo_heads=4, max_position_embeddings=64)
        model = model.cuda().half().eval()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim == 1:
                    parameter.fill_(1)
                else:
                    parameter.normal_(std=0.04)
        for module in model.modules():
            if isinstance(module, Attention):
                module.k_cache = torch.zeros(32, 4, 2, 32, device='cuda', dtype=torch.float16)
                module.v_cache = torch.zeros_like(module.k_cache)
        runner = ModelRunner.__new__(ModelRunner)
        runner.model = model
        runner.rank = 0
        runner.block_size = 4
        runner.enforce_eager = True
        runner.config = dict(max_num_sequences=3, max_model_length=64, hidden_size=128)
        return runner

    def generate(self, runner, budget):
        from myvllm.engine.scheduler import Scheduler
        from myvllm.engine.sequence import Sequence
        from myvllm.sampling_parameters import SamplingParams
        from myvllm.utils import get_context
        scheduler = Scheduler(3, budget, 32, 4, eos=-1)
        params = SamplingParams(max_tokens=4, ignore_eos=True)
        seqs = [Sequence([1, 3, 5, 7, 9, 11, 13], 4, params),
                Sequence([2, 4, 6, 8, 10, 12, 14, 16, 18], 4, params)]
        scheduler.add_sequence(seqs[0])
        added = False
        logits_by_seq = {seq.seq_id: [] for seq in seqs}
        for _ in range(100):
            if not added and seqs[0].num_completion_tokens:
                scheduler.add_sequence(seqs[1])
                added = True
            if scheduler.is_finished():
                break
            batch = scheduler.schedule()
            sampled = []
            def sampler(logits, temperatures):
                sampled.extend(logits.detach().float().unbind(0))
                return logits.argmax(-1)
            runner.sampler = sampler
            tokens = runner.run(batch)
            eligible = [r for r in batch.requests if r.should_sample]
            self.assertEqual(len(sampled), len(eligible))
            for request, logits in zip(eligible, sampled):
                logits_by_seq[request.sequence.seq_id].append(logits)
            scheduler.postprocess(batch, tokens.cpu().tolist())
            self.assertIsNone(get_context().positions)
        self.assertTrue(scheduler.is_finished())
        return ([seq.completion_token_ids for seq in seqs],
                [torch.stack(logits_by_seq[seq.seq_id]) for seq in seqs])

    def check_model(self, kind, graphs=False):
        runner = self.make_runner(kind)
        reference_tokens, reference_logits = self.generate(runner, 64)
        if graphs:
            runner.capture_cudagraph()
            runner.enforce_eager = False
        tokens, logits = self.generate(runner, 3)
        self.assertEqual(tokens, reference_tokens)
        for actual, expected in zip(logits, reference_logits):
            torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)

    def test_qwen_full_vs_chunked_and_mixed_logits(self):
        self.check_model('qwen')

    def test_llama_full_vs_chunked_and_mixed_logits(self):
        self.check_model('llama')

    def test_decode_graph_after_mixed_batches(self):
        self.check_model('qwen', graphs=True)


if __name__ == '__main__':
    unittest.main()
