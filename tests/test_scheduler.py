"""Scheduler/state tests run on CPU with only the Python standard library."""
import pickle
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from myvllm.engine.batch import build_batch
from myvllm.engine.scheduler import Scheduler, SchedulerOutput, ScheduledSequence
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.sampling_parameters import SamplingParams


def make_scheduler(tokens=8, seqs=4, blocks=32, block_size=4, chunk=0):
    return Scheduler(seqs, tokens, blocks, block_size, eos=0,
                     long_prefill_token_threshold=chunk)


def sequence(tokens, block_size=4, max_tokens=3, **kwargs):
    return Sequence(list(tokens), block_size, SamplingParams(max_tokens=max_tokens, **kwargs))


class SchedulerTests(unittest.TestCase):
    def assert_ownership(self, scheduler):
        bm = scheduler.block_manager
        owned = [b for seq in scheduler.running for b in seq.block_table]
        self.assertEqual(len(owned), len(set(owned)))
        self.assertEqual(set(owned), bm.used_block_ids)
        self.assertEqual(set(owned) | set(bm.free_block_ids), set(range(len(bm.blocks))))
        self.assertFalse(set(owned) & set(bm.free_block_ids))
        self.assertEqual(len(bm.free_block_ids), len(set(bm.free_block_ids)))
        for b in bm.blocks:
            self.assertEqual(b.ref_count, int(b.block_id in bm.used_block_ids))
        for seq in scheduler.waiting:
            self.assertEqual(seq.block_table, [])
            self.assertEqual(seq.num_computed_tokens, 0)

    def test_long_prompt_partial_chunks_do_not_sample(self):
        scheduler = make_scheduler(tokens=3)
        seq = sequence(range(10))
        scheduler.add_sequence(seq)
        for expected in [3, 6, 9]:
            batch = scheduler.schedule()
            self.assertEqual(batch.num_scheduled_tokens, 3)
            self.assertFalse(batch.requests[0].should_sample)
            scheduler.postprocess(batch, [])
            self.assertEqual(seq.num_computed_tokens, expected)
            self.assertEqual(seq.num_completion_tokens, 0)
            self.assertEqual(len(seq.block_table), (expected + 3) // 4)
        batch = scheduler.schedule()
        self.assertEqual(batch.num_scheduled_tokens, 1)
        scheduler.postprocess(batch, [99])
        self.assertEqual(seq.num_computed_tokens, 10)
        self.assertEqual(seq.completion_token_ids, [99])

    def test_unified_decode_and_prefill_share_one_budget(self):
        scheduler = make_scheduler(tokens=4)
        decode = sequence([1, 2])
        scheduler.add_sequence(decode)
        scheduler.postprocess(scheduler.schedule(), [3])
        prefill = sequence(range(9))
        scheduler.add_sequence(prefill)
        batch = scheduler.schedule()
        self.assertEqual([r.sequence for r in batch.requests], [decode, prefill])
        self.assertEqual([r.num_tokens for r in batch.requests], [1, 3])
        self.assertFalse(batch.is_decode_only)
        scheduler.postprocess(batch, [4])
        self.assertEqual(decode.completion_token_ids, [3, 4])
        self.assertEqual(prefill.num_computed_tokens, 3)
        self.assertEqual(prefill.completion_token_ids, [])

    def test_token_limit_preserves_unscheduled_requests(self):
        scheduler = make_scheduler(tokens=3)
        seqs = [sequence([i + 1]) for i in range(3)]
        for seq in seqs:
            scheduler.add_sequence(seq)
        scheduler.postprocess(scheduler.schedule(), [5, 6, 7])
        scheduler.max_num_batched_tokens = 2
        batch = scheduler.schedule()
        self.assertTrue(batch.is_decode_only)
        self.assertEqual([r.sequence for r in batch.requests], seqs[:2])
        self.assertEqual(list(scheduler.running), seqs)
        scheduler.postprocess(batch, [8, 9])
        self.assertEqual(seqs[2].num_computed_tokens, 1)

    def test_sequence_limit_counts_all_running_requests(self):
        scheduler = make_scheduler(tokens=20, seqs=2)
        seqs = [sequence([1]) for _ in range(3)]
        for seq in seqs:
            scheduler.add_sequence(seq)
        batch = scheduler.schedule()
        self.assertEqual(len(batch.requests), 2)
        self.assertEqual(list(scheduler.waiting), seqs[2:])
        self.assertEqual(len(scheduler.running), 2)

    def test_preemption_recomputes_generated_history(self):
        scheduler = make_scheduler(tokens=8, blocks=5, block_size=2)
        a, b = [sequence(range(4), block_size=2, max_tokens=4) for _ in range(2)]
        for seq in (a, b):
            scheduler.add_sequence(seq)
        scheduler.postprocess(scheduler.schedule(), [10, 20])
        batch = scheduler.schedule()
        self.assertEqual([r.sequence for r in batch.requests], [a])
        self.assertEqual(b.status, SequenceStatus.WAITING)
        self.assertEqual(b.token_ids, [0, 1, 2, 3, 20])
        self.assertEqual(b.num_computed_tokens, 0)
        scheduler.postprocess(batch, [11])
        replay_seen = False
        for _ in range(30):
            if scheduler.is_finished():
                break
            batch = scheduler.schedule()
            for r in batch.requests:
                if r.sequence is b and r.start == 0:
                    replay_seen = True
            scheduler.postprocess(batch, [42 for r in batch.requests if r.should_sample])
            self.assert_ownership(scheduler)
        self.assertTrue(replay_seen)
        self.assertTrue(scheduler.is_finished())
        self.assertEqual(b.completion_token_ids[0], 20)
        self.assertEqual(b.num_completion_tokens, 4)

    def test_preemption_of_tail_preserves_current_request(self):
        scheduler = make_scheduler(tokens=8, blocks=4, block_size=2)
        a, b = [sequence(range(4), block_size=2) for _ in range(2)]
        for seq in (a, b):
            scheduler.add_sequence(seq)
        scheduler.postprocess(scheduler.schedule(), [10, 20])
        batch = scheduler.schedule()
        self.assertEqual([r.sequence for r in batch.requests], [a])
        self.assertEqual(list(scheduler.waiting), [b])
        self.assertEqual(list(scheduler.running), [a])
        self.assert_ownership(scheduler)

    def test_invalid_output_does_not_advance_cursor(self):
        scheduler = make_scheduler(tokens=2)
        seq = sequence(range(5))
        scheduler.add_sequence(seq)
        batch = scheduler.schedule()
        with self.assertRaises(ValueError):
            scheduler.postprocess(batch, [99])
        self.assertEqual(seq.num_computed_tokens, 0)
        self.assertEqual(len(seq), 5)

    def test_identical_prompts_do_not_share_uncomputed_kv(self):
        scheduler = make_scheduler(tokens=8)
        a, b = [sequence([1, 2, 3, 4]) for _ in range(2)]
        scheduler.add_sequence(a)
        scheduler.add_sequence(b)
        scheduler.schedule()
        self.assertFalse(set(a.block_table) & set(b.block_table))
        self.assertEqual((a.num_computed_tokens, b.num_computed_tokens), (0, 0))

    def test_stopping_and_releasing_blocks(self):
        for params, output_tokens in [({}, [0]), ({'max_tokens': 1}, [7]),
                                      ({'max_model_length': 5}, [7])]:
            with self.subTest(params=params):
                scheduler = make_scheduler(tokens=3)
                seq = sequence(range(4), **params)
                scheduler.add_sequence(seq)
                scheduler.postprocess(scheduler.schedule(), [])
                scheduler.postprocess(scheduler.schedule(), output_tokens)
                self.assertTrue(seq.is_finished)
                self.assertTrue(scheduler.is_finished())
                self.assert_ownership(scheduler)

    def test_ignore_eos(self):
        scheduler = make_scheduler()
        seq = sequence([1], ignore_eos=True)
        scheduler.add_sequence(seq)
        scheduler.postprocess(scheduler.schedule(), [0])
        self.assertFalse(seq.is_finished)

    def test_invalid_limits_and_prompts(self):
        for kwargs in [{'tokens': 0}, {'seqs': 0}, {'blocks': 0}, {'block_size': 0}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_scheduler(**kwargs)
        scheduler = make_scheduler(blocks=1)
        for seq in [sequence([]), sequence(range(5)), sequence([1], block_size=2),
                    sequence([1], max_tokens=0), sequence([1], max_model_length=1)]:
            with self.subTest(seq=seq.token_ids), self.assertRaises(ValueError):
                scheduler.add_sequence(seq)

    def test_decode_outgrowing_cache_fails_instead_of_livelock(self):
        scheduler = make_scheduler(tokens=4, blocks=1)
        seq = sequence(range(4))
        scheduler.add_sequence(seq)
        scheduler.postprocess(scheduler.schedule(), [7])
        with self.assertRaisesRegex(ValueError, 'KV cache'):
            scheduler.schedule()

    def test_chunk_cap_leaves_budget_for_later_decode(self):
        scheduler = make_scheduler(tokens=8, chunk=3)
        long = sequence(range(12))
        short = sequence([1])
        scheduler.add_sequence(long)
        scheduler.add_sequence(short)
        first = scheduler.schedule()
        self.assertEqual([r.num_tokens for r in first.requests], [3, 1])
        scheduler.postprocess(first, [9])
        second = scheduler.schedule()
        self.assertEqual([r.num_tokens for r in second.requests], [3, 1])
        self.assertEqual(second.requests[1].sequence, short)
        scheduler.postprocess(second, [10])
        self.assertEqual(short.completion_token_ids, [9, 10])
        self.assertEqual(long.num_computed_tokens, 6)
        self.assert_ownership(scheduler)

    def test_chunk_cap_one_and_unlimited(self):
        for cap, expected in [(0, 8), (1, 1), (3, 3), (20, 8)]:
            scheduler = make_scheduler(tokens=8, chunk=cap)
            scheduler.add_sequence(sequence(range(12)))
            self.assertEqual(scheduler.schedule().num_scheduled_tokens, expected)
        with self.assertRaises(ValueError):
            make_scheduler(chunk=-1)

    def test_empty_scheduler(self):
        batch = make_scheduler().schedule()
        self.assertEqual(batch.requests, [])
        self.assertEqual(batch.num_scheduled_tokens, 0)

    def test_randomized_completion_and_block_conservation(self):
        for seed in range(20):
            rng = random.Random(seed)
            scheduler = make_scheduler(tokens=rng.randint(1, 9), seqs=3, blocks=6,
                                       block_size=2, chunk=rng.choice([0, 1, 3]))
            seqs = [sequence(range(rng.randint(1, 7)), block_size=2,
                             max_tokens=rng.randint(1, 4), ignore_eos=True) for _ in range(8)]
            for seq in seqs:
                scheduler.add_sequence(seq)
            for _ in range(1000):
                if scheduler.is_finished():
                    break
                batch = scheduler.schedule()
                self.assertLessEqual(batch.num_scheduled_tokens, scheduler.max_num_batched_tokens)
                self.assertLessEqual(len(scheduler.running), scheduler.max_num_sequences)
                self.assertEqual(set(scheduler.running) | set(scheduler.waiting),
                                 {seq for seq in seqs if not seq.is_finished})
                scheduler.postprocess(batch, [17 for r in batch.requests if r.should_sample])
                self.assert_ownership(scheduler)
            self.assertTrue(scheduler.is_finished(), f'seed={seed}')
            for seq in seqs:
                self.assertEqual(seq.num_completion_tokens, seq.max_tokens)


class BatchTests(unittest.TestCase):
    def test_positions_slots_and_sample_rows_with_unaligned_chunks(self):
        a = sequence(range(10))
        a.block_table = [5, 2, 9]
        a.num_computed_tokens = 3
        b = sequence([20, 21, 22])
        b.block_table = [7]
        b.num_computed_tokens = 2
        batch = build_batch(SchedulerOutput([
            ScheduledSequence(a, 3, 4), ScheduledSequence(b, 2, 1)]))
        self.assertEqual(batch.input_ids, [3, 4, 5, 6, 22])
        self.assertEqual(batch.positions, [3, 4, 5, 6, 2])
        self.assertEqual(batch.slot_mapping, [23, 8, 9, 10, 30])
        self.assertEqual(batch.cu_seqlens_q, [0, 4, 5])
        self.assertEqual(batch.context_lens, [7, 3])
        self.assertEqual(batch.block_tables, [[5, 2, 9], [7, -1, -1]])
        self.assertEqual(batch.sample_indices, [1])

    def test_worker_pickle_keeps_recompute_history_and_sampling_metadata(self):
        seq = sequence([1, 2, 3])
        seq.append_token(4)
        seq.num_computed_tokens = 2
        seq.block_table = [7]
        output = SchedulerOutput([ScheduledSequence(seq, 2, 2)])
        restored = pickle.loads(pickle.dumps(output))
        self.assertEqual(restored.requests[0].sequence.__dict__, seq.__dict__)
        self.assertEqual(build_batch(restored), build_batch(output))


if __name__ == '__main__':
    unittest.main()
