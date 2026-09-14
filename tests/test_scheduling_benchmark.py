"""Verify benchmark measurement semantics with a simulated GPU clock."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_scheduling import distribution_ms, percentile, run_workload
from myvllm.engine.scheduler import Scheduler


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class FakeRunner:
    def __init__(self, clock):
        self.clock = clock

    def call(self, name, batch):
        assert name == 'run'
        self.clock.now += 0.01
        tokens = [42 for request in batch.requests if request.should_sample]
        return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: tokens))


def fake_engine(clock):
    return SimpleNamespace(scheduler=Scheduler(4, 3, 32, 4, eos=0),
                           config={'max_model_length': 64, 'block_size': 4},
                           model_runner=FakeRunner(clock))


class BenchmarkTests(unittest.TestCase):
    def test_percentiles_and_empty_itl(self):
        self.assertEqual(percentile([4, 1, 3, 2], 0.5), 2.5)
        self.assertAlmostEqual(percentile([4, 1, 3, 2], 0.95), 3.85)
        self.assertEqual(distribution_ms([]), {'p50': None, 'p95': None, 'mean': None})

    def test_arrival_delay_sampling_and_generated_token_throughput(self):
        clock = FakeClock()
        engine = fake_engine(clock)
        result = run_workload(engine, [[1, 2], [3, 4, 5, 6, 7]], 2, 0.005,
                              clock=clock, sleep=clock.sleep)
        first, second = result['requests']
        self.assertAlmostEqual(first['ttft_ms'], 10)
        # Intended arrival is 5 ms; admission waits for first step at 10 ms.
        self.assertAlmostEqual(second['ttft_ms'], 25)
        self.assertAlmostEqual(second['latency_ms'], 35)
        self.assertAlmostEqual(result['itl_ms']['mean'], 10)
        self.assertAlmostEqual(result['output_tokens_per_second'], 100)
        self.assertEqual(result['computed_input_tokens'], 9)
        self.assertEqual(result['steps']['mixed'], 1)
        self.assertTrue(engine.scheduler.is_finished())

    def test_future_arrival_and_single_token_output(self):
        clock = FakeClock()
        result = run_workload(fake_engine(clock), [[1], [2]], 1, 0.05,
                              clock=clock, sleep=clock.sleep)
        self.assertAlmostEqual(result['elapsed_seconds'], 0.06)
        self.assertAlmostEqual(result['ttft_ms']['mean'], 10)
        self.assertIsNone(result['itl_ms']['p95'])


if __name__ == '__main__':
    unittest.main()
