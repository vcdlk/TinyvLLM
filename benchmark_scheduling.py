"""Compare scheduling budgets on one fixed model/KV pool. GPU imports are lazy."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from myvllm.engine.sequence import Sequence
from myvllm.sampling_parameters import SamplingParams


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution_ms(values):
    return {'p50': percentile(values, 0.5), 'p95': percentile(values, 0.95),
            'mean': statistics.fmean(values) if values else None}


def run_workload(engine, prompts, output_tokens, arrival_interval, clock=time.perf_counter,
                 sleep=time.sleep):
    """TTFT includes time since intended arrival, including host admission delay.

    Synchronizing each output transfer timestamps completed GPU work. No HTTP,
    tokenization or model loading is included. ITL is pooled over token gaps.
    """
    if not engine.scheduler.is_finished():
        raise ValueError('Benchmark requires an idle engine')
    params = SamplingParams(max_tokens=output_tokens, ignore_eos=True,
                            max_model_length=engine.config['max_model_length'])
    seqs = [Sequence(tokens, engine.config['block_size'], params) for tokens in prompts]
    timestamps = {seq.seq_id: [] for seq in seqs}
    submitted = 0
    steps = {'decode_only': 0, 'prefill_only': 0, 'mixed': 0, 'preempt_only': 0}
    computed_tokens = 0
    started = clock()
    while submitted < len(seqs) or not engine.scheduler.is_finished():
        now = clock()
        while submitted < len(seqs) and now - started >= submitted * arrival_interval:
            engine.scheduler.add_sequence(seqs[submitted])
            submitted += 1
        if engine.scheduler.is_finished():
            sleep(min(0.001, max(0, started + submitted * arrival_interval - clock())))
            continue
        batch = engine.scheduler.schedule()
        if not batch.requests:
            steps['preempt_only'] += 1
            continue
        # Snapshot before postprocess appends tokens and changes should_sample.
        sampled = [r.sequence for r in batch.requests if r.should_sample]
        decode = [r.num_tokens == 1 and r.start > 0 and r.should_sample for r in batch.requests]
        kind = 'decode_only' if all(decode) else 'mixed' if any(decode) else 'prefill_only'
        steps[kind] += 1
        computed_tokens += batch.num_scheduled_tokens
        tokens = engine.model_runner.call('run', batch).cpu().tolist()
        engine.scheduler.postprocess(batch, tokens)
        completed = clock()
        for seq in sampled:
            timestamps[seq.seq_id].append(completed)
    ended = clock()
    records = []
    for index, seq in enumerate(seqs):
        stamps = timestamps[seq.seq_id]
        if len(stamps) != output_tokens:
            raise RuntimeError('Workload terminated before its fixed output length')
        records.append({
            'request': index, 'prompt_tokens': seq.num_prompt_tokens,
            'output_tokens': seq.num_completion_tokens,
            'ttft_ms': (stamps[0] - started - index * arrival_interval) * 1000,
            'itl_ms': [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])],
            'latency_ms': (stamps[-1] - started - index * arrival_interval) * 1000,
        })
    elapsed = ended - started
    return {
        'elapsed_seconds': elapsed,
        'output_tokens_per_second': sum(r['output_tokens'] for r in records) / elapsed,
        'computed_input_tokens': computed_tokens,
        'ttft_ms': distribution_ms([r['ttft_ms'] for r in records]),
        'itl_ms': distribution_ms([gap for r in records for gap in r['itl_ms']]),
        'steps': steps, 'requests': records,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['qwen', 'llama'], default='qwen')
    parser.add_argument('--model-path', help='Local path or HF ID of the supported model')
    parser.add_argument('--budgets', nargs='+', type=int, default=[64, 256, 1024])
    parser.add_argument('--chunk-caps', nargs='+', type=int, default=[0, 128])
    parser.add_argument('--prompt-lengths', nargs='+', type=int, default=[32, 2048])
    parser.add_argument('--requests', type=int, default=16)
    parser.add_argument('--output-tokens', type=int, default=64)
    parser.add_argument('--arrival-ms', type=float, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cuda-graphs', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('scheduling-results.json'))
    parser.add_argument('--dry-run', action='store_true', help='Print workload plan without GPU imports')
    args = parser.parse_args(argv)
    if min(args.budgets + args.prompt_lengths + [args.requests, args.output_tokens, args.repeats]) <= 0:
        parser.error('Budgets, lengths, request count and repeats must be positive')
    if min(args.chunk_caps) < 0 or args.arrival_ms < 0:
        parser.error('Chunk caps and arrival interval must be nonnegative')
    if max(args.prompt_lengths) + args.output_tokens > 32768:
        parser.error('Prompt plus output must fit the demo model position limit (32768)')
    return args


def main():
    args = parse_args()
    lengths = [args.prompt_lengths[i % len(args.prompt_lengths)] for i in range(args.requests)]
    plan = {'budgets': args.budgets, 'chunk_caps': args.chunk_caps,
            'prompt_lengths': lengths, 'output_tokens': args.output_tokens,
            'arrival_ms': args.arrival_ms, 'repeats': args.repeats}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('This benchmark requires an NVIDIA CUDA GPU; use --dry-run on CPU.')
    from myvllm.engine.llm_engine import LLMEngine
    if args.model == 'qwen':
        from main import config as base_config
    else:
        from main_llama32 import config as base_config
    config = dict(base_config, max_num_batched_tokens=max(args.budgets),
                  max_model_length=max(lengths) + args.output_tokens,
                  enforce_eager=not args.cuda_graphs)
    if args.model_path:
        config['model_name_or_path'] = args.model_path
    engine = LLMEngine(config)
    try:
        seed_ids = engine.tokenizer.encode('Explain how language models process a sequence of tokens.',
                                           add_special_tokens=False)
        prompts = [(seed_ids * ((length + len(seed_ids) - 1) // len(seed_ids)))[:length]
                   for length in lengths]
        report = {'plan': plan, 'model': config['model_name_or_path'],
                  'gpu': torch.cuda.get_device_name(0), 'torch': str(torch.__version__),
                  'cuda': torch.version.cuda, 'dtype': str(next(engine.model_runner.model.parameters()).dtype),
                  'kv_blocks': engine.config['max_cached_blocks'], 'cuda_graphs': args.cuda_graphs,
                  'results': []}
        # A single loaded model and KV pool keep available cache capacity fixed.
        for budget in args.budgets:
            for cap in args.chunk_caps:
                engine.scheduler.max_num_batched_tokens = budget
                engine.scheduler.long_prefill_token_threshold = cap
                torch.manual_seed(0)
                run_workload(engine, prompts, args.output_tokens, args.arrival_ms / 1000)
                for repeat in range(args.repeats):
                    torch.manual_seed(repeat)
                    result = run_workload(engine, prompts, args.output_tokens, args.arrival_ms / 1000)
                    result.update(budget=budget, chunk_cap=cap, repeat=repeat)
                    report['results'].append(result)
                    print(f"budget={budget} cap={cap} repeat={repeat}: "
                          f"TTFT p95={result['ttft_ms']['p95']:.2f} ms, "
                          f"ITL p95={result['itl_ms']['p95']} ms, "
                          f"output={result['output_tokens_per_second']:.2f} tokens/s")
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2) + '\n')
    finally:
        engine.exit()


if __name__ == '__main__':
    main()
