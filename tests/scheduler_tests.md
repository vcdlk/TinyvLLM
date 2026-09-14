# Scheduler and chunked attention tests

Run the full suite from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

`test_scheduler.py` uses only the standard library. It preserves coverage for
requests lost at token/sequence limits and during tail preemption, and adds
chunked prefill, unified mixed batches, deferred sampling, incremental block
allocation, stopping conditions, replay after preemption, TP serialization,
a per-request chunk cap, and randomized block ownership/conservation checks.

`test_scheduling_benchmark.py` uses a simulated GPU clock to verify that TTFT
includes delayed admission, ITL excludes intermediate prefill chunks, and throughput
counts completion tokens. It also covers idle arrival periods and one-token outputs.

`test_chunked_attention.py` requires PyTorch, Triton and CUDA. Without them, its
four numerical/model tests are explicitly skipped. On a GPU it compares paged
attention with a dense causal reference, and compares Qwen3/Llama full-prefill
logits and generated tokens against chunked/mixed execution. It also exercises
the transition from a mixed batch to decode CUDA graphs. No model download is
required. The model tests initialize a single-rank NCCL group; multi-rank TP
still needs a separate integration run.

See [the design notes](../docs/unified_scheduling.md) for scheduling semantics
and current limitations.
