"""CPU metadata shared by the runner and its correctness tests."""
from dataclasses import dataclass

from myvllm.engine.scheduler import SchedulerOutput


@dataclass
class BatchMetadata:
    input_ids: list[int]
    positions: list[int]
    cu_seqlens_q: list[int]
    context_lens: list[int]
    slot_mapping: list[int]
    block_tables: list[list[int]]
    sample_indices: list[int]


def build_batch(output: SchedulerOutput) -> BatchMetadata:
    batch = BatchMetadata([], [], [0], [], [], [], [])
    width = max((len(r.sequence.block_table) for r in output.requests), default=0)
    for index, request in enumerate(output.requests):
        seq = request.sequence
        batch.input_ids.extend(seq.token_ids[request.start:request.end])
        batch.positions.extend(range(request.start, request.end))
        batch.cu_seqlens_q.append(batch.cu_seqlens_q[-1] + request.num_tokens)
        batch.context_lens.append(request.end)
        batch.block_tables.append(seq.block_table + [-1] * (width - len(seq.block_table)))
        # Token-based mapping handles chunk boundaries inside a KV block.
        for pos in range(request.start, request.end):
            slot = (seq.block_table[pos // seq.block_size] * seq.block_size
                    + pos % seq.block_size) if seq.block_table else -1
            batch.slot_mapping.append(slot)
        if request.should_sample:
            batch.sample_indices.append(index)
    return batch
