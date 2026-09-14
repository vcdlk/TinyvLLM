from collections import deque
from dataclasses import dataclass

from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager


@dataclass(frozen=True)
class ScheduledSequence:
    sequence: Sequence
    start: int
    num_tokens: int

    @property
    def end(self) -> int:
        return self.start + self.num_tokens

    @property
    def should_sample(self) -> bool:
        return self.end == len(self.sequence)


@dataclass
class SchedulerOutput:
    requests: list[ScheduledSequence]

    @property
    def num_scheduled_tokens(self) -> int:
        return sum(request.num_tokens for request in self.requests)

    @property
    def is_decode_only(self) -> bool:
        return bool(self.requests) and all(
            r.num_tokens == 1 and r.start > 0 and r.should_sample
            for r in self.requests
        )


class Scheduler:
    """Running-first token-budget scheduling, with incremental KV allocation.

    Prefill, partial prefill and decode all advance the same computed cursor.
    Running requests retain FCFS order; waiting requests use the remaining budget.
    """

    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int,
                 max_cached_blocks: int, block_size: int, eos: int,
                 long_prefill_token_threshold: int = 0):
        if min(max_num_sequences, max_num_batched_tokens, max_cached_blocks, block_size) <= 0:
            raise ValueError("Scheduler limits and block_size must be positive")
        if long_prefill_token_threshold < 0:
            raise ValueError("long_prefill_token_threshold must be nonnegative")
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = eos

    def is_finished(self):
        return not self.waiting and not self.running

    def add_sequence(self, sequence: Sequence):
        if not len(sequence):
            raise ValueError("Prompt must contain at least one token")
        if sequence.block_size != self.block_manager.block_size:
            raise ValueError("Sequence and scheduler block_size must match")
        if sequence.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if sequence.max_model_length is not None and len(sequence) >= sequence.max_model_length:
            raise ValueError("Prompt must leave room for output within max_model_length")
        self._check_capacity(sequence)
        self.waiting.append(sequence)

    def _check_capacity(self, seq: Sequence):
        if seq.num_blocks > len(self.block_manager.blocks):
            raise ValueError(
                f"Sequence {seq.seq_id} needs {seq.num_blocks} blocks but the KV cache "
                f"only holds {len(self.block_manager.blocks)}. Increase cache capacity "
                "or reduce the prompt/output length."
            )

    def _token_grant(self, seq: Sequence, budget: int) -> int:
        pending = len(seq) - seq.num_computed_tokens
        if self.long_prefill_token_threshold:
            pending = min(pending, self.long_prefill_token_threshold)
        return min(pending, budget)

    def schedule(self) -> SchedulerOutput:
        requests = []
        budget = self.max_num_batched_tokens
        preempted = False
        # Only unscheduled requests can be evicted: blocks referenced by this
        # output must stay allocated until the forward pass has completed.
        index = 0
        while index < len(self.running) and budget > 0:
            seq = self.running[index]
            self._check_capacity(seq)
            n = self._token_grant(seq, budget)
            assert n > 0, "Call postprocess before scheduling the next step"
            while not self.block_manager.can_allocate_slots(seq, n):
                victim = self.running.pop()
                self.preempt(victim)
                preempted = True
                if victim is seq:
                    break
            else:
                self.block_manager.allocate_slots(seq, n)
                requests.append(ScheduledSequence(seq, seq.num_computed_tokens, n))
                budget -= n
                index += 1
                continue
            break

        # Don't immediately readmit an evicted request in the same iteration.
        while (not preempted and self.waiting and budget > 0
               and len(self.running) < self.max_num_sequences):
            seq = self.waiting[0]
            self._check_capacity(seq)
            n = self._token_grant(seq, budget)
            if not self.block_manager.can_allocate_slots(seq, n):
                break
            self.block_manager.allocate_slots(seq, n)
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            requests.append(ScheduledSequence(seq, seq.num_computed_tokens, n))
            budget -= n

        if not requests and not preempted and not self.is_finished():
            raise RuntimeError("Scheduler made no progress; check KV block ownership")
        return SchedulerOutput(requests)

    def preempt(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)

    def postprocess(self, output: SchedulerOutput, token_ids: list[int]) -> None:
        # Only final chunks have logits eligible for sampling. Validate before
        # mutating any request so an incomplete runner response cannot lose work.
        eligible = [r.should_sample for r in output.requests]
        if len(token_ids) != sum(eligible):
            raise ValueError("Expected one sampled token per completed input, not per chunk")
        tokens = iter(token_ids)
        for request, sample in zip(output.requests, eligible):
            seq = request.sequence
            seq.num_computed_tokens = request.end
            if not sample:
                continue
            token_id = next(tokens)
            seq.append_token(token_id)
            if ((not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens >= seq.max_tokens
                or (seq.max_model_length is not None and len(seq) >= seq.max_model_length)):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
