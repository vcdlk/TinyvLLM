from collections import deque

from myvllm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0


class BlockManager:
    """Reserve private KV blocks only through the end of the current chunk.

    Prefix sharing is deliberately disabled: allocated slots are not necessarily
    computed KV, and partial blocks cannot be shared without copy-on-write.
    """

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def _num_new_blocks(self, seq: Sequence, num_tokens: int) -> int:
        end = seq.num_computed_tokens + num_tokens
        if not 0 < num_tokens or end > len(seq):
            raise ValueError("KV allocation must cover a nonempty range of known tokens")
        return max(0, (end + self.block_size - 1) // self.block_size - len(seq.block_table))

    def can_allocate_slots(self, seq: Sequence, num_tokens: int) -> bool:
        return self._num_new_blocks(seq, num_tokens) <= len(self.free_block_ids)

    def allocate_slots(self, seq: Sequence, num_tokens: int) -> None:
        n = self._num_new_blocks(seq, num_tokens)
        if n > len(self.free_block_ids):
            raise ValueError("Insufficient KV cache blocks")
        for _ in range(n):
            block_id = self.free_block_ids.popleft()
            block = self.blocks[block_id]
            assert block.ref_count == 0
            block.ref_count = 1
            self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence) -> None:
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            assert block.ref_count == 1
            block.ref_count = 0
            self.used_block_ids.remove(block_id)
            self.free_block_ids.append(block_id)
        seq.block_table = []
        seq.num_computed_tokens = 0
