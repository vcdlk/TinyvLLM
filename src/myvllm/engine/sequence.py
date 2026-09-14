from enum import Enum, auto
import math
from itertools import count 
from myvllm.sampling_parameters import SamplingParams
from copy import copy


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()

    def __init__(self, token_ids: list[int], block_size: int, sampling_params = SamplingParams()):
        self.block_size = block_size # number of tokens per block
        # record sequence id
        self.seq_id = next(Sequence.counter)
        # status
        self.status = SequenceStatus.WAITING
        # token ids, need copy so that it is a new list, won't be affected by outside changes
        self.token_ids = copy(token_ids)
        # last token
        self.last_token = self.token_ids[-1] if self.token_ids else None
        # num_tokens, num_prompt_tokens
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        # Number of tokens whose KV has actually been computed (not reserved).
        self.num_computed_tokens = 0
        # block_table
        self.block_table = []
        # sampling_params' related things
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.max_model_length = sampling_params.max_model_length

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, idx):
        return self.token_ids[idx]

    @property
    def num_cached_tokens(self):
        """Compatibility name for the computed prefix length."""
        return self.num_computed_tokens

    @num_cached_tokens.setter
    def num_cached_tokens(self, value):
        self.num_computed_tokens = value

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return int(math.ceil(self.num_cached_tokens / self.block_size))

    @property
    def num_blocks(self):
        return int(math.ceil(self.num_tokens / self.block_size))

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - max(self.num_blocks - 1, 0) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks, f"Block index {i} out of range [0, {self.num_blocks})"
        if i == self.num_blocks - 1:
            return self.token_ids[-self.last_block_num_tokens:]
        else:
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            return self.token_ids[start_idx : end_idx]

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1 

    def __getstate__(self):
        # A preempted request must replay prompt AND previously generated tokens.
        # Sending only last_token on decode loses that history on TP workers.
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
