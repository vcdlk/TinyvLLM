import atexit
import torch.distributed as dist
import time
import torch.multiprocessing as mp

from myvllm.engine.sequence import Sequence
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.model_runner import ModelRunner
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer


def worker_process(config, rank, event):
    """Worker process function that initializes ModelRunner and enters loop."""
    # FIRST print before any other code
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    model_runner = ModelRunner(config, rank, event)
    model_runner.loop()


class LLMEngine:
    def __init__(self, config: dict):
        config = dict(config)
        # Canonical limits are shared by scheduler, warmup and CUDA graphs.
        config.setdefault("max_num_batched_tokens", config.get("max_num_batch_tokens", 1024))
        config.setdefault("max_num_sequences", config.get("max_num_seqs", 16))
        for key in ("max_num_batched_tokens", "max_num_sequences", "max_model_length", "block_size"):
            if config[key] <= 0:
                raise ValueError(f"{key} must be positive")
        if config.get("long_prefill_token_threshold", 0) < 0:
            raise ValueError("long_prefill_token_threshold must be nonnegative")
        self.config = config
        world_size = config.get("world_size", 1)
        ctx = mp.get_context("spawn")
        self.processes = []
        self.events = []
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(config, i, event))
            self.events.append(event)
            self.processes.append(process)
            process.start()
        # start the engine only on the master thread with rank = 0
        self.model_runner = ModelRunner(config, rank=0, event=self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"))
        
        # scheduler needs to init after model_runner: when world_size > 1,
        # ModelRunner.__init__ calls dist.init_process_group() which is a
        # collective barrier — rank-0 blocks until all worker ranks have joined.
        # The scheduler should only be created after that rendezvous completes.
        # When world_size == 1 there is no barrier and no real dependency.
        self.scheduler = Scheduler(
            max_num_sequences=config.get("max_num_sequences", 16),
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
            max_cached_blocks=config.get("max_cached_blocks", 1024),
            block_size=config.get("block_size", 256),
            eos=config.get("eos", 50256),
            long_prefill_token_threshold=config.get("long_prefill_token_threshold", 0)
        )

        atexit.register(self.exit)


    def exit(self):
        if not hasattr(self, "model_runner"):
            return
        atexit.unregister(self.exit)
        self.model_runner.call("exit")
        del self.model_runner
        for process in self.processes:
            process.join()

    # Execute the unified token batch, then commit KV progress and sampled outputs.
    def step(self) -> tuple[list[tuple[int, list[int]]], int, bool]:
        batch = self.scheduler.schedule()
        # Compatibility: this flag includes mixed/partial-prefill iterations.
        is_prefill = not batch.is_decode_only
        num_processed_tokens = batch.num_scheduled_tokens
        if not batch.requests:
            return [], 0, is_prefill
        token_ids = self.model_runner.call("run", batch)
        self.scheduler.postprocess(batch, token_ids.cpu().tolist())
        outputs = [(r.sequence.seq_id, r.sequence.completion_token_ids)
                   for r in batch.requests if r.sequence.is_finished]
        return outputs, num_processed_tokens, is_prefill


    # add prompt string to the waiting queue by first transforming it to Sequence object
    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        from dataclasses import replace
        model_limit = self.config['max_model_length']
        params = replace(sampling_params, max_model_length=min(
            sampling_params.max_model_length or model_limit, model_limit))
        self.scheduler.add_sequence(Sequence(token_ids=self.tokenizer.encode(prompt),
                                            block_size=self.config['block_size'], sampling_params=params))

    # given a list of prompts
    # add_prompt for each prompt
    # call step until all sequences are finished
    # return the generated texts
    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        for prompt in prompts:
            self.add_prompt(prompt, sampling_params)
        generated_tokens = {}
        while not self.scheduler.is_finished():
            start_t = time.time()
            outputs, num_processed_tokens, is_prefill = self.step()
            end_t = time.time()
            running_time = end_t - start_t + 1e-10
            if is_prefill:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during prefill/mixed execution")
            else:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during decoding")
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})

        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        output = {'text': [self.tokenizer.decode(tokens) for tokens in generated_tokens], 'token_ids': generated_tokens}
        return output
