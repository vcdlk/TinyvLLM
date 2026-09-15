import sys, os
from pathlib import Path

from transformers import AutoTokenizer

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.engine.llm_engine import LLMEngine as LLM
from myvllm.sampling_parameters import SamplingParams

config = {
    'model_name_or_path': 'jingyaogong/minimind-3-moe',
    'world_size': 1,
    'enforce_eager': True,
    'max_model_length': 1024,
    'max_num_sequences': 16,
    'max_num_batched_tokens': 1024,
    'max_cached_blocks': 1024,
    'block_size': 256,
    'gpu_memory_utilization': 0.9,
}

def main():
    model_name = config.get('model_name_or_path', 'jingyaogong/minimind-3-moe')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config['eos'] = tokenizer.eos_token_id
    llm = LLM(config=config)

    # max_tokens is the max number of generated tokens
    # max_model_length is the max total length including prompt
    # both should be set in SamplingParams and help to determine when to stop generation
    sampling_params = SamplingParams(temperature=0.6, max_tokens=64, max_model_length=1024)
    prompts = [
        "请介绍一下自己。",
        "为什么天空是蓝色的？",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    # outputs is a dict with 'text' and 'token_ids' keys
    generated_texts = outputs['text']

    for prompt, output in zip(prompts, generated_texts):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output}")


if __name__ == "__main__":
    main()
