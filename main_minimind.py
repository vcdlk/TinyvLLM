"""Linux CUDA engine example; use scripts/check_minimind.py for CPU validation."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def main():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('TinyvLLM engine requires Linux CUDA. On Mac run scripts/check_minimind.py --device cpu.')
    from transformers import AutoTokenizer
    from myvllm.engine.llm_engine import LLMEngine
    from myvllm.sampling_parameters import SamplingParams
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='jingyaogong/minimind-3-moe')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [tokenizer.apply_chat_template(
        [{'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True
    ) for text in ['请介绍一下自己。', '为什么天空是蓝色的？']]
    llm = LLMEngine({'model_name_or_path': args.model, 'world_size': 1,
                     'enforce_eager': True, 'max_model_length': 1024,
                     'eos': tokenizer.eos_token_id})
    print(llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=64, max_model_length=1024)))


if __name__ == '__main__':
    main()
