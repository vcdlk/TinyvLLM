"""Read model metadata without executing repository Python code."""
import json
import math
from pathlib import Path


def read_model_config(path):
    local = Path(path).expanduser()
    if local.is_dir():
        filename = local / 'config.json'
    else:
        from huggingface_hub import hf_hub_download
        filename = hf_hub_download(path, 'config.json')
    with open(filename) as f:
        return json.load(f)


def validate_minimind_config(raw):
    config = dict(raw)
    model_type = config.get('model_type')
    if model_type not in ('qwen3_moe', 'minimind'):
        raise ValueError(f'Unsupported MoE model_type: {model_type}')
    if model_type == 'minimind' and not config.get('use_moe', False):
        raise ValueError('Only MiniMind-3 use_moe=True is supported')
    defaults = dict(hidden_size=768, num_hidden_layers=8, vocab_size=6400,
                    num_attention_heads=8, num_key_value_heads=4, num_experts=4,
                    num_experts_per_tok=1, norm_topk_prob=True, rms_norm_eps=1e-6,
                    rope_theta=1e6, max_position_embeddings=32768, tie_word_embeddings=True)
    for name, value in defaults.items():
        config.setdefault(name, value)
    config.setdefault('head_dim', config['hidden_size'] // config['num_attention_heads'])
    config.setdefault('intermediate_size', math.ceil(config['hidden_size'] * math.pi / 64) * 64)
    config.setdefault('moe_intermediate_size', config['intermediate_size'])
    for name in (*defaults.keys(), 'head_dim', 'moe_intermediate_size'):
        if name not in ('norm_topk_prob', 'tie_word_embeddings') and config[name] <= 0:
            raise ValueError(f'{name} must be positive')
    if not 1 <= config['num_experts_per_tok'] <= config['num_experts']:
        raise ValueError('num_experts_per_tok must be between 1 and num_experts')
    if config['num_attention_heads'] % config['num_key_value_heads'] or config['head_dim'] % 2:
        raise ValueError('Requires divisible GQA heads and even head_dim')
    if (config.get('hidden_act', 'silu') != 'silu' or config.get('attention_bias', False)
            or config.get('rope_scaling') or config.get('inference_rope_scaling', False)
            or config.get('use_sliding_window', False) or config.get('sliding_window')
            or config.get('mlp_only_layers') or config.get('decoder_sparse_step', 1) != 1
            or config.get('quantization_config') or config.get('n_shared_experts', 0)
            or config.get('num_shared_experts', 0)):
        raise ValueError('Initial MiniMind-3 support requires unquantized all-MoE SwiGLU, '
                         'no shared expert, no bias, standard RoPE and full attention')
    if config.get('rope_parameters'):
        raise ValueError('rope_parameters is not supported; use standard rope_theta config')
    return config


def prepare_model_config(config):
    """Resolve once, before workers launch. Preserve existing dense entry points."""
    config = dict(config)
    raw = read_model_config(config['model_name_or_path'])
    model_type = raw.get('model_type')
    if model_type in ('qwen3_moe', 'minimind'):
        raw = validate_minimind_config(raw)
        if config.get('world_size', 1) != 1:
            raise ValueError('MiniMind MoE currently supports world_size=1 only')
        config['model_architecture'] = 'MiniMindMoeForCausalLM'
        config['hf_config'] = raw
        for target, source in dict(vocab_size='vocab_size', hidden_size='hidden_size',
                                   num_layers='num_hidden_layers', num_heads='num_attention_heads',
                                   num_kv_heads='num_key_value_heads', head_dim='head_dim').items():
            config[target] = raw[source]
        config['enforce_eager'] = True  # Dynamic expert dispatch cannot be CUDA captured.
        config.setdefault('world_size', 1)
        config.setdefault('block_size', 256)
        config.setdefault('max_model_length', min(2048, raw['max_position_embeddings']))
        config.setdefault('max_num_sequences', 16)
        config.setdefault('max_num_batched_tokens', 4096)
        config.setdefault('max_num_batch_tokens', config['max_num_batched_tokens'])
        config.setdefault('gpu_memory_utilization', 0.85)
        config.setdefault('eos', raw.get('eos_token_id') or 2)
        if not 0 < config['max_model_length'] <= raw['max_position_embeddings']:
            raise ValueError('max_model_length must fit max_position_embeddings')
        if config['max_num_batch_tokens'] < config['max_model_length']:
            raise ValueError('Warmup token budget must cover max_model_length')
        if config['max_num_batched_tokens'] < config['max_model_length']:
            raise ValueError('Scheduler token budget must cover max_model_length')
        dtype = config.get('dtype', raw.get('dtype', raw.get('torch_dtype', 'float16')))
        if dtype not in ('float16', 'bfloat16'):
            raise ValueError('CUDA MiniMind inference requires float16 or bfloat16')
        config['dtype'] = dtype
    return config
