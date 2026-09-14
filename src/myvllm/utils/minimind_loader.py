"""Strict loader for unquantized MiniMind-3 MoE safetensors checkpoints."""
import json
from pathlib import Path
import re
import torch
from safetensors import safe_open


def expand_expert_weights(weights, config):
    """HF 5.x [E,2I,H]/[E,H,I] -> HF 4.x per-expert Linear weights."""
    result = {}
    def add(name, value):
        if name in result:
            raise ValueError(f'Duplicate checkpoint tensor: {name}')
        result[name] = value
    e, i, h = (config[k] for k in ('num_experts', 'moe_intermediate_size', 'hidden_size'))
    for name, value in weights.items():
        match = re.fullmatch(r'(model\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)(?:\.weight)?', name)
        if not match:
            add(name, value)
            continue
        prefix, kind = match.groups()
        expected = (e, 2 * i, h) if kind == 'gate_up_proj' else (e, h, i)
        if tuple(value.shape) != expected:
            raise ValueError(f'{name}: expected {expected}, got {tuple(value.shape)}')
        for expert in range(e):
            if kind == 'gate_up_proj':
                add(f'{prefix}.{expert}.gate_proj.weight', value[expert, :i])
                add(f'{prefix}.{expert}.up_proj.weight', value[expert, i:])
            else:
                add(f'{prefix}.{expert}.down_proj.weight', value[expert])
    return result


def load_minimind_state_dict(model, weights):
    weights = expand_expert_weights(weights, model.config)
    if model.config['tie_word_embeddings']:
        embed, head = 'model.embed_tokens.weight', 'lm_head.weight'
        if embed in weights and head in weights and not torch.equal(weights[embed], weights[head]):
            raise ValueError('Tied embedding and lm_head have different checkpoint values')
        if embed in weights:
            weights[head] = weights[embed]
        elif head in weights:
            weights[embed] = weights[head]
    expected = model.state_dict()
    missing, unexpected = set(expected) - set(weights), set(weights) - set(expected)
    errors = [f'{name}: expected {tuple(expected[name].shape)}, got {tuple(weights[name].shape)}'
              for name in set(expected) & set(weights) if expected[name].shape != weights[name].shape]
    if missing or unexpected or errors:
        raise ValueError(f'Invalid checkpoint: missing={sorted(missing)}, '
                         f'unexpected={sorted(unexpected)}, shapes={errors}')
    # Validate everything before mutation: never continue with random expert weights.
    model.load_state_dict(weights, strict=True)
    return set(expected)


def load_minimind_checkpoint(model, path):
    root = Path(path).expanduser()
    if not root.is_dir():
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(path, allow_patterns=['*.safetensors', '*.json']))
    index = root / 'model.safetensors.index.json'
    weight_map = None
    if index.exists():
        weight_map = json.loads(index.read_text())['weight_map']
        files = sorted(set(weight_map.values()))
    elif (root / 'model.safetensors').exists():
        files = ['model.safetensors']
    else:
        files = sorted(p.name for p in root.glob('*.safetensors'))
    if not files:
        raise ValueError('Expected safetensors checkpoint; convert raw .pth/.bin first')
    weights = {}
    for filename in files:
        file = (root / filename).resolve()
        if not file.is_relative_to(root.resolve()):
            raise ValueError('Checkpoint shard must be inside checkpoint directory')
        with safe_open(file, framework='pt', device='cpu') as shard:
            for name in shard.keys():
                if name in weights:
                    raise ValueError(f'Duplicate checkpoint tensor: {name}')
                if weight_map is not None and weight_map.get(name) != filename:
                    raise ValueError(f'Incorrect shard index for {name}')
                weights[name] = shard.get_tensor(name)
    if weight_map is not None and set(weight_map) != set(weights):
        raise ValueError('Checkpoint index does not match shard tensors')
    return load_minimind_state_dict(model, weights)
