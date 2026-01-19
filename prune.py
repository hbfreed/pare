import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "allenai/Olmo-3-7B-Instruct"
DEVICE = "cuda:1"
importance_scores = load_file(
    "importance_scores_tensors/olmo_3_7b_importance_scores.safetensors"
)

HIDDEN_SIZE = 4096
NUM_HEADS = 32
HEAD_DIM = 128
NUM_LAYERS = 32
# Olmo uses SWA, want to keep the full attention layers
FULL_ATTENTION_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}


def get_mlp_keep_indices(mlp_width):
    mlp_importance = importance_scores["mlp"]
    keep_indices = {}
    for layer_idx in range(mlp_importance.shape[0]):
        layer_scores = mlp_importance[layer_idx]
        top_k = layer_scores.argsort(descending=True)[:mlp_width]
        keep_indices[layer_idx] = top_k.sort().values
    return keep_indices


def get_attention_keep_indices(num_heads_to_keep):
    if num_heads_to_keep == NUM_HEADS:
        return []
    attn_importance = importance_scores["attention"]
    keep_indices = {}
    for layer_idx in range(attn_importance.shape[0]):
        layer_scores = attn_importance[layer_idx]
        top_k = layer_scores.argsort(descending=True)[:num_heads_to_keep]
        keep_indices[layer_idx] = top_k.sort().values
    return keep_indices


def get_layers_to_drop(num_layers_to_keep, protect_full_attention=True):
    """Returns list of layer indices to drop"""
    if num_layers_to_keep == NUM_LAYERS:
        return []

    # Layer scores are flattened as "layer.layer_N" in safetensors
    layer_importance = {k: v for k, v in importance_scores.items() if k.startswith("layer.")}
    scores = [(int(k.split("_")[1]), v) for k, v in layer_importance.items()]
    scores.sort(key=lambda x: x[1], reverse=True)  # highest first

    num_to_drop = NUM_LAYERS - num_layers_to_keep

    if protect_full_attention:
        # Only consider SWA layers for dropping
        swa_scores = [
            (idx, score) for idx, score in scores if idx not in FULL_ATTENTION_LAYERS
        ]
        swa_scores.sort(key=lambda x: x[1])  # lowest first = drop candidates
        drop = [idx for idx, _ in swa_scores[:num_to_drop]]
    else:
        # Drop lowest scoring regardless of type
        scores.sort(key=lambda x: x[1])  # lowest first
        drop = [idx for idx, _ in scores[:num_to_drop]]

    print(f"Dropping layers: {sorted(drop)}")
    if protect_full_attention:
        print(f"(Protected full attention layers: {FULL_ATTENTION_LAYERS})")

    return sorted(drop)


def apply_depth_pruning(model, num_layers_to_keep, protect_full_attention=True):
    layers_to_drop = get_layers_to_drop(num_layers_to_keep, protect_full_attention)

    if not layers_to_drop:
        return model, []

    keep_layers = [
        layer for i, layer in enumerate(model.model.layers) if i not in layers_to_drop
    ]
    model.model.layers = nn.ModuleList(keep_layers)

    # Update layer_idx for each remaining layer
    for new_idx, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = new_idx

    model.config.num_hidden_layers = num_layers_to_keep

    # Update layer_types in config if it exists
    if hasattr(model.config, "layer_types") and model.config.layer_types:
        model.config.layer_types = [
            t for i, t in enumerate(model.config.layer_types) if i not in layers_to_drop
        ]

    return model, layers_to_drop


def apply_mlp_pruning(model, mlp_width, layers_to_drop):
    keep_indices = get_mlp_keep_indices(mlp_width)
    original_indices = [i for i in range(NUM_LAYERS) if i not in layers_to_drop]
    for new_idx, layer in enumerate(model.model.layers):
        orig_idx = original_indices[new_idx]
        idx = keep_indices[orig_idx].to(layer.mlp.gate_proj.weight.device)
        layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[idx, :]
        layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[idx, :]
        layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[:, idx]

    model.config.intermediate_size = mlp_width
    return model


def apply_attention_pruning(model, num_heads_to_keep, layers_to_drop):
    if num_heads_to_keep == NUM_HEADS:
        return model
    keep_indices = get_attention_keep_indices(num_heads_to_keep)
    original_indices = [i for i in range(NUM_LAYERS) if i not in layers_to_drop]

    for new_idx, layer in enumerate(model.model.layers):
        orig_idx = original_indices[new_idx]
        idx = keep_indices[orig_idx].to(layer.self_attn.q_proj.weight.device)

        head_indices = idx.unsqueeze(1) * HEAD_DIM + torch.arange(
            HEAD_DIM, device=idx.device
        )
        flat_indices = head_indices.flatten()

        layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[
            flat_indices, :
        ]
        layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[
            flat_indices, :
        ]
        layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[
            flat_indices, :
        ]
        layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data[
            :, flat_indices
        ]
        layer.self_attn.q_norm.weight.data = layer.self_attn.q_norm.weight.data[
            flat_indices
        ]
        layer.self_attn.k_norm.weight.data = layer.self_attn.k_norm.weight.data[
            flat_indices
        ]

    model.config.num_attention_heads = num_heads_to_keep
    model.config.num_key_value_heads = num_heads_to_keep
    model.config.head_dim = HEAD_DIM
    return model


def prune_model(mlp_width, num_heads, num_layers, protect_full_attention=True):
    print(f"Pruning to: MLP={mlp_width}, heads={num_heads}, layers={num_layers}")

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)

    # Depth first so we know which layers remain
    model, layers_to_drop = apply_depth_pruning(
        model, num_layers, protect_full_attention
    )

    # Width pruning with correct layer mapping
    model = apply_mlp_pruning(model, mlp_width, layers_to_drop)
    model = apply_attention_pruning(model, num_heads, layers_to_drop)

    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e9


if __name__ == "__main__":
    config = (5120, 16, 29)
    model = prune_model(*config)  # mlp_width, num_heads, num_layers
    model = model.to(DEVICE)

    print(f"Parameters: {count_params(model):.2f}B")

    # Sanity check
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = model(**inputs)
    print(f"Forward pass works! Logits shape: {outputs.logits.shape}")

    message = [{"role": "user", "content": "What are the main ingredients in dashi?"}]
    inputs = tokenizer.apply_chat_template(
        message, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        response = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    print(
        f"Generation Check:\n {tokenizer.decode(response[0], skip_special_tokens=True)}"
    )

    # Save
    save_path = f"pruned_models/pruned_olmo3_{config[0]}_{config[1]}_{config[2]}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved to {save_path}")
