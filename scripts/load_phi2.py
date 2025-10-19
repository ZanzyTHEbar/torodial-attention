"""
Phi-2 Model Loading Utilities

Utilities for loading the Phi-2 model from Hugging Face and preparing it
for toroidal attention integration.

Phi-2 Architecture:
- 2.7B parameters
- Hidden size: 2560
- Attention heads: 32
- Layers: 32
- Context length: 2048
"""

import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from toroidal_attention import ToroidalAttention


def load_phi2_model(
    model_name: str = "microsoft/phi-2",
    device: str = "auto",
    torch_dtype: torch.dtype = torch.float32,
    low_cpu_mem_usage: bool = True
):
    """
    Load Phi-2 model and tokenizer from Hugging Face.

    Args:
        model_name (str): Model identifier on HF Hub
        device (str): Device to load model on ('auto', 'cuda', 'cpu')
        torch_dtype: Data type for model weights
        low_cpu_mem_usage (bool): Use memory-efficient loading

    Returns:
        tuple: (model, tokenizer, config)
    """
    print(f"Loading Phi-2 model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load configuration
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    print("  Model config:")
    print(f"    Hidden size: {config.hidden_size}")
    print(f"    Num attention heads: {config.num_attention_heads}")
    print(f"    Num hidden layers: {config.num_hidden_layers}")
    print(f"    Vocab size: {config.vocab_size}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device,
        low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=True,
    )

    print(f"  Model loaded on device: {device}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, tokenizer, config


def get_phi2_attention_layer(model, layer_idx: int = 0):
    """
    Extract a specific attention layer from Phi-2.

    Args:
        model: Phi-2 model
        layer_idx (int): Index of layer to extract (0-31)

    Returns:
        nn.Module: The attention module from specified layer
    """
    # Phi-2 architecture: model.transformer.h[layer_idx].attn
    # or model.model.layers[layer_idx].self_attn for newer format

    try:
        # Try Phi-2 format
        attention = model.transformer.h[layer_idx].attn
        print(f"Extracted attention from transformer.h[{layer_idx}].attn")
        return attention
    except AttributeError:
        try:
            # Try alternative format
            attention = model.model.layers[layer_idx].self_attn
            print(f"Extracted attention from model.layers[{layer_idx}].self_attn")
            return attention
        except AttributeError:
            raise ValueError(
                "Could not find attention layer. Phi-2 architecture may have changed. "
                "Expected model.transformer.h[i].attn or model.model.layers[i].self_attn"
            )


def replace_attention_layer(
    model,
    layer_idx: int,
    toroidal_attn: ToroidalAttention,
    copy_weights: bool = True
):
    """
    Replace a Phi-2 attention layer with toroidal attention.

    Args:
        model: Phi-2 model
        layer_idx (int): Which layer to replace (0-31)
        toroidal_attn (ToroidalAttention): The toroidal attention module
        copy_weights (bool): Whether to initialize from original weights

    Returns:
        nn.Module: The replaced attention module (for reference)
    """
    print(f"\nReplacing attention in layer {layer_idx} with ToroidalAttention...")

    # Get original attention
    try:
        original_attn = model.transformer.h[layer_idx].attn
        target_path = f"transformer.h[{layer_idx}].attn"
    except AttributeError:
        original_attn = model.model.layers[layer_idx].self_attn
        target_path = f"model.layers[{layer_idx}].self_attn"

    # Optionally copy weights from original attention
    if copy_weights:
        print("  Copying weights from original attention...")
        try:
            # Try to match Q, K, V projections
            # Phi-2 typically has combined QKV projection or separate Q,K,V

            # This is a simplified approach - actual weight copying would need
            # to handle the specific architecture details
            if hasattr(original_attn, 'q_proj'):
                toroidal_attn.W_q.weight.data.copy_(original_attn.q_proj.weight.data)
            if hasattr(original_attn, 'k_proj'):
                toroidal_attn.W_k.weight.data.copy_(original_attn.k_proj.weight.data)
            if hasattr(original_attn, 'v_proj'):
                toroidal_attn.W_v.weight.data.copy_(original_attn.v_proj.weight.data)
            if hasattr(original_attn, 'out_proj') or hasattr(original_attn, 'dense'):
                out_proj = getattr(original_attn, 'out_proj', None) or getattr(original_attn, 'dense', None)
                if out_proj:
                    toroidal_attn.W_o.weight.data.copy_(out_proj.weight.data)

            print("  ✓ Weights copied successfully")
        except Exception as e:
            print(f"  ⚠ Could not copy all weights: {e}")
            print("  Continuing with random initialization...")

    # Replace the attention module
    if 'transformer.h' in target_path:
        model.transformer.h[layer_idx].attn = toroidal_attn
    else:
        model.model.layers[layer_idx].self_attn = toroidal_attn

    print(f"  ✓ Replaced {target_path}")
    print(f"  ToroidalAttention parameters: {sum(p.numel() for p in toroidal_attn.parameters()):,}")

    return original_attn


def freeze_model_except_attention(model, layer_idx: int):
    """
    Freeze all model parameters except the toroidal attention layer.

    This allows fine-tuning only the toroidal attention mechanism
    while keeping the rest of Phi-2 frozen.

    Args:
        model: Phi-2 model with toroidal attention
        layer_idx (int): Index of layer with toroidal attention
    """
    print("\nFreezing model parameters...")

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze toroidal attention
    try:
        attn_module = model.transformer.h[layer_idx].attn
    except AttributeError:
        attn_module = model.model.layers[layer_idx].self_attn

    for param in attn_module.parameters():
        param.requires_grad = True

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"  Frozen parameters: {total-trainable:,}")


def print_model_structure(model, max_depth: int = 3):
    """
    Print the model structure to understand architecture.

    Args:
        model: The model to inspect
        max_depth: Maximum depth to print
    """
    def print_module(module, name='model', depth=0, max_depth=3):
        if depth > max_depth:
            return

        indent = "  " * depth
        print(f"{indent}{name}: {type(module).__name__}")

        if depth < max_depth:
            for child_name, child_module in module.named_children():
                print_module(child_module, child_name, depth+1, max_depth)

    print("\nModel Structure:")
    print("=" * 60)
    print_module(model, max_depth=max_depth)
    print("=" * 60)


def test_model_with_toroidal_attention(
    model,
    tokenizer,
    test_text: str = "The toroidal attention mechanism works by"
):
    """
    Test the model with toroidal attention on a sample text.

    Args:
        model: Phi-2 model with toroidal attention
        tokenizer: Tokenizer
        test_text: Text to generate from
    """
    print("\nTesting model generation...")
    print(f"Input: {test_text}")

    # Tokenize
    inputs = tokenizer(test_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output: {generated_text}")


def main():
    """Demo of loading Phi-2 and replacing with toroidal attention."""
    print("=" * 60)
    print("Phi-2 + Toroidal Attention Integration Demo")
    print("=" * 60)

    # Load Phi-2
    model, tokenizer, config = load_phi2_model(
        device="cpu",  # Use CPU for demo
        torch_dtype=torch.float32,
    )

    # Print structure
    print_model_structure(model, max_depth=2)

    # Create toroidal attention
    toroidal_attn = ToroidalAttention(
        d_model=config.hidden_size,  # 2560
        n_heads=config.num_attention_heads,  # 32
        max_len=2048,
        depth=4,  # 4 platters
        lambda_distance=0.1,
        fusion_mode='low_rank',
    )

    # Replace layer 0
    layer_idx = 0
    replace_attention_layer(
        model,
        layer_idx=layer_idx,
        toroidal_attn=toroidal_attn,
        copy_weights=False,  # Random init for demo
    )

    # Freeze other parameters
    freeze_model_except_attention(model, layer_idx)

    # Test generation
    test_model_with_toroidal_attention(model, tokenizer)

    print("\n✓ Integration complete!")


if __name__ == "__main__":
    main()

