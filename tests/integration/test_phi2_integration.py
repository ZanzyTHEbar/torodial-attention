import os
import torch
import pytest

from scripts.load_phi2 import load_phi2_model, replace_attention_layer, freeze_model_except_attention
from toroidal_attention import ToroidalAttention


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get('RUN_PHI2_INTEGRATION', '0') != '1',
    reason="Requires GPU and RUN_PHI2_INTEGRATION=1"
)
def test_phi2_replace_and_forward_backward():
    model, tokenizer, config = load_phi2_model(device='cpu', torch_dtype=torch.float32)
    toroidal = ToroidalAttention(
        d_model=config.hidden_size,
        n_heads=config.num_attention_heads,
        max_len=128,
        depth=2,
        lambda_distance=0.1,
        fusion_mode='low_rank',
        backend='sdpa',
    )
    replace_attention_layer(model, layer_idx=0, toroidal_attn=toroidal, copy_weights=False)
    freeze_model_except_attention(model, layer_idx=0)

    input_ids = torch.randint(0, config.vocab_size, (1, 64))
    labels = torch.randint(0, config.vocab_size, (1, 64))
    out = model(input_ids=input_ids, labels=labels)
    assert out.loss is not None
    out.loss.backward()


def test_phi2_window_flag_smoke():
    if os.environ.get('RUN_PHI2_INTEGRATION', '0') != '1':
        pytest.skip('Phi-2 integration smoke disabled by default (set RUN_PHI2_INTEGRATION=1)')
    model, tokenizer, config = load_phi2_model(device='cpu', torch_dtype=torch.float32)
    toroidal = ToroidalAttention(
        d_model=config.hidden_size,
        n_heads=config.num_attention_heads,
        max_len=128,
        depth=2,
        lambda_distance=0.0,
        fusion_mode='low_rank',
        backend='sdpa',
        window_size=8,
    )
    replace_attention_layer(model, layer_idx=0, toroidal_attn=toroidal, copy_weights=False)
    input_ids = torch.randint(0, config.vocab_size, (1, 32))
    out = model(input_ids=input_ids)
    assert out.logits.shape == (1, 32, config.vocab_size)


