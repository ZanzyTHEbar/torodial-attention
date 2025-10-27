"""
Multi-layer Phi-2 integration tests.

Tests progressive replacement of multiple layers:
- Single layer (baseline)
- Strategic layers (0, 8, 16, 24)
- Full 32-layer replacement
"""

import os
import pytest
import torch
from scripts.load_phi2 import load_phi2_model, replace_attention_layer, freeze_model_except_attention
from toroidal_attention import ToroidalAttention


@pytest.mark.skipif(
    os.environ.get('RUN_PHI2_MULTILAYER', '0') != '1',
    reason="Requires RUN_PHI2_MULTILAYER=1 (resource intensive)"
)
class TestPhi2MultiLayer:
    """Multi-layer replacement tests."""
    
    @pytest.fixture(scope="class")
    def phi2_model(self):
        """Load Phi-2 once for all tests."""
        model, tokenizer, config = load_phi2_model(device='cpu', torch_dtype=torch.float32)
        return model, tokenizer, config
    
    def test_single_layer_baseline(self, phi2_model):
        """Test single layer replacement (baseline)."""
        model, tokenizer, config = phi2_model
        
        # Create toroidal attention
        toroidal = ToroidalAttention(
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            depth=4,
            max_len=256,
            lambda_distance=0.1,
            fusion_mode='low_rank',
        )
        
        # Replace layer 0 only
        replace_attention_layer(model, layer_idx=0, toroidal_attn=toroidal, copy_weights=False)
        
        # Test forward/backward
        input_ids = torch.randint(0, config.vocab_size, (1, 64))
        labels = torch.randint(0, config.vocab_size, (1, 64))
        
        output = model(input_ids=input_ids, labels=labels)
        assert output.loss is not None
        assert not torch.isnan(output.loss)
        
        output.loss.backward()
        
        # Verify gradients
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0
    
    def test_strategic_4layer_replacement(self, phi2_model):
        """Test strategic 4-layer replacement (0, 8, 16, 24)."""
        model, tokenizer, config = phi2_model
        
        # Replace strategic layers (every 8th layer)
        layer_indices = [0, 8, 16, 24]
        
        for layer_idx in layer_indices:
            toroidal = ToroidalAttention(
                d_model=config.hidden_size,
                n_heads=config.num_attention_heads,
                depth=4,
                max_len=256,
                lambda_distance=0.1,
                fusion_mode='low_rank',
            )
            replace_attention_layer(model, layer_idx=layer_idx, toroidal_attn=toroidal, copy_weights=False)
        
        # Test forward/backward
        input_ids = torch.randint(0, config.vocab_size, (1, 64))
        labels = torch.randint(0, config.vocab_size, (1, 64))
        
        output = model(input_ids=input_ids, labels=labels)
        assert output.loss is not None
        assert not torch.isnan(output.loss)
        
        output.loss.backward()
        
        # Verify gradients for replaced layers
        for layer_idx in layer_indices:
            layer = model.model.layers[layer_idx]
            assert any(p.grad is not None for p in layer.parameters() if p.requires_grad)
    
    @pytest.mark.slow
    def test_full_32layer_replacement(self, phi2_model):
        """Test full 32-layer replacement (all layers)."""
        model, tokenizer, config = phi2_model
        
        # Replace all 32 layers
        for layer_idx in range(config.num_hidden_layers):
            toroidal = ToroidalAttention(
                d_model=config.hidden_size,
                n_heads=config.num_attention_heads,
                depth=4,
                max_len=256,
                lambda_distance=0.1,
                fusion_mode='low_rank',
            )
            replace_attention_layer(model, layer_idx=layer_idx, toroidal_attn=toroidal, copy_weights=False)
        
        # Test forward/backward (smaller batch due to memory)
        input_ids = torch.randint(0, config.vocab_size, (1, 32))
        labels = torch.randint(0, config.vocab_size, (1, 32))
        
        output = model(input_ids=input_ids, labels=labels)
        assert output.loss is not None
        assert not torch.isnan(output.loss)
        
        output.loss.backward()
        
        # Verify gradients exist
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0
    
    def test_depth_sweep_single_layer(self, phi2_model):
        """Test different depth values on single layer."""
        model, tokenizer, config = phi2_model
        
        depths_to_test = [1, 2, 4, 8]
        
        for depth in depths_to_test:
            # Create toroidal attention with specific depth
            toroidal = ToroidalAttention(
                d_model=config.hidden_size,
                n_heads=config.num_attention_heads,
                depth=depth,
                max_len=256,
                lambda_distance=0.1,
            )
            
            # Replace layer 0
            replace_attention_layer(model, layer_idx=0, toroidal_attn=toroidal, copy_weights=False)
            
            # Quick forward test
            input_ids = torch.randint(0, config.vocab_size, (1, 32))
            output = model(input_ids=input_ids)
            
            assert output.logits.shape == (1, 32, config.vocab_size)
            assert not torch.isnan(output.logits).any()
    
    @pytest.mark.gpu
    def test_single_layer_gpu(self, phi2_model, gpu_device):
        """Test single layer replacement on GPU."""
        model, tokenizer, config = phi2_model
        model = model.to(gpu_device)
        
        # Create toroidal attention
        toroidal = ToroidalAttention(
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            depth=2,
            max_len=256,
            lambda_distance=0.1,
        ).to(gpu_device)
        
        # Replace layer 0
        replace_attention_layer(model, layer_idx=0, toroidal_attn=toroidal, copy_weights=False)
        
        # Test forward
        input_ids = torch.randint(0, config.vocab_size, (1, 64), device=gpu_device)
        output = model(input_ids=input_ids)
        
        assert output.logits.device.type == gpu_device.type
        assert not torch.isnan(output.logits).any()


@pytest.mark.skipif(
    os.environ.get('RUN_PHI2_LONGCONTEXT', '0') != '1',
    reason="Requires RUN_PHI2_LONGCONTEXT=1 (memory intensive)"
)
class TestPhi2LongContext:
    """Long-context testing (>2048 tokens)."""
    
    @pytest.fixture(scope="class")
    def phi2_model_long(self):
        """Load Phi-2 for long-context tests."""
        model, tokenizer, config = load_phi2_model(device='cpu', torch_dtype=torch.float32)
        return model, tokenizer, config
    
    @pytest.mark.parametrize("seq_len", [2048, 4096, 8192])
    def test_long_sequence_forward(self, phi2_model_long, seq_len):
        """Test forward pass with long sequences."""
        model, tokenizer, config = phi2_model_long
        
        # Replace layer 0 with toroidal attention
        toroidal = ToroidalAttention(
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            depth=4,
            max_len=seq_len,
            lambda_distance=0.1,
        )
        replace_attention_layer(model, layer_idx=0, toroidal_attn=toroidal, copy_weights=False)
        
        # Test with long sequence
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
        
        # Forward only (no labels to save memory)
        with torch.no_grad():
            output = model(input_ids=input_ids)
        
        assert output.logits.shape == (1, seq_len, config.vocab_size)
        assert not torch.isnan(output.logits).any()
    
    def test_ring_attention_long_context(self, phi2_model_long):
        """Test ring attention for very long contexts."""
        model, tokenizer, config = phi2_model_long
        
        # Create toroidal attention with ring mode
        from toroidal_attention.ring import enable_ring_attention_for_long_context
        
        toroidal = ToroidalAttention(
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            depth=2,
            max_len=16384,
            lambda_distance=0.1,
        )
        
        # Enable ring attention for sequences > 2048
        enable_ring_attention_for_long_context(
            toroidal,
            block_size=512,
            enable_threshold=2048
        )
        
        replace_attention_layer(model, layer_idx=0, toroidal_attn=toroidal, copy_weights=False)
        
        # Test with 4K sequence
        input_ids = torch.randint(0, config.vocab_size, (1, 4096))
        
        with torch.no_grad():
            output = model(input_ids=input_ids)
        
        assert output.logits.shape == (1, 4096, config.vocab_size)
        assert not torch.isnan(output.logits).any()

