"""
Integration Tests

End-to-end tests that validate the full pipeline:
- Data loading → Training → Evaluation
- Model integration with Phi-2
- Checkpoint save/load
- Full forward/backward passes
- Cross-component interactions
"""

import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prepare_data import PeriodicSequenceDataset, SinusoidalDataset, create_dataloaders
from scripts.train_toroidal import MetricsTracker, TrainingConfig
from toroidal_attention import ToroidalAttention


class TestDataPipeline:
    """Test data loading and preprocessing."""

    def test_periodic_dataset_creation(self):
        """Test periodic dataset generates valid data."""
        dataset = PeriodicSequenceDataset(
            vocab_size=1000,
            seq_len=128,
            period=32,
            n_samples=100,
            noise_prob=0.1,
        )

        assert len(dataset) == 100

        # Check sample structure
        sample = dataset[0]
        assert 'input_ids' in sample
        assert 'labels' in sample
        assert sample['input_ids'].shape == (127,)  # seq_len - 1
        assert sample['labels'].shape == (127,)

        # Verify periodicity
        input_ids = sample['input_ids']
        period = 32
        # Check first two periods are similar (allowing for noise)
        if len(input_ids) >= 2 * period:
            first_period = input_ids[:period]
            second_period = input_ids[period:2*period]
            # With 10% noise, should have >80% match
            matches = (first_period == second_period).sum().item()
            assert matches > period * 0.7, f"Periodicity check failed: {matches}/{period}"

    def test_sinusoidal_dataset_creation(self):
        """Test sinusoidal dataset generates valid data."""
        dataset = SinusoidalDataset(
            vocab_size=256,
            seq_len=128,
            n_samples=50,
            n_frequencies=3,
        )

        assert len(dataset) == 50

        sample = dataset[0]
        assert 'input_ids' in sample
        assert 'labels' in sample

        # Tokens should be in vocabulary range
        assert sample['input_ids'].min() >= 0
        assert sample['input_ids'].max() < 256

    def test_dataloader_batching(self):
        """Test dataloader creates proper batches."""
        from transformers import AutoTokenizer

        # Create a dummy tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception as e:
            # If can't load, skip this test
            print(f"Skipping dataloader test (tokenizer unavailable): {e}")
            return

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            dataset_type='periodic',
            tokenizer=tokenizer,
            seq_len=64,
            batch_size=4,
            n_train=20,
            n_val=8,
        )

        # Check batch shapes
        batch = next(iter(train_loader))
        assert batch['input_ids'].shape == (4, 63)  # (batch_size, seq_len-1)
        assert batch['labels'].shape == (4, 63)


class TestModelIntegration:
    """Test model integration and compatibility."""

    def test_toroidal_layer_in_sequential(self):
        """Test ToroidalAttention works in nn.Sequential."""
        model = nn.Sequential(
            nn.Linear(128, 128),
            ToroidalAttention(d_model=128, n_heads=4, depth=2),
        )

        # Note: ToroidalAttention returns tuple (output, attn_weights)
        # This won't work directly in Sequential, but test the module
        x = torch.randn(2, 16, 128)

        # Apply layers separately
        x = model[0](x)
        output, _ = model[1](x)

        assert output.shape == (2, 16, 128)

    def test_multiple_toroidal_layers(self):
        """Test stacking multiple toroidal attention layers."""
        layers = nn.ModuleList([
            ToroidalAttention(d_model=128, n_heads=4, depth=2)
            for _ in range(3)
        ])

        x = torch.randn(2, 16, 128)

        for layer in layers:
            x, _ = layer(x)

        assert x.shape == (2, 16, 128)

    def test_mixed_attention_types(self):
        """Test mixing toroidal with standard attention."""
        toroidal = ToroidalAttention(d_model=128, n_heads=4, depth=2)
        standard = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        x = torch.randn(2, 16, 128)

        # Toroidal attention
        x1, _ = toroidal(x)

        # Standard attention
        x2, _ = standard(x, x, x)

        # Both should produce valid outputs
        assert x1.shape == (2, 16, 128)
        assert x2.shape == (2, 16, 128)


class TestForwardBackwardPass:
    """Test full forward and backward passes."""

    def test_forward_backward_single_batch(self):
        """Test forward and backward pass completes without errors."""
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2)
        x = torch.randn(2, 16, 128, requires_grad=True)

        # Forward
        output, _ = attn(x)

        # Compute loss
        loss = output.mean()

        # Backward
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        for name, param in attn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_gradient_accumulation(self):
        """Test gradient accumulation over multiple batches."""
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2)

        total_loss = 0
        for i in range(3):
            x = torch.randn(2, 16, 128, requires_grad=True)
            output, _ = attn(x)
            loss = output.mean()

            # Accumulate gradients
            loss.backward()
            total_loss += loss.item()

        # Gradients should have accumulated
        for param in attn.parameters():
            if param.requires_grad and param.grad is not None:
                assert param.grad.abs().sum() > 0

    def test_zero_grad_clears_gradients(self):
        """Test that zero_grad properly clears gradients."""
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2)
        x = torch.randn(2, 16, 128, requires_grad=True)

        # First pass
        output, _ = attn(x)
        loss = output.mean()
        loss.backward()

        # Check gradients exist
        grad_sum_before = sum(p.grad.abs().sum().item() for p in attn.parameters() if p.grad is not None)
        assert grad_sum_before > 0

        # Zero gradients
        attn.zero_grad()

        # Check gradients are cleared
        for param in attn.parameters():
            if param.grad is not None:
                assert param.grad.abs().sum().item() == 0


class TestCheckpointSaveLoad:
    """Test model checkpoint saving and loading."""

    def test_save_load_state_dict(self):
        """Test saving and loading model state dict."""
        # Create model
        attn1 = ToroidalAttention(d_model=128, n_heads=4, depth=2)

        # Forward pass to initialize buffers
        x = torch.randn(2, 16, 128)
        output1, _ = attn1(x)

        # Save state dict
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            torch.save(attn1.state_dict(), f.name)
            save_path = f.name

        try:
            # Create new model and load
            attn2 = ToroidalAttention(d_model=128, n_heads=4, depth=2)
            attn2.load_state_dict(torch.load(save_path))

            # Forward pass should produce same output
            output2, _ = attn2(x)

            assert torch.allclose(output1, output2, atol=1e-6)
        finally:
            Path(save_path).unlink()

    def test_save_load_with_optimizer(self):
        """Test saving and loading with optimizer state."""
        attn = ToroidalAttention(d_model=128, n_heads=4, depth=2)
        optimizer = torch.optim.Adam(attn.parameters(), lr=1e-4)

        # Training step
        x = torch.randn(2, 16, 128, requires_grad=True)
        output, _ = attn(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()

        # Save checkpoint
        checkpoint = {
            'model_state_dict': attn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            torch.save(checkpoint, f.name)
            save_path = f.name

        try:
            # Load checkpoint
            checkpoint_loaded = torch.load(save_path)

            attn2 = ToroidalAttention(d_model=128, n_heads=4, depth=2)
            attn2.load_state_dict(checkpoint_loaded['model_state_dict'])

            optimizer2 = torch.optim.Adam(attn2.parameters(), lr=1e-4)
            optimizer2.load_state_dict(checkpoint_loaded['optimizer_state_dict'])

            # Continue training
            output2, _ = attn2(x)
            loss2 = output2.mean()
            loss2.backward()
            optimizer2.step()

            # Should not crash
            assert True
        finally:
            Path(save_path).unlink()


class TestConfigurationManagement:
    """Test configuration saving and loading."""

    def test_training_config_serialization(self):
        """Test TrainingConfig can be saved and loaded."""
        config = TrainingConfig(
            depth=4,
            fusion_mode='low_rank',
            dataset_type='periodic',
            batch_size=8,
            num_epochs=10,
        )

        # Save to JSON
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(config.to_dict(), f)
            save_path = f.name

        try:
            # Load from JSON
            with open(save_path) as f:
                config_dict = json.load(f)

            config2 = TrainingConfig(**config_dict)

            # Check key parameters match
            assert config2.depth == 4
            assert config2.fusion_mode == 'low_rank'
            assert config2.dataset_type == 'periodic'
        finally:
            Path(save_path).unlink()


class TestMetricsTracking:
    """Test metrics tracking and aggregation."""

    def test_metrics_tracker_updates(self):
        """Test MetricsTracker properly tracks metrics."""
        tracker = MetricsTracker()

        # Simulate training
        for i in range(10):
            train_loss = 2.0 - i * 0.1  # Decreasing loss
            grad_norm = 0.5 + i * 0.01
            lr = 1e-4

            tracker.update_train(train_loss, grad_norm, lr)

        # Simulate validation
        for i in range(3):
            val_loss = 1.5 - i * 0.05
            tracker.update_val(val_loss)

        # Check metrics
        assert len(tracker.train_losses) == 10
        assert len(tracker.val_losses) == 3
        assert tracker.best_val_loss < 1.5

    def test_metrics_save_load(self):
        """Test metrics can be saved and loaded."""
        tracker = MetricsTracker()

        # Add some data
        tracker.update_train(2.0, 0.5, 1e-4)
        tracker.update_train(1.8, 0.6, 1e-4)
        tracker.update_val(1.7)

        # Save
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            tracker.save(Path(f.name))
            save_path = f.name

        try:
            # Load
            with open(save_path) as f:
                data = json.load(f)

            # Check data is present
            assert len(data['train_losses']) == 2
            assert len(data['val_losses']) == 1
            assert 'summary' in data
        finally:
            Path(save_path).unlink()


class TestCrossComponentInteraction:
    """Test interactions between different components."""

    def test_pe_with_fusion(self):
        """Test positional encoding works correctly with depth fusion."""
        from toroidal_attention.fusion import DepthFusion
        from toroidal_attention.positional_encoding import Toroidal3DPositionalEncoding

        pe = Toroidal3DPositionalEncoding(d_model=128, max_len=256, depth=4)
        fusion = DepthFusion(depth=4, fusion_mode='low_rank')

        # Generate PE
        sin_emb, cos_emb = pe(seq_len=32)

        # Create dummy tensor matching PE shape
        x = torch.randn(2, 32, 4, 128)

        # Apply fusion
        fused = fusion(x)

        assert fused.shape == (2, 32, 512)  # 4 * 128

    def test_distance_with_attention(self):
        """Test distance metric integrates correctly with attention scores."""
        from toroidal_attention.distance import create_distance_bias

        seq_len, depth = 16, 4
        bias = create_distance_bias(seq_len, depth, lambda_param=0.1)

        # Create attention scores
        B, H = 2, 8
        scores = torch.randn(B, depth, H, seq_len, seq_len)

        # Apply distance bias (simplified - just check dimensions work)
        bias_2d = bias.mean(dim=3).permute(1, 0, 2)  # Average over target depth
        bias_broadcast = bias_2d.unsqueeze(0).unsqueeze(2)

        scores_biased = scores - bias_broadcast

        assert scores_biased.shape == scores.shape


def run_integration_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("Running Integration Tests")
    print("=" * 60)

    test_classes = [
        TestDataPipeline,
        TestModelIntegration,
        TestForwardBackwardPass,
        TestCheckpointSaveLoad,
        TestConfigurationManagement,
        TestMetricsTracking,
        TestCrossComponentInteraction,
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n[{test_class.__name__}]")
        test_instance = test_class()

        # Get all test methods
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]

        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {str(e)}")

    print(f"\n{'='*60}")
    print(f"Integration Tests Complete: {passed_tests}/{total_tests} passed")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_integration_tests()

