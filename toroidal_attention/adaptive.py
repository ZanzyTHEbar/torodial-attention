"""
Adaptive Depth Learning for Toroidal Attention.

Instead of fixing depth D, learn optimal depth per layer:
- Gumbel-Softmax for differentiable depth selection
- Soft mixture of multiple depths
- Depth routing based on input statistics

Key idea: Different layers may need different depth granularity
- Early layers: shallow depth (broad patterns)
- Middle layers: deep stacking (fine-grained)
- Late layers: adaptive based on task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class AdaptiveDepthSelector(nn.Module):
    """
    Learnable depth selection module.
    
    Uses Gumbel-Softmax to select depth from candidate set:
    - Differentiable during training
    - Discrete selection during inference
    
    Args:
        d_model: Model dimension
        candidate_depths: List of candidate depth values [1, 2, 4, 8, ...]
        temperature: Gumbel-Softmax temperature
        hard: Use hard selection (straight-through estimator)
    """
    
    def __init__(
        self,
        d_model: int,
        candidate_depths: List[int] = [1, 2, 4, 8],
        temperature: float = 1.0,
        hard: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.candidate_depths = candidate_depths
        self.n_candidates = len(candidate_depths)
        self.temperature = temperature
        self.hard = hard
        
        # Learnable depth logits (initialized uniformly)
        self.depth_logits = nn.Parameter(torch.zeros(self.n_candidates))
        
        # Optional input-dependent routing
        self.use_input_routing = False
        self.router = None
    
    def enable_input_routing(self):
        """Enable input-dependent depth routing."""
        self.use_input_routing = True
        # Small MLP to compute depth logits from input statistics
        self.router = nn.Sequential(
            nn.Linear(4, 32),  # 4 input features: mean, std, min, max
            nn.ReLU(),
            nn.Linear(32, self.n_candidates),
        )
    
    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        return_depth_probs: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Select depth via Gumbel-Softmax.
        
        Args:
            x: Optional input for input-dependent routing (B, N, d_model)
            return_depth_probs: Whether to return depth probabilities
            
        Returns:
            depth_weights: Soft weights for each candidate depth (n_candidates,)
            depth_probs: Optional probability distribution over depths
        """
        if self.use_input_routing and x is not None:
            # Compute input statistics
            x_stats = torch.stack([
                x.mean(dim=(1, 2)),  # mean
                x.std(dim=(1, 2)),   # std
                x.amin(dim=(1, 2)),  # min
                x.amax(dim=(1, 2)),  # max
            ], dim=-1)  # (B, 4)
            
            # Compute depth logits from input
            logits = self.router(x_stats).mean(dim=0)  # Average over batch
        else:
            # Use learned static logits
            logits = self.depth_logits
        
        # Gumbel-Softmax sampling
        if self.training:
            depth_weights = F.gumbel_softmax(logits, tau=self.temperature, hard=self.hard, dim=0)
        else:
            # Inference: use hard selection (argmax)
            depth_weights = torch.zeros_like(logits)
            depth_weights[logits.argmax()] = 1.0
        
        depth_probs = F.softmax(logits, dim=0) if return_depth_probs else None
        
        return depth_weights, depth_probs


class AdaptiveDepthToroidalAttention(nn.Module):
    """
    Toroidal attention with learnable adaptive depth.
    
    Maintains separate attention modules for each candidate depth,
    then combines outputs using learned weights.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        candidate_depths: List of depth options [1, 2, 4, 8]
        max_len: Maximum sequence length
        lambda_distance: Distance bias strength
        temperature: Gumbel-Softmax temperature
        use_input_routing: Enable input-dependent depth selection
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        candidate_depths: List[int] = [1, 2, 4, 8],
        max_len: int = 2048,
        lambda_distance: float = 0.1,
        temperature: float = 1.0,
        use_input_routing: bool = False,
        use_orthogonal_pe: bool = True,
    ):
        super().__init__()
        
        from toroidal_attention import ToroidalAttention
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.candidate_depths = candidate_depths
        
        # Depth selector
        self.depth_selector = AdaptiveDepthSelector(
            d_model=d_model,
            candidate_depths=candidate_depths,
            temperature=temperature,
            hard=False,  # Use soft mixture during training
        )
        
        if use_input_routing:
            self.depth_selector.enable_input_routing()
        
        # Create separate attention module for each candidate depth
        self.attention_modules = nn.ModuleList([
            ToroidalAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_len=max_len,
                depth=d,
                lambda_distance=lambda_distance,
                fusion_mode='low_rank',
                fusion_rank=max(1, d // 4),
                use_orthogonal_pe=use_orthogonal_pe,
            )
            for d in candidate_depths
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_depth_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        """
        Forward pass with adaptive depth selection.
        
        Args:
            x: Input tensor (B, N, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            return_depth_info: Whether to return depth selection info
            
        Returns:
            output: Output tensor (B, N, d_model)
            attn_weights: Optional attention weights
            depth_info: Optional dict with depth selection statistics
        """
        # Select depth weights
        depth_weights, depth_probs = self.depth_selector(x, return_depth_probs=True)
        
        # Compute output for each candidate depth
        outputs = []
        attn_weights_list = []
        
        for i, attn_module in enumerate(self.attention_modules):
            out, attn_w = attn_module(x, mask=mask, return_attention=return_attention)
            outputs.append(out)
            if return_attention and attn_w is not None:
                attn_weights_list.append(attn_w)
        
        # Weighted combination of outputs
        output = sum(w * o for w, o in zip(depth_weights, outputs))
        
        # Weighted combination of attention weights
        attn_weights = None
        if return_attention and attn_weights_list:
            attn_weights = sum(w * a for w, a in zip(depth_weights, attn_weights_list))
        
        # Depth selection info
        depth_info = None
        if return_depth_info:
            selected_depth_idx = depth_weights.argmax().item()
            depth_info = {
                'depth_weights': depth_weights.detach().cpu().tolist(),
                'depth_probs': depth_probs.detach().cpu().tolist() if depth_probs is not None else None,
                'selected_depth': self.candidate_depths[selected_depth_idx],
                'candidate_depths': self.candidate_depths,
            }
        
        return output, attn_weights, depth_info
    
    def get_selected_depth(self) -> int:
        """Get currently selected depth (for inference)."""
        with torch.no_grad():
            depth_weights, _ = self.depth_selector()
            idx = depth_weights.argmax().item()
            return self.candidate_depths[idx]
    
    def set_temperature(self, temperature: float):
        """Adjust Gumbel-Softmax temperature (for annealing)."""
        self.depth_selector.temperature = temperature


class DepthAnnealer:
    """
    Temperature annealing schedule for Gumbel-Softmax depth selection.
    
    Start with high temperature (soft selection) and gradually anneal to
    low temperature (hard selection) during training.
    
    Args:
        initial_temp: Starting temperature
        final_temp: Final temperature
        anneal_steps: Number of steps to anneal over
        anneal_strategy: 'linear', 'exponential', or 'cosine'
    """
    
    def __init__(
        self,
        initial_temp: float = 5.0,
        final_temp: float = 0.5,
        anneal_steps: int = 10000,
        anneal_strategy: str = 'exponential',
    ):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.anneal_steps = anneal_steps
        self.anneal_strategy = anneal_strategy
        self.current_step = 0
    
    def step(self) -> float:
        """Compute current temperature and increment step."""
        if self.current_step >= self.anneal_steps:
            return self.final_temp
        
        progress = self.current_step / self.anneal_steps
        
        if self.anneal_strategy == 'linear':
            temp = self.initial_temp + (self.final_temp - self.initial_temp) * progress
        elif self.anneal_strategy == 'exponential':
            temp = self.initial_temp * (self.final_temp / self.initial_temp) ** progress
        elif self.anneal_strategy == 'cosine':
            temp = self.final_temp + 0.5 * (self.initial_temp - self.final_temp) * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            temp = temp.item()
        else:
            temp = self.initial_temp
        
        self.current_step += 1
        return temp


# Example integration with training
def train_with_adaptive_depth(
    model: AdaptiveDepthToroidalAttention,
    dataloader,
    optimizer,
    epochs: int = 10,
    log_interval: int = 100,
):
    """
    Example training loop with adaptive depth and temperature annealing.
    
    Args:
        model: AdaptiveDepthToroidalAttention model
        dataloader: Training data loader
        optimizer: Optimizer
        epochs: Number of training epochs
        log_interval: Logging frequency
    """
    annealer = DepthAnnealer(
        initial_temp=5.0,
        final_temp=0.5,
        anneal_steps=len(dataloader) * epochs,
        anneal_strategy='exponential',
    )
    
    model.train()
    
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(dataloader):
            # Anneal temperature
            temp = annealer.step()
            model.set_temperature(temp)
            
            # Forward pass
            output, _, depth_info = model(x, return_depth_info=True)
            
            # Compute loss (task-specific)
            loss = F.mse_loss(output, y)  # Example loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Temperature: {temp:.4f}")
                print(f"  Selected depth: {depth_info['selected_depth']}")
                print(f"  Depth weights: {[f'{w:.3f}' for w in depth_info['depth_weights']]}")


# Utility: Add adaptive depth to existing model
def convert_to_adaptive_depth(
    model: nn.Module,
    layer_idx: int,
    candidate_depths: List[int] = [1, 2, 4, 8],
    use_input_routing: bool = False,
) -> nn.Module:
    """
    Replace a standard ToroidalAttention layer with adaptive depth version.
    
    Args:
        model: Model containing ToroidalAttention layers
        layer_idx: Index of layer to replace
        candidate_depths: Depth options
        use_input_routing: Enable input-dependent routing
        
    Returns:
        Modified model
    """
    # This is a placeholder - actual implementation depends on model structure
    # For Phi-2: model.layers[layer_idx].self_attn = AdaptiveDepthToroidalAttention(...)
    
    print(f"Converting layer {layer_idx} to adaptive depth with candidates {candidate_depths}")
    
    return model

