"""
Base classes and utilities for all language models.
"""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class BaseConfig:
    """Base configuration for all models."""
    vocab_size: int = 50304
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = True

    # Training-specific
    use_flash_attention: bool = True
    activation: str = 'gelu'  # 'gelu', 'relu', 'swish'


class BaseLanguageModel(nn.Module, ABC):
    """
    Abstract base class for all language models.
    Defines the common interface and shared functionality.
    """

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        self._setup_model()
        self.apply(self._init_weights)

    @abstractmethod
    def _setup_model(self):
        """Setup model architecture - must be implemented by subclasses."""
        pass

    @abstractmethod
    def forward(self, idx, targets=None):
        """Forward pass - must be implemented by subclasses."""
        pass

    def get_num_params(self, non_embedding: bool = True):
        """
        Return the number of parameters in the model.
        For non_embedding count, position embeddings are subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self, 'transformer'):
            # Subtract position embeddings if they exist
            if 'wpe' in self.transformer:
                n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize weights - can be overridden by subclasses."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def configure_optimizers(self, weight_decay: float, learning_rate: float,
                           betas: tuple, device_type: str):
        """
        Configure AdamW optimizer with separate weight decay for different parameter groups.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Separate parameters into decay and no-decay groups
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Use fused AdamW if available (CUDA only)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int,
                temperature: float = 1.0, top_k: Optional[int] = None,
                top_p: Optional[float] = None, eos_token: Optional[int] = None):
        """
        Generate tokens autoregressively.

        Args:
            idx: Input token indices (LongTensor of shape (b, t))
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_k: If set, only sample from top k tokens
            top_p: If set, use nucleus sampling
            eos_token: If set, stop generation when this token is reached

        Returns:
            Generated token indices (LongTensor of shape (b, t + max_new_tokens))
        """
        self.eval()
        generated = idx.clone()

        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = generated
            if generated.size(1) > self.config.block_size:
                idx_cond = generated[:, -self.config.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            # Sample and append
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, idx_next), dim=1)

            # Check for EOS token
            if eos_token is not None and (idx_next == eos_token).all():
                break

        return generated

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float, device: str = 'cuda'):
        """
        Estimate Model FLOPS Utilization (MFU).

        Args:
            fwdbwd_per_iter: Number of forward-backward passes per iteration
            dt: Time per iteration in seconds
            device: 'cuda' or 'cpu'

        Returns:
            MFU as a fraction of peak FLOPS
        """
        N = self.get_num_params()
        L = getattr(self.config, 'n_layer', 1)
        H = getattr(self.config, 'n_head', 1)
        Q = getattr(self.config, 'n_embd', 1) // H
        T = self.config.block_size

        # FLOPS per token (PaLM paper Appendix B)
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # Peak FLOPS for different devices
        if device == 'cuda':
            flops_promised = 312e12  # A100 bfloat16
        else:
            flops_promised = 1e14  # Conservative estimate for CPU

        flops_achieved = flops_per_iter * (1.0 / dt)
        return flops_achieved / flops_promised

    def crop_block_size(self, block_size: int):
        """
        Reduce the block size of the model.
        Useful for loading pretrained models with larger context.
        """
        if block_size > self.config.block_size:
            raise ValueError(f"Cannot increase block size from {self.config.block_size} to {block_size}")

        self.config.block_size = block_size

        if hasattr(self, 'transformer') and 'wpe' in self.transformer:
            self.transformer.wpe.weight = nn.Parameter(
                self.transformer.wpe.weight[:block_size]
            )

        # Update attention masks if present
        for module in self.modules():
            if hasattr(module, 'causal_mask'):
                module.causal_mask = module.causal_mask[:, :, :block_size, :block_size]
            if hasattr(module, 'mask') and hasattr(module.mask, 'shape'):
                # Handle buffer masks
                if len(module.mask.shape) == 4:
                    module.mask = module.mask[:, :, :block_size, :block_size]


class ModelRegistry:
    """Registry for available model architectures."""

    _models: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class."""
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Get a model class by name."""
        if name not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(f"Unknown model: {name}. Available: {available}")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model names."""
        return list(cls._models.keys())


def create_model(model_type: str, config: BaseConfig) -> BaseLanguageModel:
    """
    Factory function to create models by name.

    Args:
        model_type: Name of the model architecture
        config: Configuration object

    Returns:
        Instantiated model
    """
    model_class = ModelRegistry.get(model_type)
    return model_class(config)


def get_model_config(model_type: str, **kwargs) -> BaseConfig:
    """
    Get a configuration object for a specific model type with sensible defaults.

    Common model presets:
    - 'mini': Small model for quick experimentation (~15M params)
    - 'small': Small GPT (~45M params)
    - 'medium': Medium GPT (~125M params, similar to GPT-2)
    - 'large': Large GPT (~350M params, similar to GPT-2 Medium)
    """
    presets = {
        'mini': {
            'n_layer': 6, 'n_head': 6, 'n_embd': 384,
            'block_size': 256, 'vocab_size': 50304
        },
        'small': {
            'n_layer': 8, 'n_head': 8, 'n_embd': 512,
            'block_size': 512, 'vocab_size': 50304
        },
        'medium': {
            'n_layer': 12, 'n_head': 12, 'n_embd': 768,
            'block_size': 1024, 'vocab_size': 50304
        },
        'large': {
            'n_layer': 24, 'n_head': 16, 'n_embd': 1024,
            'block_size': 1024, 'vocab_size': 50304
        },
    }

    if model_type in presets:
        config_dict = {**presets[model_type], **kwargs}
        return BaseConfig(**config_dict)

    return BaseConfig(**kwargs)
