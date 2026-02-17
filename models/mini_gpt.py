"""
Mini GPT: A simplified GPT model for quick experimentation.
Lighter weight than the full GPT implementation while keeping the core architecture.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

from .base import BaseLanguageModel, BaseConfig, ModelRegistry
from .common import TransformerBlock, Embeddings, LayerNorm


@dataclass
class MiniGPTConfig(BaseConfig):
    """Configuration for MiniGPT - smaller defaults for quick experimentation"""
    block_size: int = 256      # Reduced from 1024
    vocab_size: int = 50304    # GPT-2 vocab_size padded to multiple of 64
    n_layer: int = 6           # Reduced from 12
    n_head: int = 6            # Reduced from 12
    n_embd: int = 384          # Reduced from 768
    dropout: float = 0.1
    bias: bool = True
    use_flash_attention: bool = False  # Disable flash attention for simplicity


@ModelRegistry.register('mini-gpt')
class MiniGPT(BaseLanguageModel):
    """
    Mini GPT - A lightweight GPT model for quick experimentation.
    ~15M parameters with default config (vs ~124M for GPT-2).

    Uses shared components from common.py for better maintainability.
    """

    def _setup_model(self):
        """Setup the MiniGPT model architecture."""
        self.transformer = nn.ModuleDict({
            'embeddings': Embeddings(self.config),
            'h': nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.n_layer)]),
            'ln_f': LayerNorm(self.config.n_embd, bias=self.config.bias),
        })
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        # Weight tying - share embedding and lm_head weights
        self.transformer.embeddings.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        """
        Forward pass.

        Args:
            idx: Input token indices of shape (B, T)
            targets: Target token indices of shape (B, T). If None, returns last token logits only.

        Returns:
            logits: Shape (B, T, vocab_size) or (B, 1, vocab_size) if targets is None
            loss: Cross-entropy loss if targets is provided, None otherwise
        """
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"

        # Get embeddings
        x = self.transformer.embeddings(idx)

        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm
        x = self.transformer.ln_f(x)

        # Compute logits and loss
        if targets is not None:
            # Training mode: compute logits for all positions and calculate loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # Inference mode: only compute logits for the last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str, override_args: dict = None):
        """
        Load a pretrained GPT-2 model from HuggingFace and convert to MiniGPT architecture.

        Args:
            model_type: One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
            override_args: Optional dict of arguments to override (e.g., {'dropout': 0.1})

        Returns:
            MiniGPT model with pretrained weights
        """
        from transformers import GPT2LMHeadModel

        override_args = override_args or {}

        # GPT-2 model configurations
        config_args = {
            'gpt2':         {'n_layer': 12, 'n_head': 12, 'n_embd': 768},   # 124M params
            'gpt2-medium':  {'n_layer': 24, 'n_head': 16, 'n_embd': 1024},  # 350M params
            'gpt2-large':   {'n_layer': 36, 'n_head': 20, 'n_embd': 1280},  # 774M params
            'gpt2-xl':      {'n_layer': 48, 'n_head': 25, 'n_embd': 1600},  # 1558M params
        }

        if model_type not in config_args:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Get config and add fixed GPT-2 parameters
        config_dict = config_args[model_type]
        config_dict['vocab_size'] = 50257
        config_dict['block_size'] = 1024
        config_dict['bias'] = True
        config_dict.update(override_args)

        # Create model
        config = MiniGPTConfig(**config_dict)
        model = cls(config)

        # Load weights from HuggingFace
        print(f"Loading weights from pretrained GPT-2: {model_type}")
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd = model.state_dict()

        # Map HuggingFace state dict keys to our keys
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                     'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Copy weights, handling transposed linear layers
        for k in sd.keys():
            # Skip non-parameter buffers
            if not k.startswith('transformer.h.') and k not in ['transformer.embeddings.wte.weight',
                                                                'transformer.embeddings.wpe.weight',
                                                                'lm_head.weight', 'transformer.ln_f.weight',
                                                                'transformer.ln_f.bias']:
                continue

            # Convert our key naming to HF naming
            hf_key = k
            if k.startswith('transformer.embeddings.'):
                hf_key = k.replace('transformer.embeddings.', 'transformer.')
            elif k == 'lm_head.weight':
                hf_key = 'lm_head.weight'
            elif k.startswith('transformer.h.'):
                # transformer.h.{i}.xxx -> transformer.h.{i}.xxx
                hf_key = k
            elif k.startswith('transformer.ln_f'):
                hf_key = k

            # Handle transposed weights
            if any(k.endswith(w) for w in transposed):
                if hf_key in sd_hf:
                    assert sd_hf[hf_key].shape[::-1] == sd[k].shape, \
                        f"Shape mismatch for {k}: {sd_hf[hf_key].shape[::-1]} vs {sd[k].shape}"
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[hf_key].t())
            else:
                if hf_key in sd_hf:
                    assert sd_hf[hf_key].shape == sd[k].shape, \
                        f"Shape mismatch for {k}: {sd_hf[hf_key].shape} vs {sd[k].shape}"
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[hf_key])

        return model


# Convenience functions
def create_mini_gpt(**kwargs):
    """Create a MiniGPT model with optional config overrides."""
    config = MiniGPTConfig(**kwargs)
    model = MiniGPT(config)
    return model


def mini_gpt_from_pretrained(model_type: str, **kwargs):
    """Create a MiniGPT model from pretrained GPT-2 weights."""
    return MiniGPT.from_pretrained(model_type, override_args=kwargs)
