"""
Full GPT Language Model implementation using shared components.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

from .base import BaseLanguageModel, BaseConfig, ModelRegistry
from .common import TransformerBlock, Embeddings, LayerNorm, MultiHeadAttention


@dataclass
class GPTConfig(BaseConfig):
    """Configuration for GPT model - defaults match GPT-2 small"""
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    use_flash_attention: bool = True  # Enable flash attention by default


@ModelRegistry.register('gpt')
class GPT(BaseLanguageModel):
    """
    Full GPT Language Model.

    This is the complete GPT implementation matching GPT-2 architecture,
    using shared components for consistency with other models.
    """

    def _setup_model(self):
        """Setup the GPT model architecture."""
        self.transformer = nn.ModuleDict({
            'embeddings': Embeddings(self.config),
            'h': nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.n_layer)]),
            'ln_f': LayerNorm(self.config.n_embd, bias=self.config.bias),
        })
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        # Weight tying - share embedding and lm_head weights
        # https://paperswithcode.com/method/weight-tying
        self.transformer.embeddings.wte.weight = self.lm_head.weight

        # Apply special scaled init to the residual projections, per GPT-2 paper
        self._apply_residual_init()

    def _apply_residual_init(self):
        """Apply special initialization for residual projection weights."""
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0,
                    std=0.02 / math.sqrt(2 * self.config.n_layer)
                )

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
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"

        # Get embeddings (token + position)
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
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        Load a pretrained GPT-2 model from HuggingFace.

        Args:
            model_type: One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
            override_args: Optional dict of arguments to override (only 'dropout' supported)

        Returns:
            GPT model with pretrained weights
        """
        from transformers import GPT2LMHeadModel

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        override_args = override_args or {}
        # Only dropout can be overridden
        assert all(k == 'dropout' for k in override_args), \
            "Only 'dropout' can be overridden in from_pretrained"

        print(f"Loading weights from pretrained GPT-2: {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]

        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # Always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024   # Always 1024 for GPT model checkpoints
        config_args['bias'] = True         # Always True for GPT model checkpoints

        # Override dropout if specified
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        # Create model
        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # Discard mask buffer

        # Init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy while ensuring all parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                     'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # OpenAI checkpoints use Conv1D, so we transpose these weights when importing
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            # Convert HF key to our key format
            our_k = k
            if k.startswith('transformer.'):
                our_k = k.replace('transformer.', 'transformer.embeddings.' if 'wte' in k or 'wpe' in k else 'transformer.')
                if 'ln_f' in k:
                    our_k = k
                elif k.startswith('transformer.h.'):
                    # transformer.h.{i}.xxx -> transformer.h.{i}.xxx (same for blocks)
                    our_k = k

            if any(our_k.endswith(w) for w in transposed):
                # Special treatment for Conv1D weights - need to transpose
                if our_k in sd:
                    assert sd_hf[k].shape[::-1] == sd[our_k].shape
                    with torch.no_grad():
                        sd[our_k].copy_(sd_hf[k].t())
            else:
                # Vanilla copy for other parameters
                if our_k in sd:
                    assert sd_hf[k].shape == sd[our_k].shape
                    with torch.no_grad():
                        sd[our_k].copy_(sd_hf[k])

        return model


# Convenience functions
def create_gpt(**kwargs):
    """Create a GPT model with optional config overrides."""
    config = GPTConfig(**kwargs)
    model = GPT(config)
    return model


def gpt_from_pretrained(model_type: str, **kwargs):
    """Create a GPT model from pretrained GPT-2 weights."""
    return GPT.from_pretrained(model_type, override_args=kwargs)
