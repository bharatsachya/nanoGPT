"""
Common components shared across all model architectures.
Includes attention mechanisms, feed-forward networks, and layer normalization.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with configurable features.
    Supports flash attention, causal masking, and optional key-value caching.
    """

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.bias = config.bias

        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.use_flash = getattr(config, 'use_flash_attention', True) and self.flash

        if not self.flash or not self.use_flash:
            block_size = getattr(config, 'block_size', 1024)
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(block_size, block_size))
                .view(1, 1, block_size, block_size)
            )

    def forward(self, x, mask=None):
        B, T, C = x.size()

        # Q, K, V projections
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Attention computation
        if self.flash and self.use_flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # Manual attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if mask is not None:
                att = att.masked_fill(mask == 0, float('-inf'))
            else:
                att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    """
    Feed-forward network with configurable expansion factor and activation.
    """

    def __init__(self, config, expansion_factor=4):
        super().__init__()
        hidden_dim = config.n_embd * expansion_factor
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        # Activation function
        activation = getattr(config, 'activation', 'gelu')
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    Transformer block with pre-normalization.
    Architecture: LN -> Attention -> Residual -> LN -> FFN -> Residual
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Embeddings(nn.Module):
    """Token and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        return self.dropout(self.wte(idx) + self.wpe(pos))


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Often used in modern LLMs (Llama, Mistral, etc.)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit.
    Used in Llama and other modern LLMs.
    """

    def __init__(self, config, expansion_factor=4):
        super().__init__()
        hidden_dim = int(config.n_embd * expansion_factor * 2 / 3)
        hidden_dim = ((hidden_dim + 255) // 256) * 256  # Round to multiple of 256
        self.gate = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate = F.silu(self.gate(x))
        return self.dropout(self.down(gate * self.up(x)))


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Used in modern LLMs for better position encoding.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[:, None, None, :]


def apply_rotary_emb(x, emb):
    """Apply rotary embeddings to query and key."""
    cos = emb.cos()
    sin = emb.sin()
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
