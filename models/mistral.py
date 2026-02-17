"""
Mistral / LLaMA-Style Modern Transformer Model.

This module implements a modern LLM architecture inspired by Mistral and LLaMA:
- RMSNorm (Root Mean Square Layer Normalization)
- SwiGLU activation function
- Rotary Position Embeddings (RoPE)
- Grouped-Query Attention (GQA)
- Sliding Window Attention (SWA)
- No bias in linear layers
- Large vocabulary (32K - 128K)

References:
- Mistral 7B: https://arxiv.org/abs/2310.06825
- LLaMA 2: https://arxiv.org/abs/2307.09288
- LLaMA 3: https://llama.meta.com/llama3/
- "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import BaseLanguageModel, BaseConfig, ModelRegistry
from .common import RMSNorm, RotaryEmbedding, apply_rotary_emb


@dataclass
class MistralConfig(BaseConfig):
    """Configuration for Mistral/LLaMA-style model."""
    # Model size
    vocab_size: int = 32000  # Mistral/LLaMA standard vocab
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    intermediate_size: int = 14336  # 8/3 * n_embd for SwiGLU
    block_size: int = 8192  # Context length

    # Attention config
    n_kv_heads: int = 8  # Number of key-value heads for GQA
    sliding_window: int = 4096  # Sliding window attention
    attention_dropout: float = 0.0
    rope_theta: float = 10000.0  # RoPE base frequency
    rope_scaling: Optional[Dict] = None

    # MLP config
    hidden_act: str = "silu"  # SwiGLU uses SiLU

    # Training config
    dropout: float = 0.0
    bias: bool = False  # Mistral/LLaMA don't use bias

    # Initialization
    initializer_range: float = 0.02

    # Special tokens
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0


# Model size presets based on Mistral/LLaMA
MODEL_PRESETS = {
    'mistral-7b': {
        'n_layer': 32,
        'n_head': 32,
        'n_embd': 4096,
        'intermediate_size': 14336,
        'n_kv_heads': 8,
        'vocab_size': 32000,
    },
    'mistral-7b-v02': {
        'n_layer': 32,
        'n_head': 32,
        'n_embd': 4096,
        'intermediate_size': 14336,
        'n_kv_heads': 8,
        'vocab_size': 32768,
        'sliding_window': None,  # v0.2 doesn't use sliding window
    },
    'mixtral-8x7b': {
        'n_layer': 32,
        'n_head': 32,
        'n_embd': 4096,
        'intermediate_size': 14336,
        'n_kv_heads': 8,
        'vocab_size': 32000,
        'num_experts': 8,
        'num_experts_per_tok': 2,
    },
    'llama2-7b': {
        'n_layer': 32,
        'n_head': 32,
        'n_embd': 4096,
        'intermediate_size': 11008,
        'n_kv_heads': 32,  # LLaMA 2 uses MHA, not GQA
        'vocab_size': 32000,
    },
    'llama2-13b': {
        'n_layer': 40,
        'n_head': 40,
        'n_embd': 5120,
        'intermediate_size': 13824,
        'n_kv_heads': 40,
        'vocab_size': 32000,
    },
    'llama2-70b': {
        'n_layer': 80,
        'n_head': 64,
        'n_embd': 8192,
        'intermediate_size': 28672,
        'n_kv_heads': 8,  # GQA for 70B
        'vocab_size': 32000,
    },
    'llama3-8b': {
        'n_layer': 32,
        'n_head': 32,
        'n_embd': 4096,
        'intermediate_size': 14336,
        'n_kv_heads': 8,
        'vocab_size': 128256,
        'sliding_window': None,
    },
    'llama3-70b': {
        'n_layer': 80,
        'n_head': 64,
        'n_embd': 8192,
        'intermediate_size': 28672,
        'n_kv_heads': 8,
        'vocab_size': 128256,
        'sliding_window': None,
    },
    'tiny-1b': {
        'n_layer': 16,
        'n_head': 16,
        'n_embd': 2048,
        'intermediate_size': 5632,
        'n_kv_heads': 4,
        'vocab_size': 32000,
        'block_size': 2048,
    },
    'small-3b': {
        'n_layer': 26,
        'n_head': 26,
        'n_embd': 3072,
        'intermediate_size': 8192,
        'n_kv_heads': 6,
        'vocab_size': 32000,
        'block_size': 4096,
    },
}


def repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads for Grouped-Query Attention.

    Args:
        hidden: Hidden states of shape (batch, n_kv_heads, seq_len, head_dim)
        n_rep: Number of times to repeat each head

    Returns:
        Repeated hidden states of shape (batch, n_heads, seq_len, head_dim)
    """
    batch, n_kv_heads, seq_len, head_dim = hidden.shape
    hidden = hidden[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
    return hidden.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


class MistralAttention(nn.Module):
    """
    Mistral-style attention with RoPE and GQA.

    Features:
    - Grouped-Query Attention (GQA)
    - Rotary Position Embeddings (RoPE)
    - Sliding Window Attention (optional)
    - Flash Attention support
    """

    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.n_embd // config.n_head

        # QKV projections
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim, bias=config.bias)

        # Output projection
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=config.bias)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.block_size,
            base=config.rope_theta
        )

        # Sliding window
        self.sliding_window = config.sliding_window

        # Attention
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Flash attention
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV cache.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            attention_mask: Optional attention mask
            kv_cache: Optional KV cache tuple (key_cache, value_cache)

        Returns:
            Tuple of (output, new_kv_cache)
        """
        bsz, seq_len, _ = x.shape

        # QKV projections
        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(seq_len)
        query_states, key_states = apply_rotary_emb(query_states, cos), apply_rotary_emb(key_states, sin)

        # Handle KV cache
        if kv_cache is not None:
            key_cache, value_cache = kv_cache
            key_states = torch.cat([key_cache, key_states], dim=2)
            value_states = torch.cat([value_cache, value_states], dim=2)
        new_kv_cache = (key_states, value_states)

        # Repeat KV for GQA
        key_states = repeat_kv(key_states, self.n_rep)
        value_states = repeat_kv(value_states, self.n_rep)

        # Compute attention
        if self.use_flash:
            # Flash attention
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=self.is_causal and attention_mask is None
            )
        else:
            # Manual attention
            attn_weights = (query_states @ key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Apply causal mask
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            else:
                # Create causal mask
                causal_mask = torch.tril(torch.ones(seq_len, key_states.size(2), device=x.device))
                attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))

            # Apply sliding window mask
            if self.sliding_window is not None:
                sliding_mask = torch.arange(seq_len, device=x.device).view(1, -1) >= \
                               torch.arange(key_states.size(2), device=x.device).view(-1, 1)
                sliding_mask = sliding_mask.masked(
                    torch.arange(key_states.size(2), device=x.device).view(-1, 1) >=
                    torch.arange(seq_len, device=x.device).view(1, -1) - self.sliding_window
                )
                attn_weights = attn_weights.masked_fill(~sliding_mask, float('-inf'))

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = attn_weights @ value_states

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, new_kv_cache


class MistralMLP(nn.Module):
    """
    Mistral/LLaMA-style MLP with SwiGLU activation.

    SwiGLU: Swish(xW) @ (xV)
    """

    def __init__(self, config: MistralConfig):
        super().__init__()
        hidden_size = config.n_embd
        intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=config.bias)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU activation."""
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MistralDecoderLayer(nn.Module):
    """
    Mistral/LLaMA transformer decoder layer.

    Architecture:
        1. RMSNorm -> Attention
        2. Residual connection
        3. RMSNorm -> MLP
        4. Residual connection
    """

    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.n_embd
        self.self_attn = MistralAttention(config)
        self.mlp = MistralMLP(config)
        self.input_layernorm = RMSNorm(config.n_embd, eps=1e-5)
        self.post_attention_layernorm = RMSNorm(config.n_embd, eps=1e-5)
        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            kv_cache: Optional KV cache

        Returns:
            Tuple of (output, new_kv_cache)
        """
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        attn_output, new_kv_cache = self.self_attn(x, attention_mask, kv_cache)
        x = residual + attn_output

        # MLP with residual
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x)
        x = residual + mlp_output

        return x, new_kv_cache


@ModelRegistry.register('mistral')
class MistralModel(BaseLanguageModel):
    """
    Mistral/LLaMA-style modern transformer model.

    Features:
    - RMSNorm normalization
    - SwiGLU activation
    - Rotary Position Embeddings (RoPE)
    - Grouped-Query Attention (GQA)
    - Sliding Window Attention (optional)
    - No bias in layers
    - Efficient inference with KV cache

    Usage:
        >>> config = MistralConfig.from_preset('mistral-7b')
        >>> model = MistralModel(config)
        >>> output = model(input_ids)
    """

    def _setup_model(self):
        """Setup the Mistral model architecture."""
        # Embeddings
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.n_embd)

        # Transformer layers
        self.layers = nn.ModuleList([
            MistralDecoderLayer(self.config, idx)
            for idx in range(self.config.n_layer)
        ])

        # Final normalization
        self.norm = RMSNorm(self.config.n_embd, eps=1e-5)

        # LM head (share weights with embeddings)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range
            )

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            idx: Input token indices of shape (B, T)
            targets: Target token indices of shape (B, T)
            attention_mask: Optional attention mask
            use_cache: Whether to return KV cache for efficient generation

        Returns:
            Dictionary with logits, loss, and optionally cache
        """
        bsz, seq_len = idx.shape
        device = idx.device

        # Embed tokens
        inputs_embeds = self.embed_tokens(idx)

        # Cache for efficient generation
        all_cache = () if use_cache else None

        # Pass through transformer layers
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states, cache = decoder_layer(
                hidden_states,
                attention_mask,
                kv_cache=all_cache[-1] if use_cache and all_cache else None
            )
            if use_cache:
                all_cache = all_cache + (cache,)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Compute logits
        logits = self.lm_head(hidden_states)

        outputs = {'logits': logits}

        # Compute loss if targets provided
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.pad_token_id if self.config.pad_token_id is not None else -100
            )
            outputs['loss'] = loss
        else:
            outputs['loss'] = None

        if use_cache:
            outputs['past_key_values'] = all_cache

        return outputs

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively with KV cache.

        Args:
            idx: Input token indices
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            eos_token_id: End-of-sequence token ID

        Returns:
            Generated token indices
        """
        self.eval()
        generated = idx.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            # Use KV cache for efficient generation
            if past_key_values is not None:
                # Only compute for the last token
                input_ids = generated[:, -1:]
            else:
                input_ids = generated

            outputs = self(input_ids, use_cache=True)
            logits = outputs['logits']
            past_key_values = outputs.get('past_key_values', None)

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, idx_next), dim=1)

            # Check for EOS
            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break

        return generated

    @classmethod
    def from_pretrained(cls, model_type: str, **kwargs):
        """Create model from preset configuration."""
        config = MistralConfig.from_preset(model_type, **kwargs)
        model = cls(config)
        return model


class MistralForCausalLM(MistralModel):
    """Mistral model wrapper for causal language modeling."""

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for efficient generation."""
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        model_inputs = {
            'idx': input_ids,
        }

        if past_key_values is not None:
            model_inputs['use_cache'] = True

        if attention_mask is not None:
            model_inputs['attention_mask'] = attention_mask

        return model_inputs


class MistralForSequenceClassification(MistralModel):
    """Mistral model for sequence classification tasks."""

    def __init__(self, config: MistralConfig, num_labels: int = 2):
        super().__init__(config)
        self.num_labels = num_labels
        self.score = nn.Linear(config.n_embd, num_labels, bias=False)

        # Initialize classification head
        nn.init.normal_(self.score.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification.

        Args:
            idx: Input token indices
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation

        Returns:
            Dictionary with logits and optionally loss
        """
        outputs = super().forward(idx, attention_mask=attention_mask)
        hidden_states = outputs['logits']  # Actually hidden states before LM head

        # Use the last token for classification
        if attention_mask is not None:
            # Find the last non-padding token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = idx.size(0)
            pooled_hidden = hidden_states[torch.arange(batch_size), sequence_lengths]
        else:
            pooled_hidden = hidden_states[:, -1, :]

        logits = self.score(pooled_hidden)

        outputs = {'logits': logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs['loss'] = loss

        return outputs


class MistralForTokenClassification(MistralModel):
    """Mistral model for token classification (NER, POS tagging, etc.)."""

    def __init__(self, config: MistralConfig, num_labels: int = 10):
        super().__init__(config)
        self.num_labels = num_labels
        self.classifier = nn.Linear(config.n_embd, num_labels, bias=False)

        # Don't tie weights for token classification
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        delattr(self, 'lm_head')  # Remove LM head

        # Initialize classifier
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for token classification."""
        bsz, seq_len, _ = idx.shape

        # Get embeddings
        inputs_embeds = self.embed_tokens(idx)

        # Pass through layers
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states, _ = decoder_layer(hidden_states, attention_mask)

        # Normalize
        hidden_states = self.norm(hidden_states)

        # Classification logits
        logits = self.classifier(hidden_states)

        outputs = {'logits': logits}

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100
            )
            outputs['loss'] = loss

        return outputs


# Utility functions
def create_mistral_model(preset: str = 'mistral-7b', **kwargs) -> MistralModel:
    """Create a Mistral model from a preset."""
    config = MistralConfig.from_preset(preset, **kwargs)
    return MistralModel(config)


def create_llama_model(preset: str = 'llama3-8b', **kwargs) -> MistralModel:
    """Create a LLaMA model from a preset."""
    config = MistralConfig.from_preset(preset, **kwargs)
    return MistralModel(config)


def create_custom_model(**kwargs) -> MistralModel:
    """Create a custom model with specified configuration."""
    config = MistralConfig(**kwargs)
    return MistralModel(config)


# Extend MistralConfig with preset functionality
def _add_preset_method(cls):
    """Add from_preset class method to config."""
    def from_preset(cls, preset: str, **kwargs):
        """Create config from a preset."""
        if preset not in MODEL_PRESETS:
            available = ', '.join(MODEL_PRESETS.keys())
            raise ValueError(f"Unknown preset: {preset}. Available: {available}")

        config_dict = MODEL_PRESETS[preset].copy()
        config_dict.update(kwargs)
        return cls(**config_dict)

    cls.from_preset = classmethod(from_preset)


_add_preset_method(MistralConfig)
