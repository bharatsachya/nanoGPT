"""
DeepSeek-Style Mixture of Experts Reasoning Model.

This module implements a DeepSeek-style model with:
- Mixture of Experts (MoE) architecture
- Fine-grained expert routing
- Shared attention experts
- Auxiliary loss balancing for expert load
- Efficient reasoning capabilities

References:
- DeepSeek-MoE: Towards Ultimate Expert Utilization in Mixture-of-Experts Language Models
- DeepSeek-Coder: When the Large Language Model is a Coding Assistant
- "Mixture of Experts" (Shazeer et al., 2017)
- "Switch Transformers" (Fedus et al., 2021)
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Callable, Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Gumbel

from .base import BaseLanguageModel, BaseConfig, ModelRegistry
from .common import LayerNorm, MultiHeadAttention, Embeddings, RotaryEmbedding, apply_rotary_emb


@dataclass
class DeepSeekConfig(BaseConfig):
    """Configuration for DeepSeek-style MoE model."""
    # Base model config
    n_layer: int = 28
    n_head: int = 16
    n_embd: int = 2048
    block_size: int = 4096
    vocab_size: int = 102400  # Larger vocab for code + reasoning
    dropout: float = 0.1
    bias: bool = False  # DeepSeek typically doesn't use bias

    # MoE config
    n_routed_experts: int = 64  # Total number of routed experts
    n_shared_experts: int = 2   # Number of shared experts (always active)
    n_experts_per_tok: int = 6  # Number of experts to route each token to (k)
    moe_intermediate_size: int = 1408  # Size of expert intermediate layers

    # Routing config
    routing_algorithm: str = 'topk'  # 'topk', 'soft', 'gumbel'
    routing_scale_factor: float = 1.0  # Scaling factor for routing logits
    aux_loss_coef: float = 0.01  # Coefficient for auxiliary load balancing loss
    z_loss_coef: float = 0.0001  # Coefficient for z-loss (logit stabilization)

    # Reasoning-specific config
    use_reasoning_head: bool = True
    reasoning_layer_start: int = 16  # Start using MoE from this layer
    reasoning_layer_end: int = 28  # End using MoE at this layer

    # Efficiency config
    use_rotary: bool = True
    rotary_dim: int = 64
    use_flash_attention: bool = True
    fp16: bool = False


class Expert(nn.Module):
    """Individual expert network (MLP)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.activation = nn.SiLU()  # Swish activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert."""
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class SharedExpert(nn.Module):
    """Shared expert network (always active)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (SwiGLU-style)."""
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TopKRouter(nn.Module):
    """
    Top-K Router for Mixture of Experts.

    Routes each token to the top-k experts based on routing scores.
    """

    def __init__(
        self,
        n_experts: int,
        n_experts_per_tok: int,
        dim: int,
        routing_scale_factor: float = 1.0
    ):
        super().__init__()
        self.n_experts = n_experts
        self.n_experts_per_tok = n_experts_per_tok
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.routing_scale_factor = routing_scale_factor

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            x: Input tensor of shape (batch_size * seq_len, dim)

        Returns:
            expert_mask: Binary mask of shape (batch_size * seq_len, n_experts)
            weights: Routing weights of shape (batch_size * seq_len, n_experts_per_tok)
            expert_indices: Selected expert indices
        """
        # Compute routing logits
        logits = self.gate(x) * self.routing_scale_factor

        # Apply top-k
        topk_weights, expert_indices = torch.topk(
            logits, self.n_experts_per_tok, dim=-1
        )

        # Normalize weights
        topk_weights = topk_weights.softmax(dim=-1)

        # Create expert mask
        expert_mask = torch.zeros_like(logits).scatter_(
            -1, expert_indices, torch.ones_like(topk_weights)
        )

        return expert_mask, topk_weights, expert_indices


class SoftRouter(nn.Module):
    """
    Soft Router for Mixture of Experts.

    Uses soft routing (all experts get some weight) rather than hard top-k.
    """

    def __init__(
        self,
        n_experts: int,
        n_experts_per_tok: int,
        dim: int,
        routing_scale_factor: float = 1.0
    ):
        super().__init__()
        self.n_experts = n_experts
        self.n_experts_per_tok = n_experts_per_tok
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.routing_scale_factor = routing_scale_factor

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Soft routing to experts."""
        logits = self.gate(x) * self.routing_scale_factor
        weights = logits.softmax(dim=-1)

        # Still get top-k indices for compatibility, but use soft weights
        topk_weights, expert_indices = torch.topk(
            weights, self.n_experts_per_tok, dim=-1
        )

        # Use all weights (not just top-k)
        expert_mask = torch.ones_like(weights)

        return expert_mask, weights, expert_indices


class GumbelRouter(nn.Module):
    """
    Gumbel-Softmax Router for differentiable expert selection.

    Uses Gumbel-Softmax for stochastic but differentiable routing.
    """

    def __init__(
        self,
        n_experts: int,
        n_experts_per_tok: int,
        dim: int,
        routing_scale_factor: float = 1.0,
        temperature: float = 1.0
    ):
        super().__init__()
        self.n_experts = n_experts
        self.n_experts_per_tok = n_experts_per_tok
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.routing_scale_factor = routing_scale_factor
        self.temperature = temperature

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gumbel-Softmax routing."""
        logits = self.gate(x) * self.routing_scale_factor

        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log_like(
            torch.rand_like(logits) + 1e-10, 1e-10
        ))
        noisy_logits = logits + gumbel_noise

        # Apply softmax with temperature
        weights = F.softmax(noisy_logits / self.temperature, dim=-1)

        # Get top-k for efficient computation
        topk_weights, expert_indices = torch.topk(
            weights, self.n_experts_per_tok, dim=-1
        )

        # Normalize top-k weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Create mask
        expert_mask = torch.zeros_like(logits).scatter_(
            -1, expert_indices, torch.ones_like(topk_weights)
        )

        return expert_mask, topk_weights, expert_indices


class MoE(nn.Module):
    """
    Mixture of Experts Layer.

    Combines shared experts (always active) with routed experts (selectively active).
    """

    def __init__(
        self,
        dim: int,
        n_routed_experts: int,
        n_shared_experts: int,
        n_experts_per_tok: int,
        moe_intermediate_size: int,
        routing_algorithm: str = 'topk',
        routing_scale_factor: float = 1.0,
        dropout: float = 0.0,
        aux_loss_coef: float = 0.01,
        z_loss_coef: float = 0.0001
    ):
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_experts_per_tok = n_experts_per_tok
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef

        # Router
        if routing_algorithm == 'soft':
            self.router = SoftRouter(n_routed_experts, n_experts_per_tok, dim, routing_scale_factor)
        elif routing_algorithm == 'gumbel':
            self.router = GumbelRouter(n_routed_experts, n_experts_per_tok, dim, routing_scale_factor)
        else:  # topk
            self.router = TopKRouter(n_routed_experts, n_experts_per_tok, dim, routing_scale_factor)

        # Routed experts
        self.experts = nn.ModuleList([
            Expert(dim, moe_intermediate_size, dropout)
            for _ in range(n_routed_experts)
        ])

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            SharedExpert(dim, moe_intermediate_size, dropout)
            for _ in range(n_shared_experts)
        ])

        # Output projection
        self.gate_proj = nn.Linear(dim, n_shared_experts * moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, n_shared_experts * moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(n_shared_experts * moe_intermediate_size, dim, bias=False)

        # For tracking auxiliary loss
        self.last_aux_loss = torch.tensor(0.0)
        self.last_z_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)

        Returns:
            Output tensor with auxiliary loss stored in self.last_aux_loss
        """
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # (batch_size * seq_len, dim)

        # Route to experts
        expert_mask, weights, expert_indices = self.router(x_flat)

        # Compute auxiliary loss for load balancing
        aux_loss = self._compute_aux_loss(expert_mask, weights)
        self.last_aux_loss = aux_loss

        # Process through routed experts
        expert_output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_tokens = expert_mask[:, i].bool()

            if expert_tokens.any():
                expert_input = x_flat[expert_tokens]
                expert_out = expert(expert_input)

                # Weight the output by routing weights
                expert_weight = weights[expert_tokens, expert_indices[expert_tokens] == i]
                if expert_weight.ndim == 1:
                    expert_weight = expert_weight.unsqueeze(-1)

                expert_output[expert_tokens] += expert_out * expert_weight

        # Process through shared experts (always active)
        shared_output = torch.zeros_like(x_flat)
        for i, shared_expert in enumerate(self.shared_experts):
            shared_out = shared_expert(x_flat)
            shared_output += shared_out

        # Combine outputs
        output = expert_output + shared_output
        output = output.view(batch_size, seq_len, dim)

        return output

    def _compute_aux_loss(
        self,
        expert_mask: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss for load balancing.

        Encourages equal expert utilization.
        """
        # Compute fraction of tokens routed to each expert
        tokens_per_expert = expert_mask.sum(0)  # (n_experts,)
        total_tokens = expert_mask.sum()
        fraction_tokens = tokens_per_expert / (total_tokens + 1e-10)

        # Compute average routing weights per expert
        avg_weights = weights.mean(0)  # (n_experts_per_tok,) -> need to expand

        # For top-k routing, compute per-expert weights
        expert_weights = torch.zeros(self.n_routed_experts, device=weights.device)
        for i in range(self.n_experts_per_tok):
            expert_weights.scatter_add_(0, expert_indices[:, i], weights[:, i])
        expert_weights = expert_weights / (total_tokens + 1e-10)

        # Auxiliary loss: minimize difference between token fraction and weight fraction
        aux_loss = self.aux_loss_coef * (
            fraction_tokens * expert_weights
        ).sum() * self.n_routed_experts

        # Z-loss: encourage small routing logits (stability)
        if self.z_loss_coef > 0:
            z_loss = self.z_loss_coef * (
                weights.pow(2).sum() / (total_tokens + 1e-10)
            )
            aux_loss = aux_loss + z_loss
            self.last_z_loss = z_loss

        return aux_loss

    def get_aux_loss(self) -> torch.Tensor:
        """Get the last computed auxiliary loss."""
        return self.last_aux_loss


class DeepSeekAttention(nn.Module):
    """
    DeepSeek-style attention with rotary position embeddings.

    Uses grouped-query attention (GQA) for efficiency.
    """

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.q_lora_rank = 64  # Low-rank projection for Q
        self.kv_lora_rank = config.n_embd // config.n_head  # Low-rank for KV

        # Q projection with low-rank
        self.q_a_layernorm = LayerNorm(config.n_embd, bias=False)
        self.q_a_proj = nn.Linear(config.n_embd, self.q_lora_rank, bias=False)
        self.q_b_proj = nn.Linear(self.q_lora_rank, config.n_embd, bias=False)

        # KV projection with low-rank (GQA: share KV across heads)
        self.kv_a_layernorm = LayerNorm(config.n_embd, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            config.n_embd, self.kv_lora_rank + config.n_head * 2 * self.head_dim, bias=False
        )
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, config.n_embd, bias=False
        )

        # Output projection
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Rotary embeddings
        self.use_rotary = config.use_rotary
        if self.use_rotary:
            self.rotary_emb = RotaryEmbedding(
                config.rotary_dim,
                max_position_embeddings=config.block_size
            )

        # Flash attention
        self.use_flash = config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention')

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        bsz, seq_len, _ = x.shape

        # Q projection with low-rank
        q = self.q_a_layernorm(x)
        q = self.q_a_proj(q)
        q = self.q_b_proj(q)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # KV projection with low-rank (GQA)
        kv = self.kv_a_layernorm(x)
        kv = self.kv_a_proj_with_mqa(kv)
        kv = kv.view(bsz, seq_len, -1)

        # Split K and V
        k, v = torch.split(kv, [self.kv_lora_rank, self.n_head * 2 * self.head_dim], dim=-1)
        k = self.kv_b_proj(k)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, 2, self.head_dim)
        v, _ = torch.unbind(v, dim=-2)
        v = v.transpose(1, 2)

        # Apply rotary embeddings
        if self.use_rotary:
            seq_len = q.size(1)
            emb = self.rotary_emb(q, seq_len)
            q = apply_rotary_emb(q, emb)
            k = apply_rotary_emb(k, emb)

        # Attention
        if self.use_flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                is_causal=True
            )
        else:
            # Manual attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attention_mask is not None:
                att = att + attention_mask
            else:
                # Causal mask
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
                att = att.masked_fill(causal_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        y = self.o_proj(y)

        return y


class DeepSeekBlock(nn.Module):
    """
    DeepSeek transformer block.

    Alternates between attention and MoE layers.
    """

    def __init__(self, config: DeepSeekConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Layer norms (RMSNorm-style)
        self.input_layernorm = LayerNorm(config.n_embd, bias=config.bias)
        self.post_attention_layernorm = LayerNorm(config.n_embd, bias=config.bias)

        # Attention
        self.self_attn = DeepSeekAttention(config)

        # MoE (only for certain layers)
        use_moe = (config.reasoning_layer_start <= layer_idx < config.reasoning_layer_end)
        self.use_moe = use_moe

        if use_moe:
            self.moe = MoE(
                dim=config.n_embd,
                n_routed_experts=config.n_routed_experts,
                n_shared_experts=config.n_shared_experts,
                n_experts_per_tok=config.n_experts_per_tok,
                moe_intermediate_size=config.moe_intermediate_size,
                routing_algorithm=config.routing_algorithm,
                routing_scale_factor=config.routing_scale_factor,
                dropout=config.dropout,
                aux_loss_coef=config.aux_loss_coef,
                z_loss_coef=config.z_loss_coef
            )
        else:
            # Standard MLP for non-reasoning layers
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, config.moe_intermediate_size, bias=config.bias),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.moe_intermediate_size, config.n_embd, bias=config.bias),
                nn.Dropout(config.dropout)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Pre-norm attention
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = x + residual

        # Pre-norm MLP/MoE
        residual = x
        x = self.post_attention_layernorm(x)

        if self.use_moe:
            x = self.moe(x)
        else:
            x = self.mlp(x)

        x = x + residual

        return x


@ModelRegistry.register('deepseek')
class DeepSeekModel(BaseLanguageModel):
    """
    DeepSeek-Style Mixture of Experts Reasoning Model.

    Combines efficient MoE routing with reasoning capabilities:
    - Shared attention experts in early/later layers
    - Routed MoE experts in middle reasoning layers
    - Fine-grained expert specialization
    - Load balancing for expert utilization

    Architecture:
        - Embeddings (token + position)
        - N transformer blocks with alternating MoE
        - Final layernorm and LM head

    Usage:
        >>> config = DeepSeekConfig(n_layer=28, n_embd=2048)
        >>> model = DeepSeekModel(config)
        >>> output = model(input_ids)
    """

    def _setup_model(self):
        """Setup the DeepSeek model architecture."""
        # Embeddings
        self.transformer = nn.ModuleDict({
            'embeddings': Embeddings(self.config),
            'h': nn.ModuleList([
                DeepSeekBlock(self.config, i)
                for i in range(self.config.n_layer)
            ]),
            'ln_f': LayerNorm(self.config.n_embd, bias=self.config.bias),
        })

        # Language modeling head
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        # Weight tying
        self.transformer.embeddings.wte.weight = self.lm_head.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            idx: Input token indices (B, T)
            targets: Target token indices (B, T)
            return_aux_loss: Return MoE auxiliary loss

        Returns:
            Dictionary with logits, loss, and optionally aux_loss
        """
        b, t = idx.size()
        assert t <= self.config.block_size

        # Embeddings
        x = self.transformer.embeddings(idx)

        # Track auxiliary loss from MoE layers
        total_aux_loss = torch.tensor(0.0, device=idx.device)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

            # Accumulate auxiliary loss from MoE layers
            if block.use_moe and return_aux_loss:
                total_aux_loss = total_aux_loss + block.moe.get_aux_loss()

        # Final layernorm
        x = self.transformer.ln_f(x)

        outputs = {}

        # Compute logits and loss
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            outputs['logits'] = logits
            outputs['loss'] = loss

            # Add auxiliary loss
            if return_aux_loss:
                loss = loss + total_aux_loss
                outputs['loss'] = loss
        else:
            # Inference: only compute last token
            logits = self.lm_head(x[:, [-1], :])
            outputs['logits'] = logits
            outputs['loss'] = None

        outputs['aux_loss'] = total_aux_loss
        outputs['expert_stats'] = self._get_expert_stats()

        return outputs

    def _get_expert_stats(self) -> Dict[str, float]:
        """Get statistics about expert utilization."""
        stats = {
            'n_moe_layers': sum(1 for block in self.transformer.h if block.use_moe),
            'n_total_layers': len(self.transformer.h),
        }
        return stats

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        self.eval()

        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            outputs = self(idx_cond, return_aux_loss=False)
            logits = outputs['logits'][:, -1, :] / temperature

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
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @classmethod
    def from_pretrained(cls, model_type: str, **kwargs):
        """
        Load from pretrained GPT-2 and initialize as DeepSeek model.

        Note: This provides weight initialization but not the full MoE architecture
        from the original model.
        """
        from .gpt import GPT

        # Create base GPT model for weight initialization
        base_model = GPT.from_pretrained(model_type, override_args=kwargs)

        # Extract config
        config_dict = {
            'n_layer': min(base_model.config.n_layer, 28),
            'n_head': base_model.config.n_head,
            'n_embd': base_model.config.n_embd,
            'block_size': base_model.config.block_size,
            'vocab_size': min(base_model.config.vocab_size, 102400),
            'dropout': base_model.config.dropout,
            'bias': base_model.config.bias,
        }
        config_dict.update(kwargs)

        # Create DeepSeek model
        config = DeepSeekConfig(**config_dict)
        model = cls(config)

        # Copy compatible weights where possible
        # (This is a simplified version - full conversion would be more complex)
        state_dict = base_model.state_dict()
        model_state_dict = model.state_dict()

        for key in state_dict:
            if key in model_state_dict and state_dict[key].shape == model_state_dict[key].shape:
                model_state_dict[key].copy_(state_dict[key])

        return model


class DeepSeekCoderModel(DeepSeekModel):
    """
    DeepSeek-Coder variant optimized for code generation and reasoning.

    Features:
    - Larger vocabulary for code
    - Specialized code experts
    - Fill-in-the-middle capability
    """

    def __init__(self, config: DeepSeekConfig):
        # Update config for coding tasks
        config.vocab_size = max(config.vocab_size, 102400)
        config.n_routed_experts = 64  # More experts for code patterns
        super().__init__(config)

    @torch.no_grad()
    def generate_code(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.2,  # Lower temperature for code
        stop_tokens: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Generate code completion."""
        generated = self.generate(prompt, max_new_tokens, temperature)

        # Apply stop tokens if provided
        if stop_tokens:
            for i in range(1, generated.size(1)):
                if generated[:, i].item() in stop_tokens:
                    generated = generated[:, :i+1]
                    break

        return generated


# Convenience functions
def create_deepseek_model(**kwargs):
    """Create a DeepSeek model with optional config overrides."""
    config = DeepSeekConfig(**kwargs)
    model = DeepSeekModel(config)
    return model


def create_deepseek_coder(**kwargs):
    """Create a DeepSeek-Coder model."""
    config = DeepSeekConfig(**kwargs)
    model = DeepSeekCoderModel(config)
    return model


def create_deepseek_from_pretrained(model_type: str, **kwargs):
    """Create a DeepSeek model initialized from pretrained weights."""
    return DeepSeekModel.from_pretrained(model_type, **kwargs)
