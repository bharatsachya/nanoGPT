"""
Reasoning Language Model with RLHF and ReFL support.

This module implements a reasoning model that:
1. Supports Chain-of-Thought (CoT) reasoning
2. Can be trained with RLHF (Reinforcement Learning from Human Feedback)
3. Can be trained with ReFL (Reinforcement Fine-tuning with Language Feedback)
4. Includes separate reasoning and answer heads for better reasoning separation

References:
- "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- "Training language models to follow instructions with human feedback"
- "Fine-tuning with Language Feedback is (Almost) All You Need"
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import BaseLanguageModel, BaseConfig, ModelRegistry
from .common import TransformerBlock, LayerNorm, MultiHeadAttention, FeedForward


@dataclass
class ReasoningModelConfig(BaseConfig):
    """Configuration for Reasoning Model."""
    # Base model config
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 2048  # Longer context for reasoning
    vocab_size: int = 50304
    dropout: float = 0.1
    bias: bool = True

    # Reasoning-specific config
    reasoning_head_type: str = 'cot'  # 'cot' (chain-of-thought), 'split' (separate heads), 'none'
    cot_token_id: int = 50281  # Special token to trigger CoT reasoning
    answer_token_id: int = 50282  # Special token marking start of final answer

    # Training config
    use_value_head: bool = True  # Add value head for RLHF
    value_head_dim: int = 256  # Dimension of value head hidden layer

    # ReFL (Language Feedback) config
    use_critique_head: bool = True  # Add critique head for language feedback
    critique_token_id: int = 50283  # Special token for critique generation

    # Reasoning format
    reasoning_format: str = 'standard'  # 'standard', 'structured', 'tree'


class ReasoningEmbeddings(nn.Module):
    """Enhanced embeddings with reasoning-specific token embeddings."""

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        # Learnable reasoning prompts (soft prompts for reasoning)
        self.n_reasoning_prompts = 8
        self.reasoning_prompts = nn.Parameter(
            torch.randn(self.n_reasoning_prompts, config.n_embd) * 0.02
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

        # Standard token + position embeddings
        x = self.wte(idx) + self.wpe(pos)

        # Prepend reasoning prompts (soft prompts)
        prompts = self.reasoning_prompts.unsqueeze(0).expand(b, -1, -1)
        x = torch.cat([prompts, x], dim=1)

        return self.dropout(x)


class ValueHead(nn.Module):
    """Value head for RLHF training."""

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.value_head_dim)
        self.fc2 = nn.Linear(config.value_head_dim, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class CritiqueHead(nn.Module):
    """Critique head for ReFL (language feedback generation)."""

    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, x):
        return self.fc(x)


@ModelRegistry.register('reasoning')
class ReasoningModel(BaseLanguageModel):
    """
    Reasoning Language Model with RLHF and ReFL support.

    Features:
    - Chain-of-Thought (CoT) reasoning capabilities
    - Separate reasoning and answer heads
    - Value head for RLHF training
    - Critique head for ReFL (language feedback)
    - Learnable reasoning prompts
    """

    def _setup_model(self):
        """Setup the reasoning model architecture."""
        self.transformer = nn.ModuleDict({
            'embeddings': ReasoningEmbeddings(self.config),
            'h': nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.n_layer)]),
            'ln_f': LayerNorm(self.config.n_embd, bias=self.config.bias),
        })

        # Language modeling head
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        # Weight tying
        self.transformer.embeddings.wte.weight = self.lm_head.weight

        # Value head for RLHF
        if self.config.use_value_head:
            self.value_head = ValueHead(self.config)

        # Critique head for ReFL
        if self.config.use_critique_head:
            self.critique_head = CritiqueHead(self.config)

        # Track prompt offset (number of prepended soft prompts)
        self.prompt_offset = self.transformer.embeddings.n_reasoning_prompts

    def forward(
        self,
        idx,
        targets=None,
        return_value: bool = False,
        return_critique: bool = False,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional value and critique outputs.

        Args:
            idx: Input token indices of shape (B, T)
            targets: Target token indices of shape (B, T)
            return_value: Return value estimates for RLHF
            return_critique: Return critique logits for ReFL
            return_all: Return all outputs (logits, loss, value, critique, hidden_states)

        Returns:
            Dictionary with logits, loss, value, critique, and/or hidden_states
        """
        b, t = idx.size()
        total_length = t + self.prompt_offset
        assert total_length <= self.config.block_size, \
            f"Sequence length {total_length} exceeds block size {self.config.block_size}"

        # Get embeddings
        x = self.transformer.embeddings(idx)

        # Store hidden states if requested
        hidden_states = [] if return_all else None

        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)
            if return_all:
                hidden_states.append(x.clone())

        # Final layer norm
        x = self.transformer.ln_f(x)

        outputs = {}

        # Slice off the soft prompts for output
        x_main = x[:, self.prompt_offset:, :]  # (B, T, n_embd)

        # Language modeling
        if targets is not None:
            # Training mode: compute logits for all positions
            logits = self.lm_head(x_main)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            outputs['logits'] = logits
            outputs['loss'] = loss
        else:
            # Inference mode: only compute logits for the last position
            logits = self.lm_head(x_main[:, [-1], :])
            outputs['logits'] = logits
            outputs['loss'] = None

        # Value head (for RLHF)
        if return_value and self.config.use_value_head:
            value = self.value_head(x_main).squeeze(-1)  # (B, T) or (B,)
            outputs['value'] = value

        # Critique head (for ReFL)
        if return_critique and self.config.use_critique_head:
            critique_logits = self.critique_head(x_main)
            outputs['critique_logits'] = critique_logits

        # Hidden states
        if return_all:
            outputs['hidden_states'] = hidden_states
            outputs['last_hidden_state'] = x_main

        return outputs

    def generate_reasoning(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_k: int = 40,
        use_cot: bool = True,
        stop_tokens: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, str]:
        """
        Generate reasoning and answer for a given prompt.

        Args:
            prompt: Input token indices
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            use_cot: Whether to use chain-of-thought reasoning
            stop_tokens: Tokens that stop generation

        Returns:
            Tuple of (generated tokens, reasoning string)
        """
        self.eval()

        # Add CoT trigger token if requested
        if use_cot:
            cot_trigger = torch.full(
                (prompt.size(0), 1),
                self.config.cot_token_id,
                dtype=torch.long,
                device=prompt.device
            )
            prompt = torch.cat([prompt, cot_trigger], dim=1)

        generated = prompt.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = generated
                if generated.size(1) + self.prompt_offset > self.config.block_size:
                    idx_cond = generated[:, -self.config.block_size + self.prompt_offset:]

                outputs = self(idx_cond)
                logits = outputs['logits'][:, -1, :] / temperature

                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, idx_next), dim=1)

                # Check for stop tokens
                if stop_tokens and (idx_next.item() in stop_tokens):
                    break

        return generated

    def compute_policy_loss(
        self,
        idx: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_ratio: float = 0.2
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO-style policy loss for RLHF.

        Args:
            idx: Input token indices
            actions: Action tokens
            old_log_probs: Log probs from old policy
            advantages: Advantage estimates
            clip_ratio: PPO clipping parameter

        Returns:
            Tuple of (loss, info_dict)
        """
        outputs = self(idx, return_all=True)
        logits = outputs['logits']

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # PPO loss
        ratio = torch.exp(action_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Approximate KL divergence
        approx_kl = (old_log_probs - action_log_probs).mean().detach()
        clip_frac = (abs(ratio - 1) > clip_ratio).float().mean()

        info = {
            'policy_loss': policy_loss.item(),
            'approx_kl': approx_kl.item(),
            'clip_frac': clip_frac.item()
        }

        return policy_loss, info

    def compute_value_loss(
        self,
        idx: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value function loss for RLHF.

        Args:
            idx: Input token indices
            returns: Return values

        Returns:
            Value loss
        """
        outputs = self(idx, return_value=True)
        values = outputs['value']

        return F.mse_loss(values, returns)

    def compute_reflect_loss(
        self,
        idx: torch.Tensor,
        critique: torch.Tensor,
        critique_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ReFL (Reflection) loss using language feedback.

        Args:
            idx: Input token indices
            critique: Critique token indices
            critique_targets: Target critique tokens

        Returns:
            Critique loss
        """
        outputs = self(idx, return_critique=True)
        critique_logits = outputs['critique_logits']

        return F.cross_entropy(
            critique_logits.view(-1, critique_logits.size(-1)),
            critique_targets.view(-1),
            ignore_index=-1
        )

    @classmethod
    def from_pretrained(cls, model_type: str, **kwargs):
        """Load from pretrained GPT-2 and initialize for reasoning."""
        from .gpt import GPT

        # Create base GPT model
        base_model = GPT.from_pretrained(model_type, override_args=kwargs)

        # Extract config
        config_dict = {
            'n_layer': base_model.config.n_layer,
            'n_head': base_model.config.n_head,
            'n_embd': base_model.config.n_embd,
            'block_size': base_model.config.block_size,
            'vocab_size': base_model.config.vocab_size,
            'dropout': base_model.config.dropout,
            'bias': base_model.config.bias,
        }
        config_dict.update(kwargs)

        # Create reasoning model
        config = ReasoningModelConfig(**config_dict)
        model = cls(config)

        # Copy compatible weights
        state_dict = base_model.state_dict()
        model_state_dict = model.state_dict()

        for key in state_dict:
            if key in model_state_dict and state_dict[key].shape == model_state_dict[key].shape:
                model_state_dict[key].copy_(state_dict[key])

        return model


class ReasoningModelWithKL(ReasoningModel):
    """
    Reasoning Model with KL divergence constraint for RLHF.
    Adds a reference model for KL penalty computation.
    """

    def __init__(self, config: ReasoningModelConfig):
        super().__init__(config)
        # Reference model (frozen) for KL penalty
        self.reference_model = None

    def set_reference_model(self, reference_model: ReasoningModel):
        """Set the reference model for KL penalty computation."""
        self.reference_model = reference_model
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def compute_kl_penalty(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence penalty against reference model.

        Args:
            idx: Input token indices

        Returns:
            KL penalty
        """
        if self.reference_model is None:
            return torch.tensor(0.0, device=idx.device)

        with torch.no_grad():
            ref_outputs = self.reference_model(idx)
            ref_logits = ref_outputs['logits']

        outputs = self(idx)
        logits = outputs['logits']

        # Compute KL divergence
        log_p = F.log_softmax(logits, dim=-1)
        log_q = F.log_softmax(ref_logits, dim=-1)

        kl = torch.exp(log_q) * (log_q - log_p)
        return kl.sum(dim=-1).mean()


# Convenience functions
def create_reasoning_model(**kwargs):
    """Create a reasoning model with optional config overrides."""
    config = ReasoningModelConfig(**kwargs)
    model = ReasoningModel(config)
    return model


def create_reasoning_model_from_pretrained(model_type: str, **kwargs):
    """Create a reasoning model from pretrained GPT-2 weights."""
    return ReasoningModel.from_pretrained(model_type, **kwargs)
