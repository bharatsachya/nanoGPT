"""
Reward Model for RLHF (Reinforcement Learning from Human Feedback).

This module implements a reward model that can be trained on human preference data
to provide scalar rewards for language model outputs.

References:
- "Training language models to follow instructions with human feedback"
- "Learning to Summarize with Human Feedback"
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import BaseConfig, ModelRegistry
from .common import TransformerBlock, LayerNorm, Embeddings, MultiHeadAttention


@dataclass
class RewardModelConfig(BaseConfig):
    """Configuration for Reward Model."""
    # Base model config
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 2048  # May need longer context for comparing responses
    vocab_size: int = 50304
    dropout: float = 0.1
    bias: bool = True

    # Reward-specific config
    reward_head_hidden_dim: int = 512  # Hidden dimension of reward head
    reward_head_dropout: float = 0.1

    # Training config
    use_contrastive: bool = True  # Use contrastive loss (ranking)
    temperature: float = 1.0  # Temperature for contrastive loss


class RewardHead(nn.Module):
    """Reward prediction head."""

    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.n_embd, config.reward_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.reward_head_dropout),
            nn.Linear(config.reward_head_hidden_dim, config.reward_head_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.reward_head_dropout),
            nn.Linear(config.reward_head_hidden_dim // 2, 1)
        )

    def forward(self, x):
        """Return scalar reward for each sequence."""
        # x shape: (B, T, n_embd)
        # We use the mean pooling over the sequence dimension
        pooled = x.mean(dim=1)  # (B, n_embd)
        return self.layers(pooled).squeeze(-1)  # (B,)


@ModelRegistry.register('reward-model')
class RewardModel(nn.Module):
    """
    Reward Model for RLHF.

    Trains on human preference data (chosen vs rejected responses) to predict
    which response is better. Outputs a scalar reward value.

    Usage:
        >>> model = RewardModel(config)
        >>> reward_chosen = model(prompt_tokens + chosen_tokens)
        >>> reward_rejected = model(prompt_tokens + rejected_tokens)
        >>> loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected))
    """

    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.config = config

        # Transformer backbone
        self.transformer = nn.ModuleDict({
            'embeddings': Embeddings(config),
            'h': nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            'ln_f': LayerNorm(config.n_embd, bias=config.bias),
        })

        # Reward head
        self.reward_head = RewardHead(config)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass - returns scalar reward for each input sequence.

        Args:
            idx: Input token indices of shape (B, T)
            attention_mask: Optional mask of shape (B, T) for padding tokens

        Returns:
            Reward values of shape (B,)
        """
        b, t = idx.size()

        # Embeddings
        x = self.transformer.embeddings(idx)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Create mask for transformer
            mask = attention_mask[:, None, None, :]  # (B, 1, 1, T)
            x = x * mask

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm
        x = self.transformer.ln_f(x)

        # Compute reward
        reward = self.reward_head(x)

        return reward

    def compute_reward_loss(
        self,
        chosen_idx: torch.Tensor,
        rejected_idx: torch.Tensor,
        chosen_attention_mask: Optional[torch.Tensor] = None,
        rejected_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the preference loss (ranking loss).

        Args:
            chosen_idx: Chosen (better) response tokens
            rejected_idx: Rejected (worse) response tokens
            chosen_attention_mask: Attention mask for chosen
            rejected_attention_mask: Attention mask for rejected

        Returns:
            Tuple of (loss, info_dict)
        """
        # Compute rewards
        reward_chosen = self(chosen_idx, chosen_attention_mask)
        reward_rejected = self(rejected_idx, rejected_attention_mask)

        # Margin loss (sigmoid)
        # We want reward_chosen > reward_rejected
        logits = reward_chosen - reward_rejected
        loss = -F.logsigmoid(logits).mean()

        # Accuracy metrics
        with torch.no_grad():
            accuracy = (reward_chosen > reward_rejected).float().mean()
            reward_margin = (reward_chosen - reward_rejected).mean()

        info = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'reward_margin': reward_margin.item(),
            'reward_chosen': reward_chosen.mean().item(),
            'reward_rejected': reward_rejected.mean().item()
        }

        return loss, info

    def compute_contrastive_loss(
        self,
        anchor_idx: torch.Tensor,
        positive_idx: torch.Tensor,
        negative_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss with anchor-positive-negative triplet.

        Args:
            anchor_idx: Anchor (prompt) tokens
            positive_idx: Positive (chosen) tokens
            negative_idx: Negative (rejected) tokens

        Returns:
            Contrastive loss
        """
        reward_anchor = self(anchor_idx)
        reward_positive = self(positive_idx)
        reward_negative = self(negative_idx)

        # Triplet loss
        pos_distance = F.pairwise_distance(
            reward_anchor.unsqueeze(1), reward_positive.unsqueeze(1)
        )
        neg_distance = F.pairwise_distance(
            reward_anchor.unsqueeze(1), reward_negative.unsqueeze(1)
        )

        loss = F.relu(pos_distance - neg_distance + 1.0).mean()
        return loss

    @classmethod
    def from_pretrained(cls, model_type: str, **kwargs):
        """Load from pretrained GPT-2 and initialize as reward model."""
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

        # Create reward model
        config = RewardModelConfig(**config_dict)
        model = cls(config)

        # Copy compatible weights (exclude reward head)
        state_dict = base_model.state_dict()
        model_state_dict = model.state_dict()

        for key in state_dict:
            if key in model_state_dict and state_dict[key].shape == model_state_dict[key].shape:
                model_state_dict[key].copy_(state_dict[key])

        return model


class PairwiseRewardModel(RewardModel):
    """
    Pairwise Reward Model that directly compares two sequences.

    Instead of predicting separate rewards, this model takes pairs
    and directly predicts which is better.
    """

    def __init__(self, config: RewardModelConfig):
        super().__init__(config)

        # Add a pairwise comparison head
        self.pairwise_head = nn.Sequential(
            nn.Linear(config.n_embd * 2, config.reward_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.reward_head_dropout),
            nn.Linear(config.reward_head_hidden_dim, 1)
        )

    def forward_pairwise(
        self,
        idx1: torch.Tensor,
        idx2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compare two sequences directly.

        Args:
            idx1: First sequence tokens
            idx2: Second sequence tokens

        Returns:
            Logits indicating which is better (positive = first is better)
        """
        # Get embeddings for both sequences
        x1 = self._encode(idx1)
        x2 = self._encode(idx2)

        # Concatenate and compare
        combined = torch.cat([x1, x2], dim=-1)
        logits = self.pairwise_head(combined).squeeze(-1)

        return logits

    def _encode(self, idx: torch.Tensor) -> torch.Tensor:
        """Encode a sequence to a fixed-size representation."""
        x = self.transformer.embeddings(idx)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        return x.mean(dim=1)  # Mean pooling

    def compute_pairwise_loss(
        self,
        chosen_idx: torch.Tensor,
        rejected_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute pairwise comparison loss.

        Args:
            chosen_idx: Better response tokens
            rejected_idx: Worse response tokens

        Returns:
            Tuple of (loss, info_dict)
        """
        logits = self.forward_pairwise(chosen_idx, rejected_idx)

        # Binary cross entropy loss
        # We want logits > 0 (chosen is better)
        labels = torch.ones(logits.size(0), device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        with torch.no_grad():
            accuracy = (logits > 0).float().mean()

        info = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'logits_mean': logits.mean().item()
        }

        return loss, info


class EnsembleRewardModel(nn.Module):
    """
    Ensemble of reward models for more robust reward prediction.

    Uses multiple reward models and aggregates their predictions.
    """

    def __init__(self, configs: List[RewardModelConfig], aggregation: str = 'mean'):
        super().__init__()
        self.aggregation = aggregation

        self.models = nn.ModuleList([
            RewardModel(config) for config in configs
        ])

    def forward(self, idx: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through all models and aggregate."""
        rewards = [model(idx, **kwargs) for model in self.models]
        rewards = torch.stack(rewards, dim=0)  # (n_models, B)

        if self.aggregation == 'mean':
            return rewards.mean(dim=0)
        elif self.aggregation == 'median':
            return rewards.median(dim=0).values
        elif self.aggregation == 'max':
            return rewards.max(dim=0).values
        else:
            return rewards.mean(dim=0)

    def compute_individual_rewards(self, idx: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """Get rewards from each individual model."""
        return [model(idx, **kwargs) for model in self.models]


# Convenience functions
def create_reward_model(**kwargs):
    """Create a reward model with optional config overrides."""
    config = RewardModelConfig(**kwargs)
    model = RewardModel(config)
    return model


def create_reward_model_from_pretrained(model_type: str, **kwargs):
    """Create a reward model from pretrained GPT-2 weights."""
    return RewardModel.from_pretrained(model_type, **kwargs)


def create_pairwise_reward_model(**kwargs):
    """Create a pairwise reward model."""
    config = RewardModelConfig(**kwargs)
    model = PairwiseRewardModel(config)
    return model
