"""
Critique Model for ReFL (Reinforcement Fine-tuning with Language Feedback).

This module implements a critique model that provides natural language feedback
on model outputs, which can then be used to improve the model through ReFL.

References:
- "Fine-tuning with Language Feedback is (Almost) All You Need"
- "Constitutional AI: Harmlessness from AI Feedback"
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import BaseConfig, ModelRegistry
from .common import TransformerBlock, LayerNorm, Embeddings


@dataclass
class CritiqueModelConfig(BaseConfig):
    """Configuration for Critique Model."""
    # Base model config
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 2048  # Input + output
    vocab_size: int = 50304
    dropout: float = 0.1
    bias: bool = True

    # Critique-specific config
    critique_length: int = 256  # Maximum length of critique
    critique_prompt: str = "Critique the following response:\n"

    # Training config
    use_constrained_decoding: bool = True  # Constrain critique to be helpful
    use_aspect_heads: bool = True  # Separate heads for different critique aspects

    # Critique aspects (multi-dimensional critique)
    critique_aspects: List[str] = field(default_factory=lambda: [
        'accuracy', 'clarity', 'safety', 'helpfulness', 'reasoning'
    ])


class AspectHead(nn.Module):
    """Head for predicting critique aspect scores."""

    def __init__(self, n_embd: int, hidden_dim: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """Return aspect score."""
        # x shape: (B, T, n_embd) -> pooled to (B, n_embd)
        pooled = x.mean(dim=1)
        return self.layers(pooled).squeeze(-1)  # (B,)


@ModelRegistry.register('critique-model')
class CritiqueModel(nn.Module):
    """
    Critique Model for ReFL.

    Generates natural language critiques of model outputs that can be used
    as feedback for improving the model.

    Usage:
        >>> model = CritiqueModel(config)
        >>> critique_tokens = model.generate(prompt_tokens, response_tokens)
        >>> # Use critique to improve the original model
    """

    def __init__(self, config: CritiqueModelConfig):
        super().__init__()
        self.config = config

        # Transformer backbone
        self.transformer = nn.ModuleDict({
            'embeddings': Embeddings(config),
            'h': nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            'ln_f': LayerNorm(config.n_embd, bias=config.bias),
        })

        # Language modeling head for critique generation
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.embeddings.wte.weight = self.lm_head.weight

        # Aspect heads for multi-dimensional critique
        if config.use_aspect_heads:
            self.aspect_heads = nn.ModuleDict({
                aspect: AspectHead(config.n_embd)
                for aspect in config.critique_aspects
            })

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
        return_aspect_scores: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            idx: Input token indices (typically prompt + response + critique so far)
            return_aspect_scores: Return aspect scores if True

        Returns:
            Dictionary with logits and optionally aspect scores
        """
        b, t = idx.size()

        # Embeddings
        x = self.transformer.embeddings(idx)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm
        x = self.transformer.ln_f(x)

        outputs = {}

        # Language modeling
        logits = self.lm_head(x)
        outputs['logits'] = logits

        # Aspect scores (evaluated on the last token)
        if return_aspect_scores and self.config.use_aspect_heads:
            last_hidden = x[:, [-1], :]  # (B, 1, n_embd)
            aspect_scores = {
                aspect: head(last_hidden).squeeze(-1)
                for aspect, head in self.aspect_heads.items()
            }
            outputs['aspect_scores'] = aspect_scores

        return outputs

    def generate_critique(
        self,
        prompt_idx: torch.Tensor,
        response_idx: torch.Tensor,
        max_length: int = None,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        num_return_sequences: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate a critique for a given response.

        Args:
            prompt_idx: Original prompt tokens
            response_idx: Model response tokens to critique
            max_length: Maximum length of critique
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            num_return_sequences: Number of critiques to generate

        Returns:
            Tuple of (critique_tokens, aspect_scores)
        """
        self.eval()

        if max_length is None:
            max_length = self.config.critique_length

        device = prompt_idx.device

        # Combine prompt + response as context
        context = torch.cat([prompt_idx, response_idx], dim=1)

        # Add critique prompt if possible (simplified here)
        # In practice, you'd tokenize critique_prompt

        critiques = []
        all_aspect_scores = []

        with torch.no_grad():
            for _ in range(num_return_sequences):
                generated = context.clone()

                for _ in range(max_length):
                    # Truncate if needed
                    if generated.size(1) >= self.config.block_size:
                        generated = generated[:, -self.config.block_size:]

                    outputs = self(generated, return_aspect_scores=True)
                    logits = outputs['logits'][:, -1, :] / temperature

                    # Top-k filtering
                    if top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('Inf')

                    # Top-p (nucleus) filtering
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
                    generated = torch.cat([generated, idx_next], dim=1)

                    # Check for EOS (simplified - use actual EOS token in practice)
                    if idx_next.item() == 50256:  # GPT-2 EOS
                        break

                critiques.append(generated)

                # Get aspect scores for the full critique
                final_outputs = self(generated, return_aspect_scores=True)
                all_aspect_scores.append(final_outputs.get('aspect_scores', {}))

        return critiques[0], all_aspect_scores[0]

    def compute_critique_loss(
        self,
        prompt_idx: torch.Tensor,
        response_idx: torch.Tensor,
        critique_idx: torch.Tensor,
        aspect_targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the loss for training the critique model.

        Args:
            prompt_idx: Prompt tokens
            response_idx: Response tokens
            critique_idx: Target critique tokens
            aspect_targets: Optional target aspect scores

        Returns:
            Tuple of (loss, info_dict)
        """
        # Combine inputs
        combined_idx = torch.cat([prompt_idx, response_idx, critique_idx], dim=1)

        # Forward pass
        outputs = self(combined_idx, return_aspect_scores=True)

        # Language modeling loss
        logits = outputs['logits']

        # Create targets (only for critique part)
        batch_size = prompt_idx.size(0)
        prompt_len = prompt_idx.size(1) + response_idx.size(1)
        critique_len = critique_idx.size(1)

        # Shift for causal LM
        shift_logits = logits[:, prompt_len:-1, :].contiguous()
        shift_labels = critique_idx[:, 1:].contiguous()

        # Compute loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-1
        )

        info = {
            'lm_loss': loss.item(),
            'critique_tokens': critique_len
        }

        # Add aspect loss if targets provided
        if aspect_targets and self.config.use_aspect_heads:
            aspect_loss = 0.0
            aspect_count = 0

            # Get aspect scores from the critique end
            for aspect, target_score in aspect_targets.items():
                if aspect in outputs['aspect_scores']:
                    predicted_score = outputs['aspect_scores'][aspect]
                    aspect_loss += F.mse_loss(predicted_score, target_score)
                    aspect_count += 1

            if aspect_count > 0:
                aspect_loss = aspect_loss / aspect_count
                loss = loss + 0.1 * aspect_loss  # Weighted combination
                info['aspect_loss'] = aspect_loss.item()

        return loss, info

    def evaluate_response(
        self,
        prompt_idx: torch.Tensor,
        response_idx: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate a response using aspect scores.

        Args:
            prompt_idx: Prompt tokens
            response_idx: Response tokens

        Returns:
            Dictionary of aspect scores
        """
        self.eval()

        with torch.no_grad():
            combined_idx = torch.cat([prompt_idx, response_idx], dim=1)
            outputs = self(combined_idx, return_aspect_scores=True)

            if 'aspect_scores' in outputs:
                return {
                    aspect: score.item()
                    for aspect, score in outputs['aspect_scores'].items()
                }
            else:
                return {}

    @classmethod
    def from_pretrained(cls, model_type: str, **kwargs):
        """Load from pretrained GPT-2 and initialize as critique model."""
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

        # Create critique model
        config = CritiqueModelConfig(**config_dict)
        model = cls(config)

        # Copy compatible weights
        state_dict = base_model.state_dict()
        model_state_dict = model.state_dict()

        for key in state_dict:
            if key in model_state_dict and state_dict[key].shape == model_state_dict[key].shape:
                model_state_dict[key].copy_(state_dict[key])

        return model


class ConstitutionalCritiqueModel(CritiqueModel):
    """
    Constitutional Critique Model.

    Uses constitutional principles to provide critiques that align with
    specified principles (e.g., helpfulness, harmlessness, honesty).

    Reference: "Constitutional AI: Harmlessness from AI Feedback"
    """

    def __init__(self, config: CritiqueModelConfig):
        super().__init__(config)

        # Constitutional principles
        self.principles = {
            'harmlessness': "Please critique any harmful or dangerous content.",
            'honesty': "Please critique any false or misleading information.",
            'helpfulness': "Please critique whether the response is helpful and relevant.",
        }

        # Principle embeddings (learnable)
        self.principle_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, 1, config.n_embd) * 0.02)
            for name in self.principles.keys()
        })

    def forward_with_principle(
        self,
        idx: torch.Tensor,
        principle: str = 'harmlessness'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with constitutional principle conditioning.

        Args:
            idx: Input token indices
            principle: Which principle to condition on

        Returns:
            Dictionary with logits
        """
        if principle not in self.principle_embeddings:
            raise ValueError(f"Unknown principle: {principle}")

        # Get standard embeddings
        x = self.transformer.embeddings(idx)

        # Add principle embedding at the start
        principle_emb = self.principle_embeddings[principle].expand(x.size(0), -1, -1)
        x = torch.cat([principle_emb, x], dim=1)

        # Rest of forward pass
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        # Remove principle embedding from output
        x = x[:, 1:, :]

        logits = self.lm_head(x)
        return {'logits': logits}

    def critique_with_principles(
        self,
        prompt_idx: torch.Tensor,
        response_idx: torch.Tensor
    ) -> Dict[str, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Generate critiques conditioned on different principles.

        Args:
            prompt_idx: Prompt tokens
            response_idx: Response tokens

        Returns:
            Dictionary mapping principle names to (critique, aspect_scores)
        """
        results = {}

        for principle in self.principles.keys():
            # For each principle, generate a critique
            # (Simplified - full implementation would call generate_critique
            # with principle conditioning)
            results[principle] = None  # Placeholder

        return results


# Convenience functions
def create_critique_model(**kwargs):
    """Create a critique model with optional config overrides."""
    config = CritiqueModelConfig(**kwargs)
    model = CritiqueModel(config)
    return model


def create_critique_model_from_pretrained(model_type: str, **kwargs):
    """Create a critique model from pretrained GPT-2 weights."""
    return CritiqueModel.from_pretrained(model_type, **kwargs)


def create_constitutional_critique_model(**kwargs):
    """Create a constitutional critique model."""
    config = CritiqueModelConfig(**kwargs)
    model = ConstitutionalCritiqueModel(config)
    return model
