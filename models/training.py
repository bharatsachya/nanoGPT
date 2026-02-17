"""
Training utilities for RLHF and ReFL.

This module provides training loops, loss computations, and utilities for:
- RLHF (Reinforcement Learning from Human Feedback) with PPO
- ReFL (Reinforcement Fine-tuning with Language Feedback)
- Reward model training
- Critique model training

References:
- "Training language models to follow instructions with human feedback"
- "Fine-tuning with Language Feedback is (Almost) All You Need"
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable, Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from .reward import RewardModel
from .critique import CritiqueModel
from .reasoning import ReasoningModel


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # PPO hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 256
    minibatch_size: int = 64
    ppo_epochs: int = 4
    clip_range: float = 0.2
    target_kl: float = 0.01
    kl_coef: float = 0.2

    # Reward normalization
    normalize_reward: bool = True
    reward_baseline: bool = True

    # Generation
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 50

    # Early stopping
    early_stopping: bool = True
    early_stopping_threshold: float = 1.5


@dataclass
class ReFLConfig:
    """Configuration for ReFL training."""
    # ReFL hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_new_tokens: int = 256

    # Critique weighting
    critique_weight: float = 0.5
    self_consistency_weight: float = 0.3
    critique_loss_weight: float = 0.2

    # Sampling for critique
    critique_temperature: float = 0.8
    num_critique_samples: int = 4


class PPOTrainer:
    """
    PPO Trainer for RLHF.

    Implements Proximal Policy Optimization with a reward model for
    reinforcement learning from human feedback.
    """

    def __init__(
        self,
        policy_model: ReasoningModel,
        reward_model: RewardModel,
        config: PPOConfig,
        ref_policy: Optional[ReasoningModel] = None
    ):
        self.policy = policy_model
        self.reward_model = reward_model
        self.config = config
        self.ref_policy = ref_policy

        # Freeze reward model
        for param in self.reward_model.parameters():
            param.requires_grad = False
        self.reward_model.eval()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate
        )

        # Training statistics
        self.stats = {
            'loss': [],
            'policy_loss': [],
            'value_loss': [],
            'approx_kl': [],
            'clip_frac': [],
            'rewards': []
        }

    def generate_rollouts(
        self,
        prompts: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate rollouts using the current policy.

        Args:
            prompts: Prompt token indices
            attention_mask: Optional attention mask

        Returns:
            Dictionary with generated sequences, log_probs, rewards, etc.
        """
        self.policy.eval()

        batch_size = prompts.size(0)
        device = prompts.device

        with torch.no_grad():
            # Generate responses
            generated = prompts.clone()
            all_log_probs = []

            for _ in range(self.config.max_new_tokens):
                # Forward pass
                outputs = self.policy(generated)
                logits = outputs['logits'][:, -1, :]
                logits = logits / self.config.temperature

                # Top-k filtering
                if self.config.top_k > 0:
                    v, _ = torch.topk(logits, min(self.config.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # Sample
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                # Store log probs
                action_log_prob = log_probs.gather(-1, idx_next)
                all_log_probs.append(action_log_prob)

                generated = torch.cat([generated, idx_next], dim=1)

            # Compute rewards
            rewards = self.reward_model(generated)

            # Normalize rewards
            if self.config.normalize_reward:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # Compute advantages (baseline subtraction)
            if self.config.reward_baseline:
                advantages = rewards - rewards.mean()
            else:
                advantages = rewards

            # Compute old log probs from reference policy if available
            if self.ref_policy is not None:
                with torch.no_grad():
                    ref_outputs = self.ref_policy(generated)
                    ref_logits = ref_outputs['logits']
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

                    # Get log probs for generated tokens
                    old_log_probs = []
                    gen_tokens = generated[:, prompts.size(1):]
                    for i, token in enumerate(gen_tokens.unbind(-1)):
                        old_log_probs.append(ref_log_probs[:, i, token])
                    old_log_probs = torch.stack(old_log_probs, dim=-1)
            else:
                old_log_probs = torch.stack(all_log_probs, dim=-1)

        return {
            'prompts': prompts,
            'generated': generated,
            'log_probs': torch.stack(all_log_probs, dim=-1),
            'old_log_probs': old_log_probs,
            'rewards': rewards,
            'advantages': advantages
        }

    def compute_ppo_loss(
        self,
        rollouts: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO loss.

        Args:
            rollouts: Dictionary from generate_rollouts

        Returns:
            Tuple of (total_loss, info_dict)
        """
        prompts = rollouts['prompts']
        generated = rollouts['generated']
        old_log_probs = rollouts['old_log_probs']
        advantages = rollouts['advantages']

        # Forward pass with current policy
        outputs = self.policy(generated)
        logits = outputs['logits']

        # Compute new log probs
        log_probs = F.log_softmax(logits, dim=-1)
        gen_tokens = generated[:, prompts.size(1):]

        new_log_probs = []
        for i, token in enumerate(gen_tokens.unbind(-1)):
            new_log_probs.append(log_probs[:, i, token])
        new_log_probs = torch.stack(new_log_probs, dim=-1)

        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # PPO loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Approximate KL divergence
        approx_kl = (old_log_probs - new_log_probs).mean()

        # Clip fraction
        clip_frac = (abs(ratio - 1) > self.config.clip_range).float().mean()

        # Total loss
        loss = policy_loss

        info = {
            'policy_loss': policy_loss.item(),
            'approx_kl': approx_kl.item(),
            'clip_frac': clip_frac.item(),
            'mean_reward': rollouts['rewards'].mean().item(),
            'mean_advantage': advantages.mean().item()
        }

        return loss, info

    def train_step(self, prompts: torch.Tensor) -> Dict[str, float]:
        """
        Perform one PPO training step.

        Args:
            prompts: Batch of prompts

        Returns:
            Training statistics
        """
        # Generate rollouts
        rollouts = self.generate_rollouts(prompts)

        # Normalize advantages
        advantages = rollouts['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        rollouts['advantages'] = advantages

        total_info = {}

        # PPO epochs
        for epoch in range(self.config.ppo_epochs):
            # Mini-batch updates
            batch_size = prompts.size(0)
            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, self.config.minibatch_size):
                end = min(start + self.config.minibatch_size, batch_size)
                minibatch_indices = indices[start:end]

                # Create minibatch
                minibatch_rollouts = {
                    k: v[minibatch_indices] if isinstance(v, torch.Tensor) else v
                    for k, v in rollouts.items()
                }

                # Compute loss
                loss, info = self.compute_ppo_loss(minibatch_rollouts)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

                total_info.update(info)

                # Early stopping based on KL
                if self.config.early_stopping and info['approx_kl'] > self.config.target_kl * 1.5:
                    break

        return total_info

    def train(
        self,
        dataloader: DataLoader,
        num_steps: int,
        callback: Optional[Callable] = None
    ) -> Dict[str, List[float]]:
        """
        Train the policy with PPO.

        Args:
            dataloader: DataLoader with prompts
            num_steps: Number of training steps
            callback: Optional callback function

        Returns:
            Training statistics
        """
        self.policy.train()

        step = 0
        while step < num_steps:
            for batch in dataloader:
                if step >= num_steps:
                    break

                prompts = batch['input_ids'].to(next(self.policy.parameters()).device)

                # Training step
                info = self.train_step(prompts)

                # Log stats
                for key, value in info.items():
                    if key in self.stats:
                        self.stats[key].append(value)

                step += 1

                if callback is not None:
                    callback(step, info)

        return self.stats


class ReFLTrainer:
    """
    ReFL Trainer for language feedback training.

    Implements training using natural language critiques from a critique model.
    """

    def __init__(
        self,
        model: ReasoningModel,
        critique_model: CritiqueModel,
        config: ReFLConfig
    ):
        self.model = model
        self.critique_model = critique_model
        self.config = config

        # Freeze critique model
        for param in self.critique_model.parameters():
            param.requires_grad = False
        self.critique_model.eval()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )

        # Training statistics
        self.stats = {
            'loss': [],
            'lm_loss': [],
            'critique_loss': [],
            'self_consistency_loss': []
        }

    def generate_with_critique(
        self,
        prompts: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Generate responses with critiques.

        Args:
            prompts: Prompt tokens

        Returns:
            Dictionary with generated responses and critiques
        """
        self.model.eval()
        self.critique_model.eval()

        with torch.no_grad():
            # Generate initial response
            generated = self.model.generate_reasoning(
                prompts,
                max_new_tokens=self.config.max_new_tokens,
                temperature=0.8
            )

            # Generate critique
            critique, aspect_scores = self.critique_model.generate_critique(
                prompts,
                generated[:, prompts.size(1):],
                max_length=128
            )

            # Generate improved response based on critique
            # (In practice, you'd concatenate critique and regenerate)
            improved = self.model.generate_reasoning(
                torch.cat([prompts, critique[:, prompts.size(1):]], dim=1),
                max_new_tokens=self.config.max_new_tokens,
                temperature=0.8
            )

        return {
            'prompts': prompts,
            'generated': generated,
            'critique': critique,
            'improved': improved,
            'aspect_scores': aspect_scores
        }

    def compute_refl_loss(
        self,
        outputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute ReFL loss.

        Args:
            outputs: Dictionary from generate_with_critique

        Returns:
            Tuple of (loss, info_dict)
        """
        prompts = outputs['prompts']
        generated = outputs['generated']
        critique = outputs['critique']
        improved = outputs['improved']

        # Standard language modeling loss on improved response
        outputs_improved = self.model(torch.cat([prompts, improved[:, prompts.size(1):]], dim=1))
        lm_loss = outputs_improved['loss'] if outputs_improved['loss'] is not None else torch.tensor(0.0)

        # Self-consistency loss (encourage agreement)
        outputs_original = self.model(generated)
        outputs_improved_2 = self.model(improved)

        if outputs_original['logits'] is not None and outputs_improved_2['logits'] is not None:
            # KL divergence between original and improved
            log_p_original = F.log_softmax(outputs_original['logits'], dim=-1)
            log_p_improved = F.log_softmax(outputs_improved_2['logits'], dim=-1)

            # Only compare valid positions
            min_len = min(log_p_original.size(1), log_p_improved.size(1))
            kl_div = F.kl_div(
                log_p_original[:, :min_len, :].reshape(-1, log_p_original.size(-1)),
                log_p_improved[:, :min_len, :].reshape(-1, log_p_improved.size(-1)).exp(),
                reduction='batchmean'
            )
            self_consistency_loss = kl_div * self.config.self_consistency_weight
        else:
            self_consistency_loss = torch.tensor(0.0)

        # Critique loss (optional - train critique model too)
        critique_loss = torch.tensor(0.0)

        # Total loss
        loss = lm_loss + self_consistency_loss + critique_loss * self.config.critique_loss_weight

        info = {
            'lm_loss': lm_loss.item(),
            'self_consistency_loss': self_consistency_loss.item(),
            'critique_loss': critique_loss.item(),
            'total_loss': loss.item()
        }

        return loss, info

    def train_step(self, prompts: torch.Tensor) -> Dict[str, float]:
        """
        Perform one ReFL training step.

        Args:
            prompts: Batch of prompts

        Returns:
            Training statistics
        """
        # Generate with critique
        outputs = self.generate_with_critique(prompts)

        # Compute loss
        loss, info = self.compute_refl_loss(outputs)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return info

    def train(
        self,
        dataloader: DataLoader,
        num_steps: int,
        callback: Optional[Callable] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model with ReFL.

        Args:
            dataloader: DataLoader with prompts
            num_steps: Number of training steps
            callback: Optional callback function

        Returns:
            Training statistics
        """
        self.model.train()

        step = 0
        while step < num_steps:
            for batch in dataloader:
                if step >= num_steps:
                    break

                prompts = batch['input_ids'].to(next(self.model.parameters()).device)

                # Training step
                info = self.train_step(prompts)

                # Log stats
                for key, value in info.items():
                    if key in self.stats:
                        self.stats[key].append(value)

                step += 1

                if callback is not None:
                    callback(step, info)

        return self.stats


class RewardModelTrainer:
    """Trainer for reward model."""

    def __init__(
        self,
        reward_model: RewardModel,
        learning_rate: float = 1e-5,
        batch_size: int = 16
    ):
        self.reward_model = reward_model
        self.batch_size = batch_size

        self.optimizer = torch.optim.AdamW(
            reward_model.parameters(),
            lr=learning_rate
        )

    def train_step(
        self,
        chosen: torch.Tensor,
        rejected: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Train the reward model on one batch of preference data.

        Args:
            chosen: Chosen (better) responses
            rejected: Rejected (worse) responses
            chosen_mask: Attention mask for chosen
            rejected_mask: Attention mask for rejected

        Returns:
            Training statistics
        """
        self.reward_model.train()

        # Compute loss
        loss, info = self.reward_model.compute_reward_loss(
            chosen, rejected, chosen_mask, rejected_mask
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
        self.optimizer.step()

        return info

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        callback: Optional[Callable] = None
    ) -> Dict[str, List[float]]:
        """Train the reward model."""
        stats = {key: [] for key in ['loss', 'accuracy', 'reward_margin']}

        for epoch in range(num_epochs):
            for batch in dataloader:
                chosen = batch['chosen'].to(next(self.reward_model.parameters()).device)
                rejected = batch['rejected'].to(next(self.reward_model.parameters()).device)
                chosen_mask = batch.get('chosen_mask', None)
                rejected_mask = batch.get('rejected_mask', None)

                if chosen_mask is not None:
                    chosen_mask = chosen_mask.to(next(self.reward_model.parameters()).device)
                if rejected_mask is not None:
                    rejected_mask = rejected_mask.to(next(self.reward_model.parameters()).device)

                info = self.train_step(chosen, rejected, chosen_mask, rejected_mask)

                for key in stats:
                    if key in info:
                        stats[key].append(info[key])

                if callback is not None:
                    callback(epoch, info)

        return stats


class CritiqueModelTrainer:
    """Trainer for critique model."""

    def __init__(
        self,
        critique_model: CritiqueModel,
        learning_rate: float = 1e-5,
        batch_size: int = 8
    ):
        self.critique_model = critique_model
        self.batch_size = batch_size

        self.optimizer = torch.optim.AdamW(
            critique_model.parameters(),
            lr=learning_rate
        )

    def train_step(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        critiques: torch.Tensor,
        aspect_targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Train the critique model on one batch.

        Args:
            prompts: Input prompts
            responses: Model responses
            critiques: Target critiques
            aspect_targets: Optional aspect scores

        Returns:
            Training statistics
        """
        self.critique_model.train()

        # Compute loss
        loss, info = self.critique_model.compute_critique_loss(
            prompts, responses, critiques, aspect_targets
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critique_model.parameters(), 1.0)
        self.optimizer.step()

        return info

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        callback: Optional[Callable] = None
    ) -> Dict[str, List[float]]:
        """Train the critique model."""
        stats = {key: [] for key in ['lm_loss', 'aspect_loss']}

        for epoch in range(num_epochs):
            for batch in dataloader:
                prompts = batch['prompts'].to(next(self.critique_model.parameters()).device)
                responses = batch['responses'].to(next(self.critique_model.parameters()).device)
                critiques = batch['critiques'].to(next(self.critique_model.parameters()).device)
                aspect_targets = batch.get('aspect_targets', None)

                if aspect_targets is not None:
                    aspect_targets = {
                        k: v.to(next(self.critique_model.parameters()).device)
                        for k, v in aspect_targets.items()
                    }

                info = self.train_step(prompts, responses, critiques, aspect_targets)

                for key in stats:
                    if key in info:
                        stats[key].append(info[key])

                if callback is not None:
                    callback(epoch, info)

        return stats


# Convenience functions
def create_ppo_trainer(
    policy_model: ReasoningModel,
    reward_model: RewardModel,
    **kwargs
) -> PPOTrainer:
    """Create a PPO trainer."""
    config = PPOConfig(**kwargs)
    return PPOTrainer(policy_model, reward_model, config)


def create_refl_trainer(
    model: ReasoningModel,
    critique_model: CritiqueModel,
    **kwargs
) -> ReFLTrainer:
    """Create a ReFL trainer."""
    config = ReFLConfig(**kwargs)
    return ReFLTrainer(model, critique_model, config)
