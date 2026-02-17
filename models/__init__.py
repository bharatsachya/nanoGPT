"""
Models package for nanoGPT.

A modular, interconnected model library with shared components and factory functions.

Includes:
- Base language models (GPT, MiniGPT)
- Modern LLMs (Mistral, LLaMA-style with RoPE, SwiGLU, GQA)
- DeepSeek-style Mixture of Experts models
- Reasoning models with CoT and RLHF/ReFL support
- Reward models for RLHF
- Critique models for language feedback
- Training utilities

Usage:
    >>> from models import create_model, list_models, create_mistral_model
    >>> model = create_mistral_model('mistral-7b')

    >>> # Available presets
    >>> from models import MISTRAL_PRESETS
    >>> print(list(MISTRAL_PRESETS.keys()))
    ['mistral-7b', 'llama2-7b', 'llama3-8b', 'tiny-1b', ...]
"""

# Base classes and registry
from .base import (
    BaseLanguageModel,
    BaseConfig,
    ModelRegistry,
    create_model as create_model_from_registry,
    get_model_config,
)

# Common components
from .common import (
    LayerNorm,
    RMSNorm,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    Embeddings,
    SwiGLU,
    RotaryEmbedding,
    apply_rotary_emb,
)

# Model implementations
from .mini_gpt import MiniGPT, MiniGPTConfig, create_mini_gpt, mini_gpt_from_pretrained
from .gpt import GPT, GPTConfig, create_gpt, gpt_from_pretrained

# RLHF/ReFL models
from .reasoning import (
    ReasoningModel,
    ReasoningModelConfig,
    ReasoningModelWithKL,
    create_reasoning_model,
    create_reasoning_model_from_pretrained,
)
from .reward import (
    RewardModel,
    RewardModelConfig,
    PairwiseRewardModel,
    EnsembleRewardModel,
    create_reward_model,
    create_reward_model_from_pretrained,
    create_pairwise_reward_model,
)
from .critique import (
    CritiqueModel,
    CritiqueModelConfig,
    ConstitutionalCritiqueModel,
    create_critique_model,
    create_critique_model_from_pretrained,
    create_constitutional_critique_model,
)

# DeepSeek-style MoE models
from .deepseek import (
    DeepSeekModel,
    DeepSeekCoderModel,
    DeepSeekConfig,
    Expert,
    SharedExpert,
    TopKRouter,
    SoftRouter,
    GumbelRouter,
    MoE,
    DeepSeekAttention,
    DeepSeekBlock,
    create_deepseek_model,
    create_deepseek_coder,
    create_deepseek_from_pretrained,
)

# Mistral / LLaMA-style models
from .mistral import (
    MistralModel,
    MistralConfig,
    MistralAttention,
    MistralMLP,
    MistralDecoderLayer,
    MistralForCausalLM,
    MistralForSequenceClassification,
    MistralForTokenClassification,
    repeat_kv,
    create_mistral_model,
    create_llama_model,
    create_custom_model,
    MODEL_PRESETS as MISTRAL_PRESETS,
)

# Training utilities
from .training import (
    PPOConfig,
    ReFLConfig,
    PPOTrainer,
    ReFLTrainer,
    RewardModelTrainer,
    CritiqueModelTrainer,
    create_ppo_trainer,
    create_refl_trainer,
)

# Factory functions
from .factory import (
    create_model,
    create_model_from_config,
    load_from_pretrained,
    list_models,
    get_model_info,
    compare_models,
    quick_mini,
    quick_gpt2,
    quick_small,
    MODEL_PRESETS,
)

__all__ = [
    # Base
    'BaseLanguageModel',
    'BaseConfig',
    'ModelRegistry',
    'get_model_config',

    # Common components
    'LayerNorm',
    'RMSNorm',
    'MultiHeadAttention',
    'FeedForward',
    'TransformerBlock',
    'Embeddings',
    'SwiGLU',
    'RotaryEmbedding',
    'apply_rotary_emb',

    # Models
    'MiniGPT',
    'MiniGPTConfig',
    'GPT',
    'GPTConfig',

    # RLHF/ReFL models
    'ReasoningModel',
    'ReasoningModelConfig',
    'ReasoningModelWithKL',
    'create_reasoning_model',
    'create_reasoning_model_from_pretrained',

    'RewardModel',
    'RewardModelConfig',
    'PairwiseRewardModel',
    'EnsembleRewardModel',
    'create_reward_model',
    'create_reward_model_from_pretrained',
    'create_pairwise_reward_model',

    'CritiqueModel',
    'CritiqueModelConfig',
    'ConstitutionalCritiqueModel',
    'create_critique_model',
    'create_critique_model_from_pretrained',
    'create_constitutional_critique_model',

    # DeepSeek models
    'DeepSeekModel',
    'DeepSeekCoderModel',
    'DeepSeekConfig',
    'Expert',
    'SharedExpert',
    'TopKRouter',
    'SoftRouter',
    'GumbelRouter',
    'MoE',
    'DeepSeekAttention',
    'DeepSeekBlock',
    'create_deepseek_model',
    'create_deepseek_coder',
    'create_deepseek_from_pretrained',

    # Mistral / LLaMA models
    'MistralModel',
    'MistralConfig',
    'MistralAttention',
    'MistralMLP',
    'MistralDecoderLayer',
    'MistralForCausalLM',
    'MistralForSequenceClassification',
    'MistralForTokenClassification',
    'repeat_kv',
    'create_mistral_model',
    'create_llama_model',
    'create_custom_model',
    'MISTRAL_PRESETS',

    # Training
    'PPOConfig',
    'ReFLConfig',
    'PPOTrainer',
    'ReFLTrainer',
    'RewardModelTrainer',
    'CritiqueModelTrainer',
    'create_ppo_trainer',
    'create_refl_trainer',

    # Factory functions
    'create_model',
    'create_model_from_config',
    'load_from_pretrained',
    'list_models',
    'get_model_info',
    'compare_models',
    'quick_mini',
    'quick_gpt2',
    'quick_small',
    'MODEL_PRESETS',
]

# Version info
__version__ = '0.4.0'
