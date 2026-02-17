"""
Models package for nanoGPT.

A modular, interconnected model library with shared components and factory functions.

Usage:
    >>> from models import create_model, list_models
    >>> list_models()
    >>> model = create_model('mini')  # or 'gpt2', 'small', etc.

    >>> # Direct imports
    >>> from models import GPT, GPTConfig, MiniGPT, MiniGPTConfig
    >>> from models import create_model, load_from_pretrained
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
__version__ = '0.1.0'
