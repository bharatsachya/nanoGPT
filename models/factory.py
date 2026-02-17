"""
Model factory for creating language models with sensible defaults.
Provides a unified interface for instantiating different model architectures.
"""

from .base import ModelRegistry, BaseConfig
from .gpt import GPT, GPTConfig
from .mini_gpt import MiniGPT, MiniGPTConfig


# Model presets with sensible defaults
MODEL_PRESETS = {
    # Mini variants - for quick experimentation
    'mini': {
        'model_class': MiniGPT,
        'config': MiniGPTConfig(),
        'description': '~15M params, 256 context, 6 layers - Fast training & inference'
    },
    'mini-xl': {
        'model_class': MiniGPT,
        'config': MiniGPTConfig(n_layer=8, n_head=8, n_embd=512, block_size=512),
        'description': '~35M params, 512 context, 8 layers - Balanced mini model'
    },

    # Full GPT variants
    'gpt2': {
        'model_class': GPT,
        'config': GPTConfig(),
        'description': '~124M params, 1024 context, 12 layers - Original GPT-2 Small'
    },
    'gpt2-medium': {
        'model_class': GPT,
        'config': GPTConfig(n_layer=24, n_head=16, n_embd=1024),
        'description': '~350M params, 1024 context, 24 layers - GPT-2 Medium'
    },
    'gpt2-large': {
        'model_class': GPT,
        'config': GPTConfig(n_layer=36, n_head=20, n_embd=1280),
        'description': '~774M params, 1024 context, 36 layers - GPT-2 Large'
    },
    'gpt2-xl': {
        'model_class': GPT,
        'config': GPTConfig(n_layer=48, n_head=25, n_embd=1600),
        'description': '~1.5B params, 1024 context, 48 layers - GPT-2 XL'
    },

    # Custom configurations
    'small': {
        'model_class': GPT,
        'config': GPTConfig(n_layer=8, n_head=8, n_embd=512, block_size=512),
        'description': '~45M params, 512 context, 8 layers - Small GPT'
    },
    'medium': {
        'model_class': GPT,
        'config': GPTConfig(n_layer=12, n_head=12, n_embd=768, block_size=1024),
        'description': '~125M params, 1024 context, 12 layers - Medium GPT'
    },
}


def create_model(preset: str = 'mini', **kwargs):
    """
    Create a model from a preset or custom configuration.

    Args:
        preset: Name of the model preset ('mini', 'gpt2', 'small', etc.)
        **kwargs: Optional config overrides (e.g., n_layer=16, n_embd=1024)

    Returns:
        Instantiated model

    Examples:
        >>> model = create_model('mini')  # MiniGPT with defaults
        >>> model = create_model('gpt2')  # GPT-2 Small
        >>> model = create_model('mini', n_layer=8, n_embd=512)  # Custom mini
        >>> model = create_model('gpt2-medium')  # GPT-2 Medium
    """
    if preset not in MODEL_PRESETS:
        available = ', '.join(MODEL_PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset}. Available: {available}")

    preset_info = MODEL_PRESETS[preset]
    model_class = preset_info['model_class']
    config = preset_info['config']

    # Apply config overrides
    if kwargs:
        # Create a new config instance with overrides
        config = type(config)(**{**config.__dict__, **kwargs})

    model = model_class(config)
    return model


def create_model_from_config(config: BaseConfig):
    """
    Create a model from a configuration object.

    Args:
        config: BaseConfig subclass instance

    Returns:
        Instantiated model matching the config type
    """
    if isinstance(config, MiniGPTConfig):
        return MiniGPT(config)
    elif isinstance(config, GPTConfig):
        return GPT(config)
    else:
        # Try to use the registry
        config_type = type(config).__name__.replace('Config', '').lower()
        model_class = ModelRegistry.get(config_type)
        return model_class(config)


def load_from_pretrained(model_type: str, **kwargs):
    """
    Load a pretrained GPT-2 model from HuggingFace.

    Args:
        model_type: One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
        **kwargs: Optional overrides (only 'dropout' is supported)

    Returns:
        Model with pretrained weights

    Examples:
        >>> model = load_from_pretrained('gpt2')
        >>> model = load_from_pretrained('gpt2-medium', dropout=0.1)
    """
    return GPT.from_pretrained(model_type, override_args=kwargs)


def list_models():
    """List all available model presets with descriptions."""
    print("Available model presets:")
    print("-" * 60)
    for name, info in MODEL_PRESETS.items():
        print(f"{name:15s} - {info['description']}")
    print("-" * 60)


def get_model_info(preset: str):
    """Get detailed information about a model preset."""
    if preset not in MODEL_PRESETS:
        available = ', '.join(MODEL_PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset}. Available: {available}")

    info = MODEL_PRESETS[preset]
    config = info['config']

    print(f"Model: {preset}")
    print(f"Description: {info['description']}")
    print(f"Class: {info['model_class'].__name__}")
    print("-" * 40)
    print("Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")


def compare_models(*presets: str):
    """Compare multiple model presets side by side."""
    if not presets:
        presets = list(MODEL_PRESETS.keys())

    configs = [(name, MODEL_PRESETS[name]['config']) for name in presets]

    # Find all keys
    all_keys = set()
    for _, config in configs:
        all_keys.update(config.__dict__.keys())
    all_keys = sorted(all_keys)

    # Print header
    print(f"{'Key':<20s}", end='')
    for name, _ in configs:
        print(f"| {name:<15s}", end='')
    print()
    print("-" * (20 + 16 * len(configs)))

    # Print values
    for key in all_keys:
        print(f"{key:<20s}", end='')
        for _, config in configs:
            value = config.__dict__.get(key, '-')
            print(f"| {str(value):<15s}", end='')
        print()


# Shortcut aliases for common use cases
def quick_mini(**kwargs):
    """Quick shortcut to create a mini model."""
    return create_model('mini', **kwargs)


def quick_gpt2(**kwargs):
    """Quick shortcut to create a GPT-2 Small model."""
    return create_model('gpt2', **kwargs)


def quick_small(**kwargs):
    """Quick shortcut to create a small custom model."""
    return create_model('small', **kwargs)
