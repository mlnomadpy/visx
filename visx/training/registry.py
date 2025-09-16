"""Model registry for managing different model types."""

from __future__ import annotations

from typing import Dict, Type, Callable, Any
from flax import nnx
from ..models import YatCNN, LinearCNN
from ..config import Config


class ModelRegistry:
    """Registry for managing different model architectures."""
    
    _models: Dict[str, Type[nnx.Module]] = {}
    _model_builders: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[nnx.Module], 
                 builder_func: Callable = None):
        """Register a model class."""
        cls._models[name] = model_class
        if builder_func:
            cls._model_builders[name] = builder_func
    
    @classmethod
    def get_model_class(cls, name: str) -> Type[nnx.Module]:
        """Get a model class by name."""
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found in registry. Available models: {list(cls._models.keys())}")
        return cls._models[name]
    
    @classmethod
    def create_model(cls, name: str, config: Config, rngs: nnx.Rngs) -> nnx.Module:
        """Create a model instance."""
        if name in cls._model_builders:
            return cls._model_builders[name](config, rngs)
        
        model_class = cls.get_model_class(name)
        return model_class(
            rngs=rngs,
            num_classes=config.dataset.num_classes,
            input_channels=config.dataset.input_channels,
            **config.model.architecture_params
        )
    
    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered models."""
        return list(cls._models.keys())


# Builder functions for complex models
def build_yat_cnn(config: Config, rngs: nnx.Rngs) -> YatCNN:
    """Build YAT CNN model."""
    return YatCNN(
        rngs=rngs,
        num_classes=config.dataset.num_classes,
        input_channels=config.dataset.input_channels,
        **config.model.architecture_params
    )


def build_linear_cnn(config: Config, rngs: nnx.Rngs) -> LinearCNN:
    """Build Linear CNN model."""
    return LinearCNN(
        rngs=rngs,
        num_classes=config.dataset.num_classes,
        input_channels=config.dataset.input_channels,
        **config.model.architecture_params
    )


# Register default models
ModelRegistry.register("yat_cnn", YatCNN, build_yat_cnn)
ModelRegistry.register("linear_cnn", LinearCNN, build_linear_cnn)
ModelRegistry.register("yat", YatCNN, build_yat_cnn)  # Alias
ModelRegistry.register("linear", LinearCNN, build_linear_cnn)  # Alias