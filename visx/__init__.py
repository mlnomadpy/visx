"""
VISX: A modular framework for Vision eXploration with YAT (Yet Another Transformer) architectures.
"""

__version__ = "0.1.0"
__author__ = "VISX Team"

# Lazy imports to avoid dependency issues during installation
def get_models():
    from .models import YatCNN, LinearCNN, YatConv, YatNMN
    return YatCNN, LinearCNN, YatConv, YatNMN

def get_training():
    from .training import TrainingMode, ModelRegistry
    return TrainingMode, ModelRegistry

def get_config():
    from .config import Config
    return Config

__all__ = [
    "get_models",
    "get_training", 
    "get_config",
    "__version__",
    "__author__"
]