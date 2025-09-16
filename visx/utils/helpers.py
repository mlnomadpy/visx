"""Utility functions for VISX."""

from __future__ import annotations

import os
import pickle
import json
from pathlib import Path
from typing import Any, Dict
import jax.numpy as jnp
import numpy as np


def save_model_checkpoint(model, filepath: str):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # For JAX/Flax models, we need to save the state dict
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model_checkpoint(filepath: str):
    """Load model checkpoint."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_training_history(history: Dict[str, Any], filepath: str):
    """Save training history to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert JAX arrays to numpy for JSON serialization
    serializable_history = {}
    for key, value in history.items():
        if isinstance(value, list):
            # Convert any JAX arrays in the list to numpy
            serializable_history[key] = [
                float(item) if isinstance(item, (jnp.ndarray, np.ndarray)) else item
                for item in value
            ]
        else:
            serializable_history[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_history, f, indent=2)


def load_training_history(filepath: str) -> Dict[str, Any]:
    """Load training history from file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def ensure_output_dir(output_dir: str) -> Path:
    """Ensure output directory exists and return Path object."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_model_summary(model, model_name: str = "Model"):
    """Print a summary of the model architecture."""
    print(f"\n📋 {model_name.upper()} SUMMARY")
    print("=" * 50)
    
    # Count parameters (simplified)
    try:
        param_count = sum(
            p.size if hasattr(p, 'size') else np.prod(p.shape) 
            for p in jax.tree_util.tree_leaves(model) 
            if hasattr(p, 'shape')
        )
        print(f"Estimated parameters: {param_count:,}")
    except:
        print("Parameter count unavailable")
    
    print(f"Model type: {type(model).__name__}")


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics dictionary for display."""
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            formatted.append(f"{key}: {value:.{precision}f}")
        else:
            formatted.append(f"{key}: {value}")
    return ", ".join(formatted)


def create_experiment_name(config) -> str:
    """Create a unique experiment name based on configuration."""
    parts = [
        config.mode,
        config.model.name,
        config.dataset.name,
        f"lr{config.training.learning_rate}",
        f"epochs{config.dataset.num_epochs}"
    ]
    
    if config.mode == "pretraining":
        parts.append(config.pretraining.method)
    
    return "_".join(parts)


class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total_steps: int, print_every: int = 100):
        self.total_steps = total_steps
        self.print_every = print_every
        self.current_step = 0
    
    def update(self, step: int, metrics: Dict[str, float] = None):
        """Update progress."""
        self.current_step = step
        
        if step % self.print_every == 0:
            progress = (step / self.total_steps) * 100
            print(f"Progress: {progress:.1f}% ({step}/{self.total_steps})", end="")
            
            if metrics:
                print(f" - {format_metrics(metrics)}")
            else:
                print()
    
    def finish(self, final_metrics: Dict[str, float] = None):
        """Mark training as finished."""
        print(f"✅ Completed {self.total_steps} steps")
        if final_metrics:
            print(f"Final metrics: {format_metrics(final_metrics)}")


def log_experiment_info(config, output_dir: str):
    """Log experiment configuration and info."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config.to_yaml(str(output_path / "config.yaml"))
    
    # Create experiment log
    experiment_info = {
        "experiment_name": create_experiment_name(config),
        "mode": config.mode,
        "model": {
            "name": config.model.name,
            "type": config.model.type
        },
        "dataset": {
            "name": config.dataset.name,
            "num_classes": config.dataset.num_classes,
            "batch_size": config.dataset.batch_size,
            "num_epochs": config.dataset.num_epochs
        },
        "training": {
            "learning_rate": config.training.learning_rate,
            "optimizer": config.training.optimizer,
            "rng_seed": config.training.rng_seed
        }
    }
    
    with open(output_path / "experiment_info.json", 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    print(f"💾 Experiment info saved to {output_path}")
    print(f"📋 Experiment name: {experiment_info['experiment_name']}")


def validate_config(config) -> bool:
    """Validate configuration settings."""
    errors = []
    
    # Check required fields
    if not config.dataset.name:
        errors.append("Dataset name is required")
    
    if not config.model.name:
        errors.append("Model name is required")
    
    if config.training.learning_rate <= 0:
        errors.append("Learning rate must be positive")
    
    if config.dataset.num_epochs <= 0:
        errors.append("Number of epochs must be positive")
    
    if config.dataset.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    # Check mode-specific requirements
    if config.mode == "explainability" and not config.explainability.enabled:
        errors.append("Explainability must be enabled for explainability mode")
    
    if errors:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✅ Configuration validation passed")
    return True