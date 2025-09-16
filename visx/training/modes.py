"""Training modes and orchestration."""

from __future__ import annotations

from enum import Enum
from typing import Tuple, Dict, Any
from flax import nnx

from ..config import Config
from .train import train_model
from .registry import ModelRegistry
from ..pretraining.methods import pretrain_model


class TrainingMode(Enum):
    """Available training modes."""
    TRAINING = "training"
    PRETRAINING = "pretraining"
    EXPLAINABILITY = "explainability"
    COMPARISON = "comparison"


class ModelTrainer:
    """Main trainer class that orchestrates different training modes."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def run(self) -> Tuple[nnx.Module, Dict[str, Any]]:
        """Run training based on the configured mode."""
        mode = TrainingMode(self.config.mode)
        
        if mode == TrainingMode.TRAINING:
            return self.run_training()
        elif mode == TrainingMode.PRETRAINING:
            return self.run_pretraining()
        elif mode == TrainingMode.EXPLAINABILITY:
            return self.run_explainability()
        elif mode == TrainingMode.COMPARISON:
            return self.run_comparison()
        else:
            raise ValueError(f"Unknown training mode: {self.config.mode}")
    
    def run_training(self) -> Tuple[nnx.Module, Dict[str, Any]]:
        """Run standard supervised training."""
        if self.config.verbose:
            print(f"🚀 Starting training mode for {self.config.model.name} on {self.config.dataset.name}")
        
        model, history = train_model(self.config)
        
        if self.config.verbose:
            print("✅ Training completed!")
            final_acc = history['test_accuracy'][-1] if history['test_accuracy'] else 0.0
            print(f"Final test accuracy: {final_acc:.4f}")
        
        return model, history
    
    def run_pretraining(self) -> Tuple[nnx.Module, Dict[str, Any]]:
        """Run pretraining (supervised or self-supervised)."""
        if self.config.verbose:
            print(f"🚀 Starting pretraining mode with {self.config.pretraining.method} for {self.config.model.name}")
        
        model, history = pretrain_model(self.config)
        
        if self.config.verbose:
            print("✅ Pretraining completed!")
            if 'loss' in history:
                final_loss = history['loss'][-1] if history['loss'] else 0.0
                print(f"Final loss: {final_loss:.4f}")
        
        return model, history
    
    def run_explainability(self) -> Tuple[nnx.Module, Dict[str, Any]]:
        """Run explainability analysis."""
        if self.config.verbose:
            print(f"🔬 Starting explainability mode for {self.config.model.name}")
        
        # First train the model
        model, history = train_model(self.config)
        
        # Then run explainability analysis
        from ..evaluation.explainability import run_explainability_analysis
        explainability_results = run_explainability_analysis(model, self.config)
        
        # Combine results
        results = {
            'model': model,
            'training_history': history,
            'explainability': explainability_results
        }
        
        if self.config.verbose:
            print("✅ Explainability analysis completed!")
        
        return model, results
    
    def run_comparison(self) -> Tuple[nnx.Module, Dict[str, Any]]:
        """Run model comparison between YAT and Linear models."""
        if self.config.verbose:
            print(f"📊 Starting comparison mode on {self.config.dataset.name}")
        
        # Train YAT model
        yat_config = self.config
        yat_config.model.type = "yat"
        yat_config.model.name = "yat_cnn"
        
        if self.config.verbose:
            print("Training YAT model...")
        yat_model, yat_history = train_model(yat_config)
        
        # Train Linear model
        linear_config = self.config
        linear_config.model.type = "linear"
        linear_config.model.name = "linear_cnn"
        
        if self.config.verbose:
            print("Training Linear model...")
        linear_model, linear_history = train_model(linear_config)
        
        # Run comparison analysis
        from ..evaluation.comparison import run_comparison_analysis
        comparison_results = run_comparison_analysis(
            yat_model, linear_model, yat_history, linear_history, self.config
        )
        
        results = {
            'yat_model': yat_model,
            'linear_model': linear_model,
            'yat_history': yat_history,
            'linear_history': linear_history,
            'comparison': comparison_results
        }
        
        if self.config.verbose:
            print("✅ Model comparison completed!")
        
        return yat_model, results  # Return YAT model as primary


def run_training_mode(config: Config) -> Tuple[nnx.Module, Dict[str, Any]]:
    """Run training in the specified mode."""
    trainer = ModelTrainer(config)
    return trainer.run()