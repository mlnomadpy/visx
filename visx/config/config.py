"""Configuration management for VISX."""

from __future__ import annotations

import argparse
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class DatasetConfig:
    """Configuration for datasets."""
    name: str
    num_classes: int
    input_channels: int
    train_split: str = "train"
    test_split: str = "test"
    image_key: str = "image"
    label_key: str = "label"
    num_epochs: int = 5
    eval_every: int = 200
    batch_size: int = 128
    target_image_size: Optional[List[int]] = None


@dataclass
class ModelConfig:
    """Configuration for models."""
    name: str
    type: str  # "yat", "linear", "byol", "simclr", etc.
    num_classes: int = 10
    input_channels: int = 3
    architecture_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 0.003
    momentum: float = 0.9
    optimizer: str = "adamw"
    rng_seed: int = 0
    precision: str = "float32"


@dataclass
class PretrainingConfig:
    """Configuration for pretraining methods."""
    method: str = "supervised"  # "supervised", "byol", "simclr", "self_supervised"
    temperature: float = 0.1  # For contrastive methods
    projection_dim: int = 128  # For contrastive methods
    momentum_tau: float = 0.996  # For BYOL
    augmentation_strength: float = 0.5


@dataclass
class ExplainabilityConfig:
    """Configuration for explainability analysis."""
    enabled: bool = False
    methods: List[str] = field(default_factory=lambda: ["saliency", "grad_cam", "attention"])
    layer_names: List[str] = field(default_factory=lambda: ["conv1", "conv2"])
    num_samples: int = 16


@dataclass
class Config:
    """Main configuration class."""
    mode: str = "training"  # "training", "explainability", "pretraining", "comparison"
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(name="cifar10", num_classes=10, input_channels=3))
    model: ModelConfig = field(default_factory=lambda: ModelConfig(name="yat_cnn", type="yat"))
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pretraining: PretrainingConfig = field(default_factory=PretrainingConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    output_dir: str = "outputs"
    save_checkpoints: bool = True
    verbose: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create dataset config
        dataset_cfg = DatasetConfig(**config_dict.get('dataset', {}))
        
        # Create model config
        model_cfg = ModelConfig(**config_dict.get('model', {}))
        
        # Create training config
        training_cfg = TrainingConfig(**config_dict.get('training', {}))
        
        # Create pretraining config
        pretraining_cfg = PretrainingConfig(**config_dict.get('pretraining', {}))
        
        # Create explainability config
        explainability_cfg = ExplainabilityConfig(**config_dict.get('explainability', {}))
        
        # Create main config
        main_config = config_dict.copy()
        main_config.pop('dataset', None)
        main_config.pop('model', None)
        main_config.pop('training', None)
        main_config.pop('pretraining', None)
        main_config.pop('explainability', None)
        
        return cls(
            dataset=dataset_cfg,
            model=model_cfg,
            training=training_cfg,
            pretraining=pretraining_cfg,
            explainability=explainability_cfg,
            **main_config
        )

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'mode': self.mode,
            'output_dir': self.output_dir,
            'save_checkpoints': self.save_checkpoints,
            'verbose': self.verbose,
            'dataset': {
                'name': self.dataset.name,
                'num_classes': self.dataset.num_classes,
                'input_channels': self.dataset.input_channels,
                'train_split': self.dataset.train_split,
                'test_split': self.dataset.test_split,
                'image_key': self.dataset.image_key,
                'label_key': self.dataset.label_key,
                'num_epochs': self.dataset.num_epochs,
                'eval_every': self.dataset.eval_every,
                'batch_size': self.dataset.batch_size,
                'target_image_size': self.dataset.target_image_size,
            },
            'model': {
                'name': self.model.name,
                'type': self.model.type,
                'num_classes': self.model.num_classes,
                'input_channels': self.model.input_channels,
                'architecture_params': self.model.architecture_params,
            },
            'training': {
                'learning_rate': self.training.learning_rate,
                'momentum': self.training.momentum,
                'optimizer': self.training.optimizer,
                'rng_seed': self.training.rng_seed,
                'precision': self.training.precision,
            },
            'pretraining': {
                'method': self.pretraining.method,
                'temperature': self.pretraining.temperature,
                'projection_dim': self.pretraining.projection_dim,
                'momentum_tau': self.pretraining.momentum_tau,
                'augmentation_strength': self.pretraining.augmentation_strength,
            },
            'explainability': {
                'enabled': self.explainability.enabled,
                'methods': self.explainability.methods,
                'layer_names': self.explainability.layer_names,
                'num_samples': self.explainability.num_samples,
            }
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


# Predefined dataset configurations
DATASET_CONFIGS = {
    'cifar10': DatasetConfig(
        name='cifar10',
        num_classes=10,
        input_channels=3,
        train_split='train',
        test_split='test',
        image_key='image',
        label_key='label',
        num_epochs=5,
        eval_every=200,
        batch_size=128
    ),
    'cifar100': DatasetConfig(
        name='cifar100',
        num_classes=100,
        input_channels=3,
        train_split='train',
        test_split='test',
        image_key='image',
        label_key='label',
        num_epochs=5,
        eval_every=200,
        batch_size=128
    ),
    'stl10': DatasetConfig(
        name='stl10',
        num_classes=10,
        input_channels=3,
        train_split='train',
        test_split='test',
        image_key='image',
        label_key='label',
        num_epochs=5,
        eval_every=200,
        batch_size=128
    ),
    'eurosat/rgb': DatasetConfig(
        name='eurosat/rgb',
        num_classes=10,
        input_channels=3,
        train_split='train[:80%]',
        test_split='train[80%:]',
        image_key='image',
        label_key='label',
        num_epochs=5,
        eval_every=100,
        batch_size=128
    ),
    'eurosat/all': DatasetConfig(
        name='eurosat/all',
        num_classes=10,
        input_channels=13,
        train_split='train[:80%]',
        test_split='train[80%:]',
        image_key='image',
        label_key='label',
        num_epochs=5,
        eval_every=100,
        batch_size=16  # Smaller batch for more channels
    ),
}


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description='VISX: Vision eXploration with YAT architectures')
    
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--mode', type=str, choices=['training', 'explainability', 'pretraining', 'comparison'], 
                        default='training', help='Mode to run in')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--model', type=str, default='yat_cnn', help='Model name')
    parser.add_argument('--model_type', type=str, choices=['yat', 'linear', 'byol', 'simclr'], 
                        default='yat', help='Model type')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--rng_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Pretraining specific args
    parser.add_argument('--pretraining_method', type=str, choices=['supervised', 'byol', 'simclr', 'self_supervised'],
                        default='supervised', help='Pretraining method')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for contrastive learning')
    
    # Explainability specific args
    parser.add_argument('--explain_methods', nargs='+', default=['saliency', 'grad_cam'], 
                        help='Explainability methods to use')
    parser.add_argument('--explain_layers', nargs='+', default=['conv1', 'conv2'], 
                        help='Layers to analyze for explainability')
    
    return parser


def parse_config(args: argparse.Namespace) -> Config:
    """Parse configuration from command line arguments."""
    if args.config:
        config = Config.from_yaml(args.config)
        # Override with command line arguments if provided
        if hasattr(args, 'mode') and args.mode:
            config.mode = args.mode
        if hasattr(args, 'dataset') and args.dataset:
            config.dataset.name = args.dataset
        if hasattr(args, 'learning_rate'):
            config.training.learning_rate = args.learning_rate
        # Add more overrides as needed
        return config
    else:
        # Create config from command line arguments
        dataset_cfg = DATASET_CONFIGS.get(args.dataset, DatasetConfig(name=args.dataset, num_classes=10, input_channels=3))
        if hasattr(args, 'num_epochs'):
            dataset_cfg.num_epochs = args.num_epochs
        if hasattr(args, 'batch_size'):
            dataset_cfg.batch_size = args.batch_size
            
        model_cfg = ModelConfig(name=args.model, type=args.model_type)
        training_cfg = TrainingConfig(learning_rate=args.learning_rate, rng_seed=args.rng_seed)
        
        pretraining_cfg = PretrainingConfig()
        if hasattr(args, 'pretraining_method'):
            pretraining_cfg.method = args.pretraining_method
        if hasattr(args, 'temperature'):
            pretraining_cfg.temperature = args.temperature
            
        explainability_cfg = ExplainabilityConfig()
        if hasattr(args, 'explain_methods'):
            explainability_cfg.methods = args.explain_methods
        if hasattr(args, 'explain_layers'):
            explainability_cfg.layer_names = args.explain_layers
            
        return Config(
            mode=args.mode,
            dataset=dataset_cfg,
            model=model_cfg,
            training=training_cfg,
            pretraining=pretraining_cfg,
            explainability=explainability_cfg,
            output_dir=args.output_dir,
            verbose=args.verbose
        )