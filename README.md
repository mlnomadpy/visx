# VISX: Vision eXploration with YAT Architectures

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A modular framework for vision research with YAT (Yet Another Transformer) architectures, featuring support for multiple pretraining paradigms, explainability analysis, and comprehensive model comparison.

## 🚀 Features

- **Modular Architecture**: Clean separation of models, training, evaluation, and configuration
- **Multiple Training Modes**: 
  - Standard supervised training
  - Self-supervised pretraining (BYOL, SimCLR)
  - Model comparison (YAT vs Linear)
  - Explainability analysis
- **Model Registry**: Easy registration and management of different model architectures
- **Configuration Management**: YAML-based configuration with CLI argument support
- **Comprehensive Evaluation**: Training curves, convergence analysis, confusion matrices
- **Explainability Tools**: Saliency maps, activation visualization, kernel analysis
- **Production Ready**: Proper logging, checkpointing, and experiment tracking

## 📦 Installation

### From Source

```bash
git clone https://github.com/mlnomadpy/visx.git
cd visx
pip install -r requirements.txt
pip install -e .
```

### Requirements

- Python 3.8+
- JAX/Flax for neural networks
- TensorFlow for data loading
- Additional dependencies in `requirements.txt`

## 🎯 Quick Start

### 1. Basic Training

```bash
# Train a YAT model on CIFAR-10
python main_cli.py --mode training --dataset cifar10 --model yat_cnn --num_epochs 5

# Train with configuration file
python main_cli.py --config configs/training_example.yaml
```

### 2. Model Comparison

```bash
# Compare YAT vs Linear models
python main_cli.py --mode comparison --dataset cifar10 --num_epochs 3

# With configuration
python main_cli.py --config configs/comparison_example.yaml
```

### 3. Self-Supervised Pretraining

```bash
# BYOL pretraining
python main_cli.py --mode pretraining --pretraining_method byol --dataset cifar10

# SimCLR pretraining
python main_cli.py --mode pretraining --pretraining_method simclr --dataset cifar10

# With configuration
python main_cli.py --config configs/byol_pretraining.yaml
```

### 4. Explainability Analysis

```bash
# Run explainability analysis
python main_cli.py --mode explainability --dataset cifar10 --explain_methods saliency kernels

# With configuration
python main_cli.py --config configs/explainability_example.yaml
```

## 📋 Configuration

VISX uses YAML configuration files for reproducible experiments. Here's a sample configuration:

```yaml
# Basic configuration
mode: training
output_dir: outputs/my_experiment
save_checkpoints: true
verbose: true

# Dataset configuration
dataset:
  name: cifar10
  num_classes: 10
  input_channels: 3
  num_epochs: 5
  batch_size: 128

# Model configuration
model:
  name: yat_cnn
  type: yat
  architecture_params: {}

# Training configuration
training:
  learning_rate: 0.003
  optimizer: adamw
  rng_seed: 42

# Pretraining configuration (for pretraining mode)
pretraining:
  method: byol  # supervised, byol, simclr, self_supervised
  temperature: 0.1
  projection_dim: 128

# Explainability configuration (for explainability mode)
explainability:
  enabled: true
  methods: [saliency, kernels, activation]
  layer_names: [conv1, conv2]
  num_samples: 16
```

## 🏗️ Architecture

VISX follows a modular architecture:

```
visx/
├── models/           # Model definitions
│   ├── layers.py     # YAT layers (YatConv, YatNMN)
│   └── architectures.py  # Model architectures (YatCNN, LinearCNN)
├── training/         # Training logic
│   ├── registry.py   # Model registry
│   ├── train.py      # Training loops
│   └── modes.py      # Training modes orchestration
├── pretraining/      # Self-supervised pretraining
│   └── methods.py    # BYOL, SimCLR implementations
├── evaluation/       # Evaluation and analysis
│   ├── explainability.py  # Saliency, visualization
│   └── comparison.py # Model comparison utilities
├── config/           # Configuration management
│   └── config.py     # Config classes and parsing
├── utils/            # Utilities
│   └── helpers.py    # Checkpointing, logging, etc.
└── configs/          # Example configurations
    ├── training_example.yaml
    ├── byol_pretraining.yaml
    ├── explainability_example.yaml
    └── comparison_example.yaml
```

## 🔬 YAT Architecture

YAT (Yet Another Transformer) introduces distance-based transformations in convolutional and dense layers:

```python
# YAT transformation: y² / (distance + ε)
distances = inputs_squared_sum + kernel_squared_sum - 2 * y
y = y² / (distances + epsilon)

# With optional alpha scaling
if alpha is not None:
    scale = (√out_features / log(1 + out_features))^alpha
    y = y * scale
```

## 📊 Supported Datasets

- CIFAR-10/CIFAR-100
- STL-10
- EuroSAT (RGB and full spectrum)
- Extensible to other TensorFlow Datasets

## 🔍 Explainability Methods

- **Saliency Maps**: Gradient-based importance visualization
- **Activation Maps**: Intermediate layer activation visualization
- **Kernel Visualization**: Learned filter visualization
- **Attention Analysis**: For models with attention mechanisms (planned)

## 🏋️ Pretraining Methods

### Supervised Pretraining
Standard supervised learning with labeled data.

### BYOL (Bootstrap Your Own Latent)
Self-supervised learning using online and target networks with exponential moving averages.

### SimCLR
Contrastive learning with data augmentation and temperature-scaled contrastive loss.

## 📈 Model Comparison

VISX provides comprehensive model comparison:

- Training curve comparison
- Convergence analysis
- Model agreement analysis
- Confusion matrices
- Per-class accuracy analysis
- Summary reports with recommendations

## 🛠️ Advanced Usage

### Custom Models

Register new models in the model registry:

```python
from visx.training import ModelRegistry
from visx.models import YatCNN

def build_custom_model(config, rngs):
    return YatCNN(
        rngs=rngs,
        num_classes=config.dataset.num_classes,
        input_channels=config.dataset.input_channels,
        # Custom parameters
        custom_param=config.model.architecture_params.get('custom_param', 'default')
    )

ModelRegistry.register("custom_yat", YatCNN, build_custom_model)
```

### Custom Pretraining Methods

Implement new pretraining methods:

```python
from visx.pretraining import PretrainingMethod

class CustomPretraining(PretrainingMethod):
    def create_model(self, config, rngs):
        # Implement model creation
        pass
    
    def loss_fn(self, model, batch):
        # Implement custom loss
        pass
    
    def pretrain(self, config):
        # Implement pretraining loop
        pass
```

### Programmatic Usage

Use VISX programmatically:

```python
from visx.config import Config, DatasetConfig, ModelConfig
from visx.training import run_training_mode

# Create configuration
config = Config(
    mode="training",
    dataset=DatasetConfig(name="cifar10", num_classes=10, input_channels=3),
    model=ModelConfig(name="yat_cnn", type="yat")
)

# Run training
model, results = run_training_mode(config)
```

## 📝 Examples

See the `configs/` directory for example configurations:

- `training_example.yaml`: Basic supervised training
- `byol_pretraining.yaml`: BYOL self-supervised pretraining
- `explainability_example.yaml`: Explainability analysis
- `comparison_example.yaml`: Model comparison

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- JAX/Flax team for the neural network framework
- TensorFlow team for data loading utilities
- The research community for self-supervised learning methods

## 📚 Citation

If you use VISX in your research, please cite:

```bibtex
@software{visx2024,
  title={VISX: Vision eXploration with YAT Architectures},
  author={VISX Team},
  year={2024},
  url={https://github.com/mlnomadpy/visx}
}
```