# VISX New Features Guide

This document describes the new features added to VISX for JAX mesh parallelization, HuggingFace streaming datasets, and modularized data loading.

## 🌐 JAX Mesh Parallelization

### Overview
VISX now supports automatic mesh setup for distributed training on TPUs and GPUs with JAX sharding.

### Key Features
- **Automatic TPU Detection**: Creates 4x2 mesh for data/model parallelism on TPU v2/v3
- **GPU Fallback**: Uses all available devices for data parallelism
- **Partitioned Layers**: Easy creation of linear layers with weight partitioning
- **Device Information**: Comprehensive device and backend detection

### Usage Examples

```python
from visx.utils.mesh import setup_distributed_training, create_partitioned_linear
from flax import nnx

# Automatic mesh setup
mesh, device_info = setup_distributed_training()

# Create model with partitioned weights
rngs = nnx.Rngs(42)
linear_layer = create_partitioned_linear(
    in_features=128,
    out_features=10,
    mesh=mesh,
    rngs=rngs
)
```

### Manual Mesh Configuration

```python
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
import jax

if jax.default_backend() == 'tpu':
    # 4-way data parallel, 2-way tensor parallel
    mesh = Mesh(mesh_utils.create_device_mesh((4, 2)), ('batch', 'model'))
else:
    # Use all devices for data parallelism
    num_devices = len(jax.devices())
    mesh = Mesh(mesh_utils.create_device_mesh((num_devices, 1)), ('batch', 'model'))
```

## 🔄 HuggingFace Streaming Datasets

### Overview
Support for streaming large datasets from HuggingFace Hub without downloading entire datasets.

### Key Features
- **Streaming Mode**: Process datasets without local storage
- **TensorFlow Integration**: Seamless conversion to tf.data.Dataset
- **Vision Preprocessing**: Automatic image normalization and resizing
- **Graceful Fallback**: Works without HuggingFace datasets installed

### Usage Examples

```python
from visx.data.streaming import create_streaming_vision_dataset, is_huggingface_available

if is_huggingface_available():
    dataset = create_streaming_vision_dataset(
        dataset_name="cifar10",
        split="train",
        batch_size=32,
        image_size=(32, 32),
        shuffle_buffer_size=1000
    )
    
    for batch in dataset:
        # Training loop
        pass
```

### Supported Datasets
- CIFAR-10/100
- ImageNet-1K
- Food101
- Oxford-IIIT Pet
- MNIST/Fashion-MNIST
- And many more vision datasets

## 📊 Modular Data Loading

### Overview
Data loading logic has been extracted from main.py into a modular system in `visx.data`.

### Structure
```
visx/data/
├── __init__.py          # Package exports
├── configs.py           # Dataset configurations
├── loaders.py           # TensorFlow Datasets loaders
└── streaming.py         # HuggingFace streaming support
```

### Dataset Configurations

```python
from visx.data.configs import list_available_datasets, get_dataset_config

# List all configured datasets
datasets = list_available_datasets()

# Get specific configuration
config = get_dataset_config('cifar10')
print(f"Classes: {config['num_classes']}")
print(f"Batch size: {config['batch_size']}")
```

### Data Loading

```python
from visx.data.loaders import create_train_dataset, create_test_dataset

# Create datasets with automatic preprocessing
train_ds = create_train_dataset('cifar10', batch_size=64)
test_ds = create_test_dataset('cifar10', batch_size=64)
```

## 🧪 Testing

### Test Structure
Tests have been moved to the `test/` directory:
- `test/test_modularization.py` - Original modularization tests
- `test/test_new_features.py` - Tests for new features (requires dependencies)
- `test/test_minimal.py` - Minimal tests without heavy dependencies

### Running Tests

```bash
# Run modularization tests
python test/test_modularization.py

# Run minimal feature tests (no dependencies required)
python test/test_minimal.py

# Run full feature tests (requires JAX, TensorFlow, etc.)
python test/test_new_features.py
```

## 🔧 Migration Guide

### From Legacy Dataset Loading

**Old:**
```python
# Legacy global dataset loading in main.py
train_ds_global_tf = tfds.load('cifar10', split='train')
config = DATASET_CONFIGS['cifar10']
```

**New:**
```python
# Modular data loading
from visx.data.configs import get_dataset_config
from visx.data.loaders import create_train_dataset

config = get_dataset_config('cifar10')
train_ds = create_train_dataset('cifar10')
```

### Adding New Datasets

```python
from visx.data.configs import add_dataset_config

add_dataset_config('my_dataset', {
    'num_classes': 10,
    'input_channels': 3,
    'train_split': 'train',
    'test_split': 'test',
    'image_key': 'image',
    'label_key': 'label',
    'num_epochs': 10,
    'eval_every': 200,
    'batch_size': 64
})
```

## 📦 Dependencies

Updated requirements.txt includes:

```
# Core JAX/Flax dependencies
jax>=0.4.0
flax>=0.8.0

# Data dependencies
tensorflow>=2.13.0
tensorflow-datasets>=4.9.0
datasets>=2.14.0  # NEW: HuggingFace datasets

# Other dependencies...
```

## 🎯 Examples

Check the `examples/` directory for complete examples:
- `mesh_parallelization_example.py` - Comprehensive mesh usage examples
- `demo_new_features.py` - Interactive demo of all new features

## ✅ Backward Compatibility

All existing code continues to work:
- Global dataset variables still exist in main.py
- Original DATASET_CONFIGS accessible
- All analysis and training functions unchanged
- Existing model architectures work as before

The new features are additive and don't break existing functionality.