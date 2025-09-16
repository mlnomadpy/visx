"""Dataset configurations for VISX training."""

from __future__ import annotations

# Dataset configurations extracted from main.py
DATASET_CONFIGS = {
    'cifar10': {
        'num_classes': 10, 'input_channels': 3,
        'train_split': 'train', 'test_split': 'test',
        'image_key': 'image', 'label_key': 'label',
        'num_epochs': 10, 'eval_every': 200, 'batch_size': 64
    },
    'cifar100': {
        'num_classes': 100, 'input_channels': 3,
        'train_split': 'train', 'test_split': 'test',
        'image_key': 'image', 'label_key': 'label',
        'num_epochs': 15, 'eval_every': 150, 'batch_size': 32
    },
    'stl10': {
        'num_classes': 10, 'input_channels': 3,
        'train_split': 'train', 'test_split': 'test',
        'image_key': 'image', 'label_key': 'label',
        'num_epochs': 20, 'eval_every': 100, 'batch_size': 32
    },
    'eurosat/rgb': {
        'num_classes': 10, 'input_channels': 3,
        'train_split': 'train[:80%]', 'test_split': 'train[80%:]',
        'image_key': 'image', 'label_key': 'label',
        'num_epochs': 8, 'eval_every': 100, 'batch_size': 32
    },
    'eurosat/all': {
        'num_classes': 10, 'input_channels': 13,
        'train_split': 'train[:80%]', 'test_split': 'train[80%:]',
        'image_key': 'image', 'label_key': 'label',
        'num_epochs': 5, 'eval_every': 100, 'batch_size': 16
    },
}

# Default dataset for backward compatibility
DEFAULT_DATASET = 'cifar10'


def get_dataset_config(dataset_name: str) -> dict:
    """Get configuration for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset.
        
    Returns:
        dict: Dataset configuration.
        
    Raises:
        ValueError: If dataset is not supported.
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                        f"Supported datasets: {list(DATASET_CONFIGS.keys())}")
    
    return DATASET_CONFIGS[dataset_name].copy()


def list_available_datasets() -> list[str]:
    """List all available dataset configurations.
    
    Returns:
        list[str]: List of dataset names.
    """
    return list(DATASET_CONFIGS.keys())


def add_dataset_config(name: str, config: dict) -> None:
    """Add a new dataset configuration.
    
    Args:
        name: Dataset name.
        config: Dataset configuration dictionary.
    """
    required_keys = {
        'num_classes', 'input_channels', 'train_split', 'test_split',
        'image_key', 'label_key', 'num_epochs', 'eval_every', 'batch_size'
    }
    
    if not required_keys.issubset(config.keys()):
        missing = required_keys - config.keys()
        raise ValueError(f"Missing required configuration keys: {missing}")
    
    DATASET_CONFIGS[name] = config.copy()