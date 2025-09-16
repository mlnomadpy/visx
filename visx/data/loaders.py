"""Dataset loaders for VISX training."""

from __future__ import annotations

import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Dict, Any

from .configs import get_dataset_config, DEFAULT_DATASET


def preprocess_sample(sample: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess a sample according to dataset configuration.
    
    Args:
        sample: Raw sample from dataset.
        config: Dataset configuration.
        
    Returns:
        Dict[str, Any]: Preprocessed sample with 'image' and 'label' keys.
    """
    return {
        'image': tf.cast(sample[config['image_key']], tf.float32) / 255.0,
        'label': sample[config['label_key']],
    }


def create_tfds_dataset(
    dataset_name: str,
    split: str,
    config: Dict[str, Any] | None = None
) -> tf.data.Dataset:
    """Create a TensorFlow dataset from TFDS.
    
    Args:
        dataset_name: Name of the TFDS dataset.
        split: Dataset split (e.g., 'train', 'test').
        config: Dataset configuration (will be loaded if None).
        
    Returns:
        tf.data.Dataset: Preprocessed dataset.
    """
    if config is None:
        config = get_dataset_config(dataset_name)
    
    # Load the raw dataset
    ds = tfds.load(dataset_name, split=split)
    
    # Apply preprocessing
    ds = ds.map(
        lambda sample: preprocess_sample(sample, config),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return ds


def create_train_dataset(
    dataset_name: str,
    batch_size: int | None = None,
    shuffle_buffer_size: int = 10000
) -> tf.data.Dataset:
    """Create a training dataset.
    
    Args:
        dataset_name: Name of the dataset.
        batch_size: Batch size (uses config default if None).
        shuffle_buffer_size: Buffer size for shuffling.
        
    Returns:
        tf.data.Dataset: Batched training dataset.
    """
    config = get_dataset_config(dataset_name)
    
    if batch_size is None:
        batch_size = config['batch_size']
    
    ds = create_tfds_dataset(dataset_name, config['train_split'], config)
    
    # Shuffle and batch
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds


def create_test_dataset(
    dataset_name: str,
    batch_size: int | None = None
) -> tf.data.Dataset:
    """Create a test dataset.
    
    Args:
        dataset_name: Name of the dataset.
        batch_size: Batch size (uses config default if None).
        
    Returns:
        tf.data.Dataset: Batched test dataset.
    """
    config = get_dataset_config(dataset_name)
    
    if batch_size is None:
        batch_size = config['batch_size']
    
    ds = create_tfds_dataset(dataset_name, config['test_split'], config)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds


def get_dataset_info(dataset_name: str) -> tfds.core.DatasetInfo:
    """Get dataset information from TFDS.
    
    Args:
        dataset_name: Name of the dataset.
        
    Returns:
        tfds.core.DatasetInfo: Dataset information.
    """
    builder = tfds.builder(dataset_name)
    return builder.info


# Legacy compatibility functions
def create_global_datasets(dataset_name: str = DEFAULT_DATASET):
    """Create global train and test datasets for backward compatibility.
    
    Args:
        dataset_name: Name of the dataset.
        
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Training and test datasets.
    """
    config = get_dataset_config(dataset_name)
    
    # Load raw datasets
    train_ds = tfds.load(dataset_name, split=config['train_split'])
    test_ds = tfds.load(dataset_name, split=config['test_split'])
    
    # Apply preprocessing
    preprocess_fn = lambda sample: preprocess_sample(sample, config)
    train_ds = train_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    return train_ds, test_ds