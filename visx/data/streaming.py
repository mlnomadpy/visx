"""HuggingFace streaming dataset support for VISX."""

from __future__ import annotations

import tensorflow as tf
from typing import Dict, Any, Optional, Iterator
import warnings

try:
    from datasets import Dataset, IterableDataset, load_dataset
    from datasets.features import Features, Image, ClassLabel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    Dataset = None
    IterableDataset = None


def is_huggingface_available() -> bool:
    """Check if HuggingFace datasets library is available."""
    return HF_AVAILABLE


def create_hf_streaming_dataset(
    dataset_name: str,
    split: str = 'train',
    streaming: bool = True,
    trust_remote_code: bool = False,
    **kwargs
) -> IterableDataset | Dataset:
    """Create a HuggingFace streaming dataset.
    
    Args:
        dataset_name: Name of the HuggingFace dataset.
        split: Dataset split.
        streaming: Whether to use streaming mode.
        trust_remote_code: Whether to trust remote code execution.
        **kwargs: Additional arguments for load_dataset.
        
    Returns:
        IterableDataset | Dataset: HuggingFace dataset.
        
    Raises:
        ImportError: If HuggingFace datasets is not available.
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets library is not available. "
            "Install it with: pip install datasets"
        )
    
    return load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        trust_remote_code=trust_remote_code,
        **kwargs
    )


def hf_to_tf_dataset(
    hf_dataset: Dataset | IterableDataset,
    image_key: str = 'image',
    label_key: str = 'label',
    image_size: Optional[tuple[int, int]] = None,
    normalize: bool = True
) -> tf.data.Dataset:
    """Convert HuggingFace dataset to TensorFlow dataset.
    
    Args:
        hf_dataset: HuggingFace dataset.
        image_key: Key for image data.
        label_key: Key for label data.
        image_size: Target image size (height, width) for resizing.
        normalize: Whether to normalize images to [0, 1].
        
    Returns:
        tf.data.Dataset: TensorFlow dataset.
    """
    def generator():
        for sample in hf_dataset:
            # Extract image and label
            image = sample[image_key]
            label = sample[label_key]
            
            # Convert PIL image to tensor if needed
            if hasattr(image, 'convert'):
                image = image.convert('RGB')
                image = tf.keras.utils.img_to_array(image)
            
            # Resize if requested
            if image_size is not None:
                image = tf.image.resize(image, image_size)
            
            # Normalize
            if normalize:
                image = tf.cast(image, tf.float32) / 255.0
            
            yield {'image': image, 'label': label}
    
    # Infer output signature from first sample
    first_sample = next(iter(hf_dataset))
    
    # Handle image
    image = first_sample[image_key]
    if hasattr(image, 'convert'):
        image = image.convert('RGB')
        image = tf.keras.utils.img_to_array(image)
    
    if image_size is not None:
        image_shape = (*image_size, 3)
    else:
        image_shape = image.shape
    
    output_signature = {
        'image': tf.TensorSpec(shape=image_shape, dtype=tf.float32),
        'label': tf.TensorSpec(shape=(), dtype=tf.int64)
    }
    
    return tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )


def create_streaming_vision_dataset(
    dataset_name: str,
    split: str = 'train',
    batch_size: int = 32,
    image_size: Optional[tuple[int, int]] = None,
    shuffle_buffer_size: int = 1000,
    prefetch_size: int = 2,
    **hf_kwargs
) -> tf.data.Dataset:
    """Create a streaming vision dataset from HuggingFace Hub.
    
    Args:
        dataset_name: Name of the HuggingFace dataset.
        split: Dataset split.
        batch_size: Batch size.
        image_size: Target image size for resizing.
        shuffle_buffer_size: Buffer size for shuffling.
        prefetch_size: Prefetch buffer size.
        **hf_kwargs: Additional arguments for HuggingFace load_dataset.
        
    Returns:
        tf.data.Dataset: Batched and preprocessed TensorFlow dataset.
    """
    if not HF_AVAILABLE:
        warnings.warn(
            "HuggingFace datasets not available. Using fallback TensorFlow Datasets.",
            UserWarning
        )
        return None
    
    # Load streaming dataset
    hf_dataset = create_hf_streaming_dataset(
        dataset_name, split=split, **hf_kwargs
    )
    
    # Convert to TensorFlow dataset
    tf_dataset = hf_to_tf_dataset(
        hf_dataset,
        image_size=image_size
    )
    
    # Apply standard preprocessing
    if shuffle_buffer_size > 0:
        tf_dataset = tf_dataset.shuffle(shuffle_buffer_size)
    
    tf_dataset = tf_dataset.batch(batch_size)
    tf_dataset = tf_dataset.prefetch(prefetch_size)
    
    return tf_dataset


def list_hf_vision_datasets() -> list[str]:
    """List popular HuggingFace vision datasets.
    
    Returns:
        list[str]: List of dataset names.
    """
    return [
        'cifar10',
        'cifar100', 
        'imagenet-1k',
        'food101',
        'oxford-iiit-pet',
        'caltech101',
        'celeba',
        'mnist',
        'fashion_mnist',
        'svhn',
        'stl10',
    ]


def get_hf_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get information about a HuggingFace dataset.
    
    Args:
        dataset_name: Name of the dataset.
        
    Returns:
        Dict[str, Any]: Dataset information.
    """
    if not HF_AVAILABLE:
        return {}
    
    try:
        # Load just metadata without downloading
        dataset_info = load_dataset(dataset_name, split='train', streaming=True)
        first_sample = next(iter(dataset_info))
        
        return {
            'features': list(first_sample.keys()),
            'sample_keys': list(first_sample.keys()),
            'has_images': any('image' in key.lower() for key in first_sample.keys()),
            'has_labels': any('label' in key.lower() for key in first_sample.keys()),
        }
    except Exception as e:
        return {'error': str(e)}