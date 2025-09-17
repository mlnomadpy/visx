"""HuggingFace streaming dataset support for VISX."""

from __future__ import annotations

import tensorflow as tf
from typing import Dict, Any, Optional, Iterator, Union, Callable
import warnings
import logging
import time
import functools
from pathlib import Path

try:
    from datasets import Dataset, IterableDataset, load_dataset
    from datasets.features import Features, Image, ClassLabel
    from datasets.exceptions import DatasetNotFoundError
    import requests
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    Dataset = None
    IterableDataset = None
    DatasetNotFoundError = Exception
    requests = None

# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_TIMEOUT = 30
DEFAULT_CACHE_DIR = None


def retry_on_failure(max_retries: int = DEFAULT_RETRY_COUNT, 
                    delay: float = DEFAULT_RETRY_DELAY,
                    exceptions: tuple = (Exception,)):
    """Decorator to retry function calls on failure.
    
    Args:
        max_retries: Maximum number of retry attempts.
        delay: Delay between retries in seconds.
        exceptions: Tuple of exceptions to catch and retry on.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed.")
                        raise last_exception
            return None
        return wrapper
    return decorator


def is_huggingface_available() -> bool:
    """Check if HuggingFace datasets library is available."""
    return HF_AVAILABLE


def check_internet_connection() -> bool:
    """Check if internet connection is available for HuggingFace Hub."""
    if not requests:
        return False
    
    try:
        response = requests.get("https://huggingface.co", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def validate_dataset_config(
    dataset_name: str,
    split: str = 'train',
    **kwargs
) -> Dict[str, Any]:
    """Validate dataset configuration and check availability.
    
    Args:
        dataset_name: Name of the HuggingFace dataset.
        split: Dataset split.
        **kwargs: Additional arguments.
        
    Returns:
        Dict containing validation results and metadata.
    """
    if not HF_AVAILABLE:
        return {
            'valid': False,
            'error': 'HuggingFace datasets library not available',
            'suggestion': 'Install with: pip install datasets'
        }
    
    # Check internet connection
    if not check_internet_connection():
        return {
            'valid': False,
            'error': 'No internet connection available',
            'suggestion': 'Check your internet connection and try again'
        }
    
    try:
        # Try to load just metadata
        from datasets import get_dataset_config_names, get_dataset_split_names
        
        # Validate config if specified
        config_name = kwargs.get('name', kwargs.get('config_name'))
        if config_name:
            available_configs = get_dataset_config_names(dataset_name)
            if config_name not in available_configs:
                return {
                    'valid': False,
                    'error': f'Config "{config_name}" not found',
                    'suggestion': f'Available configs: {available_configs}'
                }
        
        # Validate split
        try:
            available_splits = get_dataset_split_names(dataset_name, config_name=config_name)
            if split not in available_splits:
                return {
                    'valid': False,
                    'error': f'Split "{split}" not found',
                    'suggestion': f'Available splits: {available_splits}'
                }
        except Exception:
            # Some datasets may not support split enumeration
            logger.warning(f"Could not validate splits for {dataset_name}")
        
        return {
            'valid': True,
            'dataset_name': dataset_name,
            'split': split,
            'config_name': config_name
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'suggestion': 'Check dataset name and try again'
        }


@retry_on_failure(max_retries=DEFAULT_RETRY_COUNT)
def create_hf_streaming_dataset(
    dataset_name: str,
    split: str = 'train',
    streaming: bool = True,
    trust_remote_code: bool = False,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    validate: bool = True,
    **kwargs
) -> Union[IterableDataset, Dataset]:
    """Create a HuggingFace streaming dataset with robust error handling.
    
    Args:
        dataset_name: Name of the HuggingFace dataset.
        split: Dataset split.
        streaming: Whether to use streaming mode.
        trust_remote_code: Whether to trust remote code execution.
        cache_dir: Directory to cache dataset files.
        validate: Whether to validate dataset config before loading.
        **kwargs: Additional arguments for load_dataset.
        
    Returns:
        IterableDataset | Dataset: HuggingFace dataset.
        
    Raises:
        ImportError: If HuggingFace datasets is not available.
        ValueError: If dataset validation fails.
        ConnectionError: If unable to connect to HuggingFace Hub.
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets library is not available. "
            "Install it with: pip install datasets"
        )
    
    # Validate configuration if requested
    if validate:
        validation_result = validate_dataset_config(dataset_name, split, **kwargs)
        if not validation_result['valid']:
            raise ValueError(
                f"Dataset validation failed: {validation_result['error']}. "
                f"Suggestion: {validation_result.get('suggestion', 'N/A')}"
            )
    
    try:
        logger.info(f"Loading dataset: {dataset_name}, split: {split}, streaming: {streaming}")
        
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            **kwargs
        )
        
        logger.info(f"Successfully loaded dataset: {dataset_name}")
        return dataset
        
    except DatasetNotFoundError as e:
        raise ValueError(f"Dataset '{dataset_name}' not found on HuggingFace Hub: {e}")
    except ConnectionError as e:
        raise ConnectionError(f"Failed to connect to HuggingFace Hub: {e}")
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise


def detect_dataset_format(dataset_sample: Dict[str, Any]) -> Dict[str, str]:
    """Automatically detect dataset format and key mappings.
    
    Args:
        dataset_sample: A sample from the dataset.
        
    Returns:
        Dict mapping standard keys to dataset-specific keys.
    """
    key_mapping = {}
    sample_keys = list(dataset_sample.keys())
    
    # Detect image key
    image_candidates = ['image', 'img', 'picture', 'pixel_values']
    for candidate in image_candidates:
        if candidate in sample_keys:
            key_mapping['image'] = candidate
            break
    else:
        # Look for PIL Image or array-like objects
        for key, value in dataset_sample.items():
            if hasattr(value, 'convert') or (hasattr(value, 'shape') and len(getattr(value, 'shape', [])) >= 2):
                key_mapping['image'] = key
                break
    
    # Detect label key
    label_candidates = ['label', 'labels', 'target', 'class', 'category', 'y']
    for candidate in label_candidates:
        if candidate in sample_keys:
            key_mapping['label'] = candidate
            break
    
    return key_mapping


def safe_image_preprocessing(
    image: Any,
    image_size: Optional[tuple[int, int]] = None,
    normalize: bool = True,
    dtype: tf.DType = tf.float32
) -> tf.Tensor:
    """Safely preprocess image data with error handling.
    
    Args:
        image: Input image (PIL, numpy array, or tensor).
        image_size: Target image size (height, width).
        normalize: Whether to normalize to [0, 1].
        dtype: Target dtype for the image tensor.
        
    Returns:
        Preprocessed image tensor.
    """
    try:
        # Convert PIL image to tensor if needed
        if hasattr(image, 'convert'):
            image = image.convert('RGB')
            image = tf.keras.utils.img_to_array(image)
        
        # Ensure it's a tensor
        if not isinstance(image, tf.Tensor):
            image = tf.convert_to_tensor(image)
        
        # Ensure 3D (HWC format)
        if len(image.shape) == 2:
            image = tf.expand_dims(image, -1)
        elif len(image.shape) == 4:
            image = tf.squeeze(image, 0)  # Remove batch dimension if present
        
        # Resize if requested
        if image_size is not None:
            image = tf.image.resize(image, image_size)
        
        # Convert to desired dtype
        image = tf.cast(image, dtype)
        
        # Normalize if requested
        if normalize:
            # Check if image is in [0, 255] range
            max_val = tf.reduce_max(image)
            if max_val > 1.0:
                image = image / 255.0
        
        return image
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        # Return a placeholder image of the correct shape
        if image_size:
            return tf.zeros((*image_size, 3), dtype=dtype)
        else:
            return tf.zeros((224, 224, 3), dtype=dtype)


def hf_to_tf_dataset(
    hf_dataset: Union[Dataset, IterableDataset],
    image_key: Optional[str] = None,
    label_key: Optional[str] = None,
    image_size: Optional[tuple[int, int]] = None,
    normalize: bool = True,
    dtype: tf.DType = tf.float32,
    auto_detect_format: bool = True,
    error_handling: str = 'skip'  # 'skip', 'placeholder', 'raise'
) -> tf.data.Dataset:
    """Convert HuggingFace dataset to TensorFlow dataset with robust preprocessing.
    
    Args:
        hf_dataset: HuggingFace dataset.
        image_key: Key for image data (auto-detected if None).
        label_key: Key for label data (auto-detected if None).
        image_size: Target image size (height, width) for resizing.
        normalize: Whether to normalize images to [0, 1].
        dtype: Target dtype for tensors.
        auto_detect_format: Whether to auto-detect key mappings.
        error_handling: How to handle errors ('skip', 'placeholder', 'raise').
        
    Returns:
        tf.data.Dataset: TensorFlow dataset.
        
    Raises:
        ValueError: If required keys cannot be found or detected.
    """
    def generator():
        skip_count = 0
        total_count = 0
        
        for sample in hf_dataset:
            total_count += 1
            
            try:
                # Auto-detect format on first sample if needed
                nonlocal image_key, label_key
                if auto_detect_format and (image_key is None or label_key is None):
                    detected_keys = detect_dataset_format(sample)
                    if image_key is None:
                        image_key = detected_keys.get('image')
                    if label_key is None:
                        label_key = detected_keys.get('label')
                
                if image_key is None:
                    raise ValueError("Could not detect image key in dataset")
                if label_key is None:
                    raise ValueError("Could not detect label key in dataset")
                
                # Extract and preprocess image
                image = sample[image_key]
                processed_image = safe_image_preprocessing(
                    image, image_size, normalize, dtype
                )
                
                # Extract label
                label = sample[label_key]
                if not isinstance(label, (int, tf.Tensor)):
                    label = int(label) if hasattr(label, '__int__') else 0
                
                yield {'image': processed_image, 'label': tf.cast(label, tf.int64)}
                
            except Exception as e:
                skip_count += 1
                logger.warning(f"Skipping sample {total_count}: {e}")
                
                if error_handling == 'raise':
                    raise
                elif error_handling == 'placeholder':
                    # Yield placeholder data
                    placeholder_image = tf.zeros(
                        (*image_size, 3) if image_size else (224, 224, 3), 
                        dtype=dtype
                    )
                    yield {'image': placeholder_image, 'label': tf.cast(0, tf.int64)}
                # 'skip' continues to next sample
        
        if skip_count > 0:
            logger.info(f"Skipped {skip_count}/{total_count} samples due to errors")
    
    # Infer output signature from first sample
    try:
        first_sample = next(iter(hf_dataset))
        
        # Auto-detect format if needed
        if auto_detect_format and (image_key is None or label_key is None):
            detected_keys = detect_dataset_format(first_sample)
            if image_key is None:
                image_key = detected_keys.get('image')
            if label_key is None:
                label_key = detected_keys.get('label')
        
        if image_key is None or label_key is None:
            raise ValueError(f"Could not detect required keys. Available keys: {list(first_sample.keys())}")
        
        # Process first sample to determine shapes
        sample_image = safe_image_preprocessing(
            first_sample[image_key], image_size, normalize, dtype
        )
        
        output_signature = {
            'image': tf.TensorSpec(shape=sample_image.shape, dtype=dtype),
            'label': tf.TensorSpec(shape=(), dtype=tf.int64)
        }
        
    except Exception as e:
        logger.error(f"Failed to process first sample: {e}")
        # Fallback signature
        image_shape = (*image_size, 3) if image_size else (None, None, 3)
        output_signature = {
            'image': tf.TensorSpec(shape=image_shape, dtype=dtype),
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
    normalize: bool = True,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    retry_count: int = DEFAULT_RETRY_COUNT,
    validate_config: bool = True,
    preprocessing_fn: Optional[Callable] = None,
    **hf_kwargs
) -> Optional[tf.data.Dataset]:
    """Create a robust streaming vision dataset from HuggingFace Hub.
    
    Args:
        dataset_name: Name of the HuggingFace dataset.
        split: Dataset split.
        batch_size: Batch size.
        image_size: Target image size for resizing.
        shuffle_buffer_size: Buffer size for shuffling.
        prefetch_size: Prefetch buffer size.
        normalize: Whether to normalize images.
        cache_dir: Directory to cache dataset files.
        retry_count: Number of retry attempts for failed operations.
        validate_config: Whether to validate dataset config.
        preprocessing_fn: Optional custom preprocessing function.
        **hf_kwargs: Additional arguments for HuggingFace load_dataset.
        
    Returns:
        tf.data.Dataset: Batched and preprocessed TensorFlow dataset.
        None: If HuggingFace is not available or dataset loading fails.
    """
    if not HF_AVAILABLE:
        warnings.warn(
            "HuggingFace datasets not available. Install with: pip install datasets",
            UserWarning
        )
        return None
    
    try:
        logger.info(f"Creating streaming vision dataset: {dataset_name}")
        
        # Load streaming dataset with retry logic
        hf_dataset = create_hf_streaming_dataset(
            dataset_name, 
            split=split, 
            cache_dir=cache_dir,
            validate=validate_config,
            **hf_kwargs
        )
        
        # Convert to TensorFlow dataset
        tf_dataset = hf_to_tf_dataset(
            hf_dataset,
            image_size=image_size,
            normalize=normalize,
            auto_detect_format=True,
            error_handling='skip'
        )
        
        # Apply custom preprocessing if provided
        if preprocessing_fn:
            tf_dataset = tf_dataset.map(
                preprocessing_fn,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Apply standard preprocessing pipeline
        if shuffle_buffer_size > 0:
            tf_dataset = tf_dataset.shuffle(shuffle_buffer_size)
        
        # Handle batching with error recovery
        tf_dataset = tf_dataset.batch(batch_size, drop_remainder=False)
        
        # Add prefetching for performance
        tf_dataset = tf_dataset.prefetch(prefetch_size)
        
        logger.info(f"Successfully created streaming dataset for {dataset_name}")
        return tf_dataset
        
    except Exception as e:
        logger.error(f"Failed to create streaming dataset {dataset_name}: {e}")
        warnings.warn(
            f"Failed to create streaming dataset {dataset_name}: {e}. "
            "Consider using fallback TensorFlow Datasets.",
            UserWarning
        )
        return None


def list_hf_vision_datasets(include_community: bool = False) -> Dict[str, list[str]]:
    """List popular HuggingFace vision datasets organized by category.
    
    Args:
        include_community: Whether to include community datasets.
        
    Returns:
        Dict[str, list[str]]: Datasets organized by category.
    """
    datasets = {
        'classification': [
            'cifar10',
            'cifar100',
            'imagenet-1k',
            'food101',
            'oxford-iiit-pet',
            'caltech101',
            'fashion_mnist',
            'mnist',
        ],
        'object_detection': [
            'coco',
            'voc2012',
        ],
        'face_recognition': [
            'celeba',
            'vggface2',
        ],
        'specialized': [
            'svhn',
            'stl10',
            'eurosat',
            'resisc45',
        ]
    }
    
    if include_community:
        datasets['community'] = [
            'beans',
            'cats_vs_dogs',
            'stanford_cars',
        ]
    
    return datasets


def get_dataset_categories() -> list[str]:
    """Get available dataset categories."""
    return list(list_hf_vision_datasets().keys())


@retry_on_failure(max_retries=2)
def get_hf_dataset_info(dataset_name: str, config_name: Optional[str] = None) -> Dict[str, Any]:
    """Get comprehensive information about a HuggingFace dataset.
    
    Args:
        dataset_name: Name of the dataset.
        config_name: Optional configuration name.
        
    Returns:
        Dict[str, Any]: Comprehensive dataset information.
    """
    if not HF_AVAILABLE:
        return {'error': 'HuggingFace datasets library not available'}
    
    try:
        logger.info(f"Retrieving info for dataset: {dataset_name}")
        
        # Try to get dataset info without downloading
        from datasets import get_dataset_config_names, get_dataset_split_names
        
        info = {
            'dataset_name': dataset_name,
            'available': True,
            'configs': [],
            'splits': {},
            'sample_info': {},
            'recommendations': {}
        }
        
        # Get available configurations
        try:
            configs = get_dataset_config_names(dataset_name)
            info['configs'] = configs
            if config_name and config_name not in configs:
                logger.warning(f"Config {config_name} not found. Available: {configs}")
        except Exception as e:
            logger.warning(f"Could not retrieve configs for {dataset_name}: {e}")
            info['configs'] = [config_name] if config_name else ['default']
        
        # Get splits for each config
        target_config = config_name or (info['configs'][0] if info['configs'] else None)
        try:
            splits = get_dataset_split_names(dataset_name, config_name=target_config)
            info['splits'][target_config or 'default'] = splits
        except Exception as e:
            logger.warning(f"Could not retrieve splits for {dataset_name}: {e}")
            info['splits'][target_config or 'default'] = ['train', 'test']
        
        # Sample the dataset to get structure info
        try:
            sample_dataset = load_dataset(
                dataset_name, 
                split='train[:1]',
                streaming=False,
                name=target_config
            )
            first_sample = next(iter(sample_dataset))
            
            # Analyze sample structure
            sample_info = {
                'features': list(first_sample.keys()),
                'feature_types': {k: str(type(v).__name__) for k, v in first_sample.items()}
            }
            
            # Detect data types
            detected_keys = detect_dataset_format(first_sample)
            sample_info.update(detected_keys)
            
            # Check for images
            sample_info['has_images'] = 'image' in detected_keys
            sample_info['has_labels'] = 'label' in detected_keys
            
            # Get image info if available
            if sample_info['has_images']:
                image_key = detected_keys['image']
                image = first_sample[image_key]
                if hasattr(image, 'size'):
                    sample_info['image_size'] = image.size
                elif hasattr(image, 'shape'):
                    sample_info['image_shape'] = image.shape
            
            info['sample_info'] = sample_info
            
        except Exception as e:
            logger.warning(f"Could not sample dataset {dataset_name}: {e}")
            info['sample_info'] = {'error': str(e)}
        
        # Add usage recommendations
        info['recommendations'] = _get_usage_recommendations(info)
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get info for {dataset_name}: {e}")
        return {
            'dataset_name': dataset_name,
            'available': False,
            'error': str(e),
            'suggestion': 'Check dataset name and internet connection'
        }


def _get_usage_recommendations(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    """Generate usage recommendations based on dataset info."""
    recommendations = {}
    
    sample_info = dataset_info.get('sample_info', {})
    
    # Recommend image size based on detected size
    if 'image_size' in sample_info:
        width, height = sample_info['image_size']
        if width > 224 or height > 224:
            recommendations['resize_suggestion'] = (224, 224)
        else:
            recommendations['resize_suggestion'] = (width, height)
    else:
        recommendations['resize_suggestion'] = (224, 224)
    
    # Recommend batch size based on expected image size
    if 'image_size' in sample_info or 'image_shape' in sample_info:
        size = sample_info.get('image_size', sample_info.get('image_shape', (224, 224)))
        if isinstance(size, (tuple, list)) and len(size) >= 2:
            pixels = size[0] * size[1]
            if pixels > 224 * 224:
                recommendations['batch_size_suggestion'] = 16
            else:
                recommendations['batch_size_suggestion'] = 32
    
    # Recommend preprocessing
    if sample_info.get('has_images'):
        recommendations['preprocessing'] = ['resize', 'normalize']
        if 'label' in sample_info:
            recommendations['task_type'] = 'classification'
    
    return recommendations


def clear_hf_cache(cache_dir: Optional[str] = None) -> bool:
    """Clear HuggingFace datasets cache.
    
    Args:
        cache_dir: Cache directory to clear (uses default if None).
        
    Returns:
        bool: True if successful, False otherwise.
    """
    if not HF_AVAILABLE:
        logger.warning("HuggingFace datasets not available")
        return False
    
    try:
        # Try to get default cache directory
        if cache_dir is None:
            try:
                from datasets.config import HF_DATASETS_CACHE
                cache_dir = HF_DATASETS_CACHE
            except ImportError:
                cache_dir = Path.home() / '.cache' / 'huggingface' / 'datasets'
        
        import shutil
        
        target_dir = Path(cache_dir)
        if target_dir.exists():
            shutil.rmtree(target_dir)
            logger.info(f"Cleared cache directory: {target_dir}")
            return True
        else:
            logger.info("Cache directory does not exist")
            return True
            
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return False


def get_cache_size(cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """Get information about cache usage.
    
    Args:
        cache_dir: Cache directory to analyze.
        
    Returns:
        Dict with cache information.
    """
    if not HF_AVAILABLE:
        return {'error': 'HuggingFace datasets not available'}
    
    try:
        # Try to get default cache directory from datasets
        if cache_dir is None:
            try:
                from datasets.config import HF_DATASETS_CACHE
                cache_dir = HF_DATASETS_CACHE
            except ImportError:
                # Fallback to user cache directory
                cache_dir = Path.home() / '.cache' / 'huggingface' / 'datasets'
        
        target_dir = Path(cache_dir)
        
        if not target_dir.exists():
            return {
                'cache_dir': str(target_dir),
                'exists': False,
                'size_bytes': 0,
                'size_mb': 0,
                'file_count': 0
            }
        
        total_size = 0
        file_count = 0
        
        for file_path in target_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        return {
            'cache_dir': str(target_dir),
            'exists': True,
            'size_bytes': total_size,
            'size_mb': round(total_size / (1024 * 1024), 2),
            'file_count': file_count
        }
        
    except Exception as e:
        return {'error': str(e)}


def create_dataset_benchmark(
    dataset_name: str,
    split: str = 'train',
    num_samples: int = 100,
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224)
) -> Dict[str, Any]:
    """Benchmark dataset loading performance.
    
    Args:
        dataset_name: Name of dataset to benchmark.
        split: Dataset split to use.
        num_samples: Number of samples to process.
        batch_size: Batch size for processing.
        image_size: Target image size.
        
    Returns:
        Dict with benchmark results.
    """
    if not HF_AVAILABLE:
        return {'error': 'HuggingFace datasets not available'}
    
    try:
        logger.info(f"Benchmarking dataset: {dataset_name}")
        start_time = time.time()
        
        # Create dataset
        dataset = create_streaming_vision_dataset(
            dataset_name=dataset_name,
            split=split,
            batch_size=batch_size,
            image_size=image_size,
            shuffle_buffer_size=0,  # Disable shuffle for consistent timing
            validate_config=False  # Skip validation for speed
        )
        
        if dataset is None:
            return {'error': 'Failed to create dataset'}
        
        # Process samples
        sample_count = 0
        processing_times = []
        
        for batch in dataset.take(num_samples // batch_size):
            batch_start = time.time()
            
            # Force evaluation
            _ = batch['image'].numpy()
            _ = batch['label'].numpy()
            
            batch_end = time.time()
            processing_times.append(batch_end - batch_start)
            sample_count += batch['image'].shape[0]
            
            if sample_count >= num_samples:
                break
        
        total_time = time.time() - start_time
        
        return {
            'dataset_name': dataset_name,
            'samples_processed': sample_count,
            'total_time_seconds': round(total_time, 3),
            'samples_per_second': round(sample_count / total_time, 2),
            'avg_batch_time': round(sum(processing_times) / len(processing_times), 4),
            'batch_size': batch_size,
            'image_size': image_size
        }
        
    except Exception as e:
        logger.error(f"Benchmark failed for {dataset_name}: {e}")
        return {'error': str(e)}


# Backwards compatibility functions
def list_hf_vision_datasets_legacy() -> list[str]:
    """Legacy function for backwards compatibility."""
    datasets_dict = list_hf_vision_datasets()
    all_datasets = []
    for category_datasets in datasets_dict.values():
        all_datasets.extend(category_datasets)
    return sorted(list(set(all_datasets)))