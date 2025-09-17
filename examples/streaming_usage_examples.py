#!/usr/bin/env python3
"""
Example usage of enhanced HuggingFace streaming dataset support in VISX.

This script demonstrates the robust and modular features added to the
HuggingFace streaming dataset integration.
"""

import sys
from pathlib import Path

# Add repo root to path
script_dir = Path(__file__).parent
repo_root = script_dir.parent if script_dir.name == 'examples' else script_dir
sys.path.insert(0, str(repo_root))

def example_basic_usage():
    """Example of basic enhanced streaming dataset usage."""
    print("🔄 Basic Enhanced Streaming Usage")
    print("=" * 50)
    
    try:
        from visx.data.streaming import (
            is_huggingface_available,
            create_streaming_vision_dataset,
            validate_dataset_config
        )
        
        if not is_huggingface_available():
            print("❌ HuggingFace datasets not available")
            return
        
        # Validate dataset before loading
        print("1. Validating dataset configuration...")
        validation = validate_dataset_config('cifar10', split='train')
        print(f"   Validation: {'✅ Valid' if validation['valid'] else '❌ Invalid'}")
        
        if not validation['valid']:
            print(f"   Error: {validation['error']}")
            print(f"   Suggestion: {validation['suggestion']}")
            return
        
        # Create robust streaming dataset
        print("2. Creating streaming dataset with robust error handling...")
        dataset = create_streaming_vision_dataset(
            dataset_name='cifar10',
            split='train',
            batch_size=32,
            image_size=(224, 224),
            shuffle_buffer_size=1000,
            validate_config=True,
            retry_count=3
        )
        
        if dataset is not None:
            print("   ✅ Dataset created successfully!")
            print("   Dataset is ready for training with:")
            print("     - Automatic retry on network failures")
            print("     - Format detection and preprocessing")
            print("     - Error recovery and graceful fallbacks")
        else:
            print("   ❌ Dataset creation failed (network/availability issues)")
            
    except Exception as e:
        print(f"❌ Basic usage example failed: {e}")


def example_dataset_exploration():
    """Example of exploring datasets with enhanced info."""
    print("\n📊 Dataset Exploration Example")
    print("=" * 50)
    
    try:
        from visx.data.streaming import (
            list_hf_vision_datasets,
            get_dataset_categories,
            get_hf_dataset_info
        )
        
        # Explore dataset categories
        print("1. Exploring dataset categories...")
        categories = get_dataset_categories()
        datasets_by_category = list_hf_vision_datasets()
        
        for category in categories[:2]:  # Show first 2 categories
            datasets = datasets_by_category[category]
            print(f"   📂 {category.title()}: {datasets[:3]}...")
        
        # Get detailed info about a dataset
        print("2. Getting detailed dataset information...")
        info = get_hf_dataset_info('cifar10')
        
        if 'error' not in info:
            print("   ✅ Dataset info retrieved:")
            print(f"     Available: {info.get('available', 'N/A')}")
            print(f"     Configs: {info.get('configs', 'N/A')}")
            
            if 'recommendations' in info:
                recs = info['recommendations']
                print("   💡 AI Recommendations:")
                print(f"     Image size: {recs.get('resize_suggestion', 'N/A')}")
                print(f"     Batch size: {recs.get('batch_size_suggestion', 'N/A')}")
                print(f"     Task type: {recs.get('task_type', 'N/A')}")
        else:
            print(f"   ❌ Could not get dataset info: {info['error']}")
            
    except Exception as e:
        print(f"❌ Dataset exploration example failed: {e}")


def example_advanced_preprocessing():
    """Example of advanced preprocessing with custom functions."""
    print("\n🛠️  Advanced Preprocessing Example")
    print("=" * 50)
    
    try:
        import tensorflow as tf
        from visx.data.streaming import create_streaming_vision_dataset
        
        # Define custom preprocessing function
        def custom_preprocessing(batch):
            """Custom preprocessing function for additional augmentations."""
            images = batch['image']
            labels = batch['label']
            
            # Add custom augmentations
            images = tf.image.random_flip_left_right(images)
            images = tf.image.random_brightness(images, 0.1)
            
            return {'image': images, 'label': labels}
        
        print("1. Creating dataset with custom preprocessing...")
        dataset = create_streaming_vision_dataset(
            dataset_name='cifar10',
            split='train',
            batch_size=16,
            image_size=(224, 224),
            preprocessing_fn=custom_preprocessing,
            validate_config=False  # Skip validation for demo
        )
        
        if dataset is not None:
            print("   ✅ Dataset with custom preprocessing created!")
            print("   Features:")
            print("     - Random horizontal flip")
            print("     - Random brightness adjustment")
            print("     - Automatic format detection")
            print("     - Robust error handling")
        else:
            print("   ❌ Custom preprocessing example unavailable (network issues)")
            
    except Exception as e:
        print(f"❌ Advanced preprocessing example failed: {e}")


def example_cache_management():
    """Example of cache management utilities."""
    print("\n💾 Cache Management Example")
    print("=" * 50)
    
    try:
        from visx.data.streaming import get_cache_size, clear_hf_cache
        
        # Check cache size
        print("1. Checking cache usage...")
        cache_info = get_cache_size()
        
        if 'error' not in cache_info:
            print(f"   📁 Cache location: {cache_info['cache_dir']}")
            print(f"   📊 Cache size: {cache_info['size_mb']} MB")
            print(f"   📄 Files: {cache_info['file_count']}")
            
            # Note: We won't actually clear cache in the example
            print("2. Cache clearing available...")
            print("   Use clear_hf_cache() to clear cache when needed")
            print("   ⚠️  This will remove all cached datasets")
        else:
            print(f"   ❌ Cache info error: {cache_info['error']}")
            
    except Exception as e:
        print(f"❌ Cache management example failed: {e}")


def example_performance_monitoring():
    """Example of performance monitoring and benchmarking."""
    print("\n⚡ Performance Monitoring Example")
    print("=" * 50)
    
    try:
        from visx.data.streaming import create_dataset_benchmark
        
        print("1. Running dataset benchmark...")
        print("   (This may take a moment and might fail due to network issues)")
        
        benchmark = create_dataset_benchmark(
            dataset_name='cifar10',
            split='train',
            num_samples=20,  # Small number for quick demo
            batch_size=4,
            image_size=(224, 224)
        )
        
        if 'error' not in benchmark:
            print("   ✅ Benchmark completed:")
            print(f"     Samples processed: {benchmark['samples_processed']}")
            print(f"     Total time: {benchmark['total_time_seconds']}s")
            print(f"     Throughput: {benchmark['samples_per_second']} samples/sec")
            print(f"     Avg batch time: {benchmark['avg_batch_time']}s")
        else:
            print(f"   ❌ Benchmark error (expected): {benchmark['error']}")
            print("   This is normal in environments without HuggingFace Hub access")
            
    except Exception as e:
        print(f"❌ Performance monitoring example failed: {e}")


def main():
    """Run all examples."""
    print("🚀 VISX Enhanced Streaming Dataset Examples")
    print("=" * 80)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Dataset Exploration", example_dataset_exploration),
        ("Advanced Preprocessing", example_advanced_preprocessing),
        ("Cache Management", example_cache_management),
        ("Performance Monitoring", example_performance_monitoring),
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ {name} example failed: {e}")
    
    print("\n" + "=" * 80)
    print("📋 Summary")
    print("These examples demonstrate the enhanced robustness and modularity of")
    print("HuggingFace streaming dataset support in VISX, including:")
    print("• Robust error handling and retry mechanisms")
    print("• Automatic dataset format detection")
    print("• Performance monitoring and benchmarking")
    print("• Cache management utilities")
    print("• Enhanced dataset information and recommendations")
    print("• Comprehensive validation and fallback systems")
    print("\n✅ Enhanced streaming dataset support is ready for production use!")


if __name__ == "__main__":
    main()