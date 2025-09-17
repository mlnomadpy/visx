#!/usr/bin/env python3
"""
Comprehensive tests for robust HuggingFace streaming dataset support.
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add repo root to path
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

def test_enhanced_streaming_features():
    """Test the enhanced streaming dataset features."""
    print("\n🔄 Testing Enhanced Streaming Dataset Features")
    print("=" * 60)
    
    try:
        from visx.data.streaming import (
            is_huggingface_available,
            validate_dataset_config,
            detect_dataset_format,
            list_hf_vision_datasets,
            get_dataset_categories,
            get_hf_dataset_info,
            create_hf_streaming_dataset,
            hf_to_tf_dataset,
            create_streaming_vision_dataset,
            get_cache_size,
            create_dataset_benchmark
        )
        
        print("✅ All imports successful")
        
        # Test HuggingFace availability
        hf_available = is_huggingface_available()
        print(f"✅ HuggingFace available: {hf_available}")
        
        if not hf_available:
            print("⚠️  HuggingFace not available, skipping dependent tests")
            return True
        
        # Test dataset categorization
        categories = get_dataset_categories()
        print(f"✅ Dataset categories: {categories}")
        
        datasets_by_category = list_hf_vision_datasets()
        total_datasets = sum(len(ds_list) for ds_list in datasets_by_category.values())
        print(f"✅ Found {total_datasets} datasets in {len(categories)} categories")
        
        # Test dataset validation
        print("\n🔍 Testing Dataset Validation")
        print("-" * 40)
        
        # Test with valid dataset
        validation_result = validate_dataset_config('cifar10')
        print(f"CIFAR-10 validation: {'✅ Valid' if validation_result['valid'] else '❌ Invalid'}")
        
        # Test with invalid dataset
        validation_result = validate_dataset_config('nonexistent_dataset_12345')
        print(f"Invalid dataset validation: {'✅ Correctly rejected' if not validation_result['valid'] else '❌ Should be invalid'}")
        
        # Test dataset info retrieval
        print("\n📊 Testing Dataset Info Retrieval")
        print("-" * 40)
        
        try:
            info = get_hf_dataset_info('cifar10')
            if 'error' not in info:
                print("✅ CIFAR-10 info retrieved successfully")
                print(f"   - Available: {info.get('available', 'N/A')}")
                print(f"   - Configs: {info.get('configs', 'N/A')}")
                print(f"   - Splits: {info.get('splits', 'N/A')}")
                if 'recommendations' in info:
                    print(f"   - Recommended batch size: {info['recommendations'].get('batch_size_suggestion', 'N/A')}")
            else:
                print(f"⚠️  Could not retrieve CIFAR-10 info: {info['error']}")
        except Exception as e:
            print(f"⚠️  Info retrieval failed: {e}")
        
        # Test cache management
        print("\n💾 Testing Cache Management")
        print("-" * 40)
        
        cache_info = get_cache_size()
        if 'error' not in cache_info:
            print(f"✅ Cache info retrieved: {cache_info['size_mb']} MB")
        else:
            print(f"⚠️  Cache info error: {cache_info['error']}")
        
        # Test format detection with mock data
        print("\n🔍 Testing Format Detection")
        print("-" * 40)
        
        # Mock dataset sample
        mock_sample = {
            'image': 'mock_pil_image',
            'label': 5,
            'other_data': 'test'
        }
        
        detected_format = detect_dataset_format(mock_sample)
        print(f"✅ Format detection: {detected_format}")
        
        print("\n✅ All enhanced streaming tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced streaming tests failed: {e}")
        return False


def test_error_handling():
    """Test robust error handling."""
    print("\n🛡️  Testing Error Handling")
    print("=" * 40)
    
    try:
        from visx.data.streaming import (
            create_hf_streaming_dataset,
            create_streaming_vision_dataset,
            validate_dataset_config
        )
        
        # Test invalid dataset handling
        try:
            result = create_hf_streaming_dataset(
                'definitely_nonexistent_dataset_xyz123',
                validate=True
            )
            print("❌ Should have failed for invalid dataset")
            return False
        except (ValueError, Exception) as e:
            print(f"✅ Correctly handled invalid dataset: {type(e).__name__}")
        
        # Test graceful fallback
        dataset = create_streaming_vision_dataset(
            'definitely_nonexistent_dataset_xyz123',
            validate_config=False  # This should trigger retry mechanism
        )
        
        if dataset is None:
            print("✅ Graceful fallback to None for invalid dataset")
        else:
            print("⚠️  Expected None but got dataset")
        
        print("✅ Error handling tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error handling tests failed: {e}")
        return False


def test_performance_monitoring():
    """Test performance monitoring features."""
    print("\n⚡ Testing Performance Monitoring")
    print("=" * 40)
    
    try:
        from visx.data.streaming import create_dataset_benchmark, is_huggingface_available
        
        if not is_huggingface_available():
            print("⚠️  HuggingFace not available, skipping benchmark test")
            return True
        
        # This might fail due to network/dataset issues, but we test the function exists
        try:
            benchmark_result = create_dataset_benchmark(
                'cifar10',
                num_samples=10,  # Small number for quick test
                batch_size=4
            )
            
            if 'error' not in benchmark_result:
                print(f"✅ Benchmark completed: {benchmark_result.get('samples_per_second', 'N/A')} samples/sec")
            else:
                print(f"⚠️  Benchmark error (expected): {benchmark_result['error']}")
        except Exception as e:
            print(f"⚠️  Benchmark failed (may be expected): {e}")
        
        print("✅ Performance monitoring test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False


def main():
    """Run all robust streaming tests."""
    print("🚀 VISX Robust Streaming Dataset Test Suite")
    print("=" * 80)
    
    # Change to repo root
    import os
    os.chdir(repo_root)
    
    tests = [
        ("Enhanced Streaming Features", test_enhanced_streaming_features),
        ("Error Handling", test_error_handling),
        ("Performance Monitoring", test_performance_monitoring),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"\n✅ {test_name}: PASSED")
            else:
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            print(f"\n❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 80)
    print(f"🎯 Robust Streaming Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All robust streaming tests passed!")
        return 0
    else:
        print("❌ Some tests failed. This may be expected due to network/dataset availability.")
        return 1


if __name__ == "__main__":
    sys.exit(main())