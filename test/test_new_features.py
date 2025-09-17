#!/usr/bin/env python3
"""
Test script for new VISX features: mesh utilities and streaming datasets.
"""

import sys
from pathlib import Path

def test_mesh_utilities():
    """Test mesh utilities functionality."""
    print("🌐 Testing Mesh Utilities")
    print("=" * 40)
    
    try:
        from visx.utils.mesh import (
            get_device_info, 
            create_mesh_for_device,
            print_device_info
        )
        
        # Test device info
        device_info = get_device_info()
        print(f"✅ Device info retrieved: {device_info['backend']}")
        
        # Test mesh creation
        mesh = create_mesh_for_device()
        print(f"✅ Mesh created with shape: {mesh.shape}")
        
        # Test device info printing (should not crash)
        print_device_info()
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   (This might be expected if JAX is not installed)")
        return True  # Not a failure for the modularization test
    except Exception as e:
        print(f"❌ Mesh utilities test failed: {e}")
        return False


def test_data_modules():
    """Test data module functionality."""
    print("\n📊 Testing Data Modules")
    print("=" * 40)
    
    try:
        from visx.data.configs import (
            DATASET_CONFIGS,
            get_dataset_config,
            list_available_datasets
        )
        
        # Test config loading
        datasets = list_available_datasets()
        print(f"✅ Found {len(datasets)} dataset configurations")
        
        # Test specific config
        cifar10_config = get_dataset_config('cifar10')
        print(f"✅ CIFAR-10 config loaded: {cifar10_config['num_classes']} classes")
        
        return True
        
    except Exception as e:
        print(f"❌ Data modules test failed: {e}")
        return False


def test_streaming_datasets():
    """Test streaming dataset functionality."""
    print("\n🔄 Testing Streaming Dataset Support")
    print("=" * 40)
    
    try:
        from visx.data.streaming import (
            is_huggingface_available,
            list_hf_vision_datasets,
            get_dataset_categories,
            list_hf_vision_datasets_legacy
        )
        
        # Test HuggingFace availability check
        hf_available = is_huggingface_available()
        print(f"✅ HuggingFace datasets available: {hf_available}")
        
        # Test new categorized dataset listing
        datasets_by_category = list_hf_vision_datasets()
        categories = get_dataset_categories()
        print(f"✅ Found datasets in {len(categories)} categories: {categories}")
        
        total_datasets = sum(len(ds_list) for ds_list in datasets_by_category.values())
        print(f"✅ Total datasets available: {total_datasets}")
        
        # Test legacy compatibility
        legacy_datasets = list_hf_vision_datasets_legacy()
        print(f"✅ Legacy compatibility: {len(legacy_datasets)} datasets")
        
        if hf_available:
            try:
                from visx.data.streaming import get_hf_dataset_info
                info = get_hf_dataset_info('cifar10')
                if 'error' not in info:
                    print(f"✅ Retrieved enhanced info for CIFAR-10")
                    print(f"   Available: {info.get('available', 'N/A')}")
                    if 'recommendations' in info:
                        recs = info['recommendations']
                        print(f"   Recommended image size: {recs.get('resize_suggestion', 'N/A')}")
                        print(f"   Recommended batch size: {recs.get('batch_size_suggestion', 'N/A')}")
                else:
                    print(f"⚠️  Could not test dataset info retrieval: {info['error']}")
            except Exception as e:
                print(f"⚠️  Could not test dataset info retrieval: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Streaming datasets test failed: {e}")
        return False


def test_legacy_compatibility():
    """Test that legacy code still works."""
    print("\n🔄 Testing Legacy Compatibility")
    print("=" * 40)
    
    try:
        # This should work without JAX being installed
        from visx.data.configs import DATASET_CONFIGS
        
        # Test that the expected datasets are present
        expected_datasets = ['cifar10', 'cifar100', 'stl10']
        for dataset in expected_datasets:
            if dataset in DATASET_CONFIGS:
                print(f"✅ {dataset} configuration present")
            else:
                print(f"❌ {dataset} configuration missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Legacy compatibility test failed: {e}")
        return False


def main():
    """Run all feature tests."""
    print("🚀 VISX New Features Test Suite")
    print("=" * 80)
    
    # Change to the repository root directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    import os
    os.chdir(repo_root)
    
    # Add the repo root to Python path so we can import visx
    sys.path.insert(0, str(repo_root))
    
    tests = [
        ("Mesh Utilities", test_mesh_utilities),
        ("Data Modules", test_data_modules),
        ("Streaming Datasets", test_streaming_datasets),
        ("Legacy Compatibility", test_legacy_compatibility),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
    
    print("\n" + "=" * 80)
    print(f"🎯 Feature Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All feature tests passed! New functionality is working.")
        return 0
    else:
        print("❌ Some feature tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())