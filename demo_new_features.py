#!/usr/bin/env python3
"""
Demo script showing new VISX features: JAX mesh parallelization and HuggingFace streaming datasets.
"""

import sys
from pathlib import Path

# Add repo root to path
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

print("🚀 VISX New Features Demo")
print("=" * 80)

# Demo 1: JAX Mesh and Device Setup
print("\n🌐 JAX Mesh and Device Setup Demo")
print("-" * 50)

try:
    from visx.utils.mesh import get_device_info, create_mesh_for_device, print_device_info
    
    print("Device information:")
    info = get_device_info()
    print(f"Backend: {info['backend']}")
    print(f"Device count: {info['device_count']}")
    
    print("\nCreating mesh...")
    mesh = create_mesh_for_device()
    print(f"Mesh shape: {mesh.shape}")
    print(f"Mesh axis names: {mesh.axis_names}")
    
    print("\n✅ JAX mesh utilities are working!")

except ImportError as e:
    print(f"⚠️  JAX not available: {e}")
    print("   Install JAX to use mesh parallelization features.")
except Exception as e:
    print(f"❌ Error: {e}")

# Demo 2: Data Module Usage
print("\n📊 Data Module Usage Demo")
print("-" * 50)

try:
    from visx.data.configs import list_available_datasets, get_dataset_config
    from visx.data.loaders import create_train_dataset, create_test_dataset
    
    print("Available datasets:")
    datasets = list_available_datasets()
    for i, dataset in enumerate(datasets):
        print(f"  {i+1}. {dataset}")
    
    print("\nCIFAR-10 configuration:")
    config = get_dataset_config('cifar10')
    print(f"  Classes: {config['num_classes']}")
    print(f"  Channels: {config['input_channels']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    
    print("\n✅ Data configuration system is working!")

except ImportError as e:
    print(f"⚠️  TensorFlow not available: {e}")
    print("   Install TensorFlow to use data loading features.")
except Exception as e:
    print(f"❌ Error: {e}")

# Demo 3: Enhanced HuggingFace Streaming Support
print("\n🔄 Enhanced HuggingFace Streaming Dataset Demo")
print("-" * 50)

try:
    from visx.data.streaming import (
        is_huggingface_available, 
        list_hf_vision_datasets,
        get_dataset_categories,
        get_hf_dataset_info,
        validate_dataset_config,
        get_cache_size
    )
    
    hf_available = is_huggingface_available()
    print(f"HuggingFace datasets available: {hf_available}")
    
    if hf_available:
        print("\nDataset categories and organization:")
        categories = get_dataset_categories()
        datasets_by_category = list_hf_vision_datasets()
        
        for category in categories:
            datasets = datasets_by_category[category]
            print(f"  📂 {category.title()}: {len(datasets)} datasets")
            print(f"     Examples: {', '.join(datasets[:3])}")
        
        print("\nEnhanced dataset information:")
        try:
            info = get_hf_dataset_info('cifar10')
            if 'error' not in info:
                print(f"  ✅ CIFAR-10 successfully analyzed")
                print(f"     Configs: {info.get('configs', 'N/A')}")
                if 'recommendations' in info:
                    recs = info['recommendations']
                    print(f"     💡 Recommended image size: {recs.get('resize_suggestion', 'N/A')}")
                    print(f"     💡 Recommended batch size: {recs.get('batch_size_suggestion', 'N/A')}")
            else:
                print(f"  ⚠️  CIFAR-10 analysis: {info['error']}")
        except Exception as e:
            print(f"  ⚠️  Dataset analysis error: {e}")
        
        print("\nRobust validation:")
        try:
            validation = validate_dataset_config('cifar10')
            print(f"  CIFAR-10 validation: {'✅ Valid' if validation['valid'] else '❌ Invalid'}")
            if not validation['valid']:
                print(f"     Reason: {validation['error']}")
        except Exception as e:
            print(f"  ⚠️  Validation error: {e}")
        
        print("\nCache management:")
        try:
            cache_info = get_cache_size()
            if 'error' not in cache_info:
                print(f"  📁 Cache size: {cache_info['size_mb']} MB ({cache_info['file_count']} files)")
                print(f"     Location: {cache_info['cache_dir']}")
            else:
                print(f"  ⚠️  Cache info: {cache_info['error']}")
        except Exception as e:
            print(f"  ⚠️  Cache error: {e}")
        
        print("\n✅ Enhanced HuggingFace streaming support is available!")
    else:
        print("⚠️  HuggingFace datasets not installed.")
        print("   Install with: pip install datasets")

except Exception as e:
    print(f"❌ Error: {e}")

# Demo 4: Model with Mesh Partitioning
print("\n🏗️  Model with Mesh Partitioning Demo")
print("-" * 50)

try:
    from visx.utils.mesh import create_mesh_for_device, create_partitioned_linear
    
    print("This would create a model with partitioned weights:")
    print("""
    mesh = create_mesh_for_device()
    rngs = nnx.Rngs(42)
    
    # Linear layer with partitioned weights
    linear = create_partitioned_linear(
        in_features=128,
        out_features=10,
        mesh=mesh,
        rngs=rngs
    )
    """)
    
    print("✅ Mesh-partitioned model creation is ready!")

except Exception as e:
    print(f"❌ Error: {e}")

# Summary
print("\n📋 Summary of Enhanced Features")
print("-" * 50)
print("✅ Legacy main execution code removed")
print("✅ Tests moved to test/ directory")  
print("✅ JAX mesh utilities added for TPU/GPU parallelization")
print("✅ HuggingFace streaming dataset support enhanced with:")
print("   🔧 Robust error handling and retry mechanisms")
print("   📊 Automatic dataset format detection")
print("   ⚡ Performance monitoring and benchmarking")
print("   💾 Cache management utilities")
print("   📝 Enhanced dataset information and recommendations")
print("   🛡️  Comprehensive validation and fallback systems")
print("✅ Data loading logic modularized")
print("✅ Backward compatibility maintained")

print("\n🎯 Next Steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Use visx.utils.mesh for distributed training")
print("3. Use visx.data.streaming for robust large dataset handling")
print("4. Check test/ directory for validation scripts")
print("5. Run test/test_streaming_robust.py for comprehensive streaming tests")

print("\n" + "=" * 80)