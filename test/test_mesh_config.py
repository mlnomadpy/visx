#!/usr/bin/env python3
"""
Test script for mesh configuration functionality.
"""

import sys
import tempfile
import os
from pathlib import Path


def test_mesh_config_structure():
    """Test mesh configuration dataclass structure."""
    print("🌐 Testing Mesh Configuration Structure")
    print("=" * 50)
    
    try:
        # Add the repo root to Python path so we can import visx
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        sys.path.insert(0, str(repo_root))
        
        from visx.config.config import MeshConfig, Config
        
        # Test default mesh config
        mesh_config = MeshConfig()
        assert mesh_config.enabled == True
        assert mesh_config.auto_detect == True
        assert mesh_config.shape is None
        assert mesh_config.axis_names is None
        print("✅ Default MeshConfig created successfully")
        
        # Test custom mesh config
        custom_mesh = MeshConfig(
            enabled=True,
            auto_detect=False,
            shape=[4, 2],
            axis_names=['batch', 'model'],
            tpu_mesh_shape=[4, 2],
            tpu_axis_names=['batch', 'model'],
            gpu_mesh_shape=[8, 1],
            gpu_axis_names=['batch', 'model']
        )
        assert custom_mesh.shape == [4, 2]
        assert custom_mesh.axis_names == ['batch', 'model']
        print("✅ Custom MeshConfig created successfully")
        
        # Test mesh config in main config
        config = Config()
        assert hasattr(config, 'mesh')
        assert isinstance(config.mesh, MeshConfig)
        print("✅ MeshConfig integrated into main Config")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Mesh config test failed: {e}")
        return False


def test_mesh_config_yaml():
    """Test mesh configuration in YAML files."""
    print("\n📄 Testing Mesh Configuration in YAML")
    print("=" * 50)
    
    try:
        # Add the repo root to Python path
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        sys.path.insert(0, str(repo_root))
        
        from visx.config.config import Config
        import yaml
        
        # Test YAML with mesh configuration
        test_yaml = """
mode: training
dataset:
  name: cifar10
  num_classes: 10
  input_channels: 3
model:
  name: test_model
  type: yat
training:
  learning_rate: 0.001
pretraining:
  method: supervised
explainability:
  enabled: false
mesh:
  enabled: true
  auto_detect: false
  shape: [4, 2]
  axis_names: [batch, model]
  tpu_mesh_shape: [4, 2]
  tpu_axis_names: [batch, model]
  gpu_mesh_shape: [8, 1]
  gpu_axis_names: [batch, model]
"""
        
        # Write to temporary file and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(test_yaml)
            temp_yaml_path = f.name
        
        try:
            config = Config.from_yaml(temp_yaml_path)
            
            # Test mesh configuration loaded correctly
            assert config.mesh.enabled == True
            assert config.mesh.auto_detect == False
            assert config.mesh.shape == [4, 2]
            assert config.mesh.axis_names == ['batch', 'model']
            assert config.mesh.tpu_mesh_shape == [4, 2]
            assert config.mesh.tpu_axis_names == ['batch', 'model']
            assert config.mesh.gpu_mesh_shape == [8, 1]
            assert config.mesh.gpu_axis_names == ['batch', 'model']
            print("✅ Mesh configuration loaded from YAML successfully")
            
            # Test saving to YAML
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                output_yaml_path = f.name
            
            config.to_yaml(output_yaml_path)
            
            # Load the saved file and verify
            with open(output_yaml_path, 'r') as f:
                saved_config = yaml.safe_load(f)
            
            assert 'mesh' in saved_config
            assert saved_config['mesh']['enabled'] == True
            assert saved_config['mesh']['auto_detect'] == False
            assert saved_config['mesh']['shape'] == [4, 2]
            print("✅ Mesh configuration saved to YAML successfully")
            
            return True
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_yaml_path):
                os.unlink(temp_yaml_path)
            if os.path.exists(output_yaml_path):
                os.unlink(output_yaml_path)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ YAML mesh config test failed: {e}")
        return False


def test_mesh_function_interface():
    """Test the mesh utility function interfaces."""
    print("\n🔧 Testing Mesh Utility Function Interfaces")
    print("=" * 50)
    
    try:
        # Add the repo root to Python path
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        sys.path.insert(0, str(repo_root))
        
        from visx.config.config import MeshConfig
        from visx.utils.mesh import create_mesh_from_config, setup_distributed_training
        
        # Test create_mesh_from_config function exists and accepts MeshConfig
        mesh_config = MeshConfig(enabled=False)
        result = create_mesh_from_config(mesh_config)
        assert result is None  # Should return None when disabled
        print("✅ create_mesh_from_config function works correctly")
        
        # Test setup_distributed_training accepts optional mesh_config
        try:
            mesh, device_info = setup_distributed_training(mesh_config)
            assert mesh is None  # Should be None when mesh disabled
            assert isinstance(device_info, dict)
            print("✅ setup_distributed_training function accepts mesh_config")
        except ImportError:
            print("⚠️  JAX not available, skipping mesh creation test")
        
        return True
        
    except ImportError as e:
        if "jax" in str(e).lower():
            print("⚠️  JAX not available, skipping function tests")
            return True
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Function interface test failed: {e}")
        return False


def test_existing_config_files():
    """Test that existing config files are still valid."""
    print("\n📂 Testing Existing Configuration Files")
    print("=" * 50)
    
    try:
        # Add the repo root to Python path
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        sys.path.insert(0, str(repo_root))
        
        from visx.config.config import Config
        
        config_files = [
            "configs/training_example.yaml",
            "configs/byol_pretraining.yaml",
            "configs/explainability_example.yaml",
            "configs/comparison_example.yaml"
        ]
        
        valid_configs = 0
        
        for config_file in config_files:
            config_path = repo_root / config_file
            if config_path.exists():
                try:
                    config = Config.from_yaml(str(config_path))
                    assert hasattr(config, 'mesh')
                    assert hasattr(config.mesh, 'enabled')
                    print(f"✅ {config_file} - Valid with mesh configuration")
                    valid_configs += 1
                except Exception as e:
                    print(f"❌ {config_file} - Error: {e}")
            else:
                print(f"⚠️  {config_file} - File not found")
        
        return valid_configs > 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Config file test failed: {e}")
        return False


def main():
    """Run all mesh configuration tests."""
    print("🚀 VISX Mesh Configuration Test Suite")
    print("=" * 80)
    
    # Change to the repository root directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    import os
    os.chdir(repo_root)
    
    tests = [
        ("Mesh Config Structure", test_mesh_config_structure),
        ("Mesh Config YAML", test_mesh_config_yaml),
        ("Mesh Function Interface", test_mesh_function_interface),
        ("Existing Config Files", test_existing_config_files),
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
    print(f"🎯 Mesh Configuration Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All mesh configuration tests passed! New functionality is working.")
        return 0
    else:
        print("❌ Some mesh configuration tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())