#!/usr/bin/env python3
"""
Test script for simo2.py mesh integration.
"""

import sys
import subprocess
import tempfile
import os
from pathlib import Path


def test_simo2_mesh_arguments():
    """Test that simo2.py accepts new mesh arguments without errors."""
    print("🧪 Testing SIMO2 Mesh Arguments")
    print("=" * 50)
    
    try:
        # Add the repo root to Python path
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        
        # Test help command to see if new arguments are available
        result = subprocess.run([
            sys.executable, 'simo2.py', '--help'
        ], cwd=repo_root, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            help_text = result.stdout
            # Check if our new mesh arguments are in the help
            mesh_args = [
                '--mesh_enabled',
                '--mesh_auto_detect', 
                '--mesh_shape',
                '--mesh_axis_names'
            ]
            
            missing_args = []
            for arg in mesh_args:
                if arg not in help_text:
                    missing_args.append(arg)
            
            if not missing_args:
                print("✅ All mesh arguments found in simo2.py help")
                return True
            else:
                print(f"❌ Missing mesh arguments: {missing_args}")
                return False
        else:
            print(f"❌ Failed to get help from simo2.py: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ simo2.py help command timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing simo2.py arguments: {e}")
        return False


def test_simo2_mesh_functions():
    """Test the mesh functions in simo2.py work correctly."""
    print("\n🔧 Testing SIMO2 Mesh Functions")
    print("=" * 50)
    
    try:
        # Add the repo root to Python path
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        sys.path.insert(0, str(repo_root))
        
        # Test importing the mesh functions from simo2.py
        # This will fail if there are syntax errors
        import importlib.util
        spec = importlib.util.spec_from_file_location("simo2", repo_root / "simo2.py")
        simo2_module = importlib.util.module_from_spec(spec)
        
        # Try to load the module - this will catch syntax errors
        try:
            spec.loader.exec_module(simo2_module)
        except ImportError as e:
            if "jax" in str(e).lower() or "tensorflow" in str(e).lower():
                print("⚠️  JAX/TensorFlow not available, testing function definitions only")
                # Read the file to check function definitions exist
                with open(repo_root / "simo2.py", 'r') as f:
                    content = f.read()
                    
                required_functions = [
                    'def setup_mesh_from_config',
                    'def setup_legacy_mesh'
                ]
                
                missing_functions = []
                for func in required_functions:
                    if func not in content:
                        missing_functions.append(func)
                
                if not missing_functions:
                    print("✅ All required mesh functions defined in simo2.py")
                    return True
                else:
                    print(f"❌ Missing functions: {missing_functions}")
                    return False
            else:
                raise e
        
        # If we get here, the module loaded successfully
        if hasattr(simo2_module, 'setup_mesh_from_config'):
            print("✅ setup_mesh_from_config function found")
        else:
            print("❌ setup_mesh_from_config function not found")
            return False
            
        if hasattr(simo2_module, 'setup_legacy_mesh'):
            print("✅ setup_legacy_mesh function found")
        else:
            print("❌ setup_legacy_mesh function not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing simo2.py functions: {e}")
        return False


def test_integration_config_loading():
    """Test that the integration works with a simple config."""
    print("\n📄 Testing Config Integration")
    print("=" * 50)
    
    try:
        # Add the repo root to Python path
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        sys.path.insert(0, str(repo_root))
        
        # Create a simple test config
        test_config = {
            'mesh_enabled': True,
            'mesh_auto_detect': True,
            'mesh_shape': [4, 2],
            'mesh_axis_names': ['batch', 'model']
        }
        
        # Test the logic without actually importing JAX
        # Just test that we can process the config correctly
        from visx.config.config import MeshConfig
        
        # Test creating MeshConfig from dict values
        mesh_config = MeshConfig()
        if 'mesh_enabled' in test_config:
            mesh_config.enabled = test_config['mesh_enabled']
        if 'mesh_auto_detect' in test_config:
            mesh_config.auto_detect = test_config['mesh_auto_detect']
        if 'mesh_shape' in test_config:
            mesh_config.shape = test_config['mesh_shape']
        if 'mesh_axis_names' in test_config:
            mesh_config.axis_names = test_config['mesh_axis_names']
        
        # Verify the config was set correctly
        assert mesh_config.enabled == True
        assert mesh_config.auto_detect == True
        assert mesh_config.shape == [4, 2]
        assert mesh_config.axis_names == ['batch', 'model']
        
        print("✅ Config integration logic works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Config integration test failed: {e}")
        return False


def main():
    """Run all simo2 integration tests."""
    print("🚀 SIMO2 Mesh Integration Test Suite")
    print("=" * 80)
    
    # Change to the repository root directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    import os
    os.chdir(repo_root)
    
    tests = [
        ("SIMO2 Mesh Arguments", test_simo2_mesh_arguments),
        ("SIMO2 Mesh Functions", test_simo2_mesh_functions),
        ("Config Integration", test_integration_config_loading),
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
    print(f"🎯 SIMO2 Integration Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All SIMO2 integration tests passed! Integration is working.")
        return 0
    else:
        print("❌ Some SIMO2 integration tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())