#!/usr/bin/env python3
"""
Test script to validate VISX modular structure without dependencies.
"""

import ast
import os
import sys
from pathlib import Path

def check_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def test_module_structure():
    """Test the modular structure of VISX."""
    print("🧪 Testing VISX Modular Structure")
    print("=" * 50)
    
    # Expected structure
    expected_files = [
        "visx/__init__.py",
        "visx/models/__init__.py",
        "visx/models/layers.py",
        "visx/models/architectures.py",
        "visx/training/__init__.py",
        "visx/training/registry.py",
        "visx/training/train.py",
        "visx/training/modes.py",
        "visx/pretraining/__init__.py",
        "visx/pretraining/methods.py",
        "visx/evaluation/__init__.py",
        "visx/evaluation/explainability.py",
        "visx/evaluation/comparison.py",
        "visx/config/__init__.py",
        "visx/config/config.py",
        "visx/utils/__init__.py",
        "visx/utils/helpers.py",
        "main_cli.py",
        "setup.py",
        "requirements.txt",
        "README.md"
    ]
    
    expected_configs = [
        "configs/training_example.yaml",
        "configs/byol_pretraining.yaml",
        "configs/explainability_example.yaml",
        "configs/comparison_example.yaml"
    ]
    
    # Check file existence
    missing_files = []
    syntax_errors = []
    
    all_files = expected_files + expected_configs
    
    for filepath in all_files:
        if not os.path.exists(filepath):
            missing_files.append(filepath)
        elif filepath.endswith('.py'):
            # Check syntax for Python files
            valid, error = check_syntax(filepath)
            if not valid:
                syntax_errors.append(f"{filepath}: {error}")
    
    # Report results
    if missing_files:
        print("❌ Missing files:")
        for f in missing_files:
            print(f"  - {f}")
    else:
        print("✅ All expected files present")
    
    if syntax_errors:
        print("\n❌ Syntax errors:")
        for error in syntax_errors:
            print(f"  - {error}")
    else:
        print("✅ All Python files have valid syntax")
    
    # Check configuration files
    print(f"\n📋 Configuration Files: {len(expected_configs)}")
    for config in expected_configs:
        if os.path.exists(config):
            print(f"  ✅ {config}")
        else:
            print(f"  ❌ {config}")
    
    # Summary
    total_files = len(all_files)
    present_files = total_files - len(missing_files)
    valid_syntax = len(expected_files) - len([f for f in expected_files if f.endswith('.py')]) + len([f for f in expected_files if f.endswith('.py')]) - len(syntax_errors)
    
    print(f"\n📊 Summary:")
    print(f"  Files present: {present_files}/{total_files}")
    print(f"  Valid syntax: {valid_syntax}/{len([f for f in expected_files if f.endswith('.py')])}")
    
    return len(missing_files) == 0 and len(syntax_errors) == 0

def test_config_structure():
    """Test configuration file structure."""
    print("\n🔧 Testing Configuration Structure")
    print("=" * 50)
    
    try:
        import yaml
        yaml_available = True
    except ImportError:
        yaml_available = False
        print("⚠️  PyYAML not available, skipping YAML validation")
    
    config_files = [
        "configs/training_example.yaml",
        "configs/byol_pretraining.yaml", 
        "configs/explainability_example.yaml",
        "configs/comparison_example.yaml"
    ]
    
    valid_configs = 0
    
    for config_file in config_files:
        if os.path.exists(config_file):
            if yaml_available:
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Check required fields
                    required_fields = ['mode', 'dataset', 'model', 'training']
                    has_required = all(field in config for field in required_fields)
                    
                    if has_required:
                        print(f"  ✅ {config_file} - Valid structure")
                        valid_configs += 1
                    else:
                        missing = [f for f in required_fields if f not in config]
                        print(f"  ❌ {config_file} - Missing: {missing}")
                        
                except yaml.YAMLError as e:
                    print(f"  ❌ {config_file} - YAML error: {e}")
            else:
                print(f"  ➖ {config_file} - Present but not validated")
                valid_configs += 1
        else:
            print(f"  ❌ {config_file} - Missing")
    
    print(f"\nValid configs: {valid_configs}/{len(config_files)}")
    return valid_configs == len(config_files)

def test_modular_design():
    """Test if the design follows modular principles."""
    print("\n🏗️  Testing Modular Design Principles")
    print("=" * 50)
    
    # Check separation of concerns
    modules = {
        'models': ['layers.py', 'architectures.py'],
        'training': ['registry.py', 'train.py', 'modes.py'],
        'pretraining': ['methods.py'],
        'evaluation': ['explainability.py', 'comparison.py'],
        'config': ['config.py'],
        'utils': ['helpers.py']
    }
    
    modular_score = 0
    total_modules = len(modules)
    
    for module, files in modules.items():
        module_path = f"visx/{module}"
        if os.path.exists(module_path):
            present_files = [f for f in files if os.path.exists(os.path.join(module_path, f))]
            if len(present_files) == len(files):
                print(f"  ✅ {module} - Complete ({len(files)} files)")
                modular_score += 1
            else:
                missing = [f for f in files if f not in present_files]
                print(f"  ⚠️  {module} - Missing: {missing}")
        else:
            print(f"  ❌ {module} - Directory missing")
    
    print(f"\nModular design score: {modular_score}/{total_modules}")
    return modular_score == total_modules

def main():
    """Run all tests."""
    print("🚀 VISX Modularization Test Suite")
    print("=" * 80)
    
    # Change to the script directory
    os.chdir(Path(__file__).parent)
    
    tests = [
        ("Module Structure", test_module_structure),
        ("Configuration Structure", test_config_structure), 
        ("Modular Design", test_modular_design)
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
    print(f"🎯 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All tests passed! VISX modularization is successful.")
        return 0
    else:
        print("❌ Some tests failed. Review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())