#!/usr/bin/env python3
"""
Minimal test script for new VISX features without heavy dependencies.
"""

import ast
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


def test_new_modules_syntax():
    """Test that new modules have valid syntax."""
    print("🔍 Testing New Module Syntax")
    print("=" * 40)
    
    new_modules = [
        "visx/utils/mesh.py",
        "visx/data/__init__.py", 
        "visx/data/configs.py",
        "visx/data/loaders.py",
        "visx/data/streaming.py"
    ]
    
    all_valid = True
    
    for module in new_modules:
        if Path(module).exists():
            valid, error = check_syntax(module)
            if valid:
                print(f"✅ {module}")
            else:
                print(f"❌ {module}: {error}")
                all_valid = False
        else:
            print(f"❌ {module}: File not found")
            all_valid = False
    
    return all_valid


def test_data_directory_structure():
    """Test that data directory structure is correct."""
    print("\n📁 Testing Data Directory Structure")
    print("=" * 40)
    
    expected_files = [
        "visx/data/__init__.py",
        "visx/data/configs.py", 
        "visx/data/loaders.py",
        "visx/data/streaming.py"
    ]
    
    all_present = True
    
    for filepath in expected_files:
        if Path(filepath).exists():
            print(f"✅ {filepath}")
        else:
            print(f"❌ {filepath}")
            all_present = False
    
    return all_present


def test_test_directory_structure():
    """Test that test directory structure is correct."""
    print("\n🧪 Testing Test Directory Structure")
    print("=" * 40)
    
    expected_files = [
        "test/__init__.py",
        "test/test_modularization.py",
        "test/test_new_features.py"
    ]
    
    all_present = True
    
    for filepath in expected_files:
        if Path(filepath).exists():
            print(f"✅ {filepath}")
        else:
            print(f"❌ {filepath}")
            all_present = False
    
    return all_present


def test_main_py_changes():
    """Test that main.py has been properly modified."""
    print("\n📝 Testing main.py Modifications")
    print("=" * 40)
    
    try:
        with open("main.py", "r") as f:
            content = f.read()
        
        # Check that legacy execution line was removed (not in print statements)
        import re
        lines = content.split('\n')
        actual_execution = [line for line in lines 
                          if line.strip().startswith('results_stl10 = run_complete_comparison')
                          and not line.strip().startswith('print(')]
        
        if actual_execution:
            print("❌ Legacy execution code still present")
            return False
        else:
            print("✅ Legacy execution code removed")
        
        # Check that data imports are present
        if "from visx.data.configs import" in content:
            print("✅ Data module imports added")
        else:
            print("❌ Data module imports missing")
            return False
        
        # Check basic syntax
        try:
            ast.parse(content)
            print("✅ main.py has valid syntax")
        except SyntaxError as e:
            print(f"❌ main.py syntax error: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking main.py: {e}")
        return False


def test_requirements_updated():
    """Test that requirements.txt includes new dependencies."""
    print("\n📦 Testing Requirements Updates")
    print("=" * 40)
    
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()
        
        if "datasets>=" in content:
            print("✅ HuggingFace datasets dependency added")
            return True
        else:
            print("❌ HuggingFace datasets dependency missing")
            return False
            
    except Exception as e:
        print(f"❌ Error checking requirements.txt: {e}")
        return False


def main():
    """Run all minimal tests."""
    print("🚀 VISX Minimal Feature Test Suite")
    print("=" * 80)
    
    # Change to the repository root directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    import os
    os.chdir(repo_root)
    
    tests = [
        ("New Module Syntax", test_new_modules_syntax),
        ("Data Directory Structure", test_data_directory_structure),
        ("Test Directory Structure", test_test_directory_structure),
        ("Main.py Modifications", test_main_py_changes),
        ("Requirements Updates", test_requirements_updated),
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
    print(f"🎯 Minimal Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All minimal tests passed! Changes are structurally sound.")
        return 0
    else:
        print("❌ Some minimal tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())