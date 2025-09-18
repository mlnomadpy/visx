#!/usr/bin/env python3
"""
Demonstration script showing the new mesh configuration functionality.
This script shows the API without requiring JAX installation.
"""

import sys
from pathlib import Path

# Add the repo root to Python path
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from visx.config.config import MeshConfig, Config
import tempfile
import yaml

print("🚀 VISX Mesh Configuration Demonstration")
print("=" * 60)

print("\n1️⃣ Basic Mesh Configuration")
print("-" * 40)

# Create a basic mesh config
mesh_config = MeshConfig()
print(f"Default mesh config: enabled={mesh_config.enabled}, auto_detect={mesh_config.auto_detect}")

# Create a custom mesh config
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
print(f"Custom mesh config: shape={custom_mesh.shape}, axis_names={custom_mesh.axis_names}")

print("\n2️⃣ Configuration via YAML")
print("-" * 40)

# Create a complete config with mesh settings
config = Config()
config.mesh.enabled = True
config.mesh.auto_detect = False
config.mesh.shape = [4, 2]
config.mesh.axis_names = ['batch', 'model']

# Save to temporary YAML file
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    config.to_yaml(f.name)
    temp_yaml = f.name

# Load it back
loaded_config = Config.from_yaml(temp_yaml)
print(f"Loaded from YAML: mesh shape={loaded_config.mesh.shape}")

# Show the YAML content
with open(temp_yaml, 'r') as f:
    yaml_content = f.read()
    
print("Generated YAML mesh section:")
lines = yaml_content.split('\n')
in_mesh_section = False
for line in lines:
    if line.startswith('mesh:'):
        in_mesh_section = True
    elif line and not line.startswith('  ') and in_mesh_section:
        break
    if in_mesh_section:
        print(f"  {line}")

print("\n3️⃣ Different Configuration Scenarios")
print("-" * 40)

scenarios = [
    ("Auto-detection (default)", MeshConfig(enabled=True, auto_detect=True)),
    ("Disabled mesh", MeshConfig(enabled=False)),
    ("4x2 TPU topology", MeshConfig(enabled=True, auto_detect=False, shape=[4, 2], axis_names=['batch', 'model'])),
    ("8x1 GPU topology", MeshConfig(enabled=True, auto_detect=False, shape=[8, 1], axis_names=['batch', 'model'])),
]

for name, mesh_cfg in scenarios:
    print(f"• {name}: enabled={mesh_cfg.enabled}, auto_detect={mesh_cfg.auto_detect}, shape={mesh_cfg.shape}")

print("\n4️⃣ Legacy Integration")
print("-" * 40)

# Show how simo2.py arguments would map to config
simo2_args = {
    'mesh_enabled': True,
    'mesh_auto_detect': False,
    'mesh_shape': [4, 2],
    'mesh_axis_names': ['batch', 'model']
}

print("Command line arguments for simo2.py:")
print("python simo2.py --mesh_enabled true --mesh_auto_detect false --mesh_shape 4 2 --mesh_axis_names batch model")

# Show how this maps to MeshConfig
legacy_mesh = MeshConfig()
if 'mesh_enabled' in simo2_args:
    legacy_mesh.enabled = simo2_args['mesh_enabled']
if 'mesh_auto_detect' in simo2_args:
    legacy_mesh.auto_detect = simo2_args['mesh_auto_detect']
if 'mesh_shape' in simo2_args:
    legacy_mesh.shape = simo2_args['mesh_shape']
if 'mesh_axis_names' in simo2_args:
    legacy_mesh.axis_names = simo2_args['mesh_axis_names']

print(f"Resulting MeshConfig: {legacy_mesh.shape} with axes {legacy_mesh.axis_names}")

print("\n5️⃣ Example YAML Configurations")
print("-" * 40)

yaml_examples = {
    "Automatic detection": {
        'mesh': {
            'enabled': True,
            'auto_detect': True
        }
    },
    "Custom 4x2 mesh": {
        'mesh': {
            'enabled': True,
            'auto_detect': False,
            'shape': [4, 2],
            'axis_names': ['batch', 'model']
        }
    },
    "Hardware-specific": {
        'mesh': {
            'enabled': True,
            'auto_detect': False,
            'tpu_mesh_shape': [4, 2],
            'tpu_axis_names': ['batch', 'model'],
            'gpu_mesh_shape': [8, 1],
            'gpu_axis_names': ['batch', 'model']
        }
    }
}

for example_name, example_config in yaml_examples.items():
    print(f"\n{example_name}:")
    print(yaml.dump(example_config, default_flow_style=False, indent=2))

print("✅ Key Features Implemented:")
print("  • MeshConfig dataclass with all mesh topology settings")
print("  • Integration with existing Config system")
print("  • YAML configuration support")
print("  • Command line argument support in simo2.py")
print("  • Backward compatibility with legacy mesh setup")
print("  • Hardware-specific configuration (TPU vs GPU)")
print("  • Automatic detection with manual override capability")

print("\n🎯 Usage Instructions:")
print("  1. Configure via YAML: Set mesh section in config files")
print("  2. Configure via CLI: Use --mesh_* arguments with simo2.py")
print("  3. Use in code: Import and use visx.utils.mesh functions")
print("  4. Legacy support: Existing code continues to work")

# Clean up
import os
os.unlink(temp_yaml)

print(f"\n{'=' * 60}")
print("✅ Mesh topology configuration is now fully configurable!")