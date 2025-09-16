"""JAX mesh and device utilities for distributed training."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from flax import nnx
from typing import Tuple, Optional


def create_mesh_for_device() -> Mesh:
    """Create appropriate mesh based on the available device backend.
    
    Returns:
        Mesh: JAX mesh configured for the current device backend.
    """
    if jax.default_backend() == 'tpu':
        # For 4-way data parallel and 2-way tensor parallel on TPU v2/v3
        mesh = Mesh(mesh_utils.create_device_mesh((4, 2)), ('batch', 'model'))
    else:
        # Fallback for GPUs or other setups
        # This will use 8-way data parallelism if 8 devices are available.
        # Adjust the mesh shape according to your hardware.
        num_devices = len(jax.devices())
        mesh_shape = (num_devices, 1)
        mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape), ('batch', 'model'))
    
    return mesh


def create_partitioned_linear(
    in_features: int,
    out_features: int,
    mesh: Mesh,
    rngs: nnx.rnglib.Rngs,
    *,
    use_bias: bool = True,
    dtype: Optional[jnp.dtype] = None,
    param_dtype: jnp.dtype = jnp.float32,
    kernel_init: nnx.initializers.Initializer = nnx.initializers.xavier_uniform(),
    bias_init: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
) -> nnx.Linear:
    """Create a Linear layer with partitioned weights for distributed training.
    
    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        mesh: JAX mesh for partitioning.
        rngs: Random number generators.
        use_bias: Whether to include bias term.
        dtype: Computation dtype.
        param_dtype: Parameter dtype.
        kernel_init: Kernel initializer.
        bias_init: Bias initializer.
        
    Returns:
        nnx.Linear: Linear layer with partitioned weights.
    """
    return nnx.Linear(
        in_features=in_features,
        out_features=out_features,
        use_bias=use_bias,
        dtype=dtype,
        param_dtype=param_dtype,
        kernel_init=nnx.with_partitioning(
            kernel_init, 
            NamedSharding(mesh, P(None, 'model'))
        ),
        bias_init=nnx.with_partitioning(
            bias_init, 
            NamedSharding(mesh, P('model'))
        ) if use_bias else bias_init,
        rngs=rngs
    )


def get_device_info() -> dict:
    """Get information about available devices and backend.
    
    Returns:
        dict: Device information including backend, device count, and device list.
    """
    return {
        'backend': jax.default_backend(),
        'device_count': len(jax.devices()),
        'devices': jax.devices(),
        'local_device_count': jax.local_device_count(),
        'process_count': jax.process_count(),
        'process_index': jax.process_index(),
    }


def print_device_info():
    """Print detailed device information."""
    info = get_device_info()
    print("🔧 JAX Device Information")
    print("=" * 40)
    print(f"Backend: {info['backend']}")
    print(f"Total devices: {info['device_count']}")
    print(f"Local devices: {info['local_device_count']}")
    print(f"Process count: {info['process_count']}")
    print(f"Process index: {info['process_index']}")
    print(f"Devices: {info['devices']}")
    print("=" * 40)


def setup_distributed_training() -> Tuple[Mesh, dict]:
    """Setup distributed training with appropriate mesh and device info.
    
    Returns:
        Tuple[Mesh, dict]: The mesh and device information.
    """
    device_info = get_device_info()
    mesh = create_mesh_for_device()
    
    print_device_info()
    print(f"\n🌐 Mesh Configuration")
    print(f"Mesh shape: {mesh.shape}")
    print(f"Mesh axis names: {mesh.axis_names}")
    
    return mesh, device_info