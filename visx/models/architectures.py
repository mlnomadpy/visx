"""Model architectures for VISX."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from .layers import YatConv, YatNMN


class YatCNN(nnx.Module):
    """YAT-based Convolutional Neural Network."""
    
    def __init__(self, *, rngs: nnx.Rngs, num_classes: int = 10, input_channels: int = 3):
        self.conv1 = YatConv(
            in_features=input_channels,
            out_features=32,
            kernel_size=3,
            rngs=rngs
        )
        self.conv2 = YatConv(
            in_features=32,
            out_features=64,
            kernel_size=3,
            rngs=rngs
        )
        # Add average pooling
        self.pool = nnx.avg_pool
        
        # Dense layers with YAT
        self.dense1 = YatNMN(
            in_features=64 * 8 * 8,  # After pooling
            out_features=128,
            rngs=rngs
        )
        self.dense2 = nnx.Linear(
            in_features=128,
            out_features=num_classes,
            rngs=rngs
        )
        
    def __call__(self, x, training: bool = False):
        x = self.conv1(x)
        x = nnx.relu(x)
        x = self.pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self.conv2(x)
        x = nnx.relu(x)
        x = self.pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Flatten
        x = jnp.reshape(x, (x.shape[0], -1))
        
        x = self.dense1(x)
        x = nnx.relu(x)
        x = self.dense2(x)
        
        return x


class LinearCNN(nnx.Module):
    """Standard Linear/Convolutional Neural Network for comparison."""
    
    def __init__(self, *, rngs: nnx.Rngs, num_classes: int = 10, input_channels: int = 3):
        self.conv1 = nnx.Conv(
            in_features=input_channels,
            out_features=32,
            kernel_size=3,
            rngs=rngs
        )
        self.conv2 = nnx.Conv(
            in_features=32,
            out_features=64,
            kernel_size=3,
            rngs=rngs
        )
        # Add average pooling
        self.pool = nnx.avg_pool
        
        # Standard dense layers
        self.dense1 = nnx.Linear(
            in_features=64 * 8 * 8,  # After pooling
            out_features=128,
            rngs=rngs
        )
        self.dense2 = nnx.Linear(
            in_features=128,
            out_features=num_classes,
            rngs=rngs
        )
        
    def __call__(self, x, training: bool = False):
        x = self.conv1(x)
        x = nnx.relu(x)
        x = self.pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self.conv2(x)
        x = nnx.relu(x)
        x = self.pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Flatten
        x = jnp.reshape(x, (x.shape[0], -1))
        
        x = self.dense1(x)
        x = nnx.relu(x)
        x = self.dense2(x)
        
        return x