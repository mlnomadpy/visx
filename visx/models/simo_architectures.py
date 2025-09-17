"""SIMO2 model architectures for self-supervised learning."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx


class ProjectionHead(nnx.Module):
    """Projection head for self-supervised learning."""
    
    def __init__(self, input_dim: int, embedding_dim: int, *, rngs: nnx.Rngs):
        self.projection = nnx.Linear(
            input_dim,
            embedding_dim,
            use_bias=False,
            rngs=rngs
        )

    def __call__(self, x):
        return self.projection(x)


class SimoModel(nnx.Module):
    """SIMO model with backbone and projection head."""
    
    def __init__(self, backbone: nnx.Module, feature_dim: int, embedding_dim: int, *, rngs: nnx.Rngs):
        self.backbone = backbone
        self.projection_head = ProjectionHead(feature_dim, embedding_dim, rngs=rngs)

    def __call__(self, x, training=False, return_features=False):
        features = self.backbone(x, training=training)
        if return_features:
            return features
        embeddings = self.projection_head(features)
        norm = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / jnp.maximum(norm, 1e-12)
        return embeddings


class BasicBlockLN(nnx.Module):
    """ResNet basic block with layer normalization."""
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_planes, planes, kernel_size=(3, 3), strides=(stride, stride), 
                             padding=1, use_bias=False, rngs=rngs)
        self.ln1 = nnx.LayerNorm(planes, rngs=rngs)
        self.conv2 = nnx.Conv(planes, planes, kernel_size=(3, 3), strides=(1, 1), 
                             padding=1, use_bias=False, rngs=rngs)
        self.ln2 = nnx.LayerNorm(planes, rngs=rngs)
        
        self.shortcut = nnx.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nnx.Sequential(
                nnx.Conv(in_planes, planes, kernel_size=(1, 1), strides=(stride, stride), 
                        use_bias=False, rngs=rngs),
                nnx.LayerNorm(planes, rngs=rngs)
            )

    def __call__(self, x, training: bool = False):
        out = nnx.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out += self.shortcut(x)
        out = nnx.relu(out)
        return out


class ResNetLN(nnx.Module):
    """ResNet with Layer Normalization."""
    
    def __init__(self, block, num_blocks, *, rngs: nnx.Rngs):
        self.in_planes = 64
        
        self.conv1 = nnx.Conv(3, 64, kernel_size=(3, 3), strides=(1, 1), 
                             padding=1, use_bias=False, rngs=rngs)
        self.ln1 = nnx.LayerNorm(64, rngs=rngs)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, rngs=rngs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, rngs=rngs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, rngs=rngs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, rngs=rngs)
        self.feature_dim = 512

    def _make_layer(self, block, planes, num_blocks, stride, *, rngs: nnx.Rngs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, rngs=rngs))
            self.in_planes = planes
        return layers

    def __call__(self, x, training: bool = False):
        out = nnx.relu(self.ln1(self.conv1(x)))
        for layer in self.layer1:
            out = layer(out, training=training)
        for layer in self.layer2:
            out = layer(out, training=training)
        for layer in self.layer3:
            out = layer(out, training=training)
        for layer in self.layer4:
            out = layer(out, training=training)
        out = out.mean(axis=(1, 2))  # Global Average Pooling
        return out


def ResNet18LN(rngs: nnx.Rngs):
    """ResNet-18 with Layer Normalization."""
    return ResNetLN(BasicBlockLN, [2, 2, 2, 2], rngs=rngs)


class Bottleneck(nnx.Module):
    """DenseNet bottleneck block."""
    
    def __init__(self, in_planes: int, growth_rate: int, *, rngs: nnx.Rngs):
        self.bn1 = nnx.BatchNorm(in_planes, use_running_average=True, rngs=rngs)
        self.conv1 = nnx.Conv(in_planes, 4*growth_rate, kernel_size=(1, 1), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(4*growth_rate, use_running_average=True, rngs=rngs)
        self.conv2 = nnx.Conv(4*growth_rate, growth_rate, kernel_size=(3, 3), padding=1, use_bias=False, rngs=rngs)

    def __call__(self, x, training: bool = False):
        out = self.conv1(nnx.relu(self.bn1(x, use_running_average=not training)))
        out = self.conv2(nnx.relu(self.bn2(out, use_running_average=not training)))
        out = jnp.concatenate([out, x], axis=-1)
        return out


class Transition(nnx.Module):
    """DenseNet transition block."""
    
    def __init__(self, in_planes: int, out_planes: int, *, rngs: nnx.Rngs):
        self.bn = nnx.BatchNorm(in_planes, use_running_average=True, rngs=rngs)
        self.conv = nnx.Conv(in_planes, out_planes, kernel_size=(1, 1), use_bias=False, rngs=rngs)

    def __call__(self, x, training: bool = False):
        out = self.conv(nnx.relu(self.bn(x, use_running_average=not training)))
        out = nnx.avg_pool(out, window_shape=(2, 2), strides=(2, 2))
        return out


class DenseNet(nnx.Module):
    """DenseNet architecture."""
    
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, *, rngs: nnx.Rngs):
        self.growth_rate = growth_rate
        
        num_planes = 2 * growth_rate
        self.conv1 = nnx.Conv(3, num_planes, kernel_size=(3, 3), padding=1, use_bias=False, rngs=rngs)
        
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0], rngs=rngs)
        num_planes += nblocks[0] * growth_rate
        out_planes = int(np.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes, rngs=rngs)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1], rngs=rngs)
        num_planes += nblocks[1] * growth_rate
        out_planes = int(np.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes, rngs=rngs)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2], rngs=rngs)
        num_planes += nblocks[2] * growth_rate
        out_planes = int(np.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes, rngs=rngs)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3], rngs=rngs)
        num_planes += nblocks[3] * growth_rate

        self.bn = nnx.BatchNorm(num_planes, use_running_average=True, rngs=rngs)
        self.feature_dim = num_planes

    def _make_dense_layers(self, block, in_planes, nblock, *, rngs: nnx.Rngs):
        layers = []
        for _ in range(nblock):
            layers.append(block(in_planes, self.growth_rate, rngs=rngs))
            in_planes += self.growth_rate
        return layers

    def __call__(self, x, training: bool = False):
        out = self.conv1(x)
        out = nnx.max_pool(out, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

        for layer in self.dense1: 
            out = layer(out, training=training)
        out = self.trans1(out, training=training)
        for layer in self.dense2: 
            out = layer(out, training=training)
        out = self.trans2(out, training=training)
        for layer in self.dense3: 
            out = layer(out, training=training)
        out = self.trans3(out, training=training)
        for layer in self.dense4: 
            out = layer(out, training=training)

        out = self.bn(out, use_running_average=not training)
        out = nnx.relu(out)
        out = out.mean(axis=(1, 2))  # Global Average Pooling for NHWC
        return out


def DenseNet121(rngs: nnx.Rngs):
    """DenseNet-121 architecture."""
    return DenseNet(Bottleneck, [6, 12, 24, 16], rngs=rngs)