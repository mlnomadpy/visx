"""Tests for SIMO2 integration and modularization."""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

from visx.pretraining.losses import SimoLoss, SimCLRLoss, BYOLLoss
from visx.pretraining.methods import SIMO2Pretraining, PRETRAINING_METHODS
from visx.models import ResNet18LN, DenseNet121, SimoModel
from visx.training.registry import ModelRegistry


def test_simo_loss():
    """Test SIMO loss function."""
    # Create test data
    batch_size = 4
    embedding_dim = 8
    
    key = jax.random.PRNGKey(42)
    embeddings = jax.random.normal(key, (batch_size, embedding_dim))
    # Normalize embeddings
    embeddings = embeddings / jnp.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = jnp.array([0, 0, 1, 1])
    
    # Test SIMO loss
    simo_loss = SimoLoss(epsilon=1e-3)
    total_loss, (intra_loss, inter_loss) = simo_loss(embeddings, labels)
    
    assert isinstance(total_loss, jnp.ndarray)
    assert total_loss.shape == ()
    assert isinstance(intra_loss, jnp.ndarray) 
    assert isinstance(inter_loss, jnp.ndarray)
    assert total_loss >= 0  # Loss should be non-negative


def test_simclr_loss():
    """Test SimCLR loss function."""
    batch_size = 4
    projection_dim = 8
    
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    z1 = jax.random.normal(key1, (batch_size, projection_dim))
    z2 = jax.random.normal(key2, (batch_size, projection_dim))
    
    simclr_loss = SimCLRLoss(temperature=0.1)
    loss, metrics = simclr_loss(z1, z2)
    
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()
    assert "loss" in metrics
    assert loss >= 0


def test_byol_loss():
    """Test BYOL loss function."""
    batch_size = 4
    projection_dim = 8
    
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)
    online_pred1 = jax.random.normal(keys[0], (batch_size, projection_dim))
    online_pred2 = jax.random.normal(keys[1], (batch_size, projection_dim))
    target_proj1 = jax.random.normal(keys[2], (batch_size, projection_dim))
    target_proj2 = jax.random.normal(keys[3], (batch_size, projection_dim))
    
    byol_loss = BYOLLoss()
    loss, metrics = byol_loss(online_pred1, online_pred2, target_proj1, target_proj2)
    
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()
    assert "loss" in metrics


def test_simo_model_creation():
    """Test SIMO model creation with different backbones."""
    rngs = nnx.Rngs(42)
    embedding_dim = 128
    
    # Test with ResNet18LN
    backbone_resnet = ResNet18LN(rngs=rngs)
    simo_resnet = SimoModel(
        backbone=backbone_resnet,
        feature_dim=backbone_resnet.feature_dim,
        embedding_dim=embedding_dim,
        rngs=rngs
    )
    
    # Test with DenseNet121
    backbone_densenet = DenseNet121(rngs=rngs)
    simo_densenet = SimoModel(
        backbone=backbone_densenet,
        feature_dim=backbone_densenet.feature_dim,
        embedding_dim=embedding_dim,
        rngs=rngs
    )
    
    # Test forward pass
    batch_size = 2
    input_shape = (batch_size, 32, 32, 3)  # NHWC format
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, input_shape)
    
    # Test ResNet model
    embeddings_resnet = simo_resnet(x, training=True)
    assert embeddings_resnet.shape == (batch_size, embedding_dim)
    
    # Test DenseNet model
    embeddings_densenet = simo_densenet(x, training=True)
    assert embeddings_densenet.shape == (batch_size, embedding_dim)
    
    # Test feature extraction
    features_resnet = simo_resnet(x, training=True, return_features=True)
    assert features_resnet.shape == (batch_size, backbone_resnet.feature_dim)


def test_simo2_pretraining_method():
    """Test SIMO2 pretraining method registration and basic functionality."""
    assert 'simo2' in PRETRAINING_METHODS
    
    simo2_method = PRETRAINING_METHODS['simo2']
    assert isinstance(simo2_method, SIMO2Pretraining)
    
    # Test loss function with batch format
    rngs = nnx.Rngs(42)
    backbone = ResNet18LN(rngs=rngs)
    model = SimoModel(
        backbone=backbone,
        feature_dim=backbone.feature_dim,
        embedding_dim=128,
        rngs=rngs
    )
    
    # Test with tuple batch format (views and labels)
    batch_size = 4
    input_shape = (batch_size, 32, 32, 3)
    key = jax.random.PRNGKey(42)
    view1 = jax.random.normal(key, input_shape)
    view2 = jax.random.normal(key, input_shape)
    labels = jnp.array([0, 0, 1, 1])
    
    batch = ((view1, view2), labels)
    loss, metrics = simo2_method.loss_fn(model, batch)
    
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()
    assert "loss" in metrics
    assert "intra_loss" in metrics
    assert "inter_loss" in metrics


def test_model_registry_integration():
    """Test that SIMO2 models are properly registered."""
    available_models = ModelRegistry.list_models()
    
    # Check that SIMO2 models are registered
    assert 'resnet18_ln' in available_models
    assert 'densenet121' in available_models
    assert 'simo_resnet18_ln' in available_models
    assert 'simo_densenet121' in available_models
    
    # Test model creation through registry
    rngs = nnx.Rngs(42)
    
    # Create a minimal config-like object
    class MockConfig:
        class dataset:
            num_classes = 10
            input_channels = 3
        class pretraining:
            embedding_dim = 128
        class model:
            architecture_params = {}
    
    config = MockConfig()
    
    # Test ResNet18LN creation
    resnet_model = ModelRegistry.create_model('resnet18_ln', config, rngs)
    assert hasattr(resnet_model, 'feature_dim')
    
    # Test DenseNet121 creation
    densenet_model = ModelRegistry.create_model('densenet121', config, rngs)
    assert hasattr(densenet_model, 'feature_dim')
    
    # Test SIMO models creation
    simo_resnet = ModelRegistry.create_model('simo_resnet18_ln', config, rngs)
    assert isinstance(simo_resnet, SimoModel)
    
    simo_densenet = ModelRegistry.create_model('simo_densenet121', config, rngs)
    assert isinstance(simo_densenet, SimoModel)


def test_backward_compatibility():
    """Test that existing functionality still works."""
    # Test that existing pretraining methods are still available
    required_methods = ['supervised', 'byol', 'simclr', 'self_supervised']
    for method in required_methods:
        assert method in PRETRAINING_METHODS
    
    # Test that existing models are still available
    required_models = ['yat_cnn', 'linear_cnn', 'yat', 'linear']
    available_models = ModelRegistry.list_models()
    for model in required_models:
        assert model in available_models


if __name__ == '__main__':
    pytest.main([__file__])