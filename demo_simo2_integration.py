#!/usr/bin/env python3
"""
Demo script showing SIMO2 integration with the VISX framework.

This demonstrates how to use the newly integrated SIMO2 pretraining method
with modularized losses and multiple architectures.
"""

import jax
import jax.numpy as jnp
from flax import nnx

# Import VISX components
from visx.pretraining import get_pretraining_method, PRETRAINING_METHODS
from visx.pretraining.losses import SimoLoss, SimCLRLoss, BYOLLoss
from visx.training.registry import ModelRegistry
from visx.models import SimoModel, ResNet18LN, DenseNet121


def demo_loss_functions():
    """Demonstrate the modularized loss functions."""
    print("🔬 Demo: Modularized Loss Functions")
    print("=" * 50)
    
    # Create sample data
    batch_size = 8
    embedding_dim = 16
    key = jax.random.PRNGKey(42)
    
    # Sample embeddings and labels
    embeddings = jax.random.normal(key, (batch_size, embedding_dim))
    embeddings = embeddings / jnp.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = jnp.array([0, 0, 1, 1, 2, 2, 3, 3])
    
    # Demo SIMO Loss
    print("📊 SIMO Loss:")
    simo_loss = SimoLoss(epsilon=1e-3)
    total_loss, (intra_loss, inter_loss) = simo_loss(embeddings, labels)
    print(f"  Total Loss: {total_loss:.4f}")
    print(f"  Intra-class Loss: {intra_loss:.4f}")
    print(f"  Inter-class Loss: {inter_loss:.4f}")
    
    # Demo SimCLR Loss  
    print("\n📊 SimCLR Loss:")
    key1, key2 = jax.random.split(key)
    z1 = jax.random.normal(key1, (batch_size, embedding_dim))
    z2 = jax.random.normal(key2, (batch_size, embedding_dim))
    simclr_loss = SimCLRLoss(temperature=0.1)
    loss, metrics = simclr_loss(z1, z2)
    print(f"  Contrastive Loss: {loss:.4f}")
    
    # Demo BYOL Loss
    print("\n📊 BYOL Loss:")
    keys = jax.random.split(key, 4)
    online_pred1 = jax.random.normal(keys[0], (batch_size, embedding_dim))
    online_pred2 = jax.random.normal(keys[1], (batch_size, embedding_dim))
    target_proj1 = jax.random.normal(keys[2], (batch_size, embedding_dim))
    target_proj2 = jax.random.normal(keys[3], (batch_size, embedding_dim))
    byol_loss = BYOLLoss()
    loss, metrics = byol_loss(online_pred1, online_pred2, target_proj1, target_proj2)
    print(f"  BYOL Loss: {loss:.4f}")


def demo_model_architectures():
    """Demonstrate the SIMO2 model architectures."""
    print("\n🏗️  Demo: SIMO2 Model Architectures")
    print("=" * 50)
    
    rngs = nnx.Rngs(42)
    embedding_dim = 128
    
    # Create different backbone architectures
    print("📦 Creating backbone models:")
    resnet_backbone = ResNet18LN(rngs=rngs)
    print(f"  ResNet18LN feature dim: {resnet_backbone.feature_dim}")
    
    densenet_backbone = DenseNet121(rngs=rngs)
    print(f"  DenseNet121 feature dim: {densenet_backbone.feature_dim}")
    
    # Create SIMO models with different backbones
    print("\n🤖 Creating SIMO models:")
    simo_resnet = SimoModel(
        backbone=resnet_backbone,
        feature_dim=resnet_backbone.feature_dim,
        embedding_dim=embedding_dim,
        rngs=rngs
    )
    print(f"  SIMO ResNet18LN: {type(simo_resnet).__name__}")
    
    simo_densenet = SimoModel(
        backbone=densenet_backbone,
        feature_dim=densenet_backbone.feature_dim,
        embedding_dim=embedding_dim,
        rngs=rngs
    )
    print(f"  SIMO DenseNet121: {type(simo_densenet).__name__}")
    
    # Test forward pass
    print("\n🔄 Testing forward passes:")
    batch_size = 4
    input_shape = (batch_size, 32, 32, 3)  # NHWC format
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, input_shape)
    
    # ResNet forward pass
    embeddings_resnet = simo_resnet(x, training=True)
    features_resnet = simo_resnet(x, training=True, return_features=True)
    print(f"  ResNet embeddings shape: {embeddings_resnet.shape}")
    print(f"  ResNet features shape: {features_resnet.shape}")
    
    # DenseNet forward pass
    embeddings_densenet = simo_densenet(x, training=True)
    features_densenet = simo_densenet(x, training=True, return_features=True)
    print(f"  DenseNet embeddings shape: {embeddings_densenet.shape}")
    print(f"  DenseNet features shape: {features_densenet.shape}")


def demo_model_registry():
    """Demonstrate the model registry integration."""
    print("\n📋 Demo: Model Registry Integration")
    print("=" * 50)
    
    # Show available models
    available_models = ModelRegistry.list_models()
    print("Available models in registry:")
    for model in sorted(available_models):
        print(f"  - {model}")
    
    # Create models through registry
    print("\n🏭 Creating models through registry:")
    rngs = nnx.Rngs(42)
    
    class MockConfig:
        class dataset:
            num_classes = 10
            input_channels = 3
        class pretraining:
            embedding_dim = 128
        class model:
            architecture_params = {}
    
    config = MockConfig()
    
    # Create backbone models
    resnet = ModelRegistry.create_model('resnet18_ln', config, rngs)
    print(f"  Created ResNet18LN: {type(resnet).__name__}")
    
    densenet = ModelRegistry.create_model('densenet121', config, rngs)
    print(f"  Created DenseNet121: {type(densenet).__name__}")
    
    # Create SIMO models
    simo_resnet = ModelRegistry.create_model('simo_resnet18_ln', config, rngs)
    print(f"  Created SIMO ResNet: {type(simo_resnet).__name__}")
    
    simo_densenet = ModelRegistry.create_model('simo_densenet121', config, rngs)
    print(f"  Created SIMO DenseNet: {type(simo_densenet).__name__}")


def demo_pretraining_integration():
    """Demonstrate the pretraining method integration."""
    print("\n🚀 Demo: Pretraining Integration")
    print("=" * 50)
    
    # Show available pretraining methods
    available_methods = list(PRETRAINING_METHODS.keys())
    print("Available pretraining methods:")
    for method in sorted(available_methods):
        print(f"  - {method}")
    
    # Get SIMO2 method
    print("\n🧠 SIMO2 Pretraining Method:")
    simo2_method = get_pretraining_method('simo2')
    print(f"  Method type: {type(simo2_method).__name__}")
    
    # Create a model through SIMO2 method
    rngs = nnx.Rngs(42)
    
    class MockConfig:
        class dataset:
            num_classes = 10
            input_channels = 3
        class pretraining:
            embedding_dim = 128
            learning_rate = 3e-4
            num_steps = 100
        class model:
            name = 'densenet121'
        class training:
            rng_seed = 42
        verbose = True
    
    config = MockConfig()
    
    # Create model using SIMO2 method
    model = simo2_method.create_model(config, rngs)
    print(f"  Created model: {type(model).__name__}")
    print(f"  Has backbone: {hasattr(model, 'backbone')}")
    print(f"  Has projection head: {hasattr(model, 'projection_head')}")
    
    # Test loss function
    print("\n🔄 Testing SIMO2 loss function:")
    batch_size = 4
    input_shape = (batch_size, 32, 32, 3)
    key = jax.random.PRNGKey(42)
    view1 = jax.random.normal(key, input_shape)
    view2 = jax.random.normal(key, input_shape)
    labels = jnp.array([0, 0, 1, 1])
    
    batch = ((view1, view2), labels)
    loss, metrics = simo2_method.loss_fn(model, batch)
    print(f"  Loss: {loss:.4f}")
    print(f"  Metrics: {list(metrics.keys())}")


def main():
    """Run all demos."""
    print("🎉 SIMO2 Integration Demo")
    print("========================")
    print("This demo showcases the new SIMO2 integration with:")
    print("• Modularized loss functions")
    print("• Multiple architecture support")
    print("• Registry integration")
    print("• Pretraining method integration")
    print()
    
    try:
        demo_loss_functions()
        demo_model_architectures()
        demo_model_registry()
        demo_pretraining_integration()
        
        print("\n" + "=" * 50)
        print("🎉 All demos completed successfully!")
        print("SIMO2 is now fully integrated into the VISX framework.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()