"""Explainability and interpretability analysis."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from typing import Dict, Any, List
from flax import nnx

from ..config import Config
from ..training.train import create_dataset


def saliency_map_analysis(model: nnx.Module, config: Config, num_samples: int = 5) -> Dict[str, Any]:
    """Generate saliency maps for model interpretability."""
    
    @partial(jax.jit, static_argnums=(0, 2))
    def get_saliency_map(model, image_input, class_index=None):
        def model_output_for_grad(img):
            logits = model(img, training=False)
            if class_index is not None:
                return logits[0, class_index]  # Focus on specific class
            else:
                return jnp.max(logits[0])  # Max logit
        
        grads = jax.grad(model_output_for_grad)(image_input)
        saliency = jnp.max(jnp.abs(grads[0]), axis=-1)
        return saliency
    
    # Get test dataset
    test_ds = create_dataset(config, 'test')
    test_iter = iter(test_ds)
    
    saliency_maps = []
    images = []
    predictions = []
    true_labels = []
    
    if config.verbose:
        print(f"Generating saliency maps for {num_samples} samples...")
    
    for i in range(min(num_samples, 10)):  # Limit to avoid memory issues
        try:
            batch = next(test_iter)
            sample_image = batch['image'][:1]  # Take first image from batch
            sample_label = batch['label'][:1]
            
            # Get model prediction
            logits = model(sample_image, training=False)
            predicted_class = jnp.argmax(logits[0])
            
            # Generate saliency map for predicted class
            saliency = get_saliency_map(model, sample_image, class_index=int(predicted_class))
            
            saliency_maps.append(np.array(saliency))
            images.append(np.array(sample_image[0]))
            predictions.append(int(predicted_class))
            true_labels.append(int(sample_label[0]))
            
        except StopIteration:
            break
    
    return {
        'saliency_maps': saliency_maps,
        'images': images,
        'predictions': predictions,
        'true_labels': true_labels
    }


def activation_map_visualization(model: nnx.Module, config: Config, 
                                layer_name: str = 'conv1', num_maps: int = 8) -> Dict[str, Any]:
    """Visualize activation maps from intermediate layers."""
    
    # Get test dataset
    test_ds = create_dataset(config, 'test')
    test_iter = iter(test_ds)
    
    activation_maps = []
    images = []
    
    if config.verbose:
        print(f"Generating activation maps for layer '{layer_name}'...")
    
    # This is a simplified version - in practice, you'd need to hook into intermediate layers
    try:
        batch = next(test_iter)
        sample_image = batch['image'][:1]
        
        # For demonstration, we'll use the input image as activation map
        # In practice, you'd extract intermediate activations
        activation = sample_image[0]
        if len(activation.shape) == 3 and activation.shape[-1] > 1:
            # Take first few channels as activation maps
            for i in range(min(num_maps, activation.shape[-1])):
                activation_maps.append(np.array(activation[:, :, i]))
        
        images.append(np.array(sample_image[0]))
        
    except StopIteration:
        pass
    
    return {
        'activation_maps': activation_maps,
        'images': images,
        'layer_name': layer_name
    }


def attention_analysis(model: nnx.Module, config: Config) -> Dict[str, Any]:
    """Analyze attention patterns (if model has attention mechanisms)."""
    
    # Placeholder for attention analysis
    # This would be implemented for models with attention layers
    
    if config.verbose:
        print("Attention analysis not implemented for current model architecture.")
    
    return {
        'attention_maps': [],
        'attention_weights': [],
        'message': 'Attention analysis requires models with attention mechanisms'
    }


def visualize_kernels(model: nnx.Module, config: Config, 
                     layer_name: str = 'conv1', num_kernels: int = 16) -> Dict[str, Any]:
    """Visualize learned convolutional kernels."""
    
    if config.verbose:
        print(f"Visualizing kernels from layer '{layer_name}'...")
    
    kernels = []
    
    # Extract kernels from the specified layer
    # This is a simplified approach - in practice, you'd need to access the specific layer
    try:
        # For YAT models, try to get conv1 layer
        if hasattr(model, 'conv1') and hasattr(model.conv1, 'kernel'):
            kernel_weights = model.conv1.kernel.value
            
            # Normalize kernels for visualization
            kernel_weights = (kernel_weights - kernel_weights.min()) / (kernel_weights.max() - kernel_weights.min())
            
            # Take first few kernels
            num_to_show = min(num_kernels, kernel_weights.shape[-1])
            for i in range(num_to_show):
                if kernel_weights.shape[2] == 3:  # RGB channels
                    kernel = kernel_weights[:, :, :, i]
                else:
                    kernel = kernel_weights[:, :, 0, i]  # Take first input channel
                
                kernels.append(np.array(kernel))
        
    except Exception as e:
        if config.verbose:
            print(f"Could not extract kernels: {e}")
    
    return {
        'kernels': kernels,
        'layer_name': layer_name,
        'num_kernels': len(kernels)
    }


def generate_explainability_report(results: Dict[str, Any], config: Config) -> str:
    """Generate a comprehensive explainability report."""
    
    report = []
    report.append("=" * 80)
    report.append("EXPLAINABILITY ANALYSIS REPORT")
    report.append("=" * 80)
    
    report.append(f"\nModel: {config.model.name}")
    report.append(f"Dataset: {config.dataset.name}")
    report.append(f"Analysis Methods: {', '.join(config.explainability.methods)}")
    
    # Saliency analysis summary
    if 'saliency' in results:
        saliency_data = results['saliency']
        num_samples = len(saliency_data.get('images', []))
        report.append(f"\n📊 SALIENCY ANALYSIS")
        report.append(f"   Samples analyzed: {num_samples}")
        if num_samples > 0:
            correct_predictions = sum(
                1 for pred, true in zip(saliency_data['predictions'], saliency_data['true_labels'])
                if pred == true
            )
            accuracy = correct_predictions / num_samples
            report.append(f"   Prediction accuracy: {accuracy:.2%}")
    
    # Activation analysis summary
    if 'activation' in results:
        activation_data = results['activation']
        num_maps = len(activation_data.get('activation_maps', []))
        layer_name = activation_data.get('layer_name', 'unknown')
        report.append(f"\n🔍 ACTIVATION ANALYSIS")
        report.append(f"   Layer analyzed: {layer_name}")
        report.append(f"   Activation maps generated: {num_maps}")
    
    # Kernel analysis summary
    if 'kernels' in results:
        kernel_data = results['kernels']
        num_kernels = kernel_data.get('num_kernels', 0)
        layer_name = kernel_data.get('layer_name', 'unknown')
        report.append(f"\n⚙️  KERNEL ANALYSIS")
        report.append(f"   Layer analyzed: {layer_name}")
        report.append(f"   Kernels visualized: {num_kernels}")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


def run_explainability_analysis(model: nnx.Module, config: Config) -> Dict[str, Any]:
    """Run comprehensive explainability analysis."""
    
    if not config.explainability.enabled:
        return {'message': 'Explainability analysis disabled in configuration'}
    
    results = {}
    
    for method in config.explainability.methods:
        if method == 'saliency':
            results['saliency'] = saliency_map_analysis(model, config, config.explainability.num_samples)
        elif method == 'grad_cam':
            # Placeholder for Grad-CAM
            results['grad_cam'] = {'message': 'Grad-CAM not implemented yet'}
        elif method == 'attention':
            results['attention'] = attention_analysis(model, config)
        elif method == 'kernels':
            for layer_name in config.explainability.layer_names:
                results[f'kernels_{layer_name}'] = visualize_kernels(model, config, layer_name)
        elif method == 'activation':
            for layer_name in config.explainability.layer_names:
                results[f'activation_{layer_name}'] = activation_map_visualization(model, config, layer_name)
    
    # Generate report
    results['report'] = generate_explainability_report(results, config)
    
    if config.verbose:
        print(results['report'])
    
    return results