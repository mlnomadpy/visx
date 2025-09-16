"""Model comparison and analysis utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List
from flax import nnx
from sklearn.metrics import confusion_matrix

from ..config import Config
from ..training.train import create_dataset, eval_step


def compare_training_curves(yat_history: Dict[str, List], linear_history: Dict[str, List]) -> Dict[str, Any]:
    """Compare training curves between YAT and Linear models."""
    
    print("\n📈 TRAINING CURVES COMPARISON")
    print("=" * 50)
    
    # Extract metrics
    yat_train_acc = yat_history.get('train_accuracy', [])
    yat_test_acc = yat_history.get('test_accuracy', [])
    linear_train_acc = linear_history.get('train_accuracy', [])
    linear_test_acc = linear_history.get('test_accuracy', [])
    
    # Print summary
    if yat_test_acc and linear_test_acc:
        yat_final = yat_test_acc[-1]
        linear_final = linear_test_acc[-1]
        
        print(f"YAT Model - Final Test Accuracy: {yat_final:.4f}")
        print(f"Linear Model - Final Test Accuracy: {linear_final:.4f}")
        print(f"Difference (YAT - Linear): {yat_final - linear_final:+.4f}")
        
        if yat_final > linear_final:
            print("🏆 YAT model performs better!")
        elif linear_final > yat_final:
            print("🏆 Linear model performs better!")
        else:
            print("🤝 Models perform equally!")
    
    return {
        'yat_final_accuracy': yat_test_acc[-1] if yat_test_acc else 0.0,
        'linear_final_accuracy': linear_test_acc[-1] if linear_test_acc else 0.0,
        'yat_train_curve': yat_train_acc,
        'yat_test_curve': yat_test_acc,
        'linear_train_curve': linear_train_acc,
        'linear_test_curve': linear_test_acc
    }


def analyze_convergence(yat_history: Dict[str, List], linear_history: Dict[str, List]) -> Dict[str, Any]:
    """Analyze convergence patterns of both models."""
    
    print("\n⚡ CONVERGENCE ANALYSIS")
    print("=" * 50)
    
    def calculate_convergence_metrics(history):
        test_acc = history.get('test_accuracy', [])
        if len(test_acc) < 2:
            return {'convergence_speed': 0, 'stability': 0, 'final_accuracy': 0}
        
        # Simple convergence metrics
        final_acc = test_acc[-1]
        
        # Calculate convergence speed (steps to reach 90% of final accuracy)
        target_acc = final_acc * 0.9
        convergence_step = len(test_acc)  # Default to end
        for i, acc in enumerate(test_acc):
            if acc >= target_acc:
                convergence_step = i
                break
        
        # Calculate stability (variance in last 20% of training)
        stable_portion = test_acc[int(len(test_acc) * 0.8):]
        stability = 1.0 / (1.0 + np.var(stable_portion)) if stable_portion else 0
        
        return {
            'convergence_speed': convergence_step,
            'stability': stability,
            'final_accuracy': final_acc
        }
    
    yat_metrics = calculate_convergence_metrics(yat_history)
    linear_metrics = calculate_convergence_metrics(linear_history)
    
    print(f"YAT Model:")
    print(f"  Convergence Speed: {yat_metrics['convergence_speed']} steps")
    print(f"  Stability Score: {yat_metrics['stability']:.4f}")
    print(f"  Final Accuracy: {yat_metrics['final_accuracy']:.4f}")
    
    print(f"\nLinear Model:")
    print(f"  Convergence Speed: {linear_metrics['convergence_speed']} steps")
    print(f"  Stability Score: {linear_metrics['stability']:.4f}")
    print(f"  Final Accuracy: {linear_metrics['final_accuracy']:.4f}")
    
    return {
        'yat_metrics': yat_metrics,
        'linear_metrics': linear_metrics
    }


def detailed_test_evaluation(yat_model: nnx.Module, linear_model: nnx.Module, 
                           config: Config) -> Dict[str, Any]:
    """Perform detailed evaluation on test set including per-class accuracy and model agreement."""
    
    print("\n🎯 DETAILED TEST EVALUATION")
    print("=" * 50)
    
    # Create test dataset
    test_ds = create_dataset(config, 'test')
    
    yat_predictions = []
    linear_predictions = []
    true_labels = []
    
    # Collect predictions
    for batch in test_ds.take(10):  # Limit to avoid memory issues
        # YAT predictions
        yat_logits = yat_model(batch['image'], training=False)
        yat_preds = jnp.argmax(yat_logits, axis=1)
        
        # Linear predictions
        linear_logits = linear_model(batch['image'], training=False)
        linear_preds = jnp.argmax(linear_logits, axis=1)
        
        yat_predictions.extend(np.array(yat_preds))
        linear_predictions.extend(np.array(linear_preds))
        true_labels.extend(np.array(batch['label']))
    
    yat_predictions = np.array(yat_predictions)
    linear_predictions = np.array(linear_predictions)
    true_labels = np.array(true_labels)
    
    # Calculate accuracies
    yat_accuracy = np.mean(yat_predictions == true_labels)
    linear_accuracy = np.mean(linear_predictions == true_labels)
    
    # Calculate agreement
    agreement = np.mean(yat_predictions == linear_predictions)
    
    # Calculate detailed metrics
    both_correct = np.mean((yat_predictions == true_labels) & (linear_predictions == true_labels))
    yat_correct_linear_wrong = np.mean((yat_predictions == true_labels) & (linear_predictions != true_labels))
    linear_correct_yat_wrong = np.mean((linear_predictions == true_labels) & (yat_predictions != true_labels))
    both_wrong = np.mean((yat_predictions != true_labels) & (linear_predictions != true_labels))
    
    print(f"YAT Model Accuracy: {yat_accuracy:.4f}")
    print(f"Linear Model Accuracy: {linear_accuracy:.4f}")
    print(f"\n🤝 MODEL AGREEMENT ANALYSIS")
    print(f"Overall Agreement: {agreement:.4f}")
    print(f"Both Correct: {both_correct:.4f}")
    print(f"YAT Correct, Linear Wrong: {yat_correct_linear_wrong:.4f}")
    print(f"Linear Correct, YAT Wrong: {linear_correct_yat_wrong:.4f}")
    print(f"Both Wrong: {both_wrong:.4f}")
    
    return {
        'yat_predictions': yat_predictions,
        'linear_predictions': linear_predictions,
        'true_labels': true_labels,
        'yat_accuracy': yat_accuracy,
        'linear_accuracy': linear_accuracy,
        'agreement': agreement,
        'both_correct': both_correct,
        'yat_correct_linear_wrong': yat_correct_linear_wrong,
        'linear_correct_yat_wrong': linear_correct_yat_wrong,
        'both_wrong': both_wrong
    }


def plot_confusion_matrices(predictions_data: Dict[str, Any]) -> Dict[str, Any]:
    """Plot confusion matrices for both models."""
    
    print("\n📊 CONFUSION MATRICES")
    print("=" * 50)
    
    yat_predictions = predictions_data['yat_predictions']
    linear_predictions = predictions_data['linear_predictions']
    true_labels = predictions_data['true_labels']
    
    # Calculate confusion matrices
    yat_cm = confusion_matrix(true_labels, yat_predictions)
    linear_cm = confusion_matrix(true_labels, linear_predictions)
    
    print("Confusion matrices generated (visualization would be displayed in GUI)")
    print(f"YAT Model - Confusion Matrix Shape: {yat_cm.shape}")
    print(f"Linear Model - Confusion Matrix Shape: {linear_cm.shape}")
    
    return {
        'yat_confusion_matrix': yat_cm,
        'linear_confusion_matrix': linear_cm
    }


def generate_summary_report(yat_history: Dict[str, List], linear_history: Dict[str, List], 
                          predictions_data: Dict[str, Any]) -> str:
    """Generate a comprehensive summary report of the comparison."""
    
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE MODEL COMPARISON REPORT")
    report.append("=" * 80)
    
    # Performance summary
    yat_final = yat_history.get('test_accuracy', [0])[-1]
    linear_final = linear_history.get('test_accuracy', [0])[-1]
    
    report.append(f"\n📊 PERFORMANCE SUMMARY")
    report.append(f"YAT Model Final Accuracy: {yat_final:.4f}")
    report.append(f"Linear Model Final Accuracy: {linear_final:.4f}")
    report.append(f"Performance Difference: {yat_final - linear_final:+.4f}")
    
    # Agreement analysis
    agreement = predictions_data.get('agreement', 0)
    report.append(f"\n🤝 MODEL AGREEMENT")
    report.append(f"Overall Agreement: {agreement:.4f}")
    
    # Recommendations
    report.append(f"\n💡 RECOMMENDATIONS")
    if yat_final > linear_final + 0.01:  # 1% threshold
        report.append("✅ YAT model architecture shows superior performance")
        report.append("✅ Consider using YAT layers for similar classification tasks")
    elif linear_final > yat_final + 0.01:
        report.append("✅ Linear model architecture is sufficient for this task")
        report.append("✅ Standard convolution layers perform well")
    else:
        report.append("✅ Both models perform similarly")
        report.append("✅ Consider computational efficiency when choosing")
    
    if agreement > 0.8:
        report.append("🤝 High model agreement suggests stable learning")
    else:
        report.append("🔍 Low model agreement suggests different learning patterns")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


def run_comparison_analysis(yat_model: nnx.Module, linear_model: nnx.Module,
                          yat_history: Dict[str, List], linear_history: Dict[str, List],
                          config: Config) -> Dict[str, Any]:
    """Run comprehensive comparison analysis."""
    
    print(f"\n📊 RUNNING COMPLETE COMPARISON ANALYSIS FOR: {config.dataset.name.upper()}")
    print("=" * 80)
    
    # 1. Compare training curves
    training_comparison = compare_training_curves(yat_history, linear_history)
    
    # 2. Analyze convergence
    convergence_analysis = analyze_convergence(yat_history, linear_history)
    
    # 3. Detailed test evaluation
    test_evaluation = detailed_test_evaluation(yat_model, linear_model, config)
    
    # 4. Plot confusion matrices
    confusion_analysis = plot_confusion_matrices(test_evaluation)
    
    # 5. Generate summary report
    summary_report = generate_summary_report(yat_history, linear_history, test_evaluation)
    
    if config.verbose:
        print(summary_report)
    
    return {
        'training_comparison': training_comparison,
        'convergence_analysis': convergence_analysis,
        'test_evaluation': test_evaluation,
        'confusion_analysis': confusion_analysis,
        'summary_report': summary_report
    }