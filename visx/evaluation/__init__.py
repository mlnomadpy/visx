"""Evaluation module exports."""

from .explainability import run_explainability_analysis, saliency_map_analysis, visualize_kernels
from .comparison import run_comparison_analysis, compare_training_curves, detailed_test_evaluation

__all__ = [
    "run_explainability_analysis",
    "saliency_map_analysis", 
    "visualize_kernels",
    "run_comparison_analysis",
    "compare_training_curves",
    "detailed_test_evaluation"
]
