#!/usr/bin/env python3
"""
VISX: Vision eXploration with YAT architectures

Main command-line interface for training, pretraining, explainability analysis,
and model comparison.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Add visx to path
sys.path.insert(0, str(Path(__file__).parent))

from visx.config import create_argument_parser, parse_config, Config
from visx.training import run_training_mode
from visx.utils import log_experiment_info, validate_config, ensure_output_dir


def main():
    """Main entry point for VISX CLI."""
    
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Parse configuration
    config = parse_config(args)
    
    # Validate configuration
    if not validate_config(config):
        sys.exit(1)
    
    # Setup output directory
    output_dir = ensure_output_dir(config.output_dir)
    
    # Log experiment info
    log_experiment_info(config, str(output_dir))
    
    # Print startup banner
    print("=" * 80)
    print("🚀 VISX: Vision eXploration with YAT Architectures")
    print("=" * 80)
    print(f"Mode: {config.mode}")
    print(f"Model: {config.model.name} ({config.model.type})")
    print(f"Dataset: {config.dataset.name}")
    print(f"Output: {output_dir}")
    
    if config.mode == "pretraining":
        print(f"Pretraining Method: {config.pretraining.method}")
    
    if config.mode == "explainability":
        print(f"Explainability Methods: {', '.join(config.explainability.methods)}")
    
    print("=" * 80)
    
    try:
        # Run the specified mode
        model, results = run_training_mode(config)
        
        # Save results
        if config.save_checkpoints:
            from visx.utils import save_model_checkpoint, save_training_history
            
            model_path = output_dir / "model.pkl"
            save_model_checkpoint(model, str(model_path))
            print(f"💾 Model saved to {model_path}")
            
            if isinstance(results, dict) and 'training_history' in results:
                history_path = output_dir / "training_history.json"
                save_training_history(results['training_history'], str(history_path))
                print(f"📊 Training history saved to {history_path}")
            elif isinstance(results, dict) and any(key.endswith('_history') for key in results.keys()):
                # Save all history files
                for key, value in results.items():
                    if key.endswith('_history') and isinstance(value, dict):
                        history_path = output_dir / f"{key}.json"
                        save_training_history(value, str(history_path))
                        print(f"📊 {key} saved to {history_path}")
        
        print("\n✅ VISX execution completed successfully!")
        
        # Print final summary based on mode
        if config.mode == "training":
            if isinstance(results, dict) and 'test_accuracy' in results:
                final_acc = results['test_accuracy'][-1] if results['test_accuracy'] else 0.0
                print(f"🎯 Final Test Accuracy: {final_acc:.4f}")
        
        elif config.mode == "comparison":
            if isinstance(results, dict):
                yat_history = results.get('yat_history', {})
                linear_history = results.get('linear_history', {})
                
                yat_acc = yat_history.get('test_accuracy', [0])[-1]
                linear_acc = linear_history.get('test_accuracy', [0])[-1]
                
                print(f"🎯 YAT Model Accuracy: {yat_acc:.4f}")
                print(f"🎯 Linear Model Accuracy: {linear_acc:.4f}")
                print(f"📊 Difference: {yat_acc - linear_acc:+.4f}")
        
        elif config.mode == "explainability":
            print("🔬 Explainability analysis completed!")
            if isinstance(results, dict) and 'explainability' in results:
                explainability_results = results['explainability']
                if 'report' in explainability_results:
                    print("📋 Analysis report available in results")
        
        elif config.mode == "pretraining":
            print(f"🏗️  Pretraining with {config.pretraining.method} completed!")
            if isinstance(results, dict) and 'loss' in results:
                final_loss = results['loss'][-1] if results['loss'] else 0.0
                print(f"📉 Final Loss: {final_loss:.4f}")
    
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()