#!/usr/bin/env python3
"""
GCP Training Script for Checkers AI
Creates multiple difficulty models at different training stages.

Usage:
    python gcp_train.py [--resume]
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from datetime import datetime, timedelta

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Difficulty checkpoints - when to save each difficulty model
DIFFICULTY_CHECKPOINTS = {
    'easy': 500,        # 500 games - beginner AI
    'medium': 2500,     # 2,500 games - casual player level
    'hard': 10000,      # 10,000 games - strong club player
    'impossible': 25000 # 25,000 games - very strong AI (~70h training)
}

# Training configuration - MAXIMUM QUALITY
CONFIG = {
    'games_per_iteration': 64,     # Games per training cycle
    'batches_per_iteration': 200,  # Training batches
    'mcts_simulations': 100,       # Full MCTS simulations (no compromise!)
    'batch_size': 256,
    'learning_rate': 0.001,
    'total_games': 25000,          # Target: 25k games (~70h, ~$50)
    'checkpoint_every': 500,       # Save checkpoint every N games
    'num_parallel_games': 32,      # Games to run in parallel
}


def save_difficulty_model_fast(network, difficulty, output_dir):
    """Save a model for a specific difficulty level."""
    diff_dir = os.path.join(output_dir, difficulty)
    os.makedirs(diff_dir, exist_ok=True)
    
    # Save checkpoint
    checkpoint_path = os.path.join(diff_dir, 'model.weights.h5')
    network.save(checkpoint_path)
    
    # Export to TensorFlow.js
    try:
        import tensorflowjs as tfjs
        tfjs_dir = os.path.join(diff_dir, 'tfjs')
        os.makedirs(tfjs_dir, exist_ok=True)
        tfjs.converters.save_keras_model(network.model, tfjs_dir)
        print(f"  ✅ Exported {difficulty} model to TF.js")
    except Exception as e:
        print(f"  ⚠️ Could not export to TF.js: {e}")
    
    return checkpoint_path


def update_status(status_file, status):
    """Write status to JSON file."""
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)


def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    if seconds < 0:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def main():
    parser = argparse.ArgumentParser(description='Train Checkers AI on GCP (Multiprocessing)')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--output-dir', default='trained_models', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 GCP Checkers AI Training (Multiprocessing)")
    print("  - Using 3 dedicated CPU workers for MCTS")
    print("  - Using 1 GPU process for Neural Network")
    print("  - Target: ~3-4x speedup")
    print("=" * 70)
    print()
    
    # Update config with checkpoints
    CONFIG['difficulty_checkpoints'] = DIFFICULTY_CHECKPOINTS
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Import multiprocessing training
    from mp_train import run_mp_training
    
    # Resume path
    resume_path = None
    if args.resume:
        path = os.path.join('checkpoints', 'latest.weights.h5')
        if os.path.exists(path):
            resume_path = path
            print(f"Resuming from {path}")
    
    try:
        run_mp_training(CONFIG, resume_path=resume_path)
    except KeyboardInterrupt:
        print("\nStopping training...")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
