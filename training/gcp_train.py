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
    'easy': 500,       # 500 games
    'medium': 2000,    # 2,000 games
    'hard': 5000,      # 5,000 games
    'impossible': 10000 # 10,000 games (strong AI ~12-24h with GPU parallelization)
}

# Training configuration - GPU OPTIMIZED with parallel self-play
CONFIG = {
    'games_per_iteration': 64,     # Games per training cycle
    'batches_per_iteration': 200,  # Training batches
    'mcts_simulations': 100,       # Full MCTS sims per move (no compromise!)
    'batch_size': 256,
    'learning_rate': 0.001,
    'total_games': 10000,          # Target total games (~12-24h with parallel)
    'checkpoint_every': 500,       # Save checkpoint every N games
    'num_parallel_games': 32,      # Games to run in parallel on GPU
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
    parser = argparse.ArgumentParser(description='Train Checkers AI on GCP')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--output-dir', default='trained_models', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 GCP Checkers AI Training")
    print("=" * 70)
    print()
    
    # Load TensorFlow
    print("Loading TensorFlow...")
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🎮 GPU available: {gpus[0].name}")
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️ No GPU found, using CPU (will be slower)")
    
    from trainer import TrainingExample, ReplayBuffer
    from model import CheckersNetwork
    from true_parallel import TrueParallelMCTS, generate_training_data
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Status file for monitoring
    status_file = os.path.join('checkpoints', 'gcp_training_status.json')
    
    # Initialize network and TRUE parallel MCTS
    print("\n🚀 Initializing network and TRUE GPU-parallel MCTS...")
    print(f"   Running {CONFIG.get('num_parallel_games', 32)} games simultaneously")
    print(f"   Full {CONFIG['mcts_simulations']} MCTS simulations per move")
    network = CheckersNetwork()
    
    # Warm up GPU with compiled function
    print("   Warming up GPU...")
    dummy = np.random.randn(32, 8, 8, 4).astype(np.float32)
    _ = network.predict_batch(dummy)
    print("   GPU ready!")
    
    # Initialize replay buffer for training
    buffer = ReplayBuffer(200000)
    
    # Resume if requested
    start_games = 0
    if args.resume:
        try:
            checkpoint_path = os.path.join('checkpoints', 'latest.weights.h5')
            if os.path.exists(checkpoint_path):
                network.load(checkpoint_path)
                print(f"   Loaded weights from {checkpoint_path}")
            # Try to get game count from status
            if os.path.exists(status_file):
                with open(status_file) as f:
                    status = json.load(f)
                    start_games = status.get('total_games', 0)
            print(f"Resumed from checkpoint (estimated {start_games} games)")
        except Exception as e:
            print(f"Could not resume: {e}")
            start_games = 0
    
    # Print training plan
    print()
    print("📋 Training Plan:")
    print("-" * 50)
    for diff, games in sorted(DIFFICULTY_CHECKPOINTS.items(), key=lambda x: x[1]):
        status = "✅" if start_games >= games else "⏳"
        print(f"  {status} {diff:12s}: {games:,} games")
    print("-" * 50)
    print(f"  Total target: {CONFIG['total_games']:,} games")
    print()
    
    # Training loop
    start_time = time.time()
    total_games = start_games
    iteration = 0
    games_per_iter = CONFIG['games_per_iteration']
    saved_difficulties = set()
    
    # Mark already-saved difficulties
    for diff, threshold in DIFFICULTY_CHECKPOINTS.items():
        if start_games >= threshold:
            saved_difficulties.add(diff)
    
    print("🎯 Starting training...")
    print()
    
    while total_games < CONFIG['total_games']:
        iteration += 1
        iter_start = time.time()
        
        # Status header
        elapsed = time.time() - start_time
        games_remaining = CONFIG['total_games'] - total_games
        games_per_second = (total_games - start_games) / max(elapsed, 1)
        eta_seconds = games_remaining / max(games_per_second, 0.001)
        
        print(f"{'='*70}")
        print(f"📊 Iteration {iteration}")
        print(f"   Games: {total_games:,} / {CONFIG['total_games']:,} ({100*total_games/CONFIG['total_games']:.1f}%)")
        print(f"   Elapsed: {format_time(elapsed)} | ETA: {format_time(eta_seconds)}")
        print(f"{'='*70}")
        
        # Generate self-play games with TRUE GPU-parallel MCTS
        num_parallel = CONFIG.get('num_parallel_games', 32)
        print(f"\n⏳ Generating {games_per_iter} games ({num_parallel} parallel, {CONFIG['mcts_simulations']} MCTS sims)...")
        
        import time as time_mod
        gen_start = time_mod.time()
        
        game_examples = generate_training_data(
            network=network,
            num_games=games_per_iter,
            mcts_sims=CONFIG['mcts_simulations'],
            num_parallel=num_parallel
        )
        
        total_games += games_per_iter
        gen_elapsed = time_mod.time() - gen_start
        
        # Add examples to buffer
        for state, policy, value in game_examples:
            buffer.add([TrainingExample(state, policy, value)])
        
        print(f"   ✅ {len(game_examples)} examples in {gen_elapsed:.1f}s ({games_per_iter/gen_elapsed:.2f} games/sec)")
        
        # Train on batch
        metrics = {'loss': 0, 'policy_loss': 0, 'value_loss': 0}
        if len(buffer) >= CONFIG['batch_size']:
            print(f"\n🧠 Training for {CONFIG['batches_per_iteration']} batches...")
            
            # Manual training loop since we're not using Trainer class
            losses = []
            for _ in range(CONFIG['batches_per_iteration']):
                states, policies, values = buffer.sample(CONFIG['batch_size'])
                loss = network.model.train_on_batch(
                    states, 
                    {'policy_output': policies, 'value_output': values}
                )
                losses.append(loss)
            
            avg_losses = np.mean(losses, axis=0)
            metrics = {
                'loss': float(avg_losses[0]),
                'policy_loss': float(avg_losses[1]),
                'value_loss': float(avg_losses[2])
            }
            print(f"   Loss: {metrics['loss']:.4f}")
            print(f"   Policy: {metrics['policy_loss']:.4f} | Value: {metrics['value_loss']:.4f}")
        
        # Save regular checkpoint
        checkpoint_path = os.path.join('checkpoints', 'latest.weights.h5')
        network.save(checkpoint_path)
        print(f"   Saved checkpoint: {checkpoint_path}")
        
        # Check for difficulty thresholds
        for diff, threshold in DIFFICULTY_CHECKPOINTS.items():
            if total_games >= threshold and diff not in saved_difficulties:
                print(f"\n🎉 MILESTONE: Saving {diff.upper()} model ({threshold:,} games)")
                save_difficulty_model_fast(network, diff, args.output_dir)
                saved_difficulties.add(diff)
        
        # Iteration time
        iter_time = time.time() - iter_start
        
        # Update status file
        status = {
            'total_games': total_games,
            'target_games': CONFIG['total_games'],
            'iteration': iteration,
            'elapsed_seconds': elapsed,
            'eta_seconds': eta_seconds,
            'games_per_second': games_per_second,
            'saved_difficulties': list(saved_difficulties),
            'last_update': datetime.now().isoformat(),
            'metrics': {
                'loss': float(metrics.get('loss', 0)),
                'policy_loss': float(metrics.get('policy_loss', 0)),
                'value_loss': float(metrics.get('value_loss', 0)),
                'buffer_size': len(trainer.buffer)
            }
        }
        update_status(status_file, status)
        
        print(f"\n⏱️ Iteration time: {format_time(iter_time)}")
        print()
    
    # Final save
    print("\n" + "=" * 70)
    print("🏆 Training Complete!")
    print("=" * 70)
    
    # Make sure all difficulties are saved
    for diff in DIFFICULTY_CHECKPOINTS:
        if diff not in saved_difficulties:
            print(f"Saving {diff} model...")
            save_difficulty_model(trainer, diff, args.output_dir)
    
    # Final status
    total_time = time.time() - start_time
    print(f"\n📊 Final Statistics:")
    print(f"   Total games: {total_games:,}")
    print(f"   Total time: {format_time(total_time)}")
    print(f"   Average: {total_games/total_time:.2f} games/second")
    print(f"\n📁 Models saved to: {args.output_dir}/")
    for diff in ['easy', 'medium', 'hard', 'impossible']:
        print(f"   - {diff}/tfjs/model.json")
    
    # Mark complete
    status['completed'] = True
    status['total_time'] = total_time
    update_status(status_file, status)
    
    print("\n✅ All done! Download the trained_models folder.")


if __name__ == "__main__":
    main()
