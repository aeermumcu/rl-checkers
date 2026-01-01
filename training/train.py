#!/usr/bin/env python3
"""
Main training script for Checkers AI.
Run this script to train the AI using self-play.
"""

import argparse
import os
import sys

# Ensure we can import from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='Train Checkers AI using AlphaZero-style self-play')
    
    # Training parameters
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of training iterations (default: 50)')
    parser.add_argument('--games', type=int, default=100,
                        help='Self-play games per iteration (default: 100)')
    parser.add_argument('--batches', type=int, default=200,
                        help='Training batches per iteration (default: 200)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size (default: 256)')
    parser.add_argument('--mcts-sims', type=int, default=100,
                        help='MCTS simulations per move (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    
    # Directories
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory for checkpoints (default: checkpoints)')
    parser.add_argument('--export-dir', type=str, default='../model',
                        help='Directory for TF.js export (default: ../model)')
    
    # Options
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint name (e.g., "latest")')
    parser.add_argument('--test-mode', action='store_true',
                        help='Quick test mode with minimal training')
    parser.add_argument('--export-only', action='store_true',
                        help='Only export existing model to TF.js')
    
    args = parser.parse_args()
    
    # Test mode: minimal parameters
    if args.test_mode:
        args.iterations = 1
        args.games = 2
        args.batches = 5
        args.mcts_sims = 20
        print("Running in TEST MODE with minimal parameters")
    
    # Import TensorFlow (late import for faster --help)
    print("Loading TensorFlow...")
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU available: {gpus}")
        else:
            print("No GPU found, using CPU")
        
    except ImportError:
        print("ERROR: TensorFlow not installed!")
        print("Install with: pip install tensorflow")
        sys.exit(1)
    
    from trainer import Trainer, run_training
    from model import CheckersNetwork
    
    # Export only mode
    if args.export_only:
        print(f"Exporting model to {args.export_dir}...")
        network = CheckersNetwork()
        checkpoint_path = os.path.join(args.checkpoint_dir, 'latest.weights.h5')
        if os.path.exists(checkpoint_path):
            network.load(checkpoint_path)
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            print("Warning: No checkpoint found, exporting untrained model")
        network.export_tfjs(args.export_dir)
        print("Export complete!")
        return
    
    # Create trainer
    trainer = Trainer(
        buffer_size=200000,
        batch_size=args.batch_size,
        mcts_simulations=args.mcts_sims,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume from checkpoint
    if args.resume:
        try:
            trainer.load_checkpoint(args.resume)
        except Exception as e:
            print(f"Could not load checkpoint '{args.resume}': {e}")
            print("Starting fresh...")
    
    # Run training
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Iterations: {args.iterations}")
    print(f"Games/iteration: {args.games}")
    print(f"Batches/iteration: {args.batches}")
    print(f"Batch size: {args.batch_size}")
    print(f"MCTS simulations: {args.mcts_sims}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60 + "\n")
    
    import json
    import time
    start_time = time.time()
    status_file = os.path.join(args.checkpoint_dir, 'training_status.json')
    iteration_times = []
    
    for iteration in range(1, args.iterations + 1):
        iter_start = time.time()
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{args.iterations}")
        print(f"{'='*60}")
        
        # Generate self-play games
        print(f"\nGenerating {args.games} self-play games...")
        examples = trainer.generate_games(args.games)
        print(f"Generated {examples} training examples")
        
        # Train
        metrics = {'policy_loss': 0, 'value_loss': 0}
        if len(trainer.buffer) >= trainer.batch_size:
            print(f"\nTraining for {args.batches} batches...")
            metrics = trainer.train(args.batches, args.lr)
            
            print(f"\nTraining Metrics:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            print(f"  Buffer Size: {metrics['buffer_size']}")
            print(f"  Total Games: {metrics['total_games']}")
        else:
            print(f"Skipping training, need {trainer.batch_size} examples, have {len(trainer.buffer)}")
        
        # Save checkpoint
        if iteration % 10 == 0:
            trainer.save_checkpoint(f'iter_{iteration}')
        trainer.save_checkpoint('latest')
        
        # Track iteration time
        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
        avg_iter_time = sum(iteration_times) / len(iteration_times)
        remaining_iters = args.iterations - iteration
        eta_seconds = remaining_iters * avg_iter_time
        
        # Write status file
        status = {
            'current_iteration': iteration,
            'total_iterations': args.iterations,
            'games_per_iteration': args.games,
            'policy_loss': float(metrics.get('policy_loss', 0)),
            'value_loss': float(metrics.get('value_loss', 0)),
            'total_games': trainer.total_games,
            'buffer_size': len(trainer.buffer),
            'elapsed_seconds': time.time() - start_time,
            'eta_seconds': eta_seconds,
            'iteration_times': iteration_times,
            'completed': False
        }
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    # Export final model
    print(f"\n{'='*60}")
    print("Exporting Final Model")
    print(f"{'='*60}")
    os.makedirs(args.export_dir, exist_ok=True)
    trainer.export_model(args.export_dir)
    print(f"Model exported to {args.export_dir}")
    print("\nTraining complete!")
    
    # Mark training as complete
    status['completed'] = True
    status['eta_seconds'] = 0
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)


if __name__ == "__main__":
    main()
