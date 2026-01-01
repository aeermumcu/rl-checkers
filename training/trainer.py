"""
Self-play training loop for Checkers AI.
Uses MCTS with neural network guidance following AlphaZero methodology.
"""

import numpy as np
import os
from collections import deque
from typing import List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import random

from checkers_game import CheckersGame, Piece
from mcts import MCTS, self_play_game

try:
    import tensorflow as tf
    from model import CheckersNetwork, create_checkers_model, compile_model
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("TensorFlow not available")


@dataclass
class TrainingExample:
    """A single training example from self-play."""
    state: np.ndarray        # 8x8x4 board tensor
    policy: np.ndarray       # 1024-dim policy target
    value: float             # Game outcome from this player's perspective


class ReplayBuffer:
    """Experience replay buffer for training data."""
    
    def __init__(self, max_size: int = 100000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, examples: List[TrainingExample]):
        """Add examples to the buffer."""
        self.buffer.extend(examples)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of training examples."""
        batch = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        
        states = np.array([ex.state for ex in batch], dtype=np.float32)
        policies = np.array([ex.policy for ex in batch], dtype=np.float32)
        values = np.array([ex.value for ex in batch], dtype=np.float32)
        
        return states, policies, values
    
    def __len__(self):
        return len(self.buffer)


class Trainer:
    """Self-play trainer for the Checkers AI."""
    
    def __init__(self, 
                 network: Optional['CheckersNetwork'] = None,
                 buffer_size: int = 100000,
                 batch_size: int = 256,
                 mcts_simulations: int = 100,
                 temperature: float = 1.0,
                 checkpoint_dir: str = 'checkpoints'):
        """
        Args:
            network: Neural network (creates new one if None)
            buffer_size: Maximum replay buffer size
            batch_size: Training batch size
            mcts_simulations: MCTS simulations per move
            temperature: Temperature for move selection
            checkpoint_dir: Directory for saving checkpoints
        """
        if not HAS_TF:
            raise RuntimeError("TensorFlow is required for training")
        
        self.network = network or CheckersNetwork()
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training history
        self.total_games = 0
        self.losses = []
    
    def generate_games(self, num_games: int) -> int:
        """
        Generate self-play games and add to buffer.
        
        Returns:
            Number of training examples generated
        """
        total_examples = 0
        
        for game_idx in tqdm(range(num_games), desc="Self-play"):
            examples = self._play_game()
            self.buffer.add(examples)
            total_examples += len(examples)
            self.total_games += 1
        
        return total_examples
    
    def _play_game(self) -> List[TrainingExample]:
        """Play a single self-play game and return training examples."""
        game = CheckersGame()
        mcts = MCTS(
            network=self.network,
            num_simulations=self.mcts_simulations,
            temperature=self.temperature
        )
        
        history = []  # (state, policy, current_player)
        move_count = 0
        max_moves = 200
        
        while not game.is_game_over() and move_count < max_moves:
            # Store state
            state = game.get_board_tensor()
            current_player = game.current_player
            
            # Get MCTS policy
            policy, legal_moves = mcts.search(game)
            
            # Create full policy vector
            full_policy = np.zeros(1024, dtype=np.float32)
            for i, move in enumerate(legal_moves):
                full_policy[game.move_to_index(move)] = policy[i]
            
            history.append((state, full_policy, current_player))
            
            # Select and make move
            if self.temperature > 0:
                move_idx = np.random.choice(len(legal_moves), p=policy)
            else:
                move_idx = np.argmax(policy)
            
            game.make_move(legal_moves[move_idx])
            move_count += 1
            
            # Lower temperature after opening
            if move_count > 15:
                mcts.temperature = 0.5
            if move_count > 30:
                mcts.temperature = 0.1
        
        # Determine winner
        winner = game.get_winner()
        if winner is None:
            winner = 0  # Draw
        
        # Create training examples with outcome
        examples = []
        for state, policy, player in history:
            if winner == 0:
                value = 0.0  # Draw
            elif winner == player:
                value = 1.0  # Win
            else:
                value = -1.0  # Loss
            
            examples.append(TrainingExample(state, policy, value))
        
        return examples
    
    def train(self, num_batches: int = 100, learning_rate: float = 0.001) -> dict:
        """
        Train the network on examples from the replay buffer.
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.buffer) < self.batch_size:
            return {'error': 'Not enough examples in buffer'}
        
        # Ensure model is compiled with current learning rate
        self.network.model.optimizer.learning_rate = learning_rate
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for _ in tqdm(range(num_batches), desc="Training"):
            states, policies, values = self.buffer.sample(self.batch_size)
            
            # Train step
            loss = self.network.model.train_on_batch(
                states,
                {'policy_output': policies, 'value_output': values}
            )
            
            total_loss += loss[0]
            total_policy_loss += loss[1]
            total_value_loss += loss[2]
        
        metrics = {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'buffer_size': len(self.buffer),
            'total_games': self.total_games
        }
        
        self.losses.append(metrics['loss'])
        return metrics
    
    def save_checkpoint(self, name: str = 'checkpoint'):
        """Save a training checkpoint."""
        path = os.path.join(self.checkpoint_dir, f'{name}.weights.h5')
        self.network.save(path)
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, name: str = 'checkpoint'):
        """Load a training checkpoint."""
        path = os.path.join(self.checkpoint_dir, f'{name}.weights.h5')
        self.network.load(path)
        print(f"Loaded checkpoint: {path}")
    
    def export_model(self, output_dir: str):
        """Export model to TensorFlow.js format."""
        self.network.export_tfjs(output_dir)


def run_training(
    num_iterations: int = 100,
    games_per_iteration: int = 100,
    training_batches: int = 200,
    mcts_simulations: int = 100,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    checkpoint_dir: str = 'checkpoints',
    export_dir: str = '../model'
):
    """
    Main training loop.
    
    Args:
        num_iterations: Number of training iterations
        games_per_iteration: Self-play games per iteration
        training_batches: Training batches per iteration
        mcts_simulations: MCTS simulations per move
        batch_size: Training batch size
        learning_rate: Learning rate
        checkpoint_dir: Directory for checkpoints
        export_dir: Directory for TF.js export
    """
    print("=" * 60)
    print("Checkers AlphaZero Training")
    print("=" * 60)
    
    trainer = Trainer(
        buffer_size=200000,
        batch_size=batch_size,
        mcts_simulations=mcts_simulations,
        checkpoint_dir=checkpoint_dir
    )
    
    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{num_iterations}")
        print(f"{'='*60}")
        
        # Generate self-play games
        print(f"\nGenerating {games_per_iteration} self-play games...")
        examples = trainer.generate_games(games_per_iteration)
        print(f"Generated {examples} training examples")
        
        # Train on generated data
        print(f"\nTraining for {training_batches} batches...")
        metrics = trainer.train(training_batches, learning_rate)
        
        print(f"\nTraining Metrics:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
        print(f"  Value Loss: {metrics['value_loss']:.4f}")
        print(f"  Buffer Size: {metrics['buffer_size']}")
        print(f"  Total Games: {metrics['total_games']}")
        
        # Save checkpoint
        if iteration % 10 == 0:
            trainer.save_checkpoint(f'iter_{iteration}')
        
        # Always save latest
        trainer.save_checkpoint('latest')
    
    # Export final model
    print(f"\nExporting model to {export_dir}...")
    os.makedirs(export_dir, exist_ok=True)
    trainer.export_model(export_dir)
    print("Training complete!")
    
    return trainer


if __name__ == "__main__":
    # Quick test
    print("Testing trainer...")
    
    trainer = Trainer(mcts_simulations=20)
    
    print("\nGenerating test games...")
    examples = trainer.generate_games(2)
    print(f"Generated {examples} examples")
    
    print("\nTraining test...")
    if len(trainer.buffer) >= trainer.batch_size:
        metrics = trainer.train(5)
        print(f"Metrics: {metrics}")
    else:
        print(f"Need at least {trainer.batch_size} examples, have {len(trainer.buffer)}")
