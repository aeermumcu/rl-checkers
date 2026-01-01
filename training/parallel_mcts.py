"""
Parallel self-play for GPU-optimized training.
Runs multiple games simultaneously and batches neural network predictions.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from checkers_game import CheckersGame, Move, Piece
import math


@dataclass
class ParallelGameState:
    """State for a single game in parallel simulation."""
    game: CheckersGame
    history: List[Tuple[np.ndarray, np.ndarray, int]]  # (state, policy, player)
    move_count: int = 0
    finished: bool = False
    

class ParallelMCTS:
    """
    MCTS that batches neural network predictions across multiple game states.
    This allows efficient GPU utilization.
    """
    
    def __init__(self, network, num_simulations=100, temperature=1.0):
        self.network = network
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.c_puct = 2.0
        self.dirichlet_alpha = 0.3
        self.dirichlet_weight = 0.25
    
    def search_batch(self, games: List[CheckersGame]) -> List[Tuple[np.ndarray, List[Move]]]:
        """
        Run MCTS on multiple games simultaneously with batched predictions.
        
        Returns list of (policy, legal_moves) for each game.
        """
        # Initialize all search trees
        trees = []
        for game in games:
            root = {'state': game.copy(), 'parent': None, 'children': {},
                    'visit_count': 0, 'value_sum': 0.0, 'prior': 0.0, 'move': None}
            trees.append({'root': root, 'game': game})
        
        # Batch expand all roots
        root_states = [t['root']['state'] for t in trees]
        self._batch_expand_nodes([t['root'] for t in trees], root_states)
        
        # Add dirichlet noise to roots
        for t in trees:
            self._add_noise(t['root'])
        
        # Run simulations with batching
        for _ in range(self.num_simulations):
            # Selection phase - collect leaf nodes
            leaves = []
            leaf_states = []
            
            for t in trees:
                node = t['root']
                
                # Traverse to leaf
                while node['children'] and not node['state'].is_game_over():
                    node = self._select_child(node)
                
                if node['state'].is_game_over():
                    # Terminal - backpropagate immediately
                    value = self._terminal_value(node)
                    self._backpropagate(node, value)
                else:
                    leaves.append(node)
                    leaf_states.append(node['state'])
            
            # Batch expand and evaluate all leaves
            if leaves:
                values = self._batch_expand_nodes(leaves, leaf_states)
                for node, value in zip(leaves, values):
                    self._backpropagate(node, value)
        
        # Extract policies
        results = []
        for t in trees:
            legal_moves = t['game'].get_legal_moves()
            policy = np.zeros(len(legal_moves), dtype=np.float32)
            
            for i, move in enumerate(legal_moves):
                move_idx = t['game'].move_to_index(move)
                if move_idx in t['root']['children']:
                    policy[i] = t['root']['children'][move_idx]['visit_count']
            
            # Apply temperature
            if self.temperature > 0:
                policy = policy ** (1 / self.temperature)
            policy = policy / (policy.sum() + 1e-8)
            
            results.append((policy, legal_moves))
        
        return results
    
    def _batch_expand_nodes(self, nodes: List[dict], states: List[CheckersGame]) -> List[float]:
        """Expand multiple nodes with a single batched neural network call."""
        if not nodes:
            return []
        
        # Collect all board tensors
        tensors = np.array([s.get_board_tensor() for s in states], dtype=np.float32)
        
        # Single batched prediction
        policies, values = self.network.predict_batch(tensors)
        
        # Create children for each node
        result_values = []
        for i, (node, state) in enumerate(zip(nodes, states)):
            legal_moves = state.get_legal_moves()
            
            if not legal_moves or state.is_game_over():
                result_values.append(self._terminal_value(node))
                continue
            
            policy = policies[i]
            value = values[i, 0] if values.ndim > 1 else values[i]
            
            for move in legal_moves:
                move_idx = state.move_to_index(move)
                new_state = state.copy()
                new_state.make_move(move)
                
                child = {
                    'state': new_state,
                    'parent': node,
                    'children': {},
                    'visit_count': 0,
                    'value_sum': 0.0,
                    'prior': float(policy[move_idx]),
                    'move': move
                }
                node['children'][move_idx] = child
            
            result_values.append(float(value))
        
        return result_values
    
    def _select_child(self, node: dict) -> dict:
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_child = None
        
        for child in node['children'].values():
            q = child['value_sum'] / max(child['visit_count'], 1)
            exploration = self.c_puct * child['prior'] * \
                         math.sqrt(node['visit_count']) / (1 + child['visit_count'])
            score = q + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _add_noise(self, node: dict):
        """Add Dirichlet noise to root."""
        if not node['children']:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(node['children']))
        for i, child in enumerate(node['children'].values()):
            child['prior'] = (1 - self.dirichlet_weight) * child['prior'] + \
                            self.dirichlet_weight * noise[i]
    
    def _terminal_value(self, node: dict) -> float:
        """Get value for terminal state."""
        winner = node['state'].get_winner()
        if winner == 0:
            return 0.0
        parent_player = Piece.WHITE if node['state'].current_player == Piece.BLACK else Piece.BLACK
        return 1.0 if winner == parent_player else -1.0
    
    def _backpropagate(self, node: dict, value: float):
        """Backpropagate value through tree."""
        while node is not None:
            node['visit_count'] += 1
            node['value_sum'] += value
            value = -value  # Flip for opponent
            node = node.get('parent')


class ParallelSelfPlay:
    """
    Run multiple self-play games in parallel with batched neural network predictions.
    """
    
    def __init__(self, network, num_parallel=16, mcts_sims=100, max_moves=200):
        self.network = network
        self.num_parallel = num_parallel
        self.mcts = ParallelMCTS(network, num_simulations=mcts_sims)
        self.max_moves = max_moves
    
    def generate_games(self, num_games: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Generate training examples from self-play games.
        
        Returns list of (state, policy, value) training examples.
        """
        all_examples = []
        games_completed = 0
        
        # Run in batches
        while games_completed < num_games:
            batch_size = min(self.num_parallel, num_games - games_completed)
            
            # Initialize parallel games
            states = [ParallelGameState(
                game=CheckersGame(),
                history=[]
            ) for _ in range(batch_size)]
            
            # Run games until all complete
            while any(not s.finished for s in states):
                # Get active games
                active = [(i, s) for i, s in enumerate(states) if not s.finished]
                if not active:
                    break
                
                active_indices, active_states = zip(*active)
                active_games = [s.game for s in active_states]
                
                # Batch MCTS search
                results = self.mcts.search_batch(active_games)
                
                # Process results and make moves
                for (idx, state), (policy, legal_moves) in zip(active, results):
                    if not legal_moves:
                        state.finished = True
                        continue
                    
                    # Store training example
                    board_tensor = state.game.get_board_tensor()
                    full_policy = np.zeros(1024, dtype=np.float32)
                    for i, move in enumerate(legal_moves):
                        full_policy[state.game.move_to_index(move)] = policy[i]
                    
                    state.history.append((
                        board_tensor.copy(),
                        full_policy,
                        state.game.current_player
                    ))
                    
                    # Select and make move
                    if self.mcts.temperature > 0:
                        move_idx = np.random.choice(len(legal_moves), p=policy)
                    else:
                        move_idx = np.argmax(policy)
                    
                    state.game.make_move(legal_moves[move_idx])
                    state.move_count += 1
                    
                    # Lower temperature after opening
                    if state.move_count == 15:
                        self.mcts.temperature = 0.5
                    elif state.move_count == 30:
                        self.mcts.temperature = 0.1
                    
                    # Check termination
                    if state.game.is_game_over() or state.move_count >= self.max_moves:
                        state.finished = True
            
            # Reset temperature for next batch
            self.mcts.temperature = 1.0
            
            # Convert histories to training examples
            for state in states:
                winner = state.game.get_winner()
                if winner is None:
                    winner = 0  # Draw
                
                for board, policy, player in state.history:
                    if winner == 0:
                        value = 0.0
                    elif winner == player:
                        value = 1.0
                    else:
                        value = -1.0
                    
                    all_examples.append((board, policy, value))
            
            games_completed += batch_size
        
        return all_examples


if __name__ == "__main__":
    # Test parallel self-play
    print("Testing parallel self-play...")
    
    from model import CheckersNetwork
    
    network = CheckersNetwork()
    parallel = ParallelSelfPlay(network, num_parallel=4, mcts_sims=10)
    
    print("Generating 4 games in parallel...")
    import time
    start = time.time()
    examples = parallel.generate_games(4)
    elapsed = time.time() - start
    
    print(f"Generated {len(examples)} examples in {elapsed:.2f}s")
    print(f"Speed: {4/elapsed:.2f} games/sec")
