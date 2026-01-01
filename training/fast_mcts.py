"""
Fast GPU-Optimized MCTS with Virtual Loss Parallelization.

Key optimization: Instead of evaluating one leaf at a time, we:
1. Select N leaves in parallel using virtual losses to avoid duplicates
2. Batch all N neural network predictions in one GPU call
3. Backpropagate all N results

This achieves high GPU utilization even with Python's GIL.
"""

import numpy as np
import math
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from checkers_game import CheckersGame, Move, Piece


@dataclass
class FastNode:
    """Compact MCTS node for speed."""
    state: CheckersGame
    parent: Optional['FastNode'] = None
    move: Optional[Move] = None
    children: dict = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    virtual_loss: int = 0  # For parallel selection


class FastMCTS:
    """
    GPU-optimized MCTS with virtual loss parallelization.
    
    Uses batched neural network predictions for high GPU throughput.
    """
    
    def __init__(self, network, num_simulations=100, batch_size=8,
                 c_puct=2.0, dirichlet_alpha=0.3, dirichlet_weight=0.25):
        self.network = network
        self.num_simulations = num_simulations
        self.batch_size = batch_size  # Leaves to evaluate per batch
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
    
    def search(self, game: CheckersGame, temperature: float = 1.0) -> Tuple[np.ndarray, List[Move]]:
        """
        Run MCTS search and return policy over legal moves.
        
        Returns:
            (policy, legal_moves) - probability distribution over moves
        """
        root = FastNode(state=game.copy())
        
        # Expand root
        self._expand_node(root)
        
        # Add exploration noise to root
        if root.children:
            self._add_dirichlet_noise(root)
        
        # Run batched simulations
        sims_done = 0
        while sims_done < self.num_simulations:
            # Collect batch of leaves using virtual loss
            leaves = []
            paths = []
            
            batch = min(self.batch_size, self.num_simulations - sims_done)
            for _ in range(batch):
                node, path = self._select_leaf(root)
                if node is not None:
                    leaves.append(node)
                    paths.append(path)
            
            if not leaves:
                break
            
            # Batch evaluate all leaves
            values = self._batch_evaluate(leaves)
            
            # Backpropagate all
            for path, value in zip(paths, values):
                self._backpropagate(path, value)
            
            sims_done += len(leaves)
        
        # Build policy from visit counts
        legal_moves = game.get_legal_moves()
        policy = np.zeros(len(legal_moves), dtype=np.float32)
        
        for i, move in enumerate(legal_moves):
            move_idx = game.move_to_index(move)
            if move_idx in root.children:
                policy[i] = root.children[move_idx].visit_count
        
        # Apply temperature
        if temperature > 0 and policy.sum() > 0:
            policy = policy ** (1.0 / temperature)
            policy = policy / policy.sum()
        elif policy.sum() > 0:
            best = np.argmax(policy)
            policy = np.zeros_like(policy)
            policy[best] = 1.0
        
        return policy, legal_moves
    
    def _select_leaf(self, root: FastNode) -> Tuple[Optional[FastNode], List[FastNode]]:
        """Select a leaf node, applying virtual loss to prevent re-selection."""
        node = root
        path = [node]
        
        while node.children:
            # Select child with modified UCB (accounts for virtual loss)
            best_score = -float('inf')
            best_child = None
            
            for child in node.children.values():
                # UCB with virtual loss penalty
                visits_with_vl = child.visit_count + child.virtual_loss
                value = child.value_sum / max(visits_with_vl, 1)
                
                exploration = self.c_puct * child.prior * \
                    math.sqrt(node.visit_count + node.virtual_loss) / (1 + visits_with_vl)
                
                score = value + exploration
                if score > best_score:
                    best_score = score
                    best_child = child
            
            if best_child is None:
                break
                
            node = best_child
            path.append(node)
            
            if node.state.is_game_over():
                break
        
        # Apply virtual loss along path
        for n in path:
            n.virtual_loss += 1
        
        # If already expanded and not terminal, don't re-evaluate
        if node.children or node.state.is_game_over():
            # This is a repeated terminal or already expanded - just use value
            return None, path
        
        return node, path
    
    def _batch_evaluate(self, leaves: List[FastNode]) -> List[float]:
        """Batch evaluate leaf nodes using GPU."""
        if not leaves:
            return []
        
        # Collect states
        states = []
        for leaf in leaves:
            if leaf.state.is_game_over():
                states.append(None)
            else:
                states.append(leaf.state.get_board_tensor())
        
        # Find non-terminal leaves for batched prediction
        non_terminal_indices = [i for i, s in enumerate(states) if s is not None]
        
        values = [0.0] * len(leaves)
        
        if non_terminal_indices:
            # Batch predict
            batch_states = np.array([states[i] for i in non_terminal_indices], dtype=np.float32)
            policies, nn_values = self.network.predict_batch(batch_states)
            
            # Expand nodes and get values
            for batch_idx, leaf_idx in enumerate(non_terminal_indices):
                leaf = leaves[leaf_idx]
                policy = policies[batch_idx]
                value = nn_values[batch_idx, 0] if nn_values.ndim > 1 else nn_values[batch_idx]
                
                # Expand the node
                legal_moves = leaf.state.get_legal_moves()
                for move in legal_moves:
                    move_idx = leaf.state.move_to_index(move)
                    new_state = leaf.state.copy()
                    new_state.make_move(move)
                    
                    child = FastNode(
                        state=new_state,
                        parent=leaf,
                        move=move,
                        prior=float(policy[move_idx])
                    )
                    leaf.children[move_idx] = child
                
                values[leaf_idx] = float(value)
        
        # Handle terminal nodes
        for i, leaf in enumerate(leaves):
            if states[i] is None:  # Terminal
                winner = leaf.state.get_winner()
                if winner == 0:
                    values[i] = 0.0
                else:
                    parent_player = Piece.WHITE if leaf.state.current_player == Piece.BLACK else Piece.BLACK
                    values[i] = 1.0 if winner == parent_player else -1.0
        
        return values
    
    def _expand_node(self, node: FastNode):
        """Expand a single node (used for root)."""
        if node.state.is_game_over():
            return
        
        state = node.state.get_board_tensor()
        policy, value = self.network.predict(state)
        
        legal_moves = node.state.get_legal_moves()
        for move in legal_moves:
            move_idx = node.state.move_to_index(move)
            new_state = node.state.copy()
            new_state.make_move(move)
            
            child = FastNode(
                state=new_state,
                parent=node,
                move=move,
                prior=float(policy[move_idx])
            )
            node.children[move_idx] = child
    
    def _backpropagate(self, path: List[FastNode], value: float):
        """Backpropagate value and remove virtual loss."""
        for node in reversed(path):
            node.virtual_loss -= 1
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent
    
    def _add_dirichlet_noise(self, node: FastNode):
        """Add exploration noise to root priors."""
        children = list(node.children.values())
        if not children:
            return
        
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))
        for i, child in enumerate(children):
            child.prior = (1 - self.dirichlet_weight) * child.prior + \
                         self.dirichlet_weight * noise[i]


class FastTrainer:
    """
    Fast GPU-optimized self-play trainer.
    """
    
    def __init__(self, network, mcts_sims=100, mcts_batch=8, temperature=1.0):
        self.network = network
        self.mcts = FastMCTS(network, num_simulations=mcts_sims, batch_size=mcts_batch)
        self.temperature = temperature
    
    def play_game(self) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Play one self-play game and return training examples.
        
        Returns list of (state, policy, player) tuples.
        """
        game = CheckersGame()
        history = []
        move_count = 0
        
        while not game.is_game_over() and move_count < 200:
            # Temperature schedule
            if move_count < 15:
                temp = self.temperature
            elif move_count < 30:
                temp = 0.5
            else:
                temp = 0.1
            
            # MCTS search
            policy, legal_moves = self.mcts.search(game, temperature=temp)
            
            if not legal_moves:
                break
            
            # Store training example
            state = game.get_board_tensor()
            full_policy = np.zeros(1024, dtype=np.float32)
            for i, move in enumerate(legal_moves):
                full_policy[game.move_to_index(move)] = policy[i]
            
            history.append((state.copy(), full_policy, game.current_player))
            
            # Make move
            if temp > 0:
                move_idx = np.random.choice(len(legal_moves), p=policy)
            else:
                move_idx = np.argmax(policy)
            
            game.make_move(legal_moves[move_idx])
            move_count += 1
        
        # Convert to training examples with game outcome
        winner = game.get_winner()
        examples = []
        for state, policy, player in history:
            if winner == 0:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            examples.append((state, policy, value))
        
        return examples
    
    def generate_examples(self, num_games: int, progress_callback=None) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Generate training examples from multiple games."""
        all_examples = []
        
        for i in range(num_games):
            examples = self.play_game()
            all_examples.extend(examples)
            
            if progress_callback:
                progress_callback(i + 1, num_games, len(examples))
        
        return all_examples


if __name__ == "__main__":
    import time
    from model import CheckersNetwork
    
    print("Testing Fast MCTS...")
    
    network = CheckersNetwork()
    trainer = FastTrainer(network, mcts_sims=50, mcts_batch=8)
    
    print("\nPlaying test game...")
    start = time.time()
    examples = trainer.play_game()
    elapsed = time.time() - start
    
    print(f"Game completed: {len(examples)} moves in {elapsed:.1f}s")
    print(f"Speed: {len(examples)/elapsed:.1f} moves/sec")
    
    print("\nRunning 2 games...")
    start = time.time()
    examples = trainer.generate_examples(2)
    elapsed = time.time() - start
    
    print(f"Generated {len(examples)} examples in {elapsed:.1f}s")
    print(f"Speed: {2/elapsed:.2f} games/sec")
