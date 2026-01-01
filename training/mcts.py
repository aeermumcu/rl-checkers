"""
Monte Carlo Tree Search (MCTS) implementation for Checkers.
AlphaZero-style MCTS with neural network guidance.
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from checkers_game import CheckersGame, Move, Piece


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    state: CheckersGame
    parent: Optional['MCTSNode'] = None
    move: Optional[Move] = None  # Move that led to this state
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)  # move_index -> child
    
    # Statistics
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0  # Prior probability from policy network
    
    @property
    def q_value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    @property
    def ucb_score(self) -> float:
        """Upper Confidence Bound score for selection."""
        if self.parent is None:
            return 0.0
        
        c_puct = 2.0  # Exploration constant
        
        # UCB formula with prior
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + exploration
    
    def is_expanded(self) -> bool:
        """Check if this node has been expanded."""
        return len(self.children) > 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self.state.is_game_over()


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.
    """
    
    def __init__(self, network=None, num_simulations=100, 
                 temperature=1.0, dirichlet_alpha=0.3, dirichlet_weight=0.25):
        """
        Args:
            network: Neural network for policy and value predictions
            num_simulations: Number of MCTS simulations per move
            temperature: Temperature for move selection
            dirichlet_alpha: Dirichlet noise parameter for exploration
            dirichlet_weight: Weight of Dirichlet noise at root
        """
        self.network = network
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
    
    def search(self, game: CheckersGame) -> Tuple[np.ndarray, List[Move]]:
        """
        Run MCTS from the given position.
        
        Returns:
            (policy, legal_moves) - policy is a probability distribution over legal moves
        """
        # Create root node
        root = MCTSNode(state=game.copy())
        
        # Expand root
        self._expand(root)
        
        # Add Dirichlet noise to root for exploration
        if root.children:
            self._add_dirichlet_noise(root)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            
            # Selection - traverse tree using UCB
            while node.is_expanded() and not node.is_terminal():
                node = self._select_child(node)
            
            # Check for terminal state
            if node.is_terminal():
                value = self._get_terminal_value(node)
            else:
                # Expansion and evaluation
                value = self._expand(node)
            
            # Backpropagation
            self._backpropagate(node, value)
        
        # Extract policy from visit counts
        legal_moves = game.get_legal_moves()
        policy = np.zeros(len(legal_moves), dtype=np.float32)
        
        for i, move in enumerate(legal_moves):
            move_idx = game.move_to_index(move)
            if move_idx in root.children:
                policy[i] = root.children[move_idx].visit_count
        
        # Apply temperature
        if self.temperature > 0:
            policy = policy ** (1.0 / self.temperature)
        
        # Normalize
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy /= policy_sum
        else:
            policy = np.ones(len(legal_moves)) / len(legal_moves)
        
        return policy, legal_moves
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select the child with highest UCB score."""
        best_score = -float('inf')
        best_child = None
        
        for child in node.children.values():
            score = child.ucb_score
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _expand(self, node: MCTSNode) -> float:
        """
        Expand a node and return its value.
        """
        game = node.state
        legal_moves = game.get_legal_moves()
        
        if not legal_moves or game.is_game_over():
            return self._get_terminal_value(node)
        
        # Get neural network predictions
        if self.network is not None:
            board_tensor = game.get_board_tensor()
            policy, value = self.network.predict(board_tensor)
        else:
            # Random policy if no network
            policy = np.ones(1024) / 1024
            value = 0.0
        
        # Create child nodes for all legal moves
        for move in legal_moves:
            move_idx = game.move_to_index(move)
            
            # Create new game state
            new_game = game.copy()
            new_game.make_move(move)
            
            # Create child node
            child = MCTSNode(
                state=new_game,
                parent=node,
                move=move,
                prior=policy[move_idx]
            )
            node.children[move_idx] = child
        
        # Return value from current player's perspective
        return value
    
    def _add_dirichlet_noise(self, node: MCTSNode):
        """Add Dirichlet noise to root node priors for exploration."""
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(node.children))
        
        for i, child in enumerate(node.children.values()):
            child.prior = (1 - self.dirichlet_weight) * child.prior + \
                          self.dirichlet_weight * noise[i]
    
    def _get_terminal_value(self, node: MCTSNode) -> float:
        """Get value for a terminal state."""
        winner = node.state.get_winner()
        
        if winner == 0:  # Draw
            return 0.0
        
        # Value from perspective of the player who just moved (parent's player)
        parent_player = Piece.WHITE if node.state.current_player == Piece.BLACK else Piece.BLACK
        
        if winner == parent_player:
            return 1.0
        else:
            return -1.0
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate the value up the tree."""
        while node is not None:
            node.visit_count += 1
            # Negate value as we go up (opponent's perspective)
            node.value_sum += value
            value = -value
            node = node.parent
    
    def get_best_move(self, game: CheckersGame) -> Move:
        """Get the best move according to MCTS."""
        policy, legal_moves = self.search(game)
        
        if self.temperature == 0:
            # Greedy selection
            best_idx = np.argmax(policy)
        else:
            # Probabilistic selection
            best_idx = np.random.choice(len(legal_moves), p=policy)
        
        return legal_moves[best_idx]


class MCTSPlayer:
    """A player that uses MCTS to select moves."""
    
    def __init__(self, network=None, num_simulations=100, temperature=0.0):
        self.mcts = MCTS(
            network=network,
            num_simulations=num_simulations,
            temperature=temperature,
            dirichlet_alpha=0.0,  # No noise for playing
            dirichlet_weight=0.0
        )
    
    def get_move(self, game: CheckersGame) -> Move:
        """Get the best move for the current position."""
        return self.mcts.get_best_move(game)


def self_play_game(network=None, num_simulations=100, temperature=1.0) -> Tuple[List, List, int]:
    """
    Play a game using MCTS self-play.
    
    Returns:
        (states, policies, winner) for training
    """
    game = CheckersGame()
    mcts = MCTS(
        network=network,
        num_simulations=num_simulations,
        temperature=temperature
    )
    
    states = []
    policies = []
    
    move_count = 0
    max_moves = 200  # Prevent infinite games
    
    while not game.is_game_over() and move_count < max_moves:
        # Store state
        states.append(game.get_board_tensor())
        
        # Get MCTS policy
        policy, legal_moves = mcts.search(game)
        
        # Create full policy vector
        full_policy = np.zeros(1024, dtype=np.float32)
        for i, move in enumerate(legal_moves):
            full_policy[game.move_to_index(move)] = policy[i]
        policies.append(full_policy)
        
        # Select and make move
        if temperature > 0:
            move_idx = np.random.choice(len(legal_moves), p=policy)
        else:
            move_idx = np.argmax(policy)
        
        game.make_move(legal_moves[move_idx])
        move_count += 1
        
        # Lower temperature after first few moves
        if move_count > 10:
            mcts.temperature = 0.5
        if move_count > 20:
            mcts.temperature = 0.1
    
    winner = game.get_winner()
    if winner is None:
        winner = 0  # Draw if max moves reached
    
    return states, policies, winner


if __name__ == "__main__":
    # Test MCTS without neural network
    print("Testing MCTS with random policy...")
    
    game = CheckersGame()
    mcts = MCTS(network=None, num_simulations=50)
    
    for i in range(5):
        if game.is_game_over():
            break
        
        policy, legal_moves = mcts.search(game)
        best_move = legal_moves[np.argmax(policy)]
        print(f"\nMove {i+1}: {best_move.to_notation()}")
        print(f"Policy: {policy[:5]}...")
        
        game.make_move(best_move)
        print(game)
    
    print("\n--- Self-play game test ---")
    states, policies, winner = self_play_game(num_simulations=20)
    print(f"Game length: {len(states)} moves")
    print(f"Winner: {winner}")
