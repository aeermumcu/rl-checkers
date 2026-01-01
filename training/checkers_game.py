"""
Checkers Game Logic - Python Implementation
Standard 8x8 American Checkers with mandatory captures, multi-jumps, and king promotion.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import IntEnum
import hashlib

class Piece(IntEnum):
    EMPTY = 0
    BLACK = 1       # Moves down (increasing row)
    WHITE = 2       # Moves up (decreasing row)
    BLACK_KING = 3
    WHITE_KING = 4

@dataclass
class Move:
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    captures: List[Tuple[int, int]]  # Positions of captured pieces
    is_promotion: bool = False
    
    def __hash__(self):
        return hash((self.from_pos, self.to_pos, tuple(self.captures)))
    
    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return (self.from_pos == other.from_pos and 
                self.to_pos == other.to_pos and 
                self.captures == other.captures)
    
    def to_notation(self) -> str:
        """Convert move to standard checkers notation."""
        def pos_to_num(pos):
            row, col = pos
            return (row * 4) + (col // 2) + 1
        if self.captures:
            return f"{pos_to_num(self.from_pos)}x{pos_to_num(self.to_pos)}"
        return f"{pos_to_num(self.from_pos)}-{pos_to_num(self.to_pos)}"

class CheckersGame:
    """
    8x8 American Checkers implementation.
    Black starts at top (rows 0-2), moves down.
    White starts at bottom (rows 5-7), moves up.
    """
    
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.current_player = Piece.BLACK  # Black moves first
        self.move_history: List[Move] = []
        self.no_capture_count = 0  # For draw detection
        self._setup_board()
    
    def _setup_board(self):
        """Initialize board with starting positions."""
        # Black pieces on top (rows 0-2)
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:  # Dark squares only
                    self.board[row, col] = Piece.BLACK
        
        # White pieces on bottom (rows 5-7)
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row, col] = Piece.WHITE
    
    def copy(self) -> 'CheckersGame':
        """Create a deep copy of the game state."""
        new_game = CheckersGame.__new__(CheckersGame)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_history = self.move_history.copy()
        new_game.no_capture_count = self.no_capture_count
        return new_game
    
    def get_state_hash(self) -> str:
        """Get a hash of the current board state for MCTS."""
        state = self.board.tobytes() + bytes([self.current_player])
        return hashlib.md5(state).hexdigest()
    
    def _is_player_piece(self, piece: int, player: int) -> bool:
        """Check if a piece belongs to a player."""
        if player == Piece.BLACK:
            return piece in (Piece.BLACK, Piece.BLACK_KING)
        return piece in (Piece.WHITE, Piece.WHITE_KING)
    
    def _is_king(self, piece: int) -> bool:
        """Check if a piece is a king."""
        return piece in (Piece.BLACK_KING, Piece.WHITE_KING)
    
    def _get_forward_directions(self, piece: int) -> List[Tuple[int, int]]:
        """Get forward move directions for a piece."""
        if piece == Piece.BLACK:
            return [(1, -1), (1, 1)]  # Move down
        elif piece == Piece.WHITE:
            return [(-1, -1), (-1, 1)]  # Move up
        else:  # Kings can move in all directions
            return [(1, -1), (1, 1), (-1, -1), (-1, 1)]
    
    def _in_bounds(self, row: int, col: int) -> bool:
        """Check if position is on the board."""
        return 0 <= row < 8 and 0 <= col < 8
    
    def _get_simple_moves(self, pos: Tuple[int, int]) -> List[Move]:
        """Get non-capture moves for a piece."""
        row, col = pos
        piece = self.board[row, col]
        moves = []
        
        for dr, dc in self._get_forward_directions(piece):
            new_row, new_col = row + dr, col + dc
            if self._in_bounds(new_row, new_col) and self.board[new_row, new_col] == Piece.EMPTY:
                is_promotion = self._would_promote(piece, new_row)
                moves.append(Move(pos, (new_row, new_col), [], is_promotion))
        
        return moves
    
    def _would_promote(self, piece: int, new_row: int) -> bool:
        """Check if a piece would be promoted at the given row."""
        if piece == Piece.BLACK and new_row == 7:
            return True
        if piece == Piece.WHITE and new_row == 0:
            return True
        return False
    
    def _get_captures_from_pos(self, pos: Tuple[int, int], board: np.ndarray, 
                                piece: int, captured: Set[Tuple[int, int]]) -> List[Move]:
        """Get all capture moves from a position (recursive for multi-jumps)."""
        row, col = pos
        opponent = Piece.WHITE if self._is_player_piece(piece, Piece.BLACK) else Piece.BLACK
        captures = []
        
        # Kings can capture in all directions
        directions = self._get_forward_directions(piece)
        
        for dr, dc in directions:
            mid_row, mid_col = row + dr, col + dc
            end_row, end_col = row + 2*dr, col + 2*dc
            
            if not self._in_bounds(end_row, end_col):
                continue
            
            mid_pos = (mid_row, mid_col)
            if mid_pos in captured:
                continue
                
            mid_piece = board[mid_row, mid_col]
            if self._is_player_piece(mid_piece, opponent) and board[end_row, end_col] == Piece.EMPTY:
                # Found a capture!
                new_captured = captured | {mid_pos}
                
                # Check for continuation jumps
                temp_board = board.copy()
                temp_board[row, col] = Piece.EMPTY
                temp_board[mid_row, mid_col] = Piece.EMPTY
                temp_board[end_row, end_col] = piece
                
                # Check if piece gets promoted
                actual_piece = piece
                is_promotion = self._would_promote(piece, end_row)
                if is_promotion and not self._is_king(piece):
                    actual_piece = Piece.BLACK_KING if piece == Piece.BLACK else Piece.WHITE_KING
                    temp_board[end_row, end_col] = actual_piece
                
                # Look for continuation jumps (only if not just promoted)
                if not is_promotion:
                    continuations = self._get_captures_from_pos(
                        (end_row, end_col), temp_board, piece, new_captured
                    )
                    if continuations:
                        for cont in continuations:
                            captures.append(Move(
                                pos, cont.to_pos,
                                [mid_pos] + cont.captures,
                                cont.is_promotion
                            ))
                    else:
                        captures.append(Move(pos, (end_row, end_col), [mid_pos], is_promotion))
                else:
                    captures.append(Move(pos, (end_row, end_col), [mid_pos], is_promotion))
        
        return captures
    
    def get_legal_moves(self) -> List[Move]:
        """Get all legal moves for the current player."""
        captures = []
        simple_moves = []
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row, col]
                if not self._is_player_piece(piece, self.current_player):
                    continue
                
                pos = (row, col)
                
                # Get captures
                piece_captures = self._get_captures_from_pos(pos, self.board, piece, set())
                captures.extend(piece_captures)
                
                # Get simple moves
                piece_moves = self._get_simple_moves(pos)
                simple_moves.extend(piece_moves)
        
        # Mandatory capture: if captures available, must take them
        if captures:
            return captures
        return simple_moves
    
    def make_move(self, move: Move) -> bool:
        """Execute a move. Returns True if successful."""
        legal_moves = self.get_legal_moves()
        if move not in legal_moves:
            return False
        
        from_row, from_col = move.from_pos
        to_row, to_col = move.to_pos
        piece = self.board[from_row, from_col]
        
        # Move the piece
        self.board[from_row, from_col] = Piece.EMPTY
        self.board[to_row, to_col] = piece
        
        # Remove captured pieces
        for cap_pos in move.captures:
            self.board[cap_pos[0], cap_pos[1]] = Piece.EMPTY
        
        # Check for promotion
        if move.is_promotion:
            if piece == Piece.BLACK:
                self.board[to_row, to_col] = Piece.BLACK_KING
            elif piece == Piece.WHITE:
                self.board[to_row, to_col] = Piece.WHITE_KING
        
        # Update game state
        self.move_history.append(move)
        if move.captures:
            self.no_capture_count = 0
        else:
            self.no_capture_count += 1
        
        # Switch player
        self.current_player = Piece.WHITE if self.current_player == Piece.BLACK else Piece.BLACK
        
        return True
    
    def get_winner(self) -> Optional[int]:
        """
        Get the winner of the game.
        Returns Piece.BLACK, Piece.WHITE, 0 for draw, or None if game ongoing.
        """
        # Check for draw by no captures
        if self.no_capture_count >= 80:  # 40 moves each without capture
            return 0
        
        # Count pieces
        black_count = np.sum((self.board == Piece.BLACK) | (self.board == Piece.BLACK_KING))
        white_count = np.sum((self.board == Piece.WHITE) | (self.board == Piece.WHITE_KING))
        
        if black_count == 0:
            return Piece.WHITE
        if white_count == 0:
            return Piece.BLACK
        
        # Check if current player can move
        legal_moves = self.get_legal_moves()
        if len(legal_moves) == 0:
            # Current player has no moves - they lose
            return Piece.WHITE if self.current_player == Piece.BLACK else Piece.BLACK
        
        return None  # Game still in progress
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.get_winner() is not None
    
    def get_board_tensor(self) -> np.ndarray:
        """
        Get board state as tensor for neural network input.
        Returns 8x8x4 tensor (one-hot encoding from current player's perspective).
        Channels: [own_normal, own_king, opponent_normal, opponent_king]
        """
        tensor = np.zeros((8, 8, 4), dtype=np.float32)
        
        if self.current_player == Piece.BLACK:
            own_normal, own_king = Piece.BLACK, Piece.BLACK_KING
            opp_normal, opp_king = Piece.WHITE, Piece.WHITE_KING
        else:
            own_normal, own_king = Piece.WHITE, Piece.WHITE_KING
            opp_normal, opp_king = Piece.BLACK, Piece.BLACK_KING
            # Flip board for white's perspective
        
        board = self.board
        if self.current_player == Piece.WHITE:
            board = np.flip(board, axis=0)  # Flip vertically
        
        tensor[:, :, 0] = (board == own_normal).astype(np.float32)
        tensor[:, :, 1] = (board == own_king).astype(np.float32)
        tensor[:, :, 2] = (board == opp_normal).astype(np.float32)
        tensor[:, :, 3] = (board == opp_king).astype(np.float32)
        
        return tensor
    
    def move_to_index(self, move: Move) -> int:
        """Convert a move to a policy index."""
        # Simplified indexing: from_square * 32 + offset
        # This gives us up to 32*32 = 1024 possible moves
        from_row, from_col = move.from_pos
        to_row, to_col = move.to_pos
        
        from_idx = from_row * 4 + from_col // 2
        to_idx = to_row * 4 + to_col // 2
        
        return from_idx * 32 + to_idx
    
    def index_to_move_positions(self, index: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Convert policy index back to positions."""
        from_idx = index // 32
        to_idx = index % 32
        
        from_row = from_idx // 4
        from_col = (from_idx % 4) * 2 + (1 if from_row % 2 == 0 else 0)
        
        to_row = to_idx // 4
        to_col = (to_idx % 4) * 2 + (1 if to_row % 2 == 0 else 0)
        
        return (from_row, from_col), (to_row, to_col)
    
    def __str__(self) -> str:
        """String representation of the board."""
        symbols = {
            Piece.EMPTY: '.',
            Piece.BLACK: 'b',
            Piece.WHITE: 'w',
            Piece.BLACK_KING: 'B',
            Piece.WHITE_KING: 'W'
        }
        
        lines = ['  0 1 2 3 4 5 6 7']
        for row in range(8):
            line = f'{row} '
            for col in range(8):
                if (row + col) % 2 == 0:
                    line += '  '  # Light square
                else:
                    line += symbols[self.board[row, col]] + ' '
            lines.append(line)
        
        player = 'Black' if self.current_player == Piece.BLACK else 'White'
        lines.append(f'\nCurrent player: {player}')
        
        return '\n'.join(lines)


def self_play_game(policy_fn=None, temperature=1.0) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    """
    Play a game with optional policy function.
    Returns (states, policies, winner) for training.
    """
    game = CheckersGame()
    states = []
    policies = []
    
    while not game.is_game_over():
        state = game.get_board_tensor()
        legal_moves = game.get_legal_moves()
        
        if len(legal_moves) == 0:
            break
        
        # Create policy vector
        policy = np.zeros(1024, dtype=np.float32)
        
        if policy_fn is not None:
            # Use neural network policy
            probs = policy_fn(state, legal_moves)
            for move, prob in zip(legal_moves, probs):
                policy[game.move_to_index(move)] = prob
        else:
            # Random policy
            uniform_prob = 1.0 / len(legal_moves)
            for move in legal_moves:
                policy[game.move_to_index(move)] = uniform_prob
        
        states.append(state)
        policies.append(policy)
        
        # Select move based on policy
        legal_indices = [game.move_to_index(m) for m in legal_moves]
        legal_probs = [policy[i] for i in legal_indices]
        
        if temperature > 0:
            # Apply temperature
            legal_probs = np.array(legal_probs) ** (1.0 / temperature)
            legal_probs /= legal_probs.sum()
            move_idx = np.random.choice(len(legal_moves), p=legal_probs)
        else:
            move_idx = np.argmax(legal_probs)
        
        game.make_move(legal_moves[move_idx])
    
    winner = game.get_winner()
    return states, policies, winner


if __name__ == "__main__":
    # Test the game
    game = CheckersGame()
    print(game)
    print("\nLegal moves:", len(game.get_legal_moves()))
    
    # Play a few random moves
    for _ in range(10):
        moves = game.get_legal_moves()
        if not moves:
            break
        move = np.random.choice(moves)
        print(f"\nMove: {move.to_notation()}")
        game.make_move(move)
        print(game)
