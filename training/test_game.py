"""
Unit tests for Checkers game logic.
"""

import pytest
import numpy as np
from checkers_game import CheckersGame, Piece, Move


class TestBoardSetup:
    def test_initial_board_has_12_pieces_each(self):
        game = CheckersGame()
        black_count = np.sum((game.board == Piece.BLACK) | (game.board == Piece.BLACK_KING))
        white_count = np.sum((game.board == Piece.WHITE) | (game.board == Piece.WHITE_KING))
        assert black_count == 12
        assert white_count == 12
    
    def test_black_starts_first(self):
        game = CheckersGame()
        assert game.current_player == Piece.BLACK
    
    def test_pieces_on_dark_squares_only(self):
        game = CheckersGame()
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 0:  # Light square
                    assert game.board[row, col] == Piece.EMPTY


class TestMoveGeneration:
    def test_black_initial_moves(self):
        game = CheckersGame()
        moves = game.get_legal_moves()
        # Black should have 7 possible opening moves
        assert len(moves) == 7
    
    def test_no_moves_on_empty_board_for_player(self):
        game = CheckersGame()
        game.board = np.zeros((8, 8), dtype=np.int8)
        game.board[3, 3] = Piece.WHITE  # Only white piece
        game.current_player = Piece.BLACK
        moves = game.get_legal_moves()
        assert len(moves) == 0
    
    def test_king_moves_all_directions(self):
        game = CheckersGame()
        game.board = np.zeros((8, 8), dtype=np.int8)
        game.board[3, 3] = Piece.BLACK_KING
        game.current_player = Piece.BLACK
        moves = game.get_legal_moves()
        # King in center should have 4 diagonal moves
        assert len(moves) == 4


class TestCaptures:
    def test_simple_capture(self):
        game = CheckersGame()
        game.board = np.zeros((8, 8), dtype=np.int8)
        game.board[2, 2] = Piece.BLACK
        game.board[3, 3] = Piece.WHITE
        game.current_player = Piece.BLACK
        
        moves = game.get_legal_moves()
        assert len(moves) == 1
        assert len(moves[0].captures) == 1
        assert moves[0].to_pos == (4, 4)
    
    def test_mandatory_capture(self):
        game = CheckersGame()
        game.board = np.zeros((8, 8), dtype=np.int8)
        game.board[2, 2] = Piece.BLACK
        game.board[3, 3] = Piece.WHITE
        game.board[0, 0] = Piece.BLACK  # This piece has a simple move
        game.current_player = Piece.BLACK
        
        moves = game.get_legal_moves()
        # Should only return captures (mandatory)
        assert all(len(m.captures) > 0 for m in moves)
    
    def test_multi_jump(self):
        game = CheckersGame()
        game.board = np.zeros((8, 8), dtype=np.int8)
        game.board[0, 0] = Piece.BLACK
        game.board[1, 1] = Piece.WHITE
        game.board[3, 3] = Piece.WHITE
        game.current_player = Piece.BLACK
        
        moves = game.get_legal_moves()
        # Should have a double jump
        assert any(len(m.captures) == 2 for m in moves)


class TestKingPromotion:
    def test_black_promotes_on_row_7(self):
        game = CheckersGame()
        game.board = np.zeros((8, 8), dtype=np.int8)
        game.board[6, 2] = Piece.BLACK
        game.current_player = Piece.BLACK
        
        moves = game.get_legal_moves()
        assert len(moves) == 2
        assert all(m.is_promotion for m in moves)
        
        # Make the move
        game.make_move(moves[0])
        assert game.board[7, moves[0].to_pos[1]] == Piece.BLACK_KING
    
    def test_white_promotes_on_row_0(self):
        game = CheckersGame()
        game.board = np.zeros((8, 8), dtype=np.int8)
        game.board[1, 3] = Piece.WHITE
        game.current_player = Piece.WHITE
        
        moves = game.get_legal_moves()
        assert len(moves) == 2
        assert all(m.is_promotion for m in moves)


class TestWinConditions:
    def test_win_by_capture_all(self):
        game = CheckersGame()
        game.board = np.zeros((8, 8), dtype=np.int8)
        game.board[0, 0] = Piece.BLACK
        game.current_player = Piece.BLACK
        
        winner = game.get_winner()
        assert winner == Piece.BLACK  # White has no pieces
    
    def test_win_by_blocking(self):
        game = CheckersGame()
        game.board = np.zeros((8, 8), dtype=np.int8)
        game.board[0, 0] = Piece.WHITE
        game.board[1, 1] = Piece.BLACK
        game.current_player = Piece.WHITE
        
        winner = game.get_winner()
        assert winner == Piece.BLACK  # White is blocked
    
    def test_game_not_over_initially(self):
        game = CheckersGame()
        assert game.get_winner() is None
        assert not game.is_game_over()


class TestBoardTensor:
    def test_tensor_shape(self):
        game = CheckersGame()
        tensor = game.get_board_tensor()
        assert tensor.shape == (8, 8, 4)
    
    def test_tensor_sum_equals_piece_count(self):
        game = CheckersGame()
        tensor = game.get_board_tensor()
        # Should have 12 own pieces and 12 opponent pieces
        own_count = tensor[:, :, 0].sum() + tensor[:, :, 1].sum()
        opp_count = tensor[:, :, 2].sum() + tensor[:, :, 3].sum()
        assert own_count == 12
        assert opp_count == 12


class TestGameCopy:
    def test_copy_is_independent(self):
        game = CheckersGame()
        copy = game.copy()
        
        # Make a move on original
        moves = game.get_legal_moves()
        game.make_move(moves[0])
        
        # Copy should be unchanged
        assert game.current_player != copy.current_player


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
