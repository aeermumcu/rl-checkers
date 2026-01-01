/**
 * Checkers Game Logic - JavaScript Implementation
 * Standard 8x8 American Checkers
 */

// Piece types
const Piece = {
    EMPTY: 0,
    BLACK: 1,       // AI - moves down (increasing row)
    WHITE: 2,       // Player - moves up (decreasing row)
    BLACK_KING: 3,
    WHITE_KING: 4
};

/**
 * Move representation
 */
class Move {
    constructor(fromPos, toPos, captures = [], isPromotion = false) {
        this.fromPos = fromPos;  // [row, col]
        this.toPos = toPos;      // [row, col]
        this.captures = captures; // Array of [row, col] for captured pieces
        this.isPromotion = isPromotion;
    }

    equals(other) {
        return this.fromPos[0] === other.fromPos[0] &&
               this.fromPos[1] === other.fromPos[1] &&
               this.toPos[0] === other.toPos[0] &&
               this.toPos[1] === other.toPos[1];
    }

    toNotation() {
        const posToNum = (pos) => (pos[0] * 4) + Math.floor(pos[1] / 2) + 1;
        if (this.captures.length > 0) {
            return `${posToNum(this.fromPos)}x${posToNum(this.toPos)}`;
        }
        return `${posToNum(this.fromPos)}-${posToNum(this.toPos)}`;
    }

    // Convert move to index for policy (matches Python implementation)
    toIndex() {
        const fromIdx = this.fromPos[0] * 4 + Math.floor(this.fromPos[1] / 2);
        const toIdx = this.toPos[0] * 4 + Math.floor(this.toPos[1] / 2);
        return fromIdx * 32 + toIdx;
    }
}

/**
 * Checkers Game State
 */
class CheckersGame {
    constructor() {
        this.board = this.createEmptyBoard();
        this.currentPlayer = Piece.BLACK; // Black (AI) moves first
        this.moveHistory = [];
        this.noCaptureCount = 0;
        this.setupBoard();
    }

    createEmptyBoard() {
        return Array(8).fill(null).map(() => Array(8).fill(Piece.EMPTY));
    }

    setupBoard() {
        // Black pieces on top (rows 0-2) - AI
        for (let row = 0; row < 3; row++) {
            for (let col = 0; col < 8; col++) {
                if ((row + col) % 2 === 1) {
                    this.board[row][col] = Piece.BLACK;
                }
            }
        }

        // White pieces on bottom (rows 5-7) - Player
        for (let row = 5; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                if ((row + col) % 2 === 1) {
                    this.board[row][col] = Piece.WHITE;
                }
            }
        }
    }

    copy() {
        const newGame = new CheckersGame();
        newGame.board = this.board.map(row => [...row]);
        newGame.currentPlayer = this.currentPlayer;
        newGame.moveHistory = [...this.moveHistory];
        newGame.noCaptureCount = this.noCaptureCount;
        return newGame;
    }

    isPlayerPiece(piece, player) {
        if (player === Piece.BLACK) {
            return piece === Piece.BLACK || piece === Piece.BLACK_KING;
        }
        return piece === Piece.WHITE || piece === Piece.WHITE_KING;
    }

    isKing(piece) {
        return piece === Piece.BLACK_KING || piece === Piece.WHITE_KING;
    }

    getForwardDirections(piece) {
        if (piece === Piece.BLACK) {
            return [[1, -1], [1, 1]]; // Move down
        } else if (piece === Piece.WHITE) {
            return [[-1, -1], [-1, 1]]; // Move up
        } else { // Kings
            return [[1, -1], [1, 1], [-1, -1], [-1, 1]];
        }
    }

    inBounds(row, col) {
        return row >= 0 && row < 8 && col >= 0 && col < 8;
    }

    wouldPromote(piece, newRow) {
        if (piece === Piece.BLACK && newRow === 7) return true;
        if (piece === Piece.WHITE && newRow === 0) return true;
        return false;
    }

    getSimpleMoves(pos) {
        const [row, col] = pos;
        const piece = this.board[row][col];
        const moves = [];

        for (const [dr, dc] of this.getForwardDirections(piece)) {
            const newRow = row + dr;
            const newCol = col + dc;
            
            if (this.inBounds(newRow, newCol) && this.board[newRow][newCol] === Piece.EMPTY) {
                const isPromotion = this.wouldPromote(piece, newRow);
                moves.push(new Move(pos, [newRow, newCol], [], isPromotion));
            }
        }

        return moves;
    }

    getCapturesFromPos(pos, board, piece, captured) {
        const [row, col] = pos;
        const opponent = this.isPlayerPiece(piece, Piece.BLACK) ? Piece.WHITE : Piece.BLACK;
        const captures = [];

        const directions = this.getForwardDirections(piece);

        for (const [dr, dc] of directions) {
            const midRow = row + dr;
            const midCol = col + dc;
            const endRow = row + 2 * dr;
            const endCol = col + 2 * dc;

            if (!this.inBounds(endRow, endCol)) continue;

            const midKey = `${midRow},${midCol}`;
            if (captured.has(midKey)) continue;

            const midPiece = board[midRow][midCol];
            if (this.isPlayerPiece(midPiece, opponent) && board[endRow][endCol] === Piece.EMPTY) {
                // Found a capture!
                const newCaptured = new Set(captured);
                newCaptured.add(midKey);

                // Check for continuation jumps
                const tempBoard = board.map(r => [...r]);
                tempBoard[row][col] = Piece.EMPTY;
                tempBoard[midRow][midCol] = Piece.EMPTY;
                tempBoard[endRow][endCol] = piece;

                // Check if piece gets promoted
                const isPromotion = this.wouldPromote(piece, endRow);
                let actualPiece = piece;
                if (isPromotion && !this.isKing(piece)) {
                    actualPiece = piece === Piece.BLACK ? Piece.BLACK_KING : Piece.WHITE_KING;
                    tempBoard[endRow][endCol] = actualPiece;
                }

                // Look for continuation jumps (only if not just promoted)
                if (!isPromotion) {
                    const continuations = this.getCapturesFromPos(
                        [endRow, endCol], tempBoard, piece, newCaptured
                    );
                    
                    if (continuations.length > 0) {
                        for (const cont of continuations) {
                            captures.push(new Move(
                                pos, cont.toPos,
                                [[midRow, midCol], ...cont.captures],
                                cont.isPromotion
                            ));
                        }
                    } else {
                        captures.push(new Move(pos, [endRow, endCol], [[midRow, midCol]], isPromotion));
                    }
                } else {
                    captures.push(new Move(pos, [endRow, endCol], [[midRow, midCol]], isPromotion));
                }
            }
        }

        return captures;
    }

    getLegalMoves() {
        const captures = [];
        const simpleMoves = [];

        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = this.board[row][col];
                if (!this.isPlayerPiece(piece, this.currentPlayer)) continue;

                const pos = [row, col];

                // Get captures
                const pieceCaptures = this.getCapturesFromPos(pos, this.board, piece, new Set());
                captures.push(...pieceCaptures);

                // Get simple moves
                const pieceMoves = this.getSimpleMoves(pos);
                simpleMoves.push(...pieceMoves);
            }
        }

        // Mandatory capture: if captures available, must take them
        if (captures.length > 0) {
            return captures;
        }
        return simpleMoves;
    }

    makeMove(move) {
        const [fromRow, fromCol] = move.fromPos;
        const [toRow, toCol] = move.toPos;
        const piece = this.board[fromRow][fromCol];

        // Move the piece
        this.board[fromRow][fromCol] = Piece.EMPTY;
        this.board[toRow][toCol] = piece;

        // Remove captured pieces
        for (const [capRow, capCol] of move.captures) {
            this.board[capRow][capCol] = Piece.EMPTY;
        }

        // Check for promotion
        if (move.isPromotion) {
            if (piece === Piece.BLACK) {
                this.board[toRow][toCol] = Piece.BLACK_KING;
            } else if (piece === Piece.WHITE) {
                this.board[toRow][toCol] = Piece.WHITE_KING;
            }
        }

        // Update game state
        this.moveHistory.push(move);
        if (move.captures.length > 0) {
            this.noCaptureCount = 0;
        } else {
            this.noCaptureCount++;
        }

        // Switch player
        this.currentPlayer = this.currentPlayer === Piece.BLACK ? Piece.WHITE : Piece.BLACK;

        return true;
    }

    getWinner() {
        // Check for draw by no captures
        if (this.noCaptureCount >= 80) {
            return 0; // Draw
        }

        // Count pieces
        let blackCount = 0;
        let whiteCount = 0;
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = this.board[row][col];
                if (piece === Piece.BLACK || piece === Piece.BLACK_KING) blackCount++;
                if (piece === Piece.WHITE || piece === Piece.WHITE_KING) whiteCount++;
            }
        }

        if (blackCount === 0) return Piece.WHITE;
        if (whiteCount === 0) return Piece.BLACK;

        // Check if current player can move
        const legalMoves = this.getLegalMoves();
        if (legalMoves.length === 0) {
            // Current player has no moves - they lose
            return this.currentPlayer === Piece.BLACK ? Piece.WHITE : Piece.BLACK;
        }

        return null; // Game still in progress
    }

    isGameOver() {
        return this.getWinner() !== null;
    }

    countPieces(player) {
        let count = 0;
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                if (this.isPlayerPiece(this.board[row][col], player)) {
                    count++;
                }
            }
        }
        return count;
    }

    // Get board state as tensor for neural network (matches Python format)
    getBoardTensor() {
        const tensor = [];
        
        // Determine piece types based on current player perspective
        let ownNormal, ownKing, oppNormal, oppKing;
        if (this.currentPlayer === Piece.BLACK) {
            ownNormal = Piece.BLACK;
            ownKing = Piece.BLACK_KING;
            oppNormal = Piece.WHITE;
            oppKing = Piece.WHITE_KING;
        } else {
            ownNormal = Piece.WHITE;
            ownKing = Piece.WHITE_KING;
            oppNormal = Piece.BLACK;
            oppKing = Piece.BLACK_KING;
        }

        // Create 4-channel tensor
        for (let row = 0; row < 8; row++) {
            const actualRow = this.currentPlayer === Piece.WHITE ? 7 - row : row;
            for (let col = 0; col < 8; col++) {
                const piece = this.board[actualRow][col];
                tensor.push(piece === ownNormal ? 1 : 0);   // Channel 0: own normal
                tensor.push(piece === ownKing ? 1 : 0);     // Channel 1: own king
                tensor.push(piece === oppNormal ? 1 : 0);   // Channel 2: opponent normal
                tensor.push(piece === oppKing ? 1 : 0);     // Channel 3: opponent king
            }
        }

        return tensor;
    }
}

// Export for use in other modules
window.Piece = Piece;
window.Move = Move;
window.CheckersGame = CheckersGame;
