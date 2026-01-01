/**
 * UI Controller for Checkers Game
 * Handles rendering, user interaction, and game flow
 */

class CheckersUI {
    constructor() {
        // Game state
        this.game = new CheckersGame();
        this.ai = new CheckersAI();
        this.selectedPiece = null;
        this.legalMoves = [];
        this.isPlayerTurn = false;
        this.isGameActive = false;
        this.isDragging = false;
        this.dragPiece = null;

        // Statistics
        this.stats = {
            games: parseInt(localStorage.getItem('checkers_games') || '0'),
            wins: parseInt(localStorage.getItem('checkers_wins') || '0')
        };

        // DOM elements
        this.boardEl = document.getElementById('game-board');
        this.moveListEl = document.getElementById('move-list');
        this.aiThinkingEl = document.getElementById('ai-thinking');
        this.yourTurnEl = document.getElementById('your-turn');
        this.aiPiecesEl = document.getElementById('ai-pieces');
        this.humanPiecesEl = document.getElementById('human-pieces');
        this.loadingOverlay = document.getElementById('loading-overlay');
        this.gameOverModal = document.getElementById('game-over-modal');

        // Initialize
        this.init();
    }

    async init() {
        this.setupBoard();
        this.setupEventListeners();
        this.loadTheme();
        this.updateStats();

        // Load AI model
        await this.ai.loadModel();

        // Hide loading overlay
        this.loadingOverlay.classList.add('hidden');

        // Start new game
        this.newGame();
    }

    setupBoard() {
        this.boardEl.innerHTML = '';

        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const square = document.createElement('div');
                square.className = `square ${(row + col) % 2 === 0 ? 'light' : 'dark'}`;
                square.dataset.row = row;
                square.dataset.col = col;
                this.boardEl.appendChild(square);
            }
        }
    }

    setupEventListeners() {
        // Board click/drag events
        this.boardEl.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.boardEl.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.boardEl.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.boardEl.addEventListener('mouseleave', () => this.cancelDrag());

        // Touch support
        this.boardEl.addEventListener('touchstart', (e) => this.handleTouchStart(e));
        this.boardEl.addEventListener('touchmove', (e) => this.handleTouchMove(e));
        this.boardEl.addEventListener('touchend', (e) => this.handleTouchEnd(e));

        // Control buttons
        document.getElementById('new-game-btn').addEventListener('click', () => this.newGame());
        document.getElementById('undo-btn').addEventListener('click', () => this.undoMove());
        document.getElementById('hint-btn').addEventListener('click', () => this.showHint());
        document.getElementById('play-again-btn').addEventListener('click', () => {
            this.hideModal();
            this.newGame();
        });
        document.getElementById('change-difficulty-btn').addEventListener('click', () => {
            this.hideModal();
        });

        // Difficulty buttons
        document.querySelectorAll('.diff-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.diff-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.ai.setDifficulty(btn.dataset.level);
            });
        });

        // Theme toggle
        document.getElementById('theme-toggle').addEventListener('click', () => this.toggleTheme());

        // Sound toggle
        document.getElementById('sound-toggle').addEventListener('click', () => this.toggleSound());
    }

    newGame() {
        this.game = new CheckersGame();
        this.selectedPiece = null;
        this.legalMoves = [];
        this.isGameActive = true;
        this.moveListEl.innerHTML = '<div class="move-placeholder">Game moves will appear here...</div>';

        this.renderBoard();
        this.updatePieceCounts();

        // AI plays first (black)
        this.isPlayerTurn = false;
        this.updateTurnIndicator();

        // Give a small delay then let AI move
        setTimeout(() => this.aiMove(), 500);
    }

    renderBoard() {
        const squares = this.boardEl.querySelectorAll('.square');

        squares.forEach(square => {
            const row = parseInt(square.dataset.row);
            const col = parseInt(square.dataset.col);
            const piece = this.game.board[row][col];

            // Clear existing content
            square.innerHTML = '';
            square.classList.remove('highlight', 'hint', 'capture-hint', 'last-move');

            if (piece !== Piece.EMPTY) {
                const pieceEl = document.createElement('div');
                pieceEl.className = 'piece';

                if (piece === Piece.BLACK || piece === Piece.BLACK_KING) {
                    pieceEl.classList.add('black');
                } else {
                    pieceEl.classList.add('white');
                }

                if (piece === Piece.BLACK_KING || piece === Piece.WHITE_KING) {
                    pieceEl.classList.add('king');
                }

                pieceEl.dataset.row = row;
                pieceEl.dataset.col = col;

                square.appendChild(pieceEl);
            }
        });

        // Highlight last move
        if (this.game.moveHistory.length > 0) {
            const lastMove = this.game.moveHistory[this.game.moveHistory.length - 1];
            this.getSquare(lastMove.fromPos[0], lastMove.fromPos[1]).classList.add('last-move');
            this.getSquare(lastMove.toPos[0], lastMove.toPos[1]).classList.add('last-move');
        }
    }

    getSquare(row, col) {
        return this.boardEl.querySelector(`.square[data-row="${row}"][data-col="${col}"]`);
    }

    handleMouseDown(e) {
        console.log('handleMouseDown called', {
            target: e.target,
            isPlayerTurn: this.isPlayerTurn,
            isGameActive: this.isGameActive
        });

        if (!this.isPlayerTurn || !this.isGameActive) {
            console.log('Not player turn or game not active');
            return;
        }

        const piece = e.target.closest('.piece');
        const square = e.target.closest('.square');

        console.log('Found piece and square:', { piece, square });

        // Check if clicking on a hint square (for click-to-move)
        if (square && square.classList.contains('hint') && this.selectedPiece) {
            const row = parseInt(square.dataset.row);
            const col = parseInt(square.dataset.col);
            this.makePlayerMove([row, col]);
            return;
        }

        if (!piece) {
            // Clicked on empty square - clear selection
            this.clearSelection();
            return;
        }

        const row = parseInt(piece.dataset.row);
        const col = parseInt(piece.dataset.col);
        const pieceType = this.game.board[row][col];

        // Only allow selecting player's pieces (white)
        if (!this.game.isPlayerPiece(pieceType, Piece.WHITE)) {
            return;
        }

        // If clicking on already selected piece, just toggle
        if (this.selectedPiece && this.selectedPiece[0] === row && this.selectedPiece[1] === col) {
            this.clearSelection();
            return;
        }

        // Check if this piece has any legal moves
        const pieceMoves = this.game.getLegalMoves().filter(
            m => m.fromPos[0] === row && m.fromPos[1] === col
        );

        if (pieceMoves.length === 0) {
            return;
        }

        // Clear previous selection
        this.clearSelection();

        // Select piece (don't start dragging yet - wait for mouse move)
        this.dragPiece = piece;
        this.selectedPiece = [row, col];
        this.legalMoves = pieceMoves;
        this.dragStartPos = { x: e.clientX, y: e.clientY };
        this.isDragging = false; // Will be set to true on mouse move

        piece.classList.add('selected');
        this.showLegalMoves();
    }

    handleMouseMove(e) {
        if (!this.dragPiece || !this.dragStartPos) return;

        // Check if we should start dragging (moved more than 5 pixels)
        const dx = e.clientX - this.dragStartPos.x;
        const dy = e.clientY - this.dragStartPos.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (!this.isDragging && distance > 5) {
            // Start dragging
            this.isDragging = true;
            this.dragPiece.classList.add('dragging');
        }

        if (this.isDragging) {
            this.updateDragPosition(e.clientX, e.clientY);
        }
    }

    handleMouseUp(e) {
        const square = document.elementFromPoint(e.clientX, e.clientY)?.closest('.square');

        // Check if clicking on a hint square (for click-to-move or drag-to-move)
        if (square && square.classList.contains('hint') && this.selectedPiece) {
            const row = parseInt(square.dataset.row);
            const col = parseInt(square.dataset.col);
            this.makePlayerMove([row, col]);
            return;
        }

        // If we were dragging but didn't drop on a hint, cancel the drag but keep selection
        if (this.isDragging) {
            this.cancelDrag(true); // Keep selection for click-to-move
        }

        // Reset drag start position
        this.dragStartPos = null;
    }

    cancelDrag(keepSelection = false) {
        if (this.dragPiece) {
            this.dragPiece.classList.remove('dragging');
            this.dragPiece.style.position = '';
            this.dragPiece.style.left = '';
            this.dragPiece.style.top = '';
            this.dragPiece.style.transform = '';
        }

        this.isDragging = false;
        this.dragPiece = null;

        if (!keepSelection) {
            this.clearSelection();
        }
    }

    updateDragPosition(clientX, clientY) {
        if (!this.dragPiece) return;

        const boardRect = this.boardEl.getBoundingClientRect();
        const pieceSize = boardRect.width / 8 * 0.8;

        this.dragPiece.style.position = 'fixed';
        this.dragPiece.style.left = `${clientX - pieceSize / 2}px`;
        this.dragPiece.style.top = `${clientY - pieceSize / 2}px`;
        this.dragPiece.style.width = `${pieceSize}px`;
        this.dragPiece.style.height = `${pieceSize}px`;
        this.dragPiece.style.zIndex = '1000';
    }

    // Touch event handlers
    handleTouchStart(e) {
        e.preventDefault();
        const touch = e.touches[0];
        this.handleMouseDown({
            target: document.elementFromPoint(touch.clientX, touch.clientY),
            clientX: touch.clientX,
            clientY: touch.clientY
        });
    }

    handleTouchMove(e) {
        e.preventDefault();
        const touch = e.touches[0];
        this.handleMouseMove({ clientX: touch.clientX, clientY: touch.clientY });
    }

    handleTouchEnd(e) {
        e.preventDefault();
        const touch = e.changedTouches[0];
        this.handleMouseUp({ clientX: touch.clientX, clientY: touch.clientY });
    }

    showLegalMoves() {
        for (const move of this.legalMoves) {
            const square = this.getSquare(move.toPos[0], move.toPos[1]);
            square.classList.add('hint');
            if (move.captures.length > 0) {
                square.classList.add('capture-hint');
            }
        }

        // Highlight selected piece's square
        if (this.selectedPiece) {
            this.getSquare(this.selectedPiece[0], this.selectedPiece[1]).classList.add('highlight');
        }
    }

    clearSelection() {
        this.selectedPiece = null;
        this.legalMoves = [];

        this.boardEl.querySelectorAll('.piece.selected').forEach(p => p.classList.remove('selected'));
        this.boardEl.querySelectorAll('.square.hint, .square.capture-hint, .square.highlight').forEach(s => {
            s.classList.remove('hint', 'capture-hint', 'highlight');
        });
    }

    makePlayerMove(toPos) {
        const move = this.legalMoves.find(m =>
            m.toPos[0] === toPos[0] && m.toPos[1] === toPos[1]
        );

        if (!move) return;

        this.cancelDrag();
        this.animateMove(move, () => {
            this.game.makeMove(move);
            this.addMoveToHistory(move, 'white');
            this.renderBoard();
            this.updatePieceCounts();
            this.clearSelection();

            // Check for game over
            if (this.game.isGameOver()) {
                this.handleGameOver();
                return;
            }

            // AI's turn
            this.isPlayerTurn = false;
            this.updateTurnIndicator();
            setTimeout(() => this.aiMove(), 300);
        });
    }

    async aiMove() {
        if (!this.isGameActive) return;

        // Show thinking indicator
        this.aiThinkingEl.classList.add('active');
        document.querySelector('.ai-player').classList.add('thinking');

        try {
            const move = await this.ai.getMove(this.game);

            if (!move) {
                // AI has no moves
                this.handleGameOver();
                return;
            }

            // Hide thinking indicator
            this.aiThinkingEl.classList.remove('active');
            document.querySelector('.ai-player').classList.remove('thinking');

            // Animate and execute move
            this.animateMove(move, () => {
                this.game.makeMove(move);
                this.addMoveToHistory(move, 'black');
                this.renderBoard();
                this.updatePieceCounts();

                // Check for game over
                if (this.game.isGameOver()) {
                    this.handleGameOver();
                    return;
                }

                // Player's turn
                this.isPlayerTurn = true;
                this.updateTurnIndicator();
            });

        } catch (error) {
            console.error('AI move error:', error);
            this.aiThinkingEl.classList.remove('active');
            document.querySelector('.ai-player').classList.remove('thinking');
        }
    }

    animateMove(move, callback) {
        const fromSquare = this.getSquare(move.fromPos[0], move.fromPos[1]);
        const toSquare = this.getSquare(move.toPos[0], move.toPos[1]);
        const piece = fromSquare.querySelector('.piece');

        if (!piece) {
            callback();
            return;
        }

        const fromRect = fromSquare.getBoundingClientRect();
        const toRect = toSquare.getBoundingClientRect();

        // Calculate movement
        const dx = toRect.left - fromRect.left;
        const dy = toRect.top - fromRect.top;

        piece.classList.add('moving');
        piece.style.transform = `translate(${dx}px, ${dy}px)`;

        // Animate captured pieces
        for (const [capRow, capCol] of move.captures) {
            const capSquare = this.getSquare(capRow, capCol);
            const capPiece = capSquare.querySelector('.piece');
            if (capPiece) {
                setTimeout(() => capPiece.classList.add('captured'), 150);
            }
        }

        setTimeout(() => {
            piece.classList.remove('moving');
            piece.style.transform = '';

            // Handle promotion animation
            if (move.isPromotion) {
                setTimeout(() => {
                    const newPiece = toSquare.querySelector('.piece');
                    if (newPiece) {
                        newPiece.classList.add('promoting', 'just-crowned');
                        setTimeout(() => newPiece.classList.remove('promoting'), 600);
                    }
                }, 50);
            }

            callback();
        }, 300);
    }

    addMoveToHistory(move, player) {
        // Clear placeholder
        const placeholder = this.moveListEl.querySelector('.move-placeholder');
        if (placeholder) placeholder.remove();

        const moveNumber = Math.ceil(this.game.moveHistory.length / 2);

        if (player === 'black') {
            // New entry
            const entry = document.createElement('div');
            entry.className = 'move-entry';
            entry.innerHTML = `
                <span class="move-number">${moveNumber}.</span>
                <span class="move-black">${move.toNotation()}</span>
                <span class="move-white"></span>
            `;
            this.moveListEl.appendChild(entry);
        } else {
            // Add to existing entry
            const lastEntry = this.moveListEl.querySelector('.move-entry:last-child');
            if (lastEntry) {
                lastEntry.querySelector('.move-white').textContent = move.toNotation();
            }
        }

        // Scroll to bottom
        this.moveListEl.scrollTop = this.moveListEl.scrollHeight;
    }

    updateTurnIndicator() {
        if (this.isPlayerTurn) {
            this.yourTurnEl.classList.add('active');
        } else {
            this.yourTurnEl.classList.remove('active');
        }
    }

    updatePieceCounts() {
        const aiPieces = this.game.countPieces(Piece.BLACK);
        const humanPieces = this.game.countPieces(Piece.WHITE);

        this.aiPiecesEl.textContent = `${aiPieces} piece${aiPieces !== 1 ? 's' : ''}`;
        this.humanPiecesEl.textContent = `${humanPieces} piece${humanPieces !== 1 ? 's' : ''}`;
    }

    handleGameOver() {
        this.isGameActive = false;
        const winner = this.game.getWinner();

        const modalIcon = document.getElementById('modal-icon');
        const modalTitle = document.getElementById('modal-title');
        const modalMessage = document.getElementById('modal-message');

        this.stats.games++;

        if (winner === Piece.WHITE) {
            // Player wins
            modalIcon.textContent = '🏆';
            modalIcon.className = 'modal-icon victory';
            modalTitle.textContent = 'Victory!';
            modalMessage.textContent = 'Congratulations! You defeated the AI!';
            this.stats.wins++;
            this.spawnConfetti();
        } else if (winner === Piece.BLACK) {
            // AI wins
            modalIcon.textContent = '😔';
            modalIcon.className = 'modal-icon defeat';
            modalTitle.textContent = 'Defeat';
            modalMessage.textContent = 'The AI won this time. Try again!';
        } else {
            // Draw
            modalIcon.textContent = '🤝';
            modalIcon.className = 'modal-icon';
            modalTitle.textContent = 'Draw';
            modalMessage.textContent = 'The game ended in a draw.';
        }

        // Save stats
        localStorage.setItem('checkers_games', this.stats.games.toString());
        localStorage.setItem('checkers_wins', this.stats.wins.toString());
        this.updateStats();

        // Show modal
        this.gameOverModal.classList.add('active');
    }

    hideModal() {
        this.gameOverModal.classList.remove('active');
    }

    updateStats() {
        document.getElementById('stat-games').textContent = this.stats.games;
        document.getElementById('stat-wins').textContent = this.stats.wins;
        const rate = this.stats.games > 0 ? Math.round(this.stats.wins / this.stats.games * 100) : 0;
        document.getElementById('stat-rate').textContent = `${rate}%`;
    }

    undoMove() {
        // TODO: Implement undo (would need to track more state)
        console.log('Undo not implemented yet');
    }

    showHint() {
        if (!this.isPlayerTurn || !this.isGameActive) return;

        const moves = this.game.getLegalMoves();
        if (moves.length > 0) {
            // Clear any existing selection first
            this.clearSelection();

            // Pick a random move to suggest
            const move = moves[Math.floor(Math.random() * moves.length)];

            // Get all moves for this piece so player can choose any of them
            const pieceMoves = moves.filter(
                m => m.fromPos[0] === move.fromPos[0] && m.fromPos[1] === move.fromPos[1]
            );

            // Set selection state so clicking hints will work
            this.selectedPiece = [move.fromPos[0], move.fromPos[1]];
            this.legalMoves = pieceMoves;

            // Highlight the piece and show all its legal moves
            const fromSquare = this.getSquare(move.fromPos[0], move.fromPos[1]);
            const piece = fromSquare.querySelector('.piece');

            if (piece) {
                piece.classList.add('selected');
            }
            fromSquare.classList.add('highlight');

            // Show all legal moves for this piece
            this.showLegalMoves();
        }
    }

    toggleTheme() {
        const body = document.body;
        const isDark = body.dataset.theme !== 'light';
        body.dataset.theme = isDark ? 'light' : 'dark';

        const themeIcon = document.querySelector('.theme-icon');
        themeIcon.textContent = isDark ? '☀️' : '🌙';

        localStorage.setItem('checkers_theme', body.dataset.theme);
    }

    loadTheme() {
        const savedTheme = localStorage.getItem('checkers_theme') || 'dark';
        document.body.dataset.theme = savedTheme;

        const themeIcon = document.querySelector('.theme-icon');
        themeIcon.textContent = savedTheme === 'light' ? '☀️' : '🌙';
    }

    toggleSound() {
        // TODO: Implement sound effects
        const soundIcon = document.querySelector('.sound-icon');
        const isMuted = soundIcon.textContent === '🔇';
        soundIcon.textContent = isMuted ? '🔊' : '🔇';
    }

    spawnConfetti() {
        const colors = ['#6366f1', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444'];

        for (let i = 0; i < 50; i++) {
            setTimeout(() => {
                const confetti = document.createElement('div');
                confetti.className = 'confetti';
                confetti.style.left = `${Math.random() * 100}vw`;
                confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
                confetti.style.animationDuration = `${2 + Math.random() * 2}s`;
                document.body.appendChild(confetti);

                setTimeout(() => confetti.remove(), 4000);
            }, i * 50);
        }
    }
}

// Initialize game when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.gameUI = new CheckersUI();
});
