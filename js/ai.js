/**
 * AI Module - Neural Network + MCTS for Checkers
 * Loads TensorFlow.js model and runs MCTS search
 */

class CheckersAI {
    constructor() {
        this.model = null;
        this.modelLoaded = false;
        this.difficulty = 'medium';

        // MCTS simulation counts for each difficulty
        this.simCounts = {
            easy: 50,
            medium: 200,
            hard: 800,
            impossible: 1600
        };
    }

    async loadModel(modelPath = 'model/tfjs/model.json') {
        try {
            console.log('Loading AI model...');
            this.model = await tf.loadLayersModel(modelPath);
            this.modelLoaded = true;
            console.log('AI model loaded successfully');
            return true;
        } catch (error) {
            console.warn('Could not load neural network model:', error);
            console.log('AI will use random policy with MCTS');
            this.modelLoaded = false;
            return false;
        }
    }

    setDifficulty(level) {
        this.difficulty = level;
        console.log(`AI difficulty set to: ${level} (${this.simCounts[level]} simulations)`);
    }

    // Get neural network prediction
    async predict(boardTensor) {
        if (!this.modelLoaded) {
            // Return uniform policy and neutral value
            return {
                policy: new Array(1024).fill(1 / 1024),
                value: 0
            };
        }

        // Reshape tensor for model input: [1, 8, 8, 4]
        const input = tf.tensor4d(boardTensor, [1, 8, 8, 4]);

        try {
            const [policyTensor, valueTensor] = this.model.predict(input);
            const policy = await policyTensor.data();
            const value = await valueTensor.data();

            // Clean up tensors
            input.dispose();
            policyTensor.dispose();
            valueTensor.dispose();

            return {
                policy: Array.from(policy),
                value: value[0]
            };
        } catch (error) {
            console.error('Prediction error:', error);
            input.dispose();
            return {
                policy: new Array(1024).fill(1 / 1024),
                value: 0
            };
        }
    }

    // Get best move using MCTS
    async getMove(game) {
        const numSimulations = this.simCounts[this.difficulty];
        return await this.runMCTS(game, numSimulations);
    }

    // Monte Carlo Tree Search
    async runMCTS(game, numSimulations) {
        const legalMoves = game.getLegalMoves();

        if (legalMoves.length === 0) {
            return null;
        }

        if (legalMoves.length === 1) {
            return legalMoves[0];
        }

        // Create root node
        const root = new MCTSNode(game.copy(), null, null);

        // Expand root
        await this.expand(root);

        // Add Dirichlet noise for exploration at root
        this.addDirichletNoise(root);

        // Run simulations
        for (let i = 0; i < numSimulations; i++) {
            let node = root;

            // Selection
            while (node.isExpanded() && !node.isTerminal()) {
                node = this.selectChild(node);
            }

            // Evaluation
            let value;
            if (node.isTerminal()) {
                value = this.getTerminalValue(node);
            } else {
                value = await this.expand(node);
            }

            // Backpropagation
            this.backpropagate(node, value);

            // Yield occasionally to keep UI responsive
            if (i % 50 === 0) {
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }

        // Select move based on visit counts
        let bestMove = null;
        let bestVisits = -1;

        for (const [moveIdx, child] of Object.entries(root.children)) {
            if (child.visitCount > bestVisits) {
                bestVisits = child.visitCount;
                bestMove = child.move;
            }
        }

        return bestMove;
    }

    selectChild(node) {
        let bestScore = -Infinity;
        let bestChild = null;

        for (const child of Object.values(node.children)) {
            const score = child.getUCBScore();
            if (score > bestScore) {
                bestScore = score;
                bestChild = child;
            }
        }

        return bestChild;
    }

    async expand(node) {
        const game = node.state;
        const legalMoves = game.getLegalMoves();

        if (legalMoves.length === 0 || game.isGameOver()) {
            return this.getTerminalValue(node);
        }

        // Get neural network predictions
        const boardTensor = this.reshapeTensor(game.getBoardTensor());
        const { policy, value } = await this.predict(boardTensor);

        // Create child nodes
        for (const move of legalMoves) {
            const moveIdx = move.toIndex();

            const newGame = game.copy();
            newGame.makeMove(move);

            const child = new MCTSNode(newGame, node, move);
            child.prior = policy[moveIdx] || 0.001;
            node.children[moveIdx] = child;
        }

        return value;
    }

    reshapeTensor(flatTensor) {
        // Reshape from flat [8*8*4] to [8, 8, 4]
        const tensor = [];
        for (let i = 0; i < 8; i++) {
            tensor.push([]);
            for (let j = 0; j < 8; j++) {
                tensor[i].push([]);
                for (let k = 0; k < 4; k++) {
                    tensor[i][j].push(flatTensor[(i * 8 + j) * 4 + k]);
                }
            }
        }
        return tensor;
    }

    addDirichletNoise(node, alpha = 0.3, weight = 0.25) {
        const children = Object.values(node.children);
        if (children.length === 0) return;

        const noise = this.dirichletNoise(children.length, alpha);

        children.forEach((child, i) => {
            child.prior = (1 - weight) * child.prior + weight * noise[i];
        });
    }

    dirichletNoise(n, alpha) {
        // Simple approximation of Dirichlet distribution
        const samples = [];
        let sum = 0;

        for (let i = 0; i < n; i++) {
            // Gamma distribution approximation
            let sample = 0;
            for (let j = 0; j < Math.ceil(alpha * 10); j++) {
                sample -= Math.log(Math.random());
            }
            samples.push(sample);
            sum += sample;
        }

        return samples.map(s => s / sum);
    }

    getTerminalValue(node) {
        const winner = node.state.getWinner();

        if (winner === 0) return 0; // Draw

        // Value from parent's perspective
        const parentPlayer = node.state.currentPlayer === Piece.BLACK ? Piece.WHITE : Piece.BLACK;
        return winner === parentPlayer ? 1 : -1;
    }

    backpropagate(node, value) {
        while (node !== null) {
            node.visitCount++;
            node.valueSum += value;
            value = -value; // Flip for opponent
            node = node.parent;
        }
    }
}

/**
 * MCTS Node
 */
class MCTSNode {
    constructor(state, parent, move) {
        this.state = state;
        this.parent = parent;
        this.move = move;
        this.children = {};
        this.visitCount = 0;
        this.valueSum = 0;
        this.prior = 0;
    }

    getQValue() {
        if (this.visitCount === 0) return 0;
        return this.valueSum / this.visitCount;
    }

    getUCBScore() {
        if (!this.parent) return 0;

        const cPuct = 2.0;
        const exploration = cPuct * this.prior *
            Math.sqrt(this.parent.visitCount) / (1 + this.visitCount);

        return this.getQValue() + exploration;
    }

    isExpanded() {
        return Object.keys(this.children).length > 0;
    }

    isTerminal() {
        return this.state.isGameOver();
    }
}

// Export
window.CheckersAI = CheckersAI;
