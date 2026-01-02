import os
import time
import numpy as np
import multiprocessing as mp
import logging
from datetime import datetime
import traceback
import queue  # Standard queue for Empty exception

# Set fork start method for compatibility (needed for TensorFlow on Linux)
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Import game logic
from checkers_game import CheckersGame, Move, Piece

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('MP_TRAIN')

@dataclass
class MPNode:
    """MCTS Node for Multiprocessing."""
    state: CheckersGame
    parent: Optional['MPNode'] = None
    move: Optional[Move] = None
    children: Dict[int, 'MPNode'] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0
        
    @property
    def ucb_score(self) -> float:
        if not self.parent: return 0.0
        c_puct = 2.0
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + exploration


class QueueMCTS:
    """MCTS that requests predictions via Queue."""
    def __init__(self, game: CheckersGame, worker_id: int, pred_queue: mp.Queue, result_queue: mp.Queue):
        self.root = MPNode(state=game.copy())
        self.worker_id = worker_id
        self.pred_queue = pred_queue
        self.result_queue = result_queue
        
    def run_simulation(self):
        node = self.root
        path = [node]
        
        # Selection
        while node.children and not node.state.is_game_over():
            # Select best child
            best_score = -float('inf')
            best_child = None
            for child in node.children.values():
                score = child.ucb_score
                if score > best_score:
                    best_score = score
                    best_child = child
            
            if best_child:
                node = best_child
                path.append(node)
            else:
                break
        
        # Expansion & Evaluation
        value = 0.0
        if not node.state.is_game_over():
            # Request Prediction
            self.pred_queue.put((self.worker_id, [node.state.get_board_tensor()]))
            policy, v = self.result_queue.get()
            value = float(v[0])
            policy = policy[0]
            
            # Expand
            legal_moves = node.state.get_legal_moves()
            for move in legal_moves:
                move_idx = node.state.move_to_index(move)
                new_state = node.state.copy()
                new_state.make_move(move)
                
                child = MPNode(
                    state=new_state,
                    parent=node,
                    move=move,
                    prior=policy[move_idx]
                )
                node.children[move_idx] = child
                
            # Add noise if root (simplified: do it only at specific step if needed, or always for exploration)
            if node == self.root:
                 noise = np.random.dirichlet([0.3] * len(node.children))
                 for i, child in enumerate(node.children.values()):
                     child.prior = 0.75 * child.prior + 0.25 * noise[i]
        else:
            # Terminal
            winner = node.state.get_winner()
            if winner == 0:
                value = 0.0
            else:
                # Value for parent (player who moved to get here)
                parent_player = 1 if node.state.current_player == 2 else 2
                value = 1.0 if winner == parent_player else -1.0
        
        # Backprop
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

    def get_policy(self, temp=1.0):
        legal_moves = self.root.state.get_legal_moves()
        policy = np.zeros(1024, dtype=np.float32)
        
        probs = []
        for move in legal_moves:
            idx = self.root.state.move_to_index(move)
            if idx in self.root.children:
                probs.append(self.root.children[idx].visit_count)
            else:
                probs.append(0)
        
        probs = np.array(probs)
        if probs.sum() == 0:
             return policy # Should not happen usually
             
        if temp == 0:
            best_idx = np.argmax(probs)
            probs = np.zeros_like(probs)
            probs[best_idx] = 1.0
        else:
            # Log space temp scaling
             log_probs = np.log(probs + 1e-10) / temp
             log_probs -= log_probs.max()
             probs = np.exp(log_probs)
             probs /= probs.sum()
             
        for i, move in enumerate(legal_moves):
            idx = self.root.state.move_to_index(move)
            policy[idx] = probs[i]
            
        return policy

def worker_process(worker_id, pred_queue, result_queue, data_queue, config):
    """
    Worker process running independent MCTS games.
    """
    try:
        np.random.seed(int(time.time()) + worker_id * 1000)
        logger.info(f"Worker {worker_id} started")
        
        mcts_sims = config['mcts_simulations']
        
        while True:
            # Check stop
            try:
                msg = result_queue.get_nowait()
                if msg == 'STOP': break
            except queue.Empty:
                pass
            
            game = CheckersGame()
            mcts = QueueMCTS(game, worker_id, pred_queue, result_queue)
            
            history = []
            move_count = 0
            
            while not game.is_game_over() and move_count < 200:
                # Run MCTS simulations
                # First expansion (root)
                if not mcts.root.children:
                     mcts.run_simulation() # Will expand root
                     
                for _ in range(mcts_sims):
                    mcts.run_simulation()
                
                # Select move
                temp = 1.0 if move_count < 30 else 0.1
                full_policy = mcts.get_policy(temp)
                
                # Sample move from policy
                legal_moves = game.get_legal_moves()
                if not legal_moves: break
                
                # Extract probs for legal moves only for sampling
                move_probs = []
                for move in legal_moves:
                    idx = game.move_to_index(move)
                    move_probs.append(full_policy[idx])
                
                move_probs = np.array(move_probs)
                move_probs /= move_probs.sum()
                
                if config.get('argmax_move', False):
                    move_idx = np.argmax(move_probs)
                else:
                    move_idx = np.random.choice(len(legal_moves), p=move_probs)
                    
                action = legal_moves[move_idx]
                
                # Save history
                history.append((game.get_board_tensor(), full_policy, game.current_player))
                
                # Make move
                game.make_move(action)
                move_count += 1
                
                # Reuse tree
                action_idx = game.move_to_index(action)
                if action_idx in mcts.root.children:
                    mcts.root = mcts.root.children[action_idx]
                    mcts.root.parent = None
                else:
                    mcts = QueueMCTS(game, worker_id, pred_queue, result_queue)

            # Game over
            winner = game.get_winner()
            
            examples = []
            for state, pi, player in history:
                v = 0.0
                if winner:
                   v = 1.0 if winner == player else -1.0
                examples.append((state, pi, v))
            
            data_queue.put(examples)
            
    except Exception as e:
        logger.error(f"Worker {worker_id} crashed: {e}")
        traceback.print_exc()
            
    except Exception as e:
        logger.error(f"Worker {worker_id} crashed: {e}")
        traceback.print_exc()

def inference_server(model_path, num_workers, pred_queue, result_queues, stop_event):
    """
    Main process running the GPU model.
    """
    import tensorflow as tf
    from model import CheckersNetwork
    
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Inference Server found {len(gpus)} GPUs")
        except RuntimeError as e:
            logger.error(e)
            
    # Load model
    network = CheckersNetwork()
    if model_path and os.path.exists(model_path):
        logger.info(f"Loading model weights from {model_path}")
        network.load(model_path)
    else:
        logger.info("Initialized new random model")
        
    # Compiled predict function
    @tf.function(reduce_retracing=True)
    def predict_batch(states):
        return network.model(states, training=False)
        
    # Warmup
    dummy = np.random.randn(1, 8, 8, 4).astype(np.float32)
    _ = predict_batch(dummy)
    logger.info("Model warmed up!")
    
    # Main loop
    batch_states = []
    batch_indices = [] # (worker_id, req_id) - we assume 1 req per worker at a time for simplicity
    
    # We poll workers round-robin or FIFO
    while not stop_event.is_set():
        # Try to collect a batch
        start_wait = time.time()
        
        while len(batch_states) < 16 and (time.time() - start_wait < 0.005): # Max 5ms wait or 16 items
            try:
                # Get request: (worker_id, states_list)
                worker_id, states = pred_queue.get_nowait()
                
                # Currently handling single state requests per worker for simplicity in MCTS loop
                # The worker sends 1 state at a time in current implementation
                batch_states.append(states[0])
                batch_indices.append(worker_id)
                
            except queue.Empty:
                if batch_states:
                    break # Stop waiting if we have some data
                time.sleep(0.0001) # sleep tiny bit to yield CPU
        
        if batch_states:
            # Run inference
            np_states = np.array(batch_states, dtype=np.float32)
            policy, value = predict_batch(np_states)
            
            p_numpy = policy.numpy()
            v_numpy = value.numpy()
            
            # Dispatch results directly to worker queues
            for i, worker_id in enumerate(batch_indices):
                # Send result back: (policy, value)
                result_queues[worker_id].put((
                    p_numpy[i:i+1],
                    v_numpy[i:i+1]
                ))
            
            batch_states = []
            batch_indices = []

def run_mp_training(config, resume_path=None):
    """
    Entry point for multiprocessing training.
    """
    mp.set_start_method('spawn', force=True)
    
    num_workers = 3 # Leave 1 cpu for main/inference
    
    # Queues
    pred_queue = mp.Queue()
    data_queue = mp.Queue()
    result_queues = [mp.Queue() for _ in range(num_workers)]
    stop_event = mp.Event()
    
    # Start Inference Server (in THIS process or separate?) 
    # Better to run Inference in THIS process to avoid pickling the model or dealing with CUDA context issues in separate processes easily.
    # But then who runs the Training loop?
    
    # Architecture:
    # Process 1 (Main): Inference Server + Training Loop (interleaved? No, training blocks inference)
    # Actually, training blocks inference. If we train on main process, inference stops.
    # Better:
    # Process 1 (Main): Inference Loop.
    # Process 2 (Trainer): Consumes data, builds buffer, trains periodically.
    # Process 3,4,5: MCTS Workers.
    
    # Simplification:
    # Interleave Inference and Training in Main Process.
    # While buffer < batch_size: Run Inference.
    # Once enough data: Train one step. (Inference pauses for ~100ms). This is acceptable.
    
    workers = []
    for i in range(num_workers):
        p = mp.Process(target=worker_process, 
                       args=(i, pred_queue, result_queues[i], data_queue, config),
                       name=f"Worker-{i}")
        p.start()
        workers.append(p)
        
    logger.info(f"Started {num_workers} workers")
    
    # Local Imports
    import tensorflow as tf
    from model import CheckersNetwork
    from trainer import ReplayBuffer, TrainingExample
    import json
    
    # Load Model
    network = CheckersNetwork()
    if resume_path and os.path.exists(resume_path):
        network.load(resume_path)
        logger.info(f"Loaded weights from {resume_path}")
        
    buffer = ReplayBuffer(200000)
    
    # Warmup
    dummy = np.random.randn(32, 8, 8, 4).astype(np.float32)
    network.predict_batch(dummy)
    
    # Status tracking
    total_games = 0
    start_time = time.time()
    
    # Training Loop (Interleaved Inference)
    try:
        while total_games < config['total_games']:
             
            # 1. Run inference loop for X seconds or until Y samples collected
            # We must service queues continuously. 
            # We can check data_queue periodically.
            
            # Run inference for a "tick"
            start_tick = time.time()
            max_tick = 5.0 # Train every 5 seconds?
            
            new_examples = []
            
            while time.time() - start_tick < max_tick:
                # SERVICE INFERENCE
                batch_states = []
                batch_indices = []
                
                # Drain queue up to max batch size
                for _ in range(64): 
                    try:
                        w_id, s = pred_queue.get_nowait()
                        batch_states.append(s[0])
                        batch_indices.append(w_id)
                    except queue.Empty:
                        break
                
                if batch_states:
                    pol, val = network.predict_batch(np.array(batch_states, dtype=np.float32))
                    p_np, v_np = pol.numpy(), val.numpy()
                    for k, wid in enumerate(batch_indices):
                        result_queues[wid].put((p_np[k:k+1], v_np[k:k+1]))
                
                # CHECK FOR NEW DATA
                while not data_queue.empty():
                    try:
                        ex = data_queue.get_nowait()
                        new_examples.extend(ex)
                    except queue.Empty:
                        break
                        
                # If we have lots of new data, maybe break early to train?
                if len(new_examples) > 100:
                    break
                    
                if not batch_states:
                    time.sleep(0.001)
            
            # 2. Add to buffer
            if new_examples:
                examples_obj = [TrainingExample(s, p, v) for s,p,v in new_examples]
                buffer.add(examples_obj)
                
                prev_total = total_games
                total_games += len(new_examples)
                logger.info(f"Collected {len(new_examples)} games. Total: {total_games}")
                
                # Checkpoints
                # Regular checkpoint
                if total_games // config['checkpoint_every'] > prev_total // config['checkpoint_every']:
                     path = f"checkpoints/checkpoint_{total_games}.weights.h5"
                     network.save(path)
                     network.save("checkpoints/latest.weights.h5")
                
                # Difficulty Checkpoints
                # We need to check if we crossed any threshold
                diff_checkpoints = config.get('difficulty_checkpoints', {})
                for diff_name, threshold in diff_checkpoints.items():
                    if prev_total < threshold <= total_games:
                        logger.info(f"🏆 Reached {diff_name} difficulty ({threshold} games)!")
                        # Save model
                        model_path = f"checkpoints/checkers_model_{diff_name}.weights.h5"
                        network.save(model_path)
                        
                        # Also export to TFJS if possible
                        try:
                            import tensorflowjs as tfjs
                            tfjs_path = f"checkpoints/tfjs_model_{diff_name}"
                            network.export_tfjs(tfjs_path)
                            logger.info(f"Exported TFJS model to {tfjs_path}")
                        except ImportError:
                            logger.warning("tensorflowjs not installed, skipping export")

                # Update status file
                status = {
                     "total_games": total_games,
                     "buffer_size": len(buffer),
                     "timestamp": datetime.now().isoformat(),
                     "target_games": config['total_games'],
                     "completed": False,
                     "elapsed_seconds": time.time() - start_time,
                     "eta_seconds": (config['total_games'] - total_games) / (total_games / (time.time() - start_time + 1))
                }
                with open("checkpoints/gcp_training_status.json", "w") as f:
                     json.dump(status, f)

            # 3. Train
            if len(buffer) > config['batch_size']:
                # Train on a few batches
                losses = []
                for _ in range(10): # Train 10 steps per cycle
                    batch = buffer.sample(config['batch_size'])
                    states = np.array([b.state for b in batch])
                    policies = np.array([b.policy for b in batch])
                    values = np.array([b.value for b in batch])
                    
                    metrics = network.model.train_on_batch(
                        states, [policies, values], return_dict=True
                    )
                    losses.append(metrics['loss'])
                
                logger.info(f"Training Loss: {np.mean(losses):.4f}")

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        stop_event.set()
        for p in workers:
            p.terminate()
            p.join()

if __name__ == "__main__":
    # Config matching gcp_train.py
    CONFIG = {
        'mcts_simulations': 100, 
        'total_games': 25000,
        'batch_size': 256,
        'checkpoint_every': 500
    }
    
    # Ensure checkpoint dir
    os.makedirs("checkpoints", exist_ok=True)
    
    run_mp_training(CONFIG, resume_path="checkpoints/latest.weights.h5")
