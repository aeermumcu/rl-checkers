# AI-Powered Checkers Game

## Planning Phase
- [x] Create implementation plan
- [x] Get user approval on plan

## Phase 1: Core Game Engine
- [x] Implement checkers game logic in Python
- [x] Implement checkers game logic in JavaScript (browser)
- [x] Add move validation, captures, king promotion
- [x] Add win/loss detection

## Phase 2: Neural Network Architecture
- [x] Design CNN architecture for board evaluation
- [x] Implement in TensorFlow/Keras (Python)
- [x] Create TensorFlow.js compatible model export

## Phase 3: Training Pipeline
- [x] Implement self-play training loop
- [x] Add experience replay buffer
- [x] Implement opponent pool for diversity
- [x] Set up training infrastructure
- [x] Create `mp_train.py` for multiprocessing
    - [x] Implement `worker_process` for CPU-bound MCTS
    - [x] Implement `inference_server` for GPU-bound predictions
    - [x] Use `multiprocessing.Queue` for communication
    - [x] Fix `AttributeError` with custom `QueueMCTS`
    - [x] Optimize with 32 workers to saturate CPU
- [x] Update `gcp_train.py` to use `mp_train.py` logic
- [x] Deploy and Verify
    - [x] consistent ~400% CPU usage (load avg > 4)
    - [x] robustness against deadlocks

## Phase 3.5: Scaling Hardware (16-Core Upgrade)
- [x] Upgrade VM to `g2-standard-16`
    - [x] Stop VM
    - [x] Resize
    - [x] Start VM
- [x] Optimize Code for 16 Cores
    - [x] Increase `num_parallel_games` to 96
    - [x] Increase `total_games` target to 50,000
- [x] Restart Training
    - [x] Verify 1600% CPU usage
    - [x] Verify GPU utilization increase (>20%)

## Phase 4: Browser Game UI
- [x] Create beautiful game board with dark/light themes
- [x] Implement piece animations (move, capture, promotion)
- [x] Add drag-and-drop + click-to-move controls
- [x] Show legal move hints
- [x] Add move history/game log
- [x] Implement difficulty selection

## Phase 5: Integration & Testing
- [/] Load trained model in browser
- [ ] Implement difficulty levels (limit AI search)
- [ ] Test and refine gameplay
- [ ] Create deployment package
