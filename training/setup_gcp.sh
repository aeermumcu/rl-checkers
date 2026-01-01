#!/bin/bash
# GCP VM Setup Script for Checkers AI Training
# Run this on a fresh GCP VM with T4 GPU

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           🚀 Checkers AI - GCP Training Setup                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -qq

# Install Python and pip
echo "🐍 Installing Python dependencies..."
sudo apt-get install -y -qq python3-pip python3-venv git tmux

# Clone repo (or use existing)
if [ ! -d "rl-checkers" ]; then
    echo "📥 Cloning repository..."
    git clone https://github.com/aeermumcu/rl-checkers.git
fi

cd rl-checkers/training

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Install TensorFlow with GPU support
echo "📦 Installing TensorFlow (GPU)..."
pip install tensorflow[and-cuda] -q

# Install other dependencies
echo "📦 Installing other dependencies..."
pip install numpy tqdm tensorflowjs -q

# Verify GPU
echo
echo "🎮 Checking GPU..."
python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'  GPUs found: {len(gpus)}'); [print(f'    - {g.name}') for g in gpus]"

echo
echo "════════════════════════════════════════════════════════════════"
echo "✅ Setup complete!"
echo
echo "To start training:"
echo "  cd rl-checkers/training"
echo "  source venv/bin/activate"
echo "  python gcp_train.py"
echo
echo "To run in background (recommended):"
echo "  tmux new -s training"
echo "  python gcp_train.py"
echo "  # Press Ctrl+B then D to detach"
echo
echo "To monitor progress (in another terminal):"
echo "  python monitor.py"
echo "════════════════════════════════════════════════════════════════"
