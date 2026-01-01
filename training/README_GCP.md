# GCP Training for Checkers AI

Train a strong Checkers AI using Google Cloud Platform with a T4 GPU.

## Quick Start

### 1. Create a GCP VM

Go to [Google Cloud Console](https://console.cloud.google.com/compute/instances) and create:

- **Machine type:** `n1-standard-4` (4 vCPU, 15 GB RAM)
- **GPU:** 1x NVIDIA T4
- **Boot disk:** Ubuntu 22.04 LTS, 50 GB SSD
- **Cost:** ~$0.35/hour (~$8/day)

### 2. SSH into VM and Run Setup

```bash
# SSH into your VM (from GCP console or gcloud)
gcloud compute ssh YOUR_VM_NAME

# Download and run setup script
curl -sL https://raw.githubusercontent.com/aeermumcu/rl-checkers/main/training/setup_gcp.sh | bash
```

### 3. Start Training

```bash
cd rl-checkers/training
source venv/bin/activate

# Run in tmux so it survives disconnection
tmux new -s training
python gcp_train.py

# Detach: Press Ctrl+B, then D
# Reattach later: tmux attach -t training
```

### 4. Monitor Progress

In another terminal (or tmux pane):
```bash
python monitor.py
```

## Training Milestones

| Difficulty | Games | Est. Time (T4) | Strength |
|------------|-------|----------------|----------|
| easy | 500 | ~15 min | Beginner |
| medium | 2,000 | ~1 hour | Casual |
| hard | 10,000 | ~5 hours | Strong |
| impossible | 50,000 | ~24 hours | Expert |

## Download Trained Models

After training completes:

```bash
# On VM - compress models
cd rl-checkers/training
tar -czvf trained_models.tar.gz trained_models/

# On your local machine - download
gcloud compute scp YOUR_VM:rl-checkers/training/trained_models.tar.gz .
tar -xzvf trained_models.tar.gz
```

## Stop VM When Done!

```bash
# Don't forget to stop the VM to avoid charges
gcloud compute instances stop YOUR_VM_NAME
```

## Files

- `gcp_train.py` - Main training script
- `monitor.py` - Live progress monitor  
- `setup_gcp.sh` - VM setup script
