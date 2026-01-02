#!/bin/bash
# Check GCP Training Progress
# Run this locally to see training status on the VM

VM_NAME="rl-training-2-vm"
ZONE="us-east4-c"

echo
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║            🎮 GCP Training Status                              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo

# Check if VM is running
VM_STATUS=$(gcloud compute instances describe $VM_NAME --zone=$ZONE --format="value(status)" 2>/dev/null)

if [ "$VM_STATUS" != "RUNNING" ]; then
    echo "  ⚠️  VM is $VM_STATUS (not running)"
    echo
    echo "  To start: gcloud compute instances start $VM_NAME --zone=$ZONE"
    exit 1
fi

echo "  🟢 VM: $VM_NAME ($ZONE)"
echo

# Get training status from VM (using IAP tunnel for reliability)
gcloud compute ssh $VM_NAME --zone=$ZONE --tunnel-through-iap --command='
cd ~/rl-checkers/training 2>/dev/null || { echo "  ❌ Training not set up on VM"; exit 1; }

# Check if training is running
if tmux has-session -t training 2>/dev/null; then
    echo "  Status: 🟢 TRAINING RUNNING"
else
    echo "  Status: 🔴 TRAINING NOT RUNNING"
    if [ -f "checkpoints/gcp_training_status.json" ]; then
        echo "  (May have completed or crashed)"
    fi
fi
echo

# Check status file
if [ -f "checkpoints/gcp_training_status.json" ]; then
    python3 << PYTHON
import json
from datetime import datetime, timedelta

with open("checkpoints/gcp_training_status.json") as f:
    s = json.load(f)

games = s.get("total_games", 0)
target = s.get("target_games", 50000)
elapsed = s.get("elapsed_seconds", 0)
eta = s.get("eta_seconds", 0)
completed = s.get("completed", False)
saved = set(s.get("saved_difficulties", []))
metrics = s.get("metrics", {})

pct = 100 * games / target if target > 0 else 0
bar_len = 40
filled = int(bar_len * games / target)
bar = "█" * filled + "░" * (bar_len - filled)

def fmt_time(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0: return f"{h}h {m}m {s}s"
    if m > 0: return f"{m}m {s}s"
    return f"{s}s"

print(f"  Progress: [{bar}] {pct:.1f}%")
print(f"            {games:,} / {target:,} games")
print()
print(f"  Elapsed:  {fmt_time(elapsed)}")
if not completed and eta > 0:
    eta_time = datetime.now() + timedelta(seconds=eta)
    print(f"  ETA:      {fmt_time(eta)} (around {eta_time.strftime(\"%H:%M\")})")
print()

# Difficulties
diffs = {"easy": 500, "medium": 2000, "hard": 10000, "impossible": 50000}
print("  ┌─────────────┬───────────┬──────────────────────┐")
print("  │ Difficulty  │   Games   │        Status        │")
print("  ├─────────────┼───────────┼──────────────────────┤")
for d in ["easy", "medium", "hard", "impossible"]:
    t = diffs[d]
    if d in saved:
        st = "✅ Saved"
    elif games >= t:
        st = "✅ Ready"
    else:
        st = f"⏳ {t - games:,} to go"
    print(f"  │ {d:11s} │ {t:>9,} │ {st:^20s} │")
print("  └─────────────┴───────────┴──────────────────────┘")
print()

if metrics:
    print(f"  📊 Training Metrics:")
    print(f"     Loss: {metrics.get(\"loss\", 0):.4f}")
    print(f"     Policy: {metrics.get(\"policy_loss\", 0):.4f}")
    print(f"     Value: {metrics.get(\"value_loss\", 0):.4f}")
    print(f"     Buffer: {metrics.get(\"buffer_size\", 0):,} examples")

if completed:
    print()
    print("  🏆 TRAINING COMPLETE!")
PYTHON
else
    echo "  ⏳ Waiting for first status update..."
    echo "     (Training just started, check back in a few minutes)"
fi
echo
' 2>/dev/null

echo "═══════════════════════════════════════════════════════════════════"
echo
echo "  Commands:"
echo "    View live output:  gcloud compute ssh $VM_NAME --zone=$ZONE --tunnel-through-iap -- tmux attach -t training"
echo "    Stop VM:           gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo
