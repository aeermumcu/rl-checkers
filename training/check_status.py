#!/usr/bin/env python3
"""
Training Status Viewer - Shows a nice progress report
"""

import json
import os
import subprocess
from datetime import datetime, timedelta

# Paths
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(TRAINING_DIR, 'checkpoints')
STATUS_FILE = os.path.join(CHECKPOINT_DIR, 'training_status.json')
MODEL_DIR = os.path.join(TRAINING_DIR, '..', 'model')

def format_time(seconds):
    """Format seconds as HH:MM:SS or MM:SS"""
    if seconds < 0:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def is_training_running():
    """Check if training process is running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'train.py'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def get_checkpoint_info():
    """Get info about the latest checkpoint"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest.weights.h5')
    if os.path.exists(checkpoint_path):
        stat = os.stat(checkpoint_path)
        modified = datetime.fromtimestamp(stat.st_mtime)
        size_mb = stat.st_size / (1024 * 1024)
        return modified, size_mb
    return None, None

def main():
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║              🎮 Checkers AI Training Status                   ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    
    # Check if training is running
    running = is_training_running()
    status_icon = "🟢" if running else "🔴"
    status_text = "RUNNING" if running else "STOPPED"
    print(f"  Status: {status_icon} {status_text}")
    
    # Check for status file
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'r') as f:
            status = json.load(f)
        
        current = status.get('current_iteration', 0)
        total = status.get('total_iterations', 0)
        completed = status.get('completed', False)
        elapsed = status.get('elapsed_seconds', 0)
        eta = status.get('eta_seconds', 0)
        
        if completed:
            print(f"  Progress: ✅ COMPLETE ({total}/{total} iterations)")
        else:
            pct = (current / total * 100) if total > 0 else 0
            print(f"  Progress: {current}/{total} iterations ({pct:.0f}%)")
        
        print(f"  Elapsed:  {format_time(elapsed)}")
        if not completed and eta > 0:
            eta_time = datetime.now() + timedelta(seconds=eta)
            print(f"  ETA:      {format_time(eta)} (around {eta_time.strftime('%H:%M')})")
        
        print()
        print("  ┌──────────┬─────────────┬─────────────┬────────────┐")
        print("  │ Iter     │ Policy Loss │ Value Loss  │ Status     │")
        print("  ├──────────┼─────────────┼─────────────┼────────────┤")
        
        # Show iteration history from iteration_times
        iter_times = status.get('iteration_times', [])
        policy_loss = status.get('policy_loss', 0)
        value_loss = status.get('value_loss', 0)
        
        # We only have the final losses, so show them for the current iteration
        for i in range(1, total + 1):
            if i < current:
                # Completed iterations - we don't have individual losses stored
                # Show checkmark only
                print(f"  │ {i:>3}/{total:<3}  │     ...     │     ...     │     ✅     │")
            elif i == current:
                if completed or i == current:
                    print(f"  │ {i:>3}/{total:<3}  │   {policy_loss:>6.3f}    │   {value_loss:>6.3f}    │     ✅     │")
            else:
                print(f"  │ {i:>3}/{total:<3}  │     ...     │     ...     │     ⏳     │")
        
        print("  └──────────┴─────────────┴─────────────┴────────────┘")
        print()
        print(f"  📊 Total Games: {status.get('total_games', 0)}")
        print(f"  📦 Buffer Size: {status.get('buffer_size', 0)}")
        
    else:
        print("\n  ⚠️  No training status file found.")
        print("     Status file will be created when training starts.")
        print("     (The current run uses an older version without status tracking)")
    
    # Checkpoint info
    modified, size = get_checkpoint_info()
    if modified:
        print()
        print(f"  💾 Latest checkpoint: {modified.strftime('%Y-%m-%d %H:%M:%S')} ({size:.1f} MB)")
    
    # Check for final model
    model_path = os.path.join(MODEL_DIR, 'model.keras')
    if os.path.exists(model_path):
        stat = os.stat(model_path)
        modified = datetime.fromtimestamp(stat.st_mtime)
        print()
        print(f"  ✅ Final model exported: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"     🎮 Play: open ../index.html")
    
    print()
    print("═" * 67)
    print()

if __name__ == "__main__":
    main()
