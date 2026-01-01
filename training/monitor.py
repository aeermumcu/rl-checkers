#!/usr/bin/env python3
"""
Live Training Monitor - Shows training progress in a nice format
Run in a separate terminal to watch progress.
"""

import json
import os
import time
from datetime import datetime

STATUS_FILE = os.path.join(os.path.dirname(__file__), 'checkpoints', 'gcp_training_status.json')

DIFFICULTY_THRESHOLDS = {
    'easy': 500,
    'medium': 2000,
    'hard': 10000,
    'impossible': 50000
}

def format_time(seconds):
    if seconds < 0:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

def draw_progress_bar(current, total, width=40):
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"

def main():
    print("🔍 Monitoring training progress...")
    print("   Press Ctrl+C to stop\n")
    
    while True:
        try:
            if not os.path.exists(STATUS_FILE):
                print("⏳ Waiting for training to start...")
                time.sleep(5)
                continue
            
            with open(STATUS_FILE) as f:
                status = json.load(f)
            
            clear_screen()
            
            games = status.get('total_games', 0)
            target = status.get('target_games', 50000)
            elapsed = status.get('elapsed_seconds', 0)
            eta = status.get('eta_seconds', 0)
            completed = status.get('completed', False)
            saved = set(status.get('saved_difficulties', []))
            metrics = status.get('metrics', {})
            
            print("╔══════════════════════════════════════════════════════════════════╗")
            print("║                    🎮 Checkers AI Training                       ║")
            print("╚══════════════════════════════════════════════════════════════════╝")
            print()
            
            # Status
            if completed:
                print("  Status: ✅ COMPLETE")
            else:
                print("  Status: 🟢 TRAINING")
            
            # Progress bar
            pct = 100 * games / target
            bar = draw_progress_bar(games, target)
            print(f"\n  Progress: {bar} {pct:.1f}%")
            print(f"            {games:,} / {target:,} games")
            
            # Time
            print(f"\n  Elapsed:  {format_time(elapsed)}")
            if not completed and eta > 0:
                eta_time = datetime.now().timestamp() + eta
                eta_str = datetime.fromtimestamp(eta_time).strftime("%H:%M")
                print(f"  ETA:      {format_time(eta)} (around {eta_str})")
            
            # Speed
            speed = status.get('games_per_second', 0)
            if speed > 0:
                print(f"  Speed:    {speed:.2f} games/sec")
            
            # Difficulty milestones
            print("\n  ┌─────────────┬───────────┬──────────────────────┐")
            print("  │ Difficulty  │   Games   │        Status        │")
            print("  ├─────────────┼───────────┼──────────────────────┤")
            
            for diff in ['easy', 'medium', 'hard', 'impossible']:
                threshold = DIFFICULTY_THRESHOLDS[diff]
                if diff in saved:
                    status_str = "✅ Saved"
                elif games >= threshold:
                    status_str = "✅ Ready"
                else:
                    remaining = threshold - games
                    status_str = f"⏳ {remaining:,} to go"
                print(f"  │ {diff:11s} │ {threshold:>9,} │ {status_str:^20s} │")
            
            print("  └─────────────┴───────────┴──────────────────────┘")
            
            # Metrics
            if metrics:
                print(f"\n  📊 Training Metrics:")
                print(f"     Loss: {metrics.get('loss', 0):.4f}")
                print(f"     Policy: {metrics.get('policy_loss', 0):.4f}")
                print(f"     Value: {metrics.get('value_loss', 0):.4f}")
                print(f"     Buffer: {metrics.get('buffer_size', 0):,} examples")
            
            print("\n" + "═" * 70)
            print(f"  Last updated: {status.get('last_update', 'N/A')}")
            print("  Press Ctrl+C to stop monitoring")
            
            time.sleep(10)  # Update every 10 seconds
            
        except KeyboardInterrupt:
            print("\n\n👋 Stopped monitoring.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
