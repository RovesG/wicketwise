#!/usr/bin/env python3
# Purpose: Demo script for the WicketWise replay visualizer
# Author: Assistant, Last Modified: 2024

"""
Demo script to launch the WicketWise Match Replay Visualizer.

This script demonstrates how to use the Streamlit-based replay visualizer
to analyze cricket match predictions ball-by-ball.

Usage:
    python3 demo_replay_visualizer.py
    
    Or with custom CSV file:
    python3 demo_replay_visualizer.py --csv path/to/your/eval_predictions.csv
"""

import argparse
import sys
import subprocess
from pathlib import Path

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Demo WicketWise Match Replay Visualizer")
    parser.add_argument(
        "--csv",
        type=str,
        default="sample_eval_predictions.csv",
        help="Path to eval_predictions.csv file (default: sample_eval_predictions.csv)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run Streamlit on (default: 8501)"
    )
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        print("\n💡 Available options:")
        print("1. Use the sample data: python3 demo_replay_visualizer.py")
        print("2. Generate eval_predictions.csv by running: python3 eval.py --checkpoint your_model.pth")
        print("3. Specify a different CSV file: python3 demo_replay_visualizer.py --csv path/to/your/file.csv")
        sys.exit(1)
    
    print("🏏 WicketWise Match Replay Visualizer Demo")
    print("=" * 50)
    print(f"📊 Loading data from: {csv_path}")
    print(f"🌐 Starting Streamlit server on port {args.port}")
    print("\n🚀 Launching visualizer...")
    print("\n📝 Instructions:")
    print("1. The visualizer will open in your web browser")
    print("2. Use the sidebar to load your CSV file")
    print("3. Select a match to replay")
    print("4. Navigate through balls using the slider or buttons")
    print("5. Analyze predictions, win probability, and betting insights")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Launch Streamlit with the replay visualizer
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "wicketwise/replay_visualizer.py",
            "--server.port", str(args.port),
            "--server.headless", "false",
            "--server.fileWatcherType", "none"
        ]
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running Streamlit: {e}")
        print("\n💡 Make sure Streamlit is installed:")
        print("   pip install streamlit")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main() 