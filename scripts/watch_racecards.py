#!/usr/bin/env python3
"""
File Watcher for Automatic Prediction Generation

Monitors data/raw/ folder for new racecard JSON files and automatically
runs predictions when racecards_YYYY-MM-DD.json files are created or modified.

Usage:
  python scripts/watch_racecards.py              # Start watching (Ctrl+C to stop)
  python scripts/watch_racecards.py --once       # Process existing files once and exit
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("[ERROR] watchdog library not installed")
    print("Install with: pip install watchdog")
    sys.exit(1)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
PREDICT_SCRIPT = PROJECT_ROOT / "scripts" / "predict_todays_races.py"

# Pattern to match racecard files
RACECARD_PATTERN = re.compile(r"racecards_(\d{4}-\d{2}-\d{2})\.json")


class RacecardHandler(FileSystemEventHandler):
    """Handles file system events for racecard files"""
    
    def __init__(self, auto_process=True, debounce_seconds=2):
        super().__init__()
        self.auto_process = auto_process
        self.debounce_seconds = debounce_seconds
        self.pending_files = {}  # Track files with timestamps
        self.processing = set()  # Track currently processing dates
    
    def _extract_date(self, file_path):
        """Extract date from racecard filename"""
        match = RACECARD_PATTERN.match(Path(file_path).name)
        return match.group(1) if match else None
    
    def _should_process(self, date_str):
        """Check if we should process this date"""
        if not date_str:
            return False
        
        # Skip if already processing
        if date_str in self.processing:
            return False
        
        # Check if predictions already exist
        pred_file = DATA_PROCESSED / f"predictions_{date_str}.csv"
        return not pred_file.exists()
    
    def _run_prediction(self, date_str):
        """Run prediction for a specific date"""
        if date_str in self.processing:
            return
        
        self.processing.add(date_str)
        
        try:
            print(f"\n{'='*70}")
            print(f"  🔮 Auto-generating predictions for {date_str}")
            print(f"{'='*70}")
            
            result = subprocess.run(
                [sys.executable, str(PREDICT_SCRIPT), "--date", date_str],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print(f"  ✓ Success: Predictions ready for {date_str}")
                print(f"  📁 Output: data/processed/predictions_{date_str}.csv")
            else:
                print(f"  ✗ Failed: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout: Prediction took too long")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        finally:
            self.processing.discard(date_str)
            print(f"{'='*70}\n")
    
    def on_created(self, event):
        """Called when a file is created"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if not RACECARD_PATTERN.match(file_path.name):
            return
        
        date_str = self._extract_date(file_path)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📥 New racecard detected: {file_path.name}")
        
        if self.auto_process and self._should_process(date_str):
            # Debounce: wait a bit in case file is still being written
            self.pending_files[date_str] = time.time()
            time.sleep(self.debounce_seconds)
            
            # Check if still the most recent event for this date
            if time.time() - self.pending_files.get(date_str, 0) >= self.debounce_seconds - 0.1:
                self._run_prediction(date_str)
                self.pending_files.pop(date_str, None)
        else:
            if date_str in self.processing:
                print(f"  ⏳ Already processing {date_str}...")
            else:
                print(f"  ℹ️  Predictions already exist for {date_str}")
    
    def on_modified(self, event):
        """Called when a file is modified"""
        # Treat modifications like new files (in case file was updated)
        if not event.is_directory:
            file_path = Path(event.src_path)
            if RACECARD_PATTERN.match(file_path.name):
                date_str = self._extract_date(file_path)
                
                # Only process if predictions don't exist
                if self.auto_process and self._should_process(date_str):
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📝 Racecard updated: {file_path.name}")
                    self.pending_files[date_str] = time.time()
                    time.sleep(self.debounce_seconds)
                    
                    if time.time() - self.pending_files.get(date_str, 0) >= self.debounce_seconds - 0.1:
                        self._run_prediction(date_str)
                        self.pending_files.pop(date_str, None)


def process_existing_files():
    """Process any existing racecard files that don't have predictions"""
    print("\n" + "="*70)
    print("  🔍 Checking for existing racecards...")
    print("="*70)
    
    racecards = []
    for file in DATA_RAW.glob("racecards_*.json"):
        match = RACECARD_PATTERN.match(file.name)
        if match:
            date_str = match.group(1)
            pred_file = DATA_PROCESSED / f"predictions_{date_str}.csv"
            
            if not pred_file.exists():
                racecards.append(date_str)
    
    if racecards:
        print(f"\nFound {len(racecards)} racecard(s) without predictions:")
        for date_str in sorted(racecards):
            print(f"  - {date_str}")
        
        response = input(f"\nProcess these {len(racecards)} dates now? [y/N]: ")
        if response.lower() == 'y':
            for date_str in sorted(racecards):
                handler = RacecardHandler()
                handler._run_prediction(date_str)
    else:
        print("\n✓ All existing racecards already have predictions")


def main():
    parser = argparse.ArgumentParser(
        description="Watch for new racecard files and auto-generate predictions"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process existing files once and exit (don't watch)"
    )
    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="Don't auto-process files, just notify"
    )
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not DATA_RAW.exists():
        print(f"[ERROR] Directory not found: {DATA_RAW}")
        sys.exit(1)
    
    # Process existing files if requested
    if args.once:
        process_existing_files()
        return
    
    print("\n" + "="*70)
    print("  🔭 RACECARD FILE WATCHER")
    print("="*70)
    print(f"\n  Monitoring: {DATA_RAW}")
    print(f"  Pattern:    racecards_YYYY-MM-DD.json")
    print(f"  Auto-run:   {'Enabled' if not args.no_auto else 'Disabled'}")
    print(f"\n  Press Ctrl+C to stop watching\n")
    print("="*70)
    
    # Create event handler and observer
    event_handler = RacecardHandler(auto_process=not args.no_auto)
    observer = Observer()
    observer.schedule(event_handler, str(DATA_RAW), recursive=False)
    
    # Start watching
    observer.start()
    print(f"\n✓ Watching for new racecards...\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("  🛑 Stopping file watcher...")
        print("="*70)
        observer.stop()
    
    observer.join()
    print("\n✓ File watcher stopped\n")


if __name__ == "__main__":
    main()
