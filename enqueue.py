#!/usr/bin/env python3
"""
enqueue.py — Stop hook for Claude Code (fast, < 100ms)
Drops a ticket into the queue for async processing by process_queue.py.
"""

import json
import sys
from datetime import date
from pathlib import Path

# Load config from same directory as this script
sys.path.insert(0, str(Path(__file__).parent))
try:
    from config import QUEUE_DIR as _QUEUE_DIR, LOG_FILE as _LOG_FILE, MIN_TURNS
    QUEUE_DIR = Path(_QUEUE_DIR)
    LOG_FILE = Path(_LOG_FILE)
except ImportError:
    # Fallback defaults if config.py is missing
    QUEUE_DIR = Path.home() / ".claude/hooks/queue"
    LOG_FILE = Path.home() / ".claude/hooks/auto_remember.log"
    MIN_TURNS = 5

MIN_NEW_TURNS = 10  # New turns required to re-process an already-processed session

TODAY = date.today().isoformat()


def log(msg: str):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{TODAY}] {msg}\n")
    except Exception:
        pass


def count_turns(jsonl_path: str) -> int:
    count = 0
    try:
        with open(jsonl_path) as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    if event.get("type") in ("user", "assistant"):
                        count += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return count


def main():
    try:
        raw_input = sys.stdin.read().strip()
        data = json.loads(raw_input) if raw_input else {}

        session_id = data.get("session_id", "unknown")
        transcript_path = data.get("transcript_path", "")
        cwd = data.get("cwd", "")

        if not transcript_path or not Path(transcript_path).exists():
            log(f"ENQUEUE SKIP (no transcript) session={session_id[:8]}")
            sys.exit(0)

        turn_count = count_turns(transcript_path)
        if turn_count < MIN_TURNS:
            log(f"ENQUEUE SKIP (too short: {turn_count} turns) session={session_id[:8]}")
            sys.exit(0)

        processed_path = QUEUE_DIR / "processed" / f"{session_id}.json"
        if processed_path.exists():
            # Check if the session has grown significantly since last processing
            try:
                processed_data = json.loads(processed_path.read_text(encoding="utf-8"))
                processed_turns = processed_data.get("turn_count", 0)
                new_turns = turn_count - processed_turns
                if new_turns >= MIN_NEW_TURNS:
                    processed_path.unlink()
                    log(f"RE-ENQUEUE session={session_id[:8]} (grew {processed_turns}→{turn_count} turns, +{new_turns} new)")
                    # Fall through to enqueue below
                else:
                    log(f"ENQUEUE SKIP (already processed, only +{new_turns} new turns) session={session_id[:8]}")
                    sys.exit(0)
            except Exception:
                log(f"ENQUEUE SKIP (already processed) session={session_id[:8]}")
                sys.exit(0)

        QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        ticket = {
            "session_id": session_id,
            "transcript_path": transcript_path,
            "cwd": cwd,
            "turn_count": turn_count,
            "enqueued_at": TODAY,
        }
        ticket_path = QUEUE_DIR / f"{session_id}.json"
        ticket_path.write_text(json.dumps(ticket, indent=2), encoding="utf-8")

        log(f"ENQUEUED session={session_id[:8]} turns={turn_count}")

    except Exception as e:
        log(f"ENQUEUE ERROR: {e}")

    sys.exit(0)


if __name__ == "__main__":
    main()
