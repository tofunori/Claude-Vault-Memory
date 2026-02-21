#!/usr/bin/env python3
"""
vault_session_brief.py — SessionStart hook for Claude Code.

Injects a brief of key context at the beginning of each session:
- Active projects (recently retrieved notes of type 'context' or 'module')
- Key preferences (confirmed notes of type 'preference')
- Recent decisions (notes of type 'decision' from last 14 days)

Hook config in ~/.claude/settings.json:
  "SessionStart": [{
    "hooks": [{"type": "command", "command": "python3 /path/to/vault_session_brief.py"}]
  }]
"""

import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    from config import (
        VAULT_NOTES_DIR as _VND, LOG_FILE as _LF,
        GRAPH_CACHE_PATH as _GCP,
    )
    VAULT_NOTES_DIR = Path(_VND)
    LOG_FILE = Path(_LF)
    GRAPH_CACHE_PATH = Path(_GCP)
except ImportError:
    sys.exit(0)

TODAY = date.today()
RECENT_DAYS = 14
MAX_BRIEF_NOTES = 8


def log(msg: str):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{TODAY.isoformat()}] {msg}\n")
    except Exception:
        pass


def parse_frontmatter(text: str) -> dict:
    """Extract frontmatter fields from note text."""
    fm = {}
    for field in ("description", "type", "confidence", "created"):
        m = re.search(rf'^{field}:\s*(.+)$', text, re.MULTILINE)
        if m:
            fm[field] = m.group(1).strip()
    return fm


def main():
    if not VAULT_NOTES_DIR.exists():
        sys.exit(0)

    preferences = []
    recent_decisions = []
    active_context = []
    cutoff = (TODAY - timedelta(days=RECENT_DAYS)).isoformat()

    for p in sorted(VAULT_NOTES_DIR.glob("*.md")):
        if p.name.startswith(".") or p.name.startswith("_"):
            continue
        try:
            text = p.read_text(encoding="utf-8")[:600]
            fm = parse_frontmatter(text)
            ntype = fm.get("type", "")
            confidence = fm.get("confidence", "")
            created = fm.get("created", "")
            desc = fm.get("description", p.stem)

            # Confirmed preferences (always relevant)
            if ntype == "preference" and confidence == "confirmed":
                preferences.append(f"  - [[{p.stem}]]: {desc}")

            # Recent decisions
            elif ntype == "decision" and created >= cutoff:
                preferences.append(f"  - [[{p.stem}]]: {desc}")

            # Active project context (recent)
            elif ntype in ("context", "module") and created >= cutoff:
                active_context.append(f"  - [[{p.stem}]]: {desc}")

        except Exception:
            continue

    # Build brief
    sections = []

    if preferences:
        sections.append("Key preferences & recent decisions:")
        sections.extend(preferences[:MAX_BRIEF_NOTES])

    if active_context:
        sections.append("\nActive project context:")
        sections.extend(active_context[:MAX_BRIEF_NOTES])

    if not sections:
        sys.exit(0)

    # Get vault stats for awareness
    total_notes = sum(1 for p in VAULT_NOTES_DIR.glob("*.md")
                      if not p.name.startswith(".") and not p.name.startswith("_"))

    lines = [
        f"=== Vault memory brief ({total_notes} notes) ===",
        *sections,
        f"\nNote: vault notes are automatically surfaced per-message via retrieval hook.",
    ]
    print("\n".join(lines))
    log(f"SESSION_BRIEF: {len(preferences)} prefs + {len(active_context)} context → {total_notes} total notes")


if __name__ == "__main__":
    main()
