#!/usr/bin/env python3
"""
process_queue.py — Async worker for auto_remember.
Processes tickets dropped by enqueue.py.
Triggered by launchd WatchPaths on the queue directory.
Uses Fireworks (kimi-k2) — latency doesn't matter here, quality does.
"""

import json
import os
import re
import subprocess
import sys
import traceback
from datetime import date
from pathlib import Path

# Load config from same directory as this script
sys.path.insert(0, str(Path(__file__).parent))
try:
    from config import (
        VAULT_NOTES_DIR, LOG_FILE, ENV_FILE, QUEUE_DIR, QDRANT_PATH,
        DEDUP_THRESHOLD, FIREWORKS_BASE_URL, FIREWORKS_MODEL, VOYAGE_EMBED_MODEL,
    )
    VAULT_NOTES_DIR = Path(VAULT_NOTES_DIR)
    LOG_FILE = Path(LOG_FILE)
    ENV_FILE = Path(ENV_FILE)
    QUEUE_DIR = Path(QUEUE_DIR)
    QDRANT_PATH = Path(QDRANT_PATH)
    HOOKS_DIR = Path(ENV_FILE).parent
except ImportError:
    print("ERROR: config.py not found. Copy config.example.py to config.py and edit paths.", file=sys.stderr)
    sys.exit(1)

PROCESSED_DIR = QUEUE_DIR / "processed"
COLLECTION = "vault_notes"

TODAY = date.today().isoformat()


def log(msg: str):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{TODAY}] {msg}\n")
    except Exception:
        pass


def load_env_file() -> dict:
    env = {}
    try:
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    except Exception:
        pass
    return env


def get_embed_clients():
    """Returns (voyageai.Client, QdrantClient) or (None, None) if unavailable."""
    try:
        import voyageai
        from qdrant_client import QdrantClient
    except ImportError:
        return None, None

    env = load_env_file()
    api_key = env.get("VOYAGE_API_KEY") or os.environ.get("VOYAGE_API_KEY", "")
    if not api_key or api_key.startswith("<"):
        return None, None
    if not QDRANT_PATH.exists():
        return None, None

    try:
        vo = voyageai.Client(api_key=api_key)
        qd = QdrantClient(path=str(QDRANT_PATH))
        existing = {c.name for c in qd.get_collections().collections}
        if COLLECTION not in existing:
            return None, None
        return vo, qd
    except Exception as e:
        log(f"EMBED clients error: {e}")
        return None, None


def check_semantic_dup(content: str) -> tuple[bool, str]:
    """Returns (True, target_id) if similar content already exists in Qdrant."""
    try:
        vo, qd = get_embed_clients()
        if vo is None:
            return False, ""
        result = vo.embed(
            [content[:500]],
            model=VOYAGE_EMBED_MODEL,
            input_type="query",
            truncation=True,
        )
        response = qd.query_points(
            collection_name=COLLECTION,
            query=result.embeddings[0],
            limit=1,
            score_threshold=DEDUP_THRESHOLD,
        )
        if response.points:
            return True, response.points[0].payload.get("note_id", "")
    except Exception as e:
        log(f"DEDUP error: {e}")
    return False, ""


def upsert_note_async(note_id: str):
    """Runs vault_embed.py in background to upsert a note into Qdrant."""
    try:
        script = str(HOOKS_DIR / "vault_embed.py")
        subprocess.Popen(
            ["python3", script, "--note", note_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log(f"EMBED async upsert launched: {note_id}")
    except Exception as e:
        log(f"EMBED async upsert error: {e}")


def extract_conversation(jsonl_path: str, max_chars: int = 40000) -> tuple[str, int]:
    """Extract the LAST turns that fit within max_chars.
    For long sessions, this ensures Kimi sees recent work, not the session-start summary."""
    turns = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    if event.get("type") not in ("user", "assistant"):
                        continue
                    msg = event.get("message", {})
                    role = msg.get("role", event.get("type", "unknown"))
                    content = msg.get("content", "")

                    if isinstance(content, str) and content.strip():
                        turns.append(f"{role.upper()}: {content[:2000]}")
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "").strip()
                                if text:
                                    turns.append(f"{role.upper()}: {text[:2000]}")
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        log(f"Error reading transcript: {e}")

    total_turns = len(turns)

    # Take the LAST turns that fit within max_chars (not the first)
    selected = []
    total_chars = 0
    for turn in reversed(turns):
        turn_len = len(turn) + 2  # +2 for "\n\n"
        if total_chars + turn_len > max_chars:
            break
        selected.append(turn)
        total_chars += turn_len
    selected.reverse()

    if len(selected) < total_turns:
        log(f"Conversation truncated: last {len(selected)}/{total_turns} turns ({total_chars} chars)")

    return "\n\n".join(selected), total_turns


def sanitize_note_id(note_id: str) -> str:
    """Normalize a note_id to a valid kebab-case slug (max 80 chars)."""
    import unicodedata
    note_id = unicodedata.normalize('NFKD', note_id)
    note_id = ''.join(c for c in note_id if not unicodedata.combining(c))
    note_id = note_id.lower()
    note_id = re.sub(r'[^a-z0-9\-]', '-', note_id)
    note_id = re.sub(r'-+', '-', note_id)
    note_id = note_id.strip('-')
    if len(note_id) > 80:
        note_id = note_id[:80].rstrip('-')
    return note_id


def build_title_to_id_map(notes_dir: Path) -> dict:
    """Build a mapping from lowercase H1 title (and aliases) → note_id."""
    mapping = {}
    try:
        for f in notes_dir.glob("*.md"):
            if f.name.startswith(".") or f.name.startswith("._"):
                continue
            try:
                text = f.read_text(encoding="utf-8")[:500]
                title_m = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
                if title_m:
                    mapping[title_m.group(1).strip().lower()] = f.stem
                aliases_m = re.search(r'^aliases:\s*\[(.+)\]', text, re.MULTILINE)
                if aliases_m:
                    for alias in aliases_m.group(1).split(','):
                        alias = alias.strip().strip('"').strip("'").lower()
                        if alias:
                            mapping[alias] = f.stem
            except Exception:
                pass
    except Exception as e:
        log(f"Error building title map: {e}")
    return mapping


def fix_wikilinks_in_content(content: str, title_to_id: dict, valid_ids: set) -> str:
    """Replace [[Full Title]] links with [[note-id]] links in generated content.

    - If the target is already a valid note_id → leave as-is.
    - If the target matches a known title or alias → replace with the slug.
    - If unresolvable → strip the link brackets (keeps the text, removes broken ref).
    """
    def replace_link(m):
        target = m.group(1).strip()
        display = m.group(2)
        if target in valid_ids:
            return m.group(0)
        target_lower = target.lower()
        if target_lower in title_to_id:
            corrected = title_to_id[target_lower]
            return f"[[{corrected}|{display}]]" if display else f"[[{corrected}]]"
        # Unresolvable: keep display text or target text, drop brackets
        return display if display else target

    return re.compile(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]').sub(replace_link, content)


def get_existing_notes_summary(notes_dir: Path, limit: int = 80) -> str:
    lines = []
    try:
        for f in sorted(notes_dir.glob("*.md")):
            if f.name.startswith(".") or f.name.startswith("_"):
                continue
            try:
                text = f.read_text(encoding="utf-8")[:400]
                desc_m = re.search(r'^description:\s*(.+)$', text, re.MULTILINE)
                title_m = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
                if desc_m:
                    lines.append(f"- {f.stem}: {desc_m.group(1)[:100]}")
                elif title_m:
                    lines.append(f"- {f.stem}: {title_m.group(1)[:100]}")
                else:
                    lines.append(f"- {f.stem}")
            except Exception:
                lines.append(f"- {f.stem}")
            if len(lines) >= limit:
                break
    except Exception as e:
        log(f"Error listing notes: {e}")
    return "\n".join(lines)


def extract_facts_with_llm(conversation: str, existing_notes: str) -> list:
    try:
        from openai import OpenAI
    except ImportError:
        log("openai package not installed, skipping")
        return []

    env = load_env_file()
    api_key = env.get("FIREWORKS_API_KEY") or os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        log("FIREWORKS_API_KEY missing, skipping")
        return []

    client = OpenAI(api_key=api_key, base_url=FIREWORKS_BASE_URL)

    # Strip Claude Code UI tags that confuse the extraction LLM
    clean_conversation = re.sub(r'<system-reminder>.*?</system-reminder>', '', conversation, flags=re.DOTALL)
    clean_conversation = re.sub(r'<local-command-caveat>.*?</local-command-caveat>', '', clean_conversation, flags=re.DOTALL)
    clean_conversation = re.sub(r'<[a-z-]+>|</[a-z-]+>', '', clean_conversation)
    clean_conversation = clean_conversation.strip()

    system_msg = "You are a JSON extraction bot. You output ONLY valid JSON arrays. Never output prose, reasoning, explanations, or conversational text. Your entire response must be parseable by json.loads(). If there is nothing to extract, output: []"

    user_msg = f"""Extract 0-15 durable atomic facts from this Claude Code session transcript.

WHAT TO CAPTURE (any domain):
- Technical decisions, system configs, solutions found, established workflows
- Facts learned about tools, infrastructure, methods, courses, personal projects
- Ignore: temporary debugging, small talk, reformulations, unresolved intermediate steps

RELATION TYPES:
- NEW: entirely new fact, absent from existing notes
- UPDATES:<note_id>: replaces existing info (e.g. threshold changed, value corrected)
- EXTENDS:<note_id>: adds detail without replacing (e.g. extra detail on existing method)

EXISTING VAULT NOTES (format: "- note_id: description"):
{existing_notes}

WIKI LINK RULES:
- Links MUST use kebab-case note_id slugs, NEVER full titles
- Every link target must match a slug from the existing notes list
- Every NEW note MUST end with a Topics: section linking to a relevant topic map

RESPONSE FORMAT — JSON array only, nothing else before or after:
[
  {{
    "note_id": "kebab-case-slug-max-80-chars",
    "relation": "NEW",
    "content": "---\\ndescription: one sentence\\ntype: decision\\ncreated: {TODAY}\\nconfidence: experimental\\n---\\n\\n# Title as proposition\\n\\nBody...\\n\\n## Links\\n\\n- [[existing-note-slug]]\\n\\n---\\n\\nTopics:\\n- [[relevant-topic-map]]"
  }}
]

For EXTENDS: content is the additional text to append only (not a full note).
For UPDATES: content is the complete revised note.
If nothing memorable: []

SESSION TRANSCRIPT:
{clean_conversation}"""

    raw = ""
    try:
        response = client.chat.completions.create(
            model=FIREWORKS_MODEL,
            max_tokens=10000,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        raw = _repair_json_newlines(raw)

        log(f"LLM response ({len(raw)} chars): {raw[:300]}")
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []

    except json.JSONDecodeError as e:
        log(f"Invalid JSON from LLM: {e} — raw: {raw[:300]}")
        return []
    except Exception as e:
        log(f"Fireworks API error: {e}")
        return []


def _repair_json_newlines(raw: str) -> str:
    """Fix literal newlines inside JSON strings (common LLM output issue)."""
    result = []
    in_string = False
    escaped = False
    for char in raw:
        if escaped:
            result.append(char)
            escaped = False
        elif char == '\\' and in_string:
            result.append(char)
            escaped = True
        elif char == '"':
            result.append(char)
            in_string = not in_string
        elif char == '\n' and in_string:
            result.append('\\n')
        elif char == '\r' and in_string:
            result.append('\\r')
        elif char == '\t' and in_string:
            result.append('\\t')
        else:
            result.append(char)
    return ''.join(result)


def write_note(note_id: str, content: str, relation: str):
    notes_dir = VAULT_NOTES_DIR

    if relation.startswith("UPDATES:"):
        target_id = relation.split(":", 1)[1].strip()
        target_path = notes_dir / f"{target_id}.md"
        if target_path.exists():
            target_path.write_text(content, encoding="utf-8")
            log(f"UPDATED  {target_id}")
            return
        log(f"UPDATES target not found ({target_id}), creating as NEW {note_id}")

    elif relation.startswith("EXTENDS:"):
        target_id = relation.split(":", 1)[1].strip()
        target_path = notes_dir / f"{target_id}.md"
        if target_path.exists():
            existing = target_path.read_text(encoding="utf-8")
            extension = f"\n\n---\n*Auto-extension {TODAY}:*\n\n{content}"
            target_path.write_text(existing + extension, encoding="utf-8")
            log(f"EXTENDED {target_id}")
            return
        log(f"EXTENDS target not found ({target_id}), creating as NEW {note_id}")

    note_path = notes_dir / f"{note_id}.md"
    note_path.write_text(content, encoding="utf-8")
    log(f"NEW      {note_id}")


def process_ticket(ticket_path: Path):
    try:
        ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"Error reading ticket {ticket_path.name}: {e}")
        return

    session_id = ticket.get("session_id", "unknown")
    transcript_path = ticket.get("transcript_path", "")

    log(f"--- PROCESSING session={session_id[:8]}")

    if not VAULT_NOTES_DIR.exists():
        log(f"Vault not found: {VAULT_NOTES_DIR}, skipping")
        return

    if not transcript_path or not Path(transcript_path).exists():
        log(f"Transcript not found: {transcript_path}, skipping")
        _archive(ticket_path, session_id)
        return

    conversation, turn_count = extract_conversation(transcript_path)
    log(f"Conversation: {turn_count} turns, {len(conversation)} chars")

    existing_notes = get_existing_notes_summary(VAULT_NOTES_DIR)

    # Pre-build maps for post-generation link correction
    title_to_id = build_title_to_id_map(VAULT_NOTES_DIR)
    valid_ids = {f.stem for f in VAULT_NOTES_DIR.glob("*.md") if not f.name.startswith("._")}

    facts = extract_facts_with_llm(conversation, existing_notes)

    if not facts:
        log("No memorable facts extracted")
        _archive(ticket_path, session_id, turn_count)
        return

    log(f"Facts extracted: {len(facts)}")
    written = 0
    for fact in facts:
        try:
            note_id = fact.get("note_id", "").strip()
            relation = fact.get("relation", "NEW")
            content = fact.get("content", "").strip()
            if not note_id or not content:
                log(f"Invalid fact ignored: {fact}")
                continue

            # Sanitize note_id to a valid kebab-case slug
            note_id_clean = sanitize_note_id(note_id)
            if note_id_clean != note_id:
                log(f"note_id sanitized: '{note_id}' → '{note_id_clean}'")
                note_id = note_id_clean

            # Fix any title-style [[Full Title]] links to [[note-id]] slugs
            content = fix_wikilinks_in_content(content, title_to_id, valid_ids)

            # Semantic dedup: only for NEW facts
            if relation == "NEW":
                is_dup, target_id = check_semantic_dup(content)
                if is_dup and target_id:
                    relation = f"EXTENDS:{target_id}"
                    log(f"DEDUP: {note_id} → EXTENDS:{target_id}")

            write_note(note_id, content, relation)
            written += 1

            # Incremental upsert into Qdrant after writing
            actual_id = note_id
            if relation.startswith("UPDATES:"):
                actual_id = relation.split(":", 1)[1].strip()
            elif relation.startswith("EXTENDS:"):
                actual_id = relation.split(":", 1)[1].strip()
            upsert_note_async(actual_id)

        except Exception as e:
            log(f"Error writing {fact.get('note_id', '?')}: {e}")

    log(f"Notes written: {written}/{len(facts)}")
    _archive(ticket_path, session_id, turn_count)


def _archive(ticket_path: Path, session_id: str, turn_count: int = 0):
    try:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        dest = PROCESSED_DIR / ticket_path.name
        if ticket_path.exists():
            # Update turn_count so future re-enqueue comparisons are accurate
            if turn_count > 0:
                try:
                    data = json.loads(ticket_path.read_text(encoding="utf-8"))
                    data["turn_count"] = turn_count
                    data["processed_at"] = TODAY
                    ticket_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                except Exception:
                    pass
            ticket_path.rename(dest)
            log(f"ARCHIVED session={session_id[:8]}")
        else:
            # Ticket already gone — write directly to processed/
            data = {"session_id": session_id, "turn_count": turn_count, "processed_at": TODAY}
            dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
            log(f"ARCHIVED (recreated) session={session_id[:8]}")
    except Exception as e:
        log(f"Archive error: {e}")


def main():
    try:
        tickets = [
            f for f in QUEUE_DIR.glob("*.json")
            if f.is_file() and f.parent == QUEUE_DIR
        ]

        if not tickets:
            log("Queue empty, nothing to process")
            sys.exit(0)

        log(f"=== process_queue: {len(tickets)} ticket(s) to process")

        for ticket_path in sorted(tickets, key=lambda f: f.stat().st_mtime):
            session_id = ticket_path.stem
            if (PROCESSED_DIR / ticket_path.name).exists():
                log(f"SKIP (already processed) session={session_id[:8]}")
                ticket_path.unlink(missing_ok=True)
                continue
            process_ticket(ticket_path)

        log("=== process_queue: done")

    except Exception as e:
        log(f"Fatal error in process_queue: {e}\n{traceback.format_exc()}")

    sys.exit(0)


if __name__ == "__main__":
    main()
