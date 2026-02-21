#!/usr/bin/env python3
"""
process_queue.py — Async worker for auto_remember (v7).
Processes tickets dropped by enqueue.py.
Triggered by launchd WatchPaths on the queue directory.

v7 improvements (on top of v6):
- Typed relations: relation/parent_note/superseded_by persisted in frontmatter
- Source chunk storage: conversation excerpt saved alongside extracted notes
- Smart forgetting: forget_after field + auto-archive support

v6 improvements:
- Incremental graph cache updates after writing notes
- Conflict detection in LLM extraction prompt
- Pre-query vault before extraction (reduces duplicates)
- Atomic writes (temp file + rename)
- Smart transcript truncation (filter tool noise, cap code blocks)
"""

import json
import os
import re
import subprocess
import sys
import tempfile
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

# Optional config with defaults
try:
    from config import GRAPH_CACHE_PATH as _GCP
    GRAPH_CACHE_PATH = Path(_GCP)
except ImportError:
    GRAPH_CACHE_PATH = HOOKS_DIR / "vault_graph_cache.json"

try:
    from config import MAX_CODE_BLOCK_CHARS
except ImportError:
    MAX_CODE_BLOCK_CHARS = 500

try:
    from config import VALIDATION_ENABLED
except ImportError:
    VALIDATION_ENABLED = True

try:
    from config import SOURCE_CHUNKS_ENABLED
except ImportError:
    SOURCE_CHUNKS_ENABLED = True

try:
    from config import SOURCE_CHUNK_MAX_CHARS
except ImportError:
    SOURCE_CHUNK_MAX_CHARS = 2000

try:
    from config import SOURCE_CHUNKS_DIR as _SCD
    SOURCE_CHUNKS_DIR = Path(_SCD)
except ImportError:
    SOURCE_CHUNKS_DIR = VAULT_NOTES_DIR / "_sources"

PROCESSED_DIR = QUEUE_DIR / "processed"
COLLECTION = "vault_notes"

TODAY = date.today().isoformat()

_STATUS_FILE = Path.home() / ".claude/hooks/memory_status.txt"


def _write_memory_status(msg: str):
    try:
        _STATUS_FILE.write_text(msg, encoding="utf-8")
    except Exception:
        pass


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


# ─── Smart Transcript Extraction ────────────────────────────────────────────


def _truncate_code_blocks(text: str, max_chars: int = 500) -> str:
    """Truncate large code blocks in a message to reduce noise."""
    def replace_block(m):
        lang = m.group(1) or ""
        code = m.group(2)
        if len(code) <= max_chars:
            return m.group(0)
        return f"```{lang}\n{code[:max_chars]}\n... [truncated {len(code) - max_chars} chars]\n```"
    return re.sub(r'```(\w*)\n(.*?)```', replace_block, text, flags=re.DOTALL)


def extract_conversation(jsonl_path: str, max_chars: int = 40000) -> tuple[str, int]:
    """Extract the LAST turns that fit within max_chars.
    Smart truncation: filters tool noise, caps code blocks."""
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
                        # Truncate code blocks and cap individual messages
                        cleaned = _truncate_code_blocks(content, MAX_CODE_BLOCK_CHARS)
                        turns.append(f"{role.upper()}: {cleaned[:2000]}")
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "").strip()
                                if text:
                                    cleaned = _truncate_code_blocks(text, MAX_CODE_BLOCK_CHARS)
                                    turns.append(f"{role.upper()}: {cleaned[:2000]}")
                            # Skip tool_use and tool_result blocks (noise for extraction)
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


# ─── Note ID & Link Processing ──────────────────────────────────────────────


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
    """Replace [[Full Title]] links with [[note-id]] links in generated content."""
    def replace_link(m):
        target = m.group(1).strip()
        display = m.group(2)
        if target in valid_ids:
            return m.group(0)
        target_lower = target.lower()
        if target_lower in title_to_id:
            corrected = title_to_id[target_lower]
            return f"[[{corrected}|{display}]]" if display else f"[[{corrected}]]"
        return display if display else target

    return re.compile(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]').sub(replace_link, content)


# ─── Existing Notes Summary & Pre-Query ─────────────────────────────────────


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


def pre_query_vault(conversation: str, notes_dir: Path, top_k: int = 5) -> str:
    """Search vault for notes related to the conversation topics.
    Returns formatted context of existing related notes to inject in the extraction prompt."""
    try:
        vo, qd = get_embed_clients()
        if vo is None:
            return ""

        # Use first 1000 chars of conversation as query (topic signal)
        query_text = conversation[:1000]
        result = vo.embed(
            [query_text],
            model=VOYAGE_EMBED_MODEL,
            input_type="query",
            truncation=True,
        )
        response = qd.query_points(
            collection_name=COLLECTION,
            query=result.embeddings[0],
            limit=top_k,
            score_threshold=0.50,  # Lower threshold for broader context
        )

        if not response.points:
            return ""

        related = []
        for r in response.points:
            nid = r.payload.get("note_id", "")
            note_path = notes_dir / f"{nid}.md"
            if note_path.exists():
                try:
                    content = note_path.read_text(encoding="utf-8")[:800]
                    related.append(f"### {nid} (score: {r.score:.2f})\n{content}")
                except Exception:
                    pass

        if related:
            log(f"PRE-QUERY: found {len(related)} related notes for extraction context")
            return "\n\n".join(related)
    except Exception as e:
        log(f"PRE-QUERY error: {e}")
    return ""


# ─── LLM Extraction ─────────────────────────────────────────────────────────


def extract_facts_with_llm(conversation: str, existing_notes: str, related_context: str) -> list:
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

    # Build related context section for conflict detection
    related_section = ""
    if related_context:
        related_section = f"""
RELATED EXISTING NOTES (full content — check for conflicts and overlaps):
{related_context}

CONFLICT RULES:
- If a new fact CONTRADICTS an existing note, use UPDATES:<note_id> and include the corrected content
- If a new fact is ALREADY covered by an existing note, SKIP it (do not extract)
- If a new fact ADDS DETAIL to an existing note, use EXTENDS:<note_id>
- Only use NEW for facts genuinely absent from existing notes
"""

    user_msg = f"""Extract 0-15 durable atomic facts from this Claude Code session transcript.

WHAT TO CAPTURE (any domain):
- Technical decisions, system configs, solutions found, established workflows
- Facts learned about tools, infrastructure, methods, courses, personal projects
- Ignore: temporary debugging, small talk, reformulations, unresolved intermediate steps

RELATION TYPES:
- NEW: entirely new fact, absent from existing notes
- UPDATES:<note_id>: replaces existing info (e.g. threshold changed, value corrected, fact superseded)
- EXTENDS:<note_id>: adds detail without replacing (e.g. extra detail on existing method)

EXISTING VAULT NOTES (format: "- note_id: description"):
{existing_notes}
{related_section}
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


# ─── Extraction Validation ───────────────────────────────────────────────────


def validate_extracted_facts(facts: list, conversation: str) -> list:
    """Second-pass validation: check that extracted facts are actually grounded
    in the conversation transcript. Removes hallucinated extractions."""
    if not VALIDATION_ENABLED or not facts:
        return facts

    try:
        from openai import OpenAI
    except ImportError:
        return facts

    env = load_env_file()
    api_key = env.get("FIREWORKS_API_KEY") or os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        return facts

    client = OpenAI(api_key=api_key, base_url=FIREWORKS_BASE_URL)

    # Build compact summary of facts for validation
    fact_summaries = []
    for i, f in enumerate(facts):
        desc = ""
        content = f.get("content", "")
        # Extract description from content frontmatter
        desc_m = re.search(r'description:\s*(.+)', content)
        if desc_m:
            desc = desc_m.group(1).strip()
        else:
            desc = content[:150]
        fact_summaries.append(f"{i}: [{f.get('relation', 'NEW')}] {f.get('note_id', '?')} — {desc}")

    prompt = f"""You are a fact-checker. Given a conversation transcript and a list of extracted facts,
identify which facts are NOT actually supported by the conversation.

FACTS TO VALIDATE:
{chr(10).join(fact_summaries)}

CONVERSATION (last 5000 chars):
{conversation[-5000:]}

Return ONLY a JSON array of indices (0-based) of facts that ARE valid and grounded in the conversation.
Example: [0, 2, 3] means facts 0, 2, and 3 are valid; fact 1 is hallucinated.
If all facts are valid: {list(range(len(facts)))}
If no facts are valid: []"""

    try:
        response = client.chat.completions.create(
            model=FIREWORKS_MODEL,
            max_tokens=500,
            messages=[
                {"role": "system", "content": "Output ONLY a JSON array of integers. No prose."},
                {"role": "user", "content": prompt},
            ]
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        valid_indices = json.loads(raw)

        if not isinstance(valid_indices, list):
            return facts

        validated = [facts[i] for i in valid_indices if isinstance(i, int) and 0 <= i < len(facts)]
        rejected = len(facts) - len(validated)
        if rejected > 0:
            log(f"VALIDATION: {rejected}/{len(facts)} facts rejected as hallucinated")
        else:
            log(f"VALIDATION: all {len(facts)} facts confirmed")
        return validated

    except Exception as e:
        log(f"VALIDATION error (keeping all facts): {e}")
        return facts


# ─── Atomic Write ────────────────────────────────────────────────────────────


def write_file_atomic(path: Path, content: str):
    """Write content to file atomically using temp file + rename."""
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        os.rename(tmp_path, path)
    except Exception:
        os.close(fd) if not os.get_inheritable(fd) else None
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─── Incremental Graph Cache ────────────────────────────────────────────────


def update_graph_cache_incremental(note_id: str, content: str, relation: str):
    """Update graph cache incrementally after writing a note.
    Adds/updates outbound links and backlinks for the affected note."""
    try:
        if not GRAPH_CACHE_PATH.exists():
            return  # No cache to update; will be built on next full rebuild

        cache = json.loads(GRAPH_CACHE_PATH.read_text(encoding="utf-8"))
        outbound = cache.get("outbound", {})
        backlinks = cache.get("backlinks", {})

        # Determine which note ID was actually affected
        actual_id = note_id
        if relation.startswith("UPDATES:") or relation.startswith("EXTENDS:"):
            actual_id = relation.split(":", 1)[1].strip()

        # Parse new outbound links from content
        known_ids = set(outbound.keys())
        known_ids.add(actual_id)
        raw_links = re.findall(r'\[\[([^\]|]+?)(?:\|[^\]]+)?\]\]', content)
        new_links = list(dict.fromkeys(
            l.strip() for l in raw_links
            if len(l.strip()) < 60 and ' ' not in l.strip() and l.strip() in known_ids
        ))

        # Remove old backlinks from this note
        old_links = outbound.get(actual_id, [])
        for old_target in old_links:
            if old_target in backlinks:
                backlinks[old_target] = [s for s in backlinks[old_target] if s != actual_id]

        # Set new outbound links
        outbound[actual_id] = new_links

        # Add new backlinks
        for target in new_links:
            backlinks.setdefault(target, [])
            if actual_id not in backlinks[target]:
                backlinks[target].append(actual_id)

        # Write updated cache
        cache["outbound"] = outbound
        cache["backlinks"] = backlinks
        cache["note_count"] = len(outbound)
        cache["last_incremental"] = TODAY

        write_file_atomic(GRAPH_CACHE_PATH, json.dumps(cache, ensure_ascii=False))
        log(f"GRAPH incremental update: {actual_id} → {len(new_links)} links")

    except Exception as e:
        log(f"GRAPH incremental error: {e}")


# ─── Note Writing ────────────────────────────────────────────────────────────


def _inject_frontmatter_field(content: str, field: str, value: str) -> str:
    """Insert a field into existing YAML frontmatter (before the closing ---)."""
    if f"\n{field}:" in content:
        return content  # Already present
    # Insert before the closing ---
    return content.replace("\n---\n", f"\n{field}: {value}\n---\n", 1)


def _add_superseded_by(note_path: Path, successor_id: str):
    """Mark an existing note as superseded by adding superseded_by to frontmatter."""
    try:
        content = note_path.read_text(encoding="utf-8")
        if "superseded_by:" in content:
            return  # Already superseded
        updated = _inject_frontmatter_field(content, "superseded_by", successor_id)
        if updated != content:
            write_file_atomic(note_path, updated)
    except Exception as e:
        log(f"SUPERSEDE error on {note_path.stem}: {e}")


def write_note(note_id: str, content: str, relation: str):
    notes_dir = VAULT_NOTES_DIR

    if relation.startswith("UPDATES:"):
        target_id = relation.split(":", 1)[1].strip()
        target_path = notes_dir / f"{target_id}.md"
        if target_path.exists():
            # Inject relation metadata into the new content
            content = _inject_frontmatter_field(content, "relation", "updates")
            content = _inject_frontmatter_field(content, "parent_note", target_id)
            # Mark old note as superseded (before overwriting)
            _add_superseded_by(target_path, note_id)
            write_file_atomic(target_path, content)
            log(f"UPDATED  {target_id} (superseded by {note_id})")
            return
        log(f"UPDATES target not found ({target_id}), creating as NEW {note_id}")

    elif relation.startswith("EXTENDS:"):
        target_id = relation.split(":", 1)[1].strip()
        target_path = notes_dir / f"{target_id}.md"
        if target_path.exists():
            existing = target_path.read_text(encoding="utf-8")
            extension = f"\n\n---\n*Auto-extension {TODAY} (from: {note_id}):*\n\n{content}"
            write_file_atomic(target_path, existing + extension)
            log(f"EXTENDED {target_id} (by {note_id})")
            return
        log(f"EXTENDS target not found ({target_id}), creating as NEW {note_id}")

    # NEW note — add relation: new to frontmatter
    content = _inject_frontmatter_field(content, "relation", "new")
    note_path = notes_dir / f"{note_id}.md"
    write_file_atomic(note_path, content)
    log(f"NEW      {note_id}")


# ─── Source Chunk Storage ──────────────────────────────────────────────────


def save_source_chunk(note_id: str, relation: str, conversation: str):
    """Save the conversation excerpt that generated this note.
    Stored in _sources/{note_id}.md for retrieval injection."""
    if not SOURCE_CHUNKS_ENABLED:
        return
    try:
        SOURCE_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

        # Determine which note was actually affected
        actual_id = note_id
        if relation.startswith("UPDATES:") or relation.startswith("EXTENDS:"):
            actual_id = relation.split(":", 1)[1].strip()

        # Take the last SOURCE_CHUNK_MAX_CHARS of conversation as context
        chunk = conversation[-SOURCE_CHUNK_MAX_CHARS:].strip()
        if len(conversation) > SOURCE_CHUNK_MAX_CHARS:
            chunk = f"[...truncated...]\n\n{chunk}"

        chunk_path = SOURCE_CHUNKS_DIR / f"{actual_id}.md"

        # For EXTENDS: append to existing chunk file
        if relation.startswith("EXTENDS:") and chunk_path.exists():
            existing = chunk_path.read_text(encoding="utf-8")
            new_content = f"{existing}\n\n---\n*Extension source ({TODAY}):*\n\n{chunk}"
            write_file_atomic(chunk_path, new_content)
        else:
            header = f"---\nsource_for: {actual_id}\ncaptured: {TODAY}\nrelation: {relation}\n---\n\n"
            write_file_atomic(chunk_path, header + chunk)

        log(f"SOURCE saved: {actual_id} ({len(chunk)} chars)")
    except Exception as e:
        log(f"SOURCE error for {note_id}: {e}")


# ─── Ticket Processing ──────────────────────────────────────────────────────


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

    # Pre-query vault for related context (reduces duplicates, enables conflict detection)
    related_context = pre_query_vault(conversation, VAULT_NOTES_DIR)

    # Pre-build maps for post-generation link correction
    title_to_id = build_title_to_id_map(VAULT_NOTES_DIR)
    valid_ids = {f.stem for f in VAULT_NOTES_DIR.glob("*.md") if not f.name.startswith("._")}

    facts = extract_facts_with_llm(conversation, existing_notes, related_context)

    if not facts:
        log("No memorable facts extracted")
        _archive(ticket_path, session_id, turn_count)
        return

    log(f"Facts extracted: {len(facts)}")

    # Second-pass validation: reject hallucinated facts
    facts = validate_extracted_facts(facts, conversation)
    written = 0
    written_ids: list[str] = []
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
            written_ids.append(note_id)

            # Save source conversation chunk for retrieval injection
            save_source_chunk(note_id, relation, conversation)

            # Update graph cache incrementally
            update_graph_cache_incremental(note_id, content, relation)

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
    if written > 0:
        if len(written_ids) <= 3:
            label = " ".join(f"[[{i}]]" for i in written_ids)
        else:
            label = " ".join(f"[[{i}]]" for i in written_ids[:2]) + f" +{len(written_ids) - 2}"
        _write_memory_status(f"⚡ {label}")
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
