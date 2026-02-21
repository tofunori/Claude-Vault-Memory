# Claude Vault Memory

Persistent semantic memory for [Claude Code](https://claude.ai/claude-code). Every message you send is matched against a local vector index of your markdown notes. Relevant notes are injected into Claude's context before it replies. At the end of each session, an LLM pass extracts durable facts and writes them back as new notes.

---

## How it works

```
On every message
  UserPromptSubmit → vault_retrieve.py
    hybrid search: BM25 keyword + vector (Reciprocal Rank Fusion)
    reranking via Voyage AI rerank-2 (optional)
    graph traversal:
      load vault_graph_cache.json  (pre-built by vault_embed.py, ~0.5ms)
      inject backlinks of primary notes  (notes that link TO them)
      BFS depth 1+2: outbound links scored via Qdrant cosine similarity
    temporal decay: notes accessed recently rank higher (Ebbinghaus-inspired)
    confidence boost: confirmed notes score higher than experimental
    inject primary + connected notes into Claude context

During session  (proactive memory)
  Claude detects a durable fact
    vault_add_note(note_id, content)     # write note to vault via MCP
    vault_embed.py --note {note_id}      # index immediately in Qdrant
    note is retrievable in the next session

End of session  (safety net)
  Stop hook → enqueue.py  (< 100ms, non-blocking)
    re-enqueue if session grew by MIN_NEW_TURNS since last processing

Background worker  (launchd WatchPaths)
  process_queue.py
    extract LAST N turns (not first — avoids reading session-start summary)
    strip Claude Code UI tags (<system-reminder> etc.) from transcript
    pre-query vault before LLM (reduces duplicate extractions)
    LLM extraction → 0-15 atomic facts  (system message enforces JSON-only)
    _repair_json_newlines()  →  fix unescaped newlines in JSON strings
    sanitize_note_id()  →  guarantee valid kebab-case filename
    fix_wikilinks_in_content()  →  replace [[Full Title]] with [[note-id]] slugs
    semantic dedup via Qdrant (score > 0.85 → EXTENDS existing note)
    atomic write (temp file + rename — no corruption on crash)
    incremental upsert into Qdrant + update graph cache
    archive ticket with updated turn_count for accurate future re-enqueue
```

---

## Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Embeddings | Voyage AI `voyage-4-large` | #1 on MTEB multilingual retrieval (76.90), 22 languages |
| Keyword search | BM25 (rank-bm25) | Hybrid search via Reciprocal Rank Fusion |
| Reranking | Voyage AI `rerank-2` | Precision boost after RRF fusion (~100ms) |
| Vector store | Qdrant local mode | No server, on-disk HNSW, incremental upsert |
| LLM extraction | Fireworks `kimi-k2p5` | High extraction quality, runs offline after session |
| Note format | Markdown + YAML frontmatter | Obsidian-compatible, plain text, Zettelkasten |

---

## Versions

| Version | Change |
|---------|--------|
| v1 | Semantic retrieval at session start |
| v2 | Graph traversal — connected notes surfaced via `## Links` wiki-links |
| v3 | Proactive memory — Claude writes notes mid-session via MCP, indexed immediately |
| v4 | BFS 2-level graph traversal + backlinks + Qdrant scoring for connected notes |
| v5 | Link integrity: `sanitize_note_id`, `fix_wikilinks_in_content`, enforced `Topics:` section |
| v6 | Re-enqueue grown sessions, last-turns extraction, JSON repair, system message for Kimi |
| v7 | Hybrid search (BM25+vector RRF), temporal decay, confidence weighting, conflict detection, atomic writes, `vault_status.py`, `vault_reflect.py`, `vault_session_brief.py`, 40 unit tests |

---

## Installation

### Prerequisites

- Python 3.10+
- [Voyage AI API key](https://dash.voyageai.com) — embeddings and retrieval, 200M tokens free
- [Fireworks API key](https://fireworks.ai) — end-of-session LLM extraction, pay-per-use
- Claude Code with hooks enabled
- **For proactive memory (optional):** an MCP server exposing a `vault_add_note` tool

### Setup

```bash
git clone https://github.com/tofunori/Claude-Vault-Memory
cd Claude-Vault-Memory

# Copy and edit config
cp config.example.py config.py
# Edit config.py: set VAULT_NOTES_DIR, QDRANT_PATH, ENV_FILE, QUEUE_DIR, LOG_FILE, etc.

# Run interactive installer (installs packages, prompts for API keys, builds index)
bash install.sh
```

The installer will:
1. Install Python dependencies (`voyageai`, `qdrant-client`, `openai`, `rank-bm25`)
2. Prompt for `VOYAGE_API_KEY` and `FIREWORKS_API_KEY` and write them to `.env`
3. Build the initial Qdrant index from your vault notes

### API keys

Add to your `.env` file (path set in `config.py`):

```
VOYAGE_API_KEY=<your-voyage-key>
FIREWORKS_API_KEY=<your-fireworks-key>
```

### Claude Code hooks

Add to `~/.claude/settings.json`:

```json
"hooks": {
  "UserPromptSubmit": [
    {
      "matcher": "",
      "hooks": [
        {
          "type": "command",
          "command": "python3 /path/to/Claude-Vault-Memory/vault_retrieve.py"
        }
      ]
    }
  ],
  "Stop": [
    {
      "hooks": [
        {
          "type": "command",
          "command": "python3 /path/to/Claude-Vault-Memory/enqueue.py"
        }
      ]
    }
  ]
}
```

### Background worker (macOS)

```bash
cp launchd/com.example.vault-queue-worker.plist \
   ~/Library/LaunchAgents/com.yourname.vault-queue-worker.plist

# Edit all paths in the plist, then load
launchctl load ~/Library/LaunchAgents/com.yourname.vault-queue-worker.plist
```

The worker is triggered automatically by `launchd` when a new ticket appears in the queue directory.

---

## Proactive Memory

Claude writes notes **during** the session rather than waiting for the end-of-session extraction pass. Add to your `CLAUDE.md`:

```markdown
**Proactive memory:** without waiting for an explicit request, save immediately
when you identify something clearly durable:
- A technical decision made (config established, threshold validated, tool chosen)
- A solution found (working command, bug resolved)
- A workflow established (confirmed steps, functional pipeline)
- A fact learned about infrastructure (paths, APIs, models, services)

Do NOT save: casual conversation, intermediate debugging steps, reformulations.

Process:
1. Call vault_add_note(note_id, content) — complete note with YAML frontmatter
2. Run: python3 /path/to/vault_embed.py --note {note_id}
3. Confirm in one line: "saved: [[note_id]]"
```

---

## Note format

```markdown
---
description: One-sentence summary of the note (~150 chars)
type: concept|context|argument|decision|method|result|module
created: 2026-01-15
confidence: experimental|confirmed
---

# The note argues that X causes Y under condition Z

Body of the note: mechanism, evidence, reasoning.

## Links

- [[related-note-slug]]
- [[another-note]]

---

Topics:
- [[relevant-topic-map]]
```

The title should read as a proposition ("this note argues that..."), not a label.

---

## Tooling

```bash
# Build or rebuild the full index (also rebuilds BM25 index + graph cache)
python3 vault_embed.py

# Incremental update after editing a note
python3 vault_embed.py --note my-note-slug

# Update multiple notes
python3 vault_embed.py --notes note-a note-b note-c

# Test retrieval manually
echo '{"prompt":"your query here"}' | python3 vault_retrieve.py

# Health dashboard (note counts, confidence distribution, queue status)
python3 vault_status.py

# Vault reflection: detect stale/orphan notes, merge clusters (dry run by default)
python3 vault_reflect.py

# Session start brief (inject a summary of recent notes into context)
python3 vault_session_brief.py

# Run tests
python3 -m pytest tests/
```

---

## Configuration reference

All parameters live in `config.py` (never committed). See `config.example.py` for the full list with comments. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VAULT_NOTES_DIR` | — | Directory containing your `.md` notes |
| `QDRANT_PATH` | — | On-disk path for the Qdrant collection |
| `ENV_FILE` | — | Path to `.env` file with API keys |
| `QUEUE_DIR` | — | Directory for async session tickets |
| `LOG_FILE` | — | Path to `auto_remember.log` |
| `RETRIEVE_SCORE_THRESHOLD` | `0.60` | Minimum cosine score to surface a note |
| `RETRIEVE_TOP_K` | `3` | Maximum notes returned per query |
| `DEDUP_THRESHOLD` | `0.85` | Cosine score above which a new note extends an existing one |
| `MIN_TURNS` | `5` | Minimum session turns to enqueue for extraction |
| `MIN_NEW_TURNS` | `10` | New turns required to re-process a grown session |
| `BM25_ENABLED` | `True` | Enable hybrid BM25 + vector search (RRF) |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `RERANK_ENABLED` | `True` | Enable Voyage AI reranking after RRF fusion |
| `DECAY_ENABLED` | `True` | Enable temporal decay for retrieval scoring |
| `DECAY_HALF_LIFE_DAYS` | `90` | Days until a note's score is halved without retrieval |
| `CONFIDENCE_BOOST` | `1.2` | Score multiplier for `confidence: confirmed` notes |
| `MAX_CODE_BLOCK_CHARS` | `500` | Max chars per code block in transcript before truncation |
| `VOYAGE_EMBED_MODEL` | `voyage-4-large` | Voyage AI model for embeddings |
| `BFS_DEPTH` | `2` | Graph traversal depth for connected notes |
| `GRAPH_CACHE_PATH` | — | Path to `vault_graph_cache.json` (auto-generated) |
| `BM25_INDEX_PATH` | — | Path to persistent BM25 index (auto-generated) |

---

## Logs

```
[2026-01-15] EMBED_INDEX upserted: 124 notes → /path/to/vault_qdrant
[2026-01-16] RETRIEVE query=52c → 2 notes + 1 graph (threshold 0.6)
[2026-01-16] ENQUEUED session=a3f1c9b2 turns=18
[2026-01-16] RE-ENQUEUE session=a3f1c9b2 (grew 18→45 turns, +27 new)
[2026-01-16] Conversation truncated: last 123/220 turns (39941 chars)
[2026-01-16] NEW      my-new-note
[2026-01-16] DEDUP: candidate-note → EXTENDS:existing-note
[2026-01-16] EXTENDED existing-note
[2026-01-16] ARCHIVED session=a3f1c9b2
```

---

## License

MIT
