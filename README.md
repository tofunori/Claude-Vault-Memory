# claude-vault-memory

Persistent semantic memory for [Claude Code](https://claude.ai/claude-code). Every message you send is matched against a local vector index of your markdown notes. Relevant notes are injected into Claude's context before it replies. At the end of each session, an LLM pass extracts durable facts and writes them back as new notes.

---

## How it works

```
On every message
  UserPromptSubmit → vault_retrieve.py
    embed(message) via Cohere API
    HNSW search in local Qdrant → top notes (score > 0.60)
    inject matched notes into Claude context

During session  (v3 — proactive memory)
  Claude detects a durable fact
    vault_add_note(note_id, content)     # write note to vault via MCP
    vault_embed.py --note {note_id}      # index immediately in Qdrant
    note is retrievable in the next session

End of session  (safety net)
  Stop hook → enqueue.py  (< 100ms, non-blocking)

Background worker  (launchd WatchPaths)
  process_queue.py
    parse session transcript
    LLM extraction → 0-15 atomic facts
    semantic dedup via Qdrant (score > 0.85 → EXTENDS existing note)
    write markdown notes
    incremental upsert into Qdrant
```

---

## Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Embeddings | Cohere `embed-multilingual-v3.0` | Multilingual, 1024 dims, free tier |
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

---

## Installation

### Prerequisites

- Python 3.10+
- [Cohere API key](https://dashboard.cohere.com/api-keys) — embeddings and retrieval, free tier: 1000 calls/month
- [Fireworks API key](https://fireworks.ai) — end-of-session LLM extraction, pay-per-use
- Claude Code with hooks enabled
- **For v3 proactive memory (optional):** an MCP server exposing a `vault_add_note` tool, so Claude can write notes mid-session without waiting for the end-of-session pass

### Setup

```bash
git clone https://github.com/tofunori/claude-vault-memory
cd claude-vault-memory

# Copy and edit config
cp config.example.py config.py
# Edit config.py: set VAULT_NOTES_DIR, QDRANT_PATH, ENV_FILE, QUEUE_DIR, LOG_FILE

# Run interactive installer (installs packages, prompts for API keys, builds index)
bash install.sh
```

The installer will:
1. Install Python dependencies (`cohere`, `qdrant-client`, `openai`)
2. Prompt for `COHERE_API_KEY` and `FIREWORKS_API_KEY` and write them to `.env`
3. Build the initial Qdrant index from your vault notes

### API keys

Add to your `.env` file (path set in `config.py`):

```
COHERE_API_KEY=<your-cohere-key>
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
          "command": "python3 /path/to/claude-vault-memory/vault_retrieve.py"
        }
      ]
    }
  ],
  "Stop": [
    {
      "hooks": [
        {
          "type": "command",
          "command": "python3 /path/to/claude-vault-memory/enqueue.py"
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

The worker is triggered automatically by `launchd` when a new ticket appears in the queue directory. It runs `process_queue.py`, which reads the session transcript, calls the Fireworks API, and writes the extracted notes to your vault.

---

## v3 — Proactive Memory

In v3, Claude writes notes **during** the session rather than waiting for the end-of-session extraction pass. This requires an MCP server with a `vault_add_note` tool.

Add the following to your `CLAUDE.md` to enable the behavior:

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

The reason `vault_embed.py` must be called explicitly: the `PostToolUse` hook only fires on native `Write`/`Edit` tool calls, not on MCP tool calls like `vault_add_note`.

---

## Note format

Notes are plain markdown files with YAML frontmatter. The `description` field is used as the embedding text alongside the note body.

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
```

The title should read as a proposition ("this note argues that..."), not a label. This makes retrieval more precise and forces atomic thinking.

---

## Configuration reference

All parameters live in `config.py` (never committed):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VAULT_NOTES_DIR` | — | Directory containing your `.md` notes |
| `QDRANT_PATH` | — | On-disk path for the Qdrant collection |
| `ENV_FILE` | — | Path to `.env` file with API keys |
| `QUEUE_DIR` | — | Directory for async session tickets |
| `LOG_FILE` | — | Path to `auto_remember.log` |
| `RETRIEVE_SCORE_THRESHOLD` | `0.60` | Minimum cosine score to surface a note |
| `RETRIEVE_TOP_K` | `3` | Maximum number of notes returned per query |
| `DEDUP_THRESHOLD` | `0.85` | Cosine score above which a new note extends an existing one |
| `MIN_QUERY_LENGTH` | `20` | Minimum message length in chars to trigger retrieval |
| `MIN_TURNS` | `5` | Minimum session turns to enqueue for extraction |
| `COHERE_BATCH_SIZE` | `96` | Batch size for Cohere embedding calls |

---

## Usage

```bash
# Build or rebuild the full index
python3 vault_embed.py

# Incremental update after editing a note
python3 vault_embed.py --note my-note-slug

# Update multiple notes
python3 vault_embed.py --notes note-a note-b note-c

# Test retrieval manually
echo '{"prompt":"your query here"}' | python3 vault_retrieve.py
```

---

## Logs

All events are appended to `auto_remember.log`:

```
[2026-01-15] EMBED_INDEX upserted: 124 notes → /path/to/vault_qdrant
[2026-01-16] RETRIEVE query=52c → 2 notes + 1 graph (threshold 0.6)
[2026-01-16] ENQUEUED session=a3f1c9b2 turns=18
[2026-01-16] PROCESSING session=a3f1c9b2
[2026-01-16] NEW      my-new-note
[2026-01-16] DEDUP: candidate-note → EXTENDS:existing-note
[2026-01-16] ARCHIVED session=a3f1c9b2
```

---

## License

MIT
