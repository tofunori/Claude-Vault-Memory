# Claude Vault Memory

Persistent semantic memory for [Claude Code](https://claude.ai/claude-code). Every message you send is matched against a local knowledge graph of your markdown notes. Relevant notes are injected into Claude's context before it replies. At the end of each session, an LLM pass extracts durable facts and writes them back as typed, versioned atomic notes.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  SESSION START                                                       │
│  vault_session_brief.py ──────────────────────────────────────────► │
│    scan notes/ for confirmed preferences, recent decisions,          │
│    active project context → inject brief into Claude context         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EVERY USER MESSAGE                                                  │
│  vault_retrieve.py  (UserPromptSubmit hook, ~200ms)                  │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐                               │
│  │  BM25 search │    │ Vector search│  ◄── Voyage AI voyage-4-large  │
│  │  (keyword)   │    │  (semantic)  │      embedded query            │
│  └──────┬───────┘    └──────┬───────┘                               │
│         │                   │                                        │
│         └────────┬──────────┘                                        │
│                  ▼                                                    │
│         Reciprocal Rank Fusion (RRF)                                 │
│                  │                                                    │
│                  ▼                                                    │
│         Voyage AI Reranker  ◄── rerank-2 (precision layer)           │
│                  │                                                    │
│                  ▼                                                    │
│     Apply temporal decay × confidence boost                          │
│                  │                                                    │
│         ┌────────┴────────┐                                          │
│         │  Primary notes  │  ← top 3                                 │
│         │  + source chunk │  ← conversation excerpt injected         │
│         └────────┬────────┘                                          │
│                  │  BFS graph traversal (depth 2)                    │
│                  ▼  backlinks + outbound links via graph cache        │
│         ┌────────────────┐                                           │
│         │ Connected notes│  ← up to 5                                │
│         └────────────────┘                                           │
│                  │                                                    │
│                  ▼                                                    │
│       Inject into Claude context                                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SESSION END                                                         │
│  enqueue.py  (Stop hook, < 100ms, non-blocking)                      │
│    write ticket → queue/{session_id}.json                            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                  launchd WatchPaths triggers worker
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BACKGROUND WORKER                                                   │
│  process_queue.py                                                    │
│                                                                      │
│  parse transcript ──► smart truncation (filter noise, cap code)      │
│         │                                                            │
│         ▼                                                            │
│  pre-query vault ──► embed conversation → find 5 related notes       │
│         │             inject as conflict context into LLM prompt      │
│         ▼                                                            │
│  LLM extraction (Fireworks kimi-k2p5)                               │
│    → 0–15 atomic facts with typed relations:                         │
│      NEW          → brand new fact                                   │
│      EXTENDS:id   → adds detail to existing note                     │
│      UPDATES:id   → replaces existing note (marks old as             │
│                     superseded_by: new-id)                           │
│         │                                                            │
│         ▼                                                            │
│  validation pass ──► second LLM call rejects hallucinated facts      │
│         │                                                            │
│         ▼                                                            │
│  semantic dedup ──► score > 0.85 → auto-convert NEW → EXTENDS        │
│         │                                                            │
│         ▼                                                            │
│  write note (atomic) ──► inject relation/parent_note in frontmatter  │
│  save source chunk  ──► _sources/{note-id}.md (for later retrieval)  │
│  update graph cache ──► incremental outbound + backlinks             │
│  upsert Qdrant      ──► async vault_embed.py --note {id}             │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                  weekly cron (Sundays)
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MAINTENANCE                                                         │
│  vault_reflect.py                                                    │
│                                                                      │
│  • Archive expired notes (forget_after date or type-based TTL)       │
│    → move to _archived/ + remove from Qdrant                         │
│  • Detect semantic clusters (cosine > 0.82) → suggest merges         │
│  • Flag stale notes (not retrieved in 180+ days)                     │
│  • Flag orphan notes (no incoming or outgoing links)                 │
└─────────────────────────────────────────────────────────────────────┘

Data stores
───────────
notes/                  Atomic markdown notes (.md, YAML frontmatter)
notes/_sources/         Conversation excerpts that generated each note
notes/_archived/        Expired/forgotten notes (soft delete)
vault_qdrant/           On-disk Qdrant HNSW index (Voyage AI vectors)
vault_bm25_index.json   Persistent BM25 keyword index
vault_graph_cache.json  Pre-computed outbound links + backlinks
```

---

## Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Embeddings | Voyage AI `voyage-4-large` | #1 on MTEB multilingual retrieval (76.90), 22 languages |
| Reranking | Voyage AI `rerank-2` | Precision layer after RRF fusion (+100ms, significant quality gain) |
| Vector store | Qdrant local mode | No server, on-disk HNSW, incremental upsert |
| Keyword search | BM25 (pure Python) | Hybrid search via RRF — catches exact terms vectors miss |
| LLM extraction | Fireworks `kimi-k2p5` | High extraction quality, runs async after session |
| Note format | Markdown + YAML frontmatter | Obsidian-compatible, plain text, git-trackable |

---

## Versions

| Version | Change |
|---------|--------|
| v1 | Semantic retrieval at session start |
| v2 | Graph traversal — connected notes surfaced via `## Links` wiki-links |
| v3 | Proactive memory — Claude writes notes mid-session via MCP |
| v4 | BFS 2-level graph traversal + backlinks + Qdrant scoring |
| v5 | Link integrity: `sanitize_note_id`, `fix_wikilinks_in_content` |
| v6 | Hybrid search (BM25+vector+RRF), temporal decay, confidence boost, conflict detection, observability |
| v7 | Voyage reranking, session brief, extraction validation, persistent BM25 index, reflector |
| v8 | Typed relations (relation/parent_note/superseded_by), source chunk storage, smart forgetting (forget_after + type TTL) |

---

## Installation

### Prerequisites

- Python 3.10+
- [Voyage AI API key](https://dash.voyageai.com) — embeddings and retrieval, 200M tokens free
- [Fireworks API key](https://fireworks.ai) — end-of-session LLM extraction, pay-per-use
- Claude Code with hooks enabled
- **For v3 proactive memory (optional):** an MCP server exposing a `vault_add_note` tool, so Claude can write notes mid-session without waiting for the end-of-session pass

### Setup

```bash
git clone https://github.com/tofunori/Claude-Vault-Memory
cd Claude-Vault-Memory

# Copy and edit config
cp config.example.py config.py
# Edit config.py: set VAULT_NOTES_DIR, QDRANT_PATH, ENV_FILE, QUEUE_DIR, LOG_FILE

# Run interactive installer (installs packages, prompts for API keys, builds index)
bash install.sh
```

The installer will:
1. Install Python dependencies (`voyageai`, `qdrant-client`, `openai`)
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

# Auto-populated by the system (do not edit manually):
relation: new|updates|extends        # how this note relates to the vault
parent_note: old-note-id             # set when relation=updates|extends
superseded_by: newer-note-id         # set on the old note when overwritten
forget_after: 2026-04-01             # optional: auto-archive after this date
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

All parameters live in `config.py` (never committed — copy from `config.example.py`):

**Paths**

| Parameter | Description |
|-----------|-------------|
| `VAULT_NOTES_DIR` | Directory containing your `.md` notes |
| `QDRANT_PATH` | On-disk path for the Qdrant collection |
| `ENV_FILE` | Path to `.env` file with API keys |
| `QUEUE_DIR` | Directory for async session tickets |
| `LOG_FILE` | Path to `auto_remember.log` |
| `GRAPH_CACHE_PATH` | Path to `vault_graph_cache.json` (auto-generated) |
| `BM25_INDEX_PATH` | Path to `vault_bm25_index.json` (auto-generated) |

**Retrieval**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RETRIEVE_SCORE_THRESHOLD` | `0.60` | Minimum cosine score to surface a note |
| `RETRIEVE_TOP_K` | `3` | Maximum primary notes returned per query |
| `MIN_QUERY_LENGTH` | `20` | Minimum message length (chars) to trigger retrieval |
| `BM25_ENABLED` | `True` | Enable BM25 keyword search (hybrid) |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `RERANK_ENABLED` | `True` | Enable Voyage AI reranking (adds ~100ms, improves precision) |
| `RERANK_MODEL` | `rerank-2` | Voyage reranking model |
| `RERANK_CANDIDATES` | `10` | Candidates fed to reranker |
| `CONFIDENCE_BOOST` | `1.2` | Score multiplier for `confidence: confirmed` notes |
| `DECAY_ENABLED` | `True` | Enable temporal decay (recently accessed notes rank higher) |
| `DECAY_HALF_LIFE_DAYS` | `90` | Days until retrieval score halves |
| `DECAY_FLOOR` | `0.3` | Minimum decay factor |
| `MAX_SECONDARY` | `5` | Max connected notes surfaced via BFS graph traversal |
| `MAX_BACKLINKS_PER_NOTE` | `3` | Max backlinks injected per primary note |
| `BFS_DEPTH` | `2` | BFS depth for graph traversal |

**Extraction**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FIREWORKS_MODEL` | `kimi-k2p5` | LLM for fact extraction (OpenAI-compatible API) |
| `DEDUP_THRESHOLD` | `0.85` | Cosine score to auto-convert NEW → EXTENDS existing note |
| `MIN_TURNS` | `5` | Minimum session turns to enqueue for extraction |
| `VALIDATION_ENABLED` | `True` | Second LLM pass to reject hallucinated extractions |
| `MAX_CODE_BLOCK_CHARS` | `500` | Max chars per code block in transcript (rest truncated) |

**Source chunks**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SOURCE_CHUNKS_ENABLED` | `True` | Save conversation excerpts that generated each note |
| `SOURCE_CHUNKS_DIR` | `notes/_sources` | Directory for source chunk files |
| `SOURCE_CHUNK_MAX_CHARS` | `2000` | Max chars of conversation saved per note |
| `SOURCE_INJECT_MAX_CHARS` | `800` | Max chars of source context injected during retrieval |

**Smart forgetting**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FORGET_ARCHIVE_DIR` | `notes/_archived` | Destination for archived (expired) notes |
| `FORGET_DEFAULT_TTL_DAYS` | `{}` | Per-type TTL in days, e.g. `{"context": 90}` |

**Maintenance**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `REFLECT_MIN_NOTES` | `30` | Minimum vault size to run reflection |
| `REFLECT_CLUSTER_THRESHOLD` | `0.82` | Cosine similarity to flag notes as mergeable cluster |
| `REFLECT_STALE_DAYS` | `180` | Days without retrieval to flag a note as stale |

---

## Usage

```bash
# Build or rebuild the full index (+ BM25 index + graph cache)
python3 vault_embed.py

# Incremental update after editing a note
python3 vault_embed.py --note my-note-slug

# Test retrieval manually
echo '{"prompt":"your query here"}' | python3 vault_retrieve.py

# Vault health dashboard
python3 vault_status.py

# Weekly maintenance (dry run)
python3 vault_reflect.py

# Weekly maintenance (apply: archive expired, mark stale)
python3 vault_reflect.py --apply
```

**Smart forgetting** — mark a note as temporary by adding `forget_after` to its frontmatter:

```yaml
---
description: Current sprint context
type: context
forget_after: 2026-04-01
---
```

`vault_reflect --apply` will move it to `_archived/` and remove it from Qdrant once the date passes.

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
