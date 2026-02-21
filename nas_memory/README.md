# NAS Memory V1.2

API + worker NAS-hosted for shared memory across Claude Code, Codex, and OpenClaw.

## V1 scope

- No deep refactor of `vault_retrieve.py`, `process_queue.py`, `vault_embed.py`.
- `memory-api` and `worker` run on NAS host (native Python).
- Qdrant runs in embedded/local mode (`QdrantClient(path=...)`) via existing scripts.
- Clients are thin: they only call `/retrieve` and `/events`.

## Endpoints

- `POST /retrieve`
- `POST /events`
- `GET /health`
- `POST /admin/reindex`

Auth is enforced with `Authorization: Bearer <MEMORY_API_TOKEN>`.

## Environment variables

- `MEMORY_API_TOKEN` (recommended)
- `MEMORY_ALLOWED_IPS` (comma-separated allowlist)
- `MEMORY_API_HOST` (default `0.0.0.0`)
- `MEMORY_API_PORT` (default `8766`)
- `MEMORY_ROOT` (default `/volume1/Services/memory`)
- `MEMORY_STATE_DIR` (default `$MEMORY_ROOT/state`)
- `MEMORY_REBUILD_FULL_AFTER_WRITE` (default `true`)
- `MEMORY_CORE_CONFIG` (optional explicit path to root `config.py`)
- `MEMORY_TURN_LIVE_CADENCE` (default `5`)
- `MEMORY_STAGING_TTL_HOURS` (default `24`)
- `MEMORY_STAGING_WINDOW` (default `12`)
- `MEMORY_LIVE_EXTRACT_TIMEOUT` (default `8`)
- `MEMORY_LIVE_MAX_CANDIDATES` (default `3`)
- `MEMORY_RETRIEVE_EXPERIMENTAL_MAX` (default `2`)
- `MEMORY_EXPERIMENTAL_SCORE_CAP` (default `0.35`)
- `MEMORY_PROMOTION_MIN_EVIDENCE` (default `2`)
- `MEMORY_BACKPRESSURE_QUEUE_THRESHOLD` (default `100`)
- `MEMORY_PROFILE_ENABLE` (default `true`)
- `MEMORY_PROFILE_MAX_ITEMS` (default `12`)
- `MEMORY_PROFILE_SCORE_CAP` (default `0.55`)
- `MEMORY_PROFILE_COMPACT_INTERVAL_MIN` (default `60`)
- `MEMORY_FORGET_SOFT_DELETE` (default `true`)
- `MEMORY_PROFILE_EXTRACT_TIMEOUT` (default `12`)
- `MEMORY_PROFILE_MIN_EVIDENCE` (default `2`)
- `MEMORY_PROFILE_DYNAMIC_HOURS` (default `24`)
- `MEMORY_ADMIN_GRAPH_TIMEOUT` (default `20`)
- `MEMORY_RELATION_ENABLE` (default `true`)
- `MEMORY_RELATION_WRITE` (default `false`, shadow mode)
- `MEMORY_RELATION_BATCH_MAX_PAIRS` (default `800`)
- `MEMORY_RELATION_MIN_CONFIDENCE` (default `0.72`)
- `MEMORY_RELATION_MAX_NEW_EDGES_PER_RUN` (default `120`)
- `MEMORY_RELATION_COMPACT_INTERVAL_MIN` (default `60`)
- `MEMORY_RELATION_LLM_TIMEOUT` (default `10`)

## Install

```bash
cd /volume1/Services

git clone <your-repo-url> memory
cd memory
python3 -m venv .venv
source .venv/bin/activate
pip install -r nas_memory/requirements.txt
```

Ensure root `config.py` is present and points to NAS paths.

## Run API

```bash
./nas_memory/run-api.sh
```

## Run worker

```bash
./nas_memory/run-worker.sh
```

Worker is single-writer with lockfile (`worker.lock`).

## systemd user autostart (recommended)

```bash
mkdir -p ~/.config/systemd/user
cp nas_memory/systemd/memory-api.service ~/.config/systemd/user/
cp nas_memory/systemd/memory-worker.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now memory-api.service memory-worker.service
```

To start user services at boot without interactive login:

```bash
sudo loginctl enable-linger "$USER"
```

## Quick checks

```bash
curl -sS -H "Authorization: Bearer $MEMORY_API_TOKEN" \
  "http://127.0.0.1:${MEMORY_API_PORT:-8766}/health"
```

```bash
curl -sS -X POST \
  -H "Authorization: Bearer $MEMORY_API_TOKEN" \
  -H "Content-Type: application/json" \
  "http://127.0.0.1:${MEMORY_API_PORT:-8766}/events" \
  -d '{
    "event_type":"turn",
    "agent":"codex",
    "session_id":"demo",
    "payload":{
      "turn_index": 1,
      "role": "user",
      "text": "On garde ce seuil à 0.42",
      "cwd": "/workspace/project",
      "ts": "2026-02-21T20:00:00+00:00"
    }
  }'
```

`turn` events feed live staging memory. Experimental staging is session-scoped and expires automatically.

## Burn-in 72h (mixed + strict)

Pré-vol 30 min:

```bash
python3 nas_memory/burnin/collector.py \
  --duration-hours 0.5 \
  --mode mixed \
  --gate strict
```

Run complet 72h:

```bash
bash nas_memory/burnin/run-72h.sh
```

Rapport:

```bash
python3 nas_memory/burnin/report.py \
  --run-dir /volume1/Services/memory/state/burnin/<run_id> \
  --gate strict
```

Les artefacts sont écrits dans `/volume1/Services/memory/state/burnin/<run_id>/`:
`health_samples.jsonl`, `retrieve_samples.jsonl`, `worker_live_samples.jsonl`,
`synthetic_trace.jsonl`, `summary.json`, `summary.md`, `passfail.json`.

## V1.3 profile layer (non-breaking)

V1.3 ajoute une couche de mémoire haut niveau (profil global, versioning, forget soft-delete)
sans casser les endpoints existants.

Nouveaux endpoints admin:

- `POST /admin/memory/forget`
- `POST /admin/memory/restore`
- `POST /admin/memory/upsert`
- `GET /admin/profile`
- `POST /admin/profile/compact`
- `POST /admin/relations/compact`
- `GET /admin/relations/stats`
- `GET /admin/graph`
- `GET /admin/graph/ui`

Exemples:

```bash
curl -sS -X POST \
  -H "Authorization: Bearer $MEMORY_API_TOKEN" \
  -H "Content-Type: application/json" \
  "http://127.0.0.1:${MEMORY_API_PORT:-8766}/admin/memory/upsert" \
  -d '{"fact_text":"On garde FastAPI + SQLite WAL sur NAS","fact_type":"decision","scope":"global_profile","confidence":0.82,"actor":"admin"}'
```

```bash
curl -sS -H "Authorization: Bearer $MEMORY_API_TOKEN" \
  "http://127.0.0.1:${MEMORY_API_PORT:-8766}/admin/profile"
```

Graph JSON:

```bash
curl -sS -H "Authorization: Bearer $MEMORY_API_TOKEN" \
  "http://127.0.0.1:${MEMORY_API_PORT:-8766}/admin/graph"
```

Relation compaction (queued event):

```bash
curl -sS -X POST \
  -H "Authorization: Bearer $MEMORY_API_TOKEN" \
  -H "Content-Type: application/json" \
  "http://127.0.0.1:${MEMORY_API_PORT:-8766}/admin/relations/compact" \
  -d '{"actor":"admin"}'
```

Relation stats:

```bash
curl -sS -H "Authorization: Bearer $MEMORY_API_TOKEN" \
  "http://127.0.0.1:${MEMORY_API_PORT:-8766}/admin/relations/stats"
```

Graph UI:

```bash
open "http://127.0.0.1:${MEMORY_API_PORT:-8766}/admin/graph/ui?token=$MEMORY_API_TOKEN"
```

`/admin/graph` retourne une vue unifiée:

- Notes markdown depuis `vault_graph_cache.json` (`links_to` via `outbound`).
- Nœuds mémoire V1.3 (`memory_nodes`) et relations (`memory_edges`).
- Ponts déterministes `references_note` quand `fact_text` contient `[[note_id]]`.
- Ponts alias contrôlés via table `note_aliases` (`alias -> note_id`).

La page `/admin/graph/ui` inclut:

- force-graph interactif,
- filtres nœuds/relations,
- masquage `forgotten`/`superseded`,
- recherche, centrage, isolation de composante, reset layout.

Compaction horaire (systemd user):

```bash
mkdir -p ~/.config/systemd/user
cp nas_memory/systemd/memory-profile-compact.service ~/.config/systemd/user/
cp nas_memory/systemd/memory-profile-compact.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now memory-profile-compact.timer
```

Compaction relationnelle horaire (mode shadow/activation):

```bash
mkdir -p ~/.config/systemd/user
cp nas_memory/systemd/memory-relation-compact.service ~/.config/systemd/user/
cp nas_memory/systemd/memory-relation-compact.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now memory-relation-compact.timer
```

## Codex integration (MCP bridge)

Use `nas_memory/mcp_memory` to expose memory API as MCP tools (`memory_retrieve`, `memory_enqueue_event`).
Deploy the bridge on NAS and add this to `~/.codex/config.toml`:

```toml
[mcp_servers.memory]
type = "stdio"
command = "ssh"
args = ["rorqual", "bash '/volume1/Services/mcp/memory/run-mcp.sh'"]
```

For memory-only mode in Codex, remove/disable `[mcp_servers.vault]` so agents do not bypass `memory-api`.

## Claude Code memory-only mode

- Keep only thin memory hooks:
  - `UserPromptSubmit` -> `python3 ~/.claude/hooks/vault_retrieve.py`
  - `Stop` -> `python3 ~/.claude/hooks/enqueue.py`
- Remove/disable local embedding hooks (`vault_embed_if_note.sh`, watcher launch agents).
- Set `MEMORY_API_URL` to NAS (`http://rorqual:8876` or LAN IP) and keep `MEMORY_API_TOKEN`.

## OpenClaw memory-only mode

OpenClaw should use:
- retrieve: `POST http://<nas>:8876/retrieve`
- write queue: `POST http://<nas>:8876/events`

Do not write notes/index directly from OpenClaw clients.
