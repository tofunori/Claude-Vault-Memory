# claude-vault-memory

**Supermemory pour Claude Code** — index vectoriel sémantique de vos notes, retrieval actif à chaque message, déduplication intelligente.

## Architecture

```
Session Claude Code
  └─ UserPromptSubmit → vault_retrieve.py
       └─ embed(message) via Cohere API
       └─ recherche HNSW Qdrant local → top-3 notes (score > 0.60)
       └─ output → injecté dans contexte Claude

Fin de session
  └─ Stop → enqueue.py (< 100ms)

Background worker (launchd WatchPaths)
  └─ process_queue.py
       └─ extraction LLM → 0-5 faits atomiques
       └─ check_semantic_dup() — dédup Qdrant (score > 0.85 → EXTENDS)
       └─ écriture notes markdown
       └─ upsert individuel dans Qdrant via vault_embed.py
```

## Stack

| Composant | Choix | Raison |
|-----------|-------|--------|
| Embeddings | Cohere `embed-multilingual-v3.0` | Multilingue (FR/EN), 1024 dims |
| Vector DB | Qdrant **mode local** (disque) | Zéro serveur, HNSW, incrémental |
| LLM extraction | Fireworks kimi-k2 | Qualité maximale, latence sans importance |
| Format notes | Markdown + YAML frontmatter | Compatible Obsidian, Zettelkasten |

## Installation

### Prérequis

- Python 3.10+
- Clé API Cohere (gratuit jusqu'à 1000 appels/mois) : [dashboard.cohere.com](https://dashboard.cohere.com/api-keys)
- Claude Code configuré avec hooks

### Étapes

```bash
# 1. Cloner
git clone https://github.com/yourname/claude-vault-memory
cd claude-vault-memory

# 2. Copier config
cp config.example.py config.py
# → Éditez config.py avec vos chemins

# 3. Lancer l'installation interactive
bash install.sh
```

### Configuration manuelle

Éditez `config.py` (jamais commité) :

```python
VAULT_NOTES_DIR = "/home/yourname/notes"
QDRANT_PATH = "/home/yourname/.claude/hooks/vault_qdrant"
ENV_FILE = "/home/yourname/.claude/hooks/.env"
```

Ajoutez dans votre `.env` :
```
COHERE_API_KEY=<votre-clé>
```

### settings.json Claude Code

Ajoutez dans `~/.claude/settings.json` :

```json
"UserPromptSubmit": [
  {
    "matcher": "",
    "hooks": [
      {
        "type": "command",
        "command": "python3 /path/to/hooks/vault_retrieve.py"
      }
    ]
  }
],
"Stop": [
  {
    "hooks": [
      {
        "type": "command",
        "command": "python3 /path/to/hooks/enqueue.py"
      }
    ]
  }
]
```

### launchd (macOS) — worker background

```bash
# Copier et adapter le plist
cp launchd/com.example.vault-queue-worker.plist \
   ~/Library/LaunchAgents/com.yourname.vault-queue-worker.plist

# Éditer les chemins dans le plist, puis charger
launchctl load ~/Library/LaunchAgents/com.yourname.vault-queue-worker.plist
```

## Usage

### Build initial de l'index

```bash
python3 vault_embed.py
# → EMBED_INDEX upserted: 119 notes → /path/to/vault_qdrant
```

### Test du retrieval

```bash
echo '{"prompt":"décision seuil qualité albédo glacier MODIS"}' | python3 vault_retrieve.py
# → === Notes vault pertinentes ===
# → [[decision-seuil-qa-albedo]] (decision, 78%) — Seuil de qualité MODIS albédo...
```

### Mise à jour incrémentale

```bash
# Après modification d'une note
python3 vault_embed.py --note ma-note-modifiee

# Plusieurs notes
python3 vault_embed.py --notes note-a note-b note-c
```

## Format des notes

Chaque note markdown doit avoir un frontmatter YAML :

```markdown
---
description: Description courte (~150 chars)
type: concept|context|argument|decision|method|result|module|section
created: 2026-02-20
confidence: experimental|confirmed
---

# La note affirme que X fait Y

Corps de la note...

## Connexions

- [[note-liee]]
```

## Seuils configurables

Dans `config.py` :

| Paramètre | Défaut | Rôle |
|-----------|--------|------|
| `RETRIEVE_SCORE_THRESHOLD` | 0.60 | Score min pour afficher une note |
| `RETRIEVE_TOP_K` | 3 | Nombre max de notes retournées |
| `DEDUP_THRESHOLD` | 0.85 | Score min pour détecter un doublon |
| `MIN_QUERY_LENGTH` | 20 | Longueur min d'un message (chars) |

## Logs

Tous les événements sont loggés dans `auto_remember.log` :

```
[2026-02-20] EMBED_INDEX upserted: 119 notes
[2026-02-20] RETRIEVE query=45c → 2 notes (seuil 0.6)
[2026-02-20] DEDUP: nouvelle-note → EXTENDS:note-existante
[2026-02-20] ENQUEUED session=abc123 turns=12
[2026-02-20] NEW      ma-nouvelle-note
```

## Licence

MIT
