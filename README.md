# Memo-Rize: A Computational Memory Layer for Multi-Agent Systems

Unified computational memory layer for **Claude Code, Codex, and OpenClaw**.

Ce repo a été réorganisé pour séparer clairement:

- `production NAS` (actif)
- `legacy local` (archivé)

## Ce qui est actif en production

### 1) Couche NAS (orchestration moderne)

- `nas_memory/` contient l'API, le worker, la DB SQLite WAL, la sécurité, la couche profil/relations, le graph UI, et le burn-in.
- Entrées principales:
  - `nas_memory/api.py`
  - `nas_memory/worker.py`
  - `nas_memory/db.py`
  - `nas_memory/graph_view.py`
  - `nas_memory/relation_linker.py`

### 2) Moteur core canonique

Le moteur core est maintenant dans `nas_memory/core/`:

- `nas_memory/core/vault_retrieve.py`
- `nas_memory/core/process_queue.py`
- `nas_memory/core/vault_embed.py`
- `nas_memory/core/runtime_config.py` (env-first + override fichier optionnel)

Pour compatibilite, les scripts root existent encore comme **shims**:

- `vault_retrieve.py`
- `process_queue.py`
- `vault_embed.py`

Ils deleguent vers `nas_memory/core/*` et conservent la CLI historique.

## Ce qui est archivé (legacy local)

Le pipeline local historique a été déplacé dans `legacy_local/`:

- `legacy_local/enqueue.py`
- `legacy_local/vault_session_brief.py`
- `legacy_local/vault_status.py`
- `legacy_local/vault_reflect.py`
- `legacy_local/install.sh`
- `legacy_local/launchd/`
- `legacy_local/README_LOCAL_LEGACY.md` (ancienne doc complète)

Ce dossier n'est plus le chemin recommandé pour la prod NAS.

## Structure du repo

- `nas_memory/` → stack multi-agent active (prod)
- `nas_memory/core/` → moteur core canonique (retrieval/consolidation/index)
- `vault_retrieve.py`, `process_queue.py`, `vault_embed.py` → shims de compatibilite
- `tests/` → tests core historiques
- `legacy_local/` → ancien mode local Claude-only

## Documentation à suivre

- Mode prod NAS: `nas_memory/README.md`
- Burn-in et gates: `nas_memory/burnin/README.md`
- Ancienne doc locale: `legacy_local/README_LOCAL_LEGACY.md`

## Déploiement NAS (résumé)

```bash
cd /volume1/Services

git clone <your-repo-url> memory
cd memory

python3 -m venv .venv
source .venv/bin/activate
pip install -r nas_memory/requirements.txt

# config override (optionnel)
# - MEMORY_CORE_CONFIG=/path/to/config.py
# - ou variables d'environnement directes (env-first)

systemctl --user enable --now memory-api.service memory-worker.service
systemctl --user enable --now memory-profile-compact.timer memory-relation-compact.timer
```

## État d'architecture

- Source de vérité: NAS
- Clients: thin-call only (`/retrieve`, `/events`)
- Écriture mémoire: single-writer (`memory-worker`)
- Relation compaction: shadow/write via flags d'env
