#!/usr/bin/env python3
"""
process_queue.py — Worker asynchrone auto_remember
Traite les tickets déposés par enqueue.py.
Déclenché par launchd WatchPaths sur le dossier queue/.
Utilise Fireworks (kimi-k2) — latence sans importance, qualité maximale.
"""

import json
import os
import re
import subprocess
import sys
import traceback
from datetime import date
from pathlib import Path

try:
    from config import (
        VAULT_NOTES_DIR as _VAULT_NOTES_DIR,
        QDRANT_PATH as _QDRANT_PATH,
        ENV_FILE as _ENV_FILE,
        LOG_FILE as _LOG_FILE,
        QUEUE_DIR as _QUEUE_DIR,
        COHERE_EMBED_MODEL,
        FIREWORKS_MODEL,
        FIREWORKS_BASE_URL,
        DEDUP_THRESHOLD,
    )
    VAULT_NOTES_DIR = Path(_VAULT_NOTES_DIR)
    QDRANT_PATH = Path(_QDRANT_PATH)
    ENV_FILE = Path(_ENV_FILE)
    LOG_FILE = Path(_LOG_FILE)
    QUEUE_DIR = Path(_QUEUE_DIR)
except ImportError:
    VAULT_NOTES_DIR = Path.home() / "notes"
    QDRANT_PATH = Path.home() / ".claude/hooks/vault_qdrant"
    ENV_FILE = Path.home() / ".claude/hooks/.env"
    LOG_FILE = Path.home() / ".claude/hooks/auto_remember.log"
    QUEUE_DIR = Path.home() / ".claude/hooks/queue"
    COHERE_EMBED_MODEL = "embed-multilingual-v3.0"
    FIREWORKS_MODEL = "accounts/fireworks/models/kimi-k2p5"
    FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
    DEDUP_THRESHOLD = 0.85

PROCESSED_DIR = QUEUE_DIR / "processed"
HOOKS_DIR = ENV_FILE.parent
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
    """Retourne (cohere.ClientV2, QdrantClient) ou (None, None) si indisponible."""
    try:
        import cohere
        from qdrant_client import QdrantClient
    except ImportError:
        return None, None

    env = load_env_file()
    api_key = env.get("COHERE_API_KEY") or os.environ.get("COHERE_API_KEY", "")
    if not api_key or api_key.startswith("<"):
        return None, None
    if not QDRANT_PATH.exists():
        return None, None

    try:
        co = cohere.ClientV2(api_key)
        qd = QdrantClient(path=str(QDRANT_PATH))
        existing = {c.name for c in qd.get_collections().collections}
        if COLLECTION not in existing:
            return None, None
        return co, qd
    except Exception as e:
        log(f"EMBED clients error: {e}")
        return None, None


def check_semantic_dup(content: str) -> tuple[bool, str]:
    """Retourne (True, target_id) si contenu similaire existe déjà dans Qdrant."""
    try:
        co, qd = get_embed_clients()
        if co is None:
            return False, ""
        resp = co.embed(
            model=COHERE_EMBED_MODEL,
            texts=[content[:500]],
            input_type="search_query",
            embedding_types=["float"],
        )
        qd_resp = qd.query_points(
            collection_name=COLLECTION,
            query=resp.embeddings.float_[0],
            limit=1,
            score_threshold=DEDUP_THRESHOLD,
        )
        if qd_resp.points:
            return True, qd_resp.points[0].payload.get("note_id", "")
    except Exception as e:
        log(f"DEDUP error: {e}")
    return False, ""


def upsert_note_async(note_id: str):
    """Lance vault_embed.py en arrière-plan pour upsert une note dans Qdrant."""
    try:
        script = str(HOOKS_DIR / "vault_embed.py")
        subprocess.Popen(
            ["python3", script, "--note", note_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log(f"EMBED async upsert lancé: {note_id}")
    except Exception as e:
        log(f"EMBED async upsert error: {e}")


def extract_conversation(jsonl_path: str, max_chars: int = 40000) -> tuple[str, int]:
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
        log(f"Erreur lecture transcript: {e}")

    return "\n\n".join(turns)[:max_chars], len(turns)


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
        log(f"Erreur listing notes: {e}")
    return "\n".join(lines)


def extract_facts_with_llm(conversation: str, existing_notes: str) -> list:
    try:
        from openai import OpenAI
    except ImportError:
        log("Package openai non installé, skip")
        return []

    env = load_env_file()
    api_key = env.get("FIREWORKS_API_KEY") or os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        log("FIREWORKS_API_KEY absent, skip")
        return []

    client = OpenAI(api_key=api_key, base_url=FIREWORKS_BASE_URL)

    prompt = f"""Tu es un agent de mémoire personnelle.

Extrais 0-5 faits atomiques DURABLES depuis cette session Claude Code.

RÈGLES STRICTES :
- Capture TOUT ce qui est durable : décisions techniques, configs système, solutions à des problèmes, préférences découvertes, workflows établis, insights sur n'importe quel projet, faits appris, outils configurés
- Le domaine importe peu : recherche, infra, cours, scripts, lecture, etc.
- Ignore : débogage temporaire sans résolution, bavardage, reformulations sans contenu nouveau, étapes intermédiaires
- Titre = proposition testable ("X fait Y" — pas un label générique)
- Maximum 5 notes. Zéro si vraiment rien de durable.

TYPES DE RELATION :
- NEW : fait entièrement nouveau, absent des notes existantes
- UPDATES:<note_id> : remplace une info existante (ex: seuil changé, valeur corrigée)
- EXTENDS:<note_id> : complète sans remplacer (ex: détail supplémentaire sur méthode existante)

Notes existantes dans le vault :
{existing_notes}

FORMAT DE RÉPONSE — tableau JSON uniquement, aucun texte autour :
[
  {{
    "note_id": "slug-kebab-case",
    "relation": "NEW",
    "documentDate": "{TODAY}",
    "eventDate": null,
    "content": "---\\ndescription: [~150 chars, mécanisme ou portée]\\ntype: decision|result|method|concept|context|argument|module\\ncreated: {TODAY}\\nconfidence: experimental\\n---\\n\\n# Titre comme proposition\\n\\nCorps de la note...\\n\\n## Connexions\\n\\n- [[note-liée]]"
  }}
]

Pour EXTENDS, le "content" est le texte additionnel à appendre (pas une note complète).
Pour UPDATES, le "content" est la note complète révisée.

Si zéro notes mémorables : []

CONVERSATION DE LA SESSION :
{conversation}"""

    raw = ""
    try:
        response = client.chat.completions.create(
            model=FIREWORKS_MODEL,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)

        log(f"Réponse LLM ({len(raw)} chars): {raw[:300]}")
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []

    except json.JSONDecodeError as e:
        log(f"JSON invalide depuis LLM: {e} — raw: {raw[:300]}")
        return []
    except Exception as e:
        log(f"Erreur API Fireworks: {e}")
        return []


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
            extension = f"\n\n---\n*Extension auto {TODAY}:*\n\n{content}"
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
        log(f"Erreur lecture ticket {ticket_path.name}: {e}")
        return

    session_id = ticket.get("session_id", "unknown")
    transcript_path = ticket.get("transcript_path", "")

    log(f"--- PROCESSING session={session_id[:8]}")

    if not VAULT_NOTES_DIR.exists():
        log(f"Vault introuvable: {VAULT_NOTES_DIR}, skip")
        return

    if not transcript_path or not Path(transcript_path).exists():
        log(f"Transcript introuvable: {transcript_path}, skip")
        _archive(ticket_path, session_id)
        return

    conversation, turn_count = extract_conversation(transcript_path)
    log(f"Conversation: {turn_count} tours, {len(conversation)} chars")

    existing_notes = get_existing_notes_summary(VAULT_NOTES_DIR)
    facts = extract_facts_with_llm(conversation, existing_notes)

    if not facts:
        log("Aucun fait mémorable extrait")
        _archive(ticket_path, session_id)
        return

    log(f"Faits extraits: {len(facts)}")
    written = 0
    for fact in facts:
        try:
            note_id = fact.get("note_id", "").strip()
            relation = fact.get("relation", "NEW")
            content = fact.get("content", "").strip()
            if not note_id or not content:
                log(f"Fait invalide ignoré: {fact}")
                continue

            # Déduplication sémantique : uniquement pour les faits NEW
            if relation == "NEW":
                is_dup, target_id = check_semantic_dup(content)
                if is_dup and target_id:
                    relation = f"EXTENDS:{target_id}"
                    log(f"DEDUP: {note_id} → EXTENDS:{target_id}")

            write_note(note_id, content, relation)
            written += 1

            # Upsert incrémental dans Qdrant après écriture
            actual_id = note_id
            if relation.startswith("UPDATES:"):
                actual_id = relation.split(":", 1)[1].strip()
            elif relation.startswith("EXTENDS:"):
                actual_id = relation.split(":", 1)[1].strip()
            upsert_note_async(actual_id)

        except Exception as e:
            log(f"Erreur écriture {fact.get('note_id', '?')}: {e}")

    log(f"Notes écrites: {written}/{len(facts)}")
    _archive(ticket_path, session_id)


def _archive(ticket_path: Path, session_id: str):
    try:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        dest = PROCESSED_DIR / ticket_path.name
        ticket_path.rename(dest)
        log(f"ARCHIVED session={session_id[:8]}")
    except Exception as e:
        log(f"Erreur archivage: {e}")


def main():
    try:
        tickets = [
            f for f in QUEUE_DIR.glob("*.json")
            if f.is_file() and f.parent == QUEUE_DIR
        ]

        if not tickets:
            log("Queue vide, rien à traiter")
            sys.exit(0)

        log(f"=== process_queue: {len(tickets)} ticket(s) à traiter")

        for ticket_path in sorted(tickets, key=lambda f: f.stat().st_mtime):
            session_id = ticket_path.stem
            if (PROCESSED_DIR / ticket_path.name).exists():
                log(f"SKIP (already processed) session={session_id[:8]}")
                ticket_path.unlink(missing_ok=True)
                continue
            process_ticket(ticket_path)

        log("=== process_queue: done")

    except Exception as e:
        log(f"Erreur fatale process_queue: {e}\n{traceback.format_exc()}")

    sys.exit(0)


if __name__ == "__main__":
    main()
