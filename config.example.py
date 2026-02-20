# config.example.py — Copier en config.py et adapter à votre environnement
# config.py est dans .gitignore — ne jamais committer vos chemins personnels.

# ─── Chemins ──────────────────────────────────────────────────────────────────

# Dossier contenant vos notes atomiques (.md)
VAULT_NOTES_DIR = "/home/yourname/notes"

# Dossier où Qdrant stocke son index disque
QDRANT_PATH = "/home/yourname/.claude/hooks/vault_qdrant"

# Fichier .env contenant les clés API
ENV_FILE = "/home/yourname/.claude/hooks/.env"

# Dossier queue (tickets asynchrones)
QUEUE_DIR = "/home/yourname/.claude/hooks/queue"

# Fichier log
LOG_FILE = "/home/yourname/.claude/hooks/auto_remember.log"


# ─── Modèles ──────────────────────────────────────────────────────────────────

# Cohere embeddings (multilingue, 1024 dims)
COHERE_EMBED_MODEL = "embed-multilingual-v3.0"
EMBED_DIM = 1024

# LLM pour extraction de faits (via OpenAI-compatible API)
FIREWORKS_MODEL = "accounts/fireworks/models/kimi-k2p5"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


# ─── Seuils ───────────────────────────────────────────────────────────────────

# Retrieval actif : score cosine minimum pour afficher une note
RETRIEVE_SCORE_THRESHOLD = 0.60
RETRIEVE_TOP_K = 3

# Déduplication sémantique : score minimum pour considérer un doublon
DEDUP_THRESHOLD = 0.85

# Longueur minimum d'un message pour déclencher le retrieval
MIN_QUERY_LENGTH = 20

# Nombre minimum de tours dans une session pour l'enqueue
MIN_TURNS = 5

# Taille maximale de batch pour les appels Cohere
COHERE_BATCH_SIZE = 96
