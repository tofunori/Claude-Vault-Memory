# config.py — Thierry Laurent-St-Pierre (MacBook Pro M3)
# Généré automatiquement — ne pas committer

VAULT_NOTES_DIR = "/Users/tofunori/Documents/UTQR/Master/knowledge/notes"
QDRANT_PATH = "/Users/tofunori/.claude/hooks/vault_qdrant"
ENV_FILE = "/Users/tofunori/.claude/hooks/.env"
QUEUE_DIR = "/Users/tofunori/.claude/hooks/queue"
LOG_FILE = "/Users/tofunori/.claude/hooks/auto_remember.log"
GRAPH_CACHE_PATH = "/Users/tofunori/.claude/hooks/vault_graph_cache.json"
BM25_INDEX_PATH = "/Users/tofunori/.claude/hooks/vault_bm25_index.json"
SOURCE_CHUNKS_DIR = "/Users/tofunori/Documents/UTQR/Master/knowledge/notes/_sources"
FORGET_ARCHIVE_DIR = "/Users/tofunori/Documents/UTQR/Master/knowledge/notes/_archived"

# Embedding
VOYAGE_EMBED_MODEL = "voyage-multilingual-2"
EMBED_DIM = 1024
EMBED_BATCH_SIZE = 128

# LLM extraction
CLAUDE_EXTRACT_MODEL = "claude-sonnet-4-6"
# Legacy Fireworks (kept for reference)
# FIREWORKS_MODEL = "accounts/fireworks/models/kimi-k2p5"
# FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

# Thresholds
RETRIEVE_SCORE_THRESHOLD = 0.60
RETRIEVE_TOP_K = 3
DEDUP_THRESHOLD = 0.85
MIN_QUERY_LENGTH = 20
MIN_TURNS = 3
MIN_NEW_TURNS = 10

# Graph traversal
MAX_SECONDARY = 5
MAX_BACKLINKS_PER_NOTE = 3
BFS_DEPTH = 2

# Hybrid search (BM25 + vector)
BM25_ENABLED = True
RRF_K = 60
BM25_TOP_K = 10
VECTOR_TOP_K = 10
RRF_FINAL_TOP_K = 3

# Confidence weighting
CONFIDENCE_BOOST = 1.2

# Temporal decay
DECAY_ENABLED = True
DECAY_HALF_LIFE_DAYS = 90
DECAY_FLOOR = 0.3

# Smart truncation
MAX_CODE_BLOCK_CHARS = 500

# Reranking
RERANK_ENABLED = True
RERANK_MODEL = "rerank-2"
RERANK_CANDIDATES = 10

# Extraction validation
VALIDATION_ENABLED = True

# Reflector
REFLECT_MIN_NOTES = 30
REFLECT_CLUSTER_THRESHOLD = 0.82
REFLECT_STALE_DAYS = 180

# Source chunk storage
SOURCE_CHUNKS_ENABLED = True
SOURCE_CHUNK_MAX_CHARS = 2000
SOURCE_INJECT_MAX_CHARS = 800

# Smart forgetting (désactivé par défaut)
FORGET_DEFAULT_TTL_DAYS = {}
