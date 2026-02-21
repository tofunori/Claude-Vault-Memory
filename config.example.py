# config.example.py — Copy to config.py and adapt to your environment
# config.py is in .gitignore — never commit your personal paths or keys.

# ─── Paths ────────────────────────────────────────────────────────────────────

# Directory containing your atomic notes (.md files)
VAULT_NOTES_DIR = "/home/yourname/notes"

# Directory where Qdrant stores its on-disk index
QDRANT_PATH = "/home/yourname/.claude/hooks/vault_qdrant"

# .env file containing API keys
ENV_FILE = "/home/yourname/.claude/hooks/.env"

# Queue directory (async tickets)
QUEUE_DIR = "/home/yourname/.claude/hooks/queue"

# Log file
LOG_FILE = "/home/yourname/.claude/hooks/auto_remember.log"

# Graph cache (generated automatically by vault_embed.py on full rebuild)
GRAPH_CACHE_PATH = "/home/yourname/.claude/hooks/vault_graph_cache.json"


# ─── Embedding model ──────────────────────────────────────────────────────────

# Voyage AI embedding model
VOYAGE_EMBED_MODEL = "voyage-4-large"

# Output dimension — voyage-4-large supports 256 / 512 / 1024 / 2048
# Higher = better quality, larger index. 1024 is a good default.
EMBED_DIM = 1024

# Batch size for Voyage AI embed calls (max 128 texts per request)
EMBED_BATCH_SIZE = 128


# ─── LLM extraction ───────────────────────────────────────────────────────────

# LLM for fact extraction (via OpenAI-compatible API)
FIREWORKS_MODEL = "accounts/fireworks/models/kimi-k2p5"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


# ─── Thresholds ───────────────────────────────────────────────────────────────

# Active retrieval: minimum cosine score to surface a note
RETRIEVE_SCORE_THRESHOLD = 0.60
RETRIEVE_TOP_K = 3

# Semantic deduplication: minimum score to consider a duplicate
DEDUP_THRESHOLD = 0.85

# Minimum message length (chars) to trigger retrieval
MIN_QUERY_LENGTH = 20

# Minimum number of turns in a session to enqueue for extraction
MIN_TURNS = 5

# ─── Graph traversal ──────────────────────────────────────────────────────────

# Maximum connected notes surfaced via BFS graph traversal
MAX_SECONDARY = 5

# Maximum backlinks injected per primary note (prevents MOC flooding)
MAX_BACKLINKS_PER_NOTE = 3

# BFS depth for graph traversal (2 = notes connected to connected notes)
BFS_DEPTH = 2


# ─── Hybrid search (BM25 + vector) ──────────────────────────────────────────

# Enable keyword search alongside vector search (Reciprocal Rank Fusion)
BM25_ENABLED = True

# RRF constant (higher = less emphasis on top ranks)
RRF_K = 60

# Number of keyword results to consider before fusion
BM25_TOP_K = 10

# Number of vector results to consider before fusion
VECTOR_TOP_K = 10

# Final number of primary notes after RRF fusion
RRF_FINAL_TOP_K = 3


# ─── Confidence weighting ───────────────────────────────────────────────────

# Boost factor for notes with confidence: confirmed (vs experimental)
CONFIDENCE_BOOST = 1.2


# ─── Temporal decay ─────────────────────────────────────────────────────────

# Enable temporal decay (notes accessed recently rank higher)
DECAY_ENABLED = True

# Half-life in days: after this many days without retrieval, score is halved
DECAY_HALF_LIFE_DAYS = 90

# Minimum decay factor (notes never drop below this fraction of their score)
DECAY_FLOOR = 0.3


# ─── Smart truncation ───────────────────────────────────────────────────────

# Maximum chars per individual code block in transcript (rest truncated)
MAX_CODE_BLOCK_CHARS = 500

# Minimum number of new turns to re-enqueue a grown session
MIN_NEW_TURNS = 10


# ─── Reranking ───────────────────────────────────────────────────────────────

# Enable Voyage AI reranking after RRF fusion (improves precision, adds ~100ms)
RERANK_ENABLED = True

# Voyage AI reranking model
RERANK_MODEL = "rerank-2"

# Number of candidates to feed to the reranker (before final top_k selection)
RERANK_CANDIDATES = 10


# ─── BM25 persistent index ──────────────────────────────────────────────────

# Path to the persistent BM25 index (rebuilt by vault_embed.py)
BM25_INDEX_PATH = "/home/yourname/.claude/hooks/vault_bm25_index.json"


# ─── Extraction validation ───────────────────────────────────────────────────

# Enable second-pass validation of extracted facts (catches hallucinated extractions)
VALIDATION_ENABLED = True


# ─── Reflector ───────────────────────────────────────────────────────────────

# Minimum number of notes to trigger reflection (below this, vault is too small)
REFLECT_MIN_NOTES = 30

# Cosine similarity threshold to consider notes as a mergeable cluster
REFLECT_CLUSTER_THRESHOLD = 0.82

# Maximum age in days before a never-retrieved note is flagged as stale
REFLECT_STALE_DAYS = 180


# ─── Source chunk storage ────────────────────────────────────────────────────

# Save conversation excerpts that generated each note (for retrieval injection)
SOURCE_CHUNKS_ENABLED = True

# Directory for source chunks (default: VAULT_NOTES_DIR/_sources)
SOURCE_CHUNKS_DIR = "/home/yourname/notes/_sources"

# Maximum chars of conversation to save per note
SOURCE_CHUNK_MAX_CHARS = 2000

# Maximum chars of source context injected during retrieval (per note)
SOURCE_INJECT_MAX_CHARS = 800


# ─── Smart forgetting ───────────────────────────────────────────────────────

# Directory for archived (forgotten) notes (default: VAULT_NOTES_DIR/_archived)
FORGET_ARCHIVE_DIR = "/home/yourname/notes/_archived"

# Default TTL (days) per note type. Notes of these types are auto-archived
# after this many days since creation. Set to {} to disable type-based TTL.
# Notes with an explicit forget_after frontmatter field are always respected.
FORGET_DEFAULT_TTL_DAYS = {
    # "context": 90,    # Context notes expire after 90 days
    # "result": 60,     # Result notes expire after 60 days
}
