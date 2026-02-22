from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    memory_root: Path
    state_dir: Path
    queue_db_path: Path
    worker_lock_path: Path
    qdrant_lock_path: Path
    api_host: str
    api_port: int
    api_token: str
    allowed_ips: tuple[str, ...]
    poll_interval_seconds: float
    process_queue_timeout_seconds: int
    retrieve_timeout_seconds: int
    embed_timeout_seconds: int
    reindex_timeout_seconds: int
    qdrant_lock_timeout_seconds: int
    rebuild_full_after_write: bool

    vault_notes_dir: Path
    qdrant_path: Path
    bm25_index_path: Path
    graph_cache_path: Path
    queue_dir: Path

    vault_retrieve_script: Path
    process_queue_script: Path
    vault_embed_script: Path

    search_mode: str
    core_config_loaded: bool
    live_extract_script: Path
    live_extract_timeout_seconds: int
    turn_live_cadence: int
    staging_ttl_hours: int
    staging_recent_turns_window: int
    live_extract_max_candidates: int
    retrieve_experimental_max: int
    retrieve_experimental_score_cap: float
    promotion_min_evidence: int
    backpressure_queue_threshold: int
    profile_enable: bool
    profile_max_items: int
    profile_score_cap: float
    profile_compact_interval_min: int
    forget_soft_delete: bool
    profile_extract_timeout_seconds: int
    profile_min_evidence: int
    profile_dynamic_hours: int
    profile_extract_script: Path
    admin_graph_timeout_seconds: int
    relation_enable: bool
    relation_write: bool
    relation_batch_max_pairs: int
    relation_min_confidence: float
    relation_max_new_edges_per_run: int
    relation_compact_interval_min: int
    relation_llm_timeout: int


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_core_config(repo_root: Path):
    config_path = Path(os.environ.get("MEMORY_CORE_CONFIG", repo_root / "config.py"))
    if not config_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("memory_core_config", config_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coerce_path(value: str | Path | None, fallback: Path) -> Path:
    if value is None:
        return fallback
    return Path(value)


def _resolve_script(repo_root: Path, canonical_rel: str, legacy_rel: str) -> Path:
    canonical = repo_root / canonical_rel
    if canonical.exists():
        return canonical
    return repo_root / legacy_rel


def _search_mode(core_config) -> str:
    bm25_enabled = bool(getattr(core_config, "BM25_ENABLED", True)) if core_config else True
    rerank_enabled = bool(getattr(core_config, "RERANK_ENABLED", True)) if core_config else True
    if bm25_enabled and rerank_enabled:
        return "hybrid+rerank"
    if bm25_enabled:
        return "hybrid"
    return "vector"


def load_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[1]
    core = _load_core_config(repo_root)

    memory_root = Path(os.environ.get("MEMORY_ROOT", "/volume1/Services/memory"))
    state_dir = Path(os.environ.get("MEMORY_STATE_DIR", str(memory_root / "state")))
    state_dir.mkdir(parents=True, exist_ok=True)

    queue_db_path = Path(os.environ.get("MEMORY_QUEUE_DB", str(state_dir / "memory_queue.db")))
    worker_lock_path = Path(os.environ.get("MEMORY_WORKER_LOCK", str(state_dir / "worker.lock")))
    qdrant_lock_path = Path(os.environ.get("MEMORY_QDRANT_LOCK", str(state_dir / "qdrant.lock")))

    api_host = os.environ.get("MEMORY_API_HOST", "0.0.0.0")
    api_port = int(os.environ.get("MEMORY_API_PORT", "8766"))
    api_token = os.environ.get("MEMORY_API_TOKEN", "")

    raw_allowlist = os.environ.get("MEMORY_ALLOWED_IPS", "")
    allowed_ips = tuple(ip.strip() for ip in raw_allowlist.split(",") if ip.strip())

    poll_interval_seconds = float(os.environ.get("MEMORY_POLL_INTERVAL", "2.0"))
    process_queue_timeout_seconds = int(os.environ.get("MEMORY_PROCESS_QUEUE_TIMEOUT", "600"))
    retrieve_timeout_seconds = int(os.environ.get("MEMORY_RETRIEVE_TIMEOUT", "60"))
    embed_timeout_seconds = int(os.environ.get("MEMORY_EMBED_TIMEOUT", "300"))
    reindex_timeout_seconds = int(os.environ.get("MEMORY_REINDEX_TIMEOUT", "1800"))
    qdrant_lock_timeout_seconds = int(os.environ.get("MEMORY_QDRANT_LOCK_TIMEOUT", "30"))
    rebuild_full_after_write = _parse_bool(os.environ.get("MEMORY_REBUILD_FULL_AFTER_WRITE"), True)
    live_extract_timeout_seconds = int(os.environ.get("MEMORY_LIVE_EXTRACT_TIMEOUT", "8"))
    turn_live_cadence = int(os.environ.get("MEMORY_TURN_LIVE_CADENCE", "5"))
    staging_ttl_hours = int(os.environ.get("MEMORY_STAGING_TTL_HOURS", "24"))
    staging_recent_turns_window = int(os.environ.get("MEMORY_STAGING_WINDOW", "12"))
    live_extract_max_candidates = int(os.environ.get("MEMORY_LIVE_MAX_CANDIDATES", "3"))
    retrieve_experimental_max = int(os.environ.get("MEMORY_RETRIEVE_EXPERIMENTAL_MAX", "2"))
    retrieve_experimental_score_cap = float(os.environ.get("MEMORY_EXPERIMENTAL_SCORE_CAP", "0.35"))
    promotion_min_evidence = int(os.environ.get("MEMORY_PROMOTION_MIN_EVIDENCE", "2"))
    backpressure_queue_threshold = int(os.environ.get("MEMORY_BACKPRESSURE_QUEUE_THRESHOLD", "100"))
    profile_enable = _parse_bool(os.environ.get("MEMORY_PROFILE_ENABLE"), True)
    profile_max_items = int(os.environ.get("MEMORY_PROFILE_MAX_ITEMS", "12"))
    profile_score_cap = float(os.environ.get("MEMORY_PROFILE_SCORE_CAP", "0.55"))
    profile_compact_interval_min = int(os.environ.get("MEMORY_PROFILE_COMPACT_INTERVAL_MIN", "60"))
    forget_soft_delete = _parse_bool(os.environ.get("MEMORY_FORGET_SOFT_DELETE"), True)
    profile_extract_timeout_seconds = int(os.environ.get("MEMORY_PROFILE_EXTRACT_TIMEOUT", "12"))
    profile_min_evidence = int(os.environ.get("MEMORY_PROFILE_MIN_EVIDENCE", "2"))
    profile_dynamic_hours = int(os.environ.get("MEMORY_PROFILE_DYNAMIC_HOURS", "24"))
    admin_graph_timeout_seconds = int(os.environ.get("MEMORY_ADMIN_GRAPH_TIMEOUT", "20"))
    relation_enable = _parse_bool(os.environ.get("MEMORY_RELATION_ENABLE"), True)
    relation_write = _parse_bool(os.environ.get("MEMORY_RELATION_WRITE"), False)
    relation_batch_max_pairs = int(os.environ.get("MEMORY_RELATION_BATCH_MAX_PAIRS", "800"))
    relation_min_confidence = float(os.environ.get("MEMORY_RELATION_MIN_CONFIDENCE", "0.72"))
    relation_max_new_edges_per_run = int(os.environ.get("MEMORY_RELATION_MAX_NEW_EDGES_PER_RUN", "120"))
    relation_compact_interval_min = int(os.environ.get("MEMORY_RELATION_COMPACT_INTERVAL_MIN", "60"))
    relation_llm_timeout = int(os.environ.get("MEMORY_RELATION_LLM_TIMEOUT", "10"))

    fallback_notes = memory_root / "notes"
    fallback_qdrant = memory_root / "vault_qdrant"
    fallback_bm25 = memory_root / "vault_bm25_index.json"
    fallback_graph = memory_root / "vault_graph_cache.json"
    fallback_queue = memory_root / "queue"

    vault_notes_dir = _coerce_path(getattr(core, "VAULT_NOTES_DIR", None), fallback_notes)
    qdrant_path = _coerce_path(getattr(core, "QDRANT_PATH", None), fallback_qdrant)
    bm25_index_path = _coerce_path(getattr(core, "BM25_INDEX_PATH", None), fallback_bm25)
    graph_cache_path = _coerce_path(getattr(core, "GRAPH_CACHE_PATH", None), fallback_graph)
    queue_dir = _coerce_path(getattr(core, "QUEUE_DIR", None), fallback_queue)

    queue_dir.mkdir(parents=True, exist_ok=True)

    vault_retrieve_script = _resolve_script(
        repo_root,
        canonical_rel="nas_memory/core/vault_retrieve.py",
        legacy_rel="vault_retrieve.py",
    )
    process_queue_script = _resolve_script(
        repo_root,
        canonical_rel="nas_memory/core/process_queue.py",
        legacy_rel="process_queue.py",
    )
    vault_embed_script = _resolve_script(
        repo_root,
        canonical_rel="nas_memory/core/vault_embed.py",
        legacy_rel="vault_embed.py",
    )

    return Settings(
        repo_root=repo_root,
        memory_root=memory_root,
        state_dir=state_dir,
        queue_db_path=queue_db_path,
        worker_lock_path=worker_lock_path,
        qdrant_lock_path=qdrant_lock_path,
        api_host=api_host,
        api_port=api_port,
        api_token=api_token,
        allowed_ips=allowed_ips,
        poll_interval_seconds=poll_interval_seconds,
        process_queue_timeout_seconds=process_queue_timeout_seconds,
        retrieve_timeout_seconds=retrieve_timeout_seconds,
        embed_timeout_seconds=embed_timeout_seconds,
        reindex_timeout_seconds=reindex_timeout_seconds,
        qdrant_lock_timeout_seconds=qdrant_lock_timeout_seconds,
        rebuild_full_after_write=rebuild_full_after_write,
        vault_notes_dir=vault_notes_dir,
        qdrant_path=qdrant_path,
        bm25_index_path=bm25_index_path,
        graph_cache_path=graph_cache_path,
        queue_dir=queue_dir,
        vault_retrieve_script=vault_retrieve_script,
        process_queue_script=process_queue_script,
        vault_embed_script=vault_embed_script,
        search_mode=_search_mode(core),
        core_config_loaded=core is not None,
        live_extract_script=repo_root / "nas_memory" / "live_extract.py",
        live_extract_timeout_seconds=live_extract_timeout_seconds,
        turn_live_cadence=max(1, turn_live_cadence),
        staging_ttl_hours=max(1, staging_ttl_hours),
        staging_recent_turns_window=max(3, staging_recent_turns_window),
        live_extract_max_candidates=max(1, live_extract_max_candidates),
        retrieve_experimental_max=max(0, retrieve_experimental_max),
        retrieve_experimental_score_cap=max(0.0, min(1.0, retrieve_experimental_score_cap)),
        promotion_min_evidence=max(1, promotion_min_evidence),
        backpressure_queue_threshold=max(1, backpressure_queue_threshold),
        profile_enable=profile_enable,
        profile_max_items=max(1, profile_max_items),
        profile_score_cap=max(0.0, min(1.0, profile_score_cap)),
        profile_compact_interval_min=max(1, profile_compact_interval_min),
        forget_soft_delete=forget_soft_delete,
        profile_extract_timeout_seconds=max(2, profile_extract_timeout_seconds),
        profile_min_evidence=max(1, profile_min_evidence),
        profile_dynamic_hours=max(1, profile_dynamic_hours),
        profile_extract_script=repo_root / "nas_memory" / "profile_extract.py",
        admin_graph_timeout_seconds=max(3, admin_graph_timeout_seconds),
        relation_enable=relation_enable,
        relation_write=relation_write,
        relation_batch_max_pairs=max(10, relation_batch_max_pairs),
        relation_min_confidence=max(0.0, min(1.0, relation_min_confidence)),
        relation_max_new_edges_per_run=max(1, relation_max_new_edges_per_run),
        relation_compact_interval_min=max(1, relation_compact_interval_min),
        relation_llm_timeout=max(0, relation_llm_timeout),
    )
