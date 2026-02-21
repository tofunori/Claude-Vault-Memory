from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .db import connect

WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


def _edge_id(prefix: str, source: str, target: str, relation: str) -> str:
    raw = f"{prefix}|{source}|{target}|{relation}".encode("utf-8")
    return f"{prefix}:{hashlib.sha1(raw).hexdigest()[:16]}"


def _note_node(note_id: str) -> dict[str, Any]:
    return {
        "id": f"note:{note_id}",
        "kind": "note",
        "scope": None,
        "label": note_id,
        "type": "note",
        "fact_type": None,
        "status": "active",
        "confidence": None,
        "note_id": note_id,
    }


def _memory_node(row: sqlite3.Row) -> dict[str, Any]:
    fact_text = str(row["fact_text"] or "")
    preview = fact_text if len(fact_text) <= 180 else (fact_text[:177] + "...")
    return {
        "id": f"mem:{row['id']}",
        "kind": "memory",
        "scope": row["scope"],
        "label": preview or f"memory:{row['id']}",
        "type": f"memory/{row['scope']}/{row['fact_type']}",
        "fact_type": row["fact_type"],
        "status": row["status"],
        "confidence": float(row["confidence"]) if row["confidence"] is not None else None,
        "memory_id": row["id"],
        "global_key": row["global_key"],
        "version": row["version"],
        "fact_text": fact_text,
    }


def _count_components(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> int:
    adjacency: dict[str, set[str]] = {str(n["id"]): set() for n in nodes}
    for edge in edges:
        src = str(edge.get("source", ""))
        dst = str(edge.get("target", ""))
        if src in adjacency and dst in adjacency:
            adjacency[src].add(dst)
            adjacency[dst].add(src)

    components = 0
    visited: set[str] = set()
    for node_id in adjacency:
        if node_id in visited:
            continue
        components += 1
        stack = [node_id]
        visited.add(node_id)
        while stack:
            current = stack.pop()
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
    return components


def _contains_alias(text: str, alias: str) -> bool:
    escaped = re.escape(alias.strip().lower())
    if not escaped:
        return False
    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def build_unified_graph(settings: Any) -> dict[str, Any]:
    warnings: list[str] = []
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_by_id: dict[str, dict[str, Any]] = {}
    edge_ids: set[str] = set()

    note_ids: set[str] = set()
    note_edges = 0
    memory_edges = 0
    bridge_edges = 0

    graph_cache_path = Path(settings.graph_cache_path)
    if graph_cache_path.exists():
        try:
            cache = json.loads(graph_cache_path.read_text(encoding="utf-8"))
            outbound = cache.get("outbound", {})
            if not isinstance(outbound, dict):
                outbound = {}
                warnings.append("graph cache outbound is not a dict; using empty graph")

            for src, targets in outbound.items():
                if not isinstance(src, str):
                    continue
                src = src.strip()
                if not src:
                    continue
                note_ids.add(src)
                if isinstance(targets, list):
                    for target in targets:
                        if isinstance(target, str) and target.strip():
                            note_ids.add(target.strip())

            for note_id in sorted(note_ids):
                node = _note_node(note_id)
                nodes.append(node)
                node_by_id[node["id"]] = node

            for src, targets in outbound.items():
                if not isinstance(src, str):
                    continue
                src = src.strip()
                if not src or not isinstance(targets, list):
                    continue
                for target in targets:
                    if not isinstance(target, str):
                        continue
                    target = target.strip()
                    if not target:
                        continue
                    source_id = f"note:{src}"
                    target_id = f"note:{target}"
                    if source_id not in node_by_id or target_id not in node_by_id:
                        continue
                    edge_id = _edge_id("note", source_id, target_id, "links_to")
                    if edge_id in edge_ids:
                        continue
                    edge_ids.add(edge_id)
                    edges.append(
                        {
                            "id": edge_id,
                            "source": source_id,
                            "target": target_id,
                            "relation": "links_to",
                            "confidence": None,
                        }
                    )
                    note_edges += 1
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"graph cache unreadable ({exc}); showing memory graph only")
    else:
        warnings.append("graph cache missing; showing memory graph only")

    try:
        with connect(Path(settings.queue_db_path)) as conn:
            mem_rows = conn.execute(
                """
                SELECT id, global_key, version, scope, fact_text, fact_type, status, confidence
                FROM memory_nodes
                ORDER BY updated_at DESC
                """
            ).fetchall()
            for row in mem_rows:
                node = _memory_node(row)
                nodes.append(node)
                node_by_id[node["id"]] = node

            edge_rows = conn.execute(
                """
                SELECT id, src_node_id, dst_node_id, relation, confidence
                FROM memory_edges
                ORDER BY created_at DESC
                """
            ).fetchall()
            for row in edge_rows:
                source_id = f"mem:{row['src_node_id']}"
                target_id = f"mem:{row['dst_node_id']}"
                if source_id not in node_by_id or target_id not in node_by_id:
                    continue
                relation = str(row["relation"] or "").strip() or "supports"
                edge_id = str(row["id"] or _edge_id("mem", source_id, target_id, relation))
                if edge_id in edge_ids:
                    continue
                edge_ids.add(edge_id)
                edges.append(
                    {
                        "id": edge_id,
                        "source": source_id,
                        "target": target_id,
                        "relation": relation,
                        "confidence": float(row["confidence"]) if row["confidence"] is not None else None,
                    }
                )
                memory_edges += 1
    except sqlite3.Error as exc:
        warnings.append(f"memory tables unavailable ({exc}); showing note graph only")

    note_node_ids = {f"note:{nid}" for nid in note_ids}
    alias_rows: list[dict[str, Any]] = []
    try:
        with connect(Path(settings.queue_db_path)) as conn:
            rows = conn.execute(
                """
                SELECT alias, note_id
                FROM note_aliases
                ORDER BY updated_at DESC
                LIMIT 4000
                """
            ).fetchall()
            alias_rows = [{k: row[k] for k in row.keys()} for row in rows]
    except sqlite3.Error:
        alias_rows = []

    alias_pairs: list[tuple[str, str]] = []
    for row in alias_rows:
        alias = str(row.get("alias", "")).strip().lower()
        note_id = str(row.get("note_id", "")).strip()
        note_node_id = f"note:{note_id}"
        if not alias or note_node_id not in note_node_ids:
            continue
        alias_pairs.append((alias, note_node_id))

    for node in nodes:
        if node.get("kind") != "memory":
            continue
        mem_id = str(node["id"])
        text = str(node.get("fact_text") or "")
        for match in WIKILINK_RE.findall(text):
            note_id = str(match).strip()
            if not note_id:
                continue
            note_node_id = f"note:{note_id}"
            if note_node_id not in note_node_ids:
                continue
            edge_id = _edge_id("bridge", mem_id, note_node_id, "references_note")
            if edge_id in edge_ids:
                continue
            edge_ids.add(edge_id)
            edges.append(
                {
                    "id": edge_id,
                    "source": mem_id,
                    "target": note_node_id,
                    "relation": "references_note",
                    "confidence": None,
                }
            )
            bridge_edges += 1
        lowered = text.lower()
        for alias, note_node_id in alias_pairs:
            if not _contains_alias(lowered, alias):
                continue
            edge_id = _edge_id("bridge", mem_id, note_node_id, "references_note")
            if edge_id in edge_ids:
                continue
            edge_ids.add(edge_id)
            edges.append(
                {
                    "id": edge_id,
                    "source": mem_id,
                    "target": note_node_id,
                    "relation": "references_note",
                    "confidence": None,
                }
            )
            bridge_edges += 1

    components = _count_components(nodes, edges)
    return {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stats": {
            "note_nodes": sum(1 for n in nodes if n.get("kind") == "note"),
            "memory_nodes": sum(1 for n in nodes if n.get("kind") == "memory"),
            "bridge_edges": bridge_edges,
            "note_edges": note_edges,
            "memory_edges": memory_edges,
            "components": components,
        },
        "nodes": sorted(nodes, key=lambda n: str(n.get("id", ""))),
        "edges": sorted(edges, key=lambda e: str(e.get("id", ""))),
        "warnings": warnings,
    }
