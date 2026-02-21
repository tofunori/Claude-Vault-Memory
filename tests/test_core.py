#!/usr/bin/env python3
"""
test_core.py — Unit tests for Claude Vault Memory core functions.

Run: python3 -m pytest tests/test_core.py -v
  or: python3 tests/test_core.py
"""

import json
import math
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest import TestCase, main as unittest_main

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions to test (these don't need config.py)
from process_queue import (
    sanitize_note_id,
    fix_wikilinks_in_content,
    _repair_json_newlines,
    _truncate_code_blocks,
    _inject_frontmatter_field,
    write_file_atomic,
    _add_superseded_by,
)
from vault_embed import build_graph_index


class TestSanitizeNoteId(TestCase):
    """Test sanitize_note_id: the most critical function (broken IDs = broken graph)."""

    def test_basic_kebab(self):
        self.assertEqual(sanitize_note_id("my-note-slug"), "my-note-slug")

    def test_spaces_to_hyphens(self):
        self.assertEqual(sanitize_note_id("my note slug"), "my-note-slug")

    def test_uppercase_to_lower(self):
        self.assertEqual(sanitize_note_id("My-Note-SLUG"), "my-note-slug")

    def test_special_chars_removed(self):
        self.assertEqual(sanitize_note_id("note@#$%with!chars"), "note-with-chars")

    def test_multiple_hyphens_collapsed(self):
        self.assertEqual(sanitize_note_id("note---with---hyphens"), "note-with-hyphens")

    def test_leading_trailing_hyphens_stripped(self):
        self.assertEqual(sanitize_note_id("--note-slug--"), "note-slug")

    def test_max_length_80(self):
        long_id = "a" * 100
        result = sanitize_note_id(long_id)
        self.assertLessEqual(len(result), 80)

    def test_truncation_no_trailing_hyphen(self):
        # 79 a's + a hyphen at position 80 should be trimmed
        long_id = "a" * 79 + "-" + "b" * 20
        result = sanitize_note_id(long_id)
        self.assertFalse(result.endswith("-"))
        self.assertLessEqual(len(result), 80)

    def test_unicode_normalized(self):
        # é (e + combining accent) → e
        self.assertEqual(sanitize_note_id("café-crème"), "cafe-creme")

    def test_empty_string(self):
        self.assertEqual(sanitize_note_id(""), "")

    def test_only_special_chars(self):
        self.assertEqual(sanitize_note_id("@#$%"), "")

    def test_numbers_preserved(self):
        self.assertEqual(sanitize_note_id("v2-api-endpoint-3"), "v2-api-endpoint-3")

    def test_dots_become_hyphens(self):
        self.assertEqual(sanitize_note_id("config.example.py"), "config-example-py")


class TestFixWikilinks(TestCase):
    """Test fix_wikilinks_in_content: prevents broken [[links]]."""

    def setUp(self):
        self.valid_ids = {"note-a", "note-b", "note-c"}
        self.title_to_id = {
            "my full title": "note-a",
            "another title": "note-b",
        }

    def test_valid_id_unchanged(self):
        content = "See [[note-a]] for details."
        result = fix_wikilinks_in_content(content, self.title_to_id, self.valid_ids)
        self.assertEqual(result, "See [[note-a]] for details.")

    def test_title_replaced_with_id(self):
        content = "See [[My Full Title]] for details."
        result = fix_wikilinks_in_content(content, self.title_to_id, self.valid_ids)
        self.assertEqual(result, "See [[note-a]] for details.")

    def test_unresolvable_link_stripped(self):
        content = "See [[Unknown Note]] for details."
        result = fix_wikilinks_in_content(content, self.title_to_id, self.valid_ids)
        self.assertEqual(result, "See Unknown Note for details.")

    def test_display_text_preserved(self):
        content = "See [[My Full Title|the link]] for details."
        result = fix_wikilinks_in_content(content, self.title_to_id, self.valid_ids)
        self.assertEqual(result, "See [[note-a|the link]] for details.")

    def test_unresolvable_with_display_text(self):
        content = "See [[Unknown|display]] for details."
        result = fix_wikilinks_in_content(content, self.title_to_id, self.valid_ids)
        self.assertEqual(result, "See display for details.")

    def test_multiple_links(self):
        content = "[[note-a]] and [[Another Title]] and [[unknown]]"
        result = fix_wikilinks_in_content(content, self.title_to_id, self.valid_ids)
        self.assertEqual(result, "[[note-a]] and [[note-b]] and unknown")


class TestRepairJsonNewlines(TestCase):
    """Test _repair_json_newlines: fixes LLM JSON output issues."""

    def test_newline_in_string(self):
        raw = '{"key": "line1\nline2"}'
        result = _repair_json_newlines(raw)
        self.assertEqual(result, '{"key": "line1\\nline2"}')

    def test_tab_in_string(self):
        raw = '{"key": "col1\tcol2"}'
        result = _repair_json_newlines(raw)
        self.assertEqual(result, '{"key": "col1\\tcol2"}')

    def test_newline_outside_string_unchanged(self):
        raw = '{\n  "key": "value"\n}'
        result = _repair_json_newlines(raw)
        self.assertEqual(result, raw)

    def test_already_escaped_unchanged(self):
        raw = '{"key": "line1\\nline2"}'
        result = _repair_json_newlines(raw)
        self.assertEqual(result, raw)

    def test_valid_json_roundtrip(self):
        raw = '[{"note_id": "test", "content": "line1\\nline2"}]'
        result = _repair_json_newlines(raw)
        parsed = json.loads(result)
        self.assertEqual(parsed[0]["note_id"], "test")


class TestTruncateCodeBlocks(TestCase):
    """Test _truncate_code_blocks: caps large code blocks."""

    def test_short_block_unchanged(self):
        text = "```python\nprint('hello')\n```"
        result = _truncate_code_blocks(text, max_chars=500)
        self.assertEqual(result, text)

    def test_long_block_truncated(self):
        code = "x = 1\n" * 200  # ~1200 chars
        text = f"```python\n{code}```"
        result = _truncate_code_blocks(text, max_chars=100)
        self.assertIn("[truncated", result)
        self.assertIn("```python", result)

    def test_no_code_blocks(self):
        text = "Just plain text with no code."
        result = _truncate_code_blocks(text, max_chars=100)
        self.assertEqual(result, text)


class TestBuildGraphIndex(TestCase):
    """Test build_graph_index: builds outbound + backlink indices."""

    def test_basic_graph(self):
        notes = [
            {"note_id": "note-a", "text": "See [[note-b]] and [[note-c]]"},
            {"note_id": "note-b", "text": "Back to [[note-a]]"},
            {"note_id": "note-c", "text": "No links here"},
        ]
        outbound, backlinks = build_graph_index(notes)

        self.assertEqual(set(outbound["note-a"]), {"note-b", "note-c"})
        self.assertEqual(outbound["note-b"], ["note-a"])
        self.assertEqual(outbound["note-c"], [])

        self.assertIn("note-a", backlinks.get("note-b", []))
        self.assertIn("note-a", backlinks.get("note-c", []))
        self.assertIn("note-b", backlinks.get("note-a", []))

    def test_unknown_links_excluded(self):
        notes = [
            {"note_id": "note-a", "text": "See [[nonexistent]] and [[note-a]]"},
        ]
        outbound, _ = build_graph_index(notes)
        # Self-links are allowed, unknown IDs are excluded
        self.assertNotIn("nonexistent", outbound.get("note-a", []))

    def test_empty_notes(self):
        outbound, backlinks = build_graph_index([])
        self.assertEqual(outbound, {})
        self.assertEqual(backlinks, {})

    def test_dedup_links(self):
        notes = [
            {"note_id": "note-a", "text": "[[note-b]] and again [[note-b]]"},
            {"note_id": "note-b", "text": ""},
        ]
        outbound, _ = build_graph_index(notes)
        self.assertEqual(outbound["note-a"], ["note-b"])  # No duplicates


class TestRRFMerge(TestCase):
    """Test Reciprocal Rank Fusion merge."""

    def test_basic_merge(self):
        # Import from vault_retrieve
        from vault_retrieve import rrf_merge

        vector = [
            {"note_id": "a", "description": "A", "type": "concept", "score": 0.9},
            {"note_id": "b", "description": "B", "type": "concept", "score": 0.8},
        ]
        keyword = [
            {"note_id": "b", "description": "B", "type": "concept", "score": 5.0},
            {"note_id": "c", "description": "C", "type": "concept", "score": 3.0},
        ]
        result = rrf_merge(vector, keyword, k=60, top_k=3)

        # b should rank highest (appears in both lists)
        self.assertEqual(result[0]["note_id"], "b")
        self.assertEqual(len(result), 3)

    def test_empty_lists(self):
        from vault_retrieve import rrf_merge
        result = rrf_merge([], [], k=60, top_k=3)
        self.assertEqual(result, [])


class TestDecay(TestCase):
    """Test temporal decay computation."""

    def test_today_no_decay(self):
        from vault_retrieve import compute_decay
        today = date.today().isoformat()
        decay = compute_decay(today, today)
        self.assertAlmostEqual(decay, 1.0, places=2)

    def test_old_date_decays(self):
        from vault_retrieve import compute_decay
        old_date = (date.today() - timedelta(days=180)).isoformat()
        decay = compute_decay(old_date, old_date)
        self.assertLess(decay, 1.0)
        self.assertGreaterEqual(decay, 0.3)  # Floor

    def test_very_old_hits_floor(self):
        from vault_retrieve import compute_decay
        ancient = (date.today() - timedelta(days=3650)).isoformat()  # 10 years
        decay = compute_decay(ancient, ancient)
        self.assertAlmostEqual(decay, 0.3, places=1)  # Should hit floor

    def test_none_returns_1(self):
        from vault_retrieve import compute_decay
        decay = compute_decay(None, None)
        self.assertAlmostEqual(decay, 1.0, places=2)


class TestConfidenceBoost(TestCase):
    """Test confidence boost factor."""

    def test_confirmed_gets_boost(self):
        from vault_retrieve import apply_confidence_boost
        self.assertGreater(apply_confidence_boost("confirmed"), 1.0)

    def test_experimental_no_boost(self):
        from vault_retrieve import apply_confidence_boost
        self.assertEqual(apply_confidence_boost("experimental"), 1.0)

    def test_none_no_boost(self):
        from vault_retrieve import apply_confidence_boost
        self.assertEqual(apply_confidence_boost(None), 1.0)


class TestBM25Search(TestCase):
    """Test BM25 internal functions."""

    def test_tokenize(self):
        from vault_retrieve import tokenize
        tokens = tokenize("Hello World, this is a test with Python3")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("python3", tokens)
        self.assertIn("test", tokens)
        # Stopwords removed
        self.assertNotIn("this", tokens)
        self.assertNotIn("is", tokens)

    def test_tokenize_code_terms(self):
        from vault_retrieve import tokenize
        tokens = tokenize("vault_embed.py uses qdrant-client v1.2")
        self.assertIn("vault_embed.py", tokens)
        self.assertIn("qdrant-client", tokens)
        self.assertIn("v1.2", tokens)

    def test_score_bm25_basic(self):
        from vault_retrieve import _score_bm25
        docs = [
            {"note_id": "a", "tf": {"python": 3, "code": 1}, "len": 10},
            {"note_id": "b", "tf": {"javascript": 2, "code": 1}, "len": 10},
        ]
        scored = _score_bm25(docs, ["python"])
        self.assertGreater(scored[0]["bm25_score"], 0)
        self.assertEqual(scored[1]["bm25_score"], 0)  # No match for "python"

    def test_persistent_index_load(self):
        """Test that _load_bm25_index returns None when no index exists."""
        from vault_retrieve import _load_bm25_index
        # With default path (None or non-existent), should return None
        result = _load_bm25_index()
        # Should be None (no persistent index in test env)
        self.assertIsNone(result)


class TestValidation(TestCase):
    """Test extraction validation logic."""

    def test_validation_disabled_returns_all(self):
        from process_queue import validate_extracted_facts
        import process_queue
        original = process_queue.VALIDATION_ENABLED
        process_queue.VALIDATION_ENABLED = False
        facts = [{"note_id": "test", "content": "test"}]
        result = validate_extracted_facts(facts, "conversation")
        self.assertEqual(result, facts)
        process_queue.VALIDATION_ENABLED = original

    def test_validation_empty_facts(self):
        from process_queue import validate_extracted_facts
        result = validate_extracted_facts([], "conversation")
        self.assertEqual(result, [])


class TestSessionBrief(TestCase):
    """Test session brief parsing."""

    def test_parse_frontmatter(self):
        from vault_session_brief import parse_frontmatter
        text = """---
description: Test note about Python
type: preference
confidence: confirmed
created: 2026-01-15
---

# Test note"""
        fm = parse_frontmatter(text)
        self.assertEqual(fm["description"], "Test note about Python")
        self.assertEqual(fm["type"], "preference")
        self.assertEqual(fm["confidence"], "confirmed")
        self.assertEqual(fm["created"], "2026-01-15")


class TestInjectFrontmatterField(TestCase):
    """Test _inject_frontmatter_field: inserts fields into YAML frontmatter."""

    def test_basic_injection(self):
        content = "---\ndescription: test\ntype: concept\n---\n\n# Title"
        result = _inject_frontmatter_field(content, "relation", "new")
        self.assertIn("relation: new", result)
        self.assertIn("---\n\n# Title", result)

    def test_no_duplicate_injection(self):
        content = "---\ndescription: test\nrelation: updates\n---\n\n# Title"
        result = _inject_frontmatter_field(content, "relation", "new")
        # Should not inject again
        self.assertEqual(result, content)

    def test_multiple_fields(self):
        content = "---\ndescription: test\n---\n\n# Title"
        result = _inject_frontmatter_field(content, "relation", "updates")
        result = _inject_frontmatter_field(result, "parent_note", "old-note")
        self.assertIn("relation: updates", result)
        self.assertIn("parent_note: old-note", result)


class TestSupersededBy(TestCase):
    """Test _add_superseded_by: marks old notes as superseded."""

    def test_adds_superseded_by(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            note_path = Path(tmpdir) / "old-note.md"
            note_path.write_text("---\ndescription: old\ntype: concept\n---\n\n# Old note")
            _add_superseded_by(note_path, "new-note")
            content = note_path.read_text()
            self.assertIn("superseded_by: new-note", content)

    def test_no_duplicate_superseded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            note_path = Path(tmpdir) / "old-note.md"
            original = "---\ndescription: old\nsuperseded_by: first\n---\n\n# Old note"
            note_path.write_text(original)
            _add_superseded_by(note_path, "second")
            content = note_path.read_text()
            # Should not add a second superseded_by
            self.assertEqual(content, original)


class TestSourceChunkStorage(TestCase):
    """Test source chunk saving logic."""

    def test_save_and_load_chunk(self):
        import process_queue as pq
        with tempfile.TemporaryDirectory() as tmpdir:
            sources_dir = Path(tmpdir) / "_sources"
            # Temporarily override
            orig_dir = pq.SOURCE_CHUNKS_DIR
            orig_enabled = pq.SOURCE_CHUNKS_ENABLED
            pq.SOURCE_CHUNKS_DIR = sources_dir
            pq.SOURCE_CHUNKS_ENABLED = True
            try:
                pq.save_source_chunk("test-note", "NEW", "This is the conversation content.")
                chunk_path = sources_dir / "test-note.md"
                self.assertTrue(chunk_path.exists())
                content = chunk_path.read_text()
                self.assertIn("source_for: test-note", content)
                self.assertIn("This is the conversation content.", content)
            finally:
                pq.SOURCE_CHUNKS_DIR = orig_dir
                pq.SOURCE_CHUNKS_ENABLED = orig_enabled

    def test_disabled_no_save(self):
        import process_queue as pq
        with tempfile.TemporaryDirectory() as tmpdir:
            sources_dir = Path(tmpdir) / "_sources"
            orig_dir = pq.SOURCE_CHUNKS_DIR
            orig_enabled = pq.SOURCE_CHUNKS_ENABLED
            pq.SOURCE_CHUNKS_DIR = sources_dir
            pq.SOURCE_CHUNKS_ENABLED = False
            try:
                pq.save_source_chunk("test-note", "NEW", "Content")
                self.assertFalse(sources_dir.exists())
            finally:
                pq.SOURCE_CHUNKS_DIR = orig_dir
                pq.SOURCE_CHUNKS_ENABLED = orig_enabled

    def test_extends_appends_to_existing(self):
        import process_queue as pq
        with tempfile.TemporaryDirectory() as tmpdir:
            sources_dir = Path(tmpdir) / "_sources"
            orig_dir = pq.SOURCE_CHUNKS_DIR
            orig_enabled = pq.SOURCE_CHUNKS_ENABLED
            pq.SOURCE_CHUNKS_DIR = sources_dir
            pq.SOURCE_CHUNKS_ENABLED = True
            try:
                pq.save_source_chunk("target-note", "NEW", "Original conversation.")
                pq.save_source_chunk("ext-note", "EXTENDS:target-note", "Extension conversation.")
                chunk_path = sources_dir / "target-note.md"
                content = chunk_path.read_text()
                self.assertIn("Original conversation.", content)
                self.assertIn("Extension source", content)
                self.assertIn("Extension conversation.", content)
            finally:
                pq.SOURCE_CHUNKS_DIR = orig_dir
                pq.SOURCE_CHUNKS_ENABLED = orig_enabled


class TestLoadSourceChunk(TestCase):
    """Test source chunk loading for retrieval injection."""

    def test_load_existing_chunk(self):
        from vault_retrieve import load_source_chunk
        import vault_retrieve as vr
        with tempfile.TemporaryDirectory() as tmpdir:
            sources_dir = Path(tmpdir)
            orig_dir = vr.SOURCE_CHUNKS_DIR
            orig_enabled = vr.SOURCE_CHUNKS_ENABLED
            vr.SOURCE_CHUNKS_DIR = sources_dir
            vr.SOURCE_CHUNKS_ENABLED = True
            try:
                chunk_path = sources_dir / "my-note.md"
                chunk_path.write_text("---\nsource_for: my-note\n---\n\nThe conversation excerpt here.")
                result = load_source_chunk("my-note")
                self.assertIsNotNone(result)
                self.assertIn("conversation excerpt", result)
            finally:
                vr.SOURCE_CHUNKS_DIR = orig_dir
                vr.SOURCE_CHUNKS_ENABLED = orig_enabled

    def test_load_nonexistent_returns_none(self):
        from vault_retrieve import load_source_chunk
        import vault_retrieve as vr
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_dir = vr.SOURCE_CHUNKS_DIR
            orig_enabled = vr.SOURCE_CHUNKS_ENABLED
            vr.SOURCE_CHUNKS_DIR = Path(tmpdir)
            vr.SOURCE_CHUNKS_ENABLED = True
            try:
                result = load_source_chunk("nonexistent")
                self.assertIsNone(result)
            finally:
                vr.SOURCE_CHUNKS_DIR = orig_dir
                vr.SOURCE_CHUNKS_ENABLED = orig_enabled

    def test_disabled_returns_none(self):
        from vault_retrieve import load_source_chunk
        import vault_retrieve as vr
        orig = vr.SOURCE_CHUNKS_ENABLED
        vr.SOURCE_CHUNKS_ENABLED = False
        try:
            result = load_source_chunk("any-note")
            self.assertIsNone(result)
        finally:
            vr.SOURCE_CHUNKS_ENABLED = orig


class TestFindExpiredNotes(TestCase):
    """Test smart forgetting: find_expired_notes."""

    def test_forget_after_past(self):
        from vault_reflect import find_expired_notes
        notes = [
            {"note_id": "old-note", "forget_after": "2020-01-01", "type": "context"},
        ]
        expired = find_expired_notes(notes)
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0]["note_id"], "old-note")

    def test_forget_after_future(self):
        from vault_reflect import find_expired_notes
        future = (date.today() + timedelta(days=30)).isoformat()
        notes = [
            {"note_id": "future-note", "forget_after": future, "type": "context"},
        ]
        expired = find_expired_notes(notes)
        self.assertEqual(len(expired), 0)

    def test_no_forget_after_no_ttl(self):
        from vault_reflect import find_expired_notes
        notes = [
            {"note_id": "normal-note", "type": "decision", "created": "2020-01-01"},
        ]
        expired = find_expired_notes(notes)
        self.assertEqual(len(expired), 0)

    def test_type_ttl_expired(self):
        import vault_reflect as vr
        orig = vr.FORGET_DEFAULT_TTL_DAYS
        vr.FORGET_DEFAULT_TTL_DAYS = {"context": 30}
        try:
            old_date = (date.today() - timedelta(days=60)).isoformat()
            notes = [
                {"note_id": "ctx-note", "type": "context", "created": old_date},
            ]
            expired = vr.find_expired_notes(notes)
            self.assertEqual(len(expired), 1)
        finally:
            vr.FORGET_DEFAULT_TTL_DAYS = orig

    def test_type_ttl_not_expired(self):
        import vault_reflect as vr
        orig = vr.FORGET_DEFAULT_TTL_DAYS
        vr.FORGET_DEFAULT_TTL_DAYS = {"context": 90}
        try:
            recent = (date.today() - timedelta(days=10)).isoformat()
            notes = [
                {"note_id": "recent-ctx", "type": "context", "created": recent},
            ]
            expired = vr.find_expired_notes(notes)
            self.assertEqual(len(expired), 0)
        finally:
            vr.FORGET_DEFAULT_TTL_DAYS = orig


if __name__ == "__main__":
    unittest_main()
