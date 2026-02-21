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


if __name__ == "__main__":
    unittest_main()
