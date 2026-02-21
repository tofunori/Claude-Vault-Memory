from __future__ import annotations

import unittest

try:
    from nas_memory.api import _profile_to_note
except Exception:  # pragma: no cover
    _profile_to_note = None


class RetrieveProfileMergeTests(unittest.TestCase):
    @unittest.skipIf(_profile_to_note is None, "fastapi dependencies not installed in this interpreter")
    def test_profile_note_mapping_caps_score(self) -> None:
        note = _profile_to_note(
            {
                "id": "abc",
                "fact_text": "On garde FastAPI + SQLite WAL sur NAS",
                "fact_type": "decision",
                "confidence": 0.93,
            },
            score_cap=0.55,
        )
        self.assertEqual(note.note_id, "profile-abc")
        self.assertEqual(note.type, "profile/decision")
        self.assertEqual(note.confidence, "confirmed")
        self.assertAlmostEqual(note.score or 0.0, 0.55, places=3)


if __name__ == "__main__":
    unittest.main()
