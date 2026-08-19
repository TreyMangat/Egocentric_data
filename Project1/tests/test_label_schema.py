from __future__ import annotations

import unittest

from label_schema import (
    classify_phrase,
    generate_label_document,
    validate_label_document,
)


METADATA = {
    "episode_hash": "episode-1",
    "fps": 30.0,
    "frames": 300,
    "duration_seconds": 10.0,
}


class AutomaticLabelTests(unittest.TestCase):
    def test_classifies_supported_actions(self):
        examples = {
            "pick up folded shirt": "PICK_UP",
            "place shirt on table": "PLACE",
            "move folded shirt": "MOVE",
            "fold shirt": "FOLD",
            "smoothen shirt": "SMOOTH",
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(classify_phrase(text)["label"], expected)

    def test_unknown_action_stays_other_with_a_hint(self):
        result = classify_phrase("unfold gray garment")
        self.assertEqual(result["label"], "OTHER")
        self.assertEqual(result["custom_label"], "UNFOLD")
        self.assertTrue(result["uncertain"])

    def test_multi_action_annotation_gets_approximate_suggestions(self):
        annotations = [
            {
                "segment_index": 0,
                "start_seconds": 0.0,
                "end_seconds": 4.0,
                "label": "pick up shirt, fold shirt",
            }
        ]
        document = generate_label_document(METADATA, annotations)
        self.assertEqual([item["label"] for item in document["labels"]], ["PICK_UP", "FOLD"])
        self.assertEqual(document["labels"][0]["end_seconds"], 2.0)
        self.assertEqual(document["labels"][1]["start_frame"], 60)
        self.assertTrue(all(item["uncertain"] for item in document["labels"]))
        self.assertTrue(all(not item["reviewed"] for item in document["labels"]))


class LabelValidationTests(unittest.TestCase):
    def test_recomputes_frames_from_saved_times(self):
        document = {
            "episode_hash": "episode-1",
            "labels": [
                {
                    "id": "manual-1",
                    "start_seconds": 1.0,
                    "end_seconds": 2.5,
                    "label": "OTHER",
                    "custom_label": "UNFOLD",
                    "active_hands": ["right"],
                    "reviewed": True,
                }
            ],
        }
        result = validate_label_document(document, METADATA)
        label = result["labels"][0]
        self.assertEqual(label["start_frame"], 30)
        self.assertEqual(label["end_frame_exclusive"], 75)
        self.assertEqual(label["active_hands"], ["right"])

    def test_rejects_overlapping_scene_labels(self):
        document = {
            "episode_hash": "episode-1",
            "labels": [
                {"id": "a", "start_seconds": 1, "end_seconds": 3, "label": "MOVE"},
                {"id": "b", "start_seconds": 2, "end_seconds": 4, "label": "FOLD"},
            ],
        }
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_label_document(document, METADATA)


if __name__ == "__main__":
    unittest.main()
