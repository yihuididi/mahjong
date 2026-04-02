from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mahjong_rl.generate_backward_curriculum_dataset import (
    generate_backward_curriculum_dataset,
    write_dataset,
)


class BackwardCurriculumDatasetTest(unittest.TestCase):
    def test_backward_curriculum_rows_track_terminal_distance(self) -> None:
        train_rows, test_rows, summary = generate_backward_curriculum_dataset(
            train_size=18,
            test_size=6,
            seed=11,
            max_steps_from_terminal=3,
            min_hand_size=11,
            branching_factor=2,
        )
        self.assertEqual(len(train_rows), 18)
        self.assertEqual(len(test_rows), 6)
        self.assertIn("steps_from_terminal_histogram", summary["train"])
        self.assertGreaterEqual(len(summary["train"]["steps_from_terminal_histogram"]), 2)

        for row in train_rows + test_rows:
            self.assertGreaterEqual(row["steps_from_terminal"], 1)
            self.assertLessEqual(row["steps_from_terminal"], 3)
            self.assertEqual(len(row["hand_tile_kinds"]), 14 - row["steps_from_terminal"])
            self.assertTrue(row["path_progress_tile_kinds"])
            self.assertTrue(set(row["path_progress_tile_kinds"]).issubset(set(row["improving_tile_kinds"])))

    def test_backward_curriculum_writer_outputs_metadata(self) -> None:
        train_rows, test_rows, summary = generate_backward_curriculum_dataset(
            train_size=10,
            test_size=4,
            seed=19,
            max_steps_from_terminal=2,
            min_hand_size=12,
            branching_factor=2,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            write_dataset(temp_dir, train_rows, test_rows, summary)
            output_dir = Path(temp_dir)
            self.assertTrue((output_dir / "train.jsonl").exists())
            self.assertTrue((output_dir / "test.jsonl").exists())
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["format_version"], 1)
            self.assertIn("steps_from_terminal_histogram", metadata["summary"]["train"])


if __name__ == "__main__":
    unittest.main()
