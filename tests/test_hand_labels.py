from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mahjong_rl.constants import BAMBOO_BASE, DOT_BASE, EAST, NORTH, RED, SOUTH, TILE_TYPE_COUNT
from mahjong_rl.generate_dataset import generate_dataset, write_dataset
from mahjong_rl.generate_discard_dataset import generate_discard_dataset
from mahjong_rl.hand_labels import label_concealed_discard_hand, label_concealed_hand
from mahjong_rl.rules import MeldPattern, evaluate_hand


class HandLabelsTest(unittest.TestCase):
    def test_all_straights_wait_is_legal_game_tile(self) -> None:
        hand_kinds = [
            BAMBOO_BASE + 0,
            BAMBOO_BASE + 1,
            BAMBOO_BASE + 2,
            BAMBOO_BASE + 3,
            BAMBOO_BASE + 4,
            BAMBOO_BASE + 5,
            BAMBOO_BASE + 6,
            BAMBOO_BASE + 7,
            BAMBOO_BASE + 8,
            DOT_BASE + 0,
            DOT_BASE + 1,
            DOT_BASE + 2,
            DOT_BASE + 3,
        ]
        labels = label_concealed_hand(hand_kinds, seat_wind=EAST, game_wind=SOUTH)
        self.assertEqual(labels.winning_shape_tile_kinds, (DOT_BASE + 0, DOT_BASE + 3))
        self.assertEqual(labels.legal_game_tile_kinds, (DOT_BASE + 0, DOT_BASE + 3))
        self.assertIn(DOT_BASE + 3, labels.improving_tile_kinds)

    def test_zero_tai_wait_is_shape_only_not_legal_game(self) -> None:
        hand_kinds = [
            BAMBOO_BASE + 0,
            BAMBOO_BASE + 0,
            BAMBOO_BASE + 0,
            BAMBOO_BASE + 3,
            BAMBOO_BASE + 4,
            BAMBOO_BASE + 5,
            DOT_BASE + 0,
            DOT_BASE + 1,
            DOT_BASE + 2,
            DOT_BASE + 3,
            DOT_BASE + 4,
            DOT_BASE + 5,
            RED,
        ]
        labels = label_concealed_hand(hand_kinds, seat_wind=EAST, game_wind=SOUTH)
        self.assertEqual(labels.winning_shape_tile_kinds, (RED,))
        self.assertEqual(labels.legal_game_tile_kinds, ())

    def test_exposed_north_peng_scores_for_anyone_and_prevailing_north_adds_extra_tai(self) -> None:
        concealed_counts = [0] * TILE_TYPE_COUNT
        for kind in [
            BAMBOO_BASE + 0,
            BAMBOO_BASE + 1,
            BAMBOO_BASE + 2,
            BAMBOO_BASE + 3,
            BAMBOO_BASE + 4,
            BAMBOO_BASE + 5,
            DOT_BASE + 0,
            DOT_BASE + 1,
            DOT_BASE + 2,
            DOT_BASE + 4,
            DOT_BASE + 4,
        ]:
            concealed_counts[kind] += 1

        north_peng = (MeldPattern("peng", (NORTH, NORTH, NORTH)),)
        eval_not_prevailing = evaluate_hand(concealed_counts, north_peng, seat_wind=EAST, game_wind=SOUTH)
        eval_prevailing = evaluate_hand(concealed_counts, north_peng, seat_wind=EAST, game_wind=NORTH)

        self.assertTrue(eval_not_prevailing.can_win)
        self.assertEqual(eval_not_prevailing.tai, 1)
        self.assertTrue(eval_prevailing.can_win)
        self.assertEqual(eval_prevailing.tai, 2)

    def test_discard_label_prefers_throwing_isolated_honor_over_pair(self) -> None:
        hand_kinds = [
            BAMBOO_BASE + 0,
            BAMBOO_BASE + 0,
            BAMBOO_BASE + 1,
            BAMBOO_BASE + 2,
            BAMBOO_BASE + 3,
            BAMBOO_BASE + 4,
            BAMBOO_BASE + 5,
            DOT_BASE + 0,
            DOT_BASE + 1,
            DOT_BASE + 2,
            DOT_BASE + 3,
            DOT_BASE + 4,
            DOT_BASE + 5,
            RED,
        ]
        labels = label_concealed_discard_hand(hand_kinds, seat_wind=EAST, game_wind=SOUTH)
        self.assertIn(RED, labels.best_discard_tile_kinds)
        self.assertNotIn(BAMBOO_BASE + 0, labels.best_discard_tile_kinds)

    def test_generator_writes_expected_dataset_files(self) -> None:
        train_rows, test_rows, summary = generate_dataset(train_size=8, test_size=3, seed=5, include_empty=True)
        self.assertEqual(len(train_rows), 8)
        self.assertEqual(len(test_rows), 3)
        self.assertIn("non_empty_legal_game_rows", summary["train"])

        with tempfile.TemporaryDirectory() as temp_dir:
            write_dataset(temp_dir, train_rows, test_rows, summary)
            output_dir = Path(temp_dir)
            self.assertTrue((output_dir / "train.jsonl").exists())
            self.assertTrue((output_dir / "test.jsonl").exists())
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["format_version"], 1)

    def test_include_empty_mode_keeps_structured_positive_examples(self) -> None:
        train_rows, test_rows, summary = generate_dataset(train_size=40, test_size=10, seed=5, include_empty=True)
        self.assertGreater(summary["train"]["non_empty_winning_shape_rows"], 0)
        self.assertLess(summary["train"]["non_empty_winning_shape_rows"], len(train_rows))
        self.assertTrue(any(not row["winning_shape_tile_kinds"] for row in train_rows + test_rows))

    def test_generator_default_non_empty_mode_uses_backward_sampling(self) -> None:
        train_rows, test_rows, _ = generate_dataset(train_size=6, test_size=2, seed=8)
        self.assertEqual(len(train_rows), 6)
        self.assertEqual(len(test_rows), 2)
        self.assertTrue(all(row["winning_shape_tile_kinds"] for row in train_rows + test_rows))

    def test_discard_dataset_generation_outputs_best_discard_labels(self) -> None:
        train_rows, test_rows, summary = generate_discard_dataset(train_size=6, test_size=2, seed=13)
        self.assertEqual(len(train_rows), 6)
        self.assertEqual(len(test_rows), 2)
        self.assertIn("max_best_discards", summary["train"])
        self.assertTrue(all(row["best_discard_tile_kinds"] for row in train_rows + test_rows))


if __name__ == "__main__":
    unittest.main()
