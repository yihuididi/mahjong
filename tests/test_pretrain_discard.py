from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mahjong_rl.generate_discard_dataset import generate_discard_dataset, write_dataset
from mahjong_rl.generate_dataset import generate_dataset as generate_wait_dataset, write_dataset as write_wait_dataset
from mahjong_rl.pretrain import PretrainConfig, train_pretrained_wait_model
from mahjong_rl.pretrain_discard import DiscardPretrainConfig, load_discard_rows, train_discard_encoder
from mini_training_ppo.model import PPOAgent, PPOConfig


class DiscardPretrainWorkflowTest(unittest.TestCase):
    def test_loaded_discard_rows_only_enable_present_tile_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "discard_dataset"
            train_rows, test_rows, summary = generate_discard_dataset(
                train_size=12,
                test_size=4,
                seed=18,
            )
            write_dataset(dataset_dir, train_rows, test_rows, summary)

            loaded = load_discard_rows(str(dataset_dir / "train.jsonl"))
            self.assertEqual(loaded.candidate_masks.shape[0], len(train_rows))
            for row_index, row in enumerate(train_rows):
                enabled = {
                    tile_kind
                    for tile_kind, value in enumerate(loaded.candidate_masks[row_index].tolist())
                    if value > 0.0
                }
                self.assertEqual(enabled, set(row["hand_tile_kinds"]))

    def test_discard_pretrain_checkpoint_can_initialize_ppo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "discard_dataset"
            train_rows, test_rows, summary = generate_discard_dataset(
                train_size=24,
                test_size=8,
                seed=21,
            )
            write_dataset(dataset_dir, train_rows, test_rows, summary)

            checkpoint_path = Path(temp_dir) / "pretrained_discard_encoder.npz"
            config = DiscardPretrainConfig(
                train_path=str(dataset_dir / "train.jsonl"),
                test_path=str(dataset_dir / "test.jsonl"),
                epochs=2,
                batch_size=8,
                seed=21,
                log_interval=1,
            )
            model, summaries = train_discard_encoder(config, save_path=str(checkpoint_path))
            self.assertEqual(model.output_dim, 25)
            self.assertEqual(len(summaries), 2)
            self.assertTrue(checkpoint_path.exists())

            ppo_config = PPOConfig(
                epochs=1,
                episodes_per_epoch=1,
                pretrained_encoder_path=str(checkpoint_path),
                seed=21,
            )
            agent = PPOAgent(observation_size=model.input_dim, config=ppo_config)
            self.assertEqual(agent.actor.input_dim, model.input_dim)
            self.assertEqual(agent.actor.hidden_sizes, model.hidden_sizes)

    def test_discard_pretrain_can_start_from_wait_pretrained_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wait_dataset_dir = Path(temp_dir) / "wait_dataset"
            discard_dataset_dir = Path(temp_dir) / "discard_dataset"

            wait_train_rows, wait_test_rows, wait_summary = generate_wait_dataset(
                train_size=24,
                test_size=8,
                seed=31,
                legal_only=True,
            )
            write_wait_dataset(wait_dataset_dir, wait_train_rows, wait_test_rows, wait_summary)

            wait_checkpoint_path = Path(temp_dir) / "pretrained_wait_model.npz"
            wait_config = PretrainConfig(
                train_path=str(wait_dataset_dir / "train.jsonl"),
                test_path=str(wait_dataset_dir / "test.jsonl"),
                epochs=1,
                batch_size=8,
                seed=31,
                log_interval=1,
            )
            wait_model, _ = train_pretrained_wait_model(wait_config, save_path=str(wait_checkpoint_path))

            discard_train_rows, discard_test_rows, discard_summary = generate_discard_dataset(
                train_size=24,
                test_size=8,
                seed=32,
            )
            write_dataset(discard_dataset_dir, discard_train_rows, discard_test_rows, discard_summary)

            discard_checkpoint_path = Path(temp_dir) / "pretrained_discard_encoder.npz"
            discard_config = DiscardPretrainConfig(
                train_path=str(discard_dataset_dir / "train.jsonl"),
                test_path=str(discard_dataset_dir / "test.jsonl"),
                epochs=1,
                batch_size=8,
                seed=32,
                log_interval=1,
                pretrained_encoder_path=str(wait_checkpoint_path),
            )
            discard_model, summaries = train_discard_encoder(
                discard_config,
                save_path=str(discard_checkpoint_path),
            )
            self.assertEqual(wait_model.hidden_sizes, discard_model.hidden_sizes)
            self.assertEqual(discard_model.output_dim, 25)
            self.assertEqual(len(summaries), 1)
            self.assertTrue(discard_checkpoint_path.exists())


if __name__ == "__main__":
    unittest.main()
