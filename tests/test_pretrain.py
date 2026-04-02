from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mahjong_rl.generate_dataset import generate_dataset, write_dataset
from mahjong_rl.pretrain import PRETRAIN_OUTPUT_DIM, PretrainConfig, train_pretrained_wait_model
from mini_training_ppo.model import PPOAgent, PPOConfig


class PretrainWorkflowTest(unittest.TestCase):
    def test_pretrain_checkpoint_can_initialize_ppo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            train_rows, test_rows, summary = generate_dataset(
                train_size=24,
                test_size=8,
                seed=12,
                legal_only=True,
            )
            write_dataset(dataset_dir, train_rows, test_rows, summary)

            checkpoint_path = Path(temp_dir) / "pretrained_wait_model.npz"
            config = PretrainConfig(
                train_path=str(dataset_dir / "train.jsonl"),
                test_path=str(dataset_dir / "test.jsonl"),
                epochs=2,
                batch_size=8,
                seed=12,
                log_interval=1,
            )
            model, summaries = train_pretrained_wait_model(config, save_path=str(checkpoint_path))
            self.assertEqual(model.output_dim, PRETRAIN_OUTPUT_DIM)
            self.assertEqual(len(summaries), 2)
            self.assertTrue(checkpoint_path.exists())

            ppo_config = PPOConfig(
                epochs=1,
                episodes_per_epoch=1,
                hidden_sizes=(64, 32),
                pretrained_encoder_path=str(checkpoint_path),
                seed=12,
            )
            agent = PPOAgent(observation_size=model.input_dim, config=ppo_config)
            self.assertEqual(agent.actor.input_dim, model.input_dim)
            self.assertEqual(agent.actor.hidden_sizes, model.hidden_sizes)


if __name__ == "__main__":
    unittest.main()
