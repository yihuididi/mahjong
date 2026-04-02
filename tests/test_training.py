from __future__ import annotations

import unittest

from mahjong_rl.dqn import TrainingConfig
from mahjong_rl.train import train_self_play


class TrainingSmokeTest(unittest.TestCase):
    def test_training_runs_short_self_play_loop(self) -> None:
        config = TrainingConfig(
            episodes=2,
            max_steps_per_episode=100,
            replay_capacity=256,
            batch_size=8,
            warmup_steps=8,
            hidden_sizes=(32, 16),
            target_sync_interval=20,
            updates_per_step=1,
            seed=5,
        )
        agent, summaries = train_self_play(config)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(agent.online.output_dim, 20)


if __name__ == "__main__":
    unittest.main()
