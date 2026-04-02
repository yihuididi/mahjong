from __future__ import annotations

import unittest

import numpy as np

from mini_training_ppo.env import PPOCompatibleMahjongEnv
from mini_training_ppo.train import PPOConfig, compute_gae_returns_and_advantages, train_ppo


class MiniTrainingPPOSmokeTest(unittest.TestCase):
    def test_compute_gae_returns_and_advantages_matches_manual_discounted_case(self) -> None:
        rewards = np.asarray([1.0, 2.0], dtype=np.float32)
        values = np.asarray([0.5, 0.25], dtype=np.float32)
        next_values = np.asarray([0.25, 0.0], dtype=np.float32)
        dones = np.asarray([0.0, 1.0], dtype=np.float32)

        returns, advantages = compute_gae_returns_and_advantages(
            rewards=rewards,
            values=values,
            next_values=next_values,
            dones=dones,
            gamma=1.0,
            gae_lambda=1.0,
        )

        np.testing.assert_allclose(advantages, np.asarray([2.5, 1.75], dtype=np.float32))
        np.testing.assert_allclose(returns, np.asarray([3.0, 2.0], dtype=np.float32))

    def test_ppo_env_and_training_run(self) -> None:
        env = PPOCompatibleMahjongEnv(seed=21)
        observation, action_mask, current_player = env.reset(seed=21)
        self.assertEqual(observation.shape[0], env.observation_size)
        self.assertEqual(action_mask.shape[0], 20)
        self.assertIn(current_player, (0, 1, 2))

        config = PPOConfig(
            epochs=2,
            episodes_per_epoch=2,
            max_steps_per_episode=120,
            hidden_sizes=(32, 16),
            update_epochs=2,
            minibatch_size=16,
            seed=21,
            log_interval=1,
        )
        agent, summaries = train_ppo(config)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(agent.actor.output_dim, 20)
        self.assertIn("draw_games", summaries[-1].aggregated_stats)
        self.assertIn("missed_game_actions", summaries[-1].aggregated_stats)


if __name__ == "__main__":
    unittest.main()
