from __future__ import annotations

import unittest

import numpy as np

from mahjong_rl.constants import (
    ACTION_DISCARD_BASE,
    ACTION_GAME,
    ACTION_PENG,
    BAMBOO_BASE,
    DOT_BASE,
    EAST,
    PHASE_DISCARD_OR_WIN,
)
from mahjong_rl.env import PlayerState
from minitraining.env import MiniMahjongEnv
from minitraining.train import MiniTrainingConfig, train_self_play


class MiniTrainingSmokeTest(unittest.TestCase):
    def test_compact_env_and_training_run(self) -> None:
        env = MiniMahjongEnv(seed=9)
        observation, action_mask, current_player = env.reset(seed=9)
        self.assertEqual(observation.shape[0], env.observation_size)
        self.assertEqual(action_mask.shape[0], 20)
        self.assertIn(current_player, (0, 1, 2))

        config = MiniTrainingConfig(
            episodes=2,
            max_steps_per_episode=80,
            replay_capacity=256,
            batch_size=8,
            warmup_steps=8,
            hidden_sizes=(32, 16),
            target_sync_interval=20,
            log_interval=1,
            seed=9,
        )
        agent, summaries = train_self_play(config)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(agent.online.output_dim, 20)
        self.assertIn("draw_games", summaries[-1].episode_stats)
        self.assertIn("missed_game_actions", summaries[-1].episode_stats)

    def test_missing_legal_game_is_penalized(self) -> None:
        env = MiniMahjongEnv(seed=15, progress_reward_scale=0.0, missed_game_penalty=0.15)
        env.reset(seed=15)

        for physical_tile in env.tiles:
            physical_tile.status = 0
        env.players = [PlayerState() for _ in range(3)]
        env.claim_queue = []
        env.claim_index = 0
        env.pending_discard_tile_id = None
        env.pending_discarder = None
        env.done = False
        env.winner = None
        env.winner_tai = 0
        env.terminal_reason = None
        env.game_wind = EAST

        def take_tile_id(kind: int) -> int:
            for tile in env.tiles:
                if tile.kind == kind and tile.status == 0:
                    return tile.tile_id
            raise AssertionError(f"Tile kind {kind} unavailable.")

        winning_kinds = [
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
            DOT_BASE + 3,
        ]

        env.players[0].concealed = []
        for kind in winning_kinds:
            tile_id = take_tile_id(kind)
            env.tiles[tile_id].status = 1
            env.players[0].concealed.append(tile_id)

        env.current_player = 0
        env.phase = PHASE_DISCARD_OR_WIN
        env.players[0].needs_draw = False

        action_mask = env.legal_action_mask()
        self.assertEqual(action_mask[ACTION_GAME], 1.0)

        step_result = env.step(ACTION_DISCARD_BASE)
        self.assertLess(float(step_result.rewards[0]), 0.0)
        self.assertEqual(env.episode_stats["missed_game_actions"], 1)

    def test_peng_claim_receives_positive_shaping_reward(self) -> None:
        env = MiniMahjongEnv(seed=11, progress_reward_scale=0.10)
        env.reset(seed=11)

        for physical_tile in env.tiles:
            physical_tile.status = 0
        env.players = [PlayerState() for _ in range(3)]
        env.claim_queue = []
        env.claim_index = 0
        env.pending_discard_tile_id = None
        env.pending_discarder = None
        env.done = False
        env.winner = None
        env.winner_tai = 0
        env.terminal_reason = None
        env.game_wind = EAST

        def take_tile_id(kind: int) -> int:
            for tile in env.tiles:
                if tile.kind == kind and tile.status == 0:
                    return tile.tile_id
            raise AssertionError(f"Tile kind {kind} unavailable.")

        def set_concealed(player: int, kinds: list[int]) -> None:
            tile_ids = []
            for kind in kinds:
                tile_id = take_tile_id(kind)
                env.tiles[tile_id].status = 1 + player
                tile_ids.append(tile_id)
            env.players[player].concealed = tile_ids

        set_concealed(
            0,
            [
                BAMBOO_BASE + 1,
                BAMBOO_BASE + 3,
                BAMBOO_BASE + 4,
                BAMBOO_BASE + 5,
                DOT_BASE + 1,
                DOT_BASE + 2,
                DOT_BASE + 3,
                DOT_BASE + 4,
                DOT_BASE + 5,
                DOT_BASE + 6,
                DOT_BASE + 7,
                DOT_BASE + 8,
                DOT_BASE + 0,
                BAMBOO_BASE + 6,
            ],
        )
        set_concealed(
            2,
            [
                BAMBOO_BASE + 1,
                BAMBOO_BASE + 1,
                BAMBOO_BASE + 7,
                BAMBOO_BASE + 8,
                DOT_BASE + 0,
                DOT_BASE + 1,
                DOT_BASE + 2,
                DOT_BASE + 3,
                DOT_BASE + 4,
                DOT_BASE + 5,
                DOT_BASE + 6,
                DOT_BASE + 7,
                DOT_BASE + 8,
            ],
        )

        env.current_player = 0
        env.phase = PHASE_DISCARD_OR_WIN
        env.players[0].needs_draw = False

        env.step(ACTION_DISCARD_BASE)
        step_result = env.step(ACTION_PENG)
        self.assertGreater(float(step_result.rewards[2]), 0.0)
        self.assertTrue(np.all(step_result.rewards > np.asarray([-1.0, -1.0, -1.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
