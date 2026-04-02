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
    PHASE_CLAIM_PENG,
    PHASE_DISCARD_OR_WIN,
    PHASE_POST_MELD_DISCARD,
    STATUS_HAND_BASE,
    STATUS_UNDRAWN,
)
from mahjong_rl.env import MahjongEnv, PlayerState


def tile(kind_base: int, rank: int) -> int:
    return kind_base + rank - 1


class MahjongEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MahjongEnv(seed=123)
        self.env.reset(seed=123)

    def _reset_manual_state(self) -> None:
        for physical_tile in self.env.tiles:
            physical_tile.status = STATUS_UNDRAWN
        self.env.players = [PlayerState() for _ in range(3)]
        self.env.claim_queue = []
        self.env.claim_index = 0
        self.env.pending_discard_tile_id = None
        self.env.pending_discarder = None
        self.env.done = False
        self.env.winner = None
        self.env.winner_tai = 0
        self.env.terminal_reason = None
        self.env.game_wind = EAST

    def _take_tile_ids(self, kind: int, count: int) -> list[int]:
        selected: list[int] = []
        for physical_tile in self.env.tiles:
            if physical_tile.kind == kind and physical_tile.status == STATUS_UNDRAWN:
                selected.append(physical_tile.tile_id)
                if len(selected) == count:
                    break
        if len(selected) != count:
            raise AssertionError(f"Unable to allocate {count} tiles of kind {kind}.")
        return selected

    def _set_concealed(self, player: int, kinds: list[int]) -> None:
        tile_ids: list[int] = []
        for kind in kinds:
            tile_id = self._take_tile_ids(kind, 1)[0]
            self.env.tiles[tile_id].status = STATUS_HAND_BASE + player
            tile_ids.append(tile_id)
        self.env.players[player].concealed = tile_ids

    def test_game_action_requires_minimum_tai(self) -> None:
        self._reset_manual_state()
        no_tai_complete_hand = [
            tile(BAMBOO_BASE, 1),
            tile(BAMBOO_BASE, 1),
            tile(BAMBOO_BASE, 1),
            tile(BAMBOO_BASE, 2),
            tile(BAMBOO_BASE, 3),
            tile(BAMBOO_BASE, 4),
            tile(DOT_BASE, 2),
            tile(DOT_BASE, 3),
            tile(DOT_BASE, 4),
            tile(DOT_BASE, 6),
            tile(DOT_BASE, 7),
            tile(DOT_BASE, 8),
            tile(BAMBOO_BASE, 5),
            tile(BAMBOO_BASE, 5),
        ]
        self._set_concealed(0, no_tai_complete_hand)
        self.env.current_player = 0
        self.env.phase = PHASE_DISCARD_OR_WIN
        self.env.players[0].needs_draw = False

        mask = self.env.legal_action_mask()
        self.assertEqual(mask[ACTION_GAME], 0.0)

    def test_self_draw_reward_matches_one_tai_table(self) -> None:
        self._reset_manual_state()
        one_tai_hand = [
            tile(BAMBOO_BASE, 1),
            tile(BAMBOO_BASE, 2),
            tile(BAMBOO_BASE, 3),
            tile(BAMBOO_BASE, 4),
            tile(BAMBOO_BASE, 5),
            tile(BAMBOO_BASE, 6),
            tile(BAMBOO_BASE, 7),
            tile(BAMBOO_BASE, 8),
            tile(BAMBOO_BASE, 9),
            tile(DOT_BASE, 1),
            tile(DOT_BASE, 2),
            tile(DOT_BASE, 3),
            tile(DOT_BASE, 5),
            tile(DOT_BASE, 5),
        ]
        self._set_concealed(0, one_tai_hand)
        self.env.current_player = 0
        self.env.phase = PHASE_DISCARD_OR_WIN
        self.env.players[0].needs_draw = False

        step_result = self.env.step(ACTION_GAME)
        np.testing.assert_allclose(step_result.rewards, np.asarray([4.0, -2.0, -2.0], dtype=np.float32))
        self.assertTrue(step_result.done)
        self.assertEqual(step_result.info["winner"], 0)
        self.assertEqual(step_result.info["winner_tai"], 1)

    def test_peng_claim_has_priority_over_chi(self) -> None:
        self._reset_manual_state()
        self._set_concealed(
            0,
            [
                tile(BAMBOO_BASE, 2),
                tile(BAMBOO_BASE, 4),
                tile(BAMBOO_BASE, 5),
                tile(BAMBOO_BASE, 6),
                tile(DOT_BASE, 2),
                tile(DOT_BASE, 3),
                tile(DOT_BASE, 4),
                tile(DOT_BASE, 5),
                tile(DOT_BASE, 6),
                tile(DOT_BASE, 7),
                tile(DOT_BASE, 8),
                tile(DOT_BASE, 9),
                tile(BAMBOO_BASE, 7),
            ],
        )
        self._set_concealed(
            1,
            [
                tile(BAMBOO_BASE, 1),
                tile(BAMBOO_BASE, 3),
                tile(BAMBOO_BASE, 4),
                tile(BAMBOO_BASE, 5),
                tile(DOT_BASE, 1),
                tile(DOT_BASE, 2),
                tile(DOT_BASE, 3),
                tile(DOT_BASE, 4),
                tile(DOT_BASE, 5),
                tile(DOT_BASE, 6),
                tile(DOT_BASE, 7),
                tile(DOT_BASE, 8),
                tile(DOT_BASE, 9),
            ],
        )
        self._set_concealed(
            2,
            [
                tile(BAMBOO_BASE, 2),
                tile(BAMBOO_BASE, 2),
                tile(BAMBOO_BASE, 8),
                tile(BAMBOO_BASE, 9),
                tile(DOT_BASE, 1),
                tile(DOT_BASE, 2),
                tile(DOT_BASE, 3),
                tile(DOT_BASE, 4),
                tile(DOT_BASE, 5),
                tile(DOT_BASE, 6),
                tile(DOT_BASE, 7),
                tile(DOT_BASE, 8),
                tile(DOT_BASE, 9),
            ],
        )
        self.env.current_player = 0
        self.env.phase = PHASE_DISCARD_OR_WIN
        self.env.players[0].needs_draw = False

        self.env.step(ACTION_DISCARD_BASE)

        self.assertEqual(self.env.phase, PHASE_CLAIM_PENG)
        self.assertEqual(self.env.current_player, 2)
        mask = self.env.legal_action_mask()
        self.assertEqual(mask[ACTION_PENG], 1.0)

    def test_exposed_peng_tiles_are_not_discardable(self) -> None:
        self._reset_manual_state()
        self._set_concealed(
            0,
            [
                tile(BAMBOO_BASE, 2),
                tile(BAMBOO_BASE, 4),
                tile(BAMBOO_BASE, 5),
                tile(BAMBOO_BASE, 6),
                tile(DOT_BASE, 2),
                tile(DOT_BASE, 3),
                tile(DOT_BASE, 4),
                tile(DOT_BASE, 5),
                tile(DOT_BASE, 6),
                tile(DOT_BASE, 7),
                tile(DOT_BASE, 8),
                tile(DOT_BASE, 9),
                tile(BAMBOO_BASE, 7),
            ],
        )
        self._set_concealed(
            2,
            [
                tile(BAMBOO_BASE, 2),
                tile(BAMBOO_BASE, 2),
                tile(BAMBOO_BASE, 8),
                tile(BAMBOO_BASE, 9),
                tile(DOT_BASE, 1),
                tile(DOT_BASE, 2),
                tile(DOT_BASE, 3),
                tile(DOT_BASE, 4),
                tile(DOT_BASE, 5),
                tile(DOT_BASE, 6),
                tile(DOT_BASE, 7),
                tile(DOT_BASE, 8),
                tile(DOT_BASE, 9),
            ],
        )
        self.env.current_player = 0
        self.env.phase = PHASE_DISCARD_OR_WIN
        self.env.players[0].needs_draw = False

        self.env.step(ACTION_DISCARD_BASE)
        self.env.step(ACTION_PENG)

        self.assertEqual(self.env.phase, PHASE_POST_MELD_DISCARD)
        self.assertEqual(self.env.current_player, 2)
        concealed_kinds = [self.env.tiles[tile_id].kind for tile_id in self.env.players[2].concealed]
        self.assertEqual(concealed_kinds.count(tile(BAMBOO_BASE, 2)), 0)
        discard_mask = self.env.legal_action_mask()
        self.assertEqual(int(np.sum(discard_mask[:14])), len(self.env.players[2].concealed))


if __name__ == "__main__":
    unittest.main()
