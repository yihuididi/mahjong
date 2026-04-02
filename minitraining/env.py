from __future__ import annotations

import numpy as np

from mahjong_rl.constants import (
    ACTION_GAME,
    DRAGON_KINDS,
    MAX_CONCEALED_TILES,
    MAX_TAI,
    MELD_PENG,
    NORTH,
    NUM_PLAYERS,
    PHASE_CLAIM_CHI,
    PHASE_CLAIM_PENG,
    PHASE_CLAIM_WIN,
    PHASES,
    PHASE_DISCARD_OR_WIN,
    SEAT_WINDS,
    TILE_TYPE_COUNT,
    TOTAL_TILES,
    WIND_KINDS,
)
from mahjong_rl.env import EnvStep, MahjongEnv


PROGRESS_REWARD_SCALE = 0.03
MISSED_GAME_PENALTY = 0.15


class MiniMahjongEnv(MahjongEnv):
    def __init__(
        self,
        seed: int | None = None,
        progress_reward_scale: float = PROGRESS_REWARD_SCALE,
        missed_game_penalty: float = MISSED_GAME_PENALTY,
    ) -> None:
        self.progress_reward_scale = progress_reward_scale
        self.missed_game_penalty = missed_game_penalty
        super().__init__(seed=seed)

    def _calculate_observation_size(self) -> int:
        hand_counts = TILE_TYPE_COUNT
        exposed_counts = NUM_PLAYERS * TILE_TYPE_COUNT
        discard_counts = NUM_PLAYERS * TILE_TYPE_COUNT
        phase_features = len(PHASES)
        seat_features = NUM_PLAYERS
        wind_features = len(WIND_KINDS)
        claim_tile_features = TILE_TYPE_COUNT + 1
        claim_from_features = NUM_PLAYERS + 1
        scalar_features = 11
        return (
            hand_counts
            + exposed_counts
            + discard_counts
            + phase_features
            + seat_features
            + wind_features
            + claim_tile_features
            + claim_from_features
            + scalar_features
        )

    def observe(self, player: int | None = None) -> np.ndarray:
        if player is None:
            player = self.current_player
        if player is None:
            raise ValueError("No current player is available in a terminal state.")

        hand_counts = np.zeros(TILE_TYPE_COUNT, dtype=np.float32)
        for tile_id in self.players[player].concealed:
            hand_counts[self.tiles[tile_id].kind] += 1.0

        exposed_counts = np.zeros((NUM_PLAYERS, TILE_TYPE_COUNT), dtype=np.float32)
        discard_counts = np.zeros((NUM_PLAYERS, TILE_TYPE_COUNT), dtype=np.float32)

        for absolute_player in range(NUM_PLAYERS):
            relative_player = self._relative_player(player, absolute_player)
            for meld in self.players[absolute_player].melds:
                for tile_kind in meld.tile_kinds:
                    exposed_counts[relative_player, tile_kind] += 1.0
            for tile_id in self.players[absolute_player].discards:
                discard_counts[relative_player, self.tiles[tile_id].kind] += 1.0

        phase_features = np.zeros(len(PHASES), dtype=np.float32)
        phase_features[PHASES.index(self.phase)] = 1.0

        seat_features = np.zeros(NUM_PLAYERS, dtype=np.float32)
        seat_features[player] = 1.0

        wind_features = np.zeros(len(WIND_KINDS), dtype=np.float32)
        wind_features[WIND_KINDS.index(self.game_wind)] = 1.0

        claim_tile_features = np.zeros(TILE_TYPE_COUNT + 1, dtype=np.float32)
        claim_tile_kind = self._active_claim_tile_kind()
        claim_tile_features[claim_tile_kind if claim_tile_kind is not None else TILE_TYPE_COUNT] = 1.0

        claim_from_features = np.zeros(NUM_PLAYERS + 1, dtype=np.float32)
        thrower = self._active_claim_thrower()
        if thrower is None:
            claim_from_features[0] = 1.0
        else:
            claim_from_features[self._relative_player(player, thrower) + 1] = 1.0

        current_eval = self._win_evaluation_for_observer(player)
        analysis = self._analysis_for_player(player)
        scalar_features = np.array(
            [
                self.tiles_left / float(TOTAL_TILES),
                len(self.players[player].concealed) / float(MAX_CONCEALED_TILES),
                1.0 if current_eval.is_complete else 0.0,
                current_eval.tai / float(MAX_TAI),
                analysis["total_melds"] / 4.0,
                analysis["has_pair"],
                analysis["partial_sets"] / 4.0,
                analysis["can_game"],
                analysis["complete_zero_tai"],
                analysis["tai_sources"] / float(MAX_TAI),
                analysis["completion_score"] / 12.0,
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                hand_counts,
                exposed_counts.reshape(-1),
                discard_counts.reshape(-1),
                phase_features,
                seat_features,
                wind_features,
                claim_tile_features,
                claim_from_features,
                scalar_features,
            ]
        ).astype(np.float32)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
        observation, action_mask, current_player = super().reset(seed=seed)
        self.episode_stats = self._empty_episode_stats()
        self._record_turn_state(self.current_player)
        return observation, action_mask, current_player

    def step(self, action: int) -> EnvStep:
        acting_player = self.current_player
        legal_game_before = self._legal_game_available_for_current_player()
        before_scores = self._progress_scores()
        step_result = super().step(action)
        after_scores = self._progress_scores()
        shaped_rewards = step_result.rewards + self.progress_reward_scale * (after_scores - before_scores)
        if acting_player is not None and legal_game_before and action != ACTION_GAME:
            shaped_rewards[acting_player] -= self.missed_game_penalty
            self.episode_stats["missed_game_actions"] += 1
            if step_result.info is not None:
                step_result.info["episode_stats"] = dict(self.episode_stats)
        return EnvStep(
            observation=step_result.observation,
            action_mask=step_result.action_mask,
            current_player=step_result.current_player,
            rewards=shaped_rewards.astype(np.float32),
            done=step_result.done,
            info=step_result.info,
        )

    def _build_claim_requests(self, thrower: int, tile_id: int):
        requests = super()._build_claim_requests(thrower, tile_id)
        for request in requests:
            if request.claim_type == PHASE_CLAIM_WIN:
                self.episode_stats["discard_game_opportunities"] += 1
            elif request.claim_type == PHASE_CLAIM_PENG:
                self.episode_stats["peng_opportunities"] += 1
            elif request.claim_type == PHASE_CLAIM_CHI:
                self.episode_stats["chi_opportunities"] += 1
        return requests

    def _start_regular_turn(self, player: int) -> None:
        super()._start_regular_turn(player)
        if not self.done:
            self._record_turn_state(player)

    def _resolve_turn_action(self, player: int, action: int) -> np.ndarray:
        if action == ACTION_GAME:
            self.episode_stats["self_draw_wins"] += 1
            self.episode_stats["wins"] += 1
        return super()._resolve_turn_action(player, action)

    def _resolve_claim_win(self, action: int) -> np.ndarray:
        if action == ACTION_GAME:
            self.episode_stats["discard_claim_wins"] += 1
            self.episode_stats["wins"] += 1
        return super()._resolve_claim_win(action)

    def _apply_peng(self, request) -> None:
        self.episode_stats["peng_taken"] += 1
        super()._apply_peng(request)

    def _apply_chi(self, request, action: int) -> None:
        self.episode_stats["chi_taken"] += 1
        super()._apply_chi(request, action)

    def _finish_draw(self) -> None:
        self.episode_stats["draw_games"] += 1
        super()._finish_draw()

    def _info_dict(self, acting_player: int | None) -> dict:
        info = super()._info_dict(acting_player)
        info["episode_stats"] = dict(self.episode_stats)
        return info

    def _empty_episode_stats(self) -> dict[str, int]:
        return {
            "self_draw_game_opportunities": 0,
            "discard_game_opportunities": 0,
            "complete_hand_zero_tai": 0,
            "peng_opportunities": 0,
            "chi_opportunities": 0,
            "peng_taken": 0,
            "chi_taken": 0,
            "self_draw_wins": 0,
            "discard_claim_wins": 0,
            "missed_game_actions": 0,
            "wins": 0,
            "draw_games": 0,
        }

    def _record_turn_state(self, player: int | None) -> None:
        if player is None or self.done or self.phase != PHASE_DISCARD_OR_WIN:
            return
        evaluation = self._evaluate_current_self_win(player)
        if evaluation.can_win:
            self.episode_stats["self_draw_game_opportunities"] += 1
        elif evaluation.is_complete and evaluation.tai == 0:
            self.episode_stats["complete_hand_zero_tai"] += 1

    def _progress_scores(self) -> np.ndarray:
        return np.asarray([self._progress_score_for_player(player) for player in range(NUM_PLAYERS)], dtype=np.float32)

    def _progress_score_for_player(self, player: int) -> float:
        analysis = self._analysis_for_player(player)
        score = 0.0
        score += 3.0 * analysis["total_melds"]
        score += 1.2 * analysis["has_pair"]
        score += 0.60 * min(analysis["partial_sets"], max(4 - int(analysis["total_melds"]), 0))
        score += 0.75 * analysis["tai_sources"]
        score += 6.0 * analysis["can_game"]
        score -= 1.0 * analysis["complete_zero_tai"]
        return float(score)

    def _legal_game_available_for_current_player(self) -> bool:
        if self.done or self.current_player is None:
            return False
        return bool(self.legal_action_mask(self.current_player)[ACTION_GAME] > 0.0)

    def _analysis_for_player(self, player: int) -> dict[str, float]:
        current_eval = self._win_evaluation_for_observer(player)
        counts = self._concealed_counts(player)
        melds_exposed = len(self.players[player].melds)
        tai_sources = 0
        for meld in self.players[player].melds:
            if meld.meld_type != MELD_PENG:
                continue
            tile_kind = meld.tile_kinds[0]
            if tile_kind == SEAT_WINDS[player]:
                tai_sources += 1
            if tile_kind == self.game_wind:
                tai_sources += 1
            if tile_kind == NORTH:
                tai_sources += 1
            if tile_kind in DRAGON_KINDS:
                tai_sources += 1

        sequence_first = self._greedy_completion_features(counts, prefer_sequences=True)
        triplet_first = self._greedy_completion_features(counts, prefer_sequences=False)
        concealed_melds, has_pair, partial_sets = max(
            (sequence_first, triplet_first),
            key=lambda item: (item[0], item[1], item[2]),
        )
        total_melds = melds_exposed + concealed_melds
        completion_score = (3.0 * total_melds) + (1.2 if has_pair else 0.0) + (0.60 * partial_sets) + (
            0.75 * tai_sources
        )
        return {
            "total_melds": float(min(total_melds, 4)),
            "has_pair": 1.0 if has_pair else 0.0,
            "partial_sets": float(min(partial_sets, 4)),
            "can_game": 1.0 if current_eval.can_win else 0.0,
            "complete_zero_tai": 1.0 if current_eval.is_complete and current_eval.tai == 0 else 0.0,
            "tai_sources": float(min(tai_sources, MAX_TAI)),
            "completion_score": float(completion_score),
        }

    def _greedy_completion_features(self, counts: list[int], prefer_sequences: bool) -> tuple[int, bool, int]:
        working = counts.copy()
        melds = 0

        if prefer_sequences:
            melds += self._extract_sequences(working)
            melds += self._extract_triplets(working)
            melds += self._extract_sequences(working)
        else:
            melds += self._extract_triplets(working)
            melds += self._extract_sequences(working)
            melds += self._extract_triplets(working)

        has_pair = any(value >= 2 for value in working)
        if has_pair:
            pair_kind = next(index for index, value in enumerate(working) if value >= 2)
            working[pair_kind] -= 2

        partial_sets = self._count_partials(working)
        return melds, has_pair, partial_sets

    def _extract_triplets(self, counts: list[int]) -> int:
        melds = 0
        for tile_kind in range(TILE_TYPE_COUNT):
            while counts[tile_kind] >= 3:
                counts[tile_kind] -= 3
                melds += 1
        return melds

    def _extract_sequences(self, counts: list[int]) -> int:
        melds = 0
        for suit_base in (0, 9):
            for offset in range(7):
                while counts[suit_base + offset] > 0 and counts[suit_base + offset + 1] > 0 and counts[suit_base + offset + 2] > 0:
                    counts[suit_base + offset] -= 1
                    counts[suit_base + offset + 1] -= 1
                    counts[suit_base + offset + 2] -= 1
                    melds += 1
        return melds

    def _count_partials(self, counts: list[int]) -> int:
        working = counts.copy()
        partials = 0

        for suit_base in (0, 9):
            for offset in range(8):
                while working[suit_base + offset] > 0 and working[suit_base + offset + 1] > 0:
                    working[suit_base + offset] -= 1
                    working[suit_base + offset + 1] -= 1
                    partials += 1
            for offset in range(7):
                while working[suit_base + offset] > 0 and working[suit_base + offset + 2] > 0:
                    working[suit_base + offset] -= 1
                    working[suit_base + offset + 2] -= 1
                    partials += 1

        for tile_kind in range(TILE_TYPE_COUNT):
            while working[tile_kind] >= 2:
                working[tile_kind] -= 2
                partials += 1

        return partials
