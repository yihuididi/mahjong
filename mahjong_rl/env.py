from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import (
    ACTION_CHI_CENTER,
    ACTION_CHI_LEFT,
    ACTION_CHI_RIGHT,
    ACTION_COUNT,
    ACTION_DISCARD_BASE,
    ACTION_GAME,
    ACTION_PASS,
    ACTION_PENG,
    ACTION_MEANINGS,
    COPIES_PER_TILE,
    DISCARD_WIN_REWARD,
    DRAW_END_TILES,
    MAX_CONCEALED_TILES,
    MAX_TAI,
    NUM_PLAYERS,
    PHASES,
    PHASE_CLAIM_CHI,
    PHASE_CLAIM_PENG,
    PHASE_CLAIM_WIN,
    PHASE_DISCARD_OR_WIN,
    PHASE_POST_MELD_DISCARD,
    SEAT_WINDS,
    SELF_DRAW_PAYOFF,
    STATUS_COUNT,
    STATUS_DISCARD_BASE,
    STATUS_EXPOSED_BASE,
    STATUS_HAND_BASE,
    STATUS_UNDRAWN,
    TILE_TYPE_COUNT,
    TOTAL_TILES,
    WIND_KINDS,
    MELD_CHI,
    MELD_PENG,
)
from .rules import (
    MeldPattern,
    WinEvaluation,
    available_chi_actions,
    chi_sequence_for_action,
    counts_from_kinds,
    evaluate_hand,
)


@dataclass
class PhysicalTile:
    tile_id: int
    kind: int
    copy_index: int
    status: int = STATUS_UNDRAWN


@dataclass
class ExposedMeld:
    meld_type: str
    tile_ids: tuple[int, int, int]
    tile_kinds: tuple[int, int, int]
    source_player: int


@dataclass
class PlayerState:
    concealed: list[int] = field(default_factory=list)
    melds: list[ExposedMeld] = field(default_factory=list)
    discards: list[int] = field(default_factory=list)
    needs_draw: bool = True


@dataclass(frozen=True)
class ClaimRequest:
    claim_type: str
    player: int
    thrower: int
    tile_id: int
    tile_kind: int
    chi_actions: tuple[int, ...] = ()


@dataclass
class EnvStep:
    observation: np.ndarray | None
    action_mask: np.ndarray | None
    current_player: int | None
    rewards: np.ndarray
    done: bool
    info: dict[str, Any]


class MahjongEnv:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.observation_size = self._calculate_observation_size()
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
        if seed is not None:
            self.rng.seed(seed)

        self.tiles = [
            PhysicalTile(tile_id=(kind * COPIES_PER_TILE) + copy_idx, kind=kind, copy_index=copy_idx)
            for kind in range(TILE_TYPE_COUNT)
            for copy_idx in range(COPIES_PER_TILE)
        ]
        self.players = [PlayerState() for _ in range(NUM_PLAYERS)]
        self.game_wind = self.rng.choice(WIND_KINDS)
        self.starting_player = self.rng.randrange(NUM_PLAYERS)
        self.current_player = self.starting_player
        self.phase = PHASE_DISCARD_OR_WIN
        self.claim_queue: list[ClaimRequest] = []
        self.claim_index = 0
        self.pending_discard_tile_id: int | None = None
        self.pending_discarder: int | None = None
        self.done = False
        self.winner: int | None = None
        self.winner_tai = 0
        self.terminal_reason: str | None = None

        for player_id in range(NUM_PLAYERS):
            self.players[player_id].needs_draw = player_id != self.starting_player

        self._deal_initial_hands()
        return self.observe(), self.legal_action_mask(), self.current_player

    def observe(self, player: int | None = None) -> np.ndarray:
        if player is None:
            player = self.current_player
        if player is None:
            raise ValueError("No current player is available in a terminal state.")

        tile_status_features = np.zeros((TOTAL_TILES, STATUS_COUNT), dtype=np.float32)
        for tile in self.tiles:
            rel_status = self._relative_status(tile.status, player)
            tile_status_features[tile.tile_id, rel_status] = 1.0

        hand_slot_features = np.zeros((MAX_CONCEALED_TILES, TILE_TYPE_COUNT + 1), dtype=np.float32)
        for slot_index, tile_id in enumerate(self._sorted_concealed(player)):
            hand_slot_features[slot_index, self.tiles[tile_id].kind] = 1.0
        for slot_index in range(len(self.players[player].concealed), MAX_CONCEALED_TILES):
            hand_slot_features[slot_index, TILE_TYPE_COUNT] = 1.0

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
        scalar_features = np.array(
            [
                self.tiles_left / float(TOTAL_TILES),
                len(self.players[player].concealed) / float(MAX_CONCEALED_TILES),
                1.0 if current_eval.is_complete else 0.0,
                current_eval.tai / float(MAX_TAI),
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                tile_status_features.reshape(-1),
                hand_slot_features.reshape(-1),
                phase_features,
                seat_features,
                wind_features,
                claim_tile_features,
                claim_from_features,
                scalar_features,
            ]
        ).astype(np.float32)

    def legal_action_mask(self, player: int | None = None) -> np.ndarray:
        if self.done:
            return np.zeros(ACTION_COUNT, dtype=np.float32)
        if player is None:
            player = self.current_player
        if player != self.current_player:
            raise ValueError("Legal actions are only defined for the current player.")

        mask = np.zeros(ACTION_COUNT, dtype=np.float32)

        if self.phase in (PHASE_DISCARD_OR_WIN, PHASE_POST_MELD_DISCARD):
            concealed = self._sorted_concealed(player)
            for slot_index in range(len(concealed)):
                mask[ACTION_DISCARD_BASE + slot_index] = 1.0
            if self.phase == PHASE_DISCARD_OR_WIN and self._evaluate_current_self_win(player).can_win:
                mask[ACTION_GAME] = 1.0
            return mask

        request = self._current_claim_request()
        if request is None:
            raise RuntimeError("Claim phase without an active request.")

        mask[ACTION_PASS] = 1.0
        if self.phase == PHASE_CLAIM_WIN:
            mask[ACTION_GAME] = 1.0
        elif self.phase == PHASE_CLAIM_PENG:
            mask[ACTION_PENG] = 1.0
        elif self.phase == PHASE_CLAIM_CHI:
            for action in request.chi_actions:
                mask[action] = 1.0
        else:
            raise RuntimeError(f"Unknown phase: {self.phase}")
        return mask

    def step(self, action: int) -> EnvStep:
        if self.done:
            raise RuntimeError("Cannot call step() after the game has ended.")

        mask = self.legal_action_mask()
        if action < 0 or action >= ACTION_COUNT or mask[action] == 0.0:
            raise ValueError(f"Illegal action {action} in phase {self.phase}.")

        acting_player = self.current_player
        rewards = np.zeros(NUM_PLAYERS, dtype=np.float32)

        if self.phase in (PHASE_DISCARD_OR_WIN, PHASE_POST_MELD_DISCARD):
            rewards = self._resolve_turn_action(acting_player, action)
        elif self.phase == PHASE_CLAIM_WIN:
            rewards = self._resolve_claim_win(action)
        elif self.phase == PHASE_CLAIM_PENG:
            rewards = self._resolve_claim_peng(action)
        elif self.phase == PHASE_CLAIM_CHI:
            rewards = self._resolve_claim_chi(action)
        else:
            raise RuntimeError(f"Unhandled phase: {self.phase}")

        if self.done:
            return EnvStep(
                observation=None,
                action_mask=None,
                current_player=None,
                rewards=rewards,
                done=True,
                info=self._info_dict(acting_player),
            )

        return EnvStep(
            observation=self.observe(),
            action_mask=self.legal_action_mask(),
            current_player=self.current_player,
            rewards=rewards,
            done=False,
            info=self._info_dict(acting_player),
        )

    @property
    def action_meanings(self) -> tuple[str, ...]:
        return tuple(item.label for item in ACTION_MEANINGS)

    @property
    def tiles_left(self) -> int:
        return sum(1 for tile in self.tiles if tile.status == STATUS_UNDRAWN)

    def _calculate_observation_size(self) -> int:
        tile_features = TOTAL_TILES * STATUS_COUNT
        hand_slot_features = MAX_CONCEALED_TILES * (TILE_TYPE_COUNT + 1)
        phase_features = len(PHASES)
        seat_features = NUM_PLAYERS
        wind_features = len(WIND_KINDS)
        claim_tile_features = TILE_TYPE_COUNT + 1
        claim_from_features = NUM_PLAYERS + 1
        scalar_features = 4
        return (
            tile_features
            + hand_slot_features
            + phase_features
            + seat_features
            + wind_features
            + claim_tile_features
            + claim_from_features
            + scalar_features
        )

    def _deal_initial_hands(self) -> None:
        for player_id in range(NUM_PLAYERS):
            hand_size = 14 if player_id == self.starting_player else 13
            for _ in range(hand_size):
                self._draw_random_tile(player_id)
        self.players[self.starting_player].needs_draw = False

    def _draw_random_tile(self, player: int) -> int:
        available = [tile.tile_id for tile in self.tiles if tile.status == STATUS_UNDRAWN]
        if not available:
            raise RuntimeError("The wall is empty.")
        tile_id = self.rng.choice(available)
        self.tiles[tile_id].status = STATUS_HAND_BASE + player
        self.players[player].concealed.append(tile_id)
        return tile_id

    def _resolve_turn_action(self, player: int, action: int) -> np.ndarray:
        if action == ACTION_GAME:
            evaluation = self._evaluate_current_self_win(player)
            return self._finish_self_draw_win(player, evaluation.tai)

        tile_id = self._discard_from_slot(player, action - ACTION_DISCARD_BASE)
        self.players[player].needs_draw = True
        self.pending_discard_tile_id = tile_id
        self.pending_discarder = player
        self.claim_queue = self._build_claim_requests(player, tile_id)
        self.claim_index = 0

        if self.claim_queue:
            request = self.claim_queue[0]
            self.current_player = request.player
            self.phase = self._phase_for_claim(request.claim_type)
        else:
            self._reset_pending_claim_state()
            self._advance_after_unclaimed_discard(player)

        return np.zeros(NUM_PLAYERS, dtype=np.float32)

    def _resolve_claim_win(self, action: int) -> np.ndarray:
        request = self._current_claim_request()
        if request is None:
            raise RuntimeError("No claim is active.")
        if action == ACTION_GAME:
            evaluation = self._evaluate_win_with_claim(request.player, request.tile_kind)
            return self._finish_discard_win(request.player, request.thrower, evaluation.tai)
        return self._pass_claim()

    def _resolve_claim_peng(self, action: int) -> np.ndarray:
        request = self._current_claim_request()
        if request is None:
            raise RuntimeError("No peng claim is active.")
        if action == ACTION_PENG:
            self._apply_peng(request)
            return np.zeros(NUM_PLAYERS, dtype=np.float32)
        return self._pass_claim()

    def _resolve_claim_chi(self, action: int) -> np.ndarray:
        request = self._current_claim_request()
        if request is None:
            raise RuntimeError("No chi claim is active.")
        if action != ACTION_PASS:
            self._apply_chi(request, action)
            return np.zeros(NUM_PLAYERS, dtype=np.float32)
        return self._pass_claim()

    def _pass_claim(self) -> np.ndarray:
        thrower = self.pending_discarder
        if thrower is None:
            raise RuntimeError("No discarder is available for claim resolution.")

        self.claim_index += 1
        if self.claim_index < len(self.claim_queue):
            request = self.claim_queue[self.claim_index]
            self.current_player = request.player
            self.phase = self._phase_for_claim(request.claim_type)
            return np.zeros(NUM_PLAYERS, dtype=np.float32)

        self._reset_pending_claim_state()
        self._advance_after_unclaimed_discard(thrower)
        return np.zeros(NUM_PLAYERS, dtype=np.float32)

    def _advance_after_unclaimed_discard(self, thrower: int) -> None:
        if self.tiles_left <= DRAW_END_TILES:
            self._finish_draw()
            return

        next_player = (thrower + 1) % NUM_PLAYERS
        self.current_player = next_player
        self._start_regular_turn(next_player)

    def _start_regular_turn(self, player: int) -> None:
        if self.players[player].needs_draw:
            if self.tiles_left <= DRAW_END_TILES:
                self._finish_draw()
                return
            self._draw_random_tile(player)
            self.players[player].needs_draw = False
        self.phase = PHASE_DISCARD_OR_WIN

    def _discard_from_slot(self, player: int, slot_index: int) -> int:
        concealed = self._sorted_concealed(player)
        if slot_index < 0 or slot_index >= len(concealed):
            raise ValueError(f"Discard slot {slot_index} is not available.")
        tile_id = concealed[slot_index]
        self.players[player].concealed.remove(tile_id)
        self.tiles[tile_id].status = STATUS_DISCARD_BASE + player
        self.players[player].discards.append(tile_id)
        return tile_id

    def _apply_peng(self, request: ClaimRequest) -> None:
        player = request.player
        matched_ids = self._take_matching_concealed_tile_ids(player, request.tile_kind, 2)
        claim_tile = request.tile_id
        self._claim_discard_tile_from_thrower(request.thrower, claim_tile)

        meld_ids = tuple(matched_ids + [claim_tile])
        for tile_id in meld_ids:
            self.tiles[tile_id].status = STATUS_EXPOSED_BASE + player

        self.players[player].melds.append(
            ExposedMeld(
                meld_type=MELD_PENG,
                tile_ids=meld_ids,
                tile_kinds=(request.tile_kind, request.tile_kind, request.tile_kind),
                source_player=request.thrower,
            )
        )
        self.players[player].needs_draw = False
        self.current_player = player
        self.phase = PHASE_POST_MELD_DISCARD
        self._reset_pending_claim_state()

    def _apply_chi(self, request: ClaimRequest, action: int) -> None:
        if action not in request.chi_actions:
            raise ValueError(f"Chi action {action} is not valid for the current request.")

        player = request.player
        sequence = chi_sequence_for_action(request.tile_kind, action)
        needed_kinds = list(sequence)
        needed_kinds.remove(request.tile_kind)
        matched_ids = []
        for tile_kind in needed_kinds:
            matched_ids.extend(self._take_matching_concealed_tile_ids(player, tile_kind, 1))

        claim_tile = request.tile_id
        self._claim_discard_tile_from_thrower(request.thrower, claim_tile)

        meld_ids = tuple(matched_ids + [claim_tile])
        for tile_id in meld_ids:
            self.tiles[tile_id].status = STATUS_EXPOSED_BASE + player

        self.players[player].melds.append(
            ExposedMeld(
                meld_type=MELD_CHI,
                tile_ids=meld_ids,
                tile_kinds=sequence,
                source_player=request.thrower,
            )
        )
        self.players[player].needs_draw = False
        self.current_player = player
        self.phase = PHASE_POST_MELD_DISCARD
        self._reset_pending_claim_state()

    def _claim_discard_tile_from_thrower(self, thrower: int, tile_id: int) -> None:
        self.players[thrower].discards.remove(tile_id)

    def _finish_self_draw_win(self, winner: int, tai: int) -> np.ndarray:
        capped_tai = min(max(tai, 1), MAX_TAI)
        unit = SELF_DRAW_PAYOFF[capped_tai]
        rewards = np.full(NUM_PLAYERS, -unit, dtype=np.float32)
        rewards[winner] = unit * (NUM_PLAYERS - 1)

        self.done = True
        self.winner = winner
        self.winner_tai = capped_tai
        self.terminal_reason = "self_draw"
        return rewards

    def _finish_discard_win(self, winner: int, shooter: int, tai: int) -> np.ndarray:
        capped_tai = min(max(tai, 1), MAX_TAI)
        reward = DISCARD_WIN_REWARD[capped_tai]
        rewards = np.zeros(NUM_PLAYERS, dtype=np.float32)
        rewards[winner] = reward
        rewards[shooter] = -reward

        self.done = True
        self.winner = winner
        self.winner_tai = capped_tai
        self.terminal_reason = "discard_win"
        return rewards

    def _finish_draw(self) -> None:
        self.done = True
        self.winner = None
        self.winner_tai = 0
        self.terminal_reason = "draw"
        self.current_player = None

    def _build_claim_requests(self, thrower: int, tile_id: int) -> list[ClaimRequest]:
        tile_kind = self.tiles[tile_id].kind
        order = [((thrower + offset) % NUM_PLAYERS) for offset in range(1, NUM_PLAYERS)]
        requests: list[ClaimRequest] = []

        for player in order:
            if self._evaluate_win_with_claim(player, tile_kind).can_win:
                requests.append(
                    ClaimRequest(
                        claim_type=PHASE_CLAIM_WIN,
                        player=player,
                        thrower=thrower,
                        tile_id=tile_id,
                        tile_kind=tile_kind,
                    )
                )

        for player in order:
            if self._can_peng(player, tile_kind):
                requests.append(
                    ClaimRequest(
                        claim_type=PHASE_CLAIM_PENG,
                        player=player,
                        thrower=thrower,
                        tile_id=tile_id,
                        tile_kind=tile_kind,
                    )
                )

        chi_player = order[0]
        chi_actions = available_chi_actions(self._concealed_counts(chi_player), tile_kind)
        if chi_actions:
            requests.append(
                ClaimRequest(
                    claim_type=PHASE_CLAIM_CHI,
                    player=chi_player,
                    thrower=thrower,
                    tile_id=tile_id,
                    tile_kind=tile_kind,
                    chi_actions=chi_actions,
                )
            )

        return requests

    def _phase_for_claim(self, claim_type: str) -> str:
        if claim_type == PHASE_CLAIM_WIN:
            return PHASE_CLAIM_WIN
        if claim_type == PHASE_CLAIM_PENG:
            return PHASE_CLAIM_PENG
        if claim_type == PHASE_CLAIM_CHI:
            return PHASE_CLAIM_CHI
        raise ValueError(f"Unknown claim type: {claim_type}")

    def _current_claim_request(self) -> ClaimRequest | None:
        if not self.claim_queue:
            return None
        return self.claim_queue[self.claim_index]

    def _active_claim_tile_kind(self) -> int | None:
        request = self._current_claim_request()
        return request.tile_kind if request is not None else None

    def _active_claim_thrower(self) -> int | None:
        request = self._current_claim_request()
        return request.thrower if request is not None else None

    def _reset_pending_claim_state(self) -> None:
        self.claim_queue = []
        self.claim_index = 0
        self.pending_discard_tile_id = None
        self.pending_discarder = None

    def _concealed_counts(self, player: int) -> list[int]:
        return counts_from_kinds(self.tiles[tile_id].kind for tile_id in self.players[player].concealed)

    def _exposed_patterns(self, player: int) -> tuple[MeldPattern, ...]:
        return tuple(MeldPattern(meld.meld_type, meld.tile_kinds) for meld in self.players[player].melds)

    def _evaluate_current_self_win(self, player: int) -> WinEvaluation:
        return evaluate_hand(
            self._concealed_counts(player),
            self._exposed_patterns(player),
            SEAT_WINDS[player],
            self.game_wind,
        )

    def _evaluate_win_with_claim(self, player: int, claim_tile_kind: int) -> WinEvaluation:
        counts = self._concealed_counts(player)
        counts[claim_tile_kind] += 1
        return evaluate_hand(
            counts,
            self._exposed_patterns(player),
            SEAT_WINDS[player],
            self.game_wind,
        )

    def _win_evaluation_for_observer(self, player: int) -> WinEvaluation:
        request = self._current_claim_request()
        if request is not None and request.player == player and self.phase == PHASE_CLAIM_WIN:
            return self._evaluate_win_with_claim(player, request.tile_kind)
        return self._evaluate_current_self_win(player)

    def _can_peng(self, player: int, tile_kind: int) -> bool:
        counts = self._concealed_counts(player)
        return counts[tile_kind] >= 2

    def _take_matching_concealed_tile_ids(self, player: int, tile_kind: int, count: int) -> list[int]:
        matched = sorted(
            [tile_id for tile_id in self.players[player].concealed if self.tiles[tile_id].kind == tile_kind],
            key=lambda tile_id: (self.tiles[tile_id].kind, tile_id),
        )
        if len(matched) < count:
            raise RuntimeError(f"Player {player} does not have {count} copies of tile kind {tile_kind}.")
        selected = matched[:count]
        for tile_id in selected:
            self.players[player].concealed.remove(tile_id)
        return selected

    def _sorted_concealed(self, player: int) -> list[int]:
        return sorted(self.players[player].concealed, key=lambda tile_id: (self.tiles[tile_id].kind, tile_id))

    def _relative_player(self, observer: int, absolute_player: int) -> int:
        return (absolute_player - observer) % NUM_PLAYERS

    def _relative_status(self, status: int, observer: int) -> int:
        if status == STATUS_UNDRAWN:
            return STATUS_UNDRAWN
        if STATUS_HAND_BASE <= status < STATUS_EXPOSED_BASE:
            owner = status - STATUS_HAND_BASE
            return STATUS_HAND_BASE if owner == observer else STATUS_UNDRAWN
        if STATUS_EXPOSED_BASE <= status < STATUS_DISCARD_BASE:
            owner = status - STATUS_EXPOSED_BASE
            relative = self._relative_player(observer, owner)
            return STATUS_EXPOSED_BASE if relative == 0 else STATUS_EXPOSED_BASE + relative
        owner = status - STATUS_DISCARD_BASE
        relative = self._relative_player(observer, owner)
        return STATUS_DISCARD_BASE if relative == 0 else STATUS_DISCARD_BASE + relative

    def _info_dict(self, acting_player: int | None) -> dict[str, Any]:
        return {
            "acting_player": acting_player,
            "current_player": self.current_player,
            "starting_player": self.starting_player,
            "dealer": 0,
            "game_wind": self.game_wind,
            "phase": self.phase,
            "tiles_left": self.tiles_left,
            "winner": self.winner,
            "winner_tai": self.winner_tai,
            "terminal_reason": self.terminal_reason,
        }
