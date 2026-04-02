from __future__ import annotations

import numpy as np

from .constants import (
    MAX_CONCEALED_TILES,
    MAX_TAI,
    NUM_PLAYERS,
    PHASE_DISCARD_OR_WIN,
    PHASES,
    SEAT_WINDS,
    TILE_TYPE_COUNT,
    TOTAL_TILES,
    WIND_KINDS,
)
from .hand_labels import analyze_concealed_counts
from .rules import counts_from_kinds, evaluate_hand


def compact_observation_size() -> int:
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


def compact_observation_from_concealed_hand(
    hand_kinds: list[int],
    player_index: int,
    game_wind: int,
    tiles_left: int = TOTAL_TILES,
) -> np.ndarray:
    hand_counts_list = counts_from_kinds(hand_kinds)
    hand_counts = np.asarray(hand_counts_list, dtype=np.float32)

    exposed_counts = np.zeros((NUM_PLAYERS, TILE_TYPE_COUNT), dtype=np.float32)
    discard_counts = np.zeros((NUM_PLAYERS, TILE_TYPE_COUNT), dtype=np.float32)

    phase_features = np.zeros(len(PHASES), dtype=np.float32)
    phase_features[PHASES.index(PHASE_DISCARD_OR_WIN)] = 1.0

    seat_features = np.zeros(NUM_PLAYERS, dtype=np.float32)
    seat_features[player_index] = 1.0

    wind_features = np.zeros(len(WIND_KINDS), dtype=np.float32)
    wind_features[WIND_KINDS.index(game_wind)] = 1.0

    claim_tile_features = np.zeros(TILE_TYPE_COUNT + 1, dtype=np.float32)
    claim_tile_features[TILE_TYPE_COUNT] = 1.0

    claim_from_features = np.zeros(NUM_PLAYERS + 1, dtype=np.float32)
    claim_from_features[0] = 1.0

    current_eval = evaluate_hand(hand_counts_list, (), SEAT_WINDS[player_index], game_wind)
    analysis = analyze_concealed_counts(hand_counts_list, SEAT_WINDS[player_index], game_wind)
    scalar_features = np.array(
        [
            tiles_left / float(TOTAL_TILES),
            len(hand_kinds) / float(MAX_CONCEALED_TILES),
            1.0 if current_eval.is_complete else 0.0,
            current_eval.tai / float(MAX_TAI),
            analysis["melds"] / 4.0,
            analysis["has_pair"],
            analysis["partial_sets"] / 4.0,
            analysis["can_game"],
            analysis["complete_zero_tai"],
            analysis["pair_tai_potential"] / float(MAX_TAI),
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
