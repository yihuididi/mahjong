from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .constants import COPIES_PER_TILE, DRAGON_KINDS, MAX_TAI, NORTH, TILE_TYPE_COUNT, WIND_KINDS
from .rules import counts_from_kinds, evaluate_hand, tile_name


@dataclass(frozen=True)
class HandLabel:
    winning_shape_tile_kinds: tuple[int, ...]
    legal_game_tile_kinds: tuple[int, ...]
    improving_tile_kinds: tuple[int, ...]
    completion_score: float
    best_next_completion_score: float


@dataclass(frozen=True)
class DiscardLabel:
    best_discard_tile_kinds: tuple[int, ...]
    best_discard_score: float
    discard_scores_by_kind: tuple[tuple[int, float], ...]


def sample_concealed_hand_kinds(
    rng: np.random.Generator,
    hand_size: int = 13,
) -> list[int]:
    wall = np.repeat(np.arange(TILE_TYPE_COUNT, dtype=np.int16), COPIES_PER_TILE)
    sampled = rng.choice(wall, size=hand_size, replace=False)
    return sorted(int(kind) for kind in sampled.tolist())


def label_concealed_hand(
    hand_kinds: Iterable[int],
    seat_wind: int,
    game_wind: int,
) -> HandLabel:
    hand_list = sorted(int(kind) for kind in hand_kinds)
    if len(hand_list) != 13:
        raise ValueError(f"Expected a 13-tile concealed hand, got {len(hand_list)} tiles.")

    counts = counts_from_kinds(hand_list)
    if any(count > COPIES_PER_TILE for count in counts):
        raise ValueError("A hand cannot contain more than 4 copies of a tile kind.")

    base_score = completion_score_for_counts(counts, seat_wind, game_wind)
    winning_shape_tile_kinds: list[int] = []
    legal_game_tile_kinds: list[int] = []
    improving_tile_kinds: list[int] = []
    best_next_completion_score = base_score

    for tile_kind in range(TILE_TYPE_COUNT):
        if counts[tile_kind] >= COPIES_PER_TILE:
            continue

        next_counts = counts.copy()
        next_counts[tile_kind] += 1
        evaluation = evaluate_hand(next_counts, (), seat_wind, game_wind)
        next_score = completion_score_for_counts(next_counts, seat_wind, game_wind)

        if evaluation.is_complete:
            winning_shape_tile_kinds.append(tile_kind)
        if evaluation.can_win:
            legal_game_tile_kinds.append(tile_kind)
        if next_score > base_score + 1e-6:
            improving_tile_kinds.append(tile_kind)

        if next_score > best_next_completion_score:
            best_next_completion_score = next_score

    return HandLabel(
        winning_shape_tile_kinds=tuple(winning_shape_tile_kinds),
        legal_game_tile_kinds=tuple(legal_game_tile_kinds),
        improving_tile_kinds=tuple(improving_tile_kinds),
        completion_score=float(base_score),
        best_next_completion_score=float(best_next_completion_score),
    )


def label_concealed_discard_hand(
    hand_kinds: Iterable[int],
    seat_wind: int,
    game_wind: int,
) -> DiscardLabel:
    hand_list = sorted(int(kind) for kind in hand_kinds)
    if len(hand_list) != 14:
        raise ValueError(f"Expected a 14-tile concealed hand, got {len(hand_list)} tiles.")

    counts = counts_from_kinds(hand_list)
    if any(count > COPIES_PER_TILE for count in counts):
        raise ValueError("A hand cannot contain more than 4 copies of a tile kind.")

    scored_discards: list[tuple[int, float]] = []
    for tile_kind in sorted(set(hand_list)):
        next_hand = hand_list.copy()
        next_hand.remove(tile_kind)
        labels = label_concealed_hand(next_hand, seat_wind, game_wind)
        score = discard_keep_score(labels)
        scored_discards.append((tile_kind, score))

    best_score = max(score for _, score in scored_discards)
    best_discards = tuple(
        tile_kind for tile_kind, score in scored_discards if score >= best_score - 1e-6
    )

    return DiscardLabel(
        best_discard_tile_kinds=best_discards,
        best_discard_score=float(best_score),
        discard_scores_by_kind=tuple((tile_kind, float(score)) for tile_kind, score in scored_discards),
    )


def discard_keep_score(labels: HandLabel) -> float:
    return float(
        (12.0 * len(labels.legal_game_tile_kinds))
        + (4.0 * len(labels.winning_shape_tile_kinds))
        + (1.5 * len(labels.improving_tile_kinds))
        + (0.75 * labels.completion_score)
        + (0.25 * labels.best_next_completion_score)
    )


def completion_score_for_counts(
    counts: list[int],
    seat_wind: int,
    game_wind: int,
) -> float:
    return float(analyze_concealed_counts(counts, seat_wind, game_wind)["completion_score"])


def analyze_concealed_counts(
    counts: list[int],
    seat_wind: int,
    game_wind: int,
) -> dict[str, float]:
    evaluation = evaluate_hand(counts, (), seat_wind, game_wind)
    melds, has_pair, partial_sets = _best_concealed_structure(counts)
    score = 0.0
    score += 3.0 * melds
    score += 1.2 if has_pair else 0.0
    score += 0.6 * min(partial_sets, max(4 - melds, 0))
    score += 0.75 * min(evaluation.tai, MAX_TAI)
    score += 6.0 if evaluation.can_win else 0.0
    if evaluation.is_complete and evaluation.tai == 0:
        score -= 1.0

    pair_tai_potential = 0.0
    for tile_kind in WIND_KINDS:
        if counts[tile_kind] >= 2:
            if tile_kind == seat_wind:
                pair_tai_potential += 0.25
            if tile_kind == game_wind:
                pair_tai_potential += 0.25
    if counts[NORTH] >= 2:
        pair_tai_potential += 0.25
    for tile_kind in DRAGON_KINDS:
        if counts[tile_kind] >= 2:
            pair_tai_potential += 0.25

    score += pair_tai_potential

    return {
        "melds": float(melds),
        "has_pair": 1.0 if has_pair else 0.0,
        "partial_sets": float(partial_sets),
        "can_game": 1.0 if evaluation.can_win else 0.0,
        "complete_zero_tai": 1.0 if evaluation.is_complete and evaluation.tai == 0 else 0.0,
        "tai": float(min(evaluation.tai, MAX_TAI)),
        "pair_tai_potential": float(pair_tai_potential),
        "completion_score": float(score),
    }


def tile_names_from_kinds(tile_kinds: Iterable[int]) -> list[str]:
    return [tile_name(kind) for kind in tile_kinds]


def _best_concealed_structure(counts: list[int]) -> tuple[int, bool, int]:
    sequence_first = _greedy_completion_features(counts, prefer_sequences=True)
    triplet_first = _greedy_completion_features(counts, prefer_sequences=False)
    return max((sequence_first, triplet_first), key=lambda item: (item[0], item[1], item[2]))


def _greedy_completion_features(counts: list[int], prefer_sequences: bool) -> tuple[int, bool, int]:
    working = counts.copy()
    melds = 0

    if prefer_sequences:
        melds += _extract_sequences(working)
        melds += _extract_triplets(working)
        melds += _extract_sequences(working)
    else:
        melds += _extract_triplets(working)
        melds += _extract_sequences(working)
        melds += _extract_triplets(working)

    has_pair = any(value >= 2 for value in working)
    if has_pair:
        pair_kind = next(index for index, value in enumerate(working) if value >= 2)
        working[pair_kind] -= 2

    partial_sets = _count_partials(working)
    return melds, has_pair, partial_sets


def _extract_triplets(counts: list[int]) -> int:
    melds = 0
    for tile_kind in range(TILE_TYPE_COUNT):
        while counts[tile_kind] >= 3:
            counts[tile_kind] -= 3
            melds += 1
    return melds


def _extract_sequences(counts: list[int]) -> int:
    melds = 0
    for suit_base in (0, 9):
        for offset in range(7):
            first = suit_base + offset
            second = suit_base + offset + 1
            third = suit_base + offset + 2
            while counts[first] > 0 and counts[second] > 0 and counts[third] > 0:
                counts[first] -= 1
                counts[second] -= 1
                counts[third] -= 1
                melds += 1
    return melds


def _count_partials(counts: list[int]) -> int:
    working = counts.copy()
    partials = 0

    for suit_base in (0, 9):
        for offset in range(8):
            left = suit_base + offset
            right = suit_base + offset + 1
            while working[left] > 0 and working[right] > 0:
                working[left] -= 1
                working[right] -= 1
                partials += 1
        for offset in range(7):
            left = suit_base + offset
            gap = suit_base + offset + 2
            while working[left] > 0 and working[gap] > 0:
                working[left] -= 1
                working[gap] -= 1
                partials += 1

    for tile_kind in range(TILE_TYPE_COUNT):
        while working[tile_kind] >= 2:
            working[tile_kind] -= 2
            partials += 1

    return partials
