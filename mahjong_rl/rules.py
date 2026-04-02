from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .constants import (
    ACTION_CHI_CENTER,
    ACTION_CHI_LEFT,
    ACTION_CHI_RIGHT,
    BAMBOO_BASE,
    DOT_BASE,
    DRAGON_KINDS,
    MAX_TAI,
    MELD_CHI,
    MELD_PENG,
    MIN_TAI_TO_WIN,
    NORTH,
    TILE_NAMES,
    TILE_TYPE_COUNT,
)


@dataclass(frozen=True)
class MeldPattern:
    meld_type: str
    tile_kinds: tuple[int, int, int]


@dataclass(frozen=True)
class WinEvaluation:
    is_complete: bool
    tai: int
    pair_kind: int | None = None
    concealed_melds: tuple[MeldPattern, ...] = ()

    @property
    def can_win(self) -> bool:
        return self.is_complete and self.tai >= MIN_TAI_TO_WIN


def tile_name(kind: int) -> str:
    return TILE_NAMES[kind]


def is_honor(kind: int) -> bool:
    return kind >= 18


def tile_suit(kind: int) -> int | None:
    if 0 <= kind <= 8:
        return 0
    if 9 <= kind <= 17:
        return 1
    return None


def tile_rank(kind: int) -> int | None:
    suit = tile_suit(kind)
    if suit is None:
        return None
    return (kind - (BAMBOO_BASE if suit == 0 else DOT_BASE)) + 1


def is_suit_tile(kind: int) -> bool:
    return tile_suit(kind) is not None


def same_suit(a: int, b: int) -> bool:
    return tile_suit(a) is not None and tile_suit(a) == tile_suit(b)


def empty_counts() -> list[int]:
    return [0] * TILE_TYPE_COUNT


def counts_from_kinds(tile_kinds: Iterable[int]) -> list[int]:
    counts = empty_counts()
    for kind in tile_kinds:
        counts[kind] += 1
    return counts


def available_chi_actions(hand_counts: list[int], claim_kind: int) -> tuple[int, ...]:
    if not is_suit_tile(claim_kind):
        return ()

    base = BAMBOO_BASE if claim_kind <= 8 else DOT_BASE
    rank = tile_rank(claim_kind)
    assert rank is not None

    actions: list[int] = []

    if rank <= 7:
        left_a = base + rank
        left_b = base + rank + 1
        if hand_counts[left_a] > 0 and hand_counts[left_b] > 0:
            actions.append(ACTION_CHI_LEFT)

    if 2 <= rank <= 8:
        center_a = base + rank - 2
        center_b = base + rank
        if hand_counts[center_a] > 0 and hand_counts[center_b] > 0:
            actions.append(ACTION_CHI_CENTER)

    if rank >= 3:
        right_a = base + rank - 3
        right_b = base + rank - 2
        if hand_counts[right_a] > 0 and hand_counts[right_b] > 0:
            actions.append(ACTION_CHI_RIGHT)

    return tuple(actions)


def chi_sequence_for_action(claim_kind: int, action: int) -> tuple[int, int, int]:
    if not is_suit_tile(claim_kind):
        raise ValueError("Chi is only available for suit tiles.")

    suit = tile_suit(claim_kind)
    assert suit is not None
    base = BAMBOO_BASE if suit == 0 else DOT_BASE
    rank = tile_rank(claim_kind)
    assert rank is not None

    if action == ACTION_CHI_LEFT:
        return (claim_kind, base + rank, base + rank + 1)
    if action == ACTION_CHI_CENTER:
        return (base + rank - 2, claim_kind, base + rank)
    if action == ACTION_CHI_RIGHT:
        return (base + rank - 3, base + rank - 2, claim_kind)
    raise ValueError(f"Unsupported chi action: {action}")


@lru_cache(maxsize=200_000)
def _enumerate_groupings(
    counts: tuple[int, ...],
    melds_needed: int,
    pair_kind: int,
) -> tuple[tuple[int, tuple[MeldPattern, ...]], ...]:
    total = sum(counts)
    if total == 0:
        if melds_needed == 0 and pair_kind != -1:
            return ((pair_kind, ()),)
        return ()

    if total != melds_needed * 3 + (0 if pair_kind != -1 else 2):
        return ()

    first = next(index for index, value in enumerate(counts) if value > 0)
    counts_list = list(counts)
    results: list[tuple[int, tuple[MeldPattern, ...]]] = []

    if pair_kind == -1 and counts_list[first] >= 2:
        counts_list[first] -= 2
        for resolved_pair, melds in _enumerate_groupings(tuple(counts_list), melds_needed, first):
            results.append((resolved_pair, melds))
        counts_list[first] += 2

    if melds_needed == 0:
        return tuple(results)

    if counts_list[first] >= 3:
        counts_list[first] -= 3
        pattern = MeldPattern(MELD_PENG, (first, first, first))
        for resolved_pair, melds in _enumerate_groupings(tuple(counts_list), melds_needed - 1, pair_kind):
            results.append((resolved_pair, (pattern,) + melds))
        counts_list[first] += 3

    suit = tile_suit(first)
    rank = tile_rank(first)
    if suit is not None and rank is not None and rank <= 7:
        second = first + 1
        third = first + 2
        if tile_suit(second) == suit and tile_suit(third) == suit:
            if counts_list[second] > 0 and counts_list[third] > 0:
                counts_list[first] -= 1
                counts_list[second] -= 1
                counts_list[third] -= 1
                pattern = MeldPattern(MELD_CHI, (first, second, third))
                for resolved_pair, melds in _enumerate_groupings(tuple(counts_list), melds_needed - 1, pair_kind):
                    results.append((resolved_pair, (pattern,) + melds))
                counts_list[first] += 1
                counts_list[second] += 1
                counts_list[third] += 1

    return tuple(results)


def _score_tai(
    pair_kind: int,
    concealed_melds: tuple[MeldPattern, ...],
    exposed_melds: tuple[MeldPattern, ...],
    seat_wind: int,
    game_wind: int,
) -> int:
    all_melds = exposed_melds + concealed_melds
    all_tiles = [pair_kind, pair_kind]
    for meld in all_melds:
        all_tiles.extend(meld.tile_kinds)

    if all(is_honor(kind) for kind in all_tiles):
        return MAX_TAI

    tai = 0

    for meld in exposed_melds:
        if meld.meld_type != MELD_PENG:
            continue
        tile = meld.tile_kinds[0]
        if tile == seat_wind:
            tai += 1
        if tile == game_wind:
            tai += 1
        if tile == NORTH:
            tai += 1
        if tile in DRAGON_KINDS:
            tai += 1

    if all(meld.meld_type == MELD_CHI for meld in all_melds):
        tai += 1

    if all(meld.meld_type == MELD_PENG for meld in all_melds):
        tai += 2

    suits = {tile_suit(kind) for kind in all_tiles if tile_suit(kind) is not None}
    has_honor = any(is_honor(kind) for kind in all_tiles)
    if len(suits) == 1 and not has_honor:
        tai += 4
    elif len(suits) == 1 and has_honor:
        tai += 2

    return min(tai, MAX_TAI)


def evaluate_hand(
    concealed_counts: list[int],
    exposed_melds: Iterable[MeldPattern],
    seat_wind: int,
    game_wind: int,
) -> WinEvaluation:
    exposed_tuple = tuple(exposed_melds)
    melds_needed = 4 - len(exposed_tuple)
    if melds_needed < 0:
        return WinEvaluation(is_complete=False, tai=0)

    groupings = _enumerate_groupings(tuple(concealed_counts), melds_needed, -1)
    if not groupings:
        return WinEvaluation(is_complete=False, tai=0)

    best_tai = -1
    best_pair: int | None = None
    best_melds: tuple[MeldPattern, ...] = ()

    for pair_kind, melds in groupings:
        tai = _score_tai(pair_kind, melds, exposed_tuple, seat_wind, game_wind)
        if tai > best_tai:
            best_tai = tai
            best_pair = pair_kind
            best_melds = melds

    return WinEvaluation(
        is_complete=True,
        tai=max(best_tai, 0),
        pair_kind=best_pair,
        concealed_melds=best_melds,
    )
