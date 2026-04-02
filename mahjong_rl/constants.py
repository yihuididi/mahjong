from __future__ import annotations

from dataclasses import dataclass


NUM_PLAYERS = 3
TILE_TYPE_COUNT = 25
COPIES_PER_TILE = 4
TOTAL_TILES = TILE_TYPE_COUNT * COPIES_PER_TILE
MAX_CONCEALED_TILES = 14
MAX_TAI = 5
MIN_TAI_TO_WIN = 1
DRAW_END_TILES = 15


BAMBOO_BASE = 0
DOT_BASE = 9
EAST = 18
SOUTH = 19
WEST = 20
NORTH = 21
RED = 22
GREEN = 23
WHITE = 24

WIND_KINDS = (EAST, SOUTH, WEST, NORTH)
DRAGON_KINDS = (RED, GREEN, WHITE)
SUIT_NAMES = ("bamboo", "dot")
SEAT_WINDS = (EAST, SOUTH, WEST)

TILE_NAMES = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B9",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
    "D9",
    "E",
    "S",
    "W",
    "N",
    "R",
    "G",
    "Wh",
)


STATUS_UNDRAWN = 0
STATUS_HAND_BASE = 1
STATUS_EXPOSED_BASE = 4
STATUS_DISCARD_BASE = 7
STATUS_COUNT = 10


ACTION_DISCARD_BASE = 0
ACTION_CHI_LEFT = 14
ACTION_CHI_CENTER = 15
ACTION_CHI_RIGHT = 16
ACTION_PENG = 17
ACTION_GAME = 18
ACTION_PASS = 19
ACTION_COUNT = 20

CHI_ACTIONS = (ACTION_CHI_LEFT, ACTION_CHI_CENTER, ACTION_CHI_RIGHT)


PHASE_DISCARD_OR_WIN = "discard_or_win"
PHASE_CLAIM_WIN = "claim_win"
PHASE_CLAIM_PENG = "claim_peng"
PHASE_CLAIM_CHI = "claim_chi"
PHASE_POST_MELD_DISCARD = "post_meld_discard"

PHASES = (
    PHASE_DISCARD_OR_WIN,
    PHASE_CLAIM_WIN,
    PHASE_CLAIM_PENG,
    PHASE_CLAIM_CHI,
    PHASE_POST_MELD_DISCARD,
)


MELD_CHI = "chi"
MELD_PENG = "peng"


SELF_DRAW_PAYOFF = {
    1: 2.0,
    2: 4.0,
    3: 8.0,
    4: 16.0,
    5: 32.0,
}

DISCARD_WIN_REWARD = {
    1: 4.0,
    2: 8.0,
    3: 16.0,
    4: 32.0,
    5: 64.0,
}


@dataclass(frozen=True)
class ActionMeaning:
    index: int
    label: str


ACTION_MEANINGS = (
    tuple(ActionMeaning(ACTION_DISCARD_BASE + idx, f"discard_slot_{idx}") for idx in range(MAX_CONCEALED_TILES))
    + (
        ActionMeaning(ACTION_CHI_LEFT, "chi_left"),
        ActionMeaning(ACTION_CHI_CENTER, "chi_center"),
        ActionMeaning(ACTION_CHI_RIGHT, "chi_right"),
        ActionMeaning(ACTION_PENG, "peng"),
        ActionMeaning(ACTION_GAME, "game"),
        ActionMeaning(ACTION_PASS, "pass"),
    )
)
