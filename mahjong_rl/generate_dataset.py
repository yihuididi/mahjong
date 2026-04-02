from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .constants import SEAT_WINDS, TILE_TYPE_COUNT, TILE_NAMES, WIND_KINDS
from .hand_labels import (
    label_concealed_hand,
    sample_concealed_hand_kinds,
    tile_names_from_kinds,
)
from .rules import evaluate_hand, tile_name


MELD_CANDIDATES = (
    tuple((kind, kind, kind) for kind in range(TILE_TYPE_COUNT))
    + tuple((base + offset, base + offset + 1, base + offset + 2) for base in (0, 9) for offset in range(7))
)
PAIR_CANDIDATES = tuple((kind, kind) for kind in range(TILE_TYPE_COUNT))
DEFAULT_INCLUDE_EMPTY_RANDOM_RATIO = 0.35


def generate_dataset(
    train_size: int,
    test_size: int,
    seed: int,
    include_empty: bool = False,
    legal_only: bool = False,
    include_empty_random_ratio: float = DEFAULT_INCLUDE_EMPTY_RANDOM_RATIO,
) -> tuple[list[dict], list[dict], dict[str, dict[str, int]]]:
    if not 0.0 <= include_empty_random_ratio <= 1.0:
        raise ValueError("include_empty_random_ratio must be between 0.0 and 1.0.")

    rng = np.random.default_rng(seed)
    train_rows = _generate_unique_rows(
        train_size,
        rng,
        include_empty=include_empty,
        legal_only=legal_only,
        include_empty_random_ratio=include_empty_random_ratio,
    )
    test_rows = _generate_unique_rows(test_size, rng, include_empty=include_empty, used_signatures={
        _row_signature(row) for row in train_rows
    }, legal_only=legal_only, include_empty_random_ratio=include_empty_random_ratio)
    summary = {
        "train": _summarize_rows(train_rows),
        "test": _summarize_rows(test_rows),
    }
    return train_rows, test_rows, summary


def _generate_unique_rows(
    size: int,
    rng: np.random.Generator,
    include_empty: bool,
    legal_only: bool,
    include_empty_random_ratio: float,
    used_signatures: set[tuple] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    seen = set() if used_signatures is None else set(used_signatures)
    random_row_target = 0
    structured_row_target = size
    random_row_count = 0
    structured_row_count = 0

    if include_empty:
        random_row_target = int(round(size * include_empty_random_ratio))
        structured_row_target = size - random_row_target

    attempts = 0
    max_attempts = max(size * 50, 1000)
    while len(rows) < size:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not generate {size} unique rows after {max_attempts} attempts. "
                "Try a smaller dataset size or allow empty waits."
            )

        if include_empty:
            should_sample_random = _should_sample_random_row(
                rng,
                random_row_count=random_row_count,
                structured_row_count=structured_row_count,
                random_row_target=random_row_target,
                structured_row_target=structured_row_target,
                include_empty_random_ratio=include_empty_random_ratio,
            )
            row = (
                _sample_random_row(rng)
                if should_sample_random
                else _sample_backward_row(rng, legal_only=legal_only)
            )
        else:
            should_sample_random = False
            row = _sample_backward_row(rng, legal_only=legal_only)
        signature = _row_signature(row)
        if signature in seen:
            continue
        seen.add(signature)
        rows.append(row)
        if should_sample_random:
            random_row_count += 1
        else:
            structured_row_count += 1

    return rows


def _should_sample_random_row(
    rng: np.random.Generator,
    random_row_count: int,
    structured_row_count: int,
    random_row_target: int,
    structured_row_target: int,
    include_empty_random_ratio: float,
) -> bool:
    if random_row_count >= random_row_target:
        return False
    if structured_row_count >= structured_row_target:
        return True
    return bool(rng.random() < include_empty_random_ratio)


def _row_signature(row: dict) -> tuple:
    return (
        tuple(row["hand_tile_kinds"]),
        row["seat_wind_kind"],
        row["game_wind_kind"],
    )


def _summarize_rows(rows: list[dict]) -> dict[str, int]:
    winning_shape_hist = Counter(len(row["winning_shape_tile_kinds"]) for row in rows)
    legal_game_hist = Counter(len(row["legal_game_tile_kinds"]) for row in rows)
    improving_hist = Counter(len(row["improving_tile_kinds"]) for row in rows)
    return {
        "rows": len(rows),
        "non_empty_winning_shape_rows": sum(1 for row in rows if row["winning_shape_tile_kinds"]),
        "non_empty_legal_game_rows": sum(1 for row in rows if row["legal_game_tile_kinds"]),
        "non_empty_improving_rows": sum(1 for row in rows if row["improving_tile_kinds"]),
        "max_winning_shape_tiles": max((len(row["winning_shape_tile_kinds"]) for row in rows), default=0),
        "max_legal_game_tiles": max((len(row["legal_game_tile_kinds"]) for row in rows), default=0),
        "max_improving_tiles": max((len(row["improving_tile_kinds"]) for row in rows), default=0),
        "winning_shape_histogram": dict(sorted(winning_shape_hist.items())),
        "legal_game_histogram": dict(sorted(legal_game_hist.items())),
        "improving_histogram": dict(sorted(improving_hist.items())),
    }


def _sample_random_row(rng: np.random.Generator) -> dict:
    hand_kinds = sample_concealed_hand_kinds(rng)
    seat_wind = int(rng.choice(SEAT_WINDS))
    game_wind = int(rng.choice(WIND_KINDS))
    return _row_from_hand(hand_kinds, seat_wind, game_wind)


def _sample_backward_row(rng: np.random.Generator, legal_only: bool) -> dict:
    while True:
        seat_wind = int(rng.choice(SEAT_WINDS))
        game_wind = int(rng.choice(WIND_KINDS))
        full_hand_kinds = _sample_complete_hand_kinds(rng, seat_wind, game_wind, legal_only=legal_only)
        removed_index = int(rng.integers(0, len(full_hand_kinds)))
        hand_kinds = full_hand_kinds.copy()
        hand_kinds.pop(removed_index)
        row = _row_from_hand(sorted(hand_kinds), seat_wind, game_wind)
        if legal_only and not row["legal_game_tile_kinds"]:
            continue
        if row["winning_shape_tile_kinds"]:
            return row


def _sample_complete_hand_kinds(
    rng: np.random.Generator,
    seat_wind: int,
    game_wind: int,
    legal_only: bool,
) -> list[int]:
    while True:
        counts = [0] * TILE_TYPE_COUNT
        tiles: list[int] = []

        for _ in range(4):
            options = [meld for meld in MELD_CANDIDATES if _can_add_pattern(counts, meld)]
            if not options:
                break
            meld = options[int(rng.integers(0, len(options)))]
            for tile_kind in meld:
                counts[tile_kind] += 1
                tiles.append(tile_kind)
        else:
            pair_options = [pair for pair in PAIR_CANDIDATES if _can_add_pattern(counts, pair)]
            if not pair_options:
                continue
            pair = pair_options[int(rng.integers(0, len(pair_options)))]
            for tile_kind in pair:
                counts[tile_kind] += 1
                tiles.append(tile_kind)

            evaluation = evaluate_hand(counts, (), seat_wind, game_wind)
            if not evaluation.is_complete:
                continue
            if legal_only and not evaluation.can_win:
                continue
            return sorted(tiles)


def _can_add_pattern(counts: list[int], pattern: tuple[int, ...]) -> bool:
    next_counts = counts.copy()
    for tile_kind in pattern:
        next_counts[tile_kind] += 1
        if next_counts[tile_kind] > 4:
            return False
    return True


def _row_from_hand(hand_kinds: list[int], seat_wind: int, game_wind: int) -> dict:
    labels = label_concealed_hand(hand_kinds, seat_wind, game_wind)
    return {
        "hand_tile_kinds": hand_kinds,
        "hand_tiles": tile_names_from_kinds(hand_kinds),
        "seat_wind_kind": seat_wind,
        "seat_wind": tile_name(seat_wind),
        "game_wind_kind": game_wind,
        "game_wind": tile_name(game_wind),
        "winning_shape_tile_kinds": list(labels.winning_shape_tile_kinds),
        "winning_shape_tiles": tile_names_from_kinds(labels.winning_shape_tile_kinds),
        "legal_game_tile_kinds": list(labels.legal_game_tile_kinds),
        "legal_game_tiles": tile_names_from_kinds(labels.legal_game_tile_kinds),
        "improving_tile_kinds": list(labels.improving_tile_kinds),
        "improving_tiles": tile_names_from_kinds(labels.improving_tile_kinds),
        "completion_score": labels.completion_score,
        "best_next_completion_score": labels.best_next_completion_score,
    }


def write_dataset(
    output_dir: str | Path,
    train_rows: list[dict],
    test_rows: list[dict],
    summary: dict[str, dict[str, int]],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _write_jsonl(output_path / "train.jsonl", train_rows)
    _write_jsonl(output_path / "test.jsonl", test_rows)

    metadata = {
        "format_version": 1,
        "tile_names": list(TILE_NAMES),
        "seat_winds": [tile_name(kind) for kind in SEAT_WINDS],
        "game_winds": [tile_name(kind) for kind in WIND_KINDS],
        "description": (
            "Exact-rule concealed 13-tile dataset for the custom 3-player Mahjong variant in this repo. "
            "winning_shape tiles ignore the tai requirement; legal_game tiles require tai >= 1."
        ),
        "summary": summary,
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an exact-rule concealed-hand dataset for custom 3P Mahjong.")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--include-empty", action="store_true", help="Keep hands with no winning tiles in the dataset.")
    parser.add_argument(
        "--include-empty-random-ratio",
        type=float,
        default=DEFAULT_INCLUDE_EMPTY_RANDOM_RATIO,
        help=(
            "When --include-empty is enabled, fraction of rows sampled from fully random concealed hands. "
            "The remaining rows use backward sampling so the broad curriculum still contains meaningful positives."
        ),
    )
    parser.add_argument("--legal-only", action="store_true", help="Only generate non-empty rows whose wait set includes a legal tai-valid game tile.")
    parser.add_argument("--output-dir", type=str, default="artifacts/exact_rule_dataset")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    train_rows, test_rows, summary = generate_dataset(
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        include_empty=args.include_empty,
        legal_only=args.legal_only,
        include_empty_random_ratio=args.include_empty_random_ratio,
    )
    write_dataset(args.output_dir, train_rows, test_rows, summary)

    print(
        f"generated train={len(train_rows)} test={len(test_rows)} "
        f"output_dir={args.output_dir} legal_nonempty_train={summary['train']['non_empty_legal_game_rows']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
