from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .constants import SEAT_WINDS, TILE_NAMES, WIND_KINDS
from .hand_labels import (
    label_concealed_discard_hand,
    sample_concealed_hand_kinds,
    tile_names_from_kinds,
)
from .rules import tile_name


def generate_discard_dataset(
    train_size: int,
    test_size: int,
    seed: int,
    informative_only: bool = True,
) -> tuple[list[dict], list[dict], dict[str, dict[str, int]]]:
    rng = np.random.default_rng(seed)
    train_rows = _generate_unique_rows(train_size, rng, informative_only=informative_only)
    test_rows = _generate_unique_rows(
        test_size,
        rng,
        informative_only=informative_only,
        used_signatures={_row_signature(row) for row in train_rows},
    )
    summary = {
        "train": _summarize_rows(train_rows),
        "test": _summarize_rows(test_rows),
    }
    return train_rows, test_rows, summary


def _generate_unique_rows(
    size: int,
    rng: np.random.Generator,
    informative_only: bool,
    used_signatures: set[tuple] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    seen = set() if used_signatures is None else set(used_signatures)

    attempts = 0
    max_attempts = max(size * 100, 2000)
    while len(rows) < size:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not generate {size} unique discard rows after {max_attempts} attempts."
            )

        row = _sample_random_row(rng)
        if informative_only and len(row["best_discard_tile_kinds"]) == len(set(row["hand_tile_kinds"])):
            continue
        signature = _row_signature(row)
        if signature in seen:
            continue
        seen.add(signature)
        rows.append(row)

    return rows


def _sample_random_row(rng: np.random.Generator) -> dict:
    hand_kinds = sample_concealed_hand_kinds(rng, hand_size=14)
    seat_wind = int(rng.choice(SEAT_WINDS))
    game_wind = int(rng.choice(WIND_KINDS))
    return _row_from_hand(hand_kinds, seat_wind, game_wind)


def _row_signature(row: dict) -> tuple:
    return (
        tuple(row["hand_tile_kinds"]),
        row["seat_wind_kind"],
        row["game_wind_kind"],
    )


def _row_from_hand(hand_kinds: list[int], seat_wind: int, game_wind: int) -> dict:
    labels = label_concealed_discard_hand(hand_kinds, seat_wind, game_wind)
    return {
        "hand_tile_kinds": hand_kinds,
        "hand_tiles": tile_names_from_kinds(hand_kinds),
        "seat_wind_kind": seat_wind,
        "seat_wind": tile_name(seat_wind),
        "game_wind_kind": game_wind,
        "game_wind": tile_name(game_wind),
        "best_discard_tile_kinds": list(labels.best_discard_tile_kinds),
        "best_discard_tiles": tile_names_from_kinds(labels.best_discard_tile_kinds),
        "best_discard_score": labels.best_discard_score,
        "discard_scores": [
            {"tile_kind": tile_kind, "tile": TILE_NAMES[tile_kind], "score": score}
            for tile_kind, score in labels.discard_scores_by_kind
        ],
    }


def _summarize_rows(rows: list[dict]) -> dict[str, int]:
    best_count_hist = Counter(len(row["best_discard_tile_kinds"]) for row in rows)
    return {
        "rows": len(rows),
        "max_best_discards": max((len(row["best_discard_tile_kinds"]) for row in rows), default=0),
        "best_discard_histogram": dict(sorted(best_count_hist.items())),
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
            "Exact-rule concealed 14-tile discard dataset. "
            "Each row labels the discard tile kinds that preserve the most future winning potential."
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
    parser = argparse.ArgumentParser(
        description="Generate an exact-rule concealed 14-tile discard dataset for custom 3P Mahjong."
    )
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--allow-trivial",
        action="store_true",
        help="Keep hands where every discard ties for best score.",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/exact_rule_discard_dataset")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train_rows, test_rows, summary = generate_discard_dataset(
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        informative_only=not args.allow_trivial,
    )
    write_dataset(args.output_dir, train_rows, test_rows, summary)
    print(
        f"generated train={len(train_rows)} test={len(test_rows)} "
        f"output_dir={args.output_dir} max_best_discards={summary['train']['max_best_discards']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
