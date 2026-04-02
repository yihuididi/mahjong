from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .constants import SEAT_WINDS, TILE_NAMES, WIND_KINDS
from .generate_dataset import _sample_complete_hand_kinds
from .hand_labels import completion_score_for_counts, tile_names_from_kinds
from .rules import counts_from_kinds, evaluate_hand, tile_name


def generate_backward_curriculum_dataset(
    train_size: int,
    test_size: int,
    seed: int,
    max_steps_from_terminal: int = 4,
    min_hand_size: int = 10,
    branching_factor: int = 2,
) -> tuple[list[dict], list[dict], dict[str, dict[str, int]]]:
    if max_steps_from_terminal < 1:
        raise ValueError("max_steps_from_terminal must be at least 1.")
    if min_hand_size < 1 or min_hand_size > 13:
        raise ValueError("min_hand_size must be between 1 and 13.")
    if branching_factor < 1:
        raise ValueError("branching_factor must be at least 1.")

    rng = np.random.default_rng(seed)
    train_rows = _generate_unique_rows(
        train_size,
        rng,
        max_steps_from_terminal=max_steps_from_terminal,
        min_hand_size=min_hand_size,
        branching_factor=branching_factor,
    )
    test_rows = _generate_unique_rows(
        test_size,
        rng,
        max_steps_from_terminal=max_steps_from_terminal,
        min_hand_size=min_hand_size,
        branching_factor=branching_factor,
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
    max_steps_from_terminal: int,
    min_hand_size: int,
    branching_factor: int,
    used_signatures: set[tuple] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    seen = set() if used_signatures is None else set(used_signatures)

    attempts = 0
    max_attempts = max(size * 20, 1000)
    while len(rows) < size:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not generate {size} unique backward-curriculum rows after {max_attempts} attempts."
            )

        for row in _sample_rows_from_terminal(
            rng,
            max_steps_from_terminal=max_steps_from_terminal,
            min_hand_size=min_hand_size,
            branching_factor=branching_factor,
        ):
            signature = _row_signature(row)
            if signature in seen:
                continue
            seen.add(signature)
            rows.append(row)
            if len(rows) >= size:
                break

    return rows


def _sample_rows_from_terminal(
    rng: np.random.Generator,
    max_steps_from_terminal: int,
    min_hand_size: int,
    branching_factor: int,
) -> list[dict]:
    seat_wind = int(rng.choice(SEAT_WINDS))
    game_wind = int(rng.choice(WIND_KINDS))
    terminal_hand_kinds = _sample_complete_hand_kinds(rng, seat_wind, game_wind, legal_only=True)

    rows: list[dict] = []
    frontier = [terminal_hand_kinds]
    terminal_size = len(terminal_hand_kinds)
    max_depth = min(max_steps_from_terminal, terminal_size - min_hand_size)

    for _ in range(max_depth):
        next_frontier: list[list[int]] = []
        for hand_kinds in frontier:
            if len(hand_kinds) <= min_hand_size:
                continue
            for child_kinds in _sample_unique_predecessors(hand_kinds, rng, branching_factor):
                rows.append(
                    _row_from_partial_hand(
                        hand_kinds=child_kinds,
                        seat_wind=seat_wind,
                        game_wind=game_wind,
                        terminal_hand_kinds=terminal_hand_kinds,
                    )
                )
                next_frontier.append(child_kinds)
        frontier = next_frontier
        if not frontier:
            break

    return rows


def _sample_unique_predecessors(
    hand_kinds: list[int],
    rng: np.random.Generator,
    branching_factor: int,
) -> list[list[int]]:
    unique_children: dict[tuple[int, ...], list[int]] = {}
    for tile_kind in sorted(set(hand_kinds)):
        child = hand_kinds.copy()
        child.remove(tile_kind)
        unique_children[tuple(child)] = child

    children = list(unique_children.values())
    if len(children) <= branching_factor:
        return children

    choice_indices = rng.choice(len(children), size=branching_factor, replace=False)
    return [children[int(index)] for index in sorted(choice_indices.tolist())]


def _row_from_partial_hand(
    hand_kinds: list[int],
    seat_wind: int,
    game_wind: int,
    terminal_hand_kinds: list[int],
) -> dict:
    hand_list = sorted(int(kind) for kind in hand_kinds)
    hand_counts = counts_from_kinds(hand_list)
    terminal_counts = counts_from_kinds(terminal_hand_kinds)
    path_progress_tile_kinds = [
        tile_kind
        for tile_kind, (current_count, terminal_count) in enumerate(zip(hand_counts, terminal_counts))
        if terminal_count > current_count
    ]

    completion_score = completion_score_for_counts(hand_counts, seat_wind, game_wind)
    best_next_completion_score = completion_score
    winning_shape_tile_kinds: list[int] = []
    legal_game_tile_kinds: list[int] = []
    improving_tile_kinds: list[int] = []

    for tile_kind in range(len(hand_counts)):
        if hand_counts[tile_kind] >= 4:
            continue

        next_counts = hand_counts.copy()
        next_counts[tile_kind] += 1
        next_score = completion_score_for_counts(next_counts, seat_wind, game_wind)
        if len(hand_list) == 13:
            evaluation = evaluate_hand(next_counts, (), seat_wind, game_wind)
            if evaluation.is_complete:
                winning_shape_tile_kinds.append(tile_kind)
            if evaluation.can_win:
                legal_game_tile_kinds.append(tile_kind)

        if next_score > completion_score + 1e-6 or tile_kind in path_progress_tile_kinds:
            improving_tile_kinds.append(tile_kind)
        if next_score > best_next_completion_score:
            best_next_completion_score = next_score

    steps_from_terminal = len(terminal_hand_kinds) - len(hand_list)
    return {
        "hand_tile_kinds": hand_list,
        "hand_tiles": tile_names_from_kinds(hand_list),
        "seat_wind_kind": seat_wind,
        "seat_wind": tile_name(seat_wind),
        "game_wind_kind": game_wind,
        "game_wind": tile_name(game_wind),
        "terminal_hand_tile_kinds": list(terminal_hand_kinds),
        "terminal_hand_tiles": tile_names_from_kinds(terminal_hand_kinds),
        "steps_from_terminal": steps_from_terminal,
        "path_progress_tile_kinds": path_progress_tile_kinds,
        "path_progress_tiles": tile_names_from_kinds(path_progress_tile_kinds),
        "winning_shape_tile_kinds": winning_shape_tile_kinds,
        "winning_shape_tiles": tile_names_from_kinds(winning_shape_tile_kinds),
        "legal_game_tile_kinds": legal_game_tile_kinds,
        "legal_game_tiles": tile_names_from_kinds(legal_game_tile_kinds),
        "improving_tile_kinds": improving_tile_kinds,
        "improving_tiles": tile_names_from_kinds(improving_tile_kinds),
        "completion_score": float(completion_score),
        "best_next_completion_score": float(best_next_completion_score),
    }


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
    step_hist = Counter(int(row["steps_from_terminal"]) for row in rows)
    hand_size_hist = Counter(len(row["hand_tile_kinds"]) for row in rows)
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
        "steps_from_terminal_histogram": dict(sorted(step_hist.items())),
        "hand_size_histogram": dict(sorted(hand_size_hist.items())),
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
            "Backward recursive curriculum dataset derived from legal terminal winning hands. "
            "Each row is a partial concealed hand obtained by recursively removing tiles from a legal terminal hand. "
            "path_progress tiles are tile kinds that move the hand one sampled step closer to its terminal state."
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
        description="Generate a recursive backward curriculum dataset from legal terminal Mahjong hands."
    )
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--max-steps-from-terminal", type=int, default=4)
    parser.add_argument("--min-hand-size", type=int, default=10)
    parser.add_argument("--branching-factor", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="artifacts/backward_curriculum_dataset")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train_rows, test_rows, summary = generate_backward_curriculum_dataset(
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        max_steps_from_terminal=args.max_steps_from_terminal,
        min_hand_size=args.min_hand_size,
        branching_factor=args.branching_factor,
    )
    write_dataset(args.output_dir, train_rows, test_rows, summary)
    print(
        f"generated train={len(train_rows)} test={len(test_rows)} "
        f"output_dir={args.output_dir} step_hist={summary['train']['steps_from_terminal_histogram']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
