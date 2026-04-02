from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mini_training_ppo.model import MLP

from .compact_observation import compact_observation_from_concealed_hand, compact_observation_size
from .constants import SEAT_WINDS, TILE_TYPE_COUNT


WINNING_SHAPE_SLICE = slice(0, TILE_TYPE_COUNT)
LEGAL_GAME_SLICE = slice(TILE_TYPE_COUNT, TILE_TYPE_COUNT * 2)
IMPROVING_SLICE = slice(TILE_TYPE_COUNT * 2, TILE_TYPE_COUNT * 3)
COMPLETION_INDEX = TILE_TYPE_COUNT * 3
HAS_SHAPE_INDEX = COMPLETION_INDEX + 1
HAS_LEGAL_INDEX = COMPLETION_INDEX + 2
LEGAL_WAIT_COUNT_INDEX = COMPLETION_INDEX + 3
PRETRAIN_OUTPUT_DIM = (TILE_TYPE_COUNT * 3) + 4
COMPLETION_SCORE_SCALE = 24.0
LEGAL_WAIT_COUNT_SCALE = 8.0


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    train_path: str
    test_path: str
    epochs: int


@dataclass(frozen=True)
class PretrainConfig:
    train_path: str = "artifacts/exact_rule_dataset_legal_only/train.jsonl"
    test_path: str = "artifacts/exact_rule_dataset_legal_only/test.jsonl"
    target_key: str = "legal_game_tile_kinds"
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_sizes: tuple[int, ...] = (128, 64)
    legal_positive_weight: float = 12.0
    shape_positive_weight: float = 4.0
    improving_positive_weight: float = 2.0
    legal_task_weight: float = 4.0
    shape_task_weight: float = 1.0
    improving_task_weight: float = 0.5
    completion_task_weight: float = 0.25
    has_shape_task_weight: float = 0.25
    has_legal_task_weight: float = 0.5
    legal_wait_count_task_weight: float = 0.25
    stage1_train_path: str | None = None
    stage1_test_path: str | None = None
    stage1_epochs: int = 0
    stage2_train_path: str | None = None
    stage2_test_path: str | None = None
    stage2_epochs: int = 0
    stage3_train_path: str | None = None
    stage3_test_path: str | None = None
    stage3_epochs: int = 0
    seed: int = 23
    log_interval: int = 1
    save_best_by: str = "top1_hit"


@dataclass
class LoadedSupervisedData:
    observations: np.ndarray
    winning_shape_targets: np.ndarray
    legal_game_targets: np.ndarray
    improving_targets: np.ndarray
    completion_targets: np.ndarray
    has_winning_shape_targets: np.ndarray
    has_legal_game_targets: np.ndarray
    legal_wait_count_targets: np.ndarray


@dataclass
class PretrainSummary:
    stage_name: str
    epoch: int
    global_epoch: int
    train_loss: float
    test_loss: float
    exact_match: float
    precision: float
    recall: float
    top1_hit: float
    any_legal_accuracy: float
    elapsed_seconds: float


def load_supervised_rows(path: str | Path) -> LoadedSupervisedData:
    observations: list[np.ndarray] = []
    winning_shape_targets: list[np.ndarray] = []
    legal_game_targets: list[np.ndarray] = []
    improving_targets: list[np.ndarray] = []
    completion_targets: list[float] = []
    has_winning_shape_targets: list[float] = []
    has_legal_game_targets: list[float] = []
    legal_wait_count_targets: list[float] = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            player_index = SEAT_WINDS.index(int(row["seat_wind_kind"]))
            observations.append(
                compact_observation_from_concealed_hand(
                    hand_kinds=list(row["hand_tile_kinds"]),
                    player_index=player_index,
                    game_wind=int(row["game_wind_kind"]),
                )
            )

            winning_shape_targets.append(_multihot(row["winning_shape_tile_kinds"]))
            legal_game_targets.append(_multihot(row["legal_game_tile_kinds"]))
            improving_targets.append(_multihot(row["improving_tile_kinds"]))
            completion_targets.append(float(row["completion_score"]) / COMPLETION_SCORE_SCALE)
            has_winning_shape_targets.append(1.0 if row["winning_shape_tile_kinds"] else 0.0)
            has_legal_game_targets.append(1.0 if row["legal_game_tile_kinds"] else 0.0)
            legal_wait_count_targets.append(
                min(float(len(row["legal_game_tile_kinds"])) / LEGAL_WAIT_COUNT_SCALE, 1.0)
            )

    if not observations:
        raise ValueError(f"No rows found in dataset file: {path}")

    return LoadedSupervisedData(
        observations=np.stack(observations, axis=0),
        winning_shape_targets=np.stack(winning_shape_targets, axis=0),
        legal_game_targets=np.stack(legal_game_targets, axis=0),
        improving_targets=np.stack(improving_targets, axis=0),
        completion_targets=np.asarray(completion_targets, dtype=np.float32)[:, None],
        has_winning_shape_targets=np.asarray(has_winning_shape_targets, dtype=np.float32)[:, None],
        has_legal_game_targets=np.asarray(has_legal_game_targets, dtype=np.float32)[:, None],
        legal_wait_count_targets=np.asarray(legal_wait_count_targets, dtype=np.float32)[:, None],
    )


def train_pretrained_wait_model(
    config: PretrainConfig,
    save_path: str | None = None,
) -> tuple[MLP, list[PretrainSummary]]:
    stages = _build_stages(config)
    data_cache: dict[tuple[str, str], tuple[LoadedSupervisedData, LoadedSupervisedData]] = {}

    first_train, _ = _load_stage_data(stages[0], data_cache)
    if first_train.observations.shape[1] != compact_observation_size():
        raise ValueError("Unexpected compact observation size while loading supervised dataset.")

    model = MLP(first_train.observations.shape[1], config.hidden_sizes, PRETRAIN_OUTPUT_DIM, seed=config.seed)
    rng = np.random.default_rng(config.seed)
    summaries: list[PretrainSummary] = []
    start_time = time.time()
    global_epoch = 0
    best_metric = float("-inf")
    best_payload: tuple[list[np.ndarray], list[np.ndarray]] | None = None

    for stage in stages:
        train_data, test_data = _load_stage_data(stage, data_cache)
        for stage_epoch in range(1, stage.epochs + 1):
            global_epoch += 1
            permutation = rng.permutation(train_data.observations.shape[0])
            train_losses: list[float] = []

            for start in range(0, train_data.observations.shape[0], config.batch_size):
                batch_idx = permutation[start : start + config.batch_size]
                outputs = model.predict(train_data.observations[batch_idx])
                loss, grad_output = _multitask_loss_and_grad(
                    outputs=outputs,
                    data=_slice_loaded_data(train_data, batch_idx),
                    config=config,
                )
                model.train_with_output_gradient(
                    train_data.observations[batch_idx],
                    grad_output,
                    config.learning_rate,
                )
                train_losses.append(loss)

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            test_loss, exact_match, precision, recall, top1_hit, any_legal_accuracy = evaluate_multitask_model(
                model,
                test_data,
                config,
            )
            summary = PretrainSummary(
                stage_name=stage.name,
                epoch=stage_epoch,
                global_epoch=global_epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                exact_match=exact_match,
                precision=precision,
                recall=recall,
                top1_hit=top1_hit,
                any_legal_accuracy=any_legal_accuracy,
                elapsed_seconds=time.time() - start_time,
            )
            summaries.append(summary)

            metric_value = top1_hit if config.save_best_by == "top1_hit" else (0.7 * top1_hit + 0.3 * recall)
            if metric_value > best_metric:
                best_metric = metric_value
                best_payload = ([weight.copy() for weight in model.weights], [bias.copy() for bias in model.biases])

            if stage_epoch % config.log_interval == 0 or stage_epoch == 1 or stage_epoch == stage.epochs:
                print(
                    f"stage={stage.name} epoch={stage_epoch}/{stage.epochs} global_epoch={global_epoch} "
                    f"train_loss={train_loss:.5f} test_loss={test_loss:.5f} "
                    f"exact_match={exact_match:.3f} precision={precision:.3f} recall={recall:.3f} "
                    f"top1_hit={top1_hit:.3f} any_legal_acc={any_legal_accuracy:.3f} "
                    f"elapsed={summary.elapsed_seconds:.1f}s",
                    flush=True,
                )

    if best_payload is not None:
        model.weights = [weight.copy() for weight in best_payload[0]]
        model.biases = [bias.copy() for bias in best_payload[1]]

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)

    return model, summaries


def evaluate_multitask_model(
    model: MLP,
    data: LoadedSupervisedData,
    config: PretrainConfig,
) -> tuple[float, float, float, float, float, float]:
    outputs = model.predict(data.observations)
    loss, _ = _multitask_loss_and_grad(outputs=outputs, data=data, config=config)

    legal_probs = _sigmoid(outputs[:, LEGAL_GAME_SLICE])
    legal_predictions = legal_probs >= 0.5
    legal_truth = data.legal_game_targets >= 0.5
    exact_match = float(np.mean(np.all(legal_predictions == legal_truth, axis=1)))

    true_positive = float(np.logical_and(legal_predictions, legal_truth).sum())
    predicted_positive = float(legal_predictions.sum())
    actual_positive = float(legal_truth.sum())
    precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0
    recall = true_positive / actual_positive if actual_positive > 0 else 0.0

    top_indices = np.argmax(legal_probs, axis=1)
    top1_hits = [bool(legal_truth[row_index, top_index]) for row_index, top_index in enumerate(top_indices)]
    top1_hit = float(np.mean(top1_hits))

    has_legal_probs = _sigmoid(outputs[:, HAS_LEGAL_INDEX : HAS_LEGAL_INDEX + 1])
    any_legal_accuracy = float(
        np.mean((has_legal_probs >= 0.5) == (data.has_legal_game_targets >= 0.5))
    )

    return (
        float(loss),
        _finite(exact_match),
        _finite(precision),
        _finite(recall),
        _finite(top1_hit),
        _finite(any_legal_accuracy),
    )


def _multitask_loss_and_grad(
    outputs: np.ndarray,
    data: LoadedSupervisedData,
    config: PretrainConfig,
) -> tuple[float, np.ndarray]:
    shape_logits = outputs[:, WINNING_SHAPE_SLICE]
    legal_logits = outputs[:, LEGAL_GAME_SLICE]
    improving_logits = outputs[:, IMPROVING_SLICE]
    completion_pred = outputs[:, COMPLETION_INDEX : COMPLETION_INDEX + 1]
    has_shape_logits = outputs[:, HAS_SHAPE_INDEX : HAS_SHAPE_INDEX + 1]
    has_legal_logits = outputs[:, HAS_LEGAL_INDEX : HAS_LEGAL_INDEX + 1]
    legal_wait_count_pred = outputs[:, LEGAL_WAIT_COUNT_INDEX : LEGAL_WAIT_COUNT_INDEX + 1]

    shape_probs = _sigmoid(shape_logits)
    legal_probs = _sigmoid(legal_logits)
    improving_probs = _sigmoid(improving_logits)
    has_shape_probs = _sigmoid(has_shape_logits)
    has_legal_probs = _sigmoid(has_legal_logits)

    shape_loss, shape_grad = _weighted_bce_and_grad(
        shape_probs,
        data.winning_shape_targets,
        positive_weight=config.shape_positive_weight,
        task_weight=config.shape_task_weight,
    )
    legal_loss, legal_grad = _weighted_bce_and_grad(
        legal_probs,
        data.legal_game_targets,
        positive_weight=config.legal_positive_weight,
        task_weight=config.legal_task_weight,
    )
    improving_loss, improving_grad = _weighted_bce_and_grad(
        improving_probs,
        data.improving_targets,
        positive_weight=config.improving_positive_weight,
        task_weight=config.improving_task_weight,
    )
    has_shape_loss, has_shape_grad = _weighted_bce_and_grad(
        has_shape_probs,
        data.has_winning_shape_targets,
        positive_weight=1.5,
        task_weight=config.has_shape_task_weight,
    )
    has_legal_loss, has_legal_grad = _weighted_bce_and_grad(
        has_legal_probs,
        data.has_legal_game_targets,
        positive_weight=2.0,
        task_weight=config.has_legal_task_weight,
    )

    completion_error = completion_pred - data.completion_targets
    completion_loss = config.completion_task_weight * float(np.mean(completion_error ** 2))
    completion_grad = (2.0 * config.completion_task_weight * completion_error).astype(np.float32)

    wait_count_error = legal_wait_count_pred - data.legal_wait_count_targets
    wait_count_loss = config.legal_wait_count_task_weight * float(np.mean(wait_count_error ** 2))
    wait_count_grad = (2.0 * config.legal_wait_count_task_weight * wait_count_error).astype(np.float32)

    grad_output = np.zeros_like(outputs, dtype=np.float32)
    grad_output[:, WINNING_SHAPE_SLICE] = shape_grad
    grad_output[:, LEGAL_GAME_SLICE] = legal_grad
    grad_output[:, IMPROVING_SLICE] = improving_grad
    grad_output[:, COMPLETION_INDEX : COMPLETION_INDEX + 1] = completion_grad
    grad_output[:, HAS_SHAPE_INDEX : HAS_SHAPE_INDEX + 1] = has_shape_grad
    grad_output[:, HAS_LEGAL_INDEX : HAS_LEGAL_INDEX + 1] = has_legal_grad
    grad_output[:, LEGAL_WAIT_COUNT_INDEX : LEGAL_WAIT_COUNT_INDEX + 1] = wait_count_grad

    total_loss = (
        shape_loss
        + legal_loss
        + improving_loss
        + completion_loss
        + has_shape_loss
        + has_legal_loss
        + wait_count_loss
    )
    return float(total_loss), grad_output


def _weighted_bce_and_grad(
    probs: np.ndarray,
    targets: np.ndarray,
    positive_weight: float,
    task_weight: float,
) -> tuple[float, np.ndarray]:
    clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
    weights = np.where(targets > 0.5, positive_weight, 1.0).astype(np.float32)
    loss = -(positive_weight * targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
    loss_value = task_weight * float(np.mean(loss))
    grad = task_weight * (probs - targets) * weights / float(max(targets.shape[1], 1))
    return loss_value, grad.astype(np.float32)


def _multihot(tile_kinds: list[int]) -> np.ndarray:
    target = np.zeros(TILE_TYPE_COUNT, dtype=np.float32)
    for tile_kind in tile_kinds:
        target[int(tile_kind)] = 1.0
    return target


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _finite(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def _slice_loaded_data(data: LoadedSupervisedData, batch_idx: np.ndarray) -> LoadedSupervisedData:
    return LoadedSupervisedData(
        observations=data.observations[batch_idx],
        winning_shape_targets=data.winning_shape_targets[batch_idx],
        legal_game_targets=data.legal_game_targets[batch_idx],
        improving_targets=data.improving_targets[batch_idx],
        completion_targets=data.completion_targets[batch_idx],
        has_winning_shape_targets=data.has_winning_shape_targets[batch_idx],
        has_legal_game_targets=data.has_legal_game_targets[batch_idx],
        legal_wait_count_targets=data.legal_wait_count_targets[batch_idx],
    )


def _load_stage_data(
    stage: CurriculumStage,
    cache: dict[tuple[str, str], tuple[LoadedSupervisedData, LoadedSupervisedData]],
) -> tuple[LoadedSupervisedData, LoadedSupervisedData]:
    key = (stage.train_path, stage.test_path)
    if key not in cache:
        cache[key] = (load_supervised_rows(stage.train_path), load_supervised_rows(stage.test_path))
    return cache[key]


def _build_stages(config: PretrainConfig) -> list[CurriculumStage]:
    stages: list[CurriculumStage] = []
    stage_specs = (
        ("broad", config.stage1_train_path, config.stage1_test_path, config.stage1_epochs),
        ("nonempty", config.stage2_train_path, config.stage2_test_path, config.stage2_epochs),
        ("legal", config.stage3_train_path, config.stage3_test_path, config.stage3_epochs),
    )

    for name, train_path, test_path, epochs in stage_specs:
        if epochs > 0 and train_path and test_path:
            stages.append(CurriculumStage(name=name, train_path=train_path, test_path=test_path, epochs=epochs))

    if stages:
        return stages

    return [
        CurriculumStage(
            name="single",
            train_path=config.train_path,
            test_path=config.test_path,
            epochs=config.epochs,
        )
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curriculum multi-task pretraining on exact-rule Mahjong hand labels.")
    parser.add_argument("--train-path", type=str, default=PretrainConfig.train_path)
    parser.add_argument("--test-path", type=str, default=PretrainConfig.test_path)
    parser.add_argument("--target-key", type=str, default=PretrainConfig.target_key)
    parser.add_argument("--epochs", type=int, default=PretrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=PretrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=PretrainConfig.learning_rate)
    parser.add_argument("--log-interval", type=int, default=PretrainConfig.log_interval)
    parser.add_argument("--seed", type=int, default=PretrainConfig.seed)
    parser.add_argument("--save-best-by", type=str, choices=("top1_hit", "mixed"), default=PretrainConfig.save_best_by)
    parser.add_argument("--stage1-train-path", type=str, default=PretrainConfig.stage1_train_path)
    parser.add_argument("--stage1-test-path", type=str, default=PretrainConfig.stage1_test_path)
    parser.add_argument("--stage1-epochs", type=int, default=PretrainConfig.stage1_epochs)
    parser.add_argument("--stage2-train-path", type=str, default=PretrainConfig.stage2_train_path)
    parser.add_argument("--stage2-test-path", type=str, default=PretrainConfig.stage2_test_path)
    parser.add_argument("--stage2-epochs", type=int, default=PretrainConfig.stage2_epochs)
    parser.add_argument("--stage3-train-path", type=str, default=PretrainConfig.stage3_train_path)
    parser.add_argument("--stage3-test-path", type=str, default=PretrainConfig.stage3_test_path)
    parser.add_argument("--stage3-epochs", type=int, default=PretrainConfig.stage3_epochs)
    parser.add_argument("--save-path", type=str, default="artifacts/pretrained_wait_model.npz")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = PretrainConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        target_key=args.target_key,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        log_interval=args.log_interval,
        seed=args.seed,
        stage1_train_path=args.stage1_train_path,
        stage1_test_path=args.stage1_test_path,
        stage1_epochs=args.stage1_epochs,
        stage2_train_path=args.stage2_train_path,
        stage2_test_path=args.stage2_test_path,
        stage2_epochs=args.stage2_epochs,
        stage3_train_path=args.stage3_train_path,
        stage3_test_path=args.stage3_test_path,
        stage3_epochs=args.stage3_epochs,
        save_best_by=args.save_best_by,
    )
    _, summaries = train_pretrained_wait_model(config, save_path=args.save_path)
    last = summaries[-1]
    print(
        f"finished stages={len(_build_stages(config))} epochs={len(summaries)} "
        f"test_loss={last.test_loss:.5f} exact_match={last.exact_match:.3f} "
        f"precision={last.precision:.3f} recall={last.recall:.3f} "
        f"top1_hit={last.top1_hit:.3f} any_legal_acc={last.any_legal_accuracy:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
