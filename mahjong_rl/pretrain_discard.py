from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mini_training_ppo.model import MLP

from .compact_observation import compact_observation_from_concealed_hand, compact_observation_size
from .constants import SEAT_WINDS, TILE_TYPE_COUNT


@dataclass(frozen=True)
class DiscardPretrainConfig:
    train_path: str = "artifacts/exact_rule_discard_dataset/train.jsonl"
    test_path: str = "artifacts/exact_rule_discard_dataset/test.jsonl"
    epochs: int = 12
    batch_size: int = 128
    learning_rate: float = 8e-4
    hidden_sizes: tuple[int, ...] = (128, 64)
    seed: int = 41
    log_interval: int = 1
    pretrained_encoder_path: str | None = None
    save_best_by: str = "top1_hit"


@dataclass
class LoadedDiscardData:
    observations: np.ndarray
    candidate_masks: np.ndarray
    target_distributions: np.ndarray
    target_multihot: np.ndarray


@dataclass
class DiscardPretrainSummary:
    epoch: int
    train_loss: float
    test_loss: float
    top1_hit: float
    topk_overlap: float
    elapsed_seconds: float


def load_discard_rows(path: str | Path) -> LoadedDiscardData:
    observations: list[np.ndarray] = []
    candidate_masks: list[np.ndarray] = []
    target_distributions: list[np.ndarray] = []
    target_multihot: list[np.ndarray] = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            player_index = SEAT_WINDS.index(int(row["seat_wind_kind"]))
            best_discards = [int(kind) for kind in row["best_discard_tile_kinds"]]
            observations.append(
                compact_observation_from_concealed_hand(
                    hand_kinds=list(row["hand_tile_kinds"]),
                    player_index=player_index,
                    game_wind=int(row["game_wind_kind"]),
                )
            )
            candidate_masks.append(_discard_candidate_mask(row["hand_tile_kinds"]))
            target_distributions.append(_target_distribution(best_discards))
            target_multihot.append(_multihot(best_discards))

    if not observations:
        raise ValueError(f"No rows found in discard dataset file: {path}")

    return LoadedDiscardData(
        observations=np.stack(observations, axis=0),
        candidate_masks=np.stack(candidate_masks, axis=0),
        target_distributions=np.stack(target_distributions, axis=0),
        target_multihot=np.stack(target_multihot, axis=0),
    )


def train_discard_encoder(
    config: DiscardPretrainConfig,
    save_path: str | None = None,
) -> tuple[MLP, list[DiscardPretrainSummary]]:
    train_data = load_discard_rows(config.train_path)
    test_data = load_discard_rows(config.test_path)

    if train_data.observations.shape[1] != compact_observation_size():
        raise ValueError("Unexpected compact observation size while loading discard dataset.")

    encoder_source: MLP | None = None
    hidden_sizes = config.hidden_sizes
    if config.pretrained_encoder_path:
        encoder_source = MLP.load(config.pretrained_encoder_path)
        if encoder_source.input_dim != train_data.observations.shape[1]:
            raise ValueError("Pretrained encoder input size does not match discard dataset observation size.")
        hidden_sizes = encoder_source.hidden_sizes

    model = MLP(train_data.observations.shape[1], hidden_sizes, TILE_TYPE_COUNT, seed=config.seed)
    if encoder_source is not None:
        model.copy_hidden_from(encoder_source)

    rng = np.random.default_rng(config.seed)
    summaries: list[DiscardPretrainSummary] = []
    best_metric = float("-inf")
    best_payload: tuple[list[np.ndarray], list[np.ndarray]] | None = None
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        permutation = rng.permutation(train_data.observations.shape[0])
        train_losses: list[float] = []

        for start in range(0, train_data.observations.shape[0], config.batch_size):
            batch_idx = permutation[start : start + config.batch_size]
            batch_obs = train_data.observations[batch_idx]
            batch_masks = train_data.candidate_masks[batch_idx]
            batch_targets = train_data.target_distributions[batch_idx]
            logits = model.predict(batch_obs)
            probs = _masked_softmax(logits, batch_masks)
            loss = _cross_entropy_loss(probs, batch_targets)
            grad_output = (probs - batch_targets).astype(np.float32)
            model.train_with_output_gradient(batch_obs, grad_output, config.learning_rate)
            train_losses.append(loss)

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        test_loss, top1_hit, topk_overlap = evaluate_discard_model(model, test_data)
        summaries.append(
            DiscardPretrainSummary(
                epoch=epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                top1_hit=top1_hit,
                topk_overlap=topk_overlap,
                elapsed_seconds=time.time() - start_time,
            )
        )

        metric_value = top1_hit if config.save_best_by == "top1_hit" else topk_overlap
        if metric_value > best_metric:
            best_metric = metric_value
            best_payload = ([weight.copy() for weight in model.weights], [bias.copy() for bias in model.biases])

        if epoch % config.log_interval == 0 or epoch == 1 or epoch == config.epochs:
            print(
                f"epoch={epoch} train_loss={train_loss:.5f} test_loss={test_loss:.5f} "
                f"top1_hit={top1_hit:.3f} topk_overlap={topk_overlap:.3f} "
                f"elapsed={summaries[-1].elapsed_seconds:.1f}s",
                flush=True,
            )

    if best_payload is not None:
        model.weights = [weight.copy() for weight in best_payload[0]]
        model.biases = [bias.copy() for bias in best_payload[1]]

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)

    return model, summaries


def evaluate_discard_model(
    model: MLP,
    data: LoadedDiscardData,
) -> tuple[float, float, float]:
    logits = model.predict(data.observations)
    probs = _masked_softmax(logits, data.candidate_masks)
    loss = _cross_entropy_loss(probs, data.target_distributions)
    top_indices = np.argmax(probs, axis=1)
    top1_hit = float(np.mean([bool(data.target_multihot[i, idx]) for i, idx in enumerate(top_indices)]))

    topk_hits: list[float] = []
    target_counts = np.maximum(data.target_multihot.sum(axis=1).astype(np.int32), 1)
    for row_index in range(probs.shape[0]):
        k = int(target_counts[row_index])
        topk = np.argpartition(probs[row_index], -k)[-k:]
        overlap = float(data.target_multihot[row_index, topk].sum()) / float(k)
        topk_hits.append(overlap)

    return float(loss), top1_hit, float(np.mean(topk_hits))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(shifted, -30.0, 30.0))
    denom = np.sum(exp_values, axis=1, keepdims=True)
    return (exp_values / np.maximum(denom, 1e-8)).astype(np.float32)


def _masked_softmax(logits: np.ndarray, candidate_masks: np.ndarray) -> np.ndarray:
    masks = candidate_masks.astype(np.float32, copy=False)
    shifted_logits = np.where(masks > 0.0, logits, -1e9)
    shifted_logits = shifted_logits - np.max(shifted_logits, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(shifted_logits, -30.0, 30.0)) * masks
    denom = np.sum(exp_values, axis=1, keepdims=True)
    return (exp_values / np.maximum(denom, 1e-8)).astype(np.float32)


def _cross_entropy_loss(probs: np.ndarray, target_distributions: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-8, 1.0)
    return float(-np.mean(np.sum(target_distributions * np.log(clipped), axis=1)))


def _target_distribution(tile_kinds: list[int]) -> np.ndarray:
    target = np.zeros(TILE_TYPE_COUNT, dtype=np.float32)
    if not tile_kinds:
        return target
    value = 1.0 / float(len(tile_kinds))
    for tile_kind in tile_kinds:
        target[int(tile_kind)] = value
    return target


def _multihot(tile_kinds: list[int]) -> np.ndarray:
    target = np.zeros(TILE_TYPE_COUNT, dtype=np.float32)
    for tile_kind in tile_kinds:
        target[int(tile_kind)] = 1.0
    return target


def _discard_candidate_mask(hand_tile_kinds: list[int]) -> np.ndarray:
    mask = np.zeros(TILE_TYPE_COUNT, dtype=np.float32)
    for tile_kind in set(int(kind) for kind in hand_tile_kinds):
        mask[tile_kind] = 1.0
    return mask


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a compact Mahjong encoder on exact-rule discard targets."
    )
    parser.add_argument("--train-path", type=str, default=DiscardPretrainConfig.train_path)
    parser.add_argument("--test-path", type=str, default=DiscardPretrainConfig.test_path)
    parser.add_argument("--epochs", type=int, default=DiscardPretrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=DiscardPretrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=DiscardPretrainConfig.learning_rate)
    parser.add_argument("--log-interval", type=int, default=DiscardPretrainConfig.log_interval)
    parser.add_argument("--seed", type=int, default=DiscardPretrainConfig.seed)
    parser.add_argument("--pretrained-encoder-path", type=str, default=DiscardPretrainConfig.pretrained_encoder_path)
    parser.add_argument("--save-best-by", type=str, choices=("top1_hit", "topk_overlap"), default=DiscardPretrainConfig.save_best_by)
    parser.add_argument("--save-path", type=str, default="artifacts/pretrained_discard_encoder.npz")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = DiscardPretrainConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        log_interval=args.log_interval,
        pretrained_encoder_path=args.pretrained_encoder_path,
        save_best_by=args.save_best_by,
    )
    _, summaries = train_discard_encoder(config, save_path=args.save_path)
    last = summaries[-1]
    print(
        f"finished epochs={len(summaries)} test_loss={last.test_loss:.5f} "
        f"top1_hit={last.top1_hit:.3f} topk_overlap={last.topk_overlap:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
