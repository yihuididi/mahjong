from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mahjong_rl.constants import ACTION_COUNT, NUM_PLAYERS

from .env import PPOCompatibleMahjongEnv
from .model import PPOAgent, PPOConfig


@dataclass
class PendingDecision:
    observation: np.ndarray
    action_mask: np.ndarray
    action: int
    log_prob: float
    value: float


@dataclass
class TrajectoryItem:
    player: int
    observation: np.ndarray
    action_mask: np.ndarray
    action: int
    old_log_prob: float
    value: float
    reward: float
    next_observation: np.ndarray
    next_action_mask: np.ndarray
    done: bool


@dataclass
class RolloutItem:
    observation: np.ndarray
    action_mask: np.ndarray
    action: int
    old_log_prob: float
    return_estimate: float
    advantage: float


@dataclass
class EpochSummary:
    epoch: int
    episodes: int
    transitions: int
    mean_rewards: np.ndarray
    policy_loss: float
    value_loss: float
    entropy: float
    wins: int
    draws: int
    mean_episode_steps: float
    aggregated_stats: dict[str, int]
    elapsed_seconds: float


def train_ppo(
    config: PPOConfig,
    actor_save_path: str | None = None,
    critic_save_path: str | None = None,
    log_path: str | None = None,
) -> tuple[PPOAgent, list[EpochSummary]]:
    env = PPOCompatibleMahjongEnv(seed=config.seed, progress_reward_scale=config.progress_reward_scale)
    agent = PPOAgent(observation_size=env.observation_size, config=config)
    zero_observation = np.zeros(env.observation_size, dtype=np.float32)
    zero_action_mask = np.zeros(ACTION_COUNT, dtype=np.float32)

    summaries: list[EpochSummary] = []
    start_time = time.time()
    csv_file = None
    csv_writer = None

    if log_path is not None:
        log_file_path = Path(log_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = log_file_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "epoch",
                "episodes",
                "transitions",
                "reward_p0",
                "reward_p1",
                "reward_p2",
                "wins",
                "draws",
                "mean_episode_steps",
                "policy_loss",
                "value_loss",
                "entropy",
                "self_draw_game_opportunities",
                "discard_game_opportunities",
                "complete_hand_zero_tai",
                "peng_opportunities",
                "chi_opportunities",
                "peng_taken",
                "chi_taken",
                "missed_game_actions",
                "elapsed_seconds",
            ]
        )

    try:
        for epoch in range(1, config.epochs + 1):
            rollout: list[RolloutItem] = []
            aggregate_rewards = np.zeros(NUM_PLAYERS, dtype=np.float32)
            aggregate_stats = _empty_stats()
            episode_steps: list[int] = []
            wins = 0
            draws = 0

            for _ in range(config.episodes_per_epoch):
                episode_trajectories: list[list[TrajectoryItem]] = [[] for _ in range(NUM_PLAYERS)]
                observation, action_mask, current_player = env.reset()
                pending: list[PendingDecision | None] = [None] * NUM_PLAYERS
                pending_rewards = np.zeros(NUM_PLAYERS, dtype=np.float32)
                episode_reward = np.zeros(NUM_PLAYERS, dtype=np.float32)
                step_count = 0

                for _ in range(config.max_steps_per_episode):
                    action, log_prob, value = agent.select_action(observation, action_mask)
                    actor = int(current_player)
                    pending[actor] = PendingDecision(
                        observation=observation.copy(),
                        action_mask=action_mask.copy(),
                        action=action,
                        log_prob=log_prob,
                        value=value,
                    )
                    pending_rewards[actor] = 0.0

                    step_result = env.step(action)
                    step_count += 1
                    episode_reward += step_result.rewards

                    for player in range(NUM_PLAYERS):
                        if pending[player] is not None:
                            pending_rewards[player] += step_result.rewards[player]

                    if step_result.done:
                        for player in range(NUM_PLAYERS):
                            if pending[player] is None:
                                continue
                            decision = pending[player]
                            episode_trajectories[player].append(
                                TrajectoryItem(
                                    player=player,
                                    observation=decision.observation,
                                    action_mask=decision.action_mask,
                                    action=decision.action,
                                    old_log_prob=decision.log_prob,
                                    value=decision.value,
                                    reward=float(pending_rewards[player]),
                                    next_observation=zero_observation,
                                    next_action_mask=zero_action_mask,
                                    done=True,
                                )
                            )
                            pending[player] = None
                            pending_rewards[player] = 0.0
                    else:
                        current_player = int(step_result.current_player)
                        next_observation = step_result.observation
                        next_action_mask = step_result.action_mask
                        assert next_observation is not None
                        assert next_action_mask is not None

                        if pending[current_player] is not None:
                            decision = pending[current_player]
                            episode_trajectories[current_player].append(
                                TrajectoryItem(
                                    player=current_player,
                                    observation=decision.observation,
                                    action_mask=decision.action_mask,
                                    action=decision.action,
                                    old_log_prob=decision.log_prob,
                                    value=decision.value,
                                    reward=float(pending_rewards[current_player]),
                                    next_observation=next_observation.copy(),
                                    next_action_mask=next_action_mask.copy(),
                                    done=False,
                                )
                            )
                            pending[current_player] = None
                            pending_rewards[current_player] = 0.0

                        observation = next_observation
                        action_mask = next_action_mask

                    if step_result.done:
                        stats = dict(step_result.info.get("episode_stats", {}))
                        _merge_stats(aggregate_stats, stats)
                        wins += stats.get("wins", 0)
                        draws += stats.get("draw_games", 0)
                        aggregate_rewards += episode_reward
                        episode_steps.append(step_count)
                        _extend_rollout_with_episode_sequences(
                            rollout=rollout,
                            episode_trajectories=episode_trajectories,
                            agent=agent,
                            config=config,
                        )
                        break
                else:
                    aggregate_rewards += episode_reward
                    episode_steps.append(step_count)
                    _merge_stats(aggregate_stats, dict(getattr(env, "episode_stats", {})))
                    _extend_rollout_with_episode_sequences(
                        rollout=rollout,
                        episode_trajectories=episode_trajectories,
                        agent=agent,
                        config=config,
                    )

            observations = np.stack([item.observation for item in rollout], axis=0)
            action_masks = np.stack([item.action_mask for item in rollout], axis=0)
            actions = np.asarray([item.action for item in rollout], dtype=np.int64)
            old_log_probs = np.asarray([item.old_log_prob for item in rollout], dtype=np.float32)
            returns = np.asarray([item.return_estimate for item in rollout], dtype=np.float32)
            advantages = np.asarray([item.advantage for item in rollout], dtype=np.float32)

            policy_loss, value_loss, entropy = agent.update(
                observations=observations,
                action_masks=action_masks,
                actions=actions,
                old_log_probs=old_log_probs,
                returns=returns,
                advantages=advantages,
            )

            summary = EpochSummary(
                epoch=epoch,
                episodes=config.episodes_per_epoch,
                transitions=len(rollout),
                mean_rewards=(aggregate_rewards / float(max(config.episodes_per_epoch, 1))).astype(np.float32),
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy=entropy,
                wins=wins,
                draws=draws,
                mean_episode_steps=float(np.mean(episode_steps)) if episode_steps else 0.0,
                aggregated_stats=aggregate_stats,
                elapsed_seconds=time.time() - start_time,
            )
            summaries.append(summary)
            _log_epoch_summary(summary, config.log_interval, config.epochs, csv_writer)

        if actor_save_path is not None and critic_save_path is not None:
            Path(actor_save_path).parent.mkdir(parents=True, exist_ok=True)
            Path(critic_save_path).parent.mkdir(parents=True, exist_ok=True)
            agent.save(actor_save_path, critic_save_path)

        return agent, summaries
    finally:
        if csv_file is not None:
            csv_file.close()


def _empty_stats() -> dict[str, int]:
    return {
        "self_draw_game_opportunities": 0,
        "discard_game_opportunities": 0,
        "complete_hand_zero_tai": 0,
        "peng_opportunities": 0,
        "chi_opportunities": 0,
        "peng_taken": 0,
        "chi_taken": 0,
        "missed_game_actions": 0,
        "wins": 0,
        "draw_games": 0,
    }


def _merge_stats(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def compute_gae_returns_and_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not (
        rewards.shape == values.shape == next_values.shape == dones.shape
    ):
        raise ValueError("GAE inputs must have matching shapes.")

    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for index in reversed(range(rewards.shape[0])):
        nonterminal = 1.0 - float(dones[index])
        delta = float(rewards[index]) + (gamma * float(next_values[index]) * nonterminal) - float(values[index])
        gae = delta + (gamma * gae_lambda * nonterminal * gae)
        advantages[index] = gae
    returns = advantages + values.astype(np.float32)
    return returns.astype(np.float32), advantages.astype(np.float32)


def _extend_rollout_with_episode_sequences(
    rollout: list[RolloutItem],
    episode_trajectories: list[list[TrajectoryItem]],
    agent: PPOAgent,
    config: PPOConfig,
) -> None:
    for player_trajectory in episode_trajectories:
        if not player_trajectory:
            continue
        next_observations = np.stack([item.next_observation for item in player_trajectory], axis=0)
        next_values = agent.values(next_observations)
        values = np.asarray([item.value for item in player_trajectory], dtype=np.float32)
        rewards = np.asarray([item.reward for item in player_trajectory], dtype=np.float32)
        dones = np.asarray([item.done for item in player_trajectory], dtype=np.float32)
        next_values = np.where(dones > 0.0, 0.0, next_values)
        returns, advantages = compute_gae_returns_and_advantages(
            rewards=rewards,
            values=values,
            next_values=next_values.astype(np.float32),
            dones=dones,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        for item, return_estimate, advantage in zip(player_trajectory, returns, advantages):
            rollout.append(
                RolloutItem(
                    observation=item.observation,
                    action_mask=item.action_mask,
                    action=item.action,
                    old_log_prob=item.old_log_prob,
                    return_estimate=float(return_estimate),
                    advantage=float(advantage),
                )
            )


def _log_epoch_summary(
    summary: EpochSummary,
    log_interval: int,
    total_epochs: int,
    csv_writer: csv.writer | None,
) -> None:
    stats = summary.aggregated_stats
    if csv_writer is not None:
        csv_writer.writerow(
            [
                summary.epoch,
                summary.episodes,
                summary.transitions,
                float(summary.mean_rewards[0]),
                float(summary.mean_rewards[1]),
                float(summary.mean_rewards[2]),
                summary.wins,
                summary.draws,
                summary.mean_episode_steps,
                summary.policy_loss,
                summary.value_loss,
                summary.entropy,
                stats.get("self_draw_game_opportunities", 0),
                stats.get("discard_game_opportunities", 0),
                stats.get("complete_hand_zero_tai", 0),
                stats.get("peng_opportunities", 0),
                stats.get("chi_opportunities", 0),
                stats.get("peng_taken", 0),
                stats.get("chi_taken", 0),
                stats.get("missed_game_actions", 0),
                summary.elapsed_seconds,
            ]
        )

    if summary.epoch % log_interval == 0 or summary.epoch == 1 or summary.epoch == total_epochs:
        print(
            f"epoch={summary.epoch} episodes={summary.episodes} transitions={summary.transitions} "
            f"mean_reward={summary.mean_rewards.tolist()} wins={summary.wins} draws={summary.draws} "
            f"steps={summary.mean_episode_steps:.1f} policy_loss={summary.policy_loss:.5f} "
            f"value_loss={summary.value_loss:.5f} entropy={summary.entropy:.5f} "
            f"opp(self/discard/zeroTai)="
            f"{stats.get('self_draw_game_opportunities', 0)}/"
            f"{stats.get('discard_game_opportunities', 0)}/"
            f"{stats.get('complete_hand_zero_tai', 0)} "
            f"claim(p/c)={stats.get('peng_taken', 0)}/{stats.get('chi_taken', 0)} "
            f"missedGame={stats.get('missed_game_actions', 0)} "
            f"claimOpp(p/c)={stats.get('peng_opportunities', 0)}/{stats.get('chi_opportunities', 0)} "
            f"elapsed={summary.elapsed_seconds:.1f}s",
            flush=True,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Short-epoch PPO training for 3-player Mahjong.")
    parser.add_argument("--epochs", type=int, default=PPOConfig.epochs)
    parser.add_argument("--episodes-per-epoch", type=int, default=PPOConfig.episodes_per_epoch)
    parser.add_argument("--max-steps", type=int, default=PPOConfig.max_steps_per_episode)
    parser.add_argument("--gamma", type=float, default=PPOConfig.gamma)
    parser.add_argument("--gae-lambda", type=float, default=PPOConfig.gae_lambda)
    parser.add_argument("--actor-learning-rate", type=float, default=PPOConfig.actor_learning_rate)
    parser.add_argument("--critic-learning-rate", type=float, default=PPOConfig.critic_learning_rate)
    parser.add_argument("--clip-ratio", type=float, default=PPOConfig.clip_ratio)
    parser.add_argument("--update-epochs", type=int, default=PPOConfig.update_epochs)
    parser.add_argument("--minibatch-size", type=int, default=PPOConfig.minibatch_size)
    parser.add_argument("--log-interval", type=int, default=PPOConfig.log_interval)
    parser.add_argument("--progress-reward-scale", type=float, default=PPOConfig.progress_reward_scale)
    parser.add_argument("--pretrained-encoder-path", type=str, default=PPOConfig.pretrained_encoder_path)
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--actor-save-path", type=str, default="artifacts/ppo_actor.npz")
    parser.add_argument("--critic-save-path", type=str, default="artifacts/ppo_critic.npz")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = PPOConfig(
        epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        max_steps_per_episode=args.max_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        clip_ratio=args.clip_ratio,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        log_interval=args.log_interval,
        progress_reward_scale=args.progress_reward_scale,
        pretrained_encoder_path=args.pretrained_encoder_path,
        seed=args.seed,
    )
    _, summaries = train_ppo(
        config,
        actor_save_path=args.actor_save_path,
        critic_save_path=args.critic_save_path,
        log_path=args.log_path,
    )
    last = summaries[-1]
    print(
        f"finished epochs={len(summaries)} wins={last.wins} draws={last.draws} "
        f"mean_reward={last.mean_rewards.tolist()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
