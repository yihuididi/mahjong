from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mahjong_rl.constants import ACTION_COUNT, NUM_PLAYERS
from mahjong_rl.dqn import DQNAgent, ReplayBuffer, TrainingConfig

from .env import MiniMahjongEnv


@dataclass(frozen=True)
class MiniTrainingConfig(TrainingConfig):
    episodes: int = 2000
    max_steps_per_episode: int = 300
    replay_capacity: int = 20_000
    batch_size: int = 32
    learning_rate: float = 8e-4
    hidden_sizes: tuple[int, ...] = (64, 32)
    epsilon_decay_steps: int = 10_000
    warmup_steps: int = 256
    target_sync_interval: int = 250
    log_interval: int = 50
    progress_reward_scale: float = 0.03


@dataclass
class PendingTransition:
    observation: np.ndarray
    action_mask: np.ndarray
    action: int


@dataclass
class EpisodeSummary:
    episode: int
    rewards: np.ndarray
    mean_loss: float
    epsilon: float
    winner: int | None
    winner_tai: int
    terminal_reason: str | None
    episode_stats: dict[str, int]


def train_self_play(
    config: MiniTrainingConfig,
    save_path: str | None = None,
    log_path: str | None = None,
) -> tuple[DQNAgent, list[EpisodeSummary]]:
    env = MiniMahjongEnv(seed=config.seed, progress_reward_scale=config.progress_reward_scale)
    agent = DQNAgent(observation_size=env.observation_size, config=config)
    replay_buffer = ReplayBuffer(config.replay_capacity, env.observation_size, seed=config.seed)
    zero_observation = np.zeros(env.observation_size, dtype=np.float32)
    zero_action_mask = np.zeros(ACTION_COUNT, dtype=np.float32)

    summaries: list[EpisodeSummary] = []
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
                "episode",
                "steps",
                "reward_p0",
                "reward_p1",
                "reward_p2",
                "mean_loss",
                "epsilon",
                "winner",
                "winner_tai",
                "terminal_reason",
                "self_draw_game_opportunities",
                "discard_game_opportunities",
                "complete_hand_zero_tai",
                "peng_opportunities",
                "chi_opportunities",
                "peng_taken",
                "chi_taken",
                "missed_game_actions",
                "wins",
                "elapsed_seconds",
            ]
        )

    try:
        for episode in range(1, config.episodes + 1):
            observation, action_mask, current_player = env.reset()
            pending: list[PendingTransition | None] = [None] * NUM_PLAYERS
            pending_rewards = np.zeros(NUM_PLAYERS, dtype=np.float32)
            episode_rewards = np.zeros(NUM_PLAYERS, dtype=np.float32)
            episode_losses: list[float] = []
            episode_step_count = 0

            for _ in range(config.max_steps_per_episode):
                action = agent.select_action(observation, action_mask, explore=True)
                actor = current_player
                pending[actor] = PendingTransition(
                    observation=observation.copy(),
                    action_mask=action_mask.copy(),
                    action=action,
                )
                pending_rewards[actor] = 0.0

                step_result = env.step(action)
                episode_step_count += 1

                episode_rewards += step_result.rewards
                for player in range(NUM_PLAYERS):
                    if pending[player] is not None:
                        pending_rewards[player] += step_result.rewards[player]

                if step_result.done:
                    for player in range(NUM_PLAYERS):
                        if pending[player] is None:
                            continue
                        transition = pending[player]
                        replay_buffer.add(
                            transition.observation,
                            transition.action_mask,
                            transition.action,
                            float(pending_rewards[player]),
                            zero_observation,
                            zero_action_mask,
                            True,
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
                        transition = pending[current_player]
                        replay_buffer.add(
                            transition.observation,
                            transition.action_mask,
                            transition.action,
                            float(pending_rewards[current_player]),
                            next_observation,
                            next_action_mask,
                            False,
                        )
                        pending[current_player] = None
                        pending_rewards[current_player] = 0.0

                    observation = next_observation
                    action_mask = next_action_mask

                if len(replay_buffer) >= max(config.warmup_steps, config.batch_size):
                    loss = agent.update(replay_buffer, config.batch_size)
                    if loss is not None:
                        episode_losses.append(loss)

                if step_result.done:
                    summary = EpisodeSummary(
                        episode=episode,
                        rewards=episode_rewards.copy(),
                        mean_loss=float(np.mean(episode_losses)) if episode_losses else 0.0,
                        epsilon=agent.epsilon,
                        winner=step_result.info["winner"],
                        winner_tai=step_result.info["winner_tai"],
                        terminal_reason=step_result.info["terminal_reason"],
                        episode_stats=dict(step_result.info.get("episode_stats", {})),
                    )
                    summaries.append(summary)
                    _log_episode_summary(
                        summary=summary,
                        replay_size=len(replay_buffer),
                        steps=episode_step_count,
                        elapsed_seconds=time.time() - start_time,
                        log_interval=config.log_interval,
                        total_episodes=config.episodes,
                        csv_writer=csv_writer,
                    )
                    break
            else:
                summary = EpisodeSummary(
                    episode=episode,
                    rewards=episode_rewards.copy(),
                    mean_loss=float(np.mean(episode_losses)) if episode_losses else 0.0,
                    epsilon=agent.epsilon,
                    winner=None,
                    winner_tai=0,
                    terminal_reason="step_limit",
                    episode_stats=dict(getattr(env, "episode_stats", {})),
                )
                summaries.append(summary)
                _log_episode_summary(
                    summary=summary,
                    replay_size=len(replay_buffer),
                    steps=episode_step_count,
                    elapsed_seconds=time.time() - start_time,
                    log_interval=config.log_interval,
                    total_episodes=config.episodes,
                    csv_writer=csv_writer,
                )

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            agent.save(save_path)

        return agent, summaries
    finally:
        if csv_file is not None:
            csv_file.close()


def _log_episode_summary(
    summary: EpisodeSummary,
    replay_size: int,
    steps: int,
    elapsed_seconds: float,
    log_interval: int,
    total_episodes: int,
    csv_writer: csv.writer | None,
) -> None:
    if csv_writer is not None:
        csv_writer.writerow(
            [
                summary.episode,
                steps,
                float(summary.rewards[0]),
                float(summary.rewards[1]),
                float(summary.rewards[2]),
                summary.mean_loss,
                summary.epsilon,
                summary.winner,
                summary.winner_tai,
                summary.terminal_reason,
                summary.episode_stats.get("self_draw_game_opportunities", 0),
                summary.episode_stats.get("discard_game_opportunities", 0),
                summary.episode_stats.get("complete_hand_zero_tai", 0),
                summary.episode_stats.get("peng_opportunities", 0),
                summary.episode_stats.get("chi_opportunities", 0),
                summary.episode_stats.get("peng_taken", 0),
                summary.episode_stats.get("chi_taken", 0),
                summary.episode_stats.get("missed_game_actions", 0),
                summary.episode_stats.get("wins", 0),
                elapsed_seconds,
            ]
        )

    if summary.episode % log_interval == 0 or summary.episode == 1 or summary.episode == total_episodes:
        stats = summary.episode_stats
        print(
            f"episode={summary.episode} buffer={replay_size} steps={steps} "
            f"epsilon={summary.epsilon:.3f} reward={summary.rewards.tolist()} "
            f"winner={summary.winner} tai={summary.winner_tai} reason={summary.terminal_reason} "
            f"avg_loss={summary.mean_loss:.5f} "
            f"opp(self/discard/zeroTai)={stats.get('self_draw_game_opportunities', 0)}/"
            f"{stats.get('discard_game_opportunities', 0)}/"
            f"{stats.get('complete_hand_zero_tai', 0)} "
            f"claim(p/c)={stats.get('peng_taken', 0)}/{stats.get('chi_taken', 0)} "
            f"missedGame={stats.get('missed_game_actions', 0)} "
            f"claimOpp(p/c)={stats.get('peng_opportunities', 0)}/{stats.get('chi_opportunities', 0)} "
            f"elapsed={elapsed_seconds:.1f}s",
            flush=True,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast compact-observation training for 3-player Mahjong.")
    parser.add_argument("--episodes", type=int, default=MiniTrainingConfig.episodes)
    parser.add_argument("--max-steps", type=int, default=MiniTrainingConfig.max_steps_per_episode)
    parser.add_argument("--batch-size", type=int, default=MiniTrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=MiniTrainingConfig.learning_rate)
    parser.add_argument("--gamma", type=float, default=MiniTrainingConfig.gamma)
    parser.add_argument("--warmup-steps", type=int, default=MiniTrainingConfig.warmup_steps)
    parser.add_argument("--target-sync", type=int, default=MiniTrainingConfig.target_sync_interval)
    parser.add_argument("--replay-capacity", type=int, default=MiniTrainingConfig.replay_capacity)
    parser.add_argument("--epsilon-start", type=float, default=MiniTrainingConfig.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=MiniTrainingConfig.epsilon_end)
    parser.add_argument("--epsilon-decay", type=int, default=MiniTrainingConfig.epsilon_decay_steps)
    parser.add_argument("--updates-per-step", type=int, default=MiniTrainingConfig.updates_per_step)
    parser.add_argument("--log-interval", type=int, default=MiniTrainingConfig.log_interval)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--progress-reward-scale", type=float, default=MiniTrainingConfig.progress_reward_scale)
    parser.add_argument("--seed", type=int, default=MiniTrainingConfig.seed)
    parser.add_argument("--save-path", type=str, default="artifacts/minitraining_agent.npz")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = MiniTrainingConfig(
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay,
        warmup_steps=args.warmup_steps,
        target_sync_interval=args.target_sync,
        updates_per_step=args.updates_per_step,
        log_interval=args.log_interval,
        progress_reward_scale=args.progress_reward_scale,
        seed=args.seed,
    )
    _, summaries = train_self_play(config, save_path=args.save_path, log_path=args.log_path)
    last = summaries[-1]
    print(
        f"finished episodes={len(summaries)} final_reason={last.terminal_reason} "
        f"winner={last.winner} tai={last.winner_tai} epsilon={last.epsilon:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
