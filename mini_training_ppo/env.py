from __future__ import annotations

from minitraining.env import MiniMahjongEnv


class PPOCompatibleMahjongEnv(MiniMahjongEnv):
    """Same compact-rule environment, reused for PPO experiments."""

