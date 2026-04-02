from .env import PPOCompatibleMahjongEnv

__all__ = ["PPOCompatibleMahjongEnv", "PPOConfig", "main", "train_ppo"]


def __getattr__(name: str):
    if name in {"PPOConfig", "main", "train_ppo"}:
        from .train import PPOConfig, main, train_ppo

        exports = {
            "PPOConfig": PPOConfig,
            "main": main,
            "train_ppo": train_ppo,
        }
        return exports[name]
    raise AttributeError(f"module 'mini_training_ppo' has no attribute {name!r}")
