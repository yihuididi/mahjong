from .env import MiniMahjongEnv

__all__ = ["MiniMahjongEnv", "MiniTrainingConfig", "main", "train_self_play"]


def __getattr__(name: str):
    if name in {"MiniTrainingConfig", "main", "train_self_play"}:
        from .train import MiniTrainingConfig, main, train_self_play

        exports = {
            "MiniTrainingConfig": MiniTrainingConfig,
            "main": main,
            "train_self_play": train_self_play,
        }
        return exports[name]
    raise AttributeError(f"module 'minitraining' has no attribute {name!r}")
