# 3P Mahjong RL

Training-ready 3-player Mahjong environment and RL training pipeline for the rules discussed in this repo.

## Implemented Rules

- Tile set: 100 tiles total
  - 2 suits only: `B1-B9`, `D1-D9`
  - honors: `E S W N R G Wh`
  - 4 copies of each tile type
- No flowers, animals, `万子`, `gang`, or `咬`
- Random game wind from `E/S/W/N`
- Dealer is canonicalized to internal player `0`
  - seat winds are always `player 0 = E`, `player 1 = S`, `player 2 = W`
- Starting player is random and independent from dealer
- Initial hands:
  - starting player gets `14`
  - others get `13`
- Turn flow:
  - normal turn draws one random undrawn tile unless the player just chi/peng'd
  - player may `Game` only if the hand is complete and has at least `1 tai`
  - otherwise player discards
- Claim priority after discard:
  - `Game`
  - `Peng`
  - `Chi`
  - tie-break by earliest player after the thrower
- `Chi` only from the previous player in turn order
- `Peng` from any player
- After `chi/peng`, the claimant does not draw and must immediately discard
- Draw game happens when `15` undrawn tiles remain

## Scoring

- Minimum `1 tai` to win
- Tai is capped at `5`
- Exposed `peng` scoring:
  - own wind: `+1`
  - game wind: `+1`
  - if the same exposed wind set is both own wind and game wind: `+2`
  - `N`: `+1` for any player, and `+2` total if the game wind is also `N`
  - `R/G/Wh`: `+1` each
- Pattern scoring:
  - all straights: `+1`
  - all peng: `+2`
  - pure one suit: `+4`
  - half flush: `+2`
  - all honors: capped `5 tai`

Current implementation assumes the pattern tai above can stack with each other and with the exposed `peng` tai, then the result is capped at `5`.

## RL Design

- Internal game manager tracks all 100 physical tiles with the requested tile state encoding
- Agent observation masks opponent concealed tiles to unknown
- One shared DQN policy controls all 3 players in self-play
- Environment uses legal-action masking
- RL action space is `20` actions:
  - `14` discard slots
  - `3` chi actions
  - `1` peng
  - `1` game
  - `1` pass

`pass` is included for training because the agent must be able to decline legal claims.

## Project Layout

- [mahjong_rl/constants.py](/Users/jarrellchia/Downloads/CS3263/mahjong/mahjong_rl/constants.py)
- [mahjong_rl/rules.py](/Users/jarrellchia/Downloads/CS3263/mahjong/mahjong_rl/rules.py)
- [mahjong_rl/env.py](/Users/jarrellchia/Downloads/CS3263/mahjong/mahjong_rl/env.py)
- [mahjong_rl/dqn.py](/Users/jarrellchia/Downloads/CS3263/mahjong/mahjong_rl/dqn.py)
- [mahjong_rl/train.py](/Users/jarrellchia/Downloads/CS3263/mahjong/mahjong_rl/train.py)
- [minitraining/env.py](/Users/jarrellchia/Downloads/CS3263/mahjong/minitraining/env.py)
- [minitraining/train.py](/Users/jarrellchia/Downloads/CS3263/mahjong/minitraining/train.py)
- [mini_training_ppo/env.py](/Users/jarrellchia/Downloads/CS3263/mahjong/mini_training_ppo/env.py)
- [mini_training_ppo/model.py](/Users/jarrellchia/Downloads/CS3263/mahjong/mini_training_ppo/model.py)
- [mini_training_ppo/train.py](/Users/jarrellchia/Downloads/CS3263/mahjong/mini_training_ppo/train.py)
- [tests/test_env.py](/Users/jarrellchia/Downloads/CS3263/mahjong/tests/test_env.py)
- [tests/test_training.py](/Users/jarrellchia/Downloads/CS3263/mahjong/tests/test_training.py)
- [tests/test_minitraining.py](/Users/jarrellchia/Downloads/CS3263/mahjong/tests/test_minitraining.py)
- [tests/test_mini_training_ppo.py](/Users/jarrellchia/Downloads/CS3263/mahjong/tests/test_mini_training_ppo.py)

## Run Tests

```bash
python3 -m unittest discover -s tests -v
```

## Train

```bash
python3 -m mahjong_rl.train --episodes 2000 --save-path artifacts/dqn_agent.npz
```

Example quick smoke run:

```bash
python3 -m mahjong_rl.train --episodes 1 --max-steps 50 --batch-size 8 --warmup-steps 8
```

## MiniTraining

`minitraining` keeps the same Mahjong rules and game-manager logic, but uses a much smaller observation vector and smaller default network for faster local experiments.

```bash
python3 -u -m minitraining.train --episodes 2000 --log-interval 10 --log-path artifacts/minitraining_log.csv --save-path artifacts/minitraining_agent.npz
```

## MiniTraining PPO

`mini_training_ppo` keeps the same compact environment and rules, but swaps the learner to a lightweight masked PPO actor-critic with short epochs.

```bash
python3 -u -m mini_training_ppo.train --epochs 20 --episodes-per-epoch 5 --log-interval 1 --log-path artifacts/ppo_log.csv --actor-save-path artifacts/ppo_actor.npz --critic-save-path artifacts/ppo_critic.npz
```

## Exact-Rule Dataset

You can generate a concealed 13-tile supervised dataset from this repo's exact rules. Each row includes:

- `winning_shape_tiles`: tiles that complete `4 melds + 1 pair`
- `legal_game_tiles`: tiles that complete the hand and satisfy `tai >= 1`
- `improving_tiles`: tiles that improve the hand-progress score

```bash
python3 -m mahjong_rl.generate_dataset --train-size 5000 --test-size 500 --include-empty --output-dir artifacts/exact_rule_dataset
```

`--include-empty` now uses a mixed sampler: some rows are fully random concealed hands and the rest are backward-sampled structured hands. That keeps hard negative examples without collapsing the broad stage into almost-all-empty waits.

For curriculum-style pretraining, generate three views:

```bash
python3 -m mahjong_rl.generate_dataset --train-size 10000 --test-size 1000 --include-empty --output-dir artifacts/exact_rule_dataset_with_empty
python3 -m mahjong_rl.generate_dataset --train-size 10000 --test-size 1000 --output-dir artifacts/exact_rule_dataset
python3 -m mahjong_rl.generate_dataset --train-size 10000 --test-size 1000 --legal-only --output-dir artifacts/exact_rule_dataset_legal_only
```

If you want more or fewer random empty-hand rows in the broad stage, tune `--include-empty-random-ratio`.

You can also generate a concealed 14-tile discard dataset that labels which discard preserves the most future winning potential:

```bash
python3 -m mahjong_rl.generate_discard_dataset --train-size 10000 --test-size 1000 --output-dir artifacts/exact_rule_discard_dataset
```

For a stronger curriculum, you can also generate recursive backward states from legal terminal hands. These rows include `steps_from_terminal` and `path_progress_tiles`, so the encoder sees partial states that are multiple tile additions away from a legal win:

```bash
python3 -m mahjong_rl.generate_backward_curriculum_dataset \
  --train-size 10000 \
  --test-size 1000 \
  --max-steps-from-terminal 4 \
  --min-hand-size 10 \
  --branching-factor 2 \
  --output-dir artifacts/backward_curriculum_dataset
```

## Supervised Pretraining

You can pretrain a compact multi-task hand-understanding model on the exact-rule dataset, then use its hidden layers to initialize PPO.

Current supervised targets are:

- `winning_shape_tiles`
- `legal_game_tiles`
- `improving_tiles`
- `completion_score`
- `has_any_winning_shape`
- `has_any_legal_game`
- `legal_wait_count`

```bash
python3 -m mahjong_rl.pretrain \
  --stage1-train-path artifacts/backward_curriculum_dataset/train.jsonl \
  --stage1-test-path artifacts/backward_curriculum_dataset/test.jsonl \
  --stage1-epochs 8 \
  --stage2-train-path artifacts/exact_rule_dataset/train.jsonl \
  --stage2-test-path artifacts/exact_rule_dataset/test.jsonl \
  --stage2-epochs 6 \
  --stage3-train-path artifacts/exact_rule_dataset_legal_only/train.jsonl \
  --stage3-test-path artifacts/exact_rule_dataset_legal_only/test.jsonl \
  --stage3-epochs 6 \
  --save-path artifacts/pretrained_wait_model_backward_curriculum.npz
```

The older exact-rule-only curriculum still works, but the backward curriculum stage tends to be a better first stage because it includes deeper non-terminal states instead of only one-tile waits.

```bash
python3 -m mahjong_rl.pretrain \
  --stage1-train-path artifacts/exact_rule_dataset_with_empty/train.jsonl \
  --stage1-test-path artifacts/exact_rule_dataset_with_empty/test.jsonl \
  --stage1-epochs 8 \
  --stage2-train-path artifacts/exact_rule_dataset/train.jsonl \
  --stage2-test-path artifacts/exact_rule_dataset/test.jsonl \
  --stage2-epochs 6 \
  --stage3-train-path artifacts/exact_rule_dataset_legal_only/train.jsonl \
  --stage3-test-path artifacts/exact_rule_dataset_legal_only/test.jsonl \
  --stage3-epochs 6 \
  --save-path artifacts/pretrained_wait_model.npz
```

Then start PPO with the pretrained encoder:

```bash
python3 -u -m mini_training_ppo.train \
  --epochs 100 \
  --episodes-per-epoch 5 \
  --pretrained-encoder-path artifacts/pretrained_wait_model.npz \
  --log-interval 5 \
  --log-path artifacts/ppo_pretrained.csv \
  --actor-save-path artifacts/ppo_actor_pretrained.npz \
  --critic-save-path artifacts/ppo_critic_pretrained.npz
```

## Discard-Focused Finetuning

After the wait curriculum pretraining, you can sharpen the same encoder on best-discard supervision:

```bash
python3 -m mahjong_rl.pretrain_discard \
  --train-path artifacts/exact_rule_discard_dataset/train.jsonl \
  --test-path artifacts/exact_rule_discard_dataset/test.jsonl \
  --pretrained-encoder-path artifacts/pretrained_wait_model_curriculum.npz \
  --save-best-by top1_hit \
  --save-path artifacts/pretrained_discard_encoder.npz
```

Discard pretraining masks logits to tile kinds that are actually present in the 14-tile hand, so absent tile kinds are never treated as valid discard candidates.

Then start PPO from the discard-finetuned encoder instead:

```bash
python3 -u -m mini_training_ppo.train \
  --epochs 100 \
  --episodes-per-epoch 5 \
  --pretrained-encoder-path artifacts/pretrained_discard_encoder.npz \
  --log-interval 5 \
  --log-path artifacts/ppo_discard_finetuned.csv \
  --actor-save-path artifacts/ppo_actor_discard_finetuned.npz \
  --critic-save-path artifacts/ppo_critic_discard_finetuned.npz
```
