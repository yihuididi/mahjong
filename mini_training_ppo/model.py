from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from mahjong_rl.constants import ACTION_COUNT


def masked_softmax(logits: np.ndarray, action_masks: np.ndarray) -> np.ndarray:
    masks = action_masks.astype(np.float32, copy=False)
    clipped_logits = np.where(masks > 0.0, logits, -1e9)
    max_logits = np.max(clipped_logits, axis=1, keepdims=True)
    exp_logits = np.exp(clipped_logits - max_logits) * masks
    denom = np.sum(exp_logits, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return (exp_logits / denom).astype(np.float32)


def normalize_advantages(advantages: np.ndarray) -> np.ndarray:
    if advantages.size == 0:
        return advantages
    mean = float(np.mean(advantages))
    std = float(np.std(advantages))
    return ((advantages - mean) / max(std, 1e-6)).astype(np.float32)


@dataclass(frozen=True)
class PPOConfig:
    epochs: int = 30
    episodes_per_epoch: int = 5
    max_steps_per_episode: int = 300
    gamma: float = 0.99
    gae_lambda: float = 0.95
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 6e-4
    hidden_sizes: tuple[int, ...] = (64, 32)
    clip_ratio: float = 0.2
    update_epochs: int = 3
    minibatch_size: int = 64
    value_coef: float = 0.5
    progress_reward_scale: float = 0.03
    pretrained_encoder_path: str | None = None
    seed: int = 13
    log_interval: int = 1


class MLP:
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Iterable[int],
        output_dim: int,
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.output_dim = output_dim
        self.rng = np.random.default_rng(seed)
        layer_sizes = (input_dim,) + self.hidden_sizes + (output_dim,)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(6.0 / float(fan_in + fan_out))
            self.weights.append(
                self.rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)
            )
            self.biases.append(np.zeros((fan_out,), dtype=np.float32))

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        x = inputs.astype(np.float32, copy=False)
        if x.ndim == 1:
            x = x[None, :]
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight + bias
            if index < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
        return x.astype(np.float32)

    def train_with_output_gradient(
        self,
        inputs: np.ndarray,
        grad_output: np.ndarray,
        learning_rate: float,
    ) -> None:
        x = inputs.astype(np.float32, copy=False)
        if x.ndim == 1:
            x = x[None, :]
        grad = grad_output.astype(np.float32, copy=False)

        activations = [x]
        pre_activations: list[np.ndarray] = []
        hidden = x
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = hidden @ weight + bias
            pre_activations.append(z)
            hidden = np.maximum(z, 0.0)
            activations.append(hidden)
        _ = hidden @ self.weights[-1] + self.biases[-1]
        activations.append(_)

        saved_weights = [weight.copy() for weight in self.weights]
        batch_scale = float(max(x.shape[0], 1))
        for layer_index in reversed(range(len(self.weights))):
            prev_activation = activations[layer_index]
            grad_w = (prev_activation.T @ grad) / batch_scale
            grad_b = np.sum(grad, axis=0) / batch_scale
            np.clip(grad_w, -5.0, 5.0, out=grad_w)
            np.clip(grad_b, -5.0, 5.0, out=grad_b)
            self.weights[layer_index] -= learning_rate * grad_w.astype(np.float32)
            self.biases[layer_index] -= learning_rate * grad_b.astype(np.float32)
            if layer_index == 0:
                continue
            grad = grad @ saved_weights[layer_index].T
            grad = grad * (pre_activations[layer_index - 1] > 0.0)

    def save(self, path: str | Path) -> None:
        payload: dict[str, np.ndarray] = {
            "input_dim": np.asarray([self.input_dim], dtype=np.int32),
            "hidden_sizes": np.asarray(self.hidden_sizes, dtype=np.int32),
            "output_dim": np.asarray([self.output_dim], dtype=np.int32),
        }
        for index, weight in enumerate(self.weights):
            payload[f"weight_{index}"] = weight
            payload[f"bias_{index}"] = self.biases[index]
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "MLP":
        data = np.load(path)
        input_dim = int(data["input_dim"][0])
        hidden_sizes = tuple(int(value) for value in data["hidden_sizes"])
        output_dim = int(data["output_dim"][0])
        network = cls(input_dim, hidden_sizes, output_dim)
        network.weights = []
        network.biases = []
        layer_count = len(hidden_sizes) + 1
        for index in range(layer_count):
            network.weights.append(data[f"weight_{index}"].astype(np.float32))
            network.biases.append(data[f"bias_{index}"].astype(np.float32))
        return network

    def copy_hidden_from(self, other: "MLP") -> None:
        if self.input_dim != other.input_dim or self.hidden_sizes != other.hidden_sizes:
            raise ValueError("Cannot copy hidden layers between incompatible MLP shapes.")
        hidden_layer_count = len(self.hidden_sizes)
        for index in range(hidden_layer_count):
            self.weights[index] = other.weights[index].copy()
            self.biases[index] = other.biases[index].copy()


class PPOAgent:
    def __init__(self, observation_size: int, config: PPOConfig) -> None:
        self.config = config
        encoder_source: MLP | None = None
        hidden_sizes = config.hidden_sizes
        if config.pretrained_encoder_path:
            encoder_source = MLP.load(config.pretrained_encoder_path)
            if encoder_source.input_dim != observation_size:
                raise ValueError(
                    "Pretrained encoder input size does not match PPO observation size."
                )
            hidden_sizes = encoder_source.hidden_sizes

        self.actor = MLP(observation_size, hidden_sizes, ACTION_COUNT, seed=config.seed)
        self.critic = MLP(observation_size, hidden_sizes, 1, seed=config.seed + 1)
        if encoder_source is not None:
            self.actor.copy_hidden_from(encoder_source)
            self.critic.copy_hidden_from(encoder_source)
        self.rng = np.random.default_rng(config.seed)

    def policy(self, observations: np.ndarray, action_masks: np.ndarray) -> np.ndarray:
        logits = self.actor.predict(observations)
        return masked_softmax(logits, action_masks)

    def values(self, observations: np.ndarray) -> np.ndarray:
        values = self.critic.predict(observations)
        return values.reshape(-1).astype(np.float32)

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> tuple[int, float, float]:
        probs = self.policy(observation[None, :], action_mask[None, :])[0]
        legal_actions = np.flatnonzero(action_mask > 0.0)
        if legal_actions.size == 0:
            raise RuntimeError("No legal actions are available.")
        action = int(self.rng.choice(np.arange(ACTION_COUNT), p=probs))
        log_prob = float(np.log(max(probs[action], 1e-8)))
        value = float(self.values(observation[None, :])[0])
        return action, log_prob, value

    def update(
        self,
        observations: np.ndarray,
        action_masks: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
    ) -> tuple[float, float, float]:
        batch_size = observations.shape[0]
        indices = np.arange(batch_size)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []

        normalized_advantages = normalize_advantages(advantages)

        for _ in range(self.config.update_epochs):
            self.rng.shuffle(indices)
            for start in range(0, batch_size, self.config.minibatch_size):
                batch_idx = indices[start : start + self.config.minibatch_size]
                obs_batch = observations[batch_idx]
                mask_batch = action_masks[batch_idx]
                action_batch = actions[batch_idx]
                old_logp_batch = old_log_probs[batch_idx]
                return_batch = returns[batch_idx]
                advantage_batch = normalized_advantages[batch_idx]

                probs = self.policy(obs_batch, mask_batch)
                batch_indices = np.arange(obs_batch.shape[0])
                new_logp = np.log(np.maximum(probs[batch_indices, action_batch], 1e-8))
                ratio = np.exp(new_logp - old_logp_batch)

                unclipped = np.logical_or(
                    np.logical_and(advantage_batch >= 0.0, ratio <= (1.0 + self.config.clip_ratio)),
                    np.logical_and(advantage_batch < 0.0, ratio >= (1.0 - self.config.clip_ratio)),
                )
                sample_objective = np.minimum(
                    ratio * advantage_batch,
                    np.clip(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantage_batch,
                )
                policy_losses.append(float(-np.mean(sample_objective)))

                coeff = np.zeros_like(advantage_batch, dtype=np.float32)
                coeff[unclipped] = -(ratio[unclipped] * advantage_batch[unclipped]).astype(np.float32)
                grad_logits = coeff[:, None] * probs
                grad_logits[batch_indices, action_batch] -= coeff
                self.actor.train_with_output_gradient(obs_batch, grad_logits, self.config.actor_learning_rate)

                values = self.values(obs_batch)
                value_error = values - return_batch
                value_losses.append(float(np.mean(value_error ** 2)))
                grad_values = (2.0 * value_error / float(max(obs_batch.shape[0], 1)))[:, None].astype(np.float32)
                self.critic.train_with_output_gradient(
                    obs_batch,
                    grad_values * self.config.value_coef,
                    self.config.critic_learning_rate,
                )

                entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-8)), axis=1)
                entropies.append(float(np.mean(entropy)))

        return (
            float(np.mean(policy_losses)) if policy_losses else 0.0,
            float(np.mean(value_losses)) if value_losses else 0.0,
            float(np.mean(entropies)) if entropies else 0.0,
        )

    def save(self, actor_path: str | Path, critic_path: str | Path) -> None:
        self.actor.save(actor_path)
        self.critic.save(critic_path)
