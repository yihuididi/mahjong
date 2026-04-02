from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import ACTION_COUNT


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 2000
    max_steps_per_episode: int = 2000
    replay_capacity: int = 50_000
    batch_size: int = 64
    gamma: float = 0.99
    learning_rate: float = 1e-3
    hidden_sizes: tuple[int, ...] = (256, 128)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 25_000
    warmup_steps: int = 1_000
    target_sync_interval: int = 500
    updates_per_step: int = 1
    log_interval: int = 50
    seed: int = 7


class ReplayBuffer:
    def __init__(self, capacity: int, observation_size: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.observation_size = observation_size
        self.buffer: deque[tuple[np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, bool]] = deque(
            maxlen=capacity
        )
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        next_action_mask: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            (
                observation.astype(np.float32, copy=True),
                action_mask.astype(np.float32, copy=True),
                int(action),
                float(reward),
                next_observation.astype(np.float32, copy=True),
                next_action_mask.astype(np.float32, copy=True),
                bool(done),
            )
        )

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[index] for index in indices]

        observations = np.stack([item[0] for item in batch], axis=0)
        action_masks = np.stack([item[1] for item in batch], axis=0)
        actions = np.asarray([item[2] for item in batch], dtype=np.int64)
        rewards = np.asarray([item[3] for item in batch], dtype=np.float32)
        next_observations = np.stack([item[4] for item in batch], axis=0)
        next_action_masks = np.stack([item[5] for item in batch], axis=0)
        dones = np.asarray([item[6] for item in batch], dtype=np.float32)
        return observations, action_masks, actions, rewards, next_observations, next_action_masks, dones


class MLPNetwork:
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

    def clone(self) -> "MLPNetwork":
        replica = MLPNetwork(self.input_dim, self.hidden_sizes, self.output_dim)
        replica.copy_from(self)
        return replica

    def copy_from(self, other: "MLPNetwork") -> None:
        self.weights = [weight.copy() for weight in other.weights]
        self.biases = [bias.copy() for bias in other.biases]

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        x = inputs.astype(np.float32, copy=False)
        if x.ndim == 1:
            x = x[None, :]
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight + bias
            if index < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
        return x.astype(np.float32)

    def train_selected_q(
        self,
        inputs: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
    ) -> float:
        x = inputs.astype(np.float32, copy=False)
        activations = [x]
        pre_activations: list[np.ndarray] = []
        hidden = x
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = hidden @ weight + bias
            pre_activations.append(z)
            hidden = np.maximum(z, 0.0)
            activations.append(hidden)

        q_values = hidden @ self.weights[-1] + self.biases[-1]
        activations.append(q_values)
        batch_indices = np.arange(q_values.shape[0])
        selected_q = q_values[batch_indices, actions]
        td_error = selected_q - targets
        loss = float(np.mean(td_error ** 2))

        grad_output = np.zeros_like(q_values)
        grad_output[batch_indices, actions] = (2.0 * td_error) / float(q_values.shape[0])

        grad = grad_output
        for layer_index in reversed(range(len(self.weights))):
            prev_activation = activations[layer_index]
            grad_w = prev_activation.T @ grad
            grad_b = np.sum(grad, axis=0)

            np.clip(grad_w, -5.0, 5.0, out=grad_w)
            np.clip(grad_b, -5.0, 5.0, out=grad_b)

            self.weights[layer_index] -= learning_rate * grad_w.astype(np.float32)
            self.biases[layer_index] -= learning_rate * grad_b.astype(np.float32)

            if layer_index == 0:
                continue

            grad = grad @ self.weights[layer_index].T
            grad = grad * (pre_activations[layer_index - 1] > 0.0)

        return loss

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
    def load(cls, path: str | Path) -> "MLPNetwork":
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


class DQNAgent:
    def __init__(
        self,
        observation_size: int,
        config: TrainingConfig,
    ) -> None:
        self.config = config
        self.online = MLPNetwork(
            input_dim=observation_size,
            hidden_sizes=config.hidden_sizes,
            output_dim=ACTION_COUNT,
            seed=config.seed,
        )
        self.target = self.online.clone()
        self.rng = np.random.default_rng(config.seed)
        self.epsilon = config.epsilon_start
        self.train_steps = 0

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, explore: bool = True) -> int:
        legal_actions = np.flatnonzero(action_mask > 0.0)
        if legal_actions.size == 0:
            raise RuntimeError("No legal actions are available.")

        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal_actions))

        q_values = self.online.predict(observation)[0]
        masked_q = np.where(action_mask > 0.0, q_values, -1e9)
        return int(np.argmax(masked_q))

    def update(self, replay_buffer: ReplayBuffer, batch_size: int) -> float | None:
        if len(replay_buffer) < batch_size:
            return None

        observations, _, actions, rewards, next_observations, next_action_masks, dones = replay_buffer.sample(batch_size)

        next_online_q = self.online.predict(next_observations)
        next_online_q = np.where(next_action_masks > 0.0, next_online_q, -1e9)
        best_next_actions = np.argmax(next_online_q, axis=1)

        next_target_q = self.target.predict(next_observations)
        next_state_values = next_target_q[np.arange(batch_size), best_next_actions]
        next_state_values = np.where(dones > 0.0, 0.0, next_state_values)

        targets = rewards + (1.0 - dones) * self.config.gamma * next_state_values
        loss = self.online.train_selected_q(
            observations,
            actions,
            targets.astype(np.float32),
            self.config.learning_rate,
        )

        self.train_steps += 1
        if self.train_steps % self.config.target_sync_interval == 0:
            self.target.copy_from(self.online)
        self._decay_epsilon()
        return loss

    def _decay_epsilon(self) -> None:
        progress = min(self.train_steps / float(max(self.config.epsilon_decay_steps, 1)), 1.0)
        self.epsilon = self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def save(self, path: str | Path) -> None:
        self.online.save(path)
