import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch

from snake_dqn_88.model import DQN
import random as _random


class Agent:
    """DQN agent: epsilon-greedy, replay buffer, Bellman update."""

    def __init__(
        self,
        action_size: int = 4,
        gamma: float = 0.99,
        lr: float = 1e-4,
        epsilon_start: float = 0.01,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 1.0,
        memory_size: int = 100_000,
        batch_size: int = 256,
        target_update_freq: int = 2000,
        soft_tau: float | None = 0.01,
        device: torch.device | None = None,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_update_freq = target_update_freq
        self.soft_tau = soft_tau
        self.learn_step = 0
        self.epsilon_reset_every = None
        self.epsilon_reset_value = 0.01

        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=memory_size)

        self.model = DQN(output_size=action_size).to(self.device)
        self.target_model = DQN(output_size=action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.model.reset_noise()
        with torch.no_grad():
            q_values = self.model(state_t)
        return int(torch.argmax(q_values).item())

    def replay(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states = torch.tensor(np.array([b[0] for b in batch], dtype=np.float32), device=self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([b[3] for b in batch], dtype=np.float32), device=self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.bool, device=self.device)

        self.model.reset_noise()
        self.target_model.reset_noise()
        q_pred = self.model(states)
        current_q = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: action from main model, value from target
            next_actions = self.model(next_states).argmax(1)
            next_q_target = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q_target * (~dones)

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self._maybe_update_target()
        return float(loss.item())

    def update_epsilon(self) -> None:
        # NoisyNet handles exploration; epsilon is kept constant at minimum.
        self.epsilon = self.epsilon_min

    def _maybe_update_target(self) -> None:
        self.learn_step += 1
        if self.soft_tau is not None:
            tau = self.soft_tau
            with torch.no_grad():
                for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        elif self.learn_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save_checkpoint(self, path: str, meta: dict | None = None) -> None:
        payload = {
            "format": "full_dqn_checkpoint_v3",
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "learn_step": self.learn_step,
            "memory": list(self.memory),
            "rng": {
                "python": _random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "meta": meta or {},
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> dict:
        payload = torch.load(path, map_location=self.device)
        if isinstance(payload, dict) and "model" not in payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
            self.model.load_state_dict(payload)
            self.target_model.load_state_dict(self.model.state_dict())
            return {}

        self.model.load_state_dict(payload["model"])
        self.target_model.load_state_dict(payload.get("target_model", self.model.state_dict()))
        if "optimizer" in payload:
            try:
                self.optimizer.load_state_dict(payload["optimizer"])
            except Exception:
                pass
        self.epsilon = float(payload.get("epsilon", self.epsilon))
        self.learn_step = int(payload.get("learn_step", self.learn_step))

        mem = payload.get("memory")
        if mem is not None:
            self.memory.clear()
            self.memory.extend(mem)

        rng = payload.get("rng") or {}
        try:
            if rng.get("python") is not None:
                _random.setstate(rng["python"])
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
        except Exception:
            pass

        return payload.get("meta", {}) or {}
