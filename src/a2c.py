"""
A2C (Advantage Actor-Critic) 에이전트

PyTorch 기반 Actor(정책) 네트워크와 Critic(가치) 네트워크를 구현하고,
Advantage를 사용한 정책 그래디언트와 TD 오차 기반 Critic 업데이트를 수행합니다.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """가중치 초기화 (정책/가치 학습 안정성)."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain=std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticNetwork(nn.Module):
    """
    Actor와 Critic이 공유 특징 추출기를 사용하는 네트워크.

    - 공통 MLP 백본으로 상태 임베딩
    - Actor 헤드: 정책 π(a|s) (이산 행동에 대해 softmax)
    - Critic 헤드: 상태 가치 V(s)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        shared_layers: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 공유 레이어
        layers: List[nn.Module] = []
        in_dim = state_dim
        for _ in range(shared_layers):
            layers.append(layer_init(nn.Linear(in_dim, hidden_dim)))
            in_dim = hidden_dim
        self.shared = nn.Sequential(*layers)

        # Actor 헤드: 정책 로짓
        self.actor_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

        # Critic 헤드: V(s)
        self.critic_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        state -> (action_logits, value)

        Args:
            state: (batch, state_dim)

        Returns:
            logits: (batch, action_dim), value: (batch, 1)
        """
        feat = F.relu(self.shared(state))
        logits = self.actor_head(feat)
        value = self.critic_head(feat)
        return logits, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        행동 샘플링(또는 주어진 행동)과 로그 확률, 엔트로피, V(s) 반환.

        Returns:
            action: (batch,), log_prob: (batch,), entropy: (batch,), value: (batch, 1)
        """
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


class A2CAgent:
    """
    A2C 에이전트: 여러 스텝을 모은 뒤 Advantage로 Actor/Critic 업데이트.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 128,
        device: Optional[str] = None,
    ):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 행동 개수
            lr: 학습률
            gamma: 할인 인자
            gae_lambda: GAE(Generalized Advantage Estimation) 람다
            value_coef: Critic 손실 계수
            entropy_coef: 엔트로피 보너스 계수
            max_grad_norm: 그래디언트 클리핑
            hidden_dim: 은닉층 크기
            device: 'cuda' or 'cpu'
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.net = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.optimizer = Adam(self.net.parameters(), lr=lr)

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        상태에서 행동을 샘플링하고, 학습에 쓸 로그확률/엔트로피/가치 반환.

        Returns:
            action (int), log_prob, entropy, value (텐서)
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, entropy, value = self.net.get_action_and_value(
                state_t, deterministic=deterministic
            )
        # value: (1, 1) -> 스칼라로 만들어 stack 시 (n_steps,) 유지
        return int(action.item()), log_prob.squeeze(0), entropy.squeeze(0), value.squeeze()

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_states: np.ndarray,
        next_done: bool,
    ) -> Tuple[float, float, float]:
        """
        한 번의 A2C 업데이트. GAE로 advantage 계산 후 Actor/Critic 손실로 backward.

        Returns:
            policy_loss, value_loss, entropy (평균 스칼라)
        """
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, _, _, next_value = self.net.get_action_and_value(
                torch.as_tensor(next_states, dtype=torch.float32, device=self.device).unsqueeze(0)
            )
            next_value = next_value.squeeze()
            if next_done:
                next_value = 0.0

            # GAE
            advantages = np.zeros(len(rewards), dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_val = next_value.item() if hasattr(next_value, "item") else float(next_value)
                else:
                    _, _, _, v = self.net.get_action_and_value(
                        states_t[t : t + 1]
                    )
                    next_val = v.squeeze().item()
                delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - self._last_values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae * (1 - dones[t])
                advantages[t] = last_gae
            advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
            returns_t = advantages_t + self._last_values_t
            # 정책 그래디언트용으로만 advantage 정규화 (value 타깃은 원본 returns 사용)
            advantages_normalized = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        _, log_prob, entropy, values = self.net.get_action_and_value(states_t, actions_t)
        values = values.squeeze(-1)

        # 정책 손실: -log_prob * advantage (정규화된 advantage로 학습 안정성 확보)
        policy_loss = -(log_prob * advantages_normalized).mean()
        value_loss = F.mse_loss(values, returns_t)
        entropy_loss = -entropy.mean()

        loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), -entropy_loss.item()

    def store_step_values(self, value: torch.Tensor) -> None:
        """한 스텝의 V(s)를 저장 (update에서 GAE 계산 시 사용)."""
        if not hasattr(self, "_rollout_values"):
            self._rollout_values = []
            self._rollout_values_t = []
        self._rollout_values.append(value.item())
        self._rollout_values_t.append(value)

    def finish_rollout(self, last_values: Optional[torch.Tensor] = None) -> None:
        """롤아웃 버퍼를 업데이트용으로 준비. last_values는 (n_steps,) 또는 스칼라 텐서."""
        self._last_values = getattr(self, "_rollout_values", [])
        stacked = torch.stack(getattr(self, "_rollout_values_t", []))
        # (n_steps,)로 맞춤. squeeze(0)으로 저장된 값이 (1,)일 경우 (n_steps, 1) -> (n_steps,)
        self._last_values_t = stacked.squeeze(-1) if stacked.dim() > 1 else stacked
        self._rollout_values = []
        self._rollout_values_t = []


# 업데이트 시 "last_values"는 rollout 중 저장한 value 텐서들; next_value는 별도 계산.
# 코드에서 _last_values를 리스트로 썼는데, update() 안에서는 self._last_values_t (텐서)를 써야 함.
# 수정: rollout 시 매 스텝 value를 리스트에 넣고, finish_rollout에서 텐서로 만든 뒤
# update()에서 advantages 계산 시 현재 네트워크로 다시 value를 계산하지 말고, 저장된 value를 써야 함.
# GAE 공식: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t), A_t = delta_t + gamma*lambda*A_{t+1}
# 그래서 rollout 시 저장한 V(s_t)를 쓰면 됨. _last_values_t가 (n_steps,) 텐서가 되도록 하자.