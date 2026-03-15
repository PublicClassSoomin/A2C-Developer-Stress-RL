"""
Developer Stress Gym 환경

상태: [Hours_Worked, Sleep_Hours, Bugs, Deadline_Days, Coffee_Cups, Meetings, Interruptions]
행동: 0=집중근무, 1=휴식, 2=커피, 3=디버깅
보상: StressRewardModel로 예측한 스트레스 기반 (낮을수록 높은 보상)
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .reward_model import StressRewardModel

# gymnasium이 있으면 사용, 없으면 gym
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


# 상태 각 차원의 합리적 범위 (정규화 및 클리핑용)
STATE_BOUNDS = {
    "Hours_Worked": (0.0, 16.0),
    "Sleep_Hours": (0.0, 12.0),
    "Bugs": (0.0, 20.0),
    "Deadline_Days": (0.0, 30.0),
    "Coffee_Cups": (0.0, 10.0),
    "Meetings": (0.0, 15.0),
    "Interruptions": (0.0, 15.0),
}
STATE_ORDER = [
    "Hours_Worked",
    "Sleep_Hours",
    "Bugs",
    "Deadline_Days",
    "Coffee_Cups",
    "Meetings",
    "Interruptions",
]


def get_bounds_array() -> Tuple[np.ndarray, np.ndarray]:
    """STATE_ORDER에 맞는 low, high 벡터 반환."""
    low = np.array([STATE_BOUNDS[k][0] for k in STATE_ORDER], dtype=np.float32)
    high = np.array([STATE_BOUNDS[k][1] for k in STATE_ORDER], dtype=np.float32)
    return low, high


class DevStressEnv(gym.Env):
    """
    개발자 스트레스 시뮬레이션 환경.

    - 상태: 7차원 연속 벡터 (업무시간, 수면, 버그 수, 마감일, 커피, 회의, 방해)
    - 행동: 이산 4가지 (집중근무, 휴식, 커피, 디버깅)
    - 보상: 내부 회귀 모델로 예측한 스트레스 기반 (스트레스 감소 시 높은 보상)
    """

    metadata = {"render_modes": []}

    # 행동 인덱스
    ACTION_FOCUS = 0
    ACTION_REST = 1
    ACTION_COFFEE = 2
    ACTION_DEBUG = 3

    def __init__(
        self,
        reward_model: StressRewardModel,
        initial_state: Optional[np.ndarray] = None,
        max_steps: int = 100,
        random_reset: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            reward_model: 스트레스 예측 및 보상 계산에 사용할 모델
            initial_state: 고정 초기 상태. None이면 random_reset에 따라 설정
            max_steps: 에피소드 최대 스텝
            random_reset: True면 리셋 시 구간 내 랜덤 상태, False면 중앙값
            **kwargs: gym.Env 상위 클래스 인자 (예: render_mode)
        """
        super().__init__(**kwargs)
        self.reward_model = reward_model
        self.initial_state = initial_state
        self.max_steps = max_steps
        self.random_reset = random_reset

        low, high = get_bounds_array()
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(len(STATE_ORDER),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        self._state: np.ndarray = np.zeros(len(STATE_ORDER), dtype=np.float32)
        self._step_count: int = 0

    def _clip_state(self, state: np.ndarray) -> np.ndarray:
        """상태를 정의된 범위로 클리핑."""
        low, high = get_bounds_array()
        return np.clip(state, low, high).astype(np.float32)

    def _transition(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        (state, action) -> next_state 동역학.

        - 0 집중근무: Bugs 증가, Hours_Worked 증가
        - 1 휴식: Sleep_Hours 증가, Hours_Worked 소폭 감소
        - 2 커피: Coffee_Cups 증가, Hours_Worked 약간 증가
        - 3 디버깅: Bugs 감소, Hours_Worked 약간 증가
        """
        next_state = state.copy()
        idx = {k: i for i, k in enumerate(STATE_ORDER)}

        # 스트레스 감소 행동(휴식/디버깅) 효과 강화 → 보상 신호 명확화
        if action == self.ACTION_FOCUS:
            next_state[idx["Hours_Worked"]] += 0.8
            next_state[idx["Bugs"]] += 0.5
        elif action == self.ACTION_REST:
            next_state[idx["Sleep_Hours"]] += 0.9
            next_state[idx["Hours_Worked"]] = max(0, next_state[idx["Hours_Worked"]] - 0.5)
        elif action == self.ACTION_COFFEE:
            next_state[idx["Coffee_Cups"]] += 0.5
            next_state[idx["Hours_Worked"]] += 0.2
        elif action == self.ACTION_DEBUG:
            next_state[idx["Bugs"]] = max(0, next_state[idx["Bugs"]] - 1.2)
            next_state[idx["Hours_Worked"]] += 0.2

        return self._clip_state(next_state)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """환경 초기화. 초기 상태와 info 반환."""
        if seed is not None:
            np.random.seed(seed)
        super().reset(seed=seed)

        if self.initial_state is not None:
            self._state = self._clip_state(self.initial_state.copy())
        else:
            low, high = get_bounds_array()
            if self.random_reset:
                self._state = np.random.uniform(low, high).astype(np.float32)
            else:
                self._state = ((low + high) / 2).astype(np.float32)

        self._step_count = 0
        info = {}
        return self._state.copy(), info

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        한 스텝 진행. Gymnasium API: observation, reward, terminated, truncated, info.
        """
        self._step_count += 1
        self._state = self._transition(self._state, action)

        reward = self.reward_model.compute_reward(self._state)
        terminated = False
        truncated = self._step_count >= self.max_steps
        info = {
            "stress": self.reward_model.predict_stress(self._state),
            "step": self._step_count,
        }

        return self._state.copy(), reward, terminated, truncated, info
