"""
스트레스 보상 모델

데이터셋의 Stress_Level을 예측하는 간단한 회귀 모델을 학습하고,
에이전트의 상태에 대해 예측 스트레스를 반환합니다.
보상 함수: 스트레스가 낮을수록 높은 보상 (예: reward = -stress 또는 scale 적용)
"""

from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class StressRewardModel:
    """
    Stress_Level 회귀 모델. 상태 벡터를 입력받아 예측 스트레스와 보상을 계산한다.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        stress_scale: float = -1.0,
        reward_scale: float = 0.1,
    ):
        """
        Args:
            alpha: Ridge 회귀 정규화 강도
            stress_scale: 예측 스트레스를 보상에 반영하는 계수 (음수면 낮은 스트레스 = 높은 보상)
            reward_scale: 최종 보상 스케일. 0.1 권장 → 스텝 보상 약 -1~0, 에피소드 보상 -100~0 수준
        """
        self.alpha = alpha
        self.stress_scale = stress_scale
        self.reward_scale = reward_scale
        self.scaler_x = StandardScaler()
        self.model = Ridge(alpha=alpha, random_state=42)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StressRewardModel":
        """
        피처 X와 타깃(Stress_Level) y로 회귀 모델을 학습한다.

        Args:
            X: (n_samples, n_features)
            y: (n_samples,) Stress_Level

        Returns:
            self (체이닝용)
        """
        X_scaled = self.scaler_x.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._fitted = True
        return self

    def predict_stress(self, state: np.ndarray) -> float:
        """
        단일 상태 벡터에 대한 스트레스 수준을 예측한다.

        Args:
            state: (n_features,) 또는 (1, n_features)

        Returns:
            예측 스트레스 (스칼라)
        """
        if not self._fitted:
            return 0.0
        state = np.atleast_2d(state)
        state_scaled = self.scaler_x.transform(state)
        return float(self.model.predict(state_scaled)[0])

    def compute_reward(
        self,
        state: np.ndarray,
        clip_stress: Optional[Tuple[float, float]] = (0.0, 10.0),
    ) -> float:
        """
        상태에 대한 보상을 계산한다. 스트레스가 낮을수록 보상이 높다.

        reward = reward_scale * stress_scale * clip(predicted_stress)
        reward_scale=0.1 이면 스텝 보상 약 -1~0, 100스텝 누적 시 -100~0 수준으로 학습 안정.

        Args:
            state: 현재 상태 벡터
            clip_stress: 예측 스트레스 클리핑 범위

        Returns:
            보상 (스칼라)
        """
        stress = self.predict_stress(state)
        if clip_stress is not None:
            stress = np.clip(stress, clip_stress[0], clip_stress[1])
        reward = self.reward_scale * self.stress_scale * stress
        return float(reward)
