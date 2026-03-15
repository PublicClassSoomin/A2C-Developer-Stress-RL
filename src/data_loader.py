"""
데이터 로더 모듈

Kaggle 'Developer Stress Simulation Dataset'을 kagglehub로 자동 다운로드 후
developer_stress.csv를 로드합니다. Kaggle API 오류 시 통계 기반 Mock 데이터를 생성합니다.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# 환경에서 기대하는 피처 이름 (State 벡터 순서)
FEATURE_NAMES = [
    "Hours_Worked",
    "Sleep_Hours",
    "Bugs",
    "Deadline_Days",
    "Coffee_Cups",
    "Meetings",
    "Interruptions",
]
TARGET_NAME = "Stress_Level"

# CSV에서 사용될 수 있는 컬럼 이름 별칭 (공백/언더스코어 등)
FEATURE_ALIASES = {
    "Hours_Worked": ["Hours_Worked", "Hours Worked", "hours_worked"],
    "Sleep_Hours": ["Sleep_Hours", "Sleep Hours", "sleep_hours"],
    "Bugs": ["Bugs", "bugs"],
    "Deadline_Days": ["Deadline_Days", "Deadline Days", "deadline_days"],
    "Coffee_Cups": ["Coffee_Cups", "Coffee Cups", "coffee_cups"],
    "Meetings": ["Meetings", "meetings"],
    "Interruptions": ["Interruptions", "interruptions"],
}
TARGET_ALIASES = ["Stress_Level", "Stress Level", "stress_level"]


class DataLoader:
    """
    Developer Stress 데이터셋 로더.

    - kagglehub로 데이터셋 다운로드 후 developer_stress.csv 로드
    - 실패 시 통계적 특성(평균, 표준편차)을 이용한 Mock 데이터 생성
    """

    KAGGLE_OWNER = "mabubakrsiddiq"
    KAGGLE_DATASET = "developer-stress-simulation-dataset"
    CSV_FILENAME = "developer_stress.csv"

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Args:
            cache_dir: 다운로드/캐시 디렉터리. None이면 kagglehub 기본 경로 사용.
        """
        self.cache_dir = cache_dir
        self._df: Optional[pd.DataFrame] = None
        self._used_mock = False

    def load(self) -> pd.DataFrame:
        """
        데이터셋을 로드한다. Kaggle 실패 시 Mock 데이터를 반환한다.

        Returns:
            FEATURE_NAMES + TARGET_NAME 컬럼을 가진 DataFrame (표준화된 컬럼명)
        """
        self._df = self._try_kaggle_load()
        if self._df is not None:
            self._df = self._normalize_columns(self._df)
            return self._df

        # Kaggle 실패 시 Mock 데이터 생성
        self._used_mock = True
        self._df = self._generate_mock_data()
        return self._df

    def _try_kaggle_load(self) -> Optional[pd.DataFrame]:
        """kagglehub로 데이터셋 다운로드 후 CSV 로드. 실패 시 None 반환."""
        try:
            import kagglehub

            # 데이터셋 다운로드 (최신 버전)
            path = kagglehub.dataset_download(
                f"{self.KAGGLE_OWNER}/{self.KAGGLE_DATASET}"
            )
            # path는 디렉터리; 그 안에서 developer_stress.csv 탐색
            csv_path = Path(path) / self.CSV_FILENAME
            if not csv_path.exists():
                # 하위 디렉터리에서 찾기
                for p in Path(path).rglob(self.CSV_FILENAME):
                    csv_path = p
                    break
                if not csv_path.exists():
                    for p in Path(path).rglob("*.csv"):
                        csv_path = p
                        break
            if not csv_path.exists():
                return None
            return pd.read_csv(csv_path)
        except Exception:
            return None

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame 컬럼명을 FEATURE_NAMES, TARGET_NAME에 맞게 정규화."""
        mapping = {}
        for std_name, aliases in FEATURE_ALIASES.items():
            for alias in aliases:
                if alias in df.columns:
                    mapping[alias] = std_name
                    break
        for alias in TARGET_ALIASES:
            if alias in df.columns:
                mapping[alias] = TARGET_NAME
                break
        df = df.rename(columns=mapping)
        # 필요한 컬럼만 선택 (있으면); 없으면 유지
        wanted = [c for c in FEATURE_NAMES + [TARGET_NAME] if c in df.columns]
        if wanted:
            df = df[wanted].copy()
        return df

    def _generate_mock_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """
        데이터셋의 통계적 특성을 반영한 Mock 데이터 생성.

        개발자 스트레스 시뮬레이션에 맞는 합리적인 평균/편차를 사용한다.
        """
        np.random.seed(42)
        # 합리적인 평균과 표준편차 (피처별)
        stats = {
            "Hours_Worked": (8.5, 2.5),
            "Sleep_Hours": (6.5, 1.2),
            "Bugs": (5.0, 3.0),
            "Deadline_Days": (7.0, 5.0),
            "Coffee_Cups": (3.0, 2.0),
            "Meetings": (4.0, 2.5),
            "Interruptions": (6.0, 3.0),
        }
        data = {}
        for name, (mu, sigma) in stats.items():
            values = np.random.normal(mu, sigma, n_samples)
            # 비음수/상한 적용
            if name in ("Hours_Worked", "Sleep_Hours", "Meetings", "Interruptions", "Coffee_Cups"):
                values = np.clip(values, 0, 24 if "Hours" in name or name == "Sleep_Hours" else 20)
            elif name == "Bugs":
                values = np.clip(values, 0, 20)
            elif name == "Deadline_Days":
                values = np.clip(values, 0, 30)
            data[name] = values

        # Stress_Level: 피처들의 선형 조합 + 노이즈 (높은 업무량/버그/미팅 -> 높은 스트레스)
        stress = (
            0.08 * data["Hours_Worked"]
            + (-0.15) * data["Sleep_Hours"]
            + 0.25 * data["Bugs"]
            + 0.12 * data["Deadline_Days"]
            + 0.05 * data["Coffee_Cups"]
            + 0.18 * data["Meetings"]
            + 0.15 * data["Interruptions"]
            + np.random.normal(2.0, 0.5, n_samples)
        )
        data[TARGET_NAME] = np.clip(stress, 0, 10)

        return pd.DataFrame(data)

    def get_feature_matrix_and_target(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        로드된 데이터에서 피처 행렬 X와 타깃 y를 반환한다.

        Returns:
            X: (n_samples, n_features), y: (n_samples,)
        """
        if self._df is None:
            self.load()
        df = self._df
        feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
        if not feature_cols or TARGET_NAME not in df.columns:
            raise ValueError(
                "데이터에 필요한 컬럼이 없습니다. "
                f"필요: {FEATURE_NAMES}, {TARGET_NAME}"
            )
        X = df[feature_cols].values.astype(np.float32)
        y = df[TARGET_NAME].values.astype(np.float32)
        return X, y

    @property
    def used_mock(self) -> bool:
        """Mock 데이터 사용 여부."""
        return self._used_mock

    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        """로드된 DataFrame (load 호출 후)."""
        return self._df
