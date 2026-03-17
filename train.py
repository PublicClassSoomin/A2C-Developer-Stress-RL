"""
Developer Stress A2C 학습 메인 스크립트

1. kagglehub 또는 Mock으로 데이터 로드
2. Stress 보상 모델 학습
3. DevStressEnv 생성 후 A2C 에이전트 학습
4. 보상/스트레스 곡선 시각화
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import platform

# 수정: 한글 표시 — 폰트 이름이 ttflist와 안 맞으면 DejaVu Sans로 남아 경고 발생하므로, 경로로 폰트 지정
def _get_korean_font_prop():
    """한글 지원 폰트 파일 경로를 찾아 FontProperties 반환. 없으면 None."""
    if platform.system() == "Darwin":
        # macOS 기본 한글 폰트 경로 (버전별로 다를 수 있음, .ttc 컬렉션 포함)
        paths = [
            Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
            Path("/System/Library/Fonts/AppleGothic.ttf"),
            Path("/System/Library/Fonts/AppleGothic.ttc"),
            Path("/Library/Fonts/AppleGothic.ttf"),
            Path("/System/Library/Fonts/Apple SD Gothic Neo.ttf"),
            Path("/System/Library/Fonts/Apple SD Gothic Neo.ttc"),
            Path.home() / "Library/Fonts/NanumGothic.ttf",
        ]
    elif platform.system() == "Windows":
        paths = [
            Path("C:/Windows/Fonts/malgun.ttf"),
            Path("C:/Windows/Fonts/NanumGothic.ttf"),
        ]
    else:
        paths = [Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")]
    for p in paths:
        if p.exists():
            return matplotlib.font_manager.FontProperties(fname=str(p))
    return None

from sklearn.metrics import mean_squared_error, r2_score

from src.data_loader import DataLoader
from src.env import DevStressEnv
from src.reward_model import StressRewardModel
from src.a2c import A2CAgent

# 수정: Action Distribution 시각화용 레이블
ACTION_LABELS = ["집중근무", "휴식", "커피", "디버깅"]


# 기본 하이퍼파라미터
DEFAULT_N_STEPS = 5
DEFAULT_N_EPISODES = 500
DEFAULT_MAX_STEPS = 100
DEFAULT_LR = 3e-4
DEFAULT_GAMMA = 0.99
DEFAULT_HIDDEN = 128


def parse_args():
    parser = argparse.ArgumentParser(description="Developer Stress A2C 학습")
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS, help="업데이트당 롤아웃 스텝 수")
    parser.add_argument("--n-episodes", type=int, default=DEFAULT_N_EPISODES, help="학습 에피소드 수")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="에피소드당 최대 스텝")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="학습률")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="할인 인자")
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN, help="은닉층 크기")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--save-dir", type=str, default="results", help="그래프/로그 저장 디렉터리")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 데이터 로드 ----------
    loader = DataLoader()
    loader.load()
    if loader.used_mock:
        print("[DataLoader] Kaggle 다운로드 실패 → 통계 기반 Mock 데이터 사용")
    else:
        print("[DataLoader] developer_stress.csv 로드 완료")

    X, y = loader.get_feature_matrix_and_target()
    state_dim = X.shape[1]
    action_dim = 4

    # ---------- 2. 보상 모델 학습 ----------
    # reward_scale=0.1 → 스텝 보상 약 -1~0, 에피소드 누적 -100~0 수준으로 학습 안정
    reward_model = StressRewardModel(alpha=1.0, stress_scale=-1.0, reward_scale=0.1)
    reward_model.fit(X, y)
    print("[RewardModel] Stress_Level 회귀 모델 학습 완료")
    # 수정: 학습 데이터 기준 보상 모델 성능 출력 (R², RMSE)
    X_scaled = reward_model.scaler_x.transform(X)
    y_pred = reward_model.model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"[RewardModel] 학습 데이터 기준 R²={r2:.4f}, RMSE={rmse:.4f}")

    # ---------- 3. 환경 및 에이전트 ----------
    env = DevStressEnv(
        reward_model=reward_model,
        max_steps=args.max_steps,
        random_reset=True,
    )
    agent = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        hidden_dim=args.hidden,
        entropy_coef=0.05,  # 탐색 강화: 휴식/디버깅 등 스트레스 감소 행동 발견 유도
    )

    # ---------- 4. 학습 루프 (n-step A2C: 매 n_steps마다 중간 업데이트) ----------
    episode_rewards: list = []
    episode_stresses: list = []
    # 수정: 에피소드별 액션 선택 횟수 기록 → 마지막 100 에피소드 분포 시각화용
    episode_action_counts: list = []
    global_step = 0

    for episode in range(args.n_episodes):
        # 수정: 초반 절반은 시드 고정(과적합 완화), 후반은 seed=None으로 랜덤 리셋
        reset_seed = (args.seed + episode) if episode < args.n_episodes // 2 else None
        obs, _ = env.reset(seed=reset_seed)
        episode_reward = 0.0
        episode_stress_list: list = []
        # 수정: 에피소드 내 선택한 액션 기록 (분포 집계용)
        episode_actions: list = []

        # 롤아웃 버퍼 (n_steps마다 한 번씩 업데이트에 사용)
        states_buf: list = []
        actions_buf: list = []
        rewards_buf: list = []
        dones_buf: list = []

        for step in range(args.max_steps):
            action, log_prob, entropy, value = agent.select_action(obs, deterministic=False)
            agent.store_step_values(value)
            episode_actions.append(action)

            states_buf.append(obs)
            actions_buf.append(action)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_stress_list.append(info.get("stress", 0.0))

            rewards_buf.append(reward)
            dones_buf.append(float(done))

            obs = next_obs
            global_step += 1

            # 수정: n-step A2C — 버퍼가 정확히 n_steps일 때만 업데이트 (값 버퍼와 길이 일치 유지)
            if len(states_buf) == args.n_steps:
                agent.finish_rollout()
                n_use = args.n_steps
                states_arr = np.array(states_buf[:n_use])
                actions_arr = np.array(actions_buf[:n_use])
                rewards_arr = np.array(rewards_buf[:n_use])
                dones_arr = np.array(dones_buf[:n_use])
                agent.update(
                    states=states_arr,
                    actions=actions_arr,
                    rewards=rewards_arr,
                    dones=dones_arr,
                    next_states=obs,
                    next_done=done,
                )
                states_buf = states_buf[n_use:]
                actions_buf = actions_buf[n_use:]
                rewards_buf = rewards_buf[n_use:]
                dones_buf = dones_buf[n_use:]

            if done:
                break

        # 수정: 에피소드 종료 시 남은 스텝이 있으면 한 번 더 업데이트
        if len(states_buf) > 0:
            agent.finish_rollout()
            states_arr = np.array(states_buf)
            actions_arr = np.array(actions_buf)
            rewards_arr = np.array(rewards_buf)
            dones_arr = np.array(dones_buf)
            agent.update(
                states=states_arr,
                actions=actions_arr,
                rewards=rewards_arr,
                dones=dones_arr,
                next_states=obs,
                next_done=True,
            )

        # 수정: 에피소드별 액션 카운트 저장 (0~3 인덱스별 횟수)
        counts = np.bincount(episode_actions, minlength=action_dim)
        episode_action_counts.append(counts)

        mean_stress = np.mean(episode_stress_list) if episode_stress_list else 0.0
        episode_rewards.append(episode_reward)
        episode_stresses.append(mean_stress)

        if (episode + 1) % 50 == 0 or episode == 0:
            print(
                f"Episode {episode + 1}/{args.n_episodes} | "
                f"Return={episode_reward:.1f} | Mean Stress={mean_stress:.3f}"
            )

    # ---------- 5. 시각화 ----------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1 = axes[0]
    ax1.plot(episode_rewards, color="steelblue", alpha=0.8)
    ax1.set_ylabel("Episode Return")
    ax1.set_title("A2C Training — Episode Return")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(episode_stresses, color="coral", alpha=0.8)
    ax2.set_ylabel("Mean Stress (predicted)")
    ax2.set_xlabel("Episode")
    ax2.set_title("A2C Training — Mean Stress per Episode")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = save_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[시각화] 저장됨: {plot_path}")

    # 이동 평균 곡선 추가 저장 (선택)
    window = min(50, args.n_episodes // 5)
    if window >= 2:
        fig2, ax = plt.subplots(1, 1, figsize=(10, 4))
        smooth_r = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax.plot(smooth_r, color="steelblue", label=f"Return (MA-{window})")
        smooth_s = np.convolve(episode_stresses, np.ones(window) / window, mode="valid")
        ax2_twin = ax.twinx()
        ax2_twin.plot(smooth_s, color="coral", label=f"Stress (MA-{window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax2_twin.set_ylabel("Stress")
        ax.legend(loc="upper left")
        ax2_twin.legend(loc="upper right")
        ax.set_title("Smoothed Training Curves")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "training_curves_smooth.png", dpi=150)
        plt.close()
        print(f"[시각화] 저장됨: {save_dir / 'training_curves_smooth.png'}")

    # 수정: 마지막 100 에피소드 기준 Action Distribution 막대 그래프 저장 (한글 폰트 경로로 적용해 DejaVu Sans 경고 제거)
    korean_font = _get_korean_font_prop()
    last_n = min(100, len(episode_action_counts))
    if last_n > 0:
        last_counts = np.array(episode_action_counts[-last_n:])
        action_totals = last_counts.sum(axis=0)
        fig_dist, ax_dist = plt.subplots(1, 1, figsize=(8, 4))
        ax_dist.bar(ACTION_LABELS, action_totals, color=["#2ecc71", "#3498db", "#e67e22", "#9b59b6"], edgecolor="black", alpha=0.8)
        ax_dist.set_xticks(range(len(ACTION_LABELS)))
        if korean_font is not None:
            ax_dist.set_ylabel("선택 횟수", fontproperties=korean_font)
            ax_dist.set_title(f"Action Distribution (마지막 {last_n} 에피소드)", fontproperties=korean_font)
            ax_dist.set_xticklabels(ACTION_LABELS, fontproperties=korean_font)
        else:
            ax_dist.set_ylabel("선택 횟수")
            ax_dist.set_title(f"Action Distribution (마지막 {last_n} 에피소드)")
            ax_dist.set_xticklabels(ACTION_LABELS)
        matplotlib.rcParams["axes.unicode_minus"] = False
        plt.tight_layout()
        action_dist_path = save_dir / "action_dist.png"
        plt.savefig(action_dist_path, dpi=150)
        plt.close()
        print(f"[시각화] 저장됨: {action_dist_path}")

    print("학습 완료.")


if __name__ == "__main__":
    main()
