
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import (
    check_env
)

from .rl_environment import AsterionOptimizationEnv


def train(total_timesteps=250_000):
    environment = AsterionOptimizationEnv(
        horizon=300
    )

    check_env(environment)

    model = PPO(
        "MlpPolicy",
        environment,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        seed=42
    )

    model.learn(
        total_timesteps=total_timesteps
    )

    output = (
        Path("artifacts/rl")
        / "decision_policy_ppo_candidate.zip"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save(output)

    print("Candidate model saved:", output)

    return output


if __name__ == "__main__":
    train()
