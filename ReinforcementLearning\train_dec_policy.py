"""Train a PPO policy in synthetic simulation; never deploys it automatically."""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from .decision_env import DecisionOptimizationEnv
from ..config import MODEL_DIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.timesteps < 1000:
        parser.error("--timesteps must be >= 1000")
    out = MODEL_DIR / "rl"
    out.mkdir(parents=True, exist_ok=True)
    model = PPO("MlpPolicy", DecisionOptimizationEnv(), seed=args.seed, verbose=1,
                n_steps=512, batch_size=64, learning_rate=3e-4, gamma=.97)
    model.learn(total_timesteps=args.timesteps)
    target = out / "decision_policy_ppo"
    model.save(str(target))
    print(f"Saved simulation-only policy: {target}.zip")
    print("Not promoted, not enabled by the API, and not authorized for live execution.")

if __name__ == "__main__":
    main()
