"""Paired-seed simulation comparison: PPO versus deterministic baseline."""
import argparse, json, statistics
from pathlib import Path
from stable_baselines3 import PPO
from .decision_env import DecisionOptimizationEnv, ACTION_NAMES
from ..config import MODEL_DIR

def baseline_action(obs):
    cpu, memory, disk, suspicious, workload = [float(x) for x in obs]
    if suspicious >= .55:
        return 3
    if cpu >= .72 or memory >= .78:
        return 2 if workload >= .35 else 1
    if disk >= .90:
        return 1
    return 0

def run_episode(policy, seed, horizon):
    env = DecisionOptimizationEnv(horizon=horizon)
    obs, _ = env.reset(seed=seed)
    total = 0.0; pressure_sum = 0.0; counts = {name: 0 for name in ACTION_NAMES}
    for _ in range(horizon):
        if policy == "baseline":
            action = baseline_action(obs[:5])
        else:
            action = int(
                policy.predict(obs, deterministic=True)[0])
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward); pressure_sum += float(info["pressure"]); counts[ACTION_NAMES[action]] += 1
        if terminated or truncated: break
    return {"return": total, "mean_pressure": pressure_sum / horizon, "action_counts": counts}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--horizon", type=int, default=80)
    p.add_argument("--model", default=str(MODEL_DIR / "rl" / "decision_policy_ppo.zip"))
    p.add_argument("--output", default=str(MODEL_DIR / "rl" / "policy_comparison.json"))
    a = p.parse_args()
    if a.episodes < 5 or a.horizon < 10: p.error("Use --episodes >= 5 and --horizon >= 10")
    model_path = Path(a.model)
    if not model_path.exists(): p.error(f"RL model missing: {model_path}; train it first")
    model = PPO.load(str(model_path))
    # Both policies receive the same episode seeds for a paired comparison.
    baseline = [run_episode("baseline", 1000+i, a.horizon) for i in range(a.episodes)]
    rl = [run_episode(model, 1000+i, a.horizon) for i in range(a.episodes)]
    def summarize(rows):
        return {
            "mean_return": statistics.mean(
                row["return"] for row in rows
            ),
            "stdev_return": statistics.pstdev(
                row["return"] for row in rows
            ),
            "mean_pressure": statistics.mean(
                row["mean_pressure"] for row in rows
            ),
            "action_counts": {
                action: sum(
                    row["action_counts"][action] for row in rows
                )
                for action in ACTION_NAMES
            },
            "episodes": len(rows),
        }

    baseline_summary = summarize(baseline)
    ppo_summary = summarize(rl)

    # Paired differences use the same seed for each baseline/PPO pair.
    paired_return_differences = [
        ppo_episode["return"] - baseline_episode["return"]
        for baseline_episode, ppo_episode in zip(baseline, rl)
    ]

    paired_pressure_differences = [
        ppo_episode["mean_pressure"] - baseline_episode["mean_pressure"]
        for baseline_episode, ppo_episode in zip(baseline, rl)
    ]

    def paired_summary(differences):
        mean_difference = statistics.mean(differences)

        if len(differences) > 1:
            standard_error = (
                statistics.stdev(differences) / len(differences) ** 0.5
            )
        else:
            standard_error = None

        # Approximate interval only; use a predeclared statistical method
        # for formal model promotion decisions.
        return {
            "mean_difference": mean_difference,
            "standard_error": standard_error,
            "approximate_95_percent_interval": (
                [
                    mean_difference - 1.96 * standard_error,
                    mean_difference + 1.96 * standard_error,
                ]
                if standard_error is not None
                else None
            ),
            "paired_episode_count": len(differences),
        }

    result = {
        "environment": "synthetic_only",
        "live_systems_touched": False,
        "seed_range": [
            1000,
            1000 + a.episodes - 1,
        ],
        "horizon": a.horizon,
        "baseline": baseline_summary,
        "ppo": ppo_summary,
        "paired_return_difference_ppo_minus_baseline":
            paired_summary(paired_return_differences),
        "paired_pressure_difference_ppo_minus_baseline":
            paired_summary(paired_pressure_differences),
        "interpretation": (
            "Synthetic simulation only. These results do not establish "
            "production effectiveness, causal impact, or operational safety. "
            "Do not automatically promote the policy."
        ),
    }

if __name__ == "__main__": main()
