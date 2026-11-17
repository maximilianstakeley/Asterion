
import math
from statistics import mean
from typing import Any


def evaluate_policy(
    episodes: list[dict[str, Any]]
) -> dict[str, Any]:

    if not episodes:
        raise ValueError("At least one episode is required")

    rewards = []
    risks = []

    invalid_actions = 0
    blocked_actions = 0
    total_actions = 0

    def count(episode, key):
        value = episode.get(key, 0)

        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid {key}")

        return value

    for episode in episodes:

        reward = episode.get("reward")
        risk = episode.get("cumulative_risk")

        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(reward)
        ):
            raise ValueError("Invalid reward")

        if (
            isinstance(risk, bool)
            or not isinstance(risk, (int, float))
            or not math.isfinite(risk)
            or risk < 0
        ):
            raise ValueError("Invalid cumulative risk")

        invalid = count(episode, "invalid_actions")
        blocked = count(episode, "blocked_actions")
        total = count(episode, "total_actions")

        if invalid + blocked > total:
            raise ValueError("Inconsistent action counts")

        rewards.append(float(reward))
        risks.append(float(risk))

        invalid_actions += invalid
        blocked_actions += blocked
        total_actions += total

    return {
        "episode_count": len(episodes),
        "mean_reward": mean(rewards),
        "mean_cumulative_risk": mean(risks),
        "invalid_action_rate": (
            invalid_actions / total_actions
            if total_actions else 0.0
        ),
        "blocked_action_rate": (
            blocked_actions / total_actions
            if total_actions else 0.0
        ),
        "total_actions": total_actions,
        "evaluation_status": "completed",
    }
