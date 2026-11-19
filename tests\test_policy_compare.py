
import json

import numpy as np
import pytest
from gymnasium import spaces

from Asterion.ReinforcementLearning.policy_compare import (
    baseline_action,
    build_comparison,
    paired_summary,
    run_episode,
    summarize,
    validate_model_compatibility,
    write_result,
)


class FakePPO:
    """Small deterministic policy for testing the benchmark interface."""

    def __init__(self, observation_shape=(10,), action_count=4, action=0):
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=observation_shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(action_count)
        self.action = action

    def predict(self, observation, deterministic=True):
        assert deterministic is True
        return np.asarray(self.action), None


def test_baseline_returns_valid_action():
    observation = np.asarray(
        [0.90, 0.80, 0.30, 0.10, 0.60,
         1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=np.float32,
    )

    assert baseline_action(observation) == 2


def test_baseline_prioritizes_suspicious_activity():
    observation = np.asarray(
        [0.90, 0.90, 0.90, 0.80, 0.60,
         1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=np.float32,
    )

    assert baseline_action(observation) == 3


def test_baseline_rejects_invalid_observation_shape():
    with pytest.raises(ValueError):
        baseline_action(np.zeros(5, dtype=np.float32))


def test_baseline_rejects_nonfinite_observation():
    observation = np.zeros(10, dtype=np.float32)
    observation[0] = np.nan

    with pytest.raises(ValueError):
        baseline_action(observation)


def test_model_compatibility_accepts_correct_spaces():
    validate_model_compatibility(FakePPO())


def test_model_compatibility_rejects_wrong_observation_shape():
    with pytest.raises(ValueError, match="observation shape"):
        validate_model_compatibility(
            FakePPO(observation_shape=(5,))
        )


def test_model_compatibility_rejects_wrong_action_count():
    with pytest.raises(ValueError, match="action count"):
        validate_model_compatibility(
            FakePPO(action_count=3)
        )


def test_episode_is_reproducible_for_same_seed():
    first = run_episode(None, seed=1234, horizon=20)
    second = run_episode(None, seed=1234, horizon=20)

    assert first["initial_observation"] == second["initial_observation"]
    assert first["return"] == pytest.approx(second["return"])
    assert first["mean_pressure"] == pytest.approx(second["mean_pressure"])
    assert first["action_counts"] == second["action_counts"]


def test_episode_has_correct_step_and_action_accounting():
    result = run_episode(None, seed=1234, horizon=15)

    assert result["steps"] == 15
    assert sum(result["action_counts"].values()) == 15
    assert result["mean_action_latency_ms"] >= 0
    assert result["p95_action_latency_ms"] >= 0
    assert result["total_intervention_cost"] >= 0


def test_ppo_episode_uses_deterministic_prediction():
    result = run_episode(
        FakePPO(action=0),
        seed=1234,
        horizon=12,
    )

    assert result["steps"] == 12
    assert result["action_counts"]["notify_operator"] == 12


def test_summary_aggregates_episode_metrics():
    rows = [
        {
            "return": 10.0,
            "mean_pressure": 0.4,
            "total_intervention_cost": 1.0,
            "mean_action_latency_ms": 0.1,
            "action_counts": {
                "notify_operator": 2,
                "collect_diagnostics": 0,
                "reduce_noncritical_workload": 0,
                "review_suspicious_activity": 0,
            },
        },
        {
            "return": 20.0,
            "mean_pressure": 0.2,
            "total_intervention_cost": 2.0,
            "mean_action_latency_ms": 0.2,
            "action_counts": {
                "notify_operator": 1,
                "collect_diagnostics": 1,
                "reduce_noncritical_workload": 0,
                "review_suspicious_activity": 0,
            },
        },
    ]

    result = summarize(rows)

    assert result["episodes"] == 2
    assert result["mean_return"] == pytest.approx(15.0)
    assert result["mean_pressure"] == pytest.approx(0.3)
    assert result["total_actions"] == 4


def test_paired_summary_returns_bootstrap_interval():
    result = paired_summary(
        [1.0, 2.0, 1.5, 0.5, 1.2],
        bootstrap_samples=1000,
    )

    assert result["paired_episode_count"] == 5
    assert result["bootstrap_95_percent_interval"] is not None
    assert result["confidence_interval_method"] == (
        "paired_percentile_bootstrap"
    )


def test_paired_summary_is_reproducible():
    differences = [1.0, 2.0, 1.5, 0.5, 1.2]

    first = paired_summary(
        differences,
        bootstrap_samples=1000,
        seed=42,
    )
    second = paired_summary(
        differences,
        bootstrap_samples=1000,
        seed=42,
    )

    assert first == second


def test_paired_summary_handles_single_pair_without_false_interval():
    result = paired_summary([1.0])

    assert result["mean_difference"] == 1.0
    assert result["bootstrap_95_percent_interval"] is None
    assert result["confidence_interval_status"] == "insufficient_pairs"


def test_paired_summary_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        paired_summary([1.0, float("nan")])


def test_build_comparison_uses_paired_initial_conditions():
    result = build_comparison(
        FakePPO(action=0),
        episodes=5,
        horizon=10,
        bootstrap_samples=1000,
    )

    baseline = result["episode_results"]["baseline"]
    ppo = result["episode_results"]["ppo"]

    assert len(baseline) == 5
    assert len(ppo) == 5

    for baseline_episode, ppo_episode in zip(baseline, ppo):
        assert baseline_episode["seed"] == ppo_episode["seed"]
        assert (
            baseline_episode["initial_observation"]
            == ppo_episode["initial_observation"]
        )


def test_build_comparison_marks_simulation_only():
    result = build_comparison(
        FakePPO(),
        episodes=5,
        horizon=10,
        bootstrap_samples=1000,
    )

    assert result["environment_type"] == "synthetic_only"
    assert result["live_systems_touched"] is False
    assert result["action_execution_enabled"] is False
    assert result["model_promotion_performed"] is False


def test_build_comparison_rejects_too_few_episodes():
    with pytest.raises(ValueError, match="five"):
        build_comparison(
            FakePPO(),
            episodes=4,
            horizon=10,
        )


def test_write_result_creates_valid_json(tmp_path):
    destination = tmp_path / "results" / "comparison.json"

    result = {
        "environment_type": "synthetic_only",
        "live_systems_touched": False,
        "mean_return": 10.5,
    }

    written = write_result(result, destination)

    assert written.is_file()

    loaded = json.loads(written.read_text())

    assert loaded == result
    assert not destination.with_name(
        destination.name + ".tmp"
    ).exists()
