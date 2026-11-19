
"""Paired-seed PPO versus deterministic-baseline benchmark.

This benchmark operates exclusively on DecisionOptimizationEnv, a synthetic
Gymnasium environment. It does not connect to host telemetry, invoke the
Asterion executor, authorize actions, or modify live systems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .decision_env import DecisionOptimizationEnv, ACTION_NAMES
from ..configuration import MODEL_DIR


EXPECTED_OBSERVATION_SHAPE = (10,)
EXPECTED_ACTION_COUNT = len(ACTION_NAMES)
DEFAULT_SEED = 1000
DEFAULT_BOOTSTRAP_SAMPLES = 10_000


def baseline_action(observation: np.ndarray) -> int:
    """Select an action using the deterministic five-feature baseline."""
    values = np.asarray(observation, dtype=np.float32)

    if values.shape != (10,):
        raise ValueError(
            f"Expected a 10-element observation; received {values.shape}"
        )

    if not np.isfinite(values).all():
        raise ValueError("Observation contains non-finite values")

    cpu, memory, disk, suspicious, workload = (
        float(value) for value in values[:5]
    )

    if suspicious >= 0.55:
        return 3

    if cpu >= 0.72 or memory >= 0.78:
        return 2 if workload >= 0.35 else 1

    if disk >= 0.90:
        return 1

    return 0


def validate_environment(environment: DecisionOptimizationEnv) -> None:
    """Reject incompatible environment observation or action contracts."""
    if environment.observation_space.shape != EXPECTED_OBSERVATION_SHAPE:
        raise ValueError(
            "Environment observation shape mismatch: "
            f"{environment.observation_space.shape}; "
            f"expected {EXPECTED_OBSERVATION_SHAPE}"
        )

    if environment.action_space.n != EXPECTED_ACTION_COUNT:
        raise ValueError(
            "Environment action count mismatch: "
            f"{environment.action_space.n}; "
            f"expected {EXPECTED_ACTION_COUNT}"
        )


def validate_model_compatibility(model: Any) -> None:
    """Verify PPO uses the benchmark's exact observation and action spaces."""
    observation_shape = getattr(
        getattr(model, "observation_space", None),
        "shape",
        None,
    )

    action_count = getattr(
        getattr(model, "action_space", None),
        "n",
        None,
    )

    if observation_shape != EXPECTED_OBSERVATION_SHAPE:
        raise ValueError(
            "Incompatible PPO observation shape: "
            f"{observation_shape!r}; expected {EXPECTED_OBSERVATION_SHAPE}. "
            "Train the model with DecisionOptimizationEnv."
        )

    if action_count != EXPECTED_ACTION_COUNT:
        raise ValueError(
            "Incompatible PPO action count: "
            f"{action_count!r}; expected {EXPECTED_ACTION_COUNT}."
        )


def _predict_action(model: Any, observation: np.ndarray) -> int:
    """Obtain and validate one deterministic PPO action."""
    prediction = model.predict(observation, deterministic=True)
    raw_action = prediction[0] if isinstance(prediction, tuple) else prediction

    action_array = np.asarray(raw_action)

    if action_array.size != 1:
        raise ValueError("PPO returned an action with an unexpected shape")

    action_value = action_array.item()

    if isinstance(action_value, (bool, np.bool_)):
        raise ValueError("PPO returned a boolean action")

    action = int(action_value)

    if action != action_value or not 0 <= action < EXPECTED_ACTION_COUNT:
        raise ValueError(f"PPO returned invalid action index: {action_value!r}")

    return action

def run_episode(
    policy: Any,
    seed: int,
    horizon: int,
) -> dict[str, Any]:
    """Run one isolated simulation episode.

    Pass None for the deterministic baseline or a compatible PPO model.
    Action-selection latency excludes environment transition time.

    Missing pressure measurements are excluded from the pressure mean.
    They are never interpreted as measured zero.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")

    environment = DecisionOptimizationEnv(horizon=horizon)
    validate_environment(environment)

    observation, _ = environment.reset(seed=seed)
    initial_observation = observation.astype(float).tolist()

    total_reward = 0.0
    pressure_samples: list[float] = []
    intervention_cost_sum = 0.0
    action_latency_ms: list[float] = []
    action_counts = {name: 0 for name in ACTION_NAMES}

    steps = 0

    try:
        for _ in range(horizon):
            start_ns = time.perf_counter_ns()

            if policy is None:
                action = baseline_action(observation)
            else:
                action = _predict_action(policy, observation)

            elapsed_ms = (
                time.perf_counter_ns() - start_ns
            ) / 1_000_000.0

            action_latency_ms.append(elapsed_ms)

            observation, reward, terminated, truncated, info = (
                environment.step(action)
            )

            total_reward += float(reward)

            pressure = info.get("pressure")
            if pressure is not None:
                numeric_pressure = float(pressure)

                if math.isfinite(numeric_pressure):
                    pressure_samples.append(numeric_pressure)

            intervention_cost_sum += float(
                info["intervention_cost"]
            )

            action_counts[ACTION_NAMES[action]] += 1
            steps += 1

            if terminated or truncated:
                break

    finally:
        environment.close()

    if steps == 0:
        raise RuntimeError("Simulation completed without any steps")

    mean_pressure = (
        statistics.mean(pressure_samples)
        if pressure_samples
        else None
    )

    return {
        "seed": seed,
        "initial_observation": initial_observation,
        "return": total_reward,
        "mean_pressure": mean_pressure,
        "pressure_measurement_count": len(pressure_samples),
        "steps": steps,
        "total_intervention_cost": intervention_cost_sum,
        "action_counts": action_counts,
        "mean_action_latency_ms": statistics.mean(action_latency_ms),
        "median_action_latency_ms": statistics.median(action_latency_ms),
        "p95_action_latency_ms": _percentile(action_latency_ms, 95),
    }

def _percentile(values: list[float], percentile: float) -> float:
    """Calculate a linearly interpolated percentile."""
    if not values:
        raise ValueError("Cannot calculate percentile of empty data")

    ordered = sorted(float(value) for value in values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )

def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate episode-level benchmark metrics.

    Pressure statistics include only episodes with valid pressure
    measurements. Missing pressure is not treated as zero.
    """
    if not episodes:
        raise ValueError("At least one episode is required")

    returns = [float(row["return"]) for row in episodes]

    pressures = [
        float(row["mean_pressure"])
        for row in episodes
        if row.get("mean_pressure") is not None
        and math.isfinite(float(row["mean_pressure"]))
    ]

    costs = [
        float(row["total_intervention_cost"])
        for row in episodes
    ]

    latencies = [
        float(row["mean_action_latency_ms"])
        for row in episodes
    ]

    action_counts = {
        name: sum(row["action_counts"][name] for row in episodes)
        for name in ACTION_NAMES
    }

    total_actions = sum(action_counts.values())

    return {
        "episodes": len(episodes),
        "mean_return": statistics.mean(returns),
        "median_return": statistics.median(returns),
        "stdev_return": (
            statistics.stdev(returns)
            if len(returns) > 1
            else 0.0
        ),
        "mean_pressure": (
            statistics.mean(pressures)
            if pressures
            else None
        ),
        "pressure_episode_count": len(pressures),
        "mean_intervention_cost_per_episode": statistics.mean(costs),
        "mean_action_latency_ms": statistics.mean(latencies),
        "p95_episode_mean_action_latency_ms": _percentile(
            latencies,
            95,
        ),
        "action_counts": action_counts,
        "action_frequencies": {
            name: count / total_actions if total_actions else 0.0
            for name, count in action_counts.items()
        },
        "total_actions": total_actions,
    }

def paired_summary(
    differences: list[float],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 2026,
) -> dict[str, Any]:
    """Summarize paired differences with a reproducible percentile bootstrap.

    Positive values mean PPO exceeds the baseline for the metric being
    compared. For pressure and cost, negative values favor PPO.
    """
    if not differences:
        raise ValueError("At least one paired difference is required")

    if bootstrap_samples < 1000:
        raise ValueError("bootstrap_samples must be at least 1000")

    values = np.asarray(differences, dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError("Paired differences must be finite")

    mean_difference = float(np.mean(values))

    if len(values) < 2:
        return {
            "mean_difference": mean_difference,
            "standard_error": None,
            "bootstrap_95_percent_interval": None,
            "paired_episode_count": len(values),
            "confidence_interval_status": "insufficient_pairs",
        }

    standard_error = float(
        np.std(values, ddof=1) / math.sqrt(len(values))
    )

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(values),
        size=(bootstrap_samples, len(values)),
    )

    bootstrap_means = values[indices].mean(axis=1)

    lower, upper = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return {
        "mean_difference": mean_difference,
        "standard_error": standard_error,
        "bootstrap_95_percent_interval": [
            float(lower),
            float(upper),
        ],
        "paired_episode_count": len(values),
        "bootstrap_samples": bootstrap_samples,
        "confidence_interval_method": "paired_percentile_bootstrap",
        "confidence_interval_status": "estimated",
    }


def _model_sha256(model_path: Path) -> str:
    """Hash the exact model artifact used for the comparison."""
    digest = hashlib.sha256()

    with model_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()

def paired_optional_metric(
    baseline_rows: list[dict[str, Any]],
    ppo_rows: list[dict[str, Any]],
    metric: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compare a metric using only pairs with valid measurements.

    A pair is included only when both policies have a finite numeric value.
    This is important for measurements such as pressure that can legitimately
    be unavailable.
    """
    differences: list[float] = []

    for baseline_row, ppo_row in zip(
        baseline_rows,
        ppo_rows,
    ):
        baseline_value = baseline_row.get(metric)
        ppo_value = ppo_row.get(metric)

        if baseline_value is None or ppo_value is None:
            continue

        baseline_numeric = float(baseline_value)
        ppo_numeric = float(ppo_value)

        if not (
            math.isfinite(baseline_numeric)
            and math.isfinite(ppo_numeric)
        ):
            continue

        differences.append(
            ppo_numeric - baseline_numeric
        )

    if not differences:
        return {
            "mean_difference": None,
            "standard_error": None,
            "bootstrap_95_percent_interval": None,
            "paired_episode_count": 0,
            "confidence_interval_status": (
                "unavailable_measurements"
            ),
        }

    return paired_summary(
        differences,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

def build_comparison(
    model: Any,
    *,
    episodes: int = 20,
    horizon: int = 80,
    seed_start: int = DEFAULT_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """Run paired baseline/PPO trials under identical scenario seeds."""
    if episodes < 5:
        raise ValueError("Use at least five episodes")

    if horizon < 10:
        raise ValueError("Use a horizon of at least ten steps")

    validate_model_compatibility(model)

    baseline_rows = []
    ppo_rows = []

    for episode_index in range(episodes):
        seed = seed_start + episode_index

        baseline_episode = run_episode(
            None,
            seed,
            horizon,
        )

        ppo_episode = run_episode(
            model,
            seed,
            horizon,
        )

        if not np.array_equal(
            np.asarray(baseline_episode["initial_observation"]),
            np.asarray(ppo_episode["initial_observation"]),
        ):
            raise RuntimeError(
                f"Paired initial observations differ for seed {seed}"
            )

        baseline_rows.append(baseline_episode)
        ppo_rows.append(ppo_episode)

    baseline_summary = summarize(baseline_rows)
    ppo_summary = summarize(ppo_rows)

    def paired_metric(metric: str) -> list[float]:
        return [
            float(ppo_row[metric]) - float(baseline_row[metric])
            for baseline_row, ppo_row in zip(
                baseline_rows,
                ppo_rows,
            )
        ]

    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": "DecisionOptimizationEnv",
        "environment_type": "synthetic_only",
        "live_systems_touched": False,
        "action_execution_enabled": False,
        "model_promotion_performed": False,
        "benchmark_configuration": {
            "episodes": episodes,
            "horizon": horizon,
            "seed_start": seed_start,
            "seed_end": seed_start + episodes - 1,
            "bootstrap_samples": bootstrap_samples,
            "observation_shape": list(EXPECTED_OBSERVATION_SHAPE),
            "action_names": ACTION_NAMES,
        },
        "baseline": baseline_summary,
        "ppo": ppo_summary,
        "paired_differences_ppo_minus_baseline": {
            "return": paired_summary(
                paired_metric("return"),
                bootstrap_samples=bootstrap_samples,
                seed=2026,
            ),
            "mean_pressure": paired_optional_metric(
                baseline_rows,
                ppo_rows,
                "mean_pressure",
                bootstrap_samples=bootstrap_samples,
                seed=2027,
            ),

            "intervention_cost": paired_summary(
                paired_metric("total_intervention_cost"),
                bootstrap_samples=bootstrap_samples,
                seed=2028,
            ),
            "mean_action_latency_ms": paired_summary(
                paired_metric("mean_action_latency_ms"),
                bootstrap_samples=bootstrap_samples,
                seed=2029,
            ),
        },
        "episode_results": {
            "baseline": baseline_rows,
            "ppo": ppo_rows,
        },
        "interpretation": (
            "Synthetic simulation only. Paired differences estimate "
            "performance under this environment and configuration. "
            "They do not establish production effectiveness, causal impact, "
            "or operational safety. No automatic model promotion is permitted."
        ),
    }


def write_result(result: dict[str, Any], output_path: str | Path) -> Path:
    """Write benchmark output atomically as JSON."""
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_name(destination.name + ".tmp")

    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")

        temporary.replace(destination)

    finally:
        if temporary.exists():
            temporary.unlink()

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PPO and deterministic policies in simulation."
    )

    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument(
        "--model",
        default=str(MODEL_DIR / "rl" / "decision_policy_ppo.zip"),
    )
    parser.add_argument(
        "--output",
        default=str(MODEL_DIR / "rl" / "policy_comparison.json"),
    )

    args = parser.parse_args()

    if args.episodes < 5:
        parser.error("--episodes must be at least 5")

    if args.horizon < 10:
        parser.error("--horizon must be at least 10")

    if args.bootstrap_samples < 1000:
        parser.error("--bootstrap-samples must be at least 1000")

    model_path = Path(args.model).expanduser()

    if not model_path.is_file():
        parser.error(
            f"PPO model not found: {model_path}. "
            "Train it with train_dec_policy.py first."
        )

    try:
        from stable_baselines3 import PPO

        model = PPO.load(str(model_path))
        validate_model_compatibility(model)

        result = build_comparison(
            model,
            episodes=args.episodes,
            horizon=args.horizon,
            seed_start=args.seed_start,
            bootstrap_samples=args.bootstrap_samples,
        )

        result["model"] = {
            "path": str(model_path.resolve()),
            "sha256": _model_sha256(model_path),
        }

        destination = write_result(result, args.output)

    except Exception as exc:
        parser.exit(
            1,
            f"Benchmark failed: {type(exc).__name__}: {exc}\n",
        )

    print(f"Benchmark completed: {destination}")
    print(
        "Mean return difference (PPO - baseline): "
        f"{result['paired_differences_ppo_minus_baseline']['return']['mean_difference']:.5f}"
    )
    print(
        "Mean pressure difference (PPO - baseline): "
        f"{result['paired_differences_ppo_minus_baseline']['mean_pressure']['mean_difference']:.5f}"
    )
    print("Live systems touched: False")
    print("Automatic promotion: False")


if __name__ == "__main__":
    main()
