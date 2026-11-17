
import numpy as np
import pytest

from Asterion.feature_contract import (
    CONTRACT_VERSION,
    FEATURE_NAMES,
    AVAILABILITY_NAMES,
    FEATURE_DIMENSION,
    encode_features,
    encode_feature_vector,
)

from Asterion.ReinforcementLearning.telemetry_generator import (
    generate_dataset,
    generate_observation,
)


def test_feature_order_is_fixed():
    assert FEATURE_NAMES == (
        "cpu_percent",
        "memory_percent",
        "disk_percent",
        "network_connections",
        "load_1m",
    )

    assert AVAILABILITY_NAMES == (
        "cpu_percent_available",
        "memory_percent_available",
        "disk_percent_available",
        "network_connections_available",
        "load_1m_available",
    )


def test_feature_vector_has_ten_dimensions():
    metrics = {
        "cpu_percent": 50.0,
        "memory_percent": 60.0,
        "disk_percent": 70.0,
        "network_connections": 250,
        "load_1m": 4.0,
    }

    vector = encode_feature_vector(metrics)

    assert vector.shape == (FEATURE_DIMENSION,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()


def test_normalization_uses_authoritative_scales():
    metrics = {
        "cpu_percent": 50.0,
        "memory_percent": 25.0,
        "disk_percent": 100.0,
        "network_connections": 250,
        "load_1m": 4.0,
    }

    vector = encode_feature_vector(metrics)

    np.testing.assert_allclose(
        vector,
        np.asarray(
            [
                0.50,
                0.25,
                1.00,
                0.50,
                0.50,
                1.00,
                1.00,
                1.00,
                1.00,
                1.00,
            ],
            dtype=np.float32,
        ),
    )


def test_measured_zero_differs_from_missing():
    measured_zero = {
        "cpu_percent": 0.0,
        "memory_percent": 0.0,
        "disk_percent": 0.0,
        "network_connections": 0,
        "load_1m": 0.0,
    }

    unavailable = {
        "cpu_percent": None,
        "memory_percent": None,
        "disk_percent": None,
        "network_connections": None,
        "load_1m": None,
    }

    zero_vector = encode_feature_vector(measured_zero)
    missing_vector = encode_feature_vector(unavailable)

    np.testing.assert_array_equal(
        zero_vector[:5],
        missing_vector[:5],
    )

    np.testing.assert_array_equal(
        zero_vector[5:],
        np.ones(5, dtype=np.float32),
    )

    np.testing.assert_array_equal(
        missing_vector[5:],
        np.zeros(5, dtype=np.float32),
    )


def test_structured_and_raw_metrics_encode_identically():
    raw = {
        "cpu_percent": 50.0,
        "memory_percent": 40.0,
        "disk_percent": 30.0,
        "network_connections": 200,
        "load_1m": 2.0,
    }

    structured = {
        "cpu_percent": {
            "value": 50.0,
            "unit": "%",
            "status": "ok",
            "error": None,
        },
        "memory_percent": {
            "value": 40.0,
            "unit": "%",
            "status": "ok",
            "error": None,
        },
        "disk_percent": {
            "value": 30.0,
            "unit": "%",
            "status": "ok",
            "error": None,
        },
        "network_connections": {
            "value": 200,
            "unit": "connections",
            "status": "ok",
            "error": None,
        },
        "load_1m": {
            "value": 2.0,
            "unit": "load average",
            "status": "ok",
            "error": None,
        },
    }

    np.testing.assert_array_equal(
        encode_feature_vector(raw),
        encode_feature_vector(structured),
    )


def test_structured_unavailable_measurement_requires_null():
    metrics = {
        "cpu_percent": {
            "value": 10.0,
            "unit": "%",
            "status": "unavailable",
            "error": "Permission denied",
        },
        "memory_percent": 50.0,
        "disk_percent": 50.0,
        "network_connections": 0,
        "load_1m": 1.0,
    }

    with pytest.raises(ValueError):
        encode_feature_vector(metrics)


def test_invalid_measurements_are_rejected():
    invalid_cases = [
        ("cpu_percent", 101.0),
        ("memory_percent", -1.0),
        ("disk_percent", float("nan")),
        ("network_connections", 1.5),
        ("network_connections", -1),
        ("load_1m", float("inf")),
        ("load_1m", -0.1),
    ]

    for invalid_name, invalid_value in invalid_cases:
        metrics = {
            "cpu_percent": 10.0,
            "memory_percent": 20.0,
            "disk_percent": 30.0,
            "network_connections": 10,
            "load_1m": 1.0,
        }

        metrics[invalid_name] = invalid_value

        with pytest.raises(ValueError):
            encode_feature_vector(metrics)


def test_normalization_clips_values_above_reference_scale():
    metrics = {
        "cpu_percent": 100.0,
        "memory_percent": 100.0,
        "disk_percent": 100.0,
        "network_connections": 1000,
        "load_1m": 16.0,
    }

    vector = encode_feature_vector(metrics)

    np.testing.assert_array_equal(
        vector[:5],
        np.ones(5, dtype=np.float32),
    )


def test_training_generator_uses_same_contract():
    vectors, observations = generate_dataset(
        100,
        seed=123,
    )

    assert vectors.shape == (100, 10)
    assert np.isfinite(vectors).all()

    for vector, observation in zip(vectors, observations):
        runtime_vector = encode_feature_vector(
            observation["metrics"]
        )

        np.testing.assert_array_equal(
            vector,
            runtime_vector,
        )


def test_generated_observations_preserve_failure_status():
    rng = np.random.default_rng(7)

    observation = generate_observation(
        rng,
        missing_probability=1.0,
        collection_failure_probability=0.0,
    )

    assert observation["collection_status"] == "failed"

    assert all(
        value is None
        for value in observation["metrics"].values()
    )

    assert all(
        record["status"] == "unavailable"
        for record in observation["metric_status"].values()
    )

    np.testing.assert_array_equal(
        observation["encoded"].vector[5:],
        np.zeros(5, dtype=np.float32),
    )


def test_contract_version_is_attached_to_encoding():
    metrics = {
        "cpu_percent": 1.0,
        "memory_percent": 2.0,
        "disk_percent": 3.0,
        "network_connections": 4,
        "load_1m": 0.5,
    }

    encoded = encode_features(metrics)

    assert encoded.contract_version == CONTRACT_VERSION
    assert encoded.as_dict()["contract_version"] == CONTRACT_VERSION

def test_reinforcement_learning_environment_uses_contract():
    from Asterion.ReinforcementLearning.decision_env import (
        DecisionOptimizationEnv,
    )

    env = DecisionOptimizationEnv()
    observation, info = env.reset(seed=42)

    assert observation.shape == (10,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info["contract_version"] == CONTRACT_VERSION

    for action in range(env.action_space.n):
        observation, reward, terminated, truncated, info = env.step(
            action
        )

        assert observation.shape == (10,)
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        assert info["feature_contract_version"] == CONTRACT_VERSION

        if terminated or truncated:
            break

    env.close()