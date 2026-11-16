
"""Stage 2 integration and API integrity tests."""

import pytest
from fastapi.testclient import TestClient

from Asterion.api import app
from Asterion.configuration import API_TOKEN


client = TestClient(app)
AUTH_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}


def test_required_stage2_routes_are_registered():
    """Fail if expected Stage 2 routes are absent from the API."""
    paths = app.openapi()["paths"]

    expected_routes = {
        ("/v1/detection/time-window", "post"),
        ("/v1/detection/multivariate", "post"),
        ("/v1/network-security/events", "post"),
        ("/v1/network-security/events", "get"),}

    missing = {
        f"{method.upper()} {path}"
        for path, method in expected_routes
        if method not in paths.get(path, {})}

    assert not missing, f"Missing Stage 2 API routes: {sorted(missing)}"


@pytest.mark.parametrize(
    "path",
    [
        "/v1/detection/time-window",
        "/v1/detection/multivariate",],)

def test_detection_endpoints_reject_missing_authentication(path):
    """Unauthenticated callers must not run the detectors."""
    response = client.post(path, json={"samples": []})

    assert response.status_code in (401, 403), (
        f"{path} unexpectedly accepted an unauthenticated request: "
        f"{response.status_code} {response.text}")


@pytest.mark.parametrize(
    "path",
    [
        "/v1/detection/time-window",
        "/v1/detection/multivariate",],)

def test_detection_endpoints_reject_empty_sample_windows(path):
    """An empty window must not be treated as valid telemetry."""
    response = client.post(
        path,
        headers=AUTH_HEADERS,
        json={"samples": []},)

    assert response.status_code == 422, response.text


def test_time_window_endpoint_accepts_valid_telemetry():
    """Exercise the time-window detector through the authenticated API."""
    samples = [
        {
            "timestamp": f"2026-01-01T00:{minute:02d}:00Z",
            "agent_id": "integrity-test-agent",
            "host": "integrity-test-host",
            "metrics": {
                "cpu_percent": 96.0,
                "memory_percent": 55.0,
                "disk_percent": 40.0,
                "load_1m": 2.0,},}

        for minute in (0, 1, 2, 3)]

    response = client.post(
        "/v1/detection/time-window",
        headers=AUTH_HEADERS,
        json={
            "samples": samples,
            "window_seconds": 300,
            "min_samples": 3,
            "max_gap_seconds": 120,
            "persistence_fraction": 0.60,
            "repeated_anomaly_minimum": 3,},)

    assert response.status_code == 200, response.text

    result = response.json()
    assert result["detector"] == "time_window_v1"
    assert isinstance(result["finding_count"], int)
    assert result["finding_count"] >= 0
    assert len(result["findings"]) == result["finding_count"]
    assert "do not establish malicious activity" in result["interpretation"]


def test_multivariate_endpoint_accepts_valid_telemetry():
    """Exercise multivariate detection through the authenticated API."""
    samples = [
        {
            "timestamp": f"2026-01-01T00:{minute:02d}:00Z",
            "agent_id": "integrity-test-agent",
            "host": "integrity-test-host",
            "metrics": {
                "cpu_percent": 95.0,
                "memory_percent": 90.0,
                "disk_percent": 80.0,
                "load_1m": 8.0,
            },
        }
        for minute in (0, 1, 2, 3)
    ]

    baseline = [
        {
            "timestamp": f"2025-12-31T23:{minute:02d}:00Z",
            "agent_id": "integrity-test-agent",
            "host": "integrity-test-host",
            "metrics": {
                "cpu_percent": 20.0,
                "memory_percent": 35.0,
                "disk_percent": 40.0,
                "load_1m": 1.0,
            },
        }
        for minute in (50, 51, 52, 53, 54)
    ]

    response = client.post(
        "/v1/detection/multivariate",
        headers=AUTH_HEADERS,
        json={
            "samples": samples,
            "baseline_samples": baseline,
            "features": [
                "cpu_percent",
                "memory_percent",
                "disk_percent",
                "load_1m",
            ],
            "minimum_features": 3,
            "score_threshold": 3.5,
        },
    )

    assert response.status_code == 200, response.text

    result = response.json()
    assert result["detector"] == "multivariate_robust_score_v1"
    assert isinstance(result["finding_count"], int)
    assert result["finding_count"] >= 0
    assert len(result["findings"]) == result["finding_count"]
    assert "not a calibrated intrusion probability" in result["interpretation"]


def test_detection_endpoints_reject_invalid_feature_configuration():
    """Invalid multivariate feature lists must not be silently accepted."""
    response = client.post(
        "/v1/detection/multivariate",
        headers=AUTH_HEADERS,
        json={
            "samples": [
                {
                    "timestamp": "2026-01-01T00:00:00Z",
                    "metrics": {"cpu_percent": 50.0},
                }
            ],
            "features": [],},)

    assert response.status_code == 422, response.text


def test_invalid_bearer_token_is_rejected():
    """A non-empty but incorrect token must not authorize detection."""
    response = client.post(
        "/v1/detection/time-window",
        headers={"Authorization": "Bearer invalid-integrity-test-token"},
        json={"samples": []},
    )

    assert response.status_code in (401, 403), response.text


from Asterion.evaluate_labeled_data import binary_metrics


def test_confusion_matrix_counts_are_consistent():
    """Check every prediction is accounted for exactly once."""
    result = binary_metrics(
        y_true=["normal", "normal", "intrusion", "intrusion"],
        y_pred=["normal", "intrusion", "intrusion", "normal"],
        positive_label="intrusion",
    )

    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["true_negatives"] == 1
    assert result["false_negatives"] == 1

    total = result["true_positives"] + result["false_positives"] + result["true_negatives"] + result["false_negatives"]
    assert total == 4

    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    assert result["false_positive_rate"] == pytest.approx(0.5)
    assert result["false_negative_rate"] == pytest.approx(0.5)


def test_perfect_predictions_produce_perfect_defined_metrics():
    result = binary_metrics(
        y_true=["normal", "normal", "intrusion", "intrusion"],
        y_pred=["normal", "normal", "intrusion", "intrusion"],
        positive_label="intrusion",
    )

    assert result["true_positives"] == 2
    assert result["false_positives"] == 0
    assert result["true_negatives"] == 2
    assert result["false_negatives"] == 0

    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["specificity"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
    assert result["false_positive_rate"] == pytest.approx(0.0)
    assert result["false_negative_rate"] == pytest.approx(0.0)


def test_missing_positive_class_does_not_fake_detection_performance():
    result = binary_metrics(
        y_true=["normal", "normal"],
        y_pred=["normal", "normal"],
        positive_label="intrusion",
    )

    assert result["true_negatives"] == 2
    assert result["false_positives"] == 0
    assert result["precision"] is None
    assert result["recall"] is None
    assert result["f1"] is None
    assert result["specificity"] == pytest.approx(1.0)
