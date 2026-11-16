from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from Asterion.stage2_detection import (
    detect_multivariate,
    detect_time_window,
)
from Asterion.network_security import NetworkSecurityRecord
from Asterion.evaluate_labeled_data import binary_metrics


def make_sample(host, timestamp, **metrics):
    return {
        "host": host,
        "timestamp": timestamp.isoformat(),
        "metrics": metrics,
    }


def test_time_window_detects_persistent_high_cpu():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    samples = [
        make_sample(
            "host-a",
            start + timedelta(seconds=10 * i),
            cpu_percent=96.0,
        )
        for i in range(6)
    ]

    findings = detect_time_window(
        samples,
        window_seconds=120,
        min_samples=3,
        max_gap_seconds=30,
    )

    assert any(
        item["category"] == "persistent_threshold"
        and item["evidence"]["metric"] == "cpu_percent"
        for item in findings
    )


def test_missing_samples_are_not_replaced_with_zero():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    samples = [
        make_sample("host-a", start, cpu_percent=95.0),
        make_sample("host-a", start + timedelta(seconds=10), cpu_percent=None),
        make_sample("host-a", start + timedelta(seconds=20), cpu_percent=96.0),
    ]

    findings = detect_time_window(
        samples,
        window_seconds=60,
        min_samples=3,
    )

    coverage_findings = [
        item for item in findings
        if item["category"] == "insufficient_window_data"
        and item["evidence"]["metric"] == "cpu_percent"
    ]
    assert coverage_findings
    assert coverage_findings[0]["evidence"]["available_samples"] == 2


def test_irregular_gap_does_not_create_a_rate_event():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    samples = [
        make_sample("host-a", start, cpu_percent=10.0),
        make_sample(
            "host-a",
            start + timedelta(seconds=600),
            cpu_percent=99.0,
        ),
    ]

    findings = detect_time_window(
        samples,
        window_seconds=1000,
        min_samples=2,
        max_gap_seconds=30,
        thresholds={"cpu_percent": 90},
        rate_thresholds_per_second={"cpu_percent": 0.01},
    )

    assert not any(
        item["category"] == "rapid_metric_change"
        for item in findings
    )


def test_multivariate_detection_flags_joint_deviation():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    baseline = [
        make_sample(
            "host-a",
            start + timedelta(seconds=i),
            cpu_percent=20 + (i % 3),
            memory_percent=30 + (i % 3),
            disk_percent=40 + (i % 3),
            load_1m=1 + (i % 3) * 0.1,
        )
        for i in range(20)
    ]
    target = make_sample(
        "host-a",
        start + timedelta(seconds=30),
        cpu_percent=99,
        memory_percent=98,
        disk_percent=99,
        load_1m=20,
    )

    findings = detect_multivariate(
        [target],
        baseline_samples=baseline,
        features=("cpu_percent", "memory_percent", "disk_percent", "load_1m"),
        minimum_features=3,
        score_threshold=3.5,
    )

    assert any(
        item["category"] == "multivariate_resource_anomaly"
        for item in findings
    )


def test_multivariate_missing_data_does_not_become_zero():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    baseline = [
        make_sample(
            "host-a",
            start + timedelta(seconds=i),
            cpu_percent=20 + (i % 3),
            memory_percent=30 + (i % 3),
            disk_percent=40 + (i % 3),
        )
        for i in range(10)
    ]
    target = make_sample(
        "host-a",
        start + timedelta(seconds=20),
        cpu_percent=99,
        memory_percent=None,
        disk_percent=None,
    )

    findings = detect_multivariate(
        [target],
        baseline_samples=baseline,
        features=("cpu_percent", "memory_percent", "disk_percent"),
        minimum_features=3,
    )

    assert any(
        item["category"] == "insufficient_multivariate_features"
        for item in findings
    )


def test_security_event_requires_event_name():
    with pytest.raises(ValidationError):
        NetworkSecurityRecord.model_validate({
            "record_id": "event-1",
            "record_type": "security_event",
            "timestamp": "2026-01-01T00:00:00Z",
            "source_host": "host-a",
            "provenance_source": "test-source",
            "provenance_method": "test-parser-v1",
        })


def test_flow_requires_at_least_one_flow_measurement():
    with pytest.raises(ValidationError):
        NetworkSecurityRecord.model_validate({
            "record_id": "flow-1",
            "record_type": "flow",
            "timestamp": "2026-01-01T00:00:00Z",
            "source_host": "host-a",
            "provenance_source": "test-source",
            "provenance_method": "test-parser-v1",
        })


def test_security_record_accepts_explicit_unavailable_endpoints():
    record = NetworkSecurityRecord.model_validate({
        "record_id": "event-2",
        "record_type": "security_event",
        "timestamp": "2026-01-01T00:00:00Z",
        "source_host": "host-a",
        "event_name": "connection_rejected",
        "provenance_source": "test-source",
        "provenance_method": "test-parser-v1",
        "unavailable_fields": {
            "destination_ip": "not supplied by source",
            "protocol": "not observed",
        },
    })

    assert record.destination_ip is None
    assert record.protocol is None
    assert len(record.fingerprint()) == 64


def test_binary_metrics_calculate_false_positive_and_negative_rates():
    import numpy as np

    result = binary_metrics(
        np.array(["intrusion", "normal", "normal", "intrusion"]),
        np.array(["intrusion", "intrusion", "normal", "normal"]),
        positive_label="intrusion",)

    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["true_negatives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["false_positive_rate"] == pytest.approx(0.5)
    assert result["false_negative_rate"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def test_zero_denominator_is_reported_as_undefined():
    import numpy as np

    result = binary_metrics(
        np.array(["normal", "normal"]),
        np.array(["normal", "normal"]),
        positive_label="intrusion",
    )

    assert result["true_negatives"] == 2
    assert result["false_positives"] == 0
    assert result["recall"] is None
    assert result["false_negative_rate"] is None
    assert result["precision"] is None
    assert result["specificity"] == pytest.approx(1.0)



