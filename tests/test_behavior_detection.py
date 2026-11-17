
from Asterion.behavior_detection import (
    detect_behavior,
    detect_multivariate_deviations,
    detect_persistent_violations,
    detect_security_indicators,
    detect_temporal_changes,
    detect_threshold_violations,
)


def sample(minute, *, cpu=20.0, memory=35.0, disk=40.0, load=1.0):
    return {
        "timestamp": f"2026-01-01T00:{minute:02d}:00Z",
        "agent_id": "test-agent",
        "host": "test-host",
        "metrics": {
            "cpu_percent": cpu,
            "memory_percent": memory,
            "disk_percent": disk,
            "load_1m": load,
        },
    }


def test_threshold_violation_is_reported_with_evidence():
    findings = detect_threshold_violations(
        [sample(0, cpu=96.0)],
        {"cpu_percent": 90.0},
    )

    assert len(findings) == 1
    assert findings[0]["category"] == "resource_threshold"
    assert findings[0]["evidence"]["observed_value"] == 96.0
    assert findings[0]["evidence"]["threshold"] == 90.0
    assert findings[0]["action_execution"] == "disabled"


def test_missing_or_nonfinite_metrics_do_not_become_zero():
    record = sample(0)
    record["metrics"]["cpu_percent"] = None
    record["metrics"]["memory_percent"] = float("nan")

    findings = detect_threshold_violations(
        [record],
        {"cpu_percent": 90.0, "memory_percent": 90.0},
    )

    assert findings == []


def test_invalid_percentage_is_not_used_as_valid_evidence():
    findings = detect_threshold_violations(
        [sample(0, cpu=150.0)],
        {"cpu_percent": 90.0},
    )

    assert findings == []


def test_abrupt_change_is_detected():
    records = [
        sample(0, cpu=20.0),
        sample(1, cpu=80.0),
    ]

    findings = detect_temporal_changes(
        records,
        {"cpu_percent": 25.0},
        max_gap_seconds=120,
    )

    cpu_findings = [
        item for item in findings
        if item["evidence"]["metric"] == "cpu_percent"
    ]
    assert len(cpu_findings) == 1
    assert cpu_findings[0]["evidence"]["change"] == 60.0


def test_temporal_detector_does_not_bridge_large_time_gaps():
    records = [
        sample(0, cpu=20.0),
        {
            **sample(3, cpu=95.0),
            "timestamp": "2026-01-01T01:00:00Z",
        },
    ]

    findings = detect_temporal_changes(
        records,
        {"cpu_percent": 25.0},
        max_gap_seconds=120,
    )

    assert findings == []


def test_persistent_threshold_breach_is_detected():
    records = [
        sample(0, cpu=95.0),
        sample(1, cpu=96.0),
        sample(2, cpu=94.0),
        sample(3, cpu=30.0),
    ]

    findings = detect_persistent_violations(
        records,
        {"cpu_percent": 90.0},
        min_samples=3,
        persistence_fraction=0.6,
    )

    assert len(findings) == 1
    assert findings[0]["evidence"]["violation_count"] == 3
    assert findings[0]["evidence"]["persistence_fraction"] == 0.75


def test_persistent_detector_rejects_invalid_configuration():
    records = [sample(0), sample(1), sample(2)]

    try:
        detect_persistent_violations(
            records,
            min_samples=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("min_samples=0 should be rejected")


def test_multivariate_deviation_requires_multiple_features():
    baseline = [
        sample(i, cpu=20.0 + i, memory=30.0 + i, disk=40.0 + i, load=1.0 + i * 0.1)
        for i in range(5)
    ]
    current = [
        sample(10, cpu=95.0, memory=95.0, disk=95.0, load=10.0)
    ]

    findings = detect_multivariate_deviations(
        current,
        baseline,
        minimum_features=3,
        robust_z_threshold=3.5,
    )

    assert len(findings) == 1
    assert findings[0]["category"] == "multivariate_baseline_deviation"
    assert findings[0]["evidence"]["deviating_feature_count"] >= 3


def test_security_indicator_match_is_reported():
    events = [{
        "timestamp": "2026-01-01T00:00:00Z",
        "event_type": "network_connection",
        "src_ip": "192.0.2.10",
        "dst_ip": "198.51.100.7",
        "source": "test-feed",
    }]

    findings = detect_security_indicators(
        events,
        known_indicators={"dst_ip": {"198.51.100.7"}},
    )

    assert len(findings) == 1
    assert findings[0]["category"] == "security_indicator"
    assert findings[0]["evidence"]["matched_indicators"][0]["field"] == "dst_ip"


def test_explicit_malware_alert_is_preserved_as_evidence():
    findings = detect_security_indicators([{
        "timestamp": "2026-01-01T00:00:00Z",
        "event_type": "malware_detected",
        "host": "test-host",
        "source": "endpoint-protection-test",
    }])

    assert len(findings) == 1
    assert findings[0]["evidence"]["event_type"] == "malware_detected"
    assert findings[0]["severity"] == "critical"


def test_combined_detector_returns_auditable_findings():
    records = [
        sample(0, cpu=95.0),
        sample(1, cpu=96.0),
        sample(2, cpu=97.0),
    ]

    result = detect_behavior(
        records,
        security_events=[{
            "event_type": "brute_force_detected",
            "host": "test-host",
            "source": "test-security-feed",
        }],
    )

    assert result["detector"] == "behavior_detection_v1"
    assert result["finding_count"] == len(result["findings"])
    assert result["finding_count"] >= 2
    assert result["action_execution"] == "disabled"
    assert all("evidence" in finding for finding in result["findings"])
