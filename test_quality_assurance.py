from Asterion.Intelligence.quality_assurance import (
    ModelRegistry,
    PromotionGates,
    audit_labeled_rows,
    benchmark_policies,
    compose_reward,
    evaluate_anomaly_detector,
    evaluate_classifier,
    run_live_telemetry_decision_probe,
)
import pytest


def test_data_audit_detects_duplicates_and_class_balance():
    rows = [
        {"text": "normal a", "label": "normal"},
        {"text": "normal a", "label": "normal"},
        {"text": "normal b", "label": "normal"},
        {"text": "bad a", "label": "bad"},
        {"text": "bad b", "label": "bad"},
    ]
    report = audit_labeled_rows(rows)
    assert report["unique_rows"] == 4
    assert report["exact_duplicate_rows"] == 1
    assert report["minority_to_majority_ratio"] == 1.0


def test_data_audit_rejects_conflicting_labels():
    with pytest.raises(ValueError, match="Conflicting labels"):
        audit_labeled_rows([
            {"text": "same", "label": "normal"},
            {"text": "same", "label": "anomaly"},
        ])


def test_classifier_reports_fp_recall_and_temporal_stability():
    y = [0, 0, 1, 1, 0, 1, 0, 1]
    p = [0, 1, 1, 0, 0, 1, 0, 1]
    result = evaluate_classifier(
        y, p, positive_label=1,
        timestamps=list(range(len(y))), temporal_windows=2,
    )
    assert result["confusion"] == {"tp": 3, "tn": 3, "fp": 1, "fn": 1}
    assert result["false_positive_rate"] == pytest.approx(0.25)
    assert "temporal_stability" in result


def test_anomaly_evaluator_threshold_and_metrics():
    result = evaluate_anomaly_detector(
        [0, 0, 1, 1], [0.1, 0.8, 0.9, 0.2], threshold=0.5
    )
    assert result["confusion"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}


def test_reward_composition_is_bounded_and_validated():
    assert compose_reward({"benefit": 20, "cost": -2},
                          {"benefit": 1, "cost": 1}, clip=5) == 5
    with pytest.raises(ValueError):
        compose_reward({"benefit": float("nan")}, {"benefit": 1})


def test_rl_benchmark_is_paired_and_reproducible():
    scenarios = list(range(10))
    report = benchmark_policies(
        scenarios,
        lambda s, seed: float(s) + seed / 1000,
        lambda s, seed: float(s) + seed / 1000 + 1,
        seeds=(1, 2, 3),
        minimum_pairs=30,
        bootstrap_samples=200,
    )
    assert report["paired_runs"] == 30
    assert report["mean_paired_improvement"] == pytest.approx(1.0)
    assert report["candidate_noninferior"] is True


def test_registry_requires_explicit_approval_and_gates(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"candidate model artifact")
    registry = ModelRegistry(tmp_path / "registry.sqlite")
    metrics = {
        "sample_count": 200,
        "f1": 0.91,
        "recall": 0.90,
        "false_positive_rate": 0.02,
        "temporal_stability": {"f1_standard_deviation": 0.03},
    }
    model_id = registry.register_candidate(
        artifact, model_name="detector", version="v1", metrics=metrics
    )
    with pytest.raises(ValueError, match="approver"):
        registry.promote(model_id, approved_by="", reason="test")
    promoted = registry.promote(
        model_id, approved_by="operator", reason="validated",
        gates=PromotionGates(minimum_samples=100),
    )
    assert promoted["status"] == "production"
    assert registry.production_model("detector")["model_id"] == model_id


def test_registry_rejects_tampered_artifact(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"original")
    registry = ModelRegistry(tmp_path / "registry.sqlite")
    model_id = registry.register_candidate(
        artifact, model_name="detector", version="v1",
        metrics={"sample_count": 200, "f1": .9, "recall": .9,
                 "false_positive_rate": .01,
                 "temporal_stability": {"f1_standard_deviation": .01}},
    )
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        registry.promote(model_id, approved_by="operator", reason="test")


def test_live_probe_preserves_missing_metrics_and_never_executes():
    class Collector:
        def collect(self):
            return {
                "timestamp": "2026-11-17T12:00:00+00:00",
                "agent_id": "test-agent",
                "host": "test-host",
                "collection_status": "partial",
                "metrics": {
                    "cpu_percent": 0.0,
                    "memory_percent": 20.0,
                    "disk_percent": 30.0,
                    "network_connections": None,
                    "load_1m": 0.5,
                },
            }

    result = run_live_telemetry_decision_probe(
        Collector(),
        lambda observation: {
            "decision": "review",
            "execution": {"enabled": False},
        },
    )
    assert result["status"] == "passed"
    assert result["metric_availability"]["network_connections"] is False
    assert result["execution_enabled"] is False


def test_live_probe_fails_closed_if_decision_claims_execution_enabled():
    class Collector:
        def collect(self):
            return {"collection_status": "failed", "metrics": {}}

    with pytest.raises(RuntimeError, match="Safety violation"):
        run_live_telemetry_decision_probe(
            Collector(),
            lambda observation: {"action_execution_enabled": True},
        )


def test_chronological_split_keeps_duplicate_text_in_one_partition():
    from Asterion.Intelligence.quality_assurance import chronological_split_rows
    rows = [
        {"timestamp": f"2026-01-{day:02d}T00:00:00+00:00", "text": f"event {day}", "label": "normal" if day % 2 else "anomaly"}
        for day in range(1, 11)
    ]
    rows.append({"timestamp": "2026-01-11T00:00:00+00:00", "text": "event 1", "label": "normal"})
    split = chronological_split_rows(rows)
    assert split["leakage_check"] == "passed"
    train_text = {row["text"] for row in split["train"]}
    validation_text = {row["text"] for row in split["validation"]}
    test_text = {row["text"] for row in split["test"]}
    assert not (train_text & validation_text or train_text & test_text or validation_text & test_text)


def test_generalization_report_flags_large_test_degradation():
    from Asterion.Intelligence.quality_assurance import compare_generalization
    validation = {"f1": .90, "precision": .91, "recall": .89, "false_positive_rate": .02}
    test = {"f1": .70, "precision": .72, "recall": .68, "false_positive_rate": .08}
    report = compare_generalization(validation, test, maximum_f1_drop=.10)
    assert report["within_f1_tolerance"] is False
    assert report["f1_generalization_gap"] == pytest.approx(.20)
