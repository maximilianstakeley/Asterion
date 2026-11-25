"""Versioned model registry with authenticated attestations and simulation rollout."""
from dataclasses import asdict
import hmac
from pathlib import Path
import time
from uuid import uuid4

from .contracts import digest, exact, identifier
from .ledger import sign
from .offline import MIN_EFFECTIVE_SIZE, MIN_TEST_HOSTS, estimate, evaluate, paired_host_interval, recommend
from .simulation import baseline

STAGES = ("candidate", "shadow", "canary", "simulation_active")
CANARY_PERCENT = 10


def canary_member(host):
    return int(digest(host)[:8], 16) % 100 < CANARY_PERCENT


class Registry:
    def __init__(self, ledger, reviewer_id=None, reviewer_key=None):
        self.ledger = ledger
        self.reviewer_id = reviewer_id
        self.reviewer_key = reviewer_key
        if reviewer_id is not None or reviewer_key is not None:
            identifier(reviewer_id)
            sign(reviewer_key, "key-check")
            if reviewer_key == ledger.key:
                raise ValueError("Use a separate release reviewer key")

    def _require_reviewer(self):
        if self.reviewer_key is None:
            raise ValueError("A configured reviewer credential is required")

    def admit(self, incidents, cutoff_s, holdout_hosts, variant):
        if variant not in {"snapshot", "temporal"}:
            raise ValueError("Unknown model variant")
        artifact, report = evaluate(incidents, cutoff_s, holdout_hosts)
        release = {"artifact": artifact, "report": report, "variant": variant}
        version = "bandit-" + digest(release)
        with self.ledger.transaction() as conn:
            events, _ = self.ledger.verified_events(conn)
            if version not in events:
                self.ledger.append(conn, version, "model_candidate", release)
        return version, report

    @staticmethod
    def _release(events, version):
        if events.get(version, (None,))[0] != "model_candidate":
            raise ValueError("Unknown model version")
        return events[version][1]

    @staticmethod
    def _state(events, version):
        states = [value["stage"] for kind, value in events.values()
                  if kind == "release_stage" and value["model_version"] == version]
        return states[-1] if states else "candidate"

    @staticmethod
    def _assigned(events):
        return [value["model_version"] for kind, value in events.values() if kind == "model_candidate_assignment"
                and Registry._state(events, value["model_version"]) in ("shadow", "canary", "simulation_active")]

    @staticmethod
    def _assessments(events, version, stage):
        return [value for kind, value in events.values() if kind == "release_monitor"
                and value["model_version"] == version and value["stage"] == stage]

    def monitor(self, version, incidents):
        with self.ledger.transaction() as conn:
            events, _ = self.ledger.verified_events(conn)
            entry = self._release(events, version)
            stage = self._state(events, version)
            if stage not in STAGES[1:]:
                raise ValueError("Assign shadow before monitoring")
            report = entry["report"]
            if not incidents or len({row.incident_id for row in incidents}) != len(incidents):
                raise ValueError("Monitoring needs distinct incidents")
            previous = [value for kind, value in events.values() if kind == "release_monitor"
                        and value["model_version"] == version]
            cutoff = max([report["test_label_end_s"], *(value["label_end_s"] for value in previous)])
            known_hosts = set(report["split"]["train_hosts"] + report["split"]["test_hosts"])
            if any(row.current.source != "recorded" or row.context[0].observed_at_s <= cutoff
                   or row.current.host_id in known_hosts for row in incidents):
                raise ValueError("Monitoring requires later recorded incidents from new hosts")
            if stage == "canary":
                incidents = [row for row in incidents if canary_member(row.current.host_id)]
            if len({row.current.host_id for row in incidents}) < MIN_TEST_HOSTS:
                raise ValueError("Too few monitoring hosts in this rollout cohort")
            is_temporal = entry["variant"] == "temporal"
            artifact = entry["artifact"]
            model = artifact["temporal_ablation" if is_temporal else "snapshot"]
            choices = [recommend(row, model, is_temporal) for row in incidents]
            references = [baseline(row.current.cpu_percent) for row in incidents]
            metrics, improved = estimate(incidents, choices, artifact["snapshot"])
            base, original = estimate(incidents, references, artifact["snapshot"])
            interval = paired_host_interval(incidents, improved, original)
            passed = bool(interval and interval[0] >= -0.02
                and min(base["effective_sample_size"], metrics["effective_sample_size"]) >= MIN_EFFECTIVE_SIZE
                and min(base["matched_hosts"], metrics["matched_hosts"]) >= MIN_TEST_HOSTS
                and metrics["unsafe_action_rate"] == 0)
            value = {"model_version": version, "stage": stage, "passed": passed, "metrics": metrics,
                     "baseline_metrics": base, "candidate_minus_baseline_dr_95": interval,
                     "dataset_sha256": digest([asdict(row) for row in sorted(incidents, key=lambda row: row.incident_id)]),
                     "label_end_s": max(row.label_end_s for row in incidents)}
            self.ledger.append(conn, "monitor-" + uuid4().hex, "release_monitor", value)
            if not passed:
                self._write_stage(conn, version, "rolled_back", "monitoring_threshold_failed")
        return value

    def transition(self, version, requested, envelope):
        self._require_reviewer()
        with self.ledger.transaction() as conn:
            events, _ = self.ledger.verified_events(conn)
            entry = self._release(events, version)
            stage = self._state(events, version)
            allowed = STAGES[STAGES.index(stage) + 1] if stage in STAGES[:-1] else None
            if requested != allowed or requested is None:
                raise ValueError("Invalid release transition")
            exact(envelope, {"body", "signature"})
            body = envelope["body"]
            exact(body, {"model_version", "requested", "reviewer", "regression_report_sha256",
                         "security_report_sha256", "approved", "expires_s"})
            if (body["model_version"] != version or body["requested"] != requested or body["reviewer"] != self.reviewer_id
                    or body["approved"] is not True or type(body["expires_s"]) is not int
                    or not int(time.time()) <= body["expires_s"] <= int(time.time()) + 3600):
                raise ValueError("Reviewer attestation is invalid or expired")
            for field in ("regression_report_sha256", "security_report_sha256"):
                value = body[field]
                if type(value) is not str or len(value) != 64 or any(c not in "abcdef0123456789" for c in value):
                    raise ValueError("Evidence file digests required")
            signature = envelope["signature"]
            if type(signature) is not str or not signature.isascii() or not hmac.compare_digest(
                signature, sign(self.reviewer_key, body)
            ):
                raise ValueError("Reviewer signature does not verify")
            report = entry["report"]
            metrics = report["snapshot_challenger" if entry["variant"] == "snapshot" else "temporal_challenger"]
            if (report["source"] != "recorded" or not report[f'{entry["variant"]}_has_measured_advantage']
                    or metrics["unsafe_action_rate"] != 0):
                raise ValueError("Release needs recorded evidence of advantage and zero observed unsafe actions")
            if stage == "candidate":
                if self._assigned(events):
                    raise ValueError("Roll back the assigned model before starting another release")
                self.ledger.append(conn, "assignment-" + uuid4().hex, "model_candidate_assignment",
                                   {"model_version": version})
            else:
                checks = self._assessments(events, version, stage)
                if not checks or not checks[-1]["passed"]:
                    raise ValueError(f"Passing {stage} monitoring evidence is required")
            self.ledger.append(conn, "review-" + uuid4().hex, "release_attestation", envelope)
            self._write_stage(conn, version, requested, "reviewed_release")
        return {"model_version": version, "stage": requested}

    def _write_stage(self, conn, version, stage, reason):
        self.ledger.append(conn, "stage-" + uuid4().hex, "release_stage", {
            "model_version": version, "stage": stage, "reason": reason, "live_execution": False,
        })

    def rollback(self, version):
        self._require_reviewer()
        with self.ledger.transaction() as conn:
            events, _ = self.ledger.verified_events(conn)
            self._release(events, version)
            if self._state(events, version) not in STAGES[1:]:
                raise ValueError("No rollout to roll back")
            self._write_stage(conn, version, "rolled_back", "manual_baseline_restore")
        return {"model_version": version, "stage": "rolled_back", "fallback": "cpu-threshold-v1"}

    @staticmethod
    def route(events, context):
        reference = baseline(context.current.cpu_percent)
        selected = Registry._assigned(events)
        if not selected:
            return None, reference, None
        version = selected[0]
        entry = Registry._release(events, version)
        stage = Registry._state(events, version)
        temporal = entry["variant"] == "temporal"
        model = entry["artifact"]["temporal_ablation" if temporal else "snapshot"]
        proposed = recommend(context, model, temporal)
        is_enabled = stage == "simulation_active" or (stage == "canary" and canary_member(context.current.host_id))
        return version if is_enabled else None, proposed if is_enabled else reference, {
            "model_version": version, "stage": stage, "candidate_action": proposed, "baseline_action": reference,
        }

    def sign_review(self, version, requested, regression_path, security_path):
        from hashlib import sha256

        self._require_reviewer()
        # The reviewer explicitly attests both reports after reviewing their actual contents.
        paths = [Path(regression_path), Path(security_path)]
        if paths[0].resolve() == paths[1].resolve():
            raise ValueError("Use separate regression and security reports")
        evidence = [path.read_bytes() for path in paths]
        if not all(evidence):
            raise ValueError("Review evidence files must be nonempty")
        body = {
            "model_version": version, "requested": requested, "reviewer": self.reviewer_id,
            "regression_report_sha256": sha256(evidence[0]).hexdigest(),
            "security_report_sha256": sha256(evidence[1]).hexdigest(),
            "approved": True, "expires_s": int(time.time()) + 900,
        }
        return {"body": body, "signature": sign(self.reviewer_key, body)}
