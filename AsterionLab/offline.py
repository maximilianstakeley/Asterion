"""Conservative contextual bandits and evaluation of previously logged decisions."""
from collections import defaultdict
from dataclasses import asdict, dataclass
import math
import random
from statistics import mean

from .contracts import (
    FEATURE_VERSION, SIM_ACTIONS, Observation, digest, exact, identifier, integer, number, probability_map,
)
from .simulation import POLICY_VERSION, baseline

EVALUATOR_VERSION = "host-grouped-ope-v1"
MIN_PROPENSITY = 0.05
MIN_TEST_HOSTS = 8
MIN_EFFECTIVE_SIZE = 30
ACTION = SIM_ACTIONS[1]


def admissible(obs):
    return SIM_ACTIONS if obs.cpu_percent >= 60 else SIM_ACTIONS[:1]


@dataclass(frozen=True)
class Context:
    context: tuple[Observation, ...]

    @property
    def current(self):
        return self.context[-1]


def parse_history(raw_history):
    if type(raw_history) is not list or not 1 <= len(raw_history) <= 32:
        raise ValueError("Expected 1 to 32 telemetry observations")
    history = tuple(Observation.parse(obs) for obs in raw_history)
    current = history[-1]
    if (current.cpu_status != "measured" or current.clock_uncertainty_ms > 1000
            or current.received_at_s - current.observed_at_s > 15):
        raise ValueError("The final observation must be a fresh measured CPU reading")
    if any(obs.host_id != current.host_id or obs.source != current.source
           or obs.collector_version != current.collector_version for obs in history):
        raise ValueError("Mixed host, source or collector in window")
    if len({obs.observation_id for obs in history}) != len(history):
        raise ValueError("Repeated observation in window")
    if any(obs.received_at_s > current.received_at_s for obs in history):
        raise ValueError("An observation arrived after the decision")
    if any(a.observed_at_s >= b.observed_at_s for a, b in zip(history, history[1:])):
        raise ValueError("Observations must be strictly ordered")
    return Context(history)


@dataclass(frozen=True)
class Incident:
    incident_id: str
    behavior_policy_version: str
    context: tuple[Observation, ...]
    action: str
    behavior_probabilities: dict
    label_end_s: int
    fault: bool
    success: bool
    recovery_s: float | None
    unsafe: bool
    rollback_attempted: bool
    rollback_ok: bool | None
    cost: float

    @classmethod
    def parse(cls, record):
        exact(record, set(cls.__dataclass_fields__))
        data = dict(record)
        identifier(data["incident_id"])
        identifier(data["behavior_policy_version"])
        context = parse_history(data["context"])
        data["context"] = context.context
        current = context.current
        allowed = admissible(current)
        probabilities = probability_map(data["behavior_probabilities"])
        if data["action"] not in allowed:
            raise ValueError("Logged action is outside the simulation action boundary")
        for action, propensity in probabilities.items():
            if action in allowed and propensity < MIN_PROPENSITY:
                raise ValueError("Insufficient logging support for an admissible action")
            if action not in allowed and propensity != 0:
                raise ValueError("Inadmissible action must have probability zero")
        data["behavior_probabilities"] = probabilities
        end = integer(data["label_end_s"], current.received_at_s, current.received_at_s + 86_400)
        for key in ("fault", "success", "unsafe", "rollback_attempted"):
            if type(data[key]) is not bool:
                raise ValueError(f"Expected boolean for {key}")
        if data["rollback_attempted"]:
            if type(data["rollback_ok"]) is not bool:
                raise ValueError("Rollback attempt needs a verified outcome")
        elif data["rollback_ok"] is not None:
            raise ValueError("Rollback result must be null without an attempt")
        if data["fault"] and data["success"]:
            data["recovery_s"] = number(data["recovery_s"], 0, end - current.received_at_s)
        elif data["recovery_s"] is not None:
            raise ValueError("Recovery time is defined only for recovered faults")
        data["cost"] = number(data["cost"], 0, 1)
        return cls(**data)

    @property
    def current(self):
        return self.context[-1]

    @property
    def reward(self):
        return float(self.success) - 0.25 * self.cost - float(self.unsafe)


def split(incidents, cutoff_s, holdout_hosts):
    if not holdout_hosts:
        raise ValueError("Declare holdout host IDs before fitting")
    if len({row.incident_id for row in incidents}) != len(incidents):
        raise ValueError("Duplicate incident ID")
    if len({row.current.source for row in incidents}) != 1:
        raise ValueError("Evaluate synthetic and recorded sources separately")
    # Assign every reused observation to one consistent content record.
    observations = {}
    for row in incidents:
        for obs in row.context:
            if observations.setdefault(obs.observation_id, obs) != obs:
                raise ValueError("Conflicting observation ID")
    train = [row for row in incidents if row.current.host_id not in holdout_hosts and row.label_end_s < cutoff_s]
    test = [row for row in incidents if row.current.host_id in holdout_hosts and row.context[0].observed_at_s >= cutoff_s]
    if not train or not test:
        raise ValueError("Split needs labeled pre-cutoff training and later held-out hosts")
    return train, test


def context_key(row, temporal):
    current = row.current.cpu_percent
    band = min(int(current // 10), 9)
    if not temporal:
        return str(band)
    window = row.context
    # Use a short CPU history only if provenance and timing are adequate.
    if (len(window) < 3 or any(obs.cpu_status != "measured" or obs.clock_uncertainty_ms > 1000 for obs in window)
            or window[-1].observed_at_s - window[0].observed_at_s > 60
            or any(b.observed_at_s - a.observed_at_s > 20 for a, b in zip(window, window[1:]))):
        return f"{band}:unaligned"
    elapsed = window[-1].observed_at_s - window[0].observed_at_s
    slope = (current - window[0].cpu_percent) / elapsed
    trend = "rising" if slope >= 0.5 else "falling" if slope <= -0.5 else "flat"
    return f"{band}:{trend}"


def fit(rows, temporal):
    by_host = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_host[(context_key(row, temporal), row.action)][row.current.host_id].append(row)
    result = {}
    for (context, action), hosts in sorted(by_host.items()):
        rewards = [mean(row.reward for row in records) for records in hosts.values()]
        successes = [mean(row.success for row in records) for records in hosts.values()]
        n = len(hosts)
        center = mean(rewards)
        # A deliberately conservative policy that treats hosts as units.
        radius = 2.25 * math.sqrt(math.log(40) / (2*n))
        result.setdefault(context, {})[action] = {
            "reward": center, "lower": max(-1.25, center - radius),
            "upper": min(1, center + radius), "hosts": n,
            "success_probability": (sum(successes) + 1) / (n + 2),
        }
    return result


def recommend(row, model, temporal):
    reference = baseline(row.current.cpu_percent)
    if temporal and context_key(row, temporal).endswith(":unaligned"):
        return reference
    alternatives = model.get(context_key(row, temporal), {})
    if set(admissible(row.current)) != set(SIM_ACTIONS) or not all(
        alternatives.get(action, {}).get("hosts", 0) >= 5 for action in SIM_ACTIONS
    ):
        return reference
    challenger = next(action for action in SIM_ACTIONS if action != reference)
    if alternatives[challenger]["lower"] > alternatives[reference]["upper"] + 0.02:
        return challenger
    return reference


def estimate(rows, predictions, nuisance, temporal=False):
    correction = []
    weights = []
    brier = []
    for row, selected in zip(rows, predictions, strict=True):
        propensity = row.behavior_probabilities[selected]
        weight = (1 / propensity) if selected == row.action else 0
        cell = nuisance.get(context_key(row, temporal), {})
        logged = cell.get(row.action, {})
        predicted = cell.get(selected, {})
        correction.append(predicted.get("reward", 0) + weight * (row.reward - logged.get("reward", 0)))
        weights.append(weight)
        brier.append((predicted.get("success_probability", 0.5) - row.success) ** 2)
    total = sum(weights)

    def weighted(value, include=lambda row: True):
        denominator = sum(w for w, row in zip(weights, rows) if include(row))
        if not denominator:
            return None
        return sum(w * value(row) for w, row in zip(weights, rows) if include(row)) / denominator

    return {
        "ips_reward": mean(w * row.reward for w, row in zip(weights, rows)),
        "snips_reward": weighted(lambda row: row.reward),
        "doubly_robust_reward": mean(correction),
        "effective_sample_size": total**2 / sum(w*w for w in weights) if total else 0,
        "matched_rows": sum(w > 0 for w in weights),
        "matched_hosts": len({row.current.host_id for row, weight in zip(rows, weights) if weight}),
        "task_success": weighted(lambda row: row.success),
        "mean_recovery_s": weighted(lambda row: row.recovery_s, lambda row: row.recovery_s is not None),
        "false_alarm_rate": weighted(lambda row: row.action != "observe", lambda row: not row.fault),
        "unsafe_action_rate": weighted(lambda row: row.unsafe),
        "rollback_success": weighted(lambda row: row.rollback_ok, lambda row: row.rollback_attempted),
        "normalized_resource_cost": weighted(lambda row: row.cost),
        "success_brier": sum(w * b for w, b in zip(weights, brier)) / total if total else None,
    }, correction


def paired_host_interval(rows, first, second, seed=7, replicates=1000):
    groups = defaultdict(list)
    for row, a, b in zip(rows, first, second, strict=True):
        groups[row.current.host_id].append(a - b)
    if len(groups) < MIN_TEST_HOSTS:
        return None
    clusters = [(sum(groups[host]), len(groups[host])) for host in sorted(groups)]
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sample = rng.choices(clusters, k=len(clusters))
        estimates.append(sum(total for total, _ in sample) / sum(count for _, count in sample))
    estimates.sort()
    return [estimates[int(replicates * 0.025)], estimates[int(replicates * 0.975) - 1]]


def evaluate(incidents, cutoff_s, holdout_hosts):
    train, test = split(incidents, cutoff_s, holdout_hosts)
    flat_model = fit(train, temporal=False)
    time_model = fit(train, temporal=True)
    baseline_actions = [baseline(row.current.cpu_percent) for row in test]
    flat_actions = [recommend(row, flat_model, False) for row in test]
    time_actions = [recommend(row, time_model, True) for row in test]
    baseline_report, reference = estimate(test, baseline_actions, flat_model)
    observe_report, _ = estimate(test, ["observe"] * len(test), flat_model)
    flat_report, challenger = estimate(test, flat_actions, flat_model)
    # Use the same nuisance model for all DR comparisons, so changing only
    # the reward estimator cannot masquerade as improvement in the policy.
    time_report, temporal = estimate(test, time_actions, flat_model)
    interval = paired_host_interval(test, challenger, reference)
    temporal_interval = paired_host_interval(test, temporal, challenger)
    temporal_baseline_interval = paired_host_interval(test, temporal, reference)
    data_hash = digest([asdict(row) for row in sorted(incidents, key=lambda row: row.incident_id)])
    artifact = {"format": "asterion-bandit-v1", "feature_version": FEATURE_VERSION, "dataset_sha256": data_hash,
                "baseline_version": POLICY_VERSION, "snapshot": flat_model, "temporal_ablation": time_model}
    report = {
        "evaluator_version": EVALUATOR_VERSION,
        "feature_version": FEATURE_VERSION, "dataset_sha256": data_hash,
        "artifact_sha256": digest(artifact), "source": train[0].current.source,
        "split": {"cutoff_s": cutoff_s, "train_rows": len(train), "test_rows": len(test),
                  "excluded_rows": len(incidents)-len(train)-len(test),
                  "train_hosts": sorted({row.current.host_id for row in train}),
                  "test_hosts": sorted({row.current.host_id for row in test})},
        "baseline": baseline_report, "always_observe_baseline": observe_report,
        "snapshot_challenger": flat_report, "temporal_challenger": time_report,
        "test_label_end_s": max(row.label_end_s for row in test),
        "snapshot_minus_baseline_dr_95": interval,
        "temporal_minus_snapshot_dr_95": temporal_interval,
        "temporal_minus_baseline_dr_95": temporal_baseline_interval,
        "snapshot_policy_changes": sum(a != b for a, b in zip(flat_actions, baseline_actions)),
        "temporal_policy_changes": sum(a != b for a, b in zip(time_actions, flat_actions)),
        "snapshot_has_measured_advantage": bool(interval and interval[0] > 0.02
            and flat_report["effective_sample_size"] >= MIN_EFFECTIVE_SIZE
            and baseline_report["effective_sample_size"] >= MIN_EFFECTIVE_SIZE
            and min(flat_report["matched_hosts"], baseline_report["matched_hosts"]) >= MIN_TEST_HOSTS),
        "temporal_has_measured_advantage": bool(temporal_interval and temporal_interval[0] > 0.02
            and temporal_baseline_interval and temporal_baseline_interval[0] > 0.02
            and min(time_report["effective_sample_size"], flat_report["effective_sample_size"],
                    baseline_report["effective_sample_size"]) >= MIN_EFFECTIVE_SIZE
            and min(time_report["matched_hosts"], flat_report["matched_hosts"],
                    baseline_report["matched_hosts"]) >= MIN_TEST_HOSTS),
    }
    return artifact, report
