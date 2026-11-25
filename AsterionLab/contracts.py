"""Versioned schemas for resource telemetry and bounded simulation proposals."""
from dataclasses import dataclass, fields
from hashlib import sha256
import json
import math
import re

FEATURE_VERSION = "resource-cpu-v1"
SHIELD_VERSION = "resource-simulator-v1"
SOURCES = {"synthetic", "recorded"}
STATUSES = {"measured", "missing", "stale", "invalid"}
SIM_ACTIONS = ("observe", "reduce_background_load")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def exact(data, expected):
    if type(data) is not dict or set(data) != set(expected):
        raise ValueError(f"Expected fields {', '.join(sorted(expected))}")


def identifier(value):
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value):
        raise ValueError("Invalid identifier")
    return value


def number(value, low, high):
    if type(value) not in (int, float) or not low <= value <= high:
        raise ValueError(f"Expected finite number in [{low}, {high}]")
    return float(value)


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"Expected integer in [{low}, {high}]")
    return value


def probability_map(value, minimum=0.0):
    exact(value, SIM_ACTIONS)
    result = {action: number(value[action], minimum, 1) for action in SIM_ACTIONS}
    if not math.isclose(sum(result.values()), 1, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Action probabilities must sum to one")
    return result


@dataclass(frozen=True)
class Observation:
    observation_id: str
    host_id: str
    observed_at_s: int
    received_at_s: int
    collector_version: str
    clock: str
    clock_uncertainty_ms: int
    source: str
    cpu_status: str
    cpu_percent: float | None

    def __post_init__(self):
        for value in (self.observation_id, self.host_id, self.collector_version):
            identifier(value)
        integer(self.observed_at_s, 0, 2**53 - 1)
        integer(self.received_at_s, self.observed_at_s, 2**53 - 1)
        integer(self.clock_uncertainty_ms, 0, 60_000)
        if self.clock != "utc_unix_s" or self.source not in SOURCES or self.cpu_status not in STATUSES:
            raise ValueError("Invalid observation source, clock or status")
        if self.cpu_status == "measured":
            number(self.cpu_percent, 0, 100)
        elif self.cpu_percent is not None:
            raise ValueError("Unusable reading must have null value")

    @classmethod
    def parse(cls, data):
        exact(data, {field.name for field in fields(cls)})
        return cls(**data)


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    target: str
    action: str
    evidence_ids: tuple[str, ...]
    observation_digest: str
    model_version: str
    policy_version: str
    evaluation_mode: str
    preconditions: tuple[str, ...]
    expected_effect: str
    risk: str
    reversible: bool
    timeout_s: int
    required_authorization: str
    verification_method: str
    rollback_method: str

    def __post_init__(self):
        for value in (self.proposal_id, self.target, self.model_version, self.policy_version):
            identifier(value)
        if not isinstance(self.observation_digest, str) or not re.fullmatch(r"[a-f0-9]{64}", self.observation_digest):
            raise ValueError("Missing observation digest")
        if type(self.evidence_ids) is not tuple or not 1 <= len(self.evidence_ids) <= 32:
            raise ValueError("Proposal needs evidence references")
        for item in self.evidence_ids:
            identifier(item)
        for field, required in action_rules(self.action).items():
            if type(getattr(self, field)) is not type(required) or getattr(self, field) != required:
                raise ValueError(f"Invalid action safety field: {field}")
        if self.evaluation_mode != "simulation":
            raise ValueError("Only simulation proposals are accepted")

    @classmethod
    def parse(cls, data):
        exact(data, {field.name for field in fields(cls)})
        data = dict(data)
        for key in ("evidence_ids", "preconditions"):
            if type(data[key]) is not list:
                raise ValueError(f"{key} must be a list")
            data[key] = tuple(data[key])
        return cls(**data)


def action_rules(action):
    if action not in SIM_ACTIONS:
        raise ValueError("Unsupported simulation action")
    rules = {
        "preconditions": ("target_matches", "fresh_measured_cpu"),
        "expected_effect": "preserve_virtual_state",
        "risk": "none",
        "reversible": True,
        "timeout_s": 10,
        "required_authorization": "human_simulation_operator",
        "verification_method": "compare_virtual_state",
        "rollback_method": "restore_virtual_snapshot",
    }
    if action == "reduce_background_load":
        rules.update(
            preconditions=rules["preconditions"] + ("cpu_at_least_60_percent",),
            expected_effect="lower_virtual_cpu_by_at_least_5_percentage_points",
            risk="low",
        )
    return rules
