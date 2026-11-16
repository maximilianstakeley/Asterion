"""Validated contracts shared by agents, models, policy, safety, and executor."""
from __future__ import annotations
import math
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

class Observation(StrictModel):
    agent_id: str = Field(min_length=1, max_length=128)
    host: str = Field(min_length=1, max_length=255)
    text: str = Field(min_length=1, max_length=200_000)
    metrics: dict[str, float | int | None]
    timestamp: str | None = None


    @field_validator("metrics", mode="before")
    @classmethod
    def validate_raw_load(cls, value):
        """Reject invalid load values before Pydantic coerces their types."""
        if not isinstance(value, dict):
            return value  # Let Pydantic report the invalid dictionary type.

        load = value.get("load_1m")

        if load is None:
            return value

        if isinstance(load, bool) or not isinstance(load, (int, float)):
            raise ValueError(
                "load_1m must be null or a finite, non-negative number"
            )

        try:
            valid_load = math.isfinite(load) and load >= 0
        except (OverflowError, TypeError, ValueError):
            valid_load = False

        if not valid_load:
            raise ValueError(
                "load_1m must be null or a finite, non-negative number"
            )

        return value


    @model_validator(mode="after")
    def validate_metrics(self):
        allowed = {"cpu_percent", "memory_percent", "disk_percent", "network_connections", "load_1m"}
        unknown = set(self.metrics) - allowed
        if unknown:
            raise ValueError(f"unknown metric keys: {sorted(unknown)}")
        for key in ("cpu_percent", "memory_percent", "disk_percent"):
            value = self.metrics.get(key)
            if value is not None:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0 <= value <= 100
                ):
                    raise ValueError(
                        f"{key} must be null or numeric and between 0 and 100"
                    )

        connections = self.metrics.get("network_connections")
        if connections is not None:
            if (
                isinstance(connections, bool)
                or not isinstance(connections, int)
                or not 0 <= connections <= 10_000_000
            ):
                raise ValueError(
                    "network_connections must be null or an integer "
                    "between 0 and 10000000"
                )
        load = self.metrics.get("load_1m")
        if load is not None:
            if (
                    isinstance(load, bool)
                    or not isinstance(load, (int, float))
                    or not math.isfinite(load)
                    or load < 0
            ):
                raise ValueError(
                    "load_1m must be null or a finite, non-negative number"
                )
        return self

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

class Evidence(StrictModel):
    source: str = Field(min_length=1, max_length=128)
    category: str = Field(min_length=1, max_length=128)
    severity: Literal["info", "low", "medium", "high", "critical"]
    description: str = Field(min_length=1, max_length=2000)
    details: dict[str, Any] = Field(default_factory=dict)

class CandidateAction(StrictModel):
    action: Literal["collect_diagnostics", "notify_operator", "reduce_noncritical_workload", "review_suspicious_activity"]
    rationale: str = Field(min_length=1, max_length=2000)
    risk: float = Field(ge=0, le=1)
    source_evidence_ids: list[str] = Field(default_factory=list)

class PolicyOutput(StrictModel):
    selected_action: str
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=2000)
    policy_name: str = "deterministic_baseline"

class SafetyVerdict(StrictModel):
    allowed: bool
    executed: bool = False
    verdict: Literal["blocked", "dry_run_allowed", "approval_required", "allowed_but_execution_disabled"]
    reasons: list[str] = Field(default_factory=list)

class DecisionRecord(StrictModel):
    decision_id: str
    timestamp: str
    agent_id: str
    host: str
    observation: dict[str, Any]
    model_outputs: dict[str, Any]
    evidence: list[Evidence]
    candidate_actions: list[CandidateAction]
    policy_output: PolicyOutput
    safety_verdict: SafetyVerdict
    executor_result: dict[str, Any]
