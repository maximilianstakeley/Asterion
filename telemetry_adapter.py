"""Translate Asterion agent CPU telemetry into Action Lab observations."""
from dataclasses import dataclass
from datetime import datetime, timezone

from .contracts import Observation, digest, identifier, integer, number

MAX_AGE_S = 15


@dataclass(frozen=True)
class BridgedObservation:
    observation: Observation
    provenance: dict


def parse_time(value: str) -> datetime:
    if type(value) is not str:
        raise ValueError("Telemetry timestamps must be ISO 8601 strings")
    stamp = datetime.fromisoformat(value)
    if stamp.utcoffset() is None or stamp.timestamp() < 0:
        raise ValueError(
            "Telemetry timestamps must include an offset and be after the Unix epoch"
        )
    return stamp.astimezone(timezone.utc)


def cpu_reading(payload: dict) -> tuple[str, float | None, str]:
    metrics = payload.get("metrics")
    statuses = payload.get("metric_status", {})
    if not isinstance(metrics, dict) or not isinstance(statuses, dict):
        return "invalid", None, "malformed_metrics_or_status"

    cpu = metrics.get("cpu_percent")
    outer_status = statuses.get("cpu_percent")
    if isinstance(outer_status, dict):
        outer_status = outer_status.get("status")

    if isinstance(cpu, dict):
        value = cpu.get("value")
        inner_status = cpu.get("status")
        if (
            outer_status is not None
            and inner_status is not None
            and outer_status != inner_status
        ):
            return "invalid", None, "conflicting_cpu_status"
        status = inner_status if inner_status is not None else outer_status
        if value is not None and cpu.get("unit") != "%":
            return "invalid", None, "invalid_cpu_unit"
    else:
        value, status = cpu, outer_status

    if status == "stale":
        return "stale", None, "collector_reported_stale"
    if status == "invalid":
        return "invalid", None, "collector_reported_invalid"
    if status in ("missing", "unavailable") or (status is None and value is None):
        if value is None:
            return "missing", None, "cpu_unavailable"
        return "invalid", None, "unavailable_cpu_has_value"
    if status != "ok" or payload.get("collection_status") == "failed":
        return "invalid", None, "cpu_status_unverified_or_conflicting"

    try:
        return "measured", number(value, 0, 100), "cpu_measured"
    except ValueError:
        return "invalid", None, "invalid_cpu_percentage"


def adapt_telemetry(
    payload: dict,
    *,
    collector_version: str,
    clock_uncertainty_ms: int,
    source_id: str,
    now: datetime,
) -> BridgedObservation:
    """Use CPU sample time when available and preserve receipt provenance."""
    if type(payload) is not dict:
        raise ValueError("Expected an Asterion telemetry object")
    if now.utcoffset() is None:
        raise ValueError("Adapter time must be timezone-aware")

    integer(clock_uncertainty_ms, 0, 60_000)
    agent = identifier(payload.get("agent_id"))
    host = identifier(payload.get("host"))

    recorded_receipt = payload.get("received_at")
    received = now if recorded_receipt is None else parse_time(recorded_receipt)
    metrics = payload.get("metrics")
    cpu = metrics.get("cpu_percent") if isinstance(metrics, dict) else None
    sampled_at = cpu.get("sampled_at") if isinstance(cpu, dict) else None
    observed = parse_time(
        payload.get("timestamp") if sampled_at is None else sampled_at
    )
    if not observed <= received <= now:
        raise ValueError(
            "Telemetry timestamps must be ordered as sample, receipt, then adapter time"
        )

    status, value, reason = cpu_reading(payload)
    age_s = (now - observed).total_seconds()
    if status == "measured" and age_s > MAX_AGE_S:
        status, value, reason = "stale", None, "expired_cpu_sample"

    observation = {
        "host_id": host,
        "observed_at_s": int(observed.timestamp()),
        "received_at_s": int(received.timestamp()),
        "collector_version": collector_version,
        "clock": "utc_unix_s",
        "clock_uncertainty_ms": clock_uncertainty_ms,
        "source": "recorded",
        "cpu_status": status,
        "cpu_percent": value,
    }
    # A re-read can produce a new derived observation when the source ages.
    event_id = "cpu-" + digest([source_id, agent, observation])
    provenance = {
        "source_id": source_id,
        "agent_id": agent,
        "telemetry_id": payload.get("id"),
        "collection_status": payload.get("collection_status"),
        "cpu_sample_time": observed.isoformat(),
        "received_at": received.isoformat(),
        "receipt_origin": "adapter" if recorded_receipt is None else "telemetry_store",
        "cpu_status_reason": reason,
    }
    return BridgedObservation(Observation(event_id, **observation), provenance)