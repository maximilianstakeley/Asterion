
"""Asterion defensive monitoring API.

Telemetry ingestion and retrieval are authenticated.
Action execution remains disabled.
"""

import json
import secrets
from datetime import datetime, timezone
from typing import Any
from pydantic import ValidationError
from fastapi import (
    Body,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,)
from .configuration import API_TOKEN, MODEL_DIR
from .service import analyze
from .stage2_detection import detect_time_window, detect_multivariate
from .network_security import NetworkSecurityRecord
from contextlib import asynccontextmanager
from .database import (
    add_decision_audit,
    add_experience,
    add_telemetry_observation,
    get_counts,
    get_decision_audit,
    get_findings,
    get_telemetry_timeseries,
    add_network_security_record,
    get_network_security_records,
    init_db,
    now,
)
from .schemas import AnalyzeRequest, FeedbackRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize persistent storage when the API starts."""
    init_db()
    yield


app = FastAPI(
    title="Asterion API",
    version="1.2.0",
    description=(
        "Defensive monitoring API. "
        "Telemetry access requires authentication. "
        "Action execution is disabled."
    ),
    lifespan=lifespan,)


def auth(authorization: str | None = Header(default=None)) -> bool:
    """Require a valid bearer token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    supplied_token = authorization[len("Bearer "):].strip()

    if not supplied_token or not secrets.compare_digest(
        supplied_token, API_TOKEN
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},)

    return True


@app.post("/v1/detection/time-window", dependencies=[Depends(auth)])
def run_time_window_detection(payload: dict = Body(...)):
    """Run Stage 2.1 on a supplied host telemetry window."""
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise HTTPException(
            status_code=422,
            detail="samples must be a non-empty list",)

    try:
        findings = detect_time_window(
            samples,
            window_seconds=float(payload.get("window_seconds", 300)),
            min_samples=int(payload.get("min_samples", 3)),
            max_gap_seconds=float(payload.get("max_gap_seconds", 120)),
            persistence_fraction=float(
                payload.get("persistence_fraction", 0.60)
            ),
            repeated_anomaly_minimum=int(
                payload.get("repeated_anomaly_minimum", 3)
            ),)

    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "detector": "time_window_v1",
        "finding_count": len(findings),
        "findings": findings,
        "interpretation": (
            "Findings indicate anomalies requiring investigation; "
            "they do not establish malicious activity."
        ),}


@app.post("/v1/detection/multivariate", dependencies=[Depends(auth)])
def run_multivariate_detection(payload: dict = Body(...)):
    """Run Stage 2.2 using a supplied sample set and optional baseline."""
    samples = payload.get("samples")
    baseline_samples = payload.get("baseline_samples")

    if not isinstance(samples, list) or not samples:
        raise HTTPException(
            status_code=422,
            detail="samples must be a non-empty list",
        )

    if baseline_samples is not None and not isinstance(baseline_samples, list):
        raise HTTPException(
            status_code=422,
            detail="baseline_samples must be a list when supplied",
        )

    features = payload.get("features")
    if features is not None:
        if (
            not isinstance(features, list)
            or not features
            or not all(isinstance(x, str) and x.strip() for x in features)
        ):
            raise HTTPException(
                status_code=422,
                detail="features must be a non-empty list of metric names",
            )
        features = tuple(features)

    try:
        findings = detect_multivariate(
            samples,
            baseline_samples=baseline_samples,
            features=features or (
                "cpu_percent",
                "memory_percent",
                "disk_percent",
                "load_1m",
                "network_bytes_per_sec",
                "network_packets_per_sec",
            ),
            minimum_features=int(payload.get("minimum_features", 3)),
            score_threshold=float(payload.get("score_threshold", 3.5)),
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "detector": "multivariate_robust_score_v1",
        "finding_count": len(findings),
        "findings": findings,
        "interpretation": (
            "Joint resource deviation is not a calibrated intrusion "
            "probability and is not independent proof of an attack."
        ),
    }



@app.post("/v1/network-security/events", dependencies=[Depends(auth)])
def ingest_network_security_record(payload: dict = Body(...)):
    """Validate and persist one network event or flow record."""
    try:
        record = NetworkSecurityRecord.model_validate(payload)
        record_dict = record.model_dump(mode="json")
        row_id = add_network_security_record(record_dict)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_url=False),
        ) from exc
    except Exception as exc:
        # Do not silently accept or overwrite duplicate record IDs.
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(
                status_code=409,
                detail="record_id already exists",
            ) from exc
        raise

    return {
        "status": "accepted",
        "database_id": row_id,
        "record_id": record.record_id,
        "fingerprint": record.fingerprint(),
        "interpretation": (
            "Stored as network telemetry evidence; acceptance does not "
            "mean the record is malicious."
        ),
    }


@app.get("/v1/network-security/events", dependencies=[Depends(auth)])
def list_network_security_records(
    source_host: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = Query(default=500, ge=1, le=5000),
):
    """Retrieve stored network security records."""
    start_value = _normalize_timestamp(start)
    end_value = _normalize_timestamp(end)

    if start_value and end_value and start_value > end_value:
        raise HTTPException(
            status_code=400,
            detail="start must not be later than end",
        )

    return {
        "records": get_network_security_records(
            source_host=source_host,
            start_timestamp=start_value,
            end_timestamp=end_value,
            limit=limit,
        )
    }


def _normalize_timestamp(value: datetime | None) -> str | None:
    """Normalize a timezone-aware datetime query parameter to UTC."""
    if value is None:
        return None

    if value.tzinfo is None or value.utcoffset() is None:
        raise HTTPException(
            status_code=400,
            detail="Time filters must include a timezone, e.g. +00:00",
        )

    return value.astimezone(timezone.utc).isoformat()


def _validate_observation(payload: dict) -> dict:
    """Validate and normalize an incoming telemetry observation."""
    import json
    import math
    from datetime import datetime, timezone

    if not isinstance(payload, dict):
        raise ValueError("Observation must be a JSON object")

    for field in ("timestamp", "agent_id", "host", "metrics"):
        if field not in payload:
            raise ValueError(f"Missing required field: {field}")

    for field in ("agent_id", "host"):
        value = payload[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")

    if not isinstance(payload["metrics"], dict) or not payload["metrics"]:
        raise ValueError("metrics must be a non-empty object")

    timestamp_value = payload["timestamp"]
    if not isinstance(timestamp_value, str):
        raise ValueError("timestamp must be an ISO-8601 string")

    try:
        parsed_timestamp = datetime.fromisoformat(
            timestamp_value.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("timestamp must be a valid ISO-8601 timestamp") from exc

    if (
        parsed_timestamp.tzinfo is None
        or parsed_timestamp.utcoffset() is None
    ):
        raise ValueError("timestamp must include a timezone")

    normalized_timestamp = parsed_timestamp.astimezone(timezone.utc).isoformat()

    collection_status = payload.get("collection_status")
    if collection_status not in {"ok", "partial", "failed"}:
        raise ValueError(
            "collection_status must be 'ok', 'partial', or 'failed'"
        )


    metrics = payload["metrics"]
    metric_status = payload.get("metric_status")

    if not isinstance(metric_status, dict):
        raise ValueError("metric_status must be an object")

    if set(metric_status.keys()) != set(metrics.keys()):
        raise ValueError(
            "metric_status keys must exactly match metrics keys"
        )


    for name, record in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Metric names must be non-empty strings")

        if not isinstance(record, dict):
            raise ValueError(f"Metric {name!r} must be an object")

        status = record.get("status")
        if status not in {"ok", "unavailable"}:
            raise ValueError(
                f"Metric {name!r} must have status 'ok' or 'unavailable'"
            )

        unit = record.get("unit")
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError(f"Metric {name!r} must declare its unit")

        value = record.get("value")

        if status == "unavailable":
            if value is not None:
                raise ValueError(
                    f"Unavailable metric {name!r} must have value null"
                )
            if not isinstance(record.get("error"), str) or not record["error"]:
                raise ValueError(
                    f"Unavailable metric {name!r} must include an error"
                )

        if status == "ok" and value is not None:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"Metric {name!r} has a non-finite or non-numeric value"
                )

        if name in {"cpu_percent", "memory_percent", "disk_percent"}:
            if status == "ok" and (
                value is None or not 0 <= value <= 100
            ):
                raise ValueError(
                    f"Metric {name!r} must be between 0 and 100"
                )

        if name == "network_connections" and status == "ok":
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    "network_connections must be a non-negative integer"
                )


        if name in {"load_average", "load_1m"} and status == "ok":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"Metric {name!r} must be a finite non-negative number"
                )


        # If the parallel status map is supplied, require consistency.
        if name in metric_status and metric_status[name] != status:
            raise ValueError(
                f"metric_status conflicts with metric {name!r}"
            )

        # Validate nested network counters and rates too.
        if name == "network_io" and status == "ok":
            for section in ("counters", "rates"):
                values = record.get(section)
                if not isinstance(values, dict):
                    raise ValueError(
                        f"network_io.{section} must be an object"
                    )

                for key, number in values.items():
                    if number is None:
                        # Null rates are valid for baseline samples.
                        if section == "rates":
                            continue
                        raise ValueError(
                            f"network_io counter {key!r} cannot be null"
                        )

                    if (
                        isinstance(number, bool)
                        or not isinstance(number, (int, float))
                        or not math.isfinite(number)
                        or number < 0
                    ):
                        raise ValueError(
                            f"network_io {section} {key!r} is invalid"
                        )

            if record.get("rate_status") not in {
                "ok", "baseline", "unavailable"
            }:
                raise ValueError("Invalid network_io rate_status")


    # Ensure the collection status agrees with all metric outcomes.
    metric_outcomes = [
        record["status"] for record in metrics.values()
    ]

    if all(status == "ok" for status in metric_outcomes):
        expected_collection_status = "ok"
    elif all(status == "unavailable" for status in metric_outcomes):
        expected_collection_status = "failed"
    else:
        expected_collection_status = "partial"

    if collection_status != expected_collection_status:
        raise ValueError(
            "collection_status is inconsistent with individual metric statuses"
        )


    normalized = dict(payload)
    normalized["timestamp"] = normalized_timestamp

    # Reject NaN and Infinity before persistence; preserve null values.
    try:
        json.dumps(normalized, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Observation contains values that cannot be stored as JSON"
        ) from exc

    return normalized



@app.get("/health")
def health() -> dict[str, str]:
    """Public health check; reveals no telemetry or database records."""
    return {
        "status": "ok",
        "action_execution": "disabled",
    }


@app.get("/v1/status", dependencies=[Depends(auth)])
def status() -> dict[str, Any]:
    return {
        "counts": get_counts(),
        "action_execution": "disabled",
    }


@app.get("/v1/findings", dependencies=[Depends(auth)])
def findings(
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    return {"findings": get_findings(limit)}


@app.get("/v1/decisions", dependencies=[Depends(auth)])
def decisions(
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    return {"decisions": get_decision_audit(limit)}


@app.post("/v1/analyze", dependencies=[Depends(auth)])
def analyze_route(req: AnalyzeRequest) -> Any:
    """Run the existing analysis pipeline; this does not execute actions."""
    try:
        return analyze(
            req.agent_id,
            req.host,
            req.text,
            req.metrics.model_dump(),
            persist_telemetry=False,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@app.post("/v1/feedback", dependencies=[Depends(auth)])
def feedback(req: FeedbackRequest) -> dict[str, Any]:
    """Store analyst feedback for review; do not train automatically."""
    record = {
        "timestamp": now(),
        "text": req.text,
        "label": req.label,
        "source": req.source,
        "analyst_confirmed": req.analyst_confirmed,
    }

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        with (MODEL_DIR / "analyst_feedback.jsonl").open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                ) + "\n"
            )

        add_experience(
            {"source": req.source},
            "label_feedback",
            1.0 if req.analyst_confirmed else 0.0,
            {"label": req.label, "text": req.text},
            req.analyst_confirmed,
        )

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail="Feedback contains invalid data",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Feedback persistence failed",
        ) from exc

    return {
        "stored": True,
        "analyst_confirmed": req.analyst_confirmed,
        "message": (
            "Feedback is stored for review; it is not automatically "
            "used as a training label."
        ),
    }

@app.get(
    "/v1/telemetry/timeseries",
    dependencies=[Depends(auth)],
)
def telemetry_timeseries(
    host: str | None = None,
    agent_id: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = Query(default=1000, ge=1, le=10000),
) -> dict[str, Any]:
    """Retrieve stored telemetry chronologically, with optional filters."""

    start_utc = _normalize_timestamp(start_time)
    end_utc = _normalize_timestamp(end_time)

    if start_utc is not None and end_utc is not None:
        start_dt = datetime.fromisoformat(start_utc)
        end_dt = datetime.fromisoformat(end_utc)

        if start_dt > end_dt:
            raise HTTPException(
                status_code=400,
                detail=(
                    "start_time must be earlier than or equal to end_time"
                ),
            )

    if host is not None and not host.strip():
        raise HTTPException(
            status_code=400,
            detail="host must be a non-empty string when supplied",
        )

    if agent_id is not None and not agent_id.strip():
        raise HTTPException(
            status_code=400,
            detail="agent_id must be a non-empty string when supplied",
        )

    try:
        records = get_telemetry_timeseries(
            host=host,
            agent_id=agent_id,
            start_time=start_utc,
            end_time=end_utc,
            limit=limit,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Telemetry retrieval failed",
        ) from exc

    return {
        "count": len(records),
        "records": records,
    }



@app.post(
    "/v1/telemetry",
    dependencies=[Depends(auth)],
)
def ingest_telemetry(
    observation: dict[str, Any],
) -> dict[str, Any]:
    """Validate and persist one host observation."""

    stored_observation = _validate_observation(observation)
    stored_observation["agent_id"] = observation["agent_id"].strip()
    stored_observation["host"] = observation["host"].strip()

    try:
        row_id = add_telemetry_observation(stored_observation)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        # Keep internal database details out of the response.
        raise HTTPException(
            status_code=500,
            detail="Telemetry persistence failed",
        ) from exc

    return {
        "status": "stored",
        "telemetry_id": row_id,
        "timestamp": stored_observation["timestamp"],
        "agent_id": stored_observation["agent_id"],
        "host": stored_observation["host"],
    }
