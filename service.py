
"""SQLite persistence for Asterion telemetry and decision records."""

import json
import math
import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from .configuration import DB_PATH


SCHEMA = """
PRAGMA journal_mode=WAL;


CREATE TABLE IF NOT EXISTS telemetry (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    host TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    collection_status TEXT NOT NULL DEFAULT 'unknown',
    metric_status_json TEXT NOT NULL DEFAULT '{}');


CREATE TABLE IF NOT EXISTS network_security_records (
    id INTEGER PRIMARY KEY,
    record_id TEXT NOT NULL UNIQUE,
    timestamp TEXT NOT NULL,
    source_host TEXT NOT NULL,
    record_type TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    record_json TEXT NOT NULL,
    received_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_network_security_timestamp
    ON network_security_records(timestamp, id);

CREATE INDEX IF NOT EXISTS idx_network_security_host_timestamp
    ON network_security_records(source_host, timestamp, id);


CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp_id
    ON telemetry(timestamp, id);

CREATE INDEX IF NOT EXISTS idx_telemetry_host_timestamp
    ON telemetry(host, timestamp, id);

CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    host TEXT NOT NULL,
    severity TEXT NOT NULL,
    category TEXT NOT NULL,
    detail_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS action_proposals (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    finding_id INTEGER,
    action TEXT NOT NULL,
    risk REAL NOT NULL,
    status TEXT NOT NULL,
    rationale TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiences (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    state_json TEXT NOT NULL,
    action TEXT NOT NULL,
    reward REAL NOT NULL,
    outcome_json TEXT NOT NULL,
    validated INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    model_path TEXT NOT NULL,
    model_type TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    promoted INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS decision_audit (
    decision_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    host TEXT NOT NULL,
    record_json TEXT NOT NULL
);
"""


def now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _validate_json_value(value: Any) -> None:
    """Reject non-finite floating-point values anywhere in nested data."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN and Infinity are not valid telemetry values")

    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            _validate_json_value(item)

    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_value(item)


def _json_dumps(value: Any) -> str:
    """Serialize strict JSON while preserving None as null."""
    _validate_json_value(value)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Value cannot be serialized as valid JSON: {exc}") from exc


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection with commit, rollback, and cleanup."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=15)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create the schema and migrate older telemetry tables safely."""
    with connect() as conn:
        conn.executescript(SCHEMA)

        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(telemetry)"
            ).fetchall()
        }

        if "collection_status" not in columns:
            conn.execute(
                """
                ALTER TABLE telemetry
                ADD COLUMN collection_status TEXT NOT NULL
                DEFAULT 'unknown'
                """
            )

        if "metric_status_json" not in columns:
            conn.execute(
                """
                ALTER TABLE telemetry
                ADD COLUMN metric_status_json TEXT NOT NULL
                DEFAULT '{}'
                """
            )



def add_telemetry(agent, host, metrics):
    """Store legacy telemetry using the server's insertion timestamp."""
    if not isinstance(agent, str) or not agent.strip():
        raise ValueError("agent must be a non-empty string")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty string")
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dictionary")

    metrics_json = _json_dumps(metrics)

    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO telemetry (
                timestamp, agent_id, host, metrics_json
            )
            VALUES (?, ?, ?, ?)
            """,
            (now(), agent, host, metrics_json),
        )
        return int(cursor.lastrowid)


def add_telemetry_observation(observation: dict) -> int:
    """Persist telemetry together with collection and metric-level statuses."""
    if not isinstance(observation, dict):
        raise ValueError("observation must be a dictionary")

    required = ("timestamp", "agent_id", "host", "metrics")
    missing = [key for key in required if key not in observation]
    if missing:
        raise ValueError(f"Missing observation fields: {missing}")

    timestamp = observation["timestamp"]
    agent_id = observation["agent_id"]
    host = observation["host"]
    metrics = observation["metrics"]
    collection_status = observation.get("collection_status", "unknown")
    metric_status = observation.get("metric_status", {})

    for field_name, value in (
        ("timestamp", timestamp),
        ("agent_id", agent_id),
        ("host", host),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")

    try:
        parsed_timestamp = datetime.fromisoformat(
            timestamp.strip().replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(
            "timestamp must be a valid ISO 8601 timestamp"
        ) from exc

    if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")

    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dictionary")

    if collection_status not in {"ok", "partial", "failed", "unknown"}:
        raise ValueError("Invalid collection_status")

    if not isinstance(metric_status, dict):
        raise ValueError("metric_status must be a dictionary")

    metrics_json = _json_dumps(metrics)
    metric_status_json = _json_dumps(metric_status)

    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO telemetry (
                timestamp,
                agent_id,
                host,
                metrics_json,
                collection_status,
                metric_status_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp.strip(),
                agent_id.strip(),
                host.strip(),
                metrics_json,
                collection_status,
                metric_status_json,
            ),
        )
        return int(cursor.lastrowid)



def add_finding(agent, host, severity, category, detail):
    """Persist a finding."""
    detail_json = _json_dumps(detail)

    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO findings (
                timestamp, agent_id, host, severity, category, detail_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (now(), agent, host, severity, category, detail_json),
        )
        return int(cursor.lastrowid)


def add_proposal(fid, action, risk, status, rationale):
    """Persist an action proposal; this does not execute the action."""
    if not isinstance(risk, (int, float)) or isinstance(risk, bool):
        raise ValueError("risk must be a finite number")
    if not math.isfinite(risk):
        raise ValueError("risk must be a finite number")

    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO action_proposals (
                timestamp, finding_id, action, risk, status, rationale
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (now(), fid, action, float(risk), status, rationale),
        )
        return int(cursor.lastrowid)


def add_experience(state, action, reward, outcome, validated=False):
    """Persist a recorded experience."""
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise ValueError("reward must be a finite number")
    if not math.isfinite(reward):
        raise ValueError("reward must be a finite number")

    state_json = _json_dumps(state)
    outcome_json = _json_dumps(outcome)

    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO experiences (
                timestamp, state_json, action, reward, outcome_json, validated
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                now(),
                state_json,
                action,
                float(reward),
                outcome_json,
                int(bool(validated)),
            ),
        )
        return int(cursor.lastrowid)


def get_counts():
    """Return counts for the primary Asterion persistence tables."""
    tables = (
        "telemetry",
        "findings",
        "action_proposals",
        "experiences",
        "model_registry",
    )

    with connect() as conn:
        return {
            table: conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            for table in tables
        }


def get_findings(limit=50):
    """Return the newest findings first."""
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")

    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM findings ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]


def add_decision_audit(
    decision_id,
    timestamp,
    agent_id,
    host,
    record,
):
    """Append a decision audit record; duplicate IDs are rejected."""
    if not all(
        isinstance(value, str) and value.strip()
        for value in (decision_id, timestamp, agent_id, host)
    ):
        raise ValueError(
            "decision_id, timestamp, agent_id and host must be non-empty strings"
        )

    record_json = _json_dumps(record)

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO decision_audit (
                decision_id, timestamp, agent_id, host, record_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                timestamp,
                agent_id,
                host,
                record_json,
            ),
        )

    return decision_id



def get_decision_audit(limit=50, decision_id=None):
    """Return audit records with top-level fields and a nested record payload."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer")
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")

    if decision_id is not None:
        if not isinstance(decision_id, str) or not decision_id.strip():
            raise ValueError("decision_id must be a non-empty string")
        decision_id = decision_id.strip()

    with connect() as conn:
        if decision_id is not None:
            rows = conn.execute(
                """
                SELECT decision_id, timestamp, agent_id, host, record_json
                FROM decision_audit
                WHERE decision_id = ?
                ORDER BY timestamp DESC, rowid DESC
                LIMIT ?
                """,
                (decision_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT decision_id, timestamp, agent_id, host, record_json
                FROM decision_audit
                ORDER BY timestamp DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    results = []
    for row in rows:
        record = json.loads(row["record_json"])

        if not isinstance(record, dict):
            record = {"payload": record}

        # Keep the stored record's fields available at the top level.
        result = dict(record)

        # Database metadata is authoritative.
        result.update({
            "decision_id": row["decision_id"],
            "timestamp": row["timestamp"],
            "agent_id": row["agent_id"],
            "host": row["host"],
            "record": record,
        })

        results.append(result)

    return results


def get_telemetry_timeseries(
    host: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = 1000,
    agent_id: str | None = None,
) -> list[dict]:
    """Return chronological telemetry filtered by host, agent, and time."""

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer")

    if not 1 <= limit <= 10000:
        raise ValueError("limit must be between 1 and 10000")

    clauses = []
    params = []

    # Filter by machine hostname.
    if host is not None:
        if not isinstance(host, str) or not host.strip():
            raise ValueError("host must be a non-empty string")

        clauses.append("host = ?")
        params.append(host.strip())

    # Filter by monitoring agent identifier.
    if agent_id is not None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")

        clauses.append("agent_id = ?")
        params.append(agent_id.strip())

    # Validate and normalize timestamps to UTC.
    normalized_times = {}

    for field, value, operator in (
        ("start_time", start_time, ">="),
        ("end_time", end_time, "<="),
    ):
        if value is None:
            continue

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")

        try:
            parsed = datetime.fromisoformat(
                value.strip().replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(
                f"{field} must be a valid ISO 8601 timestamp"
            ) from exc

        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(
                f"{field} must include a timezone"
            )

        parsed_utc = parsed.astimezone(timezone.utc)
        normalized_times[field] = parsed_utc

        clauses.append(f"timestamp {operator} ?")
        params.append(parsed_utc.isoformat())

    if (
        "start_time" in normalized_times
        and "end_time" in normalized_times
        and normalized_times["start_time"] > normalized_times["end_time"]
    ):
        raise ValueError(
            "start_time must be earlier than or equal to end_time"
        )

    where = (
        f"WHERE {' AND '.join(clauses)}"
        if clauses
        else ""
    )

    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT
                id,
                timestamp,
                agent_id,
                host,
                metrics_json,
                collection_status,
                metric_status_json
            FROM (
                SELECT
                    id,
                    timestamp,
                    agent_id,
                    host,
                    metrics_json,
                    collection_status,
                    metric_status_json
                FROM telemetry
                {where}
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
            ) AS recent
            ORDER BY timestamp ASC, id ASC
            """,
            (*params, limit),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "agent_id": row["agent_id"],
            "host": row["host"],
            "collection_status": row["collection_status"],
            "metric_status": json.loads(row["metric_status_json"]),
            "metrics": json.loads(row["metrics_json"]),
        }
        for row in rows
    ]


def add_network_security_record(record: dict) -> int:
    """Persist a validated network security record.

    The caller should validate the payload with NetworkSecurityRecord before
    calling this function. Duplicate record IDs are rejected, not overwritten.
    """
    import hashlib
    import json

    if not isinstance(record, dict):
        raise ValueError("record must be a dictionary")

    required = ("record_id", "timestamp", "source_host", "record_type")
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(f"Missing record fields: {missing}")

    canonical = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO network_security_records (
                record_id, timestamp, source_host, record_type,
                fingerprint, record_json, received_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["record_id"],
                record["timestamp"],
                record["source_host"],
                record["record_type"],
                fingerprint,
                canonical,
                now(),
            ),
        )
        return int(cursor.lastrowid)


def get_network_security_records(
    *,
    source_host: str | None = None,
    start_timestamp: str | None = None,
    end_timestamp: str | None = None,
    limit: int = 500,
) -> list[dict]:
    """Read stored network security records in event-time order."""
    if not 1 <= limit <= 5000:
        raise ValueError("limit must be between 1 and 5000")

    clauses = []
    parameters: list[object] = []

    if source_host is not None:
        clauses.append("source_host = ?")
        parameters.append(source_host)

    if start_timestamp is not None:
        clauses.append("timestamp >= ?")
        parameters.append(start_timestamp)

    if end_timestamp is not None:
        clauses.append("timestamp <= ?")
        parameters.append(end_timestamp)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    parameters.append(limit)

    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, record_id, timestamp, source_host, record_type,
                   fingerprint, record_json, received_at
            FROM network_security_records
            {where}
            ORDER BY timestamp ASC, id ASC
            LIMIT ?
            """,
            parameters,
        ).fetchall()

    return [
        {
            "id": row["id"],
            "record_id": row["record_id"],
            "timestamp": row["timestamp"],
            "source_host": row["source_host"],
            "record_type": row["record_type"],
            "fingerprint": row["fingerprint"],
            "record": json.loads(row["record_json"]),
            "received_at": row["received_at"],
        }
        for row in rows]


