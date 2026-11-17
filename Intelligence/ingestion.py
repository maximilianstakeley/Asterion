
from __future__ import annotations

import json
import math
import sqlite3
import uuid

from datetime import datetime, timezone
from typing import Any

from ..configuration import DB_PATH


def initialize_ingestion():
    with sqlite3.connect(DB_PATH) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS telemetry_analysis_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telemetry_id INTEGER NOT NULL UNIQUE,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status
        ON telemetry_analysis_jobs(status, id);
        """)


def validate_observation(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Observation must be an object")

    required = (
        "timestamp", "agent_id", "host",
        "metrics", "metric_status"
    )

    for key in required:
        if key not in payload:
            raise ValueError(f"Missing {key}")

    timestamp = datetime.fromisoformat(
        payload["timestamp"].replace("Z", "+00:00")
    )

    if timestamp.tzinfo is None:
        raise ValueError("Timestamp must include timezone")

    metrics = payload["metrics"]
    statuses = payload["metric_status"]

    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Metrics must be a non-empty dictionary")

    if not isinstance(statuses, dict):
        raise ValueError("metric_status must be a dictionary")

    if set(metrics) != set(statuses):
        raise ValueError("Metric/status keys do not match")

    outcomes = []

    for name, record in metrics.items():
        if not isinstance(record, dict):
            raise ValueError(f"{name} must be a structured metric")

        status = record.get("status")
        value = record.get("value")

        if status not in ("ok", "unavailable"):
            raise ValueError(f"Invalid status for {name}")

        if statuses[name] != status:
            raise ValueError(f"Conflicting status for {name}")

        if status == "unavailable":
            if value is not None:
                raise ValueError(
                    f"{name}: unavailable value must be None"
                )

            if not record.get("error"):
                raise ValueError(
                    f"{name}: unavailable metric requires an error"
                )

        else:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name}: invalid numeric measurement")

            if name in (
                "cpu_percent",
                "memory_percent",
                "disk_percent"
            ) and not 0 <= value <= 100:
                raise ValueError(f"{name} outside [0, 100]")

        outcomes.append(status)

    expected = (
        "ok" if all(x == "ok" for x in outcomes)
        else "failed" if all(x == "unavailable" for x in outcomes)
        else "partial"
    )

    if payload.get("collection_status") != expected:
        raise ValueError("Incorrect collection_status")

    normalized = dict(payload)
    normalized["timestamp"] = timestamp.astimezone(
        timezone.utc
    ).isoformat()

    json.dumps(normalized, allow_nan=False)

    return normalized


def ingest_observation(payload: dict) -> int:
    """Atomically persist telemetry and queue its analysis."""

    observation = validate_observation(payload)

    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH, timeout=30) as db:
        db.execute("BEGIN IMMEDIATE")

        cursor = db.execute("""
            INSERT INTO telemetry (
                timestamp,
                agent_id,
                host,
                metrics_json,
                collection_status,
                metric_status_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            observation["timestamp"],
            observation["agent_id"],
            observation["host"],
            json.dumps(observation["metrics"], allow_nan=False),
            observation["collection_status"],
            json.dumps(observation["metric_status"]),
        ))

        telemetry_id = cursor.lastrowid

        db.execute("""
            INSERT INTO telemetry_analysis_jobs (
                telemetry_id, status, created_at
            )
            VALUES (?, 'pending', ?)
        """, (telemetry_id, now))

        db.commit()

    return telemetry_id


def get_pending_jobs(limit=100):
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row

        return [
            dict(row)
            for row in db.execute("""
                SELECT *
                FROM telemetry_analysis_jobs
                WHERE status IN ('pending', 'retry')
                ORDER BY id
                LIMIT ?
            """, (limit,)).fetchall()
        ]


def mark_job_complete(job_id):
    with sqlite3.connect(DB_PATH) as db:
        db.execute("""
            UPDATE telemetry_analysis_jobs
            SET status = 'complete',
                completed_at = ?
            WHERE id = ?
        """, (
            datetime.now(timezone.utc).isoformat(),
            job_id
        ))


def mark_job_failed(job_id, error):
    with sqlite3.connect(DB_PATH) as db:
        db.execute("""
            UPDATE telemetry_analysis_jobs
            SET attempts = attempts + 1,
                status = CASE
                    WHEN attempts >= 4 THEN 'dead_letter'
                    ELSE 'retry'
                END,
                last_error = ?
            WHERE id = ?
        """, (str(error)[:2000], job_id))
