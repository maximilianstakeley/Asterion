
from __future__ import annotations

import json
import sqlite3
import uuid

from datetime import datetime, timezone

from ..configuration import DB_PATH


def initialize_outcomes():
    with sqlite3.connect(DB_PATH) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS action_outcomes (
            outcome_id TEXT PRIMARY KEY,
            action_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            host TEXT NOT NULL,
            baseline_telemetry_id INTEGER NOT NULL,
            post_telemetry_id INTEGER,
            expected_effect_json TEXT NOT NULL,
            observed_effect_json TEXT,
            status TEXT NOT NULL,
            rollback_plan_json TEXT,
            created_at TEXT NOT NULL,
            verified_at TEXT
        );
        """)


def create_outcome_record(
    action_id,
    agent_id,
    host,
    baseline_telemetry_id,
    expected_effect,
    rollback_plan=None
):
    outcome_id = str(uuid.uuid4())

    with sqlite3.connect(DB_PATH) as db:
        db.execute("""
            INSERT INTO action_outcomes (
                outcome_id,
                action_id,
                agent_id,
                host,
                baseline_telemetry_id,
                expected_effect_json,
                status,
                rollback_plan_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'awaiting_measurement', ?, ?)
        """, (
            outcome_id,
            action_id,
            agent_id,
            host,
            baseline_telemetry_id,
            json.dumps(expected_effect, allow_nan=False),
            json.dumps(rollback_plan, allow_nan=False)
            if rollback_plan is not None else None,
            datetime.now(timezone.utc).isoformat()
        ))

    return outcome_id


def _extract(metrics):
    result = {}

    for name, record in metrics.items():
        if (
            isinstance(record, dict)
            and record.get("status") == "ok"
            and isinstance(record.get("value"), (int, float))
            and not isinstance(record.get("value"), bool)
        ):
            result[name] = float(record["value"])

    return result


def verify_outcome(
    outcome_id,
    post_telemetry_id
):
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row

        outcome = db.execute("""
            SELECT *
            FROM action_outcomes
            WHERE outcome_id = ?
        """, (outcome_id,)).fetchone()

        if outcome is None:
            raise ValueError("Unknown outcome")

        if outcome["status"] != "awaiting_measurement":
            raise ValueError("Outcome already processed")

        before = db.execute("""
            SELECT metrics_json
            FROM telemetry
            WHERE id = ?
              AND host = ?
              AND agent_id = ?
        """, (
            outcome["baseline_telemetry_id"],
            outcome["host"],
            outcome["agent_id"]
        )).fetchone()

        after = db.execute("""
            SELECT metrics_json
            FROM telemetry
            WHERE id = ?
              AND host = ?
              AND agent_id = ?
        """, (
            post_telemetry_id,
            outcome["host"],
            outcome["agent_id"]
        )).fetchone()

        if before is None or after is None:
            raise ValueError("Telemetry evidence missing")

        baseline = _extract(
            json.loads(before["metrics_json"])
        )

        current = _extract(
            json.loads(after["metrics_json"])
        )

        comparable = (
            set(baseline.keys()) & set(current.keys())
        )

        if not comparable:
            raise ValueError(
                "No comparable measurements available"
            )

        changes = {
            name: {
                "before": baseline[name],
                "after": current[name],
                "delta": current[name] - baseline[name]
            }
            for name in comparable
        }

        expected = json.loads(
            outcome["expected_effect_json"]
        )

        # Expected effects define the desired direction.
        # Missing or unrecognized directions are not considered success.
        checks = []

        for metric, direction in expected.items():
            if metric not in changes:
                continue

            delta = changes[metric]["delta"]

            if direction == "decrease":
                checks.append(delta < 0)

            elif direction == "increase":
                checks.append(delta > 0)

            elif direction == "stable":
                checks.append(abs(delta) < 1.0)

        if not checks:
            status = "inconclusive"

        elif all(checks):
            status = "verified_improvement"

        elif not any(checks):
            status = "no_improvement"

        else:
            status = "mixed_outcome"

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        db.execute("""
            UPDATE action_outcomes
            SET post_telemetry_id = ?,
                observed_effect_json = ?,
                status = ?,
                verified_at = ?
            WHERE outcome_id = ?
        """, (
            post_telemetry_id,
            json.dumps(changes, allow_nan=False),
            status,
            timestamp,
            outcome_id
        ))

        db.commit()

    return {
        "outcome_id": outcome_id,
        "status": status,
        "changes": changes,
        "rollback_required": status in (
            "no_improvement",
            "mixed_outcome"
        )
    }


def request_rollback(outcome_id, operator_authorized=False):
    """Create a rollback request; never execute it."""

    if not operator_authorized:
        raise PermissionError(
            "Rollback requires explicit operator authorization"
        )

    with sqlite3.connect(DB_PATH) as db:
        row = db.execute("""
            SELECT status, rollback_plan_json
            FROM action_outcomes
            WHERE outcome_id = ?
        """, (outcome_id,)).fetchone()

        if row is None:
            raise ValueError("Unknown outcome")

        status, rollback_plan = row

        if status not in (
            "no_improvement",
            "mixed_outcome"
        ):
            raise ValueError(
                "Rollback is not indicated by verified outcome"
            )

        if not rollback_plan:
            raise ValueError(
                "No rollback plan was recorded"
            )

        plan = json.loads(rollback_plan)

        if not isinstance(plan, dict):
            raise ValueError("Invalid rollback plan")

        db.execute("""
            UPDATE action_outcomes
            SET status = 'rollback_requested'
            WHERE outcome_id = ?
        """, (outcome_id,))

        db.commit()

    return {
        "outcome_id": outcome_id,
        "status": "rollback_requested",
        "plan": plan,
        "executed": False
    }
