
from __future__ import annotations

import json
import sqlite3
import time

from ..configuration import DB_PATH

from .ingestion import (
    get_pending_jobs,
    mark_job_complete,
    mark_job_failed,
)

from .detection import analyze_host, load_history

from .root_cause import rank_root_causes

from .correlation import correlate_finding


def process_job(job):
    job_id = job["id"]
    telemetry_id = job["telemetry_id"]

    try:
        with sqlite3.connect(DB_PATH) as db:
            db.row_factory = sqlite3.Row

            row = db.execute("""
                SELECT agent_id, host
                FROM telemetry
                WHERE id = ?
            """, (telemetry_id,)).fetchone()

        if row is None:
            raise ValueError("Telemetry record not found")

        agent_id = row["agent_id"]
        host = row["host"]

        result = analyze_host(host, agent_id)

        history = load_history(
            host,
            agent_id
        )

        hypotheses = rank_root_causes(
            history,
            result["findings"]
        )

        with sqlite3.connect(DB_PATH) as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS root_cause_assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telemetry_id INTEGER NOT NULL,
                    agent_id TEXT NOT NULL,
                    host TEXT NOT NULL,
                    hypotheses_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(telemetry_id)
                )
            """)

            from datetime import datetime, timezone

            db.execute("""
                INSERT OR IGNORE INTO root_cause_assessments (
                    telemetry_id,
                    agent_id,
                    host,
                    hypotheses_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                telemetry_id,
                agent_id,
                host,
                json.dumps(hypotheses, allow_nan=False),
                datetime.now(timezone.utc).isoformat()
            ))

        for finding_id in result["persisted_finding_ids"]:
            correlate_finding(finding_id)

        mark_job_complete(job_id)

        return {
            "job_id": job_id,
            "telemetry_id": telemetry_id,
            "status": "complete",
            "findings": len(result["findings"]),
            "hypotheses": hypotheses
        }

    except Exception as exc:
        mark_job_failed(job_id, exc)

        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(exc)
        }


def run_worker_once():
    jobs = get_pending_jobs()

    return [
        process_job(job)
        for job in jobs
    ]


def run_forever(poll_interval=2):
    print("Asterion Intelligence worker started")

    while True:
        results = run_worker_once()

        for result in results:
            print(json.dumps(result, default=str))

        time.sleep(poll_interval)


if __name__ == "__main__":
    run_forever()
