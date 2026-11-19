
import json
import sqlite3

from ..configuration import DB_PATH


def initialize_correlation():
    with sqlite3.connect(DB_PATH) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS finding_evidence_links (
            finding_id TEXT NOT NULL,
            telemetry_id INTEGER NOT NULL,
            relationship TEXT NOT NULL,
            PRIMARY KEY (
                finding_id, telemetry_id, relationship
            )
        );

        CREATE TABLE IF NOT EXISTS finding_decision_links (
            finding_id TEXT NOT NULL,
            decision_id TEXT NOT NULL,
            relationship TEXT NOT NULL,
            PRIMARY KEY (
                finding_id, decision_id, relationship
            )
        );
        """)


def correlate_finding(finding_id):
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row

        finding = db.execute("""
            SELECT *
            FROM intelligence_findings
            WHERE evidence_id = ?
        """, (finding_id,)).fetchone()

        if finding is None:
            raise ValueError("Finding not found")

        telemetry = db.execute("""
            SELECT id, timestamp, agent_id, host, metrics_json
            FROM telemetry
            WHERE id = ?
        """, (finding["telemetry_id"],)).fetchone()

        if telemetry is None:
            raise ValueError("Source telemetry missing")

        detail = json.loads(finding["detail_json"])

        db.execute("""
            INSERT OR IGNORE INTO finding_evidence_links
            VALUES (?, ?, 'originating_observation')
        """, (finding_id, telemetry["id"]))

        # Associate decisions on the same host and agent within
        # the same evidence time window.
        decisions = db.execute("""
            SELECT decision_id, timestamp, record_json
            FROM decision_audit
            WHERE host = ?
              AND agent_id = ?
              AND julianday(timestamp) BETWEEN
    julianday(?) - (10.0 / 1440.0)
    AND julianday(?) + (10.0 / 1440.0)
        """, (
            finding["host"],
            finding["agent_id"],
            finding["timestamp"],
            finding["timestamp"]
        )).fetchall()

        linked = []

        for decision in decisions:
            db.execute("""
                INSERT OR IGNORE INTO finding_decision_links
                VALUES (?, ?, 'temporally_associated')
            """, (
                finding_id,
                decision["decision_id"]
            ))

            linked.append(decision["decision_id"])

        db.commit()

    return {
        "finding": dict(finding),
        "source_telemetry": dict(telemetry),
        "decision_records": linked,
        "interpretation": (
            "Temporal association does not establish causation."
        )
    }
