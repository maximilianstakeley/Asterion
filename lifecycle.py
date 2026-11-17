
from __future__ import annotations

import hashlib
import json
import sqlite3

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..configuration import DB_PATH


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def initialize_registry():
    with sqlite3.connect(DB_PATH) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS intelligence_models (
            model_id TEXT PRIMARY KEY,
            model_type TEXT NOT NULL,
            model_path TEXT NOT NULL,
            checksum TEXT NOT NULL,
            feature_contract TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            dataset_fingerprint TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            promoted_at TEXT
        );

        CREATE TABLE IF NOT EXISTS model_promotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            previous_model_id TEXT,
            promoted_at TEXT NOT NULL,
            reason TEXT NOT NULL
        );
        """)


def register_candidate(
    model_path,
    model_type,
    feature_contract,
    metrics,
    dataset_fingerprint
):
    path = Path(model_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    if not dataset_fingerprint:
        raise ValueError("Dataset fingerprint is required")

    model_id = str(uuid4())

    checksum = sha256_file(path)

    with sqlite3.connect(DB_PATH) as db:
        db.execute("""
            INSERT INTO intelligence_models (
                model_id,
                model_type,
                model_path,
                checksum,
                feature_contract,
                metrics_json,
                dataset_fingerprint,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'candidate', ?)
        """, (
            model_id,
            model_type,
            str(path),
            checksum,
            feature_contract,
            json.dumps(metrics, allow_nan=False),
            dataset_fingerprint,
            datetime.now(timezone.utc).isoformat()
        ))

    return model_id


def promote_model(
    model_id,
    minimum_f1=0.70,
    maximum_false_positive_rate=0.05
):
    with sqlite3.connect(DB_PATH) as db:
        db.execute("BEGIN IMMEDIATE")

        candidate = db.execute("""
            SELECT model_path,
                   checksum,
                   metrics_json,
                   status
            FROM intelligence_models
            WHERE model_id = ?
        """, (model_id,)).fetchone()

        if candidate is None:
            raise ValueError("Unknown model")

        path, checksum, metrics_json, status = candidate

        if status != "candidate":
            raise ValueError("Model is not a candidate")

        if sha256_file(path) != checksum:
            raise ValueError("Model checksum mismatch")

        metrics = json.loads(metrics_json)

        required = (
            "f1",
            "false_positive_rate",
            "false_negative_rate"
        )

        if any(key not in metrics for key in required):
            raise ValueError("Incomplete evaluation metrics")

        if metrics["f1"] < minimum_f1:
            raise ValueError("F1 requirement not met")

        if (
            metrics["false_positive_rate"]
            > maximum_false_positive_rate
        ):
            raise ValueError("False-positive limit exceeded")

        previous = db.execute("""
            SELECT model_id
            FROM intelligence_models
            WHERE status = 'production'
            ORDER BY promoted_at DESC
            LIMIT 1
        """).fetchone()

        previous_id = previous[0] if previous else None

        db.execute("""
            UPDATE intelligence_models
            SET status = 'archived'
            WHERE status = 'production'
        """)

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        db.execute("""
            UPDATE intelligence_models
            SET status = 'production',
                promoted_at = ?
            WHERE model_id = ?
        """, (timestamp, model_id))

        db.execute("""
            INSERT INTO model_promotions (
                model_id,
                previous_model_id,
                promoted_at,
                reason
            )
            VALUES (?, ?, ?, ?)
        """, (
            model_id,
            previous_id,
            timestamp,
            "Passed configured promotion gates"
        ))

        db.commit()

    return {
        "model_id": model_id,
        "previous_model_id": previous_id,
        "status": "production"
    }


def rollback_model(model_id):
    with sqlite3.connect(DB_PATH) as db:
        db.execute("BEGIN IMMEDIATE")

        target = db.execute("""
            SELECT model_path, checksum, status
            FROM intelligence_models
            WHERE model_id = ?
        """, (model_id,)).fetchone()

        if target is None:
            raise ValueError("Unknown rollback model")

        path, checksum, status = target

        if status != "archived":
            raise ValueError("Rollback target must be archived")

        if sha256_file(path) != checksum:
            raise ValueError("Rollback artifact checksum mismatch")

        current = db.execute("""
            SELECT model_id
            FROM intelligence_models
            WHERE status = 'production'
        """).fetchone()

        current_id = current[0] if current else None

        db.execute("""
            UPDATE intelligence_models
            SET status = 'archived'
            WHERE status = 'production'
        """)

        db.execute("""
            UPDATE intelligence_models
            SET status = 'production',
                promoted_at = ?
            WHERE model_id = ?
        """, (
            datetime.now(timezone.utc).isoformat(),
            model_id
        ))

        db.execute("""
            INSERT INTO model_promotions (
                model_id,
                previous_model_id,
                promoted_at,
                reason
            )
            VALUES (?, ?, ?, ?)
        """, (
            model_id,
            current_id,
            datetime.now(timezone.utc).isoformat(),
            "Controlled model rollback"
        ))

        db.commit()

    return {
        "restored_model": model_id,
        "replaced_model": current_id
    }
