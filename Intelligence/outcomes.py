
from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time

from dataclasses import dataclass
from datetime import datetime, timezone

from ..configuration import DB_PATH


@dataclass(frozen=True)
class AgentCredential:
    agent_id: str
    token: str
    signing_secret: str


def initialize_security():
    with sqlite3.connect(DB_PATH) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS agent_credentials (
            agent_id TEXT PRIMARY KEY,
            token_hash TEXT NOT NULL,
            signing_secret_hash TEXT NOT NULL,
            signing_secret_encrypted TEXT,
            allowed_hosts_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            revoked_at TEXT,
            active INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS agent_nonces (
            agent_id TEXT NOT NULL,
            nonce TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            PRIMARY KEY(agent_id, nonce)
        );

        CREATE INDEX IF NOT EXISTS idx_agent_nonce_timestamp
        ON agent_nonces(timestamp);
        """)


def provision_agent(agent_id: str, allowed_hosts: list[str]):
    """Create a new agent identity. Return secrets only once."""

    if not agent_id.strip():
        raise ValueError("agent_id is required")

    if not allowed_hosts:
        raise ValueError("At least one host must be authorized")

    token = secrets.token_urlsafe(48)
    signing_secret = secrets.token_urlsafe(48)

    token_hash = hashlib.sha256(
        token.encode()
    ).hexdigest()

    signing_hash = hashlib.sha256(
        signing_secret.encode()
    ).hexdigest()

    import json

    with sqlite3.connect(DB_PATH) as db:
        db.execute("""
            INSERT INTO agent_credentials (
                agent_id,
                token_hash,
                signing_secret_hash,
                allowed_hosts_json,
                created_at,
                active
            )
            VALUES (?, ?, ?, ?, ?, 1)
        """, (
            agent_id,
            token_hash,
            signing_hash,
            json.dumps(sorted(set(allowed_hosts))),
            datetime.now(timezone.utc).isoformat()
        ))

    return AgentCredential(
        agent_id=agent_id,
        token=token,
        signing_secret=signing_secret
    )


def revoke_agent(agent_id: str):
    with sqlite3.connect(DB_PATH) as db:
        cursor = db.execute("""
            UPDATE agent_credentials
            SET active = 0,
                revoked_at = ?
            WHERE agent_id = ?
        """, (
            datetime.now(timezone.utc).isoformat(),
            agent_id
        ))

        if cursor.rowcount != 1:
            raise ValueError("Unknown agent")


def signature_for(
    method: str,
    path: str,
    timestamp: str,
    nonce: str,
    body: bytes,
    signing_secret: str
):
    body_hash = hashlib.sha256(body).hexdigest()

    canonical = "\n".join([
        method.upper(),
        path,
        timestamp,
        nonce,
        body_hash
    ])

    return hmac.new(
        signing_secret.encode(),
        canonical.encode(),
        hashlib.sha256
    ).hexdigest()


def authenticate_agent(
    agent_id: str,
    token: str,
    timestamp: str,
    nonce: str,
    signature: str,
    method: str,
    path: str,
    body: bytes,
    host: str,
    tolerance_seconds: int = 60
):
    import json

    try:
        request_time = int(timestamp)
    except (ValueError, TypeError):
        raise PermissionError("Invalid timestamp")

    if abs(time.time() - request_time) > tolerance_seconds:
        raise PermissionError("Expired request")

    if not nonce or len(nonce) < 16:
        raise PermissionError("Invalid nonce")

    with sqlite3.connect(DB_PATH) as db:
        row = db.execute("""
            SELECT token_hash,
                   signing_secret_hash,
                   allowed_hosts_json,
                   active
            FROM agent_credentials
            WHERE agent_id = ?
        """, (agent_id,)).fetchone()

        if row is None or row[3] != 1:
            raise PermissionError("Unknown or revoked agent")

        expected_token_hash = hashlib.sha256(
            token.encode()
        ).hexdigest()

        if not hmac.compare_digest(
            expected_token_hash,
            row[0]
        ):
            raise PermissionError("Invalid agent token")

        # In this first version the signing secret is not retrievable
        # from its stored hash. Use the credential-verification
        # variant below, which verifies an HMAC with the supplied
        # signing secret against its stored hash.
        raise PermissionError(
            "Use authenticate_agent_signed() with signing_secret"
        )


def authenticate_agent_signed(
    agent_id: str,
    token: str,
    signing_secret: str,
    timestamp: str,
    nonce: str,
    signature: str,
    method: str,
    path: str,
    body: bytes,
    host: str,
    tolerance_seconds: int = 60
):
    import json

    try:
        request_time = int(timestamp)
    except (TypeError, ValueError):
        raise PermissionError("Invalid timestamp")

    if abs(time.time() - request_time) > tolerance_seconds:
        raise PermissionError("Expired request")

    with sqlite3.connect(DB_PATH) as db:
        db.execute("BEGIN IMMEDIATE")

        row = db.execute("""
            SELECT token_hash,
                   signing_secret_hash,
                   allowed_hosts_json,
                   active
            FROM agent_credentials
            WHERE agent_id = ?
        """, (agent_id,)).fetchone()

        if row is None or row[3] != 1:
            raise PermissionError("Agent revoked or unknown")

        token_hash = hashlib.sha256(
            token.encode()
        ).hexdigest()

        secret_hash = hashlib.sha256(
            signing_secret.encode()
        ).hexdigest()

        if not hmac.compare_digest(token_hash, row[0]):
            raise PermissionError("Invalid token")

        if not hmac.compare_digest(secret_hash, row[1]):
            raise PermissionError("Invalid signing secret")

        allowed_hosts = json.loads(row[2])

        if host not in allowed_hosts:
            raise PermissionError("Host identity mismatch")

        expected = signature_for(
            method,
            path,
            timestamp,
            nonce,
            body,
            signing_secret
        )

        if not hmac.compare_digest(expected, signature):
            raise PermissionError("Invalid signature")

        # BEGIN IMMEDIATE ensures competing requests cannot both
        # accept the same nonce.
        try:
            db.execute("""
                INSERT INTO agent_nonces (
                    agent_id, nonce, timestamp
                )
                VALUES (?, ?, ?)
            """, (agent_id, nonce, request_time))
        except sqlite3.IntegrityError:
            raise PermissionError("Replay detected")

        db.execute("""
            DELETE FROM agent_nonces
            WHERE timestamp < ?
        """, (int(time.time()) - 300,))

        db.commit()

        return True
