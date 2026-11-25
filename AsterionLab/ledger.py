"""Append-only SQLite event history authenticated by an external key."""
from contextlib import contextmanager
import hashlib
import hmac
import json
import os
from pathlib import Path
import sqlite3
import time

from .contracts import canonical, identifier


def sign(key, value):
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("Use a key containing at least 32 bytes")
    return hmac.new(key, canonical(value).encode(), hashlib.sha256).hexdigest()


class Ledger:
    def __init__(self, path, key):
        self.path = Path(path)
        self.key = key
        sign(key, "key-check")
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        with self.transaction() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS events(
                seq INTEGER PRIMARY KEY, event_id TEXT UNIQUE NOT NULL,
                kind TEXT NOT NULL, unix_s INTEGER NOT NULL,
                payload TEXT NOT NULL, signature TEXT NOT NULL)""")
            for verb in ("UPDATE", "DELETE"):
                conn.execute(f"""CREATE TRIGGER IF NOT EXISTS events_no_{verb}
                    BEFORE {verb} ON events BEGIN
                    SELECT RAISE(ABORT, 'append-only event history'); END""")

    @contextmanager
    def transaction(self):
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def append(self, conn, event_id, kind, payload):
        identifier(event_id)
        identifier(kind)
        last = conn.execute("SELECT seq, signature FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        seq = last["seq"] + 1 if last else 1
        previous = last["signature"] if last else ""
        stamp = int(time.time())
        signature = sign(self.key, [seq, event_id, kind, stamp, payload, previous])
        conn.execute("INSERT INTO events VALUES (?,?,?,?,?,?)",
                     (seq, event_id, kind, stamp, canonical(payload), signature))
        return signature

    def verified_events(self, conn):
        events = {}
        previous = ""
        for expected, row in enumerate(conn.execute("SELECT * FROM events ORDER BY seq"), 1):
            payload = json.loads(row["payload"])
            signature = sign(self.key, [expected, row["event_id"], row["kind"], row["unix_s"], payload, previous])
            if row["seq"] != expected or not hmac.compare_digest(signature, row["signature"]):
                raise ValueError(f"Audit integrity verification failed at event {expected}")
            events[row["event_id"]] = (row["kind"], payload)
            previous = signature
        return events, previous

    def inspect(self):
        with self.transaction() as conn:
            events, head = self.verified_events(conn)
        return {"event_count": len(events), "head_hmac": head,
                "counts_by_kind": {kind: sum(k == kind for k, _ in events.values())
                                   for kind in sorted({k for k, _ in events.values()})}}
