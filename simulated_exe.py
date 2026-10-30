
"""Simulation-only transactional adapter with scoped HMAC authorization.

This module modifies in-memory simulated state only. It does not access,
inspect, or modify real hosts. Production/live execution is intentionally
absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import math
import secrets
import threading
import time
from collections.abc import Mapping
from typing import Any, Callable

from .contracts import CandidateAction, SafetyVerdict


SIMULATED_ACTIONS = frozenset({"reduce_noncritical_workload"})

MIN_SECRET_LENGTH = 32
MIN_TTL_SECONDS = 1
MAX_TTL_SECONDS = 300
MAX_SIMULATED_RISK = 0.20

MIN_WORKLOAD_LEVEL = 0.10
MAX_WORKLOAD_LEVEL = 1.00
WORKLOAD_REDUCTION = 0.15


@dataclass(frozen=True)
class AuthorizationGrant:
    """Short-lived authorization capability scoped to one target and action."""

    target: str
    action: str
    expires_at: int
    nonce: str
    signature: str


class AuthorizationAuthority:
    """Issue and verify short-lived HMAC-signed simulation grants.

    The secret must be generated and managed independently. This in-memory
    authority is suitable for controlled simulation/testing, not a complete
    production identity or authorization system.
    """

    def __init__(self, secret: bytes) -> None:
        if not isinstance(secret, bytes) or len(secret) < MIN_SECRET_LENGTH:
            raise ValueError(
                f"authorization secret must be bytes of at least "
                f"{MIN_SECRET_LENGTH} bytes"
            )

        self._secret = secret
        self._used_nonces: set[str] = set()
        self._lock = threading.Lock()

    @staticmethod
    def _payload(
        *,
        target: str,
        action: str,
        expires_at: int,
        nonce: str,
    ) -> dict[str, Any]:
        """Construct the canonical fields covered by the signature."""
        return {
            "target": target,
            "action": action,
            "expires_at": expires_at,
            "nonce": nonce,
        }

    def _sign(self, payload: Mapping[str, Any]) -> str:
        canonical = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

        return hmac.new(
            self._secret,
            canonical,
            sha256,
        ).hexdigest()

    def issue(
        self,
        *,
        target: str,
        action: str,
        ttl_seconds: int = 60,
    ) -> AuthorizationGrant:
        """Issue a grant limited to an allowlisted simulated action."""
        if not isinstance(target, str) or not target.strip():
            raise ValueError("target must be a non-empty string")

        if action not in SIMULATED_ACTIONS:
            raise ValueError(
                "authority only issues grants for allowlisted simulated actions"
            )

        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
            raise ValueError("ttl_seconds must be an integer")

        if not MIN_TTL_SECONDS <= ttl_seconds <= MAX_TTL_SECONDS:
            raise ValueError(
                f"ttl_seconds must be {MIN_TTL_SECONDS}..{MAX_TTL_SECONDS}"
            )

        payload = self._payload(
            target=target,
            action=action,
            expires_at=int(time.time()) + ttl_seconds,
            nonce=secrets.token_hex(16),
        )

        return AuthorizationGrant(
            **payload,
            signature=self._sign(payload),
        )

    def verify(
        self,
        grant: AuthorizationGrant | None,
        *,
        target: str,
        action: str,
    ) -> tuple[bool, str]:
        """Verify signature, expiry, scope, allowlist, and single-use nonce."""
        if grant is None:
            return False, "missing_authorization_grant"

        if not isinstance(grant, AuthorizationGrant):
            return False, "invalid_authorization_grant"

        if (
            not isinstance(grant.target, str)
            or not isinstance(grant.action, str)
            or not isinstance(grant.nonce, str)
            or not isinstance(grant.signature, str)
            or isinstance(grant.expires_at, bool)
            or not isinstance(grant.expires_at, int)
        ):
            return False, "invalid_authorization_grant_fields"

        if not grant.target.strip() or not grant.nonce or not grant.signature:
            return False, "invalid_authorization_grant_fields"

        payload = self._payload(
            target=grant.target,
            action=grant.action,
            expires_at=grant.expires_at,
            nonce=grant.nonce,
        )

        expected_signature = self._sign(payload)

        if not hmac.compare_digest(expected_signature, grant.signature):
            return False, "invalid_grant_signature"

        if grant.expires_at <= int(time.time()):
            return False, "authorization_grant_expired"

        if grant.target != target or grant.action != action:
            return False, "grant_scope_mismatch"

        if grant.action not in SIMULATED_ACTIONS:
            return False, "action_not_allowlisted_for_simulation"

        # The check-and-add is atomic, so concurrent verification attempts
        # cannot successfully reuse the same grant.
        with self._lock:
            if grant.nonce in self._used_nonces:
                return False, "authorization_grant_replayed"

            self._used_nonces.add(grant.nonce)

        return True, "authorized_for_simulation_only"


class SimulatedHostAdapter:
    """Thread-safe in-memory stand-in for a host.

    All state belongs to this Python object. No operating-system, process,
    network, or remote-host operations are performed.
    """

    def __init__(
        self,
        target: str,
        workload_level: float = 0.70,
    ) -> None:
        if not isinstance(target, str) or not target.strip():
            raise ValueError("target must be a non-empty string")

        if isinstance(workload_level, bool):
            raise ValueError("workload_level must be a finite number in [0, 1]")

        try:
            level = float(workload_level)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "workload_level must be a finite number in [0, 1]"
            ) from exc

        if (
            not math.isfinite(level)
            or not MIN_WORKLOAD_LEVEL <= level <= MAX_WORKLOAD_LEVEL
        ):
            raise ValueError("workload_level must be finite and in [0.1, 1.0]")

        self.target = target
        self._state = {
            "workload_level": level,
            "revision": 0,
        }
        self._transaction_lock = threading.RLock()

    def snapshot(self) -> dict[str, Any]:
        """Return a copy of the current simulated state."""
        with self._transaction_lock:
            return dict(self._state)

    def apply(self, action: str) -> dict[str, Any]:
        """Apply the supported simulated action to in-memory state."""
        if action not in SIMULATED_ACTIONS:
            raise ValueError("action_not_supported_by_simulator")

        with self._transaction_lock:
            current_level = self._state["workload_level"]

            if current_level <= MIN_WORKLOAD_LEVEL:
                raise ValueError("simulated_workload_already_at_minimum")

            next_level = max(
                MIN_WORKLOAD_LEVEL,
                current_level - WORKLOAD_REDUCTION,
            )

            if next_level >= current_level:
                raise RuntimeError("simulated_state_transition_not_progressing")

            self._state["workload_level"] = next_level
            self._state["revision"] += 1

            return dict(self._state)

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a previously captured, validated in-memory snapshot."""
        if not isinstance(snapshot, Mapping):
            raise ValueError("snapshot must be a mapping")

        level = snapshot.get("workload_level")
        revision = snapshot.get("revision")

        if (
            isinstance(level, bool)
            or not isinstance(level, (int, float))
            or not math.isfinite(float(level))
            or not MIN_WORKLOAD_LEVEL <= float(level) <= MAX_WORKLOAD_LEVEL
        ):
            raise ValueError("snapshot contains an invalid workload level")

        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 0
        ):
            raise ValueError("snapshot contains an invalid revision")

        if set(snapshot.keys()) != {"workload_level", "revision"}:
            raise ValueError("snapshot contains unexpected fields")

        with self._transaction_lock:
            self._state = {
                "workload_level": float(level),
                "revision": revision,
            }


class SimulatedTransactionalExecutor:
    """Execute an authorized action against simulated state with rollback.

    A successful result means only that the in-memory simulation changed.
    It does not mean a real host action was executed.
    """

    def __init__(
        self,
        authority: AuthorizationAuthority,
        adapter: SimulatedHostAdapter,
        fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        if not isinstance(authority, AuthorizationAuthority):
            raise TypeError("authority must be an AuthorizationAuthority")

        if not isinstance(adapter, SimulatedHostAdapter):
            raise TypeError("adapter must be a SimulatedHostAdapter")

        if fault_hook is not None and not callable(fault_hook):
            raise TypeError("fault_hook must be callable or None")

        self.authority = authority
        self.adapter = adapter
        self.fault_hook = fault_hook or (lambda stage: None)

    @staticmethod
    def _blocked(reason: str) -> dict[str, Any]:
        return {
            "status": "blocked",
            "executed": False,
            "live_system_changes": False,
            "reason": reason,
        }

    @staticmethod
    def _valid_candidate_risk(candidate: Any) -> float | None:
        risk = getattr(candidate, "risk", None)

        if isinstance(risk, bool):
            return None

        try:
            value = float(risk)
        except (TypeError, ValueError, OverflowError):
            return None

        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return None

        return value

    def execute(
        self,
        candidate: CandidateAction,
        verdict: SafetyVerdict,
        grant: AuthorizationGrant | None,
    ) -> dict[str, Any]:
        """Run the simulated transaction, verifying and rolling back as needed."""
        if candidate is None:
            return self._blocked("missing_candidate")

        action = getattr(candidate, "action", None)
        if not isinstance(action, str) or not action.strip():
            return self._blocked("invalid_candidate_action")

        if action not in SIMULATED_ACTIONS:
            return self._blocked("action_not_supported_by_simulator")

        if verdict is None or getattr(verdict, "allowed", None) is not True:
            return self._blocked("safety_gate_blocked")

        risk = self._valid_candidate_risk(candidate)
        if risk is None:
            return self._blocked("invalid_candidate_risk")

        if risk > MAX_SIMULATED_RISK:
            return self._blocked("simulator_risk_limit_exceeded")

        valid, reason = self.authority.verify(
            grant,
            target=self.adapter.target,
            action=action,
        )

        if not valid:
            return self._blocked(reason)

        # Hold the lock across snapshot, mutation, verification, and rollback.
        # This prevents another transaction using this adapter from interleaving
        # with the current transaction.
        with self.adapter._transaction_lock:
            before = self.adapter.snapshot()

            try:
                self.fault_hook("before_apply")

                after = self.adapter.apply(action)

                self.fault_hook("after_apply")

                # Verify the expected bounded state transition.
                valid_transition = (
                    isinstance(after, dict)
                    and isinstance(after.get("workload_level"), (int, float))
                    and not isinstance(after.get("workload_level"), bool)
                    and math.isfinite(float(after["workload_level"]))
                    and before["workload_level"] > after["workload_level"]
                    and MIN_WORKLOAD_LEVEL <= after["workload_level"] <= MAX_WORKLOAD_LEVEL
                    and after.get("revision") == before["revision"] + 1
                )

                if not valid_transition:
                    raise RuntimeError("postcondition_verification_failed")

                self.fault_hook("after_verify")

                return {
                    "status": "simulated_success",
                    "executed": True,
                    "live_system_changes": False,
                    "execution_mode": "in_memory_simulation",
                    "before": before,
                    "after": dict(after),
                    "rollback": "not_needed",
                    "rollback_verified": None,
                }

            except Exception as exc:
                try:
                    self.adapter.restore(before)
                    after_rollback = self.adapter.snapshot()
                    restored = after_rollback == before
                except Exception:
                    restored = False
                    after_rollback = {"state_unavailable": True}

                return {
                    "status": "rolled_back" if restored else "rollback_failed",
                    "executed": True,
                    "live_system_changes": False,
                    "execution_mode": "in_memory_simulation",
                    "error": type(exc).__name__,
                    "before": before,
                    "after": after_rollback,
                    "rollback": "verified" if restored else "failed",
                    "rollback_verified": restored,
                }
