"""A deterministic shield and pure simulator. No system command adapters."""
from dataclasses import asdict
import hmac
import time
from uuid import uuid4

from .contracts import (
    FEATURE_VERSION, SHIELD_VERSION, Observation, Proposal, action_rules,
    digest, exact, identifier, integer,
)
from .ledger import sign

POLICY_VERSION = "cpu-threshold-v1"
APPROVAL_ROLE = "human_simulation_operator"


def baseline(cpu_percent):
    return "reduce_background_load" if cpu_percent >= 80 else "observe"


def approval(proposal, operator_id, key, now, ttl_s=60):
    identifier(operator_id)
    integer(now, 0, 2**53 - 301)
    integer(ttl_s, 1, 300)
    body = {
        "proposal_id": proposal.proposal_id,
        "proposal_digest": digest(asdict(proposal)),
        "operator_id": operator_id,
        "role": APPROVAL_ROLE,
        "issued_s": now,
        "expires_s": now + ttl_s,
        "nonce": uuid4().hex,
    }
    return {"body": body, "signature": sign(key, body)}


def check_approval(envelope, proposal, key_by_operator, now):
    exact(envelope, {"body", "signature"})
    body = envelope["body"]
    exact(body, {"proposal_id", "proposal_digest", "operator_id", "role", "issued_s", "expires_s", "nonce"})
    operator = identifier(body["operator_id"])
    identifier(body["nonce"])
    issued = integer(body["issued_s"], 0, now)
    integer(body["expires_s"], now, issued + 300)
    if not issued < body["expires_s"]:
        raise ValueError("Empty approval interval")
    if body["proposal_id"] != proposal.proposal_id or body["proposal_digest"] != digest(asdict(proposal)):
        raise ValueError("Approval is for a different proposal")
    if body["role"] != APPROVAL_ROLE or operator not in key_by_operator:
        raise ValueError("Operator is not authorized")
    signature = envelope["signature"]
    if type(signature) is not str or not signature.isascii() or not hmac.compare_digest(
        signature, sign(key_by_operator[operator], body)
    ):
        raise ValueError("Approval signature does not verify")
    return body


class Simulation:
    def __init__(self, ledger, key_by_operator):
        self.ledger = ledger
        self.key_by_operator = dict(key_by_operator)
        for key in self.key_by_operator.values():
            sign(key, "key-check")
            if key == ledger.key:
                raise ValueError("Use separate operator and ledger keys")

    def propose(self, obs, *, context=None, routing=None):
        evidence = asdict(obs)
        observations = context.context if context else (obs,)
        if observations[-1] != obs:
            raise ValueError("Latest context observation must match the proposal")
        model_version = "deterministic-baseline-v1"
        action = baseline(obs.cpu_percent) if obs.cpu_status == "measured" else "observe"
        with self.ledger.transaction() as conn:
            events, _ = self.ledger.verified_events(conn)
            release = None
            if routing is not None:
                version, action, release = routing(events)
                if version:
                    model_version = version
            proposal = Proposal(
                proposal_id="proposal-" + uuid4().hex, target=obs.host_id, action=action,
                evidence_ids=tuple(item.observation_id for item in observations), observation_digest=digest(evidence), model_version=model_version,
                policy_version=POLICY_VERSION, evaluation_mode="simulation", **action_rules(action),
            )
            wire_proposal = asdict(proposal)
            for item in observations:
                payload = asdict(item)
                previous = events.get(item.observation_id)
                if previous is None:
                    self.ledger.append(conn, item.observation_id, "observation", payload)
                elif previous != ("observation", payload):
                    raise ValueError("Observation ID was reused with different content")
            candidate_id = "candidate-" + proposal.proposal_id
            self.ledger.append(conn, candidate_id, "candidate_action", wire_proposal)
            self.ledger.append(conn, proposal.proposal_id, "policy_decision", {
                "proposal": wire_proposal, "candidate_id": candidate_id, "feature_version": FEATURE_VERSION,
                "reason": "released_model_proposal" if model_version != "deterministic-baseline-v1"
                          else "deterministic_threshold",
                "release": release,
            })
        return proposal

    def run(self, proposal_id, envelope, *, fail_verification=False):
        evidence_prefix = "result-" + proposal_id
        with self.ledger.transaction() as conn:
            events, _ = self.ledger.verified_events(conn)
            kind, event = events.get(proposal_id, (None, None))
            if kind != "policy_decision":
                raise ValueError("Unknown proposal")
            proposal = Proposal.parse(event["proposal"])
            obs_kind, obs_payload = events.get(proposal.evidence_ids[-1], (None, None))
            if obs_kind != "observation" or digest(obs_payload) != proposal.observation_digest:
                raise ValueError("Proposal evidence is unavailable or differs")
            obs = Observation.parse(obs_payload)
            now = int(time.time())
            try:
                operator = check_approval(envelope, proposal, self.key_by_operator, now)
                if evidence_prefix in events:
                    recorded = events.get("approval-" + operator["nonce"])
                    if recorded == ("operator_approval", envelope):
                        return events[evidence_prefix][1]
                    raise ValueError("This proposal already has an execution result")
                if "approval-" + operator["nonce"] in events:
                    raise ValueError("Approval already used")
                version = proposal.model_version
                if version.startswith("bandit-"):
                    stages = [record["stage"] for kind, record in events.values()
                              if kind == "release_stage" and record["model_version"] == version]
                    if not stages or stages[-1] not in {"canary", "simulation_active"}:
                        raise ValueError("Model was withdrawn before simulation")
                self._shield(proposal, obs, now)
            except ValueError as error:
                self.ledger.append(conn, "denial-" + uuid4().hex, "shield_denial", {
                    "proposal_id": proposal_id, "reason": str(error), "shield_version": SHIELD_VERSION,
                })
                return {"status": "denied", "reason": str(error)}
            self.ledger.append(conn, "approval-" + operator["nonce"], "operator_approval", envelope)
            self.ledger.append(conn, "shield-" + proposal_id, "shield_verdict",
                               {"allowed": True, "mode": "simulation", "shield_version": SHIELD_VERSION})
            result = self._simulate(proposal, obs.cpu_percent, fail_verification)
            self.ledger.append(conn, evidence_prefix, "execution_result", result)
            self.ledger.append(conn, "outcome-" + proposal_id, "verified_outcome", {
                "proposal_id": proposal_id, "result_id": evidence_prefix, "mode": "simulation",
                "effect_verified": result["effect_verified"], "rollback_verified": result["rollback_verified"],
                "eligible_as_production_feedback": False,
            })
        return result

    @staticmethod
    def _shield(proposal, observation, now):
        if proposal.target != observation.host_id:
            raise ValueError("Target does not match observed host")
        age_s = now - observation.observed_at_s + observation.clock_uncertainty_ms / 1000
        if (observation.cpu_status != "measured" or not 0 <= age_s <= 15
                or observation.observed_at_s > now or observation.received_at_s > now
                or observation.clock_uncertainty_ms > 1000):
            raise ValueError("Fresh measured CPU with a bounded clock is required")
        if proposal.action == "reduce_background_load" and observation.cpu_percent < 60:
            raise ValueError("Simulated load reduction requires CPU at or above 60%")

    @staticmethod
    def _simulate(proposal, before, force_failure):
        deadline = time.monotonic() + proposal.timeout_s
        state = {"cpu_percent": before}
        snapshot = state.copy()
        if proposal.action == "reduce_background_load":
            state["cpu_percent"] = max(0.0, state["cpu_percent"] - 10)
        after = state["cpu_percent"]
        verified = (before - after >= 5 if proposal.action == "reduce_background_load" else state == snapshot)
        verified = verified and not force_failure and time.monotonic() <= deadline
        if not verified:
            state.update(snapshot)
        return {
            "proposal_id": proposal.proposal_id, "mode": "simulation", "live_executed": False,
            "status": "verified" if verified else "rolled_back",
            "before_cpu_percent": before, "attempted_cpu_percent": after,
            "final_cpu_percent": state["cpu_percent"], "effect_verified": verified,
            "rollback_verified": None if verified else state == snapshot,
        }
