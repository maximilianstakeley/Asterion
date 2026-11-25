"""Local commands for review, evaluation and simulation."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import secrets
import sqlite3
import sys
from tempfile import TemporaryDirectory
import time

from .contracts import Observation, Proposal, canonical, identifier
from .demo import fixture, run_demo
from .ledger import Ledger
from .offline import Incident, parse_history
from .registry import Registry
from .simulation import Simulation, approval

MAX_ROW_BYTES = 262_144
MAX_ROWS = 100_000
MAX_LOG_BYTES = 64 * 1024**2


def load_json(text):
    def unique(pairs):
        data = {}
        for name, value in pairs:
            if name in data:
                raise ValueError(f"Repeated JSON field {name}")
            data[name] = value
        return data

    def invalid_number(value):
        raise ValueError(f"Invalid JSON number {value}")

    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid_number)


def document(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_ROW_BYTES + 1)
    if len(raw) > MAX_ROW_BYTES:
        raise ValueError("JSON document exceeds size limit")
    return load_json(raw)


def incidents(path):
    rows = []
    total_bytes = 0
    with Path(path).open("rb") as stream:
        while raw := stream.readline(MAX_ROW_BYTES + 1):
            total_bytes += len(raw)
            if len(raw) > MAX_ROW_BYTES or total_bytes > MAX_LOG_BYTES or len(rows) >= MAX_ROWS:
                raise ValueError("Incident file exceeds size limit")
            if raw.strip():
                rows.append(Incident.parse(load_json(raw)))
    return rows


def key(name):
    if not os.getenv(name):
        raise ValueError(f"Set {name} to a randomly generated secret of at least 32 bytes encoded as hex")
    return bytes.fromhex(os.environ[name])


def reviewer(ledger):
    return Registry(ledger, os.environ.get("ASTERION_REVIEWER_ID", "reviewer"), key("ASTERION_REVIEWER_KEY"))


def demo_in(directory):
    directory.mkdir(mode=0o700, exist_ok=True)
    ledger_key = secrets.token_bytes(32)
    # Exclusive creation avoids overwriting credentials belonging to an existing audit.
    fd = os.open(directory / "audit.key", os.O_WRONLY | os.O_EXCL | os.O_CREAT, 0o600)
    with os.fdopen(fd, "w") as out:
        out.write(ledger_key.hex() + "\n")
    ledger = Ledger(directory / "simulation.sqlite", ledger_key)
    return run_demo(ledger, secrets.token_bytes(32))


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    main.add_argument("--ledger", type=Path, default=Path("asterion-actions.sqlite"))
    sub = main.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo", help="run synthetic benchmarks and exercise the shield and rollback")
    p.add_argument("--dir", type=Path, help="persist the demo audit and private key in a new directory")
    sub.add_parser("fixture", help="emit labeled synthetic incident JSON Lines")
    p = sub.add_parser("verify-log")
    p.add_argument("--expected-head", help="previously stored event-chain HMAC for an unchanged checkpoint")
    p = sub.add_parser("propose")
    p.add_argument("observation", help="JSON observation or JSON list of observations")
    p = sub.add_parser("approve")
    p.add_argument("proposal", help="JSON proposal file")
    p.add_argument("--confirm", action="store_true", required=True)
    p = sub.add_parser("simulate")
    p.add_argument("proposal_id")
    p.add_argument("approval", help="signed JSON approval")
    p.add_argument("--fail-verification", action="store_true", help="exercise virtual rollback")
    p = sub.add_parser("evaluate")
    p.add_argument("incidents", help="JSON Lines of logged incidents")
    p.add_argument("--cutoff-s", type=int, required=True)
    p.add_argument("--holdout", nargs="+", required=True, metavar="HOST")
    p.add_argument("--variant", choices=["snapshot", "temporal"], default="snapshot")
    p = sub.add_parser("review")
    p.add_argument("model")
    p.add_argument("stage", choices=["shadow", "canary", "simulation_active"])
    p.add_argument("--regression-report", required=True)
    p.add_argument("--security-report", required=True)
    p.add_argument("--attest-passing", action="store_true", required=True)
    p = sub.add_parser("advance")
    p.add_argument("model")
    p.add_argument("stage", choices=["shadow", "canary", "simulation_active"])
    p.add_argument("attestation")
    p = sub.add_parser("monitor")
    p.add_argument("model")
    p.add_argument("incidents")
    p = sub.add_parser("rollback")
    p.add_argument("model")
    return main


def dispatch(args):
    command = args.command
    if command == "demo":
        if args.dir:
            return demo_in(args.dir)
        with TemporaryDirectory(prefix="asterion-demo-") as directory:
            return demo_in(Path(directory))
    if command == "fixture":
        rows, _, _, _ = fixture()
        for row in rows:
            print(canonical(asdict(row)))
        return None
    operator = os.environ.get("ASTERION_OPERATOR_ID", "operator")
    if command == "approve":
        return approval(Proposal.parse(document(args.proposal)), operator, key("ASTERION_OPERATOR_KEY"), int(time.time()))
    ledger = Ledger(args.ledger, key("ASTERION_AUDIT_KEY"))
    if command == "verify-log":
        checked = ledger.inspect()
        if args.expected_head is not None and args.expected_head != checked["head_hmac"]:
            raise ValueError("Ledger head does not match external checkpoint")
        return checked
    if command == "propose":
        source = document(args.observation)
        context = parse_history(source) if type(source) is list else None
        obs = context.current if context else Observation.parse(source)
        simulation = Simulation(ledger, {})
        if obs.cpu_status != "measured":
            return asdict(simulation.propose(obs))
        if context is None:
            context = parse_history([source])
        return asdict(simulation.propose(obs, context=context, routing=lambda events: Registry.route(events, context)))
    if command == "simulate":
        simulation = Simulation(ledger, {operator: key("ASTERION_OPERATOR_KEY")})
        return simulation.run(identifier(args.proposal_id), document(args.approval),
                              fail_verification=args.fail_verification)
    if command == "evaluate":
        hosts = {identifier(host) for host in args.holdout}
        version, report = Registry(ledger).admit(incidents(args.incidents), args.cutoff_s, hosts, args.variant)
        return {"model_version": version, **report}
    registry = reviewer(ledger)
    if command == "review":
        return registry.sign_review(args.model, args.stage, args.regression_report, args.security_report)
    if command == "advance":
        return registry.transition(args.model, args.stage, document(args.attestation))
    if command == "monitor":
        return registry.monitor(args.model, incidents(args.incidents))
    if command == "rollback":
        return registry.rollback(args.model)
    raise ValueError("Unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
        if result is not None:
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 2 if isinstance(result, dict) and result.get("status") == "denied" else 0
    except (ValueError, OSError, sqlite3.Error, TypeError) as error:
        print(f"asterion-action-lab: {error}", file=sys.stderr)
        return 2
