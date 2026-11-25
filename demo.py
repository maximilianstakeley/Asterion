"""Reproducible synthetic incidents with a held-out counterfactual oracle."""
from dataclasses import asdict
import random
from statistics import mean
import time

from .contracts import Observation
from .offline import Incident, evaluate, fit, recommend, split
from .simulation import Simulation, approval, baseline


def observation(host, ident, at, cpu):
    return Observation(ident, host, at, at, "fixture-v1", "utc_unix_s", 0, "synthetic", "measured", cpu)


def outcome(cpu, action, fault, noise):
    acted = action == "reduce_background_load"
    success = not fault or noise < (0.86 if acted else max(0.2, 0.8 - (cpu - 75) / 50))
    rollback = acted and not success
    return {
        "fault": fault, "success": success, "recovery_s": 24 if acted and fault and success
        else 45 if fault and success else None, "unsafe": False,
        "rollback_attempted": rollback, "rollback_ok": True if rollback else None,
        "cost": 0.24 if acted else 0.02,
    }


def fixture():
    rng = random.Random(42)
    start = 1_700_000_000
    cutoff = start + 100_000
    rows = []
    latent = {}
    test_hosts = {f"host-{i}" for i in range(40, 52)}
    for i in range(52):
        host = f"host-{i}"
        for j in range(30):
            stamp = (cutoff if host in test_hosts else start) + 100*j + 30
            ident = f"incident-{i}-{j}"
            cpu = rng.randint(42, 99)
            trend = rng.uniform(-0.8, 0.8)
            history = [asdict(observation(host, f"{ident}-{offset}", stamp + offset,
                                          min(100, max(0, cpu + trend*offset))))
                       for offset in (-20, -10, 0)]
            action = "reduce_background_load" if cpu >= 60 and rng.random() < 0.5 else "observe"
            fault = cpu >= 75 and rng.random() < 0.9
            noise = rng.random()
            latent[ident] = (cpu, fault, noise)
            rows.append(Incident.parse({
                "incident_id": ident, "behavior_policy_version": "fixture-logging-v1",
                "context": history, "action": action,
                "behavior_probabilities": {"observe": 0.5 if cpu >= 60 else 1,
                                           "reduce_background_load": 0.5 if cpu >= 60 else 0},
                "label_end_s": stamp + 60, **outcome(cpu, action, fault, noise),
            }))
    return rows, cutoff, test_hosts, latent


def direct_metrics(outputs, predictions):
    faults = [entry for entry in outputs if entry["fault"]]
    normals = [entry for entry in outputs if not entry["fault"]]
    recoveries = [entry["recovery_s"] for entry in faults if entry["recovery_s"] is not None]
    rollbacks = [entry for entry in outputs if entry["rollback_attempted"]]
    return {
        "task_success": mean(entry["success"] for entry in outputs),
        "mean_recovery_s": mean(recoveries) if recoveries else None,
        "false_alarm_rate": mean(entry["action"] != "observe" for entry in normals) if normals else None,
        "unsafe_action_rate": mean(entry["unsafe"] for entry in outputs),
        "rollback_success": mean(entry["rollback_ok"] for entry in rollbacks) if rollbacks else None,
        "normalized_resource_cost": mean(entry["cost"] for entry in outputs),
        "success_brier": mean((probability - int(entry["success"]))**2
                              for entry, probability in zip(outputs, predictions, strict=True)),
    }


def run_demo(ledger, operator_key):
    from .offline import context_key

    simulation = Simulation(ledger, {"demo-operator": operator_key})
    now = int(time.time())
    from uuid import uuid4

    run_id = uuid4().hex
    proposal = simulation.propose(observation("virtual-host", f"demo-success-{run_id}", now, 87))
    denied = simulation.run(proposal.proposal_id, None)
    allowed = simulation.run(proposal.proposal_id, approval(proposal, "demo-operator", operator_key, now))
    failure = simulation.propose(observation("virtual-host", f"demo-failure-{run_id}", now, 92))
    rolled_back = simulation.run(
        failure.proposal_id, approval(failure, "demo-operator", operator_key, now), fail_verification=True
    )
    rows, cutoff, hosts, latent = fixture()
    _, report = evaluate(rows, cutoff, hosts)
    train, test = split(rows, cutoff, hosts)
    model = fit(train, temporal=False)
    choices = {
        "always_observe": ["observe"] * len(test),
        "deterministic_threshold": [baseline(row.current.cpu_percent) for row in test],
        "snapshot_bandit": [recommend(row, model, False) for row in test],
    }
    benchmark = {}
    for name, actions in choices.items():
        outputs = []
        probabilities = []
        for row, action in zip(test, actions, strict=True):
            cpu, fault, noise = latent[row.incident_id]
            outputs.append({"action": action, **outcome(cpu, action, fault, noise)})
            probabilities.append(model.get(context_key(row, False), {}).get(action, {}).get("success_probability", 0.5))
        benchmark[name] = direct_metrics(outputs, probabilities)
    return {"missing_approval": denied, "verified_simulation": allowed, "verified_rollback": rolled_back,
            "synthetic_direct_benchmark": benchmark, "offline_evaluation": report, "audit": ledger.inspect()}
