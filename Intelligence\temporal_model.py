
from __future__ import annotations

import math


def rank_actions(
    candidates,
    lambda_cost=0.25,
    lambda_risk=1.0,
    lambda_uncertainty=0.5,
):
    """
    Rank candidate actions.

    Candidates must provide:
      action
      expected_benefit
      cost
      risk
      uncertainty
      authorized
      reversible
    """

    ranked = []

    for candidate in candidates:
        required = (
            "action",
            "expected_benefit",
            "cost",
            "risk",
            "uncertainty",
            "authorized",
            "reversible",
        )

        if any(key not in candidate for key in required):
            continue

        if candidate["authorized"] is not True:
            continue

        if candidate["reversible"] is not True:
            continue

        numbers = [
            candidate["expected_benefit"],
            candidate["cost"],
            candidate["risk"],
            candidate["uncertainty"],
        ]

        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            for x in numbers
        ):
            continue

        benefit, cost, risk, uncertainty = numbers

        if (
            benefit < 0
            or cost < 0
            or not 0 <= risk <= 1
            or not 0 <= uncertainty <= 1
        ):
            continue

        objective = (
            benefit
            - lambda_cost * cost
            - lambda_risk * risk
            - lambda_uncertainty * uncertainty
        )

        ranked.append({
            **candidate,
            "objective_score": objective,
        })

    return sorted(
        ranked,
        key=lambda x: x["objective_score"],
        reverse=True
    )


def select_action(
    candidates,
    maximum_risk=0.25,
    minimum_benefit=0.0,
):
    ranked = rank_actions(candidates)

    for candidate in ranked:
        if candidate["risk"] > maximum_risk:
            continue

        if candidate["expected_benefit"] <= minimum_benefit:
            continue

        return {
            "selected_action": candidate["action"],
            "objective_score": candidate["objective_score"],
            "expected_benefit": candidate["expected_benefit"],
            "risk": candidate["risk"],
            "cost": candidate["cost"],
            "uncertainty": candidate["uncertainty"],
            "selection_status": "candidate_selected",
            "execution_enabled": False,
        }

    return {
        "selected_action": None,
        "selection_status": "no_safe_candidate",
        "execution_enabled": False,
    }
