
import math

import pytest
from pydantic import ValidationError

from Asterion.contracts import Observation


def make_observation(load_1m):
    return Observation(
        agent_id="test-agent",
        host="test-host",
        text="Validation test",
        metrics={
            "cpu_percent": 10.0,
            "memory_percent": 20.0,
            "disk_percent": 30.0,
            "network_connections": None,
            "load_1m": load_1m,
        },
    )


@pytest.mark.parametrize(
    "value",
    [
        0,
        0.0,
        1,
        1.5,
    ],
)
def test_load_accepts_finite_nonnegative_numbers(value):
    observation = make_observation(value)
    assert observation.metrics["load_1m"] == value


def test_load_accepts_none_when_unavailable():
    observation = make_observation(None)
    assert observation.metrics["load_1m"] is None


@pytest.mark.parametrize(
    "value",
    [
        -1,
        -0.001,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        False,
        "1.5",
    ],
)
def test_load_rejects_invalid_values(value):
    with pytest.raises(ValidationError):
        make_observation(value)


def test_load_rejects_all_nonfinite_float_categories():
    for value in (float("nan"), float("inf"), float("-inf")):
        assert not math.isfinite(value)

        with pytest.raises(ValidationError):
            make_observation(value)
