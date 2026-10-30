
import psutil

from Asterion.network_telemetry import (
    NetworkMeasurement,
    measure_network_connections,
)


def test_genuine_zero_is_not_unavailable():
    result = measure_network_connections(
        provider=lambda kind="all": []
    )

    assert result == NetworkMeasurement(
        available=True,
        connection_count=0,
        reason=None,
    )


def test_positive_count_is_preserved():
    fake_connections = [object(), object(), object()]

    result = measure_network_connections(
        provider=lambda kind="all": fake_connections
    )

    assert result.available is True
    assert result.connection_count == 3
    assert result.reason is None


def test_access_denied_is_unavailable_not_zero():
    def denied(kind="all"):
        raise psutil.AccessDenied(pid=1234)

    result = measure_network_connections(provider=denied)

    assert result.available is False
    assert result.connection_count is None
    assert result.reason == "AccessDenied"


def test_operating_system_error_is_unavailable():
    def failed(kind="all"):
        raise OSError("simulated measurement failure")

    result = measure_network_connections(provider=failed)

    assert result.available is False
    assert result.connection_count is None
    assert result.reason == "OSError"


def test_none_from_provider_is_not_a_zero_measurement():
    result = measure_network_connections(
        provider=lambda kind="all": None
    )

    assert result.available is False
    assert result.connection_count is None
    assert result.reason == "provider_returned_no_measurement"
