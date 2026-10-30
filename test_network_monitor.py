
import time

import pytest

from Asterion import network_monitor as nm


class Counter:
    def __init__(
        self,
        bytes_sent=100,
        bytes_recv=200,
        packets_sent=10,
        packets_recv=20,
        errout=0,
        errin=0,
        dropout=0,
        dropin=0,
    ):
        self.bytes_sent = bytes_sent
        self.bytes_recv = bytes_recv
        self.packets_sent = packets_sent
        self.packets_recv = packets_recv
        self.errout = errout
        self.errin = errin
        self.dropout = dropout
        self.dropin = dropin


def test_no_recommendation_without_interface_evidence():
    assert nm.propose_network_optimizations({}) == []
    assert nm.propose_network_optimizations(
        {"interfaces": None}
    ) == []


def test_high_receive_rate_produces_advisory_only():
    proposals = nm.propose_network_optimizations({
        "interfaces": {
            "en0": {
                "bytes_recv_per_second": 60_000_000,
            }
        }
    })

    assert len(proposals) == 1
    assert proposals[0]["category"] == "capacity_review"
    assert proposals[0]["execution"] == "disabled"
    assert proposals[0]["evidence"]["bytes_recv_per_second"] == 60_000_000


def test_missing_or_invalid_rates_do_not_trigger_recommendations():
    proposals = nm.propose_network_optimizations({
        "interfaces": {
            "en0": {"bytes_recv_per_second": None},
            "en1": {"bytes_recv_per_second": float("nan")},
            "en2": {"bytes_recv_per_second": -1},
        }
    })

    assert proposals == []


def test_permission_denial_is_unavailable_not_zero(monkeypatch):
    class FakePsutil:
        AccessDenied = type("AccessDenied", (Exception,), {})

        @staticmethod
        def net_connections(kind="inet"):
            raise FakePsutil.AccessDenied()

    monkeypatch.setattr(nm, "psutil", FakePsutil)

    summary, status = nm.NetworkMonitor._collect_connections()

    assert summary is None
    assert status["status"] == "unavailable"


def test_first_sample_has_no_rate(monkeypatch):
    class FakePsutil:
        AccessDenied = PermissionError

        @staticmethod
        def net_io_counters(pernic=True, nowrap=False):
            return {"en0": Counter()}

        @staticmethod
        def net_connections(kind="inet"):
            raise PermissionError("restricted")

    monkeypatch.setattr(nm, "psutil", FakePsutil)

    monitor = nm.NetworkMonitor()
    result = monitor.collect()

    assert result["interfaces"]["en0"]["bytes_recv_per_second"] is None
    assert result["interfaces"]["en0"]["rate_status"] == "unavailable"
    assert result["metric_status"]["network_connections"]["status"] == "unavailable"


def test_second_sample_calculates_actual_rates(monkeypatch):
    snapshots = [
        {"en0": Counter(bytes_sent=100, bytes_recv=200)},
        {"en0": Counter(bytes_sent=300, bytes_recv=600)},
    ]
    ticks = iter([10.0, 12.0])

    class FakePsutil:
        AccessDenied = PermissionError

        @staticmethod
        def net_io_counters(pernic=True, nowrap=False):
            return snapshots.pop(0)

        @staticmethod
        def net_connections(kind="inet"):
            return []

    monkeypatch.setattr(nm, "psutil", FakePsutil)
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))

    monitor = nm.NetworkMonitor()
    monitor.collect()
    result = monitor.collect()

    assert result["interfaces"]["en0"]["bytes_recv_per_second"] == pytest.approx(200)
    assert result["interfaces"]["en0"]["bytes_sent_per_second"] == pytest.approx(100)


def test_counter_reset_does_not_create_rate(monkeypatch):
    snapshots = [
        {"en0": Counter(bytes_recv=500)},
        {"en0": Counter(bytes_recv=20)},
    ]
    ticks = iter([10.0, 12.0])

    class FakePsutil:
        AccessDenied = PermissionError

        @staticmethod
        def net_io_counters(pernic=True, nowrap=False):
            return snapshots.pop(0)

        @staticmethod
        def net_connections(kind="inet"):
            return []

    monkeypatch.setattr(nm, "psutil", FakePsutil)
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))

    monitor = nm.NetworkMonitor()
    monitor.collect()
    result = monitor.collect()

    assert result["interfaces"]["en0"]["bytes_recv_per_second"] is None
