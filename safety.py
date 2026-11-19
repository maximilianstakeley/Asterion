
"""Validated request schemas for the Asterion API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Metrics(BaseModel):
    """Optional host metrics; None means unavailable, not measured zero."""

    model_config = ConfigDict(
        extra="allow",
        allow_inf_nan=False,
    )

    cpu_percent: float | None = Field(default=None, ge=0, le=100)
    memory_percent: float | None = Field(default=None, ge=0, le=100)
    disk_percent: float | None = Field(default=None, ge=0, le=100)

    network_connections: int | None = Field(
        default=None,
        strict=True,
        ge=0,
        le=10_000_000,
        description="None means unavailable; zero means measured zero.",
    )

    load_1m: float | None = Field(default=None, ge=0)

    # Structured measurements from TelemetryCollector.
    disk: dict[str, Any] | None = None
    load_average: dict[str, Any] | None = None
    network_traffic: dict[str, Any] | None = None
    network_interfaces: dict[str, Any] | None = None
    connection_metadata: dict[str, Any] | None = None


class AnalyzeRequest(BaseModel):
    """Request to analyze text and optional host telemetry."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    agent_id: str = Field(
        default="api-client",
        min_length=1,
        max_length=128,
    )
    host: str = Field(
        default="unknown-host",
        min_length=1,
        max_length=255,
    )
    text: str = Field(
        min_length=1,
        max_length=200_000,
    )
    metrics: Metrics


class FeedbackRequest(BaseModel):
    """Analyst or external-source feedback for classifier evaluation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    text: str = Field(
        min_length=1,
        max_length=200_000,
    )
    label: str = Field(
        min_length=1,
        max_length=64,
    )
    source: str = Field(
        default="analyst",
        min_length=1,
        max_length=128,
    )
    analyst_confirmed: bool = False
