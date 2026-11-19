
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    agent_id: str
    host: str

    category: str
    severity: Literal[
        "info",
        "low",
        "medium",
        "high",
        "critical"
    ]

    description: str

    source: Literal[
        "rule",
        "statistical",
        "behavioral",
        "network",
        "machine_learning"
    ]

    evidence: dict[str, Any] = Field(
        default_factory=dict
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0
    )

    hypothesis_ids: list[str] = Field(
        default_factory=list
    )

    interpretation: str | None = None
