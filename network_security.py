"""Validated network security event and flow ingestion contract."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class NetworkSecurityRecord(BaseModel):
    """One timestamped network security event or aggregated flow record.

    This model validates evidence structure, not whether the event is malicious.
    Unknown fields are rejected to prevent silent schema drift.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    record_id: str = Field(min_length=1, max_length=256)
    record_type: Literal["security_event", "flow"]
    timestamp: datetime
    source_host: str = Field(min_length=1, max_length=255)

    # Endpoint values are optional: collectors and providers differ in what
    # they can observe. Missing endpoints are not fabricated.
    source_ip: str | None = Field(default=None, max_length=64)
    source_port: int | None = Field(default=None, ge=0, le=65535)
    destination_ip: str | None = Field(default=None, max_length=64)
    destination_port: int | None = Field(default=None, ge=0, le=65535)
    protocol: str | None = Field(default=None, max_length=32)

    # A flow can carry counts/bytes; an event may carry a provider-specific
    # event name. These are observations, not automatic attack verdicts.
    event_name: str | None = Field(default=None, max_length=256)
    packets: int | None = Field(default=None, ge=0)
    bytes_transferred: int | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(default=None, ge=0)

    # Provenance must say where the record came from and how it was produced.
    provenance_source: str = Field(min_length=1, max_length=256)
    provenance_method: str = Field(min_length=1, max_length=256)
    provider_record_id: str | None = Field(default=None, max_length=256)
    collector_version: str | None = Field(default=None, max_length=128)

    # Preserve unavailable fields explicitly. Example:
    # {"destination_ip": "not_observed", "protocol": "not_provided"}
    unavailable_fields: dict[str, str] = Field(default_factory=dict)
    labels: dict[str, str | int | bool | None] = Field(default_factory=dict)
    attributes: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value.astimezone(timezone.utc)

    @field_validator("protocol")
    @classmethod
    def normalize_protocol(cls, value: str | None) -> str | None:
        return value.lower() if value is not None else None

    @field_validator("attributes")
    @classmethod
    def require_finite_attributes(
        cls,
        value: dict[str, str | int | float | bool | None],
    ) -> dict[str, str | int | float | bool | None]:
        for key, item in value.items():
            if not key.strip():
                raise ValueError("attribute keys must not be blank")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("attributes cannot contain NaN or infinity")
        return value

    @model_validator(mode="after")
    def validate_record_type(self) -> "NetworkSecurityRecord":
        if self.record_type == "security_event" and not self.event_name:
            raise ValueError(
                "security_event records require event_name"
            )

        if self.record_type == "flow":
            if (
                self.packets is None
                and self.bytes_transferred is None
                and self.duration_seconds is None
            ):
                raise ValueError(
                    "flow records require at least one measured flow quantity"
                )

        present = {
            "source_ip": self.source_ip,
            "source_port": self.source_port,
            "destination_ip": self.destination_ip,
            "destination_port": self.destination_port,
            "protocol": self.protocol,
        }

        for field_name, reason in self.unavailable_fields.items():
            if field_name not in present:
                raise ValueError(
                    f"unavailable_fields contains unknown field {field_name!r}"
                )
            if present[field_name] is not None:
                raise ValueError(
                    f"{field_name} has a value and cannot also be unavailable"
                )
            if not reason.strip():
                raise ValueError(
                    f"unavailability reason for {field_name} cannot be blank"
                )

        return self

    def canonical_json(self) -> str:
        """Stable JSON representation for audit and deduplication."""
        payload = self.model_dump(mode="json")
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def fingerprint(self) -> str:
        """Content fingerprint for integrity and duplicate detection."""
        return hashlib.sha256(
            self.canonical_json().encode("utf-8")
        ).hexdigest()
