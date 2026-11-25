"""Research EEG ingest and deterministic signal summaries. Not a diagnostic device."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Sequence
import math

@dataclass(frozen=True)
class EEGFrame:
    host_id: str
    stream_id: str
    timestamp_ns: int
    sample_rate_hz: float
    channel_names: tuple[str, ...]
    samples_uv: tuple[tuple[float, ...], ...]  # channel-major window
    units: str = "uV"
    source: str = "unknown"

    def validate(self) -> None:
        if not self.host_id or not self.stream_id: raise ValueError("host_id and stream_id required")
        if not math.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0: raise ValueError("sample_rate_hz must be positive")
        if self.units != "uV": raise ValueError("samples must be explicitly converted to microvolts")
        if len(self.channel_names) != len(self.samples_uv) or not self.channel_names: raise ValueError("channel metadata mismatch")
        n = len(self.samples_uv[0])
        if n < 2 or any(len(ch) != n for ch in self.samples_uv): raise ValueError("channels must have equal window lengths >=2")
        if any(not math.isfinite(x) for ch in self.samples_uv for x in ch): raise ValueError("non-finite EEG sample")

    def observation(self) -> dict:
        self.validate()
        return {"schema":"asterion.eeg.observation.v1", **asdict(self), "channel_names":list(self.channel_names), "samples_uv":[list(c) for c in self.samples_uv], "quality":{"status":"unassessed","flags":[]}}

def summarize(frame: EEGFrame) -> dict:
    """Calculate auditable window features; does not infer diagnosis or state."""
    frame.validate(); result=[]
    for name, xs in zip(frame.channel_names, frame.samples_uv):
        mean=sum(xs)/len(xs); centered=[x-mean for x in xs]
        rms=math.sqrt(sum(x*x for x in centered)/len(xs))
        result.append({"channel":name,"mean_uv":mean,"rms_uv":rms,"peak_to_peak_uv":max(xs)-min(xs),"samples":len(xs)})
    return {"schema":"asterion.eeg.features.v1","host_id":frame.host_id,"stream_id":frame.stream_id,"timestamp_ns":frame.timestamp_ns,"sample_rate_hz":frame.sample_rate_hz,"source":frame.source,"features":result,"interpretation":{"status":"not_performed","reason":"features are descriptive; no validated clinical model configured"}}
