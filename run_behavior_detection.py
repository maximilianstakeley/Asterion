
"""Run Asterion's behavior detector against a JSON evidence file.

Input format:
{
  "samples": [],
  "baseline_samples": [],
  "security_events": [],
  "thresholds": {"cpu_percent": 90.0},
  "change_thresholds": {"cpu_percent": 25.0},
  "known_indicators": {"dst_ip": ["203.0.113.10"]}
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from Asterion.behavior_detection import detect_behavior


def load_input(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON input: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object")

    for key in ("samples", "baseline_samples", "security_events"):
        if key in data and not isinstance(data[key], list):
            raise ValueError(f"'{key}' must be a JSON array")

    indicators = data.get("known_indicators", {})
    if not isinstance(indicators, dict):
        raise ValueError("'known_indicators' must be a JSON object")

    normalized_indicators = {}
    for field, values in indicators.items():
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(
                f"Indicator values for '{field}' must be an array of strings"
            )
        normalized_indicators[field] = set(values)

    return {
        "samples": data.get("samples", []),
        "baseline_samples": data.get("baseline_samples", []),
        "security_events": data.get("security_events", []),
        "thresholds": data.get("thresholds"),
        "change_thresholds": data.get("change_thresholds"),
        "known_indicators": normalized_indicators,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze telemetry and security events for suspicious behavior."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a JSON file containing telemetry and optional security events",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON findings report; defaults to stdout",
    )
    args = parser.parse_args()

    try:
        evidence = load_input(args.input)
        result = detect_behavior(**evidence)
        rendered = json.dumps(result, indent=2, allow_nan=False)

        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
            print(
                f"Wrote {result['finding_count']} findings to {args.output}",
                file=sys.stderr,
            )
        else:
            print(rendered)

        return 0
    except (ValueError, TypeError, KeyError) as exc:
        print(f"Detection input error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"File error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
