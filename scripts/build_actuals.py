"""Extract actuals.json from a resolved events file for local Brier evaluation.

Usage:
    python scripts/build_actuals.py events_resolved.json actuals.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def build_actuals(events_path: Path, out_path: Path) -> int:
    events = json.loads(events_path.read_text(encoding="utf-8"))
    actuals: dict[str, str] = {}

    for event in events:
        resolved = event.get("resolved_outcome")
        if not resolved:
            continue
        values = resolved.get("value")
        if not values:
            continue
        # `value` is always a list of strings; for single-winner events take the
        # first one. Multi-winner rows just take the first too — Brier still
        # rewards mass on it.
        actuals[event["market_ticker"]] = values[0]

    out_path.write_text(json.dumps(actuals, indent=2), encoding="utf-8")
    print(f"Wrote {len(actuals)} actuals to {out_path}")
    return len(actuals)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/build_actuals.py <events.json> <actuals.json>")
        sys.exit(1)
    build_actuals(Path(sys.argv[1]), Path(sys.argv[2]))
