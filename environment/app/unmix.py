#!/usr/bin/env python3
"""Starter two-source count mixture entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("REFUSED: expected input and output paths", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("REFUSED: unreadable input", file=sys.stderr)
        return 2

    observations = payload.get("observations", []) if isinstance(payload, dict) else []
    counts = sorted(
        item.get("count", 0)
        for item in observations
        if isinstance(item, dict)
    )
    split = max(1, len(counts) // 2)
    low = counts[:split]
    high = counts[split:]
    result = {
        "components": [
            {"rate": sum(low) / len(low), "weight": 0.5},
            {"rate": sum(high) / len(high), "weight": 0.5},
        ],
        "log_likelihood": 0.0,
        "status": "ok",
    }
    target.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
