#!/usr/bin/env python3
"""Reference solver for a two-component Poisson mixture."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any


INPUT_KEYS = {"schema", "rate_ceiling", "observations"}
OBSERVATION_KEYS = {"id", "count"}
OBSERVATION_ID = re.compile(r"CNT-[A-Z0-9]{4}\Z")


class Refused(ValueError):
    pass


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Refused("duplicate JSON key")
        result[key] = value
    return result


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refused("expected finite number")
    result = float(value)
    if not math.isfinite(result):
        raise Refused("expected finite number")
    return result


def parse_input(path: Path) -> tuple[float, list[int]]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
        )
    except OSError as exc:
        raise Refused("input cannot be read") from exc
    except json.JSONDecodeError as exc:
        raise Refused("input is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != INPUT_KEYS:
        raise Refused("top-level key set is invalid")
    if payload.get("schema") != "count-mixture-v1":
        raise Refused("unsupported schema")
    ceiling = _number(payload.get("rate_ceiling"))
    if not 5.0 <= ceiling <= 200.0:
        raise Refused("rate_ceiling is outside the public interval")
    raw = payload.get("observations")
    if not isinstance(raw, list) or not 20 <= len(raw) <= 300:
        raise Refused("observations must contain 20 to 300 windows")
    identifiers: set[str] = set()
    counts: list[int] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != OBSERVATION_KEYS:
            raise Refused("observation key set is invalid")
        identifier = item.get("id")
        if not isinstance(identifier, str) or OBSERVATION_ID.fullmatch(identifier) is None:
            raise Refused("observation id is invalid")
        if identifier in identifiers:
            raise Refused("observation id is duplicated")
        identifiers.add(identifier)
        count = item.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 99:
            raise Refused("count is outside the public interval")
        counts.append(count)
    return ceiling, counts


def _log_probability(count: int, rate: float) -> float:
    return count * math.log(rate) - rate - math.lgamma(count + 1.0)


def _log_likelihood(
    counts: list[int],
    low_rate: float,
    high_rate: float,
    low_weight: float,
) -> float:
    terms: list[float] = []
    for count in counts:
        low = math.log(low_weight) + _log_probability(count, low_rate)
        high = math.log(1.0 - low_weight) + _log_probability(count, high_rate)
        maximum = max(low, high)
        terms.append(maximum + math.log(math.exp(low - maximum) + math.exp(high - maximum)))
    return math.fsum(terms)


def _sigmoid(value: float) -> float:
    return 0.5 * (1.0 + math.tanh(value / 2.0))


def _em(
    counts: list[int],
    ceiling: float,
    low_rate: float,
    high_rate: float,
    low_weight: float,
) -> tuple[float, float, float, float] | None:
    low_rate = min(ceiling - 0.01, max(1e-6, low_rate))
    high_rate = min(ceiling, max(low_rate + 0.01, high_rate))
    low_weight = min(0.99, max(0.01, low_weight))
    previous = -math.inf
    for _ in range(500):
        previous_weight = low_weight
        responsibilities: list[float] = []
        for count in counts:
            low = math.log(low_weight) + _log_probability(count, low_rate)
            high = math.log(1.0 - low_weight) + _log_probability(count, high_rate)
            responsibilities.append(_sigmoid(low - high))
        low_mass = math.fsum(responsibilities)
        high_mass = len(counts) - low_mass
        if low_mass < 1e-8 or high_mass < 1e-8:
            return None
        low_weight = min(0.99, max(0.01, low_mass / len(counts)))
        new_low = math.fsum(
            responsibility * count
            for responsibility, count in zip(responsibilities, counts)
        ) / low_mass
        new_high = math.fsum(
            (1.0 - responsibility) * count
            for responsibility, count in zip(responsibilities, counts)
        ) / high_mass
        new_low = min(ceiling - 0.01, max(1e-6, new_low))
        new_high = min(ceiling, max(new_low + 0.01, new_high))
        if new_low > new_high:
            new_low, new_high = new_high, new_low
            low_weight = 1.0 - low_weight
        likelihood = _log_likelihood(counts, new_low, new_high, low_weight)
        movement = abs(new_low - low_rate) + abs(new_high - high_rate) + abs(
            low_weight - previous_weight
        )
        low_rate, high_rate = new_low, new_high
        if likelihood - previous < 1e-10 and movement < 1e-9:
            return low_rate, high_rate, low_weight, likelihood
        previous = likelihood
    return None


def recover(counts: list[int], ceiling: float) -> dict[str, Any]:
    # 多个初值必须在同一连续参数区间内收敛到全局似然最优。
    ordered = sorted(counts)
    size = len(ordered)
    candidates = [
        (1.0, max(ordered[int(0.75 * (size - 1))], 2.0), 0.5),
        (max(ordered[int(0.1 * (size - 1))], 1.0), ordered[int(0.9 * (size - 1))], 0.35),
        (max(ordered[int(0.25 * (size - 1))], 1.0), ordered[int(0.75 * (size - 1))], 0.6),
        (2.0, max(6.0, ceiling * 0.25), 0.25),
        (ceiling * 0.05, ceiling * 0.35, 0.75),
    ]
    best: tuple[float, float, float, float] | None = None
    for low_rate, high_rate, low_weight in candidates:
        if low_rate >= high_rate:
            continue
        fitted = _em(counts, ceiling, low_rate, high_rate, low_weight)
        if fitted is None:
            continue
        if best is None or fitted[3] > best[3]:
            best = fitted
    if best is None:
        raise Refused("two-component likelihood did not converge")
    low_rate, high_rate, low_weight, likelihood = best
    if high_rate - low_rate < 0.2:
        raise Refused("the sample does not support two separated components")
    return {
        "components": [
            {"rate": round(low_rate, 6), "weight": round(low_weight, 6)},
            {"rate": round(high_rate, 6), "weight": round(1.0 - low_weight, 6)},
        ],
        "log_likelihood": round(likelihood, 6),
        "status": "ok",
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("REFUSED: expected input and output paths", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    try:
        ceiling, counts = parse_input(source)
        result = recover(counts, ceiling)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    # 混合结果按排序键编码，固定多初值搜索的可重放字节。
    target.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
