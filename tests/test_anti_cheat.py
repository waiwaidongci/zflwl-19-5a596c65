from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


ENTRY = os.environ.get("COUNT_UNMIXER", "/app/unmix.py")
VARIANTS = (
    (1.6, 8.5, 0.72, 270, 4171, 40.0),
    (3.4, 12.0, 0.38, 300, 4172, 40.0),
    (2.2, 6.4, 0.58, 240, 4173, 16.0),
)


def _load_output_helpers():
    path = Path(__file__).with_name("test_outputs.py")
    spec = importlib.util.spec_from_file_location("yel019_output_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = _load_output_helpers()


def _signature(result: dict[str, object]) -> str:
    raw = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _run_variants(tmp_path: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for index, (low, high, weight, size, seed, ceiling) in enumerate(VARIANTS):
        payload = HELPERS._mixture(low, high, weight, size, seed, ceiling)
        source = tmp_path / f"anti-input-{index}.json"
        output = tmp_path / f"anti-output-{index}.json"
        source.write_text(json.dumps(payload), encoding="utf-8")
        completed = HELPERS._run(source, output)
        actual = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
        components = actual["components"] if actual else None
        results.append(
            {
                "returncode": completed.returncode,
                "low_rate": components[0]["rate"] if components else None,
                "high_rate": components[1]["rate"] if components else None,
                "low_weight": components[0]["weight"] if components else None,
                "weight_sum": sum(item["weight"] for item in components) if components else None,
                "signature": _signature(actual) if actual else None,
            }
        )
    return results


def test_anti_cheat_varied_mixtures_cannot_reuse_one_fit(tmp_path: Path) -> None:
    """题面：「结果要在 `0.01 <= q <= 0.99`、`0 < r_low < r_high <= rate_ceiling` 内求对数似然最大的参数。」

    三组未公开的泊松混合样本具有不同率、权重和率上限。固定参数表无法同时落在
    三组独立样本的最大似然邻域，也无法保持每组的权重和与率排序。
    """

    variants = _run_variants(tmp_path)
    assert [item["returncode"] for item in variants] == [0, 0, 0]
    assert [item["low_rate"] for item in variants] == pytest.approx([1.6, 3.4, 2.2], abs=0.65)
    assert [item["high_rate"] for item in variants] == pytest.approx([8.5, 12.0, 6.4], abs=0.9)
    assert [item["low_weight"] for item in variants] == pytest.approx([0.72, 0.38, 0.58], abs=0.16)
    assert [item["weight_sum"] for item in variants] == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)
    assert len({item["signature"] for item in variants}) == len(VARIANTS)
