from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest


ENTRY = os.environ.get("COUNT_UNMIXER", "/app/unmix.py")


def _run(payload: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, ENTRY, str(payload), str(output)],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )


def _load_entry():
    spec = importlib.util.spec_from_file_location("count_unmix_entry", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _poisson(rng: random.Random, rate: float) -> int:
    threshold = math.exp(-rate)
    product = 1.0
    count = 0
    while product > threshold:
        product *= rng.random()
        count += 1
    return count - 1


def _base36(value: int) -> str:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digits: list[str] = []
    current = value
    while current:
        current, remainder = divmod(current, 36)
        digits.append(alphabet[remainder])
    return "".join(reversed(digits or ["0"])).rjust(4, "0")


def _mixture(
    low_rate: float,
    high_rate: float,
    low_weight: float,
    size: int,
    seed: int,
    ceiling: float = 40.0,
) -> dict[str, object]:
    rng = random.Random(seed)
    observations: list[dict[str, object]] = []
    for index in range(size):
        rate = low_rate if rng.random() < low_weight else high_rate
        count = _poisson(rng, rate)
        identifier = (index * 7919 + seed * 131) % (36**4)
        observations.append({"id": f"CNT-{_base36(identifier)}", "count": count})
    return {
        "schema": "count-mixture-v1",
        "rate_ceiling": ceiling,
        "observations": observations,
    }


def test_gate_recovers_two_component_likelihood(tmp_path: Path) -> None:
    # case_id: COUNT-CMD-01 COUNT-OUT-01 COUNT-OUT-02 COUNT-OUT-03 COUNT-OUT-04
    """题面：
    「结果文件恰好带 status、components、log_likelihood 三个键。`status` 为 `"ok"`。components 恰好有两项并按 rate 升序排列。每项恰好带 weight、rate，并保留六位小数。两个权重之和按原始未取整值计算，应为 1。`log_likelihood` 是当前窗口计数在原参数下的对数似然，保留六位小数。」
    """

    payload = _mixture(2.0, 9.0, 0.65, 240, 701)
    source = tmp_path / "windows.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["status"] == "ok"
    assert len(result["components"]) == 2
    assert result["log_likelihood"] < 0


def test_balanced_counts_recover_rates_and_weight(tmp_path: Path) -> None:
    # case_id: COUNT-SCENARIO-01 COUNT-CORE-01
    """题面：
    「低源权重记为 `q`，低源率记为 `r_low`，高源率记为 `r_high`。结果要在 `0.01 <= q <= 0.99`、`0 < r_low < r_high <= rate_ceiling` 内求对数似然最大的参数。两个来源的排列不另算结果；输出时低率分量总排在前面。」
    """

    payload = _mixture(2.0, 9.0, 0.65, 260, 811)
    source = tmp_path / "windows.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    components = result["components"]
    assert completed.returncode == 0
    assert components[0]["rate"] == pytest.approx(2.0, abs=0.35)
    assert components[1]["rate"] == pytest.approx(9.0, abs=0.55)
    assert components[0]["weight"] == pytest.approx(0.65, abs=0.08)


def test_minor_source_is_not_collapsed_into_background(tmp_path: Path) -> None:
    # case_id: COUNT-SCENARIO-02 COUNT-CORE-02
    """题面：
    「字段或样本量不合格时也走同一拒收路径：旧文件不动，退出码 2，标准错误留一行 `REFUSED: ` 开头的说明。」
    """

    payload = _mixture(1.2, 13.0, 0.18, 300, 1907)
    source = tmp_path / "windows.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    components = result["components"]
    assert completed.returncode == 0
    assert components[0]["weight"] == pytest.approx(0.18, abs=0.07)
    assert components[1]["rate"] == pytest.approx(13.0, abs=0.8)


def test_large_window_set_converges_to_global_likelihood(tmp_path: Path) -> None:
    # case_id: COUNT-LATENT-01 COUNT-CORE-03
    """题面：
    「如果公开参数域里凑不出两个分离分量，处理直接终止。不会产生结果文件，退出码为 2，且标准错误只给一行 `REFUSED: ` 前缀的原因。」
    """

    payload = _mixture(3.0, 11.5, 0.45, 300, 3331)
    source = tmp_path / "windows.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    components = result["components"]
    assert completed.returncode == 0
    assert components[0]["rate"] == pytest.approx(3.0, abs=0.5)
    assert components[1]["rate"] == pytest.approx(11.5, abs=0.7)
    assert sum(item["weight"] for item in components) == pytest.approx(1.0, abs=1e-6)


def test_public_upper_window_set_finishes_within_deadline(tmp_path: Path) -> None:
    # case_id: COUNT-PERF-01 COUNT-HINT-01
    """题面：
    「`observations` 至少 20 项、至多 300 项；每项仅含 `id` 与 `count`。探测器编号采用 `CNT-` 前缀，后面固定四格大写字母数字；同批不能出现两次。`count` 采用 0 至 99 的十进制整数，JSON 的 true/false 不在允许类型里。」
    """

    payload = _mixture(2.5, 10.5, 0.55, 300, 6401)
    source = tmp_path / "windows.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, ENTRY, str(source), str(output)],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    elapsed = time.monotonic() - started
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["status"] == "ok"
    assert elapsed < 6


def test_minimum_window_count_is_accepted(tmp_path: Path) -> None:
    # case_id: COUNT-RANGE-01
    """题面：
    「三百个窗口组成的公开上限批次，需要在 6 秒内给出结果。估计逻辑只读当前输入，不可访问测试目录，也不能写死样本参数。」
    """

    payload = _mixture(1.5, 8.0, 0.5, 20, 901)
    source = tmp_path / "windows.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert len(result["components"]) == 2


def test_rate_ceiling_lower_bound_is_accepted(tmp_path: Path) -> None:
    """题面：
    「处理成功时，标准输出为空，标准错误也不留文字。交付文件使用无多余空格的 JSON，键按字典序组织，末尾恰好一个换行。重复运行同一输入应得到相同字节。」
    """

    payload = _mixture(1.2, 4.0, 0.55, 220, 908, ceiling=5.0)
    source = tmp_path / "windows.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["status"] == "ok"
    assert result["components"][1]["rate"] <= payload["rate_ceiling"], "COUNT-RANGE-03"


def test_rate_ceiling_below_five_is_refused(tmp_path: Path) -> None:
    """题面：
    「输入只接受下面列出的字段，名单外的键会让本次处理停止。」
    """

    payload = _mixture(1.2, 4.0, 0.55, 220, 909, ceiling=4.99)
    source = tmp_path / "windows.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists(), "COUNT-RANGE-04"


def test_equal_initial_rates_are_skipped_before_em(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """题面：
    「每个窗口只由一个隐藏来源产生。两个来源的窗口计数服从泊松分布：」
    """

    starts: list[tuple[float, float]] = []

    def fake_em(
        counts: list[int],
        ceiling: float,
        low_rate: float,
        high_rate: float,
        low_weight: float,
    ) -> tuple[float, float, float, float]:
        starts.append((low_rate, high_rate))
        return low_rate, high_rate, low_weight, -1.0

    module = _load_entry()
    monkeypatch.setattr(module, "_em", fake_em)
    result = module.recover([0] + [1] * 19, 5.0)
    assert (1.0, 1.0) not in starts, "COUNT-INIT-01"
    assert result["components"][0]["rate"] < result["components"][1]["rate"]


def test_observations_require_identifier_and_count(tmp_path: Path) -> None:
    # case_id: COUNT-SCHEMA-05
    """题面：
    「辐射监测组换了一台记录器，窗口总数还完整，但每个计数窗口来自哪只探头已经丢了。两只探头的发射率不同，值班员又需要在下一次巡检前给出各自的平均计数率和低源占比，才能判断其中一只是否漂移。」
    """

    payload = _mixture(2.0, 9.0, 0.6, 40, 907)
    payload["observations"][2].pop("count")
    source = tmp_path / "windows.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_too_few_windows_are_refused(tmp_path: Path) -> None:
    # case_id: COUNT-RANGE-02
    """题面：
    「字段或样本量不合格时也走同一拒收路径：旧文件不动，退出码 2，标准错误留一行 `REFUSED: ` 开头的说明。」
    """

    payload = _mixture(1.5, 8.0, 0.5, 19, 902)
    source = tmp_path / "windows.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_boolean_count_is_refused(tmp_path: Path) -> None:
    # case_id: COUNT-SCHEMA-02
    """题面：
    「字段或样本量不合格时也走同一拒收路径：旧文件不动，退出码 2，标准错误留一行 `REFUSED: ` 开头的说明。」
    """

    payload = _mixture(1.5, 8.0, 0.5, 50, 903)
    payload["observations"][3]["count"] = True
    source = tmp_path / "windows.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_unknown_schema_is_refused(tmp_path: Path) -> None:
    # case_id: COUNT-SCHEMA-04
    """题面：
    「字段或样本量不合格时也走同一拒收路径：旧文件不动，退出码 2，标准错误留一行 `REFUSED: ` 开头的说明。」
    """

    payload = _mixture(2.0, 9.0, 0.6, 40, 906)
    payload["schema"] = "count-mixture-v2"
    source = tmp_path / "windows.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_extra_input_key_keeps_existing_result_bytes(tmp_path: Path) -> None:
    # case_id: COUNT-SCHEMA-03 COUNT-FAIL-01 COUNT-FAIL-02 COUNT-FAIL-03 COUNT-FAIL-04
    """题面：
    「字段或样本量不合格时也走同一拒收路径：旧文件不动，退出码 2，标准错误留一行 `REFUSED: ` 开头的说明。」
    """

    payload = _mixture(2.0, 9.0, 0.6, 80, 904)
    payload["analyst"] = "night-shift"
    source = tmp_path / "windows.json"
    output = tmp_path / "result.json"
    old_payload = b'{"prior":true}\n'
    output.write_bytes(old_payload)
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert output.read_bytes() == old_payload


def test_success_writes_silent_canonical_json(tmp_path: Path) -> None:
    # case_id: COUNT-SERIAL-01 COUNT-SERIAL-02 COUNT-OUTPUT-EMPTY-01
    """题面：
    「`observations` 至少 20 项、至多 300 项；每项仅含 `id` 与 `count`。探测器编号采用 `CNT-` 前缀，后面固定四格大写字母数字；同批不能出现两次。`count` 采用 0 至 99 的十进制整数，JSON 的 true/false 不在允许类型里。」
    """

    payload = _mixture(2.0, 9.0, 0.6, 160, 905)
    source = tmp_path / "windows.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    second_output = tmp_path / "result-copy.json"
    second_run = _run(source, second_output)
    raw = output.read_bytes()
    parsed = json.loads(raw)
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert second_run.returncode == 0
    assert second_output.read_bytes() == raw
    assert raw == canonical
