"""Unit tests for evaluator edge cases, resource guardrails, and micro-benchmark runner."""

from __future__ import annotations

import pytest

from benchmarks.evaluators.base import EvalCase
from benchmarks.evaluators.math_reasoning import extract_final_number
from benchmarks.evaluators.registry import build_evaluator_registry, evaluate_case
from benchmarks.runners.perf_runner import run_chunked_tensor_microbenchmarks
from core.exceptions import ResourceExhaustedError
from core.resource_manager import ResourceManager


@pytest.mark.unit
def test_math_fractions_gsm8k_hash_and_zero_division() -> None:
    val_frac, src_frac = extract_final_number("Therefore the probability is 3/8")
    assert src_frac == "last_number" and val_frac == pytest.approx(0.375)

    val_div0, _ = extract_final_number("Undefined ratio 5/0")
    assert val_div0 is None

    val_hash, src_hash = extract_final_number("Total items = 120.\n#### -45.5")
    assert src_hash == "gsm8k_hash" and val_hash == pytest.approx(-45.5)


@pytest.mark.unit
def test_instruction_and_json_array_edge_cases() -> None:
    inst = EvalCase(
        "inst-edge",
        "instruction_following",
        "",
        "",
        {
            "min_words": 4,
            "must_exclude": ["forbidden"],
            "starts_with": "Result:",
            "ends_with": "done.",
        },
    )
    res_ok = evaluate_case(inst, "Result: computation is now done.")
    res_fail = evaluate_case(inst, "Result: forbidden word done.")
    assert res_ok.passed is True
    assert res_fail.passed is False

    json_arr = EvalCase(
        "j-arr",
        "json_reliability",
        "",
        "",
        {"root_type": "array", "min_items": 3},
    )
    assert evaluate_case(json_arr, "```json\n[10, 20, 30]\n```").passed is True

    with pytest.raises(ValueError):
        evaluate_case(EvalCase("bad", "unknown_category", "", "", {}), "hi")
    assert len(build_evaluator_registry()) == 8


@pytest.mark.unit
def test_resource_manager_headroom_guard_and_cpu_mode() -> None:
    rm_cpu = ResourceManager(preferred_device="cpu")
    assert rm_cpu.resolve_device() == "cpu"
    assert "CPU" in rm_cpu.device_reason

    # Requesting impossible RAM floor (10 TB) must trigger ResourceExhaustedError
    rm_impossible = ResourceManager(preferred_device="cpu", min_free_ram_mb=10_000_000.0)
    with pytest.raises(ResourceExhaustedError):
        rm_impossible.check_memory_headroom()


@pytest.mark.unit
def test_chunked_microbenchmark_runner_produces_real_measurements() -> None:
    res = run_chunked_tensor_microbenchmarks()
    assert len(res["chunked_microbenchmarks"]) == 6
    for entry in res["chunked_microbenchmarks"]:
        assert entry["duration_seconds"] > 0.0
        assert entry["millions_elements_per_sec"] > 0.0
    assert res["safetensors_lazy_streaming"]["tensor_count"] == 8
