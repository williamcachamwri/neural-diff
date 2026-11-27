"""Unit tests for `neural-diff` and Charmbracelet `ndiff` CLI subcommands."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from core.cli import find_charm_binary, main
from tests.fixtures.checkpoint_factory import make_synthetic_hf_checkpoint_pair


@pytest.mark.unit
def test_cli_suites_status_dag_and_svd_subcommands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    rc_suites = main(["suites"])
    assert rc_suites == 0
    suites_out = json.loads(capsys.readouterr().out)
    assert suites_out["total_cases"] == 24
    assert len(suites_out["categories"]) == 8

    rc_status = main(["status", "--json"])
    assert rc_status == 0
    status_out = json.loads(capsys.readouterr().out)
    assert status_out["suite_id"] == "neural-diff-core-v1"
    assert status_out["total_cases"] == 24

    rc_pretty = main(["suites", "--pretty", "--no-color"])
    assert rc_pretty == 0
    pretty_text = capsys.readouterr().out
    assert "NEURAL DIFF" in pretty_text
    assert "neural-diff-core-v1" in pretty_text

    rc_dag = main(["dag", "--pretty", "--no-color"])
    assert rc_dag == 0
    dag_text = capsys.readouterr().out
    assert "Evolution DAG" in dag_text

    from core import security

    orig_init = security.SecurityPolicy.__init__

    def _patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        orig_init(self, *args, **kwargs)
        if tmp_path.resolve() not in self.allowed_roots:
            self.allowed_roots.append(tmp_path.resolve())

    monkeypatch.setattr(security.SecurityPolicy, "__init__", _patched_init)

    dir_a, dir_b = make_synthetic_hf_checkpoint_pair(tmp_path / "cli_pair")
    rc_qd = main(["quickdiff", "--model-a", str(dir_a), "--model-b", str(dir_b), "--local-only", "--json"])
    assert rc_qd == 0
    qd_out = json.loads(capsys.readouterr().out)
    assert qd_out["compatible"] is True

    rc_svd = main(
        [
            "svd",
            "--model-a",
            str(dir_a),
            "--model-b",
            str(dir_b),
            "--tensor",
            "model.layers.0.self_attn.q_proj.weight",
            "--local-only",
        ]
    )
    assert rc_svd == 0
    svd_out = json.loads(capsys.readouterr().out)
    assert svd_out["tensor"] == "model.layers.0.self_attn.q_proj.weight"
    assert svd_out["shannon_effective_rank"] > 0.0


@pytest.mark.unit
def test_charm_ndiff_binary_when_compiled(tmp_path: Path) -> None:
    charm_bin = find_charm_binary()
    if charm_bin is None:
        pytest.skip("Charmbracelet Go binary `bin/ndiff` not built in this environment")

    dir_a, dir_b = make_synthetic_hf_checkpoint_pair(tmp_path / "charm_pair")

    # 1. Deterministic JSON mode parity (`ndiff suites --json`)
    proc_suites = subprocess.run(
        [str(charm_bin), "suites", "--json"], capture_output=True, text=True, check=True
    )
    parsed = json.loads(proc_suites.stdout)
    assert parsed["total_cases"] == 24
    assert len(parsed["categories"]) == 8

    # 2. Fast native `.safetensors` header quickdiff (`ndiff quickdiff`)
    proc_qd = subprocess.run(
        [str(charm_bin), "quickdiff", "--model-a", str(dir_a), "--model-b", str(dir_b), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    qd = json.loads(proc_qd.stdout)
    assert qd["status"] == "COMPATIBLE"
    assert qd["aligned_tensors"] > 0

    # 3. Lip Gloss v2 rounded table & DAG tree rendering (`--pretty --no-color`)
    proc_dag = subprocess.run(
        [str(charm_bin), "--pretty", "--no-color", "dag"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "DAG Lineage" in proc_dag.stdout
    assert "╭" in proc_dag.stdout
    assert "╰──" in proc_dag.stdout
