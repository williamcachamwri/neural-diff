"""Unit tests for `neural-diff` CLI subcommands (`inspect`, `validate`, `suites`, `svd`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.cli import main
from tests.fixtures.checkpoint_factory import make_synthetic_hf_checkpoint_pair


@pytest.mark.unit
def test_cli_suites_and_svd_subcommands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    rc_suites = main(["suites"])
    assert rc_suites == 0
    suites_out = json.loads(capsys.readouterr().out)
    assert suites_out["total_cases"] == 24
    assert len(suites_out["categories"]) == 8

    # Create synthetic pair inside workspace data/ scratch test dir or patch SecurityPolicy
    from core import security

    orig_init = security.SecurityPolicy.__init__

    def _patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        orig_init(self, *args, **kwargs)
        if tmp_path.resolve() not in self.allowed_roots:
            self.allowed_roots.append(tmp_path.resolve())

    monkeypatch.setattr(security.SecurityPolicy, "__init__", _patched_init)

    dir_a, dir_b = make_synthetic_hf_checkpoint_pair(tmp_path / "cli_pair")
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
