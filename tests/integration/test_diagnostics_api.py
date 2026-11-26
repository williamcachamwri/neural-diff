"""Integration tests for on-demand `/api/diagnostics/*` endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from core.security import SecurityPolicy
from tests.fixtures.checkpoint_factory import make_synthetic_hf_checkpoint_pair


@pytest.mark.integration
def test_diagnostics_tensor_probe_weight_tying_and_trends(tmp_path: Path) -> None:
    dir_a, dir_b = make_synthetic_hf_checkpoint_pair(tmp_path / "pair")
    policy = SecurityPolicy(allowed_roots=[tmp_path])
    app = create_app(db_path=tmp_path / "diag.db", policy=policy)
    client = TestClient(app)

    # 1. Probe a 2D attention projection matrix for SVD spectrum, sparsity, histogram & top deltas
    resp = client.post(
        "/api/diagnostics/tensor-probe",
        json={
            "model_a": str(dir_a),
            "model_b": str(dir_b),
            "tensor_name": "model.layers.0.self_attn.q_proj.weight",
            "local_files_only": True,
            "histogram_bins": 16,
            "top_k": 5,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["svd_spectrum"] is not None
    assert body["svd_spectrum"]["shannon_effective_rank"] > 0.0
    assert body["sparsity_b"] is not None
    assert body["histogram"]["bins"] == 16
    assert len(body["top_elements"]) == 5

    # 2. Verify weight-tying inspection endpoint
    wt = client.post(
        "/api/diagnostics/weight-tying",
        json={"model_a": str(dir_a), "model_b": str(dir_b), "local_files_only": True},
    )
    assert wt.status_code == 200
    assert "tying_changed" in wt.json()

    # 3. Verify history-trends gracefully reports <2 runs on fresh DB
    tr = client.get("/api/diagnostics/history-trends")
    assert tr.status_code == 200
    assert tr.json()["available_behavioral_runs"] == 0
    assert tr.json()["comparison"] is None
