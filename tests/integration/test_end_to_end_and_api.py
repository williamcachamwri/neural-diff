"""Integration tests across sharded Safetensors, Safe PyTorch, FastAPI, and exports."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from api.app import create_app
from core.loaders.model_loader import ModelLoader
from core.security import SecurityPolicy
from core.tensor_engine.comparator import TensorDiffEngine
from tests.fixtures.checkpoint_factory import (
    make_sharded_safetensors_checkpoint,
    make_synthetic_hf_checkpoint_pair,
)


@pytest.mark.integration
def test_sharded_safetensors_vs_safe_pytorch_checkpoint_equivalence(tmp_path: Path) -> None:
    policy = SecurityPolicy(allowed_roots=[tmp_path])
    g = torch.Generator().manual_seed(2026)
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn((32, 32), generator=g),
        "model.layers.0.mlp.up_proj.weight": torch.randn((64, 32), generator=g),
        "model.norm.weight": torch.ones((32,)),
    }

    sharded_dir = tmp_path / "sharded_model"
    index_path = make_sharded_safetensors_checkpoint(sharded_dir, tensors)

    pt_file = tmp_path / "weights_only_model.pt"
    torch.save({"state_dict": tensors}, str(pt_file))

    loader = ModelLoader(policy=policy)
    with loader.load(index_path) as h_sharded, loader.load(pt_file) as h_pt:
        assert h_sharded.metadata.shard_count == 2
        assert h_pt.metadata.format.value == "pytorch_weights_only"
        diff = TensorDiffEngine(chunk_elements=256).compare(h_sharded, h_pt)
        assert diff.identical_tensors == 3
        assert diff.global_changed_pct == 0.0
        assert diff.global_mean_cosine == pytest.approx(1.0)


@pytest.mark.integration
def test_full_fastapi_workflow_comparisons_filtering_exports_and_settings(tmp_path: Path) -> None:
    policy = SecurityPolicy(allowed_roots=[tmp_path])
    dir_a, dir_b = make_synthetic_hf_checkpoint_pair(tmp_path)
    app = create_app(db_path=tmp_path / "integration.db", policy=policy)
    client = TestClient(app)

    # 1. Health & benchmarks discovery
    h = client.get("/api/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"

    suites = client.get("/api/benchmarks/suites")
    assert suites.status_code == 200
    assert suites.json()["total_cases"] == 24

    # 2. Inspect & Validate compatibility
    insp = client.post("/api/models/inspect", json={"source": str(dir_a), "local_files_only": True})
    assert insp.status_code == 200
    assert insp.json()["metadata"]["tensor_count"] == 21

    val = client.post(
        "/api/models/validate",
        json={"model_a": str(dir_a), "model_b": str(dir_b), "local_files_only": True},
    )
    assert val.status_code == 200
    assert val.json()["compatibility"]["compatible_for_direct_tensor_diff"] is True

    # 3. Synchronous full weight/structural/neuron/regression comparison
    cmp_resp = client.post(
        "/api/comparisons/run",
        json={
            "model_a": str(dir_a),
            "model_b": str(dir_b),
            "modules": ["structural", "tensor", "neuron", "regression"],
            "experiment_type": "SYNTHETIC FIXTURE EXPERIMENT",
            "preferred_device": "cpu",
            "local_files_only": True,
            "async_job": False,
        },
    )
    assert cmp_resp.status_code == 200
    comp = cmp_resp.json()
    comp_id = comp["comparison_id"]
    assert comp["status"] == "completed"
    assert comp["verdict"] == "PASSED"

    # 4. Tensor explorer query filtering & sorting
    t_query = client.get(
        f"/api/comparisons/{comp_id}/tensors",
        params={"role": "attn_q", "sort_by": "relative_l2"},
    )
    assert t_query.status_code == 200
    assert t_query.json()["total"] == 2

    # 5. Export JSON, Markdown, and HTML reports
    for fmt, needle in (
        ("json", '"comparison_id"'),
        ("markdown", "# NEURAL DIFF Report"),
        ("html", "<!DOCTYPE html>"),
    ):
        exp_r = client.get(f"/api/comparisons/{comp_id}/export/{fmt}")
        assert exp_r.status_code == 200
        assert needle in exp_r.text

    # 6. Async job execution polling
    async_resp = client.post(
        "/api/comparisons/run",
        json={
            "model_a": str(dir_a),
            "model_b": str(dir_b),
            "modules": ["structural", "tensor"],
            "preferred_device": "cpu",
            "local_files_only": True,
            "async_job": True,
        },
    )
    assert async_resp.status_code == 200
    async_id = async_resp.json()["comparison_id"]
    for _ in range(40):
        poll = client.get(f"/api/comparisons/{async_id}").json()
        if poll["status"] == "completed":
            break
        time.sleep(0.05)
    assert poll["status"] == "completed"

    # 7. Settings GET & PUT
    put_s = client.put(
        "/api/settings",
        json={
            "preferred_device": "cpu",
            "atol": 1e-5,
            "rtol": 1e-4,
            "zero_threshold": 0.0,
            "chunk_elements": 65536,
            "allow_pytorch_weights_only": False,
            "max_new_tokens": 64,
        },
    )
    assert put_s.status_code == 200
    assert put_s.json()["allow_pytorch_weights_only"] is False
