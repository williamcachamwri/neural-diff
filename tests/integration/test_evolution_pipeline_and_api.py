"""Integration, API, Representation Compatibility, Caching, Cancellation, Recovery & Export tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from core.cli import main as cli_main
from core.evolution.exporter import EvolutionReportExporter
from core.evolution.orchestrator import EvolutionAnalysisConfig, EvolutionOrchestrator
from core.evolution.registry import inspect_and_build_checkpoint_record
from core.evolution.store import EvolutionStore
from core.exceptions import OperationCancelledError
from core.resource_manager import CancellationToken
from core.security import SecurityPolicy
from tests.fixtures.evolution_factory import create_known_evolution_checkpoints


@pytest.mark.integration
def test_orchestrator_representation_compat_caching_cancellation_and_exports(tmp_path: Path) -> None:
    ckpts = create_known_evolution_checkpoints(tmp_path)
    policy = SecurityPolicy(allowed_roots=[tmp_path])
    store = EvolutionStore(db_path=tmp_path / "orch.db")
    proj = store.create_project(name="Representation & Export Integration")

    c_base = store.add_checkpoint(
        inspect_and_build_checkpoint_record(
            project_id=proj.project_id,
            source=ckpts["a"],
            version_label="Base",
            transformation_type="Base Model",
            policy=policy,
            local_files_only=True,
        )
    )
    c_sft = store.add_checkpoint(
        inspect_and_build_checkpoint_record(
            project_id=proj.project_id,
            source=ckpts["b"],
            version_label="SFT",
            parent_checkpoint_ids=[c_base.checkpoint_id],
            transformation_type="SFT",
            policy=policy,
            local_files_only=True,
        )
    )
    # Add an incompatible architecture branch to verify explicit compatibility warnings (never silent failure!)
    c_incompat = store.add_checkpoint(
        inspect_and_build_checkpoint_record(
            project_id=proj.project_id,
            source=ckpts["incompat"],
            version_label="Incompat-24d",
            parent_checkpoint_ids=[c_base.checkpoint_id],
            transformation_type="Distillation",
            policy=policy,
            local_files_only=True,
        )
    )

    orch = EvolutionOrchestrator(store=store, policy=policy)
    cfg = EvolutionAnalysisConfig(
        include_tensor_evolution=True,
        include_layer_evolution=True,
        include_behavioral_evolution=False,  # synthetic dirs have no BPE tokenizer.json
        include_representation_evolution=True,
        enable_causal_patching=True,
        chunk_elements=4096,  # exercises chunked streaming path
        local_files_only=True,
    )

    # 1. Initial run computes CKA matrix, Logit Lens, causal patching, and surfaces incompat warnings
    res1 = orch.run_analysis(proj.project_id, cfg)
    assert res1["status"] == "completed"
    assert res1["cache_hit"] is False

    rep = res1["representation_evolution"]
    assert rep is not None
    rows_by_id = {r["checkpoint_id"]: r for r in rep["evolution_matrix"]["rows"]}
    assert rows_by_id[c_base.checkpoint_id]["layer_cka"] == [1.0, 1.0, 1.0]
    assert rows_by_id[c_sft.checkpoint_id]["compatible_with_base"] is True
    # Incompatible vocab_size (64 vs 80) produces different fallback token IDs -> explicit warning & blocked!
    assert rows_by_id[c_incompat.checkpoint_id]["compatible_with_base"] is False
    assert len(rows_by_id[c_incompat.checkpoint_id]["compatibility_warnings"]) >= 1

    # Verify tensor shape mismatches explicitly tracked on Base -> Incompat-24d
    incompat_tensor_tr = next(
        t
        for t in res1["tensor_transitions"]
        if t["scope"] == "consecutive" and t["target_checkpoint_id"] == c_incompat.checkpoint_id
    )
    assert incompat_tensor_tr["compatible_for_direct_tensor_diff"] is False
    assert len(incompat_tensor_tr["shape_mismatched_tensors"]) > 0

    # 2. Content-addressed cache hit when re-running identical inputs & config
    res2 = orch.run_analysis(proj.project_id, cfg)
    assert res2["cache_hit"] is True
    assert res2["run_id"] == res1["run_id"]

    # Changing seed invalidates content cache key -> computes fresh run!
    cfg_changed = EvolutionAnalysisConfig(
        include_behavioral_evolution=False,
        include_representation_evolution=True,
        seed=999,
        local_files_only=True,
    )
    res3 = orch.run_analysis(proj.project_id, cfg_changed)
    assert res3["cache_hit"] is False
    assert res3["run_id"] != res1["run_id"]

    # 3. Export verification across JSON, CSV, HTML, and PDF
    exporter = EvolutionReportExporter()
    out_dir = tmp_path / "exports"
    p_json = exporter.export_file(res1, "json", out_dir)
    p_csv = exporter.export_file(res1, "csv", out_dir)
    p_md = exporter.export_file(res1, "markdown", out_dir)
    p_html = exporter.export_file(res1, "html", out_dir)
    p_pdf = exporter.export_file(res1, "pdf", out_dir)

    parsed_json = json.loads(p_json.read_text(encoding="utf-8"))
    assert parsed_json["run_id"] == res1["run_id"]
    assert "[CHECKPOINT_REGISTRY]" in p_csv.read_text(encoding="utf-8")
    md_txt = p_md.read_text(encoding="utf-8")
    assert "```mermaid" in md_txt and "flowchart LR" in md_txt
    html_txt = p_html.read_text(encoding="utf-8")
    assert "<svg" in html_txt and "NEURAL DIFF" in html_txt
    pdf_bytes = p_pdf.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-1.4")
    assert b"/Type /Page" in pdf_bytes
    assert b"xref" in pdf_bytes
    assert pdf_bytes.strip().endswith(b"%%EOF")

    # 4. CancellationToken interruption test
    tok = CancellationToken()
    tok.cancel("Simulated user abort")
    with pytest.raises(OperationCancelledError):
        orch.run_analysis(
            proj.project_id,
            EvolutionAnalysisConfig(force_rerun=True, local_files_only=True, include_behavioral_evolution=False),
            cancellation_token=tok,
        )

    # 5. Missing/deleted checkpoint graceful partial recovery test
    temp_copy = tmp_path / "deletable_ckpt"
    shutil.copytree(ckpts["d"], temp_copy)
    c_del = store.add_checkpoint(
        inspect_and_build_checkpoint_record(
            project_id=proj.project_id,
            source=temp_copy,
            version_label="Deleted-Stage",
            parent_checkpoint_ids=[c_sft.checkpoint_id],
            policy=policy,
            local_files_only=True,
        )
    )
    shutil.rmtree(temp_copy)
    res_partial = orch.run_analysis(
        proj.project_id,
        EvolutionAnalysisConfig(
            include_behavioral_evolution=False,
            force_rerun=True,
            local_files_only=True,
        ),
    )
    assert res_partial["status"] == "completed"
    assert any(c_del.checkpoint_id in w or "Deleted-Stage" in w for w in res_partial["warnings"])


@pytest.mark.integration
def test_evolution_rest_api_and_cli_end_to_end(tmp_path: Path) -> None:
    ckpts = create_known_evolution_checkpoints(tmp_path)
    policy = SecurityPolicy(allowed_roots=[tmp_path])
    db_path = tmp_path / "api_evo.db"
    app = create_app(db_path=db_path, policy=policy)
    client = TestClient(app)

    # 1. Create & list projects with pagination
    r_create = client.post(
        "/api/evolution/projects",
        json={"name": "API Lifecycle Project", "description": "REST E2E", "tags": ["test"]},
    )
    assert r_create.status_code == 200
    pid = r_create.json()["project_id"]

    r_list = client.get("/api/evolution/projects?limit=10&offset=0")
    assert r_list.status_code == 200
    assert r_list.json()["total"] == 1

    # 2. Register Base, SFT-A, SFT-B, and Merged
    c_a = client.post(
        f"/api/evolution/projects/{pid}/checkpoints",
        json={"source": str(ckpts["a"]), "version_label": "v0-base", "transformation_type": "Base Model", "local_files_only": True},
    ).json()
    c_b = client.post(
        f"/api/evolution/projects/{pid}/checkpoints",
        json={
            "source": str(ckpts["b"]),
            "version_label": "v1-sft-a",
            "parent_checkpoint_ids": [c_a["checkpoint_id"]],
            "transformation_type": "SFT",
            "local_files_only": True,
        },
    ).json()
    c_sb = client.post(
        f"/api/evolution/projects/{pid}/checkpoints",
        json={
            "source": str(ckpts["sft_b"]),
            "version_label": "v1-sft-b",
            "parent_checkpoint_ids": [c_a["checkpoint_id"]],
            "transformation_type": "SFT",
            "local_files_only": True,
        },
    ).json()

    # Cycle via REST relationship endpoint returns HTTP 422 CYCLIC_DEPENDENCY_ERROR
    r_cyc = client.post(
        f"/api/evolution/projects/{pid}/relationships",
        json={
            "parent_checkpoint_id": c_b["checkpoint_id"],
            "child_checkpoint_id": c_a["checkpoint_id"],
        },
    )
    assert r_cyc.status_code == 422
    assert r_cyc.json()["error"] == "CYCLIC_DEPENDENCY_ERROR"

    # Patch checkpoint label
    r_patch = client.patch(
        f"/api/evolution/projects/{pid}/checkpoints/{c_b['checkpoint_id']}",
        json={"version_label": "v1-sft-alpha", "experiment_notes": "lr=2e-5"},
    )
    assert r_patch.status_code == 200
    assert r_patch.json()["version_label"] == "v1-sft-alpha"

    # Compare branch-to-branch versions
    r_cmp = client.post(
        f"/api/evolution/projects/{pid}/compare",
        json={
            "checkpoint_a_id": c_b["checkpoint_id"],
            "checkpoint_b_id": c_sb["checkpoint_id"],
            "local_files_only": True,
        },
    )
    assert r_cmp.status_code == 200
    assert r_cmp.json()["comparison"]["scope"] == "branch_to_branch"
    assert r_cmp.json()["lowest_common_ancestor_id"] == c_a["checkpoint_id"]

    # Run synchronous evolution analysis
    r_run = client.post(
        f"/api/evolution/projects/{pid}/analyze",
        json={
            "include_behavioral_evolution": False,
            "include_representation_evolution": True,
            "local_files_only": True,
        },
    )
    assert r_run.status_code == 200
    run_id = r_run.json()["run_id"]

    # Verify all 4 HTTP export formats
    for fmt, expected_mime in (
        ("json", "application/json"),
        ("csv", "text/csv"),
        ("html", "text/html"),
        ("pdf", "application/pdf"),
    ):
        resp_exp = client.get(f"/api/evolution/runs/{run_id}/export/{fmt}")
        assert resp_exp.status_code == 200
        assert expected_mime in resp_exp.headers["content-type"]
        if fmt == "pdf":
            assert resp_exp.content.startswith(b"%PDF-1.4")

    # Overview & runs history endpoints
    assert client.get("/api/evolution/overview").status_code == 200
    assert client.get(f"/api/evolution/projects/{pid}/graph").status_code == 200
    assert client.get(f"/api/evolution/projects/{pid}/runs").json()["total"] == 1
    assert client.get(f"/api/evolution/projects/{pid}/regressions").status_code == 200

    # CLI smoke verification
    cli_db = tmp_path / "cli_evo.db"
    rc = cli_main(
        [
            "evolution",
            "create-project",
            "--name",
            "CLI Project",
            "--db-path",
            str(cli_db),
        ]
    )
    assert rc == 0
