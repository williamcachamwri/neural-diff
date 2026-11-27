"""Security tests for Neural Evolution Tracking (path traversal, unsafe pickle, tampered SHA-256, schema injection)."""

from __future__ import annotations

import os
import pickle
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from core.evolution.registry import (
    inspect_and_build_checkpoint_record,
    revalidate_checkpoint_health,
)
from core.evolution.types import CheckpointValidationStatus
from core.security import SecurityPolicy
from tests.fixtures.evolution_factory import create_known_evolution_checkpoints


class _EvilEvolutionPickle:
    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (os.system, ("echo exploited",))


@pytest.mark.security
def test_evolution_api_blocks_traversal_pickle_rce_and_schema_injection(tmp_path: Path) -> None:
    policy = SecurityPolicy(allowed_roots=[tmp_path])
    app = create_app(db_path=tmp_path / "sec_evo.db", policy=policy)
    client = TestClient(app)

    pid = client.post("/api/evolution/projects", json={"name": "SecProj"}).json()["project_id"]

    # 1. Path traversal rejected HTTP 403
    r_trav = client.post(
        f"/api/evolution/projects/{pid}/checkpoints",
        json={"source": "../../etc/passwd", "local_files_only": True},
    )
    assert r_trav.status_code == 403
    assert r_trav.json()["error"] == "SECURITY_VALIDATION_ERROR"

    # 2. Malicious PyTorch pickle RCE archive rejected HTTP 403
    evil_pt = tmp_path / "trojan_ckpt.pt"
    with zipfile.ZipFile(evil_pt, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps({"w": _EvilEvolutionPickle()}, protocol=4))

    r_pkl = client.post(
        f"/api/evolution/projects/{pid}/checkpoints",
        json={"source": str(evil_pt), "local_files_only": True},
    )
    assert r_pkl.status_code == 403
    assert r_pkl.json()["error"] == "SECURITY_VALIDATION_ERROR"

    # 3. Unknown extra JSON fields rejected HTTP 422
    r_extra = client.post(
        "/api/evolution/projects",
        json={"name": "Valid", "injected_admin_flag": True},
    )
    assert r_extra.status_code == 422


@pytest.mark.security
def test_tampered_checkpoint_detected_via_sha256_revalidation(tmp_path: Path) -> None:
    ckpts = create_known_evolution_checkpoints(tmp_path)
    policy = SecurityPolicy(allowed_roots=[tmp_path])
    rec = inspect_and_build_checkpoint_record(
        project_id="evo-sec",
        source=ckpts["a"],
        version_label="Base",
        policy=policy,
        local_files_only=True,
    )
    st_ok, err_ok = revalidate_checkpoint_health(rec, policy=policy)
    assert st_ok == CheckpointValidationStatus.VERIFIED
    assert err_ok is None

    # Overwrite model.safetensors with another valid checkpoint's weights -> SHA-256 mismatch detected!
    (ckpts["a"] / "model.safetensors").write_bytes(
        (ckpts["b"] / "model.safetensors").read_bytes()
    )
    st_tampered, err_tampered = revalidate_checkpoint_health(rec, policy=policy)
    assert st_tampered == CheckpointValidationStatus.CORRUPTED_OR_MISSING
    assert err_tampered is not None and "SHA-256 mismatch" in err_tampered
