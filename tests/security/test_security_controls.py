"""Security test suite covering path traversal, malformed headers, unsafe pickle, and API hardening."""

from __future__ import annotations

import json
import os
import pickle
import struct
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from core.exceptions import MalformedCheckpointError, SecurityValidationError
from core.loaders.model_loader import ModelLoader
from core.security import (
    SecurityPolicy,
    inspect_pytorch_archive_safety,
    validate_safe_path,
    validate_safe_shard_filename,
    validate_safetensors_header,
)


@pytest.mark.security
def test_path_traversal_null_bytes_and_unc_rejected(tmp_path: Path) -> None:
    policy = SecurityPolicy(allowed_roots=[tmp_path])

    with pytest.raises(SecurityValidationError):
        validate_safe_path("../../Windows/System32/drivers/etc/hosts", policy=policy)

    with pytest.raises(SecurityValidationError):
        validate_safe_path(f"{tmp_path}/valid\x00evil.safetensors", policy=policy)

    with pytest.raises(SecurityValidationError):
        validate_safe_path("\\\\?\\C:\\Windows\\win.ini", policy=policy)

    with pytest.raises(SecurityValidationError):
        validate_safe_path("file:///etc/passwd", policy=policy)

    outside = Path("C:/Windows/System32")
    with pytest.raises(SecurityValidationError):
        validate_safe_path(outside, policy=policy, must_exist=False)


@pytest.mark.security
def test_shard_index_traversal_rejected(tmp_path: Path) -> None:
    policy = SecurityPolicy(allowed_roots=[tmp_path])
    sub = tmp_path / "model_dir"
    sub.mkdir()
    secret = tmp_path / "secret.safetensors"
    secret.write_bytes(b"12345678")

    with pytest.raises(SecurityValidationError):
        validate_safe_shard_filename("../secret.safetensors", sub, policy=policy)


@pytest.mark.security
def test_malformed_and_adversarial_safetensors_headers(tmp_path: Path) -> None:
    policy = SecurityPolicy(allowed_roots=[tmp_path], max_header_bytes=1024)

    # 1. Truncated < 8 bytes
    trunc = tmp_path / "trunc.safetensors"
    trunc.write_bytes(b"short")
    with pytest.raises(MalformedCheckpointError):
        validate_safetensors_header(trunc, policy=policy)

    # 2. Zero header length
    zero_h = tmp_path / "zero.safetensors"
    zero_h.write_bytes(struct.pack("<Q", 0) + b"{}")
    with pytest.raises(MalformedCheckpointError):
        validate_safetensors_header(zero_h, policy=policy)

    # 3. Oversized header DoS claim (> max_header_bytes)
    bomb = tmp_path / "bomb.safetensors"
    bomb.write_bytes(struct.pack("<Q", 50_000_000) + b"{}")
    with pytest.raises(SecurityValidationError):
        validate_safetensors_header(bomb, policy=policy)

    # 4. Out-of-bounds data_offsets
    hdr_oob = json.dumps({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 64]}}).encode("utf-8")
    oob = tmp_path / "oob.safetensors"
    oob.write_bytes(struct.pack("<Q", len(hdr_oob)) + hdr_oob + b"\x00" * 16)
    with pytest.raises(MalformedCheckpointError):
        validate_safetensors_header(oob, policy=policy)

    # 5. Overlapping tensor spans
    hdr_ov = json.dumps(
        {
            "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
        }
    ).encode("utf-8")
    ov = tmp_path / "overlap.safetensors"
    ov.write_bytes(struct.pack("<Q", len(hdr_ov)) + hdr_ov + b"\x00" * 16)
    with pytest.raises(MalformedCheckpointError):
        validate_safetensors_header(ov, policy=policy)


class _MaliciousPayload:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


@pytest.mark.security
def test_unsafe_pickle_rce_gadget_rejected_before_execution(tmp_path: Path) -> None:
    policy = SecurityPolicy(allowed_roots=[tmp_path], allow_pytorch_weights_only=True)
    evil_pt = tmp_path / "evil_model.pt"

    # Craft a ZIP-wrapped PyTorch-like archive containing an os.system RCE gadget
    raw_pkl = pickle.dumps({"state_dict": _MaliciousPayload()}, protocol=4)
    with zipfile.ZipFile(evil_pt, "w") as zf:
        zf.writestr("archive/data.pkl", raw_pkl)

    with pytest.raises(SecurityValidationError):
        inspect_pytorch_archive_safety(evil_pt)

    loader = ModelLoader(policy=policy)
    with pytest.raises(SecurityValidationError):
        loader.load(evil_pt)


@pytest.mark.security
def test_api_security_headers_payload_limit_and_traversal_blocking(tmp_path: Path) -> None:
    policy = SecurityPolicy(allowed_roots=[tmp_path])
    app = create_app(db_path=tmp_path / "sec_test.db", policy=policy, max_body_bytes=4096)
    client = TestClient(app)

    # Security response headers
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.headers["X-Content-Type-Options"] == "nosniff"
    assert health.headers["X-Frame-Options"] == "DENY"
    assert health.headers["Referrer-Policy"] == "no-referrer"

    # Path traversal blocked via API
    trav = client.post("/api/models/inspect", json={"source": "../../etc/passwd", "local_files_only": True})
    assert trav.status_code == 403
    assert trav.json()["error"] == "SECURITY_VALIDATION_ERROR"

    # Unknown extra field rejected by strict Pydantic schema
    extra = client.post(
        "/api/models/inspect",
        json={"source": "some_model", "unexpected_field": "injected"},
    )
    assert extra.status_code == 422

    # Oversized body rejected with HTTP 413
    huge_payload = json.dumps({"source": "x" * 10000})
    too_large = client.post(
        "/api/models/inspect",
        content=huge_payload,
        headers={"Content-Type": "application/json"},
    )
    assert too_large.status_code == 413
    assert too_large.json()["error"] == "PAYLOAD_TOO_LARGE"
