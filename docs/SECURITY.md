# Security Architecture & Audit Report — NEURAL DIFF

**Security Audit Status:** `VERIFIED` (Automated secret scanning, dependency vulnerability auditing, and adversarial unit/API security tests executed; logs preserved under [`evidence/security/`](../evidence/security/)).

> **Residual Risk Statement:** No software system processing complex binary ML artifacts and third-party model repositories is completely secure. While NEURAL DIFF eliminates automatic remote code execution and hardens filesystem, serialization, and HTTP boundaries, residual risks remain (detailed in Section 4).

---

## 1. Threat Model & Implemented Controls

| Threat Vector | Control Implemented | Source Location |
| :--- | :--- | :--- |
| **Arbitrary Code Execution via Hugging Face Repositories** | `trust_remote_code=False` enforced unconditionally; `snapshot_download` explicitly passes `ignore_patterns=["*.py", "*.pyc", "*.so", "*.dll", "*.dylib"]` so untrusted Python modules in model repos are never downloaded or imported. | `adapters/huggingface/loader.py`, `core/activation_analysis/comparator.py`, `core/behavioral_analysis/runner.py` |
| **Python `pickle` Deserialization RCE (`.pt`, `.pth`, `.bin`)** | Safetensors is strictly preferred when present. Legacy PyTorch files must pass static `pickletools.genops` opcode inspection (`inspect_pytorch_archive_safety`) allowing only whitelisted tensor storage rebuild globals, followed by `torch.load(..., weights_only=True)`. Users can disable PyTorch archives entirely via `allow_pytorch_weights_only=False`. | `core/security.py`, `adapters/pytorch/loader.py` |
| **Path Traversal, UNC Device Paths & Symlink Escape** | `validate_safe_path` blocks null bytes (`\x00`), URI schemes (`file://`, `http://`), Windows raw/UNC device paths (`\\?\`, `\\.\`, `\\server\share`), `..` parent traversal, and symlinks escaping configured `allowed_roots`. Shard filenames inside `*.index.json` are confined to the index directory via `validate_safe_shard_filename`. | `core/security.py` |
| **Malformed / Adversarial Safetensors Binary Headers** | `validate_safetensors_header` inspects the 8-byte little-endian `u64` prefix and JSON header *before* invoking C/Rust `mmap` bindings, enforcing a `100 MiB` header cap, `4B` single-tensor element ceiling, exact `shape * dtype` byte-length matching, file-size bounds, and non-overlapping tensor buffer spans. | `core/security.py` |
| **HTTP Payload Exhaustion & Header Hardening** | `SecurityHardeningMiddleware` enforces a `2 MiB` JSON body ceiling (`HTTP 413`), sets `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, and `Cache-Control: no-store`. Pydantic v2 schemas enforce `extra="forbid"` and strict numeric/string bounds. | `api/security_middleware.py`, `api/schemas.py` |
| **Host RAM / GPU VRAM Exhaustion** | `SingleSafetensorsReader.iter_tensor_chunks` streams slices in bounded chunks (`1,048,576` elements default), `ActivationDiffEngine` and `BehavioralDiffEngine` load/unload models sequentially with `gc.collect()` + `torch.cuda.empty_cache()`, and `CancellationToken` enables cooperative job termination. | `core/resource_manager.py`, `adapters/safetensors/loader.py` |

---

## 2. Dependency Auditing & Failure-to-Fix Trace

### 2.1 Initial Findings & Remediation (Preserved Evidence)
1. **Frontend (`npm audit`):**
   - Initial install with `vitest@^3.0.5` surfaced 3 advisories in transitive `tinypool` / `@vitest/mocker`.
   - Remediated by upgrading `vitest` (`^5.1.2`) and `vite` (`^8.4.0`). Re-audit (`evidence/security/npm_audit.log`) exited with `0` (`found 0 vulnerabilities`).
2. **Python (`pip-audit`):**
   - Initial scan (`evidence/security/pip_audit_initial_failure.log`, exit code `1`) detected `setuptools 81.0.0` (`PYSEC-2026-3447`, `GHSA-xmj3-9gf7-h52w`).
   - Remediated by pinning `setuptools>=84.1.0` in `pyproject.toml`, upgrading `.venv` to `setuptools==84.2.1` (`torch==2.10.0+cu128`).
   - Rerun (`evidence/security/pip_audit.log`, exit code `0`) verified: **`No known vulnerabilities found`**.

### 2.2 Secret Scanning
- Executed `scripts/run_security_audit.py` across tracked and untracked repository files (`evidence/security/secret_scan.json`).
- **Result:** `0` secret/credential findings across AWS, GitHub PAT, Hugging Face `hf_`, Slack, and PEM private-key patterns.

---

## 3. Security Test Suite Verification

Adversarial tests in [`tests/security/test_security_controls.py`](../tests/security/test_security_controls.py) directly test:
- Relative and absolute path traversal outside allowed roots (`SecurityValidationError`, HTTP `403`).
- Null-byte, URI-scheme, and Windows UNC device path rejection.
- Malicious shard index (`../../outside.safetensors`) containment failure.
- Truncated, zero-length, oversized (`>100 MiB` header claim), out-of-bounds, and overlapping-offset Safetensors headers (`MalformedCheckpointError` / `SecurityValidationError`).
- Crafted PyTorch `.pt` archive containing `os.system` `__reduce__` RCE gadget blocked statically prior to deserialization.
- Oversized HTTP POST payload rejected with `HTTP 413 Payload Too Large` and strict Pydantic schema rejection (`HTTP 422`).

---

## 4. Known Limitations & Unresolved Residual Risks

1. **Localhost Single-Tenant API Design:** The FastAPI backend binds for local workstation analysis without built-in OAuth/JWT multi-tenant authentication. Deploying the API on an untrusted network interface requires placing it behind an authenticated reverse proxy (e.g., mTLS / OAuth2 proxy).
2. **Custom CUDA Wheel Audit Coverage:** Because `torch==2.10.0+cu128` uses a local version identifier (`+cu128`) hosted on `download.pytorch.org` rather than standard PyPI, `pip-audit` skips querying PyPI advisories for that wheel specifically (`Skip Reason: Dependency not found on PyPI`). Users must track official PyTorch security advisories directly.
3. **C++/Rust Native Library Memory Bugs:** While NEURAL DIFF pre-validates Safetensors headers in pure Python before calling native `safetensors` / `libtorch` routines, zero-day memory safety bugs inside upstream native tensor/tokenizer C++/Rust libraries cannot be ruled out completely without OS-level sandboxing/containers.
