# Environment Audit — NEURAL DIFF

**Audit Status:** `VERIFIED` (Baseline Host Inspection) — Post-installation toolchain verification recorded separately after Phase 4 dependency provisioning.

**Raw Evidence Artifacts:**
- Baseline JSON: [environment_baseline.json](../evidence/environment/environment_baseline.json)
- Baseline Terminal Log: [environment_baseline.log](../evidence/environment/environment_baseline.log)
- `uv` Installation Log: [uv_install.log](../evidence/installation/uv_install.log)
- Post-Install Environment Audit (generated after Phase 4): [environment_post_install.json](../evidence/environment/environment_post_install.json) and [environment_post_install.log](../evidence/environment/environment_post_install.log)

---

## 1. Workspace & User Data Safety Audit

Before creating or modifying any files, `D:\LLM` and its parent directory `D:\` were inspected:

| Check | Finding | Status |
| :--- | :--- | :--- |
| Active Workspace (`D:\LLM`) | Empty directory (`0` existing files or folders) | `VERIFIED` |
| Existing Git Repository | None (`fatal: not a git repository`) prior to initialization | `VERIFIED` |
| Uncommitted User Changes | None present in `D:\LLM` | `VERIFIED` |
| Sibling Directories on `D:\` | Unrelated directories (`Ollama`, `OllamaModels`, `Zust`, etc.) left untouched | `VERIFIED` |
| Git Identity | Derived locally from authenticated GitHub CLI user (`William Cachamwri`, `2741582+williamcachamwri@users.noreply.github.com`) without altering global system config | `VERIFIED` |

`D:\LLM` serves directly as the `neural-diff` repository root so that all relative paths (`core/`, `api/`, `frontend/`, `evidence/`, `docs/`, `scripts/`) resolve cleanly from the workspace root.

---

## 2. Hardware & Operating System Profile

All values below were captured directly via OS CIM/WMI queries, `GlobalMemoryStatusEx`, `shutil.disk_usage`, and `nvidia-smi`:

| Component | Detected Value | Engineering Impact |
| :--- | :--- | :--- |
| **Operating System** | Microsoft Windows 11 Home Insider Preview Single Language (`10.0.26320`, 64-bit) | Windows path normalization, drive-root path traversal protection, and non-POSIX symlink handling required |
| **CPU** | Intel(R) Core(TM) i5-14450HX (10 physical cores, 16 logical threads, `AMD64`) | Multi-threaded CPU fallback and chunked NumPy/PyTorch reductions viable |
| **System RAM** | `15.71 GB` total physical RAM (`~4.14 GB` free at baseline snapshot) | Memory-bounded execution is critical; must implement lazy Safetensors slicing, chunked tensor diffing, and explicit garbage/CUDA cache reclamation |
| **Workspace Storage (`D:\`)** | `98.10 GB` free / `200.05 GB` total | Sufficient for `uv` cache (`D:\LLM\.cache\uv`), Hugging Face model cache (`D:\LLM\.cache\huggingface`), virtual environment, Node modules, and real sub-1B model benchmarks |
| **System Drive (`C:\`)** | `43.34 GB` free / `247.94 GB` total | Redirect heavy ML/package caches to `D:\LLM\.cache` to preserve system drive space |
| **Discrete GPU** | NVIDIA GeForce RTX 5050 Laptop GPU (`8,151 MiB` total VRAM, `~6,142 MiB` free at baseline) | Sufficient VRAM for `Qwen2.5-0.5B` (~1 GB bf16) and `SmolLM2-135M` (~270 MB bf16) real inference and activation extraction |
| **NVIDIA Driver & CUDA** | Driver `591.44`, CUDA Driver API `13.1`, Compute Capability `12.0` (`sm_120`, Blackwell) | Requires PyTorch built with **CUDA 12.8 (`cu128`) or newer** on Python 3.12/3.13; older `cu121`/`cu124` builds lack `sm_120` kernel images |
| **Integrated / Virtual Display** | Intel(R) UHD Graphics + virtual display adapters | Discrete CUDA device `cuda:0` maps to RTX 5050 Laptop GPU |
| **Apple Silicon / Metal (MPS)** | `False` (`AMD64` Windows host) | MPS code paths remain guarded by runtime `torch.backends.mps.is_available()` checks (`NOT VERIFIED` on non-Apple hardware) |

---

## 3. Baseline Software & Toolchain Audit

| Tool / Package | Baseline State | Resolution Strategy | Status |
| :--- | :--- | :--- | :--- |
| **System Python** | `Python 3.15.2` (`C:\Users\MSl\AppData\Local\Python\pythoncore-3.15-64\python.exe`) | System Python is >= 3.11, but prebuilt PyTorch `cu128` wheels target Python 3.12/3.13; `uv` will provision a pinned Python `3.12` virtual environment (`.venv`) | `VERIFIED` |
| **uv** | Not installed at baseline; installed `uv 0.13.10` via `python -m pip install uv` | Invoked via `python -m uv` (or `C:\Users\MSl\AppData\Local\Python\pythoncore-3.15-64\Scripts\uv.exe`) with pinned `uv.lock` | `VERIFIED` |
| **Node.js / npm / npx** | `Node v26.13.0`, `npm 11.20.0`, `npx 11.20.0` | Ready for React + TypeScript + Vite frontend and Playwright E2E automation | `VERIFIED` |
| **Git** | `git version 2.56.0.windows.4` | Ready for repository lifecycle and Conventional Commits history | `VERIFIED` |
| **GitHub CLI (`gh`)** | `gh version 2.103.3` | Available locally; no remote pushes executed without user instruction | `VERIFIED` |
| **Docker** | Not installed on `PATH` | Optional containerization not required for local execution; documented in limitations | `VERIFIED` |
| **ML & QA Python Stack** (`torch`, `transformers`, `safetensors`, `numpy`, `scipy`, `pytest`, `hypothesis`, `ruff`, `mypy`, `fastapi`, `pydantic`, `playwright`) | Not installed in global Python 3.15 environment | Provisioned cleanly inside reproducible `.venv` (Python 3.12) via `pyproject.toml` + `uv.lock` in Phase 4 | `VERIFIED` (Baseline) |

---

## 4. Key Architectural Constraints Derived from Environment Audit

1. **Blackwell (`sm_120`) GPU Verification:** PyTorch must be sourced from the official `https://download.pytorch.org/whl/cu128` wheel index under Python 3.12, and actual GPU kernel execution (`torch.dot` on `cuda:0`) must be verified rather than trusting `torch.cuda.is_available()` alone.
2. **RAM Headroom Management:** With ~4.1 GB free physical RAM at idle and ~6.1 GB free VRAM on RTX 5050, tensor diffing must stream tensors lazily from Safetensors files and process large arrays in bounded chunks rather than materializing two full state dicts simultaneously in host RAM.
3. **Cache Isolation on `D:\`:** Setting `UV_CACHE_DIR=D:\LLM\.cache\uv` and `HF_HOME=D:\LLM\.cache\huggingface` prevents exhausting `C:\` free space during wheel installation and real model downloads.
