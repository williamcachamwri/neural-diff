# Installation & Reproducible Setup Guide — NEURAL DIFF

**Verification Status:** `VERIFIED` on Windows 11 (`AMD64`, CPython `3.12.16`, `uv 0.13.10`, Node `v26.13.0`, NVIDIA GeForce RTX 5050 Laptop GPU `sm_120`). Logs preserved under [`evidence/installation/`](../evidence/installation/).

---

## 1. Prerequisites

- **Git** (`2.40+`)
- **Python** (`3.11`–`3.13`; `3.12` pinned via `.python-version` for official PyTorch CUDA wheels)
- **uv** (`0.5+` — recommended for deterministic lockfile installation via `uv.lock`)
- **Node.js** (`20+`) & **npm** (`10+`) for the React + TypeScript + Vite frontend and Playwright E2E tests
- *(Optional)* **NVIDIA GPU** with Driver `570+` supporting CUDA `12.8+` (CPU fallback is automatic when no supported GPU is present)

---

## 2. Backend & Core Engine Setup (`uv`)

From the repository root (`neural-diff` / `d:\LLM`):

```powershell
# Optional: keep large package and model caches off system drive
$env:UV_CACHE_DIR = "$PWD\.cache\uv"
$env:HF_HOME = "$PWD\.cache\huggingface"

# Install uv if not already available
python -m pip install uv

# Sync pinned Python 3.12 environment and PyTorch cu128 wheels from uv.lock
python -m uv sync --locked --python 3.12
```

Verify the environment and GPU/CPU compute engine:

```powershell
.\.venv\Scripts\python.exe scripts/capture_environment.py --stage verify
.\.venv\Scripts\python.exe -m core.cli --help
```

---

## 3. Frontend Setup (`npm`)

```powershell
npm --prefix frontend ci
npm --prefix frontend run build
```

*(Optional — required only for live browser E2E tests and automated screenshot capture)*:

```powershell
npx --prefix frontend playwright install chromium
```

---

## 4. Running the Full-Stack Application

### 4.1 Start the FastAPI Backend (`127.0.0.1:8000`)

```powershell
$env:HF_HOME = "$PWD\.cache\huggingface"
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Verify health endpoint: `http://127.0.0.1:8000/api/health`

### 4.2 Start the React Workbench (`127.0.0.1:5173`)

In a second terminal:

```powershell
npm --prefix frontend run dev
```

Open `http://127.0.0.1:5173` in your browser. Vite proxies `/api/*` requests directly to `http://127.0.0.1:8000`.

---

## 5. CLI Quickstart

```powershell
# Inspect a local checkpoint or open-access Hugging Face model
.\.venv\Scripts\python.exe -m core.cli inspect HuggingFaceTB/SmolLM2-135M

# Validate structural & architectural compatibility
.\.venv\Scripts\python.exe -m core.cli validate `
  --model-a Qwen/Qwen2.5-0.5B `
  --model-b Qwen/Qwen2.5-0.5B-Instruct

# Run full comparison and export JSON, Markdown, and interactive HTML reports
.\.venv\Scripts\python.exe -m core.cli compare `
  --model-a Qwen/Qwen2.5-0.5B `
  --model-b Qwen/Qwen2.5-0.5B-Instruct `
  --modules structural,tensor,neuron,activation,behavioral,regression `
  --real-experiment `
  --output-dir ./reports
```
