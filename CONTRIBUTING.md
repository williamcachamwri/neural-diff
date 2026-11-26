# Contributing to NEURAL DIFF

Thank you for contributing to **NEURAL DIFF — Git for AI Brains**.

---

## 1. Core Engineering Standards

Every pull request must uphold NEURAL DIFF's non-negotiable engineering rules:

1. **Evidence Over Claims:** Never submit placeholder functions, mock production paths, hardcoded benchmark outputs, or unverified test claims.
2. **Epistemic Discipline:** Any new analytical module must classify its claims accurately using `EpistemicStatus` (`OBSERVED`, `INFERRED`, `CORRELATED`, `CAUSALLY_VERIFIED`, `NOT_VERIFIED`). Never label basis-index shifts as monosemantic concepts or correlational drift as causal proof without intervention evidence.
3. **Security By Default:** Never enable `trust_remote_code=True` automatically, never call `torch.load(..., weights_only=False)`, and validate all filesystem paths through `core.security.validate_safe_path`.

---

## 2. Local Development Workflow

```powershell
# 1. Sync Python 3.12 virtual environment
python -m uv sync --locked --python 3.12

# 2. Install frontend dependencies
npm --prefix frontend ci

# 3. Run full lint, typecheck, unit, property, integration, regression, security & frontend suites
.\.venv\Scripts\python.exe scripts/run_all_tests.py

# 4. Run secret scan & dependency vulnerability audits
.\.venv\Scripts\python.exe scripts/run_security_audit.py
```

---

## 3. Commit Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat(<scope>): ...` — New genuine capability (`loader`, `diff`, `structural`, `neuron`, `activation`, `behavior`, `regression`, `report`, `api`, `frontend`, `cli`)
- `fix(<scope>): ...` — Bug or security remediation with regression test
- `perf(<scope>): ...` — Measured performance or memory improvement
- `test(<scope>): ...` — Unit, Hypothesis property, integration, security, or E2E tests
- `docs(<scope>): ...` — Technical documentation aligned with actual code
- `chore(<scope>): ...` / `build(<scope>): ...` — Tooling, CI, or lockfile maintenance
