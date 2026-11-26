#!/usr/bin/env python3
"""Generate deterministic synthetic HF checkpoints under `data/demo_fixtures/` for E2E & local dev."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.fixtures.checkpoint_factory import make_synthetic_hf_checkpoint_pair  # noqa: E402


def main() -> int:
    target = REPO_ROOT / "data" / "demo_fixtures"
    dir_a, dir_b = make_synthetic_hf_checkpoint_pair(target)
    print(f"Seeded demo fixtures: {dir_a.relative_to(REPO_ROOT)} & {dir_b.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
