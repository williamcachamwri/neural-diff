#!/usr/bin/env python3
"""Runnable Example: Create a branched Evolution DAG, run analysis, and export JSON/CSV/HTML/PDF."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.evolution.exporter import EvolutionReportExporter  # noqa: E402
from core.evolution.orchestrator import (  # noqa: E402
    EvolutionAnalysisConfig,
    EvolutionOrchestrator,
)
from core.evolution.registry import inspect_and_build_checkpoint_record  # noqa: E402
from core.evolution.store import EvolutionStore  # noqa: E402
from core.security import SecurityPolicy  # noqa: E402
from tests.fixtures.evolution_factory import create_known_evolution_checkpoints  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory(
        prefix="neural_diff_evo_demo_", ignore_cleanup_errors=True
    ) as tmp:
        root = Path(tmp)
        ckpts = create_known_evolution_checkpoints(root)
        policy = SecurityPolicy(allowed_roots=[root])
        store = EvolutionStore(db_path=root / "demo_evolution.db")

        proj = store.create_project(
            name="Demo Multi-Branch & Merge Evolution",
            description="Base -> SFT-A -> DPO-A + SFT-B -> Merged -> Quantized -> Pruned",
        )

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
        c_sft_a = store.add_checkpoint(
            inspect_and_build_checkpoint_record(
                project_id=proj.project_id,
                source=ckpts["b"],
                version_label="SFT-A",
                parent_checkpoint_ids=[c_base.checkpoint_id],
                transformation_type="SFT",
                policy=policy,
                local_files_only=True,
            )
        )
        c_dpo_a = store.add_checkpoint(
            inspect_and_build_checkpoint_record(
                project_id=proj.project_id,
                source=ckpts["c"],
                version_label="DPO-A",
                parent_checkpoint_ids=[c_sft_a.checkpoint_id],
                transformation_type="DPO",
                policy=policy,
                local_files_only=True,
            )
        )
        c_sft_b = store.add_checkpoint(
            inspect_and_build_checkpoint_record(
                project_id=proj.project_id,
                source=ckpts["sft_b"],
                version_label="SFT-B",
                parent_checkpoint_ids=[c_base.checkpoint_id],
                transformation_type="SFT",
                policy=policy,
                local_files_only=True,
            )
        )
        _ = store.add_checkpoint(
            inspect_and_build_checkpoint_record(
                project_id=proj.project_id,
                source=ckpts["merged"],
                version_label="Merged-AB",
                parent_checkpoint_ids=[c_dpo_a.checkpoint_id, c_sft_b.checkpoint_id],
                transformation_type="Model Merging",
                policy=policy,
                local_files_only=True,
            )
        )

        bundle = EvolutionOrchestrator(store=store, policy=policy).run_analysis(
            proj.project_id,
            EvolutionAnalysisConfig(
                include_behavioral_evolution=False,
                include_representation_evolution=True,
                local_files_only=True,
            ),
        )
        exporter = EvolutionReportExporter()
        out_dir = root / "reports"
        exported = {
            fmt: exporter.export_file(bundle, fmt, out_dir).stat().st_size
            for fmt in ("json", "csv", "html", "pdf")
        }
        print(
            json.dumps(
                {
                    "run_id": bundle["run_id"],
                    "overview": bundle["overview"],
                    "insights_count": len(bundle["insights"]),
                    "exported_report_bytes": exported,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
