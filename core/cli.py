"""Command-line interface (`neural-diff`) for reproducible checkpoint comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmarks.datasets.loader import load_benchmark_suite
from core.evolution.exporter import EvolutionReportExporter
from core.evolution.orchestrator import EvolutionAnalysisConfig, EvolutionOrchestrator
from core.evolution.registry import inspect_and_build_checkpoint_record
from core.evolution.store import EvolutionStore
from core.exceptions import ModelNotFoundError, NeuralDiffError
from core.loaders.model_loader import load_model
from core.orchestrator import VALID_MODULES, ComparisonOrchestrator, ComparisonRequestConfig
from core.regression_engine.thresholds import RegressionThresholds
from core.report_engine.exporter import ReportEngine
from core.structural_analysis.compatibility import validate_compatibility
from core.tensor_engine.metrics import DiffTolerances
from core.tensor_engine.task_vector import analyze_task_vector_spectrum


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neural-diff",
        description="NEURAL DIFF — Git for AI Brains (Structural, Neural & Behavioral Diffing)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_insp = sub.add_parser("inspect", help="Inspect model checkpoint metadata safely.")
    p_insp.add_argument("model", help="Path to checkpoint/directory or Hugging Face repo ID.")
    p_insp.add_argument("--local-only", action="store_true", help="Do not download from Hub.")

    # validate
    p_val = sub.add_parser("validate", help="Validate compatibility between Model A and Model B.")
    p_val.add_argument("--model-a", required=True)
    p_val.add_argument("--model-b", required=True)
    p_val.add_argument("--local-only", action="store_true")

    # compare
    p_cmp = sub.add_parser("compare", help="Run selected analysis modules and export reports.")
    p_cmp.add_argument("--model-a", required=True)
    p_cmp.add_argument("--model-b", required=True)
    p_cmp.add_argument(
        "--modules",
        default="structural,tensor,neuron,regression",
        help=f"Comma-separated modules ({','.join(VALID_MODULES)}).",
    )
    p_cmp.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p_cmp.add_argument("--atol", type=float, default=1e-6)
    p_cmp.add_argument("--rtol", type=float, default=1e-5)
    p_cmp.add_argument("--seed", type=int, default=42)
    p_cmp.add_argument("--max-new-tokens", type=int, default=128)
    p_cmp.add_argument("--real-experiment", action="store_true")
    p_cmp.add_argument("--local-only", action="store_true")
    p_cmp.add_argument("--output-dir", default=None, help="Directory to export JSON/MD/HTML reports.")

    # behavior (direct reproduction command target)
    p_beh = sub.add_parser("behavior", help="Run deterministic behavioral diff suite.")
    p_beh.add_argument("--model-a", required=True)
    p_beh.add_argument("--model-b", required=True)
    p_beh.add_argument("--category", default=None, help="Optional single category filter.")
    p_beh.add_argument("--max-cases", type=int, default=None)
    p_beh.add_argument("--seed", type=int, default=42)
    p_beh.add_argument("--device", default="auto")
    p_beh.add_argument("--output-dir", default=None)

    # suites
    sub.add_parser("suites", help="List registered deterministic behavioral benchmark suites.")

    # svd
    p_svd = sub.add_parser(
        "svd",
        help="Analyze SVD singular spectrum and Shannon effective rank of a 2D weight delta.",
    )
    p_svd.add_argument("--model-a", required=True)
    p_svd.add_argument("--model-b", required=True)
    p_svd.add_argument("--tensor", required=True, help="Exact 2D matrix parameter name.")
    p_svd.add_argument("--local-only", action="store_true")

    # evolution
    p_evo = sub.add_parser("evolution", help="Manage Neural Evolution Tracking projects, checkpoints, and runs.")
    evo_sub = p_evo.add_subparsers(dest="evo_action", required=True)

    p_ep = evo_sub.add_parser("create-project", help="Create a new evolution tracking project.")
    p_ep.add_argument("--name", required=True)
    p_ep.add_argument("--description", default="")
    p_ep.add_argument("--db-path", default=None)

    p_er = evo_sub.add_parser("register", help="Register a checkpoint into an evolution project.")
    p_er.add_argument("--project-id", required=True)
    p_er.add_argument("--source", required=True)
    p_er.add_argument("--label", default=None)
    p_er.add_argument("--parents", default="", help="Comma-separated parent checkpoint IDs.")
    p_er.add_argument("--transformation", default=None)
    p_er.add_argument("--notes", default="")
    p_er.add_argument("--local-only", action="store_true")
    p_er.add_argument("--db-path", default=None)

    p_ea = evo_sub.add_parser("analyze", help="Run multi-generation evolution analysis on a project.")
    p_ea.add_argument("--project-id", required=True)
    p_ea.add_argument("--skip-behavioral", action="store_true")
    p_ea.add_argument("--skip-representation", action="store_true")
    p_ea.add_argument("--force-rerun", action="store_true")
    p_ea.add_argument("--local-only", action="store_true")
    p_ea.add_argument("--output-dir", default=None, help="Export JSON/CSV/HTML/PDF reports to directory.")
    p_ea.add_argument("--db-path", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "inspect":
            with load_model(args.model, local_files_only=args.local_only) as handle:
                print(json.dumps(handle.metadata.to_dict(), indent=2))
            return 0

        if args.command == "validate":
            with (
                load_model(args.model_a, local_files_only=args.local_only) as ha,
                load_model(args.model_b, local_files_only=args.local_only) as hb,
            ):
                rep = validate_compatibility(ha, hb)
                print(json.dumps(rep.to_dict(), indent=2))
            return 0

        if args.command == "compare":
            mods = [m.strip() for m in str(args.modules).split(",") if m.strip()]
            cfg = ComparisonRequestConfig(
                model_a=args.model_a,
                model_b=args.model_b,
                modules=mods,
                experiment_type=(
                    "REAL MODEL EXPERIMENT"
                    if args.real_experiment
                    else "SYNTHETIC FIXTURE EXPERIMENT"
                ),
                tolerances=DiffTolerances(atol=args.atol, rtol=args.rtol),
                thresholds=RegressionThresholds(),
                preferred_device=args.device,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                local_files_only=args.local_only,
            )
            bundle = ComparisonOrchestrator().run(cfg)
            exporter = ReportEngine()
            if args.output_dir:
                paths = exporter.export_all(bundle, Path(args.output_dir))
                print(
                    json.dumps(
                        {
                            "comparison_id": bundle.comparison_id,
                            "verdict": bundle.regression.verdict if bundle.regression else None,
                            "exported_reports": {k: str(v) for k, v in paths.items()},
                        },
                        indent=2,
                    )
                )
            else:
                print(exporter.to_json(bundle, include_full_tensor_records=False))
            return 0

        if args.command == "behavior":
            cats = [args.category] if args.category else None
            cfg = ComparisonRequestConfig(
                model_a=args.model_a,
                model_b=args.model_b,
                modules=["behavioral", "regression"],
                preferred_device=args.device,
                seed=args.seed,
                behavioral_categories=cats,
                behavioral_max_cases=args.max_cases,
            )
            bundle = ComparisonOrchestrator().run(cfg)
            exporter = ReportEngine()
            if args.output_dir:
                paths = exporter.export_all(bundle, Path(args.output_dir))
                print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
            else:
                print(exporter.to_json(bundle))
            return 0

        if args.command == "suites":
            suite = load_benchmark_suite()
            counts: dict[str, int] = {}
            for c in suite.cases:
                counts[c.category] = counts.get(c.category, 0) + 1
            print(
                json.dumps(
                    {
                        "suite_id": suite.suite_id,
                        "version": suite.version,
                        "total_cases": len(suite.cases),
                        "categories": counts,
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "svd":
            with (
                load_model(args.model_a, local_files_only=args.local_only) as ha,
                load_model(args.model_b, local_files_only=args.local_only) as hb,
            ):
                if args.tensor not in ha.tensors or args.tensor not in hb.tensors:
                    raise ModelNotFoundError(
                        f"Tensor '{args.tensor}' not found in both checkpoints."
                    )
                spec = analyze_task_vector_spectrum(
                    ha.get_tensor(args.tensor), hb.get_tensor(args.tensor)
                )
                print(json.dumps({"tensor": args.tensor, **spec.to_dict()}, indent=2))
            return 0

        if args.command == "evolution":
            evo_store = EvolutionStore(db_path=args.db_path)
            if args.evo_action == "create-project":
                proj = evo_store.create_project(name=args.name, description=args.description)
                print(json.dumps(proj.to_dict(), indent=2))
                return 0
            if args.evo_action == "register":
                parents = [p.strip() for p in str(args.parents).split(",") if p.strip()]
                ckpt = inspect_and_build_checkpoint_record(
                    project_id=args.project_id,
                    source=args.source,
                    version_label=args.label,
                    parent_checkpoint_ids=parents,
                    transformation_type=args.transformation,
                    experiment_notes=args.notes,
                    existing_project_checkpoints=evo_store.list_checkpoints(args.project_id),
                    local_files_only=args.local_only,
                )
                saved = evo_store.add_checkpoint(ckpt)
                print(json.dumps(saved.to_dict(), indent=2))
                return 0
            if args.evo_action == "analyze":
                cfg_evo = EvolutionAnalysisConfig(
                    include_behavioral_evolution=not args.skip_behavioral,
                    include_representation_evolution=not args.skip_representation,
                    force_rerun=args.force_rerun,
                    local_files_only=args.local_only,
                )
                run_res = EvolutionOrchestrator(store=evo_store).run_analysis(
                    args.project_id, cfg_evo
                )
                if args.output_dir:
                    exp = EvolutionReportExporter()
                    exported = {
                        fmt: str(exp.export_file(run_res, fmt, args.output_dir))
                        for fmt in ("json", "csv", "markdown", "html", "pdf")
                    }
                    print(
                        json.dumps(
                            {
                                "run_id": run_res["run_id"],
                                "cache_hit": run_res.get("cache_hit", False),
                                "exported_reports": exported,
                            },
                            indent=2,
                        )
                    )
                else:
                    print(json.dumps({"run_id": run_res["run_id"], "overview": run_res["overview"]}, indent=2))
                return 0

    except NeuralDiffError as exc:
        sys.stderr.write(json.dumps(exc.to_dict(), indent=2) + "\n")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
