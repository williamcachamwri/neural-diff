"""Command-line interface (`neural-diff`) for reproducible checkpoint comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.exceptions import NeuralDiffError
from core.loaders.model_loader import load_model
from core.orchestrator import VALID_MODULES, ComparisonOrchestrator, ComparisonRequestConfig
from core.regression_engine.thresholds import RegressionThresholds
from core.report_engine.exporter import ReportEngine
from core.structural_analysis.compatibility import validate_compatibility
from core.tensor_engine.metrics import DiffTolerances


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

    except NeuralDiffError as exc:
        sys.stderr.write(json.dumps(exc.to_dict(), indent=2) + "\n")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
