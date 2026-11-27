"""Command-line interface (`neural-diff`) with Charmbracelet `ndiff` native TUI/CLI delegation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

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

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_charm_binary() -> Path | None:
    """Locate the compiled Charmbracelet Go CLI (`ndiff`) if available."""
    env_bin = os.environ.get("NEURAL_DIFF_CHARM_BIN")
    if env_bin:
        p = Path(env_bin)
        if p.is_file():
            return p

    exe_name = "ndiff.exe" if os.name == "nt" else "ndiff"
    local_bin = REPO_ROOT / "bin" / exe_name
    if local_bin.is_file():
        return local_bin

    which_bin = shutil.which("ndiff")
    if which_bin:
        return Path(which_bin)
    return None


def _should_emit_json(args: argparse.Namespace) -> bool:
    if getattr(args, "json", False) or os.environ.get("NEURAL_DIFF_CLI_JSON") == "1":
        return True
    if getattr(args, "pretty", False):
        return False
    return not sys.stdout.isatty()


def _format_rounded_kv_table(title: str, rows: list[tuple[str, str]], *, no_color: bool = False) -> str:
    """Minimalist rounded-border fallback table when `ndiff` Go binary is not compiled."""
    use_ansi = not no_color and "NO_COLOR" not in os.environ
    cyan = "\033[1;36m" if use_ansi else ""
    dim = "\033[90m" if use_ansi else ""
    bold = "\033[1m" if use_ansi else ""
    reset = "\033[0m" if use_ansi else ""

    w_key = max([len("FIELD"), *(len(k) for k, _ in rows)])
    w_val = max([len("VALUE"), *(min(84, len(v)) for _, v in rows)])

    top = f"{dim}╭─{'─' * w_key}─┬─{'─' * w_val}─╮{reset}"
    hdr = f"{dim}│{reset} {bold}{'FIELD'.ljust(w_key)}{reset} {dim}│{reset} {bold}{'VALUE'.ljust(w_val)}{reset} {dim}│{reset}"
    sep = f"{dim}├─{'─' * w_key}─┼─{'─' * w_val}─┤{reset}"
    bot = f"{dim}╰─{'─' * w_key}─┴─{'─' * w_val}─╯{reset}"

    lines = [f"{cyan}NEURAL DIFF{reset}  {bold}{title}{reset}", "", top, hdr, sep]
    for k, v in rows:
        cell = v if len(v) <= 84 else v[:81] + "..."
        lines.append(f"{dim}│{reset} {k.ljust(w_key)} {dim}│{reset} {cell.ljust(w_val)} {dim}│{reset}")
    lines.append(bot)
    return "\n".join(lines)


def _emit_payload(title: str, payload: dict[str, Any], args: argparse.Namespace) -> None:
    if _should_emit_json(args):
        print(json.dumps(payload, indent=2))
        return
    no_color = bool(getattr(args, "no_color", False))
    rows: list[tuple[str, str]] = []
    for k, v in payload.items():
        if isinstance(v, (dict, list)):
            rows.append((k, json.dumps(v, ensure_ascii=False)))
        else:
            rows.append((k, str(v)))
    print(_format_rounded_kv_table(title, rows, no_color=no_color))


def _add_shared_display_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="Emit deterministic machine-readable JSON.")
    p.add_argument("--pretty", action="store_true", help="Force rounded terminal table output.")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI color codes.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neural-diff",
        description="NEURAL DIFF — Structural, Neural & Behavioral Checkpoint Diffing",
    )
    _add_shared_display_flags(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    # status
    p_stat = sub.add_parser("status", help="Show workspace, SQLite, manifest, and suite readiness.")
    _add_shared_display_flags(p_stat)

    # tui
    p_tui = sub.add_parser("tui", help="Launch interactive Charm Bubble Tea v2 Terminal Studio.")
    p_tui.add_argument("--file", default="", help="Optional evolution benchmark JSON path.")
    p_tui.add_argument("--model", default="", help="Optional .safetensors checkpoint path.")

    # wizard
    sub.add_parser("wizard", help="Launch interactive Charm Huh v2 comparison configurator.")

    # dag
    p_dag = sub.add_parser("dag", help="Render the branched & merged Neural Evolution DAG.")
    p_dag.add_argument("--file", default="", help="Optional evolution benchmark JSON path.")
    _add_shared_display_flags(p_dag)

    # quickdiff
    p_qd = sub.add_parser("quickdiff", help="Fast structural header diff between Model A and Model B.")
    p_qd.add_argument("--model-a", required=True)
    p_qd.add_argument("--model-b", required=True)
    p_qd.add_argument("--local-only", action="store_true")
    _add_shared_display_flags(p_qd)

    # inspect
    p_insp = sub.add_parser("inspect", help="Inspect model checkpoint metadata safely.")
    p_insp.add_argument("model", help="Path to checkpoint/directory or Hugging Face repo ID.")
    p_insp.add_argument("--local-only", action="store_true", help="Do not download from Hub.")
    p_insp.add_argument("--filter", default="", help="Optional tensor name substring filter.")
    p_insp.add_argument("--top", type=int, default=24, help="Max tensors shown in pretty mode.")
    _add_shared_display_flags(p_insp)

    # validate
    p_val = sub.add_parser("validate", help="Validate compatibility between Model A and Model B.")
    p_val.add_argument("--model-a", required=True)
    p_val.add_argument("--model-b", required=True)
    p_val.add_argument("--local-only", action="store_true")
    _add_shared_display_flags(p_val)

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
    _add_shared_display_flags(p_cmp)

    # behavior (direct reproduction command target)
    p_beh = sub.add_parser("behavior", help="Run deterministic behavioral diff suite.")
    p_beh.add_argument("--model-a", required=True)
    p_beh.add_argument("--model-b", required=True)
    p_beh.add_argument("--category", default=None, help="Optional single category filter.")
    p_beh.add_argument("--max-cases", type=int, default=None)
    p_beh.add_argument("--seed", type=int, default=42)
    p_beh.add_argument("--device", default="auto")
    p_beh.add_argument("--output-dir", default=None)
    _add_shared_display_flags(p_beh)

    # suites
    p_suites = sub.add_parser("suites", help="List registered deterministic behavioral benchmark suites.")
    _add_shared_display_flags(p_suites)

    # svd
    p_svd = sub.add_parser(
        "svd",
        help="Analyze SVD singular spectrum and Shannon effective rank of a 2D weight delta.",
    )
    p_svd.add_argument("--model-a", required=True)
    p_svd.add_argument("--model-b", required=True)
    p_svd.add_argument("--tensor", required=True, help="Exact 2D matrix parameter name.")
    p_svd.add_argument("--local-only", action="store_true")
    _add_shared_display_flags(p_svd)

    # evolution
    p_evo = sub.add_parser("evolution", help="Manage Neural Evolution Tracking projects, checkpoints, and runs.")
    _add_shared_display_flags(p_evo)
    evo_sub = p_evo.add_subparsers(dest="evo_action", required=True)

    p_ep = evo_sub.add_parser("create-project", help="Create a new evolution tracking project.")
    p_ep.add_argument("--name", required=True)
    p_ep.add_argument("--description", default="")
    p_ep.add_argument("--db-path", default=None)
    _add_shared_display_flags(p_ep)

    p_er = evo_sub.add_parser("register", help="Register a checkpoint into an evolution project.")
    p_er.add_argument("--project-id", required=True)
    p_er.add_argument("--source", required=True)
    p_er.add_argument("--label", default=None)
    p_er.add_argument("--parents", default="", help="Comma-separated parent checkpoint IDs.")
    p_er.add_argument("--transformation", default=None)
    p_er.add_argument("--notes", default="")
    p_er.add_argument("--local-only", action="store_true")
    p_er.add_argument("--db-path", default=None)
    _add_shared_display_flags(p_er)

    p_ea = evo_sub.add_parser("analyze", help="Run multi-generation evolution analysis on a project.")
    p_ea.add_argument("--project-id", required=True)
    p_ea.add_argument("--skip-behavioral", action="store_true")
    p_ea.add_argument("--skip-representation", action="store_true")
    p_ea.add_argument("--force-rerun", action="store_true")
    p_ea.add_argument("--local-only", action="store_true")
    p_ea.add_argument("--output-dir", default=None, help="Export JSON/CSV/HTML/PDF reports to directory.")
    p_ea.add_argument("--db-path", default=None)
    _add_shared_display_flags(p_ea)

    return parser


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)

    # Delegate interactive TTY / `--pretty` CLI sessions to the compiled Charmbracelet `ndiff` binary.
    charm_bin = find_charm_binary()
    if (
        argv is None
        and charm_bin is not None
        and os.environ.get("NEURAL_DIFF_CLI_JSON") != "1"
        and "--json" not in effective_argv
        and (
            sys.stdout.isatty()
            or "--pretty" in effective_argv
            or (effective_argv and effective_argv[0] in {"tui", "wizard"})
        )
    ):
        env = dict(os.environ)
        env["NEURAL_DIFF_PYTHON"] = sys.executable
        proc = subprocess.run([str(charm_bin), *effective_argv], check=False, env=env)
        return int(proc.returncode)

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command in {"tui", "wizard"}:
            sys.stderr.write(
                json.dumps(
                    {
                        "error_code": "CHARM_BINARY_NOT_BUILT",
                        "message": (
                            f"`neural-diff {args.command}` uses the compiled Charmbracelet Go binary (`bin/ndiff`). "
                            "Build it via `python scripts/build_cli.py` or `cd cli && go build -o ../bin/ndiff ./cmd/ndiff`."
                        ),
                    },
                    indent=2,
                )
                + "\n"
            )
            return 2

        if args.command == "status":
            suite = load_benchmark_suite()
            db_file = REPO_ROOT / "data" / "neural_diff.db"
            manifest_file = REPO_ROOT / "evidence" / "EVIDENCE_MANIFEST.json"
            payload = {
                "repo_root": str(REPO_ROOT),
                "charm_binary": str(charm_bin) if charm_bin else None,
                "sqlite_present": db_file.is_file(),
                "evidence_manifest_present": manifest_file.is_file(),
                "suite_id": suite.suite_id,
                "total_cases": len(suite.cases),
            }
            _emit_payload("Workspace Status", payload, args)
            return 0

        if args.command == "dag":
            target = (
                Path(args.file)
                if args.file
                else REPO_ROOT / "evidence" / "evolution" / "evolution_benchmark.json"
            )
            if not target.is_file():
                raise ModelNotFoundError(f"Evolution ledger not found at '{target}'.")
            evo_data = json.loads(target.read_text(encoding="utf-8"))
            if _should_emit_json(args):
                print(json.dumps(evo_data, indent=2))
            else:
                ov = evo_data.get("overview", {})
                rs = evo_data.get("regression_summary", {})
                _emit_payload(
                    f"Evolution DAG · {evo_data.get('run_id', 'unknown')}",
                    {
                        "experiment_type": evo_data.get("experiment_type"),
                        "checkpoints": ov.get("total_checkpoints"),
                        "branches": ov.get("branch_count"),
                        "peak_cumulative_l2": ov.get("peak_cumulative_l2"),
                        "behavioral_regressions": rs.get("behavioral_regression_observed_count"),
                        "score_regressions": rs.get("score_regression_observed_count"),
                        "causally_verified": rs.get("causal_intervention_verified_count"),
                    },
                    args,
                )
            return 0

        if args.command == "validate":
            with (
                load_model(args.model_a, local_files_only=args.local_only) as ha,
                load_model(args.model_b, local_files_only=args.local_only) as hb,
            ):
                rep = validate_compatibility(ha, hb)
                _emit_payload("Compatibility Validation", rep.to_dict(), args)
            return 0

        if args.command == "quickdiff":
            with (
                load_model(args.model_a, local_files_only=args.local_only) as ha,
                load_model(args.model_b, local_files_only=args.local_only) as hb,
            ):
                rep = validate_compatibility(ha, hb)
                only_a = sorted(set(ha.tensors) - set(hb.tensors))
                only_b = sorted(set(hb.tensors) - set(ha.tensors))
                status = (
                    "INCOMPATIBLE"
                    if rep.shape_matched_tensor_count == 0
                    else (
                        "COMPATIBLE"
                        if not only_a
                        and not only_b
                        and rep.shape_matched_tensor_count == rep.shared_tensor_count
                        else "PARTIAL"
                    )
                )
                _emit_payload(
                    "Fast Structural Header Diff",
                    {
                        "model_a": ha.metadata.model_id,
                        "model_b": hb.metadata.model_id,
                        "compatible": rep.shape_matched_tensor_count > 0,
                        "status": status,
                        "aligned_tensors": rep.shape_matched_tensor_count,
                        "only_in_a_count": len(only_a),
                        "only_in_b_count": len(only_b),
                        "shape_mismatch_count": rep.shared_tensor_count - rep.shape_matched_tensor_count,
                        "parameter_count_a": ha.metadata.total_parameters,
                        "parameter_count_b": hb.metadata.total_parameters,
                        "parameter_delta": hb.metadata.total_parameters - ha.metadata.total_parameters,
                    },
                    args,
                )
            return 0

        if args.command == "inspect":
            with load_model(args.model, local_files_only=args.local_only) as handle:
                _emit_payload("Checkpoint Metadata", handle.metadata.to_dict(), args)
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
                _emit_payload(
                    "Comparison Result",
                    {
                        "comparison_id": bundle.comparison_id,
                        "verdict": bundle.regression.verdict if bundle.regression else None,
                        "exported_reports": {k: str(v) for k, v in paths.items()},
                    },
                    args,
                )
            elif _should_emit_json(args):
                print(exporter.to_json(bundle, include_full_tensor_records=False))
            else:
                _emit_payload(
                    "Comparison Result",
                    {
                        "comparison_id": bundle.comparison_id,
                        "model_a": bundle.model_a.model_id,
                        "model_b": bundle.model_b.model_id,
                        "verdict": bundle.regression.verdict if bundle.regression else "COMPLETED",
                    },
                    args,
                )
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
                _emit_payload("Behavioral Diff Exports", {k: str(v) for k, v in paths.items()}, args)
            elif _should_emit_json(args):
                print(exporter.to_json(bundle))
            else:
                _emit_payload(
                    "Behavioral Diff Result",
                    {
                        "comparison_id": bundle.comparison_id,
                        "verdict": bundle.regression.verdict if bundle.regression else "COMPLETED",
                    },
                    args,
                )
            return 0

        if args.command == "suites":
            suite = load_benchmark_suite()
            counts: dict[str, int] = {}
            for c in suite.cases:
                counts[c.category] = counts.get(c.category, 0) + 1
            _emit_payload(
                "Deterministic Behavioral Suite",
                {
                    "suite_id": suite.suite_id,
                    "version": suite.version,
                    "total_cases": len(suite.cases),
                    "categories": counts,
                },
                args,
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
                _emit_payload("SVD Task Vector Spectrum", {"tensor": args.tensor, **spec.to_dict()}, args)
            return 0

        if args.command == "evolution":
            evo_store = EvolutionStore(db_path=args.db_path)
            if args.evo_action == "create-project":
                proj = evo_store.create_project(name=args.name, description=args.description)
                _emit_payload("Evolution Project Created", proj.to_dict(), args)
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
                _emit_payload("Checkpoint Registered", saved.to_dict(), args)
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
                    _emit_payload(
                        "Evolution Analysis Exported",
                        {
                            "run_id": run_res["run_id"],
                            "cache_hit": run_res.get("cache_hit", False),
                            "exported_reports": exported,
                        },
                        args,
                    )
                else:
                    _emit_payload(
                        "Evolution Analysis Summary",
                        {"run_id": run_res["run_id"], "overview": run_res["overview"]},
                        args,
                    )
                return 0

    except NeuralDiffError as exc:
        sys.stderr.write(json.dumps(exc.to_dict(), indent=2) + "\n")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
