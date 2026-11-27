#!/usr/bin/env python3
"""Refresh persisted pairwise & evolution Markdown/HTML/PDF/CSV reports from stored JSON bundles."""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.evolution.exporter import EvolutionReportExporter  # noqa: E402
from core.evolution.store import EvolutionStore  # noqa: E402
from core.report_engine.exporter import ReportEngine  # noqa: E402


def _enum(val: Any) -> SimpleNamespace:
    return SimpleNamespace(value=str(val))


def _wrap_bundle_dict(d: dict[str, Any]) -> Any:
    def _wrap_model(m: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            model_id=m["model_id"],
            total_parameters=int(m["total_parameters"]),
            format=_enum(m["format"]),
            revision=m.get("revision"),
        )

    ledger = [
        SimpleNamespace(
            epistemic_status=_enum(c["epistemic_status"]),
            statement=c["statement"],
            evidence_source=c["evidence_source"],
        )
        for c in d.get("epistemic_ledger", [])
    ]

    st_raw = d.get("structural")
    structural = (
        SimpleNamespace(
            shared_tensors=st_raw.get("shared_tensors", []),
            added_tensors=st_raw.get("added_tensors", []),
            removed_tensors=st_raw.get("removed_tensors", []),
            inferred_renames=st_raw.get("inferred_renames", []),
            shape_changes=st_raw.get("shape_changes", []),
            dtype_changes=st_raw.get("dtype_changes", []),
            delta_parameters=int(st_raw.get("delta_parameters", 0)),
        )
        if st_raw
        else None
    )

    td_raw = d.get("tensor_diff")
    tensor_diff = None
    if td_raw:
        ts = td_raw.get("summary", {})
        layer_summaries = [
            SimpleNamespace(
                layer_index=int(ls["layer_index"]),
                mean_relative_l2=ls.get("mean_relative_l2"),
                mean_cosine_similarity=ls.get("mean_cosine_similarity"),
                changed_pct=float(ls.get("changed_pct", 0.0)),
                role_breakdown=ls.get("role_breakdown", {}),
            )
            for ls in td_raw.get("layer_summaries", [])
        ]
        records = []
        for rec in td_raw.get("records", []):
            m = rec.get("metrics")
            records.append(
                SimpleNamespace(
                    name=rec["name"],
                    canonical_role=_enum(rec["canonical_role"]),
                    shape_a=rec.get("shape_a"),
                    shape_b=rec.get("shape_b"),
                    status=_enum(rec["status"]),
                    metrics=(
                        SimpleNamespace(
                            cosine_similarity=m.get("cosine_similarity"),
                            relative_l2=m.get("relative_l2"),
                            max_abs_diff=m.get("max_abs_diff"),
                            changed_pct=float(m.get("changed_pct", 0.0)),
                        )
                        if m
                        else None
                    ),
                )
            )
        tensor_diff = SimpleNamespace(
            compared_tensors=int(ts.get("compared_tensors", 0)),
            identical_tensors=int(ts.get("identical_tensors", 0)),
            within_tolerance_tensors=int(ts.get("within_tolerance_tensors", 0)),
            modified_tensors=int(ts.get("modified_tensors", 0)),
            total_changed_parameters=int(ts.get("total_changed_parameters", 0)),
            total_compared_parameters=int(ts.get("total_compared_parameters", 0)),
            global_changed_pct=float(ts.get("global_changed_pct", 0.0)),
            global_mean_cosine=ts.get("global_mean_cosine"),
            global_mean_relative_l2=ts.get("global_mean_relative_l2"),
            global_max_abs_diff=ts.get("global_max_abs_diff"),
            layer_summaries=layer_summaries,
            top_diverged_names=td_raw.get("top_diverged_names", []),
            records=records,
        )

    act_raw = d.get("activation")
    activation = None
    if act_raw:
        asum = act_raw.get("summary", {})
        activation = SimpleNamespace(
            prompt_count=int(asum.get("prompt_count", 0)),
            mean_final_token_js_divergence=asum.get("mean_final_token_js_divergence"),
            cross_prompt_pooled_cka_final_layer=asum.get("cross_prompt_pooled_cka_final_layer"),
            mean_hidden_cosine_by_layer={
                int(k): v for k, v in (asum.get("mean_hidden_cosine_by_layer") or {}).items()
            },
            mean_hidden_cka_by_layer={
                int(k): v for k, v in (asum.get("mean_hidden_cka_by_layer") or {}).items()
            },
        )

    beh_raw = d.get("behavioral")
    behavioral = None
    if beh_raw:
        ov = beh_raw["overall_statistics"]
        behavioral = SimpleNamespace(
            suite_id=beh_raw["suite_id"],
            suite_version=beh_raw["suite_version"],
            overall_statistics=SimpleNamespace(**ov),
            category_summaries=[SimpleNamespace(**cs) for cs in beh_raw.get("category_summaries", [])],
            cases=[
                SimpleNamespace(
                    case=SimpleNamespace(**pc["case"]),
                    eval_a=SimpleNamespace(**pc["eval_a"]),
                    eval_b=SimpleNamespace(**pc["eval_b"]),
                    score_delta=float(pc["score_delta"]),
                    status_change=pc["status_change"],
                    output_a=pc["output_a"],
                    output_b=pc["output_b"],
                )
                for pc in beh_raw.get("cases", [])
            ],
        )

    reg_raw = d.get("regression")
    regression = None
    if reg_raw:
        regression = SimpleNamespace(
            verdict=reg_raw["verdict"],
            highest_severity=_enum(reg_raw["highest_severity"]),
            thresholds=SimpleNamespace(policy_name=reg_raw["thresholds"]["policy_name"]),
            findings=[
                SimpleNamespace(
                    severity=_enum(f["severity"]),
                    epistemic_status=_enum(f["epistemic_status"]),
                    title=f["title"],
                    description=f["description"],
                )
                for f in reg_raw.get("findings", [])
            ],
        )

    return SimpleNamespace(
        comparison_id=d["comparison_id"],
        created_at_utc=d["created_at_utc"],
        experiment_type=d["experiment_type"],
        model_a=_wrap_model(d["model_a"]),
        model_b=_wrap_model(d["model_b"]),
        selected_modules=d.get("selected_modules", []),
        structural=structural,
        tensor_diff=tensor_diff,
        activation=activation,
        behavioral=behavioral,
        regression=regression,
        execution_telemetry=d.get("execution_telemetry", {}),
        reproducibility=d.get("reproducibility", {}),
        build_epistemic_ledger=lambda: ledger,
    )


def main() -> int:
    engine = ReportEngine()
    db_path = REPO_ROOT / "data" / "neural_diff.db"

    # 1. Refresh all completed pairwise rows in SQLite
    updated_db = 0
    if db_path.is_file():
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT comparison_id, bundle_json FROM experiments WHERE status='completed' AND bundle_json IS NOT NULL"
            ).fetchall()
            for comp_id, b_json in rows:
                wrapped = _wrap_bundle_dict(json.loads(b_json))
                md = engine.to_markdown(wrapped)
                ht = engine.to_html(wrapped)
                conn.execute(
                    "UPDATE experiments SET markdown_report = ?, html_report = ? WHERE comparison_id = ?",
                    (md, ht, comp_id),
                )
                updated_db += 1
            conn.commit()
    print(f"Refreshed {updated_db} pairwise report rows in {db_path}.")

    # 2. Refresh all persisted pairwise report files under `evidence/model_comparisons` & `evidence/releases`
    for json_file in (REPO_ROOT / "evidence" / "model_comparisons").rglob("*.json"):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        if "comparison_id" not in data or "model_a" not in data:
            continue
        wrapped = _wrap_bundle_dict(data)
        md_file = json_file.with_suffix(".md")
        html_file = json_file.with_suffix(".html")
        md_file.write_text(engine.to_markdown(wrapped), encoding="utf-8")
        html_file.write_text(engine.to_html(wrapped), encoding="utf-8")
        print(f"Refreshed {md_file.relative_to(REPO_ROOT)} & {html_file.relative_to(REPO_ROOT)}")

    qwen_dir = REPO_ROOT / "evidence" / "model_comparisons" / "real_qwen25_05b"
    rel_dir = REPO_ROOT / "evidence" / "releases"
    if (qwen_dir / "ndiff-real-qwen25-05b.md").is_file() and rel_dir.is_dir():
        shutil.copy2(qwen_dir / "ndiff-real-qwen25-05b.md", rel_dir / "ndiff-real-qwen25-05b.md")
        shutil.copy2(qwen_dir / "ndiff-real-qwen25-05b.html", rel_dir / "ndiff-real-qwen25-05b.html")

    # 3. Refresh evolution exports (`json`, `csv`, `markdown`, `html`, `pdf`)
    evo_store = EvolutionStore(db_path=db_path)
    try:
        evo_bundle = evo_store.get_run("evorun-real-smollm2")
        evo_exp = EvolutionReportExporter()
        evo_out = REPO_ROOT / "evidence" / "evolution" / "exports"
        for fmt in ("json", "csv", "markdown", "html", "pdf"):
            evo_exp.export_file(evo_bundle, fmt, evo_out)
        print(f"Refreshed 5 evolution export formats in {evo_out.relative_to(REPO_ROOT)}.")
    except Exception as exc:
        print(f"Skipped evolution export refresh: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
