"""Multi-format Evolution Report Exporter: JSON, CSV, Markdown, HTML (5 SVGs), and PDF."""

from __future__ import annotations

import csv
import html
import io
import json
from pathlib import Path
from typing import Any

from core.evolution.pdf_writer import MinimalPDFBuilder
from core.report_engine.visuals import (
    human_label,
    md_bar,
    svg_dual_line_chart,
    svg_evolution_dag,
    svg_matrix_heatmap,
    svg_radar_polygon,
)


def _fmt_num(val: Any, digits: int = 6) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def _fmt_pct(val: Any, digits: int = 1) -> str:
    if val is None:
        return "—"
    if isinstance(val, (int, float)):
        return f"{float(val) * 100.0:.{digits}f}%"
    return str(val)


class EvolutionReportExporter:
    """Exports complete `EvolutionStore.get_run(...)` bundles into JSON, CSV, HTML, and PDF."""

    def to_json(self, run_bundle: dict[str, Any]) -> str:
        return json.dumps(run_bundle, indent=2, allow_nan=False)

    def to_csv(self, run_bundle: dict[str, Any]) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")

        proj = run_bundle.get("project", {})
        prov = run_bundle.get("provenance", {})
        writer.writerow(["# NEURAL DIFF EVOLUTION REPORT", run_bundle.get("run_id", "")])
        writer.writerow(["project_id", proj.get("project_id", ""), "project_name", proj.get("name", "")])
        writer.writerow(["content_cache_key", run_bundle.get("cache_key", ""), "seed", prov.get("random_seed", "")])
        writer.writerow([])

        # Section 1: Checkpoints
        writer.writerow(["[CHECKPOINT_REGISTRY]"])
        writer.writerow(
            [
                "checkpoint_id",
                "version_label",
                "transformation_type",
                "parent_ids",
                "parameter_count",
                "layer_count",
                "hidden_size",
                "vocab_size",
                "tensor_count",
                "format",
                "file_size_bytes",
                "sha256_fingerprint",
                "validation_status",
            ]
        )
        for c in run_bundle.get("checkpoints", []):
            writer.writerow(
                [
                    c["checkpoint_id"],
                    c["version_label"],
                    c["transformation_type"],
                    ";".join(c.get("parent_checkpoint_ids", [])),
                    c["parameter_count"],
                    c.get("layer_count") or "",
                    c.get("hidden_size") or "",
                    c.get("vocab_size") or "",
                    c["tensor_count"],
                    c["checkpoint_format"],
                    c["file_size_bytes"],
                    c["sha256_fingerprint"],
                    c["validation_status"],
                ]
            )
        writer.writerow([])

        # Section 2: Tensor Transitions (Consecutive, Base-to-Current, Branch-to-Branch)
        writer.writerow(["[TENSOR_EVOLUTION_TRANSITIONS]"])
        writer.writerow(
            [
                "scope",
                "source_label",
                "target_label",
                "transformation_type",
                "epistemic_status",
                "l2_distance",
                "relative_l2_distance",
                "rmse",
                "max_abs_diff",
                "global_cosine_similarity",
                "modified_parameter_pct",
                "sparsity_delta",
                "zero_to_nonzero",
                "nonzero_to_zero",
            ]
        )
        for t in run_bundle.get("tensor_transitions", []):
            writer.writerow(
                [
                    t["scope"],
                    t["source_label"],
                    t["target_label"],
                    t["transformation_type"],
                    t["epistemic_status"],
                    t.get("l2_distance") if t.get("l2_distance") is not None else "",
                    t.get("relative_l2_distance") if t.get("relative_l2_distance") is not None else "",
                    t.get("rmse") if t.get("rmse") is not None else "",
                    t.get("max_abs_diff") if t.get("max_abs_diff") is not None else "",
                    t.get("global_cosine_similarity") if t.get("global_cosine_similarity") is not None else "",
                    t.get("modified_parameter_pct", 0.0),
                    t.get("sparsity_delta") if t.get("sparsity_delta") is not None else "",
                    t.get("zero_to_nonzero_transitions", 0),
                    t.get("nonzero_to_zero_transitions", 0),
                ]
            )
        writer.writerow([])

        # Section 3: Behavioral Timeline
        beh = run_bundle.get("behavioral_evolution")
        if beh:
            writer.writerow(["[BEHAVIORAL_EVOLUTION_TIMELINE]"])
            writer.writerow(
                [
                    "checkpoint_id",
                    "version_label",
                    "transformation_type",
                    "sample_size",
                    "pass_rate",
                    "wilson_ci95_low",
                    "wilson_ci95_high",
                    "mean_score",
                    "factual_contradiction_rate",
                ]
            )
            for row in beh.get("timeline", []):
                ci = row.get("pass_rate_wilson_ci95", [0.0, 0.0])
                writer.writerow(
                    [
                        row["checkpoint_id"],
                        row["version_label"],
                        row["transformation_type"],
                        row["sample_size"],
                        row["pass_rate"],
                        ci[0],
                        ci[1],
                        row["mean_score"],
                        row.get("factual_contradiction_rate") if row.get("factual_contradiction_rate") is not None else "",
                    ]
                )
            writer.writerow([])

        # Section 4: Representation CKA Matrix
        rep = run_bundle.get("representation_evolution")
        if rep and rep.get("evolution_matrix"):
            mat = rep["evolution_matrix"]
            lids = mat.get("layer_indices", [])
            writer.writerow(["[REPRESENTATION_CKA_MATRIX]"])
            writer.writerow(
                [
                    "checkpoint_id",
                    "version_label",
                    "mean_cka_to_base",
                    "representation_drift",
                    *[f"layer_{lid}_cka" for lid in lids],
                ]
            )
            for r in mat.get("rows", []):
                writer.writerow(
                    [
                        r["checkpoint_id"],
                        r["version_label"],
                        r.get("mean_cka_to_base") if r.get("mean_cka_to_base") is not None else "",
                        r.get("representation_drift") if r.get("representation_drift") is not None else "",
                        *[v if v is not None else "" for v in r.get("layer_cka", [])],
                    ]
                )
            writer.writerow([])

        # Section 5: Regression Origin Events
        writer.writerow(["[REGRESSION_ORIGIN_EVENTS]"])
        writer.writerow(
            [
                "event_id",
                "previous_label",
                "first_affected_label",
                "transformation_type",
                "affected_category",
                "pass_rate_before",
                "pass_rate_after",
                "score_drop",
                "failed_cases",
                "epistemic_status",
                "causal_verification_exists",
            ]
        )
        for ev in run_bundle.get("regression_events", []):
            writer.writerow(
                [
                    ev["event_id"],
                    ev["previous_label"],
                    ev["first_affected_label"],
                    ev["transformation_type"],
                    ev["affected_category"],
                    ev["pass_rate_before"],
                    ev["pass_rate_after"],
                    ev["score_drop"],
                    ";".join(ev.get("failed_test_case_ids", [])),
                    ev["epistemic_status"],
                    ev["causal_verification"]["causal_verification_exists"],
                ]
            )

        return buf.getvalue()

    def to_markdown(self, run_bundle: dict[str, Any]) -> str:
        """Render a minimalist Markdown evolution report with Mermaid DAG and Unicode gauges."""
        proj = run_bundle.get("project", {})
        ov = run_bundle.get("overview") or {}
        prov = run_bundle.get("provenance") or {}
        ckpts = run_bundle.get("checkpoints", [])
        graph = run_bundle.get("graph", {})
        t_trans = run_bundle.get("tensor_transitions", [])
        beh = run_bundle.get("behavioral_evolution")
        rep = run_bundle.get("representation_evolution")
        regs = run_bundle.get("regression_events", [])
        insights = run_bundle.get("insights", [])

        lines: list[str] = [
            f"# NEURAL DIFF · Evolution Report — `{proj.get('name', 'Project')}`",
            "",
            "| Project ID | Run ID | Checkpoints | Branches | Regressions | Peak Direct L2 |",
            "| :--- | :--- | ---: | ---: | ---: | ---: |",
            (
                f"| `{proj.get('project_id', '')}` | `{run_bundle.get('run_id', '')}` "
                f"| {ov.get('total_checkpoints', len(ckpts))} "
                f"| {ov.get('branch_count', 0)} "
                f"| {ov.get('detected_regressions', len(regs))} "
                f"| `{_fmt_num(ov.get('peak_cumulative_l2'), 4)}` |"
            ),
        ]

        # Dynamic Mermaid Branched & Merged DAG
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        if nodes:
            lines.extend(["", "```mermaid", "flowchart LR"])
            for n in nodes:
                nid = str(n["checkpoint_id"]).replace("-", "_")
                lbl = str(n.get("version_label", n["checkpoint_id"])).replace('"', "'")
                stg = str(n.get("transformation_type", "")).replace('"', "'")
                lines.append(f'  {nid}["{lbl}\\n{stg}"]')
            for e in edges:
                s_id = str(e["parent_checkpoint_id"]).replace("-", "_")
                t_id = str(e["child_checkpoint_id"]).replace("-", "_")
                mw = e.get("merge_weight")
                edge_lbl = (
                    f"{e.get('transformation_type', '')} ({mw:.2f})"
                    if isinstance(mw, (int, float))
                    else str(e.get("transformation_type", ""))
                )
                lines.append(f'  {s_id} -->|"{edge_lbl}"| {t_id}')
            lines.append("```")

        if insights:
            lines.extend(
                [
                    "",
                    "## Empirical Findings",
                    "",
                    "| Domain | Evidence | Observation | Scope |",
                    "| :--- | :--- | :--- | :--- |",
                ]
            )
            for ins in insights:
                scope_txt = (
                    ", ".join(ins.get("evaluation_cases", [])[:4])
                    or f"{len(ins.get('checkpoints', []))} checkpoints"
                )
                lines.append(
                    f"| {human_label(ins.get('domain'))} "
                    f"| {human_label(ins.get('epistemic_status'))} "
                    f"| {ins.get('statement', '')} | `{scope_txt}` |"
                )

        # Direct Cumulative Base-to-Current Weight Distance
        base_trans = [t for t in t_trans if t.get("scope") == "base_to_current"]
        if base_trans:
            max_l2 = max((float(t.get("l2_distance") or 0.0) for t in base_trans), default=1.0) or 1.0
            lines.extend(
                [
                    "",
                    "## Direct Cumulative Weight Trajectory (`Base → Current`)",
                    "",
                    "| Checkpoint | Stage | Direct L2 | Step Sum L2 | Direct / Sum | Cosine | Magnitude |",
                    "| :--- | :--- | ---: | ---: | ---: | ---: | :--- |",
                ]
            )
            for t in base_trans:
                l2_v = float(t.get("l2_distance") or 0.0)
                step_v = t.get("sum_of_step_l2_along_path")
                ratio_v = t.get("path_cancellation_ratio")
                lines.append(
                    f"| `{t['target_label']}` | {t['transformation_type']} "
                    f"| `{l2_v:.4f}` | `{_fmt_num(step_v, 4)}` "
                    f"| `{_fmt_num(ratio_v, 3)}` | `{_fmt_num(t.get('global_cosine_similarity'), 5)}` "
                    f"| `{md_bar(l2_v, max_l2, 12)}` |"
                )

        # Behavioral Evolution Timeline
        if beh and beh.get("timeline"):
            lines.extend(
                [
                    "",
                    f"## Behavioral Capability Trajectory (`{beh.get('suite_id')}@{beh.get('suite_version')}`, n={beh.get('sample_size')})",
                    "",
                    "| Version | Stage | Pass Rate | 95% Wilson CI | Mean Score | Pass Profile |",
                    "| :--- | :--- | ---: | :--- | ---: | :--- |",
                ]
            )
            for row in beh["timeline"]:
                pr = float(row["pass_rate"])
                ci = row.get("pass_rate_wilson_ci95", [0.0, 0.0])
                lines.append(
                    f"| `{row['version_label']}` | {row['transformation_type']} "
                    f"| `{pr * 100:.1f}%` | `[{ci[0] * 100:.0f}%, {ci[1] * 100:.0f}%]` "
                    f"| `{float(row['mean_score']):.3f}` | `{md_bar(pr, 1.0, 12)}` |"
                )

        # Representation CKA Matrix
        if rep and rep.get("evolution_matrix") and rep["evolution_matrix"].get("rows"):
            rows = rep["evolution_matrix"]["rows"]
            lines.extend(
                [
                    "",
                    "## Representation Alignment (`Centered Linear CKA vs Base`)",
                    "",
                    "| Version | Stage | Mean CKA | Drift (1−CKA) | Alignment |",
                    "| :--- | :--- | ---: | ---: | :--- |",
                ]
            )
            for r in rows:
                m_cka = r.get("mean_cka_to_base")
                lines.append(
                    f"| `{r['version_label']}` | {r.get('transformation_type', '')} "
                    f"| `{_fmt_num(m_cka, 4)}` | `{_fmt_num(r.get('representation_drift'), 4)}` "
                    f"| `{md_bar(m_cka, 1.0, 12)}` |"
                )

        # Regression Origin Events
        if regs:
            lines.extend(
                [
                    "",
                    "## Regression Origin Pinpointing",
                    "",
                    "| Transition | Stage | Category | Pass Shift | Evidence | Failed Cases |",
                    "| :--- | :--- | :--- | ---: | :--- | :--- |",
                ]
            )
            for ev in regs:
                cases_str = ", ".join(ev.get("failed_test_case_ids", [])[:5]) or "—"
                lines.append(
                    f"| `{ev['previous_label']} → {ev['first_affected_label']}` "
                    f"| {ev['transformation_type']} | {human_label(ev['affected_category'])} "
                    f"| `{_fmt_pct(ev['pass_rate_before'])} → {_fmt_pct(ev['pass_rate_after'])}` "
                    f"| {human_label(ev['epistemic_status'])} | `{cases_str}` |"
                )

        lines.extend(
            [
                "",
                "## Provenance",
                "",
                "| Parameter | Value |",
                "| :--- | :--- |",
                f"| Content Cache Key | `{run_bundle.get('cache_key', '')}` |",
                f"| Device & Seed | `{prov.get('selected_device', 'cpu')}` · seed `{prov.get('random_seed', 42)}` |",
                f"| Dataset SHA-256 | `{str(prov.get('evaluation_dataset_fingerprint') or 'N/A')[:24]}` |",
                "",
            ]
        )
        return "\n".join(lines)

    def to_html(self, run_bundle: dict[str, Any]) -> str:
        """Render a minimalist Architectural Monochrome HTML evolution report with 5 rich SVGs."""
        esc = html.escape
        proj = run_bundle.get("project", {})
        ov = run_bundle.get("overview") or {}
        prov = run_bundle.get("provenance") or {}
        ckpts = run_bundle.get("checkpoints", [])
        graph = run_bundle.get("graph", {})
        t_trans = run_bundle.get("tensor_transitions", [])
        layer_evo = run_bundle.get("layer_evolution")
        beh = run_bundle.get("behavioral_evolution")
        rep = run_bundle.get("representation_evolution")
        regs = run_bundle.get("regression_events", [])
        insights = run_bundle.get("insights", [])

        reg_target_ids = {str(r["first_affected_checkpoint_id"]) for r in regs}
        pass_by_id = {
            str(item["checkpoint_id"]): float(item["pass_rate"])
            for item in (beh.get("timeline", []) if beh else [])
        }
        base_trans = [t for t in t_trans if t.get("scope") == "base_to_current"]
        base_l2_by_id = {
            str(t["target_checkpoint_id"]): float(t["l2_distance"])
            for t in base_trans
            if t.get("l2_distance") is not None
        }

        # 1. Vector Branched & Merged DAG SVG
        dag_svg = svg_evolution_dag(
            nodes=graph.get("nodes", []),
            edges=graph.get("edges", []),
            regression_target_ids=reg_target_ids,
            pass_rate_by_id=pass_by_id,
            base_l2_by_id=base_l2_by_id,
        )

        # 2. Direct Cumulative Base-to-Current L2 vs Step Sum Dual-Curve SVG
        svg_weight = ""
        if base_trans:
            xlbls = [str(t["target_label"]) for t in base_trans]
            direct_vals = [float(t.get("l2_distance") or 0.0) for t in base_trans]
            step_vals = [
                float(t["sum_of_step_l2_along_path"])
                if t.get("sum_of_step_l2_along_path") is not None
                else float(t.get("l2_distance") or 0.0)
                for t in base_trans
            ]
            hi_y = max([*direct_vals, *step_vals, 1.0]) * 1.08
            svg_weight = svg_dual_line_chart(
                x_labels=xlbls,
                series_primary=direct_vals,
                label_primary="Direct ||W_current - W_base||_2",
                series_secondary=step_vals,
                label_secondary="Sum of Consecutive Step L2",
                y_min=0.0,
                y_max=hi_y,
            )

        # 3. Layer Evolution Matrix Heatmap (Transitions × Layers)
        svg_layer_heat = ""
        if layer_evo and layer_evo.get("heatmap") and layer_evo["heatmap"].get("rows"):
            hm = layer_evo["heatmap"]
            svg_layer_heat = svg_matrix_heatmap(
                row_labels=[str(r["transition_label"]) for r in hm["rows"]],
                col_labels=[f"L{lid}" for lid in hm.get("layer_indices", [])],
                matrix=[r.get("relative_l2_by_layer", []) for r in hm["rows"]],
            )

        # 4. Behavioral Pass-Rate Curve + 8-Domain Capability Radar Polygon
        svg_beh_curve = ""
        svg_beh_radar = ""
        if beh and beh.get("timeline"):
            tl = beh["timeline"]
            svg_beh_curve = svg_dual_line_chart(
                x_labels=[str(x["version_label"]) for x in tl],
                series_primary=[float(x["pass_rate"]) for x in tl],
                label_primary="Pass Rate",
                series_secondary=[float(x["mean_score"]) for x in tl],
                label_secondary="Mean Rubric Score",
                y_min=0.0,
                y_max=1.0,
            )
            cats = sorted({k for item in tl for k in item.get("category_scores", {})})
            if len(cats) >= 3:
                sample_ckpts = [tl[0]]
                if len(tl) > 2:
                    best_mid = max(tl[1:-1], key=lambda item: float(item["pass_rate"]))
                    sample_ckpts.append(best_mid)
                if len(tl) > 1:
                    sample_ckpts.append(tl[-1])
                palette = ["#71717a", "#4ade80", "#d6c5a9"]
                svg_beh_radar = svg_radar_polygon(
                    axes=[human_label(c) for c in cats],
                    series=[
                        (
                            str(ck["version_label"]),
                            [float(ck.get("category_scores", {}).get(c, 0.0)) for c in cats],
                            palette[i % len(palette)],
                        )
                        for i, ck in enumerate(sample_ckpts)
                    ],
                    size=275,
                )

        # 5. Representation Centered Linear CKA Matrix Heatmap (Versions × Layers)
        svg_cka_heat = ""
        if rep and rep.get("evolution_matrix") and rep["evolution_matrix"].get("rows"):
            em = rep["evolution_matrix"]
            svg_cka_heat = svg_matrix_heatmap(
                row_labels=[str(r["version_label"]) for r in em["rows"]],
                col_labels=[f"L{lid}" for lid in em.get("layer_indices", [])],
                matrix=[r.get("layer_cka", []) for r in em["rows"]],
                min_val=0.0,
                max_val=1.0,
            )

        def _dot_col(status_str: str, domain: str = "") -> str:
            if domain == "regression_origin":
                return "#f87171"
            if status_str == "CAUSALLY VERIFIED":
                return "#4ade80"
            if status_str == "OBSERVED":
                return "#e4e4e7"
            return "#d6c5a9"

        def _ins_scope(ins: dict[str, Any]) -> str:
            cases = ins.get("evaluation_cases", [])
            return ", ".join(cases[:4]) if cases else f"{len(ins.get('checkpoints', []))} ckpts"

        insight_rows = "".join(
            f"<tr>"
            f"<td class='muted nowrap'>{esc(human_label(ins.get('domain')))}</td>"
            f"<td class='nowrap'><span class='dot' style='background:{_dot_col(ins.get('epistemic_status', ''), ins.get('domain', ''))}'></span>"
            f"{esc(human_label(ins.get('epistemic_status')))}</td>"
            f"<td>{esc(ins.get('statement', ''))}</td>"
            f"<td class='mono muted'>{esc(_ins_scope(ins))}</td>"
            f"</tr>"
            for ins in insights
        )

        ckpt_rows = "".join(
            f"<tr>"
            f"<td class='mono'>{esc(c['version_label'])}</td>"
            f"<td>{esc(c['transformation_type'])}</td>"
            f"<td class='num mono'>{c['parameter_count']:,}</td>"
            f"<td class='num mono'>{c.get('layer_count') or '—'}</td>"
            f"<td class='mono muted'>{esc(', '.join(c.get('dtypes_present', [])))}</td>"
            f"<td class='mono muted'>{esc(c['sha256_fingerprint'][:16])}…</td>"
            f"<td>{esc(human_label(c['validation_status']))}</td>"
            f"</tr>"
            for c in ckpts
        )

        t_rows = "".join(
            f"<tr>"
            f"<td class='muted'>{esc(human_label(t['scope']))}</td>"
            f"<td class='mono'>{esc(t['source_label'])} → {esc(t['target_label'])}</td>"
            f"<td>{esc(t['transformation_type'])}</td>"
            f"<td class='num mono'>{_fmt_num(t.get('l2_distance'), 4)}</td>"
            f"<td class='num mono'>{_fmt_num(t.get('relative_l2_distance'), 5)}</td>"
            f"<td class='num mono'>{_fmt_num(t.get('global_cosine_similarity'), 6)}</td>"
            f"<td class='num mono'>{float(t.get('modified_parameter_pct', 0.0)):.2f}%</td>"
            f"<td class='num mono'>{_fmt_num(t.get('sparsity_delta'), 4)}</td>"
            f"</tr>"
            for t in t_trans
        )

        reg_rows = "".join(
            f"<tr>"
            f"<td class='mono'>{esc(ev['previous_label'])} → {esc(ev['first_affected_label'])}</td>"
            f"<td>{esc(ev['transformation_type'])}</td>"
            f"<td>{esc(human_label(ev['affected_category']))}</td>"
            f"<td class='num mono' style='color:#f87171'>{_fmt_pct(ev['pass_rate_before'])} → {_fmt_pct(ev['pass_rate_after'])}</td>"
            f"<td class='mono muted'>{esc(', '.join(ev.get('failed_test_case_ids', [])[:5]) or '—')}</td>"
            f"<td class='nowrap'><span class='dot' style='background:{_dot_col(ev['epistemic_status'])}'></span>{esc(human_label(ev['epistemic_status']))}</td>"
            f"</tr>"
            for ev in regs
        )

        reg_col = "#f87171" if len(regs) > 0 else "#f4f4f5"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NEURAL DIFF Evolution — {esc(proj.get('name', 'Project'))}</title>
<style>
  :root {{
    --bg: #09090b; --surface: #121216; --border: rgba(255,255,255,0.08);
    --border-faint: rgba(255,255,255,0.05); --text: #f4f4f5; --muted: #71717a; --champagne: #d6c5a9;
  }}
  * {{ box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--text); font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 32px 28px; line-height: 1.5; -webkit-font-smoothing: antialiased; }}
  .container {{ max-width: 1180px; margin: 0 auto; }}
  header {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap; padding-bottom: 18px; border-bottom: 1px solid var(--border); margin-bottom: 20px; }}
  h1 {{ margin: 0; font-size: 20px; font-weight: 600; letter-spacing: -0.02em; }}
  h2 {{ margin: 0 0 12px 0; font-size: 13.5px; font-weight: 600; letter-spacing: -0.01em; display: flex; justify-content: space-between; align-items: center; }}
  .sub {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-bottom: 18px; }}
  .kpi {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }}
  .kpi .lbl {{ color: var(--muted); font-size: 11px; }}
  .kpi .val {{ font-size: 18px; font-weight: 600; margin-top: 4px; font-variant-numeric: tabular-nums; }}
  .panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; margin-bottom: 16px; }}
  .grid-beh {{ display: grid; grid-template-columns: 1.5fr 1fr; gap: 16px; align-items: center; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border-faint); vertical-align: top; }}
  th {{ color: var(--muted); font-weight: 500; font-size: 11.5px; border-bottom: 1px solid var(--border); }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .mono {{ font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 11.5px; }}
  .muted {{ color: var(--muted); }}
  .nowrap {{ white-space: nowrap; }}
  .dot {{ display: inline-block; width: 6px; height: 6px; border-radius: 50%; margin-right: 7px; vertical-align: middle; }}
  .kv-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; font-size: 12px; }}
  .kv-item {{ padding: 8px 10px; border: 1px solid var(--border-faint); border-radius: 6px; background: rgba(255,255,255,0.015); }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>NEURAL DIFF · {esc(proj.get('name', 'Evolution Report'))}</h1>
      <div class="sub mono">Project {esc(proj.get('project_id', ''))} · Run {esc(run_bundle.get('run_id', ''))} · Cache {esc(str(run_bundle.get('cache_key', ''))[:16])}…</div>
    </div>
    <div class="mono muted">{esc(str(prov.get('selected_device', 'cpu')))} · seed {prov.get('random_seed', 42)}</div>
  </header>

  <div class="kpis">
    <div class="kpi"><div class="lbl">Checkpoints</div><div class="val mono">{ov.get('total_checkpoints', len(ckpts))}</div></div>
    <div class="kpi"><div class="lbl">Transformations</div><div class="val mono">{ov.get('total_transformations', 0)}</div></div>
    <div class="kpi"><div class="lbl">DAG Branches</div><div class="val mono">{ov.get('branch_count', 0)}</div></div>
    <div class="kpi"><div class="lbl">Regression Origins</div><div class="val mono" style="color:{reg_col}">{ov.get('detected_regressions', len(regs))}</div></div>
    <div class="kpi"><div class="lbl">Peak Direct L2</div><div class="val mono">{_fmt_num(ov.get('peak_cumulative_l2'), 4)}</div></div>
  </div>

  <div class="panel">
    <h2><span>Branched &amp; Merged Evolution Topology</span><span class="muted mono">{len(ckpts)} registered checkpoints</span></h2>
    <div style="overflow-x:auto">{dag_svg}</div>
    <table style="margin-top:14px">
      <thead><tr><th>Version</th><th>Stage</th><th class="num">Parameters</th><th class="num">Layers</th><th>Dtypes</th><th>SHA-256</th><th>Validation</th></tr></thead>
      <tbody>{ckpt_rows}</tbody>
    </table>
  </div>

  <div class="panel">
    <h2><span>Empirical Findings</span><span class="muted mono">{len(insights)} threshold-verified findings</span></h2>
    <table>
      <thead><tr><th>Domain</th><th>Evidence</th><th>Observation</th><th>Scope</th></tr></thead>
      <tbody>{insight_rows or '<tr><td colspan="4" class="muted">No threshold-exceeding insights triggered.</td></tr>'}</tbody>
    </table>
  </div>

  <div class="panel">
    <h2><span>Direct Cumulative Weight Distance vs Step Summation</span><span class="muted mono">Exact float64 ||W_current − W_base||_2</span></h2>
    {svg_weight or "<div class='muted'>No base-to-current tensor transitions recorded.</div>"}
    {f'<div style="margin-top:14px">{svg_layer_heat}</div>' if svg_layer_heat else ''}
    <table style="margin-top:14px">
      <thead><tr><th>Scope</th><th>Transition</th><th>Stage</th><th class="num">L2 Dist</th><th class="num">Rel L2</th><th class="num">Cosine</th><th class="num">Mod %</th><th class="num">Sparsity Δ</th></tr></thead>
      <tbody>{t_rows}</tbody>
    </table>
  </div>

  {f'''<div class="panel">
    <h2><span>Behavioral Capability Trajectory</span><span class="muted mono">{esc(str((beh or {}).get("suite_id", "")))} · n={(beh or {}).get("sample_size", 0)}</span></h2>
    <div class="grid-beh">
      <div>{svg_beh_curve}</div>
      <div>{svg_beh_radar}</div>
    </div>
  </div>''' if svg_beh_curve else ''}

  {f'''<div class="panel">
    <h2><span>Representation Alignment Matrix</span><span class="muted mono">Centered Linear CKA vs Base across depth</span></h2>
    {svg_cka_heat}
  </div>''' if svg_cka_heat else ''}

  <div class="panel">
    <h2><span>Regression Origin Pinpointing</span><span class="muted mono">First observed transition per drop</span></h2>
    <table>
      <thead><tr><th>Origin Transition</th><th>Stage</th><th>Category</th><th class="num">Pass Shift</th><th>Failed Cases</th><th>Evidence</th></tr></thead>
      <tbody>{reg_rows or '<tr><td colspan="6" class="muted">Zero regressions exceeded configured thresholds.</td></tr>'}</tbody>
    </table>
  </div>

  <div class="panel">
    <h2><span>Reproducibility &amp; Provenance</span><span class="muted mono">Content-addressed SHA-256 cache</span></h2>
    <div class="kv-grid mono">
      <div class="kv-item"><div class="muted">Cache Key</div><div>{esc(str(run_bundle.get("cache_key", ""))[:32])}…</div></div>
      <div class="kv-item"><div class="muted">Dataset Fingerprint</div><div>{esc(str(prov.get("evaluation_dataset_fingerprint") or "N/A")[:32])}</div></div>
      <div class="kv-item"><div class="muted">Toolchain</div><div>PyTorch {esc(str((prov.get("software_versions") or {}).get("torch", "—")))}</div></div>
    </div>
  </div>
</div>
</body>
</html>
"""

    def to_pdf(self, run_bundle: dict[str, Any]) -> bytes:
        proj = run_bundle.get("project", {})
        ov = run_bundle.get("overview") or {}
        prov = run_bundle.get("provenance") or {}
        ckpts = run_bundle.get("checkpoints", [])
        t_trans = run_bundle.get("tensor_transitions", [])
        beh = run_bundle.get("behavioral_evolution")
        rep = run_bundle.get("representation_evolution")
        regs = run_bundle.get("regression_events", [])
        insights = run_bundle.get("insights", [])

        pdf = MinimalPDFBuilder(title=f"NEURAL DIFF Evolution - {proj.get('name', 'Report')}")
        pdf.add_banner(
            f"NEURAL DIFF  |  {proj.get('name', 'Evolution Report')}",
            f"Run {run_bundle.get('run_id', '')}  |  Checkpoints: {len(ckpts)}  |  Branches: {ov.get('branch_count', 0)}  |  Regressions: {len(regs)}",
        )

        # 1. Empirical Findings Table (Zero [OBSERVED] paragraph clutter)
        pdf.add_section_heading("Empirical Findings")
        if insights:
            pdf.add_table(
                ["Domain", "Evidence", "Observation"],
                [
                    [
                        human_label(ins.get("domain")),
                        human_label(ins.get("epistemic_status")),
                        str(ins.get("statement", "")),
                    ]
                    for ins in insights
                ],
                col_weights=[1.0, 1.0, 3.8],
            )
        else:
            pdf.add_paragraph("No threshold-exceeding insights recorded for this run.")

        # 2. Checkpoint Registry
        pdf.add_section_heading("Checkpoint Topology & Fingerprints")
        pdf.add_table(
            ["Version", "Stage", "Params", "Layers", "SHA-256", "Status"],
            [
                [
                    c["version_label"],
                    c["transformation_type"],
                    f"{c['parameter_count']:,}",
                    str(c.get("layer_count") or "-"),
                    c["sha256_fingerprint"][:14],
                    human_label(c["validation_status"]),
                ]
                for c in ckpts
            ],
            col_weights=[1.4, 1.3, 1.0, 0.6, 1.3, 1.0],
        )

        # 3. Direct Cumulative Weight Chart + Transition Table
        pdf.add_section_heading("Direct Cumulative Weight Distance & Transitions")
        base_trans = [t for t in t_trans if t.get("scope") == "base_to_current"]
        if base_trans:
            pdf.add_horizontal_bar_chart(
                "Direct Base-to-Current L2 Distance ||W_current - W_base||_2",
                [
                    (
                        t["target_label"],
                        float(t.get("l2_distance") or 0.0),
                        f"L2={float(t.get('l2_distance') or 0.0):.4f}",
                    )
                    for t in base_trans
                ],
            )
        pdf.add_table(
            ["Scope", "Transition", "Rel L2", "Cosine", "Mod %", "Sparsity D"],
            [
                [
                    human_label(t["scope"]),
                    f"{t['source_label']}->{t['target_label']}",
                    _fmt_num(t.get("relative_l2_distance"), 5),
                    _fmt_num(t.get("global_cosine_similarity"), 5),
                    f"{float(t.get('modified_parameter_pct', 0.0)):.1f}%",
                    _fmt_num(t.get("sparsity_delta"), 4),
                ]
                for t in t_trans
            ],
            col_weights=[1.1, 2.0, 1.0, 1.0, 0.8, 0.9],
        )

        # 4. Behavioral Evolution Trajectory
        pdf.add_section_heading("Behavioral Capability Trajectory")
        if beh and beh.get("timeline"):
            tl = beh["timeline"]
            pdf.add_polyline_chart(
                f"Pass-Rate Trajectory ({beh.get('suite_id')}@{beh.get('suite_version')}, n={beh.get('sample_size')})",
                [str(item["version_label"]) for item in tl],
                [float(item["pass_rate"]) for item in tl],
                max_value=1.0,
            )
            pdf.add_horizontal_bar_chart(
                "Pass Rate by Checkpoint",
                [
                    (
                        item["version_label"],
                        float(item["pass_rate"]),
                        f"{float(item['pass_rate']) * 100.0:.1f}%",
                    )
                    for item in tl
                ],
                max_value=1.0,
            )
        else:
            pdf.add_paragraph("Behavioral evaluation module not executed.")

        # 5. Representation CKA
        pdf.add_section_heading("Representation Alignment (Centered Linear CKA vs Base)")
        if rep and rep.get("evolution_matrix") and rep["evolution_matrix"].get("rows"):
            rows = rep["evolution_matrix"]["rows"]
            lids = rep["evolution_matrix"]["layer_indices"][:6]
            pdf.add_table(
                ["Version", "Mean CKA", "Drift", *[f"L{lid}" for lid in lids]],
                [
                    [
                        r["version_label"],
                        _fmt_num(r.get("mean_cka_to_base"), 4),
                        _fmt_num(r.get("representation_drift"), 4),
                        *[_fmt_num(v, 3) for v in r.get("layer_cka", [])[:6]],
                    ]
                    for r in rows
                ],
            )
        else:
            pdf.add_paragraph("Representation evolution matrix not available.")

        # 6. Regression Origins
        pdf.add_section_heading("Regression Origin Pinpointing")
        if regs:
            pdf.add_table(
                ["Origin Transition", "Stage", "Category", "Drop", "Evidence", "Cases"],
                [
                    [
                        f"{e['previous_label']}->{e['first_affected_label']}",
                        e["transformation_type"],
                        human_label(e["affected_category"]),
                        _fmt_pct(e["score_drop"]),
                        human_label(e["epistemic_status"]),
                        ",".join(e.get("failed_test_case_ids", [])) or "-",
                    ]
                    for e in regs
                ],
                col_weights=[1.8, 1.1, 1.1, 0.7, 1.1, 1.2],
            )
        else:
            pdf.add_paragraph("Zero regressions exceeded configured thresholds.")

        # 7. Provenance
        pdf.add_section_heading("Provenance")
        pdf.add_paragraph(
            f"Cache Key: {run_bundle.get('cache_key', '')}  |  Seed: {prov.get('random_seed', 42)}  |  "
            f"Device: {prov.get('selected_device', 'cpu')}  |  Dataset SHA-256: {str(prov.get('evaluation_dataset_fingerprint') or 'N/A')[:24]}",
            mono=True,
            font_size=8.0,
        )

        return pdf.to_bytes()

    def export_file(self, run_bundle: dict[str, Any], fmt: str, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rid = str(run_bundle.get("run_id", "evolution_report"))
        fmt_l = fmt.lower().strip()
        if fmt_l == "json":
            p = out / f"{rid}.json"
            p.write_text(self.to_json(run_bundle), encoding="utf-8")
            return p
        if fmt_l == "csv":
            p = out / f"{rid}.csv"
            p.write_text(self.to_csv(run_bundle), encoding="utf-8")
            return p
        if fmt_l in ("markdown", "md"):
            p = out / f"{rid}.md"
            p.write_text(self.to_markdown(run_bundle), encoding="utf-8")
            return p
        if fmt_l == "html":
            p = out / f"{rid}.html"
            p.write_text(self.to_html(run_bundle), encoding="utf-8")
            return p
        if fmt_l == "pdf":
            p = out / f"{rid}.pdf"
            p.write_bytes(self.to_pdf(run_bundle))
            return p
        raise ValueError(
            f"Unsupported export format '{fmt}'. Expected json, csv, markdown, html, or pdf."
        )
