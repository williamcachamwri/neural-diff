"""Multi-format Evolution Report Exporter: JSON, CSV, HTML (SVG charts), and PDF (Section 12)."""

from __future__ import annotations

import csv
import html
import io
import json
from pathlib import Path
from typing import Any

from core.evolution.pdf_writer import MinimalPDFBuilder


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

    def to_html(self, run_bundle: dict[str, Any]) -> str:
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
        warnings = run_bundle.get("warnings", [])

        # Build SVG Cumulative L2 chart
        base_trans = [t for t in t_trans if t.get("scope") == "base_to_current"]
        max_l2 = max((float(t.get("l2_distance") or 0.0) for t in base_trans), default=1.0) or 1.0
        l2_bars_svg: list[str] = []
        for idx, t in enumerate(base_trans):
            val = float(t.get("l2_distance") or 0.0)
            w = max(2, int((val / max_l2) * 380))
            y = 12 + idx * 24
            lbl = html.escape(str(t["target_label"])[:24])
            l2_bars_svg.append(
                f"<text x='4' y='{y + 12}' fill='#e4e4e7' font-size='11' font-family='monospace'>{lbl}</text>"
                f"<rect x='170' y='{y + 2}' width='{w}' height='12' fill='#d6c5a9' rx='2'/>"
                f"<text x='{178 + w}' y='{y + 12}' fill='#a1a1aa' font-size='11' font-family='monospace'>L2={val:.5f}</text>"
            )
        svg_weight = (
            f"<svg viewBox='0 0 680 {max(40, len(base_trans) * 24 + 20)}' width='100%'>{''.join(l2_bars_svg)}</svg>"
            if l2_bars_svg
            else "<p class='muted'>No base-to-current tensor transitions recorded.</p>"
        )

        # Build SVG Behavioral Pass Rate chart
        beh_bars_svg: list[str] = []
        if beh and beh.get("timeline"):
            for idx, item in enumerate(beh["timeline"]):
                pr = float(item["pass_rate"])
                w = max(2, int(pr * 380))
                y = 12 + idx * 24
                lbl = html.escape(str(item["version_label"])[:24])
                ci = item.get("pass_rate_wilson_ci95", [0.0, 0.0])
                beh_bars_svg.append(
                    f"<text x='4' y='{y + 12}' fill='#e4e4e7' font-size='11' font-family='monospace'>{lbl}</text>"
                    f"<rect x='170' y='{y + 2}' width='{w}' height='12' fill='#e4e4e7' rx='2'/>"
                    f"<text x='{178 + w}' y='{y + 12}' fill='#a1a1aa' font-size='11' font-family='monospace'>"
                    f"{pr * 100:.1f}% (95% CI [{ci[0] * 100:.0f}%, {ci[1] * 100:.0f}%])</text>"
                )
        svg_beh = (
            f"<svg viewBox='0 0 720 {max(40, len(beh_bars_svg) * 24 + 20)}' width='100%'>{''.join(beh_bars_svg)}</svg>"
            if beh_bars_svg
            else "<p class='muted'>Behavioral evolution module was not executed or no compatible checkpoints.</p>"
        )

        # Checkpoint table rows
        ckpt_rows = "".join(
            f"<tr>"
            f"<td class='mono'>{html.escape(c['version_label'])}</td>"
            f"<td>{html.escape(c['transformation_type'])}</td>"
            f"<td class='mono'>{c['parameter_count']:,}</td>"
            f"<td class='mono'>{c.get('layer_count') or '—'}</td>"
            f"<td class='mono'>{html.escape(', '.join(c.get('dtypes_present', [])))}</td>"
            f"<td class='mono'>{html.escape(c['sha256_fingerprint'][:16])}…</td>"
            f"<td>{html.escape(c['validation_status'])}</td>"
            f"</tr>"
            for c in ckpts
        )

        # Tensor transitions rows
        t_rows = "".join(
            f"<tr>"
            f"<td class='mono'>{html.escape(t['scope'])}</td>"
            f"<td class='mono'>{html.escape(t['source_label'])} → {html.escape(t['target_label'])}</td>"
            f"<td>{html.escape(t['transformation_type'])}</td>"
            f"<td class='mono'>{_fmt_num(t.get('l2_distance'), 5)}</td>"
            f"<td class='mono'>{_fmt_num(t.get('relative_l2_distance'), 6)}</td>"
            f"<td class='mono'>{_fmt_num(t.get('global_cosine_similarity'), 6)}</td>"
            f"<td class='mono'>{float(t.get('modified_parameter_pct', 0.0)):.2f}%</td>"
            f"<td class='mono'>{_fmt_num(t.get('sparsity_delta'), 5)}</td>"
            f"</tr>"
            for t in t_trans
        )

        # Representation CKA rows
        rep_html = "<p class='muted'>Representation evolution matrix not available.</p>"
        if rep and rep.get("evolution_matrix") and rep["evolution_matrix"].get("rows"):
            lids = rep["evolution_matrix"]["layer_indices"]
            hdrs = "".join(f"<th class='mono'>L{lid}</th>" for lid in lids)
            r_rows = []
            for r in rep["evolution_matrix"]["rows"]:
                cells = "".join(
                    f"<td class='mono'>{f'{v:.4f}' if isinstance(v, (int, float)) else '—'}</td>"
                    for v in r.get("layer_cka", [])
                )
                r_rows.append(
                    f"<tr><td class='mono'>{html.escape(r['version_label'])}</td>"
                    f"<td class='mono'>{_fmt_num(r.get('mean_cka_to_base'), 4)}</td>"
                    f"<td class='mono'>{_fmt_num(r.get('representation_drift'), 4)}</td>{cells}</tr>"
                )
            rep_html = (
                f"<table><thead><tr><th>Version</th><th>Mean CKA</th><th>Drift</th>{hdrs}</tr></thead>"
                f"<tbody>{''.join(r_rows)}</tbody></table>"
            )

        # Regression origin rows
        reg_rows = "".join(
            f"<tr>"
            f"<td class='mono'>{html.escape(ev['previous_label'])} → {html.escape(ev['first_affected_label'])}</td>"
            f"<td>{html.escape(ev['transformation_type'])}</td>"
            f"<td class='mono'>{html.escape(ev['affected_category'])}</td>"
            f"<td class='mono'>{_fmt_pct(ev['pass_rate_before'])} → {_fmt_pct(ev['pass_rate_after'])}</td>"
            f"<td class='mono'>{html.escape(', '.join(ev.get('failed_test_case_ids', [])) or '—')}</td>"
            f"<td class='mono'>{html.escape(ev['epistemic_status'])}</td>"
            f"</tr>"
            for ev in regs
        )

        insight_items = "".join(
            f"<li><strong>[{html.escape(ins['epistemic_status'])}]</strong> {html.escape(ins['statement'])}</li>"
            for ins in insights
        )
        warn_items = "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
        branch_items = "".join(
            f"<li class='mono'>{html.escape(b['branch_id'])}: {' → '.join(html.escape(lbl) for lbl in b['version_labels'])}</li>"
            for b in graph.get("branches", [])
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>NEURAL DIFF Evolution Report — {html.escape(proj.get('name', 'Project'))}</title>
<style>
  body {{ background:#09090b; color:#f4f4f5; font-family:Inter,-apple-system,sans-serif; margin:0; padding:32px; line-height:1.5; }}
  .container {{ max-width:1120px; margin:0 auto; }}
  h1, h2, h3 {{ font-weight:600; letter-spacing:-0.02em; }}
  h2 {{ margin-top:28px; padding-bottom:6px; border-bottom:1px solid rgba(255,255,255,0.09); font-size:16px; }}
  .card {{ background:#121216; border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:18px; margin-bottom:16px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin:16px 0; }}
  .kpi {{ background:#121216; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:12px 14px; }}
  .kpi .lbl {{ color:#71717a; font-size:11px; text-transform:uppercase; letter-spacing:0.04em; }}
  .kpi .val {{ font-size:20px; font-weight:600; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; margin-top:8px; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid rgba(255,255,255,0.06); }}
  th {{ color:#a1a1aa; font-weight:500; }}
  .mono {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:12px; }}
  .muted {{ color:#71717a; font-size:12.5px; }}
</style>
</head>
<body>
<div class="container">
  <h1>NEURAL DIFF · Neural Evolution Report</h1>
  <p class="muted">Project: <strong>{html.escape(proj.get('name', ''))}</strong> (<span class="mono">{html.escape(proj.get('project_id', ''))}</span>)
  · Run <span class="mono">{html.escape(run_bundle.get('run_id', ''))}</span>
  · Cache Key <span class="mono">{html.escape(str(run_bundle.get('cache_key', ''))[:16])}…</span></p>

  <div class="kpis">
    <div class="kpi"><div class="lbl">Checkpoints</div><div class="val">{ov.get('total_checkpoints', len(ckpts))}</div></div>
    <div class="kpi"><div class="lbl">Transformations</div><div class="val">{ov.get('total_transformations', 0)}</div></div>
    <div class="kpi"><div class="lbl">DAG Branches</div><div class="val">{ov.get('branch_count', 0)}</div></div>
    <div class="kpi"><div class="lbl">Detected Regressions</div><div class="val">{ov.get('detected_regressions', len(regs))}</div></div>
    <div class="kpi"><div class="lbl">Peak Cumulative L2</div><div class="val mono">{_fmt_num(ov.get('peak_cumulative_l2'), 4)}</div></div>
  </div>

  <h2>1. Evidence-Backed Evolution Insights</h2>
  <div class="card"><ul>{insight_items or '<li>No threshold-exceeding insights triggered.</li>'}</ul></div>

  <h2>2. Evolution Lineages & Checkpoint Registry</h2>
  <div class="card">
    <ul>{branch_items}</ul>
    <table>
      <thead><tr><th>Version</th><th>Transformation</th><th>Params</th><th>Layers</th><th>Dtypes</th><th>SHA-256</th><th>Status</th></tr></thead>
      <tbody>{ckpt_rows}</tbody>
    </table>
  </div>

  <h2>3. Cumulative Weight Distance (Direct Base → Current) & Multi-Gen Transitions</h2>
  <div class="card">
    {svg_weight}
    <table>
      <thead><tr><th>Scope</th><th>Transition</th><th>Type</th><th>L2 Dist</th><th>Rel L2</th><th>Cosine</th><th>Modified %</th><th>Sparsity Δ</th></tr></thead>
      <tbody>{t_rows}</tbody>
    </table>
  </div>

  <h2>4. Behavioral Evolution Timeline (Deterministic Suite)</h2>
  <div class="card">
    {svg_beh}
    <p class="muted">{html.escape(beh.get('statistical_disclosure', '')) if beh else ''}</p>
  </div>

  <h2>5. Representation Evolution Matrix (Centered Linear CKA vs Base)</h2>
  <div class="card">{rep_html}</div>

  <h2>6. Automatic Regression Origin Analysis (OBSERVED / CORRELATED / CAUSALLY VERIFIED)</h2>
  <div class="card">
    <table>
      <thead><tr><th>Origin Transition</th><th>Transformation</th><th>Category</th><th>Pass Rate</th><th>Failed Cases</th><th>Epistemic Status</th></tr></thead>
      <tbody>{reg_rows or '<tr><td colspan="6" class="muted">No regressions exceeded configured thresholds.</td></tr>'}</tbody>
    </table>
  </div>

  <h2>7. Reproducibility Provenance & Limitations</h2>
  <div class="card">
    <p class="mono">Device: {html.escape(str(prov.get('selected_device', 'cpu')))} · Seed: {prov.get('random_seed', 42)} · Dataset SHA-256: {html.escape(str(prov.get('evaluation_dataset_fingerprint') or 'N/A')[:24])}</p>
    <p class="muted">Epistemic Policy: Direct tensor and rubric pass-rate diffs are OBSERVED. Co-occurring layer weight or CKA shifts during a behavioral drop are strictly CORRELATED unless an interventional residual-stream patch recovers the clean margin (CAUSALLY VERIFIED).</p>
    {f'<ul>{warn_items}</ul>' if warn_items else ''}
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
        warnings = run_bundle.get("warnings", [])

        pdf = MinimalPDFBuilder(title=f"NEURAL DIFF Evolution - {proj.get('name', 'Report')}")
        pdf.add_banner(
            f"NEURAL DIFF  |  Neural Evolution Report: {proj.get('name', '')}",
            f"Run ID: {run_bundle.get('run_id', '')}  |  Checkpoints: {len(ckpts)}  |  Branches: {ov.get('branch_count', 0)}  |  Regressions: {len(regs)}",
        )

        # 1. Insights
        pdf.add_section_heading("1. Evidence-Backed Evolution Insights")
        if insights:
            for ins in insights:
                pdf.add_paragraph(f"[{ins['epistemic_status']}] {ins['statement']}")
        else:
            pdf.add_paragraph("No threshold-exceeding insights recorded for this run.")

        # 2. Checkpoint Registry
        pdf.add_section_heading("2. Checkpoint Registry & Provenance")
        pdf.add_table(
            ["Version", "Transformation", "Params", "Layers", "SHA-256", "Status"],
            [
                [
                    c["version_label"],
                    c["transformation_type"],
                    f"{c['parameter_count']:,}",
                    str(c.get("layer_count") or "-"),
                    c["sha256_fingerprint"][:14],
                    c["validation_status"],
                ]
                for c in ckpts
            ],
            col_weights=[1.4, 1.3, 1.0, 0.6, 1.3, 1.1],
        )

        # 3. Cumulative Weight Chart + Transition Table
        pdf.add_section_heading("3. Multi-Generation & Direct Cumulative Weight Evolution")
        base_trans = [t for t in t_trans if t.get("scope") == "base_to_current"]
        if base_trans:
            pdf.add_horizontal_bar_chart(
                "Direct Base-to-Current L2 Weight Distance (Computed Directly vs Root)",
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
                    t["scope"],
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

        # 4. Behavioral Evolution
        pdf.add_section_heading("4. Behavioral Capability Evolution")
        if beh and beh.get("timeline"):
            pdf.add_horizontal_bar_chart(
                f"Deterministic Pass Rate Timeline (Suite: {beh.get('suite_id')}@{beh.get('suite_version')}, n={beh.get('sample_size')})",
                [
                    (
                        item["version_label"],
                        float(item["pass_rate"]),
                        f"{float(item['pass_rate']) * 100.0:.1f}%",
                    )
                    for item in beh["timeline"]
                ],
                max_value=1.0,
            )
            pdf.add_paragraph(beh.get("statistical_disclosure", ""))
        else:
            pdf.add_paragraph("Behavioral evaluation module not executed.")

        # 5. Representation CKA
        pdf.add_section_heading("5. Representation Evolution (Centered Linear CKA vs Base)")
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
        pdf.add_section_heading("6. Automatic Regression Origin Analysis")
        if regs:
            pdf.add_table(
                ["Origin Transition", "Type", "Category", "Drop", "Epistemic", "Cases"],
                [
                    [
                        f"{e['previous_label']}->{e['first_affected_label']}",
                        e["transformation_type"],
                        e["affected_category"],
                        _fmt_pct(e["score_drop"]),
                        e["epistemic_status"],
                        ",".join(e.get("failed_test_case_ids", [])) or "-",
                    ]
                    for e in regs
                ],
                col_weights=[1.8, 1.1, 1.1, 0.7, 1.2, 1.2],
            )
        else:
            pdf.add_paragraph("No regressions exceeded configured thresholds.")

        # 7. Provenance & Epistemic Notice
        pdf.add_section_heading("7. Experiment Reproducibility & Epistemic Disclosures")
        pdf.add_paragraph(
            f"Content Cache Key: {run_bundle.get('cache_key', '')}  |  Seed: {prov.get('random_seed', 42)}  |  "
            f"Device: {prov.get('selected_device', 'cpu')}  |  Dataset SHA-256: {str(prov.get('evaluation_dataset_fingerprint') or 'N/A')[:24]}",
            mono=True,
            font_size=8.0,
        )
        pdf.add_paragraph(
            "Epistemic Boundary: Direct weight norms and deterministic rubric pass rates are OBSERVED. "
            "Layer weight changes and Linear CKA shifts coinciding with behavioral regressions are strictly CORRELATED "
            "and never treated as standalone proof of causation unless interventional activation patching is verified."
        )
        for w in warnings:
            pdf.add_paragraph(f"Warning: {w}", font_size=8.2)

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
        if fmt_l == "html":
            p = out / f"{rid}.html"
            p.write_text(self.to_html(run_bundle), encoding="utf-8")
            return p
        if fmt_l == "pdf":
            p = out / f"{rid}.pdf"
            p.write_bytes(self.to_pdf(run_bundle))
            return p
        raise ValueError(f"Unsupported export format '{fmt}'. Expected json, csv, html, or pdf.")
