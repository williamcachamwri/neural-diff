"""Report Engine (Module H) — JSON, Markdown, and standalone SVG-equipped HTML reports."""

from __future__ import annotations

import datetime as dt
import html
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import transformers

import safetensors
from core import __version__
from core.activation_analysis.comparator import ActivationDiffReport
from core.behavioral_analysis.runner import BehavioralDiffReport
from core.regression_engine.engine import RegressionReport
from core.report_engine.visuals import (
    human_label,
    md_bar,
    svg_dual_line_chart,
    svg_lollipop_chart,
    svg_matrix_heatmap,
    svg_radar_polygon,
    svg_segmented_ratio_bar,
)
from core.structural_analysis.diff import StructuralDiffReport
from core.structural_analysis.neuron_attention import NeuronAttentionReport
from core.tensor_engine.comparator import TensorDiffReport
from core.types import EpistemicClaim, EpistemicStatus, ModelMetadata


@dataclass
class ComparisonBundle:
    """Complete execution artifact assembled across selected NEURAL DIFF modules."""

    comparison_id: str
    created_at_utc: str
    experiment_type: str  # "REAL MODEL EXPERIMENT" | "SYNTHETIC FIXTURE EXPERIMENT"
    model_a: ModelMetadata
    model_b: ModelMetadata
    selected_modules: list[str]
    structural: StructuralDiffReport | None = None
    tensor_diff: TensorDiffReport | None = None
    neuron_attention: NeuronAttentionReport | None = None
    activation: ActivationDiffReport | None = None
    behavioral: BehavioralDiffReport | None = None
    regression: RegressionReport | None = None
    execution_telemetry: dict[str, Any] = field(default_factory=dict)
    reproducibility: dict[str, Any] = field(default_factory=dict)
    epistemic_claims: list[EpistemicClaim] = field(default_factory=list)

    def build_epistemic_ledger(self) -> list[EpistemicClaim]:
        """Construct explicit epistemic status ledger distinguishing OBSERVED, INFERRED, CORRELATED, etc."""
        if self.epistemic_claims:
            return self.epistemic_claims

        claims: list[EpistemicClaim] = []
        if self.structural is not None:
            claims.append(
                EpistemicClaim(
                    statement=(
                        f"Model A ({self.model_a.total_parameters:,} params) vs Model B "
                        f"({self.model_b.total_parameters:,} params) share {len(self.structural.shared_tensors)} tensor keys."
                    ),
                    epistemic_status=EpistemicStatus.OBSERVED,
                    evidence_source="Module C — StructuralDiffEngine",
                    metric_values={
                        "shared_tensors": len(self.structural.shared_tensors),
                        "added": len(self.structural.added_tensors),
                        "removed": len(self.structural.removed_tensors),
                    },
                )
            )
            if self.structural.inferred_renames:
                claims.append(
                    EpistemicClaim(
                        statement=(
                            f"{len(self.structural.inferred_renames)} unmatched tensor pair(s) were matched as "
                            "probable renames via identical shape, role, and high cosine similarity."
                        ),
                        epistemic_status=EpistemicStatus.INFERRED,
                        evidence_source="Module C — Rename heuristic",
                        metric_values={"inferred_rename_count": len(self.structural.inferred_renames)},
                    )
                )

        if self.tensor_diff is not None:
            claims.append(
                EpistemicClaim(
                    statement=(
                        f"{self.tensor_diff.global_changed_pct:.4f}% of aligned parameters exceeded numerical tolerance "
                        f"(mean cosine={self.tensor_diff.global_mean_cosine}, max abs diff={self.tensor_diff.global_max_abs_diff})."
                    ),
                    epistemic_status=EpistemicStatus.OBSERVED,
                    evidence_source="Module B — TensorDiffEngine",
                    metric_values={
                        "changed_pct": self.tensor_diff.global_changed_pct,
                        "mean_cosine": self.tensor_diff.global_mean_cosine,
                    },
                )
            )

        if self.activation is not None and self.behavioral is not None:
            claims.append(
                EpistemicClaim(
                    statement=(
                        "Internal hidden-state and MLP activation shifts co-occurred alongside observed behavioral "
                        "score deltas; without targeted activation patching, internal-to-behavioral linkage is correlational."
                    ),
                    epistemic_status=EpistemicStatus.CORRELATED,
                    evidence_source="Module E + Module F joint comparison",
                    metric_values={
                        "pooled_final_cka": self.activation.cross_prompt_pooled_cka_final_layer,
                        "behavioral_mean_score_delta": self.behavioral.overall_statistics.mean_score_delta,
                    },
                )
            )

        claims.append(
            EpistemicClaim(
                statement=(
                    "Causal mediation of individual neurons or circuits via counterfactual activation patching "
                    "was not executed in this observational diff run."
                ),
                epistemic_status=EpistemicStatus.NOT_VERIFIED,
                evidence_source="Epistemic boundary declaration (Causal Patching Gate)",
                metric_values={"causally_verified_circuits": 0},
            )
        )
        return claims

    def to_dict(self, *, include_full_tensor_records: bool = True) -> dict[str, Any]:
        ledger = self.build_epistemic_ledger()
        return {
            "schema_version": "1.0.0",
            "application": {"name": "NEURAL DIFF", "version": __version__},
            "comparison_id": self.comparison_id,
            "created_at_utc": self.created_at_utc,
            "experiment_type": self.experiment_type,
            "selected_modules": self.selected_modules,
            "model_a": self.model_a.to_dict(),
            "model_b": self.model_b.to_dict(),
            "epistemic_ledger": [c.to_dict() for c in ledger],
            "structural": self.structural.to_dict() if self.structural is not None else None,
            "tensor_diff": (
                self.tensor_diff.to_dict(include_records=include_full_tensor_records)
                if self.tensor_diff is not None
                else None
            ),
            "neuron_attention": (
                self.neuron_attention.to_dict() if self.neuron_attention is not None else None
            ),
            "activation": self.activation.to_dict() if self.activation is not None else None,
            "behavioral": self.behavioral.to_dict() if self.behavioral is not None else None,
            "regression": self.regression.to_dict() if self.regression is not None else None,
            "execution_telemetry": self.execution_telemetry,
            "reproducibility": self.reproducibility or default_reproducibility_metadata(),
        }


def default_reproducibility_metadata() -> dict[str, Any]:
    return {
        "captured_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "safetensors_version": safetensors.__version__,
        "neural_diff_version": __version__,
    }


def _fmt_f(val: float | None, digits: int = 6) -> str:
    if val is None:
        return "—"
    return f"{val:.{digits}f}"


class ReportEngine:
    """Exports `ComparisonBundle` measurements into strict JSON, Markdown, and SVG-rich HTML."""

    def to_json(self, bundle: ComparisonBundle, *, include_full_tensor_records: bool = True) -> str:
        payload = bundle.to_dict(include_full_tensor_records=include_full_tensor_records)
        return json.dumps(payload, indent=2, allow_nan=False)

    def to_markdown(self, bundle: ComparisonBundle) -> str:
        ledger = bundle.build_epistemic_ledger()
        verdict_raw = bundle.regression.verdict if bundle.regression else "NOT_RUN"
        verdict_lbl = human_label(verdict_raw)

        chg_pct_lbl = (
            f"{bundle.tensor_diff.global_changed_pct:.2f}% params Δ"
            if bundle.tensor_diff is not None
            else "Structural only"
        )
        cka_lbl = (
            f"Final CKA {bundle.activation.cross_prompt_pooled_cka_final_layer:.4f}"
            if (
                bundle.activation is not None
                and bundle.activation.cross_prompt_pooled_cka_final_layer is not None
            )
            else "No hooks"
        )
        beh_lbl = (
            f"Pass {bundle.behavioral.overall_statistics.pass_rate_a:.0%} -> "
            f"{bundle.behavioral.overall_statistics.pass_rate_b:.0%}"
            if bundle.behavioral is not None
            else "No behavioral run"
        )

        lines: list[str] = [
            f"# NEURAL DIFF Report — `{bundle.comparison_id}`",
            "",
            "| Baseline (A) | Candidate (B) | Mode | Gate | Generated (UTC) |",
            "| :--- | :--- | :--- | :--- | :--- |",
            (
                f"| `{bundle.model_a.model_id}` ({bundle.model_a.total_parameters:,}) "
                f"| `{bundle.model_b.model_id}` ({bundle.model_b.total_parameters:,}) "
                f"| {human_label(bundle.experiment_type)} "
                f"| **{verdict_lbl}** "
                f"| `{bundle.created_at_utc[:19]}` |"
            ),
            "",
            "```mermaid",
            "flowchart LR",
            f'  A["Model A\\n{bundle.model_a.model_id.split("/")[-1]}"] --> W["Weights\\n{chg_pct_lbl}"]',
            f'  B["Model B\\n{bundle.model_b.model_id.split("/")[-1]}"] --> W',
            f'  W --> R["Representations\\n{cka_lbl}"]',
            f'  R --> E["Evaluation\\n{beh_lbl}"]',
            f'  E --> G["Gate\\n{verdict_lbl}"]',
            "```",
            "",
            "## Key Findings",
            "",
            "| Evidence | Observation | Source |",
            "| :--- | :--- | :--- |",
        ]
        for c in ledger:
            lines.append(
                f"| {human_label(c.epistemic_status.value)} | {c.statement} | {c.evidence_source} |"
            )

        if bundle.structural is not None:
            st = bundle.structural
            lines.extend(
                [
                    "",
                    "## Architecture & Alignment",
                    "",
                    "| Shared Tensors | Added | Removed | Renamed (Inferred) | Shape Δ | Dtype Δ | Param Δ |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                    (
                        f"| {len(st.shared_tensors)} | {len(st.added_tensors)} | {len(st.removed_tensors)} "
                        f"| {len(st.inferred_renames)} | {len(st.shape_changes)} "
                        f"| {len(st.dtype_changes)} | `{st.delta_parameters:+,}` |"
                    ),
                ]
            )

        if bundle.tensor_diff is not None:
            td = bundle.tensor_diff
            lines.extend(
                [
                    "",
                    "## Weight Perturbation",
                    "",
                    "| Compared | Modified | Changed Params | Mean Cosine | Mean Rel L2 | Max \\|Δ\\| |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: |",
                    (
                        f"| {td.compared_tensors} | {td.modified_tensors} "
                        f"| `{td.global_changed_pct:.2f}%` ({td.total_changed_parameters:,}) "
                        f"| `{_fmt_f(td.global_mean_cosine, 6)}` "
                        f"| `{_fmt_f(td.global_mean_relative_l2, 6)}` "
                        f"| `{_fmt_f(td.global_max_abs_diff, 5)}` |"
                    ),
                ]
            )

            if td.layer_summaries:
                max_layer_l2 = max(
                    (ls.mean_relative_l2 or 0.0 for ls in td.layer_summaries), default=1.0
                ) or 1.0
                lines.extend(
                    [
                        "",
                        "### Layer Perturbation Profile",
                        "",
                        "| Layer | Mean Rel L2 | Mean Cosine | Changed % | Magnitude |",
                        "| :--- | ---: | ---: | ---: | :--- |",
                    ]
                )
                for ls in td.layer_summaries[:28]:
                    bar = md_bar(ls.mean_relative_l2, max_layer_l2, width=14)
                    lines.append(
                        f"| `L{ls.layer_index}` | `{_fmt_f(ls.mean_relative_l2, 5)}` "
                        f"| `{_fmt_f(ls.mean_cosine_similarity, 6)}` | `{ls.changed_pct:.1f}%` | `{bar}` |"
                    )

            rec_map = {r.name: r for r in td.records}
            top_recs = [
                rec_map[n]
                for n in td.top_diverged_names[:10]
                if n in rec_map and rec_map[n].metrics is not None
            ]
            if top_recs:
                max_rec_l2 = max(
                    (r.metrics.relative_l2 or 0.0 for r in top_recs if r.metrics), default=1.0
                ) or 1.0
                lines.extend(
                    [
                        "",
                        "### Top Diverged Tensors",
                        "",
                        "| Tensor | Role | Cosine | Rel L2 | Max \\|Δ\\| | Profile |",
                        "| :--- | :--- | ---: | ---: | ---: | :--- |",
                    ]
                )
                for rec in top_recs:
                    m = rec.metrics
                    if m is None:
                        continue
                    lines.append(
                        f"| `{rec.name}` | `{rec.canonical_role.value}` "
                        f"| `{_fmt_f(m.cosine_similarity, 6)}` | `{_fmt_f(m.relative_l2, 5)}` "
                        f"| `{_fmt_f(m.max_abs_diff, 5)}` | `{md_bar(m.relative_l2, max_rec_l2, 10)}` |"
                    )

        if bundle.activation is not None:
            act = bundle.activation
            lines.extend(
                [
                    "",
                    "## Representation Geometry",
                    "",
                    "| Prompts | Final-Layer Linear CKA | Next-Token Logit JS |",
                    "| ---: | ---: | ---: |",
                    (
                        f"| {act.prompt_count} "
                        f"| `{_fmt_f(act.cross_prompt_pooled_cka_final_layer, 5)}` "
                        f"| `{_fmt_f(act.mean_final_token_js_divergence, 5)}` |"
                    ),
                ]
            )
            lkeys = sorted(act.mean_hidden_cosine_by_layer.keys(), key=int)
            if lkeys:
                lines.extend(
                    [
                        "",
                        "| Layer | Linear CKA | Hidden Cosine | CKA Alignment |",
                        "| :--- | ---: | ---: | :--- |",
                    ]
                )
                for lk in lkeys[:28]:
                    cka_v = act.mean_hidden_cka_by_layer.get(lk)
                    cos_v = act.mean_hidden_cosine_by_layer.get(lk)
                    lines.append(
                        f"| `L{lk}` | `{_fmt_f(cka_v, 5)}` | `{_fmt_f(cos_v, 5)}` "
                        f"| `{md_bar(cka_v, 1.0, 14)}` |"
                    )

        if bundle.behavioral is not None:
            beh = bundle.behavioral
            st_ov = beh.overall_statistics
            lines.extend(
                [
                    "",
                    f"## Behavioral Evaluation (`{beh.suite_id}@{beh.suite_version}`)",
                    "",
                    "| Cases | Pass A → B | Mean Score A → B | 95% Bootstrap CI | McNemar p |",
                    "| ---: | :--- | :--- | :--- | ---: |",
                    (
                        f"| {st_ov.n_cases} | `{st_ov.pass_rate_a:.1%} → {st_ov.pass_rate_b:.1%}` "
                        f"(`{st_ov.pass_rate_delta:+.1%}`) "
                        f"| `{st_ov.mean_score_a:.3f} → {st_ov.mean_score_b:.3f}` (`{st_ov.mean_score_delta:+.3f}`) "
                        f"| `[{st_ov.bootstrap_ci95_low:+.3f}, {st_ov.bootstrap_ci95_high:+.3f}]` "
                        f"| `{st_ov.mcnemar_p_value:.4f}` |"
                    ),
                    "",
                    "| Suite | Cases | Score A | Score B | Delta | Candidate Profile |",
                    "| :--- | ---: | ---: | ---: | ---: | :--- |",
                ]
            )
            for cs in beh.category_summaries:
                lines.append(
                    f"| {human_label(cs.category)} | {cs.case_count} | `{cs.mean_score_a:.3f}` "
                    f"| `{cs.mean_score_b:.3f}` | `{cs.mean_score_delta:+.3f}` "
                    f"| `{md_bar(cs.mean_score_b, 1.0, 12)}` |"
                )

        if bundle.regression is not None and bundle.regression.findings:
            reg = bundle.regression
            lines.extend(
                [
                    "",
                    f"## Regression Gate — {verdict_lbl} (`{reg.thresholds.policy_name}`)",
                    "",
                    "| Severity | Evidence | Finding | Detail |",
                    "| :--- | :--- | :--- | :--- |",
                ]
            )
            for f in reg.findings:
                lines.append(
                    f"| {human_label(f.severity.value)} | {human_label(f.epistemic_status.value)} "
                    f"| **{f.title}** | {f.description} |"
                )

        repro = bundle.reproducibility or default_reproducibility_metadata()
        lines.extend(
            [
                "",
                "## Provenance",
                "",
                "| Parameter | Value |",
                "| :--- | :--- |",
                f"| Platform | `{repro.get('platform', '—')}` |",
                f"| Python / PyTorch | `{repro.get('python_version', '—')}` / `{repro.get('torch_version', '—')}` |",
                f"| Transformers / Safetensors | `{repro.get('transformers_version', '—')}` / `{repro.get('safetensors_version', '—')}` |",
                f"| Executed Modules | `{', '.join(bundle.selected_modules)}` |",
                "",
            ]
        )
        return "\n".join(lines)

    def to_html(self, bundle: ComparisonBundle) -> str:
        """Render a minimalist Architectural Monochrome HTML report with 5 embedded SVG visuals."""
        ledger = bundle.build_epistemic_ledger()
        esc = html.escape

        ratio_svg = self._render_param_composition_svg(bundle.tensor_diff)
        heatmap_svg = self._render_layer_heatmap_svg(bundle.tensor_diff)
        lollipop_svg = self._render_top_tensors_lollipop_svg(bundle.tensor_diff)
        act_curve_svg = self._render_activation_curve_svg(bundle.activation)
        radar_svg, beh_bars_svg = self._render_behavioral_visuals_svg(bundle.behavioral)

        def _dot_for_status(st_val: str) -> str:
            if st_val == "OBSERVED":
                return "#4ade80"
            if st_val in ("CORRELATED", "INFERRED"):
                return "#d6c5a9"
            return "#71717a"

        ledger_rows = "".join(
            f"<tr>"
            f"<td class='nowrap'><span class='dot' style='background:{_dot_for_status(c.epistemic_status.value)}'></span>"
            f"{esc(human_label(c.epistemic_status.value))}</td>"
            f"<td>{esc(c.statement)}</td>"
            f"<td class='muted mono'>{esc(c.evidence_source)}</td>"
            f"</tr>"
            for c in ledger
        )

        tensor_rows = ""
        if bundle.tensor_diff is not None:
            rec_map = {r.name: r for r in bundle.tensor_diff.records}
            target_names = bundle.tensor_diff.top_diverged_names[:14] or [
                r.name for r in bundle.tensor_diff.records[:14]
            ]
            for tname in target_names:
                r = rec_map.get(tname)
                if r is None:
                    continue
                m = r.metrics
                tensor_rows += (
                    f"<tr><td class='mono'>{esc(r.name)}</td>"
                    f"<td class='muted'>{esc(r.canonical_role.value)}</td>"
                    f"<td class='mono muted'>{esc('×'.join(str(d) for d in (r.shape_a or r.shape_b or [])))}</td>"
                    f"<td class='num mono'>{_fmt_f(m.cosine_similarity, 6) if m else '—'}</td>"
                    f"<td class='num mono'>{_fmt_f(m.relative_l2, 6) if m else '—'}</td>"
                    f"<td class='num mono'>{_fmt_f(m.max_abs_diff, 5) if m else '—'}</td>"
                    f"<td class='num mono'>{f'{m.changed_pct:.2f}%' if m else '—'}</td></tr>"
                )

        beh_case_rows = ""
        if bundle.behavioral is not None:
            for pc in bundle.behavioral.cases[:16]:
                d_col = "#4ade80" if pc.score_delta > 0.01 else ("#f87171" if pc.score_delta < -0.01 else "#a1a1aa")
                beh_case_rows += (
                    f"<tr><td class='mono'>{esc(pc.case.case_id)}</td>"
                    f"<td class='muted'>{esc(human_label(pc.case.category))}</td>"
                    f"<td class='num mono' style='color:{d_col}'>{pc.eval_a.score:.2f} → {pc.eval_b.score:.2f} ({pc.score_delta:+.2f})</td>"
                    f"<td><div class='prompt'>{esc(pc.case.prompt)}</div>"
                    f"<div class='out-grid'><div><span class='lbl-inline'>A</span> {esc(pc.output_a[:220])}</div>"
                    f"<div><span class='lbl-inline'>B</span> {esc(pc.output_b[:220])}</div></div></td></tr>"
                )

        verdict_raw = bundle.regression.verdict if bundle.regression else "NOT_RUN"
        verdict_col = (
            "#f87171"
            if "FAIL" in verdict_raw
            else ("#4ade80" if verdict_raw == "PASSED" else "#d6c5a9")
        )
        chg_kpi = (
            f"{bundle.tensor_diff.global_changed_pct:.2f}%"
            if bundle.tensor_diff is not None
            else "—"
        )
        cos_kpi = (
            _fmt_f(bundle.tensor_diff.global_mean_cosine, 5)
            if bundle.tensor_diff is not None
            else "—"
        )
        cka_kpi = (
            _fmt_f(bundle.activation.cross_prompt_pooled_cka_final_layer, 4)
            if bundle.activation is not None
            else "—"
        )
        pass_kpi = (
            f"{bundle.behavioral.overall_statistics.pass_rate_a:.0%} → {bundle.behavioral.overall_statistics.pass_rate_b:.0%}"
            if bundle.behavioral is not None
            else "—"
        )

        repro = bundle.reproducibility or default_reproducibility_metadata()

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>NEURAL DIFF — {esc(bundle.comparison_id)}</title>
<style>
  :root {{
    --bg: #09090b; --surface: #121216; --border: rgba(255,255,255,0.08);
    --border-faint: rgba(255,255,255,0.05); --text: #f4f4f5; --secondary: #a1a1aa;
    --muted: #71717a; --champagne: #d6c5a9;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 32px 28px; line-height: 1.5; -webkit-font-smoothing: antialiased; }}
  .container {{ max-width: 1180px; margin: 0 auto; }}
  header {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap; padding-bottom: 18px; border-bottom: 1px solid var(--border); margin-bottom: 20px; }}
  h1 {{ margin: 0; font-size: 20px; font-weight: 600; letter-spacing: -0.02em; }}
  h2 {{ font-size: 13.5px; font-weight: 600; letter-spacing: -0.01em; color: var(--text); margin: 0 0 12px 0; display: flex; justify-content: space-between; align-items: center; }}
  .sub {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-bottom: 18px; }}
  .kpi {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }}
  .kpi .k-lbl {{ color: var(--muted); font-size: 11px; }}
  .kpi .k-val {{ font-size: 18px; font-weight: 600; margin-top: 4px; font-variant-numeric: tabular-nums; }}
  .panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; margin-bottom: 16px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1.35fr; gap: 16px; align-items: center; }}
  .mono {{ font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 11.5px; }}
  .muted {{ color: var(--muted); }}
  .nowrap {{ white-space: nowrap; }}
  .dot {{ display: inline-block; width: 6px; height: 6px; border-radius: 50%; margin-right: 7px; vertical-align: middle; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--border-faint); text-align: left; vertical-align: top; }}
  th {{ color: var(--muted); font-weight: 500; font-size: 11.5px; border-bottom: 1px solid var(--border); }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .prompt {{ color: var(--text); font-size: 12px; margin-bottom: 5px; }}
  .out-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 11.5px; color: var(--secondary); }}
  .lbl-inline {{ display: inline-block; color: var(--muted); font-family: monospace; margin-right: 4px; }}
  .kv-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; font-size: 12px; }}
  .kv-item {{ padding: 8px 10px; border: 1px solid var(--border-faint); border-radius: 6px; background: rgba(255,255,255,0.015); }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>NEURAL DIFF · {esc(bundle.comparison_id)}</h1>
      <div class="sub mono">{esc(bundle.model_a.model_id)} ({bundle.model_a.total_parameters:,})  →  {esc(bundle.model_b.model_id)} ({bundle.model_b.total_parameters:,})</div>
    </div>
    <div class="mono muted">{esc(human_label(bundle.experiment_type))} · {esc(bundle.created_at_utc[:19])}Z</div>
  </header>

  <div class="kpis">
    <div class="kpi"><div class="k-lbl">Modified Parameters</div><div class="k-val mono">{esc(chg_kpi)}</div></div>
    <div class="kpi"><div class="k-lbl">Mean Weight Cosine</div><div class="k-val mono">{esc(cos_kpi)}</div></div>
    <div class="kpi"><div class="k-lbl">Final-Layer CKA</div><div class="k-val mono">{esc(cka_kpi)}</div></div>
    <div class="kpi"><div class="k-lbl">Pass Rate (A → B)</div><div class="k-val mono">{esc(pass_kpi)}</div></div>
    <div class="kpi"><div class="k-lbl">Gate Verdict</div><div class="k-val mono" style="color:{verdict_col}">{esc(human_label(verdict_raw))}</div></div>
  </div>

  <div class="panel">
    <h2><span>Weight Perturbation Matrix &amp; Parameter Composition</span><span class="muted mono">Relative L2 across transformer depth</span></h2>
    {ratio_svg}
    <div style="margin-top:12px">{heatmap_svg}</div>
  </div>

  <div class="panel">
    <h2><span>Empirical Findings</span><span class="muted mono">{len(ledger)} verified records</span></h2>
    <table><thead><tr><th>Evidence</th><th>Observation</th><th>Source</th></tr></thead><tbody>{ledger_rows}</tbody></table>
  </div>

  <div class="panel">
    <h2><span>Peak Diverged Tensors</span><span class="muted mono">Ranked by float64 Relative L2</span></h2>
    {lollipop_svg}
    <table style="margin-top:12px"><thead><tr><th>Tensor</th><th>Role</th><th>Shape</th><th class="num">Cosine</th><th class="num">Rel L2</th><th class="num">Max |Δ|</th><th class="num">Changed</th></tr></thead><tbody>{tensor_rows or "<tr><td colspan='7' class='muted'>Tensor diff module not executed.</td></tr>"}</tbody></table>
  </div>

  {f'<div class="panel"><h2><span>Layer Representation Trajectory</span><span class="muted mono">Centered Linear CKA vs Hidden Cosine</span></h2>{act_curve_svg}</div>' if act_curve_svg else ''}

  <div class="panel">
    <h2><span>Behavioral Capability Profile</span><span class="muted mono">Deterministic programmatic suites</span></h2>
    <div class="grid-2">
      <div>{radar_svg}</div>
      <div>{beh_bars_svg}</div>
    </div>
    {f'<table style="margin-top:14px"><thead><tr><th>Case</th><th>Suite</th><th class="num">Score A → B</th><th>Prompt &amp; Paired Responses</th></tr></thead><tbody>{beh_case_rows}</tbody></table>' if beh_case_rows else ''}
  </div>

  <div class="panel">
    <h2><span>Provenance &amp; Runtime Environment</span><span class="muted mono">{esc(", ".join(bundle.selected_modules))}</span></h2>
    <div class="kv-grid mono">
      <div class="kv-item"><div class="muted">Platform</div><div>{esc(str(repro.get("platform", "—"))[:42])}</div></div>
      <div class="kv-item"><div class="muted">Python / PyTorch</div><div>{esc(str(repro.get("python_version", "—")))} · {esc(str(repro.get("torch_version", "—")))}</div></div>
      <div class="kv-item"><div class="muted">Transformers / Safetensors</div><div>{esc(str(repro.get("transformers_version", "—")))} · {esc(str(repro.get("safetensors_version", "—")))}</div></div>
    </div>
  </div>
</div>
</body>
</html>"""

    def export_all(self, bundle: ComparisonBundle, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"{bundle.comparison_id}.json"
        md_path = out / f"{bundle.comparison_id}.md"
        html_path = out / f"{bundle.comparison_id}.html"

        json_path.write_text(self.to_json(bundle), encoding="utf-8")
        md_path.write_text(self.to_markdown(bundle), encoding="utf-8")
        html_path.write_text(self.to_html(bundle), encoding="utf-8")
        return {"json": json_path, "markdown": md_path, "html": html_path}

    @staticmethod
    def _render_param_composition_svg(tensor_diff: TensorDiffReport | None) -> str:
        if tensor_diff is None or tensor_diff.compared_tensors <= 0:
            return ""
        return svg_segmented_ratio_bar(
            [
                ("Modified", float(tensor_diff.modified_tensors), "#d6c5a9"),
                ("Within tol", float(tensor_diff.within_tolerance_tensors), "#71717a"),
                ("Identical", float(tensor_diff.identical_tensors), "rgba(255,255,255,0.22)"),
            ]
        )

    @staticmethod
    def _render_layer_heatmap_svg(tensor_diff: TensorDiffReport | None) -> str:
        if tensor_diff is None or not tensor_diff.layer_summaries:
            return "<div class='muted'>No layered tensor diff measurements available.</div>"

        roles = ["attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down", "norm"]
        layers = tensor_diff.layer_summaries
        col_labels = [f"L{ls.layer_index}" for ls in layers]
        matrix: list[list[float | None]] = []
        for rname in roles:
            matrix.append(
                [(ls.role_breakdown.get(rname) or {}).get("mean_relative_l2") for ls in layers]
            )
        return svg_matrix_heatmap(row_labels=roles, col_labels=col_labels, matrix=matrix)

    @staticmethod
    def _render_top_tensors_lollipop_svg(tensor_diff: TensorDiffReport | None) -> str:
        if tensor_diff is None or not tensor_diff.records:
            return ""
        rec_map = {r.name: r for r in tensor_diff.records}
        names = tensor_diff.top_diverged_names[:10]
        items: list[tuple[str, float, str]] = []
        for n in names:
            rec = rec_map.get(n)
            if rec and rec.metrics and rec.metrics.relative_l2 is not None:
                items.append(
                    (rec.name, float(rec.metrics.relative_l2), f"Rel L2 {rec.metrics.relative_l2:.5f}")
                )
        return svg_lollipop_chart(items)

    @staticmethod
    def _render_activation_curve_svg(activation: ActivationDiffReport | None) -> str:
        if activation is None or not activation.mean_hidden_cosine_by_layer:
            return ""
        lkeys = sorted(activation.mean_hidden_cosine_by_layer.keys(), key=int)
        xlbls = [f"L{k}" for k in lkeys]
        cka_vals = [activation.mean_hidden_cka_by_layer.get(k) for k in lkeys]
        cos_vals = [activation.mean_hidden_cosine_by_layer.get(k) for k in lkeys]
        valid = [v for v in [*cka_vals, *cos_vals] if v is not None]
        lo = max(0.0, min(valid) - 0.05) if valid else 0.0
        return svg_dual_line_chart(
            x_labels=xlbls,
            series_primary=cka_vals,
            label_primary="Centered Linear CKA",
            series_secondary=cos_vals,
            label_secondary="Hidden Cosine",
            y_min=lo,
            y_max=1.0,
        )

    @staticmethod
    def _render_behavioral_visuals_svg(
        behavioral: BehavioralDiffReport | None,
    ) -> tuple[str, str]:
        if behavioral is None or not behavioral.category_summaries:
            return ("", "<div class='muted'>Behavioral diff module not executed.</div>")

        cats = behavioral.category_summaries
        axes = [human_label(c.category) for c in cats]
        vals_a = [float(c.mean_score_a) for c in cats]
        vals_b = [float(c.mean_score_b) for c in cats]
        radar = svg_radar_polygon(
            axes=axes,
            series=[
                ("Model A", vals_a, "#71717a"),
                ("Model B", vals_b, "#d6c5a9"),
            ],
            size=270,
        )

        row_h = 26
        left_pad = 148
        bar_max = 260
        width = left_pad + bar_max + 118
        height = len(cats) * row_h + 16
        elems: list[str] = []
        for idx, cs in enumerate(cats):
            y = 10 + idx * row_h
            wa = max(2, int(cs.mean_score_a * bar_max))
            wb = max(2, int(cs.mean_score_b * bar_max))
            col_b = "#f87171" if cs.mean_score_delta < -0.01 else "#f4f4f5"
            elems.append(
                f"<text x='{left_pad - 8}' y='{y + 12}' fill='#a1a1aa' font-size='10.5' text-anchor='end' font-family='ui-monospace,Consolas,monospace'>{html.escape(human_label(cs.category)[:18])}</text>"
                f"<rect x='{left_pad}' y='{y + 1}' width='{bar_max}' height='6' fill='rgba(255,255,255,0.04)' rx='2'/>"
                f"<rect x='{left_pad}' y='{y + 1}' width='{wa}' height='6' fill='#71717a' rx='2'/>"
                f"<rect x='{left_pad}' y='{y + 9}' width='{bar_max}' height='6' fill='rgba(255,255,255,0.04)' rx='2'/>"
                f"<rect x='{left_pad}' y='{y + 9}' width='{wb}' height='6' fill='{col_b}' rx='2'/>"
                f"<text x='{left_pad + bar_max + 8}' y='{y + 12}' fill='#d6c5a9' font-size='10' font-family='ui-monospace,Consolas,monospace'>{cs.mean_score_a:.2f} → {cs.mean_score_b:.2f}</text>"
            )
        bars = f"<svg viewBox='0 0 {width} {height}' width='100%' style='display:block'>{''.join(elems)}</svg>"
        return (radar, bars)
