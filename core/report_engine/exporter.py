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

import safetensors
import torch
import transformers

from core import __version__
from core.activation_analysis.comparator import ActivationDiffReport
from core.behavioral_analysis.runner import BehavioralDiffReport
from core.regression_engine.engine import RegressionReport
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
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "safetensors_version": safetensors.__version__,
        "neural_diff_version": __version__,
    }


class ReportEngine:
    """Exports `ComparisonBundle` measurements into strict JSON, Markdown, and SVG-rich HTML."""

    def to_json(self, bundle: ComparisonBundle, *, include_full_tensor_records: bool = True) -> str:
        payload = bundle.to_dict(include_full_tensor_records=include_full_tensor_records)
        # allow_nan=False enforces strict RFC-8259 JSON compliance
        return json.dumps(payload, indent=2, allow_nan=False)

    def to_markdown(self, bundle: ComparisonBundle) -> str:
        ledger = bundle.build_epistemic_ledger()
        lines: list[str] = [
            f"# NEURAL DIFF Report — `{bundle.comparison_id}`",
            "",
            f"- **Experiment Classification:** `{bundle.experiment_type}`",
            f"- **Generated (UTC):** `{bundle.created_at_utc}`",
            f"- **Model A (Baseline):** `{bundle.model_a.model_id}` (`{bundle.model_a.total_parameters:,}` params, format=`{bundle.model_a.format.value}`, rev=`{bundle.model_a.revision or 'local'}`)",
            f"- **Model B (Candidate):** `{bundle.model_b.model_id}` (`{bundle.model_b.total_parameters:,}` params, format=`{bundle.model_b.format.value}`, rev=`{bundle.model_b.revision or 'local'}`)",
            f"- **Executed Modules:** `{', '.join(bundle.selected_modules)}`",
            "",
            "## 1. Epistemic Classification Ledger",
            "",
            "| Status | Finding | Source |",
            "| :--- | :--- | :--- |",
        ]
        for c in ledger:
            lines.append(f"| `{c.epistemic_status.value}` | {c.statement} | {c.evidence_source} |")

        if bundle.structural is not None:
            st = bundle.structural
            lines.extend(
                [
                    "",
                    "## 2. Structural & Compatibility Diff (`OBSERVED` / `INFERRED`)",
                    "",
                    f"- **Compatibility Summary:** {st.compatibility.summary}",
                    f"- **Parameter Delta:** `{st.delta_parameters:+,}` (`{st.model_a_parameters:,}` -> `{st.model_b_parameters:,}`)",
                    f"- **Shared Keys:** `{len(st.shared_tensors)}` | **Added:** `{len(st.added_tensors)}` | **Removed:** `{len(st.removed_tensors)}` | **Inferred Renames (`INFERRED`):** `{len(st.inferred_renames)}` | **Shape Changes:** `{len(st.shape_changes)}` | **Dtype Changes:** `{len(st.dtype_changes)}`",
                ]
            )

        if bundle.tensor_diff is not None:
            td = bundle.tensor_diff
            lines.extend(
                [
                    "",
                    "## 3. Tensor Diff Summary (`OBSERVED`)",
                    "",
                    f"- **Compared Tensors:** `{td.compared_tensors}` (Identical: `{td.identical_tensors}`, Within Tolerance: `{td.within_tolerance_tensors}`, Modified: `{td.modified_tensors}`)",
                    f"- **Global Changed Parameters:** `{td.total_changed_parameters:,} / {td.total_compared_parameters:,}` (`{td.global_changed_pct:.4f}%`)",
                    f"- **Global Mean Cosine Similarity:** `{td.global_mean_cosine}`",
                    f"- **Global Mean Relative L2:** `{td.global_mean_relative_l2}`",
                    f"- **Global Max Absolute Diff:** `{td.global_max_abs_diff}`",
                    "",
                    "### Top Diverged Tensors",
                    "",
                    "| Tensor | Role | Shape | Cosine | Relative L2 | Max Abs Diff | Changed % |",
                    "| :--- | :--- | :--- | ---: | ---: | ---: | ---: |",
                ]
            )
            rec_map = {r.name: r for r in td.records}
            for tname in td.top_diverged_names[:10]:
                rec = rec_map.get(tname)
                if rec and rec.metrics:
                    m = rec.metrics
                    lines.append(
                        f"| `{rec.name}` | `{rec.canonical_role.value}` | `{list(rec.shape_a or [])}` | "
                        f"`{m.cosine_similarity}` | `{m.relative_l2}` | `{m.max_abs_diff}` | `{m.changed_pct:.2f}%` |"
                    )

        if bundle.activation is not None:
            act = bundle.activation
            lines.extend(
                [
                    "",
                    "## 4. Activation Analysis (`OBSERVED`)",
                    "",
                    f"- **Controlled Prompts Executed:** `{act.prompt_count}`",
                    f"- **Mean Final-Token Logit JS Divergence:** `{act.mean_final_token_js_divergence}`",
                    f"- **Cross-Prompt Final-Layer Linear CKA:** `{act.cross_prompt_pooled_cka_final_layer}`",
                ]
            )

        if bundle.behavioral is not None:
            beh = bundle.behavioral
            st_overall = beh.overall_statistics
            lines.extend(
                [
                    "",
                    "## 5. Deterministic Behavioral Comparison (`OBSERVED`)",
                    "",
                    f"- **Suite:** `{beh.suite_id}@{beh.suite_version}` (`{st_overall.n_cases}` paired cases)",
                    f"- **Pass Rate:** Model A `{st_overall.pass_rate_a:.1%}` -> Model B `{st_overall.pass_rate_b:.1%}` (Delta `{st_overall.pass_rate_delta:+.1%}`)",
                    f"- **Mean Score:** Model A `{st_overall.mean_score_a:.4f}` -> Model B `{st_overall.mean_score_b:.4f}` (Delta `{st_overall.mean_score_delta:+.4f}`, 95% Bootstrap CI `[{st_overall.bootstrap_ci95_low:+.4f}, {st_overall.bootstrap_ci95_high:+.4f}]`, McNemar p=`{st_overall.mcnemar_p_value:.4f}`)",
                    "",
                    "| Category | Cases | Pass A | Pass B | Mean Score A | Mean Score B | Delta |",
                    "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for cs in beh.category_summaries:
                lines.append(
                    f"| `{cs.category}` | {cs.case_count} | {cs.pass_rate_a:.0%} | {cs.pass_rate_b:.0%} | "
                    f"{cs.mean_score_a:.3f} | {cs.mean_score_b:.3f} | {cs.mean_score_delta:+.3f} |"
                )

        if bundle.regression is not None:
            reg = bundle.regression
            lines.extend(
                [
                    "",
                    "## 6. Regression Gate Verdict",
                    "",
                    f"- **Verdict:** **`{reg.verdict}`** (Highest Severity: `{reg.highest_severity.value}`)",
                    f"- **Policy:** `{reg.thresholds.policy_name}` | **Findings:** `{len(reg.findings)}`",
                ]
            )
            for f in reg.findings:
                lines.append(
                    f"  - `[{f.severity.value}]` (`{f.epistemic_status.value}`) **{f.title}**: {f.description}"
                )

        repro = bundle.reproducibility or default_reproducibility_metadata()
        lines.extend(
            [
                "",
                "## 7. Execution & Reproducibility Metadata",
                "",
                "```json",
                json.dumps(
                    {"reproducibility": repro, "telemetry": bundle.execution_telemetry},
                    indent=2,
                ),
                "```",
                "",
            ]
        )
        return "\n".join(lines)

    def to_html(self, bundle: ComparisonBundle) -> str:
        """Render a self-contained HTML report with data-driven SVG layer heatmaps and charts."""
        ledger = bundle.build_epistemic_ledger()
        esc = html.escape

        heatmap_svg = self._render_layer_heatmap_svg(bundle.tensor_diff)
        behavior_svg = self._render_behavioral_svg(bundle.behavioral)

        ledger_rows = "".join(
            f"<tr><td><span class='badge ep-{esc(c.epistemic_status.value.lower().replace(' ', '-'))}'>"
            f"{esc(c.epistemic_status.value)}</span></td>"
            f"<td>{esc(c.statement)}</td><td>{esc(c.evidence_source)}</td></tr>"
            for c in ledger
        )

        tensor_rows = ""
        if bundle.tensor_diff is not None:
            rec_map = {r.name: r for r in bundle.tensor_diff.records}
            target_names = bundle.tensor_diff.top_diverged_names[:15] or [
                r.name for r in bundle.tensor_diff.records[:15]
            ]
            for tname in target_names:
                r = rec_map.get(tname)
                if r is None:
                    continue
                m = r.metrics
                tensor_rows += (
                    f"<tr><td class='mono'>{esc(r.name)}</td>"
                    f"<td>{esc(r.canonical_role.value)}</td>"
                    f"<td class='mono'>{esc(str(list(r.shape_a or r.shape_b or [])))}</td>"
                    f"<td>{esc(r.status.value)}</td>"
                    f"<td class='num'>{m.cosine_similarity if m else '—'}</td>"
                    f"<td class='num'>{m.relative_l2 if m else '—'}</td>"
                    f"<td class='num'>{m.max_abs_diff if m else '—'}</td>"
                    f"<td class='num'>{f'{m.changed_pct:.2f}%' if m else '—'}</td></tr>"
                )

        beh_case_rows = ""
        if bundle.behavioral is not None:
            for pc in bundle.behavioral.cases:
                beh_case_rows += (
                    f"<tr><td class='mono'>{esc(pc.case.case_id)}</td>"
                    f"<td>{esc(pc.case.category)}</td>"
                    f"<td>{esc(pc.status_change)}</td>"
                    f"<td class='num'>{pc.eval_a.score:.2f} → {pc.eval_b.score:.2f} ({pc.score_delta:+.2f})</td>"
                    f"<td><div class='prompt'>{esc(pc.case.prompt)}</div>"
                    f"<div class='out-grid'><div><strong>Model A:</strong> {esc(pc.output_a[:280])}</div>"
                    f"<div><strong>Model B:</strong> {esc(pc.output_b[:280])}</div></div></td></tr>"
                )

        verdict_str = bundle.regression.verdict if bundle.regression else "NOT_RUN"
        repro_json = esc(
            json.dumps(
                {
                    "reproducibility": bundle.reproducibility or default_reproducibility_metadata(),
                    "telemetry": bundle.execution_telemetry,
                },
                indent=2,
            )
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>NEURAL DIFF Report — {esc(bundle.comparison_id)}</title>
<style>
  :root {{
    --bg: #0b1020; --panel: #131a2e; --border: #253152; --text: #e5ecff;
    --muted: #93a3c7; --accent: #4fd1c5; --warn: #f6ad55; --danger: #fc8181; --ok: #68d391;
  }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 28px; line-height: 1.5; }}
  .container {{ max-width: 1240px; margin: 0 auto; }}
  header {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 20px 24px; margin-bottom: 20px; }}
  h1 {{ margin: 0 0 6px 0; font-size: 24px; }}
  h2 {{ font-size: 18px; border-bottom: 1px solid var(--border); padding-bottom: 8px; margin-top: 26px; }}
  .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-top: 14px; }}
  .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12.5px; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; font-size: 13px; }}
  th, td {{ padding: 9px 11px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }}
  th {{ background: #192340; color: var(--muted); font-weight: 600; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: 0.03em; background: #233258; }}
  .ep-observed {{ background: #1c4532; color: #9ae6b4; }}
  .ep-inferred {{ background: #44337a; color: #d6bcfa; }}
  .ep-correlated {{ background: #744210; color: #fbd38d; }}
  .ep-not-verified {{ background: #2d3748; color: #cbd5e0; }}
  .out-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 6px; font-size: 12px; color: var(--muted); }}
  pre {{ background: var(--panel); border: 1px solid var(--border); padding: 14px; border-radius: 8px; overflow-x: auto; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="badge">{esc(bundle.experiment_type)}</div>
    <h1>NEURAL DIFF — {esc(bundle.comparison_id)}</h1>
    <div style="color:var(--muted)">Git for AI Brains • Generated {esc(bundle.created_at_utc)} • Regression Verdict: <strong>{esc(verdict_str)}</strong></div>
    <div class="meta-grid">
      <div class="card"><strong>Model A (Baseline)</strong><div class="mono">{esc(bundle.model_a.model_id)}</div><div>{bundle.model_a.total_parameters:,} params • {esc(bundle.model_a.format.value)}</div></div>
      <div class="card"><strong>Model B (Candidate)</strong><div class="mono">{esc(bundle.model_b.model_id)}</div><div>{bundle.model_b.total_parameters:,} params • {esc(bundle.model_b.format.value)}</div></div>
      <div class="card"><strong>Modules Executed</strong><div class="mono">{esc(", ".join(bundle.selected_modules))}</div></div>
    </div>
  </header>

  <h2>1. Epistemic Status Ledger</h2>
  <table><thead><tr><th>Epistemic Label</th><th>Finding</th><th>Evidence Source</th></tr></thead><tbody>{ledger_rows}</tbody></table>

  <h2>2. Layer-by-Component Weight Diff Heatmap (OBSERVED)</h2>
  <div class="card">{heatmap_svg}</div>

  <h2>3. Tensor Diff Explorer Summary (OBSERVED)</h2>
  <table><thead><tr><th>Tensor</th><th>Role</th><th>Shape</th><th>Status</th><th>Cosine</th><th>Rel L2</th><th>Max Abs</th><th>Changed</th></tr></thead><tbody>{tensor_rows or "<tr><td colspan='8'>Tensor diff module not selected.</td></tr>"}</tbody></table>

  <h2>4. Behavioral Intelligence Comparison (OBSERVED)</h2>
  <div class="card">{behavior_svg}</div>
  <table style="margin-top:12px"><thead><tr><th>Case</th><th>Category</th><th>Change</th><th>Score A → B</th><th>Prompt &amp; Paired Outputs</th></tr></thead><tbody>{beh_case_rows or "<tr><td colspan='5'>Behavioral module not selected.</td></tr>"}</tbody></table>

  <h2>5. Execution &amp; Reproducibility Manifest</h2>
  <pre class="mono">{repro_json}</pre>
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
    def _render_layer_heatmap_svg(tensor_diff: TensorDiffReport | None) -> str:
        if tensor_diff is None or not tensor_diff.layer_summaries:
            return "<div style='color:#93a3c7'>No layered tensor diff measurements available for heatmap.</div>"

        roles = ["attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down", "norm"]
        layers = tensor_diff.layer_summaries
        cell_w = 26
        cell_h = 22
        left_pad = 92
        top_pad = 28
        width = left_pad + max(len(layers) * cell_w, 240) + 30
        height = top_pad + len(roles) * cell_h + 32

        # Normalize color intensity by observed max relative_l2 across cells
        max_rel = 1e-6
        for ls in layers:
            for rname in roles:
                val = (ls.role_breakdown.get(rname) or {}).get("mean_relative_l2")
                if val is not None and val > max_rel:
                    max_rel = val

        rects: list[str] = []
        for l_pos, ls in enumerate(layers):
            x = left_pad + l_pos * cell_w
            if l_pos % max(1, len(layers) // 12) == 0:
                rects.append(
                    f"<text x='{x + cell_w // 2}' y='18' fill='#93a3c7' font-size='10' text-anchor='middle'>L{ls.layer_index}</text>"
                )
            for r_pos, rname in enumerate(roles):
                y = top_pad + r_pos * cell_h
                val = (ls.role_breakdown.get(rname) or {}).get("mean_relative_l2")
                if val is None:
                    fill = "#18223d"
                    tip = f"L{ls.layer_index} {rname}: N/A"
                else:
                    ratio = min(1.0, max(0.0, val / max_rel))
                    r_col = int(30 + 215 * ratio)
                    g_col = int(55 + 135 * (1.0 - abs(ratio - 0.5)))
                    b_col = int(130 * (1.0 - ratio) + 45)
                    fill = f"rgb({r_col},{g_col},{b_col})"
                    tip = f"L{ls.layer_index} {rname}: rel_L2={val:.6f}"
                rects.append(
                    f"<rect x='{x}' y='{y}' width='{cell_w - 2}' height='{cell_h - 2}' rx='2' fill='{fill}'><title>{html.escape(tip)}</title></rect>"
                )

        for r_pos, rname in enumerate(roles):
            y = top_pad + r_pos * cell_h + 14
            rects.append(f"<text x='6' y='{y}' fill='#cbd5e0' font-size='11'>{html.escape(rname)}</text>")

        return (
            f"<svg viewBox='0 0 {width} {height}' width='100%' style='max-height:260px'>"
            + "".join(rects)
            + "</svg>"
        )

    @staticmethod
    def _render_behavioral_svg(behavioral: BehavioralDiffReport | None) -> str:
        if behavioral is None or not behavioral.category_summaries:
            return "<div style='color:#93a3c7'>Behavioral diff module not executed.</div>"

        cats = behavioral.category_summaries
        row_h = 30
        left_pad = 165
        bar_max = 360
        width = left_pad + bar_max + 130
        height = len(cats) * row_h + 24

        elems: list[str] = []
        for idx, cs in enumerate(cats):
            y = 14 + idx * row_h
            wa = int(cs.mean_score_a * bar_max)
            wb = int(cs.mean_score_b * bar_max)
            elems.append(f"<text x='6' y='{y + 14}' fill='#e5ecff' font-size='11'>{html.escape(cs.category)}</text>")
            elems.append(f"<rect x='{left_pad}' y='{y}' width='{max(2, wa)}' height='9' fill='#63b3ed' rx='2' />")
            elems.append(f"<rect x='{left_pad}' y='{y + 11}' width='{max(2, wb)}' height='9' fill='#4fd1c5' rx='2' />")
            elems.append(
                f"<text x='{left_pad + bar_max + 10}' y='{y + 14}' fill='#93a3c7' font-size='11'>"
                f"A={cs.mean_score_a:.2f} / B={cs.mean_score_b:.2f}</text>"
            )
        return f"<svg viewBox='0 0 {width} {height}' width='100%'>" + "".join(elems) + "</svg>"
