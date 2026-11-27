"""Minimalist, high-density SVG & Markdown visual primitives for NEURAL DIFF reports.

Follows Architectural Monochrome principles: `#09090b` obsidian canvas, hairline `rgba(255,255,255,0.07)`
gridlines, `#f4f4f5` titanium primary signal, `#d6c5a9` champagne secondary signal, and semantic
`#4ade80` / `#f87171` delta accents. Zero decorative gradients or colored pill badges.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from typing import Any


def md_bar(value: float | None, max_value: float = 1.0, width: int = 12) -> str:
    """Render a proportional Unicode block gauge (`████░░░░░░░░`) for Markdown tables."""
    if value is None or math.isnan(value) or max_value <= 0:
        return "·" * width
    ratio = max(0.0, min(1.0, float(value) / float(max_value)))
    filled = int(round(ratio * width))
    return ("█" * filled) + ("░" * (width - filled))


def human_label(raw: str | None) -> str:
    """Convert raw snake_case / SCREAMING_SNAKE_CASE identifiers to clean sentence case."""
    if not raw:
        return "—"
    mapping = {
        "OBSERVED": "Observed",
        "INFERRED": "Inferred",
        "CORRELATED": "Correlated",
        "CAUSAL SUPPORT": "Causal support",
        "CAUSALLY VERIFIED": "Causally verified",
        "NOT VERIFIED": "Unverified",
        "ancestor_to_descendant": "Ancestor → Descendant",
        "descendant_to_ancestor": "Descendant → Ancestor",
        "base_to_current": "Base → Current",
        "branch_to_branch": "Branch ↔ Branch",
        "consecutive": "Step → Step",
        "unrelated": "Disjoint roots",
        "pass_rate_and_score": "Pass rate & score",
        "pass_rate_only": "Pass rate drop",
        "score_only": "Score drop",
        "exploratory_small_sample": "Exploratory sample",
        "statistically_significant": "Statistically significant",
        "PASSED": "Passed",
        "WARNING": "Warning",
        "FAILED_REGRESSION": "Failed gate",
        "NOT_RUN": "Not run",
        "REAL MODEL EXPERIMENT": "Real model",
        "SYNTHETIC FIXTURE EXPERIMENT": "Synthetic fixture",
    }
    if raw in mapping:
        return mapping[raw]
    cleaned = raw.replace("_", " ").strip()
    return cleaned[:1].upper() + cleaned[1:].lower() if cleaned.isupper() or "_" in raw else cleaned


def svg_segmented_ratio_bar(
    segments: list[tuple[str, float, str]],
    *,
    width: int = 760,
) -> str:
    """Render a single sleek horizontal composition bar with inline dot legend."""
    total = sum(max(0.0, v) for _, v, _ in segments)
    if total <= 0:
        return ""
    bar_w = width - 16
    x = 8.0
    rects: list[str] = []
    legend: list[str] = []
    lx = 8.0
    for label, val, color in segments:
        frac = max(0.0, val) / total
        seg_w = frac * bar_w
        if seg_w >= 1.0:
            rects.append(
                f"<rect x='{x:.1f}' y='8' width='{max(1.5, seg_w - 1.5):.1f}' height='8' rx='2' fill='{color}'/>"
            )
        x += seg_w
        legend.append(
            f"<circle cx='{lx + 4:.1f}' cy='30' r='3' fill='{color}'/>"
            f"<text x='{lx + 12:.1f}' y='33.5' fill='#a1a1aa' font-size='10.5' font-family='ui-monospace,Consolas,monospace'>"
            f"{html.escape(label)} {(frac * 100):.1f}%</text>"
        )
        lx += max(125.0, len(label) * 7.2 + 64.0)

    return (
        f"<svg viewBox='0 0 {width} 42' width='100%' style='display:block'>"
        f"<rect x='8' y='8' width='{bar_w}' height='8' rx='2' fill='rgba(255,255,255,0.04)'/>"
        + "".join(rects)
        + "".join(legend)
        + "</svg>"
    )


def svg_matrix_heatmap(
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    matrix: Sequence[Sequence[float | None]],
    invert_scale: bool = False,
    min_val: float | None = None,
    max_val: float | None = None,
) -> str:
    """Render a crisp `rows × cols` luminance matrix heatmap in champagne/titanium tones."""
    if not row_labels or not col_labels or not matrix:
        return ""

    flat = [v for row in matrix for v in row if v is not None and math.isfinite(v)]
    if not flat:
        return ""

    lo = min_val if min_val is not None else min(flat)
    hi = max_val if max_val is not None else max(flat)
    span = max(1e-9, hi - lo)

    left_pad = max(108, min(195, max(len(r) for r in row_labels) * 6 + 18))
    cell_w = max(20, min(36, int(680 / max(1, len(col_labels)))))
    cell_h = 22
    top_pad = 24
    width = left_pad + len(col_labels) * cell_w + 16
    height = top_pad + len(row_labels) * cell_h + 12

    elems: list[str] = []
    label_stride = max(1, len(col_labels) // 16)
    for c_idx, clbl in enumerate(col_labels):
        if c_idx % label_stride == 0 or c_idx == len(col_labels) - 1:
            cx = left_pad + c_idx * cell_w + cell_w / 2
            elems.append(
                f"<text x='{cx:.1f}' y='15' fill='#71717a' font-size='9.5' text-anchor='middle' font-family='ui-monospace,Consolas,monospace'>{html.escape(clbl)}</text>"
            )

    show_cell_text = cell_w >= 32 and len(col_labels) <= 14
    for r_idx, rlbl in enumerate(row_labels):
        y = top_pad + r_idx * cell_h
        elems.append(
            f"<text x='{left_pad - 8}' y='{y + 14.5:.1f}' fill='#a1a1aa' font-size='10.5' text-anchor='end' font-family='ui-monospace,Consolas,monospace'>{html.escape(rlbl[:26])}</text>"
        )
        row_vals = matrix[r_idx] if r_idx < len(matrix) else []
        for c_idx in range(len(col_labels)):
            x = left_pad + c_idx * cell_w
            val = row_vals[c_idx] if c_idx < len(row_vals) else None
            if val is None or not math.isfinite(val):
                elems.append(
                    f"<rect x='{x}' y='{y}' width='{cell_w - 2}' height='{cell_h - 2}' rx='2' fill='rgba(255,255,255,0.03)'/>"
                )
                continue
            norm = max(0.0, min(1.0, (val - lo) / span))
            intensity = (1.0 - norm) if invert_scale else norm
            alpha = 0.06 + 0.76 * intensity
            tip = f"{rlbl} · {col_labels[c_idx]}: {val:.5f}"
            elems.append(
                f"<rect x='{x}' y='{y}' width='{cell_w - 2}' height='{cell_h - 2}' rx='2' fill='rgba(214,197,169,{alpha:.3f})'><title>{html.escape(tip)}</title></rect>"
            )
            if show_cell_text:
                txt_col = "#09090b" if alpha > 0.52 else "#e4e4e7"
                elems.append(
                    f"<text x='{x + (cell_w - 2) / 2:.1f}' y='{y + 13.5:.1f}' fill='{txt_col}' font-size='8.5' text-anchor='middle' font-family='ui-monospace,Consolas,monospace'>{val:.2f}</text>"
                )

    return f"<svg viewBox='0 0 {width} {height}' width='100%' style='display:block'>{''.join(elems)}</svg>"


def svg_dual_line_chart(
    *,
    x_labels: Sequence[str],
    series_primary: Sequence[float | None],
    label_primary: str,
    series_secondary: Sequence[float | None] | None = None,
    label_secondary: str = "",
    y_min: float = 0.0,
    y_max: float = 1.0,
    width: int = 780,
    height: int = 195,
) -> str:
    """Render an oscilloscope-style area + dual-curve trajectory chart."""
    n = len(x_labels)
    if n == 0:
        return ""

    pad_l, pad_r, pad_t, pad_b = 46.0, 20.0, 26.0, 28.0
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    span_y = max(1e-9, y_max - y_min)

    def _xy(idx: int, val: float) -> tuple[float, float]:
        px = pad_l + (idx / max(1, n - 1)) * plot_w if n > 1 else pad_l + plot_w / 2
        clamped = max(y_min, min(y_max, val))
        py = pad_t + (1.0 - (clamped - y_min) / span_y) * plot_h
        return px, py

    elems: list[str] = [
        "<defs>"
        "<linearGradient id='ndAreaGrad' x1='0' y1='0' x2='0' y2='1'>"
        "<stop offset='0%' stop-color='#f4f4f5' stop-opacity='0.18'/>"
        "<stop offset='100%' stop-color='#f4f4f5' stop-opacity='0.0'/>"
        "</linearGradient>"
        "</defs>"
    ]

    # Horizontal gridlines
    for step in (0.0, 0.5, 1.0):
        gv = y_min + step * span_y
        gy = pad_t + (1.0 - step) * plot_h
        elems.append(
            f"<line x1='{pad_l}' y1='{gy:.1f}' x2='{width - pad_r}' y2='{gy:.1f}' stroke='rgba(255,255,255,0.06)' stroke-width='1'/>"
            f"<text x='{pad_l - 6}' y='{gy + 3.5:.1f}' fill='#71717a' font-size='9.5' text-anchor='end' font-family='ui-monospace,Consolas,monospace'>{gv:.2f}</text>"
        )

    # X labels
    stride = max(1, n // 12)
    for i, xlbl in enumerate(x_labels):
        if i % stride == 0 or i == n - 1:
            px, _ = _xy(i, y_min)
            elems.append(
                f"<text x='{px:.1f}' y='{height - 8}' fill='#71717a' font-size='9.5' text-anchor='middle' font-family='ui-monospace,Consolas,monospace'>{html.escape(xlbl[:14])}</text>"
            )

    pts_p = [(i, float(v)) for i, v in enumerate(series_primary) if v is not None and math.isfinite(v)]
    if pts_p:
        coords_p = [_xy(i, v) for i, v in pts_p]
        poly_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords_p)
        if len(coords_p) >= 2:
            area_d = (
                f"M {coords_p[0][0]:.1f} {pad_t + plot_h:.1f} "
                + " ".join(f"L {x:.1f} {y:.1f}" for x, y in coords_p)
                + f" L {coords_p[-1][0]:.1f} {pad_t + plot_h:.1f} Z"
            )
            elems.append(f"<path d='{area_d}' fill='url(#ndAreaGrad)'/>")
            elems.append(
                f"<polyline points='{poly_str}' fill='none' stroke='#f4f4f5' stroke-width='1.75'/>"
            )
        for x, y in coords_p:
            elems.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='2.4' fill='#f4f4f5'/>")

    if series_secondary:
        pts_s = [
            (i, float(v)) for i, v in enumerate(series_secondary) if v is not None and math.isfinite(v)
        ]
        if pts_s:
            coords_s = [_xy(i, v) for i, v in pts_s]
            poly_s = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords_s)
            if len(coords_s) >= 2:
                elems.append(
                    f"<polyline points='{poly_s}' fill='none' stroke='#d6c5a9' stroke-width='1.5' stroke-dasharray='4 3'/>"
                )
            for x, y in coords_s:
                elems.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='2.1' fill='#d6c5a9'/>")

    # Legend top-right
    elems.append(
        f"<line x1='{pad_l}' y1='11' x2='{pad_l + 14}' y2='11' stroke='#f4f4f5' stroke-width='2'/>"
        f"<text x='{pad_l + 20}' y='14' fill='#e4e4e7' font-size='10.5' font-family='ui-monospace,Consolas,monospace'>{html.escape(label_primary)}</text>"
    )
    if series_secondary and label_secondary:
        off = pad_l + 36 + len(label_primary) * 6.8
        elems.append(
            f"<line x1='{off:.1f}' y1='11' x2='{off + 14:.1f}' y2='11' stroke='#d6c5a9' stroke-width='2' stroke-dasharray='3 2'/>"
            f"<text x='{off + 20:.1f}' y='14' fill='#d6c5a9' font-size='10.5' font-family='ui-monospace,Consolas,monospace'>{html.escape(label_secondary)}</text>"
        )

    return f"<svg viewBox='0 0 {width} {height}' width='100%' style='display:block'>{''.join(elems)}</svg>"


def svg_radar_polygon(
    *,
    axes: list[str],
    series: list[tuple[str, list[float], str]],
    size: int = 290,
) -> str:
    """Render a minimalist polar radar / capability polygon chart (`[0..1]` normalized)."""
    m = len(axes)
    if m < 3 or not series:
        return ""

    cx, cy = size / 2.0 + 18.0, size / 2.0 + 4.0
    radius = size * 0.33
    total_w = size + 36

    elems: list[str] = []
    # Concentric web polygons
    for ring in (0.25, 0.5, 0.75, 1.0):
        r = radius * ring
        ring_pts = []
        for i in range(m):
            ang = -math.pi / 2.0 + (2.0 * math.pi * i) / m
            ring_pts.append(f"{cx + r * math.cos(ang):.1f},{cy + r * math.sin(ang):.1f}")
        elems.append(
            f"<polygon points='{' '.join(ring_pts)}' fill='none' stroke='rgba(255,255,255,0.07)' stroke-width='1'/>"
        )

    # Spokes + axis labels
    for i, lbl in enumerate(axes):
        ang = -math.pi / 2.0 + (2.0 * math.pi * i) / m
        ex = cx + radius * math.cos(ang)
        ey = cy + radius * math.sin(ang)
        lx = cx + (radius + 18.0) * math.cos(ang)
        ly = cy + (radius + 14.0) * math.sin(ang)
        anchor = "middle"
        if math.cos(ang) > 0.28:
            anchor = "start"
        elif math.cos(ang) < -0.28:
            anchor = "end"
        short_lbl = lbl.replace("_", " ")[:14]
        elems.append(
            f"<line x1='{cx:.1f}' y1='{cy:.1f}' x2='{ex:.1f}' y2='{ey:.1f}' stroke='rgba(255,255,255,0.06)' stroke-width='1'/>"
            f"<text x='{lx:.1f}' y='{ly + 3:.1f}' fill='#a1a1aa' font-size='9.5' text-anchor='{anchor}' font-family='ui-monospace,Consolas,monospace'>{html.escape(short_lbl)}</text>"
        )

    # Data polygons
    for idx, (s_label, vals, color) in enumerate(series):
        pts = []
        for i in range(m):
            v = max(0.0, min(1.0, float(vals[i]) if i < len(vals) else 0.0))
            ang = -math.pi / 2.0 + (2.0 * math.pi * i) / m
            pts.append(f"{cx + radius * v * math.cos(ang):.1f},{cy + radius * v * math.sin(ang):.1f}")
        elems.append(
            f"<polygon points='{' '.join(pts)}' fill='{color}' fill-opacity='0.14' stroke='{color}' stroke-width='1.6'/>"
        )
        lx = 12 + idx * 140
        elems.append(
            f"<circle cx='{lx}' cy='{size - 8}' r='3' fill='{color}'/>"
            f"<text x='{lx + 8}' y='{size - 4.5}' fill='#a1a1aa' font-size='9.5' font-family='ui-monospace,Consolas,monospace'>{html.escape(s_label[:20])}</text>"
        )

    return f"<svg viewBox='0 0 {total_w} {size}' width='100%' style='max-height:280px;display:block'>{''.join(elems)}</svg>"


def svg_lollipop_chart(
    items: list[tuple[str, float, str]],
    *,
    width: int = 760,
) -> str:
    """Render a minimalist horizontal stem-and-dot lollipop chart `(label, value, formatted_right)`."""
    if not items:
        return ""
    max_v = max((abs(v) for _, v, _ in items), default=1.0) or 1.0
    row_h = 22
    left_pad = 250
    track_w = width - left_pad - 105
    height = len(items) * row_h + 14

    elems: list[str] = []
    for idx, (lbl, val, right_txt) in enumerate(items):
        y = 12 + idx * row_h
        w = max(2.0, (max(0.0, val) / max_v) * track_w)
        short = lbl if len(lbl) <= 36 else ("…" + lbl[-35:])
        elems.append(
            f"<text x='{left_pad - 10}' y='{y + 4}' fill='#a1a1aa' font-size='10.5' text-anchor='end' font-family='ui-monospace,Consolas,monospace'>{html.escape(short)}</text>"
            f"<line x1='{left_pad}' y1='{y}' x2='{left_pad + track_w}' y2='{y}' stroke='rgba(255,255,255,0.04)' stroke-width='1'/>"
            f"<line x1='{left_pad}' y1='{y}' x2='{left_pad + w:.1f}' y2='{y}' stroke='#d6c5a9' stroke-width='1.8'/>"
            f"<circle cx='{left_pad + w:.1f}' cy='{y}' r='3.2' fill='#f4f4f5'/>"
            f"<text x='{left_pad + track_w + 10}' y='{y + 3.5}' fill='#e4e4e7' font-size='10.5' font-family='ui-monospace,Consolas,monospace'>{html.escape(right_txt)}</text>"
        )
    return f"<svg viewBox='0 0 {width} {height}' width='100%' style='display:block'>{''.join(elems)}</svg>"


def svg_evolution_dag(
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    regression_target_ids: set[str],
    pass_rate_by_id: dict[str, float],
    base_l2_by_id: dict[str, float],
) -> str:
    """Render a standalone vector Branched & Merged DAG diagram directly as crisp SVG."""
    if not nodes:
        return ""

    max_depth = max((int(n.get("generation_depth", 0)) for n in nodes), default=0)
    max_lanes = max((int(n.get("lane_count", 1)) for n in nodes), default=1)

    node_w, node_h = 166.0, 62.0
    col_step = 216.0
    row_step = 90.0
    width = int(48 + (max_depth + 1) * col_step)
    height = int(max(140, max_lanes * row_step + 56))
    mid_y = height / 2.0

    pos_map: dict[str, tuple[float, float]] = {}
    for n in nodes:
        d = int(n.get("generation_depth", 0))
        li = int(n.get("lane_index", 0))
        lc = int(n.get("lane_count", 1))
        x = 24.0 + d * col_step
        y = mid_y + (li - (lc - 1) / 2.0) * row_step - node_h / 2.0
        pos_map[str(n["checkpoint_id"])] = (x, y)

    elems: list[str] = []
    # Bezier edges
    for e in edges:
        src = str(e["parent_checkpoint_id"])
        tgt = str(e["child_checkpoint_id"])
        if src not in pos_map or tgt not in pos_map:
            continue
        sx, sy = pos_map[src]
        tx, ty = pos_map[tgt]
        x1, y1 = sx + node_w, sy + node_h / 2.0
        x2, y2 = tx, ty + node_h / 2.0
        cx1, cx2 = x1 + 26.0, x2 - 26.0
        is_reg = tgt in regression_target_ids
        stroke = "#f87171" if is_reg else "rgba(214,197,169,0.55)"
        dash = " stroke-dasharray='4 3'" if is_reg else ""
        elems.append(
            f"<path d='M {x1:.1f} {y1:.1f} C {cx1:.1f} {y1:.1f}, {cx2:.1f} {y2:.1f}, {x2:.1f} {y2:.1f}' fill='none' stroke='{stroke}' stroke-width='1.6'{dash}/>"
        )

    # Nodes
    for n in nodes:
        cid = str(n["checkpoint_id"])
        x, y = pos_map[cid]
        is_reg = cid in regression_target_ids
        is_base = bool(n.get("is_base_model"))
        border = "rgba(248,113,113,0.65)" if is_reg else ("rgba(74,222,128,0.45)" if is_base else "rgba(255,255,255,0.13)")
        dot_col = "#f87171" if is_reg else ("#4ade80" if is_base else "#e4e4e7")
        lbl = str(n.get("version_label", cid))[:20]
        stage = str(n.get("transformation_type", ""))[:20]
        pr = pass_rate_by_id.get(cid)
        l2 = base_l2_by_id.get(cid)
        sub_metric = (
            f"Pass {pr * 100:.1f}% · L2 {l2:.1f}"
            if (pr is not None and l2 is not None)
            else (f"L2 {l2:.2f}" if l2 is not None else f"{int(n.get('parameter_count', 0)) // 1_000_000}M")
        )

        elems.append(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{node_w}' height='{node_h}' rx='8' fill='#121216' stroke='{border}' stroke-width='1'/>"
            f"<circle cx='{x + 12:.1f}' cy='{y + 15:.1f}' r='3' fill='{dot_col}'/>"
            f"<text x='{x + 21:.1f}' y='{y + 18.5:.1f}' fill='#f4f4f5' font-size='10.5' font-weight='600' font-family='Inter,system-ui,sans-serif'>{html.escape(lbl)}</text>"
            f"<text x='{x + 12:.1f}' y='{y + 35:.1f}' fill='#71717a' font-size='9.5' font-family='ui-monospace,Consolas,monospace'>{html.escape(stage)}</text>"
            f"<text x='{x + 12:.1f}' y='{y + 50:.1f}' fill='#d6c5a9' font-size='9.5' font-family='ui-monospace,Consolas,monospace'>{html.escape(sub_metric)}</text>"
        )

    return f"<svg viewBox='0 0 {width} {height}' width='100%' style='display:block'>{''.join(elems)}</svg>"
