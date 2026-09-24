#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Generate static, screenshot-ready SVG chart HTML files from eval-harness
JSON reports for pre-embedded (BEIR) datasets.

Only colbert and rrf rescorer results are charted -- any stray cross-encoder
reports left over from earlier runs are ignored.

Usage:
    uv run python generate_charts.py [results_dir] [out_dir]
"""

import json
import math
import sys
from pathlib import Path
from typing import Any

WASH = "#e2e7f5"
CARD = "#ffffff"
N30 = "#28324d"
N50 = "#576280"
N70 = "#8f98b2"
N90 = "#d4d9e6"
GRID = "#e1e5f0"
BLUE30 = "#0040a1"
BLUE50 = "#2f6ff0"
BLUE70 = "#89a9ff"
TEAL30 = "#004f4f"
TEAL50 = "#038585"

FONT = 'ui-monospace, "JetBrains Mono", "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace'

# Display order + two-line labels for quant/dtype config variants, matching
# the naming already used in packages/qdrant-load/configs/<dataset>/*.yml.
VARIANT_ORDER = [
    "float32_noquant",
    "float16_noquant",
    "turbo4_noquant",
    "scalar_int8",
    "product_x16",
    "binary",
    "float16_binary",
    "tubo4_bits1",
]
VARIANT_LABELS = {
    "float32_noquant": ("no-quant", "(f32)"),
    "float16_noquant": ("no-quant", "(f16)"),
    "turbo4_noquant": ("no-quant", "(turbo4)"),
    "scalar_int8": ("scalar", "int8"),
    "product_x16": ("product", "x16"),
    "binary": ("binary", "(f32)"),
    "float16_binary": ("binary", "(f16)"),
    "tubo4_bits1": ("turbo4", "tq1"),
}


def load_reports(results_dir: Path) -> list[dict[str, Any]]:
    reports = []
    for path in sorted(results_dir.glob("*.json")):
        d = json.loads(path.read_text())
        parts = d["config"].split("/")
        d["_dataset"] = parts[-2]
        d["_variant"] = parts[-1].replace("config_", "").replace(".yml", "")
        d["_mode"] = "hybrid" if d.get("use_dense_prefetch", True) else "sparseonly"
        reports.append(d)
    return reports


def pick(reports: list[dict[str, Any]], **filters: Any) -> dict[str, dict[str, Any]]:
    """Returns {variant: report} for reports matching all filters."""
    out: dict[str, dict[str, Any]] = {}
    for r in reports:
        if all(r.get(k) == v for k, v in filters.items()):
            out[r["_variant"]] = r
    return out


def nice_ticks(max_val: float, n_ticks: int = 5) -> list[float]:
    """Rounds max_val up to a human-friendly axis ceiling and returns evenly
    spaced tick values from 0 to that ceiling."""
    if max_val <= 0:
        return [0, 1]
    raw_step = max_val / (n_ticks - 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    residual = raw_step / magnitude
    if residual > 5:
        step = 10 * magnitude
    elif residual > 2:
        step = 5 * magnitude
    elif residual > 1:
        step = 2 * magnitude
    else:
        step = magnitude
    ticks = [step * i for i in range(n_ticks)]
    while ticks[-1] < max_val:
        step_count = len(ticks)
        ticks.append(step * step_count)
    return ticks


def fmt_tick(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def y_pos(value: float, y_max: float, plot_top: float, plot_bottom: float) -> float:
    frac = value / y_max if y_max else 0
    return plot_bottom - frac * (plot_bottom - plot_top)


def page(body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  :root {{
    --wash: {WASH}; --card: {CARD};
    --n30: {N30}; --n50: {N50}; --n70: {N70}; --n90: {N90}; --grid: {GRID};
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:40px; background:var(--wash); font-family: {FONT}; }}
  .frame {{ max-width:1180px; margin:0 auto; background:var(--card); border-radius:16px; padding:44px 48px 36px; }}
  .legend {{ display:flex; gap:32px; flex-wrap:wrap; align-items:center; }}
  .legend-group {{ display:flex; align-items:center; gap:10px; }}
  .swatch {{ width:15px; height:15px; border-radius:3px; display:inline-block; }}
  .legend-label {{ font-size:14px; color:var(--n30); }}
  svg text {{ font-family: {FONT}; }}
</style>
</head>
<body>
<div class="frame">
{body}
</div>
</body>
</html>
"""


def grouped_bar_chart(
    categories: list[str],
    series: list[tuple[str, str, dict[str, float]]],
    y_max: float,
    y_ticks: list[float],
    x_title: str,
    y_title: str,
    value_fmt: str = "{:.1f}",
) -> str:
    """categories: variant keys, in display order.
    series: list of (legend label, color, {variant: value})."""
    plot_left, plot_right = 76, 1156
    plot_top, plot_bottom = 24, 400
    width = plot_right - plot_left
    n_cat = len(categories)
    group_w = width / n_cat if n_cat else width
    bar_w = 20
    gap = 8
    n_series = len(series)
    group_content_w = n_series * bar_w + (n_series - 1) * gap

    svg: list[str] = []

    for tick in y_ticks:
        y = y_pos(tick, y_max, plot_top, plot_bottom)
        svg.append(
            f'<line x1="{plot_left}" x2="{plot_right}" y1="{y:.2f}" y2="{y:.2f}" stroke="var(--grid)" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{plot_left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="14" fill="var(--n70)">{fmt_tick(tick)}</text>'
        )
    svg.append(
        f'<line x1="{plot_left}" x2="{plot_right}" y1="{plot_bottom}" y2="{plot_bottom}" stroke="var(--n90)" stroke-width="1.5"/>'
    )

    mid_y = (plot_top + plot_bottom) / 2
    svg.append(
        f'<text x="-40" y="{mid_y:.2f}" text-anchor="middle" font-size="14.5" fill="var(--n50)" '
        f'transform="rotate(-90 -40 {mid_y:.2f})">{y_title}</text>'
    )
    mid_x = (plot_left + plot_right) / 2
    svg.append(
        f'<text x="{mid_x:.2f}" y="480" text-anchor="middle" font-size="14.5" fill="var(--n50)">{x_title}</text>'
    )

    for ci, cat in enumerate(categories):
        group_center = plot_left + group_w * (ci + 0.5)
        group_start = group_center - group_content_w / 2
        label = VARIANT_LABELS.get(cat, (cat, ""))
        for si, (_label, color, values) in enumerate(series):
            val = values.get(cat)
            if val is None:
                continue
            bx = group_start + si * (bar_w + gap)
            by = y_pos(val, y_max, plot_top, plot_bottom)
            bh = plot_bottom - by
            svg.append(
                f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w}" height="{bh:.2f}" rx="4" fill="{color}"/>'
            )
            svg.append(
                f'<text x="{bx + bar_w / 2:.2f}" y="{by - 10:.2f}" text-anchor="middle" font-size="12.5" fill="var(--n30)">{value_fmt.format(val)}</text>'
            )
        cx = group_start + group_content_w / 2
        svg.append(
            f'<text x="{cx:.2f}" y="{plot_bottom + 24}" text-anchor="middle" font-size="13" fill="var(--n50)">'
            f'{label[0]}<tspan x="{cx:.2f}" dy="16">{label[1]}</tspan></text>'
        )

    legend = '<div class="legend">' + "".join(
        f'<span class="legend-group"><span class="swatch" style="background:{color}"></span>'
        f'<span class="legend-label">{lbl}</span></span>'
        for lbl, color, _ in series
    ) + "</div>"

    svg_tag = (
        '<svg viewBox="-60 0 1240 500" width="100%" height="auto" style="overflow:visible; display:block; margin-top:22px">\n'
        + "\n".join(svg)
        + "\n</svg>"
    )
    return legend + "\n" + svg_tag


def quality_series(slice_: dict[str, dict[str, Any]], k: int, cats: list[str]) -> list[tuple[str, str, dict[str, float]]]:
    return [
        (f"recall@{k}", BLUE30, {c: slice_[c]["quality"]["recall@k"] * 100 for c in cats}),
        (f"mrr@{k}", BLUE50, {c: slice_[c]["quality"]["mrr@k"] * 100 for c in cats}),
        (f"ndcg@{k}", BLUE70, {c: slice_[c]["quality"]["ndcg@k"] * 100 for c in cats}),
    ]


def main() -> None:
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/kubench-shared/results")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "pages/charts")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed representative slice for the "by config" charts: hybrid prefetch,
    # k=20, the widest prefetch_limit in the sweep (best-quality end).
    K = 20
    PF = 100

    reports = [r for r in load_reports(results_dir) if r["rescorer"] == "colbert"]
    datasets = sorted({r["_dataset"] for r in reports})

    if not datasets:
        print(f"No colbert reports found in {results_dir}")
        return

    for dataset in datasets:
        ds_reports = [r for r in reports if r["_dataset"] == dataset]
        slug = dataset.replace("-", "_")

        colbert_slice = pick(ds_reports, rescorer="colbert", _mode="hybrid", k=K, prefetch_limit=PF)
        colbert_cats = [c for c in VARIANT_ORDER if c in colbert_slice]

        if not colbert_cats:
            print(f"[{dataset}] no colbert reports at k={K}, pf={PF}, hybrid -- skipped")
            continue

        body = grouped_bar_chart(
            colbert_cats,
            quality_series(colbert_slice, K, colbert_cats),
            100,
            [0, 25, 50, 75, 100],
            f"quantization config, colbert rescorer (hybrid prefetch, k={K}, pf={PF})",
            "score (0–100 scale)",
        )
        (out_dir / f"quality-by-config-colbert-{slug}.html").write_text(page(body))

        max_latency = max(
            max(colbert_slice[c]["latency"]["p50_ms"] for c in colbert_cats),
            max(colbert_slice[c]["latency"]["p99_ms"] for c in colbert_cats),
        )
        ticks = nice_ticks(max_latency)
        body = grouped_bar_chart(
            colbert_cats,
            [
                ("p50", BLUE50, {c: colbert_slice[c]["latency"]["p50_ms"] for c in colbert_cats}),
                ("p99", TEAL50, {c: colbert_slice[c]["latency"]["p99_ms"] for c in colbert_cats}),
            ],
            ticks[-1],
            ticks,
            f"quantization config, colbert rescorer (hybrid prefetch, k={K}, pf={PF})",
            "latency, ms",
            value_fmt="{:.0f}",
        )
        (out_dir / f"latency-by-config-{slug}.html").write_text(page(body))

        max_qps = max(colbert_slice[c]["throughput_qps"] for c in colbert_cats)
        ticks = nice_ticks(max_qps)
        body = grouped_bar_chart(
            colbert_cats,
            [("colbert qps", TEAL50, {c: colbert_slice[c]["throughput_qps"] for c in colbert_cats})],
            ticks[-1],
            ticks,
            f"quantization config, colbert rescorer (hybrid prefetch, k={K}, pf={PF}, "
            f"concurrency={next(iter(colbert_slice.values()))['concurrency']})",
            "throughput, qps",
            value_fmt="{:.0f}",
        )
        (out_dir / f"throughput-by-config-{slug}.html").write_text(page(body))

        print(f"[{dataset}] wrote charts ({len(colbert_cats)} colbert configs)")

    print(f"\nDone. Charts written to {out_dir}")


if __name__ == "__main__":
    main()
