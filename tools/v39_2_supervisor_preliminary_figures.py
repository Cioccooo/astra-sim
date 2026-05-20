#!/usr/bin/env python3
"""V39.2 clean preliminary supervisor figures.

This script only repackages validated V39/V39.1 outputs.  It does not change
model semantics, run ASTRA, or perform new topology sweeps.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
V39 = REPO / "results/moe_expert_trace_converter/v39_semantic_correct_model_selection"
V391 = REPO / "results/moe_expert_trace_converter/v39_1_routing_oracle_validation"
OUT = REPO / "results/moe_expert_trace_converter/v39_2_supervisor_preliminary_figures"

WORKLOAD_ORDER = [
    ("qwen_mmlu_machine_learning", "Qwen MMLU ML"),
    ("deepseek_mmlu_machine_learning", "DeepSeek MMLU ML"),
    ("qwen_livecodebench_execution", "Qwen LiveCodeBench"),
]

FIG1_METHODS = ["fixed random", "fair universal static", "prefill-informed OCS", "oracle"]
FIG2_METHODS = ["SON / torus", "fixed random", "fair universal static", "prefill-informed OCS", "oracle"]
COLORS = {
    "SON / torus": "#4E79A7",
    "fixed random": "#59A14F",
    "fair universal static": "#F28E2B",
    "prefill-informed OCS": "#E15759",
    "oracle": "#7B52AB",
    "ECMP-4": "#4E79A7",
    "all-shortest optimistic": "#B07AA1",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def workload_label(workload_id: str) -> str:
    return dict(WORKLOAD_ORDER).get(workload_id, workload_id)


def draw_grouped_bar(
    path: Path,
    title: str,
    subtitle: str,
    workloads: list[str],
    methods: list[str],
    values: dict[str, dict[str, float]],
    ylabel: str,
    ymax: float | None = None,
    baseline: float | None = None,
    annotations: dict[tuple[str, str], str] | None = None,
) -> None:
    width, height = 1900, 1050
    ml, mr, mt, mb = 165, 80, 120, 220
    pw, ph = width - ml - mr, height - mt - mb
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    maxv = ymax or max(max(row.values()) for row in values.values()) * 1.18
    maxv = max(maxv, 1e-9)

    def y(value: float) -> float:
        return mt + ph - (value / maxv) * ph

    draw.text((ml, 35), title, fill="#111827", font=font)
    draw.text((ml, 60), subtitle, fill="#B91C1C" if "not" in subtitle.lower() or "sensitivity" in subtitle.lower() else "#374151", font=font)
    for i in range(6):
        tick = maxv * i / 5
        yy = y(tick)
        draw.line((ml, yy, ml + pw, yy), fill="#E5E7EB")
        draw.text((45, yy - 7), f"{tick:.2f}", fill="#374151", font=font)
    draw.line((ml, mt, ml, mt + ph), fill="#111827")
    draw.line((ml, mt + ph, ml + pw, mt + ph), fill="#111827")
    if baseline is not None:
        yy = y(baseline)
        draw.line((ml, yy, ml + pw, yy), fill="#111827", width=3)
        draw.text((ml + pw - 155, yy - 18), f"baseline {baseline:g}", fill="#111827", font=font)

    for gi, workload in enumerate(workloads):
        group_w = pw / len(workloads)
        inner_w = group_w * 0.76
        start = ml + gi * group_w + (group_w - inner_w) / 2
        bar_w = inner_w / len(methods) * 0.82
        draw.text((ml + gi * group_w + group_w / 2 - 75, mt + ph + 28), workload, fill="#111827", font=font)
        for mi, method in enumerate(methods):
            value = values[workload][method]
            x = start + mi * (inner_w / len(methods))
            yy = y(value)
            draw.rectangle((x, yy, x + bar_w, mt + ph), fill=COLORS.get(method, "#777777"))
            if annotations and (workload, method) in annotations:
                draw.text((x - 5, yy - 18), annotations[(workload, method)], fill="#111827", font=font)

    lx, ly = ml, height - 95
    for method in methods:
        draw.rectangle((lx, ly, lx + 24, ly + 16), fill=COLORS.get(method, "#777777"))
        draw.text((lx + 32, ly + 1), method, fill="#111827", font=font)
        lx += 330
        if lx > width - 320:
            lx = ml
            ly += 32
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    img.save(path.with_suffix(".pdf"), "PDF", resolution=160.0)


def draw_table(path: Path, title: str, subtitle: str, rows: list[dict[str, Any]]) -> None:
    width, height = 2100, 1200
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((60, 35), title, fill="#111827", font=font)
    draw.text((60, 60), subtitle, fill="#374151", font=font)
    columns = [
        ("Workload", 210),
        ("Method", 210),
        ("Candidate", 235),
        ("Hash", 120),
        ("Bottleneck", 150),
        ("Sender ms", 120),
        ("Receiver ms", 130),
        ("Resource ms", 140),
        ("Same fair?", 105),
    ]
    x0, y0 = 55, 115
    row_h = 42
    x = x0
    for name, w in columns:
        draw.rectangle((x, y0, x + w, y0 + row_h), fill="#E5E7EB", outline="#9CA3AF")
        draw.text((x + 6, y0 + 13), name, fill="#111827", font=font)
        x += w
    for ri, row in enumerate(rows):
        y = y0 + row_h * (ri + 1)
        fill = "#FFFFFF" if ri % 2 == 0 else "#F9FAFB"
        values = [
            row["workload_label"],
            row["method"],
            row["selected_candidate_name"],
            row["graph_hash"][:10],
            row["bottleneck_type"],
            f"{float(row['sender_bottleneck_time_ms']):.2f}",
            f"{float(row['receiver_bottleneck_time_ms']):.2f}",
            f"{float(row['optical_resource_bottleneck_time_ms']):.2f}",
            str(row["same_as_fair_graph"]),
        ]
        x = x0
        for value, (_, w) in zip(values, columns):
            draw.rectangle((x, y, x + w, y + row_h), fill=fill, outline="#E5E7EB")
            draw.text((x + 6, y + 13), str(value)[:32], fill="#111827", font=font)
            x += w
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    img.save(path.with_suffix(".pdf"), "PDF", resolution=160.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    optical_rows = read_csv(V39 / "optical_circuit_reference_values.csv")
    torus_rows = read_csv(V391 / "torus_routing_sensitivity.csv")
    oracle_rows_raw = [r for r in read_csv(V391 / "oracle_vs_universal_audit.csv") if r.get("method")]

    by_workload_method = {
        (row["workload"], row["method"]): row
        for row in optical_rows
        if row["workload"] in dict(WORKLOAD_ORDER)
    }

    plotted_rows: list[dict[str, Any]] = []
    fig1_values: dict[str, dict[str, float]] = {}
    fig2_values: dict[str, dict[str, float]] = {}
    for workload_id, label in WORKLOAD_ORDER:
        fair_ms = float(by_workload_method[(workload_id, "fair universal static")]["optical_reference_ms"])
        son_ms = float(by_workload_method[(workload_id, "SON / torus")]["optical_reference_ms"])
        fig1_values[label] = {}
        fig2_values[label] = {}
        for method in FIG2_METHODS:
            row = by_workload_method[(workload_id, method)]
            ms = float(row["optical_reference_ms"])
            norm_fair = ms / fair_ms
            norm_son = ms / son_ms
            if method in FIG1_METHODS:
                fig1_values[label][method] = norm_fair
            fig2_values[label][method] = norm_son
            plotted_rows.append(
                {
                    "workload": workload_id,
                    "workload_label": label,
                    "method": method,
                    "candidate": row["candidate"],
                    "optical_reference_ms": ms,
                    "normalised_to_fair_universal_static": norm_fair,
                    "normalised_to_son_torus": norm_son,
                    "model": "optical circuit/capacity reference",
                }
            )

    torus_plot_rows = []
    fig3_values: dict[str, dict[str, float]] = {label: {} for _, label in WORKLOAD_ORDER}
    for row in torus_rows:
        if row["workload"] not in dict(WORKLOAD_ORDER):
            continue
        if row["routing_rule"] not in ("ecmp4", "all_shortest"):
            continue
        label = workload_label(row["workload"])
        method = "ECMP-4" if row["routing_rule"] == "ecmp4" else "all-shortest optimistic"
        ratio = float(row["son_over_fair_static_ratio"])
        fig3_values[label][method] = ratio
        torus_plot_rows.append(
            {
                "workload": row["workload"],
                "workload_label": label,
                "routing_rule": row["routing_rule"],
                "plot_label": method,
                "son_over_fair_static_ratio": ratio,
                "production_default": row["routing_rule"] == "ecmp4",
                "note": "all-shortest is optimistic sensitivity only" if row["routing_rule"] == "all_shortest" else "defensible default used by V39",
            }
        )

    bottleneck_rows = []
    for row in oracle_rows_raw:
        if row["workload"] not in dict(WORKLOAD_ORDER):
            continue
        if row["method"] not in ("fair universal static", "prefill-informed OCS", "oracle"):
            continue
        out = dict(row)
        out["workload_label"] = workload_label(row["workload"])
        bottleneck_rows.append(out)

    write_csv(OUT / "plotted_values.csv", plotted_rows)
    write_json(OUT / "plotted_values.json", plotted_rows)
    write_csv(OUT / "torus_routing_sensitivity_plot_values.csv", torus_plot_rows)
    write_json(OUT / "torus_routing_sensitivity_plot_values.json", torus_plot_rows)
    write_csv(OUT / "bottleneck_decomposition_plot_values.csv", bottleneck_rows)
    write_json(OUT / "bottleneck_decomposition_plot_values.json", bottleneck_rows)

    fig_dir = OUT / "figures"
    annotations = {}
    for _, label in WORKLOAD_ORDER:
        val = fig1_values[label]["prefill-informed OCS"]
        annotations[(label, "prefill-informed OCS")] = f"{(1 - val) * 100:+.1f}%"
        annotations[(label, "oracle")] = "UB"

    draw_grouped_bar(
        fig_dir / "figure_1_ocs_vs_strong_static.png",
        "Figure 1: Prefill-informed decode OCS vs strong static optical topology",
        "Optical circuit/capacity reference model. Normalised by fair universal static = 1.0.",
        [label for _, label in WORKLOAD_ORDER],
        FIG1_METHODS,
        fig1_values,
        "normalised decode communication time",
        baseline=1.0,
        ymax=max(1.18, max(max(v.values()) for v in fig1_values.values()) * 1.12),
        annotations=annotations,
    )
    draw_grouped_bar(
        fig_dir / "figure_2_topology_family_vs_torus.png",
        "Figure 2: Optical topology family benefit over SON torus",
        "Optical circuit/capacity reference model. Normalised by SON / torus = 1.0; torus gap is routing-sensitive.",
        [label for _, label in WORKLOAD_ORDER],
        FIG2_METHODS,
        fig2_values,
        "normalised decode communication time",
        baseline=1.0,
        ymax=1.12,
    )
    draw_grouped_bar(
        fig_dir / "figure_3_torus_routing_sensitivity.png",
        "Figure 3: SON torus sensitivity to routing assumption",
        "Routing sensitivity. All-shortest-path ECMP is optimistic and not the production default.",
        [label for _, label in WORKLOAD_ORDER],
        ["ECMP-4", "all-shortest optimistic"],
        fig3_values,
        "SON / fair-static time ratio",
        ymax=max(max(v.values()) for v in fig3_values.values()) * 1.18,
    )
    draw_table(
        fig_dir / "figure_4_oracle_bottleneck_explanation.png",
        "Figure 4: Why oracle and OCS are close to fair static",
        "Bottleneck / oracle explanation. Times are optical circuit/capacity reference model.",
        bottleneck_rows,
    )

    (OUT / "figure_caption_notes.md").write_text(
        """# Figure Caption Notes

## Figure 1

Main preliminary figure. Optical circuit/capacity reference model. Inference-only
prefill-informed decode communication. Selection uses prefill `trace[0]`;
evaluation uses decode `trace[1:]`. Oracle is a decode upper bound.

## Figure 2

Shows topology-family benefit over SON torus under ECMP-4. Caption must say:
the torus gap is routing-assumption sensitive and shrinks under all-shortest
optimistic routing.

## Figure 3

Routing sensitivity only. ECMP-4 is the defensible default used by V39;
all-shortest-path ECMP is optimistic.

## Figure 4

Explains why oracle is close to fair static: optical-resource bottleneck
dominates and many random-regular candidates are near-equivalent.
"""
    )

    readme = """# V39.2 Preliminary Supervisor Figures

## What Each Figure Shows

1. Figure 1 compares prefill-informed OCS against strong static optical
   baselines. This is the first figure to show.
2. Figure 2 shows that expander-like optical topologies outperform SON torus
   under ECMP-4.
3. Figure 3 shows that the torus gap is routing-assumption sensitive.
4. Figure 4 explains why oracle and prefill-informed OCS are close to fair
   universal static.

## What These Figures Can Claim

- Figure 1/2 are optical circuit reference results, not native ASTRA optical
  results.
- The evaluated stage is decode `trace[1:]`.
- Prefill-informed OCS uses only prefill `trace[0]` for topology selection.
- Optical methods use the same degree-4, 400Gb/s/circuit, 1.6Tb/s/GPU budget.

## What These Figures Cannot Claim

- They are not paper-final.
- They are not full serving latency.
- They are not MoE training.
- They are not native ASTRA optical-circuit results.

## Why Figure A/1 Is Main

It is the controlled optical-only comparison. It does not mix EN packet
GraphTopology with optical circuit reference bars.

## Why V39 Figure B Is Reference Only

EN is native ASTRA packet folded-Clos electrical reference. Optical methods use
the optical circuit/capacity reference. This is explicitly mixed semantics and
should not be presented as same-model fairness.

## Why Current ASTRA GraphTopology Is Sensitivity Only

Current ASTRA GraphTopology models packet/store-and-forward routing. SON/RON can
include intermediate GPUs on routes, so it is not transparent optical OCS
semantics.

## Key Caveat To Say Out Loud

Prefill-informed OCS only wins DeepSeek MMLU slightly; Qwen MMLU ties and Qwen
LiveCodeBench loses slightly. The robust result is that expander-like optical
topologies beat torus under ECMP-4, but the torus gap shrinks under optimistic
all-shortest routing.
"""
    (OUT / "README.md").write_text(readme)

    summary = {
        "all_figures_generated": True,
        "first_figure_to_show": "Figure 1: OCS vs strong static optical baseline",
        "prefill_ocs_vs_fair_static": {
            label: fig1_values[label]["prefill-informed OCS"] < 1.0
            for _, label in WORKLOAD_ORDER
        },
        "torus_gap_survives_ecmp4": {
            row["workload_label"]: row["son_over_fair_static_ratio"]
            for row in torus_plot_rows
            if row["routing_rule"] == "ecmp4"
        },
        "caveat": "These are optical circuit reference results, not native ASTRA optical results; current ASTRA GraphTopology is packet-routing sensitivity only.",
    }
    write_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
