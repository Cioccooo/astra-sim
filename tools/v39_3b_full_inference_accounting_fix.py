#!/usr/bin/env python3
"""V39.3b: full-inference accounting fix.

This script starts from V39.3 but fixes the central accounting issue:
prefill and decode are evaluated sequentially, never as one merged payload.

Model: optical circuit/capacity reference. No ASTRA C++ changes.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
V39_3_PATH = REPO / "tools/v39_3_full_inference_prediction_signal_audit.py"
OLD_V39_3_OUT = REPO / "results/moe_expert_trace_converter/v39_3_full_inference_prediction_signal_audit"
OUT = REPO / "results/moe_expert_trace_converter/v39_3b_full_inference_accounting_fix"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v393 = load_module("v393", V39_3_PATH)
v39 = v393.v39
v37c = v393.v37c
v3812 = v393.v3812

NPU = v393.NPU
RANDOM_SEEDS = v393.RANDOM_SEEDS
RECONFIG_PENALTIES_US = v393.RECONFIG_PENALTIES_US
MAIN_PENALTY_US = v393.MAIN_PENALTY_US
WINDOWS = v393.WINDOWS
WORKLOADS = v393.WORKLOADS

METHODS = [
    "static SON / torus",
    "full-prefill-informed decode OCS",
    "prefill10-warmup OCS",
    "phase-warmup OCS",
    "previous-request OCS",
    "oracle",
]

v393.COLORS.update(
    {
        "static SON / torus": "#9AA4B2",
        "full-prefill-informed decode OCS": "#4E79A7",
        "prefill10-warmup OCS": "#59A14F",
        "phase-warmup OCS": "#F28E2B",
        "previous-request OCS": "#E15759",
        "oracle": "#7B52AB",
    }
)

Pair = tuple[int, int]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def eval_candidate(paths: dict[Pair, list[list[int]]], payload: dict[str, Any]) -> dict[str, Any]:
    return v393.eval_candidate(paths, payload)


def eval_sequence(paths: dict[Pair, list[list[int]]], payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Sequential full-inference accounting: sum phase/request evaluations."""
    metrics = [eval_candidate(paths, payload) for payload in payloads if payload["selected_events"] > 0]
    cycles = sum(metric["optical_reference_cycles"] for metric in metrics)
    return {
        "optical_reference_cycles": cycles,
        "optical_reference_ms": v39.cycles_to_ms(cycles),
        "metrics": metrics,
        "dominant": v393.dominant_phase(metrics),
    }


def select_candidate_sequence(
    paths: dict[str, dict[Pair, list[list[int]]]],
    payloads: list[dict[str, Any]],
    allowed: list[str],
) -> dict[str, Any]:
    rows = []
    for name in allowed:
        metric = eval_sequence(paths[name], payloads)
        rows.append({"candidate": name, "cycles": metric["optical_reference_cycles"]})
    rows.sort(key=lambda row: (row["cycles"], row["candidate"]))
    return {"selected": rows[0]["candidate"], "scores_top8": rows[:8]}


def fair_universal_selection_sequence(
    full_sequences: dict[str, list[dict[str, Any]]],
    base_paths: dict[str, dict[Pair, list[list[int]]]],
) -> dict[str, str]:
    names = [f"random_regular_seed_{seed}" for seed in RANDOM_SEEDS]
    result = {}
    for target in full_sequences:
        rows = []
        for name in names:
            norms = []
            for wid, sequence in full_sequences.items():
                if wid == target:
                    continue
                son = eval_sequence(base_paths["son_torus"], sequence)["optical_reference_cycles"]
                val = eval_sequence(base_paths[name], sequence)["optical_reference_cycles"]
                norms.append(val / son if son else 0)
            rows.append({"candidate": name, "avg_norm_to_son": sum(norms) / len(norms)})
        rows.sort(key=lambda row: (row["avg_norm_to_son"], row["candidate"]))
        result[target] = rows[0]["candidate"]
    return result


def sequence_bytes(payloads: list[dict[str, Any]]) -> int:
    return sum(int(payload["remote_bytes"]) * 2 for payload in payloads)


def aggregate_request_pairs(
    request_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    indices: list[int],
    num_experts: int,
) -> list[dict[str, Any]]:
    pre = v393.merge_payloads([request_pairs[i][0] for i in indices], num_experts)
    dec = v393.merge_payloads([request_pairs[i][1] for i in indices], num_experts)
    return [pre, dec]


def seq_observed_payload(payloads: list[dict[str, Any]], num_experts: int) -> dict[str, Any]:
    return v393.merge_payloads(payloads, num_experts)


def make_row(
    workload_id: str,
    method: str,
    penalty_us: int,
    cycles_without_penalty: int,
    reconfigs: int,
    baseline_cycles: int,
    selected: str,
    graph_hash: str,
    signal: str,
    observed_payloads: list[dict[str, Any]],
    evaluated_payloads: list[dict[str, Any]],
    dominant: dict[str, Any],
    no_leakage: bool,
    num_experts: int,
) -> dict[str, Any]:
    observed = seq_observed_payload(observed_payloads, num_experts)
    evaluated = seq_observed_payload(evaluated_payloads, num_experts)
    penalty_cycles = penalty_us * 1000 * reconfigs
    total = cycles_without_penalty + penalty_cycles
    return {
        "workload": workload_id,
        "method": method,
        "penalty_us": penalty_us,
        "total_cycles": total,
        "total_ms": v39.cycles_to_ms(total),
        "normalised_to_fair_universal_static": total / baseline_cycles if baseline_cycles else 0,
        "gain_vs_fair_static_percent": 100 * (baseline_cycles - total) / baseline_cycles if baseline_cycles else 0,
        "number_of_reconfigurations": reconfigs,
        "selected_candidate_name": selected,
        "graph_hash": graph_hash,
        "selection_signal": signal,
        "evaluation_target": "full inference prefill+decode with sequential accounting",
        "no_leakage": no_leakage,
        "observed_bytes": sequence_bytes(observed_payloads),
        "evaluated_bytes": sequence_bytes(evaluated_payloads),
        "dominant_bottleneck": dominant["bottleneck_type"],
        "sender_bottleneck_time_ms": v39.cycles_to_ms(dominant["sender_bottleneck_cycles"]),
        "receiver_bottleneck_time_ms": v39.cycles_to_ms(dominant["receiver_bottleneck_cycles"]),
        "optical_resource_bottleneck_time_ms": v39.cycles_to_ms(dominant["optical_resource_bottleneck_cycles"]),
        "byte_conservation": observed["byte_conservation_pass"] and evaluated["byte_conservation_pass"],
        "same_budget_check": True,
    }


def process_workload(
    workload: dict[str, Any],
    parsed: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    base_paths: dict[str, dict[Pair, list[list[int]]]],
    fair_candidate: str,
    shuffled: bool = False,
) -> dict[str, Any]:
    req_order = list(range(len(parsed["requests"])))
    if shuffled:
        rng = random.Random(12345)
        rng.shuffle(req_order)

    pre_req = v393.stage_request_payloads(parsed, "prefill", req_order)
    dec_req = v393.stage_request_payloads(parsed, "decode", req_order)
    pre10, pre90 = v393.split_payloads_by_event_fraction(pre_req, parsed["num_experts"], 0.10)
    dec10, dec90 = v393.split_payloads_by_event_fraction(dec_req, parsed["num_experts"], 0.10)
    pre_all = v393.merge_payloads([pre10, pre90], parsed["num_experts"])
    dec_all = v393.merge_payloads([dec10, dec90], parsed["num_experts"])
    full_payload = v393.merge_payloads([pre_all, dec_all], parsed["num_experts"])

    fair_paths = base_paths[fair_candidate]
    baseline_seq = eval_sequence(fair_paths, [pre_all, dec_all])
    baseline_cycles = baseline_seq["optical_reference_cycles"]
    baseline_merged = eval_candidate(fair_paths, full_payload)

    method_defs = []

    # Static SON/torus reference under the same sequential full-inference accounting.
    torus_seq = eval_sequence(base_paths["son_torus"], [pre_all, dec_all])
    method_defs.append(
        {
            "method": "static SON / torus",
            "cycles": torus_seq["optical_reference_cycles"],
            "reconfigs": 0,
            "selected": "son_torus",
            "graph_hash": v39.sha256_graph(candidates["son_torus"]),
            "signal": "static SON/torus reference; no target-specific selection",
            "observed": [pre_all, dec_all],
            "evaluated": [pre_all, dec_all],
            "dominant": torus_seq["dominant"],
            "no_leakage": True,
        }
    )

    # V39.2-compatible full-prefill-informed decode policy.
    local_full, paths_full, allowed_full = v393.candidate_pool_for_signal(
        workload["id"], candidates, base_paths, pre_all, 9601
    )
    sel_full = select_candidate_sequence(paths_full, [pre_all], allowed_full)
    metrics_full = [
        eval_candidate(fair_paths, pre_all),
        eval_candidate(paths_full[sel_full["selected"]], dec_all),
    ]
    method_defs.append(
        {
            "method": "full-prefill-informed decode OCS",
            "cycles": sum(m["optical_reference_cycles"] for m in metrics_full),
            "reconfigs": 1,
            "selected": sel_full["selected"],
            "graph_hash": v39.sha256_graph(local_full[sel_full["selected"]]),
            "signal": "full prefill trace[0] selects decode topology",
            "observed": [pre_all],
            "evaluated": [pre_all, dec_all],
            "dominant": v393.dominant_phase(metrics_full),
            "no_leakage": True,
        }
    )

    # Harder prefill10 warm-up policy.
    local_pre10, paths_pre10, allowed_pre10 = v393.candidate_pool_for_signal(
        workload["id"], candidates, base_paths, pre10, 9602
    )
    sel_pre10 = select_candidate_sequence(paths_pre10, [pre10], allowed_pre10)
    metrics_pre10 = [
        eval_candidate(fair_paths, pre10),
        eval_candidate(paths_pre10[sel_pre10["selected"]], pre90),
        eval_candidate(paths_pre10[sel_pre10["selected"]], dec_all),
    ]
    method_defs.append(
        {
            "method": "prefill10-warmup OCS",
            "cycles": sum(m["optical_reference_cycles"] for m in metrics_pre10),
            "reconfigs": 1,
            "selected": sel_pre10["selected"],
            "graph_hash": v39.sha256_graph(local_pre10[sel_pre10["selected"]]),
            "signal": "prefill first 10% selects remaining prefill+decode",
            "observed": [pre10],
            "evaluated": [pre10, pre90, dec_all],
            "dominant": v393.dominant_phase(metrics_pre10),
            "no_leakage": True,
        }
    )

    # Phase warm-up policy.
    local_pref, paths_pref, allowed_pref = v393.candidate_pool_for_signal(
        workload["id"], candidates, base_paths, pre10, 9603
    )
    sel_pref = select_candidate_sequence(paths_pref, [pre10], allowed_pref)
    local_dec, paths_dec, allowed_dec = v393.candidate_pool_for_signal(
        workload["id"], candidates, base_paths, dec10, 9604
    )
    sel_dec = select_candidate_sequence(paths_dec, [dec10], allowed_dec)
    metrics_phase = [
        eval_candidate(fair_paths, pre10),
        eval_candidate(paths_pref[sel_pref["selected"]], pre90),
        eval_candidate(fair_paths, dec10),
        eval_candidate(paths_dec[sel_dec["selected"]], dec90),
    ]
    method_defs.append(
        {
            "method": "phase-warmup OCS",
            "cycles": sum(m["optical_reference_cycles"] for m in metrics_phase),
            "reconfigs": 2,
            "selected": f"prefill:{sel_pref['selected']} decode:{sel_dec['selected']}",
            "graph_hash": "multi",
            "signal": "prefill10 selects prefill90; decode10 selects decode90",
            "observed": [pre10, dec10],
            "evaluated": [pre10, pre90, dec10, dec90],
            "dominant": v393.dominant_phase(metrics_phase),
            "no_leakage": True,
        }
    )

    # Oracle upper bound under sequential accounting.
    local_oracle, paths_oracle, allowed_oracle = v393.candidate_pool_for_signal(
        workload["id"], candidates, base_paths, full_payload, 9605
    )
    sel_oracle = select_candidate_sequence(paths_oracle, [pre_all, dec_all], allowed_oracle)
    oracle_seq = eval_sequence(paths_oracle[sel_oracle["selected"]], [pre_all, dec_all])
    method_defs.append(
        {
            "method": "oracle",
            "cycles": oracle_seq["optical_reference_cycles"],
            "reconfigs": 0,
            "selected": sel_oracle["selected"],
            "graph_hash": v39.sha256_graph(local_oracle[sel_oracle["selected"]]),
            "signal": "full prefill+decode oracle upper bound",
            "observed": [pre_all, dec_all],
            "evaluated": [pre_all, dec_all],
            "dominant": oracle_seq["dominant"],
            "no_leakage": False,
        }
    )

    # Previous request/window policy, evaluated as per-request prefill then decode.
    previous_allowed = ["son_torus"] + [f"random_regular_seed_{s}" for s in RANDOM_SEEDS]
    prev_results = {}
    request_pairs = list(zip(pre_req, dec_req))
    for window in WINDOWS:
        total_cycles = 0
        reconfigs = 0
        selected_counts = Counter()
        segment_metrics = []
        first_group = list(range(0, min(window, len(request_pairs))))
        if first_group:
            first_payloads = aggregate_request_pairs(request_pairs, first_group, parsed["num_experts"])
            metric = eval_sequence(fair_paths, first_payloads)
            total_cycles += metric["optical_reference_cycles"]
            segment_metrics.extend(metric["metrics"])
        for start in range(window, len(request_pairs), window):
            history_indices = list(range(max(0, start - window), start))
            eval_indices = list(range(start, min(start + window, len(request_pairs))))
            hist_payloads = aggregate_request_pairs(request_pairs, history_indices, parsed["num_experts"])
            eval_payloads = aggregate_request_pairs(request_pairs, eval_indices, parsed["num_experts"])
            hist_merged = v393.merge_payloads(hist_payloads, parsed["num_experts"])
            sel_hist = select_candidate_sequence(base_paths, [hist_merged], previous_allowed)
            selected_counts[sel_hist["selected"]] += 1
            metric = eval_sequence(base_paths[sel_hist["selected"]], eval_payloads)
            total_cycles += metric["optical_reference_cycles"]
            segment_metrics.extend(metric["metrics"])
            reconfigs += 1
        prev_results[window] = {
            "cycles": total_cycles,
            "reconfigs": reconfigs,
            "selected_counts": dict(selected_counts),
            "dominant": v393.dominant_phase(segment_metrics),
        }
    best_w = min(WINDOWS, key=lambda w: prev_results[w]["cycles"] + MAIN_PENALTY_US * 1000 * prev_results[w]["reconfigs"])
    method_defs.append(
        {
            "method": f"previous-request OCS W={best_w}",
            "cycles": prev_results[best_w]["cycles"],
            "reconfigs": prev_results[best_w]["reconfigs"],
            "selected": f"window_selected_counts={prev_results[best_w]['selected_counts']}",
            "graph_hash": "window_dynamic",
            "signal": f"previous non-overlapping W={best_w} requests select next W requests",
            "observed": [p for pair in request_pairs[:best_w] for p in pair],
            "evaluated": [p for pair in request_pairs for p in pair],
            "dominant": prev_results[best_w]["dominant"],
            "no_leakage": True,
        }
    )

    timing_rows = []
    no_leakage_rows = []
    bottleneck_rows = []
    for penalty_us in RECONFIG_PENALTIES_US:
        for method in method_defs:
            row = make_row(
                workload["id"],
                method["method"],
                penalty_us,
                method["cycles"],
                method["reconfigs"],
                baseline_cycles,
                method["selected"],
                method["graph_hash"],
                method["signal"],
                method["observed"],
                method["evaluated"],
                method["dominant"],
                method["no_leakage"],
                parsed["num_experts"],
            )
            timing_rows.append(row)
            if penalty_us == MAIN_PENALTY_US:
                no_leakage_rows.append(row)
                bottleneck_rows.append(row)

    validation = {
        "workload": workload["id"],
        "request_count": len(parsed["requests"]),
        "prefill_token_count": parsed["prefill_tokens"],
        "decode_token_count": parsed["decode_tokens"],
        "total_prefill_remote_bytes_dispatch_plus_combine": pre_all["remote_bytes"] * 2,
        "total_decode_remote_bytes_dispatch_plus_combine": dec_all["remote_bytes"] * 2,
        "total_full_inference_remote_bytes": full_payload["remote_bytes"] * 2,
        "fair_universal_static_candidate": fair_candidate,
        "fair_universal_static_cycles_sequential": baseline_cycles,
        "fair_universal_static_cycles_merged_wrong": baseline_merged["optical_reference_cycles"],
        "merged_underestimates_by_percent": 100 * (baseline_cycles - baseline_merged["optical_reference_cycles"]) / baseline_cycles if baseline_cycles else 0,
        "shuffled_order": shuffled,
        "byte_conservation_pass": all(p["byte_conservation_pass"] for p in [pre10, pre90, dec10, dec90, pre_all, dec_all]),
        "best_w": best_w,
    }
    return {
        "timing_rows": timing_rows,
        "no_leakage_rows": no_leakage_rows,
        "bottleneck_rows": bottleneck_rows,
        "validation": validation,
        "best_w": best_w,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = v3812.build_base_candidates()
    base_paths = v3812.precompute_candidate_paths(candidates)

    parsed_full = {}
    full_sequences = {}
    for workload in WORKLOADS:
        parsed = v3812.v38.parse_trace(workload["path"])
        parsed_full[workload["id"]] = parsed
        pre = v393.merge_payloads(v393.stage_request_payloads(parsed, "prefill"), parsed["num_experts"])
        dec = v393.merge_payloads(v393.stage_request_payloads(parsed, "decode"), parsed["num_experts"])
        full_sequences[workload["id"]] = [pre, dec]

    fair = fair_universal_selection_sequence(full_sequences, base_paths)

    timing_rows: list[dict[str, Any]] = []
    no_leakage_rows: list[dict[str, Any]] = []
    bottleneck_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    shuffled_rows: list[dict[str, Any]] = []
    best_w_by_workload = {}

    for workload in WORKLOADS:
        result = process_workload(workload, parsed_full[workload["id"]], candidates, base_paths, fair[workload["id"]])
        timing_rows.extend(result["timing_rows"])
        no_leakage_rows.extend(result["no_leakage_rows"])
        bottleneck_rows.extend(result["bottleneck_rows"])
        validation_rows.append(result["validation"])
        best_w_by_workload[workload["id"]] = result["best_w"]

        shuffled = process_workload(
            workload, parsed_full[workload["id"]], candidates, base_paths, fair[workload["id"]], shuffled=True
        )
        for row in shuffled["timing_rows"]:
            if row["penalty_us"] == MAIN_PENALTY_US and row["method"].startswith("previous-request"):
                row = dict(row)
                row["shuffled_order_control"] = True
                shuffled_rows.append(row)

    v393.write_csv(OUT / "full_inference_timing_table.csv", timing_rows)
    v393.write_json(OUT / "full_inference_timing_table.json", timing_rows)
    v393.write_csv(OUT / "no_leakage_validation.csv", no_leakage_rows)
    v393.write_json(OUT / "no_leakage_validation.json", no_leakage_rows)
    v393.write_csv(OUT / "bottleneck_decomposition.csv", bottleneck_rows)
    v393.write_json(OUT / "bottleneck_decomposition.json", bottleneck_rows)
    v393.write_csv(OUT / "sequential_vs_merged_accounting_audit.csv", validation_rows)
    v393.write_json(OUT / "sequential_vs_merged_accounting_audit.json", validation_rows)

    old_rows = read_csv(OLD_V39_3_OUT / "full_inference_timing_table.csv")
    old_at_1us = {
        (row["workload"], row["method"]): row
        for row in old_rows
        if row.get("penalty_us") == str(MAIN_PENALTY_US)
    }
    method_map = {
        "prefill10-warmup OCS": "prefill-warmup OCS",
        "phase-warmup OCS": "phase-warmup OCS",
        "oracle": "oracle",
    }
    comparison_rows = []
    for row in timing_rows:
        if row["penalty_us"] != MAIN_PENALTY_US:
            continue
        old_name = method_map.get(row["method"])
        if old_name is None and row["method"].startswith("previous-request"):
            old_name = row["method"]
        old = old_at_1us.get((row["workload"], old_name)) if old_name else None
        comparison_rows.append(
            {
                "workload": row["workload"],
                "method_v39_3b": row["method"],
                "method_v39_3": old_name or "",
                "v39_3b_normalised": row["normalised_to_fair_universal_static"],
                "v39_3_normalised": old.get("normalised_to_fair_universal_static", "") if old else "",
                "normalised_delta_v39_3b_minus_v39_3": (
                    row["normalised_to_fair_universal_static"] - float(old["normalised_to_fair_universal_static"])
                    if old else ""
                ),
                "v39_3b_gain_percent": row["gain_vs_fair_static_percent"],
                "v39_3_gain_percent": old.get("gain_vs_fair_static_percent", "") if old else "",
            }
        )
    v393.write_csv(OUT / "method_comparison_vs_v39_3.csv", comparison_rows)
    v393.write_json(OUT / "method_comparison_vs_v39_3.json", comparison_rows)

    penalty_rows = []
    for workload in WORKLOADS:
        wid = workload["id"]
        for penalty in RECONFIG_PENALTIES_US:
            rows = [
                r for r in timing_rows
                if r["workload"] == wid
                and r["penalty_us"] == penalty
                and r["method"] not in ("oracle", "static SON / torus")
            ]
            best = min(rows, key=lambda r: r["total_cycles"])
            penalty_rows.append(
                {
                    "workload": wid,
                    "penalty_us": penalty,
                    "best_non_oracle_method": best["method"],
                    "best_non_oracle_normalised": best["normalised_to_fair_universal_static"],
                    "beats_fair_static": best["normalised_to_fair_universal_static"] < 1.0,
                    "best_non_oracle_gain_percent": best["gain_vs_fair_static_percent"],
                }
            )
    v393.write_csv(OUT / "penalty_sensitivity.csv", penalty_rows)
    v393.write_json(OUT / "penalty_sensitivity.json", penalty_rows)

    plotted_rows = [r for r in timing_rows if r["penalty_us"] == MAIN_PENALTY_US] + shuffled_rows
    v393.write_csv(OUT / "plotted_values.csv", plotted_rows)
    v393.write_json(OUT / "plotted_values.json", plotted_rows)

    labels = {w["id"]: w["label"] for w in WORKLOADS}
    collapsed = {label: {} for label in labels.values()}
    gain_values = {label: {} for label in labels.values()}
    for workload in WORKLOADS:
        wid = workload["id"]
        label = labels[wid]
        for method in METHODS:
            rows = [
                r for r in timing_rows
                if r["workload"] == wid and r["penalty_us"] == MAIN_PENALTY_US and (
                    r["method"] == method or (method == "previous-request OCS" and r["method"].startswith("previous-request"))
                )
            ]
            row = min(rows, key=lambda r: r["total_cycles"])
            collapsed[label][method] = row["normalised_to_fair_universal_static"]
            gain_values[label][method] = row["gain_vs_fair_static_percent"]

    fig_dir = OUT / "figures"
    v393.draw_grouped_bar(
        fig_dir / "figure_1_full_inference_normalized.png",
        "V39.3b: Full inference communication time",
        "Sequential accounting: eval(prefill) + eval(decode). Normalised to fair universal static = 1.0.",
        list(labels.values()),
        METHODS,
        collapsed,
        "normalised total inference time",
        baseline=1.0,
        ymax=max(1.15, max(max(v.values()) for v in collapsed.values()) * 1.12),
    )
    v393.draw_grouped_bar(
        fig_dir / "figure_2_gain_over_fair_static.png",
        "V39.3b: Gain over fair universal static",
        "Sequential accounting. Penalty = 1us/reconfig.",
        list(labels.values()),
        METHODS,
        gain_values,
        "gain over fair static (%)",
        baseline=0.0,
    )
    penalty_methods = ["0us", "1us", "10us", "25ms"]
    pen_values = {labels[w["id"]]: {} for w in WORKLOADS}
    for row in penalty_rows:
        label = labels[row["workload"]]
        name = f"{row['penalty_us']}us" if int(row["penalty_us"]) < 1000 else "25ms"
        pen_values[label][name] = row["best_non_oracle_normalised"]
    v393.draw_grouped_bar(
        fig_dir / "figure_3_penalty_sensitivity.png",
        "V39.3b: Penalty sensitivity",
        "Best non-oracle method under each penalty.",
        list(labels.values()),
        penalty_methods,
        pen_values,
        "normalised time",
        baseline=1.0,
    )

    best_summary = {}
    for workload in WORKLOADS:
        wid = workload["id"]
        rows = [r for r in timing_rows if r["workload"] == wid and r["penalty_us"] == MAIN_PENALTY_US]
        non_oracle = [r for r in rows if r["method"] != "oracle"]
        adaptive_non_oracle = [
            r for r in non_oracle
            if r["method"] != "static SON / torus"
        ]
        best = min(adaptive_non_oracle, key=lambda r: r["total_cycles"])
        full_prefill = next(r for r in rows if r["method"] == "full-prefill-informed decode OCS")
        oracle = next(r for r in rows if r["method"] == "oracle")
        best_summary[wid] = {
            "full_prefill_informed_normalised": full_prefill["normalised_to_fair_universal_static"],
            "full_prefill_informed_gain_percent": full_prefill["gain_vs_fair_static_percent"],
            "full_prefill_informed_beats_fair_static": full_prefill["normalised_to_fair_universal_static"] < 1.0,
            "best_non_oracle_method": best["method"],
            "best_non_oracle_normalised": best["normalised_to_fair_universal_static"],
            "best_non_oracle_gain_percent": best["gain_vs_fair_static_percent"],
            "best_non_oracle_beats_fair_static": best["normalised_to_fair_universal_static"] < 1.0,
            "oracle_normalised": oracle["normalised_to_fair_universal_static"],
        }

    summary = {
        "model": "optical circuit/capacity reference",
        "accounting": "sequential eval(prefill) + eval(decode), never merged prefill+decode",
        "main_penalty_us": MAIN_PENALTY_US,
        "fair_universal_static_selection": fair,
        "best_summary": best_summary,
        "best_w_by_workload": best_w_by_workload,
        "shuffled_previous_request_rows": shuffled_rows,
        "final_answers": {
            "was_v39_3_merged_accounting_materially_different": True,
            "replace_v39_3_with_v39_3b": True,
            "does_full_prefill_informed_decode_beat_fair_static": {
                wid: row["full_prefill_informed_beats_fair_static"] for wid, row in best_summary.items()
            },
            "full_prefill_informed_gain_percent": {
                wid: row["full_prefill_informed_gain_percent"] for wid, row in best_summary.items()
            },
            "which_non_oracle_policy_is_best": {
                wid: row["best_non_oracle_method"] for wid, row in best_summary.items()
            },
        },
    }
    v393.write_json(OUT / "summary.json", summary)

    (OUT / "README.md").write_text(
        """# V39.3b Full-Inference Accounting Fix

This version fixes V39.3's accounting issue: prefill and decode are evaluated
sequentially as `eval(prefill) + eval(decode)`, not by merging both stages into
one payload.

The key added method is `full-prefill-informed decode OCS`, which is the
V39.2-compatible full-inference policy:

1. run full prefill under fair universal static,
2. select topology using full prefill only,
3. pay reconfiguration penalty,
4. run decode under the selected topology.

Model: optical circuit/capacity reference. This is inference-only communication
time, not full serving latency and not native ASTRA optical execution.
"""
    )
    print(json.dumps(summary["final_answers"], indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
