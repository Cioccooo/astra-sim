#!/usr/bin/env python3
"""V37 four-workload static topology candidate-family validation.

This reuses the validated V36 aggregated native ASTRA pipeline and extends it to
four HF-derived MoE prefill workloads. The goal is to decompose whether static
RON gains come from torus weakness, random-regular family strength,
calibration selection, greedy demand-aware construction, or oracle advantage.
"""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation"
V36_PATH = REPO / "tools/v36_static_ron_one_workload_validation.py"

spec = importlib.util.spec_from_file_location("v36_static_ron", V36_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {V36_PATH}")
v36 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v36)
v36.OUT = OUT


WORKLOADS = [
    {
        "id": "qwen_mmlu_machine_learning",
        "label": "Qwen MMLU machine_learning",
        "path": Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/mmlu/machine_learning"),
    },
    {
        "id": "qwen_livecodebench_execution",
        "label": "Qwen LiveCodeBench execution",
        "path": Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/livecodebench/execution"),
    },
    {
        "id": "qwen_mmlu_zh_cn_anatomy",
        "label": "Qwen MMLU_ZH_CN anatomy",
        "path": Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/mmlu_ZH_CN/anatomy"),
    },
    {
        "id": "deepseek_livecodebench_execution",
        "label": "DeepSeek LiveCodeBench execution",
        "path": Path("/Users/dfx/Python/trace/cognitivecomputations/DeepSeek-R1-AWQ/livecodebench/execution"),
    },
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def candidate_type(name: str) -> str:
    if name == "son_torus":
        return "torus"
    if name == "calibration_greedy":
        return "greedy_calibration"
    if name == "evaluation_greedy":
        return "greedy_evaluation"
    if name.startswith("random_regular_seed_"):
        return "random_regular"
    return "unknown"


def fingerprint_label(stats: dict[str, Any]) -> str:
    g = float(stats["gini"])
    top16 = float(stats["top16_share"])
    if g >= 0.30 or top16 >= 0.10:
        return "hot-pair skewed"
    if g >= 0.15 or top16 >= 0.04:
        return "moderately structured"
    return "broad / near-uniform"


def matrix_from_payload(payload: dict[str, Any], phase: str) -> list[list[int]]:
    return payload[phase + "_matrix"]


def score_candidates(
    candidates: dict[str, dict[str, Any]],
    dispatch: list[list[int]],
    combine: list[list[int]],
    allowed: list[str],
    path_cache: dict[str, dict[tuple[int, int], list[list[int]]]],
) -> list[dict[str, Any]]:
    rows = []
    for name in allowed:
        score = max(
            link_load_estimate_cached(path_cache[name], dispatch)["max_link_load_bytes"],
            link_load_estimate_cached(path_cache[name], combine)["max_link_load_bytes"],
        )
        rows.append(
            {
                "name": name,
                "candidate_type": candidate_type(name),
                "score_max_link_load_bytes": score,
            }
        )
    return sorted(rows, key=lambda item: (item["score_max_link_load_bytes"], item["name"]))


def precompute_selected_paths(graph: dict[str, Any]) -> dict[tuple[int, int], list[list[int]]]:
    paths: dict[tuple[int, int], list[list[int]]] = {}
    for src in range(v36.NPU_COUNT):
        for dst in range(v36.NPU_COUNT):
            if src != dst:
                paths[(src, dst)] = v36.selected_paths(graph, src, dst, 4)
    return paths


def link_load_estimate_cached(
    paths_by_pair: dict[tuple[int, int], list[list[int]]],
    matrix: list[list[int]],
) -> dict[str, Any]:
    loads: dict[tuple[int, int], int] = defaultdict(int)
    selected_counts: list[int] = []
    hop_counts: list[int] = []
    byte_weighted_hops = 0
    total_bytes = 0
    for src in range(v36.NPU_COUNT):
        for dst in range(v36.NPU_COUNT):
            size = matrix[src][dst]
            if src == dst or size <= 0:
                continue
            paths = paths_by_pair[(src, dst)]
            if not paths:
                raise RuntimeError(f"no path {src}->{dst}")
            selected_counts.append(len(paths))
            for path, subbytes in zip(paths, v36.split_bytes(size, len(paths))):
                hop_count = len(path) - 1
                hop_counts.append(hop_count)
                byte_weighted_hops += subbytes * hop_count
                total_bytes += subbytes
                for u, vv in zip(path, path[1:]):
                    loads[(u, vv)] += subbytes
    max_load = max(loads.values()) if loads else 0
    vals = list(loads.values())
    return {
        "selected_path_count_min": min(selected_counts) if selected_counts else 0,
        "selected_path_count_median": statistics.median(selected_counts) if selected_counts else 0,
        "selected_path_count_mean": statistics.mean(selected_counts) if selected_counts else 0,
        "selected_path_count_max": max(selected_counts) if selected_counts else 0,
        "average_hop_count": statistics.mean(hop_counts) if hop_counts else 0,
        "byte_weighted_average_hop_count": byte_weighted_hops / total_bytes if total_bytes else 0,
        "max_link_load_bytes": max_load,
        "median_link_load_bytes": statistics.median(vals) if vals else 0,
        "average_link_load_bytes": statistics.mean(vals) if vals else 0,
        "fluid_cycles": int(max_load / v36.ASTRA_BYTES_PER_NS) if max_load else 0,
        "hot_links": [
            {"src": u, "dst": vv, "bytes": b}
            for (u, vv), b in sorted(loads.items(), key=lambda x: -x[1])[:8]
        ],
    }


def tiny_subchunk_audit_cached(
    paths_by_pair: dict[tuple[int, int], list[list[int]]],
    matrix: list[list[int]],
) -> dict[str, Any]:
    subchunks: list[int] = []
    selected_counts: list[int] = []
    for src in range(v36.NPU_COUNT):
        for dst in range(v36.NPU_COUNT):
            size = matrix[src][dst]
            if src == dst or size <= 0:
                continue
            paths = paths_by_pair[(src, dst)]
            selected_counts.append(len(paths))
            subchunks.extend(v36.split_bytes(size, len(paths)))
    return {
        "one_cycle_threshold_bytes": v36.ONE_CYCLE_THRESHOLD_BYTES,
        "subchunk_bytes_min": min(subchunks) if subchunks else 0,
        "subchunk_bytes_median": statistics.median(subchunks) if subchunks else 0,
        "subchunks_total": len(subchunks),
        "subchunks_lt_54B": sum(1 for value in subchunks if value < 54),
        "subchunks_lt_128B": sum(1 for value in subchunks if value < 128),
        "subchunks_lt_256B": sum(1 for value in subchunks if value < 256),
        "selected_path_count_min": min(selected_counts) if selected_counts else 0,
        "selected_path_count_median": statistics.median(selected_counts) if selected_counts else 0,
        "selected_path_count_mean": statistics.mean(selected_counts) if selected_counts else 0,
        "selected_path_count_max": max(selected_counts) if selected_counts else 0,
        "zero_delay_risk": any(value < v36.ONE_CYCLE_THRESHOLD_BYTES for value in subchunks),
    }


def rank_by_name(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {row["name"]: idx + 1 for idx, row in enumerate(rows)}


def median_random_name(eval_scores: list[dict[str, Any]]) -> str:
    random_rows = [row for row in eval_scores if row["name"].startswith("random_regular_seed_")]
    random_rows = sorted(random_rows, key=lambda row: (row["score_max_link_load_bytes"], row["name"]))
    return random_rows[len(random_rows) // 2]["name"]


def best_random_name(eval_scores: list[dict[str, Any]]) -> str:
    random_rows = [row for row in eval_scores if row["name"].startswith("random_regular_seed_")]
    return min(random_rows, key=lambda row: (row["score_max_link_load_bytes"], row["name"]))["name"]


def representative_names(cal_selected: str, oracle_selected: str, eval_scores: list[dict[str, Any]]) -> list[str]:
    names = [
        "son_torus",
        "random_regular_seed_0",
        median_random_name(eval_scores),
        best_random_name(eval_scores),
        cal_selected,
        oracle_selected,
        "calibration_greedy",
        "evaluation_greedy",
    ]
    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def graph_audit_compact(graph: dict[str, Any]) -> dict[str, Any]:
    audit = v36.graph_audit(graph)
    quality = v36.graph_quality(graph)
    return {
        "name": graph["name"],
        "edge_circuit_count": audit["edge_circuit_count"],
        "degree_distribution": audit["degree_distribution"],
        "connected_components": audit["connected_components"],
        "duplicate_edges": audit["duplicate_edges"],
        "self_loops": audit["self_loops"],
        "valid": audit["valid"],
        "same_degree_bandwidth_budget_as_son": audit["same_degree_bandwidth_budget_as_son"],
        "average_shortest_path_length": quality["average_shortest_path_length"],
        "diameter": quality["diameter"],
        "ecmp_path_count_distribution_cap4": quality["ecmp_path_count_distribution_cap4"],
    }


def graph_audit_budget_only(graph: dict[str, Any]) -> dict[str, Any]:
    audit = v36.graph_audit(graph)
    return {
        "name": graph["name"],
        "edge_circuit_count": audit["edge_circuit_count"],
        "degree_distribution": audit["degree_distribution"],
        "connected_components": audit["connected_components"],
        "duplicate_edges": audit["duplicate_edges"],
        "self_loops": audit["self_loops"],
        "valid": audit["valid"],
        "same_degree_bandwidth_budget_as_son": audit["same_degree_bandwidth_budget_as_son"],
    }


def repair_to_degree4(edges: set[tuple[int, int]]) -> set[tuple[int, int]]:
    edges = set(edges)

    def degs() -> dict[int, int]:
        deg = {node: 0 for node in range(v36.NPU_COUNT)}
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
        return deg

    for _ in range(10000):
        deg = degs()
        if all(value == v36.DEGREE for value in deg.values()):
            if v36.is_connected(edges):
                return edges
            raise RuntimeError("degree-4 greedy repair produced disconnected graph")
        lows = [node for node, value in deg.items() if value < v36.DEGREE]

        added = False
        for i, a in enumerate(lows):
            for b in lows[i + 1 :]:
                edge = tuple(sorted((a, b)))
                if edge not in edges:
                    edges.add(edge)
                    added = True
                    break
            if added:
                break
        if added:
            continue

        # If low-degree nodes are already connected to each other, split an
        # unrelated edge into two edges. This preserves the old endpoints'
        # degree and increases both low endpoints by one.
        if len(lows) < 2:
            raise RuntimeError(f"cannot repair odd low-degree set: {lows}")
        a, b = lows[0], lows[1]
        repaired = False
        for u, vv in sorted(edges):
            if len({a, b, u, vv}) < 4:
                continue
            option1 = (tuple(sorted((a, u))), tuple(sorted((b, vv))))
            option2 = (tuple(sorted((a, vv))), tuple(sorted((b, u))))
            for e1, e2 in (option1, option2):
                if e1 in edges or e2 in edges or e1 == e2:
                    continue
                trial = set(edges)
                trial.remove((u, vv))
                trial.add(e1)
                trial.add(e2)
                if v36.is_connected(trial):
                    edges = trial
                    repaired = True
                    break
            if repaired:
                break
        if not repaired:
            raise RuntimeError(f"could not repair greedy graph low nodes={lows}")
    raise RuntimeError("degree repair exceeded iteration limit")


def strict_greedy_demand_graph(demand: Any, seed_edges: set[tuple[int, int]]) -> set[tuple[int, int]]:
    edges = v36.greedy_demand_graph(demand, seed_edges)
    return repair_to_degree4(edges)


def build_candidate_graphs_strict(cal_demand: Any, eval_demand: Any) -> dict[str, dict[str, Any]]:
    candidates = {
        "son_torus": v36.graph_from_edges("son_torus_4x8_32gpu", v36.torus_edges(), {"construction": "static_4x8_torus"}),
        "calibration_greedy": v36.graph_from_edges(
            "ron_calibration_greedy_degree4",
            strict_greedy_demand_graph(cal_demand, v36.ring_edges()),
            {
                "construction": "strict_greedy_demand_graph",
                "traffic_source": "calibration_dispatch_plus_combine",
                "seed_edges": "ring_degree2",
                "repair": "edge-swap to enforce exact degree-4 if greedy fill stalls",
            },
        ),
        "evaluation_greedy": v36.graph_from_edges(
            "ron_evaluation_greedy_degree4",
            strict_greedy_demand_graph(eval_demand, v36.ring_edges()),
            {
                "construction": "strict_greedy_demand_graph",
                "traffic_source": "evaluation_dispatch_plus_combine",
                "seed_edges": "ring_degree2",
                "repair": "edge-swap to enforce exact degree-4 if greedy fill stalls",
            },
        ),
    }
    for seed in range(v36.RANDOM_CANDIDATES):
        candidates[f"random_regular_seed_{seed}"] = v36.graph_from_edges(
            f"ron_random_regular_seed_{seed}",
            v36.random_regular_graph(seed),
            {"construction": "random_regular_degree4", "seed": seed},
        )
    return candidates


def phase_totals(native: dict[str, Any], fluid: dict[str, Any], names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in names:
        d_run = native["dispatch"][name]
        c_run = native["combine"][name]
        d_fluid = fluid["dispatch"][name]
        c_fluid = fluid["combine"][name]
        total = (
            d_run["max_cycles"] + c_run["max_cycles"]
            if d_run["max_cycles"] is not None and c_run["max_cycles"] is not None
            else None
        )
        fluid_total = d_fluid["fluid_cycles"] + c_fluid["fluid_cycles"]
        out[name] = {
            "candidate_type": candidate_type(name),
            "dispatch_cycles": d_run["max_cycles"],
            "combine_cycles": c_run["max_cycles"],
            "total_cycles": total,
            "dispatch_fluid_cycles": d_fluid["fluid_cycles"],
            "combine_fluid_cycles": c_fluid["fluid_cycles"],
            "total_fluid_cycles": fluid_total,
            "astra_over_fluid_total": (total / fluid_total) if total is not None and fluid_total else None,
            "success": d_run["success"] and c_run["success"],
            "runtime_s": d_run["runtime_s"] + c_run["runtime_s"],
        }
    return out


def interpret_workload(
    workload_id: str,
    totals: dict[str, Any],
    cal_selected: str,
    oracle_selected: str,
    ranks: dict[str, int],
    traffic_stats: dict[str, Any],
) -> dict[str, Any]:
    son = totals["son_torus"]["total_cycles"]
    fixed = totals.get("random_regular_seed_0", {}).get("total_cycles")
    cal = totals[cal_selected]["total_cycles"]
    oracle = totals[oracle_selected]["total_cycles"]
    median_random = next((name for name, row in totals.items() if row.get("role") == "median_random"), None)
    best_random = next((name for name, row in totals.items() if row.get("role") == "best_random"), None)
    if cal_selected.startswith("random_regular_seed_"):
        main = "workload-selected random-regular topology search"
    elif cal_selected == "calibration_greedy":
        main = "hot-pair-aware greedy construction"
    elif cal_selected == "son_torus":
        main = "torus remains best under calibration"
    else:
        main = "mixed/unknown"
    if fixed is not None and cal is not None and fixed <= cal:
        main += "; fixed random is already competitive"
    return {
        "workload": workload_id,
        "calibrated_candidate": cal_selected,
        "calibrated_candidate_type": candidate_type(cal_selected),
        "oracle_candidate": oracle_selected,
        "oracle_candidate_type": candidate_type(oracle_selected),
        "traffic_fingerprint": fingerprint_label(traffic_stats),
        "torus_rank_by_eval_fluid": ranks.get("son_torus"),
        "greedy_calibration_rank_by_eval_fluid": ranks.get("calibration_greedy"),
        "greedy_evaluation_rank_by_eval_fluid": ranks.get("evaluation_greedy"),
        "fixed_seed0_rank_by_eval_fluid": ranks.get("random_regular_seed_0"),
        "calibrated_rank_by_eval_fluid": ranks.get(cal_selected),
        "oracle_rank_by_eval_fluid": ranks.get(oracle_selected),
        "calibrated_beats_son": cal is not None and son is not None and cal < son,
        "calibrated_beats_fixed_random": cal is not None and fixed is not None and cal < fixed,
        "oracle_beats_calibrated": oracle is not None and cal is not None and oracle < cal,
        "calibrated_gain_vs_son_percent": (100 * (son - cal) / son) if son and cal is not None else None,
        "calibrated_gain_vs_fixed_random_percent": (100 * (fixed - cal) / fixed) if fixed and cal is not None else None,
        "oracle_gap_vs_calibrated_percent": (100 * (cal - oracle) / cal) if cal and oracle is not None else None,
        "main_explanation": main,
        "median_random_name": median_random,
        "best_random_name": best_random,
    }


def process_workload(workload: dict[str, Any]) -> dict[str, Any]:
    wid = workload["id"]
    wout = OUT / "workloads" / wid
    if not workload["path"].exists() or not list(workload["path"].glob("*.json")):
        return {
            "workload": workload,
            "available": False,
            "error": f"Missing or empty trace directory: {workload['path']}",
        }
    v36.TRACE_DIR = workload["path"]
    parsed = v36.parse_requests()
    full = parsed["full"]
    cal = parsed["calibration"]
    ev = parsed["evaluation"]

    matrices_dir = wout / "traffic_matrices"
    graphs_dir = wout / "graphs"
    configs_dir = wout / "network_configs"
    traces_dir = wout / "chakra_traces"
    write_json(matrices_dir / "full_dispatch_matrix.json", {"phase": "dispatch", "matrix": full["dispatch_matrix"], "checksum": full["dispatch_checksum"]})
    write_json(matrices_dir / "full_combine_matrix.json", {"phase": "combine", "matrix": full["combine_matrix"], "checksum": full["combine_checksum"]})
    write_json(matrices_dir / "calibration_dispatch_matrix.json", {"phase": "dispatch", "matrix": cal["dispatch_matrix"], "checksum": cal["dispatch_checksum"]})
    write_json(matrices_dir / "calibration_combine_matrix.json", {"phase": "combine", "matrix": cal["combine_matrix"], "checksum": cal["combine_checksum"]})
    write_json(matrices_dir / "evaluation_dispatch_matrix.json", {"phase": "dispatch", "matrix": ev["dispatch_matrix"], "checksum": ev["dispatch_checksum"]})
    write_json(matrices_dir / "evaluation_combine_matrix.json", {"phase": "combine", "matrix": ev["combine_matrix"], "checksum": ev["combine_checksum"]})

    cal_demand = v36.matrix_to_demand(cal["dispatch_matrix"], cal["combine_matrix"])
    eval_demand = v36.matrix_to_demand(ev["dispatch_matrix"], ev["combine_matrix"])
    candidates = build_candidate_graphs_strict(cal_demand, eval_demand)
    path_cache = {name: precompute_selected_paths(graph) for name, graph in candidates.items()}
    calibrated_allowed = ["son_torus", "calibration_greedy"] + [f"random_regular_seed_{seed}" for seed in range(v36.RANDOM_CANDIDATES)]
    oracle_allowed = ["son_torus", "evaluation_greedy"] + [f"random_regular_seed_{seed}" for seed in range(v36.RANDOM_CANDIDATES)]
    cal_scores = score_candidates(candidates, cal["dispatch_matrix"], cal["combine_matrix"], calibrated_allowed, path_cache)
    eval_scores = score_candidates(candidates, ev["dispatch_matrix"], ev["combine_matrix"], oracle_allowed + ["calibration_greedy"], path_cache)
    cal_selected = cal_scores[0]["name"]
    oracle_selected = min([row for row in eval_scores if row["name"] in oracle_allowed], key=lambda row: (row["score_max_link_load_bytes"], row["name"]))["name"]
    reps = representative_names(cal_selected, oracle_selected, eval_scores)

    median_random = median_random_name(eval_scores)
    best_random = best_random_name(eval_scores)
    roles = {
        "son_torus": "son",
        "random_regular_seed_0": "fixed_random",
        median_random: "median_random",
        best_random: "best_random",
        cal_selected: "ron_calibrated",
        oracle_selected: "ron_oracle",
        "calibration_greedy": "greedy_calibration",
        "evaluation_greedy": "greedy_evaluation",
    }

    candidate_audits = {name: graph_audit_budget_only(graph) for name, graph in candidates.items()}
    graph_budget_pass = all(item["same_degree_bandwidth_budget_as_son"] for item in candidate_audits.values())
    valid_graphs = all(item["valid"] for item in candidate_audits.values())

    graph_paths: dict[str, Path] = {}
    config_paths: dict[str, Path] = {}
    for name in reps:
        graph_paths[name] = graphs_dir / f"{name}.json"
        write_json(graph_paths[name], candidates[name])
        config_paths[name] = configs_dir / f"{name}.yml"
        v36.network_config(config_paths[name], graph_paths[name])

    trace_meta = {
        "evaluation_dispatch": v36.write_matrix_trace(traces_dir / "evaluation_dispatch" / "workload", ev["dispatch_matrix"]),
        "evaluation_combine": v36.write_matrix_trace(traces_dir / "evaluation_combine" / "workload", ev["combine_matrix"]),
    }

    native: dict[str, Any] = {"dispatch": {}, "combine": {}}
    fluid: dict[str, Any] = {"dispatch": {}, "combine": {}}
    tiny: dict[str, Any] = {"dispatch": {}, "combine": {}}
    for phase, matrix, workload_prefix in (
        ("dispatch", ev["dispatch_matrix"], traces_dir / "evaluation_dispatch" / "workload"),
        ("combine", ev["combine_matrix"], traces_dir / "evaluation_combine" / "workload"),
    ):
        for name in reps:
            graph = candidates[name]
            fluid[phase][name] = link_load_estimate_cached(path_cache[name], matrix)
            tiny[phase][name] = tiny_subchunk_audit_cached(path_cache[name], matrix)
            native[phase][name] = v36.run_astra(f"{wid}_{phase}_{name}", workload_prefix, config_paths[name])

    totals = phase_totals(native, fluid, reps)
    for name, role in roles.items():
        if name in totals:
            totals[name]["role"] = role

    traffic_fingerprint = {
        "calibration_dispatch": {**v36.matrix_stats(cal["dispatch_matrix"]), "interpretation": fingerprint_label(v36.matrix_stats(cal["dispatch_matrix"]))},
        "calibration_combine": {**v36.matrix_stats(cal["combine_matrix"]), "interpretation": fingerprint_label(v36.matrix_stats(cal["combine_matrix"]))},
        "evaluation_dispatch": {**v36.matrix_stats(ev["dispatch_matrix"]), "interpretation": fingerprint_label(v36.matrix_stats(ev["dispatch_matrix"]))},
        "evaluation_combine": {**v36.matrix_stats(ev["combine_matrix"]), "interpretation": fingerprint_label(v36.matrix_stats(ev["combine_matrix"]))},
    }
    eval_ranks = rank_by_name(eval_scores)
    interpretation = interpret_workload(wid, totals, cal_selected, oracle_selected, eval_ranks, traffic_fingerprint["evaluation_dispatch"])

    split = {
        "calibration_request_count": cal["request_count"],
        "evaluation_request_count": ev["request_count"],
        "calibration_request_ids": cal["request_ids"],
        "evaluation_request_ids": ev["request_ids"],
        "calibration_rule": "front ceil(10%) requests",
    }
    write_json(wout / "split.json", split)
    write_json(wout / "candidate_scores_calibration.json", cal_scores)
    write_json(wout / "candidate_scores_evaluation.json", eval_scores)
    write_json(wout / "candidate_audits.json", candidate_audits)

    return {
        "workload": {
            "id": wid,
            "label": workload["label"],
            "path": str(workload["path"]),
        },
        "available": True,
        "trace_parse": {
            "files_found": parsed["files_found"],
            "files_used": parsed["files_used"],
            "moe_layer_count": parsed["moe_layer_count"],
            "inferred_num_experts": parsed["inferred_num_experts"],
            "malformed_records": parsed["malformed_records"],
            "full": {k: v for k, v in full.items() if not k.endswith("_matrix")},
            "calibration": {k: v for k, v in cal.items() if not k.endswith("_matrix")},
            "evaluation": {k: v for k, v in ev.items() if not k.endswith("_matrix")},
        },
        "split": split,
        "anti_leakage": {
            "calibrated_uses_calibration_only": True,
            "oracle_uses_evaluation_only": True,
            "oracle_reference_only": True,
        },
        "traffic_fingerprint": traffic_fingerprint,
        "candidate_pool": {
        "candidate_count": len(candidates),
        "random_candidate_count": v36.RANDOM_CANDIDATES,
            "candidate_types": dict(sorted({kind: sum(1 for name in candidates if candidate_type(name) == kind) for kind in {candidate_type(name) for name in candidates}}.items())),
            "graph_budget_pass": graph_budget_pass,
            "all_candidate_graphs_valid": valid_graphs,
        },
        "candidate_scores": {
            "calibration_top12": cal_scores[:12],
            "evaluation_top12": eval_scores[:12],
            "evaluation_all": eval_scores,
        },
        "selected": {
            "ron_calibrated": cal_selected,
            "ron_calibrated_type": candidate_type(cal_selected),
            "ron_oracle": oracle_selected,
            "ron_oracle_type": candidate_type(oracle_selected),
            "fixed_random": "random_regular_seed_0",
            "median_random": median_random,
            "best_random": best_random,
            "representatives_run_in_astra": reps,
            "roles": roles,
        },
        "candidate_audits_all_budget_only": candidate_audits,
        "candidate_audits_representatives": {name: graph_audit_compact(candidates[name]) for name in reps},
        "native_astra_results": native,
        "native_astra_totals": totals,
        "fluid_lower_bound": fluid,
        "tiny_subchunk_audit": tiny,
        "candidate_family_interpretation": interpretation,
        "validation_pass": {
            "byte_conservation": full["byte_conservation_pass"] and cal["byte_conservation_pass"] and ev["byte_conservation_pass"],
            "graph_budget": graph_budget_pass,
            "graphs_valid": valid_graphs,
            "native_runs": all(row["success"] for row in totals.values()),
            "no_tiny_subchunk_risk": not any(
                item["zero_delay_risk"]
                for phase_payload in tiny.values()
                for item in phase_payload.values()
            ),
        },
    }


def compact_summary_row(result: dict[str, Any]) -> dict[str, Any]:
    interp = result["candidate_family_interpretation"]
    totals = result["native_astra_totals"]
    fixed = totals.get("random_regular_seed_0", {})
    median_name = result["selected"]["median_random"]
    best_name = result["selected"]["best_random"]
    cal_name = result["selected"]["ron_calibrated"]
    oracle_name = result["selected"]["ron_oracle"]
    eval_stats = result["traffic_fingerprint"]["evaluation_dispatch"]
    return {
        "workload": result["workload"]["id"],
        "traffic_gini": eval_stats["gini"],
        "top16_pair_share": eval_stats["top16_share"],
        "traffic_interpretation": eval_stats["interpretation"],
        "calibrated_candidate_type": result["selected"]["ron_calibrated_type"],
        "oracle_candidate_type": result["selected"]["ron_oracle_type"],
        "son_cycles": totals["son_torus"]["total_cycles"],
        "fixed_random_cycles": fixed.get("total_cycles"),
        "median_random_cycles": totals.get(median_name, {}).get("total_cycles"),
        "best_random_cycles": totals.get(best_name, {}).get("total_cycles"),
        "calibrated_cycles": totals[cal_name]["total_cycles"],
        "oracle_cycles": totals[oracle_name]["total_cycles"],
        "calibrated_gain_vs_son_percent": interp["calibrated_gain_vs_son_percent"],
        "calibrated_gain_vs_fixed_random_percent": interp["calibrated_gain_vs_fixed_random_percent"],
        "oracle_gap_vs_calibrated_percent": interp["oracle_gap_vs_calibrated_percent"],
        "main_explanation": interp["main_explanation"],
    }


def write_report(summary: dict[str, Any]) -> None:
    (OUT / "README.md").write_text(
        f"""# V3.7 Four-Workload Static Candidate-Family Validation

## Scope

This extends the V36 aggregated native ASTRA pipeline from one workload to four workloads. It does not implement W=4 dynamic reconfiguration, generate paper figures, use all-path ECMP, or switch to token/layer-level traces.

## Final Answers

1. Do all four workloads run through the aggregated native ASTRA pipeline? **{summary["final_answers"]["all_workloads_run"]}.**
2. Are all calibrated selections leakage-free? **{summary["final_answers"]["all_calibrated_leakage_free"]}.**
3. Are all topologies under the same degree/bandwidth budget? **{summary["final_answers"]["all_topologies_same_budget"]}.**
4. Which candidate type does calibrated select for each workload? See cross-workload summary.
5. Which candidate type does oracle select for each workload? See cross-workload summary.
6. Does calibrated beat SON? **{summary["final_answers"]["calibrated_beats_son_by_workload"]}.**
7. Does calibrated beat fixed random? **{summary["final_answers"]["calibrated_beats_fixed_random_by_workload"]}.**
8. Does calibrated beat median random? **{summary["final_answers"]["calibrated_beats_median_random_by_workload"]}.**
9. Is the gain mainly workload-aware, greedy hot-pair-aware, or random-regular topology-family strength? See candidate-family interpretation.
10. Which traffic fingerprints predict gain? See cross-workload summary and traffic fingerprints.
11. Is it safe to proceed to RON W=4 segmented validation next? **{summary["final_recommendation"]["safe_to_proceed_to_w4"]}.**

## Cross-Workload Summary

```json
{json.dumps(summary["cross_workload_summary"], indent=2)}
```

## Candidate-Family Interpretation

```json
{json.dumps(summary["candidate_family_interpretation"], indent=2)}
```

## Workload Results

```json
{json.dumps(summary["workload_results"], indent=2)}
```

## Anti-Overclaiming

Can claim:

- Static topology candidate selection can be evaluated across multiple HF-derived MoE prefill workloads.
- RON calibrated/oracle can be compared against SON under the same degree/bandwidth budget.
- Random-regular controls help separate topology-family effects from workload-aware selection.

Cannot claim:

- W=4 dynamic reconfiguration works.
- Real serving latency.
- Physical transparent OCS modelling.
- Token/layer-level execution timing.
- Generality beyond these workloads.
"""
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = [process_workload(workload) for workload in WORKLOADS]
    available_results = [result for result in results if result.get("available")]
    cross_rows = [compact_summary_row(result) for result in available_results]
    candidate_interp = {
        result["workload"]["id"]: result["candidate_family_interpretation"]
        for result in available_results
    }

    all_workloads_run = len(available_results) == len(WORKLOADS) and all(
        all(result["validation_pass"].values()) for result in available_results
    )
    final_answers = {
        "all_workloads_run": all_workloads_run,
        "all_calibrated_leakage_free": all(result["anti_leakage"]["calibrated_uses_calibration_only"] for result in available_results),
        "all_topologies_same_budget": all(result["candidate_pool"]["graph_budget_pass"] for result in available_results),
        "calibrated_beats_son_by_workload": {
            result["workload"]["id"]: result["candidate_family_interpretation"]["calibrated_beats_son"]
            for result in available_results
        },
        "calibrated_beats_fixed_random_by_workload": {
            result["workload"]["id"]: result["candidate_family_interpretation"]["calibrated_beats_fixed_random"]
            for result in available_results
        },
        "calibrated_beats_median_random_by_workload": {
            result["workload"]["id"]: (
                result["native_astra_totals"][result["selected"]["ron_calibrated"]]["total_cycles"]
                < result["native_astra_totals"][result["selected"]["median_random"]]["total_cycles"]
            )
            for result in available_results
        },
    }
    # If calibrated often does not beat strong random controls, W=4 can still be
    # attempted, but the RON story should be framed as candidate-family search.
    calibrated_vs_fixed = final_answers["calibrated_beats_fixed_random_by_workload"]
    calibrated_wins_fixed_count = sum(1 for value in calibrated_vs_fixed.values() if value)
    safe_to_w4 = all_workloads_run
    story_revision_needed = calibrated_wins_fixed_count < max(1, len(available_results) // 2)
    summary = {
        "scope": "four-workload static topology candidate-family validation",
        "workloads_requested": [
            {"id": w["id"], "label": w["label"], "path": str(w["path"])} for w in WORKLOADS
        ],
        "workload_results": results,
        "cross_workload_summary": cross_rows,
        "candidate_family_interpretation": candidate_interp,
        "final_answers": final_answers,
        "final_recommendation": {
            "safe_to_proceed_to_w4": safe_to_w4,
            "story_revision_needed": story_revision_needed,
            "recommended_next_step": (
                "Proceed to RON W=4 segmented validation, but frame static gains as workload-selected degree-4 topology search if random controls explain most gains."
                if safe_to_w4
                else "Fix failed workload/topology validation before W=4."
            ),
        },
    }
    write_json(OUT / "summary.json", summary)
    write_report(summary)
    print(json.dumps(summary, indent=2))
    if not all_workloads_run:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
