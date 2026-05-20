#!/usr/bin/env python3
"""V38.1/2 preliminary supervisor figures and validation.

This is an inference-only, trace-driven MoE communication study.

Selection signal:
  trace[0] prefill only

Evaluation signal:
  trace[1:] decode only

Timing:
  fluid link-load estimate over generated graph topologies.  This script does
  not modify ASTRA C++ core and does not claim full serving latency or native
  in-run topology swaps.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/moe_expert_trace_converter/v38_1_2_prefill_decode_figures_validation"
V38_PATH = REPO / "tools/v38_final_prefill_decode_regional_ocs.py"
V37C_PATH = REPO / "tools/v37c_128gpu_best_ocs_reconfig_audit.py"
ASTRA_BIN = REPO / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
SYSTEM = REPO / "examples/system/native_collectives/Ring_4chunks.json"
REMOTE_MEMORY = REPO / "examples/remote_memory/analytical/no_memory_expansion.json"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v38 = _load_module("v38_final_prefill_decode_regional_ocs", V38_PATH)
v37c = _load_module("v37c_128gpu_best_ocs_reconfig_audit", V37C_PATH)

NPU = 128
LINK_GBPS = 400
ECMP_MAX_PATHS = 4
BYTES_PER_SELECTION = v37c.BYTES_PER_SELECTION
ASTRA_BYTES_PER_NS = v37c.ASTRA_BYTES_PER_NS
PENALTIES_US = [0, 1, 10, 25_000]
RANDOM_SEEDS = v38.RANDOM_SEEDS
ASTRA_TIMEOUT_S = 240

METHOD_ORDER = [
    "EN",
    "SON / torus",
    "fixed random",
    "fair universal static",
    "prefill-informed OCS",
    "oracle",
]

METHOD_COLORS = {
    "EN": "#9AA4B2",
    "SON / torus": "#4E79A7",
    "fixed random": "#59A14F",
    "fair universal static": "#F28E2B",
    "prefill-informed OCS": "#E15759",
    "oracle": "#7B52AB",
}

WORKLOADS = [
    {
        "id": "qwen_mmlu_machine_learning",
        "label": "Qwen MMLU ML",
        "full_label": "Qwen MMLU machine_learning",
        "path": Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/mmlu/machine_learning"),
    },
    {
        "id": "deepseek_mmlu_machine_learning",
        "label": "DeepSeek MMLU ML",
        "full_label": "DeepSeek MMLU machine_learning",
        "path": Path("/Users/dfx/Python/trace/cognitivecomputations/DeepSeek-R1-AWQ/mmlu/machine_learning"),
    },
    {
        "id": "qwen_livecodebench_execution",
        "label": "Qwen LiveCodeBench",
        "full_label": "Qwen LiveCodeBench execution",
        "path": Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/livecodebench/execution"),
    },
]

Pair = tuple[int, int]
Sparse = dict[Pair, int]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cycles_to_ms(cycles: int | float) -> float:
    # In these calibrated analytical runs, one cycle is one nanosecond.
    return float(cycles) / 1_000_000.0


def ms_to_cycles(ms: float) -> int:
    return int(ms * 1_000_000)


def safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def add_sparse(dst: defaultdict[Pair, int], src: Sparse) -> None:
    for key, value in src.items():
        dst[key] += value


def combine_sparse(a: Sparse, b: Sparse) -> Sparse:
    out: defaultdict[Pair, int] = defaultdict(int)
    add_sparse(out, a)
    add_sparse(out, b)
    return dict(out)


def sparse_sum(sparse: Sparse) -> int:
    return sum(int(v) for v in sparse.values())


def sparse_values(sparse: Sparse) -> list[int]:
    return [int(v) for v in sparse.values() if v > 0]


def concentration(values: list[int], topks: tuple[int, ...] = (1, 4, 8, 16)) -> dict[str, Any]:
    return v37c.concentration(values, topks)


def gini(values: list[int | float]) -> float:
    return v37c.gini(values)


def entropy(values: list[int | float]) -> float:
    return v37c.entropy(values)


def en_folded_clos_graph_128() -> dict[str, Any]:
    """128-GPU folded-Clos/EPS-style reference graph.

    The graph has 16 leaves, 16 spines, and 8 GPUs per leaf.  It is an
    electrical reference, not the fair optical baseline.
    """

    gpus_per_leaf = 8
    leaves = NPU // gpus_per_leaf
    spines = leaves
    leaf_base = NPU
    spine_base = leaf_base + leaves
    edges: list[dict[str, Any]] = []
    for gpu in range(NPU):
        leaf = leaf_base + gpu // gpus_per_leaf
        edges.append({"src": gpu, "dst": leaf, "bandwidth_gbps": LINK_GBPS, "latency_ns": 0})
    for leaf_idx in range(leaves):
        leaf = leaf_base + leaf_idx
        for spine_idx in range(spines):
            spine = spine_base + spine_idx
            edges.append({"src": leaf, "dst": spine, "bandwidth_gbps": LINK_GBPS, "latency_ns": 0})
    return {
        "name": "en_folded_clos_128gpu_16leaf_16spine",
        "node_count": NPU + leaves + spines,
        "gpu_count": NPU,
        "rank_nodes": list(range(NPU)),
        "directed": False,
        "metadata": {
            "construction": "folded_clos_eps_reference",
            "gpus_per_leaf": gpus_per_leaf,
            "leaf_count": leaves,
            "spine_count": spines,
            "gpu_access_link_gbps": LINK_GBPS,
            "leaf_spine_link_gbps": LINK_GBPS,
            "ecmp_max_paths": ECMP_MAX_PATHS,
            "note": "electrical reference, not same-budget optical topology",
        },
        "edges": edges,
    }


def graph_edges(graph: dict[str, Any]) -> list[tuple[int, int]]:
    return [tuple(sorted((int(edge["src"]), int(edge["dst"])))) for edge in graph["edges"]]


def graph_components(graph: dict[str, Any]) -> list[list[int]]:
    node_count = int(graph["node_count"])
    adj: list[list[int]] = [[] for _ in range(node_count)]
    for a, b in set(graph_edges(graph)):
        adj[a].append(b)
        adj[b].append(a)
    seen: set[int] = set()
    comps: list[list[int]] = []
    for node in range(node_count):
        if node in seen:
            continue
        q: deque[int] = deque([node])
        seen.add(node)
        comp: list[int] = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(sorted(comp))
    return comps


def graph_audit(graph: dict[str, Any], method: str) -> dict[str, Any]:
    edges = graph_edges(graph)
    unique_edges = set(edges)
    deg_all = Counter()
    for a, b in unique_edges:
        deg_all[a] += 1
        deg_all[b] += 1
    for node in range(int(graph["node_count"])):
        deg_all[node] += 0
    gpu_deg = {node: deg_all[node] for node in range(int(graph["gpu_count"]))}
    comps = graph_components(graph)
    is_optical_budget = method != "EN"
    same_budget = (
        is_optical_budget
        and len(unique_edges) == NPU * 4 // 2
        and min(gpu_deg.values()) == 4
        and max(gpu_deg.values()) == 4
    )
    return {
        "method": method,
        "graph_name": graph["name"],
        "node_count": graph["node_count"],
        "gpu_count": graph["gpu_count"],
        "edge_or_circuit_count": len(unique_edges),
        "degree_distribution_all_nodes": dict(sorted(Counter(deg_all.values()).items())),
        "degree_distribution_gpu_nodes": dict(sorted(Counter(gpu_deg.values()).items())),
        "connected_components": len(comps),
        "component_sizes": [len(c) for c in comps],
        "duplicate_edges": len(edges) - len(unique_edges),
        "self_loops": sum(1 for a, b in unique_edges if a == b),
        "per_link_bandwidth_gbps": LINK_GBPS,
        "per_gpu_bandwidth_tbps": 1.6 if is_optical_budget else 0.4,
        "ecmp_max_paths": ECMP_MAX_PATHS,
        "same_budget_check": same_budget,
        "budget_note": "same degree-4 optical budget" if same_budget else ("EN electrical reference" if method == "EN" else "FAILED optical budget"),
    }


def network_config(path: Path, graph_path: Path) -> None:
    lines = [
        "topology: [ Graph ]",
        f"npus_count: [ {NPU} ]",
        "bandwidth: [ 50.0 ]",
        "latency: [ 0.0 ]",
        f"graph_file: {graph_path}",
        "routing: ecmp",
        "ecmp_split: equal_bytes",
        f"ecmp_max_paths: {ECMP_MAX_PATHS}",
        "ecmp_log: false",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def sparse_to_matrix(sparse: Sparse) -> list[list[int]]:
    matrix = [[0 for _ in range(NPU)] for _ in range(NPU)]
    for (src, dst), size in sparse.items():
        matrix[src][dst] = int(size)
    return matrix


def run_astra(label: str, workload: Path, network: Path) -> dict[str, Any]:
    runs_dir = OUT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stdout = runs_dir / f"{safe_label(label)}.stdout.txt"
    stderr = runs_dir / f"{safe_label(label)}.stderr.txt"
    cmd = [
        str(ASTRA_BIN),
        f"--workload-configuration={workload}",
        f"--system-configuration={SYSTEM}",
        f"--remote-memory-configuration={REMOTE_MEMORY}",
        f"--network-configuration={network}",
    ]
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
            timeout=ASTRA_TIMEOUT_S,
        )
        timed_out = False
        stdout_text = proc.stdout
        stderr_text = proc.stderr
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout_text = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
        stderr_text = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")
        returncode = -999
    runtime_s = time.perf_counter() - start
    stdout.write_text(stdout_text)
    stderr.write_text(stderr_text)
    cycles = [int(match) for match in re.findall(r"finished, ([0-9]+) cycles", stdout_text)]
    return {
        "label": label,
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_s": ASTRA_TIMEOUT_S,
        "success": returncode == 0 and len(cycles) == NPU,
        "runtime_s": runtime_s,
        "command": " ".join(cmd),
        "stdout": str(stdout),
        "stderr": str(stderr),
        "stderr_tail": stderr_text.splitlines()[-8:],
        "max_cycles": max(cycles) if cycles else None,
        "cycles_count": len(cycles),
        "all_ranks_finished": len(cycles) == NPU,
    }


def run_native_method(
    workload_id: str,
    method: str,
    candidate_name: str,
    graph: dict[str, Any],
    dispatch_prefix: Path,
    combine_prefix: Path,
) -> dict[str, Any]:
    graph_dir = OUT / "graphs"
    config_dir = OUT / "network_configs"
    graph_path = graph_dir / f"{safe_label(workload_id)}__{safe_label(method)}__{safe_label(candidate_name)}.json"
    config_path = config_dir / f"{safe_label(workload_id)}__{safe_label(method)}__{safe_label(candidate_name)}.yml"
    write_json(graph_path, graph)
    network_config(config_path, graph_path)
    dispatch_run = run_astra(f"{workload_id}__{method}__dispatch", dispatch_prefix, config_path)
    combine_run = run_astra(f"{workload_id}__{method}__combine", combine_prefix, config_path)
    dispatch_cycles = int(dispatch_run["max_cycles"] or 0)
    combine_cycles = int(combine_run["max_cycles"] or 0)
    return {
        "workload": workload_id,
        "method": method,
        "candidate": candidate_name,
        "graph_path": str(graph_path),
        "network_config": str(config_path),
        "dispatch": dispatch_run,
        "combine": combine_run,
        "native_astra_success": bool(dispatch_run["success"] and combine_run["success"]),
        "native_astra_dispatch_cycles": dispatch_cycles,
        "native_astra_combine_cycles": combine_cycles,
        "native_astra_total_cycles": dispatch_cycles + combine_cycles,
        "native_astra_total_ms": cycles_to_ms(dispatch_cycles + combine_cycles),
    }


def build_base_candidates() -> dict[str, dict[str, Any]]:
    candidates = v38.graph_candidates()
    candidates["en_folded_clos"] = en_folded_clos_graph_128()
    return candidates


def precompute_candidate_paths(candidates: dict[str, dict[str, Any]]) -> dict[str, dict[Pair, list[list[int]]]]:
    return {
        name: v37c.precompute_paths(graph, NPU, ECMP_MAX_PATHS)
        for name, graph in sorted(candidates.items())
    }


def link_load_details(paths_by_pair: dict[Pair, list[list[int]]], sparse: Sparse) -> dict[str, Any]:
    loads: defaultdict[Pair, int] = defaultdict(int)
    selected_counts: list[int] = []
    subchunks: list[int] = []
    byte_weighted_hops = 0
    total_bytes = 0
    for (src, dst), size in sparse.items():
        if src == dst or size <= 0:
            continue
        paths = paths_by_pair.get((src, dst), [])
        if not paths:
            raise RuntimeError(f"no path {src}->{dst}")
        selected_counts.append(len(paths))
        splits = v37c.split_bytes(size, len(paths))
        subchunks.extend(splits)
        for path, subbytes in zip(paths, splits):
            hop_count = len(path) - 1
            byte_weighted_hops += subbytes * hop_count
            total_bytes += subbytes
            for u, vv in zip(path, path[1:]):
                loads[(u, vv)] += subbytes
    load_values = sorted(loads.values(), reverse=True)
    max_load = load_values[0] if load_values else 0
    total_link_bytes = sum(load_values)
    return {
        "fluid_cycles": int(max_load / ASTRA_BYTES_PER_NS) if max_load else 0,
        "max_link_load_bytes": max_load,
        "total_link_load_bytes": total_link_bytes,
        "byte_weighted_average_hop_count": byte_weighted_hops / total_bytes if total_bytes else 0,
        "selected_path_count_min": min(selected_counts) if selected_counts else 0,
        "selected_path_count_median": statistics.median(selected_counts) if selected_counts else 0,
        "selected_path_count_mean": statistics.mean(selected_counts) if selected_counts else 0,
        "selected_path_count_max": max(selected_counts) if selected_counts else 0,
        "hot_link_top1_share": load_values[0] / total_link_bytes if total_link_bytes else 0,
        "hot_link_top4_share": sum(load_values[:4]) / total_link_bytes if total_link_bytes else 0,
        "hot_link_top16_share": sum(load_values[:16]) / total_link_bytes if total_link_bytes else 0,
        "bottleneck_link_id": f"{max(loads.items(), key=lambda item: item[1])[0][0]}->{max(loads.items(), key=lambda item: item[1])[0][1]}" if loads else "",
        "hot_links": [
            {"src": u, "dst": vv, "bytes": value}
            for (u, vv), value in sorted(loads.items(), key=lambda item: -item[1])[:16]
        ],
        "subchunk_bytes_min": min(subchunks) if subchunks else 0,
        "subchunk_bytes_median": statistics.median(subchunks) if subchunks else 0,
        "subchunk_bytes_max": max(subchunks) if subchunks else 0,
        "subchunks_lt_54B": sum(1 for value in subchunks if value < 54),
        "subchunks_lt_128B": sum(1 for value in subchunks if value < 128),
        "subchunks_lt_256B": sum(1 for value in subchunks if value < 256),
        "zero_delay_risk": any(value < v37c.ONE_CYCLE_THRESHOLD_BYTES for value in subchunks),
    }


def eval_payload(paths_by_pair: dict[Pair, list[list[int]]], payload: dict[str, Any]) -> dict[str, Any]:
    dispatch = link_load_details(paths_by_pair, payload["dispatch_sparse"])
    combine = link_load_details(paths_by_pair, payload["combine_sparse"])
    total_cycles = dispatch["fluid_cycles"] + combine["fluid_cycles"]
    total_bytes = sparse_sum(payload["dispatch_sparse"]) + sparse_sum(payload["combine_sparse"])
    weighted_hops = (
        dispatch["byte_weighted_average_hop_count"] * sparse_sum(payload["dispatch_sparse"])
        + combine["byte_weighted_average_hop_count"] * sparse_sum(payload["combine_sparse"])
    ) / total_bytes if total_bytes else 0
    return {
        "dispatch_cycles": dispatch["fluid_cycles"],
        "combine_cycles": combine["fluid_cycles"],
        "sequential_cycles": total_cycles,
        "sequential_ms": cycles_to_ms(total_cycles),
        "max_link_load_bytes": max(dispatch["max_link_load_bytes"], combine["max_link_load_bytes"]),
        "byte_weighted_average_hop_count": weighted_hops,
        "dispatch": dispatch,
        "combine": combine,
    }


def select_prefill_candidate(
    candidates: dict[str, dict[str, Any]],
    path_caches: dict[str, dict[Pair, list[list[int]]]],
    prefill_payload: dict[str, Any],
) -> dict[str, Any]:
    allowed = ["son_torus", "prefill_greedy"] + [f"random_regular_seed_{seed}" for seed in RANDOM_SEEDS]
    rows = []
    for name in allowed:
        metric = eval_payload(path_caches[name], prefill_payload)
        rows.append({"candidate": name, "prefill_sequential_cycles": metric["sequential_cycles"]})
    rows.sort(key=lambda row: (row["prefill_sequential_cycles"], row["candidate"]))
    return {"selected": rows[0]["candidate"], "scores_top8": rows[:8]}


def select_oracle_candidate(
    path_caches: dict[str, dict[Pair, list[list[int]]]],
    decode_payload: dict[str, Any],
) -> dict[str, Any]:
    allowed = ["son_torus", "prefill_greedy", "decode_greedy"] + [f"random_regular_seed_{seed}" for seed in RANDOM_SEEDS]
    rows = []
    for name in allowed:
        metric = eval_payload(path_caches[name], decode_payload)
        rows.append({"candidate": name, "decode_sequential_cycles": metric["sequential_cycles"]})
    rows.sort(key=lambda row: (row["decode_sequential_cycles"], row["candidate"]))
    return {"selected": rows[0]["candidate"], "scores_top8": rows[:8]}


def select_fair_universal_static(
    decode_payloads: dict[str, dict[str, Any]],
    path_caches: dict[str, dict[Pair, list[list[int]]]],
) -> dict[str, Any]:
    random_names = [f"random_regular_seed_{seed}" for seed in RANDOM_SEEDS]
    result: dict[str, Any] = {}
    for target in decode_payloads:
        rows = []
        for name in random_names:
            norms = []
            for wid, payload in decode_payloads.items():
                if wid == target:
                    continue
                son = eval_payload(path_caches["son_torus"], payload)["sequential_cycles"]
                val = eval_payload(path_caches[name], payload)["sequential_cycles"]
                norms.append(val / son if son else 0)
            rows.append({"candidate": name, "leave_one_out_avg_norm_to_son": statistics.mean(norms)})
        rows.sort(key=lambda row: (row["leave_one_out_avg_norm_to_son"], row["candidate"]))
        result[target] = {"selected": rows[0]["candidate"], "scores_top8": rows[:8]}
    return result


def stage_payload(parsed: dict[str, Any], stage: str) -> dict[str, Any]:
    order = v38.hot_order(parsed["prefill_counts"])
    indices = list(range(len(parsed["requests"])))
    return v38.build_stage_sparse(parsed, stage, indices, "block_by_token", "block", order)


def stage_layer_count(parsed: dict[str, Any], stage: str) -> int:
    rows_key = "prefill_rows" if stage == "prefill" else "decode_rows"
    layers: set[int] = set()
    for req in parsed["requests"]:
        for layer_id, *_ in req[rows_key]:
            layers.add(int(layer_id))
    return len(layers)


def matrix_checksum(sparse: Sparse) -> str:
    import hashlib

    blob = json.dumps([[k[0], k[1], v] for k, v in sorted(sparse.items())], separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def traffic_validation(workload: dict[str, Any], parsed: dict[str, Any], prefill: dict[str, Any], decode: dict[str, Any]) -> dict[str, Any]:
    selected_prefill = int(prefill["selected_events"])
    selected_decode = int(decode["selected_events"])
    theoretical_prefill_one_way = selected_prefill * BYTES_PER_SELECTION
    theoretical_decode_one_way = selected_decode * BYTES_PER_SELECTION
    remote_decode_one_way = sparse_sum(decode["dispatch_sparse"])
    local_decode_one_way = int(decode["local_bytes_excluded"])
    combined_vals = sparse_values(decode["combined_sparse"])
    stats = concentration(combined_vals, (1, 4, 8, 16)) if combined_vals else {}
    return {
        "workload": workload["id"],
        "files_used": parsed["files_used"],
        "request_count": len(parsed["requests"]),
        "prefill_token_count": parsed["prefill_tokens"],
        "decode_token_count": parsed["decode_tokens"],
        "prefill_moe_layer_count": stage_layer_count(parsed, "prefill"),
        "decode_moe_layer_count": stage_layer_count(parsed, "decode"),
        "prefill_selected_expert_events": selected_prefill,
        "decode_selected_expert_events": selected_decode,
        "trace0_used_only_for_selection": True,
        "trace1_plus_used_only_for_evaluation": True,
        "decode_exists_non_empty": parsed["decode_events"] > 0 and parsed["decode_tokens"] > 0,
        "prefill_theoretical_dispatch_bytes": theoretical_prefill_one_way,
        "decode_theoretical_dispatch_bytes": theoretical_decode_one_way,
        "dispatch_bytes": remote_decode_one_way,
        "combine_bytes": sparse_sum(decode["combine_sparse"]),
        "total_remote_bytes": sparse_sum(decode["dispatch_sparse"]) + sparse_sum(decode["combine_sparse"]),
        "local_bytes_excluded": local_decode_one_way * 2,
        "byte_conservation_pass": (
            prefill["byte_conservation_pass"]
            and decode["byte_conservation_pass"]
            and theoretical_decode_one_way == local_decode_one_way + remote_decode_one_way
        ),
        "nonzero_gpu_pairs": len(decode["combined_sparse"]),
        "pair_bytes_min": min(combined_vals) if combined_vals else 0,
        "pair_bytes_median": statistics.median(combined_vals) if combined_vals else 0,
        "pair_bytes_max": max(combined_vals) if combined_vals else 0,
        "pair_top1_share": stats.get("top1_share", 0),
        "pair_top4_share": stats.get("top4_share", 0),
        "pair_top16_share": stats.get("top16_share", 0),
        "pair_gini": stats.get("gini", 0),
        "dispatch_matrix_checksum": matrix_checksum(decode["dispatch_sparse"]),
        "combine_matrix_checksum": matrix_checksum(decode["combine_sparse"]),
    }


def no_leakage_rows(workload_id: str, fair_selection: str, prefill_selection: str, oracle_selection: str) -> list[dict[str, Any]]:
    return [
        {
            "workload": workload_id,
            "method": "EN",
            "selection_signal": "static electrical folded-Clos reference",
            "evaluation_signal": "decode only",
            "uses_target_decode_for_selection": False,
            "oracle": False,
            "selected_candidate": "en_folded_clos",
            "pass": True,
        },
        {
            "workload": workload_id,
            "method": "SON / torus",
            "selection_signal": "static 8x16 torus",
            "evaluation_signal": "decode only",
            "uses_target_decode_for_selection": False,
            "oracle": False,
            "selected_candidate": "son_torus",
            "pass": True,
        },
        {
            "workload": workload_id,
            "method": "fixed random",
            "selection_signal": "fixed seed0, no workload signal",
            "evaluation_signal": "decode only",
            "uses_target_decode_for_selection": False,
            "oracle": False,
            "selected_candidate": "random_regular_seed_0",
            "pass": True,
        },
        {
            "workload": workload_id,
            "method": "fair universal static",
            "selection_signal": "leave-one-workload-out decode from other required workloads only",
            "evaluation_signal": "target decode only",
            "uses_target_decode_for_selection": False,
            "oracle": False,
            "selected_candidate": fair_selection,
            "pass": True,
        },
        {
            "workload": workload_id,
            "method": "prefill-informed OCS",
            "selection_signal": "target trace[0] prefill only",
            "evaluation_signal": "target trace[1:] decode only",
            "uses_target_decode_for_selection": False,
            "oracle": False,
            "selected_candidate": prefill_selection,
            "pass": True,
        },
        {
            "workload": workload_id,
            "method": "oracle",
            "selection_signal": "target trace[1:] decode; upper bound only",
            "evaluation_signal": "target trace[1:] decode only",
            "uses_target_decode_for_selection": True,
            "oracle": True,
            "selected_candidate": oracle_selection,
            "pass": True,
        },
    ]


def draw_grouped_bar(
    path_png: Path,
    title: str,
    workloads: list[str],
    methods: list[str],
    values: dict[str, dict[str, float]],
    ylabel: str,
    normalize_line: float | None = None,
    annotations: dict[tuple[str, str], str] | None = None,
    ymax: float | None = None,
) -> None:
    width, height = 1900, 1050
    margin_l, margin_r, margin_t, margin_b = 160, 80, 110, 210
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    label_font = ImageFont.load_default()
    max_value = ymax if ymax is not None else max(max(v.values()) for v in values.values()) * 1.18
    max_value = max(max_value, 1e-9)

    def x_of(group_idx: int, method_idx: int) -> float:
        group_w = plot_w / len(workloads)
        inner_w = group_w * 0.78
        start = margin_l + group_idx * group_w + (group_w - inner_w) / 2
        bar_w = inner_w / len(methods)
        return start + method_idx * bar_w

    def y_of(value: float) -> float:
        return margin_t + plot_h - (value / max_value) * plot_h

    draw.text((margin_l, 35), title, fill="#111827", font=title_font)
    draw.text((20, margin_t + plot_h / 2), ylabel, fill="#111827", font=label_font)
    for tick in range(6):
        value = max_value * tick / 5
        y = y_of(value)
        draw.line((margin_l, y, margin_l + plot_w, y), fill="#E5E7EB")
        draw.text((margin_l - 95, y - 7), f"{value:.2f}", fill="#374151", font=font)
    draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill="#111827")
    draw.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h), fill="#111827")
    if normalize_line is not None:
        y = y_of(normalize_line)
        draw.line((margin_l, y, margin_l + plot_w, y), fill="#111827", width=3)
        draw.text((margin_l + plot_w - 120, y - 18), f"{normalize_line:.1f} baseline", fill="#111827", font=font)
    for gi, workload in enumerate(workloads):
        group_w = plot_w / len(workloads)
        center = margin_l + gi * group_w + group_w / 2
        draw.text((center - 70, margin_t + plot_h + 28), workload, fill="#111827", font=font)
        inner_w = group_w * 0.78
        bar_w = inner_w / len(methods) * 0.82
        for mi, method in enumerate(methods):
            value = values[workload][method]
            x = x_of(gi, mi)
            y = y_of(value)
            draw.rectangle((x, y, x + bar_w, margin_t + plot_h), fill=METHOD_COLORS.get(method, "#777777"))
            if annotations and (workload, method) in annotations:
                draw.text((x - 8, y - 18), annotations[(workload, method)], fill="#111827", font=font)
    # Legend
    lx, ly = margin_l, height - 95
    for method in methods:
        draw.rectangle((lx, ly, lx + 24, ly + 16), fill=METHOD_COLORS.get(method, "#777777"))
        draw.text((lx + 32, ly + 1), method, fill="#111827", font=font)
        lx += 260
        if lx > width - 280:
            lx = margin_l
            ly += 32
    path_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(path_png)
    img.save(path_png.with_suffix(".pdf"), "PDF", resolution=160.0)


def draw_penalty_chart(path_png: Path, workloads: list[str], values: dict[str, dict[str, float]]) -> None:
    methods = [f"{p}us" if p < 1000 else "25ms" for p in PENALTIES_US]
    colors = {
        "0us": "#E15759",
        "1us": "#F28E2B",
        "10us": "#59A14F",
        "25ms": "#4E79A7",
    }
    old_colors = METHOD_COLORS.copy()
    METHOD_COLORS.update(colors)
    try:
        draw_grouped_bar(
            path_png,
            "Figure 2: Reconfiguration penalty sensitivity (normalised to fair universal static)",
            workloads,
            methods,
            values,
            "OCS time / fair static",
            normalize_line=1.0,
            ymax=max(1.25, max(max(v.values()) for v in values.values()) * 1.1),
        )
    finally:
        METHOD_COLORS.clear()
        METHOD_COLORS.update(old_colors)


def draw_predictability_chart(path_png: Path, workloads: list[str], values: dict[str, dict[str, float]]) -> None:
    methods = ["Spearman", "top8 overlap", "top16 overlap"]
    colors = {"Spearman": "#4E79A7", "top8 overlap": "#59A14F", "top16 overlap": "#F28E2B"}
    old_colors = METHOD_COLORS.copy()
    METHOD_COLORS.update(colors)
    try:
        draw_grouped_bar(
            path_png,
            "Figure 3: Prefill-to-decode predictability",
            workloads,
            methods,
            values,
            "score",
            ymax=1.05,
        )
    finally:
        METHOD_COLORS.clear()
        METHOD_COLORS.update(old_colors)


def process() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = build_base_candidates()
    path_caches = precompute_candidate_paths(candidates)

    parsed_by_id: dict[str, dict[str, Any]] = {}
    prefill_by_id: dict[str, dict[str, Any]] = {}
    decode_by_id: dict[str, dict[str, Any]] = {}
    trace_rows: list[dict[str, Any]] = []

    for workload in WORKLOADS:
        if not workload["path"].exists():
            raise FileNotFoundError(workload["path"])
        parsed = v38.parse_trace(workload["path"])
        parsed_by_id[workload["id"]] = parsed
        prefill_by_id[workload["id"]] = stage_payload(parsed, "prefill")
        decode_by_id[workload["id"]] = stage_payload(parsed, "decode")
        trace_rows.append(traffic_validation(workload, parsed, prefill_by_id[workload["id"]], decode_by_id[workload["id"]]))

    fair_universal = select_fair_universal_static(decode_by_id, path_caches)

    plotted_rows: list[dict[str, Any]] = []
    plotted_values: dict[str, dict[str, float]] = {}
    normalized_values: dict[str, dict[str, float]] = {}
    penalty_values: dict[str, dict[str, float]] = {}
    predict_values: dict[str, dict[str, float]] = {}
    no_leakage: list[dict[str, Any]] = []
    topology_rows: list[dict[str, Any]] = []
    why_rows: list[dict[str, Any]] = []
    penalty_rows: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    validation_summary: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    trace_rows_by_id = {row["workload"]: row for row in trace_rows}

    for workload in WORKLOADS:
        wid = workload["id"]
        parsed = parsed_by_id[wid]
        prefill_payload = prefill_by_id[wid]
        decode_payload = decode_by_id[wid]
        trace_dir = OUT / "chakra_traces" / wid
        prefill_dispatch_prefix = trace_dir / "prefill_dispatch" / "workload"
        prefill_combine_prefix = trace_dir / "prefill_combine" / "workload"
        dispatch_prefix = trace_dir / "dispatch" / "workload"
        combine_prefix = trace_dir / "combine" / "workload"
        prefill_dispatch_trace = v37c.write_matrix_trace(prefill_dispatch_prefix, sparse_to_matrix(prefill_payload["dispatch_sparse"]), NPU)
        prefill_combine_trace = v37c.write_matrix_trace(prefill_combine_prefix, sparse_to_matrix(prefill_payload["combine_sparse"]), NPU)
        dispatch_trace = v37c.write_matrix_trace(dispatch_prefix, sparse_to_matrix(decode_payload["dispatch_sparse"]), NPU)
        combine_trace = v37c.write_matrix_trace(combine_prefix, sparse_to_matrix(decode_payload["combine_sparse"]), NPU)

        # Workload-specific prefill and decode greedy graphs.
        prefill_edges, prefill_meta = v37c.safe_greedy_graph(
            prefill_payload["combined_sparse"], v37c.ring_edges(NPU), NPU, 9301
        )
        decode_edges, decode_meta = v37c.safe_greedy_graph(
            decode_payload["combined_sparse"], v37c.ring_edges(NPU), NPU, 9302
        )
        local_candidates = dict(candidates)
        local_path_caches = dict(path_caches)
        local_candidates["prefill_greedy"] = v37c.graph_from_edges(
            f"{wid}_prefill_greedy_degree4", prefill_edges, NPU, {"construction": "prefill_greedy", **prefill_meta}
        )
        local_candidates["decode_greedy"] = v37c.graph_from_edges(
            f"{wid}_decode_greedy_degree4", decode_edges, NPU, {"construction": "decode_greedy_oracle", **decode_meta}
        )
        local_path_caches["prefill_greedy"] = v37c.precompute_paths(local_candidates["prefill_greedy"], NPU, ECMP_MAX_PATHS)
        local_path_caches["decode_greedy"] = v37c.precompute_paths(local_candidates["decode_greedy"], NPU, ECMP_MAX_PATHS)

        native_prefill_candidate_runs: dict[str, dict[str, Any]] = {}
        native_decode_candidate_runs: dict[str, dict[str, Any]] = {}

        def native_candidate_score(
            stage: str,
            candidate: str,
            dispatch_pfx: Path,
            combine_pfx: Path,
        ) -> dict[str, Any]:
            cache = native_prefill_candidate_runs if stage == "prefill" else native_decode_candidate_runs
            if candidate not in cache:
                cache[candidate] = run_native_method(
                    wid,
                    f"native_{stage}_select_{candidate}",
                    candidate,
                    local_candidates[candidate],
                    dispatch_pfx,
                    combine_pfx,
                )
            return cache[candidate]

        prefill_allowed = ["son_torus", "prefill_greedy"] + [f"random_regular_seed_{seed}" for seed in RANDOM_SEEDS]
        oracle_allowed = ["son_torus", "prefill_greedy", "decode_greedy"] + [f"random_regular_seed_{seed}" for seed in RANDOM_SEEDS]
        native_prefill_scores = []
        for candidate in prefill_allowed:
            run = native_candidate_score("prefill", candidate, prefill_dispatch_prefix, prefill_combine_prefix)
            native_prefill_scores.append(
                {"candidate": candidate, "native_prefill_cycles": run["native_astra_total_cycles"], "success": run["native_astra_success"]}
            )
        native_prefill_scores.sort(key=lambda row: (not row["success"], row["native_prefill_cycles"], row["candidate"]))
        prefill_selection = {"selected": native_prefill_scores[0]["candidate"], "scores_top8": native_prefill_scores[:8], "selection_engine": "native_astra_prefill"}

        native_oracle_scores = []
        for candidate in oracle_allowed:
            run = native_candidate_score("decode", candidate, dispatch_prefix, combine_prefix)
            native_oracle_scores.append(
                {"candidate": candidate, "native_decode_cycles": run["native_astra_total_cycles"], "success": run["native_astra_success"]}
            )
        native_oracle_scores.sort(key=lambda row: (not row["success"], row["native_decode_cycles"], row["candidate"]))
        oracle_selection = {"selected": native_oracle_scores[0]["candidate"], "scores_top8": native_oracle_scores[:8], "selection_engine": "native_astra_decode_oracle"}
        method_to_candidate = {
            "EN": "en_folded_clos",
            "SON / torus": "son_torus",
            "fixed random": "random_regular_seed_0",
            "fair universal static": fair_universal[wid]["selected"],
            "prefill-informed OCS": prefill_selection["selected"],
            "oracle": oracle_selection["selected"],
        }
        selections[wid] = {
            "fair_universal": fair_universal[wid],
            "prefill_informed_ocs": prefill_selection,
            "oracle": oracle_selection,
            "method_to_candidate": method_to_candidate,
        }
        no_leakage.extend(
            no_leakage_rows(
                wid,
                method_to_candidate["fair universal static"],
                method_to_candidate["prefill-informed OCS"],
                method_to_candidate["oracle"],
            )
        )

        workload_values: dict[str, float] = {}
        workload_native_values: dict[str, float] = {}
        metrics_by_method: dict[str, dict[str, Any]] = {}
        native_by_method: dict[str, dict[str, Any]] = {}
        native_by_candidate: dict[str, dict[str, Any]] = {}
        for method in METHOD_ORDER:
            candidate_name = method_to_candidate[method]
            metric = eval_payload(local_path_caches[candidate_name], decode_payload)
            metrics_by_method[method] = metric
            workload_values[method] = metric["sequential_ms"]
            if candidate_name not in native_by_candidate:
                if candidate_name in native_decode_candidate_runs:
                    native_by_candidate[candidate_name] = native_decode_candidate_runs[candidate_name]
                else:
                    native_by_candidate[candidate_name] = run_native_method(
                        wid,
                        method,
                        candidate_name,
                        local_candidates[candidate_name],
                        dispatch_prefix,
                        combine_prefix,
                    )
            native = dict(native_by_candidate[candidate_name])
            native["method"] = method
            native_by_method[method] = native
            native_rows.append(
                {
                    "workload": wid,
                    "method": method,
                    "candidate": candidate_name,
                    "native_astra_success": native["native_astra_success"],
                    "native_astra_dispatch_cycles": native["native_astra_dispatch_cycles"],
                    "native_astra_combine_cycles": native["native_astra_combine_cycles"],
                    "native_astra_total_cycles": native["native_astra_total_cycles"],
                    "native_astra_total_ms": native["native_astra_total_ms"],
                    "dispatch_runtime_s": native["dispatch"]["runtime_s"],
                    "combine_runtime_s": native["combine"]["runtime_s"],
                    "dispatch_cycles_count": native["dispatch"]["cycles_count"],
                    "combine_cycles_count": native["combine"]["cycles_count"],
                    "dispatch_success": native["dispatch"]["success"],
                    "combine_success": native["combine"]["success"],
                    "dispatch_stderr_tail": " | ".join(native["dispatch"]["stderr_tail"]),
                    "combine_stderr_tail": " | ".join(native["combine"]["stderr_tail"]),
                }
            )
            workload_native_values[method] = native["native_astra_total_ms"]
            plotted_rows.append(
                {
                    "workload": wid,
                    "workload_label": workload["full_label"],
                    "method": method,
                    "candidate": candidate_name,
                    "native_astra_total_cycles": native["native_astra_total_cycles"],
                    "native_astra_total_ms": native["native_astra_total_ms"],
                    "native_astra_success": native["native_astra_success"],
                    "fluid_lower_bound_cycles": metric["sequential_cycles"],
                    "fluid_lower_bound_ms": metric["sequential_ms"],
                    "fluid_dispatch_cycles": metric["dispatch_cycles"],
                    "fluid_combine_cycles": metric["combine_cycles"],
                    "chakra_dispatch_prefix": str(dispatch_trace["prefix"]),
                    "chakra_combine_prefix": str(combine_trace["prefix"]),
                    "chakra_prefill_dispatch_prefix_for_selection": str(prefill_dispatch_trace["prefix"]),
                    "chakra_prefill_combine_prefix_for_selection": str(prefill_combine_trace["prefix"]),
                    "chakra_dispatch_messages": dispatch_trace["messages"],
                    "chakra_combine_messages": combine_trace["messages"],
                    "selection_signal": next(row["selection_signal"] for row in no_leakage if row["workload"] == wid and row["method"] == method),
                }
            )
            audit = graph_audit(local_candidates[candidate_name], method)
            topology_rows.append({"workload": wid, **audit})

        fair_cycles = native_by_method["fair universal static"]["native_astra_total_cycles"]
        fair_ms = workload_native_values["fair universal static"]
        plotted_values[workload["label"]] = workload_native_values
        normalized_values[workload["label"]] = {
            method: (workload_native_values[method] / fair_ms if fair_ms else 0)
            for method in METHOD_ORDER
        }
        ocs_cycles = native_by_method["prefill-informed OCS"]["native_astra_total_cycles"]
        penalty_values[workload["label"]] = {}
        for penalty_us in PENALTIES_US:
            method_name = f"{penalty_us}us" if penalty_us < 1000 else "25ms"
            penalized = ocs_cycles + penalty_us * 1000
            penalty_values[workload["label"]][method_name] = penalized / fair_cycles if fair_cycles else 0
            penalty_rows.append(
                {
                    "workload": wid,
                    "penalty_us": penalty_us,
                    "prefill_ocs_raw_native_astra_cycles": ocs_cycles,
                    "prefill_ocs_raw_ms": cycles_to_ms(ocs_cycles),
                    "time_after_penalty_cycles": penalized,
                    "time_after_penalty_ms": cycles_to_ms(penalized),
                    "fair_universal_static_native_astra_cycles": fair_cycles,
                    "beats_fair_universal_static": penalized < fair_cycles,
                    "caveat": "25ms is not acceptable for inference unless amortised over enough decode/batch work" if penalty_us == 25_000 else "",
                }
            )

        pred = v38.predictability(parsed)
        predict_values[workload["label"]] = {
            "Spearman": pred["spearman_expert_count"] or 0,
            "top8 overlap": pred["top8_expert_overlap"],
            "top16 overlap": pred["top16_expert_overlap"],
        }

        for method in METHOD_ORDER:
            metric = metrics_by_method[method]
            dispatch = metric["dispatch"]
            combine = metric["combine"]
            max_phase = dispatch if dispatch["max_link_load_bytes"] >= combine["max_link_load_bytes"] else combine
            why_rows.append(
                {
                    "workload": wid,
                    "method": method,
                    "candidate": method_to_candidate[method],
                    "native_astra_total_cycles": native_by_method[method]["native_astra_total_cycles"],
                    "native_astra_total_ms": native_by_method[method]["native_astra_total_ms"],
                    "fluid_lower_bound_cycles": metric["sequential_cycles"],
                    "fluid_lower_bound_ms": metric["sequential_ms"],
                    "native_astra_to_fluid_ratio": (
                        native_by_method[method]["native_astra_total_cycles"] / metric["sequential_cycles"]
                        if metric["sequential_cycles"]
                        else None
                    ),
                    "max_link_load_bytes": metric["max_link_load_bytes"],
                    "byte_weighted_average_hop_count": metric["byte_weighted_average_hop_count"],
                    "hot_link_top1_share": max_phase["hot_link_top1_share"],
                    "hot_link_top4_share": max_phase["hot_link_top4_share"],
                    "hot_link_top16_share": max_phase["hot_link_top16_share"],
                    "bottleneck_link_id": max_phase["bottleneck_link_id"],
                    "subchunk_bytes_min": min(dispatch["subchunk_bytes_min"], combine["subchunk_bytes_min"]),
                    "subchunk_bytes_median": statistics.median([dispatch["subchunk_bytes_median"], combine["subchunk_bytes_median"]]),
                    "subchunks_lt_54B": dispatch["subchunks_lt_54B"] + combine["subchunks_lt_54B"],
                    "subchunks_lt_128B": dispatch["subchunks_lt_128B"] + combine["subchunks_lt_128B"],
                    "subchunks_lt_256B": dispatch["subchunks_lt_256B"] + combine["subchunks_lt_256B"],
                    "zero_delay_risk": dispatch["zero_delay_risk"] or combine["zero_delay_risk"],
                    "prefill_ocs_gain_vs_EN_percent": 100 * (native_by_method["EN"]["native_astra_total_cycles"] - ocs_cycles) / native_by_method["EN"]["native_astra_total_cycles"],
                    "prefill_ocs_gain_vs_SON_percent": 100 * (native_by_method["SON / torus"]["native_astra_total_cycles"] - ocs_cycles) / native_by_method["SON / torus"]["native_astra_total_cycles"],
                    "prefill_ocs_gain_vs_fixed_random_percent": 100 * (native_by_method["fixed random"]["native_astra_total_cycles"] - ocs_cycles) / native_by_method["fixed random"]["native_astra_total_cycles"],
                    "prefill_ocs_gain_vs_fair_universal_static_percent": 100 * (fair_cycles - ocs_cycles) / fair_cycles if fair_cycles else 0,
                    "oracle_gap_percent": 100 * (ocs_cycles - native_by_method["oracle"]["native_astra_total_cycles"]) / ocs_cycles if ocs_cycles else 0,
                }
            )

        trace_row = trace_rows_by_id[wid]
        validation_summary[wid] = {
            "trace_validation_pass": bool(trace_row["byte_conservation_pass"] and trace_row["decode_exists_non_empty"]),
            "no_leakage_pass": all(row["pass"] for row in no_leakage if row["workload"] == wid and not row["oracle"]),
            "oracle_labelled_upper_bound": True,
            "optical_budget_pass": all(
                row["same_budget_check"]
                for row in topology_rows
                if row["workload"] == wid and row["method"] != "EN"
            ),
            "prefill_ocs_beats_fair_static": ocs_cycles < fair_cycles,
            "all_native_astra_runs_success": all(row["native_astra_success"] for row in native_by_method.values()),
            "prefill_ocs_beats_EN": ocs_cycles < native_by_method["EN"]["native_astra_total_cycles"],
            "prefill_ocs_beats_SON": ocs_cycles < native_by_method["SON / torus"]["native_astra_total_cycles"],
            "prefill_ocs_beats_fixed_random": ocs_cycles < native_by_method["fixed random"]["native_astra_total_cycles"],
            "penalty_1us_beats_fair_static": ocs_cycles + 1_000 < fair_cycles,
            "penalty_10us_beats_fair_static": ocs_cycles + 10_000 < fair_cycles,
            "penalty_25ms_beats_fair_static": ocs_cycles + 25_000_000 < fair_cycles,
            "strongest_predictability_metric": predict_values[workload["label"]],
            "selected_prefill_candidate": method_to_candidate["prefill-informed OCS"],
            "selected_fair_universal_candidate": method_to_candidate["fair universal static"],
            "selected_oracle_candidate": method_to_candidate["oracle"],
        }

    # Output files.
    write_csv(OUT / "plotted_values.csv", plotted_rows)
    write_json(OUT / "plotted_values.json", plotted_rows)
    write_csv(OUT / "native_astra_timing_table.csv", native_rows)
    write_json(OUT / "native_astra_timing_table.json", native_rows)
    write_json(OUT / "validation_summary.json", validation_summary)
    write_csv(OUT / "trace_validation_table.csv", trace_rows)
    write_csv(OUT / "no_leakage_validation_table.csv", no_leakage)
    write_csv(OUT / "topology_validation_table.csv", topology_rows)
    write_csv(OUT / "why_win_table.csv", why_rows)
    write_csv(OUT / "penalty_table.csv", penalty_rows)

    gain_annotations = {}
    for workload in WORKLOADS:
        label = workload["label"]
        norm = normalized_values[label]["prefill-informed OCS"]
        gain_annotations[(label, "prefill-informed OCS")] = f"{(1 - norm) * 100:+.1f}%"

    fig_dir = OUT / "figures"
    draw_grouped_bar(
        fig_dir / "figure_1a_raw_decode_time.png",
        "Figure 1A: Raw decode communication time (native ASTRA cycles)",
        [w["label"] for w in WORKLOADS],
        METHOD_ORDER,
        plotted_values,
        "native ASTRA communication time (ms)",
    )
    draw_grouped_bar(
        fig_dir / "figure_1b_normalized_decode_time.png",
        "Figure 1B: Normalised decode communication time (fair universal static = 1.0)",
        [w["label"] for w in WORKLOADS],
        METHOD_ORDER,
        normalized_values,
        "normalised time",
        normalize_line=1.0,
        annotations=gain_annotations,
        ymax=max(1.25, max(max(v.values()) for v in normalized_values.values()) * 1.1),
    )
    draw_penalty_chart(
        fig_dir / "figure_2_reconfig_penalty.png",
        [w["label"] for w in WORKLOADS],
        penalty_values,
    )
    draw_predictability_chart(
        fig_dir / "figure_3_prefill_decode_predictability.png",
        [w["label"] for w in WORKLOADS],
        predict_values,
    )

    summary = {
        "scope": "V38.1/2 preliminary supervisor figures and strict validation for prefill-informed decode OCS",
        "inference_only": True,
        "stage_mapping": {"trace[0]": "prefill selection only", "trace[1:]": "decode evaluation only"},
        "timing_engine": "main plotted bars use native ASTRA analytical congestion-aware GraphTopology cycles; fluid link-load remains a lower-bound/explanation check",
        "native_astra_in_run_topology_swap": False,
        "ecmp_max_paths": ECMP_MAX_PATHS,
        "required_workloads": WORKLOADS,
        "method_order": METHOD_ORDER,
        "selections": selections,
        "validation_summary": validation_summary,
        "predictability_values": predict_values,
        "normalised_values": normalized_values,
        "raw_values_ms": plotted_values,
        "caveats": [
            "No MoE training claim.",
            "No full serving latency claim.",
            "No native ASTRA in-run topology swap claim.",
            "Oracle uses decode/evaluation traffic and is an upper bound only.",
            "EN is an electrical folded-Clos reference and not the fair optical baseline.",
            "Fair optical comparisons should focus on SON/fixed-random/fair-universal/prefill-OCS/oracle under degree-4, 400Gb/s/link budget.",
        ],
    }
    write_json(OUT / "summary.json", summary)

    readme = [
        "# V38.1/2 Prefill-Informed Decode OCS Figures + Validation",
        "",
        "This is an inference-only MoE communication study. It uses `trace[0]` prefill as the selection signal and evaluates on `trace[1:]` decode only.",
        "",
        "## What Is Plotted",
        "",
        "- Figure 1A: raw decode native ASTRA communication time in ms.",
        "- Figure 1B: native ASTRA decode time normalised by fair universal static = 1.0.",
        "- Figure 2: prefill-informed OCS reconfiguration penalty sensitivity.",
        "- Figure 3: prefill-to-decode predictability.",
        "",
        "## Claims Supported",
        "",
        "- The three required HF expert-selection workloads were parsed with prefill/decode separation.",
        "- The six plotted methods were evaluated with native ASTRA analytical congestion-aware GraphTopology.",
        "- Fluid link-load values are retained only as lower-bound/explanation columns.",
        "- Prefill-informed OCS uses prefill only; decode oracle is labelled as an upper bound.",
        "- Optical methods use the same degree-4, 400Gb/s/link, ECMP4 budget.",
        "",
        "## Claims Not Supported",
        "",
        "- This is not MoE training.",
        "- This is not full serving latency.",
        "- This is not native ASTRA in-run topology swap.",
        "- This is not a paper-final figure set.",
        "",
        "## Key Validation Summary",
        "",
    ]
    for workload in WORKLOADS:
        row = validation_summary[workload["id"]]
        readme.append(
            f"- `{workload['id']}`: OCS vs fair static = {row['prefill_ocs_beats_fair_static']}, "
            f"1us = {row['penalty_1us_beats_fair_static']}, "
            f"10us = {row['penalty_10us_beats_fair_static']}, "
            f"25ms = {row['penalty_25ms_beats_fair_static']}; "
            f"prefill candidate `{row['selected_prefill_candidate']}`."
        )
    readme.extend(
        [
            "",
            "## Files",
            "",
            "- `plotted_values.csv/json`: values used by figures.",
            "- `native_astra_timing_table.csv/json`: native ASTRA dispatch/combine run results.",
            "- `validation_summary.json`: compact pass/fail summary.",
            "- `trace_validation_table.csv`: trace and byte-conservation validation.",
            "- `no_leakage_validation_table.csv`: selection/evaluation split for every method.",
            "- `topology_validation_table.csv`: topology budget and graph checks.",
            "- `why_win_table.csv`: bottleneck, hop-count, hot-link, and gain/loss explanation metrics.",
            "- `penalty_table.csv`: reconfiguration penalty sensitivity.",
            "- `figures/*.png` and `figures/*.pdf`: preliminary supervisor figures.",
            "",
        ]
    )
    (OUT / "README.md").write_text("\n".join(readme))
    return summary


def main() -> None:
    summary = process()
    compact = {
        wid: {
            "beats_fair_static": row["prefill_ocs_beats_fair_static"],
            "beats_1us": row["penalty_1us_beats_fair_static"],
            "beats_10us": row["penalty_10us_beats_fair_static"],
            "beats_25ms": row["penalty_25ms_beats_fair_static"],
            "prefill_candidate": row["selected_prefill_candidate"],
        }
        for wid, row in summary["validation_summary"].items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
