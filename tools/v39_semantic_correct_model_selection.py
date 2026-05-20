#!/usr/bin/env python3
"""V39 FINAL: semantic-correct model selection and controlled figures.

This script separates three semantics that were mixed in earlier drafts:

1. EN electrical packet folded-Clos reference: native ASTRA GraphTopology.
2. Optical SON/RON/OCS circuit-capacity reference: explicit optical model.
3. Current ASTRA GraphTopology packet/store-and-forward sensitivity.

It intentionally does not call optical circuit reference results "ASTRA".
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/moe_expert_trace_converter/v39_semantic_correct_model_selection"
V3812_OUT = REPO / "results/moe_expert_trace_converter/v38_1_2_prefill_decode_figures_validation"
V3812_PATH = REPO / "tools/v38_1_2_prefill_decode_figures_validation.py"
V37C_PATH = REPO / "tools/v37c_128gpu_best_ocs_reconfig_audit.py"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v3812 = load_module("v3812", V3812_PATH)
v37c = load_module("v37c", V37C_PATH)

NPU = 128
DEGREE = 4
LINK_GBPS = 400
ECMP_MAX_PATHS = 4
BYTES_PER_SELECTION = v37c.BYTES_PER_SELECTION
LINK_BYTES_PER_NS = v37c.ASTRA_BYTES_PER_NS
GPU_OPTICAL_BYTES_PER_NS = DEGREE * LINK_BYTES_PER_NS
RANDOM_SEEDS = v3812.RANDOM_SEEDS

WORKLOADS = v3812.WORKLOADS
METHODS_OPTICAL = [
    "SON / torus",
    "fixed random",
    "fair universal static",
    "prefill-informed OCS",
    "oracle",
]
METHODS_WITH_EN = [
    "EN electrical reference",
    "SON optical circuit",
    "fair universal static optical",
    "prefill-informed OCS optical",
    "oracle optical",
]
METHODS_PACKET = v3812.METHOD_ORDER

COLORS = {
    "EN": "#9AA4B2",
    "EN electrical reference": "#9AA4B2",
    "SON / torus": "#4E79A7",
    "SON optical circuit": "#4E79A7",
    "fixed random": "#59A14F",
    "fair universal static": "#F28E2B",
    "fair universal static optical": "#F28E2B",
    "prefill-informed OCS": "#E15759",
    "prefill-informed OCS optical": "#E15759",
    "oracle": "#7B52AB",
    "oracle optical": "#7B52AB",
}

Pair = tuple[int, int]
Sparse = dict[Pair, int]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_graph(graph: dict[str, Any]) -> str:
    blob = json.dumps(graph, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def cycles_to_ms(cycles: int | float) -> float:
    return float(cycles) / 1_000_000.0


def graph_edges(graph: dict[str, Any]) -> list[tuple[int, int]]:
    return [tuple(sorted((int(edge["src"]), int(edge["dst"])))) for edge in graph["edges"]]


def connected_components(graph: dict[str, Any]) -> list[list[int]]:
    node_count = int(graph["node_count"])
    adj = [[] for _ in range(node_count)]
    for a, b in set(graph_edges(graph)):
        adj[a].append(b)
        adj[b].append(a)
    seen: set[int] = set()
    comps: list[list[int]] = []
    for node in range(node_count):
        if node in seen:
            continue
        q = deque([node])
        seen.add(node)
        comp = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(sorted(comp))
    return comps


def route_quality(graph: dict[str, Any]) -> dict[str, Any]:
    paths = v37c.precompute_paths(graph, NPU, ECMP_MAX_PATHS)
    lengths = [len(p[0]) - 1 for p in paths.values() if p]
    counts = [len(p) for p in paths.values() if p]
    intermediate_gpu_paths = 0
    total_paths = 0
    for (src, dst), plist in paths.items():
        for path in plist:
            total_paths += 1
            if any((node < NPU and node not in (src, dst)) for node in path[1:-1]):
                intermediate_gpu_paths += 1
                break
    return {
        "average_route_hop_count": statistics.mean(lengths) if lengths else None,
        "diameter": max(lengths) if lengths else None,
        "route_hop_distribution": dict(sorted(Counter(lengths).items())),
        "ecmp_path_count_distribution_cap4": dict(sorted(Counter(counts).items())),
        "intermediate_gpu_appears_on_any_route": intermediate_gpu_paths > 0,
        "intermediate_gpu_route_pair_count": intermediate_gpu_paths,
    }


def graph_audit(graph: dict[str, Any], method: str, semantics: str) -> dict[str, Any]:
    edges = graph_edges(graph)
    unique_edges = set(edges)
    deg = Counter()
    for a, b in unique_edges:
        deg[a] += 1
        deg[b] += 1
    for node in range(int(graph["node_count"])):
        deg[node] += 0
    gpu_deg = {node: deg[node] for node in range(int(graph["gpu_count"]))}
    comps = connected_components(graph)
    q = route_quality(graph)
    is_optical = method != "EN" and "electrical" not in method.lower()
    return {
        "method": method,
        "graph_name": graph["name"],
        "graph_hash": sha256_graph(graph),
        "topology_type": graph.get("metadata", {}).get("construction", "unknown"),
        "model_semantics": semantics,
        "link_bandwidth_gbps": LINK_GBPS,
        "per_gpu_or_server_bandwidth_tbps": 1.6 if is_optical else 0.4,
        "node_count": graph["node_count"],
        "gpu_count": graph["gpu_count"],
        "non_gpu_switch_nodes": int(graph["node_count"]) - int(graph["gpu_count"]),
        "edge_or_circuit_count": len(unique_edges),
        "degree_distribution_all_nodes": dict(sorted(Counter(deg.values()).items())),
        "degree_distribution_gpu_nodes": dict(sorted(Counter(gpu_deg.values()).items())),
        "connected_components": len(comps),
        "component_sizes": [len(c) for c in comps],
        "duplicate_edges": len(edges) - len(unique_edges),
        "self_loops": sum(1 for a, b in unique_edges if a == b),
        "same_optical_budget_check": (
            is_optical
            and len(unique_edges) == NPU * DEGREE // 2
            and min(gpu_deg.values()) == DEGREE
            and max(gpu_deg.values()) == DEGREE
        ),
        **q,
    }


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


def split_bytes(total: int, parts: int) -> list[int]:
    return v37c.split_bytes(total, parts)


def optical_phase_eval(paths_by_pair: dict[Pair, list[list[int]]], sparse: Sparse) -> dict[str, Any]:
    src_out = [0 for _ in range(NPU)]
    dst_in = [0 for _ in range(NPU)]
    resource_loads: defaultdict[Pair, int] = defaultdict(int)
    selected_counts = []
    byte_weighted_hops = 0
    total_path_bytes = 0
    for (src, dst), size in sparse.items():
        if src == dst or size <= 0:
            continue
        src_out[src] += size
        dst_in[dst] += size
        paths = paths_by_pair[(src, dst)]
        selected_counts.append(len(paths))
        for path, subbytes in zip(paths, split_bytes(size, len(paths))):
            byte_weighted_hops += subbytes * (len(path) - 1)
            total_path_bytes += subbytes
            for u, v in zip(path, path[1:]):
                resource_loads[(u, v)] += subbytes
    src_cycles = int(max(src_out) / GPU_OPTICAL_BYTES_PER_NS) if src_out else 0
    dst_cycles = int(max(dst_in) / GPU_OPTICAL_BYTES_PER_NS) if dst_in else 0
    resource_cycles = int(max(resource_loads.values()) / LINK_BYTES_PER_NS) if resource_loads else 0
    bottleneck_cycles = max(src_cycles, dst_cycles, resource_cycles)
    bottleneck_type = "sender"
    if bottleneck_cycles == dst_cycles and dst_cycles >= src_cycles and dst_cycles >= resource_cycles:
        bottleneck_type = "receiver"
    if bottleneck_cycles == resource_cycles and resource_cycles >= src_cycles and resource_cycles >= dst_cycles:
        bottleneck_type = "optical_resource"
    load_values = sorted(resource_loads.values(), reverse=True)
    total_resource_bytes = sum(load_values)
    return {
        "phase_cycles": bottleneck_cycles,
        "phase_ms": cycles_to_ms(bottleneck_cycles),
        "bottleneck_type": bottleneck_type,
        "sender_bottleneck_cycles": src_cycles,
        "receiver_bottleneck_cycles": dst_cycles,
        "optical_resource_bottleneck_cycles": resource_cycles,
        "max_src_out_bytes": max(src_out) if src_out else 0,
        "max_dst_in_bytes": max(dst_in) if dst_in else 0,
        "max_optical_resource_load_bytes": max(load_values) if load_values else 0,
        "hot_resource_top1_share": load_values[0] / total_resource_bytes if total_resource_bytes else 0,
        "hot_resource_top4_share": sum(load_values[:4]) / total_resource_bytes if total_resource_bytes else 0,
        "hot_resource_top16_share": sum(load_values[:16]) / total_resource_bytes if total_resource_bytes else 0,
        "byte_weighted_resource_hop_count": byte_weighted_hops / total_path_bytes if total_path_bytes else 0,
        "selected_path_count_median": statistics.median(selected_counts) if selected_counts else 0,
        "selected_path_count_max": max(selected_counts) if selected_counts else 0,
    }


def optical_eval(paths_by_pair: dict[Pair, list[list[int]]], payload: dict[str, Any]) -> dict[str, Any]:
    dispatch = optical_phase_eval(paths_by_pair, payload["dispatch_sparse"])
    combine = optical_phase_eval(paths_by_pair, payload["combine_sparse"])
    cycles = dispatch["phase_cycles"] + combine["phase_cycles"]
    return {
        "optical_reference_cycles": cycles,
        "optical_reference_ms": cycles_to_ms(cycles),
        "dispatch": dispatch,
        "combine": combine,
    }


def select_fair_universal(decode_payloads: dict[str, dict[str, Any]], path_caches: dict[str, dict[Pair, list[list[int]]]]) -> dict[str, Any]:
    result = {}
    names = [f"random_regular_seed_{seed}" for seed in RANDOM_SEEDS]
    for target in decode_payloads:
        rows = []
        for name in names:
            norms = []
            for wid, payload in decode_payloads.items():
                if wid == target:
                    continue
                son = optical_eval(path_caches["son_torus"], payload)["optical_reference_cycles"]
                val = optical_eval(path_caches[name], payload)["optical_reference_cycles"]
                norms.append(val / son if son else 0)
            rows.append({"candidate": name, "leave_one_out_avg_norm_to_son": statistics.mean(norms)})
        rows.sort(key=lambda row: (row["leave_one_out_avg_norm_to_son"], row["candidate"]))
        result[target] = {"selected": rows[0]["candidate"], "scores_top8": rows[:8]}
    return result


def select_by_payload(
    path_caches: dict[str, dict[Pair, list[list[int]]]],
    payload: dict[str, Any],
    allowed: list[str],
    score_key: str,
) -> dict[str, Any]:
    rows = []
    for name in allowed:
        val = optical_eval(path_caches[name], payload)["optical_reference_cycles"]
        rows.append({"candidate": name, score_key: val})
    rows.sort(key=lambda row: (row[score_key], row["candidate"]))
    return {"selected": rows[0]["candidate"], "scores_top8": rows[:8]}


def stage_payload(parsed: dict[str, Any], stage: str) -> dict[str, Any]:
    return v3812.stage_payload(parsed, stage)


def traffic_validation(workload: dict[str, Any], parsed: dict[str, Any], decode: dict[str, Any]) -> dict[str, Any]:
    dispatch = sparse_sum(decode["dispatch_sparse"])
    combine = sparse_sum(decode["combine_sparse"])
    combined = combine_sparse(decode["dispatch_sparse"], decode["combine_sparse"])
    vals = [v for v in combined.values() if v > 0]
    return {
        "workload": workload["id"],
        "request_count": parsed["files_used"],
        "prefill_token_count": parsed["prefill_tokens"],
        "decode_token_count": parsed["decode_tokens"],
        "trace0_selection_only": True,
        "trace1_plus_evaluation_only": True,
        "decode_dispatch_bytes": dispatch,
        "decode_combine_bytes": combine,
        "local_bytes_excluded_total": decode["local_bytes_excluded"] * 2,
        "byte_conservation_pass": decode["byte_conservation_pass"],
        "nonzero_gpu_pairs": len(combined),
        "pair_bytes_min": min(vals) if vals else 0,
        "pair_bytes_median": statistics.median(vals) if vals else 0,
        "pair_bytes_max": max(vals) if vals else 0,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def forensic_audit() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    plotted = read_csv(V3812_OUT / "plotted_values.csv")
    native = {(row["workload"], row["method"]): row for row in read_csv(V3812_OUT / "native_astra_timing_table.csv")}
    current_rows = []
    method_rows = []
    en_rows = []
    oracle_rows = []
    by_workload: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in plotted:
        by_workload[row["workload"]][row["method"]] = row
        graph_file = V3812_OUT / "graphs" / f"{row['workload']}__{row['method'].replace(' / ', '_').replace(' ', '_')}__{row['candidate']}.json"
        # The saved graph names use the same safe-label transformation as V38.1/2.
        if not graph_file.exists():
            matches = list((V3812_OUT / "graphs").glob(f"{row['workload']}__*__{row['candidate']}.json"))
            graph_file = matches[0] if matches else graph_file
        graph = json.loads(graph_file.read_text()) if graph_file.exists() else None
        graph_hash = sha256_file(graph_file) if graph_file.exists() else ""
        config_matches = list((V3812_OUT / "network_configs").glob(f"{row['workload']}__*__{row['candidate']}.yml"))
        config_path = config_matches[0] if config_matches else Path("")
        semantics = "native ASTRA GraphTopology packet/store-and-forward"
        audit = graph_audit(graph, row["method"], semantics) if graph else {}
        native_row = native.get((row["workload"], row["method"]), {})
        astra_cmd = ""
        if config_path:
            astra_cmd = (
                f"AstraSim_Analytical_Congestion_Aware --workload-configuration=<dispatch/combine prefix> "
                f"--network-configuration={config_path}"
            )
        out = {
            "workload": row["workload"],
            "method": row["method"],
            "selected_candidate_name": row["candidate"],
            "graph_file": str(graph_file),
            "graph_hash": graph_hash,
            "network_config_path": str(config_path),
            "model_semantics": semantics,
            "astra_command_if_native_run": astra_cmd,
            "dispatch_cycles": row["native_astra_total_cycles"] and native_row.get("native_astra_dispatch_cycles", row.get("native_astra_total_cycles")),
            "combine_cycles": native_row.get("native_astra_combine_cycles", ""),
            "total_cycles": row["native_astra_total_cycles"],
            **audit,
        }
        current_rows.append(out)
        method_rows.append(out)
        if row["method"] == "EN":
            en_rows.append(out)
    for workload, rows in by_workload.items():
        fair = int(rows["fair universal static"]["native_astra_total_cycles"])
        prefill = int(rows["prefill-informed OCS"]["native_astra_total_cycles"])
        oracle = int(rows["oracle"]["native_astra_total_cycles"])
        fair_hash = next((r["graph_hash"] for r in current_rows if r["workload"] == workload and r["method"] == "fair universal static"), "")
        pre_hash = next((r["graph_hash"] for r in current_rows if r["workload"] == workload and r["method"] == "prefill-informed OCS"), "")
        oracle_hash = next((r["graph_hash"] for r in current_rows if r["workload"] == workload and r["method"] == "oracle"), "")
        oracle_rows.append(
            {
                "workload": workload,
                "fair_cycles": fair,
                "prefill_cycles": prefill,
                "oracle_cycles": oracle,
                "oracle_lte_prefill": oracle <= prefill,
                "oracle_lte_fair": oracle <= fair,
                "fair_prefill_same_graph_hash": fair_hash == pre_hash,
                "prefill_oracle_same_graph_hash": pre_hash == oracle_hash,
                "fair_graph_hash": fair_hash,
                "prefill_graph_hash": pre_hash,
                "oracle_graph_hash": oracle_hash,
            }
        )
    return current_rows, en_rows, method_rows, oracle_rows


def draw_grouped_bar(
    path: Path,
    title: str,
    workloads: list[str],
    methods: list[str],
    values: dict[str, dict[str, float]],
    ylabel: str,
    note: str,
    baseline: float | None = None,
    ymax: float | None = None,
) -> None:
    width, height = 1900, 1050
    ml, mr, mt, mb = 160, 80, 115, 215
    pw, ph = width - ml - mr, height - mt - mb
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    maxv = ymax or max(max(row.values()) for row in values.values()) * 1.18
    maxv = max(maxv, 1e-9)

    def y(v: float) -> float:
        return mt + ph - (v / maxv) * ph

    draw.text((ml, 35), title, fill="#111827", font=font)
    draw.text((ml, 58), note, fill="#B91C1C" if "sensitivity" not in note.lower() else "#374151", font=font)
    for i in range(6):
        val = maxv * i / 5
        yy = y(val)
        draw.line((ml, yy, ml + pw, yy), fill="#E5E7EB")
        draw.text((45, yy - 7), f"{val:.2f}", fill="#374151", font=font)
    draw.line((ml, mt, ml, mt + ph), fill="#111827")
    draw.line((ml, mt + ph, ml + pw, mt + ph), fill="#111827")
    if baseline is not None:
        yy = y(baseline)
        draw.line((ml, yy, ml + pw, yy), fill="#111827", width=3)
    for gi, workload in enumerate(workloads):
        gw = pw / len(workloads)
        iw = gw * 0.78
        start = ml + gi * gw + (gw - iw) / 2
        bw = iw / len(methods) * 0.82
        draw.text((ml + gi * gw + gw / 2 - 75, mt + ph + 30), workload, fill="#111827", font=font)
        for mi, method in enumerate(methods):
            val = values[workload][method]
            x = start + mi * (iw / len(methods))
            yy = y(val)
            draw.rectangle((x, yy, x + bw, mt + ph), fill=COLORS.get(method, "#777777"))
            if len(methods) <= 5:
                draw.text((x, yy - 18), f"{val:.2f}", fill="#111827", font=font)
    lx, ly = ml, height - 95
    for method in methods:
        draw.rectangle((lx, ly, lx + 24, ly + 16), fill=COLORS.get(method, "#777777"))
        draw.text((lx + 32, ly + 1), method, fill="#111827", font=font)
        lx += 310
        if lx > width - 300:
            lx = ml
            ly += 32
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    img.save(path.with_suffix(".pdf"), "PDF", resolution=160.0)


def write_modelling_notes() -> None:
    (OUT / "V39_modelling_note.md").write_text(
        """# V39 Modelling Note

## A. EN Electrical Reference

EN is an electrical packet folded-Clos / EPS-style reference. Native ASTRA
GraphTopology is appropriate here because it models graph routing, link queues,
serialization, multi-hop forwarding, and ECMP-like path splitting. EN must be
labelled as an electrical reference unless bandwidth and topology are explicitly
normalised against optical methods.

## B. Current SON/RON GraphTopology

The current ASTRA GraphTopology representation of SON/RON uses rank/GPU nodes as
graph vertices. If a route is `GPU -> GPU -> GPU`, the current backend treats it
as packet/store-and-forward routing through intermediate GPUs. This is a valid
packet-routing sensitivity, but it is not transparent optical OCS semantics.

## C. Intended Optical SON/RON/OCS

The intended optical model is a circuit/capacity fabric. Topology controls which
optical resources/circuits carry traffic. Intermediate optical resources consume
capacity but should not imply GPU packet forwarding. Source and destination
injection limits must be included.

The V39 optical reference timing is:

```text
T_phase = max(
  max_src_out_bytes / B_src,
  max_dst_in_bytes / B_dst,
  max_optical_resource_load / B_link
)
```

with `B_src = B_dst = degree * 400Gb/s = 1.6Tb/s` and `B_link = 400Gb/s`.
Dispatch and combine are sequential.

## D. Figure Policy

- Figure A: optical-only controlled comparison. Use optical circuit reference.
- Figure B: EN electrical reference vs optical methods. Label mixed semantics.
- Figure C: packet-routing ASTRA GraphTopology sensitivity. Do not use as main
  optical OCS result.
- Never mix semantics without labels.
"""
    )
    (OUT / "V39_native_circuit_topology_design.md").write_text(
        """# V39 Native Circuit-Aware ASTRA Design

## Current Blocker

`GraphTopology` routes Chakra SEND/RECV chunks hop by hop. Intermediate graph
nodes are forwarding/queuing points. If SON/RON graph nodes are GPUs, the result
is GPU-as-router packet routing.

## Proposed Addition

Add a `CircuitCapacityNetwork` or `CircuitGraphTopology` path in the analytical
backend. It should:

- load endpoint ranks and optical resource graph separately;
- route SEND/RECV over optical resource paths;
- charge capacity to optical links/resources;
- charge injection only to source and destination ranks;
- avoid treating intermediate rank IDs as GPU forwarding work unless explicitly
  marked as packet-router nodes;
- split messages over ECMP/circuit paths deterministically;
- aggregate callbacks only after all circuit subflows finish.

## Minimal API

```yaml
topology: [ CircuitGraph ]
graph_file: topology.json
routing: ecmp
ecmp_max_paths: 4
endpoint_injection_gbps: 1600
resource_link_gbps: 400
```

## Files Likely Involved

- `extern/network_backend/analytical/congestion_aware/topology/*`
- `CongestionAwareNetworkApi::sim_send`
- Graph route/path-cache code added in V31/V33
- network parser for `CircuitGraph`

## Smoke Tests

1. 4-node direct circuit graph.
2. Optical switch/resource intermediate node without GPU forwarding.
3. 32/128-node SON optical circuit graph.
4. Compare circuit-aware ASTRA timing to V39 optical reference; explain any gap.

Do not generate paper figures from this prototype until smoke tests pass.
"""
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_modelling_notes()

    current_rows, en_rows, method_rows, oracle_rows = forensic_audit()
    write_csv(OUT / "current_result_audit.csv", current_rows)
    write_json(OUT / "current_result_audit.json", current_rows)
    write_csv(OUT / "en_baseline_audit.csv", en_rows)
    write_json(OUT / "en_baseline_audit.json", en_rows)
    write_csv(OUT / "method_topology_audit.csv", method_rows)
    write_json(OUT / "method_topology_audit.json", method_rows)
    write_csv(OUT / "oracle_sanity_table.csv", oracle_rows)
    write_json(OUT / "oracle_sanity_table.json", oracle_rows)

    candidates = v3812.build_base_candidates()
    base_path_caches = v3812.precompute_candidate_paths(candidates)
    parsed_by_id: dict[str, dict[str, Any]] = {}
    prefill_by_id: dict[str, dict[str, Any]] = {}
    decode_by_id: dict[str, dict[str, Any]] = {}
    traffic_rows = []
    for workload in WORKLOADS:
        parsed = v3812.v38.parse_trace(workload["path"])
        parsed_by_id[workload["id"]] = parsed
        prefill_by_id[workload["id"]] = v3812.stage_payload(parsed, "prefill")
        decode_by_id[workload["id"]] = v3812.stage_payload(parsed, "decode")
        traffic_rows.append(traffic_validation(workload, parsed, decode_by_id[workload["id"]]))

    fair = select_fair_universal(decode_by_id, base_path_caches)
    optical_values: list[dict[str, Any]] = []
    optical_validation: list[dict[str, Any]] = []
    no_leakage: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    why_rows: list[dict[str, Any]] = []
    figure_values: dict[str, dict[str, float]] = {}
    normalized: dict[str, dict[str, float]] = {}
    predictability: dict[str, dict[str, float]] = {}
    en_reference_values: dict[str, dict[str, float]] = {}

    packet_values: dict[str, dict[str, float]] = defaultdict(dict)
    for row in read_csv(V3812_OUT / "plotted_values.csv"):
        label = next(w["label"] for w in WORKLOADS if w["id"] == row["workload"])
        packet_values[label][row["method"]] = float(row["native_astra_total_ms"])

    for workload in WORKLOADS:
        wid = workload["id"]
        label = workload["label"]
        prefill = prefill_by_id[wid]
        decode = decode_by_id[wid]
        local_candidates = dict(candidates)
        local_paths = dict(base_path_caches)
        pre_edges, pre_meta = v37c.safe_greedy_graph(prefill["combined_sparse"], v37c.ring_edges(NPU), NPU, 9301)
        dec_edges, dec_meta = v37c.safe_greedy_graph(decode["combined_sparse"], v37c.ring_edges(NPU), NPU, 9302)
        local_candidates["prefill_greedy"] = v37c.graph_from_edges(
            f"{wid}_prefill_greedy_degree4", pre_edges, NPU, {"construction": "prefill_greedy", **pre_meta}
        )
        local_candidates["decode_greedy"] = v37c.graph_from_edges(
            f"{wid}_decode_greedy_degree4", dec_edges, NPU, {"construction": "decode_greedy_oracle", **dec_meta}
        )
        local_paths["prefill_greedy"] = v37c.precompute_paths(local_candidates["prefill_greedy"], NPU, ECMP_MAX_PATHS)
        local_paths["decode_greedy"] = v37c.precompute_paths(local_candidates["decode_greedy"], NPU, ECMP_MAX_PATHS)

        prefill_sel = select_by_payload(
            local_paths,
            prefill,
            ["son_torus", "prefill_greedy"] + [f"random_regular_seed_{s}" for s in RANDOM_SEEDS],
            "prefill_optical_cycles",
        )
        oracle_sel = select_by_payload(
            local_paths,
            decode,
            ["son_torus", "prefill_greedy", "decode_greedy"] + [f"random_regular_seed_{s}" for s in RANDOM_SEEDS],
            "decode_optical_cycles",
        )
        method_to_candidate = {
            "SON / torus": "son_torus",
            "fixed random": "random_regular_seed_0",
            "fair universal static": fair[wid]["selected"],
            "prefill-informed OCS": prefill_sel["selected"],
            "oracle": oracle_sel["selected"],
        }
        figure_values[label] = {}
        for method, cand in method_to_candidate.items():
            graph = local_candidates[cand]
            metric = optical_eval(local_paths[cand], decode)
            figure_values[label][method] = metric["optical_reference_ms"]
            dispatch = metric["dispatch"]
            combine = metric["combine"]
            bottleneck_phase = dispatch if dispatch["phase_cycles"] >= combine["phase_cycles"] else combine
            optical_values.append(
                {
                    "workload": wid,
                    "method": method,
                    "candidate": cand,
                    "graph_hash": sha256_graph(graph),
                    "model_used": "optical circuit/capacity reference",
                    "selection_signal": "decode oracle upper bound" if method == "oracle" else ("prefill only" if method == "prefill-informed OCS" else "static/no target decode"),
                    "evaluation_target": "decode trace[1:] only",
                    "optical_reference_cycles": metric["optical_reference_cycles"],
                    "optical_reference_ms": metric["optical_reference_ms"],
                    "dispatch_cycles": dispatch["phase_cycles"],
                    "combine_cycles": combine["phase_cycles"],
                    "dispatch_bottleneck_type": dispatch["bottleneck_type"],
                    "combine_bottleneck_type": combine["bottleneck_type"],
                    "max_src_out_bytes": max(dispatch["max_src_out_bytes"], combine["max_src_out_bytes"]),
                    "max_dst_in_bytes": max(dispatch["max_dst_in_bytes"], combine["max_dst_in_bytes"]),
                    "max_optical_resource_load_bytes": max(
                        dispatch["max_optical_resource_load_bytes"],
                        combine["max_optical_resource_load_bytes"],
                    ),
                }
            )
            audit = graph_audit(graph, method, "optical circuit/capacity reference")
            budget_rows.append({"workload": wid, **audit})
            why_rows.append(
                {
                    "workload": wid,
                    "method": method,
                    "candidate": cand,
                    "model_used": "optical circuit/capacity reference",
                    "timing_ms": metric["optical_reference_ms"],
                    "bottleneck_type": bottleneck_phase["bottleneck_type"],
                    "sender_bottleneck_cycles": bottleneck_phase["sender_bottleneck_cycles"],
                    "receiver_bottleneck_cycles": bottleneck_phase["receiver_bottleneck_cycles"],
                    "optical_resource_bottleneck_cycles": bottleneck_phase["optical_resource_bottleneck_cycles"],
                    "max_src_out_bytes": bottleneck_phase["max_src_out_bytes"],
                    "max_dst_in_bytes": bottleneck_phase["max_dst_in_bytes"],
                    "max_resource_link_load_bytes": bottleneck_phase["max_optical_resource_load_bytes"],
                    "hot_resource_top1_share": bottleneck_phase["hot_resource_top1_share"],
                    "hot_resource_top4_share": bottleneck_phase["hot_resource_top4_share"],
                    "hot_resource_top16_share": bottleneck_phase["hot_resource_top16_share"],
                    "byte_weighted_resource_hop_count": bottleneck_phase["byte_weighted_resource_hop_count"],
                }
            )
            no_leakage.append(
                {
                    "workload": wid,
                    "method": method,
                    "selection_signal": "target decode/evaluation" if method == "oracle" else ("target prefill only" if method == "prefill-informed OCS" else "static/leave-one-out no target decode"),
                    "uses_target_decode_for_selection": method == "oracle",
                    "oracle_upper_bound": method == "oracle",
                    "leakage_free_non_oracle": method != "oracle",
                    "candidate": cand,
                }
            )
        base = figure_values[label]["fair universal static"]
        normalized[label] = {method: figure_values[label][method] / base if base else 0 for method in METHODS_OPTICAL}
        pred = v3812.v38.predictability(parsed_by_id[wid])
        predictability[label] = {
            "Spearman": pred["spearman_expert_count"] or 0,
            "top8 overlap": pred["top8_expert_overlap"],
            "top16 overlap": pred["top16_expert_overlap"],
        }
        native_en = next(
            float(row["native_astra_total_ms"])
            for row in read_csv(V3812_OUT / "plotted_values.csv")
            if row["workload"] == wid and row["method"] == "EN"
        )
        en_reference_values[label] = {
            "EN electrical reference": native_en,
            "SON optical circuit": figure_values[label]["SON / torus"],
            "fair universal static optical": figure_values[label]["fair universal static"],
            "prefill-informed OCS optical": figure_values[label]["prefill-informed OCS"],
            "oracle optical": figure_values[label]["oracle"],
        }

    for row in optical_values:
        fair_ms = next(x["optical_reference_ms"] for x in optical_values if x["workload"] == row["workload"] and x["method"] == "fair universal static")
        oracle_ms = next(x["optical_reference_ms"] for x in optical_values if x["workload"] == row["workload"] and x["method"] == "oracle")
        row["normalised_to_fair_universal_static"] = row["optical_reference_ms"] / fair_ms if fair_ms else 0
        row["oracle_gap_vs_this_method_percent"] = 100 * (row["optical_reference_ms"] - oracle_ms) / row["optical_reference_ms"] if row["optical_reference_ms"] else 0

    write_csv(OUT / "optical_circuit_reference_values.csv", optical_values)
    write_json(OUT / "optical_circuit_reference_values.json", optical_values)
    write_csv(OUT / "optical_circuit_validation.csv", traffic_rows + optical_values)
    write_json(OUT / "optical_circuit_validation.json", {"traffic": traffic_rows, "values": optical_values})
    write_csv(OUT / "no_leakage_validation.csv", no_leakage)
    write_json(OUT / "no_leakage_validation.json", no_leakage)
    write_csv(OUT / "topology_budget_validation.csv", budget_rows)
    write_json(OUT / "topology_budget_validation.json", budget_rows)
    write_csv(OUT / "why_win_table.csv", why_rows)
    write_json(OUT / "why_win_table.json", why_rows)

    figure_rows = [
        {
            "figure": "A",
            "title": "Optical-only controlled comparison",
            "model": "optical circuit/capacity reference",
            "role": "main optical result",
        },
        {
            "figure": "B",
            "title": "EN electrical reference vs optical methods",
            "model": "EN native ASTRA GraphTopology; optical methods optical circuit reference",
            "role": "reference with explicitly mixed semantics",
        },
        {
            "figure": "C",
            "title": "Packet-routing ASTRA GraphTopology sensitivity",
            "model": "native ASTRA GraphTopology packet/store-and-forward",
            "role": "sensitivity, not main optical OCS result",
        },
        {
            "figure": "D",
            "title": "Prefill-to-decode predictability",
            "model": "trace statistics only",
            "role": "diagnostic",
        },
    ]
    write_csv(OUT / "figure_semantics_table.csv", figure_rows)
    write_json(OUT / "figure_semantics_table.json", figure_rows)

    labels = [w["label"] for w in WORKLOADS]
    fig_dir = OUT / "figures"
    draw_grouped_bar(
        fig_dir / "figure_A1_optical_only_raw.png",
        "Figure A1: Optical-only raw decode communication time",
        labels,
        METHODS_OPTICAL,
        figure_values,
        "optical reference time (ms)",
        "MODEL: optical circuit/capacity reference. Main optical-only controlled comparison.",
    )
    draw_grouped_bar(
        fig_dir / "figure_A2_optical_only_normalized.png",
        "Figure A2: Optical-only normalised decode time",
        labels,
        METHODS_OPTICAL,
        normalized,
        "normalised time",
        "MODEL: optical circuit/capacity reference. Fair universal static = 1.0.",
        baseline=1.0,
        ymax=max(1.25, max(max(v.values()) for v in normalized.values()) * 1.1),
    )
    draw_grouped_bar(
        fig_dir / "figure_B_en_reference_vs_optical.png",
        "Figure B: EN electrical reference vs optical circuit methods",
        labels,
        METHODS_WITH_EN,
        en_reference_values,
        "time (ms)",
        "MIXED SEMANTICS: EN is native ASTRA packet Clos; optical bars use optical circuit reference.",
    )
    draw_grouped_bar(
        fig_dir / "figure_C_packet_graph_topology_sensitivity.png",
        "Figure C: Native ASTRA GraphTopology packet-routing sensitivity",
        labels,
        METHODS_PACKET,
        packet_values,
        "native ASTRA time (ms)",
        "SENSITIVITY ONLY: packet/store-and-forward GraphTopology, not main optical OCS result.",
    )
    draw_grouped_bar(
        fig_dir / "figure_D_prefill_decode_predictability.png",
        "Figure D: Prefill-to-decode predictability",
        labels,
        ["Spearman", "top8 overlap", "top16 overlap"],
        predictability,
        "score",
        "Trace statistics only.",
        ymax=1.05,
    )

    final_answers = {
        "q1_model_for_EN": "Native ASTRA GraphTopology is semantically appropriate for electrical packet folded-Clos EN.",
        "q2_model_for_optical": "The optical circuit/capacity reference is more semantically correct than current GraphTopology for transparent optical SON/RON/OCS.",
        "q3_current_ASTRA_SON_RON_valid_for_optical_claims": False,
        "q4_EN_same_figure_policy": "Separate optical-only Figure A from mixed-reference Figure B; never present EN and optical methods as same-budget without labels.",
        "q5_prefill_ocs_beats_fair_static_optical_only": {
            w["id"]: figure_values[w["label"]]["prefill-informed OCS"] < figure_values[w["label"]]["fair universal static"]
            for w in WORKLOADS
        },
        "q6_oracle_correct_upper_bound": {
            w["id"]: figure_values[w["label"]]["oracle"] <= min(figure_values[w["label"]].values())
            for w in WORKLOADS
        },
        "q7_why_astra_and_reference_differ": "ASTRA GraphTopology models hop-by-hop packet/store-and-forward; optical reference charges sender, receiver, and optical resource capacity without GPU forwarding semantics.",
        "q8_supervisor_discussion_figure": "Figure A plus Figure B, with Figure C only as sensitivity.",
        "q9_paper_direction_figure": "Figure A if the paper targets optical OCS capacity fabrics; Figure C only for packet-routing sensitivity appendix.",
        "q10_next_step": "Design/prototype native circuit-aware ASTRA backend if ASTRA-native optical results are required.",
    }

    summary = {
        "scope": "V39 semantic-correct model selection and controlled-variable figures",
        "workloads": WORKLOADS,
        "traffic_validation": traffic_rows,
        "current_forensic_audit_findings": {
            "EN_faster_than_SON_reason": "current ASTRA Figure C compares packet folded-Clos EN to GPU-as-router multi-hop SON torus; EN has lower average hop count and non-GPU switch nodes.",
            "current_SON_is_GPU_as_router_packet_routing": True,
            "current_graph_topology_not_main_optical_result": True,
        },
        "optical_reference_values": optical_values,
        "figure_values": {
            "figure_A_optical_only_ms": figure_values,
            "figure_A_normalized": normalized,
            "figure_B_en_reference_ms": en_reference_values,
            "figure_C_packet_sensitivity_ms": packet_values,
            "figure_D_predictability": predictability,
        },
        "final_answers": final_answers,
    }
    write_json(OUT / "summary.json", summary)
    (OUT / "final_recommendation.md").write_text(
        f"""# V39 Final Recommendation

## Decision

Use the **optical circuit/capacity reference model** for the main optical-only
comparison. Use native ASTRA GraphTopology for EN electrical reference and for a
separate packet-routing sensitivity figure.

## Why EN Was Faster Than SON Before

The current native ASTRA GraphTopology figure compared electrical packet
folded-Clos EN against SON represented as GPU-as-router multi-hop torus. EN has
switch nodes and shorter routes; SON had long GPU transit paths. That is a
packet-routing sensitivity, not transparent optical OCS semantics.

## Main Paper-Direction Figure

Use Figure A:

- `figures/figure_A1_optical_only_raw.png`
- `figures/figure_A2_optical_only_normalized.png`

## Supervisor Discussion

Use Figure A and Figure B. Show Figure C only to explain why the earlier ASTRA
GraphTopology result was not the right optical main model.

## Final Answers

```json
{json.dumps(final_answers, indent=2)}
```
"""
    )
    (OUT / "README.md").write_text(
        """# V39 Semantic-Correct Model Selection

This output separates electrical packet, optical circuit, and packet-routing
sensitivity semantics. It intentionally avoids one mixed misleading figure.

Key files:

- `V39_modelling_note.md`
- `V39_native_circuit_topology_design.md`
- `current_result_audit.csv/json`
- `optical_circuit_reference_values.csv/json`
- `figure_semantics_table.csv/json`
- `figures/figure_A1_optical_only_raw.png`
- `figures/figure_A2_optical_only_normalized.png`
- `figures/figure_B_en_reference_vs_optical.png`
- `figures/figure_C_packet_graph_topology_sensitivity.png`
- `figures/figure_D_prefill_decode_predictability.png`

Main result: Figure A uses the optical circuit/capacity reference model. Figure C
is native ASTRA packet GraphTopology sensitivity only.
"""
    )
    print(json.dumps(final_answers, indent=2))


if __name__ == "__main__":
    main()
