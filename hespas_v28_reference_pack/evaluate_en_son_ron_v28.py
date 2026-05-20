#!/usr/bin/env python3
"""V2.8 fair EN / SON / RON comparison with torus ECMP and overhead sensitivities."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


Pair = tuple[int, int]
Edge = tuple[int, int]
UnitKey = tuple[str, int, str]
Route = list[tuple[Edge, float]]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    label: str
    path: Path


@dataclass
class Traffic:
    spec: DatasetSpec
    request_ids: list[str]
    files_found: int
    files_used: int
    inferred_num_experts: int
    moe_layers: list[int]
    directed_by_unit: dict[UnitKey, Counter[Pair]]
    pair_by_request: dict[str, Counter[Pair]]
    local_bytes: int
    remote_bytes: int
    malformed_records: int


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def numeric_json_sort(path: Path) -> int | str:
    return int(path.stem) if path.stem.isdigit() else path.stem


def gbps_to_bytes_per_us(gbps: float) -> float:
    return gbps * 1e9 / 8 / 1e6


def expert_rank(expert_id: int, ep_size: int, num_experts: int) -> int:
    experts_per_rank = num_experts / ep_size
    return min(int(expert_id / experts_per_rank), ep_size - 1)


def block_source_rank(global_token_index: int, ep_size: int, block_size: int) -> int:
    return (global_token_index // block_size) % ep_size


def load_traffic(spec: DatasetSpec, ep_size: int, *, hidden_size: int, bytes_per_value: int, block_size: int) -> Traffic:
    files = sorted(spec.path.glob("*.json"), key=numeric_json_sort)
    request_ids = [path.stem for path in files]
    raw: list[tuple[str, int, int, list[int]]] = []
    max_expert = -1
    moe_layers: set[int] = set()
    malformed = 0
    global_token_offset = 0
    for path in files:
        try:
            trace = json.loads(path.read_text())
        except Exception:
            malformed += 1
            continue
        if not isinstance(trace, list) or not trace or not isinstance(trace[0], dict):
            malformed += 1
            continue
        max_rows = 0
        for layer_str, rows in trace[0].items():
            if rows is None:
                continue
            try:
                layer_id = int(layer_str)
            except ValueError:
                malformed += 1
                continue
            if not isinstance(rows, list):
                malformed += 1
                continue
            if rows:
                moe_layers.add(layer_id)
            max_rows = max(max_rows, len(rows))
            for row_index, experts in enumerate(rows):
                if not isinstance(experts, list):
                    malformed += 1
                    continue
                parsed: list[int] = []
                for expert in experts:
                    try:
                        expert_id = int(expert)
                    except Exception:
                        malformed += 1
                        continue
                    parsed.append(expert_id)
                    max_expert = max(max_expert, expert_id)
                raw.append((path.stem, layer_id, global_token_offset + row_index, parsed))
        global_token_offset += max_rows
    if max_expert < 0:
        raise ValueError(f"no expert ids found for {spec.dataset_id}")
    num_experts = max_expert + 1
    bytes_per_selection = hidden_size * bytes_per_value
    directed_by_unit: dict[UnitKey, Counter[Pair]] = defaultdict(Counter)
    pair_by_request: dict[str, Counter[Pair]] = defaultdict(Counter)
    local_bytes = 0
    remote_bytes = 0
    for request_id, layer_id, global_token_index, experts in raw:
        src = block_source_rank(global_token_index, ep_size, block_size)
        for expert_id in experts:
            dst = expert_rank(expert_id, ep_size, num_experts)
            size = bytes_per_selection
            if src == dst:
                local_bytes += size
                continue
            directed_by_unit[(request_id, layer_id, "dispatch")][(src, dst)] += size
            directed_by_unit[(request_id, layer_id, "combine")][(dst, src)] += size
            pair_by_request[request_id][(src, dst)] += size
            pair_by_request[request_id][(dst, src)] += size
            remote_bytes += 2 * size
    return Traffic(spec, request_ids, len(files), len(files), num_experts, sorted(moe_layers), directed_by_unit, pair_by_request, local_bytes, remote_bytes, malformed)


def ring_graph(n: int) -> set[Edge]:
    return {tuple(sorted((i, (i + 1) % n))) for i in range(n)}


def torus_dims(n: int) -> tuple[int, int]:
    if n == 16:
        return 4, 4
    if n == 32:
        return 4, 8
    if n == 64:
        return 8, 8
    raise ValueError(n)


def torus_2d_graph(rows: int, cols: int) -> set[Edge]:
    edges: set[Edge] = set()
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            edges.add(tuple(sorted((node, r * cols + ((c + 1) % cols)))))
            edges.add(tuple(sorted((node, ((r + 1) % rows) * cols + c))))
    return edges


def graph_degree(n: int, edges: set[Edge]) -> Counter[int]:
    deg: Counter[int] = Counter()
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    for i in range(n):
        deg[i] += 0
    return deg


def is_connected(n: int, edges: set[Edge]) -> bool:
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen = {0}
    q = deque([0])
    while q:
        x = q.popleft()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                q.append(y)
    return len(seen) == n


def random_regular_graph(n: int, degree: int, seed: int) -> set[Edge]:
    rng = random.Random(seed)
    stubs = [rank for rank in range(n) for _ in range(degree)]
    for _ in range(50000):
        rng.shuffle(stubs)
        edges: set[Edge] = set()
        ok = True
        for idx in range(0, len(stubs), 2):
            a, b = sorted((stubs[idx], stubs[idx + 1]))
            if a == b or (a, b) in edges:
                ok = False
                break
            edges.add((a, b))
        if ok and is_connected(n, edges):
            return edges
    raise RuntimeError(f"could not build random regular graph n={n}")


def greedy_demand_graph(n: int, degree: int, demand: Counter[Pair], seed_edges: set[Edge]) -> set[Edge]:
    edges = set(seed_edges)
    deg = graph_degree(n, edges)
    undirected: Counter[Edge] = Counter()
    for (src, dst), size in demand.items():
        if src != dst:
            undirected[tuple(sorted((src, dst)))] += size
    for edge, _ in undirected.most_common():
        a, b = edge
        if edge in edges or deg[a] >= degree or deg[b] >= degree:
            continue
        edges.add(edge)
        deg[a] += 1
        deg[b] += 1
    rng = random.Random(777)
    candidates = [tuple(sorted((i, j))) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(candidates)
    for edge in candidates:
        a, b = edge
        if edge in edges or deg[a] >= degree or deg[b] >= degree:
            continue
        edges.add(edge)
        deg[a] += 1
        deg[b] += 1
        if all(deg[i] == degree for i in range(n)):
            break
    return edges


def shortest_paths(n: int, edges: set[Edge]) -> dict[Pair, list[Edge]]:
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    paths: dict[Pair, list[Edge]] = {}
    for src in range(n):
        prev: dict[int, int | None] = {src: None}
        q = deque([src])
        while q:
            x = q.popleft()
            for y in sorted(adj[x]):
                if y not in prev:
                    prev[y] = x
                    q.append(y)
        if len(prev) != n:
            raise ValueError("graph is disconnected")
        for dst in range(n):
            if src == dst:
                continue
            cur = dst
            nodes = [dst]
            while prev[cur] is not None:
                cur = prev[cur]  # type: ignore[assignment]
                nodes.append(cur)
            nodes.reverse()
            paths[(src, dst)] = [tuple(sorted((nodes[i], nodes[i + 1]))) for i in range(len(nodes) - 1)]
    return paths


def torus_ecmp_routes(rows: int, cols: int) -> dict[Pair, Route]:
    """Split evenly across minimal dimension-order torus routes.

    For each dimension, both directions are used when they are tied. When both row
    and column movement are needed, row-first and column-first orders are both used.
    """
    n = rows * cols
    routes: dict[Pair, Route] = {}
    for src in range(n):
        sr, sc = divmod(src, cols)
        for dst in range(n):
            if src == dst:
                continue
            dr, dc = divmod(dst, cols)
            row_steps: list[tuple[int, int]] = []
            down = (dr - sr) % rows
            up = (sr - dr) % rows
            if down and down <= up:
                row_steps.append((1, down))
            if up and up <= down:
                row_steps.append((-1, up))
            col_steps: list[tuple[int, int]] = []
            right = (dc - sc) % cols
            left = (sc - dc) % cols
            if right and right <= left:
                col_steps.append((1, right))
            if left and left <= right:
                col_steps.append((-1, left))
            if not row_steps:
                row_steps = [(0, 0)]
            if not col_steps:
                col_steps = [(0, 0)]
            path_options: list[list[Edge]] = []
            for rdir, rcount in row_steps:
                for cdir, ccount in col_steps:
                    orders = ["row_col", "col_row"] if rcount and ccount else ["row_col"]
                    for order in orders:
                        r, c = sr, sc
                        path: list[Edge] = []
                        dims = [order[:3], order[4:]] if order == "row_col" else [order[:3], order[4:]]
                        # Explicit for readability.
                        sequence = ["row", "col"] if order == "row_col" else ["col", "row"]
                        for dim in sequence:
                            if dim == "row":
                                for _ in range(rcount):
                                    nr = (r + rdir) % rows
                                    path.append(tuple(sorted((r * cols + c, nr * cols + c))))
                                    r = nr
                            else:
                                for _ in range(ccount):
                                    nc = (c + cdir) % cols
                                    path.append(tuple(sorted((r * cols + c, r * cols + nc))))
                                    c = nc
                        path_options.append(path)
            # Deduplicate equal paths, then split equally by path.
            unique = list(dict.fromkeys(tuple(p) for p in path_options))
            route: Counter[Edge] = Counter()
            for path in unique:
                for edge in path:
                    route[edge] += 1 / len(unique)
            routes[(src, dst)] = list(route.items())
    return routes


def en_phase_time(load: Counter[Pair], ep_size: int, bpu: float, leaf_size: int, imbalance_factor: float = 1.0) -> tuple[float, int]:
    num_leaves = math.ceil(ep_size / leaf_size)
    num_spines = max(1, num_leaves)
    link_load: Counter[tuple[str, int, int]] = Counter()
    for (src, dst), size in load.items():
        src_leaf = src // leaf_size
        dst_leaf = dst // leaf_size
        link_load[("gpu_leaf", src, src_leaf)] += size
        link_load[("leaf_gpu", dst_leaf, dst)] += size
        if src_leaf != dst_leaf:
            shard = size / num_spines
            for spine in range(num_spines):
                link_load[("leaf_spine", src_leaf, spine)] += shard
                link_load[("spine_leaf", spine, dst_leaf)] += shard
    max_load = int(max(link_load.values(), default=0))
    return max_load * imbalance_factor / bpu, max_load


def optical_phase_time(load: Counter[Pair], routes: dict[Pair, Route], bpu: float) -> tuple[float, int, float, int, float, float]:
    edge_load: Counter[Edge] = Counter()
    total_bytes = 0
    hop_weighted = 0
    max_hop = 0
    for pair, size in load.items():
        route = routes[pair]
        hop_count = sum(weight for _, weight in route)
        total_bytes += size
        hop_weighted += size * hop_count
        max_hop = max(max_hop, math.ceil(hop_count))
        for edge, weight in route:
            edge_load[edge] += size * weight
    vals = list(edge_load.values())
    max_load = int(max(vals, default=0))
    median_load = float(median(vals)) if vals else 0.0
    avg_load = float(mean(vals)) if vals else 0.0
    return max_load / bpu, max_load, hop_weighted / total_bytes if total_bytes else 0.0, max_hop, median_load, avg_load


def deterministic_routes(paths: dict[Pair, list[Edge]]) -> dict[Pair, Route]:
    return {pair: [(edge, 1.0) for edge in path] for pair, path in paths.items()}


def request_time_en(
    traffic: Traffic,
    reqs: list[str],
    ep_size: int,
    bpu: float,
    leaf_size: int,
    *,
    imbalance_factor: float = 1.0,
    per_message_overhead_us: float = 0.0,
    switch_hop_latency_us: float = 0.0,
) -> dict[str, Any]:
    total_us = 0.0
    max_load = 0
    message_count = 0
    for req in reqs:
        for layer in traffic.moe_layers:
            for phase in ("dispatch", "combine"):
                load = traffic.directed_by_unit.get((req, layer, phase), Counter())
                t, max_link = en_phase_time(load, ep_size, bpu, leaf_size, imbalance_factor)
                cross_leaf = sum(1 for (src, dst), size in load.items() if size and src // leaf_size != dst // leaf_size)
                same_leaf = sum(1 for (src, dst), size in load.items() if size and src // leaf_size == dst // leaf_size)
                latency = (cross_leaf * 4 + same_leaf * 2) * switch_hop_latency_us + len(load) * per_message_overhead_us
                total_us += t + latency
                message_count += len(load)
                max_load = max(max_load, max_link)
    return {"completion_time_ms": total_us / 1000, "max_link_load_bytes": max_load, "message_count": message_count}


def request_time_optical(traffic: Traffic, reqs: list[str], routes: dict[Pair, Route], bpu: float) -> dict[str, Any]:
    total_us = 0.0
    max_load = 0
    median_sum = 0.0
    avg_sum = 0.0
    hop_sum = 0.0
    max_hop = 0
    units = 0
    for req in reqs:
        for layer in traffic.moe_layers:
            for phase in ("dispatch", "combine"):
                t, load, hop, mh, med_load, avg_load = optical_phase_time(traffic.directed_by_unit.get((req, layer, phase), Counter()), routes, bpu)
                total_us += t
                max_load = max(max_load, load)
                median_sum += med_load
                avg_sum += avg_load
                hop_sum += hop
                max_hop = max(max_hop, mh)
                units += 1
    return {
        "completion_time_ms": total_us / 1000,
        "max_link_load_bytes": max_load,
        "median_link_load_bytes": median_sum / units if units else 0,
        "average_link_load_bytes": avg_sum / units if units else 0,
        "average_hop_count": hop_sum / units if units else 0,
        "max_hop_count": max_hop,
    }


def sum_optical_results(items: list[dict[str, Any]], reconfig_us: float = 0.0) -> dict[str, Any]:
    total = sum(float(item["completion_time_ms"]) for item in items) + len(items) * reconfig_us / 1000
    weight = total or len(items) or 1
    return {
        "completion_time_ms": total,
        "max_link_load_bytes": max(int(item["max_link_load_bytes"]) for item in items) if items else 0,
        "median_link_load_bytes": mean(float(item["median_link_load_bytes"]) for item in items) if items else 0,
        "average_link_load_bytes": mean(float(item["average_link_load_bytes"]) for item in items) if items else 0,
        "average_hop_count": sum(float(item["average_hop_count"]) * (float(item["completion_time_ms"]) or 1.0) for item in items) / weight if items else 0,
        "max_hop_count": max(int(item["max_hop_count"]) for item in items) if items else 0,
        "exposed_reconfiguration_time_ms": len(items) * reconfig_us / 1000,
    }


def aggregate_load(traffic: Traffic, reqs: list[str]) -> Counter[Pair]:
    out: Counter[Pair] = Counter()
    for req in reqs:
        out.update(traffic.pair_by_request.get(req, Counter()))
    return out


def proxy_time(load: Counter[Pair], routes: dict[Pair, Route], bpu: float) -> float:
    edge_load: Counter[Edge] = Counter()
    for pair, size in load.items():
        for edge, weight in routes[pair]:
            edge_load[edge] += size * weight
    return max(edge_load.values(), default=0) / bpu


def plot_grouped(path_png: Path, path_pdf: Path, rows: list[dict[str, Any]], title: str) -> None:
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(path_png.parent / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = list(dict.fromkeys(row["dataset_label"] for row in rows))
    modes = list(dict.fromkeys(row["network_mode"] for row in rows))
    width = 0.82 / len(modes)
    xs = list(range(len(datasets)))
    fig, ax = plt.subplots(figsize=(16, 6.4))
    for i, mode in enumerate(modes):
        vals = [float(next(row for row in rows if row["dataset_label"] == ds and row["network_mode"] == mode)["completion_time_ms"]) for ds in datasets]
        offsets = [x + (i - (len(modes) - 1) / 2) * width for x in xs]
        ax.bar(offsets, vals, width=width, label=mode)
    ax.set_title(title)
    ax.set_ylabel("Communication completion time (ms)")
    ax.set_xticks(xs)
    ax.set_xticklabels(datasets, rotation=12, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncols=3)
    fig.tight_layout()
    fig.savefig(path_png, dpi=220)
    fig.savefig(path_pdf)
    plt.close(fig)


def plot_lines(path_png: Path, path_pdf: Path, rows: list[dict[str, Any]], title: str, y_field: str, ylabel: str) -> None:
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(path_png.parent / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(dict.fromkeys(row["network_mode"] for row in rows))
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    for mode in modes:
        subset = sorted([row for row in rows if row["network_mode"] == mode], key=lambda row: int(row["ep_size"]))
        ax.plot([int(row["ep_size"]) for row in subset], [float(row[y_field]) for row in subset], marker="o", label=mode)
    ax.set_title(title)
    ax.set_xlabel("GPU count")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_png, dpi=220)
    fig.savefig(path_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/moe_expert_trace_converter/v28_fair_en_son_ron_ecmp_overhead"))
    parser.add_argument("--ep-sizes", default="16,32,64")
    parser.add_argument("--random-seeds", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        DatasetSpec("qwen_mmlu_ml", "Qwen MMLU ML", Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/mmlu/machine_learning")),
        DatasetSpec("qwen_livecode", "Qwen LiveCodeBench", Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/livecodebench/execution")),
        DatasetSpec("qwen_mmlu_zh_anatomy", "Qwen ZH Anatomy", Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/mmlu_ZH_CN/anatomy")),
        DatasetSpec("deepseek_livecode", "DeepSeek LiveCodeBench", Path("/Users/dfx/Python/trace/cognitivecomputations/DeepSeek-R1-AWQ/livecodebench/execution")),
    ]
    ep_sizes = [int(x) for x in args.ep_sizes.split(",") if x]
    en_bpu = gbps_to_bytes_per_us(400)
    ring_bpu = gbps_to_bytes_per_us(800)
    optical_bpu = gbps_to_bytes_per_us(400)
    leaf_size = 8

    rows: list[dict[str, Any]] = []
    topo_rows: list[dict[str, Any]] = []
    en_overhead_rows: list[dict[str, Any]] = []
    torus_compare_rows: list[dict[str, Any]] = []
    ron_reconfig_rows: list[dict[str, Any]] = []
    oversub_rows: list[dict[str, Any]] = []
    path_cache: dict[tuple[Edge, ...], dict[Pair, Route]] = {}

    for ep_size in ep_sizes:
        leaves = math.ceil(ep_size / leaf_size)
        spines = leaves
        down = leaf_size * 400
        up = spines * 400
        oversub_rows.append({"ep_size": ep_size, "gpus_per_leaf": leaf_size, "num_leaves": leaves, "num_spines": spines, "gpu_downlink_per_leaf_gbps": down, "leaf_spine_uplink_per_leaf_gbps": up, "oversubscription_ratio": down / up})

    def det_routes(graph: set[Edge]) -> dict[Pair, Route]:
        key = tuple(sorted(graph))
        if key not in path_cache:
            path_cache[key] = deterministic_routes(shortest_paths(max(max(e) for e in graph) + 1, graph))
        return path_cache[key]

    for spec in specs:
        for ep_size in ep_sizes:
            print(f"V2.8 {spec.dataset_id} ep={ep_size}", flush=True)
            traffic = load_traffic(spec, ep_size, hidden_size=4096, bytes_per_value=2, block_size=args.block_size)
            cal_count = max(1, min(math.ceil(traffic.files_used * 0.10), traffic.files_used - 1))
            cal = traffic.request_ids[:cal_count]
            ev = traffic.request_ids[cal_count:]
            base = {
                "dataset_id": spec.dataset_id,
                "dataset_label": spec.label,
                "ep_size": ep_size,
                "experts_per_gpu": traffic.inferred_num_experts / ep_size,
                "files_used": traffic.files_used,
                "calibration_requests": len(cal),
                "evaluation_requests": len(ev),
                "source_policy": "block_by_token",
                "block_size": args.block_size,
                "expert_placement": "block",
                "local_bytes_excluded": traffic.local_bytes,
                "remote_bytes": traffic.remote_bytes,
            }
            ring = ring_graph(ep_size)
            torus = torus_2d_graph(*torus_dims(ep_size))
            ring_routes = det_routes(ring)
            torus_det_routes = det_routes(torus)
            torus_ecmp = torus_ecmp_routes(*torus_dims(ep_size))

            en_ideal = request_time_en(traffic, ev, ep_size, en_bpu, leaf_size)
            row_en = {**base, "network_mode": "EN ideal Clos", "degree": "", "per_gpu_bandwidth_tbps": 0.4, "per_link_bandwidth_gbps": 400, "routing": "ideal_ECMP", "reconfigurations": 0, **en_ideal}
            rows.append(row_en)
            en13 = request_time_en(traffic, ev, ep_size, en_bpu, leaf_size, imbalance_factor=1.3)
            rows.append({**base, "network_mode": "EN ECMP-imbalance 1.3x", "degree": "", "per_gpu_bandwidth_tbps": 0.4, "per_link_bandwidth_gbps": 400, "routing": "ideal_ECMP_times_1.3", "reconfigurations": 0, **en13})
            for factor in (1.2, 1.3):
                r = request_time_en(traffic, ev, ep_size, en_bpu, leaf_size, imbalance_factor=factor)
                en_overhead_rows.append({**base, "variant": f"ECMP-imbalance {factor}x", **r})
            rlat = request_time_en(traffic, ev, ep_size, en_bpu, leaf_size, per_message_overhead_us=0.5, switch_hop_latency_us=0.05)
            en_overhead_rows.append({**base, "variant": "latency-sensitive 0.5us_msg_0.05us_hop", **rlat})
            rcons = request_time_en(traffic, ev, ep_size, en_bpu, leaf_size, imbalance_factor=1.3, per_message_overhead_us=0.5, switch_hop_latency_us=0.05)
            en_overhead_rows.append({**base, "variant": "conservative 1.3x_plus_latency", **rcons})

            ring_r = request_time_optical(traffic, ev, ring_routes, ring_bpu)
            rows.append({**base, "network_mode": "SON ring", "degree": 2, "per_gpu_bandwidth_tbps": 1.6, "per_link_bandwidth_gbps": 800, "routing": "deterministic_shortest_path", "reconfigurations": 0, **ring_r})
            tor_det = request_time_optical(traffic, ev, torus_det_routes, optical_bpu)
            rows.append({**base, "network_mode": "SON torus deterministic", "degree": 4, "per_gpu_bandwidth_tbps": 1.6, "per_link_bandwidth_gbps": 400, "routing": "deterministic_shortest_path", "reconfigurations": 0, **tor_det})
            tor_ecmp = request_time_optical(traffic, ev, torus_ecmp, optical_bpu)
            rows.append({**base, "network_mode": "SON torus ECMP", "degree": 4, "per_gpu_bandwidth_tbps": 1.6, "per_link_bandwidth_gbps": 400, "routing": "equal_cost_shortest_path_split", "reconfigurations": 0, **tor_ecmp})
            torus_compare_rows.append({**base, "det_time_ms": tor_det["completion_time_ms"], "ecmp_time_ms": tor_ecmp["completion_time_ms"], "det_max_link_load_bytes": tor_det["max_link_load_bytes"], "ecmp_max_link_load_bytes": tor_ecmp["max_link_load_bytes"], "det_median_link_load_bytes": tor_det["median_link_load_bytes"], "ecmp_median_link_load_bytes": tor_ecmp["median_link_load_bytes"], "det_max_over_median": tor_det["max_link_load_bytes"] / tor_det["median_link_load_bytes"] if tor_det["median_link_load_bytes"] else "", "ecmp_max_over_median": tor_ecmp["max_link_load_bytes"] / tor_ecmp["median_link_load_bytes"] if tor_ecmp["median_link_load_bytes"] else ""})

            random_graphs = [random_regular_graph(ep_size, 4, seed) for seed in range(args.random_seeds)]
            greedy = greedy_demand_graph(ep_size, 4, aggregate_load(traffic, cal), ring)
            if not is_connected(ep_size, greedy) or not all(v == 4 for v in graph_degree(ep_size, greedy).values()):
                greedy = torus
            candidate_graphs = [torus, greedy] + random_graphs
            candidate_routes = [det_routes(g) for g in candidate_graphs]

            def score(idx: int, reqs: list[str]) -> float:
                return proxy_time(aggregate_load(traffic, reqs), candidate_routes[idx], optical_bpu)

            best_cal = min(range(len(candidate_graphs)), key=lambda idx: score(idx, cal))
            ron_cal = request_time_optical(traffic, ev, candidate_routes[best_cal], optical_bpu)
            rows.append({**base, "network_mode": "RON calibrated", "degree": 4, "per_gpu_bandwidth_tbps": 1.6, "per_link_bandwidth_gbps": 400, "routing": "deterministic_shortest_path_on_selected_graph", "candidate_pool_size": len(candidate_graphs), "selection_objective": "minimize_calibration_aggregate_max_link_load", "candidate_index": best_cal, "reconfigurations": 1, **ron_cal})

            w_items = []
            oracle_items = []
            for idx, req in enumerate(ev):
                prev = ev[max(0, idx - 4):idx] or cal
                best_w = min(range(len(candidate_graphs)), key=lambda graph_idx: score(graph_idx, prev))
                best_o = min(range(len(candidate_graphs)), key=lambda graph_idx: score(graph_idx, [req]))
                w_items.append(request_time_optical(traffic, [req], candidate_routes[best_w], optical_bpu))
                oracle_items.append(request_time_optical(traffic, [req], candidate_routes[best_o], optical_bpu))
            for reconfig_us in (0, 1, 10):
                ron_w = sum_optical_results(w_items, reconfig_us=reconfig_us)
                row = {**base, "network_mode": f"RON W=4 {reconfig_us}us", "degree": 4, "per_gpu_bandwidth_tbps": 1.6, "per_link_bandwidth_gbps": 400, "routing": "deterministic_shortest_path_on_selected_graph", "candidate_pool_size": len(candidate_graphs), "selection_objective": "previous_4_requests_minimize_max_link_load", "reconfigurations": len(ev), "reconfig_us_per_request": reconfig_us, **ron_w}
                rows.append(row)
                ron_reconfig_rows.append(row)
            ron_oracle = sum_optical_results(oracle_items)
            rows.append({**base, "network_mode": "RON oracle", "degree": 4, "per_gpu_bandwidth_tbps": 1.6, "per_link_bandwidth_gbps": 400, "routing": "deterministic_shortest_path_on_selected_graph", "candidate_pool_size": len(candidate_graphs), "selection_objective": "true_current_request_minimize_max_link_load", "reconfigurations": 0, **ron_oracle})

            topo_rows.extend([
                {**base, "topology": "SON ring", "connected": is_connected(ep_size, ring), "degree_values": json.dumps(dict(graph_degree(ep_size, ring))), "routing": "deterministic"},
                {**base, "topology": "SON torus deterministic", "connected": is_connected(ep_size, torus), "degree_values": json.dumps(dict(graph_degree(ep_size, torus))), "routing": "deterministic"},
                {**base, "topology": "SON torus ECMP", "connected": is_connected(ep_size, torus), "degree_values": json.dumps(dict(graph_degree(ep_size, torus))), "routing": "ECMP"},
                {**base, "topology": "RON candidate selected calibrated", "connected": is_connected(ep_size, candidate_graphs[best_cal]), "degree_values": json.dumps(dict(graph_degree(ep_size, candidate_graphs[best_cal]))), "candidate_pool_size": len(candidate_graphs), "candidate_index": best_cal},
            ])

    for row in rows:
        group = [r for r in rows if r["dataset_id"] == row["dataset_id"] and int(r["ep_size"]) == int(row["ep_size"])]
        en = next(r for r in group if r["network_mode"] == "EN ideal Clos")
        ring = next(r for r in group if r["network_mode"] == "SON ring")
        torus = next(r for r in group if r["network_mode"] == "SON torus ECMP")
        t = float(row["completion_time_ms"])
        row["improvement_over_en_ideal"] = (float(en["completion_time_ms"]) - t) / float(en["completion_time_ms"])
        row["improvement_over_son_ring"] = (float(ring["completion_time_ms"]) - t) / float(ring["completion_time_ms"])
        row["improvement_over_son_torus_ecmp"] = (float(torus["completion_time_ms"]) - t) / float(torus["completion_time_ms"])

    write_csv(args.output_dir / "v28_full_summary.csv", rows)
    (args.output_dir / "v28_full_summary.json").write_text(json.dumps(rows, indent=2, default=str))
    write_csv(args.output_dir / "topology_routing_summary.csv", topo_rows)
    write_csv(args.output_dir / "en_oversubscription_summary.csv", oversub_rows)
    write_csv(args.output_dir / "en_overhead_sensitivity_summary.csv", en_overhead_rows)
    write_csv(args.output_dir / "torus_deterministic_vs_ecmp_link_load_summary.csv", torus_compare_rows)
    write_csv(args.output_dir / "ron_reconfiguration_sensitivity_summary.csv", ron_reconfig_rows)

    fig1_modes = ["EN ideal Clos", "EN ECMP-imbalance 1.3x", "SON ring", "SON torus deterministic", "SON torus ECMP", "RON calibrated", "RON W=4 0us", "RON W=4 1us", "RON oracle"]
    fig1 = [r for r in rows if int(r["ep_size"]) == 32 and r["network_mode"] in fig1_modes]
    fig1 = sorted(fig1, key=lambda r: (r["dataset_id"], fig1_modes.index(r["network_mode"])))
    write_csv(args.output_dir / "fig1_32gpu_four_workloads.csv", fig1)
    plot_grouped(args.output_dir / "fig1_32gpu_four_workloads.png", args.output_dir / "fig1_32gpu_four_workloads.pdf", fig1, "EN / SON / RON at 32 GPUs Across Four MoE Traces")

    fig2_modes = ["SON torus deterministic", "SON torus ECMP", "RON calibrated", "RON W=4 0us", "RON W=4 1us", "RON oracle"]
    fig2 = [r for r in rows if r["dataset_id"] == "qwen_mmlu_ml" and r["network_mode"] in fig2_modes]
    fig2 = sorted(fig2, key=lambda r: (fig2_modes.index(r["network_mode"]), int(r["ep_size"])))
    write_csv(args.output_dir / "fig2_qwen_mmlu_scaling.csv", fig2)
    plot_lines(args.output_dir / "fig2_qwen_mmlu_scaling.png", args.output_dir / "fig2_qwen_mmlu_scaling.pdf", fig2, "Qwen MMLU: SON Torus ECMP vs RON as GPU Count Increases", "completion_time_ms", "Communication completion time (ms)")

    fig3_modes = ["RON calibrated", "RON W=4 0us", "RON W=4 1us", "RON oracle"]
    fig3 = [dict(r, gain_over_son_torus_ecmp_percent=100 * float(r["improvement_over_son_torus_ecmp"])) for r in rows if r["dataset_id"] == "qwen_mmlu_ml" and r["network_mode"] in fig3_modes]
    fig3 = sorted(fig3, key=lambda r: (r["network_mode"], int(r["ep_size"])))
    write_csv(args.output_dir / "fig3_ron_gain_over_torus_ecmp.csv", fig3)
    plot_lines(args.output_dir / "fig3_ron_gain_over_torus_ecmp.png", args.output_dir / "fig3_ron_gain_over_torus_ecmp.pdf", fig3, "RON Gain over SON 2D Torus ECMP", "gain_over_son_torus_ecmp_percent", "Improvement over SON torus ECMP (%)")

    validation = {
        "results_from_astra_sim": False,
        "results_from_custom_analytical_evaluator": True,
        "astra_sim_only_used_for_earlier_smoke_tests": True,
        "en_uses_no_optical_paths": True,
        "son_ron_use_no_eps_fallback": True,
        "torus_deterministic_and_ecmp_same_topology_bandwidth": True,
        "son_torus_and_ron_degree4_1p6tbps_per_gpu_400gbps_per_link": True,
        "son_ring_degree2_1p6tbps_per_gpu_800gbps_per_link": True,
        "local_traffic_excluded": True,
        "dispatch_combine_sequential": True,
        "ron_calibrated_uses_first_10_percent": True,
        "ron_w4_reconfigures_once_per_request": True,
        "multi_hop_semantics": "abstract optical switching fabric path; if interpreted as GPU/NIC forwarding this is a serious modelling limitation",
    }
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2))

    readme = """# V2.8 Fair EN / SON / RON with Torus ECMP and Overhead Sensitivity

These results are from our custom analytical evaluator, not ASTRA-sim. ASTRA-sim was only used for earlier smoke tests.

The pipeline is:

HF expert-selection JSON -> parser -> block-by-token source mapping -> block expert placement -> dispatch/combine traffic matrix -> analytical network timing model -> CSV/figures.

V2.8 keeps V2.7's prefill-only workload assumptions and adds:

- `SON torus deterministic`: old deterministic shortest-path torus.
- `SON torus ECMP`: same torus topology and bandwidth, but splits traffic across equal-cost shortest paths.
- EN sensitivity variants: ECMP imbalance factors and latency-sensitive/conservative variants.
- RON W=4 reconfiguration sensitivity: 0us, 1us, 10us per evaluated request.

Main fair comparison:

- `RON calibrated` vs `SON torus ECMP`
- `RON W=4 0us/1us` vs `SON torus ECMP`

Diagnostic baselines:

- `SON ring`: weak degree-2 optical baseline.
- `SON torus deterministic`: shows the penalty from deterministic shortest-path tie-breaking.
- `EN ECMP-imbalance 1.3x`: sensitivity, not the main EN baseline.

Multi-hop optical semantics:

The SON/RON graph-routing model should be interpreted as an abstract optical switching fabric path. If intermediate graph hops are interpreted as GPU/NIC forwarding through intermediate GPUs, that is a serious modelling limitation and the optical multi-hop results should not be treated as physically valid without a forwarding/switching model.

Why EN beat SON in V2.7:

V2.7 gave EN ideal ECMP over a Clos abstraction, while SON torus used deterministic shortest paths. That deterministic torus routing could overload a small number of torus edges. SON ring also has degree 2 but 800Gb/s per link, which can outperform a 400Gb/s/link torus for some traffic patterns despite longer paths. Therefore EN beating SON in some V2.7 cases was mostly a combination of weak SON routing/topology and EN idealisation, not a definitive real-world result.

Do not double count:

Max-link-load / bandwidth already accounts for serialization. Additional EN latency and imbalance rows are sensitivity analyses, not the main model.
"""
    (args.output_dir / "README_V2_8.md").write_text(readme)
    print(args.output_dir)


if __name__ == "__main__":
    main()
