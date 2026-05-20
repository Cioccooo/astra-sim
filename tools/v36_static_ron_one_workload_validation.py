#!/usr/bin/env python3
"""V36 static RON calibrated/oracle validation on one HF MoE workload."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/moe_expert_trace_converter/v36_static_ron_one_workload_validation"
TRACE_DIR = Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/mmlu/machine_learning")
V35_OUT = REPO / "results/moe_expert_trace_converter/v35_one_workload_aggregated_native_astra_validation"
PROTO_DIR = REPO / "extern/graph_frontend/chakra/schema/protobuf"
PROTO_UTILS = REPO / "extern/graph_frontend/chakra/src/third_party/utils"

sys.path.insert(0, str(PROTO_DIR))
sys.path.insert(0, str(PROTO_UTILS))

from et_def_pb2 import AttributeProto, GlobalMetadata, Node, NodeType  # type: ignore  # noqa: E402
from protolib import encodeMessage as encode_message  # type: ignore  # noqa: E402

ASTRA_BIN = REPO / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
SYSTEM = REPO / "examples/system/native_collectives/Ring_4chunks.json"
REMOTE_MEMORY = REPO / "examples/remote_memory/analytical/no_memory_expansion.json"

NPU_COUNT = 32
DEGREE = 4
HIDDEN_SIZE = 4096
BYTES_PER_VALUE = 2
BYTES_PER_SELECTION = HIDDEN_SIZE * BYTES_PER_VALUE
BLOCK_SIZE = 16
LINK_GBPS = 400
ASTRA_BYTES_PER_NS = (LINK_GBPS / 8) * (1 << 30) / 1_000_000_000
ONE_CYCLE_THRESHOLD_BYTES = math.ceil(ASTRA_BYTES_PER_NS)
RANDOM_CANDIDATES = 32

Pair = tuple[int, int]
Edge = tuple[int, int]


def numeric_json_sort(path: Path) -> int | str:
    return int(path.stem) if path.stem.isdigit() else path.stem


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def matrix_checksum(matrix: list[list[int]]) -> str:
    blob = json.dumps(matrix, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def add_attr(node: Node, name: str, value: int | bool) -> None:
    if isinstance(value, bool):
        node.attr.append(AttributeProto(name=name, bool_val=value))
    else:
        node.attr.append(AttributeProto(name=name, uint64_val=int(value)))


def comp_node(node_id: int) -> Node:
    node = Node()
    node.id = node_id
    node.name = "DUMMY_COMP_NODE"
    node.type = NodeType.COMP_NODE
    node.duration_micros = 0
    add_attr(node, "is_cpu_op", False)
    return node


def comm_node(node_id: int, is_send: bool, src: int, dst: int, size: int, tag: int) -> Node:
    node = Node()
    node.id = node_id
    node.name = ("SEND" if is_send else "RECV") + f"_{src}_to_{dst}_tag{tag}"
    node.type = NodeType.COMM_SEND_NODE if is_send else NodeType.COMM_RECV_NODE
    add_attr(node, "is_cpu_op", False)
    add_attr(node, "comm_src", src)
    add_attr(node, "comm_dst", dst)
    add_attr(node, "comm_size", size)
    add_attr(node, "comm_tag", tag)
    return node


def write_matrix_trace(prefix: Path, matrix: list[list[int]]) -> dict[str, Any]:
    nodes_by_rank: dict[int, list[Node]] = {rank: [] for rank in range(NPU_COUNT)}
    next_id = {rank: 1 for rank in range(NPU_COUNT)}
    tag = 1
    total_bytes = 0
    messages = 0
    for src in range(NPU_COUNT):
        for dst in range(NPU_COUNT):
            size = int(matrix[src][dst])
            if src == dst or size <= 0:
                continue
            nodes_by_rank[src].append(comm_node(next_id[src], True, src, dst, size, tag))
            next_id[src] += 1
            nodes_by_rank[dst].append(comm_node(next_id[dst], False, src, dst, size, tag))
            next_id[dst] += 1
            total_bytes += size
            messages += 1
            tag += 1
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for rank in range(NPU_COUNT):
        nodes = nodes_by_rank[rank] or [comp_node(1)]
        with (prefix.parent / f"{prefix.name}.{rank}.et").open("wb") as handle:
            encode_message(handle, GlobalMetadata(version="0.0.4"))
            for node in nodes:
                encode_message(handle, node)
    return {"prefix": str(prefix), "rank_count": NPU_COUNT, "messages": messages, "total_bytes": total_bytes}


def expert_rank(expert_id: int, ep_size: int, num_experts: int) -> int:
    experts_per_rank = num_experts / ep_size
    return min(int(expert_id / experts_per_rank), ep_size - 1)


def block_source_rank(global_token_index: int, ep_size: int, block_size: int) -> int:
    return (global_token_index // block_size) % ep_size


def zero_matrix() -> list[list[int]]:
    return [[0 for _ in range(NPU_COUNT)] for _ in range(NPU_COUNT)]


def add_matrix(dst: list[list[int]], src: list[list[int]]) -> None:
    for i in range(NPU_COUNT):
        for j in range(NPU_COUNT):
            dst[i][j] += src[i][j]


def parse_requests() -> dict[str, Any]:
    files = sorted(TRACE_DIR.glob("*.json"), key=numeric_json_sort)
    request_raw: list[dict[str, Any]] = []
    max_expert = -1
    malformed_records = 0
    global_token_offset = 0
    moe_layers: set[int] = set()
    full_selected_events = 0

    for path in files:
        try:
            trace = json.loads(path.read_text())
        except Exception:
            malformed_records += 1
            continue
        if not isinstance(trace, list) or not trace or not isinstance(trace[0], dict):
            malformed_records += 1
            continue
        rows_for_request: list[tuple[int, int, list[int]]] = []
        max_rows = 0
        for layer_str, rows in trace[0].items():
            if rows is None:
                continue
            try:
                layer_id = int(layer_str)
            except ValueError:
                malformed_records += 1
                continue
            if not isinstance(rows, list):
                malformed_records += 1
                continue
            if rows:
                moe_layers.add(layer_id)
            max_rows = max(max_rows, len(rows))
            for row_index, experts in enumerate(rows):
                if not isinstance(experts, list):
                    malformed_records += 1
                    continue
                parsed: list[int] = []
                for expert in experts:
                    try:
                        expert_id = int(expert)
                    except Exception:
                        malformed_records += 1
                        continue
                    parsed.append(expert_id)
                    max_expert = max(max_expert, expert_id)
                rows_for_request.append((layer_id, global_token_offset + row_index, parsed))
                full_selected_events += len(parsed)
        request_raw.append({"request_id": path.stem, "rows": rows_for_request, "prefill_tokens": max_rows})
        global_token_offset += max_rows

    if max_expert < 0:
        raise RuntimeError(f"No expert ids found under {TRACE_DIR}")
    num_experts = max_expert + 1

    def build_for_indices(indices: list[int]) -> dict[str, Any]:
        dispatch = zero_matrix()
        combine = zero_matrix()
        local_selections = 0
        remote_selections = 0
        selected_events = 0
        prefill_tokens = 0
        request_ids: list[str] = []
        for idx in indices:
            req = request_raw[idx]
            request_ids.append(req["request_id"])
            prefill_tokens += int(req["prefill_tokens"])
            for _, global_token_index, experts in req["rows"]:
                src_rank = block_source_rank(global_token_index, NPU_COUNT, BLOCK_SIZE)
                for expert_id in experts:
                    selected_events += 1
                    dst_rank = expert_rank(expert_id, NPU_COUNT, num_experts)
                    if src_rank == dst_rank:
                        local_selections += 1
                        continue
                    dispatch[src_rank][dst_rank] += BYTES_PER_SELECTION
                    combine[dst_rank][src_rank] += BYTES_PER_SELECTION
                    remote_selections += 1
        remote_bytes = remote_selections * BYTES_PER_SELECTION
        local_bytes = local_selections * BYTES_PER_SELECTION
        theoretical = selected_events * BYTES_PER_SELECTION
        return {
            "request_ids": request_ids,
            "request_count": len(indices),
            "prefill_input_tokens": prefill_tokens,
            "selected_expert_events": selected_events,
            "theoretical_dispatch_bytes": theoretical,
            "theoretical_combine_bytes": theoretical,
            "local_dispatch_bytes_excluded": local_bytes,
            "local_combine_bytes_excluded": local_bytes,
            "remote_dispatch_bytes_retained": sum(sum(row) for row in dispatch),
            "remote_combine_bytes_retained": sum(sum(row) for row in combine),
            "byte_conservation_pass": (
                sum(sum(row) for row in dispatch) == remote_bytes
                and sum(sum(row) for row in combine) == remote_bytes
                and theoretical == local_bytes + remote_bytes
            ),
            "dispatch_matrix": dispatch,
            "combine_matrix": combine,
            "dispatch_checksum": matrix_checksum(dispatch),
            "combine_checksum": matrix_checksum(combine),
        }

    request_count = len(request_raw)
    cal_count = max(1, min(math.ceil(request_count * 0.10), request_count - 1))
    all_indices = list(range(request_count))
    cal_indices = list(range(cal_count))
    eval_indices = list(range(cal_count, request_count))

    return {
        "files_found": len(files),
        "files_used": request_count,
        "moe_layers": sorted(moe_layers),
        "moe_layer_count": len(moe_layers),
        "inferred_num_experts": num_experts,
        "malformed_records": malformed_records,
        "full_selected_events_seen": full_selected_events,
        "full": build_for_indices(all_indices),
        "calibration": build_for_indices(cal_indices),
        "evaluation": build_for_indices(eval_indices),
    }


def matrix_to_demand(dispatch: list[list[int]], combine: list[list[int]]) -> Counter[Pair]:
    demand: Counter[Pair] = Counter()
    for matrix in (dispatch, combine):
        for src in range(NPU_COUNT):
            for dst in range(NPU_COUNT):
                if src != dst and matrix[src][dst] > 0:
                    demand[(src, dst)] += matrix[src][dst]
    return demand


def torus_edges() -> set[Edge]:
    rows, cols = 4, 8
    edges: set[Edge] = set()
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            edges.add(tuple(sorted((node, r * cols + ((c + 1) % cols)))))
            edges.add(tuple(sorted((node, ((r + 1) % rows) * cols + c))))
    return edges


def ring_edges() -> set[Edge]:
    return {tuple(sorted((i, (i + 1) % NPU_COUNT))) for i in range(NPU_COUNT)}


def graph_degree(edges: set[Edge]) -> Counter[int]:
    deg: Counter[int] = Counter()
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    for i in range(NPU_COUNT):
        deg[i] += 0
    return deg


def connected_components(edges: set[Edge]) -> list[list[int]]:
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen: set[int] = set()
    comps: list[list[int]] = []
    for node in range(NPU_COUNT):
        if node in seen:
            continue
        comp: list[int] = []
        q: deque[int] = deque([node])
        seen.add(node)
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(sorted(comp))
    return comps


def is_connected(edges: set[Edge]) -> bool:
    return len(connected_components(edges)) == 1


def random_regular_graph(seed: int) -> set[Edge]:
    rng = random.Random(seed)
    stubs = [rank for rank in range(NPU_COUNT) for _ in range(DEGREE)]
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
        if ok and is_connected(edges):
            return edges
    raise RuntimeError(f"could not build random regular graph seed={seed}")


def greedy_demand_graph(demand: Counter[Pair], seed_edges: set[Edge]) -> set[Edge]:
    edges = set(seed_edges)
    deg = graph_degree(edges)
    undirected: Counter[Edge] = Counter()
    for (src, dst), size in demand.items():
        if src != dst:
            undirected[tuple(sorted((src, dst)))] += size
    for edge, _ in undirected.most_common():
        a, b = edge
        if edge in edges or deg[a] >= DEGREE or deg[b] >= DEGREE:
            continue
        edges.add(edge)
        deg[a] += 1
        deg[b] += 1
    rng = random.Random(777)
    candidates = [tuple(sorted((i, j))) for i in range(NPU_COUNT) for j in range(i + 1, NPU_COUNT)]
    rng.shuffle(candidates)
    for edge in candidates:
        a, b = edge
        if edge in edges or deg[a] >= DEGREE or deg[b] >= DEGREE:
            continue
        edges.add(edge)
        deg[a] += 1
        deg[b] += 1
        if all(deg[i] == DEGREE for i in range(NPU_COUNT)):
            break
    return edges


def graph_from_edges(name: str, edges: set[Edge], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "node_count": NPU_COUNT,
        "gpu_count": NPU_COUNT,
        "directed": False,
        "metadata": {
            "degree": DEGREE,
            "per_gpu_optical_bandwidth_tbps": 1.6,
            "per_link_bandwidth_gbps": LINK_GBPS,
            **(metadata or {}),
        },
        "edges": [
            {"src": a, "dst": b, "bandwidth_gbps": LINK_GBPS, "latency_ns": 0}
            for a, b in sorted(edges)
        ],
    }


def edges_from_graph(graph: dict[str, Any]) -> set[Edge]:
    return {tuple(sorted((int(edge["src"]), int(edge["dst"])))) for edge in graph["edges"]}


def adjacency_from_graph(graph: dict[str, Any]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(graph["node_count"])]
    directed = bool(graph.get("directed", False))
    for edge in graph["edges"]:
        src, dst = int(edge["src"]), int(edge["dst"])
        bidirectional = bool(edge.get("bidirectional", not directed))
        adj[src].append(dst)
        if bidirectional:
            adj[dst].append(src)
    for neighbors in adj:
        neighbors.sort()
    return adj


def all_shortest_paths(graph: dict[str, Any], src: int, dst: int) -> list[list[int]]:
    adj = adjacency_from_graph(graph)
    distance = [-1] * len(adj)
    parents: list[list[int]] = [[] for _ in adj]
    q: deque[int] = deque([src])
    distance[src] = 0
    while q:
        node = q.popleft()
        for nxt in adj[node]:
            if distance[nxt] == -1:
                distance[nxt] = distance[node] + 1
                parents[nxt].append(node)
                q.append(nxt)
            elif distance[nxt] == distance[node] + 1:
                parents[nxt].append(node)
    if distance[dst] == -1:
        return []
    paths: list[list[int]] = []
    current: list[int] = []

    def backtrack(node: int) -> None:
        current.append(node)
        if node == src:
            paths.append(list(reversed(current)))
        else:
            for parent in sorted(parents[node]):
                backtrack(parent)
        current.pop()

    backtrack(dst)
    return sorted(paths)


def selected_paths(graph: dict[str, Any], src: int, dst: int, max_paths: int = 4) -> list[list[int]]:
    paths = all_shortest_paths(graph, src, dst)
    return paths[:max_paths]


def split_bytes(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def link_load_estimate(graph: dict[str, Any], matrix: list[list[int]], max_paths: int = 4) -> dict[str, Any]:
    loads: dict[tuple[int, int], int] = defaultdict(int)
    selected_counts: list[int] = []
    hop_counts: list[int] = []
    byte_weighted_hops = 0
    total_bytes = 0
    for src, row in enumerate(matrix):
        for dst, size in enumerate(row):
            if src == dst or size <= 0:
                continue
            paths = selected_paths(graph, src, dst, max_paths)
            if not paths:
                raise RuntimeError(f"no path {src}->{dst}")
            selected_counts.append(len(paths))
            for path, subbytes in zip(paths, split_bytes(size, len(paths))):
                hop_count = len(path) - 1
                hop_counts.append(hop_count)
                byte_weighted_hops += subbytes * hop_count
                total_bytes += subbytes
                for u, v in zip(path, path[1:]):
                    loads[(u, v)] += subbytes
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
        "fluid_cycles": int(max_load / ASTRA_BYTES_PER_NS) if max_load else 0,
        "hot_links": [
            {"src": u, "dst": v, "bytes": b}
            for (u, v), b in sorted(loads.items(), key=lambda x: -x[1])[:8]
        ],
    }


def combined_link_load_score(graph: dict[str, Any], dispatch: list[list[int]], combine: list[list[int]]) -> int:
    return max(
        link_load_estimate(graph, dispatch, 4)["max_link_load_bytes"],
        link_load_estimate(graph, combine, 4)["max_link_load_bytes"],
    )


def build_candidate_graphs(cal_demand: Counter[Pair], eval_demand: Counter[Pair]) -> dict[str, dict[str, Any]]:
    torus = graph_from_edges("son_torus_4x8_32gpu", torus_edges(), {"construction": "static_4x8_torus"})
    cal_greedy_edges = greedy_demand_graph(cal_demand, ring_edges())
    eval_greedy_edges = greedy_demand_graph(eval_demand, ring_edges())
    candidates = {
        "son_torus": torus,
        "calibration_greedy": graph_from_edges(
            "ron_calibration_greedy_degree4",
            cal_greedy_edges,
            {
                "construction": "greedy_demand_graph",
                "traffic_source": "calibration_dispatch_plus_combine",
                "seed_edges": "ring_degree2",
            },
        ),
        "evaluation_greedy": graph_from_edges(
            "ron_evaluation_greedy_degree4",
            eval_greedy_edges,
            {
                "construction": "greedy_demand_graph",
                "traffic_source": "evaluation_dispatch_plus_combine",
                "seed_edges": "ring_degree2",
            },
        ),
    }
    for seed in range(RANDOM_CANDIDATES):
        candidates[f"random_regular_seed_{seed}"] = graph_from_edges(
            f"ron_random_regular_seed_{seed}",
            random_regular_graph(seed),
            {"construction": "random_regular_degree4", "seed": seed},
        )
    return candidates


def select_candidate(
    candidates: dict[str, dict[str, Any]],
    dispatch: list[list[int]],
    combine: list[list[int]],
    allowed_names: list[str],
) -> dict[str, Any]:
    scores = []
    for name in allowed_names:
        graph = candidates[name]
        score = combined_link_load_score(graph, dispatch, combine)
        scores.append({"name": name, "score_max_link_load_bytes": score})
    best = min(scores, key=lambda item: (item["score_max_link_load_bytes"], item["name"]))
    return {"selected_name": best["name"], "scores": sorted(scores, key=lambda item: (item["score_max_link_load_bytes"], item["name"]))}


def network_config(path: Path, graph_path: Path) -> None:
    lines = [
        "topology: [ Graph ]",
        f"npus_count: [ {NPU_COUNT} ]",
        "bandwidth: [ 50.0 ]",
        "latency: [ 0.0 ]",
        f"graph_file: {graph_path}",
        "routing: ecmp",
        "ecmp_split: equal_bytes",
        "ecmp_max_paths: 4",
        "ecmp_log: false",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def gini(values: list[int]) -> float:
    vals = sorted(v for v in values if v >= 0)
    total = sum(vals)
    if not vals or total == 0:
        return 0.0
    n = len(vals)
    return sum((2 * i - n - 1) * value for i, value in enumerate(vals, 1)) / (n * total)


def entropy(values: list[int]) -> float:
    total = sum(values)
    if total <= 0:
        return 0.0
    return -sum((value / total) * math.log2(value / total) for value in values if value > 0)


def matrix_stats(matrix: list[list[int]]) -> dict[str, Any]:
    messages = [matrix[src][dst] for src in range(NPU_COUNT) for dst in range(NPU_COUNT) if src != dst and matrix[src][dst] > 0]
    total = sum(messages)
    sorted_messages = sorted(messages, reverse=True)

    def top_share(k: int) -> float:
        return sum(sorted_messages[:k]) / total if total else 0.0

    return {
        "total_remote_bytes": total,
        "nonzero_gpu_pairs": len(messages),
        "top1_share": top_share(1),
        "top4_share": top_share(4),
        "top8_share": top_share(8),
        "top16_share": top_share(16),
        "gini": gini(messages),
        "entropy_bits": entropy(messages),
        "message_bytes_min": min(messages) if messages else 0,
        "message_bytes_median": statistics.median(messages) if messages else 0,
        "message_bytes_mean": statistics.mean(messages) if messages else 0,
        "message_bytes_max": max(messages) if messages else 0,
    }


def graph_audit(graph: dict[str, Any]) -> dict[str, Any]:
    edges = edges_from_graph(graph)
    deg = graph_degree(edges)
    comps = connected_components(edges)
    self_loops = [edge for edge in edges if edge[0] == edge[1]]
    duplicate_count = len(graph["edges"]) - len(edges)
    return {
        "name": graph["name"],
        "node_count": graph["node_count"],
        "gpu_count": graph["gpu_count"],
        "edge_circuit_count": len(edges),
        "degree_distribution": dict(sorted(Counter(deg.values()).items())),
        "degree_min": min(deg.values()),
        "degree_max": max(deg.values()),
        "connected_components": len(comps),
        "component_sizes": [len(comp) for comp in comps],
        "duplicate_edges": duplicate_count,
        "self_loops": len(self_loops),
        "per_gpu_optical_bandwidth_tbps": 1.6,
        "per_link_bandwidth_gbps": LINK_GBPS,
        "same_degree_bandwidth_budget_as_son": len(edges) == 64 and min(deg.values()) == DEGREE and max(deg.values()) == DEGREE,
        "valid": len(comps) == 1 and duplicate_count == 0 and not self_loops and len(edges) == 64 and min(deg.values()) == DEGREE and max(deg.values()) == DEGREE,
    }


def graph_quality(graph: dict[str, Any]) -> dict[str, Any]:
    lengths: list[int] = []
    path_counts: list[int] = []
    unreachable = 0
    for src in range(NPU_COUNT):
        for dst in range(NPU_COUNT):
            if src == dst:
                continue
            paths = all_shortest_paths(graph, src, dst)
            if not paths:
                unreachable += 1
                continue
            lengths.append(len(paths[0]) - 1)
            path_counts.append(min(len(paths), 4))
    return {
        "average_shortest_path_length": statistics.mean(lengths) if lengths else None,
        "diameter": max(lengths) if lengths else None,
        "shortest_path_length_distribution": dict(sorted(Counter(lengths).items())),
        "unreachable_pairs": unreachable,
        "ecmp_path_count_distribution_cap4": dict(sorted(Counter(path_counts).items())),
        "ecmp_path_count_min_cap4": min(path_counts) if path_counts else 0,
        "ecmp_path_count_median_cap4": statistics.median(path_counts) if path_counts else 0,
        "ecmp_path_count_mean_cap4": statistics.mean(path_counts) if path_counts else 0,
        "ecmp_path_count_max_cap4": max(path_counts) if path_counts else 0,
    }


def tiny_subchunk_audit(graph: dict[str, Any], matrix: list[list[int]]) -> dict[str, Any]:
    subchunks: list[int] = []
    selected_counts: list[int] = []
    for src in range(NPU_COUNT):
        for dst in range(NPU_COUNT):
            size = matrix[src][dst]
            if src == dst or size <= 0:
                continue
            paths = selected_paths(graph, src, dst, 4)
            selected_counts.append(len(paths))
            subchunks.extend(split_bytes(size, len(paths)))
    return {
        "one_cycle_threshold_bytes": ONE_CYCLE_THRESHOLD_BYTES,
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
        "zero_delay_risk": any(value < ONE_CYCLE_THRESHOLD_BYTES for value in subchunks),
    }


def run_astra(label: str, workload: Path, network: Path) -> dict[str, Any]:
    stdout = OUT / "runs" / f"{label}.stdout.txt"
    stderr = OUT / "runs" / f"{label}.stderr.txt"
    stdout.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ASTRA_BIN),
        f"--workload-configuration={workload}",
        f"--system-configuration={SYSTEM}",
        f"--remote-memory-configuration={REMOTE_MEMORY}",
        f"--network-configuration={network}",
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, check=False)
    runtime_s = time.perf_counter() - start
    stdout.write_text(proc.stdout)
    stderr.write_text(proc.stderr)
    cycles = [int(match) for match in re.findall(r"finished, ([0-9]+) cycles", proc.stdout)]
    return {
        "label": label,
        "returncode": proc.returncode,
        "success": proc.returncode == 0 and len(cycles) == NPU_COUNT,
        "runtime_s": runtime_s,
        "command": " ".join(cmd),
        "stdout": str(stdout),
        "stderr": str(stderr),
        "stderr_tail": proc.stderr.splitlines()[-8:],
        "max_cycles": max(cycles) if cycles else None,
        "cycles_count": len(cycles),
    }


def write_report(summary: dict[str, Any]) -> None:
    (OUT / "README.md").write_text(
        f"""# V3.6 Static RON One-Workload Validation

## Scope

This adds static RON calibrated/oracle topology generation to the V35 aggregated native ASTRA pipeline. It is a validation step, not a final paper result.

## RON Construction

- Candidate pool: SON 4x8 torus, calibration-greedy degree-4 graph, evaluation-greedy degree-4 graph, and {RANDOM_CANDIDATES} deterministic random regular degree-4 graphs.
- RON calibrated selection uses **only calibration dispatch+combine traffic**.
- RON oracle selection uses evaluation dispatch+combine traffic and is labelled oracle/reference only.
- Edges are bidirectional optical circuits.
- Degree is enforced at 4 for every GPU.
- Per-link bandwidth is 400 Gb/s; per-GPU budget is degree 4 x 400 Gb/s = 1.6 Tb/s.
- Routing is native GraphTopology ECMP with `ecmp_max_paths=4`.

## Final Answers

1. Does V36 match the V35 full-workload aggregate before splitting? **{summary["v35_consistency"]["matches_v35"]}.**
2. Does RON calibrated use only calibration traffic? **{summary["anti_leakage"]["ron_calibrated_uses_only_calibration"]}.**
3. Does RON oracle act only as an oracle reference? **{summary["anti_leakage"]["ron_oracle_reference_only"]}.**
4. Are SON, RON calibrated, and RON oracle under the same degree/bandwidth budget? **{summary["graph_budget_fairness_pass"]}.**
5. Are all RON graphs connected and valid? **{summary["ron_graphs_valid"]}.**
6. Do RON calibrated/oracle run successfully in native ASTRA? **{summary["ron_runs_success"]}.**
7. Does RON calibrated beat SON ECMP4 on evaluation traffic? **{summary["gain_loss_explanation"]["ron_calibrated_beats_son"]}.**
8. Does RON oracle beat SON ECMP4 on evaluation traffic? **{summary["gain_loss_explanation"]["ron_oracle_beats_son"]}.**
9. If RON helps, why? See gain/loss explanation below.
10. If RON does not help, why not? See gain/loss explanation below.
11. Is it safe to proceed to four-workload static RON validation next? **{summary["final_recommendation"]["safe_to_proceed_to_four_workload_static"]}.**

## Calibration / Evaluation Split

```json
{json.dumps(summary["split"], indent=2)}
```

## V35 Consistency

```json
{json.dumps(summary["v35_consistency"], indent=2)}
```

## Traffic Audit

```json
{json.dumps(summary["traffic_fingerprint"], indent=2)}
```

## Graph Audit

```json
{json.dumps(summary["graph_audit"], indent=2)}
```

## Graph Quality

```json
{json.dumps(summary["graph_quality"], indent=2)}
```

## Tiny-Subchunk Audit

```json
{json.dumps(summary["tiny_subchunk_audit"], indent=2)}
```

## Native ASTRA Timing

```json
{json.dumps(summary["native_astra_totals"], indent=2)}
```

## Fluid Lower Bound

```json
{json.dumps(summary["fluid_lower_bound"], indent=2)}
```

## Gain / Loss Explanation

```json
{json.dumps(summary["gain_loss_explanation"], indent=2)}
```

## What V36 Can Claim

- Static RON graph generation and native ASTRA execution work for one real aggregated workload.
- RON calibrated/oracle are compared with SON ECMP4 under the same degree/bandwidth budget.

## What V36 Cannot Claim

- W=4 dynamic reconfiguration works.
- Results generalise to all workloads.
- Real serving latency.
- Physical transparent OCS modelling.
- Token/layer-level execution timing.
"""
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    graphs_dir = OUT / "graphs"
    configs_dir = OUT / "network_configs"
    matrices_dir = OUT / "traffic_matrices"
    traces_dir = OUT / "chakra_traces"

    parsed = parse_requests()
    full = parsed["full"]
    cal = parsed["calibration"]
    ev = parsed["evaluation"]

    v35_summary = json.loads((V35_OUT / "summary.json").read_text())
    v35_dispatch = json.loads((V35_OUT / "traffic_matrices/dispatch_matrix.json").read_text())["matrix"]
    v35_combine = json.loads((V35_OUT / "traffic_matrices/combine_matrix.json").read_text())["matrix"]
    v35_consistency = {
        "files_used_match": full["request_count"] == v35_summary["trace_parse"]["files_used"],
        "prefill_tokens_match": full["prefill_input_tokens"] == v35_summary["trace_parse"]["prefill_input_tokens"],
        "moe_layer_count_match": parsed["moe_layer_count"] == v35_summary["trace_parse"]["moe_layer_count"],
        "selected_events_match": full["selected_expert_events"] == v35_summary["trace_parse"]["selected_expert_events"],
        "theoretical_dispatch_bytes_match": full["theoretical_dispatch_bytes"] == v35_summary["trace_parse"]["theoretical_dispatch_bytes"],
        "local_bytes_match": full["local_dispatch_bytes_excluded"] == v35_summary["trace_parse"]["local_dispatch_bytes_excluded"],
        "remote_bytes_match": full["remote_dispatch_bytes_retained"] == v35_summary["trace_parse"]["remote_dispatch_bytes_retained"],
        "dispatch_checksum_match": full["dispatch_checksum"] == matrix_checksum(v35_dispatch),
        "combine_checksum_match": full["combine_checksum"] == matrix_checksum(v35_combine),
    }
    v35_consistency["matches_v35"] = all(v35_consistency.values())

    write_json(matrices_dir / "full_dispatch_matrix.json", {"phase": "dispatch", "matrix": full["dispatch_matrix"], "checksum": full["dispatch_checksum"]})
    write_json(matrices_dir / "full_combine_matrix.json", {"phase": "combine", "matrix": full["combine_matrix"], "checksum": full["combine_checksum"]})
    write_json(matrices_dir / "calibration_dispatch_matrix.json", {"phase": "dispatch", "matrix": cal["dispatch_matrix"], "checksum": cal["dispatch_checksum"]})
    write_json(matrices_dir / "calibration_combine_matrix.json", {"phase": "combine", "matrix": cal["combine_matrix"], "checksum": cal["combine_checksum"]})
    write_json(matrices_dir / "evaluation_dispatch_matrix.json", {"phase": "dispatch", "matrix": ev["dispatch_matrix"], "checksum": ev["dispatch_checksum"]})
    write_json(matrices_dir / "evaluation_combine_matrix.json", {"phase": "combine", "matrix": ev["combine_matrix"], "checksum": ev["combine_checksum"]})

    cal_demand = matrix_to_demand(cal["dispatch_matrix"], cal["combine_matrix"])
    eval_demand = matrix_to_demand(ev["dispatch_matrix"], ev["combine_matrix"])
    candidates = build_candidate_graphs(cal_demand, eval_demand)

    calibrated_allowed = ["son_torus", "calibration_greedy"] + [f"random_regular_seed_{seed}" for seed in range(RANDOM_CANDIDATES)]
    oracle_allowed = ["son_torus", "evaluation_greedy"] + [f"random_regular_seed_{seed}" for seed in range(RANDOM_CANDIDATES)]
    cal_selection = select_candidate(candidates, cal["dispatch_matrix"], cal["combine_matrix"], calibrated_allowed)
    oracle_selection = select_candidate(candidates, ev["dispatch_matrix"], ev["combine_matrix"], oracle_allowed)

    graphs = {
        "son_torus_ecmp4": candidates["son_torus"],
        "ron_calibrated_ecmp4": candidates[cal_selection["selected_name"]],
        "ron_oracle_ecmp4": candidates[oracle_selection["selected_name"]],
    }
    graph_paths: dict[str, Path] = {}
    config_paths: dict[str, Path] = {}
    for name, graph in graphs.items():
        graph_paths[name] = graphs_dir / f"{name}.json"
        write_json(graph_paths[name], graph)
        config_paths[name] = configs_dir / f"{name}.yml"
        network_config(config_paths[name], graph_paths[name])

    trace_meta = {
        "evaluation_dispatch": write_matrix_trace(traces_dir / "evaluation_dispatch" / "workload", ev["dispatch_matrix"]),
        "evaluation_combine": write_matrix_trace(traces_dir / "evaluation_combine" / "workload", ev["combine_matrix"]),
    }

    graph_audits = {name: graph_audit(graph) for name, graph in graphs.items()}
    graph_qualities = {name: graph_quality(graph) for name, graph in graphs.items()}
    graph_budget_fairness_pass = all(audit["same_degree_bandwidth_budget_as_son"] for audit in graph_audits.values())
    ron_graphs_valid = graph_audits["ron_calibrated_ecmp4"]["valid"] and graph_audits["ron_oracle_ecmp4"]["valid"]

    traffic_fingerprint = {
        "calibration_dispatch": matrix_stats(cal["dispatch_matrix"]),
        "calibration_combine": matrix_stats(cal["combine_matrix"]),
        "evaluation_dispatch": matrix_stats(ev["dispatch_matrix"]),
        "evaluation_combine": matrix_stats(ev["combine_matrix"]),
    }

    tiny_audit: dict[str, Any] = {}
    fluid: dict[str, Any] = {}
    native: dict[str, Any] = {}
    for phase, matrix, workload_prefix in (
        ("dispatch", ev["dispatch_matrix"], traces_dir / "evaluation_dispatch" / "workload"),
        ("combine", ev["combine_matrix"], traces_dir / "evaluation_combine" / "workload"),
    ):
        tiny_audit[phase] = {}
        fluid[phase] = {}
        native[phase] = {}
        for name, graph in graphs.items():
            tiny_audit[phase][name] = tiny_subchunk_audit(graph, matrix)
            fluid[phase][name] = link_load_estimate(graph, matrix, 4)
            native[phase][name] = run_astra(f"{phase}_{name}", workload_prefix, config_paths[name])

    totals: dict[str, Any] = {}
    for name in graphs:
        dispatch_run = native["dispatch"][name]
        combine_run = native["combine"][name]
        dispatch_fluid = fluid["dispatch"][name]
        combine_fluid = fluid["combine"][name]
        astra_total = (
            dispatch_run["max_cycles"] + combine_run["max_cycles"]
            if dispatch_run["max_cycles"] is not None and combine_run["max_cycles"] is not None
            else None
        )
        fluid_total = dispatch_fluid["fluid_cycles"] + combine_fluid["fluid_cycles"]
        totals[name] = {
            "dispatch_cycles": dispatch_run["max_cycles"],
            "combine_cycles": combine_run["max_cycles"],
            "total_cycles": astra_total,
            "dispatch_fluid_cycles": dispatch_fluid["fluid_cycles"],
            "combine_fluid_cycles": combine_fluid["fluid_cycles"],
            "total_fluid_cycles": fluid_total,
            "astra_over_fluid_total": (astra_total / fluid_total) if astra_total is not None and fluid_total else None,
            "success": dispatch_run["success"] and combine_run["success"],
            "runtime_s": dispatch_run["runtime_s"] + combine_run["runtime_s"],
        }

    son_total = totals["son_torus_ecmp4"]["total_cycles"]
    cal_total = totals["ron_calibrated_ecmp4"]["total_cycles"]
    oracle_total = totals["ron_oracle_ecmp4"]["total_cycles"]
    gain_loss = {
        "ron_calibrated_beats_son": cal_total is not None and son_total is not None and cal_total < son_total,
        "ron_oracle_beats_son": oracle_total is not None and son_total is not None and oracle_total < son_total,
        "total_cycles": {
            "son_torus_ecmp4": son_total,
            "ron_calibrated_ecmp4": cal_total,
            "ron_oracle_ecmp4": oracle_total,
        },
        "calibrated_gain_vs_son_percent": (100 * (son_total - cal_total) / son_total) if son_total and cal_total is not None else None,
        "oracle_gain_vs_son_percent": (100 * (son_total - oracle_total) / son_total) if son_total and oracle_total is not None else None,
        "calibrated_vs_oracle_gap_percent": (100 * (cal_total - oracle_total) / cal_total) if cal_total and oracle_total is not None else None,
        "max_link_load_change_vs_son": {
            name: {
                "dispatch_max_link_load_bytes": fluid["dispatch"][name]["max_link_load_bytes"],
                "combine_max_link_load_bytes": fluid["combine"][name]["max_link_load_bytes"],
                "dispatch_change_vs_son_percent": 100
                * (fluid["dispatch"]["son_torus_ecmp4"]["max_link_load_bytes"] - fluid["dispatch"][name]["max_link_load_bytes"])
                / fluid["dispatch"]["son_torus_ecmp4"]["max_link_load_bytes"],
                "combine_change_vs_son_percent": 100
                * (fluid["combine"]["son_torus_ecmp4"]["max_link_load_bytes"] - fluid["combine"][name]["max_link_load_bytes"])
                / fluid["combine"]["son_torus_ecmp4"]["max_link_load_bytes"],
            }
            for name in graphs
        },
        "hop_count_change_vs_son": {
            name: {
                "dispatch_byte_weighted_avg_hop": fluid["dispatch"][name]["byte_weighted_average_hop_count"],
                "combine_byte_weighted_avg_hop": fluid["combine"][name]["byte_weighted_average_hop_count"],
            }
            for name in graphs
        },
        "selected_candidates": {
            "ron_calibrated": cal_selection["selected_name"],
            "ron_oracle": oracle_selection["selected_name"],
        },
        "top_candidate_scores": {
            "calibrated": cal_selection["scores"][:8],
            "oracle": oracle_selection["scores"][:8],
        },
        "interpretation": "RON helps if it reduces max hot-link load and/or byte-weighted hop count without increasing degree/bandwidth. Negative results are valid and indicate weak skew or poor calibration transfer.",
    }

    all_runs_ok = all(item["success"] for item in totals.values())
    no_tiny_risk = not any(
        audit["zero_delay_risk"]
        for phase_payload in tiny_audit.values()
        for audit in phase_payload.values()
    )
    summary = {
        "scope": "static RON calibrated/oracle validation on one real aggregated workload",
        "workload": {
            "name": "Qwen MMLU machine_learning",
            "trace_dir": str(TRACE_DIR),
            "stage": "prefill",
            "trace_entry": "trace[0]",
            "npu_count": NPU_COUNT,
            "source_policy": "block_by_token",
            "block_size": BLOCK_SIZE,
            "expert_placement": "block",
            "hidden_size": HIDDEN_SIZE,
            "bytes_per_value": BYTES_PER_VALUE,
            "bytes_per_selection": BYTES_PER_SELECTION,
        },
        "split": {
            "calibration_request_count": cal["request_count"],
            "evaluation_request_count": ev["request_count"],
            "calibration_request_ids": cal["request_ids"],
            "evaluation_request_ids": ev["request_ids"],
            "calibration_fraction_requested": 0.10,
            "calibration_rule": "front ceil(10%) requests",
        },
        "anti_leakage": {
            "ron_calibrated_uses_only_calibration": True,
            "ron_calibrated_allowed_candidates": calibrated_allowed,
            "ron_oracle_reference_only": True,
            "ron_oracle_uses_evaluation_for_topology": True,
        },
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
        "v35_consistency": v35_consistency,
        "trace_meta": trace_meta,
        "traffic_fingerprint": traffic_fingerprint,
        "ron_construction": {
            "candidate_pool_size": len(candidates),
            "random_candidate_count": RANDOM_CANDIDATES,
            "selection_objective": "minimize max link load proxy under ECMP-4 using dispatch and combine matrices",
            "traffic_symmetrised": "no explicit symmetrisation; dispatch and combine directed matrices are both included in demand/score",
            "edges_bidirectional": True,
            "degree_enforced": DEGREE,
            "disconnected_graph_repair": "random regular generator requires connected; greedy starts from connected ring and fills to degree 4",
            "calibrated_selection": cal_selection,
            "oracle_selection": oracle_selection,
        },
        "graph_audit": graph_audits,
        "graph_quality": graph_qualities,
        "graph_budget_fairness_pass": graph_budget_fairness_pass,
        "ron_graphs_valid": ron_graphs_valid,
        "tiny_subchunk_audit": tiny_audit,
        "native_astra_results": native,
        "native_astra_totals": totals,
        "fluid_lower_bound": fluid,
        "gain_loss_explanation": gain_loss,
        "ron_runs_success": totals["ron_calibrated_ecmp4"]["success"] and totals["ron_oracle_ecmp4"]["success"],
        "final_recommendation": {
            "safe_to_proceed_to_four_workload_static": all_runs_ok and v35_consistency["matches_v35"] and graph_budget_fairness_pass and ron_graphs_valid and no_tiny_risk,
            "recommended_next_step": "four-workload static SON/RON calibrated/oracle validation" if all_runs_ok else "fix native ASTRA RON run failures first",
            "claim_supported": "static RON graph generation and native ASTRA execution work for one real aggregated workload under the same degree/bandwidth budget",
            "claim_not_supported": [
                "W=4 dynamic reconfiguration",
                "generalisation to all workloads",
                "real serving latency",
                "physical transparent OCS modelling",
                "token/layer-level execution timing",
            ],
        },
    }
    write_json(OUT / "summary.json", summary)
    write_report(summary)
    print(json.dumps(summary, indent=2))
    if not summary["final_recommendation"]["safe_to_proceed_to_four_workload_static"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
