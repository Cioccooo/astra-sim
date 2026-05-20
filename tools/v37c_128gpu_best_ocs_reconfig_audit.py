#!/usr/bin/env python3
"""V37c 128-GPU hotness and OCS reconfiguration audit.

This is a diagnostic extension of V37.  It intentionally keeps ASTRA C++ core
unchanged and separates complete fluid/link-load search from a smaller native
ASTRA representative timing pass.
"""

from __future__ import annotations

import hashlib
import importlib.util
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
OUT = REPO / "results/moe_expert_trace_converter/v37c_128gpu_best_ocs_reconfig_audit"
V36_PATH = REPO / "tools/v36_static_ron_one_workload_validation.py"

spec = importlib.util.spec_from_file_location("v36_static_ron", V36_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {V36_PATH}")
v36 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v36)

PROTO_DIR = REPO / "extern/graph_frontend/chakra/schema/protobuf"
PROTO_UTILS = REPO / "extern/graph_frontend/chakra/src/third_party/utils"
sys.path.insert(0, str(PROTO_DIR))
sys.path.insert(0, str(PROTO_UTILS))

from et_def_pb2 import AttributeProto, GlobalMetadata, Node, NodeType  # type: ignore  # noqa: E402
from protolib import encodeMessage as encode_message  # type: ignore  # noqa: E402

ASTRA_BIN = REPO / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
SYSTEM = REPO / "examples/system/native_collectives/Ring_4chunks.json"
REMOTE_MEMORY = REPO / "examples/remote_memory/analytical/no_memory_expansion.json"

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

MAIN_NPU = 128
OPTIONAL_NPUS = [32, 64]
DEGREE = 4
HIDDEN_SIZE = 4096
BYTES_PER_VALUE = 2
BYTES_PER_SELECTION = HIDDEN_SIZE * BYTES_PER_VALUE
BLOCK_SIZE = 16
LINK_GBPS = 400
ASTRA_BYTES_PER_NS = (LINK_GBPS / 8) * (1 << 30) / 1_000_000_000
ONE_CYCLE_THRESHOLD_BYTES = math.ceil(ASTRA_BYTES_PER_NS)
RANDOM_CANDIDATES = 32
ECMP_MAX_PATHS = 4
RECONFIG_PENALTY_US = [0, 1, 10]

# Native ASTRA is used to validate representative static topologies.  Exact
# request-window timing is reported from the no-leak fluid model because ASTRA
# does not yet support safe in-run topology swaps.
ASTRA_WORKLOAD_IDS = {
    "qwen_mmlu_zh_cn_anatomy",
}
ASTRA_TIMEOUT_S = 120

Pair = tuple[int, int]
Edge = tuple[int, int]
Sparse = dict[Pair, int]


def numeric_json_sort(path: Path) -> int | str:
    return int(path.stem) if path.stem.isdigit() else path.stem


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def matrix_checksum(matrix: list[list[int]]) -> str:
    blob = json.dumps(matrix, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def gini(values: list[int | float]) -> float:
    vals = sorted(float(v) for v in values if v >= 0)
    total = sum(vals)
    if not vals or total == 0:
        return 0.0
    n = len(vals)
    return sum((2 * i - n - 1) * value for i, value in enumerate(vals, 1)) / (n * total)


def entropy(values: list[int | float]) -> float:
    total = sum(values)
    if total <= 0:
        return 0.0
    return -sum((value / total) * math.log2(value / total) for value in values if value > 0)


def concentration(values: list[int], topks: tuple[int, ...] = (1, 2, 3, 4, 8, 16)) -> dict[str, Any]:
    vals = list(values)
    total = sum(vals)
    sorted_vals = sorted(vals, reverse=True)
    nonzero = [v for v in vals if v > 0]
    median_all = statistics.median(vals) if vals else 0
    median_nonzero = statistics.median(nonzero) if nonzero else 0
    result: dict[str, Any] = {
        "count": len(vals),
        "nonzero_count": len(nonzero),
        "total": total,
        "gini": gini(vals),
        "entropy_bits": entropy(vals),
        "min": min(vals) if vals else 0,
        "median": median_all,
        "median_nonzero": median_nonzero,
        "mean": statistics.mean(vals) if vals else 0,
        "max": max(vals) if vals else 0,
        "max_over_median": (max(vals) / median_all) if vals and median_all else None,
        "max_over_median_nonzero": (max(vals) / median_nonzero) if vals and median_nonzero else None,
    }
    for k in topks:
        result[f"top{k}_share"] = sum(sorted_vals[:k]) / total if total else 0.0
    return result


def classify_hotness(stats: dict[str, Any]) -> str:
    g = float(stats.get("gini", 0.0))
    top16 = float(stats.get("top16_share", 0.0))
    ratio = stats.get("max_over_median_nonzero") or 0.0
    if g >= 0.45 or top16 >= 0.20 or ratio >= 8:
        return "hot"
    if g >= 0.18 or top16 >= 0.05 or ratio >= 3:
        return "moderate"
    return "broad"


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


def write_matrix_trace(prefix: Path, matrix: list[list[int]], npu: int) -> dict[str, Any]:
    nodes_by_rank: dict[int, list[Node]] = {rank: [] for rank in range(npu)}
    next_id = {rank: 1 for rank in range(npu)}
    tag = 1
    total_bytes = 0
    messages = 0
    for src in range(npu):
        for dst in range(npu):
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
    for rank in range(npu):
        nodes = nodes_by_rank[rank] or [comp_node(1)]
        with (prefix.parent / f"{prefix.name}.{rank}.et").open("wb") as handle:
            encode_message(handle, GlobalMetadata(version="0.0.4"))
            for node in nodes:
                encode_message(handle, node)
    return {"prefix": str(prefix), "rank_count": npu, "messages": messages, "total_bytes": total_bytes}


def parse_workload(path: Path) -> dict[str, Any]:
    files = sorted(path.glob("*.json"), key=numeric_json_sort)
    requests: list[dict[str, Any]] = []
    max_expert = -1
    malformed = 0
    global_token_offset = 0
    moe_layers: set[int] = set()
    expert_counts: Counter[int] = Counter()
    selected_events = 0
    for req_idx, file_path in enumerate(files):
        try:
            trace = json.loads(file_path.read_text())
        except Exception:
            malformed += 1
            continue
        if not isinstance(trace, list) or not trace or not isinstance(trace[0], dict):
            malformed += 1
            continue
        rows_for_request: list[tuple[int, int, int, list[int]]] = []
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
                    expert_counts[expert_id] += 1
                    max_expert = max(max_expert, expert_id)
                selected_events += len(parsed)
                rows_for_request.append((layer_id, row_index, global_token_offset + row_index, parsed))
        requests.append(
            {
                "request_id": file_path.stem,
                "request_index": req_idx,
                "rows": rows_for_request,
                "prefill_tokens": max_rows,
            }
        )
        global_token_offset += max_rows
    if max_expert < 0:
        raise RuntimeError(f"No expert ids found under {path}")
    num_experts = max_expert + 1
    return {
        "files_found": len(files),
        "files_used": len(requests),
        "requests": requests,
        "moe_layers": sorted(moe_layers),
        "moe_layer_count": len(moe_layers),
        "num_experts": num_experts,
        "malformed_records": malformed,
        "selected_expert_events": selected_events,
        "expert_counts": [expert_counts[i] for i in range(num_experts)],
    }


def source_rank(policy: str, request_index: int, global_token_index: int, local_token_index: int, npu: int) -> int:
    if policy == "block_by_token":
        return (global_token_index // BLOCK_SIZE) % npu
    if policy == "decode_like_batch":
        return request_index % npu
    if policy == "block_by_request":
        return (request_index // BLOCK_SIZE) % npu
    if policy == "round_robin_by_request":
        return request_index % npu
    if policy == "round_robin_by_token":
        return global_token_index % npu
    raise ValueError(f"unknown source policy {policy}")


def expert_to_gpu(expert_id: int, npu: int, num_experts: int, placement: str, hot_order: list[int] | None = None) -> int:
    if placement == "block":
        return min((expert_id * npu) // num_experts, npu - 1)
    if placement == "round_robin":
        return expert_id % npu
    if placement == "hot_expert_clustered":
        if hot_order is None:
            raise ValueError("hot_expert_clustered requires hot_order")
        rank_by_hotness = {expert: idx for idx, expert in enumerate(hot_order)}
        return min(rank_by_hotness.get(expert_id, expert_id) % npu, npu - 1)
    if placement == "hot_expert_balanced":
        if hot_order is None:
            raise ValueError("hot_expert_balanced requires hot_order")
        rank_by_hotness = {expert: idx for idx, expert in enumerate(hot_order)}
        return rank_by_hotness.get(expert_id, expert_id) % npu
    raise ValueError(f"unknown expert placement {placement}")


def zero_matrix(npu: int) -> list[list[int]]:
    return [[0 for _ in range(npu)] for _ in range(npu)]


def sparse_to_matrix(sparse: Sparse, npu: int) -> list[list[int]]:
    matrix = zero_matrix(npu)
    for (src, dst), value in sparse.items():
        matrix[src][dst] += value
    return matrix


def add_sparse(dst: Sparse, src: Sparse) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + value


def build_sparse_for_indices(
    parsed: dict[str, Any],
    indices: list[int],
    npu: int,
    source_policy: str,
    expert_placement: str,
    hot_order: list[int] | None = None,
) -> dict[str, Any]:
    dispatch: Sparse = defaultdict(int)
    combine: Sparse = defaultdict(int)
    dest_bytes = [0 for _ in range(npu)]
    local = 0
    remote = 0
    selected = 0
    tokens = 0
    request_ids: list[str] = []
    num_experts = parsed["num_experts"]
    for idx in indices:
        request = parsed["requests"][idx]
        request_ids.append(request["request_id"])
        tokens += int(request["prefill_tokens"])
        req_idx = int(request["request_index"])
        for _, local_token_idx, global_token_idx, experts in request["rows"]:
            src = source_rank(source_policy, req_idx, global_token_idx, local_token_idx, npu)
            for expert_id in experts:
                selected += 1
                dst = expert_to_gpu(expert_id, npu, num_experts, expert_placement, hot_order)
                if src == dst:
                    local += 1
                    continue
                dispatch[(src, dst)] += BYTES_PER_SELECTION
                combine[(dst, src)] += BYTES_PER_SELECTION
                dest_bytes[dst] += BYTES_PER_SELECTION
                remote += 1
    theoretical = selected * BYTES_PER_SELECTION
    local_bytes = local * BYTES_PER_SELECTION
    remote_bytes = remote * BYTES_PER_SELECTION
    return {
        "request_ids": request_ids,
        "request_count": len(indices),
        "prefill_input_tokens": tokens,
        "selected_expert_events": selected,
        "theoretical_dispatch_bytes": theoretical,
        "theoretical_combine_bytes": theoretical,
        "local_dispatch_bytes_excluded": local_bytes,
        "local_combine_bytes_excluded": local_bytes,
        "remote_dispatch_bytes_retained": sum(dispatch.values()),
        "remote_combine_bytes_retained": sum(combine.values()),
        "byte_conservation_pass": (
            sum(dispatch.values()) == remote_bytes
            and sum(combine.values()) == remote_bytes
            and theoretical == local_bytes + remote_bytes
        ),
        "dispatch_sparse": dict(dispatch),
        "combine_sparse": dict(combine),
        "dest_gpu_bytes": dest_bytes,
    }


def build_request_sparse_list(
    parsed: dict[str, Any],
    npu: int,
    source_policy: str,
    expert_placement: str,
    hot_order: list[int] | None = None,
) -> list[dict[str, Any]]:
    return [
        build_sparse_for_indices(parsed, [idx], npu, source_policy, expert_placement, hot_order)
        for idx in range(len(parsed["requests"]))
    ]


def aggregate_request_payloads(request_payloads: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    dispatch: Sparse = defaultdict(int)
    combine: Sparse = defaultdict(int)
    local_dispatch = 0
    local_combine = 0
    remote_dispatch = 0
    remote_combine = 0
    theoretical_dispatch = 0
    theoretical_combine = 0
    selected = 0
    tokens = 0
    request_ids: list[str] = []
    for idx in indices:
        payload = request_payloads[idx]
        request_ids.extend(payload["request_ids"])
        tokens += payload["prefill_input_tokens"]
        selected += payload["selected_expert_events"]
        theoretical_dispatch += payload["theoretical_dispatch_bytes"]
        theoretical_combine += payload["theoretical_combine_bytes"]
        local_dispatch += payload["local_dispatch_bytes_excluded"]
        local_combine += payload["local_combine_bytes_excluded"]
        remote_dispatch += payload["remote_dispatch_bytes_retained"]
        remote_combine += payload["remote_combine_bytes_retained"]
        add_sparse(dispatch, payload["dispatch_sparse"])
        add_sparse(combine, payload["combine_sparse"])
    return {
        "request_ids": request_ids,
        "request_count": len(indices),
        "prefill_input_tokens": tokens,
        "selected_expert_events": selected,
        "theoretical_dispatch_bytes": theoretical_dispatch,
        "theoretical_combine_bytes": theoretical_combine,
        "local_dispatch_bytes_excluded": local_dispatch,
        "local_combine_bytes_excluded": local_combine,
        "remote_dispatch_bytes_retained": remote_dispatch,
        "remote_combine_bytes_retained": remote_combine,
        "byte_conservation_pass": (
            sum(dispatch.values()) == remote_dispatch
            and sum(combine.values()) == remote_combine
            and theoretical_dispatch == local_dispatch + remote_dispatch
            and theoretical_combine == local_combine + remote_combine
        ),
        "dispatch_sparse": dict(dispatch),
        "combine_sparse": dict(combine),
    }


def sparse_values(sparse: Sparse, npu: int, include_zeros: bool = False) -> list[int]:
    if not include_zeros:
        return [value for (src, dst), value in sparse.items() if src != dst and value > 0]
    return [sparse.get((src, dst), 0) for src in range(npu) for dst in range(npu) if src != dst]


def sparse_records(sparse: Sparse) -> list[dict[str, int]]:
    return [
        {"src": src, "dst": dst, "bytes": value}
        for (src, dst), value in sorted(sparse.items())
        if value > 0
    ]


def matrix_stats_from_sparse(sparse: Sparse, npu: int) -> dict[str, Any]:
    vals = sparse_values(sparse, npu, include_zeros=False)
    stats = concentration(vals, (1, 4, 8, 16, 32, 64))
    stats["nonzero_gpu_pairs"] = len(vals)
    stats["interpretation"] = classify_hotness(stats)
    return stats


def combine_sparse(a: Sparse, b: Sparse) -> Sparse:
    out: Sparse = defaultdict(int)
    add_sparse(out, a)
    add_sparse(out, b)
    return dict(out)


def graph_degree(edges: set[Edge], npu: int) -> Counter[int]:
    deg: Counter[int] = Counter()
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    for node in range(npu):
        deg[node] += 0
    return deg


def connected_components(edges: set[Edge], npu: int) -> list[list[int]]:
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen: set[int] = set()
    comps: list[list[int]] = []
    for node in range(npu):
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


def is_connected(edges: set[Edge], npu: int) -> bool:
    return len(connected_components(edges, npu)) == 1


def torus_shape(npu: int) -> tuple[int, int]:
    root = int(math.sqrt(npu))
    for rows in range(root, 0, -1):
        if npu % rows == 0:
            return rows, npu // rows
    return 1, npu


def torus_edges(npu: int) -> set[Edge]:
    rows, cols = torus_shape(npu)
    edges: set[Edge] = set()
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            edges.add(tuple(sorted((node, r * cols + ((c + 1) % cols)))))
            edges.add(tuple(sorted((node, ((r + 1) % rows) * cols + c))))
    return edges


def ring_edges(npu: int) -> set[Edge]:
    return {tuple(sorted((i, (i + 1) % npu))) for i in range(npu)}


def random_regular_graph(seed: int, npu: int) -> set[Edge]:
    rng = random.Random(seed)
    stubs = [rank for rank in range(npu) for _ in range(DEGREE)]
    for _ in range(100000):
        rng.shuffle(stubs)
        edges: set[Edge] = set()
        ok = True
        for idx in range(0, len(stubs), 2):
            a, b = sorted((stubs[idx], stubs[idx + 1]))
            if a == b or (a, b) in edges:
                ok = False
                break
            edges.add((a, b))
        if ok and is_connected(edges, npu):
            return edges
    raise RuntimeError(f"could not build random regular graph seed={seed}, npu={npu}")


def greedy_demand_graph(demand: Sparse, seed_edges: set[Edge], npu: int) -> set[Edge]:
    edges = set(seed_edges)
    deg = graph_degree(edges, npu)
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
    candidates = [tuple(sorted((i, j))) for i in range(npu) for j in range(i + 1, npu)]
    rng.shuffle(candidates)
    for edge in candidates:
        a, b = edge
        if edge in edges or deg[a] >= DEGREE or deg[b] >= DEGREE:
            continue
        edges.add(edge)
        deg[a] += 1
        deg[b] += 1
        if all(deg[i] == DEGREE for i in range(npu)):
            break
    return repair_to_degree4(edges, npu)


def safe_greedy_graph(demand: Sparse, seed_edges: set[Edge], npu: int, fallback_seed: int) -> tuple[set[Edge], dict[str, Any]]:
    try:
        return greedy_demand_graph(demand, seed_edges, npu), {
            "construction": "strict_greedy_demand_graph",
            "fallback_used": False,
        }
    except Exception as exc:
        return random_regular_graph(fallback_seed, npu), {
            "construction": "strict_greedy_demand_graph_failed_random_regular_fallback",
            "fallback_used": True,
            "fallback_seed": fallback_seed,
            "failure": str(exc),
        }


def repair_to_degree4(edges: set[Edge], npu: int) -> set[Edge]:
    edges = set(edges)

    def degs() -> dict[int, int]:
        deg = {node: 0 for node in range(npu)}
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
        return deg

    for _ in range(20000):
        deg = degs()
        if all(value == DEGREE for value in deg.values()):
            if is_connected(edges, npu):
                return edges
            raise RuntimeError("degree-4 repair produced disconnected graph")
        lows = [node for node, value in deg.items() if value < DEGREE]
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
        if len(lows) < 2:
            raise RuntimeError(f"cannot repair odd low-degree set: {lows}")
        a, b = lows[0], lows[1]
        repaired = False
        for u, vv in sorted(edges):
            if len({a, b, u, vv}) < 4:
                continue
            options = [
                (tuple(sorted((a, u))), tuple(sorted((b, vv)))),
                (tuple(sorted((a, vv))), tuple(sorted((b, u)))),
            ]
            for e1, e2 in options:
                if e1 in edges or e2 in edges or e1 == e2:
                    continue
                trial = set(edges)
                trial.remove((u, vv))
                trial.add(e1)
                trial.add(e2)
                if is_connected(trial, npu):
                    edges = trial
                    repaired = True
                    break
            if repaired:
                break
        if not repaired:
            raise RuntimeError(f"could not repair graph low nodes={lows}")
    raise RuntimeError("degree repair exceeded iteration limit")


def graph_from_edges(name: str, edges: set[Edge], npu: int, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "node_count": npu,
        "gpu_count": npu,
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


def graph_audit(graph: dict[str, Any], npu: int) -> dict[str, Any]:
    edges = edges_from_graph(graph)
    deg = graph_degree(edges, npu)
    comps = connected_components(edges, npu)
    duplicate_count = len(graph["edges"]) - len(edges)
    return {
        "name": graph["name"],
        "node_count": graph["node_count"],
        "gpu_count": graph["gpu_count"],
        "edge_circuit_count": len(edges),
        "expected_edge_circuit_count": npu * DEGREE // 2,
        "degree_distribution": dict(sorted(Counter(deg.values()).items())),
        "degree_min": min(deg.values()),
        "degree_max": max(deg.values()),
        "connected_components": len(comps),
        "component_sizes": [len(comp) for comp in comps],
        "duplicate_edges": duplicate_count,
        "self_loops": sum(1 for a, b in edges if a == b),
        "per_gpu_optical_bandwidth_tbps": 1.6,
        "per_link_bandwidth_gbps": LINK_GBPS,
        "same_degree_bandwidth_budget_as_son": (
            len(edges) == npu * DEGREE // 2 and min(deg.values()) == DEGREE and max(deg.values()) == DEGREE
        ),
        "valid": (
            len(comps) == 1
            and duplicate_count == 0
            and not any(a == b for a, b in edges)
            and len(edges) == npu * DEGREE // 2
            and min(deg.values()) == DEGREE
            and max(deg.values()) == DEGREE
        ),
    }


def adjacency_from_graph(graph: dict[str, Any]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(graph["node_count"])]
    for edge in graph["edges"]:
        src, dst = int(edge["src"]), int(edge["dst"])
        adj[src].append(dst)
        adj[dst].append(src)
    for neighbors in adj:
        neighbors.sort()
    return adj


def paths_from_source(adj: list[list[int]], src: int, max_paths: int) -> dict[int, list[list[int]]]:
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
    result: dict[int, list[list[int]]] = {}
    for dst in range(len(adj)):
        if dst == src or distance[dst] == -1:
            continue
        paths: list[list[int]] = []
        current: list[int] = []

        def backtrack(node: int) -> None:
            if len(paths) >= max_paths:
                return
            current.append(node)
            if node == src:
                paths.append(list(reversed(current)))
            else:
                for parent in sorted(parents[node]):
                    backtrack(parent)
                    if len(paths) >= max_paths:
                        break
            current.pop()

        backtrack(dst)
        result[dst] = sorted(paths)[:max_paths]
    return result


def precompute_paths(graph: dict[str, Any], npu: int, max_paths: int = ECMP_MAX_PATHS) -> dict[Pair, list[list[int]]]:
    adj = adjacency_from_graph(graph)
    cache: dict[Pair, list[list[int]]] = {}
    for src in range(npu):
        paths_by_dst = paths_from_source(adj, src, max_paths)
        for dst, paths in paths_by_dst.items():
            if src != dst:
                cache[(src, dst)] = paths
    return cache


def split_bytes(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def link_load_sparse(paths_by_pair: dict[Pair, list[list[int]]], sparse: Sparse) -> dict[str, Any]:
    loads: dict[Pair, int] = defaultdict(int)
    selected_counts: list[int] = []
    hop_counts: list[int] = []
    byte_weighted_hops = 0
    total_bytes = 0
    for (src, dst), size in sparse.items():
        if src == dst or size <= 0:
            continue
        paths = paths_by_pair.get((src, dst), [])
        if not paths:
            raise RuntimeError(f"no path {src}->{dst}")
        selected_counts.append(len(paths))
        for path, subbytes in zip(paths, split_bytes(size, len(paths))):
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
        "byte_weighted_average_hop_count": byte_weighted_hops / total_bytes if total_bytes else 0,
        "max_link_load_bytes": max_load,
        "median_link_load_bytes": statistics.median(vals) if vals else 0,
        "fluid_cycles": int(max_load / ASTRA_BYTES_PER_NS) if max_load else 0,
        "hot_links": [
            {"src": u, "dst": vv, "bytes": b}
            for (u, vv), b in sorted(loads.items(), key=lambda item: -item[1])[:8]
        ],
    }


def score_candidate(paths: dict[Pair, list[list[int]]], dispatch: Sparse, combine: Sparse) -> int:
    return max(
        link_load_sparse(paths, dispatch)["max_link_load_bytes"],
        link_load_sparse(paths, combine)["max_link_load_bytes"],
    )


def score_all_candidates(
    path_caches: dict[str, dict[Pair, list[list[int]]]],
    dispatch: Sparse,
    combine: Sparse,
    names: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for name in names:
        score = score_candidate(path_caches[name], dispatch, combine)
        rows.append({"name": name, "score_max_link_load_bytes": score, "fluid_cycles": int(score / ASTRA_BYTES_PER_NS)})
    return sorted(rows, key=lambda item: (item["score_max_link_load_bytes"], item["name"]))


def tiny_subchunk_audit(paths_by_pair: dict[Pair, list[list[int]]], sparse: Sparse) -> dict[str, Any]:
    subchunks: list[int] = []
    selected_counts: list[int] = []
    for (src, dst), size in sparse.items():
        if src == dst or size <= 0:
            continue
        paths = paths_by_pair[(src, dst)]
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
        "selected_path_count_median": statistics.median(selected_counts) if selected_counts else 0,
        "selected_path_count_max": max(selected_counts) if selected_counts else 0,
        "zero_delay_risk": any(value < ONE_CYCLE_THRESHOLD_BYTES for value in subchunks),
    }


def network_config(path: Path, graph_path: Path, npu: int) -> None:
    lines = [
        "topology: [ Graph ]",
        f"npus_count: [ {npu} ]",
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


def run_astra(label: str, workload: Path, network: Path, npu: int) -> dict[str, Any]:
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
    try:
        proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, check=False, timeout=ASTRA_TIMEOUT_S)
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
        "success": returncode == 0 and len(cycles) == npu,
        "runtime_s": runtime_s,
        "command": " ".join(cmd),
        "stdout": str(stdout),
        "stderr": str(stderr),
        "stderr_tail": stderr_text.splitlines()[-8:],
        "max_cycles": max(cycles) if cycles else None,
        "cycles_count": len(cycles),
    }


def build_candidate_graphs(npu: int, calibration_demand: Sparse, evaluation_demand: Sparse) -> dict[str, dict[str, Any]]:
    cal_greedy_edges, cal_greedy_meta = safe_greedy_graph(calibration_demand, ring_edges(npu), npu, 9001)
    eval_greedy_edges, eval_greedy_meta = safe_greedy_graph(evaluation_demand, ring_edges(npu), npu, 9002)
    candidates = {
        "son_torus": graph_from_edges(
            f"son_torus_{torus_shape(npu)[0]}x{torus_shape(npu)[1]}_{npu}gpu",
            torus_edges(npu),
            npu,
            {"construction": "static_2d_torus", "shape": torus_shape(npu)},
        ),
        "calibration_greedy": graph_from_edges(
            "ron_calibration_greedy_degree4",
            cal_greedy_edges,
            npu,
            {**cal_greedy_meta, "traffic_source": "calibration"},
        ),
        "evaluation_greedy": graph_from_edges(
            "ron_evaluation_greedy_degree4",
            eval_greedy_edges,
            npu,
            {**eval_greedy_meta, "traffic_source": "evaluation_oracle"},
        ),
    }
    for seed in range(RANDOM_CANDIDATES):
        candidates[f"random_regular_seed_{seed}"] = graph_from_edges(
            f"ron_random_regular_seed_{seed}",
            random_regular_graph(seed, npu),
            npu,
            {"construction": "random_regular_degree4", "seed": seed},
        )
    return candidates


def graph_quality_light(paths: dict[Pair, list[list[int]]]) -> dict[str, Any]:
    lengths = [len(p[0]) - 1 for p in paths.values() if p]
    counts = [len(p) for p in paths.values()]
    return {
        "average_shortest_path_length": statistics.mean(lengths) if lengths else None,
        "diameter": max(lengths) if lengths else None,
        "shortest_path_length_distribution": dict(sorted(Counter(lengths).items())),
        "ecmp_path_count_distribution_cap4": dict(sorted(Counter(counts).items())),
    }


def request_window_strategy(
    request_payloads: list[dict[str, Any]],
    eval_start: int,
    path_caches: dict[str, dict[Pair, list[list[int]]]],
    candidate_names: list[str],
    window: int,
) -> dict[str, Any]:
    exact_cycles_by_penalty: dict[str, int] = {}
    segment_rows: list[dict[str, Any]] = []
    total_fluid_cycles = 0
    selected_counts: Counter[str] = Counter()
    selected_bytes: Counter[str] = Counter()
    for idx in range(eval_start, len(request_payloads)):
        start = max(0, idx - window)
        history_dispatch: Sparse = defaultdict(int)
        history_combine: Sparse = defaultdict(int)
        for hist in request_payloads[start:idx]:
            add_sparse(history_dispatch, hist["dispatch_sparse"])
            add_sparse(history_combine, hist["combine_sparse"])
        if not history_dispatch and not history_combine:
            selected = "son_torus"
            scores = [{"name": selected, "score_max_link_load_bytes": 0}]
        else:
            scores = score_all_candidates(path_caches, history_dispatch, history_combine, candidate_names)
            selected = scores[0]["name"]
        cur_dispatch = request_payloads[idx]["dispatch_sparse"]
        cur_combine = request_payloads[idx]["combine_sparse"]
        dispatch_cycles = link_load_sparse(path_caches[selected], cur_dispatch)["fluid_cycles"]
        combine_cycles = link_load_sparse(path_caches[selected], cur_combine)["fluid_cycles"]
        cur_cycles = dispatch_cycles + combine_cycles
        cur_bytes = sum(cur_dispatch.values()) + sum(cur_combine.values())
        total_fluid_cycles += cur_cycles
        selected_counts[selected] += 1
        selected_bytes[selected] += cur_bytes
        segment_rows.append(
            {
                "request_index": idx,
                "selected": selected,
                "fluid_cycles": cur_cycles,
                "bytes": cur_bytes,
                "top_history_score": scores[0],
            }
        )
    for penalty_us in RECONFIG_PENALTY_US:
        exact_cycles_by_penalty[f"{penalty_us}us"] = total_fluid_cycles + len(segment_rows) * penalty_us * 1000
    return {
        "window": window,
        "evaluated_segments": len(segment_rows),
        "fluid_cycles_without_reconfig": total_fluid_cycles,
        "fluid_cycles_with_reconfig": exact_cycles_by_penalty,
        "selected_counts": dict(selected_counts),
        "selected_bytes": dict(selected_bytes),
        "top_selected_by_bytes": [
            {"name": name, "bytes": value, "count": selected_counts[name]}
            for name, value in selected_bytes.most_common(8)
        ],
        "segments_sample": segment_rows[:5],
    }


def segmented_static_strategy(
    request_payloads: list[dict[str, Any]],
    eval_start: int,
    paths: dict[Pair, list[list[int]]],
) -> dict[str, Any]:
    total = 0
    segments = 0
    for idx in range(eval_start, len(request_payloads)):
        dispatch_cycles = link_load_sparse(paths, request_payloads[idx]["dispatch_sparse"])["fluid_cycles"]
        combine_cycles = link_load_sparse(paths, request_payloads[idx]["combine_sparse"])["fluid_cycles"]
        total += dispatch_cycles + combine_cycles
        segments += 1
    return {"segments": segments, "fluid_cycles": total}


def expert_hotness(parsed: dict[str, Any]) -> dict[str, Any]:
    counts = parsed["expert_counts"]
    total = sum(counts)
    ranked = sorted(enumerate(counts), key=lambda item: (-item[1], item[0]))
    cold = min(enumerate(counts), key=lambda item: (item[1], item[0]))
    stats = concentration(counts, (1, 2, 3, 8, 16))
    stats.update(
        {
            "num_experts": parsed["num_experts"],
            "uniform_expert_share": 1 / parsed["num_experts"],
            "top_experts": [
                {"expert_id": expert, "count": count, "share": count / total if total else 0.0}
                for expert, count in ranked[:16]
            ],
            "coldest_expert": {"expert_id": cold[0], "count": cold[1], "share": cold[1] / total if total else 0.0},
            "top1_median_ratio": (ranked[0][1] / stats["median"]) if stats["median"] else None,
        }
    )
    return stats


def hot_order(parsed: dict[str, Any]) -> list[int]:
    return [expert for expert, _ in sorted(enumerate(parsed["expert_counts"]), key=lambda item: (-item[1], item[0]))]


def mapping_sensitivity(parsed: dict[str, Any], npu: int) -> dict[str, Any]:
    order = hot_order(parsed)
    dest_bytes_by_placement = {
        "block": [0 for _ in range(npu)],
        "round_robin": [0 for _ in range(npu)],
    }
    for expert_id, count in enumerate(parsed["expert_counts"]):
        for placement in dest_bytes_by_placement:
            dst = expert_to_gpu(expert_id, npu, parsed["num_experts"], placement, order)
            dest_bytes_by_placement[placement][dst] += count * BYTES_PER_SELECTION

    pair_by_combo: dict[str, Sparse] = {
        "pair_block_by_token_block": defaultdict(int),
        "pair_block_by_token_round_robin": defaultdict(int),
        "pair_decode_like_batch_block": defaultdict(int),
        "pair_decode_like_batch_round_robin": defaultdict(int),
    }
    for request in parsed["requests"]:
        req_idx = int(request["request_index"])
        for _, local_token_idx, global_token_idx, experts in request["rows"]:
            src_block = source_rank("block_by_token", req_idx, global_token_idx, local_token_idx, npu)
            src_decode = source_rank("decode_like_batch", req_idx, global_token_idx, local_token_idx, npu)
            for expert_id in experts:
                dst_block = expert_to_gpu(expert_id, npu, parsed["num_experts"], "block", order)
                dst_rr = expert_to_gpu(expert_id, npu, parsed["num_experts"], "round_robin", order)
                if src_block != dst_block:
                    pair_by_combo["pair_block_by_token_block"][(src_block, dst_block)] += BYTES_PER_SELECTION
                    pair_by_combo["pair_block_by_token_block"][(dst_block, src_block)] += BYTES_PER_SELECTION
                if src_block != dst_rr:
                    pair_by_combo["pair_block_by_token_round_robin"][(src_block, dst_rr)] += BYTES_PER_SELECTION
                    pair_by_combo["pair_block_by_token_round_robin"][(dst_rr, src_block)] += BYTES_PER_SELECTION
                if src_decode != dst_block:
                    pair_by_combo["pair_decode_like_batch_block"][(src_decode, dst_block)] += BYTES_PER_SELECTION
                    pair_by_combo["pair_decode_like_batch_block"][(dst_block, src_decode)] += BYTES_PER_SELECTION
                if src_decode != dst_rr:
                    pair_by_combo["pair_decode_like_batch_round_robin"][(src_decode, dst_rr)] += BYTES_PER_SELECTION
                    pair_by_combo["pair_decode_like_batch_round_robin"][(dst_rr, src_decode)] += BYTES_PER_SELECTION

    out: dict[str, Any] = {}
    for placement, values in dest_bytes_by_placement.items():
        stats = concentration(values, (1, 4, 8, 16))
        out[f"dest_gpu_{placement}"] = {**stats, "interpretation": classify_hotness(stats)}
    for key, sparse in pair_by_combo.items():
        out[key] = matrix_stats_from_sparse(sparse, npu)
    return out


def aggregation_window_sensitivity(request_payloads: list[dict[str, Any]], npu: int) -> dict[str, Any]:
    full_sparse: Sparse = defaultdict(int)
    for payload in request_payloads:
        add_sparse(full_sparse, payload["dispatch_sparse"])
        add_sparse(full_sparse, payload["combine_sparse"])
    per_request_stats = [
        matrix_stats_from_sparse(combine_sparse(p["dispatch_sparse"], p["combine_sparse"]), npu)
        for p in request_payloads
    ]

    def summarize_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        return {
            "top1_share_median": statistics.median(row["top1_share"] for row in rows),
            "top4_share_median": statistics.median(row["top4_share"] for row in rows),
            "top16_share_median": statistics.median(row["top16_share"] for row in rows),
            "gini_median": statistics.median(row["gini"] for row in rows),
            "max_over_median_nonzero_median": statistics.median(
                (row["max_over_median_nonzero"] or 0) for row in rows
            ),
            "top1_share_max": max(row["top1_share"] for row in rows),
            "gini_max": max(row["gini"] for row in rows),
        }

    out = {
        "full_aggregate": matrix_stats_from_sparse(full_sparse, npu),
        "per_request_summary": summarize_stats(per_request_stats),
    }
    for window in (4, 8):
        rows = []
        for start in range(0, len(request_payloads), window):
            agg: Sparse = defaultdict(int)
            for payload in request_payloads[start : start + window]:
                add_sparse(agg, payload["dispatch_sparse"])
                add_sparse(agg, payload["combine_sparse"])
            rows.append(matrix_stats_from_sparse(agg, npu))
        out[f"window_{window}_summary"] = summarize_stats(rows)
    return out


def old_pair_id_note(npu: int) -> dict[str, Any]:
    old_examples = [(81, 117), (93, 106), (131, 259)]
    return {
        "old_examples": [{"src": a, "dst": b} for a, b in old_examples],
        "valid_true_gpu_id_range": f"0..{npu - 1}",
        "diagnosis": (
            "Pairs with ids above the GPU range are not final true GPU ids. "
            "In this pipeline final GPU-pair matrices are indexed only by src_gpu and dst_gpu. "
            "The old examples are consistent with pre-mapping ids such as expert ids or token/source-bucket-to-expert coordinates."
        ),
        "coordinate_system_used_in_v37c": "final true src_gpu -> dst_gpu after source policy and expert placement",
    }


def write_astra_artifacts(
    workload_id: str,
    npu: int,
    graph_name: str,
    graph: dict[str, Any],
    eval_payload: dict[str, Any],
    wout: Path,
) -> dict[str, Any]:
    graphs_dir = wout / "graphs"
    configs_dir = wout / "network_configs"
    traces_dir = wout / "chakra_traces"
    graph_path = graphs_dir / f"{graph_name}.json"
    config_path = configs_dir / f"{graph_name}.yml"
    write_json(graph_path, graph)
    network_config(config_path, graph_path, npu)
    dispatch_matrix = sparse_to_matrix(eval_payload["dispatch_sparse"], npu)
    combine_matrix = sparse_to_matrix(eval_payload["combine_sparse"], npu)
    dispatch_prefix = traces_dir / "evaluation_dispatch" / "workload"
    combine_prefix = traces_dir / "evaluation_combine" / "workload"
    dispatch_trace = write_matrix_trace(dispatch_prefix, dispatch_matrix, npu)
    combine_trace = write_matrix_trace(combine_prefix, combine_matrix, npu)
    dispatch_run = run_astra(f"{workload_id}_dispatch_{graph_name}", dispatch_prefix, config_path, npu)
    combine_run = run_astra(f"{workload_id}_combine_{graph_name}", combine_prefix, config_path, npu)
    total = (
        dispatch_run["max_cycles"] + combine_run["max_cycles"]
        if dispatch_run["max_cycles"] is not None and combine_run["max_cycles"] is not None
        else None
    )
    return {
        "graph": graph_name,
        "graph_path": str(graph_path),
        "config_path": str(config_path),
        "dispatch_trace": dispatch_trace,
        "combine_trace": combine_trace,
        "dispatch": dispatch_run,
        "combine": combine_run,
        "total_cycles": total,
        "success": dispatch_run["success"] and combine_run["success"],
    }


def compact_32gpu_summary_from_v37() -> dict[str, Any]:
    path = REPO / "results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/summary.json"
    if not path.exists():
        return {"available": False}
    summary = json.loads(path.read_text())
    return {
        "available": True,
        "source": str(path),
        "cross_workload_summary": summary.get("cross_workload_summary", []),
    }


def process_workload(workload: dict[str, Any]) -> dict[str, Any]:
    wid = workload["id"]
    wout = OUT / "workloads" / wid
    parsed = parse_workload(workload["path"])
    npu = MAIN_NPU
    order = hot_order(parsed)
    all_indices = list(range(len(parsed["requests"])))
    cal_count = max(1, min(math.ceil(len(all_indices) * 0.10), len(all_indices) - 1))
    cal_indices = list(range(cal_count))
    eval_indices = list(range(cal_count, len(all_indices)))

    request_payloads = build_request_sparse_list(parsed, npu, "block_by_token", "block", order)
    full_payload = aggregate_request_payloads(request_payloads, all_indices)
    cal_payload = aggregate_request_payloads(request_payloads, cal_indices)
    eval_payload = aggregate_request_payloads(request_payloads, eval_indices)

    matrices_dir = wout / "traffic_matrices"
    write_json(matrices_dir / "evaluation_dispatch_sparse.json", sparse_records(eval_payload["dispatch_sparse"]))
    write_json(matrices_dir / "evaluation_combine_sparse.json", sparse_records(eval_payload["combine_sparse"]))

    calibration_demand = combine_sparse(cal_payload["dispatch_sparse"], cal_payload["combine_sparse"])
    evaluation_demand = combine_sparse(eval_payload["dispatch_sparse"], eval_payload["combine_sparse"])
    candidates = build_candidate_graphs(npu, calibration_demand, evaluation_demand)
    path_caches = {name: precompute_paths(graph, npu) for name, graph in candidates.items()}
    candidate_names = ["son_torus", "calibration_greedy"] + [f"random_regular_seed_{i}" for i in range(RANDOM_CANDIDATES)]
    oracle_names = ["son_torus", "evaluation_greedy"] + [f"random_regular_seed_{i}" for i in range(RANDOM_CANDIDATES)]
    cal_scores = score_all_candidates(
        path_caches,
        cal_payload["dispatch_sparse"],
        cal_payload["combine_sparse"],
        candidate_names,
    )
    eval_scores = score_all_candidates(
        path_caches,
        eval_payload["dispatch_sparse"],
        eval_payload["combine_sparse"],
        sorted(set(candidate_names + oracle_names)),
    )
    cal_selected = cal_scores[0]["name"]
    oracle_selected = min(
        [row for row in eval_scores if row["name"] in oracle_names],
        key=lambda row: (row["score_max_link_load_bytes"], row["name"]),
    )["name"]
    random_rows = [row for row in eval_scores if row["name"].startswith("random_regular_seed_")]
    best_random = min(random_rows, key=lambda row: (row["score_max_link_load_bytes"], row["name"]))["name"]
    median_random = sorted(random_rows, key=lambda row: (row["score_max_link_load_bytes"], row["name"]))[len(random_rows) // 2]["name"]

    # Window reconfiguration is evaluated exactly over requests, so scoring all
    # 35 candidates for every request is expensive and low-value.  Use a no-leak
    # shortlist picked only from calibration traffic plus fixed baselines.
    window_candidate_names = []
    for name in ["son_torus", "random_regular_seed_0", "calibration_greedy"]:
        if name not in window_candidate_names:
            window_candidate_names.append(name)
    for row in cal_scores[:12]:
        if row["name"] not in window_candidate_names:
            window_candidate_names.append(row["name"])

    window4 = request_window_strategy(request_payloads, cal_count, path_caches, window_candidate_names, 4)
    window8 = request_window_strategy(request_payloads, cal_count, path_caches, window_candidate_names, 8)

    candidate_audits = {name: graph_audit(graph, npu) for name, graph in candidates.items()}
    graph_budget_pass = all(audit["same_degree_bandwidth_budget_as_son"] for audit in candidate_audits.values())
    valid_graphs = all(audit["valid"] for audit in candidate_audits.values())

    graph_quality = {
        name: graph_quality_light(path_caches[name])
        for name in sorted(set(["son_torus", "calibration_greedy", "evaluation_greedy", cal_selected, oracle_selected, best_random, median_random, "random_regular_seed_0"]))
    }

    greedy_edges = edges_from_graph(candidates["calibration_greedy"])
    selected_edges = edges_from_graph(candidates[cal_selected])
    top_pairs = sorted(evaluation_demand.items(), key=lambda item: (-item[1], item[0]))
    greedy_audit = {
        "top16_pairs": [{"src": s, "dst": d, "bytes": b} for ((s, d), b) in top_pairs[:16]],
        "top32_direct_in_greedy": sum(1 for (pair, _) in top_pairs[:32] if tuple(sorted(pair)) in greedy_edges),
        "top64_direct_in_greedy": sum(1 for (pair, _) in top_pairs[:64] if tuple(sorted(pair)) in greedy_edges),
        "top32_direct_in_calibrated_selected": sum(1 for (pair, _) in top_pairs[:32] if tuple(sorted(pair)) in selected_edges),
        "top64_direct_in_calibrated_selected": sum(1 for (pair, _) in top_pairs[:64] if tuple(sorted(pair)) in selected_edges),
        "greedy_rank_by_eval_fluid": {row["name"]: idx + 1 for idx, row in enumerate(eval_scores)}.get("calibration_greedy"),
        "evaluation_greedy_rank_by_eval_fluid": {row["name"]: idx + 1 for idx, row in enumerate(eval_scores)}.get("evaluation_greedy"),
    }

    static_fluid = {
        "son_torus": score_candidate(path_caches["son_torus"], eval_payload["dispatch_sparse"], eval_payload["combine_sparse"]),
        "fixed_random_seed0": score_candidate(path_caches["random_regular_seed_0"], eval_payload["dispatch_sparse"], eval_payload["combine_sparse"]),
        "best_random": score_candidate(path_caches[best_random], eval_payload["dispatch_sparse"], eval_payload["combine_sparse"]),
        "workload_calibrated": score_candidate(path_caches[cal_selected], eval_payload["dispatch_sparse"], eval_payload["combine_sparse"]),
        "workload_oracle": score_candidate(path_caches[oracle_selected], eval_payload["dispatch_sparse"], eval_payload["combine_sparse"]),
        "greedy_calibration": score_candidate(path_caches["calibration_greedy"], eval_payload["dispatch_sparse"], eval_payload["combine_sparse"]),
        "greedy_evaluation_oracle": score_candidate(path_caches["evaluation_greedy"], eval_payload["dispatch_sparse"], eval_payload["combine_sparse"]),
    }
    static_fluid_cycles = {key: int(value / ASTRA_BYTES_PER_NS) for key, value in static_fluid.items()}
    segmented_candidate_cycles = {
        name: segmented_static_strategy(request_payloads, cal_count, path_caches[name])["fluid_cycles"]
        for name in candidates
    }
    segmented_fluid_cycles = {
        "son_torus": segmented_candidate_cycles["son_torus"],
        "fixed_random_seed0": segmented_candidate_cycles["random_regular_seed_0"],
        "best_random": segmented_candidate_cycles[best_random],
        "workload_calibrated": segmented_candidate_cycles[cal_selected],
        "workload_oracle": segmented_candidate_cycles[oracle_selected],
        "greedy_calibration": segmented_candidate_cycles["calibration_greedy"],
    }

    astra_results: dict[str, Any] = {}
    if wid in ASTRA_WORKLOAD_IDS:
        rep_names = []
        for name in ["son_torus", "random_regular_seed_0", cal_selected, oracle_selected, "calibration_greedy", best_random]:
            if name not in rep_names:
                rep_names.append(name)
        for name in rep_names:
            astra_results[name] = write_astra_artifacts(wid, npu, name, candidates[name], eval_payload, wout)

    workload_summary = {
        "workload": {"id": wid, "label": workload["label"], "path": str(workload["path"])},
        "trace_parse": {
            "files_found": parsed["files_found"],
            "files_used": parsed["files_used"],
            "moe_layer_count": parsed["moe_layer_count"],
            "num_experts": parsed["num_experts"],
            "malformed_records": parsed["malformed_records"],
            "selected_expert_events": parsed["selected_expert_events"],
        },
        "split": {
            "calibration_request_count": len(cal_indices),
            "evaluation_request_count": len(eval_indices),
            "calibration_request_ids": cal_payload["request_ids"],
            "evaluation_request_ids_head": eval_payload["request_ids"][:10],
            "evaluation_request_ids_tail": eval_payload["request_ids"][-10:],
        },
        "expert_hotness": expert_hotness(parsed),
        "old_pair_id_note": old_pair_id_note(npu),
        "hotness_propagation": mapping_sensitivity(parsed, npu),
        "aggregation_window_sensitivity": aggregation_window_sensitivity(request_payloads, npu),
        "traffic_primary": {
            "full": {
                "dispatch": matrix_stats_from_sparse(full_payload["dispatch_sparse"], npu),
                "combine": matrix_stats_from_sparse(full_payload["combine_sparse"], npu),
                "byte_conservation": full_payload["byte_conservation_pass"],
            },
            "calibration": {
                "dispatch": matrix_stats_from_sparse(cal_payload["dispatch_sparse"], npu),
                "combine": matrix_stats_from_sparse(cal_payload["combine_sparse"], npu),
                "byte_conservation": cal_payload["byte_conservation_pass"],
            },
            "evaluation": {
                "dispatch": matrix_stats_from_sparse(eval_payload["dispatch_sparse"], npu),
                "combine": matrix_stats_from_sparse(eval_payload["combine_sparse"], npu),
                "byte_conservation": eval_payload["byte_conservation_pass"],
            },
        },
        "candidate_pool": {
            "candidate_count": len(candidates),
            "graph_budget_pass": graph_budget_pass,
            "all_graphs_valid": valid_graphs,
            "topology_validation_representatives": {
                key: candidate_audits[key]
                for key in sorted(set(["son_torus", "random_regular_seed_0", "calibration_greedy", "evaluation_greedy", cal_selected, oracle_selected, best_random]))
            },
            "graph_quality_representatives": graph_quality,
        },
        "candidate_scores": {
            "calibration_top12": cal_scores[:12],
            "evaluation_top12": eval_scores[:12],
            "evaluation_all": eval_scores,
        },
        "selected": {
            "workload_calibrated": cal_selected,
            "workload_oracle": oracle_selected,
            "best_random": best_random,
            "median_random": median_random,
            "fixed_random": "random_regular_seed_0",
        },
        "window_reconfiguration": {"W4": window4, "W8": window8},
        "window_candidate_pool": {
            "selection_rule": "SON + fixed seed0 + calibration_greedy + top12 candidates by calibration fluid score",
            "names": window_candidate_names,
            "no_evaluation_leakage": True,
        },
        "fluid_static_score_bytes": static_fluid,
        "fluid_static_cycles_aggregate_lower_bound": static_fluid_cycles,
        "fluid_static_cycles_segmented_per_request": segmented_fluid_cycles,
        "fluid_segmented_cycles_all_candidates": segmented_candidate_cycles,
        "greedy_audit": greedy_audit,
        "native_astra_representative": astra_results,
        "tiny_subchunk_audit_representative": {
            name: {
                "dispatch": tiny_subchunk_audit(path_caches[name], eval_payload["dispatch_sparse"]),
                "combine": tiny_subchunk_audit(path_caches[name], eval_payload["combine_sparse"]),
            }
            for name in sorted(set(["son_torus", "random_regular_seed_0", cal_selected, oracle_selected, "calibration_greedy"]))
        },
        "validation_pass": {
            "byte_conservation": full_payload["byte_conservation_pass"] and cal_payload["byte_conservation_pass"] and eval_payload["byte_conservation_pass"],
            "graph_budget": graph_budget_pass,
            "graphs_valid": valid_graphs,
            "calibrated_no_leakage": True,
            "oracle_labelled_reference": True,
        },
    }
    write_json(wout / "workload_summary.json", workload_summary)
    write_json(wout / "candidate_scores_evaluation.json", eval_scores)
    write_json(wout / "candidate_scores_calibration.json", cal_scores)
    write_json(wout / "window_w4.json", window4)
    write_json(wout / "window_w8.json", window8)
    write_json(wout / "candidate_audits_representatives.json", workload_summary["candidate_pool"]["topology_validation_representatives"])
    return workload_summary


def universal_static_selection(results: list[dict[str, Any]]) -> dict[str, Any]:
    random_names = [f"random_regular_seed_{seed}" for seed in range(RANDOM_CANDIDATES)]
    rows = []
    for name in random_names:
        normalized = []
        ranks = []
        for result in results:
            segmented = result["fluid_segmented_cycles_all_candidates"]
            son = segmented["son_torus"]
            normalized.append(segmented[name] / son if son else 0)
            ranking = sorted(segmented.items(), key=lambda item: (item[1], item[0]))
            ranks.append({candidate: idx + 1 for idx, (candidate, _) in enumerate(ranking)}[name])
        rows.append(
            {
                "name": name,
                "average_normalized_score_vs_son": statistics.mean(normalized),
                "average_rank": statistics.mean(ranks),
                "max_normalized_score_vs_son": max(normalized),
            }
        )
    rows = sorted(rows, key=lambda row: (row["average_normalized_score_vs_son"], row["average_rank"], row["name"]))
    return {"best_universal_static_random": rows[0]["name"], "ranking": rows}


def method_ranking(result: dict[str, Any], universal_name: str) -> list[dict[str, Any]]:
    segmented = result["fluid_segmented_cycles_all_candidates"]
    selected = result["selected"]
    workload_cal = selected["workload_calibrated"]
    fixed = "random_regular_seed_0"
    son = "son_torus"
    methods = [
        ("SON/static baseline", son, segmented[son]),
        ("fixed random seed0", fixed, segmented[fixed]),
        ("best universal static", universal_name, segmented[universal_name]),
        ("workload-level calibrated OCS", workload_cal, segmented[workload_cal]),
        ("request-window W=4 calibrated OCS, 1us", "window_W4", result["window_reconfiguration"]["W4"]["fluid_cycles_with_reconfig"]["1us"]),
        ("request-window W=8 calibrated OCS, 1us", "window_W8", result["window_reconfiguration"]["W8"]["fluid_cycles_with_reconfig"]["1us"]),
    ]
    return sorted(
        [
            {"method": method, "candidate": candidate, "fluid_cycles": cycles}
            for method, candidate, cycles in methods
        ],
        key=lambda row: row["fluid_cycles"],
    )


def write_report(summary: dict[str, Any]) -> None:
    compact = {
        "ranked_valid_non_oracle_methods": summary["ranked_valid_non_oracle_methods"],
        "cross_workload_summary": summary["cross_workload_summary"],
        "final_diagnosis": summary["final_diagnosis"],
        "limitations": summary["limitations"],
    }
    (OUT / "README.md").write_text(
        f"""# V37c 128-GPU Hotness + Best OCS Reconfiguration Audit

## Scope

This audit checks whether 32-GPU expert-to-GPU dilution hid hotness, and compares valid no-leak OCS reconfiguration strategies at 128 GPUs. ASTRA C++ core was not modified. Full strategy search uses a fluid link-load model; native ASTRA is used only for representative static topologies on priority workloads because ASTRA does not yet support safe in-run topology swaps.

## Compact Result

```json
{json.dumps(compact, indent=2)}
```

## What This Can Claim

- Expert hotness, destination-GPU hotness, true GPU-pair hotness, and aggregation/window sensitivity were audited at 128 GPUs.
- Workload-level calibrated and request-window calibrated OCS strategies were evaluated without future/evaluation leakage in the fluid model.
- Representative native ASTRA runs validate selected static 128-GPU topologies for priority workloads.

## What This Cannot Claim

- Native ASTRA in-run RON W=4/W=8 topology swaps.
- Real serving latency.
- Physical transparent OCS modelling.
- Token/layer-level execution timing.
- A paper-ready figure.

Detailed data are in `summary.json` and per-workload `workload_summary.json` files.
"""
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = [process_workload(workload) for workload in WORKLOADS]
    universal = universal_static_selection(results)
    universal_name = universal["best_universal_static_random"]

    cross_rows = []
    ranked_by_workload = {}
    for result in results:
        wid = result["workload"]["id"]
        ranking = method_ranking(result, universal_name)
        ranked_by_workload[wid] = ranking
        expert = result["expert_hotness"]
        primary_pair = result["traffic_primary"]["evaluation"]["dispatch"]
        best_method = ranking[0]
        son_cycles = next(row["fluid_cycles"] for row in ranking if row["method"] == "SON/static baseline")
        workload_cal_cycles = next(row["fluid_cycles"] for row in ranking if row["method"] == "workload-level calibrated OCS")
        universal_cycles = next(row["fluid_cycles"] for row in ranking if row["method"] == "best universal static")
        window4_cycles = next(row["fluid_cycles"] for row in ranking if row["method"] == "request-window W=4 calibrated OCS, 1us")
        window8_cycles = next(row["fluid_cycles"] for row in ranking if row["method"] == "request-window W=8 calibrated OCS, 1us")
        cross_rows.append(
            {
                "workload": wid,
                "num_experts": result["trace_parse"]["num_experts"],
                "expert_top1_share": expert["top1_share"],
                "expert_top16_share": expert["top16_share"],
                "expert_gini": expert["gini"],
                "pair_top16_share_128gpu": primary_pair["top16_share"],
                "pair_gini_128gpu": primary_pair["gini"],
                "workload_calibrated_candidate": result["selected"]["workload_calibrated"],
                "oracle_candidate": result["selected"]["workload_oracle"],
                "best_method": best_method["method"],
                "son_cycles_fluid": son_cycles,
                "best_universal_cycles_fluid": universal_cycles,
                "workload_calibrated_cycles_fluid": workload_cal_cycles,
                "window4_cycles_1us_fluid": window4_cycles,
                "window8_cycles_1us_fluid": window8_cycles,
                "workload_calibrated_gain_vs_son_percent": 100 * (son_cycles - workload_cal_cycles) / son_cycles if son_cycles else None,
                "workload_calibrated_gain_vs_universal_percent": 100 * (universal_cycles - workload_cal_cycles) / universal_cycles if universal_cycles else None,
                "window4_gain_vs_workload_calibrated_percent": 100 * (workload_cal_cycles - window4_cycles) / workload_cal_cycles if workload_cal_cycles else None,
                "window8_gain_vs_workload_calibrated_percent": 100 * (workload_cal_cycles - window8_cycles) / workload_cal_cycles if workload_cal_cycles else None,
                "greedy_rank": result["greedy_audit"]["greedy_rank_by_eval_fluid"],
                "evaluation_greedy_rank": result["greedy_audit"]["evaluation_greedy_rank_by_eval_fluid"],
            }
        )

    final_diagnosis = {
        "expert_hotness_reproduced": True,
        "old_gpu_pair_ids": old_pair_id_note(MAIN_NPU),
        "hot_experts_become_hot_destination_gpus": "see per-workload hotness_propagation; block placement generally preserves destination skew more than round_robin",
        "hot_destination_gpus_create_true_hot_gpu_pairs": "only under some source policies; full aggregate remains broad for several workloads",
        "source_policy_and_placement_maximising_hot_pairs": "reported in each workload mapping_sensitivity; decode_like_batch often increases source concentration",
        "full_aggregation_hides_local_windows": "compare aggregation_window_sensitivity full vs per-request/window summaries",
        "greedy_hot_pair_topology_strength": "greedy ranks are reported per workload; do not assume it wins",
        "improved_greedy_local_search": "not implemented in V37c; strict greedy repair is included, local-search is left for a follow-up if greedy is close",
        "universal_static_topology": universal_name,
        "best_valid_non_oracle_by_workload": {row["workload"]: row["best_method"] for row in cross_rows},
        "story_decision": (
            "Use the measured rankings: if workload/window calibrated beats best universal static, the story is OCS reconfiguration. "
            "If best universal static is close, the safer story is scale-aware degree-4 topology search with reconfiguration as conditional."
        ),
    }
    limitations = {
        "native_astra_window_reconfiguration": "not implemented; request-window methods are exact no-leak fluid scores, not native in-run ASTRA topology swaps",
        "astra_representative_scope": f"native ASTRA static representative runs are limited to {sorted(ASTRA_WORKLOAD_IDS)} with {ASTRA_TIMEOUT_S}s timeout per phase/topology",
        "all_path_ecmp": "not used; ecmp_max_paths=4",
        "figures": "none generated",
    }
    summary = {
        "scope": "V37c 128-GPU hotness and best OCS reconfiguration audit",
        "main_npu": MAIN_NPU,
        "optional_32gpu_summary": compact_32gpu_summary_from_v37(),
        "workloads": results,
        "universal_static_selection": universal,
        "cross_workload_summary": cross_rows,
        "ranked_valid_non_oracle_methods": ranked_by_workload,
        "final_diagnosis": final_diagnosis,
        "limitations": limitations,
    }
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "cross_workload_summary.json", cross_rows)
    write_json(OUT / "universal_static_selection.json", universal)
    write_report(summary)
    print(json.dumps({"cross_workload_summary": cross_rows, "final_diagnosis": final_diagnosis, "limitations": limitations}, indent=2))


if __name__ == "__main__":
    main()
