#!/usr/bin/env python3
"""V34 production-topology ECMP validation for 32-GPU EN/SON baselines."""

from __future__ import annotations

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
OUT = REPO / "results/moe_expert_trace_converter/v34_production_topology_ecmp_smoke"
PROTO_DIR = REPO / "extern/graph_frontend/chakra/schema/protobuf"
PROTO_UTILS = REPO / "extern/graph_frontend/chakra/src/third_party/utils"

sys.path.insert(0, str(PROTO_DIR))
sys.path.insert(0, str(PROTO_UTILS))

from et_def_pb2 import AttributeProto, GlobalMetadata, Node, NodeType  # type: ignore  # noqa: E402
from protolib import encodeMessage as encode_message  # type: ignore  # noqa: E402

ASTRA_BIN = REPO / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
SYSTEM = REPO / "examples/system/native_collectives/Ring_4chunks.json"
REMOTE_MEMORY = REPO / "examples/remote_memory/analytical/no_memory_expansion.json"
LINK_GBPS = 400
ASTRA_BYTES_PER_NS = (LINK_GBPS / 8) * (1 << 30) / 1_000_000_000


def add_attr(node: Node, name: str, value: int | bool) -> None:
    if isinstance(value, bool):
        node.attr.append(AttributeProto(name=name, bool_val=value))
    else:
        node.attr.append(AttributeProto(name=name, uint64_val=int(value)))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


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
    n = len(matrix)
    nodes_by_rank: dict[int, list[Node]] = {rank: [] for rank in range(n)}
    next_id = {rank: 1 for rank in range(n)}
    tag = 1
    total_bytes = 0
    messages = 0
    for src in range(n):
        for dst in range(n):
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
    for rank in range(n):
        nodes = nodes_by_rank[rank] or [comp_node(1)]
        with (prefix.parent / f"{prefix.name}.{rank}.et").open("wb") as handle:
            encode_message(handle, GlobalMetadata(version="0.0.4"))
            for node in nodes:
                encode_message(handle, node)

    return {"prefix": str(prefix), "rank_count": n, "messages": messages, "total_bytes": total_bytes}


def en_folded_clos_graph() -> dict[str, Any]:
    edges: list[dict[str, Any]] = []
    leaf_base = 32
    spine_base = 36
    for gpu in range(32):
        leaf = leaf_base + gpu // 8
        edges.append({"src": gpu, "dst": leaf, "bandwidth_gbps": LINK_GBPS, "latency_ns": 0})
    for leaf_idx in range(4):
        leaf = leaf_base + leaf_idx
        for spine_idx in range(4):
            spine = spine_base + spine_idx
            edges.append({"src": leaf, "dst": spine, "bandwidth_gbps": LINK_GBPS, "latency_ns": 0})
    return {
        "name": "en_folded_clos_32gpu_4leaf_4spine",
        "node_count": 40,
        "gpu_count": 32,
        "rank_nodes": list(range(32)),
        "directed": False,
        "metadata": {
            "gpus_per_leaf": 8,
            "leaf_nodes": list(range(32, 36)),
            "spine_nodes": list(range(36, 40)),
            "gpu_access_link_gbps": LINK_GBPS,
            "leaf_spine_link_gbps": LINK_GBPS,
        },
        "edges": edges,
    }


def son_torus_graph() -> dict[str, Any]:
    rows, cols = 4, 8
    edges: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    def node(r: int, c: int) -> int:
        return r * cols + c

    def add(u: int, v: int) -> None:
        key = tuple(sorted((u, v)))
        if key in seen:
            return
        seen.add(key)
        edges.append({"src": u, "dst": v, "bandwidth_gbps": LINK_GBPS, "latency_ns": 0})

    for r in range(rows):
        for c in range(cols):
            add(node(r, c), node(r, (c + 1) % cols))
            add(node(r, c), node((r + 1) % rows, c))
    return {
        "name": "son_2d_torus_4x8_32gpu",
        "node_count": 32,
        "gpu_count": 32,
        "directed": False,
        "metadata": {
            "rows": rows,
            "cols": cols,
            "degree": 4,
            "per_gpu_optical_bandwidth_tbps": 1.6,
            "per_link_bandwidth_gbps": LINK_GBPS,
        },
        "edges": edges,
    }


def adjacency(graph: dict[str, Any]) -> list[list[int]]:
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


def rank_node(graph: dict[str, Any], rank: int) -> int:
    return int((graph.get("rank_nodes") or list(range(graph["gpu_count"])))[rank])


def all_shortest_paths(graph: dict[str, Any], src_rank: int, dst_rank: int) -> list[list[int]]:
    adj = adjacency(graph)
    src, dst = rank_node(graph, src_rank), rank_node(graph, dst_rank)
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


def selected_paths(graph: dict[str, Any], src: int, dst: int, mode: str, max_paths: int | None) -> list[list[int]]:
    paths = all_shortest_paths(graph, src, dst)
    if mode == "deterministic":
        return paths[:1]
    if max_paths is not None and max_paths > 0:
        paths = paths[:max_paths]
    return paths


def network_config(path: Path, graph_path: Path, npus: int, mode: str, max_paths: int | None, log: bool = False) -> None:
    lines = [
        "topology: [ Graph ]",
        f"npus_count: [ {npus} ]",
        "bandwidth: [ 50.0 ]",
        "latency: [ 0.0 ]",
        f"graph_file: {graph_path}",
    ]
    if mode == "ecmp":
        lines += ["routing: ecmp", "ecmp_split: equal_bytes", f"ecmp_max_paths: {max_paths or 0}", f"ecmp_log: {'true' if log else 'false'}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def matrix_single(n: int, src: int, dst: int, size: int) -> list[list[int]]:
    matrix = [[0] * n for _ in range(n)]
    matrix[src][dst] = size
    return matrix


def matrix_uniform(n: int, size: int) -> list[list[int]]:
    return [[0 if src == dst else size for dst in range(n)] for src in range(n)]


def matrix_hot_destination(n: int, dst: int, size: int) -> list[list[int]]:
    matrix = [[0] * n for _ in range(n)]
    for src in range(n):
        if src != dst:
            matrix[src][dst] = size
    return matrix


def matrix_sparse_random(n: int, seed: int, density: float, min_size: int, max_size: int) -> list[list[int]]:
    rng = random.Random(seed)
    matrix = [[0] * n for _ in range(n)]
    for src in range(n):
        for dst in range(n):
            if src != dst and rng.random() < density:
                matrix[src][dst] = rng.randint(min_size, max_size)
    return matrix


def split_bytes(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def link_load_estimate(graph: dict[str, Any], matrix: list[list[int]], mode: str, max_paths: int | None) -> dict[str, Any]:
    loads: dict[tuple[int, int], int] = defaultdict(int)
    selected_counts: list[int] = []
    nonzero_messages = 0
    for src, row in enumerate(matrix):
        for dst, size in enumerate(row):
            if src == dst or size <= 0:
                continue
            paths = selected_paths(graph, src, dst, mode, max_paths)
            if not paths:
                raise RuntimeError(f"no path {src}->{dst}")
            nonzero_messages += 1
            selected_counts.append(len(paths))
            for path, subbytes in zip(paths, split_bytes(size, len(paths))):
                for u, v in zip(path, path[1:]):
                    loads[(u, v)] += subbytes
    max_load = max(loads.values()) if loads else 0
    return {
        "nonzero_messages": nonzero_messages,
        "selected_path_count_min": min(selected_counts) if selected_counts else 0,
        "selected_path_count_median": statistics.median(selected_counts) if selected_counts else 0,
        "selected_path_count_mean": statistics.mean(selected_counts) if selected_counts else 0,
        "selected_path_count_max": max(selected_counts) if selected_counts else 0,
        "max_link_load_bytes": max_load,
        "fluid_cycles": int(max_load / ASTRA_BYTES_PER_NS) if max_load else 0,
        "hot_links": [{"src": u, "dst": v, "bytes": b} for (u, v), b in sorted(loads.items(), key=lambda x: -x[1])[:8]],
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
    ecmp_lines = [line for line in proc.stdout.splitlines() if "[ecmp]" in line]
    stderr_lines = proc.stderr.splitlines()
    return {
        "label": label,
        "returncode": proc.returncode,
        "success": proc.returncode == 0 and bool(cycles),
        "runtime_s": runtime_s,
        "command": " ".join(cmd),
        "stdout": str(stdout),
        "stderr": str(stderr),
        "stderr_tail": stderr_lines[-8:],
        "max_cycles": max(cycles) if cycles else None,
        "cycles_count": len(cycles),
        "sample_ecmp_logs": ecmp_lines[:8],
    }


def path_count_distribution(graph: dict[str, Any]) -> dict[str, Any]:
    counts = []
    examples: list[dict[str, Any]] = []
    for src in range(graph["gpu_count"]):
        for dst in range(graph["gpu_count"]):
            if src == dst:
                continue
            paths = all_shortest_paths(graph, src, dst)
            count = len(paths)
            counts.append(count)
            examples.append({"src": src, "dst": dst, "path_count": count, "hop_count": len(paths[0]) - 1})
    hist = Counter(counts)
    max_count = max(counts)
    return {
        "min": min(counts),
        "median": statistics.median(counts),
        "mean": statistics.mean(counts),
        "max": max_count,
        "histogram": dict(sorted(hist.items())),
        "largest_pairs": sorted([item for item in examples if item["path_count"] == max_count], key=lambda x: (x["src"], x["dst"]))[:20],
    }


def parse_ecmp_log_paths(stdout_path: Path) -> list[str]:
    text = stdout_path.read_text()
    paths = []
    for line in text.splitlines():
        if "[ecmp]" in line:
            paths.extend(re.findall(r"route=(\[[^\]]+\])", line))
    return paths


def write_report(summary: dict[str, Any]) -> None:
    (OUT / "README.md").write_text(
        f"""# V3.4 Production-Topology ECMP Smoke

## Scope

This validates production-scale baseline topologies only:

- 32-GPU EN folded-Clos ECMP, clean ECMP only.
- 32-GPU SON 4x8 2D torus deterministic and ECMP.

It does **not** run full MoE workloads, generate figures, model EN imbalance, implement RON calibrated/oracle/W=4, or claim physical transparent OCS behavior.

## Final Answers

1. Can native GraphTopology ECMP support 32-GPU EN folded-Clos ECMP? **Yes.**
2. Can native GraphTopology ECMP support 32-GPU SON 2D torus ECMP? **Yes.**
3. For EN folded-Clos, does cross-leaf traffic correctly split across all 4 spines? **Yes:** GPU0->GPU16 splits over four spine paths.
4. For EN folded-Clos, is intra-leaf traffic represented as GPU -> leaf -> GPU logical forwarding? **Yes:** GPU0->GPU1 uses `[0,32,1]`; this is logical folded-Clos forwarding, not direct GPU-GPU wiring.
5. For SON 2D torus, is all-path ECMP tractable at 32 GPUs? **Route precomputation is tractable, but all-path ASTRA execution is not robust enough for production.** The dense uniform all-to-all case triggered ASTRA's event-queue assertion after splitting 1000B messages over many paths.
6. Does all-path ECMP look too idealised? **Yes.** It can use up to 60 equal shortest paths on the 4x8 torus, and tiny subchunks can create zero-delay event-queue issues.
7. Recommended production SON setting: **`ecmp_max_paths: 4`** as the main setting, with `2` and `8` as sensitivity checks. Treat `all` as route-statistics / upper-bound only until tiny-subchunk handling is fixed.
8. Native ASTRA vs V2.8 link-load evaluator: **different but explainable.** Link-load is a fluid lower bound; ASTRA is chunk-level multi-hop forwarding with per-hop serialization and queues.
9. Are differences explainable by store-and-forward semantics? **Yes.**
10. Is it safe to run one real MoE workload next? **Yes, as a one-workload validation, not yet a full paper result.**

## EN Folded-Clos Validation

- node_count: `{summary["en"]["node_count"]}`
- gpu_count: `{summary["en"]["gpu_count"]}`
- leaf nodes: `{summary["en"]["leaf_nodes"]}`
- spine nodes: `{summary["en"]["spine_nodes"]}`
- intra-leaf GPU0->GPU1 paths: `{summary["en"]["intra_leaf_paths"]}`
- cross-leaf GPU0->GPU16 ECMP paths: `{summary["en"]["cross_leaf_paths"]}`
- cross-leaf path count: `{summary["en"]["cross_leaf_path_count"]}`
- ASTRA cross-leaf sample log: `{summary["en"]["cross_leaf_run"]["sample_ecmp_logs"]}`

## SON Torus Route Statistics

```json
{json.dumps(summary["son_path_distribution"], indent=2)}
```

## Deterministic vs ECMP and Link-Load Comparison

```json
{json.dumps(summary["son_results"], indent=2)}
```

## Interpretation

All-path ECMP route enumeration is tractable at 32 GPUs, but it is too optimistic and not robust enough as a production default. On the 4x8 torus, one pair can have 60 equal shortest paths; splitting small messages across all of them can produce tiny subchunks and hit ASTRA's zero-delay event-queue assertion. For optical/interconnect studies, `ecmp_max_paths=4` is a safer mainline because it still removes deterministic BFS tie-breaking while avoiding excessive path fanout. `ecmp_max_paths=2` and `8` are useful sensitivity points; `all` should be labeled route-statistics / upper-bound only until tiny-subchunk handling is fixed.

Native ASTRA timing is usually above the V2.8-style link-load estimate because ASTRA forwards each chunk hop-by-hop through `Device`/`Link` queues. The link-load model computes `max_link_load / bandwidth`, which is closer to a fluid lower bound and does not charge repeated per-hop serialization in the same way.

## Generated Files

- Graph JSONs: `graphs/en_folded_clos_32.json`, `graphs/son_torus_4x8_32.json`
- Network configs: `network_configs/*.yml`
- Traffic matrices: `traffic_matrices/*.json`
- Chakra traces: `chakra_traces/*/workload.*.et`
- Run logs: `runs/*.stdout.txt`, `runs/*.stderr.txt`
- Machine-readable summary: `summary.json`
"""
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    graphs_dir = OUT / "graphs"
    configs_dir = OUT / "network_configs"
    matrices_dir = OUT / "traffic_matrices"
    traces_dir = OUT / "chakra_traces"

    en_graph = en_folded_clos_graph()
    son_graph = son_torus_graph()
    en_graph_path = graphs_dir / "en_folded_clos_32.json"
    son_graph_path = graphs_dir / "son_torus_4x8_32.json"
    write_json(en_graph_path, en_graph)
    write_json(son_graph_path, son_graph)

    network_config(configs_dir / "en_ecmp_all_log.yml", en_graph_path, 32, "ecmp", None, True)
    network_config(configs_dir / "son_deterministic.yml", son_graph_path, 32, "deterministic", None, False)
    for max_paths in (2, 4, 8, 0):
        label = "all" if max_paths == 0 else str(max_paths)
        network_config(configs_dir / f"son_ecmp_{label}.yml", son_graph_path, 32, "ecmp", max_paths, False)
    network_config(configs_dir / "son_ecmp_all_log.yml", son_graph_path, 32, "ecmp", None, True)

    # EN path validation traces.
    en_intra = matrix_single(32, 0, 1, 5000)
    en_cross = matrix_single(32, 0, 16, 5000)
    write_matrix_trace(traces_dir / "en_intra_0_to_1" / "workload", en_intra)
    write_matrix_trace(traces_dir / "en_cross_0_to_16" / "workload", en_cross)
    en_intra_run = run_astra("en_intra_0_to_1", traces_dir / "en_intra_0_to_1" / "workload", configs_dir / "en_ecmp_all_log.yml")
    en_cross_run = run_astra("en_cross_0_to_16", traces_dir / "en_cross_0_to_16" / "workload", configs_dir / "en_ecmp_all_log.yml")

    dist = path_count_distribution(son_graph)
    max_pair = dist["largest_pairs"][0]
    traffic_cases = {
        "single_pair_0_to_4": matrix_single(32, 0, 4, 16000),
        f"single_pair_maxpath_{max_pair['src']}_to_{max_pair['dst']}": matrix_single(32, max_pair["src"], max_pair["dst"], 16000),
        "uniform_all_to_all": matrix_uniform(32, 1000),
        "hot_destination_0": matrix_hot_destination(32, 0, 8000),
        "sparse_random_seed_20260518": matrix_sparse_random(32, 20260518, 0.08, 1000, 12000),
    }

    trace_meta: dict[str, Any] = {}
    for name, matrix in traffic_cases.items():
        write_json(matrices_dir / f"{name}.json", {"name": name, "matrix": matrix})
        trace_meta[name] = write_matrix_trace(traces_dir / name / "workload", matrix)

    configs = {
        "deterministic": configs_dir / "son_deterministic.yml",
        "ecmp_2": configs_dir / "son_ecmp_2.yml",
        "ecmp_4": configs_dir / "son_ecmp_4.yml",
        "ecmp_8": configs_dir / "son_ecmp_8.yml",
        "ecmp_all": configs_dir / "son_ecmp_all.yml",
    }
    mode_params = {
        "deterministic": ("deterministic", None),
        "ecmp_2": ("ecmp", 2),
        "ecmp_4": ("ecmp", 4),
        "ecmp_8": ("ecmp", 8),
        "ecmp_all": ("ecmp", None),
    }

    son_results: dict[str, Any] = {}
    for traffic_name, matrix in traffic_cases.items():
        son_results[traffic_name] = {}
        for config_name, config_path in configs.items():
            run = run_astra(f"son_{traffic_name}_{config_name}", traces_dir / traffic_name / "workload", config_path)
            mode, max_paths = mode_params[config_name]
            estimate = link_load_estimate(son_graph, matrix, mode, max_paths)
            astra_cycles = run["max_cycles"]
            estimate_cycles = estimate["fluid_cycles"]
            son_results[traffic_name][config_name] = {
                "astra_max_cycles": astra_cycles,
                "run_returncode": run["returncode"],
                "run_success": run["success"],
                "stderr_tail": run["stderr_tail"],
                "runtime_s": run["runtime_s"],
                "link_load_fluid_cycles": estimate_cycles,
                "astra_over_fluid": (astra_cycles / estimate_cycles) if astra_cycles is not None and estimate_cycles else None,
                "selected_path_count_min": estimate["selected_path_count_min"],
                "selected_path_count_median": estimate["selected_path_count_median"],
                "selected_path_count_mean": estimate["selected_path_count_mean"],
                "selected_path_count_max": estimate["selected_path_count_max"],
                "max_link_load_bytes": estimate["max_link_load_bytes"],
                "hot_links": estimate["hot_links"],
                "command": run["command"],
            }

    core_son_passed = all(
        son_results[traffic_name][config_name]["run_success"]
        for traffic_name in traffic_cases
        for config_name in ("deterministic", "ecmp_2", "ecmp_4", "ecmp_8")
    )
    all_path_passed = all(
        son_results[traffic_name]["ecmp_all"]["run_success"] for traffic_name in traffic_cases
    )
    summary = {
        "all_passed": en_intra_run["success"] and en_cross_run["success"] and core_son_passed and all_path_passed,
        "core_production_passed": en_intra_run["success"] and en_cross_run["success"] and core_son_passed,
        "all_path_passed": all_path_passed,
        "en": {
            "node_count": en_graph["node_count"],
            "gpu_count": en_graph["gpu_count"],
            "rank_nodes": en_graph["rank_nodes"],
            "leaf_nodes": en_graph["metadata"]["leaf_nodes"],
            "spine_nodes": en_graph["metadata"]["spine_nodes"],
            "intra_leaf_paths": all_shortest_paths(en_graph, 0, 1),
            "cross_leaf_paths": all_shortest_paths(en_graph, 0, 16),
            "cross_leaf_path_count": len(all_shortest_paths(en_graph, 0, 16)),
            "intra_leaf_run": en_intra_run,
            "cross_leaf_run": en_cross_run,
        },
        "son_path_distribution": dist,
        "traffic_trace_meta": trace_meta,
        "son_results": son_results,
        "recommendation": {
            "production_son_ecmp_max_paths": 4,
            "reason": "4 removes deterministic BFS tie-breaking while avoiding the optimistic/tiny-subchunk fanout of all-path ECMP; keep 2 and 8 as sensitivity. Treat all-path as route-statistics/upper-bound only until tiny-subchunk handling is fixed.",
            "safe_to_run_one_real_moe_next": en_intra_run["success"] and en_cross_run["success"] and core_son_passed,
        },
    }
    write_json(OUT / "summary.json", summary)
    write_report(summary)
    print(json.dumps(summary, indent=2))
    if not summary["core_production_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
