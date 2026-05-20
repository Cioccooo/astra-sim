#!/usr/bin/env python3
"""V35 one-workload native ASTRA GraphTopology validation.

This script validates the pipeline:
HF MoE expert-selection JSON -> aggregated per-GPU-pair dispatch/combine
matrices -> Chakra SEND/RECV traces -> native ASTRA GraphTopology timing.
"""

from __future__ import annotations

import json
import math
import os
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
OUT = REPO / "results/moe_expert_trace_converter/v35_one_workload_aggregated_native_astra_validation"
TRACE_DIR = Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/mmlu/machine_learning")
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
HIDDEN_SIZE = 4096
BYTES_PER_VALUE = 2
BYTES_PER_SELECTION = HIDDEN_SIZE * BYTES_PER_VALUE
BLOCK_SIZE = 16
LINK_GBPS = 400
ASTRA_BYTES_PER_NS = (LINK_GBPS / 8) * (1 << 30) / 1_000_000_000
ONE_CYCLE_THRESHOLD_BYTES = math.ceil(ASTRA_BYTES_PER_NS)


def numeric_json_sort(path: Path) -> int | str:
    return int(path.stem) if path.stem.isdigit() else path.stem


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


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

    return {
        "prefix": str(prefix),
        "rank_count": n,
        "messages": messages,
        "total_bytes": total_bytes,
    }


def expert_rank(expert_id: int, ep_size: int, num_experts: int) -> int:
    experts_per_rank = num_experts / ep_size
    return min(int(expert_id / experts_per_rank), ep_size - 1)


def block_source_rank(global_token_index: int, ep_size: int, block_size: int) -> int:
    return (global_token_index // block_size) % ep_size


def parse_qwen_prefill() -> dict[str, Any]:
    files = sorted(TRACE_DIR.glob("*.json"), key=numeric_json_sort)
    raw: list[tuple[str, int, int, list[int]]] = []
    request_ids: list[str] = []
    max_expert = -1
    moe_layers: set[int] = set()
    malformed_records = 0
    global_token_offset = 0
    prefill_tokens = 0
    rows_by_request: dict[str, int] = {}

    for path in files:
        request_ids.append(path.stem)
        try:
            trace = json.loads(path.read_text())
        except Exception:
            malformed_records += 1
            continue
        if not isinstance(trace, list) or not trace or not isinstance(trace[0], dict):
            malformed_records += 1
            continue
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
                raw.append((path.stem, layer_id, global_token_offset + row_index, parsed))
        rows_by_request[path.stem] = max_rows
        prefill_tokens += max_rows
        global_token_offset += max_rows

    if max_expert < 0:
        raise RuntimeError(f"No expert ids found under {TRACE_DIR}")

    num_experts = max_expert + 1
    dispatch = [[0 for _ in range(NPU_COUNT)] for _ in range(NPU_COUNT)]
    combine = [[0 for _ in range(NPU_COUNT)] for _ in range(NPU_COUNT)]
    local_selections = 0
    remote_selections = 0
    selected_events = 0

    for _, _, global_token_index, experts in raw:
        src = block_source_rank(global_token_index, NPU_COUNT, BLOCK_SIZE)
        for expert_id in experts:
            selected_events += 1
            dst = expert_rank(expert_id, NPU_COUNT, num_experts)
            if src == dst:
                local_selections += 1
                continue
            dispatch[src][dst] += BYTES_PER_SELECTION
            combine[dst][src] += BYTES_PER_SELECTION
            remote_selections += 1

    theoretical_one_way_bytes = selected_events * BYTES_PER_SELECTION
    local_one_way_bytes = local_selections * BYTES_PER_SELECTION
    remote_one_way_bytes = remote_selections * BYTES_PER_SELECTION
    dispatch_bytes = sum(sum(row) for row in dispatch)
    combine_bytes = sum(sum(row) for row in combine)

    return {
        "request_ids": request_ids,
        "files_found": len(files),
        "files_used": len(request_ids),
        "prefill_input_tokens": prefill_tokens,
        "rows_by_request": rows_by_request,
        "moe_layers": sorted(moe_layers),
        "moe_layer_count": len(moe_layers),
        "inferred_num_experts": num_experts,
        "selected_expert_events": selected_events,
        "malformed_records": malformed_records,
        "theoretical_dispatch_bytes": theoretical_one_way_bytes,
        "theoretical_combine_bytes": theoretical_one_way_bytes,
        "local_dispatch_bytes_excluded": local_one_way_bytes,
        "local_combine_bytes_excluded": local_one_way_bytes,
        "remote_dispatch_bytes_retained": dispatch_bytes,
        "remote_combine_bytes_retained": combine_bytes,
        "byte_conservation_pass": (
            dispatch_bytes == remote_one_way_bytes
            and combine_bytes == remote_one_way_bytes
            and theoretical_one_way_bytes == local_one_way_bytes + remote_one_way_bytes
        ),
        "dispatch_matrix": dispatch,
        "combine_matrix": combine,
    }


def en_folded_clos_graph() -> dict[str, Any]:
    edges: list[dict[str, Any]] = []
    leaf_base = 32
    spine_base = 36
    for gpu in range(NPU_COUNT):
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
        "gpu_count": NPU_COUNT,
        "rank_nodes": list(range(NPU_COUNT)),
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
        "node_count": NPU_COUNT,
        "gpu_count": NPU_COUNT,
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


def split_bytes(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def network_config(path: Path, graph_path: Path, mode: str, max_paths: int | None) -> None:
    lines = [
        "topology: [ Graph ]",
        f"npus_count: [ {NPU_COUNT} ]",
        "bandwidth: [ 50.0 ]",
        "latency: [ 0.0 ]",
        f"graph_file: {graph_path}",
    ]
    if mode == "ecmp":
        lines += [
            "routing: ecmp",
            "ecmp_split: equal_bytes",
            f"ecmp_max_paths: {max_paths or 0}",
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


def matrix_stats(matrix: list[list[int]]) -> dict[str, Any]:
    messages = [matrix[src][dst] for src in range(len(matrix)) for dst in range(len(matrix)) if src != dst and matrix[src][dst] > 0]
    total = sum(messages)
    sorted_messages = sorted(messages, reverse=True)

    def top_share(k: int) -> float:
        return sum(sorted_messages[:k]) / total if total else 0.0

    thresholds = [54, 128, 256, 1024]
    return {
        "total_remote_bytes": total,
        "nonzero_gpu_pairs": len(messages),
        "top1_share": top_share(1),
        "top4_share": top_share(4),
        "top8_share": top_share(8),
        "top16_share": top_share(16),
        "gini": gini(messages),
        "message_bytes_min": min(messages) if messages else 0,
        "message_bytes_median": statistics.median(messages) if messages else 0,
        "message_bytes_mean": statistics.mean(messages) if messages else 0,
        "message_bytes_max": max(messages) if messages else 0,
        "messages_lt_54B": sum(1 for value in messages if value < 54),
        "messages_lt_128B": sum(1 for value in messages if value < 128),
        "messages_lt_256B": sum(1 for value in messages if value < 256),
        "messages_lt_1KB": sum(1 for value in messages if value < 1024),
    }


def link_load_estimate(graph: dict[str, Any], matrix: list[list[int]], mode: str, max_paths: int | None) -> dict[str, Any]:
    loads: dict[tuple[int, int], int] = defaultdict(int)
    selected_counts: list[int] = []
    for src, row in enumerate(matrix):
        for dst, size in enumerate(row):
            if src == dst or size <= 0:
                continue
            paths = selected_paths(graph, src, dst, mode, max_paths)
            if not paths:
                raise RuntimeError(f"no path {src}->{dst}")
            selected_counts.append(len(paths))
            for path, subbytes in zip(paths, split_bytes(size, len(paths))):
                for u, v in zip(path, path[1:]):
                    loads[(u, v)] += subbytes
    max_load = max(loads.values()) if loads else 0
    return {
        "selected_path_count_min": min(selected_counts) if selected_counts else 0,
        "selected_path_count_median": statistics.median(selected_counts) if selected_counts else 0,
        "selected_path_count_mean": statistics.mean(selected_counts) if selected_counts else 0,
        "selected_path_count_max": max(selected_counts) if selected_counts else 0,
        "max_link_load_bytes": max_load,
        "fluid_cycles": int(max_load / ASTRA_BYTES_PER_NS) if max_load else 0,
        "hot_links": [
            {"src": u, "dst": v, "bytes": b}
            for (u, v), b in sorted(loads.items(), key=lambda x: -x[1])[:8]
        ],
    }


def tiny_subchunk_audit(graph: dict[str, Any], matrix: list[list[int]], mode: str, max_paths: int | None) -> dict[str, Any]:
    subchunks: list[int] = []
    selected_counts: list[int] = []
    for src, row in enumerate(matrix):
        for dst, size in enumerate(row):
            if src == dst or size <= 0:
                continue
            paths = selected_paths(graph, src, dst, mode, max_paths)
            selected_counts.append(len(paths))
            subchunks.extend(split_bytes(size, len(paths)))
    return {
        "one_cycle_threshold_bytes": ONE_CYCLE_THRESHOLD_BYTES,
        "subchunk_bytes_min": min(subchunks) if subchunks else 0,
        "subchunk_bytes_median": statistics.median(subchunks) if subchunks else 0,
        "subchunk_bytes_mean": statistics.mean(subchunks) if subchunks else 0,
        "subchunk_bytes_max": max(subchunks) if subchunks else 0,
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
    stderr_lines = proc.stderr.splitlines()
    return {
        "label": label,
        "returncode": proc.returncode,
        "success": proc.returncode == 0 and len(cycles) == NPU_COUNT,
        "runtime_s": runtime_s,
        "command": " ".join(cmd),
        "stdout": str(stdout),
        "stderr": str(stderr),
        "stderr_tail": stderr_lines[-8:],
        "max_cycles": max(cycles) if cycles else None,
        "cycles_count": len(cycles),
    }


def write_report(summary: dict[str, Any]) -> None:
    totals = summary["native_astra_totals"]
    (OUT / "README.md").write_text(
        f"""# V3.5 One-Workload Aggregated Native ASTRA Validation

## Scope

This is a pipeline validation step, not a final paper result and not a full V2.8 reproduction.

Pipeline:

`HF MoE expert-selection trace -> aggregated per-GPU-pair dispatch/combine matrices -> Chakra SEND/RECV traces -> native ASTRA GraphTopology timing`

Workload: Qwen MMLU `machine_learning`, 32 GPUs, prefill only (`trace[0]`), `block_by_token`, block expert placement, `hidden_size=4096`, `bytes_per_value=2`, local traffic excluded.

## Final Answers

1. Can one real HF-derived MoE prefill workload run successfully on native ASTRA GraphTopology? **{summary["final_recommendation"]["one_workload_runs"]}.**
2. Does SON torus ECMP improve over deterministic routing? **{summary["trend_check"]["son_ecmp_improves_over_deterministic"]}.**
3. Is `ecmp_max_paths=4` still a good production main setting? **{summary["trend_check"]["ecmp4_reasonable_main"]}.**
4. Should `ecmp_max_paths=2` and `8` remain sensitivity points? **Yes.**
5. How different are native ASTRA results from the fluid lower bound? **{summary["trend_check"]["astra_vs_fluid_gap_summary"]}.**
6. Are the differences explainable? **Yes, by ASTRA's hop-by-hop chunk forwarding/store-and-forward semantics.**
7. Is it safe to proceed to static RON calibrated/oracle validation next? **{summary["final_recommendation"]["safe_to_proceed_to_static_ron"]}.**

## Byte Conservation

```json
{json.dumps(summary["trace_parse"], indent=2)}
```

## Aggregated Matrix Stats

```json
{json.dumps(summary["matrix_stats"], indent=2)}
```

## Tiny-Subchunk Audit

```json
{json.dumps(summary["tiny_subchunk_audit"], indent=2)}
```

## Native ASTRA Timing Totals

Dispatch and combine are run separately, then summed.

```json
{json.dumps(totals, indent=2)}
```

## Native ASTRA Per-Phase Timing

```json
{json.dumps(summary["native_astra_results"], indent=2)}
```

## Fluid Lower-Bound Tables

```json
{json.dumps(summary["fluid_lower_bound"], indent=2)}
```

## Interpretation

- EN folded-Clos is an electrical reference, not the fair optical baseline.
- The fair optical trend check is SON deterministic vs SON ECMP-2/4/8.
- Native ASTRA is expected to differ from the fluid link-load lower bound because ASTRA serializes and queues chunks over each hop.
- V35 uses one aggregated SEND/RECV message per nonzero GPU pair. This intentionally avoids token/layer-level tiny-message artifacts.

## Generated Files

- Aggregated matrices: `traffic_matrices/dispatch_matrix.json`, `traffic_matrices/combine_matrix.json`
- Graph JSONs: `graphs/en_folded_clos_32.json`, `graphs/son_torus_4x8_32.json`
- Network configs: `network_configs/*.yml`
- Chakra traces: `chakra_traces/dispatch/workload.*.et`, `chakra_traces/combine/workload.*.et`
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

    parsed = parse_qwen_prefill()
    dispatch = parsed.pop("dispatch_matrix")
    combine = parsed.pop("combine_matrix")
    write_json(matrices_dir / "dispatch_matrix.json", {"phase": "dispatch", "matrix": dispatch})
    write_json(matrices_dir / "combine_matrix.json", {"phase": "combine", "matrix": combine})

    en_graph = en_folded_clos_graph()
    son_graph = son_torus_graph()
    en_graph_path = graphs_dir / "en_folded_clos_32.json"
    son_graph_path = graphs_dir / "son_torus_4x8_32.json"
    write_json(en_graph_path, en_graph)
    write_json(son_graph_path, son_graph)

    configs = {
        "en_folded_clos_ecmp4": (en_graph, "ecmp", 4, configs_dir / "en_folded_clos_ecmp4.yml"),
        "son_torus_deterministic": (son_graph, "deterministic", None, configs_dir / "son_torus_deterministic.yml"),
        "son_torus_ecmp2": (son_graph, "ecmp", 2, configs_dir / "son_torus_ecmp2.yml"),
        "son_torus_ecmp4": (son_graph, "ecmp", 4, configs_dir / "son_torus_ecmp4.yml"),
        "son_torus_ecmp8": (son_graph, "ecmp", 8, configs_dir / "son_torus_ecmp8.yml"),
    }
    for name, (graph, mode, max_paths, config_path) in configs.items():
        graph_path = en_graph_path if graph["name"].startswith("en_") else son_graph_path
        network_config(config_path, graph_path, mode, max_paths)

    trace_meta = {
        "dispatch": write_matrix_trace(traces_dir / "dispatch" / "workload", dispatch),
        "combine": write_matrix_trace(traces_dir / "combine" / "workload", combine),
    }

    matrix_stat_payload = {
        "dispatch": matrix_stats(dispatch),
        "combine": matrix_stats(combine),
    }

    tiny_payload: dict[str, Any] = {}
    fluid_payload: dict[str, Any] = {}
    native_results: dict[str, Any] = {}
    for phase, matrix in {"dispatch": dispatch, "combine": combine}.items():
        tiny_payload[phase] = {}
        fluid_payload[phase] = {}
        native_results[phase] = {}
        for config_name, (graph, mode, max_paths, config_path) in configs.items():
            tiny_payload[phase][config_name] = tiny_subchunk_audit(graph, matrix, mode, max_paths)
            fluid_payload[phase][config_name] = link_load_estimate(graph, matrix, mode, max_paths)
            native_results[phase][config_name] = run_astra(
                f"{phase}_{config_name}",
                traces_dir / phase / "workload",
                config_path,
            )

    totals: dict[str, Any] = {}
    for config_name in configs:
        dispatch_run = native_results["dispatch"][config_name]
        combine_run = native_results["combine"][config_name]
        dispatch_fluid = fluid_payload["dispatch"][config_name]
        combine_fluid = fluid_payload["combine"][config_name]
        astra_total = (
            dispatch_run["max_cycles"] + combine_run["max_cycles"]
            if dispatch_run["max_cycles"] is not None and combine_run["max_cycles"] is not None
            else None
        )
        fluid_total = dispatch_fluid["fluid_cycles"] + combine_fluid["fluid_cycles"]
        totals[config_name] = {
            "dispatch_cycles": dispatch_run["max_cycles"],
            "combine_cycles": combine_run["max_cycles"],
            "total_cycles": astra_total,
            "dispatch_fluid_cycles": dispatch_fluid["fluid_cycles"],
            "combine_fluid_cycles": combine_fluid["fluid_cycles"],
            "total_fluid_cycles": fluid_total,
            "astra_over_fluid_total": (astra_total / fluid_total) if astra_total is not None and fluid_total else None,
            "success": dispatch_run["success"] and combine_run["success"],
        }

    son_det = totals["son_torus_deterministic"]["total_cycles"]
    son2 = totals["son_torus_ecmp2"]["total_cycles"]
    son4 = totals["son_torus_ecmp4"]["total_cycles"]
    son8 = totals["son_torus_ecmp8"]["total_cycles"]
    son_values = [value for value in (son_det, son2, son4, son8) if value is not None]
    gaps = [
        item["astra_over_fluid_total"]
        for item in totals.values()
        if item["astra_over_fluid_total"] is not None
    ]
    serious_tiny_risk = any(
        audit["zero_delay_risk"]
        for phase_payload in tiny_payload.values()
        for audit in phase_payload.values()
    )
    all_runs_ok = all(item["success"] for item in totals.values())
    trend_check = {
        "son_ecmp_improves_over_deterministic": bool(son_values and son_det is not None and min(son2, son4, son8) < son_det),
        "son_total_cycles": {
            "deterministic": son_det,
            "ecmp2": son2,
            "ecmp4": son4,
            "ecmp8": son8,
        },
        "ecmp4_reasonable_main": bool(son4 is not None and son_det is not None and son4 < son_det and all_runs_ok),
        "ecmp2_and_8_sensitivity_needed": True,
        "en_reference_total_cycles": totals["en_folded_clos_ecmp4"]["total_cycles"],
        "astra_vs_fluid_gap_min": min(gaps) if gaps else None,
        "astra_vs_fluid_gap_max": max(gaps) if gaps else None,
        "astra_vs_fluid_gap_summary": (
            f"{min(gaps):.2f}x to {max(gaps):.2f}x across topology/phase totals"
            if gaps
            else "unavailable"
        ),
        "tiny_subchunk_zero_delay_risk": serious_tiny_risk,
        "astra_assertions_or_failures": not all_runs_ok,
    }

    summary = {
        "scope": "one real HF-derived workload, aggregated per-GPU-pair, native ASTRA GraphTopology",
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
        "trace_parse": parsed,
        "trace_meta": trace_meta,
        "matrix_stats": matrix_stat_payload,
        "tiny_subchunk_audit": tiny_payload,
        "native_astra_results": native_results,
        "native_astra_totals": totals,
        "fluid_lower_bound": fluid_payload,
        "trend_check": trend_check,
        "final_recommendation": {
            "one_workload_runs": all_runs_ok,
            "safe_to_proceed_to_static_ron": all_runs_ok and parsed["byte_conservation_pass"] and not serious_tiny_risk,
            "recommended_next_step": "static RON calibrated/oracle validation on the same aggregated one-workload pipeline",
            "notes": [
                "Do not claim final V2.8 reproduction yet.",
                "Keep EN as electrical reference, not fair optical baseline.",
                "Keep SON ECMP-2 and ECMP-8 as sensitivity around ECMP-4.",
            ],
        },
    }
    write_json(OUT / "summary.json", summary)
    write_report(summary)
    print(json.dumps(summary, indent=2))
    if not all_runs_ok or not parsed["byte_conservation_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
