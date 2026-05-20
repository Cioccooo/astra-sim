#!/usr/bin/env python3
"""Generate and run native GraphTopology smoke tests.

This script intentionally keeps the workload tiny. Its job is to verify that
the congestion-aware analytical backend can load an arbitrary graph topology,
route pairwise Chakra SEND/RECV traffic through a deterministic shortest path,
and still run the existing built-in topology blocks.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

# The local tree currently builds Chakra protobufs with protoc 3.6.1 to match
# the C++ protobuf headers available on this macOS machine. Newer Python
# protobuf runtimes need the pure-Python implementation for these generated
# bindings.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/moe_expert_trace_converter/v31_native_graph_topology_smoke"
PROTO_DIR = REPO / "extern/graph_frontend/chakra/schema/protobuf"
PROTO_UTILS = REPO / "extern/graph_frontend/chakra/src/third_party/utils"

sys.path.insert(0, str(PROTO_DIR))
sys.path.insert(0, str(PROTO_UTILS))

from et_def_pb2 import AttributeProto, GlobalMetadata, Node, NodeType  # type: ignore  # noqa: E402
from protolib import encodeMessage as encode_message  # type: ignore  # noqa: E402


ASTRA_BIN = REPO / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
SYSTEM = REPO / "examples/system/native_collectives/Ring_4chunks.json"
REMOTE_MEMORY = REPO / "examples/remote_memory/analytical/no_memory_expansion.json"


def add_attr(node: Node, name: str, value: int | bool) -> None:
    if isinstance(value, bool):
        node.attr.append(AttributeProto(name=name, bool_val=value))
    else:
        node.attr.append(AttributeProto(name=name, uint64_val=int(value)))


def write_pair_trace(prefix: Path, rank_count: int, src: int, dst: int, size_bytes: int) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    nodes_by_rank: dict[int, list[Node]] = {rank: [] for rank in range(rank_count)}

    send = Node()
    send.id = 1
    send.name = f"SEND_{src}_to_{dst}"
    send.type = NodeType.COMM_SEND_NODE
    add_attr(send, "is_cpu_op", False)
    add_attr(send, "comm_src", src)
    add_attr(send, "comm_dst", dst)
    add_attr(send, "comm_size", size_bytes)
    add_attr(send, "comm_tag", 1)
    nodes_by_rank[src].append(send)

    recv = Node()
    recv.id = 1
    recv.name = f"RECV_{src}_to_{dst}"
    recv.type = NodeType.COMM_RECV_NODE
    add_attr(recv, "is_cpu_op", False)
    add_attr(recv, "comm_src", src)
    add_attr(recv, "comm_dst", dst)
    add_attr(recv, "comm_size", size_bytes)
    add_attr(recv, "comm_tag", 1)
    nodes_by_rank[dst].append(recv)

    for rank in range(rank_count):
        if not nodes_by_rank[rank]:
            dummy = Node()
            dummy.id = 1
            dummy.name = "DUMMY_COMP_NODE"
            dummy.type = NodeType.COMP_NODE
            dummy.duration_micros = 0
            add_attr(dummy, "is_cpu_op", False)
            nodes_by_rank[rank].append(dummy)
        with (prefix.parent / f"{prefix.name}.{rank}.et").open("wb") as handle:
            encode_message(handle, GlobalMetadata(version="0.0.4"))
            for node in nodes_by_rank[rank]:
                encode_message(handle, node)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_network_yaml(path: Path, topology: str, npus: int, graph_file: Path | None = None) -> None:
    lines = [
        f"topology: [ {topology} ]",
        f"npus_count: [ {npus} ]",
        "bandwidth: [ 50.0 ]  # GB/s; Graph edges use bandwidth_gbps and convert /8 internally",
        "latency: [ 0.0 ]  # ns; Graph can override this per edge",
    ]
    if graph_file is not None:
        lines.append(f"graph_file: {graph_file}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def line4_graph(path: Path) -> None:
    write_json(
        path,
        {
            "name": "line4_400gbps",
            "node_count": 4,
            "gpu_count": 4,
            "directed": False,
            "edges": [
                {"src": 0, "dst": 1, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 1, "dst": 2, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 2, "dst": 3, "bandwidth_gbps": 400, "latency_ns": 0},
            ],
        },
    )


def torus4x8_graph(path: Path) -> None:
    rows, cols = 4, 8
    edges: list[dict[str, Any]] = []

    def node(row: int, col: int) -> int:
        return row * cols + col

    for row in range(rows):
        for col in range(cols):
            src = node(row, col)
            right = node(row, (col + 1) % cols)
            down = node((row + 1) % rows, col)
            if src < right:
                edges.append({"src": src, "dst": right, "bandwidth_gbps": 400, "latency_ns": 0})
            if src < down:
                edges.append({"src": src, "dst": down, "bandwidth_gbps": 400, "latency_ns": 0})
            # Wrap edges have right/down node IDs lower than src, so add them explicitly once.
            if col == cols - 1:
                edges.append({"src": src, "dst": right, "bandwidth_gbps": 400, "latency_ns": 0})
            if row == rows - 1:
                edges.append({"src": src, "dst": down, "bandwidth_gbps": 400, "latency_ns": 0})
    write_json(
        path,
        {
            "name": "son_2d_torus_4x8_400gbps",
            "node_count": 32,
            "gpu_count": 32,
            "directed": False,
            "edges": edges,
        },
    )


def graph_with_switch_nodes(path: Path) -> None:
    write_json(
        path,
        {
            "name": "four_gpu_two_switch_nodes",
            "node_count": 6,
            "gpu_count": 4,
            "rank_nodes": [0, 1, 2, 3],
            "directed": False,
            "edges": [
                {"src": 0, "dst": 4, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 1, "dst": 4, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 2, "dst": 5, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 3, "dst": 5, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 4, "dst": 5, "bandwidth_gbps": 400, "latency_ns": 0},
            ],
        },
    )


def shortest_path_hops(graph: dict[str, Any], src: int, dst: int) -> int:
    adjacency: list[list[int]] = [[] for _ in range(int(graph["node_count"]))]
    directed = bool(graph.get("directed", False))
    rank_nodes = graph.get("rank_nodes") or list(range(int(graph["gpu_count"])))
    src_node = int(rank_nodes[src])
    dst_node = int(rank_nodes[dst])
    for edge in graph["edges"]:
        u, v = int(edge["src"]), int(edge["dst"])
        bidirectional = bool(edge.get("bidirectional", not directed))
        adjacency[u].append(v)
        if bidirectional:
            adjacency[v].append(u)
    for neighbors in adjacency:
        neighbors.sort()

    queue: deque[tuple[int, int]] = deque([(src_node, 0)])
    seen = {src_node}
    while queue:
        node, depth = queue.popleft()
        if node == dst_node:
            return depth
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, depth + 1))
    raise ValueError(f"no route {src}->{dst}")


def run_astra(label: str, workload_prefix: Path, network: Path) -> dict[str, Any]:
    stdout_path = OUT / "runs" / f"{label}.stdout.txt"
    stderr_path = OUT / "runs" / f"{label}.stderr.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ASTRA_BIN),
        f"--workload-configuration={workload_prefix}",
        f"--system-configuration={SYSTEM}",
        f"--remote-memory-configuration={REMOTE_MEMORY}",
        f"--network-configuration={network}",
    ]
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, check=False)
    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)
    cycles = [int(match) for match in re.findall(r"finished, ([0-9]+) cycles", proc.stdout)]
    return {
        "label": label,
        "returncode": proc.returncode,
        "command": " ".join(cmd),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "cycles": cycles,
        "max_cycles": max(cycles) if cycles else None,
    }


def write_readme(summary: dict[str, Any]) -> None:
    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# V3.1 Native GraphTopology Smoke Test",
                "",
                "This folder validates the first native ASTRA-sim integration step: deterministic arbitrary graph topology support inside the analytical congestion-aware backend.",
                "",
                "## What This Claims",
                "",
                "- Native congestion-aware analytical backend can load a `Graph` topology from JSON/YAML.",
                "- Graph edges support `bandwidth_gbps` and optional `latency_ns`; bandwidth is converted to ASTRA internal GB/s by dividing by 8.",
                "- Pairwise Chakra `COMM_SEND_NODE` / `COMM_RECV_NODE` traffic can route over one deterministic shortest path.",
                "- Existing `Ring`, `Switch`, and `FullyConnected` topology names still run.",
                "",
                "## What This Does Not Claim",
                "",
                "- No ECMP splitting yet.",
                "- No in-run RON W=4 topology swap yet.",
                "- No full EN/SON/RON reproduction yet.",
                "- Multi-hop paths are analytical graph paths through ASTRA `Device`/`Link` objects, not a device-level optical physics model.",
                "",
                "## Build Command",
                "",
                "```bash",
                "./build/astra_analytical/build.sh -t congestion_aware",
                "```",
                "",
                "Note: this local macOS tree has `/usr/local` protobuf 3.6.1, so Chakra protobuf C++/Python bindings were regenerated with that version for build consistency.",
                "",
                "## Smoke Results",
                "",
                "```json",
                json.dumps(summary, indent=2),
                "```",
                "",
                "## Code Files Changed",
                "",
                "- `extern/network_backend/analytical/include/astra-network-analytical/common/Type.h` adds `TopologyBuildingBlock::Graph`.",
                "- `extern/network_backend/analytical/include/astra-network-analytical/common/NetworkParser.h` stores and exposes `graph_file`.",
                "- `extern/network_backend/analytical/common/network-parser/NetworkParser.cpp` parses `Graph` and resolves relative `graph_file` paths.",
                "- `extern/network_backend/analytical/include/astra-network-analytical/congestion_aware/GraphTopology.h` declares native graph topology support.",
                "- `extern/network_backend/analytical/congestion_aware/basic-topology/GraphTopology.cpp` loads JSON/YAML graphs, creates `Device`/`Link`s, validates rank mappings, and precomputes deterministic BFS shortest paths.",
                "- `extern/network_backend/analytical/congestion_aware/topology/Helper.cpp` constructs `GraphTopology` for congestion-aware analytical runs.",
                "- `tools/v31_native_graph_topology_smoke.py` generates and runs this smoke test.",
                "",
                "## Manual Timing Sanity Check",
                "",
                "For a single non-contended message of 5000 bytes over 400 Gb/s links:",
                "",
                "`400 Gb/s = 50 GB/s = 50 bytes/ns`, so serialization is `5000 / 50 = 100 ns` per hop.",
                "",
                "The 4-node line test routes rank 0 to rank 3 over 3 hops, so the network-only lower-bound is about `300 ns` plus ASTRA endpoint/scheduling overhead.",
                "",
                "The 4x8 torus test records the deterministic BFS hop count in `summary.json` and uses the same per-hop serialization estimate.",
                "",
                "## ECMP Blockers",
                "",
                "- `GraphTopology::route(src, dst)` returns one `Route`; there is no API yet to return multiple equal-cost paths.",
                "- `CongestionAwareNetworkApi::sim_send` creates one `Chunk`, so splitting one logical message across multiple paths needs API/workload changes or chunkization before routing.",
                "- A fair ECMP model also needs deterministic path hashing or byte-splitting semantics and validation against folded-Clos / torus expectations.",
                "",
                "## RON Segmented-Run Blockers",
                "",
                "- In-run topology mutation is not supported yet; topology is built once by `NetworkParser`/`Helper` at simulator startup.",
                "- The safe next step is one ASTRA run per request/request group with a selected Graph JSON, then external aggregation plus reconfiguration penalty.",
                "- Native in-run swapping would require a topology reload API and careful draining semantics for in-flight `Chunk`s.",
                "",
            ]
        )
        + "\n"
    )


def main() -> None:
    if not ASTRA_BIN.exists():
        raise SystemExit(f"missing ASTRA binary: {ASTRA_BIN}")

    graphs = OUT / "graphs"
    configs = OUT / "network_configs"
    traces = OUT / "chakra_traces"

    line_graph_path = graphs / "line4.json"
    torus_graph_path = graphs / "torus4x8.json"
    switch_nodes_graph_path = graphs / "four_gpu_two_switch_nodes.json"
    line4_graph(line_graph_path)
    torus4x8_graph(torus_graph_path)
    graph_with_switch_nodes(switch_nodes_graph_path)

    write_network_yaml(configs / "graph_line4.yml", "Graph", 4, line_graph_path)
    write_network_yaml(configs / "graph_torus4x8.yml", "Graph", 32, torus_graph_path)
    write_network_yaml(configs / "graph_non_gpu_switch_nodes.yml", "Graph", 4, switch_nodes_graph_path)
    write_network_yaml(configs / "ring4.yml", "Ring", 4)
    write_network_yaml(configs / "switch4.yml", "Switch", 4)
    write_network_yaml(configs / "fully_connected4.yml", "FullyConnected", 4)

    msg_bytes = 5000
    write_pair_trace(traces / "line4_0_to_3" / "workload", 4, 0, 3, msg_bytes)
    write_pair_trace(traces / "torus32_0_to_31" / "workload", 32, 0, 31, msg_bytes)
    write_pair_trace(traces / "basic4_0_to_3" / "workload", 4, 0, 3, msg_bytes)

    line_graph = json.loads(line_graph_path.read_text())
    torus_graph = json.loads(torus_graph_path.read_text())
    switch_nodes_graph = json.loads(switch_nodes_graph_path.read_text())
    line_hops = shortest_path_hops(line_graph, 0, 3)
    torus_hops = shortest_path_hops(torus_graph, 0, 31)
    switch_nodes_hops = shortest_path_hops(switch_nodes_graph, 0, 3)
    per_hop_ns = msg_bytes / (400 / 8)

    runs = [
        run_astra("graph_line4", traces / "line4_0_to_3" / "workload", configs / "graph_line4.yml"),
        run_astra("graph_torus4x8", traces / "torus32_0_to_31" / "workload", configs / "graph_torus4x8.yml"),
        run_astra(
            "graph_non_gpu_switch_nodes",
            traces / "basic4_0_to_3" / "workload",
            configs / "graph_non_gpu_switch_nodes.yml",
        ),
        run_astra("existing_ring4", traces / "basic4_0_to_3" / "workload", configs / "ring4.yml"),
        run_astra("existing_switch4", traces / "basic4_0_to_3" / "workload", configs / "switch4.yml"),
        run_astra(
            "existing_fully_connected4",
            traces / "basic4_0_to_3" / "workload",
            configs / "fully_connected4.yml",
        ),
    ]
    all_passed = all(run["returncode"] == 0 and run["cycles"] for run in runs)
    summary = {
        "all_passed": all_passed,
        "message_bytes": msg_bytes,
        "bandwidth_gbps": 400,
        "per_hop_serialization_ns": per_hop_ns,
        "manual_expectations": {
            "line4_0_to_3_hops": line_hops,
            "line4_network_only_ns": line_hops * per_hop_ns,
            "torus4x8_0_to_31_hops": torus_hops,
            "torus4x8_network_only_ns": torus_hops * per_hop_ns,
            "non_gpu_switch_nodes_0_to_3_hops": switch_nodes_hops,
            "non_gpu_switch_nodes_network_only_ns": switch_nodes_hops * per_hop_ns,
        },
        "runs": runs,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_readme(summary)

    print(json.dumps(summary, indent=2))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
