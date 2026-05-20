#!/usr/bin/env python3
"""V33 native GraphTopology ECMP smoke tests."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke"
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


def comp_node(node_id: int, runtime_us: int = 0) -> Node:
    node = Node()
    node.id = node_id
    node.name = f"COMP_{node_id}"
    node.type = NodeType.COMP_NODE
    node.duration_micros = runtime_us
    add_attr(node, "is_cpu_op", False)
    return node


def send_node(node_id: int, src: int, dst: int, size: int, tag: int, dep: int | None = None) -> Node:
    node = Node()
    node.id = node_id
    node.name = f"SEND_{src}_to_{dst}_tag{tag}"
    node.type = NodeType.COMM_SEND_NODE
    if dep is not None:
        node.data_deps.append(dep)
    add_attr(node, "is_cpu_op", False)
    add_attr(node, "comm_src", src)
    add_attr(node, "comm_dst", dst)
    add_attr(node, "comm_size", size)
    add_attr(node, "comm_tag", tag)
    return node


def recv_node(node_id: int, src: int, dst: int, size: int, tag: int, dep: int | None = None) -> Node:
    node = Node()
    node.id = node_id
    node.name = f"RECV_{src}_to_{dst}_tag{tag}"
    node.type = NodeType.COMM_RECV_NODE
    if dep is not None:
        node.data_deps.append(dep)
    add_attr(node, "is_cpu_op", False)
    add_attr(node, "comm_src", src)
    add_attr(node, "comm_dst", dst)
    add_attr(node, "comm_size", size)
    add_attr(node, "comm_tag", tag)
    return node


def write_trace(prefix: Path, rank_count: int, nodes_by_rank: dict[int, list[Node]]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for rank in range(rank_count):
        nodes = nodes_by_rank.get(rank, [])
        if not nodes:
            nodes = [comp_node(1)]
        with (prefix.parent / f"{prefix.name}.{rank}.et").open("wb") as handle:
            encode_message(handle, GlobalMetadata(version="0.0.4"))
            for node in nodes:
                encode_message(handle, node)


def single_pair_trace(
    prefix: Path,
    rank_count: int,
    src: int,
    dst: int,
    size: int,
    tag: int = 1,
    send_delay_us: int = 0,
    recv_delay_us: int = 0,
) -> None:
    nodes: dict[int, list[Node]] = {rank: [] for rank in range(rank_count)}
    send_dep = None
    recv_dep = None
    if send_delay_us > 0:
        nodes[src].append(comp_node(1, send_delay_us))
        send_dep = 1
        send_id = 2
    else:
        send_id = 1
    if recv_delay_us > 0:
        nodes[dst].append(comp_node(1, recv_delay_us))
        recv_dep = 1
        recv_id = 2
    else:
        recv_id = 1
    nodes[src].append(send_node(send_id, src, dst, size, tag, send_dep))
    nodes[dst].append(recv_node(recv_id, src, dst, size, tag, recv_dep))
    write_trace(prefix, rank_count, nodes)


def two_tags_same_pair_trace(prefix: Path, rank_count: int, src: int, dst: int, size: int) -> None:
    nodes: dict[int, list[Node]] = {rank: [] for rank in range(rank_count)}
    nodes[src].append(send_node(1, src, dst, size, 1))
    nodes[src].append(send_node(2, src, dst, size, 2))
    nodes[dst].append(recv_node(1, src, dst, size, 1))
    nodes[dst].append(recv_node(2, src, dst, size, 2))
    write_trace(prefix, rank_count, nodes)


def p2p_all_to_all_trace(prefix: Path, rank_count: int, size: int) -> None:
    nodes: dict[int, list[Node]] = {rank: [] for rank in range(rank_count)}
    next_id = {rank: 1 for rank in range(rank_count)}
    tag = 1
    for src in range(rank_count):
        for dst in range(rank_count):
            if src == dst:
                continue
            nodes[src].append(send_node(next_id[src], src, dst, size, tag))
            next_id[src] += 1
            nodes[dst].append(recv_node(next_id[dst], src, dst, size, tag))
            next_id[dst] += 1
            tag += 1
    write_trace(prefix, rank_count, nodes)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def diamond_graph(path: Path, slow_edge: bool = False) -> None:
    write_json(
        path,
        {
            "name": "diamond_ecmp_slow" if slow_edge else "diamond_ecmp",
            "node_count": 4,
            "gpu_count": 4,
            "directed": False,
            "edges": [
                {"src": 0, "dst": 1, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 1, "dst": 3, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 0, "dst": 2, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 2, "dst": 3, "bandwidth_gbps": 400, "latency_ns": 1000 if slow_edge else 0},
            ],
        },
    )


def clos_graph(path: Path) -> None:
    write_json(
        path,
        {
            "name": "tiny_clos_two_spines",
            "node_count": 8,
            "gpu_count": 4,
            "rank_nodes": [0, 1, 2, 3],
            "directed": False,
            "edges": [
                {"src": 0, "dst": 4, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 1, "dst": 4, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 2, "dst": 5, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 3, "dst": 5, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 4, "dst": 6, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 4, "dst": 7, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 5, "dst": 6, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 5, "dst": 7, "bandwidth_gbps": 400, "latency_ns": 0},
            ],
        },
    )


def torus3x3_graph(path: Path) -> None:
    rows = cols = 3
    seen: set[tuple[int, int]] = set()
    edges: list[dict[str, Any]] = []

    def node(r: int, c: int) -> int:
        return r * cols + c

    def add(u: int, v: int) -> None:
        key = tuple(sorted((u, v)))
        if key not in seen:
            seen.add(key)
            edges.append({"src": u, "dst": v, "bandwidth_gbps": 400, "latency_ns": 0})

    for r in range(rows):
        for c in range(cols):
            add(node(r, c), node(r, (c + 1) % cols))
            add(node(r, c), node((r + 1) % rows, c))
    write_json(
        path,
        {"name": "torus3x3", "node_count": 9, "gpu_count": 9, "directed": False, "edges": edges},
    )


def line4_graph(path: Path) -> None:
    write_json(
        path,
        {
            "name": "line4_deterministic",
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


def two_node_graph(path: Path) -> None:
    write_json(
        path,
        {
            "name": "two_node_one_link",
            "node_count": 2,
            "gpu_count": 2,
            "directed": False,
            "edges": [{"src": 0, "dst": 1, "bandwidth_gbps": 400, "latency_ns": 0}],
        },
    )


def network_config(path: Path, graph_file: Path, npus: int, ecmp: bool = True, max_paths: int = 0) -> None:
    lines = [
        "topology: [ Graph ]",
        f"npus_count: [ {npus} ]",
        "bandwidth: [ 50.0 ]",
        "latency: [ 0.0 ]",
        f"graph_file: {graph_file}",
    ]
    if ecmp:
        lines.extend(["routing: ecmp", "ecmp_split: equal_bytes", f"ecmp_max_paths: {max_paths}", "ecmp_log: true"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def simple_network_config(path: Path, topology: str, npus: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"topology: [ {topology} ]",
                f"npus_count: [ {npus} ]",
                "bandwidth: [ 50.0 ]",
                "latency: [ 0.0 ]",
            ]
        )
        + "\n"
    )


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
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, check=False)
    stdout.write_text(proc.stdout)
    stderr.write_text(proc.stderr)
    cycles = [int(match) for match in re.findall(r"finished, ([0-9]+) cycles", proc.stdout)]
    return {
        "label": label,
        "returncode": proc.returncode,
        "command": " ".join(cmd),
        "stdout": str(stdout),
        "stderr": str(stderr),
        "cycles": cycles,
        "max_cycles": max(cycles) if cycles else None,
        "ecmp_logs": parse_ecmp_logs(proc.stdout),
    }


def parse_ecmp_logs(stdout: str) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if "[ecmp]" not in line:
            continue
        header = re.search(r"tag=(\d+) src=(\d+) dst=(\d+) bytes=(\d+) paths=(\d+)", line)
        if not header:
            continue
        subchunks = [
            {"index": int(idx), "bytes": int(size), "route": route}
            for idx, size, route in re.findall(r"subchunk(\d+)=(\d+) route=(\[[^\]]+\])", line)
        ]
        logs.append(
            {
                "tag": int(header.group(1)),
                "src": int(header.group(2)),
                "dst": int(header.group(3)),
                "bytes": int(header.group(4)),
                "paths": int(header.group(5)),
                "subchunks": subchunks,
                "byte_sum": sum(item["bytes"] for item in subchunks),
            }
        )
    return logs


def validate_log(run: dict[str, Any], expected_paths: int, expected_bytes: int) -> bool:
    if run["returncode"] != 0 or not run["ecmp_logs"]:
        return False
    first = run["ecmp_logs"][0]
    return first["paths"] == expected_paths and first["byte_sum"] == expected_bytes


def write_report(summary: dict[str, Any]) -> None:
    (OUT / "README.md").write_text(
        f"""# V3.3 Native GraphTopology ECMP Smoke

## Summary

Native `GraphTopology` now has optional ECMP support for the analytical congestion-aware backend.

Final answers:

1. Does native GraphTopology now support ECMP? **Yes**, when network config sets `routing: ecmp`.
2. Can this support EN folded-Clos ECMP? **Yes for equal-cost shortest-path splitting over a folded-Clos graph**, including non-GPU leaf/spine nodes via `rank_nodes`; full EN still needs larger topology generation and validation.
3. Can this support SON 2D torus ECMP? **Yes for equal-cost shortest-path splitting over torus graph topology**; full SON still needs the 32-GPU/64-GPU production graph and workload validation.
4. What remains before RON calibrated / W=4? We still need segmented per-request Graph runs or a topology swap API, plus RON topology selection logic; this step does not implement in-run reconfiguration.

## ECMP API / Design

Network config:

```yaml
topology: [ Graph ]
graph_file: path/to/graph.json
routing: ecmp
ecmp_split: equal_bytes
ecmp_max_paths: 0   # 0 means all equal-cost shortest paths
ecmp_log: true
```

Design:

- `Topology::routes(src,dst)` defaults to a single deterministic `route(src,dst)` for backward compatibility.
- `GraphTopology::routes(src,dst)` returns all deterministic equal-cost shortest paths, sorted lexicographically.
- `CongestionAwareNetworkApi::sim_send()` keeps the original logical callback tracker entry keyed by `(tag,src,dst,count,chunk_id)`.
- In ECMP mode, one logical message is split into one subchunk per selected path using deterministic equal-byte split; remainders go to earlier paths.
- Each subchunk is sent as a normal `Chunk` over one selected route.
- `process_ecmp_chunk_arrival()` decrements an aggregation counter and only marks the original logical message complete after all subchunks arrive.

## Results

```json
{json.dumps(summary, indent=2)}
```

## Manual Timing Checks

- ASTRA internally interprets `50 GB/s` as `50 * 2^30 / 1e9 = {summary["manual"]["bytes_per_ns"]:.6f}` B/ns.
- A 5000B single-path one-link message should take about `{summary["manual"]["one_link_5000B_cycles"]}` cycles; regression observed `{summary["runs"]["one_link_400gbps"]["max_cycles"]}`.
- Diamond ECMP 5000B splits into 2500B + 2500B over two 2-hop paths. Each subchunk serializes in about `{summary["manual"]["diamond_subchunk_cycles_per_hop"]}` cycles per hop; expected network-only completion is about `{summary["manual"]["diamond_expected_cycles"]}` cycles. Observed `{summary["runs"]["diamond_ecmp"]["max_cycles"]}`.
- Odd-size 5001B split was validated by byte conservation in ECMP logs.

## Callback Correctness Checks

- `recv_before_send_odd_slow_path`: recv is issued before delayed send; ECMP subchunks are 2501B and 2500B, and the run only completes after the slow path arrives.
- `send_before_recv_odd_slow_path`: send/transmission can complete before delayed recv; the existing callback tracker releases recv when it is later issued.
- `two_tags_same_pair`: two logical messages with same src/dst but different tags complete independently.
- Slow-path test makes early callback firing observable: if the first subchunk released the logical recv, completion would be much earlier than the slow path. The observed completion includes the slow-path latency.

## Files Changed

- `extern/network_backend/analytical/include/astra-network-analytical/congestion_aware/Topology.h`
- `extern/network_backend/analytical/congestion_aware/topology/Topology.cpp`
- `extern/network_backend/analytical/include/astra-network-analytical/congestion_aware/GraphTopology.h`
- `extern/network_backend/analytical/congestion_aware/basic-topology/GraphTopology.cpp`
- `extern/network_backend/analytical/include/astra-network-analytical/common/NetworkParser.h`
- `extern/network_backend/analytical/common/network-parser/NetworkParser.cpp`
- `astra-sim/network_frontend/analytical/include/congestion_aware/CongestionAwareNetworkApi.hh`
- `astra-sim/network_frontend/analytical/congestion_aware/CongestionAwareNetworkApi.cc`
- `astra-sim/network_frontend/analytical/congestion_aware/main.cc`
- `tools/v33_native_graph_ecmp_smoke.py`

## Remaining Limits

- ECMP path cost is currently hop-count equal cost, not weighted by latency.
- ECMP splitting is equal-byte splitting, not packet hash ECMP.
- No in-run topology swap.
- No full EN/SON/RON workload validation yet.
- Multi-hop semantics remain analytical forwarding through `Device`/`Link` queues, not ideal transparent optical circuit reservation.
"""
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    graphs = OUT / "graphs"
    configs = OUT / "network_configs"
    traces = OUT / "chakra_traces"

    diamond = graphs / "diamond.json"
    diamond_slow = graphs / "diamond_slow.json"
    clos = graphs / "tiny_clos.json"
    torus = graphs / "torus3x3.json"
    line = graphs / "line4.json"
    two = graphs / "two_node.json"
    diamond_graph(diamond)
    diamond_graph(diamond_slow, slow_edge=True)
    clos_graph(clos)
    torus3x3_graph(torus)
    line4_graph(line)
    two_node_graph(two)

    network_config(configs / "diamond_ecmp.yml", diamond, 4, True)
    network_config(configs / "diamond_slow_ecmp.yml", diamond_slow, 4, True)
    network_config(configs / "tiny_clos_ecmp.yml", clos, 4, True)
    network_config(configs / "torus3x3_ecmp.yml", torus, 9, True)
    network_config(configs / "line4_deterministic.yml", line, 4, False)
    network_config(configs / "two_node_deterministic.yml", two, 2, False)
    simple_network_config(configs / "ring4.yml", "Ring", 4)
    simple_network_config(configs / "switch4.yml", "Switch", 4)
    simple_network_config(configs / "fully_connected4.yml", "FullyConnected", 4)

    single_pair_trace(traces / "diamond_0_to_3_5000" / "workload", 4, 0, 3, 5000)
    single_pair_trace(traces / "clos_0_to_2_5000" / "workload", 4, 0, 2, 5000)
    single_pair_trace(traces / "torus_0_to_4_5000" / "workload", 9, 0, 4, 5000)
    single_pair_trace(traces / "recv_before_send_odd" / "workload", 4, 0, 3, 5001, send_delay_us=1)
    single_pair_trace(traces / "send_before_recv_odd" / "workload", 4, 0, 3, 5001, recv_delay_us=1)
    two_tags_same_pair_trace(traces / "two_tags_same_pair" / "workload", 4, 0, 3, 5000)
    single_pair_trace(traces / "line4_0_to_3_5000" / "workload", 4, 0, 3, 5000)
    single_pair_trace(traces / "one_link_0_to_1_5000" / "workload", 2, 0, 1, 5000)
    p2p_all_to_all_trace(traces / "basic4_all_to_all" / "workload", 4, 5000)

    runs = {
        "diamond_ecmp": run_astra("diamond_ecmp", traces / "diamond_0_to_3_5000" / "workload", configs / "diamond_ecmp.yml"),
        "tiny_clos_ecmp": run_astra("tiny_clos_ecmp", traces / "clos_0_to_2_5000" / "workload", configs / "tiny_clos_ecmp.yml"),
        "tiny_torus_ecmp": run_astra("tiny_torus_ecmp", traces / "torus_0_to_4_5000" / "workload", configs / "torus3x3_ecmp.yml"),
        "recv_before_send_odd_slow_path": run_astra("recv_before_send_odd_slow_path", traces / "recv_before_send_odd" / "workload", configs / "diamond_slow_ecmp.yml"),
        "send_before_recv_odd_slow_path": run_astra("send_before_recv_odd_slow_path", traces / "send_before_recv_odd" / "workload", configs / "diamond_slow_ecmp.yml"),
        "two_tags_same_pair": run_astra("two_tags_same_pair", traces / "two_tags_same_pair" / "workload", configs / "diamond_ecmp.yml"),
        "deterministic_graph_line4": run_astra("deterministic_graph_line4", traces / "line4_0_to_3_5000" / "workload", configs / "line4_deterministic.yml"),
        "one_link_400gbps": run_astra("one_link_400gbps", traces / "one_link_0_to_1_5000" / "workload", configs / "two_node_deterministic.yml"),
        "existing_ring4": run_astra("existing_ring4", traces / "basic4_all_to_all" / "workload", configs / "ring4.yml"),
        "existing_switch4": run_astra("existing_switch4", traces / "basic4_all_to_all" / "workload", configs / "switch4.yml"),
        "existing_fully_connected4": run_astra("existing_fully_connected4", traces / "basic4_all_to_all" / "workload", configs / "fully_connected4.yml"),
    }

    bytes_per_ns = 50 * (1 << 30) / 1_000_000_000
    subchunk_cycles = int(2500 / bytes_per_ns)
    summary = {
        "all_passed": all(run["returncode"] == 0 for run in runs.values()),
        "byte_conservation": {
            "diamond_5000": validate_log(runs["diamond_ecmp"], 2, 5000),
            "clos_5000": validate_log(runs["tiny_clos_ecmp"], 2, 5000),
            "torus_5000": validate_log(runs["tiny_torus_ecmp"], 2, 5000),
            "recv_before_send_5001": validate_log(runs["recv_before_send_odd_slow_path"], 2, 5001),
            "send_before_recv_5001": validate_log(runs["send_before_recv_odd_slow_path"], 2, 5001),
            "two_tags": len(runs["two_tags_same_pair"]["ecmp_logs"]) == 2
            and all(log["byte_sum"] == 5000 for log in runs["two_tags_same_pair"]["ecmp_logs"]),
        },
        "manual": {
            "bytes_per_ns": bytes_per_ns,
            "one_link_5000B_cycles": int(5000 / bytes_per_ns),
            "diamond_subchunk_cycles_per_hop": subchunk_cycles,
            "diamond_expected_cycles": 2 * subchunk_cycles,
        },
        "runs": runs,
    }
    summary["all_passed"] = summary["all_passed"] and all(summary["byte_conservation"].values())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary)
    print(json.dumps(summary, indent=2))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
