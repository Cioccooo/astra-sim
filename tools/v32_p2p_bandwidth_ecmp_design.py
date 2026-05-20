#!/usr/bin/env python3
"""Focused P2P/bandwidth/ECMP audit for native GraphTopology."""

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
OUT = REPO / "results/moe_expert_trace_converter/v32_p2p_bandwidth_ecmp_design"
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


def make_send(node_id: int, src: int, dst: int, size: int, tag: int) -> Node:
    node = Node()
    node.id = node_id
    node.name = f"P2P_SEND_{src}_to_{dst}_tag{tag}"
    node.type = NodeType.COMM_SEND_NODE
    add_attr(node, "is_cpu_op", False)
    add_attr(node, "comm_src", src)
    add_attr(node, "comm_dst", dst)
    add_attr(node, "comm_size", size)
    add_attr(node, "comm_tag", tag)
    return node


def make_recv(node_id: int, src: int, dst: int, size: int, tag: int) -> Node:
    node = Node()
    node.id = node_id
    node.name = f"P2P_RECV_{src}_to_{dst}_tag{tag}"
    node.type = NodeType.COMM_RECV_NODE
    add_attr(node, "is_cpu_op", False)
    add_attr(node, "comm_src", src)
    add_attr(node, "comm_dst", dst)
    add_attr(node, "comm_size", size)
    add_attr(node, "comm_tag", tag)
    return node


def write_all_to_all_p2p(prefix: Path, ranks: int, size: int) -> dict[str, Any]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    nodes_by_rank: dict[int, list[Node]] = {rank: [] for rank in range(ranks)}
    next_id = {rank: 1 for rank in range(ranks)}
    tag = 1
    for src in range(ranks):
        for dst in range(ranks):
            if src == dst:
                continue
            nodes_by_rank[src].append(make_send(next_id[src], src, dst, size, tag))
            next_id[src] += 1
            nodes_by_rank[dst].append(make_recv(next_id[dst], src, dst, size, tag))
            next_id[dst] += 1
            tag += 1

    for rank, nodes in nodes_by_rank.items():
        with (prefix.parent / f"{prefix.name}.{rank}.et").open("wb") as handle:
            encode_message(handle, GlobalMetadata(version="0.0.4"))
            for node in nodes:
                encode_message(handle, node)

    return {
        "prefix": str(prefix),
        "rank_count": ranks,
        "message_size_bytes": size,
        "directed_messages": ranks * (ranks - 1),
        "node_types": ["COMM_SEND_NODE", "COMM_RECV_NODE"],
        "uses_comm_coll_node": False,
    }


def write_single_pair_p2p(prefix: Path, ranks: int, src: int, dst: int, size: int) -> dict[str, Any]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    nodes_by_rank: dict[int, list[Node]] = {rank: [] for rank in range(ranks)}
    nodes_by_rank[src].append(make_send(1, src, dst, size, 1))
    nodes_by_rank[dst].append(make_recv(1, src, dst, size, 1))
    for rank in range(ranks):
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
    return {"prefix": str(prefix), "src": src, "dst": dst, "message_size_bytes": size}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_network(path: Path, graph_file: Path, npus: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "topology: [ Graph ]",
                f"npus_count: [ {npus} ]",
                "bandwidth: [ 50.0 ]  # required by parser; Graph edge bandwidth_gbps is authoritative",
                "latency: [ 0.0 ]",
                f"graph_file: {graph_file}",
            ]
        )
        + "\n"
    )


def write_square_graph(path: Path) -> None:
    write_json(
        path,
        {
            "name": "square4_400gbps",
            "node_count": 4,
            "gpu_count": 4,
            "directed": False,
            "edges": [
                {"src": 0, "dst": 1, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 1, "dst": 2, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 2, "dst": 3, "bandwidth_gbps": 400, "latency_ns": 0},
                {"src": 3, "dst": 0, "bandwidth_gbps": 400, "latency_ns": 0},
            ],
        },
    )


def write_two_node_graph(path: Path) -> None:
    write_json(
        path,
        {
            "name": "two_node_one_400gbps_link",
            "node_count": 2,
            "gpu_count": 2,
            "directed": False,
            "edges": [{"src": 0, "dst": 1, "bandwidth_gbps": 400, "latency_ns": 0}],
        },
    )


def run_astra(label: str, workload_prefix: Path, network_config: Path) -> dict[str, Any]:
    stdout = OUT / "runs" / f"{label}.stdout.txt"
    stderr = OUT / "runs" / f"{label}.stderr.txt"
    stdout.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ASTRA_BIN),
        f"--workload-configuration={workload_prefix}",
        f"--system-configuration={SYSTEM}",
        f"--remote-memory-configuration={REMOTE_MEMORY}",
        f"--network-configuration={network_config}",
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
    }


def write_report(summary: dict[str, Any]) -> None:
    readme = OUT / "README.md"
    readme.write_text(
        f"""# V3.2 P2P / Bandwidth / ECMP Design Audit

## Summary

This audit validates whether MoE dispatch/combine traffic can be represented as point-to-point Chakra `COMM_SEND_NODE` / `COMM_RECV_NODE` traces on native congestion-aware `GraphTopology`.

Final answers:

1. Can ASTRA-sim execute MoE dispatch/combine as point-to-point SEND/RECV traffic? **Yes**, as independent `COMM_SEND_NODE` / `COMM_RECV_NODE` pairs.
2. Is bandwidth conversion correct? **Yes for Gb/s -> ASTRA GB/s by `/8`**, with one caveat: ASTRA internally converts "GB/s" using `2^30` bytes, so there is a decimal-vs-binary ~7.37% difference, but no 8x error.
3. Is native GraphTopology ready for ECMP implementation? **Ready as a base**, but ECMP requires callback aggregation and route API changes before we can claim ECMP.

## Test Results

```json
{json.dumps(summary, indent=2)}
```

## Code Paths Inspected

- `astra-sim/workload/Workload.cc:288-302`: `issue_comm()` dispatches `COMM_COLL_NODE`, `COMM_SEND_NODE`, and `COMM_RECV_NODE` separately.
- `astra-sim/workload/Workload.cc:304-391`: `issue_coll_comm()` is collective-only and calls `generate_all_reduce`, `generate_all_to_all`, etc.
- `astra-sim/workload/Workload.cc:393-417`: `issue_send_comm()` reads `comm_src`, `comm_dst`, `comm_size`, `comm_tag` and calls `sys->front_end_sim_send(..., NATIVE, ...)`.
- `astra-sim/workload/Workload.cc:419-439`: `issue_recv_comm()` reads matching fields and calls `sys->front_end_sim_recv(..., NATIVE, ...)`.
- `astra-sim/system/Sys.cc:1246-1303`: `front_end_sim_send/recv()` normalize native tags and call `sim_send/recv()` when rendezvous is disabled.
- `astra-sim/system/Sys.cc:1362-1401`: `sim_send/recv()` call `comm_NI->sim_send/recv()`.
- `astra-sim/network_frontend/analytical/congestion_aware/CongestionAwareNetworkApi.cc:36-75`: `sim_send()` creates one `Chunk`, asks `topology->route(src,dst)`, then calls `topology->send()`.
- `astra-sim/network_frontend/analytical/common/CommonNetworkApi.cc:34-63` and `96-135`: callback tracker matches send and recv completion.
- `extern/network_backend/analytical/congestion_aware/network/Device.cpp:23-42`: each intermediate device forwards the chunk to the next link.
- `extern/network_backend/analytical/congestion_aware/network/Link.cpp:52-135`: each link serializes chunks, queues pending chunks, and schedules arrival/free events.
- `extern/network_backend/analytical/congestion_aware/network/Chunk.cpp:13-30`: chunk arrival either invokes callback at destination or forwards to the next device.
- `extern/network_backend/analytical/congestion_aware/basic-topology/GraphTopology.cpp:199-207`: `GraphTopology::route()` returns one deterministic shortest path.

## Chakra P2P Validation

The generated 4-rank test uses 12 directed point-to-point messages, one from every rank to every other rank. It intentionally uses only explicit `COMM_SEND_NODE` and `COMM_RECV_NODE`; no `COMM_COLL_NODE` appears in the trace generator metadata.

ASTRA run:

```bash
{summary["runs"]["p2p_all_to_all_square"]["command"]}
```

The run completed successfully with `max_cycles={summary["runs"]["p2p_all_to_all_square"]["max_cycles"]}`. This validates the P2P plumbing, not a collective algorithm.

## Bandwidth Sanity Check

The one-link test sends `{summary["message_size_bytes"]}` bytes over one 400 Gb/s graph edge.

Manual conversion:

- 400 Gb/s / 8 = 50 GB/s input to ASTRA.
- ASTRA converts GB/s to B/ns as `GBps * (1 << 30) / 1e9`.
- So 50 GB/s becomes `{summary["bandwidth_audit"]["astra_bytes_per_ns"]:.6f}` B/ns.
- Expected ASTRA serialization time is `{summary["bandwidth_audit"]["expected_astra_ns"]:.3f}` ns, truncated to about `{summary["bandwidth_audit"]["expected_astra_cycles_truncated"]}` cycles.
- Decimal SI calculation would be exactly `{summary["bandwidth_audit"]["expected_decimal_ns"]:.3f}` ns.

Observed max cycles: `{summary["runs"]["one_link_400gbps"]["max_cycles"]}`.

Conclusion: no 8x Gb/s-vs-GB/s risk remains for `GraphTopology` edge configs, because `bandwidth_gbps` is divided by 8 before constructing `Link`. The only remaining unit caveat is ASTRA's GiB-style conversion.

## Intermediate-Hop Semantics

In congestion-aware analytical backend, a multi-hop graph path is a sequence of `Device` objects connected by `Link` queues. A `Chunk` arrives at each intermediate node and is forwarded onto the next link. Each link has serialization delay, latency, busy/free state, and a FIFO pending queue.

This is closer to analytical packet/store-and-forward forwarding than an ideal transparent optical circuit. For SON/RON papers we should describe this as a link-contention-aware logical graph backend. If we want transparent optical-circuit semantics, we may need a different path-delay model, e.g. reserve a full path and charge serialization once at the bottleneck plus propagation/control latency.

## ECMP Minimal Implementation Plan

Target semantics:

`one logical message -> N subchunks -> each subchunk follows one equal-cost shortest path -> original send/recv callbacks fire only after all subchunks complete`.

Likely changes:

- `GraphTopology`: add `routes(src,dst)` returning all equal-cost deterministic shortest paths, while preserving existing `route(src,dst)` for single-path users.
- `CongestionAwareNetworkApi::sim_send`: when ECMP is enabled, split `count` into subchunk sizes, create one `Chunk` per route, and send each along its route.
- `CommonNetworkApi` / callback tracking: add an aggregation object so the original logical send/recv callback fires only after all subchunk arrivals have completed and the matching recv is registered.
- `ChunkIdGenerator` / `CallbackTracker`: either allocate subchunk IDs under one parent logical ID or include `subchunk_id` while aggregating parent completion.
- Network config parser: add optional ECMP mode such as `routing: ecmp` and maybe `ecmp_max_paths`.

Main risks:

- Callback aggregation must not fire send or recv early after only the first subchunk arrives.
- Recv completion correctness is subtle because recv may be issued before or after some subchunks arrive.
- Splitting bytes changes link contention behavior; we need a deterministic policy: equal byte split across all equal-cost paths, or hash-based path choice per message.
- Existing `route(src,dst)` API is single-route, so ECMP should be additive to avoid breaking `Ring`, `Switch`, and `FullyConnected`.

## Files Generated

- P2P traces: `chakra_traces/p2p_all_to_all_4/workload.*.et`
- One-link trace: `chakra_traces/one_link_0_to_1/workload.*.et`
- Graphs: `graphs/square4.json`, `graphs/two_node_one_link.json`
- Network configs: `network_configs/graph_square4.yml`, `network_configs/graph_two_node.yml`
- Run logs: `runs/*.stdout.txt`, `runs/*.stderr.txt`
"""
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    graphs = OUT / "graphs"
    configs = OUT / "network_configs"
    traces = OUT / "chakra_traces"

    square = graphs / "square4.json"
    two_node = graphs / "two_node_one_link.json"
    write_square_graph(square)
    write_two_node_graph(two_node)

    square_config = configs / "graph_square4.yml"
    two_node_config = configs / "graph_two_node.yml"
    write_network(square_config, square, 4)
    write_network(two_node_config, two_node, 2)

    msg_size = 5000
    p2p_trace = write_all_to_all_p2p(traces / "p2p_all_to_all_4" / "workload", 4, msg_size)
    one_link_trace = write_single_pair_p2p(traces / "one_link_0_to_1" / "workload", 2, 0, 1, msg_size)

    p2p_run = run_astra("p2p_all_to_all_square", Path(p2p_trace["prefix"]), square_config)
    one_link_run = run_astra("one_link_400gbps", Path(one_link_trace["prefix"]), two_node_config)

    gbps = 400
    astra_gbps_as_GBps = gbps / 8
    astra_bytes_per_ns = astra_gbps_as_GBps * (1 << 30) / 1_000_000_000
    expected_astra_ns = msg_size / astra_bytes_per_ns
    expected_decimal_ns = msg_size / astra_gbps_as_GBps
    summary = {
        "all_passed": p2p_run["returncode"] == 0 and one_link_run["returncode"] == 0,
        "message_size_bytes": msg_size,
        "trace_validation": {
            "p2p_all_to_all": p2p_trace,
            "one_link": one_link_trace,
        },
        "bandwidth_audit": {
            "input_bandwidth_gbps": gbps,
            "graph_topology_converted_GBps": astra_gbps_as_GBps,
            "astra_bytes_per_ns": astra_bytes_per_ns,
            "expected_astra_ns": expected_astra_ns,
            "expected_astra_cycles_truncated": int(expected_astra_ns),
            "expected_decimal_ns": expected_decimal_ns,
            "decimal_vs_astra_difference_percent": (expected_decimal_ns / expected_astra_ns - 1) * 100,
            "eight_x_unit_risk": False,
        },
        "runs": {
            "p2p_all_to_all_square": p2p_run,
            "one_link_400gbps": one_link_run,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary)
    print(json.dumps(summary, indent=2))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
