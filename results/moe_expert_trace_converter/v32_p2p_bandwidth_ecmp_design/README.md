# V3.2 P2P / Bandwidth / ECMP Design Audit

## Summary

This audit validates whether MoE dispatch/combine traffic can be represented as point-to-point Chakra `COMM_SEND_NODE` / `COMM_RECV_NODE` traces on native congestion-aware `GraphTopology`.

Final answers:

1. Can ASTRA-sim execute MoE dispatch/combine as point-to-point SEND/RECV traffic? **Yes**, as independent `COMM_SEND_NODE` / `COMM_RECV_NODE` pairs.
2. Is bandwidth conversion correct? **Yes for Gb/s -> ASTRA GB/s by `/8`**, with one caveat: ASTRA internally converts "GB/s" using `2^30` bytes, so there is a decimal-vs-binary ~7.37% difference, but no 8x error.
3. Is native GraphTopology ready for ECMP implementation? **Ready as a base**, but ECMP requires callback aggregation and route API changes before we can claim ECMP.

## Test Results

```json
{
  "all_passed": true,
  "message_size_bytes": 5000,
  "trace_validation": {
    "p2p_all_to_all": {
      "prefix": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v32_p2p_bandwidth_ecmp_design/chakra_traces/p2p_all_to_all_4/workload",
      "rank_count": 4,
      "message_size_bytes": 5000,
      "directed_messages": 12,
      "node_types": [
        "COMM_SEND_NODE",
        "COMM_RECV_NODE"
      ],
      "uses_comm_coll_node": false
    },
    "one_link": {
      "prefix": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v32_p2p_bandwidth_ecmp_design/chakra_traces/one_link_0_to_1/workload",
      "src": 0,
      "dst": 1,
      "message_size_bytes": 5000
    }
  },
  "bandwidth_audit": {
    "input_bandwidth_gbps": 400,
    "graph_topology_converted_GBps": 50.0,
    "astra_bytes_per_ns": 53.6870912,
    "expected_astra_ns": 93.13225746154785,
    "expected_astra_cycles_truncated": 93,
    "expected_decimal_ns": 100.0,
    "decimal_vs_astra_difference_percent": 7.374182400000007,
    "eight_x_unit_risk": false
  },
  "runs": {
    "p2p_all_to_all_square": {
      "label": "p2p_all_to_all_square",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v32_p2p_bandwidth_ecmp_design/chakra_traces/p2p_all_to_all_4/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v32_p2p_bandwidth_ecmp_design/network_configs/graph_square4.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v32_p2p_bandwidth_ecmp_design/runs/p2p_all_to_all_square.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v32_p2p_bandwidth_ecmp_design/runs/p2p_all_to_all_square.stderr.txt",
      "cycles": [
        372,
        372,
        465,
        465
      ],
      "max_cycles": 465
    },
    "one_link_400gbps": {
      "label": "one_link_400gbps",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v32_p2p_bandwidth_ecmp_design/chakra_traces/one_link_0_to_1/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v32_p2p_bandwidth_ecmp_design/network_configs/graph_two_node.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v32_p2p_bandwidth_ecmp_design/runs/one_link_400gbps.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v32_p2p_bandwidth_ecmp_design/runs/one_link_400gbps.stderr.txt",
      "cycles": [
        93,
        93
      ],
      "max_cycles": 93
    }
  }
}
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
/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v32_p2p_bandwidth_ecmp_design/chakra_traces/p2p_all_to_all_4/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v32_p2p_bandwidth_ecmp_design/network_configs/graph_square4.yml
```

The run completed successfully with `max_cycles=465`. This validates the P2P plumbing, not a collective algorithm.

## Bandwidth Sanity Check

The one-link test sends `5000` bytes over one 400 Gb/s graph edge.

Manual conversion:

- 400 Gb/s / 8 = 50 GB/s input to ASTRA.
- ASTRA converts GB/s to B/ns as `GBps * (1 << 30) / 1e9`.
- So 50 GB/s becomes `53.687091` B/ns.
- Expected ASTRA serialization time is `93.132` ns, truncated to about `93` cycles.
- Decimal SI calculation would be exactly `100.000` ns.

Observed max cycles: `93`.

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
