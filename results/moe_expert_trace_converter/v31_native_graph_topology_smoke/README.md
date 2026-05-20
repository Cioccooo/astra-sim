# V3.1 Native GraphTopology Smoke Test

This folder validates the first native ASTRA-sim integration step: deterministic arbitrary graph topology support inside the analytical congestion-aware backend.

## What This Claims

- Native congestion-aware analytical backend can load a `Graph` topology from JSON/YAML.
- Graph edges support `bandwidth_gbps` and optional `latency_ns`; bandwidth is converted to ASTRA internal GB/s by dividing by 8.
- Pairwise Chakra `COMM_SEND_NODE` / `COMM_RECV_NODE` traffic can route over one deterministic shortest path.
- Existing `Ring`, `Switch`, and `FullyConnected` topology names still run.

## What This Does Not Claim

- No ECMP splitting yet.
- No in-run RON W=4 topology swap yet.
- No full EN/SON/RON reproduction yet.
- Multi-hop paths are analytical graph paths through ASTRA `Device`/`Link` objects, not a device-level optical physics model.

## Build Command

```bash
./build/astra_analytical/build.sh -t congestion_aware
```

Note: this local macOS tree has `/usr/local` protobuf 3.6.1, so Chakra protobuf C++/Python bindings were regenerated with that version for build consistency.

## Smoke Results

```json
{
  "all_passed": true,
  "message_bytes": 5000,
  "bandwidth_gbps": 400,
  "per_hop_serialization_ns": 100.0,
  "manual_expectations": {
    "line4_0_to_3_hops": 3,
    "line4_network_only_ns": 300.0,
    "torus4x8_0_to_31_hops": 2,
    "torus4x8_network_only_ns": 200.0,
    "non_gpu_switch_nodes_0_to_3_hops": 3,
    "non_gpu_switch_nodes_network_only_ns": 300.0
  },
  "runs": [
    {
      "label": "graph_line4",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/chakra_traces/line4_0_to_3/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/network_configs/graph_line4.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/runs/graph_line4.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/runs/graph_line4.stderr.txt",
      "cycles": [
        1,
        1,
        279,
        279
      ],
      "max_cycles": 279
    },
    {
      "label": "graph_torus4x8",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/chakra_traces/torus32_0_to_31/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/network_configs/graph_torus4x8.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/runs/graph_torus4x8.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/runs/graph_torus4x8.stderr.txt",
      "cycles": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        186,
        186
      ],
      "max_cycles": 186
    },
    {
      "label": "graph_non_gpu_switch_nodes",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/chakra_traces/basic4_0_to_3/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/network_configs/graph_non_gpu_switch_nodes.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/runs/graph_non_gpu_switch_nodes.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/runs/graph_non_gpu_switch_nodes.stderr.txt",
      "cycles": [
        1,
        1,
        279,
        279
      ],
      "max_cycles": 279
    },
    {
      "label": "existing_ring4",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/chakra_traces/basic4_0_to_3/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/network_configs/ring4.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/runs/existing_ring4.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/runs/existing_ring4.stderr.txt",
      "cycles": [
        1,
        1,
        93,
        93
      ],
      "max_cycles": 93
    },
    {
      "label": "existing_switch4",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/chakra_traces/basic4_0_to_3/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/network_configs/switch4.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/runs/existing_switch4.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/runs/existing_switch4.stderr.txt",
      "cycles": [
        1,
        1,
        186,
        186
      ],
      "max_cycles": 186
    },
    {
      "label": "existing_fully_connected4",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/chakra_traces/basic4_0_to_3/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/network_configs/fully_connected4.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/runs/existing_fully_connected4.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v31_native_graph_topology_smoke/runs/existing_fully_connected4.stderr.txt",
      "cycles": [
        1,
        1,
        93,
        93
      ],
      "max_cycles": 93
    }
  ]
}
```

## Code Files Changed

- `extern/network_backend/analytical/include/astra-network-analytical/common/Type.h` adds `TopologyBuildingBlock::Graph`.
- `extern/network_backend/analytical/include/astra-network-analytical/common/NetworkParser.h` stores and exposes `graph_file`.
- `extern/network_backend/analytical/common/network-parser/NetworkParser.cpp` parses `Graph` and resolves relative `graph_file` paths.
- `extern/network_backend/analytical/include/astra-network-analytical/congestion_aware/GraphTopology.h` declares native graph topology support.
- `extern/network_backend/analytical/congestion_aware/basic-topology/GraphTopology.cpp` loads JSON/YAML graphs, creates `Device`/`Link`s, validates rank mappings, and precomputes deterministic BFS shortest paths.
- `extern/network_backend/analytical/congestion_aware/topology/Helper.cpp` constructs `GraphTopology` for congestion-aware analytical runs.
- `tools/v31_native_graph_topology_smoke.py` generates and runs this smoke test.

## Manual Timing Sanity Check

For a single non-contended message of 5000 bytes over 400 Gb/s links:

`400 Gb/s = 50 GB/s = 50 bytes/ns`, so serialization is `5000 / 50 = 100 ns` per hop.

The 4-node line test routes rank 0 to rank 3 over 3 hops, so the network-only lower-bound is about `300 ns` plus ASTRA endpoint/scheduling overhead.

The 4x8 torus test records the deterministic BFS hop count in `summary.json` and uses the same per-hop serialization estimate.

## ECMP Blockers

- `GraphTopology::route(src, dst)` returns one `Route`; there is no API yet to return multiple equal-cost paths.
- `CongestionAwareNetworkApi::sim_send` creates one `Chunk`, so splitting one logical message across multiple paths needs API/workload changes or chunkization before routing.
- A fair ECMP model also needs deterministic path hashing or byte-splitting semantics and validation against folded-Clos / torus expectations.

## RON Segmented-Run Blockers

- In-run topology mutation is not supported yet; topology is built once by `NetworkParser`/`Helper` at simulator startup.
- The safe next step is one ASTRA run per request/request group with a selected Graph JSON, then external aggregation plus reconfiguration penalty.
- Native in-run swapping would require a topology reload API and careful draining semantics for in-flight `Chunk`s.

