# V3.3 Native GraphTopology ECMP Smoke

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
{
  "all_passed": true,
  "byte_conservation": {
    "diamond_5000": true,
    "clos_5000": true,
    "torus_5000": true,
    "recv_before_send_5001": true,
    "send_before_recv_5001": true,
    "two_tags": true
  },
  "manual": {
    "bytes_per_ns": 53.6870912,
    "one_link_5000B_cycles": 93,
    "diamond_subchunk_cycles_per_hop": 46,
    "diamond_expected_cycles": 92
  },
  "runs": {
    "diamond_ecmp": {
      "label": "diamond_ecmp",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/chakra_traces/diamond_0_to_3_5000/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/network_configs/diamond_ecmp.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/diamond_ecmp.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/diamond_ecmp.stderr.txt",
      "cycles": [
        1,
        1,
        92,
        92
      ],
      "max_cycles": 92,
      "ecmp_logs": [
        {
          "tag": 1,
          "src": 0,
          "dst": 3,
          "bytes": 5000,
          "paths": 2,
          "subchunks": [
            {
              "index": 0,
              "bytes": 2500,
              "route": "[0,1,3]"
            },
            {
              "index": 1,
              "bytes": 2500,
              "route": "[0,2,3]"
            }
          ],
          "byte_sum": 5000
        }
      ]
    },
    "tiny_clos_ecmp": {
      "label": "tiny_clos_ecmp",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/chakra_traces/clos_0_to_2_5000/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/network_configs/tiny_clos_ecmp.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/tiny_clos_ecmp.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/tiny_clos_ecmp.stderr.txt",
      "cycles": [
        1,
        1,
        230,
        230
      ],
      "max_cycles": 230,
      "ecmp_logs": [
        {
          "tag": 1,
          "src": 0,
          "dst": 2,
          "bytes": 5000,
          "paths": 2,
          "subchunks": [
            {
              "index": 0,
              "bytes": 2500,
              "route": "[0,4,6,5,2]"
            },
            {
              "index": 1,
              "bytes": 2500,
              "route": "[0,4,7,5,2]"
            }
          ],
          "byte_sum": 5000
        }
      ]
    },
    "tiny_torus_ecmp": {
      "label": "tiny_torus_ecmp",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/chakra_traces/torus_0_to_4_5000/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/network_configs/torus3x3_ecmp.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/tiny_torus_ecmp.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/tiny_torus_ecmp.stderr.txt",
      "cycles": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        92,
        92
      ],
      "max_cycles": 92,
      "ecmp_logs": [
        {
          "tag": 1,
          "src": 0,
          "dst": 4,
          "bytes": 5000,
          "paths": 2,
          "subchunks": [
            {
              "index": 0,
              "bytes": 2500,
              "route": "[0,1,4]"
            },
            {
              "index": 1,
              "bytes": 2500,
              "route": "[0,3,4]"
            }
          ],
          "byte_sum": 5000
        }
      ]
    },
    "recv_before_send_odd_slow_path": {
      "label": "recv_before_send_odd_slow_path",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/chakra_traces/recv_before_send_odd/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/network_configs/diamond_slow_ecmp.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/recv_before_send_odd_slow_path.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/recv_before_send_odd_slow_path.stderr.txt",
      "cycles": [
        1,
        1,
        2092,
        2092
      ],
      "max_cycles": 2092,
      "ecmp_logs": [
        {
          "tag": 1,
          "src": 0,
          "dst": 3,
          "bytes": 5001,
          "paths": 2,
          "subchunks": [
            {
              "index": 0,
              "bytes": 2501,
              "route": "[0,1,3]"
            },
            {
              "index": 1,
              "bytes": 2500,
              "route": "[0,2,3]"
            }
          ],
          "byte_sum": 5001
        }
      ]
    },
    "send_before_recv_odd_slow_path": {
      "label": "send_before_recv_odd_slow_path",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/chakra_traces/send_before_recv_odd/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/network_configs/diamond_slow_ecmp.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/send_before_recv_odd_slow_path.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/send_before_recv_odd_slow_path.stderr.txt",
      "cycles": [
        1,
        1,
        1092,
        1092
      ],
      "max_cycles": 1092,
      "ecmp_logs": [
        {
          "tag": 1,
          "src": 0,
          "dst": 3,
          "bytes": 5001,
          "paths": 2,
          "subchunks": [
            {
              "index": 0,
              "bytes": 2501,
              "route": "[0,1,3]"
            },
            {
              "index": 1,
              "bytes": 2500,
              "route": "[0,2,3]"
            }
          ],
          "byte_sum": 5001
        }
      ]
    },
    "two_tags_same_pair": {
      "label": "two_tags_same_pair",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/chakra_traces/two_tags_same_pair/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/network_configs/diamond_ecmp.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/two_tags_same_pair.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/two_tags_same_pair.stderr.txt",
      "cycles": [
        1,
        1,
        184,
        184
      ],
      "max_cycles": 184,
      "ecmp_logs": [
        {
          "tag": 1,
          "src": 0,
          "dst": 3,
          "bytes": 5000,
          "paths": 2,
          "subchunks": [
            {
              "index": 0,
              "bytes": 2500,
              "route": "[0,1,3]"
            },
            {
              "index": 1,
              "bytes": 2500,
              "route": "[0,2,3]"
            }
          ],
          "byte_sum": 5000
        },
        {
          "tag": 2,
          "src": 0,
          "dst": 3,
          "bytes": 5000,
          "paths": 2,
          "subchunks": [
            {
              "index": 0,
              "bytes": 2500,
              "route": "[0,1,3]"
            },
            {
              "index": 1,
              "bytes": 2500,
              "route": "[0,2,3]"
            }
          ],
          "byte_sum": 5000
        }
      ]
    },
    "deterministic_graph_line4": {
      "label": "deterministic_graph_line4",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/chakra_traces/line4_0_to_3_5000/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/network_configs/line4_deterministic.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/deterministic_graph_line4.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/deterministic_graph_line4.stderr.txt",
      "cycles": [
        1,
        1,
        279,
        279
      ],
      "max_cycles": 279,
      "ecmp_logs": []
    },
    "one_link_400gbps": {
      "label": "one_link_400gbps",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/chakra_traces/one_link_0_to_1_5000/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/network_configs/two_node_deterministic.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/one_link_400gbps.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/one_link_400gbps.stderr.txt",
      "cycles": [
        93,
        93
      ],
      "max_cycles": 93,
      "ecmp_logs": []
    },
    "existing_ring4": {
      "label": "existing_ring4",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/chakra_traces/basic4_all_to_all/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/network_configs/ring4.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/existing_ring4.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/existing_ring4.stderr.txt",
      "cycles": [
        372,
        465,
        465,
        465
      ],
      "max_cycles": 465,
      "ecmp_logs": []
    },
    "existing_switch4": {
      "label": "existing_switch4",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/chakra_traces/basic4_all_to_all/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/network_configs/switch4.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/existing_switch4.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/existing_switch4.stderr.txt",
      "cycles": [
        558,
        651,
        744,
        744
      ],
      "max_cycles": 744,
      "ecmp_logs": []
    },
    "existing_fully_connected4": {
      "label": "existing_fully_connected4",
      "returncode": 0,
      "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/chakra_traces/basic4_all_to_all/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/network_configs/fully_connected4.yml",
      "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/existing_fully_connected4.stdout.txt",
      "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v33_native_graph_ecmp_smoke/runs/existing_fully_connected4.stderr.txt",
      "cycles": [
        279,
        279,
        279,
        279
      ],
      "max_cycles": 279,
      "ecmp_logs": []
    }
  }
}
```

## Manual Timing Checks

- ASTRA internally interprets `50 GB/s` as `50 * 2^30 / 1e9 = 53.687091` B/ns.
- A 5000B single-path one-link message should take about `93` cycles; regression observed `93`.
- Diamond ECMP 5000B splits into 2500B + 2500B over two 2-hop paths. Each subchunk serializes in about `46` cycles per hop; expected network-only completion is about `92` cycles. Observed `92`.
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
