# V39 Native Circuit-Aware ASTRA Design

## Current Blocker

`GraphTopology` routes Chakra SEND/RECV chunks hop by hop. Intermediate graph
nodes are forwarding/queuing points. If SON/RON graph nodes are GPUs, the result
is GPU-as-router packet routing.

## Proposed Addition

Add a `CircuitCapacityNetwork` or `CircuitGraphTopology` path in the analytical
backend. It should:

- load endpoint ranks and optical resource graph separately;
- route SEND/RECV over optical resource paths;
- charge capacity to optical links/resources;
- charge injection only to source and destination ranks;
- avoid treating intermediate rank IDs as GPU forwarding work unless explicitly
  marked as packet-router nodes;
- split messages over ECMP/circuit paths deterministically;
- aggregate callbacks only after all circuit subflows finish.

## Minimal API

```yaml
topology: [ CircuitGraph ]
graph_file: topology.json
routing: ecmp
ecmp_max_paths: 4
endpoint_injection_gbps: 1600
resource_link_gbps: 400
```

## Files Likely Involved

- `extern/network_backend/analytical/congestion_aware/topology/*`
- `CongestionAwareNetworkApi::sim_send`
- Graph route/path-cache code added in V31/V33
- network parser for `CircuitGraph`

## Smoke Tests

1. 4-node direct circuit graph.
2. Optical switch/resource intermediate node without GPU forwarding.
3. 32/128-node SON optical circuit graph.
4. Compare circuit-aware ASTRA timing to V39 optical reference; explain any gap.

Do not generate paper figures from this prototype until smoke tests pass.
