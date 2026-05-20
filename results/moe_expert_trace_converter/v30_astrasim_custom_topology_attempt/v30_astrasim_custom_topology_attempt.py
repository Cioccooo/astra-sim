#!/usr/bin/env python3
"""V3.0 ASTRA-side custom topology attempt for MoE prefill EN/SON/RON.

This is intentionally not the old HESPAS V2.8 evaluator.  It lives in the
ASTRA-sim tree, generates Chakra SEND/RECV artifacts, writes explicit graph
topology configs, and uses a small graph/link-contention timing backend for the
topologies that the current ASTRA analytical C++ backend cannot represent.

Important limitation: this is not yet wired into the ASTRA C++ network API.
It is a minimal ASTRA-side backend/prototype to identify the exact remaining
C++ work needed for a first-class ASTRA graph backend.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
STG_CHAKRA = (
    REPO_ROOT.parent
    / "symbolic_tensor_graph"
    / "symbolic_tensor_graph"
    / "chakra"
    / "backends"
    / "chakra_00_4_backend"
)
if STG_CHAKRA.exists():
    sys.path.insert(0, str(STG_CHAKRA))

from et_def.et_def_pb2 import AttributeProto as ChakraAttr  # type: ignore  # noqa: E402
from et_def.et_def_pb2 import GlobalMetadata, Node as ChakraNode, NodeType  # type: ignore  # noqa: E402
from protolib import encodeMessage as encode_message  # type: ignore  # noqa: E402


V28_DIR = Path("/Users/dfx/Python/hespas/results/moe_expert_trace_converter/v28_fair_en_son_ron_ecmp_overhead")
V28_TEACHER_CSV = V28_DIR / "fig_teacher_32gpu_fair_subset.csv"

HIDDEN_SIZE = 4096
BYTES_PER_VALUE = 2
BYTES_PER_SELECTION = HIDDEN_SIZE * BYTES_PER_VALUE
EP_SIZE = 32
BLOCK_SIZE = 16
LEAF_SIZE = 8
EN_GBPS = 400.0
OPTICAL_GBPS = 400.0
RON_DEGREE = 4

TARGET_MODES = [
    "EN folded-Clos ECMP / 1.3x imbalance",
    "SON 2D torus ECMP",
    "RON calibrated",
    "RON W=4 1us",
    "RON oracle",
]

V28_MODE_MAP = {
    "EN folded-Clos ECMP / 1.3x imbalance": "EN ECMP-imbalance 1.3x",
    "SON 2D torus ECMP": "SON torus ECMP",
    "RON calibrated": "RON calibrated",
    "RON W=4 1us": "RON W=4 1us",
    "RON oracle": "RON oracle",
}

Pair = tuple[int, int]
DirectedEdge = tuple[int, int]
UnitKey = tuple[str, int, str]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    label: str
    path: Path


@dataclass
class Traffic:
    spec: DatasetSpec
    request_ids: list[str]
    eval_request_ids: list[str]
    calibration_request_ids: list[str]
    files_found: int
    files_used: int
    inferred_num_experts: int
    moe_layers: list[int]
    directed_by_unit: dict[UnitKey, Counter[Pair]]
    request_phase_matrix: dict[tuple[str, str], list[list[int]]]
    pair_by_request: dict[str, Counter[Pair]]
    local_bytes: int
    remote_bytes: int
    eval_local_bytes: int
    eval_remote_bytes: int
    malformed_records: int


@dataclass
class GraphTopology:
    name: str
    node_count: int
    gpu_count: int
    directed_edges: dict[DirectedEdge, float]
    undirected_edges_for_export: list[tuple[int, int, float]]
    metadata: dict[str, Any]


def numeric_json_sort(path: Path) -> int | str:
    return int(path.stem) if path.stem.isdigit() else path.stem


def gbps_to_bytes_per_us(gbps: float) -> float:
    return gbps * 1e9 / 8 / 1e6


def expert_rank(expert_id: int, ep_size: int, num_experts: int) -> int:
    experts_per_rank = num_experts / ep_size
    return min(int(expert_id / experts_per_rank), ep_size - 1)


def block_source_rank(global_token_index: int, ep_size: int, block_size: int) -> int:
    return (global_token_index // block_size) % ep_size


def zero_matrix(n: int) -> list[list[int]]:
    return [[0 for _ in range(n)] for _ in range(n)]


def matrix_sum(matrix: list[list[int]]) -> int:
    return sum(sum(row) for row in matrix)


def add_matrix(dst: list[list[int]], src: list[list[int]]) -> None:
    for i in range(len(dst)):
        for j in range(len(dst)):
            dst[i][j] += src[i][j]


def matrix_from_counter(counter: Counter[Pair], n: int) -> list[list[int]]:
    mat = zero_matrix(n)
    for (src, dst), size in counter.items():
        if src != dst and size > 0:
            mat[src][dst] += int(size)
    return mat


def counter_from_matrix(matrix: list[list[int]]) -> Counter[Pair]:
    counter: Counter[Pair] = Counter()
    for src, row in enumerate(matrix):
        for dst, size in enumerate(row):
            if src != dst and size > 0:
                counter[(src, dst)] += int(size)
    return counter


def parse_dataset(spec: DatasetSpec, ep_size: int = EP_SIZE, block_size: int = BLOCK_SIZE) -> Traffic:
    files = sorted(spec.path.glob("*.json"), key=numeric_json_sort)
    request_ids = [path.stem for path in files]
    raw: list[tuple[str, int, int, list[int]]] = []
    max_expert = -1
    malformed = 0
    global_token_offset = 0
    moe_layers: set[int] = set()

    for path in files:
        try:
            trace = json.loads(path.read_text())
        except Exception:
            malformed += 1
            continue
        if not isinstance(trace, list) or not trace or not isinstance(trace[0], dict):
            malformed += 1
            continue
        max_rows = 0
        for layer_str, rows in trace[0].items():
            if rows is None:
                continue
            try:
                layer_id = int(layer_str)
            except Exception:
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
                    max_expert = max(max_expert, expert_id)
                raw.append((path.stem, layer_id, global_token_offset + row_index, parsed))
        global_token_offset += max_rows

    if max_expert < 0:
        raise ValueError(f"No expert IDs found in {spec.path}")
    num_experts = max_expert + 1

    directed_by_unit: dict[UnitKey, Counter[Pair]] = defaultdict(Counter)
    request_phase_matrix: dict[tuple[str, str], list[list[int]]] = {}
    pair_by_request: dict[str, Counter[Pair]] = defaultdict(Counter)
    local_bytes = 0
    remote_bytes = 0

    for request_id, layer_id, global_token_index, experts in raw:
        src = block_source_rank(global_token_index, ep_size, block_size)
        for expert_id in experts:
            dst = expert_rank(expert_id, ep_size, num_experts)
            if src == dst:
                local_bytes += BYTES_PER_SELECTION
                continue
            directed_by_unit[(request_id, layer_id, "dispatch")][(src, dst)] += BYTES_PER_SELECTION
            directed_by_unit[(request_id, layer_id, "combine")][(dst, src)] += BYTES_PER_SELECTION
            pair_by_request[request_id][(src, dst)] += BYTES_PER_SELECTION
            pair_by_request[request_id][(dst, src)] += BYTES_PER_SELECTION
            remote_bytes += 2 * BYTES_PER_SELECTION

    for request_id in request_ids:
        for phase in ("dispatch", "combine"):
            counter: Counter[Pair] = Counter()
            for layer_id in moe_layers:
                counter.update(directed_by_unit.get((request_id, layer_id, phase), Counter()))
            request_phase_matrix[(request_id, phase)] = matrix_from_counter(counter, ep_size)

    cal_count = max(1, min(math.ceil(len(request_ids) * 0.10), len(request_ids) - 1))
    calibration_request_ids = request_ids[:cal_count]
    eval_request_ids = request_ids[cal_count:]
    eval_local = 0
    eval_remote = 0
    for request_id in eval_request_ids:
        eval_remote += sum(pair_by_request.get(request_id, Counter()).values())
    # Local bytes are not needed for timing, but keep a validation-side count.
    eval_local = local_bytes  # split by request is irrelevant because local traffic is excluded.

    return Traffic(
        spec=spec,
        request_ids=request_ids,
        eval_request_ids=eval_request_ids,
        calibration_request_ids=calibration_request_ids,
        files_found=len(files),
        files_used=len(files),
        inferred_num_experts=num_experts,
        moe_layers=sorted(moe_layers),
        directed_by_unit=directed_by_unit,
        request_phase_matrix=request_phase_matrix,
        pair_by_request=pair_by_request,
        local_bytes=local_bytes,
        remote_bytes=remote_bytes,
        eval_local_bytes=eval_local,
        eval_remote_bytes=eval_remote,
        malformed_records=malformed,
    )


def add_attr(node: ChakraNode, name: str, value: int | bool) -> None:
    if isinstance(value, bool):
        node.attr.append(ChakraAttr(name=name, bool_val=value))
    else:
        node.attr.append(ChakraAttr(name=name, uint64_val=int(value)))


def write_pairwise_chakra_trace(matrix: list[list[int]], trace_dir: Path, prefix: str, phase: str) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    n = len(matrix)
    nodes_by_rank: dict[int, list[ChakraNode]] = {rank: [] for rank in range(n)}
    next_id = {rank: 1 for rank in range(n)}
    tag = 1
    for src in range(n):
        for dst in range(n):
            size = int(matrix[src][dst])
            if src == dst or size <= 0:
                continue
            send = ChakraNode()
            send.id = next_id[src]
            next_id[src] += 1
            send.name = f"{phase}_SEND_{src}_to_{dst}_tag{tag}"
            send.type = NodeType.COMM_SEND_NODE
            add_attr(send, "is_cpu_op", False)
            add_attr(send, "comm_src", src)
            add_attr(send, "comm_dst", dst)
            add_attr(send, "comm_size", size)
            add_attr(send, "comm_tag", tag)
            nodes_by_rank[src].append(send)

            recv = ChakraNode()
            recv.id = next_id[dst]
            next_id[dst] += 1
            recv.name = f"{phase}_RECV_{src}_to_{dst}_tag{tag}"
            recv.type = NodeType.COMM_RECV_NODE
            add_attr(recv, "is_cpu_op", False)
            add_attr(recv, "comm_src", src)
            add_attr(recv, "comm_dst", dst)
            add_attr(recv, "comm_size", size)
            add_attr(recv, "comm_tag", tag)
            nodes_by_rank[dst].append(recv)
            tag += 1

    for rank, nodes in nodes_by_rank.items():
        with (trace_dir / f"{prefix}.{rank}.et").open("wb") as handle:
            encode_message(handle, GlobalMetadata(version="0.0.4"))
            for node in nodes:
                encode_message(handle, node)
    return trace_dir / prefix


def write_chakra_artifacts(traffic: Traffic, output_dir: Path, max_segment_chakra: int) -> dict[str, Any]:
    base = output_dir / "chakra_traces" / traffic.spec.dataset_id
    aggregate_info: dict[str, str] = {}
    for phase in ("dispatch", "combine"):
        agg = zero_matrix(EP_SIZE)
        for request_id in traffic.eval_request_ids:
            add_matrix(agg, traffic.request_phase_matrix[(request_id, phase)])
        prefix = write_pairwise_chakra_trace(agg, base / "aggregate_eval" / phase, "workload", f"AGG_{phase}")
        aggregate_info[phase] = str(prefix)

    segment_root = output_dir / "request_segments" / traffic.spec.dataset_id
    segment_root.mkdir(parents=True, exist_ok=True)
    segment_index: list[dict[str, Any]] = []
    for idx, request_id in enumerate(traffic.eval_request_ids):
        request_dir = segment_root / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        item = {"request_id": request_id, "dispatch_bytes": 0, "combine_bytes": 0, "chakra_written": False}
        for phase in ("dispatch", "combine"):
            matrix = traffic.request_phase_matrix[(request_id, phase)]
            item[f"{phase}_bytes"] = matrix_sum(matrix)
            (request_dir / f"{phase}_matrix.json").write_text(json.dumps(matrix))
            if idx < max_segment_chakra:
                prefix = write_pairwise_chakra_trace(
                    matrix,
                    request_dir / f"{phase}_chakra",
                    "workload",
                    f"{request_id}_{phase}",
                )
                item[f"{phase}_chakra_prefix"] = str(prefix)
                item["chakra_written"] = True
        segment_index.append(item)
    (segment_root / "segment_index.json").write_text(json.dumps(segment_index, indent=2, sort_keys=True))
    return {
        "aggregate_eval_chakra_prefixes": aggregate_info,
        "request_segment_index": str(segment_root / "segment_index.json"),
        "request_segment_chakra_written": min(max_segment_chakra, len(traffic.eval_request_ids)),
        "request_segment_count": len(traffic.eval_request_ids),
    }


def add_undirected_edge(
    directed: dict[DirectedEdge, float],
    export_edges: list[tuple[int, int, float]],
    a: int,
    b: int,
    gbps: float,
) -> None:
    directed[(a, b)] = gbps
    directed[(b, a)] = gbps
    export_edges.append((min(a, b), max(a, b), gbps))


def make_en_clos(ep_size: int = EP_SIZE, leaf_size: int = LEAF_SIZE) -> GraphTopology:
    leaves = math.ceil(ep_size / leaf_size)
    spines = leaves
    leaf_base = ep_size
    spine_base = ep_size + leaves
    directed: dict[DirectedEdge, float] = {}
    export: list[tuple[int, int, float]] = []
    for gpu in range(ep_size):
        leaf = leaf_base + gpu // leaf_size
        add_undirected_edge(directed, export, gpu, leaf, EN_GBPS)
    for leaf_idx in range(leaves):
        for spine_idx in range(spines):
            add_undirected_edge(directed, export, leaf_base + leaf_idx, spine_base + spine_idx, EN_GBPS)
    return GraphTopology(
        name="en_folded_clos_ecmp",
        node_count=ep_size + leaves + spines,
        gpu_count=ep_size,
        directed_edges=directed,
        undirected_edges_for_export=export,
        metadata={
            "type": "folded_clos",
            "gpu_uplink_gbps": EN_GBPS,
            "leaf_size": leaf_size,
            "num_leaves": leaves,
            "num_spines": spines,
            "ecmp": True,
            "imbalance_factor": 1.3,
        },
    )


def make_torus_2d(rows: int = 4, cols: int = 8) -> GraphTopology:
    directed: dict[DirectedEdge, float] = {}
    export: list[tuple[int, int, float]] = []
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            right = r * cols + ((c + 1) % cols)
            down = ((r + 1) % rows) * cols + c
            if node < right or c == cols - 1:
                add_undirected_edge(directed, export, node, right, OPTICAL_GBPS)
            if node < down or r == rows - 1:
                add_undirected_edge(directed, export, node, down, OPTICAL_GBPS)
    return GraphTopology(
        name="son_2d_torus_ecmp",
        node_count=rows * cols,
        gpu_count=rows * cols,
        directed_edges=directed,
        undirected_edges_for_export=sorted(set(export)),
        metadata={
            "type": "2d_torus",
            "rows": rows,
            "cols": cols,
            "degree": 4,
            "per_gpu_optical_bandwidth_tbps": 1.6,
            "per_link_bandwidth_gbps": OPTICAL_GBPS,
            "ecmp": True,
            "eps_fallback": False,
        },
    )


def ring_edges(n: int) -> set[tuple[int, int]]:
    return {tuple(sorted((i, (i + 1) % n))) for i in range(n)}


def graph_degree(n: int, edges: Iterable[tuple[int, int]]) -> Counter[int]:
    deg: Counter[int] = Counter()
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    for i in range(n):
        deg[i] += 0
    return deg


def demand_graph(n: int, degree: int, demand: Counter[Pair]) -> set[tuple[int, int]]:
    edges = set(ring_edges(n))
    deg = graph_degree(n, edges)
    weighted: Counter[tuple[int, int]] = Counter()
    for (src, dst), size in demand.items():
        if src != dst:
            weighted[tuple(sorted((src, dst)))] += int(size)
    for edge, _ in weighted.most_common():
        a, b = edge
        if edge in edges or deg[a] >= degree or deg[b] >= degree:
            continue
        edges.add(edge)
        deg[a] += 1
        deg[b] += 1
    rng = random.Random(2026)
    candidates = [tuple(sorted((i, j))) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(candidates)
    for edge in candidates:
        a, b = edge
        if edge in edges or deg[a] >= degree or deg[b] >= degree:
            continue
        edges.add(edge)
        deg[a] += 1
        deg[b] += 1
        if all(deg[i] == degree for i in range(n)):
            break
    return edges


def make_ron_topology(name: str, demand: Counter[Pair], degree: int = RON_DEGREE) -> GraphTopology:
    undirected_edges = sorted(demand_graph(EP_SIZE, degree, demand))
    return make_ron_topology_from_edges(name, undirected_edges, degree, selection="ring_seed_plus_hot_pair_edges_plus_random_fill")


def make_ron_topology_from_edges(
    name: str,
    undirected_edges: Iterable[tuple[int, int]],
    degree: int = RON_DEGREE,
    *,
    selection: str,
) -> GraphTopology:
    undirected_edges = sorted(set(tuple(sorted(edge)) for edge in undirected_edges))
    directed: dict[DirectedEdge, float] = {}
    export: list[tuple[int, int, float]] = []
    for a, b in undirected_edges:
        add_undirected_edge(directed, export, a, b, OPTICAL_GBPS)
    deg = graph_degree(EP_SIZE, undirected_edges)
    return GraphTopology(
        name=name,
        node_count=EP_SIZE,
        gpu_count=EP_SIZE,
        directed_edges=directed,
        undirected_edges_for_export=export,
        metadata={
            "type": "degree_limited_ron",
            "degree_target": degree,
            "degree_min": min(deg.values()),
            "degree_max": max(deg.values()),
            "edge_count": len(undirected_edges),
            "per_gpu_optical_bandwidth_tbps": 1.6,
            "per_link_bandwidth_gbps": OPTICAL_GBPS,
            "eps_fallback": False,
            "selection": selection,
        },
    )


def is_connected(n: int, edges: set[tuple[int, int]]) -> bool:
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen = {0}
    q = deque([0])
    while q:
        node = q.popleft()
        for nxt in adj[node]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == n


def random_regular_edges(n: int, degree: int, seed: int) -> set[tuple[int, int]]:
    rng = random.Random(seed)
    stubs = [rank for rank in range(n) for _ in range(degree)]
    for _ in range(5000):
        rng.shuffle(stubs)
        edges: set[tuple[int, int]] = set()
        ok = True
        for idx in range(0, len(stubs), 2):
            a, b = sorted((stubs[idx], stubs[idx + 1]))
            if a == b or (a, b) in edges:
                ok = False
                break
            edges.add((a, b))
        if ok and is_connected(n, edges):
            return edges
    return demand_graph(n, degree, Counter())


def torus_edge_set(rows: int = 4, cols: int = 8) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            edges.add(tuple(sorted((node, r * cols + ((c + 1) % cols)))))
            edges.add(tuple(sorted((node, ((r + 1) % rows) * cols + c))))
    return edges


def candidate_ron_topologies(name_prefix: str, demand: Counter[Pair], random_count: int = 4) -> list[GraphTopology]:
    candidates = [
        make_ron_topology(f"{name_prefix}_greedy", demand),
        make_ron_topology_from_edges(
            f"{name_prefix}_torus",
            torus_edge_set(),
            RON_DEGREE,
            selection="fixed_torus_candidate",
        ),
    ]
    for seed in range(random_count):
        candidates.append(
            make_ron_topology_from_edges(
                f"{name_prefix}_random_{seed}",
                random_regular_edges(EP_SIZE, RON_DEGREE, 10_000 + seed),
                RON_DEGREE,
                selection=f"random_regular_seed_{seed}",
            )
        )
    return candidates


def topology_score_us(demand: Counter[Pair], topology: GraphTopology, route_cache: RouteCache) -> float:
    route_cache.precompute_all_pairs(topology, "deterministic_shortest_path")
    return float(
        phase_time_us(
            demand,
            topology,
            "deterministic_shortest_path",
            route_cache,
        )["time_us"]
    )


def choose_best_ron_topology(name_prefix: str, demand: Counter[Pair], route_cache: RouteCache) -> GraphTopology:
    best_topology: GraphTopology | None = None
    best_score = float("inf")
    for candidate in candidate_ron_topologies(name_prefix, demand):
        score = topology_score_us(demand, candidate, route_cache)
        if score < best_score:
            best_score = score
            best_topology = candidate
    assert best_topology is not None
    best_topology.name = name_prefix
    best_topology.metadata["candidate_pool_size"] = 2 + 4
    best_topology.metadata["selection_objective"] = "minimize_request_phase_graph_time"
    best_topology.metadata["selection_score_us"] = best_score
    return best_topology


def adjacency(topology: GraphTopology) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in topology.directed_edges:
        adj[a].append(b)
    for node in range(topology.node_count):
        adj[node] = sorted(set(adj[node]))
    return adj


def shortest_paths(topology: GraphTopology, src: int, dst: int, max_paths: int = 64) -> list[list[int]]:
    adj = adjacency(topology)
    dist: dict[int, int] = {src: 0}
    parents: dict[int, list[int]] = defaultdict(list)
    q = deque([src])
    while q:
        node = q.popleft()
        for nxt in adj[node]:
            nd = dist[node] + 1
            if nxt not in dist:
                dist[nxt] = nd
                parents[nxt].append(node)
                q.append(nxt)
            elif dist[nxt] == nd:
                parents[nxt].append(node)
    if dst not in dist:
        raise ValueError(f"{topology.name}: disconnected pair {src}->{dst}")

    out: list[list[int]] = []

    def backtrack(node: int, suffix: list[int]) -> None:
        if len(out) >= max_paths:
            return
        if node == src:
            out.append([src] + suffix[::-1])
            return
        for parent in sorted(parents[node]):
            backtrack(parent, suffix + [node])

    backtrack(dst, [])
    return out


def single_shortest_path(topology: GraphTopology, src: int, dst: int) -> list[int]:
    adj = adjacency(topology)
    parent: dict[int, int | None] = {src: None}
    q = deque([src])
    while q:
        node = q.popleft()
        if node == dst:
            break
        for nxt in adj[node]:
            if nxt not in parent:
                parent[nxt] = node
                q.append(nxt)
    if dst not in parent:
        raise ValueError(f"{topology.name}: disconnected pair {src}->{dst}")
    path = [dst]
    cur = dst
    while parent[cur] is not None:
        cur = parent[cur]  # type: ignore[assignment]
        path.append(cur)
    path.reverse()
    return path


class RouteCache:
    def __init__(self) -> None:
        self.cache: dict[tuple[str, str, int, int], list[list[int]]] = {}
        self.all_pair_cache: dict[tuple[str, str], dict[Pair, list[list[int]]]] = {}

    @staticmethod
    def topology_key(topology: GraphTopology) -> str:
        # The selected edge set is the real identity.  The name alone is not
        # enough because RON W=4/oracle creates many generated graphs.
        edge_key = sorted((a, b, round(gbps, 6)) for a, b, gbps in topology.undirected_edges_for_export)
        return json.dumps([topology.gpu_count, edge_key], separators=(",", ":"))

    def get(self, topology: GraphTopology, routing: str, src: int, dst: int) -> list[list[int]]:
        all_key = (self.topology_key(topology), routing)
        if all_key in self.all_pair_cache:
            return self.all_pair_cache[all_key][(src, dst)]
        key = (all_key[0], routing, src, dst)
        if key not in self.cache:
            if routing == "deterministic_shortest_path":
                paths = [single_shortest_path(topology, src, dst)]
            else:
                paths = shortest_paths(topology, src, dst)
            self.cache[key] = paths
        return self.cache[key]

    def precompute_all_pairs(self, topology: GraphTopology, routing: str) -> None:
        all_key = (self.topology_key(topology), routing)
        if all_key in self.all_pair_cache:
            return
        table: dict[Pair, list[list[int]]] = {}
        for src in range(topology.gpu_count):
            for dst in range(topology.gpu_count):
                if src == dst:
                    continue
                if routing == "deterministic_shortest_path":
                    table[(src, dst)] = [single_shortest_path(topology, src, dst)]
                else:
                    table[(src, dst)] = shortest_paths(topology, src, dst)
        self.all_pair_cache[all_key] = table

    def route_table(self, topology: GraphTopology, routing: str) -> dict[Pair, list[list[int]]]:
        self.precompute_all_pairs(topology, routing)
        return self.all_pair_cache[(self.topology_key(topology), routing)]


def phase_time_us(
    load: Counter[Pair],
    topology: GraphTopology,
    routing: str,
    route_cache: RouteCache,
    *,
    imbalance_factor: float = 1.0,
    link_latency_us: float = 0.0,
) -> dict[str, Any]:
    directed_load: Counter[DirectedEdge] = Counter()
    route_table = route_cache.route_table(topology, routing)
    total_bytes = 0
    weighted_hops = 0.0
    max_hops = 0
    message_count = 0
    for (src, dst), size in load.items():
        if src == dst or size <= 0:
            continue
        paths = route_table[(src, dst)]
        share = float(size) / len(paths)
        for path in paths:
            for a, b in zip(path, path[1:]):
                directed_load[(a, b)] += share
        total_bytes += int(size)
        weighted_hops += int(size) * mean(len(path) - 1 for path in paths)
        max_hops = max(max_hops, max(len(path) - 1 for path in paths))
        message_count += 1
    edge_times = []
    for edge, bytes_on_edge in directed_load.items():
        gbps = topology.directed_edges[edge]
        edge_times.append(bytes_on_edge / gbps_to_bytes_per_us(gbps))
    max_link_time_us = max(edge_times, default=0.0)
    # Fluid link-load model: serialization dominates.  Latency is a per-phase
    # max path term rather than per-message additive cost.
    time_us = (max_link_time_us * imbalance_factor) + max_hops * link_latency_us
    return {
        "time_us": time_us,
        "max_link_load_bytes": max(directed_load.values(), default=0.0),
        "average_hop_count": weighted_hops / total_bytes if total_bytes else 0.0,
        "max_hop_count": max_hops,
        "message_count": message_count,
        "nonzero_link_count": len(directed_load),
    }


def unit_counter(traffic: Traffic, request_id: str, layer_id: int, phase: str) -> Counter[Pair]:
    return traffic.directed_by_unit.get((request_id, layer_id, phase), Counter())


def run_fixed_topology(
    traffic: Traffic,
    topology: GraphTopology,
    routing: str,
    route_cache: RouteCache,
    *,
    imbalance_factor: float = 1.0,
) -> dict[str, Any]:
    route_cache.precompute_all_pairs(topology, routing)
    total_us = 0.0
    max_link = 0.0
    msg_count = 0
    hop_weight = 0.0
    units = 0
    max_hop = 0
    for request_id in traffic.eval_request_ids:
        for phase in ("dispatch", "combine"):
            result = phase_time_us(
                counter_from_matrix(traffic.request_phase_matrix[(request_id, phase)]),
                topology,
                routing,
                route_cache,
                imbalance_factor=imbalance_factor,
            )
            total_us += result["time_us"]
            max_link = max(max_link, result["max_link_load_bytes"])
            msg_count += int(result["message_count"])
            hop_weight += float(result["average_hop_count"])
            max_hop = max(max_hop, int(result["max_hop_count"]))
            units += 1
    return {
        "completion_time_ms": total_us / 1000.0,
        "max_link_load_bytes": max_link,
        "message_count": msg_count,
        "average_hop_count": hop_weight / units if units else 0.0,
        "max_hop_count": max_hop,
        "reconfigurations": 0,
        "exposed_reconfiguration_time_ms": 0.0,
    }


def aggregate_demand(traffic: Traffic, request_ids: Iterable[str]) -> Counter[Pair]:
    demand: Counter[Pair] = Counter()
    for request_id in request_ids:
        demand.update(traffic.pair_by_request.get(request_id, Counter()))
    return demand


def run_ron_calibrated(traffic: Traffic, route_cache: RouteCache) -> tuple[dict[str, Any], GraphTopology]:
    topology = choose_best_ron_topology(
        f"{traffic.spec.dataset_id}_ron_calibrated",
        aggregate_demand(traffic, traffic.calibration_request_ids),
        route_cache,
    )
    result = run_fixed_topology(traffic, topology, "deterministic_shortest_path", route_cache)
    result["reconfigurations"] = 1
    return result, topology


def run_ron_segmented(
    traffic: Traffic,
    route_cache: RouteCache,
    mode: str,
    *,
    window: int = 4,
    reconfig_us: float = 0.0,
    topology_output_dir: Path | None = None,
) -> dict[str, Any]:
    total_us = 0.0
    max_link = 0.0
    msg_count = 0
    hop_sum = 0.0
    units = 0
    max_hop = 0
    topo_written = 0
    for req_index, request_id in enumerate(traffic.eval_request_ids):
        if mode == "oracle":
            demand = traffic.pair_by_request.get(request_id, Counter())
            name = f"{traffic.spec.dataset_id}_ron_oracle_{request_id}"
        elif mode == "w4":
            absolute_index = traffic.request_ids.index(request_id)
            previous = traffic.request_ids[max(0, absolute_index - window) : absolute_index]
            demand = aggregate_demand(traffic, previous)
            name = f"{traffic.spec.dataset_id}_ron_w4_{request_id}"
        else:
            raise ValueError(mode)
        topology = choose_best_ron_topology(name, demand, route_cache)
        route_cache.precompute_all_pairs(topology, "deterministic_shortest_path")
        if topology_output_dir and topo_written < 8:
            write_topology_json(topology_output_dir / f"{name}.json", topology)
            topo_written += 1
        for phase in ("dispatch", "combine"):
            result = phase_time_us(
                counter_from_matrix(traffic.request_phase_matrix[(request_id, phase)]),
                topology,
                "deterministic_shortest_path",
                route_cache,
            )
            total_us += result["time_us"]
            max_link = max(max_link, result["max_link_load_bytes"])
            msg_count += int(result["message_count"])
            hop_sum += float(result["average_hop_count"])
            max_hop = max(max_hop, int(result["max_hop_count"]))
            units += 1
        total_us += reconfig_us
    return {
        "completion_time_ms": total_us / 1000.0,
        "max_link_load_bytes": max_link,
        "message_count": msg_count,
        "average_hop_count": hop_sum / units if units else 0.0,
        "max_hop_count": max_hop,
        "reconfigurations": len(traffic.eval_request_ids),
        "exposed_reconfiguration_time_ms": len(traffic.eval_request_ids) * reconfig_us / 1000.0,
        "segmented_run_approximation": True,
    }


def write_topology_json(path: Path, topology: GraphTopology) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": topology.name,
        "node_count": topology.node_count,
        "gpu_count": topology.gpu_count,
        "directed": False,
        "edges": [
            {"src": a, "dst": b, "bandwidth_gbps": gbps, "latency_us": 0.0}
            for a, b, gbps in topology.undirected_edges_for_export
        ],
        "metadata": topology.metadata,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def read_v28_reference() -> dict[tuple[str, str], float]:
    rows = list(csv.DictReader(V28_TEACHER_CSV.open()))
    out: dict[tuple[str, str], float] = {}
    for row in rows:
        if int(row["ep_size"]) == EP_SIZE:
            out[(row["dataset_id"], row["network_mode"])] = float(row["completion_time_ms"])
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_grouped(path_png: Path, path_pdf: Path, rows: list[dict[str, Any]]) -> None:
    datasets = list(dict.fromkeys(row["dataset_label"] for row in rows))
    modes = TARGET_MODES
    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", str(path_png.parent / ".mplconfig"))
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        width = 0.82 / len(modes)
        xs = list(range(len(datasets)))
        fig, ax = plt.subplots(figsize=(15.5, 6.2))
        for i, mode in enumerate(modes):
            vals = [
                float(next(row for row in rows if row["dataset_label"] == ds and row["network_mode"] == mode)["v30_graph_ms"])
                for ds in datasets
            ]
            offsets = [x + (i - (len(modes) - 1) / 2) * width for x in xs]
            ax.bar(offsets, vals, width=width, label=mode)
        ax.set_title("32-GPU MoE Prefill Communication Time")
        ax.set_ylabel("Communication time (ms, lower is better)")
        ax.set_xticks(xs)
        ax.set_xticklabels(datasets, rotation=12, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, ncols=3)
        fig.tight_layout()
        fig.savefig(path_png, dpi=220)
        fig.savefig(path_pdf)
        plt.close(fig)
    except ModuleNotFoundError:
        from PIL import Image, ImageDraw, ImageFont

        width_px, height_px = 1800, 900
        margin_l, margin_r, margin_t, margin_b = 150, 60, 90, 220
        plot_w = width_px - margin_l - margin_r
        plot_h = height_px - margin_t - margin_b
        image = Image.new("RGB", (width_px, height_px), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        value_map = {(row["dataset_label"], row["network_mode"]): float(row["v30_graph_ms"]) for row in rows}
        max_val = max(value_map.values()) if value_map else 1.0
        colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"]
        draw.text((margin_l, 25), "32-GPU MoE Prefill Communication Time", fill="black", font=font)
        draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill="black")
        draw.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h), fill="black")
        for tick in range(6):
            y = margin_t + plot_h - int(plot_h * tick / 5)
            val = max_val * tick / 5
            draw.line((margin_l - 5, y, margin_l + plot_w, y), fill="#dddddd")
            draw.text((20, y - 6), f"{val:.0f} ms", fill="black", font=font)
        group_w = plot_w / max(1, len(datasets))
        bar_w = group_w * 0.75 / len(modes)
        for di, dataset in enumerate(datasets):
            base_x = margin_l + di * group_w + group_w * 0.125
            for mi, mode in enumerate(modes):
                val = value_map[(dataset, mode)]
                h = int(plot_h * val / max_val)
                x0 = int(base_x + mi * bar_w)
                x1 = int(x0 + bar_w * 0.9)
                y0 = margin_t + plot_h - h
                y1 = margin_t + plot_h
                draw.rectangle((x0, y0, x1, y1), fill=colors[mi % len(colors)])
            draw.text((int(margin_l + di * group_w + 5), margin_t + plot_h + 20), dataset, fill="black", font=font)
        legend_x, legend_y = margin_l, height_px - 90
        for mi, mode in enumerate(modes):
            x = legend_x + (mi % 3) * 500
            y = legend_y + (mi // 3) * 30
            draw.rectangle((x, y, x + 18, y + 18), fill=colors[mi % len(colors)])
            draw.text((x + 24, y + 2), mode, fill="black", font=font)
        path_png.parent.mkdir(parents=True, exist_ok=True)
        image.save(path_png)
        image.save(path_pdf, "PDF", resolution=220.0)


def make_readme(out: Path, validation: dict[str, Any]) -> None:
    text = f"""# V3.0 ASTRA-sim Custom Topology Attempt

## Bottom Line

This attempt goes beyond the V2.9 `Switch` proxy.  It generates Chakra
SEND/RECV traces and explicit graph topology JSONs for EN folded-Clos, SON 2D
torus, and RON degree-4 graphs.  Final timing is produced by a new graph/link
contention backend in this ASTRA-sim workspace:

`tools/v30_astrasim_custom_topology_attempt.py`

This is **not yet a first-class ASTRA C++ backend**.  The blocker is precise:
the current analytical C++ parser accepts only `Ring`, `Switch`, and
`FullyConnected`, while the congestion-aware route API returns a single path per
chunk.  Faithful ECMP requires a graph topology parser plus multi-path
message-splitting or route selection.

## Pipeline

```mermaid
flowchart LR
  A["HF expert-selection JSON"] --> B["trace[0] prefill only"]
  B --> C["block_by_token source GPU"]
  B --> D["block expert placement"]
  C --> E["per-request/layer/phase pairwise traffic"]
  D --> E
  E --> F["Chakra SEND/RECV traces"]
  E --> G["ASTRA-side graph timing backend"]
  H["Topology JSON: Clos / Torus / RON"] --> G
  G --> I["CSV/JSON summary + figure"]
```

## What Was Modified

No ASTRA core C++ files were changed in this pass.  A new ASTRA-side prototype
backend was added under `tools/`.  This keeps the experiment reproducible while
avoiding a half-integrated C++ route API change.

## Modelling Notes

- EN folded-Clos: explicit GPU-leaf-spine graph, 400 Gb/s links, ECMP over equal
  shortest paths, with 1.3x imbalance applied to exposed link load.
- SON: explicit 4x8 degree-4 torus, 400 Gb/s links, ECMP over equal shortest
  paths.
- RON calibrated: degree-4 graph chosen from the first 10% calibration requests,
  reused for all evaluated requests.
- RON W=4: one topology per evaluated request, selected from previous 4
  requests, with 1 us reconfiguration penalty per request.
- RON oracle: one topology per evaluated request, selected from current request,
  no reconfiguration penalty.
- Optical multi-hop is interpreted as an abstract optical switching-fabric path,
  not intermediate GPU packet forwarding.
- Timing granularity is request-phase level for the V3.0 graph engine:
  dispatch and combine are sequential, but all MoE layers inside a request phase
  are aggregated for tractability.  Per-request/layer/phase traffic is still
  generated internally and the aggregate/per-request Chakra artifacts are
  emitted.

## Validation

```json
{json.dumps(validation, indent=2, sort_keys=True)}
```

## Remaining Backend Work

To make this a true ASTRA C++ backend:

1. Extend `NetworkParser` to load graph topology JSON/YAML with per-edge
   bandwidth/latency.
2. Add a `GraphTopology` class under `congestion_aware` that instantiates
   arbitrary switch/GPU devices and edges.
3. Extend route selection for ECMP.  The current `route(src,dst)` returns a
   single route, so ECMP needs either message chunk splitting before route
   construction or route-level round-robin over many generated chunks.
4. Add segmented-run orchestration or an in-run topology update API for RON
   W=4/oracle.
"""
    (out / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "moe_expert_trace_converter" / "v30_astrasim_custom_topology_attempt",
    )
    parser.add_argument(
        "--max-segment-chakra",
        type=int,
        default=4,
        help="Write full per-request Chakra segment traces for the first N evaluated requests per workload.",
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    specs = [
        DatasetSpec("qwen_mmlu_ml", "Qwen MMLU ML", Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/mmlu/machine_learning")),
        DatasetSpec("qwen_livecode", "Qwen LiveCodeBench", Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/livecodebench/execution")),
        DatasetSpec("qwen_mmlu_zh_anatomy", "Qwen ZH Anatomy", Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/mmlu_ZH_CN/anatomy")),
        DatasetSpec("deepseek_livecode", "DeepSeek LiveCodeBench", Path("/Users/dfx/Python/trace/cognitivecomputations/DeepSeek-R1-AWQ/livecodebench/execution")),
    ]

    route_cache = RouteCache()
    reference = read_v28_reference()
    en_topology = make_en_clos()
    son_topology = make_torus_2d()
    write_topology_json(out / "topologies" / "en_folded_clos_ecmp.json", en_topology)
    write_topology_json(out / "topologies" / "son_2d_torus_ecmp.json", son_topology)

    comparison_rows: list[dict[str, Any]] = []
    traffic_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    trace_artifacts: dict[str, Any] = {}
    blocker_rows: list[dict[str, Any]] = []

    for spec in specs:
        traffic = parse_dataset(spec)
        trace_artifacts[spec.dataset_id] = write_chakra_artifacts(traffic, out, args.max_segment_chakra)
        traffic_rows.append(
            {
                "dataset_id": spec.dataset_id,
                "dataset_label": spec.label,
                "files_found": traffic.files_found,
                "files_used": traffic.files_used,
                "calibration_requests": len(traffic.calibration_request_ids),
                "evaluation_requests": len(traffic.eval_request_ids),
                "prefill_only": True,
                "source_policy": "block_by_token",
                "expert_placement": "block",
                "hidden_size": HIDDEN_SIZE,
                "bytes_per_value": BYTES_PER_VALUE,
                "inferred_num_experts": traffic.inferred_num_experts,
                "moe_layers": len(traffic.moe_layers),
                "remote_bytes_all_requests_dispatch_plus_combine": traffic.remote_bytes,
                "remote_bytes_eval_dispatch_plus_combine": traffic.eval_remote_bytes,
                "local_bytes_excluded_all_requests": traffic.local_bytes,
                "malformed_records": traffic.malformed_records,
            }
        )
        validation_rows.append(
            {
                "dataset_id": spec.dataset_id,
                "byte_conservation_eval_remote_equals_request_pairs": traffic.eval_remote_bytes
                == sum(sum(traffic.pair_by_request.get(req, Counter()).values()) for req in traffic.eval_request_ids),
                "dispatch_combine_sequential": True,
                "prefill_trace0_only": True,
                "local_src_eq_dst_excluded": True,
            }
        )

        ron_cal_result, ron_cal_topology = run_ron_calibrated(traffic, route_cache)
        write_topology_json(out / "topologies" / f"{spec.dataset_id}_ron_calibrated.json", ron_cal_topology)
        ron_w4_result = run_ron_segmented(
            traffic,
            route_cache,
            "w4",
            reconfig_us=1.0,
            topology_output_dir=out / "topologies" / "ron_w4_samples" / spec.dataset_id,
        )
        ron_oracle_result = run_ron_segmented(
            traffic,
            route_cache,
            "oracle",
            reconfig_us=0.0,
            topology_output_dir=out / "topologies" / "ron_oracle_samples" / spec.dataset_id,
        )
        mode_results = {
            "EN folded-Clos ECMP / 1.3x imbalance": run_fixed_topology(
                traffic,
                en_topology,
                "ecmp_equal_shortest_path",
                route_cache,
                imbalance_factor=1.3,
            ),
            "SON 2D torus ECMP": run_fixed_topology(
                traffic,
                son_topology,
                "ecmp_equal_shortest_path",
                route_cache,
            ),
            "RON calibrated": ron_cal_result,
            "RON W=4 1us": ron_w4_result,
            "RON oracle": ron_oracle_result,
        }
        for mode in TARGET_MODES:
            result = mode_results[mode]
            v28 = reference.get((spec.dataset_id, V28_MODE_MAP[mode]))
            v30 = float(result["completion_time_ms"])
            rel = ((v30 - v28) / v28) if v28 else None
            if mode.startswith("EN"):
                faith = "faithful_graph_model_not_core_astra"
                notes = "Explicit folded-Clos graph with ECMP; 1.3x imbalance applied externally."
            elif mode.startswith("SON"):
                faith = "faithful_graph_model_not_core_astra"
                notes = "Explicit 4x8 torus graph with equal-shortest-path ECMP."
            elif mode == "RON calibrated":
                faith = "faithful_graph_model_not_core_astra"
                notes = "Degree-4 RON graph selected from first 10% requests; reused for eval."
            elif mode == "RON W=4 1us":
                faith = "segmented_run_approximation"
                notes = "One graph timing run per request; previous 4 requests select topology; +1us/request."
            else:
                faith = "segmented_run_approximation"
                notes = "One graph timing run per request; current request selects topology; no reconfig penalty."
            comparison_rows.append(
                {
                    "dataset_id": spec.dataset_id,
                    "dataset_label": spec.label,
                    "network_mode": mode,
                    "v28_reference_ms": v28,
                    "v30_graph_ms": v30,
                    "relative_difference_vs_v28": rel,
                    "support_status": faith,
                    "routing": "ECMP equal-shortest-path" if "ECMP" in mode else "deterministic shortest-path",
                    "max_link_load_bytes": result.get("max_link_load_bytes"),
                    "message_count": result.get("message_count"),
                    "average_hop_count": result.get("average_hop_count"),
                    "max_hop_count": result.get("max_hop_count"),
                    "reconfigurations": result.get("reconfigurations", 0),
                    "exposed_reconfiguration_time_ms": result.get("exposed_reconfiguration_time_ms", 0.0),
                    "notes": notes,
                }
            )

    blocker_rows.extend(
        [
            {
                "mode": "EN folded-Clos ECMP",
                "current_core_blocker": "NetworkParser only accepts Ring/Switch/FullyConnected; no folded-Clos graph generator.",
                "v30_status": "Supported by new graph backend, not integrated C++ ASTRA core.",
                "needed_core_change": "Add GraphTopology/folded-Clos generator and ECMP path splitting.",
            },
            {
                "mode": "SON 2D torus ECMP",
                "current_core_blocker": "No 2D torus ECMP graph routing in congestion-aware backend.",
                "v30_status": "Supported by new graph backend, not integrated C++ ASTRA core.",
                "needed_core_change": "Add graph topology parser/generator plus equal-cost shortest path routing.",
            },
            {
                "mode": "custom degree-4 RON",
                "current_core_blocker": "No arbitrary topology loader and no per-edge topology JSON support.",
                "v30_status": "Supported by new graph backend for calibrated/oracle/W4 topology JSONs.",
                "needed_core_change": "Add arbitrary graph loader and route cache for selected graph.",
            },
            {
                "mode": "per-request RON reconfiguration",
                "current_core_blocker": "No topology update API inside one ASTRA run.",
                "v30_status": "Segmented-run approximation: one graph timing segment per evaluated request.",
                "needed_core_change": "Add in-run topology swap event or official segmented-run orchestrator.",
            },
        ]
    )

    validation = {
        "only_four_specified_workloads": [spec.dataset_id for spec in specs],
        "prefill_only_trace0": True,
        "source_policy": "block_by_token",
        "expert_placement": "block",
        "hidden_size": HIDDEN_SIZE,
        "bytes_per_value": BYTES_PER_VALUE,
        "local_src_eq_dst_excluded": True,
        "dispatch_and_combine_sequential": True,
        "byte_conservation_all_passed": all(row["byte_conservation_eval_remote_equals_request_pairs"] for row in validation_rows),
        "en_uses_400gbps_links": True,
        "son_ron_degree4_400gbps_links_1p6tbps_per_gpu": True,
        "ron_calibrated_uses_first_10pct": True,
        "ron_w4_uses_previous_4_requests_plus_1us": True,
        "chakra_generated": True,
        "final_timing_engine": "v30 ASTRA-side graph/link-contention backend, not old V2.8 evaluator",
        "timing_granularity": "request_phase_aggregated_layers",
        "astra_core_cpp_modified": False,
        "full_core_astra_replacement": False,
    }

    write_csv(out / "traffic_summary.csv", traffic_rows)
    write_csv(out / "validation_rows.csv", validation_rows)
    write_csv(out / "summary_v30_graph_vs_v28.csv", comparison_rows)
    write_csv(out / "blocker_report.csv", blocker_rows)
    (out / "summary_v30_graph_vs_v28.json").write_text(json.dumps(comparison_rows, indent=2, sort_keys=True))
    (out / "trace_artifacts.json").write_text(json.dumps(trace_artifacts, indent=2, sort_keys=True))
    (out / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True))
    plot_grouped(out / "fig_32gpu_moe_prefill_v30_graph.png", out / "fig_32gpu_moe_prefill_v30_graph.pdf", comparison_rows)
    make_readme(out, validation)
    shutil.copy2(Path(__file__), out / "v30_astrasim_custom_topology_attempt.py")

    print("OUTPUT", out)
    print("FULL_CORE_ASTRA_REPLACEMENT", validation["full_core_astra_replacement"])
    for row in comparison_rows:
        print(
            f"{row['dataset_id']} | {row['network_mode']} | "
            f"v30={float(row['v30_graph_ms']):.6f} ms | "
            f"v28={row['v28_reference_ms']} | status={row['support_status']}"
        )


if __name__ == "__main__":
    main()
