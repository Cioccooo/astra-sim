#!/usr/bin/env python3
"""More realistic MoE hot-pair lambda-CoCo-lite feasibility test.

This keeps ASTRA-sim stock. ASTRA executes true pairwise Chakra send/recv
AllToAllv traces under a uniform Switch network. Per-pair wavelength / OCS
capacity policies are evaluated by an external fixed-budget capacity model,
then scaled to the ASTRA uniform baseline.

Approximations are explicit:
  * Synthetic MixNet-informed source-destination traffic, not production traces.
  * Dispatch and combine are modeled as two sequential AllToAllv phases.
  * OCS capacity is a finite number of circuits/wavelength slots per rank.
  * A uniform EPS overflow path carries traffic not favored by OCS circuits.
  * Sender and receiver row/column capacity budgets are fixed for every policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
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

from et_def.et_def_pb2 import (  # type: ignore  # noqa: E402
    AttributeProto as ChakraAttr,
    GlobalMetadata,
    Node as ChakraNode,
    NodeType,
)
from protolib import encodeMessage as encode_message  # type: ignore  # noqa: E402


ASTRA_BIN = (
    REPO_ROOT
    / "build"
    / "astra_analytical"
    / "build"
    / "bin"
    / "AstraSim_Analytical_Congestion_Unaware"
)
SYSTEM_CONFIG = REPO_ROOT / "examples" / "system" / "native_collectives" / "HGX-H100-validated.json"
REMOTE_MEMORY_CONFIG = (
    REPO_ROOT / "examples" / "remote_memory" / "analytical" / "no_memory_expansion.json"
)


@dataclass(frozen=True)
class MoeConfig:
    name: str
    ranks: int
    ep_degree: int
    experts: int
    top_k: int
    seq_len: int
    microbatch: int
    hidden_size: int
    bytes_per_value: int

    @property
    def phase_network_bytes(self) -> int:
        # Approximate remote EP payload for one dispatch or one combine phase.
        # Token-to-local-expert traffic is excluded by the (ranks-1)/ranks term.
        token_assignments = self.ranks * self.microbatch * self.seq_len * self.top_k
        payload = token_assignments * self.hidden_size * self.bytes_per_value
        return int(payload * (self.ranks - 1) / self.ranks)


@dataclass
class AstraPhaseRun:
    phase: str
    exit_code: int
    finished_ranks: int
    max_cycles: int | None
    log_path: str


@dataclass
class PolicyPhaseMetrics:
    time_model: float
    scaled_cycles: float
    speedup: float
    hot_p95_scaled_cycles: float
    cold_p95_scaled_cycles: float
    overflow_p95_scaled_cycles: float


@dataclass
class PolicyMetrics:
    dispatch: PolicyPhaseMetrics
    combine: PolicyPhaseMetrics
    ep_total_cycles: float
    ep_speedup: float
    full_iter_speedup_30: float
    full_iter_speedup_50: float
    full_iter_speedup_80: float


@dataclass
class CaseResult:
    config: str
    ranks: int
    ep_degree: int
    experts: int
    top_k: int
    skew: str
    matrix_kind: str
    total_phase_bytes: int
    nonzero_pairs: int
    top_1pct_byte_share: float
    max_source_byte_share: float
    max_dest_byte_share: float
    dispatch_finished: str
    combine_finished: str
    true_pairwise_alltoallv: bool
    uniform: PolicyMetrics
    hot_pair: PolicyMetrics
    mixnet_greedy: PolicyMetrics
    oracle: PolicyMetrics


def offdiag_mask(n: int) -> np.ndarray:
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    return mask


def normalize_offdiag(matrix: np.ndarray, total_bytes: int) -> np.ndarray:
    mask = offdiag_mask(matrix.shape[0])
    matrix = matrix.astype(float)
    matrix[~mask] = 0.0
    matrix *= float(total_bytes) / float(matrix.sum())
    return matrix


def integerize_matrix(matrix: np.ndarray, total_bytes: int) -> np.ndarray:
    flat = matrix.reshape(-1)
    floors = np.floor(flat).astype(int)
    remainder = int(total_bytes - floors.sum())
    if remainder > 0:
        fractions = flat - floors
        for idx in np.argsort(fractions)[-remainder:]:
            floors[idx] += 1
    out = floors.reshape(matrix.shape)
    np.fill_diagonal(out, 0)
    return out


def make_uniform_matrix(n: int, total_bytes: int) -> np.ndarray:
    return normalize_offdiag(offdiag_mask(n).astype(float), total_bytes)


def make_pair_zipf_matrix(n: int, alpha: float, seed: int, total_bytes: int) -> np.ndarray:
    mask = offdiag_mask(n)
    weights = 1.0 / np.power(np.arange(1, int(mask.sum()) + 1, dtype=float), alpha)
    shuffled = list(weights)
    random.Random(seed).shuffle(shuffled)
    matrix = np.zeros((n, n), dtype=float)
    matrix[mask] = shuffled
    return normalize_offdiag(matrix, total_bytes)


def uniform_capacity(n: int) -> np.ndarray:
    return offdiag_mask(n).astype(float) / float(n - 1)


def circuit_capacity_from_counts(
    circuit_counts: np.ndarray,
    slots_per_rank: int,
    overflow_fraction: float,
) -> np.ndarray:
    n = circuit_counts.shape[0]
    overflow = overflow_fraction * uniform_capacity(n)
    ocs = (1.0 - overflow_fraction) * circuit_counts.astype(float) / float(slots_per_rank)
    return overflow + ocs


def weighted_derangement(weights: np.ndarray) -> np.ndarray:
    """Return one src->dst assignment with no self edges.

    This is a greedy max-weight derangement, not an exact Hungarian solver. It
    is sufficient for this feasibility pass and, importantly, always preserves
    one outgoing and one incoming circuit per rank for a wavelength slot.
    """
    n = weights.shape[0]
    assignment = np.full(n, -1, dtype=int)
    available: set[int] = set(range(n))
    row_order = sorted(
        range(n),
        key=lambda src: float(np.max(np.delete(weights[src], src))),
        reverse=True,
    )

    for src in row_order:
        candidates = [dst for dst in available if dst != src]
        if not candidates:
            # The only remaining destination is the diagonal. Swap with any
            # prior row; this preserves a perfect matching and avoids self-edge.
            diagonal_dst = src
            previous = next(row for row in range(n) if assignment[row] >= 0)
            previous_dst = int(assignment[previous])
            assignment[previous] = diagonal_dst
            assignment[src] = previous_dst
            available.remove(diagonal_dst)
            continue
        dst = max(candidates, key=lambda item: float(weights[src, item]))
        assignment[src] = dst
        available.remove(dst)

    if np.any(assignment < 0) or any(src == dst for src, dst in enumerate(assignment)):
        raise RuntimeError("Failed to build non-diagonal assignment")
    return assignment


def allocate_volume_ranked_circuits(
    matrix: np.ndarray,
    slots_per_rank: int,
    overflow_fraction: float,
) -> np.ndarray:
    """lambda-CoCo-lite: each wavelength slot follows pair byte volume."""
    n = matrix.shape[0]
    counts = np.zeros((n, n), dtype=int)
    weights = matrix.astype(float).copy()
    weights[~offdiag_mask(n)] = -1.0
    for _ in range(slots_per_rank):
        assignment = weighted_derangement(weights)
        for src, dst in enumerate(assignment):
            counts[src, dst] += 1
    return circuit_capacity_from_counts(counts, slots_per_rank, overflow_fraction)


def allocate_mixnet_greedy_circuits(
    matrix: np.ndarray,
    slots_per_rank: int,
    overflow_fraction: float,
) -> np.ndarray:
    """MixNet-style bottleneck greedy: each slot targets current worst pairs."""
    n = matrix.shape[0]
    counts = np.zeros((n, n), dtype=int)
    base = overflow_fraction * uniform_capacity(n)
    circuit_unit = (1.0 - overflow_fraction) / float(slots_per_rank)
    for _ in range(slots_per_rank):
        capacity = base + counts * circuit_unit
        score = np.divide(matrix, capacity, out=np.zeros_like(matrix, dtype=float), where=capacity > 0)
        score[~offdiag_mask(n)] = -1.0
        assignment = weighted_derangement(score)
        for src, dst in enumerate(assignment):
            counts[src, dst] += 1
    return circuit_capacity_from_counts(counts, slots_per_rank, overflow_fraction)


def transfer_time(matrix: np.ndarray, capacity: np.ndarray) -> float:
    active = matrix > 0
    return float(np.max(matrix[active] / capacity[active]))


def oracle_time(matrix: np.ndarray) -> float:
    return float(max(matrix.sum(axis=1).max(), matrix.sum(axis=0).max()))


def p95(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, 95))


def classify_pair_sets(matrix: np.ndarray, capacity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = matrix > 0
    bytes_active = matrix[active]
    top_count = max(1, int(math.ceil(0.01 * len(bytes_active))))
    active_indices = np.argwhere(active)
    order = np.argsort(bytes_active)
    hot_pairs = active_indices[order[-top_count:]]
    cold_pairs = active_indices[order[: max(1, len(order) // 2)]]
    # Overflow-dominant means no extra OCS circuit beyond the uniform overflow.
    min_cap = capacity[active].min()
    overflow_pairs = active_indices[np.isclose(capacity[active], min_cap)]
    return hot_pairs, cold_pairs, overflow_pairs


def phase_policy_metrics(
    matrix: np.ndarray,
    capacity: np.ndarray | None,
    uniform_model_time: float,
    astra_uniform_cycles: int,
) -> PolicyPhaseMetrics:
    if capacity is None:
        model_time = oracle_time(matrix)
        pair_times = np.full(matrix.shape, model_time, dtype=float)
        hot_pairs, cold_pairs, overflow_pairs = classify_pair_sets(matrix, uniform_capacity(matrix.shape[0]))
    else:
        model_time = transfer_time(matrix, capacity)
        pair_times = np.divide(matrix, capacity, out=np.zeros_like(matrix, dtype=float), where=capacity > 0)
        hot_pairs, cold_pairs, overflow_pairs = classify_pair_sets(matrix, capacity)

    scale = astra_uniform_cycles / uniform_model_time

    def scaled_p95(pairs: np.ndarray) -> float:
        return p95(np.array([pair_times[src, dst] for src, dst in pairs])) * scale

    return PolicyPhaseMetrics(
        time_model=model_time,
        scaled_cycles=model_time * scale,
        speedup=uniform_model_time / model_time,
        hot_p95_scaled_cycles=scaled_p95(hot_pairs),
        cold_p95_scaled_cycles=scaled_p95(cold_pairs),
        overflow_p95_scaled_cycles=scaled_p95(overflow_pairs),
    )


def full_iter_speedup(ep_speedup: float, ep_fraction: float) -> float:
    return 1.0 / ((1.0 - ep_fraction) + ep_fraction / ep_speedup)


def policy_metrics(
    dispatch: PolicyPhaseMetrics,
    combine: PolicyPhaseMetrics,
    uniform_total_cycles: float,
) -> PolicyMetrics:
    total = dispatch.scaled_cycles + combine.scaled_cycles
    ep_speedup = uniform_total_cycles / total
    return PolicyMetrics(
        dispatch=dispatch,
        combine=combine,
        ep_total_cycles=total,
        ep_speedup=ep_speedup,
        full_iter_speedup_30=full_iter_speedup(ep_speedup, 0.30),
        full_iter_speedup_50=full_iter_speedup(ep_speedup, 0.50),
        full_iter_speedup_80=full_iter_speedup(ep_speedup, 0.80),
    )


def add_attr(node: ChakraNode, name: str, value: int | bool) -> None:
    if isinstance(value, bool):
        node.attr.append(ChakraAttr(name=name, bool_val=value))
    else:
        node.attr.append(ChakraAttr(name=name, uint64_val=int(value)))


def write_pairwise_trace(matrix: np.ndarray, trace_dir: Path, prefix: str) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    n = matrix.shape[0]
    nodes_by_rank: dict[int, list[ChakraNode]] = {rank: [] for rank in range(n)}
    next_id = {rank: 1 for rank in range(n)}
    tag = 1
    for src in range(n):
        for dst in range(n):
            size = int(matrix[src, dst])
            if src == dst or size <= 0:
                continue

            send = ChakraNode()
            send.id = next_id[src]
            next_id[src] += 1
            send.name = f"EP_ALLTOALLV_SEND_{src}_to_{dst}_tag{tag}"
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
            recv.name = f"EP_ALLTOALLV_RECV_{src}_to_{dst}_tag{tag}"
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


def write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["src", "dst", "bytes"])
        for src in range(matrix.shape[0]):
            for dst in range(matrix.shape[1]):
                if src != dst and matrix[src, dst] > 0:
                    writer.writerow([src, dst, int(matrix[src, dst])])


def write_network_config(path: Path, n: int, bandwidth_gbps: float, latency_ns: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "topology: [ Switch ]",
                f"npus_count: [ {n} ]",
                f"bandwidth: [ {bandwidth_gbps / 8.0:.6f} ]",
                f"latency: [ {latency_ns:.6f} ]",
                "",
            ]
        )
    )


def run_astra(prefix_path: Path, network_config: Path, log_path: Path, phase: str) -> AstraPhaseRun:
    cmd = [
        str(ASTRA_BIN),
        f"--workload-configuration={prefix_path}",
        f"--system-configuration={SYSTEM_CONFIG}",
        f"--network-configuration={network_config}",
        f"--remote-memory-configuration={REMOTE_MEMORY_CONFIG}",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout)
    cycles = [int(match) for match in re.findall(r"finished, ([0-9]+) cycles", proc.stdout)]
    return AstraPhaseRun(
        phase=phase,
        exit_code=proc.returncode,
        finished_ranks=len(cycles),
        max_cycles=max(cycles) if cycles else None,
        log_path=str(log_path),
    )


def run_case(
    config: MoeConfig,
    case_name: str,
    skew: str,
    matrix_kind: str,
    dispatch_matrix: np.ndarray,
    output_dir: Path,
    network_config: Path,
    slots_per_rank: int,
    overflow_fraction: float,
) -> CaseResult:
    combine_matrix = dispatch_matrix.T.copy()
    case_dir = output_dir / "traces" / config.name / case_name
    prefix = "moe_ep_alltoallv"

    phase_runs: dict[str, AstraPhaseRun] = {}
    for phase, matrix in (("dispatch", dispatch_matrix), ("combine", combine_matrix)):
        phase_dir = case_dir / phase
        write_matrix_csv(phase_dir / "traffic_matrix.csv", matrix)
        write_pairwise_trace(matrix, phase_dir, prefix)
        phase_runs[phase] = run_astra(
            phase_dir / prefix,
            network_config,
            output_dir / "logs" / config.name / f"{case_name}.{phase}.astra.log",
            phase,
        )

    if phase_runs["dispatch"].max_cycles is None or phase_runs["combine"].max_cycles is None:
        raise RuntimeError(f"ASTRA failed for {config.name}/{case_name}")

    capacities = {
        "uniform": uniform_capacity(config.ranks),
        "hot_pair": allocate_volume_ranked_circuits(
            dispatch_matrix, slots_per_rank, overflow_fraction
        ),
        "mixnet_greedy": allocate_mixnet_greedy_circuits(
            dispatch_matrix, slots_per_rank, overflow_fraction
        ),
        "oracle": None,
    }

    dispatch_uniform_model = transfer_time(dispatch_matrix, capacities["uniform"])
    combine_uniform_model = transfer_time(combine_matrix, capacities["uniform"].T)

    phase_metrics: dict[str, dict[str, PolicyPhaseMetrics]] = {}
    for policy, cap in capacities.items():
        dispatch_cap = None if cap is None else cap
        combine_cap = None if cap is None else cap.T
        phase_metrics[policy] = {
            "dispatch": phase_policy_metrics(
                dispatch_matrix,
                dispatch_cap,
                dispatch_uniform_model,
                phase_runs["dispatch"].max_cycles,
            ),
            "combine": phase_policy_metrics(
                combine_matrix,
                combine_cap,
                combine_uniform_model,
                phase_runs["combine"].max_cycles,
            ),
        }

    uniform_total = (
        phase_metrics["uniform"]["dispatch"].scaled_cycles
        + phase_metrics["uniform"]["combine"].scaled_cycles
    )

    active_bytes = dispatch_matrix[dispatch_matrix > 0]
    top_count = max(1, int(math.ceil(0.01 * len(active_bytes))))
    true_alltoallv = len(set(active_bytes.astype(int).tolist())) > 1 or matrix_kind == "uniform"
    return CaseResult(
        config=config.name,
        ranks=config.ranks,
        ep_degree=config.ep_degree,
        experts=config.experts,
        top_k=config.top_k,
        skew=skew,
        matrix_kind=matrix_kind,
        total_phase_bytes=int(dispatch_matrix.sum()),
        nonzero_pairs=int((dispatch_matrix > 0).sum()),
        top_1pct_byte_share=float(np.sort(active_bytes)[-top_count:].sum() / dispatch_matrix.sum()),
        max_source_byte_share=float(dispatch_matrix.sum(axis=1).max() / dispatch_matrix.sum()),
        max_dest_byte_share=float(dispatch_matrix.sum(axis=0).max() / dispatch_matrix.sum()),
        dispatch_finished=f"{phase_runs['dispatch'].finished_ranks}/{config.ranks}",
        combine_finished=f"{phase_runs['combine'].finished_ranks}/{config.ranks}",
        true_pairwise_alltoallv=true_alltoallv,
        uniform=policy_metrics(
            phase_metrics["uniform"]["dispatch"],
            phase_metrics["uniform"]["combine"],
            uniform_total,
        ),
        hot_pair=policy_metrics(
            phase_metrics["hot_pair"]["dispatch"],
            phase_metrics["hot_pair"]["combine"],
            uniform_total,
        ),
        mixnet_greedy=policy_metrics(
            phase_metrics["mixnet_greedy"]["dispatch"],
            phase_metrics["mixnet_greedy"]["combine"],
            uniform_total,
        ),
        oracle=policy_metrics(
            phase_metrics["oracle"]["dispatch"],
            phase_metrics["oracle"]["combine"],
            uniform_total,
        ),
    )


def make_cases(config: MoeConfig, seed: int) -> Iterable[tuple[str, str, str, np.ndarray]]:
    yield (
        "uniform",
        "uniform",
        "uniform",
        integerize_matrix(make_uniform_matrix(config.ranks, config.phase_network_bytes), config.phase_network_bytes),
    )
    for alpha in (0.8, 1.2, 1.5):
        matrix = make_pair_zipf_matrix(
            config.ranks,
            alpha,
            seed + int(alpha * 100),
            config.phase_network_bytes,
        )
        yield (
            f"pair_zipf_{alpha:.1f}",
            f"Zipf alpha={alpha:.1f}",
            "pair_zipf",
            integerize_matrix(matrix, config.phase_network_bytes),
        )


def verdict(result: CaseResult) -> str:
    gain_50 = (result.hot_pair.full_iter_speedup_50 - 1.0) * 100.0
    if result.matrix_kind == "uniform":
        return "drop/no skew"
    if gain_50 < 5.0:
        return "drop"
    if gain_50 < 15.0:
        return "keep serious"
    return "strong continue"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "lambda_coco_moe_realistic")
    parser.add_argument("--ranks", type=int, default=64)
    parser.add_argument("--config", choices=["mixtral", "qwen"], default="mixtral")
    parser.add_argument("--bandwidth-gbps", type=float, default=800.0)
    parser.add_argument("--latency-ns", type=float, default=100.0)
    parser.add_argument("--slots-per-rank", type=int, default=8)
    parser.add_argument("--overflow-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260514)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> MoeConfig:
    if args.config == "mixtral":
        return MoeConfig(
            name=f"mixtral_like_{args.ranks}ep",
            ranks=args.ranks,
            ep_degree=args.ranks,
            experts=args.ranks,
            top_k=2,
            seq_len=4096,
            microbatch=4,
            hidden_size=4096,
            bytes_per_value=2,
        )
    return MoeConfig(
        name=f"qwen_moe_like_{args.ranks}ep",
        ranks=args.ranks,
        ep_degree=args.ranks,
        experts=args.ranks * 2,
        top_k=8,
        seq_len=4096,
        microbatch=1,
        hidden_size=4096,
        bytes_per_value=2,
    )


def main() -> None:
    args = parse_args()
    if not ASTRA_BIN.exists():
        raise SystemExit(f"Missing ASTRA binary: {ASTRA_BIN}")

    config = build_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs" / config.name).mkdir(parents=True, exist_ok=True)
    network_config = args.output_dir / "configs" / f"switch_{config.ranks}.yml"
    write_network_config(network_config, config.ranks, args.bandwidth_gbps, args.latency_ns)

    results = [
        run_case(
            config=config,
            case_name=case_name,
            skew=skew,
            matrix_kind=matrix_kind,
            dispatch_matrix=matrix,
            output_dir=args.output_dir,
            network_config=network_config,
            slots_per_rank=args.slots_per_rank,
            overflow_fraction=args.overflow_fraction,
        )
        for case_name, skew, matrix_kind, matrix in make_cases(config, args.seed)
    ]

    summary = {
        "approximations": [
            "Synthetic MixNet-informed pairwise skew; no public production router trace is used.",
            "ASTRA runs true pairwise Chakra send/recv AllToAllv traces under a uniform Switch network.",
            "Hot-pair, MixNet-greedy, and oracle policies are evaluated by an external capacity model and scaled to ASTRA uniform cycles.",
            "Capacity model fixes sender row sums and receiver column sums for every policy.",
            f"OCS budget is {args.slots_per_rank} circuit/wavelength slots per rank plus {args.overflow_fraction:.2f} uniform EPS overflow path.",
            "Dispatch and combine are sequential; combine uses the transpose of dispatch traffic and transpose of the chosen capacity.",
        ],
        "args": {
            **vars(args),
            "output_dir": str(args.output_dir),
            "network_config": str(network_config),
        },
        "config": asdict(config),
        "results": [asdict(result) for result in results],
    }
    summary_path = args.output_dir / f"{config.name}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(
        "config\tranks\tep\tskew\tuniform\thot_pair\tmixnet_greedy\toracle\t"
        "ep_speedup\tfull30\tfull50\tfull80\tverdict"
    )
    for result in results:
        print(
            f"{result.config}\t{result.ranks}\t{result.ep_degree}\t{result.skew}\t"
            f"{result.uniform.ep_total_cycles:.1f}\t"
            f"{result.hot_pair.ep_total_cycles:.1f}\t"
            f"{result.mixnet_greedy.ep_total_cycles:.1f}\t"
            f"{result.oracle.ep_total_cycles:.1f}\t"
            f"{result.hot_pair.ep_speedup:.4f}\t"
            f"{result.hot_pair.full_iter_speedup_30:.4f}\t"
            f"{result.hot_pair.full_iter_speedup_50:.4f}\t"
            f"{result.hot_pair.full_iter_speedup_80:.4f}\t"
            f"{verdict(result)}"
        )
    print(f"summary_json\t{summary_path}")


if __name__ == "__main__":
    main()
