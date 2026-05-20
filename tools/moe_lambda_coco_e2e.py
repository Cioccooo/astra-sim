#!/usr/bin/env python3
"""Minimal end-to-end feasibility test for skew-based MoE lambda-CoCo-lite.

Pipeline:
  synthetic MoE router matrix -> pairwise AllToAllv Chakra send/recv trace
  -> stock ASTRA-sim run for the real trace -> analytical wavelength/circuit
  allocation model for uniform, hot-pair-biased, hot-expert-only, and oracle.

The allocation model is intentionally outside ASTRA-sim because the current
analytical backend has uniform per-topology bandwidth and cannot express
per-source-destination wavelength/circuit capacities without backend changes.
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

from et_def.et_def_pb2 import (  # type: ignore
    AttributeProto as ChakraAttr,
    GlobalMetadata,
    Node as ChakraNode,
    NodeType,
)
from protolib import encodeMessage as encode_message  # type: ignore


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


@dataclass
class AstraRun:
    exit_code: int
    finished_ranks: int
    max_cycles: int | None
    log_path: str


@dataclass
class AllocationMetrics:
    name: str
    ep_time_model: float
    ep_time_scaled_cycles: float
    speedup_vs_uniform: float
    hot_p95_model: float
    cold_p95_model: float
    overflow_capacity_share: float


@dataclass
class CaseResult:
    case: str
    skew: str
    matrix_kind: str
    astra_uniform_cycles: int | None
    finished_ranks: int
    all_ranks_finished: bool
    true_pairwise_alltoallv: bool
    total_bytes: int
    nonzero_pairs: int
    top_1pct_byte_share: float
    max_source_byte_share: float
    max_dest_byte_share: float
    uniform: AllocationMetrics
    hot_pair: AllocationMetrics
    hot_expert: AllocationMetrics
    oracle: AllocationMetrics


def offdiag_mask(n: int) -> np.ndarray:
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    return mask


def normalize_offdiag(matrix: np.ndarray, total_bytes: int) -> np.ndarray:
    mask = offdiag_mask(matrix.shape[0])
    matrix = matrix.astype(float)
    matrix[~mask] = 0.0
    current = matrix.sum()
    if current <= 0:
        raise ValueError("matrix has no off-diagonal traffic")
    matrix *= float(total_bytes) / current
    return matrix


def make_uniform_matrix(n: int, total_bytes: int) -> np.ndarray:
    matrix = offdiag_mask(n).astype(float)
    return normalize_offdiag(matrix, total_bytes)


def make_pair_zipf_matrix(n: int, alpha: float, seed: int, total_bytes: int) -> np.ndarray:
    mask = offdiag_mask(n)
    count = int(mask.sum())
    weights = 1.0 / np.power(np.arange(1, count + 1, dtype=float), alpha)
    rng = random.Random(seed)
    shuffled = list(weights)
    rng.shuffle(shuffled)
    matrix = np.zeros((n, n), dtype=float)
    matrix[mask] = shuffled
    return normalize_offdiag(matrix, total_bytes)


def make_expert_zipf_matrix(n: int, alpha: float, total_bytes: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=float)
    global_expert_probs = 1.0 / np.power(np.arange(1, n + 1, dtype=float), alpha)
    global_expert_probs /= global_expert_probs.sum()
    for src in range(n):
        row = global_expert_probs.copy()
        row[src] = 0.0
        row /= row.sum()
        matrix[src, :] = row
    return normalize_offdiag(matrix, total_bytes)


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


def masked_sinkhorn(weights: np.ndarray, iterations: int = 800) -> np.ndarray:
    mask = offdiag_mask(weights.shape[0])
    capacity = np.where(mask, np.maximum(weights, 1e-30), 0.0).astype(float)
    for _ in range(iterations):
        row_sums = capacity.sum(axis=1)
        capacity *= np.where(row_sums > 0, 1.0 / row_sums, 0.0)[:, None]
        capacity[~mask] = 0.0
        col_sums = capacity.sum(axis=0)
        capacity *= np.where(col_sums > 0, 1.0 / col_sums, 0.0)[None, :]
        capacity[~mask] = 0.0
    return capacity


def uniform_capacity(n: int) -> np.ndarray:
    capacity = offdiag_mask(n).astype(float) / float(n - 1)
    return capacity


def hot_pair_capacity(matrix: np.ndarray, overflow_fraction: float, floor: float) -> np.ndarray:
    normalized = matrix / max(float(matrix[matrix > 0].mean()), 1.0)
    biased = masked_sinkhorn(normalized + floor)
    return overflow_fraction * uniform_capacity(matrix.shape[0]) + (1.0 - overflow_fraction) * biased


def hot_expert_capacity(matrix: np.ndarray, overflow_fraction: float, floor: float) -> np.ndarray:
    # Destination-only weights intentionally ignore source-destination pair skew.
    # Under fixed receiver capacity, Sinkhorn largely collapses this to uniform.
    col_weights = matrix.sum(axis=0)
    weights = np.tile(col_weights[None, :], (matrix.shape[0], 1))
    biased = masked_sinkhorn(weights + floor)
    return overflow_fraction * uniform_capacity(matrix.shape[0]) + (1.0 - overflow_fraction) * biased


def transfer_time(matrix: np.ndarray, capacity: np.ndarray) -> float:
    active = matrix > 0
    return float(np.max(matrix[active] / capacity[active]))


def oracle_time(matrix: np.ndarray) -> float:
    return float(max(matrix.sum(axis=1).max(), matrix.sum(axis=0).max()))


def p95(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, 95))


def allocation_metrics(
    name: str,
    matrix: np.ndarray,
    capacity: np.ndarray | None,
    uniform_model_time: float,
    astra_uniform_cycles: int,
    overflow_fraction: float,
) -> AllocationMetrics:
    if capacity is None:
        model_time = oracle_time(matrix)
        pair_times = np.full(int((matrix > 0).sum()), model_time, dtype=float)
    else:
        model_time = transfer_time(matrix, capacity)
        pair_times = matrix[matrix > 0] / capacity[matrix > 0]

    scale = float(astra_uniform_cycles) / uniform_model_time
    bytes_active = matrix[matrix > 0]
    top_count = max(1, int(math.ceil(0.01 * len(bytes_active))))
    order = np.argsort(bytes_active)
    hot_idx = order[-top_count:]
    cold_idx = order[: max(1, len(order) // 2)]
    return AllocationMetrics(
        name=name,
        ep_time_model=model_time,
        ep_time_scaled_cycles=model_time * scale,
        speedup_vs_uniform=uniform_model_time / model_time,
        hot_p95_model=p95(pair_times[hot_idx]),
        cold_p95_model=p95(pair_times[cold_idx]),
        overflow_capacity_share=overflow_fraction if capacity is not None else 0.0,
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


def write_network_config(path: Path, n: int, bandwidth_gbps: float, latency_ns: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bandwidth_gb_per_s = bandwidth_gbps / 8.0
    path.write_text(
        "\n".join(
            [
                "topology: [ Switch ]",
                f"npus_count: [ {n} ]",
                f"bandwidth: [ {bandwidth_gb_per_s:.6f} ]",
                f"latency: [ {latency_ns:.6f} ]",
                "",
            ]
        )
    )


def run_astra(prefix_path: Path, network_config: Path, log_path: Path) -> AstraRun:
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
    return AstraRun(
        exit_code=proc.returncode,
        finished_ranks=len(cycles),
        max_cycles=max(cycles) if cycles else None,
        log_path=str(log_path),
    )


def write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["src", "dst", "bytes"])
        for src in range(matrix.shape[0]):
            for dst in range(matrix.shape[1]):
                if src != dst and matrix[src, dst] > 0:
                    writer.writerow([src, dst, int(matrix[src, dst])])


def summarize_case(
    case: str,
    skew: str,
    matrix_kind: str,
    matrix: np.ndarray,
    astra: AstraRun,
    overflow_fraction: float,
    floor: float,
) -> CaseResult:
    if astra.max_cycles is None:
        raise RuntimeError(f"ASTRA run failed for {case}; see {astra.log_path}")

    uni_cap = uniform_capacity(matrix.shape[0])
    hot_pair_cap = hot_pair_capacity(matrix, overflow_fraction, floor)
    hot_expert_cap = hot_expert_capacity(matrix, overflow_fraction, floor)
    uniform_model_time = transfer_time(matrix, uni_cap)

    uniform = allocation_metrics(
        "uniform", matrix, uni_cap, uniform_model_time, astra.max_cycles, overflow_fraction=1.0
    )
    hot_pair = allocation_metrics(
        "hot_pair",
        matrix,
        hot_pair_cap,
        uniform_model_time,
        astra.max_cycles,
        overflow_fraction=overflow_fraction,
    )
    hot_expert = allocation_metrics(
        "hot_expert_only",
        matrix,
        hot_expert_cap,
        uniform_model_time,
        astra.max_cycles,
        overflow_fraction=overflow_fraction,
    )
    oracle = allocation_metrics(
        "oracle", matrix, None, uniform_model_time, astra.max_cycles, overflow_fraction=0.0
    )

    active_bytes = matrix[matrix > 0]
    top_count = max(1, int(math.ceil(0.01 * len(active_bytes))))
    top_share = float(np.sort(active_bytes)[-top_count:].sum() / matrix.sum())
    return CaseResult(
        case=case,
        skew=skew,
        matrix_kind=matrix_kind,
        astra_uniform_cycles=astra.max_cycles,
        finished_ranks=astra.finished_ranks,
        all_ranks_finished=astra.finished_ranks == matrix.shape[0] and astra.exit_code == 0,
        true_pairwise_alltoallv=len(set(active_bytes.tolist())) > 1 or matrix_kind == "uniform",
        total_bytes=int(matrix.sum()),
        nonzero_pairs=int((matrix > 0).sum()),
        top_1pct_byte_share=top_share,
        max_source_byte_share=float(matrix.sum(axis=1).max() / matrix.sum()),
        max_dest_byte_share=float(matrix.sum(axis=0).max() / matrix.sum()),
        uniform=uniform,
        hot_pair=hot_pair,
        hot_expert=hot_expert,
        oracle=oracle,
    )


def make_cases(n: int, total_bytes: int, seed: int) -> Iterable[tuple[str, str, str, np.ndarray]]:
    yield "uniform", "none", "uniform", make_uniform_matrix(n, total_bytes)
    for alpha in (0.8, 1.2, 1.5):
        yield (
            f"pair_zipf_{alpha:.1f}",
            f"alpha={alpha:.1f}",
            "pair_zipf",
            make_pair_zipf_matrix(n, alpha, seed + int(alpha * 100), total_bytes),
        )
    for alpha in (0.8, 1.2, 1.5):
        yield (
            f"expert_zipf_{alpha:.1f}",
            f"alpha={alpha:.1f}",
            "expert_zipf_sanity",
            make_expert_zipf_matrix(n, alpha, total_bytes),
        )


def verdict(speedup: float, matrix_kind: str) -> str:
    gain = (speedup - 1.0) * 100.0
    if matrix_kind == "expert_zipf_sanity":
        return "sanity: receiver-bound"
    if gain < 5.0:
        return "drop"
    if gain < 10.0:
        return "weak/possible"
    return "strong-continue"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "lambda_coco_moe_feasibility")
    parser.add_argument("--ranks", type=int, default=16)
    parser.add_argument("--total-mbytes", type=float, default=256.0)
    parser.add_argument("--bandwidth-gbps", type=float, default=800.0)
    parser.add_argument("--latency-ns", type=float, default=100.0)
    parser.add_argument("--overflow-fraction", type=float, default=0.2)
    parser.add_argument("--bias-floor", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260514)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not ASTRA_BIN.exists():
        raise SystemExit(f"Missing ASTRA binary: {ASTRA_BIN}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_root = args.output_dir / "traces"
    log_root = args.output_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    network_config = args.output_dir / "configs" / f"switch_{args.ranks}.yml"
    write_network_config(network_config, args.ranks, args.bandwidth_gbps, args.latency_ns)

    total_bytes = int(args.total_mbytes * 1024 * 1024)
    results: list[CaseResult] = []
    for case, skew, matrix_kind, float_matrix in make_cases(args.ranks, total_bytes, args.seed):
        case_dir = trace_root / case
        matrix = integerize_matrix(float_matrix, total_bytes)
        matrix_path = case_dir / "traffic_matrix.csv"
        case_dir.mkdir(parents=True, exist_ok=True)
        write_matrix_csv(matrix_path, matrix)
        prefix = "moe_ep_alltoallv"
        write_pairwise_trace(matrix, case_dir, prefix)
        astra = run_astra(case_dir / prefix, network_config, log_root / f"{case}.astra.log")
        result = summarize_case(
            case=case,
            skew=skew,
            matrix_kind=matrix_kind,
            matrix=matrix,
            astra=astra,
            overflow_fraction=args.overflow_fraction,
            floor=args.bias_floor,
        )
        results.append(result)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "approximations": [
                    "Chakra trace is true pairwise send/recv AllToAllv; it is not a single balanced ALL_TO_ALL collective.",
                    "Stock ASTRA-sim analytical backend is used only for the uniform Switch execution of the real trace.",
                    "Per-pair wavelength/circuit allocation is evaluated by a separate row/column-capacity model, then scaled to ASTRA uniform cycles.",
                    "Row and column sums of capacity are fixed, preserving per-rank injection and receive bandwidth.",
                    "Hot-pair and hot-expert allocations include a uniform overflow path.",
                ],
                "args": vars(args) | {"output_dir": str(args.output_dir)},
                "results": [
                    {
                        **{
                            key: value
                            for key, value in asdict(result).items()
                            if key not in {"uniform", "hot_pair", "hot_expert", "oracle"}
                        },
                        "uniform": asdict(result.uniform),
                        "hot_pair": asdict(result.hot_pair),
                        "hot_expert": asdict(result.hot_expert),
                        "oracle": asdict(result.oracle),
                    }
                    for result in results
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )

    print("case\tskew\tkind\tfinished\tpairwise\tuniform_cycles\thot_pair_cycles\toracle_cycles\thot_expert_cycles\tep_speedup\titer_speedup\tverdict")
    for result in results:
        hot_cycles = result.hot_pair.ep_time_scaled_cycles
        oracle_cycles = result.oracle.ep_time_scaled_cycles
        expert_cycles = result.hot_expert.ep_time_scaled_cycles
        speedup = result.hot_pair.speedup_vs_uniform
        print(
            f"{result.case}\t{result.skew}\t{result.matrix_kind}\t"
            f"{result.finished_ranks}/{args.ranks}\t{result.true_pairwise_alltoallv}\t"
            f"{result.astra_uniform_cycles}\t{hot_cycles:.1f}\t{oracle_cycles:.1f}\t"
            f"{expert_cycles:.1f}\t{speedup:.4f}\t{speedup:.4f}\t"
            f"{verdict(speedup, result.matrix_kind)}"
        )

    print(f"summary_json\t{summary_path}")


if __name__ == "__main__":
    main()
