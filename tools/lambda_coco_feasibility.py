#!/usr/bin/env python3
"""Minimal feasibility checks for lambda-CoCo-lite ideas.

This script deliberately stays trace-level and analytical. It does not modify
ASTRA-sim backends. The goal is to decide whether overlap-based wavelength
planes or skew-based MoE hot-pair wavelength binding are worth deeper simulator
work.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_trace_diagnosis() -> Any:
    path = REPO_ROOT / "tools" / "trace_diagnosis.py"
    spec = importlib.util.spec_from_file_location("trace_diagnosis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["trace_diagnosis"] = module
    spec.loader.exec_module(module)
    return module


def copy_groups(td: Any, groups: list[Any]) -> list[Any]:
    return [
        td.Group(
            gid=group.gid,
            key=group.key,
            kind=group.kind,
            label=group.label,
            comm_class=group.comm_class,
            members=group.members,
            ranks=set(group.ranks),
            deps=set(group.deps),
            children=set(group.children),
            duration_ns=group.duration_ns,
        )
        for group in groups
    ]


@dataclass
class OverlapResult:
    name: str
    astra_wall_us: float
    fair_two_plane_wall_us: float
    upper_two_plane_wall_us: float
    fair_speedup: float
    upper_speedup: float
    cross_type_ratio: float
    same_rank_ratio: float
    pairs: dict[tuple[str, str], tuple[float, float]]


def run_overlap_case(
    td: Any,
    name: str,
    trace_dir: Path,
    bytes_per_ns: float,
    latency_ns: int,
    all_reduce_factor: float,
    compute_tflops: float,
) -> OverlapResult:
    traces = td.load_trace_dir(trace_dir)

    def build(speed_bytes_per_ns: float) -> list[Any]:
        return td.build_groups(
            traces=traces,
            bytes_per_ns=speed_bytes_per_ns,
            latency_ns=latency_ns,
            all_reduce_factor=all_reduce_factor,
            compute_tflops=compute_tflops,
        )

    astra_groups = copy_groups(td, build(bytes_per_ns))
    td.schedule_groups(astra_groups, rank_resources=True)
    astra_metrics = td.overlap_metrics([g for g in astra_groups if g.kind == "comm"])
    astra_wall = max(g.end_ns for g in astra_groups) / 1_000.0

    upper_groups = copy_groups(td, build(bytes_per_ns))
    td.schedule_groups(
        upper_groups,
        rank_resources=False,
        rank_plane_resources=True,
        two_plane=True,
    )
    upper_wall = max(g.end_ns for g in upper_groups) / 1_000.0

    fair_groups = copy_groups(td, build(bytes_per_ns / 2.0))
    td.schedule_groups(
        fair_groups,
        rank_resources=False,
        rank_plane_resources=True,
        two_plane=True,
    )
    fair_wall = max(g.end_ns for g in fair_groups) / 1_000.0

    pairs: dict[tuple[str, str], tuple[float, float]] = {}
    all_pairs = set(astra_metrics["pair_overlap"]) | set(
        astra_metrics["pair_shared_rank_overlap"]
    )
    for pair in sorted(all_pairs):
        pairs[pair] = (
            astra_metrics["pair_overlap"].get(pair, 0) / 1_000.0,
            astra_metrics["pair_shared_rank_overlap"].get(pair, 0) / 1_000.0,
        )

    return OverlapResult(
        name=name,
        astra_wall_us=astra_wall,
        fair_two_plane_wall_us=fair_wall,
        upper_two_plane_wall_us=upper_wall,
        fair_speedup=astra_wall / fair_wall,
        upper_speedup=astra_wall / upper_wall,
        cross_type_ratio=astra_metrics["multi_type_ratio"],
        same_rank_ratio=astra_metrics["same_rank_multi_type_ratio"],
        pairs=pairs,
    )


def sinkhorn_capacity(weights: np.ndarray, iterations: int = 500) -> np.ndarray:
    capacity = np.maximum(weights.astype(float), 1e-30)
    n = capacity.shape[0]
    for _ in range(iterations):
        capacity *= (1.0 / capacity.sum(axis=1))[:, None]
        capacity *= (1.0 / capacity.sum(axis=0))[None, :]
    # Small numerical drift is harmless, but normalize total exactly for reports.
    capacity *= n / capacity.sum()
    return capacity


def transfer_time(matrix: np.ndarray, capacity: np.ndarray) -> float:
    active = matrix > 0
    return float(np.max(matrix[active] / capacity[active]))


def oracle_time_with_nic_caps(matrix: np.ndarray) -> float:
    n = matrix.shape[0]
    total_lower_bound = float(matrix.sum() / n)
    row_lower_bound = float(np.max(matrix.sum(axis=1)))
    col_lower_bound = float(np.max(matrix.sum(axis=0)))
    return max(total_lower_bound, row_lower_bound, col_lower_bound)


def make_pair_zipf_matrix(n: int, alpha: float, seed: int, total_bytes: float) -> np.ndarray:
    ranks = np.arange(1, n * n + 1, dtype=float)
    weights = 1.0 / np.power(ranks, alpha)
    rng = random.Random(seed)
    order = list(range(n * n))
    rng.shuffle(order)
    shuffled = np.empty_like(weights)
    shuffled[order] = weights
    matrix = shuffled.reshape((n, n))
    matrix *= total_bytes / matrix.sum()
    return matrix


def make_expert_zipf_matrix(n: int, alpha: float, total_bytes: float) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=float)
    expert_probs = 1.0 / np.power(ranks, alpha)
    expert_probs /= expert_probs.sum()
    # Every source sees the same expert popularity distribution. This isolates
    # hot-destination pressure from true source-destination hot-pair skew.
    matrix = np.tile(expert_probs[None, :], (n, 1))
    matrix *= total_bytes / matrix.sum()
    return matrix


@dataclass
class MoeResult:
    workload: str
    alpha: float | None
    uniform_time: float
    biased_time: float
    oracle_time: float
    biased_speedup: float
    oracle_speedup: float
    hot_pair_share: float
    max_row_share: float
    max_col_share: float


def run_moe_case(
    n: int,
    alpha: float | None,
    seed: int,
    total_bytes: float,
    bias_floor: float,
    skew_mode: str,
) -> MoeResult:
    if alpha is None:
        matrix = np.full((n, n), total_bytes / (n * n), dtype=float)
        workload = "uniform"
    elif skew_mode == "pair":
        matrix = make_pair_zipf_matrix(n, alpha, seed, total_bytes)
        workload = f"pair_zipf_{alpha:.1f}"
    elif skew_mode == "expert":
        matrix = make_expert_zipf_matrix(n, alpha, total_bytes)
        workload = f"expert_zipf_{alpha:.1f}"
    else:
        raise ValueError(f"Unknown skew_mode {skew_mode}")

    uniform_capacity = np.full((n, n), 1.0 / n, dtype=float)
    normalized = matrix / matrix.mean()
    biased_weights = bias_floor + normalized
    biased_capacity = sinkhorn_capacity(biased_weights)

    uniform_time = transfer_time(matrix, uniform_capacity)
    biased_time = transfer_time(matrix, biased_capacity)
    oracle_time = oracle_time_with_nic_caps(matrix)

    top_k = max(1, int(0.01 * n * n))
    hot_pair_share = float(np.sort(matrix.reshape(-1))[-top_k:].sum() / matrix.sum())

    return MoeResult(
        workload=workload,
        alpha=alpha,
        uniform_time=uniform_time,
        biased_time=biased_time,
        oracle_time=oracle_time,
        biased_speedup=uniform_time / biased_time,
        oracle_speedup=uniform_time / oracle_time,
        hot_pair_share=hot_pair_share,
        max_row_share=float(matrix.sum(axis=1).max() / matrix.sum()),
        max_col_share=float(matrix.sum(axis=0).max() / matrix.sum()),
    )


def aggregate_moe(results: list[MoeResult]) -> MoeResult:
    first = results[0]

    def mean(attr: str) -> float:
        return sum(float(getattr(result, attr)) for result in results) / len(results)

    return MoeResult(
        workload=first.workload,
        alpha=first.alpha,
        uniform_time=mean("uniform_time"),
        biased_time=mean("biased_time"),
        oracle_time=mean("oracle_time"),
        biased_speedup=mean("biased_speedup"),
        oracle_speedup=mean("oracle_speedup"),
        hot_pair_share=mean("hot_pair_share"),
        max_row_share=mean("max_row_share"),
        max_col_share=mean("max_col_share"),
    )


def decision_for_overlap(speedup: float) -> str:
    gain = (speedup - 1.0) * 100.0
    if gain < 3.0:
        return "drop"
    if gain < 8.0:
        return "weak/conditional"
    return "keep"


def decision_for_moe(speedup: float) -> str:
    gain = (speedup - 1.0) * 100.0
    if gain < 5.0:
        return "drop"
    if gain < 10.0:
        return "keep/conditional"
    return "strong-keep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bytes-per-ns",
        type=float,
        default=100.0,
        help="Trace-level bandwidth model. 100 bytes/ns equals 800 Gbps.",
    )
    parser.add_argument("--latency-ns", type=int, default=10)
    parser.add_argument("--all-reduce-factor", type=float, default=2.0)
    parser.add_argument("--compute-tflops", type=float, default=700.0)
    parser.add_argument("--moe-n", type=int, default=64)
    parser.add_argument("--moe-trials", type=int, default=8)
    parser.add_argument("--moe-bias-floor", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    td = load_trace_diagnosis()

    trace_cases = {
        "GPT-3-175B-like": REPO_ROOT
        / "trace_diagnosis_workloads"
        / "stage_output"
        / "gpt3_175b_dp2_pp8_tp4",
        "Megatron-310B-like": REPO_ROOT
        / "trace_diagnosis_workloads"
        / "stage_output"
        / "megatron_310b_dp2_pp8_tp4",
    }

    print("OVERLAP_BASED_LAMBDA_COCO_LITE")
    print(
        "trace\tcross_type_overlap\tsame_rank_overlap\t"
        "fair_two_plane_speedup\tupper_same_bw_speedup\tdecision"
    )
    overlap_results = []
    for name, trace_dir in trace_cases.items():
        result = run_overlap_case(
            td=td,
            name=name,
            trace_dir=trace_dir,
            bytes_per_ns=args.bytes_per_ns,
            latency_ns=args.latency_ns,
            all_reduce_factor=args.all_reduce_factor,
            compute_tflops=args.compute_tflops,
        )
        overlap_results.append(result)
        print(
            f"{result.name}\t{result.cross_type_ratio:.4f}\t"
            f"{result.same_rank_ratio:.4f}\t{result.fair_speedup:.4f}\t"
            f"{result.upper_speedup:.4f}\t{decision_for_overlap(result.fair_speedup)}"
        )
        for pair, (overlap_us, shared_rank_us) in result.pairs.items():
            pair_name = "+".join(pair)
            print(
                f"pair\t{result.name}\t{pair_name}\t"
                f"overlap_us={overlap_us:.3f}\tshared_rank_us={shared_rank_us:.3f}"
            )

    print("\nSKEW_BASED_MOE_LAMBDA_COCO_LITE")
    print(
        "matrix\talpha\thot_top1pct_share\tmax_row_share\tmax_col_share\t"
        "biased_speedup\toracle_speedup\tdecision"
    )
    cases: list[tuple[float | None, str]] = [(None, "pair")]
    cases.extend((alpha, "pair") for alpha in (0.8, 1.2, 1.5))
    cases.extend((alpha, "expert") for alpha in (0.8, 1.2, 1.5))
    total_bytes = float(args.moe_n)
    moe_summaries = []
    for alpha, skew_mode in cases:
        trial_results = [
            run_moe_case(
                n=args.moe_n,
                alpha=alpha,
                seed=1000 + trial,
                total_bytes=total_bytes,
                bias_floor=args.moe_bias_floor,
                skew_mode=skew_mode,
            )
            for trial in range(args.moe_trials)
        ]
        summary = aggregate_moe(trial_results)
        moe_summaries.append(summary)
        alpha_text = "na" if summary.alpha is None else f"{summary.alpha:.1f}"
        print(
            f"{summary.workload}\t{alpha_text}\t"
            f"{summary.hot_pair_share:.4f}\t{summary.max_row_share:.4f}\t"
            f"{summary.max_col_share:.4f}\t{summary.biased_speedup:.4f}\t"
            f"{summary.oracle_speedup:.4f}\t{decision_for_moe(summary.biased_speedup)}"
        )


if __name__ == "__main__":
    main()
