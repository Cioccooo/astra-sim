#!/usr/bin/env python3
"""Minimal feasibility test for PathBudget.

This is a decision test, not an ASTRA core extension.

What is actually run in ASTRA:
  * Small Chakra all-reduce smoke traces on ASTRA-supported logical topologies:
    depth 1: [Switch] / [Ring]
    depth 2: [Switch, Switch] / [Switch, Ring]
    depth 3: [Switch, Switch, Switch] / [Switch, Switch, Ring] /
             [Switch, Switch, FullyConnected]

What is modeled externally:
  * OCS path-depth reach, bandwidth, latency, loss, port/radix budget, and
    depth assignment. ASTRA dimensions are not treated as physical OCS depth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


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

from et_def.et_def_pb2 import (  # type: ignore  # noqa: E402
    AttributeProto as ChakraAttr,
    BoolList,
    CollectiveCommType,
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


AXES = ("TP", "PP", "DP")


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    params_b: float
    layers: int
    hidden: int
    seq_len: int
    tp: int
    pp: int
    dp: int
    microbatches: int
    microbatch_size: int
    critical_tp: float
    critical_pp: float
    critical_dp: float


@dataclass(frozen=True)
class AxisSignature:
    axis: str
    payload_gb: float
    events: int
    group_size: int
    min_depth: int
    critical_weight: float
    frequency: str


@dataclass(frozen=True)
class DepthSpec:
    depth: int
    reach: int
    bandwidth_gbps: float
    latency_us: float
    port_cost: float
    loss_db: float
    reconfig_us: float


@dataclass(frozen=True)
class Regime:
    name: str
    description: str
    depths: dict[int, DepthSpec]


@dataclass(frozen=True)
class Budget:
    loss_name: str
    loss_db: float
    port_name: str
    port_units: float


@dataclass
class AssignmentEval:
    workload: str
    regime: str
    loss_budget: str
    port_budget: str
    assignment: str
    time_s: float | None
    port_usage: float
    max_loss_db: float
    feasible: bool
    failure_reason: str


@dataclass
class ResultRow:
    workload: str
    regime: str
    loss_budget: str
    port_budget: str
    best_fixed_depth: str
    pathbudget_assignment: str
    fixed_time_s: float | None
    pathbudget_time_s: float | None
    oracle_time_s: float | None
    speedup: float | None
    oracle_gap_pct: float | None
    port_usage: float | None
    max_loss_db: float | None
    feasible: bool
    feasibility_failures: dict[str, int]
    verdict: str


@dataclass
class CalibrationConfig:
    name: str
    logical_depth: int
    topology: list[str]
    npus_count: list[int]


def workload_specs() -> list[WorkloadSpec]:
    return [
        WorkloadSpec(
            name="LLaMA-13B-like_DP-heavy_TP8_PP2_DP64",
            params_b=13.0,
            layers=40,
            hidden=5120,
            seq_len=4096,
            tp=8,
            pp=2,
            dp=64,
            microbatches=32,
            microbatch_size=1,
            critical_tp=0.55,
            critical_pp=0.35,
            critical_dp=0.75,
        ),
        WorkloadSpec(
            name="GPT-3-175B-like_mixed_TP8_PP8_DP16",
            params_b=175.0,
            layers=96,
            hidden=12288,
            seq_len=2048,
            tp=8,
            pp=8,
            dp=16,
            microbatches=32,
            microbatch_size=1,
            critical_tp=0.75,
            critical_pp=0.65,
            critical_dp=0.55,
        ),
        WorkloadSpec(
            name="Megatron-310B-like_PP-heavy_TP8_PP16_DP8",
            params_b=310.0,
            layers=96,
            hidden=16384,
            seq_len=2048,
            tp=8,
            pp=16,
            dp=8,
            microbatches=32,
            microbatch_size=1,
            critical_tp=0.70,
            critical_pp=0.90,
            critical_dp=0.45,
        ),
        WorkloadSpec(
            name="Megatron-530B-like_large_PP-heavy_TP8_PP16_DP8",
            params_b=530.0,
            layers=105,
            hidden=20480,
            seq_len=2048,
            tp=8,
            pp=16,
            dp=8,
            microbatches=32,
            microbatch_size=1,
            critical_tp=0.70,
            critical_pp=0.95,
            critical_dp=0.45,
        ),
    ]


def min_depth_for_reach(group_size: int) -> int:
    if group_size <= 16:
        return 1
    if group_size <= 64:
        return 2
    return 3


def signatures(workload: WorkloadSpec) -> dict[str, AxisSignature]:
    bytes_per_elem = 2
    activation_bytes = (
        workload.microbatch_size
        * workload.seq_len
        * workload.hidden
        * bytes_per_elem
        / workload.tp
    )
    tp_collective_factor = 2.0 * (workload.tp - 1) / workload.tp
    dp_collective_factor = 2.0 * (workload.dp - 1) / workload.dp

    tp_payload = (
        workload.layers
        * workload.microbatches
        * 4
        * activation_bytes
        * tp_collective_factor
    )
    pp_payload = (
        max(1, workload.pp - 1)
        * workload.microbatches
        * 2
        * activation_bytes
    )
    dp_payload = (
        workload.params_b
        * 1e9
        * bytes_per_elem
        / (workload.tp * workload.pp)
        * dp_collective_factor
    )

    tp_events = workload.layers * workload.microbatches * 4
    pp_events = max(1, workload.pp - 1) * workload.microbatches * 2
    dp_events = max(1, math.ceil(workload.layers / 4))

    # Placement model used for this first feasibility pass:
    #   TP is local to a tensor-parallel group.
    #   PP uses the span of all pipeline stages in one model-parallel replica.
    #   DP/FSDP is treated as global synchronization, so it needs depth 3 even
    #   when the DP degree could fit inside one pod by rank count alone.
    pp_span = workload.tp * workload.pp
    return {
        "TP": AxisSignature(
            axis="TP",
            payload_gb=tp_payload / 1e9,
            events=tp_events,
            group_size=workload.tp,
            min_depth=min_depth_for_reach(workload.tp),
            critical_weight=workload.critical_tp,
            frequency="layer/microbatch frequent",
        ),
        "PP": AxisSignature(
            axis="PP",
            payload_gb=pp_payload / 1e9,
            events=pp_events,
            group_size=pp_span,
            min_depth=min_depth_for_reach(pp_span),
            critical_weight=workload.critical_pp,
            frequency="microbatch pipeline boundary",
        ),
        "DP": AxisSignature(
            axis="DP",
            payload_gb=dp_payload / 1e9,
            events=dp_events,
            group_size=workload.dp,
            min_depth=3,
            critical_weight=workload.critical_dp,
            frequency="bucketed global gradient sync",
        ),
    }


def regimes() -> list[Regime]:
    return [
        Regime(
            name="reach_only_depth_is_costly",
            description="deeper paths add reach, latency, loss, and port cost; bandwidth is not magically better",
            depths={
                1: DepthSpec(1, 16, 800, 0.45, 1.0, 2.5, 0),
                2: DepthSpec(2, 64, 700, 1.50, 2.0, 6.0, 1),
                3: DepthSpec(3, 1024, 600, 5.00, 4.0, 10.0, 10),
            },
        ),
        Regime(
            name="extra_parallelism_bandwidth_steering",
            description="deeper/global paths receive more wavelengths/circuits, but consume more port budget",
            depths={
                1: DepthSpec(1, 16, 600, 0.45, 1.0, 2.5, 0),
                2: DepthSpec(2, 64, 900, 1.50, 3.0, 6.5, 1),
                3: DepthSpec(3, 1024, 1200, 5.00, 6.0, 10.5, 10),
            },
        ),
        Regime(
            name="deep_oversubscribed_global_bottleneck",
            description="global reach is available, but shared global OCS resources reduce per-flow bandwidth",
            depths={
                1: DepthSpec(1, 16, 800, 0.45, 1.0, 2.5, 0),
                2: DepthSpec(2, 64, 600, 1.80, 2.5, 6.5, 1),
                3: DepthSpec(3, 1024, 300, 6.50, 5.0, 11.0, 10),
            },
        ),
    ]


def budgets() -> list[Budget]:
    loss_budgets = {
        "strict": 10.0,
        "medium": 14.0,
        "relaxed": 20.0,
    }
    port_budgets = {
        "low": 7.0,
        "medium": 12.5,
        "high": 18.0,
    }
    return [
        Budget(loss_name=loss_name, loss_db=loss_db, port_name=port_name, port_units=port_units)
        for loss_name, loss_db in loss_budgets.items()
        for port_name, port_units in port_budgets.items()
    ]


def calibration_configs() -> list[CalibrationConfig]:
    return [
        CalibrationConfig("depth1_switch", 1, ["Switch"], [16]),
        CalibrationConfig("depth1_ring", 1, ["Ring"], [16]),
        CalibrationConfig("depth2_switch_switch", 2, ["Switch", "Switch"], [16, 4]),
        CalibrationConfig("depth2_switch_ring", 2, ["Switch", "Ring"], [16, 4]),
        CalibrationConfig("depth3_switch_switch_switch", 3, ["Switch", "Switch", "Switch"], [16, 4, 16]),
        CalibrationConfig("depth3_switch_switch_ring", 3, ["Switch", "Switch", "Ring"], [16, 4, 16]),
        CalibrationConfig("depth3_switch_switch_fc", 3, ["Switch", "Switch", "FullyConnected"], [16, 4, 16]),
    ]


def collective_steps(axis: AxisSignature) -> int:
    if axis.axis == "PP":
        return 1
    group = max(2, axis.group_size)
    return max(1, math.ceil(math.log2(group)) * 2)


def axis_port_weight(axis: str) -> float:
    return {"TP": 1.20, "PP": 0.80, "DP": 1.00}[axis]


def assignment_name(assignment: dict[str, int]) -> str:
    return f"TP{assignment['TP']}-PP{assignment['PP']}-DP{assignment['DP']}"


def assignment_time_s(
    sigs: dict[str, AxisSignature],
    regime: Regime,
    budget: Budget,
    assignment: dict[str, int],
    ignore_port_budget: bool = False,
) -> AssignmentEval:
    port_usage = 0.0
    max_loss = 0.0
    failure_counts: list[str] = []
    for axis_name, depth in assignment.items():
        axis = sigs[axis_name]
        spec = regime.depths[depth]
        if depth < axis.min_depth:
            failure_counts.append("reach")
        if spec.loss_db > budget.loss_db:
            failure_counts.append("loss")
        port_usage += axis_port_weight(axis_name) * spec.port_cost
        max_loss = max(max_loss, spec.loss_db)

    if port_usage > budget.port_units and not ignore_port_budget:
        failure_counts.append("port")

    if failure_counts:
        return AssignmentEval(
            workload="",
            regime=regime.name,
            loss_budget=budget.loss_name,
            port_budget=budget.port_name,
            assignment=assignment_name(assignment),
            time_s=None,
            port_usage=port_usage,
            max_loss_db=max_loss,
            feasible=False,
            failure_reason="+".join(sorted(set(failure_counts))),
        )

    total = 0.0
    for axis_name, depth in assignment.items():
        axis = sigs[axis_name]
        spec = regime.depths[depth]
        latency_s = (spec.latency_us + spec.reconfig_us) * 1e-6
        latency_component = latency_s * axis.events * collective_steps(axis)
        bandwidth_bps = spec.bandwidth_gbps * 1e9
        data_component = (axis.payload_gb * 8e9) / bandwidth_bps
        total += axis.critical_weight * (latency_component + data_component)

    return AssignmentEval(
        workload="",
        regime=regime.name,
        loss_budget=budget.loss_name,
        port_budget=budget.port_name,
        assignment=assignment_name(assignment),
        time_s=total,
        port_usage=port_usage,
        max_loss_db=max_loss,
        feasible=True,
        failure_reason="",
    )


def all_assignments() -> Iterable[dict[str, int]]:
    for tp in (1, 2, 3):
        for pp in (1, 2, 3):
            for dp in (1, 2, 3):
                yield {"TP": tp, "PP": pp, "DP": dp}


def best_eval(evals: Iterable[AssignmentEval]) -> AssignmentEval | None:
    feasible = [item for item in evals if item.feasible and item.time_s is not None]
    if not feasible:
        return None
    return min(feasible, key=lambda item: item.time_s or float("inf"))


def failure_hist(evals: Iterable[AssignmentEval]) -> dict[str, int]:
    hist: dict[str, int] = {"reach": 0, "loss": 0, "port": 0}
    for item in evals:
        if item.feasible:
            continue
        for part in item.failure_reason.split("+"):
            if part:
                hist[part] = hist.get(part, 0) + 1
    return hist


def evaluate() -> tuple[list[dict[str, object]], list[ResultRow], list[dict[str, object]]]:
    workload_summaries = []
    rows: list[ResultRow] = []
    assignment_rows: list[dict[str, object]] = []

    for workload in workload_specs():
        sigs = signatures(workload)
        workload_summaries.append(
            {
                "name": workload.name,
                "tp": workload.tp,
                "pp": workload.pp,
                "dp": workload.dp,
                "axes": {axis: asdict(sig) for axis, sig in sigs.items()},
            }
        )
        for regime in regimes():
            for budget in budgets():
                evals = [
                    assignment_time_s(sigs, regime, budget, assignment)
                    for assignment in all_assignments()
                ]
                for item in evals:
                    item.workload = workload.name
                    assignment_rows.append(asdict(item))

                fixed_evals = []
                for depth in (1, 2, 3):
                    fixed_evals.append(
                        assignment_time_s(
                            sigs,
                            regime,
                            budget,
                            {"TP": depth, "PP": depth, "DP": depth},
                        )
                    )
                best_fixed = best_eval(fixed_evals)
                pathbudget = best_eval(evals)
                oracle = best_eval(
                    assignment_time_s(sigs, regime, budget, assignment, ignore_port_budget=True)
                    for assignment in all_assignments()
                )

                if best_fixed is None or pathbudget is None:
                    speedup = None
                else:
                    speedup = (best_fixed.time_s or 0.0) / (pathbudget.time_s or float("inf"))
                if oracle is None or pathbudget is None:
                    oracle_gap = None
                else:
                    oracle_gap = ((pathbudget.time_s or 0.0) / (oracle.time_s or 1.0) - 1.0) * 100.0

                if pathbudget is None:
                    verdict = "infeasible"
                elif best_fixed is None:
                    verdict = "keep_no_fixed_feasible"
                elif speedup is not None and speedup >= 1.10 and (oracle_gap is None or oracle_gap <= 10.0):
                    verdict = "strong_keep"
                elif speedup is not None and speedup >= 1.05:
                    verdict = "weak_keep"
                elif speedup is not None and speedup >= 0.98:
                    verdict = "small_ablation"
                else:
                    verdict = "drop"

                rows.append(
                    ResultRow(
                        workload=workload.name,
                        regime=regime.name,
                        loss_budget=budget.loss_name,
                        port_budget=budget.port_name,
                        best_fixed_depth=best_fixed.assignment if best_fixed else "none",
                        pathbudget_assignment=pathbudget.assignment if pathbudget else "none",
                        fixed_time_s=best_fixed.time_s if best_fixed else None,
                        pathbudget_time_s=pathbudget.time_s if pathbudget else None,
                        oracle_time_s=oracle.time_s if oracle else None,
                        speedup=speedup,
                        oracle_gap_pct=oracle_gap,
                        port_usage=pathbudget.port_usage if pathbudget else None,
                        max_loss_db=pathbudget.max_loss_db if pathbudget else None,
                        feasible=pathbudget is not None,
                        feasibility_failures=failure_hist(evals),
                        verdict=verdict,
                    )
                )
    return workload_summaries, rows, assignment_rows


def write_network_config(path: Path, topology: list[str], counts: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bw = [50.0] * len(topology)
    lat = [500.0] * len(topology)
    path.write_text(
        "\n".join(
            [
                f"topology: [ {', '.join(topology)} ]",
                f"npus_count: [ {', '.join(str(count) for count in counts)} ]",
                f"bandwidth: [ {', '.join(str(item) for item in bw)} ]",
                f"latency: [ {', '.join(str(item) for item in lat)} ]",
                "",
            ]
        )
    )


def write_calibration_workload(workload_dir: Path, ranks: int, dims: int) -> Path:
    workload_dir.mkdir(parents=True, exist_ok=True)
    prefix = workload_dir / "calib"
    for rank in range(ranks):
        node = ChakraNode()
        node.id = 1
        node.name = "PATHBUDGET_CALIB_ALL_REDUCE_1MB"
        node.type = NodeType.COMM_COLL_NODE
        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        node.attr.append(ChakraAttr(name="comm_type", int64_val=int(CollectiveCommType.ALL_REDUCE)))
        node.attr.append(ChakraAttr(name="comm_size", uint64_val=1024 * 1024))
        node.attr.append(ChakraAttr(name="involved_dim", bool_list=BoolList(values=[True] * dims)))
        with (workload_dir / f"calib.{rank}.et").open("wb") as handle:
            encode_message(handle, GlobalMetadata(version="0.0.4"))
            encode_message(handle, node)
    return prefix


def run_astra_calibration(output_dir: Path, timeout_s: int) -> list[dict[str, object]]:
    results = []
    for config in calibration_configs():
        ranks = math.prod(config.npus_count)
        workload_prefix = write_calibration_workload(
            output_dir / "astra_workloads" / config.name,
            ranks=ranks,
            dims=len(config.topology),
        )
        network_config = output_dir / "astra_configs" / f"{config.name}.yml"
        write_network_config(network_config, config.topology, config.npus_count)
        log_path = output_dir / "astra_logs" / f"{config.name}.log"
        cmd = [
            str(ASTRA_BIN),
            f"--workload-configuration={workload_prefix}",
            f"--system-configuration={SYSTEM_CONFIG}",
            f"--network-configuration={network_config}",
            f"--remote-memory-configuration={REMOTE_MEMORY_CONFIG}",
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_s,
            )
            output = proc.stdout
            exit_code = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            exit_code = None
            timed_out = True
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output)
        cycles = [int(match) for match in re.findall(r"finished, ([0-9]+) cycles", output)]
        results.append(
            {
                "name": config.name,
                "logical_depth": config.logical_depth,
                "topology": config.topology,
                "npus_count": config.npus_count,
                "ranks": ranks,
                "network_config": str(network_config),
                "log": str(log_path),
                "exit_code": exit_code,
                "timed_out": timed_out,
                "finished_ranks": len(cycles),
                "max_cycles": max(cycles) if cycles else None,
                "min_cycles": min(cycles) if cycles else None,
            }
        )
    return results


def write_outputs(
    output_dir: Path,
    astra_calibration: list[dict[str, object]],
    workloads: list[dict[str, object]],
    rows: list[ResultRow],
    assignments: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "astra_calibration.json").write_text(json.dumps(astra_calibration, indent=2, sort_keys=True))
    (output_dir / "workload_signatures.json").write_text(json.dumps(workloads, indent=2, sort_keys=True))
    (output_dir / "results.json").write_text(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))
    (output_dir / "assignments.json").write_text(json.dumps(assignments, indent=2, sort_keys=True))
    with (output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    with (output_dir / "assignments.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignments[0].keys()))
        writer.writeheader()
        for row in assignments:
            writer.writerow(row)


def selected_rows(rows: list[ResultRow]) -> list[ResultRow]:
    wanted = []
    for regime in ("reach_only_depth_is_costly", "extra_parallelism_bandwidth_steering", "deep_oversubscribed_global_bottleneck"):
        for workload in (
            "LLaMA-13B-like_DP-heavy_TP8_PP2_DP64",
            "GPT-3-175B-like_mixed_TP8_PP8_DP16",
            "Megatron-310B-like_PP-heavy_TP8_PP16_DP8",
            "Megatron-530B-like_large_PP-heavy_TP8_PP16_DP8",
        ):
            matches = [
                row
                for row in rows
                if row.workload == workload
                and row.regime == regime
                and row.loss_budget == "medium"
                and row.port_budget == "medium"
            ]
            if matches:
                wanted.append(matches[0])
    return wanted


def summarize(rows: list[ResultRow]) -> dict[str, object]:
    feasible = [row for row in rows if row.feasible and row.speedup is not None]
    medium_nonoptimistic = [
        row
        for row in feasible
        if row.regime == "reach_only_depth_is_costly"
        and row.loss_budget == "medium"
        and row.port_budget == "medium"
    ]
    speedups = [row.speedup for row in feasible if row.speedup is not None]
    nonopt_speedups = [row.speedup for row in medium_nonoptimistic if row.speedup is not None]
    worst_slowdown = min(speedups) if speedups else None
    oracle_gaps = [row.oracle_gap_pct for row in feasible if row.oracle_gap_pct is not None]
    selected = {}
    for row in feasible:
        selected[row.pathbudget_assignment] = selected.get(row.pathbudget_assignment, 0) + 1
    return {
        "num_rows": len(rows),
        "num_feasible_pathbudget_rows": len([row for row in rows if row.feasible]),
        "average_speedup_all_feasible": mean(speedups) if speedups else None,
        "average_speedup_medium_nonoptimistic": mean(nonopt_speedups) if nonopt_speedups else None,
        "worst_case_speedup_all_feasible": worst_slowdown,
        "average_oracle_gap_pct": mean(oracle_gaps) if oracle_gaps else None,
        "selected_assignment_histogram": dict(sorted(selected.items(), key=lambda item: item[0])),
    }


def fmt_time(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}s"


def fmt_speedup(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}x"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "pathbudget_feasibility")
    parser.add_argument("--skip-astra", action="store_true")
    parser.add_argument("--astra-timeout-s", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workloads, rows, assignments = evaluate()
    astra_calibration = [] if args.skip_astra else run_astra_calibration(args.output_dir, args.astra_timeout_s)
    write_outputs(args.output_dir, astra_calibration, workloads, rows, assignments)

    print("ASTRA_CALIBRATION")
    if not astra_calibration:
        print("skipped")
    for item in astra_calibration:
        topo = "/".join(item["topology"])
        print(
            f"{item['name']}\tdepth={item['logical_depth']}\t{topo}\t"
            f"finished={item['finished_ranks']}/{item['ranks']}\t"
            f"cycles={item['max_cycles']}\ttimeout={item['timed_out']}"
        )

    print("\nWORKLOAD_SIGNATURES")
    for workload in workloads:
        axes = workload["axes"]
        print(f"{workload['name']}")
        for axis in AXES:
            sig = axes[axis]
            print(
                f"  {axis}: payload={sig['payload_gb']:.2f}GB, events={sig['events']}, "
                f"group={sig['group_size']}, min_depth={sig['min_depth']}, "
                f"critical={sig['critical_weight']}"
            )

    print("\nSELECTED_RESULTS_MEDIUM_BUDGETS")
    print("workload\tregime\tbest_fixed\tpathbudget\tfixed\tpathbudget\toracle\tspeedup\tverdict")
    for row in selected_rows(rows):
        print(
            f"{row.workload}\t{row.regime}\t{row.best_fixed_depth}\t{row.pathbudget_assignment}\t"
            f"{fmt_time(row.fixed_time_s)}\t{fmt_time(row.pathbudget_time_s)}\t"
            f"{fmt_time(row.oracle_time_s)}\t{fmt_speedup(row.speedup)}\t{row.verdict}"
        )

    summary = summarize(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
