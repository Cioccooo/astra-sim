#!/usr/bin/env python3
"""Minimal feasibility test for StratOCS.

This is intentionally a decision test, not a proof. It uses only ASTRA-sim
analytical topology blocks: Ring, Switch, and FullyConnected.

What is actually run in ASTRA:
  * A small 2 x 2 multi-dimensional topology smoke test for each topology YAML.

What is modeled externally:
  * The 1024-GPU, 20-job queue sweep. Generating and running all large traces for
    every topology/window/reconfig point would be overkill for this first pass.
    The cost model is transparent and only uses communication signatures.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


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
    CollectiveCommType,
    AttributeProto as ChakraAttr,
    BoolList,
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
class TopologyTemplate:
    name: str
    inner: str
    outer: str
    outer_class: str
    meaning: str
    dp_mult: float
    tp_mult: float
    pp_mult: float

    @property
    def astra_topology(self) -> list[str]:
        return [self.inner, "Ring" if self.outer.startswith("Ring") else self.outer]


@dataclass(frozen=True)
class Job:
    job_id: str
    model: str
    tp: int
    pp: int
    dp: int
    dp_volume: float
    tp_volume: float
    pp_volume: float
    comm_fraction: float
    preferred_outer: str

    @property
    def dominant(self) -> str:
        volumes = {"DP": self.dp_volume, "TP": self.tp_volume, "PP": self.pp_volume}
        return max(volumes, key=volumes.get)


@dataclass
class PolicyResult:
    policy: str
    makespan_s: float
    runtime_s: float
    reconfig_s: float
    slow_reconfigs: int
    fast_reconfigs: int


@dataclass
class ScenarioResult:
    queue_order: str
    diversity: str
    job_duration_s: float
    slow_reconfig_s: float
    fast_reconfig_s: float
    window_size: int
    static_makespan_s: float
    per_job_makespan_s: float
    stratocs_makespan_s: float
    oracle_makespan_s: float
    static_speedup: float
    per_job_speedup: float
    oracle_loss_pct: float
    reconfig_fraction: float
    slow_reconfig_reduction_pct: float
    retained_improvement_pct: float
    verdict: str


def topology_library() -> list[TopologyTemplate]:
    return [
        TopologyTemplate(
            name="cheap_optical",
            inner="Ring",
            outer="RingGeneric",
            outer_class="RingGeneric",
            meaning="conservative low-radix all-optical fabric",
            dp_mult=1.40,
            tp_mult=1.30,
            pp_mult=1.05,
        ),
        TopologyTemplate(
            name="balanced",
            inner="Switch",
            outer="Switch",
            outer_class="Switch",
            meaning="general-purpose two-layer OCS fabric",
            dp_mult=1.00,
            tp_mult=1.00,
            pp_mult=1.00,
        ),
        TopologyTemplate(
            name="dp_ring_outer",
            inner="Switch",
            outer="RingDP",
            outer_class="RingDP",
            meaning="pod-local switch plus DP-friendly outer ring placement",
            dp_mult=0.82,
            tp_mult=1.00,
            pp_mult=1.10,
        ),
        TopologyTemplate(
            name="pp_chain_outer",
            inner="Switch",
            outer="RingPP",
            outer_class="RingPP",
            meaning="pod-local switch plus PP-stage-adjacent outer ring placement",
            dp_mult=1.20,
            tp_mult=1.00,
            pp_mult=0.65,
        ),
        TopologyTemplate(
            name="global_upper",
            inner="Switch",
            outer="FullyConnected",
            outer_class="FullyConnected",
            meaning="ideal high-radix outer OCS upper bound",
            dp_mult=0.55,
            tp_mult=0.95,
            pp_mult=0.85,
        ),
        TopologyTemplate(
            name="local_upper",
            inner="FullyConnected",
            outer="Switch",
            outer_class="Switch",
            meaning="ideal inner OCS upper bound",
            dp_mult=1.00,
            tp_mult=0.55,
            pp_mult=0.90,
        ),
    ]


def build_jobs() -> list[Job]:
    specs: list[tuple[str, int, int, int, int, float, float, float, float, str]] = []
    specs.extend(("LLaMA-7B-like", i, 4, 2, 128, 0.65, 0.25, 0.10, 0.25, "RingDP") for i in range(5))
    specs.extend(("LLaMA-13B-like", i, 8, 2, 64, 0.55, 0.35, 0.10, 0.30, "RingDP") for i in range(4))
    specs.extend(("LLaMA-70B-like", i, 8, 8, 16, 0.25, 0.35, 0.40, 0.38, "RingPP") for i in range(3))
    specs.extend(("GPT-3-13B-like", i, 4, 4, 64, 0.45, 0.25, 0.30, 0.30, "Switch") for i in range(3))
    specs.extend(("GPT-3-175B-like", i, 8, 8, 16, 0.20, 0.35, 0.45, 0.45, "RingPP") for i in range(3))
    specs.extend(("Megatron-310B-like", i, 8, 16, 8, 0.10, 0.30, 0.60, 0.50, "RingPP") for i in range(2))
    jobs = []
    counters: dict[str, int] = {}
    for model, _, tp, pp, dp, dp_v, tp_v, pp_v, comm_f, preferred in specs:
        counters[model] = counters.get(model, 0) + 1
        jobs.append(
            Job(
                job_id=f"{model}-{counters[model]}",
                model=model,
                tp=tp,
                pp=pp,
                dp=dp,
                dp_volume=dp_v,
                tp_volume=tp_v,
                pp_volume=pp_v,
                comm_fraction=comm_f,
                preferred_outer=preferred,
            )
        )
    return jobs


def clustered_queue(jobs: list[Job]) -> list[Job]:
    return sorted(jobs, key=lambda job: (job.preferred_outer != "RingDP", job.preferred_outer, job.job_id))


def random_queue(jobs: list[Job], seed: int) -> list[Job]:
    out = list(jobs)
    random.Random(seed).shuffle(out)
    return out


def diversity_scale(name: str) -> float:
    return {"low": 0.35, "medium": 1.0, "high": 1.45}[name]


def effective_mult(base: float, diversity: str) -> float:
    return max(0.05, 1.0 + diversity_scale(diversity) * (base - 1.0))


def relative_runtime(job: Job, topo: TopologyTemplate, diversity: str) -> float:
    comm_mult = (
        job.dp_volume * effective_mult(topo.dp_mult, diversity)
        + job.tp_volume * effective_mult(topo.tp_mult, diversity)
        + job.pp_volume * effective_mult(topo.pp_mult, diversity)
    )
    return (1.0 - job.comm_fraction) + job.comm_fraction * comm_mult


def topology_time_s(
    job: Job,
    topo: TopologyTemplate,
    diversity: str,
    static_baseline_duration_s: float,
    static_topo: TopologyTemplate,
) -> float:
    static_relative = relative_runtime(job, static_topo, diversity)
    topo_relative = relative_runtime(job, topo, diversity)
    return static_baseline_duration_s * topo_relative / static_relative


def best_template(
    job: Job,
    templates: list[TopologyTemplate],
    diversity: str,
    duration_s: float,
    static_topo: TopologyTemplate,
    outer_class: str | None = None,
) -> tuple[TopologyTemplate, float]:
    candidates = templates
    if outer_class is not None:
        candidates = [template for template in templates if template.outer_class == outer_class]
    if not candidates:
        raise ValueError(f"No templates for outer_class={outer_class}")
    scored = [
        (
            topology_time_s(job, template, diversity, duration_s, static_topo),
            template,
        )
        for template in candidates
    ]
    time_s, template = min(scored, key=lambda item: item[0])
    return template, time_s


def static_policy(
    queue: list[Job],
    templates: list[TopologyTemplate],
    diversity: str,
    duration_s: float,
) -> PolicyResult:
    static_topo = next(template for template in templates if template.name == "balanced")
    runtime = sum(topology_time_s(job, static_topo, diversity, duration_s, static_topo) for job in queue)
    return PolicyResult("static", runtime, runtime, 0.0, 0, 0)


def oracle_policy(
    queue: list[Job],
    templates: list[TopologyTemplate],
    diversity: str,
    duration_s: float,
) -> PolicyResult:
    static_topo = next(template for template in templates if template.name == "balanced")
    runtime = sum(best_template(job, templates, diversity, duration_s, static_topo)[1] for job in queue)
    return PolicyResult("oracle", runtime, runtime, 0.0, 0, 0)


def per_job_policy(
    queue: list[Job],
    templates: list[TopologyTemplate],
    diversity: str,
    duration_s: float,
    slow_s: float,
    fast_s: float,
) -> PolicyResult:
    oracle = oracle_policy(queue, templates, diversity, duration_s)
    reconfig = len(queue) * (slow_s + fast_s)
    return PolicyResult("per_job_full_reconfig", oracle.runtime_s + reconfig, oracle.runtime_s, reconfig, len(queue), len(queue))


def stratocs_policy(
    queue: list[Job],
    templates: list[TopologyTemplate],
    diversity: str,
    duration_s: float,
    slow_s: float,
    fast_s: float,
    window_size: int,
) -> PolicyResult:
    static_topo = next(template for template in templates if template.name == "balanced")
    outer_classes = sorted({template.outer_class for template in templates})
    runtime = 0.0
    slow_reconfigs = 0
    for start in range(0, len(queue), window_size):
        window = queue[start : start + window_size]
        best_outer_time = None
        for outer in outer_classes:
            total = sum(
                best_template(job, templates, diversity, duration_s, static_topo, outer_class=outer)[1]
                for job in window
            )
            if best_outer_time is None or total < best_outer_time:
                best_outer_time = total
        runtime += float(best_outer_time)
        slow_reconfigs += 1
    fast_reconfigs = len(queue)
    reconfig = slow_reconfigs * slow_s + fast_reconfigs * fast_s
    return PolicyResult(f"stratocs_w{window_size}", runtime + reconfig, runtime, reconfig, slow_reconfigs, fast_reconfigs)


def retained_improvement(static: PolicyResult, oracle: PolicyResult, strat: PolicyResult) -> float:
    denom = static.runtime_s - oracle.runtime_s
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, (static.runtime_s - strat.runtime_s) / denom))


def scenario_verdict(result: ScenarioResult) -> str:
    static_gain = (result.static_speedup - 1.0) * 100.0
    per_job_gain = (result.per_job_speedup - 1.0) * 100.0
    if static_gain >= 10.0 and result.retained_improvement_pct >= 80.0 and result.slow_reconfig_reduction_pct >= 70.0:
        return "serious"
    if static_gain >= 5.0 and result.retained_improvement_pct >= 80.0:
        return "narrow_keep"
    if per_job_gain >= 0.0 and result.slow_reconfig_reduction_pct >= 70.0:
        return "conditional"
    return "drop"


def run_scenario(
    queue: list[Job],
    order_name: str,
    diversity: str,
    duration_s: float,
    slow_s: float,
    fast_s: float,
    window_size: int,
    templates: list[TopologyTemplate],
) -> ScenarioResult:
    static = static_policy(queue, templates, diversity, duration_s)
    oracle = oracle_policy(queue, templates, diversity, duration_s)
    per_job = per_job_policy(queue, templates, diversity, duration_s, slow_s, fast_s)
    strat = stratocs_policy(queue, templates, diversity, duration_s, slow_s, fast_s, window_size)
    result = ScenarioResult(
        queue_order=order_name,
        diversity=diversity,
        job_duration_s=duration_s,
        slow_reconfig_s=slow_s,
        fast_reconfig_s=fast_s,
        window_size=window_size,
        static_makespan_s=static.makespan_s,
        per_job_makespan_s=per_job.makespan_s,
        stratocs_makespan_s=strat.makespan_s,
        oracle_makespan_s=oracle.makespan_s,
        static_speedup=static.makespan_s / strat.makespan_s,
        per_job_speedup=per_job.makespan_s / strat.makespan_s,
        oracle_loss_pct=(strat.makespan_s / oracle.makespan_s - 1.0) * 100.0,
        reconfig_fraction=strat.reconfig_s / strat.makespan_s,
        slow_reconfig_reduction_pct=(1.0 - strat.slow_reconfigs / len(queue)) * 100.0,
        retained_improvement_pct=retained_improvement(static, oracle, strat) * 100.0,
        verdict="",
    )
    result.verdict = scenario_verdict(result)
    return result


def write_network_config(path: Path, topologies: list[str], counts: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"topology: [ {', '.join(topologies)} ]",
                f"npus_count: [ {', '.join(str(count) for count in counts)} ]",
                "bandwidth: [ 400.0, 50.0 ]",
                "latency: [ 100.0, 1000.0 ]",
                "",
            ]
        )
    )


def write_smoke_workload(workload_dir: Path, ranks: int, dims: int) -> Path:
    workload_dir.mkdir(parents=True, exist_ok=True)
    prefix = workload_dir / "smoke"
    for rank in range(ranks):
        node = ChakraNode()
        node.id = 1
        node.name = "SMOKE_ALL_REDUCE"
        node.type = NodeType.COMM_COLL_NODE
        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        node.attr.append(ChakraAttr(name="comm_type", int64_val=int(CollectiveCommType.ALL_REDUCE)))
        node.attr.append(ChakraAttr(name="comm_size", uint64_val=1024 * 1024))
        node.attr.append(ChakraAttr(name="involved_dim", bool_list=BoolList(values=[True] * dims)))
        with (workload_dir / f"smoke.{rank}.et").open("wb") as handle:
            encode_message(handle, GlobalMetadata(version="0.0.4"))
            encode_message(handle, node)
    return prefix


def run_astra_smoke(output_dir: Path, templates: list[TopologyTemplate]) -> list[dict[str, object]]:
    workload_prefix = write_smoke_workload(output_dir / "smoke_workload", ranks=4, dims=2)
    seen: set[tuple[str, str]] = set()
    results = []
    for template in templates:
        topo = template.astra_topology
        key = tuple(topo)
        if key in seen:
            continue
        seen.add(key)
        config_path = output_dir / "smoke_configs" / f"{'_'.join(topo)}.yml"
        write_network_config(config_path, topo, [2, 2])
        log_path = output_dir / "smoke_logs" / f"{'_'.join(topo)}.log"
        cmd = [
            str(ASTRA_BIN),
            f"--workload-configuration={workload_prefix}",
            f"--system-configuration={SYSTEM_CONFIG}",
            f"--network-configuration={config_path}",
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
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout)
        cycles = [int(match) for match in re.findall(r"finished, ([0-9]+) cycles", proc.stdout)]
        results.append(
            {
                "topology": topo,
                "config": str(config_path),
                "exit_code": proc.returncode,
                "finished_ranks": len(cycles),
                "max_cycles": max(cycles) if cycles else None,
                "log": str(log_path),
            }
        )
    return results


def average_random_results(results: list[ScenarioResult]) -> ScenarioResult:
    first = results[0]

    def avg(attr: str) -> float:
        return mean(getattr(result, attr) for result in results)

    averaged = ScenarioResult(
        queue_order=first.queue_order,
        diversity=first.diversity,
        job_duration_s=first.job_duration_s,
        slow_reconfig_s=first.slow_reconfig_s,
        fast_reconfig_s=first.fast_reconfig_s,
        window_size=first.window_size,
        static_makespan_s=avg("static_makespan_s"),
        per_job_makespan_s=avg("per_job_makespan_s"),
        stratocs_makespan_s=avg("stratocs_makespan_s"),
        oracle_makespan_s=avg("oracle_makespan_s"),
        static_speedup=avg("static_speedup"),
        per_job_speedup=avg("per_job_speedup"),
        oracle_loss_pct=avg("oracle_loss_pct"),
        reconfig_fraction=avg("reconfig_fraction"),
        slow_reconfig_reduction_pct=avg("slow_reconfig_reduction_pct"),
        retained_improvement_pct=avg("retained_improvement_pct"),
        verdict="",
    )
    averaged.verdict = scenario_verdict(averaged)
    return averaged


def run_sweep(args: argparse.Namespace, templates: list[TopologyTemplate]) -> list[ScenarioResult]:
    jobs = build_jobs()
    durations = [10 * 60, 60 * 60, 12 * 60 * 60, 3 * 24 * 60 * 60]
    slow_costs = [1, 5, 30, 120]
    fast_costs = [0, 0.001, 0.010]
    window_sizes = [2, 5, 10, 20]
    diversities = ["low", "medium", "high"]
    all_results: list[ScenarioResult] = []
    for diversity in diversities:
        for duration in durations:
            for slow in slow_costs:
                for fast in fast_costs:
                    for window in window_sizes:
                        all_results.append(
                            run_scenario(
                                clustered_queue(jobs),
                                "clustered",
                                diversity,
                                duration,
                                slow,
                                fast,
                                window,
                                templates,
                            )
                        )
                        random_runs = [
                            run_scenario(
                                random_queue(jobs, args.random_seed + seed),
                                "random_avg",
                                diversity,
                                duration,
                                slow,
                                fast,
                                window,
                                templates,
                            )
                            for seed in range(args.random_seeds)
                        ]
                        all_results.append(average_random_results(random_runs))
    return all_results


def write_outputs(output_dir: Path, templates: list[TopologyTemplate], smoke: list[dict[str, object]], results: list[ScenarioResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "templates.json").write_text(json.dumps([asdict(t) for t in templates], indent=2))
    (output_dir / "astra_smoke.json").write_text(json.dumps(smoke, indent=2, sort_keys=True))
    (output_dir / "results.json").write_text(json.dumps([asdict(r) for r in results], indent=2, sort_keys=True))
    with (output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def selected_rows(results: list[ScenarioResult]) -> list[ScenarioResult]:
    rows = []
    for order in ["clustered", "random_avg"]:
        for duration in [10 * 60, 60 * 60, 12 * 60 * 60]:
            for slow in [5, 30, 120]:
                candidates = [
                    r
                    for r in results
                    if r.queue_order == order
                    and r.diversity == "high"
                    and r.job_duration_s == duration
                    and r.slow_reconfig_s == slow
                    and r.fast_reconfig_s == 0.001
                ]
                if candidates:
                    rows.append(min(candidates, key=lambda r: r.stratocs_makespan_s))
    return rows


def fmt_hours(seconds: float) -> str:
    return f"{seconds / 3600.0:.2f}h"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "stratocs_feasibility")
    parser.add_argument("--random-seeds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260514)
    parser.add_argument("--skip-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    templates = topology_library()
    smoke = [] if args.skip_smoke else run_astra_smoke(args.output_dir, templates)
    results = run_sweep(args, templates)
    write_outputs(args.output_dir, templates, smoke, results)

    print("ASTRA_SMOKE")
    for item in smoke:
        print(
            f"{'/'.join(item['topology'])}\tfinished={item['finished_ranks']}/4\t"
            f"exit={item['exit_code']}\tcycles={item['max_cycles']}"
        )

    print("\nSELECTED_SCENARIOS")
    print(
        "order\tjob_duration\tslow_reconfig\tW\tstatic\tper_job\tstratocs\toracle\t"
        "speedup_static\tspeedup_per_job\tretained\tverdict"
    )
    for result in selected_rows(results):
        print(
            f"{result.queue_order}\t{fmt_hours(result.job_duration_s)}\t"
            f"{result.slow_reconfig_s:.0f}s\t{result.window_size}\t"
            f"{fmt_hours(result.static_makespan_s)}\t"
            f"{fmt_hours(result.per_job_makespan_s)}\t"
            f"{fmt_hours(result.stratocs_makespan_s)}\t"
            f"{fmt_hours(result.oracle_makespan_s)}\t"
            f"{result.static_speedup:.3f}x\t{result.per_job_speedup:.3f}x\t"
            f"{result.retained_improvement_pct:.1f}%\t{result.verdict}"
        )

    serious = [r for r in results if r.verdict == "serious"]
    print(f"\nresults_csv\t{args.output_dir / 'results.csv'}")
    print(f"serious_scenarios\t{len(serious)}/{len(results)}")


if __name__ == "__main__":
    main()
