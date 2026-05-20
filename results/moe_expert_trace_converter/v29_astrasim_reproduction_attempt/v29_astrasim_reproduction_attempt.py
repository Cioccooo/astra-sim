#!/usr/bin/env python3
"""V2.9 ASTRA-sim reproduction attempt for MoE prefill EN/SON/RON.

This script intentionally separates three things:

1. Real local work:
   HF expert-selection JSON -> prefill-only pairwise dispatch/combine traffic
   -> Chakra SEND/RECV traces -> ASTRA-sim congestion-aware Switch runs.

2. V2.8 analytical reference:
   Read only for comparison and figure reproduction.

3. Unsupported modes:
   Folded Clos, 2D torus ECMP, and custom RON topologies are not implemented in
   this ASTRA tree. They are reported as unsupported rather than silently faked.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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


ASTRA_AWARE_BIN = (
    REPO_ROOT
    / "build"
    / "astra_analytical"
    / "build"
    / "bin"
    / "AstraSim_Analytical_Congestion_Aware"
)
SYSTEM_CONFIG = REPO_ROOT / "examples" / "system" / "native_collectives" / "HGX-H100-validated.json"
REMOTE_MEMORY_CONFIG = (
    REPO_ROOT / "examples" / "remote_memory" / "analytical" / "no_memory_expansion.json"
)
V28_DIR = Path("/Users/dfx/Python/hespas/results/moe_expert_trace_converter/v28_fair_en_son_ron_ecmp_overhead")
V28_TEACHER_CSV = V28_DIR / "fig_teacher_32gpu_fair_subset.csv"

HIDDEN_SIZE = 4096
BYTES_PER_VALUE = 2
BYTES_PER_SELECTION = HIDDEN_SIZE * BYTES_PER_VALUE
EP_SIZE = 32
BLOCK_SIZE = 16

TARGET_MODES = [
    "EN ECMP-imbalance 1.3x",
    "SON torus ECMP",
    "RON calibrated",
    "RON W=4 1us",
    "RON oracle",
]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    label: str
    path: Path


@dataclass
class Traffic:
    spec: DatasetSpec
    request_ids: list[str]
    files_found: int
    files_used: int
    inferred_num_experts: int
    moe_layers: list[int]
    dispatch_eval: list[list[int]]
    combine_eval: list[list[int]]
    all_remote_bytes: int
    eval_remote_bytes: int
    all_local_bytes: int
    eval_local_bytes: int
    calibration_requests: int
    evaluation_requests: int
    malformed_records: int
    prefill_only: bool


@dataclass
class AstraRun:
    dataset_id: str
    proxy_mode: str
    bandwidth_gbps: float
    dispatch_cycles: int | None
    combine_cycles: int | None
    total_ms: float | None
    dispatch_finished_ranks: int
    combine_finished_ranks: int
    exit_code_dispatch: int | None
    exit_code_combine: int | None
    log_dispatch: str
    log_combine: str


def numeric_json_sort(path: Path) -> int | str:
    return int(path.stem) if path.stem.isdigit() else path.stem


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


def parse_dataset(spec: DatasetSpec, ep_size: int = EP_SIZE, block_size: int = BLOCK_SIZE) -> Traffic:
    files = sorted(spec.path.glob("*.json"), key=numeric_json_sort)
    request_ids = [path.stem for path in files]
    raw_by_request: dict[str, tuple[list[list[int]], list[list[int]], int, int]] = {}
    max_expert = -1
    malformed = 0
    global_token_offset = 0
    moe_layers: set[int] = set()
    per_request_records: dict[str, list[tuple[int, int, list[int]]]] = defaultdict(list)
    per_request_max_rows: dict[str, int] = {}

    for path in files:
        try:
            trace = json.loads(path.read_text())
        except Exception:
            malformed += 1
            continue
        if not isinstance(trace, list) or not trace or not isinstance(trace[0], dict):
            malformed += 1
            continue
        prefill = trace[0]
        max_rows = 0
        for layer_str, rows in prefill.items():
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
                per_request_records[path.stem].append((layer_id, global_token_offset + row_index, parsed))
        per_request_max_rows[path.stem] = max_rows
        global_token_offset += max_rows

    if max_expert < 0:
        raise ValueError(f"No expert IDs found in {spec.path}")
    num_experts = max_expert + 1

    for request_id, records in per_request_records.items():
        dispatch = zero_matrix(ep_size)
        combine = zero_matrix(ep_size)
        local_bytes = 0
        remote_bytes = 0
        for _layer_id, global_token_index, experts in records:
            src = block_source_rank(global_token_index, ep_size, block_size)
            for expert_id in experts:
                dst = expert_rank(expert_id, ep_size, num_experts)
                if src == dst:
                    local_bytes += BYTES_PER_SELECTION
                    continue
                dispatch[src][dst] += BYTES_PER_SELECTION
                combine[dst][src] += BYTES_PER_SELECTION
                remote_bytes += 2 * BYTES_PER_SELECTION
        raw_by_request[request_id] = (dispatch, combine, local_bytes, remote_bytes)

    cal_count = max(1, min(math.ceil(len(request_ids) * 0.10), len(request_ids) - 1))
    eval_ids = request_ids[cal_count:]
    dispatch_eval = zero_matrix(ep_size)
    combine_eval = zero_matrix(ep_size)
    eval_local = 0
    eval_remote = 0
    all_local = 0
    all_remote = 0
    for request_id in request_ids:
        dispatch, combine, local_bytes, remote_bytes = raw_by_request[request_id]
        all_local += local_bytes
        all_remote += remote_bytes
        if request_id in eval_ids:
            add_matrix(dispatch_eval, dispatch)
            add_matrix(combine_eval, combine)
            eval_local += local_bytes
            eval_remote += remote_bytes

    return Traffic(
        spec=spec,
        request_ids=request_ids,
        files_found=len(files),
        files_used=len(raw_by_request),
        inferred_num_experts=num_experts,
        moe_layers=sorted(moe_layers),
        dispatch_eval=dispatch_eval,
        combine_eval=combine_eval,
        all_remote_bytes=all_remote,
        eval_remote_bytes=eval_remote,
        all_local_bytes=all_local,
        eval_local_bytes=eval_local,
        calibration_requests=cal_count,
        evaluation_requests=len(eval_ids),
        malformed_records=malformed,
        prefill_only=True,
    )


def add_attr(node: ChakraNode, name: str, value: int | bool) -> None:
    if isinstance(value, bool):
        node.attr.append(ChakraAttr(name=name, bool_val=value))
    else:
        node.attr.append(ChakraAttr(name=name, uint64_val=int(value)))


def write_pairwise_trace(matrix: list[list[int]], trace_dir: Path, prefix: str, phase: str) -> Path:
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


def write_network_config(path: Path, n: int, bandwidth_gbps: float, latency_ns: float = 0.0) -> None:
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


def run_astra_phase(prefix_path: Path, network_config: Path, log_path: Path) -> tuple[int | None, int, int | None]:
    cmd = [
        str(ASTRA_AWARE_BIN),
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
        timeout=180,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout)
    cycles = [int(match) for match in re.findall(r"finished, ([0-9]+) cycles", proc.stdout)]
    return proc.returncode, len(cycles), max(cycles) if cycles else None


def run_astra_proxy(traffic: Traffic, output_dir: Path, proxy_mode: str, bandwidth_gbps: float) -> AstraRun:
    trace_base = output_dir / "chakra_traces" / traffic.spec.dataset_id
    dispatch_prefix = write_pairwise_trace(
        traffic.dispatch_eval,
        trace_base / "dispatch",
        "workload",
        "PREFILL_DISPATCH",
    )
    combine_prefix = write_pairwise_trace(
        traffic.combine_eval,
        trace_base / "combine",
        "workload",
        "PREFILL_COMBINE",
    )
    network_config = output_dir / "astra_configs" / f"switch_{int(bandwidth_gbps)}gbps.yml"
    write_network_config(network_config, EP_SIZE, bandwidth_gbps)
    logs = output_dir / "astra_logs" / traffic.spec.dataset_id / proxy_mode.replace(" ", "_")
    d_exit, d_finished, d_cycles = run_astra_phase(dispatch_prefix, network_config, logs / "dispatch.log")
    c_exit, c_finished, c_cycles = run_astra_phase(combine_prefix, network_config, logs / "combine.log")
    total_ms = None
    if d_cycles is not None and c_cycles is not None:
        # ASTRA analytical time unit is cycles/nanoseconds for these configs.
        total_ms = (d_cycles + c_cycles) / 1e6
    return AstraRun(
        dataset_id=traffic.spec.dataset_id,
        proxy_mode=proxy_mode,
        bandwidth_gbps=bandwidth_gbps,
        dispatch_cycles=d_cycles,
        combine_cycles=c_cycles,
        total_ms=total_ms,
        dispatch_finished_ranks=d_finished,
        combine_finished_ranks=c_finished,
        exit_code_dispatch=d_exit,
        exit_code_combine=c_exit,
        log_dispatch=str(logs / "dispatch.log"),
        log_combine=str(logs / "combine.log"),
    )


def read_v28_reference() -> dict[tuple[str, str], dict[str, str]]:
    rows = list(csv.DictReader(V28_TEACHER_CSV.open()))
    return {
        (row["dataset_id"], row["network_mode"]): row
        for row in rows
        if row["network_mode"] in TARGET_MODES and int(row["ep_size"]) == EP_SIZE
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def plot_reference(path_png: Path, path_pdf: Path, rows: list[dict[str, Any]]) -> None:
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
        fig, ax = plt.subplots(figsize=(15, 6))
        for i, mode in enumerate(modes):
            values = [
                float(next(row for row in rows if row["dataset_label"] == dataset and row["network_mode"] == mode)["analytical_v28_ms"])
                for dataset in datasets
            ]
            offsets = [x + (i - (len(modes) - 1) / 2) * width for x in xs]
            ax.bar(offsets, values, width=width, label=mode)
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
        # Minimal dependency-free fallback for this local environment.
        from PIL import Image, ImageDraw, ImageFont

        width_px, height_px = 1800, 900
        margin_l, margin_r, margin_t, margin_b = 150, 60, 90, 220
        plot_w = width_px - margin_l - margin_r
        plot_h = height_px - margin_t - margin_b
        image = Image.new("RGB", (width_px, height_px), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        values_by_dataset_mode = {
            (row["dataset_label"], row["network_mode"]): float(row["analytical_v28_ms"])
            for row in rows
        }
        max_val = max(values_by_dataset_mode.values()) if values_by_dataset_mode else 1.0
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
                val = values_by_dataset_mode[(dataset, mode)]
                h = int(plot_h * val / max_val)
                x0 = int(base_x + mi * bar_w)
                x1 = int(x0 + bar_w * 0.9)
                y0 = margin_t + plot_h - h
                y1 = margin_t + plot_h
                draw.rectangle((x0, y0, x1, y1), fill=colors[mi % len(colors)])
            draw.text((int(margin_l + di * group_w + 5), margin_t + plot_h + 20), dataset, fill="black", font=font)
        legend_x, legend_y = margin_l, height_px - 90
        for mi, mode in enumerate(modes):
            x = legend_x + (mi % 3) * 430
            y = legend_y + (mi // 3) * 30
            draw.rectangle((x, y, x + 18, y + 18), fill=colors[mi % len(colors)])
            draw.text((x + 24, y + 2), mode, fill="black", font=font)
        path_png.parent.mkdir(parents=True, exist_ok=True)
        image.save(path_png)
        image.save(path_pdf, "PDF", resolution=220.0)


def make_readme(output_dir: Path, validation: dict[str, Any]) -> None:
    readme = f"""# V2.9 ASTRA-sim Reproduction Attempt

## Bottom Line

ASTRA-sim was actually used in this folder, but only for congestion-aware
1D `Switch` proxy runs generated from Chakra SEND/RECV traces. This ASTRA tree
cannot faithfully model the five target EN/SON/RON modes:

- Folded-Clos ECMP is not an ASTRA analytical topology here.
- 2D torus ECMP is not supported as arbitrary ECMP graph routing.
- RON calibrated / W=4 / oracle require per-request custom degree-4 topologies;
  the analytical parser only supports `Ring`, `Switch`, and `FullyConnected`.
- Congestion-aware analytical backend supports only 1D topology.

Therefore ASTRA-sim cannot replace the V2.8 analytical graph evaluator for final
timing without extending the network backend.

## Pipeline

```mermaid
flowchart LR
  A["HF expert-selection JSON folders"] --> B["prefill only: trace[0]"]
  B --> C["block_by_token source GPU"]
  B --> D["block expert placement"]
  C --> E["pairwise dispatch matrix"]
  D --> E
  E --> F["Chakra COMM_SEND/COMM_RECV traces"]
  F --> G["ASTRA congestion-aware Switch proxy"]
  E --> H["V2.8 analytical reference comparison"]
```

## ASTRA Support Audit

| Question | Answer |
|---|---|
| Arbitrary topology/link bandwidth/link latency? | No arbitrary topology. YAML supports per-dimension bandwidth/latency, but topology names are limited to `Ring`, `Switch`, `FullyConnected`. |
| Pairwise send/recv traffic? | Yes. Chakra `COMM_SEND_NODE` / `COMM_RECV_NODE` works and calls `front_end_sim_send/recv`. |
| Consume Chakra traces? | Yes. The generated `.et` traces are consumed by ASTRA in this run. |
| Chakra represent dispatch/combine? | Yes, as pairwise send/recv AllToAllv-like traffic. |
| Folded-Clos, 2D torus ECMP, custom RON? | Not faithfully in this ASTRA backend. |
| Per-request topology changes? | Not inside one ASTRA run. Would require separate runs/config swaps plus external accounting. |
| Congestion-aware routing/queueing? | Only for 1D `Ring`/`Switch`/`FullyConnected`; no custom ECMP graph routing. |
| MLSynth/STAGE needed? | No. This is communication-only prefill from real router traces; direct Chakra generation is simpler. |

## Validation

```json
{json.dumps(validation, indent=2, sort_keys=True)}
```
"""
    (output_dir / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "moe_expert_trace_converter" / "v29_astrasim_reproduction_attempt",
    )
    parser.add_argument("--skip-astra", action="store_true")
    parser.add_argument("--force-astra", action="store_true")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    specs = [
        DatasetSpec("qwen_mmlu_ml", "Qwen MMLU ML", Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/mmlu/machine_learning")),
        DatasetSpec("qwen_livecode", "Qwen LiveCodeBench", Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/livecodebench/execution")),
        DatasetSpec("qwen_mmlu_zh_anatomy", "Qwen ZH Anatomy", Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/mmlu_ZH_CN/anatomy")),
        DatasetSpec("deepseek_livecode", "DeepSeek LiveCodeBench", Path("/Users/dfx/Python/trace/cognitivecomputations/DeepSeek-R1-AWQ/livecodebench/execution")),
    ]
    traffic_items = [parse_dataset(spec) for spec in specs]
    traffic_rows = []
    validation_rows = []
    for traffic in traffic_items:
        dispatch_bytes = matrix_sum(traffic.dispatch_eval)
        combine_bytes = matrix_sum(traffic.combine_eval)
        traffic_rows.append(
            {
                "dataset_id": traffic.spec.dataset_id,
                "dataset_label": traffic.spec.label,
                "path": str(traffic.spec.path),
                "files_found": traffic.files_found,
                "files_used": traffic.files_used,
                "calibration_requests": traffic.calibration_requests,
                "evaluation_requests": traffic.evaluation_requests,
                "prefill_only": traffic.prefill_only,
                "source_policy": "block_by_token",
                "block_size": BLOCK_SIZE,
                "expert_placement": "block",
                "hidden_size": HIDDEN_SIZE,
                "bytes_per_value": BYTES_PER_VALUE,
                "inferred_num_experts": traffic.inferred_num_experts,
                "moe_layers": len(traffic.moe_layers),
                "all_local_bytes_excluded": traffic.all_local_bytes,
                "all_remote_bytes_dispatch_plus_combine": traffic.all_remote_bytes,
                "eval_local_bytes_excluded": traffic.eval_local_bytes,
                "eval_remote_bytes_dispatch_plus_combine": traffic.eval_remote_bytes,
                "eval_dispatch_bytes": dispatch_bytes,
                "eval_combine_bytes": combine_bytes,
                "malformed_records": traffic.malformed_records,
            }
        )
        validation_rows.append(
            {
                "dataset_id": traffic.spec.dataset_id,
                "dispatch_equals_combine": dispatch_bytes == combine_bytes,
                "eval_remote_equals_dispatch_plus_combine": traffic.eval_remote_bytes == dispatch_bytes + combine_bytes,
                "only_prefill_trace0": traffic.prefill_only,
                "local_traffic_excluded": True,
            }
        )

    astra_runs_path = out / "astra_runs.json"
    astra_runs: list[AstraRun] = []
    if astra_runs_path.exists() and not args.force_astra:
        astra_runs = [AstraRun(**item) for item in json.loads(astra_runs_path.read_text())]
    elif not args.skip_astra:
        for traffic in traffic_items:
            astra_runs.append(run_astra_proxy(traffic, out, "astra_ca_switch_400g", 400))
            astra_runs.append(run_astra_proxy(traffic, out, "astra_ca_switch_1600g", 1600))

    astra_by_dataset_mode = {(run.dataset_id, run.proxy_mode): run for run in astra_runs}
    reference = read_v28_reference()
    comparison_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []
    for spec in specs:
        for mode in TARGET_MODES:
            ref = reference.get((spec.dataset_id, mode))
            analytical_ms = float(ref["completion_time_ms"]) if ref else None
            if mode == "EN ECMP-imbalance 1.3x":
                run = astra_by_dataset_mode.get((spec.dataset_id, "astra_ca_switch_400g"))
                proxy_ms = (run.total_ms * 1.3) if run and run.total_ms is not None else None
                proxy_name = "ASTRA congestion-aware 1D Switch 400Gbps x 1.3"
                support = "approximation"
                notes = "Not folded-Clos ECMP; 1.3x imbalance applied externally."
            else:
                run = astra_by_dataset_mode.get((spec.dataset_id, "astra_ca_switch_1600g"))
                proxy_ms = run.total_ms if run else None
                proxy_name = "ASTRA congestion-aware 1D Switch 1.6Tbps"
                support = "unsupported_target_topology"
                notes = "Proxy only. Does not model degree-4 torus ECMP or RON custom topology/reconfiguration."
                if mode == "RON W=4 1us" and ref:
                    notes += " V2.8 includes per-request 1us reconfiguration externally; ASTRA proxy does not."
            rel_diff = None
            if analytical_ms and proxy_ms is not None:
                rel_diff = (proxy_ms - analytical_ms) / analytical_ms
            row = {
                "dataset_id": spec.dataset_id,
                "dataset_label": spec.label,
                "network_mode": mode,
                "analytical_v28_ms": analytical_ms,
                "astra_proxy_ms": proxy_ms,
                "relative_difference_vs_v28": rel_diff,
                "astra_proxy_name": proxy_name,
                "support_status": support,
                "notes": notes,
            }
            comparison_rows.append(row)
            figure_rows.append(row)

    write_csv(out / "traffic_summary.csv", traffic_rows)
    write_csv(out / "validation_rows.csv", validation_rows)
    write_csv(out / "summary_analytical_vs_astra_proxy.csv", comparison_rows)
    (out / "summary_analytical_vs_astra_proxy.json").write_text(json.dumps(comparison_rows, indent=2, sort_keys=True))
    (out / "astra_runs.json").write_text(json.dumps([asdict(run) for run in astra_runs], indent=2, sort_keys=True))

    validation = {
        "only_four_specified_workloads": [spec.dataset_id for spec in specs],
        "prefill_only_trace0": all(row["only_prefill_trace0"] for row in validation_rows),
        "source_policy": "block_by_token",
        "expert_placement": "block",
        "local_traffic_excluded": True,
        "dispatch_combine_sequential": "ASTRA proxy runs dispatch and combine as separate phases and sums them.",
        "byte_conservation_all_passed": all(row["eval_remote_equals_dispatch_plus_combine"] for row in validation_rows),
        "dispatch_equals_combine_all_passed": all(row["dispatch_equals_combine"] for row in validation_rows),
        "en_400gbps_settings": True,
        "son_ron_pure_optical_no_eps_fallback": "Not faithfully modeled by ASTRA; preserved only in V2.8 reference rows.",
        "son_ron_degree4_1p6tbps": "Not faithfully modeled by ASTRA; ASTRA proxy is 1D Switch at 1.6Tbps.",
        "ron_calibrated_first_10pct": "Parsed and counted; topology choice not modeled in ASTRA.",
        "ron_w4_previous4_plus_1us": "Available only in V2.8 reference; not modeled in ASTRA proxy.",
        "astra_sim_actually_used": bool(astra_runs),
        "full_reproduction_feasible": False,
    }
    (out / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True))
    plot_reference(out / "fig_32gpu_moe_prefill_v28_reference.png", out / "fig_32gpu_moe_prefill_v28_reference.pdf", figure_rows)
    make_readme(out, validation)
    shutil.copy2(Path(__file__), out / "v29_astrasim_reproduction_attempt.py")

    print("OUTPUT", out)
    print("ASTRA_RUNS", len(astra_runs))
    for run in astra_runs:
        print(
            f"{run.dataset_id} {run.proxy_mode} total_ms={run.total_ms} "
            f"finished={run.dispatch_finished_ranks}/{EP_SIZE}+{run.combine_finished_ranks}/{EP_SIZE}"
        )
    print("FULL_REPRODUCTION_FEASIBLE false")


if __name__ == "__main__":
    main()
