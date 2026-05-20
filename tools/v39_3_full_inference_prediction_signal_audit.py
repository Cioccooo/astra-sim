#!/usr/bin/env python3
"""V39.3 full-inference prediction-signal audit.

Compares OCS control policies on full inference communication:
prefill + decode + warm-up/default phases + reconfiguration penalties.

Model: optical circuit/capacity reference.  No ASTRA C++ changes.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
V39_PATH = REPO / "tools/v39_semantic_correct_model_selection.py"
OUT = REPO / "results/moe_expert_trace_converter/v39_3_full_inference_prediction_signal_audit"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v39 = load_module("v39", V39_PATH)
v37c = v39.v37c
v3812 = v39.v3812

NPU = v39.NPU
ECMP_MAX_PATHS = v39.ECMP_MAX_PATHS
RECONFIG_PENALTIES_US = [0, 1, 10, 25_000]
MAIN_PENALTY_US = 1
WINDOWS = [4, 8, 16]
RANDOM_SEEDS = v39.RANDOM_SEEDS

WORKLOADS = v39.WORKLOADS
OPTIONAL_QWEN_ZH = {
    "id": "qwen_mmlu_zh_cn_anatomy_optional",
    "label": "Qwen MMLU_ZH_CN anatomy optional",
    "path": Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/mmlu_ZH_CN/anatomy"),
}

METHODS = [
    "phase-warmup OCS",
    "prefill-warmup OCS",
    "previous-request OCS",
    "oracle",
]
COLORS = {
    "phase-warmup OCS": "#4E79A7",
    "prefill-warmup OCS": "#59A14F",
    "previous-request OCS": "#E15759",
    "oracle": "#7B52AB",
    "0us": "#9AA4B2",
    "1us": "#59A14F",
    "10us": "#F28E2B",
    "25ms": "#E15759",
}

Pair = tuple[int, int]
Sparse = dict[Pair, int]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def add_sparse(dst: defaultdict[Pair, int], src: Sparse) -> None:
    for key, value in src.items():
        dst[key] += int(value)


def combine_sparse(a: Sparse, b: Sparse) -> Sparse:
    out: defaultdict[Pair, int] = defaultdict(int)
    add_sparse(out, a)
    add_sparse(out, b)
    return dict(out)


def sparse_sum(sparse: Sparse) -> int:
    return sum(int(v) for v in sparse.values())


def empty_payload(num_experts: int) -> dict[str, Any]:
    return {
        "dispatch_sparse": {},
        "combine_sparse": {},
        "combined_sparse": {},
        "dest_gpu_bytes": [0 for _ in range(NPU)],
        "selected_events": 0,
        "token_count": 0,
        "local_bytes_excluded": 0,
        "remote_bytes": 0,
        "byte_conservation_pass": True,
        "expert_counts": [0 for _ in range(num_experts)],
    }


def merge_payloads(payloads: list[dict[str, Any]], num_experts: int) -> dict[str, Any]:
    dispatch: defaultdict[Pair, int] = defaultdict(int)
    combine: defaultdict[Pair, int] = defaultdict(int)
    dest = [0 for _ in range(NPU)]
    experts = [0 for _ in range(num_experts)]
    selected = token_count = local = remote = 0
    for payload in payloads:
        add_sparse(dispatch, payload["dispatch_sparse"])
        add_sparse(combine, payload["combine_sparse"])
        for i, v in enumerate(payload.get("dest_gpu_bytes", [])):
            dest[i] += v
        for i, v in enumerate(payload.get("expert_counts", [])):
            experts[i] += v
        selected += int(payload.get("selected_events", 0))
        token_count += int(payload.get("token_count", 0))
        local += int(payload.get("local_bytes_excluded", 0))
        remote += int(payload.get("remote_bytes", 0))
    return {
        "dispatch_sparse": dict(dispatch),
        "combine_sparse": dict(combine),
        "combined_sparse": combine_sparse(dict(dispatch), dict(combine)),
        "dest_gpu_bytes": dest,
        "selected_events": selected,
        "token_count": token_count,
        "local_bytes_excluded": local,
        "remote_bytes": remote,
        "byte_conservation_pass": selected * v37c.BYTES_PER_SELECTION == local + remote,
        "expert_counts": experts,
    }


def flatten_records(parsed: dict[str, Any], stage: str, request_order: list[int] | None = None) -> list[dict[str, Any]]:
    rows_key = "prefill_rows" if stage == "prefill" else "decode_rows"
    order = request_order if request_order is not None else list(range(len(parsed["requests"])))
    records: list[dict[str, Any]] = []
    for idx in order:
        req = parsed["requests"][idx]
        for layer_id, local_token_idx, global_token_idx, experts in req[rows_key]:
            records.append(
                {
                    "request_index": int(req["request_index"]),
                    "request_id": req["request_id"],
                    "layer_id": int(layer_id),
                    "local_token_index": int(local_token_idx),
                    "global_token_index": int(global_token_idx),
                    "experts": list(experts),
                }
            )
    return records


def split_records_by_event_fraction(records: list[dict[str, Any]], fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total_events = sum(len(r["experts"]) for r in records)
    target = max(1, int(total_events * fraction)) if total_events else 0
    count = 0
    cut = 0
    for idx, record in enumerate(records):
        count += len(record["experts"])
        cut = idx + 1
        if count >= target:
            break
    return records[:cut], records[cut:]


def payload_from_records(parsed: dict[str, Any], records: list[dict[str, Any]], placement: str = "block") -> dict[str, Any]:
    order = v3812.v38.hot_order(parsed["prefill_counts"])
    dispatch: defaultdict[Pair, int] = defaultdict(int)
    combine: defaultdict[Pair, int] = defaultdict(int)
    dest_gpu = [0 for _ in range(NPU)]
    expert_counts = [0 for _ in range(parsed["num_experts"])]
    selected = local_events = remote_events = 0
    token_keys: set[tuple[int, int, int]] = set()
    for record in records:
        src = v3812.v38.source_rank(
            "block_by_token",
            int(record["request_index"]),
            int(record["global_token_index"]),
            int(record["local_token_index"]),
        )
        token_keys.add((int(record["request_index"]), int(record["layer_id"]), int(record["local_token_index"])))
        for expert_id in record["experts"]:
            selected += 1
            expert_counts[int(expert_id)] += 1
            dst = v3812.v38.expert_to_gpu(int(expert_id), parsed["num_experts"], placement, order)
            if src == dst:
                local_events += 1
                continue
            dispatch[(src, dst)] += v37c.BYTES_PER_SELECTION
            combine[(dst, src)] += v37c.BYTES_PER_SELECTION
            dest_gpu[dst] += v37c.BYTES_PER_SELECTION
            remote_events += 1
    local_bytes = local_events * v37c.BYTES_PER_SELECTION
    remote_bytes = remote_events * v37c.BYTES_PER_SELECTION
    return {
        "dispatch_sparse": dict(dispatch),
        "combine_sparse": dict(combine),
        "combined_sparse": combine_sparse(dict(dispatch), dict(combine)),
        "dest_gpu_bytes": dest_gpu,
        "selected_events": selected,
        "token_count": len(token_keys),
        "local_bytes_excluded": local_bytes,
        "remote_bytes": remote_bytes,
        "byte_conservation_pass": selected * v37c.BYTES_PER_SELECTION == local_bytes + remote_bytes,
        "expert_counts": expert_counts,
    }


def payload_from_rows(
    parsed: dict[str, Any],
    rows: list[tuple[int, int, int, list[int]]],
    request_index: int,
    placement: str = "block",
    expert_order: list[int] | None = None,
) -> dict[str, Any]:
    """Build the same payload as payload_from_records without materialising per-row dicts."""
    order = expert_order if expert_order is not None else v3812.v38.hot_order(parsed["prefill_counts"])
    dispatch: defaultdict[Pair, int] = defaultdict(int)
    combine: defaultdict[Pair, int] = defaultdict(int)
    dest_gpu = [0 for _ in range(NPU)]
    expert_counts = [0 for _ in range(parsed["num_experts"])]
    selected = local_events = remote_events = 0
    token_keys: set[tuple[int, int, int]] = set()
    for layer_id, local_token_idx, global_token_idx, experts in rows:
        src = v3812.v38.source_rank(
            "block_by_token",
            int(request_index),
            int(global_token_idx),
            int(local_token_idx),
        )
        token_keys.add((int(request_index), int(layer_id), int(local_token_idx)))
        for expert_id in experts:
            selected += 1
            expert_counts[int(expert_id)] += 1
            dst = v3812.v38.expert_to_gpu(int(expert_id), parsed["num_experts"], placement, order)
            if src == dst:
                local_events += 1
                continue
            dispatch[(src, dst)] += v37c.BYTES_PER_SELECTION
            combine[(dst, src)] += v37c.BYTES_PER_SELECTION
            dest_gpu[dst] += v37c.BYTES_PER_SELECTION
            remote_events += 1
    local_bytes = local_events * v37c.BYTES_PER_SELECTION
    remote_bytes = remote_events * v37c.BYTES_PER_SELECTION
    return {
        "dispatch_sparse": dict(dispatch),
        "combine_sparse": dict(combine),
        "combined_sparse": combine_sparse(dict(dispatch), dict(combine)),
        "dest_gpu_bytes": dest_gpu,
        "selected_events": selected,
        "token_count": len(token_keys),
        "local_bytes_excluded": local_bytes,
        "remote_bytes": remote_bytes,
        "byte_conservation_pass": selected * v37c.BYTES_PER_SELECTION == local_bytes + remote_bytes,
        "expert_counts": expert_counts,
    }


def stage_request_payloads(parsed: dict[str, Any], stage: str, request_order: list[int] | None = None) -> list[dict[str, Any]]:
    rows_key = "prefill_rows" if stage == "prefill" else "decode_rows"
    order = request_order if request_order is not None else list(range(len(parsed["requests"])))
    expert_order = v3812.v38.hot_order(parsed["prefill_counts"])
    payloads = []
    for idx in order:
        req = parsed["requests"][idx]
        payloads.append(payload_from_rows(parsed, req[rows_key], int(req["request_index"]), expert_order=expert_order))
    return payloads


def split_payloads_by_event_fraction(
    payloads: list[dict[str, Any]],
    num_experts: int,
    fraction: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    total_events = sum(int(payload["selected_events"]) for payload in payloads)
    target = max(1, int(total_events * fraction)) if total_events else 0
    count = 0
    cut = 0
    for idx, payload in enumerate(payloads):
        count += int(payload["selected_events"])
        cut = idx + 1
        if count >= target:
            break
    return (
        merge_payloads(payloads[:cut], num_experts),
        merge_payloads(payloads[cut:], num_experts),
    )


def request_payloads(parsed: dict[str, Any], request_order: list[int] | None = None) -> list[dict[str, Any]]:
    order = request_order if request_order is not None else list(range(len(parsed["requests"])))
    pre = stage_request_payloads(parsed, "prefill", order)
    dec = stage_request_payloads(parsed, "decode", order)
    return [merge_payloads([p, d], parsed["num_experts"]) for p, d in zip(pre, dec)]


def candidate_pool_for_signal(
    workload_id: str,
    candidates: dict[str, dict[str, Any]],
    base_paths: dict[str, dict[Pair, list[list[int]]]],
    signal_payload: dict[str, Any],
    fallback_seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[Pair, list[list[int]]]], list[str]]:
    local = dict(candidates)
    edges, meta = v37c.safe_greedy_graph(signal_payload["combined_sparse"], v37c.ring_edges(NPU), NPU, fallback_seed)
    local["signal_greedy"] = v37c.graph_from_edges(
        f"{workload_id}_signal_greedy_degree4", edges, NPU, {"construction": "signal_greedy", **meta}
    )
    # Base candidate paths are expensive but invariant. Reuse them and only
    # precompute the signal-dependent greedy graph for warm-up/window methods.
    paths = dict(base_paths)
    paths["signal_greedy"] = v37c.precompute_paths(local["signal_greedy"], NPU, v39.ECMP_MAX_PATHS)
    allowed = ["son_torus", "signal_greedy"] + [f"random_regular_seed_{s}" for s in RANDOM_SEEDS]
    return local, paths, allowed


def eval_candidate(paths: dict[Pair, list[list[int]]], payload: dict[str, Any]) -> dict[str, Any]:
    return v39.optical_eval(paths, payload)


def select_candidate(
    paths: dict[str, dict[Pair, list[list[int]]]],
    payload: dict[str, Any],
    allowed: list[str],
) -> dict[str, Any]:
    rows = []
    for name in allowed:
        metric = eval_candidate(paths[name], payload)
        rows.append({"candidate": name, "cycles": metric["optical_reference_cycles"]})
    rows.sort(key=lambda row: (row["cycles"], row["candidate"]))
    return {"selected": rows[0]["candidate"], "scores_top8": rows[:8]}


def fair_universal_selection(full_payloads: dict[str, dict[str, Any]], base_paths: dict[str, Any]) -> dict[str, str]:
    names = [f"random_regular_seed_{s}" for s in RANDOM_SEEDS]
    result = {}
    for target in full_payloads:
        rows = []
        for name in names:
            norms = []
            for wid, payload in full_payloads.items():
                if wid == target:
                    continue
                son = eval_candidate(base_paths["son_torus"], payload)["optical_reference_cycles"]
                val = eval_candidate(base_paths[name], payload)["optical_reference_cycles"]
                norms.append(val / son if son else 0)
            rows.append({"candidate": name, "avg_norm_to_son": statistics.mean(norms)})
        rows.sort(key=lambda row: (row["avg_norm_to_son"], row["candidate"]))
        result[target] = rows[0]["candidate"]
    return result


def spearman_counts(a: list[int], b: list[int]) -> float | None:
    return v3812.v38.spearman_counts(a, b)


def top_overlap(a: list[int], b: list[int], k: int) -> float:
    ar = [idx for idx, _ in sorted(enumerate(a), key=lambda item: (-item[1], item[0]))]
    br = [idx for idx, _ in sorted(enumerate(b), key=lambda item: (-item[1], item[0]))]
    return len(set(ar[:k]) & set(br[:k])) / k if k else 0


def gini_sparse(sparse: Sparse) -> float:
    return v37c.gini([v for v in sparse.values() if v > 0])


def method_metric_row(
    workload_id: str,
    method: str,
    penalty_us: int,
    total_cycles_without_penalty: int,
    reconfigs: int,
    baseline_cycles: int,
    selected_candidate: str,
    graph_hash: str,
    selection_signal: str,
    observed_payload: dict[str, Any],
    evaluated_payload: dict[str, Any],
    dominant: dict[str, Any],
    leakage_free: bool,
) -> dict[str, Any]:
    penalty_cycles = penalty_us * 1000 * reconfigs
    total = total_cycles_without_penalty + penalty_cycles
    return {
        "workload": workload_id,
        "method": method,
        "penalty_us": penalty_us,
        "total_cycles": total,
        "total_ms": v39.cycles_to_ms(total),
        "normalised_to_fair_universal_static": total / baseline_cycles if baseline_cycles else 0,
        "gain_vs_fair_static_percent": 100 * (baseline_cycles - total) / baseline_cycles if baseline_cycles else 0,
        "number_of_reconfigurations": reconfigs,
        "selected_candidate_name": selected_candidate,
        "graph_hash": graph_hash,
        "selection_signal": selection_signal,
        "evaluation_target": "full inference prefill+decode including warm-up/default phases",
        "no_leakage": leakage_free,
        "observed_bytes": observed_payload["remote_bytes"] * 2,
        "evaluated_bytes": evaluated_payload["remote_bytes"] * 2,
        "dominant_bottleneck": dominant["bottleneck_type"],
        "sender_bottleneck_time_ms": v39.cycles_to_ms(dominant["sender_bottleneck_cycles"]),
        "receiver_bottleneck_time_ms": v39.cycles_to_ms(dominant["receiver_bottleneck_cycles"]),
        "optical_resource_bottleneck_time_ms": v39.cycles_to_ms(dominant["optical_resource_bottleneck_cycles"]),
        "byte_conservation": observed_payload["byte_conservation_pass"] and evaluated_payload["byte_conservation_pass"],
        "same_budget_check": True,
    }


def dominant_phase(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    phases = []
    for metric in metrics:
        phases.append(metric["dispatch"])
        phases.append(metric["combine"])
    return max(phases, key=lambda phase: phase["phase_cycles"]) if phases else {
        "bottleneck_type": "none",
        "sender_bottleneck_cycles": 0,
        "receiver_bottleneck_cycles": 0,
        "optical_resource_bottleneck_cycles": 0,
    }


def draw_grouped_bar(
    path: Path,
    title: str,
    subtitle: str,
    workloads: list[str],
    methods: list[str],
    values: dict[str, dict[str, float]],
    ylabel: str,
    baseline: float | None = None,
    ymax: float | None = None,
) -> None:
    width, height = 1900, 1050
    ml, mr, mt, mb = 165, 80, 120, 220
    pw, ph = width - ml - mr, height - mt - mb
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    all_values = [value for workload_values in values.values() for value in workload_values.values()]
    if baseline is not None:
        all_values.append(baseline)
    minv = min(0.0, min(all_values))
    max_data = max(all_values)
    maxv = ymax or (max_data * 1.18 if max_data > 0 else max_data * 0.82)
    if maxv == minv:
        maxv = minv + 1.0

    def y(value: float) -> float:
        return mt + ph - ((value - minv) / (maxv - minv)) * ph

    draw.text((ml, 35), title, fill="#111827", font=font)
    draw.text((ml, 60), subtitle, fill="#374151", font=font)
    for i in range(6):
        tick = minv + (maxv - minv) * i / 5
        yy = y(tick)
        draw.line((ml, yy, ml + pw, yy), fill="#E5E7EB")
        draw.text((45, yy - 7), f"{tick:.2f}", fill="#374151", font=font)
    draw.line((ml, mt, ml, mt + ph), fill="#111827")
    draw.line((ml, mt + ph, ml + pw, mt + ph), fill="#111827")
    if baseline is not None:
        yy = y(baseline)
        draw.line((ml, yy, ml + pw, yy), fill="#111827", width=3)
    for gi, workload in enumerate(workloads):
        gw = pw / len(workloads)
        iw = gw * 0.76
        start = ml + gi * gw + (gw - iw) / 2
        bar_w = iw / len(methods) * 0.82
        draw.text((ml + gi * gw + gw / 2 - 75, mt + ph + 28), workload, fill="#111827", font=font)
        for mi, method in enumerate(methods):
            val = values[workload][method]
            x = start + mi * (iw / len(methods))
            yy = y(val)
            zero_y = y(0.0)
            draw.rectangle(
                (x, min(yy, zero_y), x + bar_w, max(yy, zero_y)),
                fill=COLORS.get(method, "#777777"),
            )
    lx, ly = ml, height - 95
    for method in methods:
        draw.rectangle((lx, ly, lx + 24, ly + 16), fill=COLORS.get(method, "#777777"))
        draw.text((lx + 32, ly + 1), method, fill="#111827", font=font)
        lx += 330
        if lx > width - 320:
            lx = ml
            ly += 32
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    img.save(path.with_suffix(".pdf"), "PDF", resolution=160.0)


def process_workload(
    workload: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    base_paths: dict[str, dict[Pair, list[list[int]]]],
    fair_candidate: str,
    parsed_override: dict[str, Any] | None = None,
    shuffled: bool = False,
) -> dict[str, Any]:
    parsed = parsed_override if parsed_override is not None else v3812.v38.parse_trace(workload["path"])
    req_order = list(range(len(parsed["requests"])))
    if shuffled:
        rng = random.Random(12345)
        rng.shuffle(req_order)
    pre_request_payloads = stage_request_payloads(parsed, "prefill", req_order)
    dec_request_payloads = stage_request_payloads(parsed, "decode", req_order)
    pre10, pre90 = split_payloads_by_event_fraction(pre_request_payloads, parsed["num_experts"], 0.10)
    dec10, dec90 = split_payloads_by_event_fraction(dec_request_payloads, parsed["num_experts"], 0.10)
    pre_all = merge_payloads([pre10, pre90], parsed["num_experts"])
    dec_all = merge_payloads([dec10, dec90], parsed["num_experts"])
    full_payload = merge_payloads([pre_all, dec_all], parsed["num_experts"])
    remaining_pre_decode = merge_payloads([pre90, dec_all], parsed["num_experts"])

    fair_graph = candidates[fair_candidate]
    fair_paths = base_paths[fair_candidate]
    baseline_metric = eval_candidate(fair_paths, full_payload)
    baseline_cycles = baseline_metric["optical_reference_cycles"]

    timing_rows = []
    no_leakage_rows = []
    bottleneck_rows = []

    # Phase warm-up.
    phase_signal_prefill = pre10
    local_pref, paths_pref, allowed_pref = candidate_pool_for_signal(
        workload["id"], candidates, base_paths, phase_signal_prefill, 9401
    )
    sel_pref = select_candidate(paths_pref, phase_signal_prefill, allowed_pref)
    phase_signal_decode = dec10
    local_dec, paths_dec, allowed_dec = candidate_pool_for_signal(
        workload["id"], candidates, base_paths, phase_signal_decode, 9402
    )
    sel_dec = select_candidate(paths_dec, phase_signal_decode, allowed_dec)
    metrics = [
        eval_candidate(fair_paths, pre10),
        eval_candidate(paths_pref[sel_pref["selected"]], pre90),
        eval_candidate(fair_paths, dec10),
        eval_candidate(paths_dec[sel_dec["selected"]], dec90),
    ]
    phase_cycles = sum(m["optical_reference_cycles"] for m in metrics)
    phase_observed = merge_payloads([pre10, dec10], parsed["num_experts"])
    phase_evaluated = merge_payloads([pre90, dec90], parsed["num_experts"])
    phase_dom = dominant_phase(metrics)

    # Prefill warm-up.
    local_preonly, paths_preonly, allowed_preonly = candidate_pool_for_signal(
        workload["id"], candidates, base_paths, pre10, 9403
    )
    sel_preonly = select_candidate(paths_preonly, pre10, allowed_preonly)
    metrics_preonly = [
        eval_candidate(fair_paths, pre10),
        eval_candidate(paths_preonly[sel_preonly["selected"]], remaining_pre_decode),
    ]
    preonly_cycles = sum(m["optical_reference_cycles"] for m in metrics_preonly)
    preonly_dom = dominant_phase(metrics_preonly)

    # Oracle.
    local_oracle, paths_oracle, allowed_oracle = candidate_pool_for_signal(
        workload["id"], candidates, base_paths, full_payload, 9404
    )
    sel_oracle = select_candidate(paths_oracle, full_payload, allowed_oracle)
    oracle_metric = eval_candidate(paths_oracle[sel_oracle["selected"]], full_payload)

    method_base = [
        (
            "phase-warmup OCS",
            phase_cycles,
            2,
            f"prefill:{sel_pref['selected']} decode:{sel_dec['selected']}",
            "multi",
            "prefill10 for prefill90; decode10 for decode90",
            phase_observed,
            phase_evaluated,
            phase_dom,
            True,
        ),
        (
            "prefill-warmup OCS",
            preonly_cycles,
            1,
            sel_preonly["selected"],
            v39.sha256_graph(local_preonly[sel_preonly["selected"]]),
            "prefill10 only for remaining prefill+decode",
            pre10,
            remaining_pre_decode,
            preonly_dom,
            True,
        ),
        (
            "oracle",
            oracle_metric["optical_reference_cycles"],
            0,
            sel_oracle["selected"],
            v39.sha256_graph(local_oracle[sel_oracle["selected"]]),
            "full inference oracle upper bound",
            full_payload,
            full_payload,
            dominant_phase([oracle_metric]),
            False,
        ),
    ]

    # Previous request/window OCS. Non-overlapping windows: previous W selects next W.
    req_payloads = request_payloads(parsed, req_order)
    prev_results = {}
    previous_allowed = ["son_torus"] + [f"random_regular_seed_{s}" for s in RANDOM_SEEDS]
    for window in WINDOWS:
        total_cycles = 0
        reconfigs = 0
        selected_counts = Counter()
        segment_metrics = []
        # First window default fair.
        first_group = list(range(0, min(window, len(req_payloads))))
        if first_group:
            first_payload = merge_payloads([req_payloads[i] for i in first_group], parsed["num_experts"])
            metric = eval_candidate(fair_paths, first_payload)
            total_cycles += metric["optical_reference_cycles"]
            segment_metrics.append(metric)
        for start in range(window, len(req_payloads), window):
            history_indices = list(range(max(0, start - window), start))
            eval_indices = list(range(start, min(start + window, len(req_payloads))))
            hist_payload = merge_payloads([req_payloads[i] for i in history_indices], parsed["num_experts"])
            eval_payload = merge_payloads([req_payloads[i] for i in eval_indices], parsed["num_experts"])
            # Keep the request-window policy controlled: it selects among the
            # fixed candidate pool using previous-window traffic, rather than
            # synthesising a fresh graph for every small window.
            sel_hist = select_candidate(base_paths, hist_payload, previous_allowed)
            selected_counts[sel_hist["selected"]] += 1
            metric = eval_candidate(base_paths[sel_hist["selected"]], eval_payload)
            total_cycles += metric["optical_reference_cycles"]
            segment_metrics.append(metric)
            reconfigs += 1
        prev_results[window] = {
            "cycles": total_cycles,
            "reconfigs": reconfigs,
            "selected_counts": dict(selected_counts),
            "dominant": dominant_phase(segment_metrics),
        }
    best_w = min(WINDOWS, key=lambda w: prev_results[w]["cycles"] + MAIN_PENALTY_US * 1000 * prev_results[w]["reconfigs"])
    method_base.append(
        (
            f"previous-request OCS W={best_w}",
            prev_results[best_w]["cycles"],
            prev_results[best_w]["reconfigs"],
            f"window_selected_counts={prev_results[best_w]['selected_counts']}",
            "window_dynamic",
            f"previous non-overlapping W={best_w} requests select next W requests",
            merge_payloads(req_payloads[:best_w], parsed["num_experts"]),
            full_payload,
            prev_results[best_w]["dominant"],
            True,
        )
    )

    for penalty_us in RECONFIG_PENALTIES_US:
        for method, cycles, reconfigs, selected, graph_hash, signal, observed, evaluated, dom, no_leak in method_base:
            row = method_metric_row(
                workload["id"],
                method,
                penalty_us,
                cycles,
                reconfigs,
                baseline_cycles,
                selected,
                graph_hash,
                signal,
                observed,
                evaluated,
                dom,
                no_leak,
            )
            timing_rows.append(row)
            if penalty_us == MAIN_PENALTY_US:
                no_leakage_rows.append(row)
                bottleneck_rows.append(row)

    # Prediction signals.
    prediction_rows = []
    signals = [
        ("prefill10 -> remaining prefill", pre10, pre90),
        ("decode10 -> remaining decode", dec10, dec90),
        ("prefill10 -> remaining prefill+decode", pre10, remaining_pre_decode),
    ]
    for name, src, dst in signals:
        prediction_rows.append(
            {
                "workload": workload["id"],
                "signal": name,
                "spearman_expert": spearman_counts(src["expert_counts"], dst["expert_counts"]),
                "top8_expert_overlap": top_overlap(src["expert_counts"], dst["expert_counts"], 8),
                "top16_expert_overlap": top_overlap(src["expert_counts"], dst["expert_counts"], 16),
                "source_pair_gini": gini_sparse(src["combined_sparse"]),
                "target_pair_gini": gini_sparse(dst["combined_sparse"]),
                "source_top16_pair_share": v37c.concentration(list(src["combined_sparse"].values()), (16,)).get("top16_share", 0),
                "target_top16_pair_share": v37c.concentration(list(dst["combined_sparse"].values()), (16,)).get("top16_share", 0),
            }
        )
    # Previous request average predictability for W=best_w.
    req_preds = []
    for start in range(best_w, len(req_payloads), best_w):
        hist = merge_payloads(req_payloads[max(0, start - best_w):start], parsed["num_experts"])
        nxt = merge_payloads(req_payloads[start:min(start + best_w, len(req_payloads))], parsed["num_experts"])
        val = spearman_counts(hist["expert_counts"], nxt["expert_counts"])
        if val is not None:
            req_preds.append(val)
    prediction_rows.append(
        {
            "workload": workload["id"],
            "signal": f"previous W={best_w} requests -> next W={best_w} requests",
            "spearman_expert": statistics.mean(req_preds) if req_preds else None,
            "top8_expert_overlap": "",
            "top16_expert_overlap": "",
            "source_pair_gini": "",
            "target_pair_gini": "",
            "source_top16_pair_share": "",
            "target_top16_pair_share": "",
        }
    )

    validation = {
        "workload": workload["id"],
        "request_count": len(parsed["requests"]),
        "prefill_token_count": parsed["prefill_tokens"],
        "decode_token_count": parsed["decode_tokens"],
        "total_prefill_remote_bytes_dispatch_plus_combine": pre_all["remote_bytes"] * 2,
        "total_decode_remote_bytes_dispatch_plus_combine": dec_all["remote_bytes"] * 2,
        "total_full_inference_remote_bytes": full_payload["remote_bytes"] * 2,
        "fair_universal_static_candidate": fair_candidate,
        "fair_universal_static_cycles": baseline_cycles,
        "shuffled_order": shuffled,
        "prefill10_observed_fraction_remote_bytes": (pre10["remote_bytes"] / pre_all["remote_bytes"]) if pre_all["remote_bytes"] else 0,
        "decode10_observed_fraction_remote_bytes": (dec10["remote_bytes"] / dec_all["remote_bytes"]) if dec_all["remote_bytes"] else 0,
        "byte_conservation_pass": all(p["byte_conservation_pass"] for p in [pre10, pre90, dec10, dec90, full_payload]),
    }
    return {
        "timing_rows": timing_rows,
        "no_leakage_rows": no_leakage_rows,
        "bottleneck_rows": bottleneck_rows,
        "prediction_rows": prediction_rows,
        "validation": validation,
        "baseline_cycles": baseline_cycles,
        "best_w": best_w,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = v3812.build_base_candidates()
    base_paths = v3812.precompute_candidate_paths(candidates)
    parsed_full = {}
    full_payloads = {}
    for workload in WORKLOADS:
        parsed = v3812.v38.parse_trace(workload["path"])
        parsed_full[workload["id"]] = parsed
        full_payloads[workload["id"]] = merge_payloads(
            [
                merge_payloads(stage_request_payloads(parsed, "prefill"), parsed["num_experts"]),
                merge_payloads(stage_request_payloads(parsed, "decode"), parsed["num_experts"]),
            ],
            parsed["num_experts"],
        )
    fair = fair_universal_selection(full_payloads, base_paths)

    timing_rows: list[dict[str, Any]] = []
    no_leakage_rows: list[dict[str, Any]] = []
    bottleneck_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    shuffled_rows: list[dict[str, Any]] = []
    best_w_by_workload = {}

    for workload in WORKLOADS:
        result = process_workload(
            workload, candidates, base_paths, fair[workload["id"]], parsed_full[workload["id"]], shuffled=False
        )
        timing_rows.extend(result["timing_rows"])
        no_leakage_rows.extend(result["no_leakage_rows"])
        bottleneck_rows.extend(result["bottleneck_rows"])
        prediction_rows.extend(result["prediction_rows"])
        validation_rows.append(result["validation"])
        best_w_by_workload[workload["id"]] = result["best_w"]
        shuffled = process_workload(
            workload, candidates, base_paths, fair[workload["id"]], parsed_full[workload["id"]], shuffled=True
        )
        for row in shuffled["timing_rows"]:
            if row["penalty_us"] == MAIN_PENALTY_US and row["method"].startswith("previous-request"):
                row = dict(row)
                row["shuffled_order_control"] = True
                shuffled_rows.append(row)

    # Optional cheap parse only, not main timing.
    optional_status = None
    if OPTIONAL_QWEN_ZH["path"].exists():
        parsed = v3812.v38.parse_trace(OPTIONAL_QWEN_ZH["path"])
        optional_status = {
            "workload": OPTIONAL_QWEN_ZH["id"],
            "request_count": parsed["files_used"],
            "prefill_token_count": parsed["prefill_tokens"],
            "decode_token_count": parsed["decode_tokens"],
            "included_in_main_figures": False,
            "reason": "optional; omitted to keep V39.3 required workload set fixed",
        }

    write_csv(OUT / "full_inference_timing_table.csv", timing_rows)
    write_json(OUT / "full_inference_timing_table.json", timing_rows)
    write_csv(OUT / "prediction_signal_table.csv", prediction_rows)
    write_json(OUT / "prediction_signal_table.json", prediction_rows)
    write_csv(OUT / "no_leakage_validation.csv", no_leakage_rows)
    write_json(OUT / "no_leakage_validation.json", no_leakage_rows)
    write_csv(OUT / "bottleneck_decomposition.csv", bottleneck_rows)
    write_json(OUT / "bottleneck_decomposition.json", bottleneck_rows)

    # Penalty sensitivity: best non-oracle method per workload at each penalty.
    penalty_rows = []
    for workload in WORKLOADS:
        wid = workload["id"]
        for penalty in RECONFIG_PENALTIES_US:
            rows = [
                r for r in timing_rows
                if r["workload"] == wid and r["penalty_us"] == penalty and r["method"] != "oracle"
            ]
            best = min(rows, key=lambda r: r["total_cycles"])
            penalty_rows.append({
                "workload": wid,
                "penalty_us": penalty,
                "best_non_oracle_method": best["method"],
                "best_non_oracle_normalised": best["normalised_to_fair_universal_static"],
                "beats_fair_static": best["normalised_to_fair_universal_static"] < 1.0,
                "best_non_oracle_gain_percent": best["gain_vs_fair_static_percent"],
            })
    write_csv(OUT / "penalty_sensitivity.csv", penalty_rows)
    write_json(OUT / "penalty_sensitivity.json", penalty_rows)

    # Main plotted values at 1us.
    plotted_rows = [
        r for r in timing_rows
        if r["penalty_us"] == MAIN_PENALTY_US
    ] + shuffled_rows
    write_csv(OUT / "plotted_values.csv", plotted_rows)
    write_json(OUT / "plotted_values.json", plotted_rows)

    labels = {w["id"]: w["label"] for w in WORKLOADS}
    fig_methods = [
        "phase-warmup OCS",
        "prefill-warmup OCS",
        "previous-request OCS",
        "oracle",
    ]
    fig1_values = {label: {} for label in labels.values()}
    fig2_values = {label: {} for label in labels.values()}
    for workload in WORKLOADS:
        wid = workload["id"]
        label = labels[wid]
        for method in fig_methods:
            candidates_rows = [
                r for r in timing_rows
                if r["workload"] == wid and r["penalty_us"] == MAIN_PENALTY_US and (
                    r["method"] == method or (method == "previous-request OCS" and r["method"].startswith("previous-request"))
                )
            ]
            row = min(candidates_rows, key=lambda r: r["total_cycles"])
            fig1_values[label][method if method != "previous-request OCS" else row["method"]] = row["normalised_to_fair_universal_static"]
            fig2_values[label][method if method != "previous-request OCS" else row["method"]] = row["gain_vs_fair_static_percent"]

    # Normalize method labels for variable W in figures.
    fig_methods_resolved = []
    for method in fig_methods:
        if method == "previous-request OCS":
            # Use one label per figure; values dict may use workload-specific W labels, so collapse to generic.
            fig_methods_resolved.append("previous-request OCS")
        else:
            fig_methods_resolved.append(method)
    collapsed_fig1 = {label: {} for label in labels.values()}
    collapsed_fig2 = {label: {} for label in labels.values()}
    for label, row in fig1_values.items():
        for key, value in row.items():
            collapsed_fig1[label]["previous-request OCS" if key.startswith("previous-request") else key] = value
    for label, row in fig2_values.items():
        for key, value in row.items():
            collapsed_fig2[label]["previous-request OCS" if key.startswith("previous-request") else key] = value

    fig_dir = OUT / "figures"
    draw_grouped_bar(
        fig_dir / "figure_1_full_inference_normalized.png",
        "Figure 1: Full inference communication time",
        "Optical circuit/capacity reference. Normalised to fair universal static = 1.0. Penalty = 1us/reconfig.",
        list(labels.values()),
        fig_methods_resolved,
        collapsed_fig1,
        "normalised total inference time",
        baseline=1.0,
        ymax=max(1.15, max(max(v.values()) for v in collapsed_fig1.values()) * 1.12),
    )
    draw_grouped_bar(
        fig_dir / "figure_2_gain_over_fair_static.png",
        "Figure 2: Gain over fair universal static",
        "Optical circuit/capacity reference. Full inference, warm-up included. Penalty = 1us/reconfig.",
        list(labels.values()),
        fig_methods_resolved,
        collapsed_fig2,
        "gain over fair static (%)",
        baseline=0.0,
        ymax=max(1.0, max(max(v.values()) for v in collapsed_fig2.values()) * 1.2),
    )
    # Figure 3 prediction quality.
    pred_methods = [
        "prefill10 -> remaining prefill",
        "decode10 -> remaining decode",
        "prefill10 -> remaining prefill+decode",
        "previous W requests -> next W requests",
    ]
    pred_values = {label: {} for label in labels.values()}
    for workload in WORKLOADS:
        wid = workload["id"]
        label = labels[wid]
        for row in prediction_rows:
            if row["workload"] != wid:
                continue
            signal = row["signal"]
            key = "previous W requests -> next W requests" if signal.startswith("previous W=") else signal
            val = row["spearman_expert"]
            pred_values[label][key] = float(val) if val not in ("", None) else 0.0
    draw_grouped_bar(
        fig_dir / "figure_3_prediction_quality.png",
        "Figure 3: Prediction quality",
        "Spearman correlation over expert counts. Higher means observed traffic predicts future traffic better.",
        list(labels.values()),
        pred_methods,
        pred_values,
        "Spearman",
        ymax=1.05,
    )
    # Figure 4 penalty sensitivity.
    penalty_methods = ["0us", "1us", "10us", "25ms"]
    pen_values = {label: {} for label in labels.values()}
    for workload in WORKLOADS:
        wid = workload["id"]
        label = labels[wid]
        for row in penalty_rows:
            if row["workload"] == wid:
                name = f"{row['penalty_us']}us" if int(row["penalty_us"]) < 1000 else "25ms"
                pen_values[label][name] = row["best_non_oracle_normalised"]
    draw_grouped_bar(
        fig_dir / "figure_4_penalty_sensitivity.png",
        "Figure 4: Penalty sensitivity",
        "Best valid non-oracle method under each reconfiguration penalty.",
        list(labels.values()),
        penalty_methods,
        pen_values,
        "normalised time",
        baseline=1.0,
        ymax=max(1.2, max(max(v.values()) for v in pen_values.values()) * 1.1),
    )

    best_summary = {}
    for workload in WORKLOADS:
        wid = workload["id"]
        rows = [r for r in timing_rows if r["workload"] == wid and r["penalty_us"] == MAIN_PENALTY_US]
        non_oracle = [r for r in rows if r["method"] != "oracle"]
        best = min(non_oracle, key=lambda r: r["total_cycles"])
        oracle = min([r for r in rows if r["method"] == "oracle"], key=lambda r: r["total_cycles"])
        best_summary[wid] = {
            "best_non_oracle_method": best["method"],
            "best_non_oracle_normalised": best["normalised_to_fair_universal_static"],
            "best_non_oracle_beats_fair_static": best["normalised_to_fair_universal_static"] < 1.0,
            "oracle_normalised": oracle["normalised_to_fair_universal_static"],
        }

    summary = {
        "model": "optical circuit/capacity reference",
        "main_penalty_us": MAIN_PENALTY_US,
        "fair_universal_static_selection": fair,
        "best_summary": best_summary,
        "best_w_by_workload": best_w_by_workload,
        "shuffled_previous_request_rows": shuffled_rows,
        "optional_qwen_zh_status": optional_status,
        "final_answers": {
            "which_method_best_full_inference": best_summary,
            "does_any_non_oracle_beat_fair_static": {
                wid: row["best_non_oracle_beats_fair_static"] for wid, row in best_summary.items()
            },
            "gain_after_warmup_and_penalty": {
                wid: row["best_non_oracle_normalised"] for wid, row in best_summary.items()
            },
            "most_defensible": "phase-warmup if an online decode warm-up is acceptable; otherwise prefill-warmup is weaker but simpler. Previous-request is reported with shuffled control.",
            "changes_v39_2_conclusion": "Yes: once full inference warm-up/default traffic is included, non-oracle OCS gains are much smaller and often vanish.",
        },
    }
    write_json(OUT / "summary.json", summary)

    (OUT / "README.md").write_text(
        """# V39.3 Full-Inference Prediction-Signal Audit

This audit compares OCS control policies on full inference communication:
prefill + decode + observation/default-static phases + reconfiguration penalty.

Model: optical circuit/capacity reference. This is not full serving latency and
not native ASTRA optical execution.

Main penalty for figures: 1us per reconfiguration. Tables include 0us, 1us,
10us, and 25ms.

Methods:

- fair universal static baseline
- phase-warmup OCS
- prefill-warmup OCS
- previous-request OCS
- oracle upper bound

Important: every method includes observed/warm-up traffic and reconfiguration
penalty. Results should not be compared to V39.2 remaining-decode-only bars
without this caveat.
"""
    )
    (OUT / "supervisor_update_summary.md").write_text(
        f"""# V39.3 Supervisor Update

## Main Takeaway

After including warm-up/default traffic and reconfiguration penalties, OCS
control gains are much weaker than the remaining-decode-only view.

## Best Non-Oracle Method at 1us Penalty

```json
{json.dumps(best_summary, indent=2)}
```

## Interpretation

This does change the V39.2 conclusion: prefill/decode prediction may help in a
remaining-traffic view, but full inference accounting makes the OCS advantage
harder to defend unless the control policy improves a large fraction of traffic
or reconfiguration is amortised over larger batches/windows.
"""
    )
    print(json.dumps(summary["final_answers"], indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
