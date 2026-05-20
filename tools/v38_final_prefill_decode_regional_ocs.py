#!/usr/bin/env python3
"""V38 final: prefill-informed decode OCS for MoE inference.

This script uses only MoE inference expert-selection traces:
trace[0] is prefill, trace[1:] is decode.  Prefill may select topology,
placement, and server circuits; decode is the evaluation target.

The full search is a fluid link-load study.  Native ASTRA is not used by
default because 128-GPU decode traces are large and ASTRA does not support
safe in-run topology swaps.
"""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/moe_expert_trace_converter/v38_final_prefill_decode_regional_ocs"
V37C_PATH = REPO / "tools/v37c_128gpu_best_ocs_reconfig_audit.py"

spec = importlib.util.spec_from_file_location("v37c", V37C_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {V37C_PATH}")
v37c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v37c)

NPU = 128
SERVER_COUNT = 16
GPUS_PER_SERVER = 8
LINK_GBPS = 400
ASTRA_BYTES_PER_NS = v37c.ASTRA_BYTES_PER_NS
BYTES_PER_SELECTION = v37c.BYTES_PER_SELECTION
# Keep the V38 search controlled.  The seed set includes fixed seed0, V37c
# universal/best seeds, and a small deterministic exploration set.
RANDOM_SEEDS = sorted(set([0, 1, 2, 3, 4, 7, 8, 10, 11, 14, 24, 26, 27]))
ECMP_MAX_PATHS = 4
BATCH_SIZES = [1, 4, 8, 16, 32]
SOURCE_POLICIES = ["block_by_token", "decode_like_batch", "block_by_request"]
PLACEMENTS = ["block", "round_robin", "hot_expert_balanced"]
ALPHAS = [2, 4, 6]
PENALTIES_US = [0, 1, 10, 25_000]

WORKLOADS = [
    {
        "id": "qwen_livecodebench_execution",
        "label": "Qwen LiveCodeBench execution",
        "path": Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/livecodebench/execution"),
        "priority": True,
    },
    {
        "id": "qwen_mmlu_zh_cn_anatomy",
        "label": "Qwen MMLU_ZH_CN anatomy",
        "path": Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/mmlu_ZH_CN/anatomy"),
        "priority": True,
    },
    {
        "id": "qwen_mmlu_machine_learning",
        "label": "Qwen MMLU machine_learning",
        "path": Path("/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/mmlu/machine_learning"),
        "priority": False,
    },
    {
        "id": "deepseek_livecodebench_execution",
        "label": "DeepSeek LiveCodeBench execution",
        "path": Path("/Users/dfx/Python/trace/cognitivecomputations/DeepSeek-R1-AWQ/livecodebench/execution"),
        "priority": False,
    },
]

Sparse = dict[tuple[int, int], int]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def numeric_json_sort(path: Path) -> int | str:
    return int(path.stem) if path.stem.isdigit() else path.stem


def rankdata_desc(values: list[int]) -> list[float]:
    pairs = sorted(enumerate(values), key=lambda item: (-item[1], item[0]))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[pairs[k][0]] = avg_rank
        i = j
    return ranks


def pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    ma = statistics.mean(a)
    mb = statistics.mean(b)
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    denom = math.sqrt(sum(x * x for x in da) * sum(x * x for x in db))
    return sum(x * y for x, y in zip(da, db)) / denom if denom else None


def spearman_counts(prefill: list[int], decode: list[int]) -> float | None:
    return pearson(rankdata_desc(prefill), rankdata_desc(decode))


def concentration(values: list[int], topks: tuple[int, ...] = (1, 4, 8, 16)) -> dict[str, Any]:
    return v37c.concentration(values, topks)


def sparse_stats(sparse: Sparse, n: int = NPU) -> dict[str, Any]:
    return v37c.matrix_stats_from_sparse(sparse, n)


def add_sparse(dst: Sparse, src: Sparse) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + value


def combine_sparse(a: Sparse, b: Sparse) -> Sparse:
    out: Sparse = defaultdict(int)
    add_sparse(out, a)
    add_sparse(out, b)
    return dict(out)


def parse_trace(path: Path) -> dict[str, Any]:
    files = sorted(path.glob("*.json"), key=numeric_json_sort)
    requests = []
    prefill_counts: Counter[int] = Counter()
    decode_counts: Counter[int] = Counter()
    max_expert = -1
    malformed = 0
    global_prefill_offset = 0
    global_decode_offset = 0
    prefill_events = 0
    decode_events = 0
    prefill_tokens = 0
    decode_tokens = 0
    for req_idx, file_path in enumerate(files):
        try:
            trace = json.loads(file_path.read_text())
        except Exception:
            malformed += 1
            continue
        if not isinstance(trace, list) or not trace:
            malformed += 1
            continue
        prefill_rows = []
        decode_rows = []
        req_prefill_tokens = 0
        req_decode_tokens = 0
        for stage_idx, stage in enumerate(trace):
            if not isinstance(stage, dict):
                continue
            stage_max_rows = 0
            for layer_str, rows in stage.items():
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
                stage_max_rows = max(stage_max_rows, len(rows))
                for row_index, experts in enumerate(rows):
                    if not isinstance(experts, list):
                        malformed += 1
                        continue
                    parsed = []
                    for expert in experts:
                        try:
                            eid = int(expert)
                        except Exception:
                            malformed += 1
                            continue
                        parsed.append(eid)
                        max_expert = max(max_expert, eid)
                    if stage_idx == 0:
                        prefill_counts.update(parsed)
                        prefill_events += len(parsed)
                        prefill_rows.append((layer_id, row_index, global_prefill_offset + row_index, parsed))
                    else:
                        decode_counts.update(parsed)
                        decode_events += len(parsed)
                        # Keep global decode token index monotonic over all decode rows.
                        decode_rows.append((layer_id, row_index, global_decode_offset + row_index, parsed))
            if stage_idx == 0:
                req_prefill_tokens = max(req_prefill_tokens, stage_max_rows)
            else:
                req_decode_tokens += stage_max_rows
                global_decode_offset += stage_max_rows
        prefill_tokens += req_prefill_tokens
        decode_tokens += req_decode_tokens
        requests.append(
            {
                "request_id": file_path.stem,
                "request_index": req_idx,
                "prefill_rows": prefill_rows,
                "decode_rows": decode_rows,
                "prefill_tokens": req_prefill_tokens,
                "decode_tokens": req_decode_tokens,
            }
        )
        global_prefill_offset += req_prefill_tokens
    if max_expert < 0:
        raise RuntimeError(f"No experts in {path}")
    num_experts = max_expert + 1
    return {
        "files_found": len(files),
        "files_used": len(requests),
        "requests": requests,
        "num_experts": num_experts,
        "malformed_records": malformed,
        "prefill_counts": [prefill_counts[i] for i in range(num_experts)],
        "decode_counts": [decode_counts[i] for i in range(num_experts)],
        "prefill_events": prefill_events,
        "decode_events": decode_events,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
    }


def hot_order(counts: list[int]) -> list[int]:
    return [idx for idx, _ in sorted(enumerate(counts), key=lambda item: (-item[1], item[0]))]


def expert_to_gpu(expert_id: int, num_experts: int, placement: str, prefill_order: list[int]) -> int:
    if placement == "block":
        return min((expert_id * NPU) // num_experts, NPU - 1)
    if placement == "round_robin":
        return expert_id % NPU
    if placement == "hot_expert_balanced":
        rank = {expert: idx for idx, expert in enumerate(prefill_order)}
        return rank.get(expert_id, expert_id) % NPU
    raise ValueError(f"unknown placement {placement}")


def source_rank(policy: str, request_index: int, global_token_index: int, local_token_index: int) -> int:
    if policy == "block_by_token":
        return (global_token_index // v37c.BLOCK_SIZE) % NPU
    if policy == "decode_like_batch":
        return request_index % NPU
    if policy in ("block_by_request", "request_sticky"):
        return (request_index // v37c.BLOCK_SIZE) % NPU
    raise ValueError(f"unknown source policy {policy}")


def build_stage_sparse(
    parsed: dict[str, Any],
    stage: str,
    request_indices: list[int],
    source_policy: str,
    placement: str,
    prefill_order: list[int],
) -> dict[str, Any]:
    dispatch: Sparse = defaultdict(int)
    combine: Sparse = defaultdict(int)
    dest_gpu = [0 for _ in range(NPU)]
    selected = 0
    local = 0
    remote = 0
    token_count = 0
    for idx in request_indices:
        req = parsed["requests"][idx]
        rows = req["prefill_rows"] if stage == "prefill" else req["decode_rows"]
        token_count += req["prefill_tokens"] if stage == "prefill" else req["decode_tokens"]
        for _, local_token_idx, global_token_idx, experts in rows:
            src = source_rank(source_policy, int(req["request_index"]), global_token_idx, local_token_idx)
            for expert_id in experts:
                selected += 1
                dst = expert_to_gpu(expert_id, parsed["num_experts"], placement, prefill_order)
                if src == dst:
                    local += 1
                    continue
                dispatch[(src, dst)] += BYTES_PER_SELECTION
                combine[(dst, src)] += BYTES_PER_SELECTION
                dest_gpu[dst] += BYTES_PER_SELECTION
                remote += 1
    return {
        "dispatch_sparse": dict(dispatch),
        "combine_sparse": dict(combine),
        "combined_sparse": combine_sparse(dispatch, combine),
        "dest_gpu_bytes": dest_gpu,
        "selected_events": selected,
        "token_count": token_count,
        "local_bytes_excluded": local * BYTES_PER_SELECTION,
        "remote_bytes": remote * BYTES_PER_SELECTION,
        "byte_conservation_pass": selected * BYTES_PER_SELECTION == local * BYTES_PER_SELECTION + remote * BYTES_PER_SELECTION,
    }


def batch_indices(request_count: int, batch_size: int, shuffled: bool = False) -> list[list[int]]:
    indices = list(range(request_count))
    if shuffled:
        rng = __import__("random").Random(12345)
        rng.shuffle(indices)
    return [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]


def gpu_to_server(gpu: int) -> int:
    return gpu // GPUS_PER_SERVER


def server_sparse_from_gpu(sparse: Sparse) -> Sparse:
    out: Sparse = defaultdict(int)
    for (src, dst), value in sparse.items():
        ss, dd = gpu_to_server(src), gpu_to_server(dst)
        if ss != dd:
            out[(ss, dd)] += value
    return dict(out)


def undirected_pair_bytes(sparse: Sparse) -> Counter[tuple[int, int]]:
    out: Counter[tuple[int, int]] = Counter()
    for (src, dst), value in sparse.items():
        if src != dst:
            out[tuple(sorted((src, dst)))] += value
    return out


def graph_candidates() -> dict[str, dict[str, Any]]:
    candidates = {
        "son_torus": v37c.graph_from_edges(
            f"son_torus_{v37c.torus_shape(NPU)[0]}x{v37c.torus_shape(NPU)[1]}_{NPU}gpu",
            v37c.torus_edges(NPU),
            NPU,
            {"construction": "static_2d_torus", "shape": v37c.torus_shape(NPU)},
        )
    }
    for seed in RANDOM_SEEDS:
        candidates[f"random_regular_seed_{seed}"] = v37c.graph_from_edges(
            f"ron_random_regular_seed_{seed}",
            v37c.random_regular_graph(seed, NPU),
            NPU,
            {"construction": "random_regular_degree4", "seed": seed},
        )
    return candidates


def score_graph(paths: dict[tuple[int, int], list[list[int]]], payload: dict[str, Any]) -> int:
    return max(
        v37c.link_load_sparse(paths, payload["dispatch_sparse"])["fluid_cycles"],
        v37c.link_load_sparse(paths, payload["combine_sparse"])["fluid_cycles"],
    )


def score_graph_bytes(paths: dict[tuple[int, int], list[list[int]]], payload: dict[str, Any]) -> int:
    return max(
        v37c.link_load_sparse(paths, payload["dispatch_sparse"])["max_link_load_bytes"],
        v37c.link_load_sparse(paths, payload["combine_sparse"])["max_link_load_bytes"],
    )


def select_by_signal(
    candidates: dict[str, dict[str, Any]],
    path_caches: dict[str, dict[tuple[int, int], list[list[int]]]],
    signal_payload: dict[str, Any],
    allowed: list[str],
) -> dict[str, Any]:
    rows = []
    for name in allowed:
        rows.append({"name": name, "score_bytes": score_graph_bytes(path_caches[name], signal_payload)})
    rows = sorted(rows, key=lambda row: (row["score_bytes"], row["name"]))
    return {"selected": rows[0]["name"], "scores_top8": rows[:8]}


def evaluate_static_methods(
    candidates: dict[str, dict[str, Any]],
    path_caches: dict[str, dict[tuple[int, int], list[list[int]]]],
    prefill_payload: dict[str, Any],
    decode_payload: dict[str, Any],
    fair_universal: str,
) -> dict[str, Any]:
    allowed_cal = ["son_torus"] + [f"random_regular_seed_{i}" for i in RANDOM_SEEDS]
    # Rebuild calibrated greedy from actual prefill signal by replacing the generic graph.
    cal_edges = v37c.safe_greedy_graph(prefill_payload["combined_sparse"], v37c.ring_edges(NPU), NPU, 9101)[0]
    candidates["prefill_greedy"] = v37c.graph_from_edges("prefill_greedy_degree4", cal_edges, NPU, {"construction": "prefill_greedy"})
    path_caches["prefill_greedy"] = v37c.precompute_paths(candidates["prefill_greedy"], NPU)
    allowed_cal = ["son_torus", "prefill_greedy"] + [f"random_regular_seed_{i}" for i in RANDOM_SEEDS]
    selection = select_by_signal(candidates, path_caches, prefill_payload, allowed_cal)
    oracle = select_by_signal(candidates, path_caches, decode_payload, ["son_torus", "prefill_greedy"] + [f"random_regular_seed_{i}" for i in RANDOM_SEEDS])
    method_names = {
        "son_torus": "son_torus",
        "fixed_random_seed0": "random_regular_seed_0",
        "fair_universal_static": fair_universal,
        "prefill_informed_workload_ocs": selection["selected"],
        "decode_oracle_ocs": oracle["selected"],
    }
    rows = {}
    for method, name in method_names.items():
        rows[method] = {
            "candidate": name,
            "decode_fluid_cycles": score_graph(path_caches[name], decode_payload),
            "decode_score_bytes": score_graph_bytes(path_caches[name], decode_payload),
        }
    return {"methods": rows, "prefill_selection": selection, "oracle_selection": oracle}


def server_circuit_alloc(signal: Sparse, alpha: int, oracle_name: str) -> dict[str, Any]:
    demand = undirected_pair_bytes(signal)
    degree = [0 for _ in range(SERVER_COUNT)]
    circuits: Counter[tuple[int, int]] = Counter()
    for pair, _ in demand.most_common():
        a, b = pair
        while degree[a] < alpha and degree[b] < alpha and circuits[pair] < alpha:
            circuits[pair] += 1
            degree[a] += 1
            degree[b] += 1
    # Fill unused ports deterministically.
    rng_pairs = [(i, j) for i in range(SERVER_COUNT) for j in range(i + 1, SERVER_COUNT)]
    idx = 0
    while any(d < alpha for d in degree) and idx < len(rng_pairs) * 20:
        a, b = rng_pairs[idx % len(rng_pairs)]
        idx += 1
        if degree[a] < alpha and degree[b] < alpha:
            circuits[(a, b)] += 1
            degree[a] += 1
            degree[b] += 1
    return {"name": oracle_name, "alpha": alpha, "circuits": {f"{a}-{b}": c for (a, b), c in sorted(circuits.items())}, "degree": degree}


def random_regular_server_edges(seed: int, degree: int) -> set[tuple[int, int]]:
    if degree % 2 != 0 or degree >= SERVER_COUNT:
        raise ValueError(f"server degree must be even and < {SERVER_COUNT}: {degree}")
    rng = __import__("random").Random(seed)
    perm = list(range(SERVER_COUNT))
    rng.shuffle(perm)
    edges = set()
    for idx in range(SERVER_COUNT):
        for offset in range(1, degree // 2 + 1):
            a = perm[idx]
            b = perm[(idx + offset) % SERVER_COUNT]
            edges.add(tuple(sorted((a, b))))
    if len(v37c.connected_components(edges, SERVER_COUNT)) != 1:
        raise RuntimeError(f"server circulant graph unexpectedly disconnected degree={degree} seed={seed}")
    return edges


def server_torus_like_edges(alpha: int) -> set[tuple[int, int]]:
    if alpha == 2:
        return v37c.ring_edges(SERVER_COUNT)
    if alpha == 4:
        return v37c.torus_edges(SERVER_COUNT)
    return random_regular_server_edges(600, alpha)


def static_server_circuits(alpha: int, kind: str, seed: int = 0) -> dict[str, Any]:
    if kind == "torus":
        edges = server_torus_like_edges(alpha)
    else:
        edges = random_regular_server_edges(seed, alpha)
    circuits = Counter(edges)
    degree = [0 for _ in range(SERVER_COUNT)]
    for (a, b), c in circuits.items():
        degree[a] += c
        degree[b] += c
    return {"name": f"{kind}_alpha{alpha}_seed{seed}", "alpha": alpha, "circuits": {f"{a}-{b}": c for (a, b), c in sorted(circuits.items())}, "degree": degree}


def eval_server_hybrid(circuit_obj: dict[str, Any], decode_server: Sparse) -> dict[str, Any]:
    circuits = {}
    for key, c in circuit_obj["circuits"].items():
        a, b = map(int, key.split("-"))
        circuits[(a, b)] = c
    ocs_loads = []
    eps_load_by_server = [0 for _ in range(SERVER_COUNT)]
    ocs_bytes = 0
    eps_bytes = 0
    undirected = undirected_pair_bytes(decode_server)
    for pair, bytes_ in undirected.items():
        c = circuits.get(pair, 0)
        if c > 0:
            ocs_loads.append(math.ceil(bytes_ / c))
            ocs_bytes += bytes_
        else:
            a, b = pair
            eps_load_by_server[a] += bytes_
            eps_load_by_server[b] += bytes_
            eps_bytes += bytes_
    max_ocs = max(ocs_loads) if ocs_loads else 0
    max_eps = max(eps_load_by_server) if eps_load_by_server else 0
    total = ocs_bytes + eps_bytes
    # EPS fallback is intentionally explicit and uses the same 400 Gb/s unit.
    cycles = max(int(max_ocs / ASTRA_BYTES_PER_NS), int(max_eps / ASTRA_BYTES_PER_NS))
    return {
        "estimated_cycles": cycles,
        "ocs_served_fraction": ocs_bytes / total if total else 0,
        "eps_residual_fraction": eps_bytes / total if total else 0,
        "max_ocs_load_bytes": max_ocs,
        "max_eps_residual_load_bytes": max_eps,
        "degree_distribution": dict(Counter(circuit_obj["degree"])),
        "same_alpha_budget": min(circuit_obj["degree"]) == circuit_obj["alpha"] and max(circuit_obj["degree"]) == circuit_obj["alpha"],
    }


def predictability(parsed: dict[str, Any]) -> dict[str, Any]:
    pre = parsed["prefill_counts"]
    dec = parsed["decode_counts"]
    pre_rank = [idx for idx, _ in sorted(enumerate(pre), key=lambda item: (-item[1], item[0]))]
    dec_rank = [idx for idx, _ in sorted(enumerate(dec), key=lambda item: (-item[1], item[0]))]
    total_pre = sum(pre)
    total_dec = sum(dec)
    out = {
        "request_count": parsed["files_used"],
        "prefill_token_count": parsed["prefill_tokens"],
        "decode_token_count": parsed["decode_tokens"],
        "prefill_events": parsed["prefill_events"],
        "decode_events": parsed["decode_events"],
        "num_experts": parsed["num_experts"],
        "spearman_expert_count": spearman_counts(pre, dec),
        "prefill": concentration(pre, (1, 8, 16)),
        "decode": concentration(dec, (1, 8, 16)),
    }
    for k in (1, 8, 16):
        out[f"top{k}_expert_overlap"] = len(set(pre_rank[:k]) & set(dec_rank[:k])) / k
    out["top_prefill_experts"] = [{"expert": i, "share": pre[i] / total_pre if total_pre else 0} for i in pre_rank[:16]]
    out["top_decode_experts"] = [{"expert": i, "share": dec[i] / total_dec if total_dec else 0} for i in dec_rank[:16]]
    return out


def universal_static_by_leave_one_out(workload_payloads: dict[str, dict[str, Any]], path_caches: dict[str, dict[tuple[int, int], list[list[int]]]]) -> dict[str, str]:
    random_names = [f"random_regular_seed_{i}" for i in RANDOM_SEEDS]
    result = {}
    for target in workload_payloads:
        rows = []
        for name in random_names:
            norm = []
            for wid, payload in workload_payloads.items():
                if wid == target:
                    continue
                son = score_graph(path_caches["son_torus"], payload["decode"])
                val = score_graph(path_caches[name], payload["decode"])
                norm.append(val / son if son else 0)
            rows.append({"name": name, "avg_norm": statistics.mean(norm)})
        result[target] = min(rows, key=lambda row: (row["avg_norm"], row["name"]))["name"]
    return result


def request_stage_payloads(
    parsed: dict[str, Any],
    stage: str,
    source_policy: str,
    placement: str,
    prefill_order: list[int],
) -> list[dict[str, Any]]:
    return [
        build_stage_sparse(parsed, stage, [idx], source_policy, placement, prefill_order)
        for idx in range(len(parsed["requests"]))
    ]


def aggregate_payloads(payloads: list[dict[str, Any]], group: list[int]) -> dict[str, Any]:
    combined: Sparse = defaultdict(int)
    dispatch: Sparse = defaultdict(int)
    combine: Sparse = defaultdict(int)
    dest_gpu = [0 for _ in range(NPU)]
    for idx in group:
        payload = payloads[idx]
        add_sparse(combined, payload["combined_sparse"])
        add_sparse(dispatch, payload["dispatch_sparse"])
        add_sparse(combine, payload["combine_sparse"])
        for i, value in enumerate(payload["dest_gpu_bytes"]):
            dest_gpu[i] += value
    return {
        "combined_sparse": dict(combined),
        "dispatch_sparse": dict(dispatch),
        "combine_sparse": dict(combine),
        "dest_gpu_bytes": dest_gpu,
    }


def process_workload(
    workload: dict[str, Any],
    parsed: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    path_caches: dict[str, dict[tuple[int, int], list[list[int]]]],
    fair_universal: str,
) -> dict[str, Any]:
    order = hot_order(parsed["prefill_counts"])
    req_count = len(parsed["requests"])
    all_indices = list(range(req_count))

    primary_prefill = build_stage_sparse(parsed, "prefill", all_indices, "block_by_token", "block", order)
    primary_decode = build_stage_sparse(parsed, "decode", all_indices, "block_by_token", "block", order)
    static_eval = evaluate_static_methods(candidates, path_caches, primary_prefill, primary_decode, fair_universal)

    batch_rows = []
    best_batch = None
    decode_request_payloads: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for source_policy in SOURCE_POLICIES:
        for placement in ("block", "round_robin"):
            decode_request_payloads[(source_policy, placement)] = request_stage_payloads(
                parsed, "decode", source_policy, placement, order
            )
    for batch_size in BATCH_SIZES:
        for source_policy in SOURCE_POLICIES:
            for placement in ("block", "round_robin"):
                decode_agg: Sparse = defaultdict(int)
                dest_gpu = [0 for _ in range(NPU)]
                payloads = decode_request_payloads[(source_policy, placement)]
                for group in batch_indices(req_count, batch_size):
                    dec = aggregate_payloads(payloads, group)
                    add_sparse(decode_agg, dec["combined_sparse"])
                    for i, v in enumerate(dec["dest_gpu_bytes"]):
                        dest_gpu[i] += v
                pair_stats = sparse_stats(dict(decode_agg), NPU)
                server_stats = sparse_stats(server_sparse_from_gpu(dict(decode_agg)), SERVER_COUNT)
                dest_stats = concentration(dest_gpu, (1, 4, 8, 16))
                row = {
                    "batch_size": batch_size,
                    "source_policy": source_policy,
                    "expert_placement": placement,
                    "synthetic_source_policy": source_policy != "block_by_token",
                    "destination_gpu_top16_share": dest_stats["top16_share"],
                    "destination_gpu_gini": dest_stats["gini"],
                    "gpu_pair_top1_share": pair_stats["top1_share"],
                    "gpu_pair_top4_share": pair_stats["top4_share"],
                    "gpu_pair_top16_share": pair_stats["top16_share"],
                    "gpu_pair_gini": pair_stats["gini"],
                    "server_pair_top1_share": server_stats["top1_share"],
                    "server_pair_top4_share": server_stats["top4_share"],
                    "server_pair_top16_share": server_stats["top16_share"],
                    "server_pair_gini": server_stats["gini"],
                }
                batch_rows.append(row)
                # For OCS opportunity, prefer strong server-pair concentration.
                opportunity = server_stats["top16_share"] + server_stats["gini"]
                if best_batch is None or opportunity > best_batch["opportunity"]:
                    best_batch = {**row, "opportunity": opportunity}
    shuffled_control = None
    if best_batch:
        # Same B/policy/placement, shuffled consecutive grouping.
        decode_agg = defaultdict(int)
        payloads = decode_request_payloads[(str(best_batch["source_policy"]), str(best_batch["expert_placement"]))]
        for group in batch_indices(req_count, int(best_batch["batch_size"]), shuffled=True):
            dec = aggregate_payloads(payloads, group)
            add_sparse(decode_agg, dec["combined_sparse"])
        shuffled_control = {
            "batch_size": best_batch["batch_size"],
            "source_policy": best_batch["source_policy"],
            "expert_placement": best_batch["expert_placement"],
            "server_pair": sparse_stats(server_sparse_from_gpu(dict(decode_agg)), SERVER_COUNT),
            "gpu_pair": sparse_stats(dict(decode_agg), NPU),
        }

    placement_rows = []
    for placement in PLACEMENTS:
        pre = build_stage_sparse(parsed, "prefill", all_indices, "block_by_token", placement, order)
        dec = build_stage_sparse(parsed, "decode", all_indices, "block_by_token", placement, order)
        eval_row = evaluate_static_methods(candidates, path_caches, pre, dec, fair_universal)
        best_static = eval_row["methods"]["fair_universal_static"]["decode_fluid_cycles"]
        ocs = eval_row["methods"]["prefill_informed_workload_ocs"]["decode_fluid_cycles"]
        placement_rows.append(
            {
                "placement": placement,
                "selected_candidate": eval_row["methods"]["prefill_informed_workload_ocs"]["candidate"],
                "destination_gpu": concentration(dec["dest_gpu_bytes"], (1, 4, 8, 16)),
                "gpu_pair": sparse_stats(dec["combined_sparse"], NPU),
                "server_pair": sparse_stats(server_sparse_from_gpu(dec["combined_sparse"]), SERVER_COUNT),
                "ocs_gain_vs_fair_static_percent": 100 * (best_static - ocs) / best_static if best_static else None,
                "fair_static_cycles": best_static,
                "ocs_cycles": ocs,
            }
        )

    server_rows = []
    decode_server = server_sparse_from_gpu(primary_decode["combined_sparse"])
    prefill_server = server_sparse_from_gpu(primary_prefill["combined_sparse"])
    for alpha in ALPHAS:
        circuits = {
            "son_server_torus": static_server_circuits(alpha, "torus"),
            "fixed_random_server": static_server_circuits(alpha, "random", 0),
            "prefill_mixnet_greedy": server_circuit_alloc(prefill_server, alpha, "prefill_mixnet_greedy"),
            "decode_oracle_server": server_circuit_alloc(decode_server, alpha, "decode_oracle_server"),
        }
        for name, circ in circuits.items():
            row = eval_server_hybrid(circ, decode_server)
            server_rows.append({"alpha": alpha, "method": name, **row})

    penalty_rows = []
    fair = static_eval["methods"]["fair_universal_static"]["decode_fluid_cycles"]
    ocs_base = static_eval["methods"]["prefill_informed_workload_ocs"]["decode_fluid_cycles"]
    server_best = min((row for row in server_rows if "oracle" not in row["method"]), key=lambda row: row["estimated_cycles"])
    for penalty in PENALTIES_US:
        penalty_cycles = penalty * 1000
        penalty_rows.append(
            {
                "penalty_us": penalty,
                "gpu_level_prefill_ocs_cycles": ocs_base + penalty_cycles,
                "gpu_level_beats_fair_static": ocs_base + penalty_cycles < fair,
                "server_level_best_method": server_best["method"],
                "server_level_cycles": server_best["estimated_cycles"] + penalty_cycles,
                "server_level_beats_fair_static": server_best["estimated_cycles"] + penalty_cycles < fair,
            }
        )

    best_static_cycles = min(
        static_eval["methods"]["son_torus"]["decode_fluid_cycles"],
        static_eval["methods"]["fixed_random_seed0"]["decode_fluid_cycles"],
        static_eval["methods"]["fair_universal_static"]["decode_fluid_cycles"],
    )
    best_ocs_cycles = min(
        static_eval["methods"]["prefill_informed_workload_ocs"]["decode_fluid_cycles"],
        min(row["estimated_cycles"] for row in server_rows if "oracle" not in row["method"]),
    )
    best_server = min((row for row in server_rows if "oracle" not in row["method"]), key=lambda row: row["estimated_cycles"])
    oracle_cycles = min(
        static_eval["methods"]["decode_oracle_ocs"]["decode_fluid_cycles"],
        min(row["estimated_cycles"] for row in server_rows if "oracle" in row["method"]),
    )

    result = {
        "workload": {"id": workload["id"], "label": workload["label"], "path": str(workload["path"])},
        "predictability": predictability(parsed),
        "primary_static_vs_ocs": static_eval,
        "batch_source_locality": batch_rows,
        "best_batch_source_locality": best_batch,
        "shuffled_order_control_for_best_batch": shuffled_control,
        "expert_placement": placement_rows,
        "server_level_regional_ocs": server_rows,
        "reconfiguration_penalty": penalty_rows,
        "final_ranking": {
            "best_static_cycles": best_static_cycles,
            "best_gpu_level_prefill_ocs_cycles": static_eval["methods"]["prefill_informed_workload_ocs"]["decode_fluid_cycles"],
            "best_server_level_non_oracle": best_server,
            "best_ocs_cycles": best_ocs_cycles,
            "best_decode_oracle_cycles": oracle_cycles,
            "ocs_gain_vs_best_static_percent": 100 * (best_static_cycles - best_ocs_cycles) / best_static_cycles if best_static_cycles else None,
            "oracle_gap_percent": 100 * (best_ocs_cycles - oracle_cycles) / best_ocs_cycles if best_ocs_cycles else None,
            "best_valid_non_oracle_method": "server_level_regional_hybrid" if best_server["estimated_cycles"] <= static_eval["methods"]["prefill_informed_workload_ocs"]["decode_fluid_cycles"] else "gpu_level_prefill_informed_ocs",
        },
        "validation": {
            "prefill_only_selection": True,
            "decode_only_evaluation": True,
            "no_native_in_run_topology_swap_claim": True,
            "same_gpu_level_budget": "degree=4, 400Gb/s/link, ECMP4",
            "same_server_level_budget": "alpha circuits/server, 400Gb/s/circuit; EPS residual explicit in hybrid model",
        },
    }
    write_json(OUT / "workloads" / workload["id"] / "workload_summary.json", result)
    return result


def write_report(summary: dict[str, Any]) -> None:
    compact = {
        "final_ranking_table": summary["final_ranking_table"],
        "prefill_decode_predictability": summary["prefill_decode_predictability_table"],
        "final_diagnosis": summary["final_diagnosis"],
        "limitations": summary["limitations"],
    }
    (OUT / "README.md").write_text(
        f"""# V38-FINAL Prefill-Informed Decode OCS for MoE Inference

## Scope

This is inference-only. It uses `trace[0]` as prefill and `trace[1:]` as decode. Prefill may select topology, placement, or server circuits; decode is the evaluation target. No ASTRA C++ core changes, no figures, no all-path ECMP, and no native in-run topology swap claims.

## Compact Result

```json
{json.dumps(compact, indent=2)}
```

## Claims Allowed

- Prefill-vs-decode expert predictability is measured on HF inference traces.
- Prefill-informed GPU-level and server-level OCS strategies are evaluated without decode leakage in a fluid model.
- Server-level regional hybrid OCS includes explicit residual EPS traffic.

## Claims Not Allowed

- MoE training behavior.
- Real serving latency.
- Native ASTRA in-run topology swaps.
- Physical OCS device-level timing.
- Paper-ready figures.

Detailed data are in `summary.json` and per-workload `workload_summary.json`.
"""
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = graph_candidates()
    path_caches = {name: v37c.precompute_paths(graph, NPU) for name, graph in candidates.items()}

    full_workloads = [workload for workload in WORKLOADS if workload["priority"]]

    # Pre-parse primary workloads once to choose leave-one-workload-out universal random static.
    lightweight_payloads: dict[str, dict[str, Any]] = {}
    parsed_cache: dict[str, dict[str, Any]] = {}
    for workload in full_workloads:
        parsed = parse_trace(workload["path"])
        parsed_cache[workload["id"]] = parsed
        order = hot_order(parsed["prefill_counts"])
        decode = build_stage_sparse(parsed, "decode", list(range(len(parsed["requests"]))), "block_by_token", "block", order)
        lightweight_payloads[workload["id"]] = {"decode": decode}
    fair_universal_by_workload = universal_static_by_leave_one_out(lightweight_payloads, path_caches)

    results = []
    for workload in full_workloads:
        result = process_workload(workload, parsed_cache[workload["id"]], candidates, path_caches, fair_universal_by_workload[workload["id"]])
        results.append(result)

    optional_compact = []
    for workload in [workload for workload in WORKLOADS if not workload["priority"]]:
        parsed = parse_trace(workload["path"])
        optional_compact.append({"workload": workload["id"], "predictability": predictability(parsed)})

    predict_table = []
    final_table = []
    batch_table = []
    placement_table = []
    server_table = []
    penalty_table = []
    for result in results:
        wid = result["workload"]["id"]
        pred = result["predictability"]
        predict_table.append(
            {
                "workload": wid,
                "request_count": pred["request_count"],
                "prefill_tokens": pred["prefill_token_count"],
                "decode_tokens": pred["decode_token_count"],
                "prefill_top1": pred["prefill"]["top1_share"],
                "decode_top1": pred["decode"]["top1_share"],
                "top8_overlap": pred["top8_expert_overlap"],
                "top16_overlap": pred["top16_expert_overlap"],
                "spearman": pred["spearman_expert_count"],
            }
        )
        rank = result["final_ranking"]
        static_methods = result["primary_static_vs_ocs"]["methods"]
        final_table.append(
            {
                "workload": wid,
                "stage_evaluated": "decode",
                "selection_signal": "prefill only",
                "batch_size": "all_requests",
                "source_policy": "block_by_token",
                "expert_placement": "block",
                "architecture": rank["best_valid_non_oracle_method"],
                "best_static_method": min(
                    ["son_torus", "fixed_random_seed0", "fair_universal_static"],
                    key=lambda key: static_methods[key]["decode_fluid_cycles"],
                ),
                "best_ocs_method": rank["best_valid_non_oracle_method"],
                "best_static_time": rank["best_static_cycles"],
                "best_ocs_time": rank["best_ocs_cycles"],
                "ocs_gain_vs_fair_static": rank["ocs_gain_vs_best_static_percent"],
                "ocs_gain_vs_son": 100 * (static_methods["son_torus"]["decode_fluid_cycles"] - rank["best_ocs_cycles"]) / static_methods["son_torus"]["decode_fluid_cycles"],
                "reconfiguration_penalty": "0us in ranking; see penalty table",
                "oracle_gap": rank["oracle_gap_percent"],
                "astra_validated": "no",
                "interpretation": "fluid-only inference decode audit",
            }
        )
        batch_table.extend({"workload": wid, **row} for row in result["batch_source_locality"])
        placement_table.extend({"workload": wid, **row} for row in result["expert_placement"])
        server_table.extend({"workload": wid, **row} for row in result["server_level_regional_ocs"])
        penalty_table.extend({"workload": wid, **row} for row in result["reconfiguration_penalty"])

    final_diagnosis = {
        "q1_prefill_predicts_decode": "yes/moderately" if all((row["spearman"] or 0) > 0.5 for row in predict_table) else "mixed",
        "q2_prefill_informed_ocs_beats_fair_static": {
            row["workload"]: row["ocs_gain_vs_fair_static"] > 0 for row in final_table
        },
        "q3_batching_source_locality": "see batch_source_locality_table; synthetic source policies are labelled",
        "q4_placement_vs_topology": "see expert_placement_table; placement changes hotness but static baselines receive the same placement",
        "q5_server_regional_vs_gpu_ocs": "see GPU-level vs server-level table; server regional hybrid is often the strongest OCS candidate if residual EPS is allowed",
        "q6_mixnet_greedy_vs_static_expander": "reported per alpha; negative cases are kept",
        "q7_penalty_survival": "see penalty table for 1us/10us/25ms",
        "q8_closest_case_if_not_winning": "see final ranking table",
        "q9_story": "The strongest defensible story is selected from A-E based on final_ranking_table; do not claim training.",
    }
    summary = {
        "scope": "V38 final prefill-informed decode OCS for MoE inference",
        "workloads": results,
        "optional_compact_predictability": optional_compact,
        "fair_universal_by_workload": fair_universal_by_workload,
        "prefill_decode_predictability_table": predict_table,
        "batch_source_locality_table": batch_table,
        "expert_placement_table": placement_table,
        "gpu_level_vs_server_level_ocs_table": server_table,
        "fair_static_vs_ocs_comparison": final_table,
        "reconfiguration_penalty_table": penalty_table,
        "final_ranking_table": final_table,
        "final_diagnosis": final_diagnosis,
        "limitations": {
            "timing_engine": "fluid link-load scoring only; native ASTRA skipped for V38 because decode traces are large and in-run topology swap is unsupported",
            "no_training_claim": True,
            "no_real_serving_latency_claim": True,
            "no_figures": True,
            "ecmp_max_paths": ECMP_MAX_PATHS,
        },
    }
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "prefill_decode_predictability_table.json", predict_table)
    write_json(OUT / "batch_source_locality_table.json", batch_table)
    write_json(OUT / "expert_placement_table.json", placement_table)
    write_json(OUT / "gpu_level_vs_server_level_ocs_table.json", server_table)
    write_json(OUT / "fair_static_vs_ocs_comparison.json", final_table)
    write_json(OUT / "reconfiguration_penalty_table.json", penalty_table)
    write_report(summary)
    print(json.dumps({"final_ranking_table": final_table, "final_diagnosis": final_diagnosis}, indent=2))


if __name__ == "__main__":
    main()
