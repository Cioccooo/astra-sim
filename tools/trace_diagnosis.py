#!/usr/bin/env python3
"""Diagnose communication overlap in Chakra execution traces.

The checker is intentionally trace-level: it reads rank-local Chakra ET files,
groups collective and send/recv nodes into global communication events, schedules
the dependency graph with a simple bandwidth model, and reports which
communication classes overlap in time.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import itertools
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
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

try:
    from et_def.et_def_pb2 import (  # type: ignore
        CollectiveCommType,
        GlobalMetadata,
        Node,
        NodeType,
    )
    from protolib import decodeMessage  # type: ignore
except Exception as exc:  # pragma: no cover - environment diagnostic.
    raise SystemExit(
        "Could not import Chakra v0.0.4 protobuf support. "
        f"Tried {STG_CHAKRA}. Original error: {exc}"
    )


COMM_TYPE_NAMES = {
    int(CollectiveCommType.ALL_REDUCE): "ALL_REDUCE",
    int(CollectiveCommType.REDUCE): "REDUCE",
    int(CollectiveCommType.ALL_GATHER): "ALL_GATHER",
    int(CollectiveCommType.GATHER): "GATHER",
    int(CollectiveCommType.SCATTER): "SCATTER",
    int(CollectiveCommType.BROADCAST): "BROADCAST",
    int(CollectiveCommType.ALL_TO_ALL): "ALL_TO_ALL",
    int(CollectiveCommType.REDUCE_SCATTER): "REDUCE_SCATTER",
    int(CollectiveCommType.REDUCE_SCATTER_BLOCK): "REDUCE_SCATTER_BLOCK",
    int(CollectiveCommType.BARRIER): "BARRIER",
}


@dataclass(frozen=True)
class NodeRef:
    rank: int
    node_id: int


@dataclass
class TraceNode:
    rank: int
    node: Any
    attrs: dict[str, Any]
    deps: list[int]


@dataclass
class Group:
    gid: int
    key: tuple[Any, ...]
    kind: str
    label: str
    comm_class: str | None
    members: list[TraceNode] = field(default_factory=list)
    ranks: set[int] = field(default_factory=set)
    deps: set[int] = field(default_factory=set)
    children: set[int] = field(default_factory=set)
    duration_ns: int = 1
    start_ns: int = 0
    end_ns: int = 0


def attr_value(attr: Any) -> Any:
    which = attr.WhichOneof("value")
    if which is None:
        return None
    value = getattr(attr, which)
    if which.endswith("_list"):
        return list(value.values)
    return value


def attrs_for(node: Any) -> dict[str, Any]:
    return {attr.name: attr_value(attr) for attr in node.attr}


def rank_from_path(path: Path) -> int:
    match = re.search(r"\.(\d+)\.et$", path.name)
    if not match:
        raise ValueError(f"Cannot infer rank from {path}")
    return int(match.group(1))


def read_trace_file(path: Path) -> list[TraceNode]:
    rank = rank_from_path(path)
    out: list[TraceNode] = []
    with path.open("rb") as handle:
        metadata = GlobalMetadata()
        if not decodeMessage(handle, metadata):
            raise ValueError(f"Failed to read GlobalMetadata from {path}")
        while True:
            node = Node()
            if not decodeMessage(handle, node):
                break
            deps = list(node.data_deps) + list(node.ctrl_deps)
            out.append(TraceNode(rank=rank, node=node, attrs=attrs_for(node), deps=deps))
    return out


def load_trace_dir(trace_dir: Path) -> dict[int, list[TraceNode]]:
    files = sorted(trace_dir.glob("*.et"), key=rank_from_path)
    if not files:
        raise ValueError(f"No .et files found in {trace_dir}")
    return {rank_from_path(path): read_trace_file(path) for path in files}


def int_attr(attrs: dict[str, Any], name: str, default: int = 0) -> int:
    value = attrs.get(name, default)
    if value is None:
        return default
    return int(value)


def str_attr(attrs: dict[str, Any], name: str, default: str = "") -> str:
    value = attrs.get(name, default)
    if value is None:
        return default
    return str(value)


def collective_name(attrs: dict[str, Any]) -> str:
    return COMM_TYPE_NAMES.get(int_attr(attrs, "comm_type", -1), "COLLECTIVE")


def classify_comm(node: Any, attrs: dict[str, Any]) -> str:
    name = node.name.lower()
    comm_type = int_attr(attrs, "comm_type", -1)
    coll_name = COMM_TYPE_NAMES.get(comm_type, "")

    if node.type in (NodeType.COMM_SEND_NODE, NodeType.COMM_RECV_NODE):
        if "_y_send" in name or "_y_recv" in name:
            return "PP"
        if "fwd" in name or "bckwd" in name or "send_node" in name or "recv_node" in name:
            return "PP"
        return "P2P"

    if "dp_all-reduce" in name or "dp_all_reduce" in name:
        return "DP"
    if "alltoall" in name or "all_to_all" in name or "moe" in name or "expert" in name:
        return "EP" if coll_name == "ALL_TO_ALL" else "MOE"
    if "assembled_weight" in name:
        return "FSDP"
    if "sharded_grad" in name:
        return "FSDP" if coll_name == "REDUCE_SCATTER" else "DP"
    if coll_name == "ALL_REDUCE" and (
        ".w@" in name or name.endswith(".w_x2_comm") or "_grad" in name
    ):
        return "DP"
    if coll_name == "ALL_TO_ALL":
        return "EP"
    if coll_name in {"ALL_GATHER", "REDUCE_SCATTER"}:
        return "TP"
    if coll_name == "ALL_REDUCE":
        return "TP"
    return coll_name or "COMM"


def group_key(trace_node: TraceNode) -> tuple[Any, ...]:
    node = trace_node.node
    attrs = trace_node.attrs
    if node.type == NodeType.COMM_COLL_NODE:
        return (
            "coll",
            int(node.id),
            node.name,
            int_attr(attrs, "comm_type", -1),
            str_attr(attrs, "pg_name", ""),
            int_attr(attrs, "comm_size", 0),
        )
    if node.type in (NodeType.COMM_SEND_NODE, NodeType.COMM_RECV_NODE):
        if "comm_src" in attrs and "comm_dst" in attrs:
            endpoint_key: tuple[Any, ...] = (
                int_attr(attrs, "comm_src", -1),
                int_attr(attrs, "comm_dst", -1),
            )
        else:
            endpoint_key = ("partial-endpoint",)
        return (
            "p2p",
            int_attr(attrs, "comm_tag", -1),
            *endpoint_key,
            int_attr(attrs, "comm_size", 0),
        )
    return ("rank-node", trace_node.rank, int(node.id))


def group_kind(node_type: int) -> str:
    if node_type in (NodeType.COMM_COLL_NODE, NodeType.COMM_SEND_NODE, NodeType.COMM_RECV_NODE):
        return "comm"
    if node_type == NodeType.COMP_NODE:
        return "comp"
    return "other"


def comm_duration_ns(
    members: list[TraceNode],
    bytes_per_ns: float,
    latency_ns: int,
    all_reduce_factor: float,
) -> int:
    node = members[0].node
    attrs = members[0].attrs
    size = int_attr(attrs, "comm_size", 0)
    factor = 1.0
    if node.type == NodeType.COMM_COLL_NODE:
        comm_type = int_attr(attrs, "comm_type", -1)
        if comm_type == int(CollectiveCommType.ALL_REDUCE):
            factor = all_reduce_factor
    return max(1, int(math.ceil(latency_ns + factor * size / bytes_per_ns)))


def non_comm_duration_ns(member: TraceNode, compute_tflops: float) -> int:
    node = member.node
    if int(node.duration_micros):
        return max(1, int(node.duration_micros) * 1000)
    if node.type == NodeType.COMP_NODE:
        num_ops = int_attr(member.attrs, "num_ops", 0)
        if num_ops > 0 and compute_tflops > 0:
            return max(1, int(math.ceil(num_ops / (compute_tflops * 1_000.0))))
    return 1


def build_groups(
    traces: dict[int, list[TraceNode]],
    bytes_per_ns: float,
    latency_ns: int,
    all_reduce_factor: float,
    compute_tflops: float,
) -> list[Group]:
    key_to_gid: dict[tuple[Any, ...], int] = {}
    groups: list[Group] = []
    node_to_gid: dict[NodeRef, int] = {}

    for rank, nodes in traces.items():
        for trace_node in nodes:
            key = group_key(trace_node)
            gid = key_to_gid.get(key)
            if gid is None:
                node = trace_node.node
                kind = group_kind(node.type)
                comm_class = classify_comm(node, trace_node.attrs) if kind == "comm" else None
                if kind == "comm":
                    if node.type == NodeType.COMM_COLL_NODE:
                        label = f"{comm_class}:{collective_name(trace_node.attrs)}:{node.name}"
                    else:
                        label = f"{comm_class}:P2P:tag{int_attr(trace_node.attrs, 'comm_tag', -1)}"
                else:
                    label = node.name
                gid = len(groups)
                key_to_gid[key] = gid
                groups.append(
                    Group(
                        gid=gid,
                        key=key,
                        kind=kind,
                        label=label,
                        comm_class=comm_class,
                    )
                )
            groups[gid].members.append(trace_node)
            groups[gid].ranks.add(rank)
            node_to_gid[NodeRef(rank, int(trace_node.node.id))] = gid

    for group in groups:
        if group.kind == "comm":
            group.duration_ns = comm_duration_ns(
                group.members, bytes_per_ns, latency_ns, all_reduce_factor
            )
        else:
            group.duration_ns = max(
                non_comm_duration_ns(member, compute_tflops) for member in group.members
            )

    for group in groups:
        for member in group.members:
            for dep_id in member.deps:
                dep_gid = node_to_gid.get(NodeRef(member.rank, int(dep_id)))
                if dep_gid is not None and dep_gid != group.gid:
                    group.deps.add(dep_gid)

    for group in groups:
        for dep_gid in group.deps:
            groups[dep_gid].children.add(group.gid)

    return groups


def plane_for_group(group: Group, two_plane: bool) -> str:
    if not two_plane:
        return "single"
    if group.comm_class in {"DP", "FSDP"}:
        return "data"
    return "model_pipe"


def schedule_groups(
    groups: list[Group],
    rank_resources: bool,
    *,
    rank_plane_resources: bool = False,
    two_plane: bool = False,
    global_plane_resources: bool = False,
) -> None:
    indegree = {group.gid: len(group.deps) for group in groups}
    dep_ready = {group.gid: 0 for group in groups}
    heap: list[tuple[int, int]] = [(0, group.gid) for group in groups if indegree[group.gid] == 0]
    heapq.heapify(heap)

    comp_ready: defaultdict[int, int] = defaultdict(int)
    comm_ready: defaultdict[int, int] = defaultdict(int)
    rank_plane_ready: defaultdict[tuple[int, str], int] = defaultdict(int)
    global_plane_ready: defaultdict[str, int] = defaultdict(int)
    scheduled = 0

    while heap:
        _, gid = heapq.heappop(heap)
        group = groups[gid]
        start = dep_ready[gid]
        if rank_plane_resources and group.kind == "comm":
            plane = plane_for_group(group, two_plane=two_plane)
            start = max(start, *(rank_plane_ready[(rank, plane)] for rank in group.ranks))
            if global_plane_resources:
                start = max(start, global_plane_ready[plane])
        elif rank_resources:
            if group.kind == "comm":
                start = max(start, *(comm_ready[rank] for rank in group.ranks))
            elif group.kind == "comp":
                start = max(start, *(comp_ready[rank] for rank in group.ranks))
        group.start_ns = start
        group.end_ns = start + group.duration_ns
        if rank_plane_resources and group.kind == "comm":
            plane = plane_for_group(group, two_plane=two_plane)
            for rank in group.ranks:
                rank_plane_ready[(rank, plane)] = group.end_ns
            if global_plane_resources:
                global_plane_ready[plane] = group.end_ns
        elif rank_resources:
            if group.kind == "comm":
                for rank in group.ranks:
                    comm_ready[rank] = group.end_ns
            elif group.kind == "comp":
                for rank in group.ranks:
                    comp_ready[rank] = group.end_ns
        scheduled += 1
        for child in group.children:
            dep_ready[child] = max(dep_ready[child], group.end_ns)
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(heap, (dep_ready[child], child))

    if scheduled != len(groups):
        raise ValueError(
            f"Could not schedule all groups: scheduled {scheduled}, total {len(groups)}"
        )


def union_duration(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals = sorted((s, e) for s, e in intervals if e > s)
    if not intervals:
        return 0
    total = 0
    cur_s, cur_e = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_e:
            cur_e = max(cur_e, end)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = start, end
    total += cur_e - cur_s
    return total


def overlap_metrics(events: list[Group]) -> dict[str, Any]:
    boundaries: list[tuple[int, int, Group]] = []
    for event in events:
        boundaries.append((event.start_ns, 1, event))
        boundaries.append((event.end_ns, -1, event))
    boundaries.sort(key=lambda item: (item[0], item[1]))

    active: set[int] = set()
    last: int | None = None
    any_active_time = 0
    any_multi_event = 0
    multi_type = 0
    same_rank_multi_type = 0
    pair_overlap: defaultdict[tuple[str, str], int] = defaultdict(int)
    pair_shared_rank_overlap: defaultdict[tuple[str, str], int] = defaultdict(int)
    sample_overlaps: list[tuple[int, int, tuple[str, ...]]] = []

    event_by_gid = {event.gid: event for event in events}

    for time, delta, event in boundaries:
        if last is not None and time > last and active:
            duration = time - last
            any_active_time += duration
            classes = sorted({event_by_gid[gid].comm_class or "COMM" for gid in active})
            if len(active) >= 2:
                any_multi_event += duration
            if len(classes) >= 2:
                ranks_by_class: dict[str, set[int]] = defaultdict(set)
                classes_by_rank: dict[int, set[str]] = defaultdict(set)
                for gid in active:
                    active_event = event_by_gid[gid]
                    cls = active_event.comm_class or "COMM"
                    ranks_by_class[cls].update(active_event.ranks)
                    for rank in active_event.ranks:
                        classes_by_rank[rank].add(cls)
                rank_local_pairs: set[tuple[str, str]] = set()
                for rank_classes in classes_by_rank.values():
                    if len(rank_classes) >= 2:
                        rank_local_pairs.update(itertools.combinations(sorted(rank_classes), 2))
                if rank_local_pairs:
                    same_rank_multi_type += duration
                multi_type += duration
                for pair in itertools.combinations(classes, 2):
                    pair_overlap[pair] += duration
                    if pair in rank_local_pairs:
                        pair_shared_rank_overlap[pair] += duration
                if len(sample_overlaps) < 12:
                    sample_overlaps.append((last, time, tuple(classes)))
        if delta > 0:
            active.add(event.gid)
        else:
            active.discard(event.gid)
        last = time

    # Use the sweep-derived active time as the denominator. It is equivalent to
    # interval union for well-formed inputs and stays consistent with the same
    # active-set semantics used for cross-type overlap.
    comm_union = any_active_time
    by_class: dict[str, dict[str, Any]] = {}
    for cls in sorted({event.comm_class or "COMM" for event in events}):
        class_events = [event for event in events if (event.comm_class or "COMM") == cls]
        by_class[cls] = {
            "events": len(class_events),
            "active_ns": union_duration([(event.start_ns, event.end_ns) for event in class_events]),
            "first_start_ns": min(event.start_ns for event in class_events),
            "last_end_ns": max(event.end_ns for event in class_events),
            "total_event_ns": sum(event.end_ns - event.start_ns for event in class_events),
        }

    return {
        "comm_union_ns": comm_union,
        "any_multi_event_ns": any_multi_event,
        "multi_type_ns": multi_type,
        "same_rank_multi_type_ns": same_rank_multi_type,
        "any_multi_event_ratio": any_multi_event / comm_union if comm_union else 0.0,
        "multi_type_ratio": multi_type / comm_union if comm_union else 0.0,
        "same_rank_multi_type_ratio": same_rank_multi_type / comm_union if comm_union else 0.0,
        "pair_overlap": dict(pair_overlap),
        "pair_shared_rank_overlap": dict(pair_shared_rank_overlap),
        "samples": sample_overlaps,
        "by_class": by_class,
    }


def ns_to_us(value: int) -> float:
    return value / 1_000.0


def print_report(name: str, trace_dir: Path, groups: list[Group], mode_name: str) -> dict[str, Any]:
    events = [group for group in groups if group.kind == "comm"]
    metrics = overlap_metrics(events)
    mode = mode_name
    print(f"\n=== {name} | {mode} ===")
    print(f"trace_dir\t{trace_dir}")
    print(f"comm_events\t{len(events)}")
    print(f"comm_union_us\t{ns_to_us(metrics['comm_union_ns']):.3f}")
    print(f"any_comm_event_overlap_ratio\t{metrics['any_multi_event_ratio']:.4f}")
    print(f"cross_type_overlap_ratio\t{metrics['multi_type_ratio']:.4f}")
    print(f"same_rank_cross_type_overlap_ratio\t{metrics['same_rank_multi_type_ratio']:.4f}")
    print("by_class\tclass\tevents\tactive_us\tfirst_start_us\tlast_end_us\ttotal_event_us")
    for cls, item in metrics["by_class"].items():
        print(
            "by_class\t"
            f"{cls}\t{item['events']}\t{ns_to_us(item['active_ns']):.3f}\t"
            f"{ns_to_us(item['first_start_ns']):.3f}\t"
            f"{ns_to_us(item['last_end_ns']):.3f}\t"
            f"{ns_to_us(item['total_event_ns']):.3f}"
        )

    print("top_cross_type_pairs\tpair\toverlap_us")
    for (a, b), value in sorted(
        metrics["pair_overlap"].items(), key=lambda item: item[1], reverse=True
    )[:8]:
        shared = metrics["pair_shared_rank_overlap"].get((a, b), 0)
        print(f"top_cross_type_pairs\t{a}+{b}\t{ns_to_us(value):.3f}\tshared_rank_us={ns_to_us(shared):.3f}")

    print("sample_overlap_windows\tstart_us\tend_us\tclasses")
    for start, end, classes in metrics["samples"][:8]:
        print(
            f"sample_overlap_windows\t{ns_to_us(start):.3f}\t"
            f"{ns_to_us(end):.3f}\t{','.join(classes)}"
        )
    return metrics


def dump_events_csv(output_dir: Path, name: str, groups: list[Group], mode: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    path = output_dir / f"{safe_name}.{mode}.events.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "comm_class",
                "start_us",
                "end_us",
                "duration_us",
                "ranks",
                "label",
            ]
        )
        for group in sorted(
            (group for group in groups if group.kind == "comm"),
            key=lambda item: (item.start_ns, item.end_ns, item.label),
        ):
            writer.writerow(
                [
                    group.comm_class,
                    f"{ns_to_us(group.start_ns):.6f}",
                    f"{ns_to_us(group.end_ns):.6f}",
                    f"{ns_to_us(group.end_ns - group.start_ns):.6f}",
                    " ".join(str(rank) for rank in sorted(group.ranks)),
                    group.label,
                ]
            )


def run_case(args: argparse.Namespace, case: str) -> None:
    if "=" in case:
        name, raw_path = case.split("=", 1)
    else:
        path = Path(case)
        name, raw_path = path.name, case
    trace_dir = Path(raw_path).resolve()
    traces = load_trace_dir(trace_dir)

    def make_groups(bytes_per_ns: float) -> list[Group]:
        return build_groups(
            traces=traces,
            bytes_per_ns=bytes_per_ns,
            latency_ns=args.latency_ns,
            all_reduce_factor=args.all_reduce_factor,
            compute_tflops=args.compute_tflops,
        )

    scenario_specs: list[tuple[str, float, dict[str, Any]]] = []
    for mode_name in args.modes:
        if mode_name == "dag":
            scenario_specs.append(
                ("dag_only_unlimited_comm", args.bytes_per_ns, {"rank_resources": False})
            )
        elif mode_name == "astra-like":
            scenario_specs.append(
                ("astra_like_rank_serial", args.bytes_per_ns, {"rank_resources": True})
            )
        elif mode_name == "two-plane-upper":
            scenario_specs.append(
                (
                    "two_plane_upper_same_bandwidth",
                    args.bytes_per_ns,
                    {
                        "rank_resources": False,
                        "rank_plane_resources": True,
                        "two_plane": True,
                    },
                )
            )
        elif mode_name == "two-plane-fair":
            scenario_specs.append(
                (
                    "two_plane_fair_half_bandwidth_per_plane",
                    args.bytes_per_ns / 2.0,
                    {
                        "rank_resources": False,
                        "rank_plane_resources": True,
                        "two_plane": True,
                    },
                )
            )
        elif mode_name == "single-plane-global":
            scenario_specs.append(
                (
                    "single_plane_global_slot",
                    args.bytes_per_ns,
                    {
                        "rank_resources": False,
                        "rank_plane_resources": True,
                        "two_plane": False,
                        "global_plane_resources": True,
                    },
                )
            )
        elif mode_name == "two-plane-global-fair":
            scenario_specs.append(
                (
                    "two_plane_global_slot_fair_half_bandwidth",
                    args.bytes_per_ns / 2.0,
                    {
                        "rank_resources": False,
                        "rank_plane_resources": True,
                        "two_plane": True,
                        "global_plane_resources": True,
                    },
                )
            )
        else:
            raise ValueError(f"Unknown mode {mode_name}")

    wall_times: dict[str, int] = {}
    for mode_name, bytes_per_ns, schedule_kwargs in scenario_specs:
        base_groups = make_groups(bytes_per_ns)
        groups = [
            Group(
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
            for group in base_groups
        ]
        schedule_groups(groups, **schedule_kwargs)
        print_report(name, trace_dir, groups, mode_name=mode_name)
        wall_times[mode_name] = max(group.end_ns for group in groups)
        if args.dump_events_csv:
            dump_events_csv(Path(args.dump_events_csv), name, groups, mode_name)

    baseline = wall_times.get("astra_like_rank_serial")
    if baseline:
        print(f"\n=== {name} | speedup_vs_astra_like_rank_serial ===")
        for mode_name, wall_time in wall_times.items():
            print(f"speedup\t{mode_name}\t{baseline / wall_time:.4f}\twall_us={ns_to_us(wall_time):.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases",
        nargs="+",
        help="Trace directories, optionally named as name=/path/to/trace_dir.",
    )
    parser.add_argument(
        "--bytes-per-ns",
        type=float,
        default=100.0,
        help="Per-event bandwidth model. 100 bytes/ns equals 800 Gbps.",
    )
    parser.add_argument(
        "--latency-ns",
        type=int,
        default=10,
        help="Fixed latency added to each communication event.",
    )
    parser.add_argument(
        "--all-reduce-factor",
        type=float,
        default=2.0,
        help="Duration multiplier for ALL_REDUCE events.",
    )
    parser.add_argument(
        "--compute-tflops",
        type=float,
        default=700.0,
        help="Fallback compute model for COMP_NODE num_ops when duration_micros is absent.",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "both",
            "dag",
            "astra-like",
            "two-plane-upper",
            "two-plane-fair",
            "single-plane-global",
            "two-plane-global-fair",
            "plane-toy",
            "all",
        ),
        default="both",
        help="Schedule with DAG, ASTRA-like, and/or wavelength-plane toy resource models.",
    )
    parser.add_argument(
        "--dump-events-csv",
        help="Optional output directory for per-communication-event start/end CSV files.",
    )
    args = parser.parse_args()
    if args.mode == "both":
        args.modes = ["dag", "astra-like"]
    elif args.mode == "plane-toy":
        args.modes = ["astra-like", "two-plane-upper", "two-plane-fair"]
    elif args.mode == "all":
        args.modes = [
            "dag",
            "astra-like",
            "two-plane-upper",
            "two-plane-fair",
            "single-plane-global",
            "two-plane-global-fair",
        ]
    else:
        args.modes = [args.mode]
    return args


def main() -> None:
    args = parse_args()
    for case in args.cases:
        run_case(args, case)


if __name__ == "__main__":
    main()
