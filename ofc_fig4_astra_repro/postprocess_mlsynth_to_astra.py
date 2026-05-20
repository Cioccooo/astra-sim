#!/usr/bin/env python3
"""Convert MLSynth Chakra traces to ASTRA-sim input and patch GPT-3 Fig. 4 semantics.

MLSynth emits Chakra traces with string process-group names. Current ASTRA-sim
expects numeric process-group IDs, so this script performs the same mapping as
mlsynth_tests/convert_mlsynth_to_astra.py.

For the OFC Fig. 4 GPT-3 SON sanity check, two additional corrections are applied:

1. Divide compute-node ops by TP degree. MLSynth's TransformerLayer formulas emit
   full-layer FLOPs, while the Megatron-style TP=8 execution shards GEMMs across
   tensor-parallel ranks.
2. Optionally append an optimizer/weight-update compute node after the final node
   on each rank. AMPeD's Fig. 4 compute stack includes weight-update time, but
   MLSynth's trace only models forward/backward plus collectives.
"""

import argparse
import json
from pathlib import Path

from chakra.schema.protobuf.et_def_pb2 import AttributeProto, Node, COMP_NODE
from chakra.src.third_party.utils.protolib import decodeMessage, encodeMessage


def read_nodes(path):
    nodes = []
    with open(path, "rb") as src:
        while True:
            node = Node()
            if not decodeMessage(src, node):
                break
            nodes.append(node)
    return nodes


def write_nodes(path, nodes):
    with open(path, "wb") as dst:
        for node in nodes:
            encodeMessage(dst, node)


def rewrite_pg_names(nodes, group_name_to_id):
    for node in nodes:
        for attr in node.attr:
            if attr.name == "pg_name" and attr.string_val in group_name_to_id:
                attr.string_val = str(group_name_to_id[attr.string_val])


def scale_compute_nodes(nodes, divisor):
    if divisor == 1:
        return
    for node in nodes:
        if node.type != COMP_NODE:
            continue
        for attr in node.attr:
            if attr.name == "num_ops":
                attr.int64_val = max(1, int(round(attr.int64_val / divisor)))
            elif attr.name == "tensor_size":
                attr.uint64_val = max(1, int(round(attr.uint64_val / divisor)))


def append_optimizer_node(nodes, optimizer_ops, optimizer_tensor_size):
    if optimizer_ops <= 0:
        return

    max_id = max((node.id for node in nodes), default=-1)
    last_id = max_id

    opt = Node()
    opt.id = max_id + 1
    opt.name = "COMP_NODE_OPTIMIZER_WEIGHT_UPDATE"
    opt.type = COMP_NODE
    opt.attr.append(AttributeProto(name="is_cpu_op", bool_val=False))
    opt.attr.append(AttributeProto(name="num_ops", int64_val=int(optimizer_ops)))
    opt.attr.append(AttributeProto(name="tensor_size", uint64_val=int(optimizer_tensor_size)))
    if last_id >= 0:
        opt.data_deps.append(last_id)
    nodes.append(opt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="MLSynth output directory")
    parser.add_argument("--dst", required=True, help="ASTRA-compatible output directory")
    parser.add_argument("--trace-prefix", required=True, help="Trace filename prefix before .<rank>.et")
    parser.add_argument("--compute-divisor", type=float, default=1.0)
    parser.add_argument("--optimizer-ops", type=float, default=0.0)
    parser.add_argument("--optimizer-tensor-size", type=int, default=1)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    src_et = src / "et"
    dst.mkdir(parents=True, exist_ok=True)

    comm_groups = json.loads((src / "comm_groups.json").read_text())
    used_group_names = set()
    trace_files = sorted(src_et.glob(f"{args.trace_prefix}.*.et"))

    for trace_file in trace_files:
        for node in read_nodes(trace_file):
            for attr in node.attr:
                if attr.name == "pg_name" and attr.string_val:
                    used_group_names.add(attr.string_val)

    group_name_to_id = {
        name: idx + 1 for idx, name in enumerate(sorted(used_group_names))
    }
    astra_groups = {
        str(group_name_to_id[name]): ranks
        for name, ranks in comm_groups.items()
        if name in group_name_to_id
    }
    (dst / "comm_groups.json").write_text(json.dumps(astra_groups, indent=2, sort_keys=True))
    (dst / "pg_name_map.json").write_text(json.dumps(group_name_to_id, indent=2, sort_keys=True))

    for trace_file in trace_files:
        nodes = read_nodes(trace_file)
        rewrite_pg_names(nodes, group_name_to_id)
        scale_compute_nodes(nodes, args.compute_divisor)
        append_optimizer_node(nodes, args.optimizer_ops, args.optimizer_tensor_size)
        rank_suffix = trace_file.name.removeprefix(args.trace_prefix)
        write_nodes(dst / f"workload{rank_suffix}", nodes)

    summary = {
        "num_traces": len(trace_files),
        "num_process_groups": len(astra_groups),
        "compute_divisor": args.compute_divisor,
        "optimizer_ops_per_rank": args.optimizer_ops,
        "optimizer_tensor_size": args.optimizer_tensor_size,
    }
    (dst / "postprocess_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(dst)


if __name__ == "__main__":
    main()

