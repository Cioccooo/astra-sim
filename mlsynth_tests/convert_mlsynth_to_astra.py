#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from chakra.schema.protobuf.et_def_pb2 import AttributeProto, Node
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="MLSynth output directory")
    parser.add_argument("--dst", required=True, help="ASTRA-compatible output directory")
    parser.add_argument("--trace-prefix", required=True, help="Trace filename prefix before .<rank>.et")
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
        rank_suffix = trace_file.name.removeprefix(args.trace_prefix)
        write_nodes(dst / f"workload{rank_suffix}", nodes)


if __name__ == "__main__":
    main()
