#!/usr/bin/env python3
"""Generate a tiny MoE/expert-parallel Chakra trace for ASTRA-sim.

This is intentionally small: it proves that ASTRA-sim can execute Chakra traces
containing expert-parallel all-to-all token shuffles. It is not a full MoE
router model; traffic is balanced across the EP group.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


MLSYNTH_ROOT = Path("/Users/dfx/Python/MLSynth")
sys.path.insert(0, str(MLSYNTH_ROOT))

from Layer.TransformerMoeLayer import TransformerMoeLayer  # noqa: E402
from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata  # noqa: E402
from chakra.src.third_party.utils.protolib import encodeMessage  # noqa: E402
from utils import add_dependencies  # noqa: E402


def write_nodes(nodes_by_rank, trace_prefix, output_dir):
    et_dir = output_dir / "et"
    et_dir.mkdir(parents=True, exist_ok=True)
    for rank, nodes in nodes_by_rank.items():
        with (et_dir / f"{trace_prefix}.{rank}.et").open("wb") as dst:
            for node in nodes:
                encodeMessage(dst, node)


def generate_trace(args):
    layer = TransformerMoeLayer(
        num_layers=args.layers,
        hidden_size=args.hidden_size,
        sequence_len=args.sequence_len,
        vocab_size=args.vocab_size,
        ep_size=args.ep_size,
        tp_size=1,
        top_k=args.top_k,
        capacity_factor=args.capacity_factor,
        bytes_per_val=args.bytes_per_val,
        scale=args.scale,
    )

    nodes_by_rank = defaultdict(list)
    ep_group_name = "ep_0"
    micro_batch = args.batch_size / args.microbatches

    for rank in range(args.ep_size):
        nodes = nodes_by_rank[rank]
        nodes.append(GlobalMetadata(version="0.0.4"))
        previous = None

        for microbatch in range(args.microbatches):
            for layer_id in range(args.layers):
                fwd_nodes = layer.fwd(
                    name=f"COMP_NODE_FWD_mb{microbatch}_layer{layer_id}_rank{rank}",
                    pg_name=ep_group_name,
                    num_batches=micro_batch,
                )
                add_dependencies(fwd_nodes[0], [previous])
                nodes.extend(fwd_nodes)
                previous = fwd_nodes[-1]

        if not args.forward_only:
            for microbatch in range(args.microbatches):
                for layer_id in reversed(range(args.layers)):
                    bwd_nodes = layer.bckwd(
                        name=f"COMP_NODE_BWD_mb{microbatch}_layer{layer_id}_rank{rank}",
                        pg_name=ep_group_name,
                        num_batches=micro_batch,
                    )
                    add_dependencies(bwd_nodes[0], [previous])
                    nodes.extend(bwd_nodes)
                    previous = bwd_nodes[-1]

    return nodes_by_rank, {ep_group_name: list(range(args.ep_size))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trace-prefix", default="tiny_moe_ep")
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatches", type=int, default=1)
    parser.add_argument("--sequence-len", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--bytes-per-val", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_by_rank, comm_groups = generate_trace(args)

    (output_dir / "comm_groups.json").write_text(
        json.dumps(comm_groups, indent=2, sort_keys=True)
    )
    write_nodes(nodes_by_rank, args.trace_prefix, output_dir)

    summary = {
        "trace_prefix": args.trace_prefix,
        "ranks": args.ep_size,
        "layers": args.layers,
        "forward_only": args.forward_only,
        "nodes_per_rank": {str(rank): len(nodes) for rank, nodes in nodes_by_rank.items()},
        "comm_groups": comm_groups,
        "moe_limitations": [
            "balanced all-to-all traffic",
            "no dynamic router distribution",
            "no imbalanced expert load",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(output_dir)


if __name__ == "__main__":
    main()
