#!/usr/bin/env python3
"""Run MLSynth for the OFC Fig. 4 GPT-3 SON case.

This runner is intentionally separate from mlsynth_tests/run_mlsynth.py because
that helper validates `batch_size / dp_size >= num_microbatches`, assuming
`batch_size` is global. MLSynth's current MegatronLM implementation uses
`batch_size` directly in each DP replica's compute/PP activation formulas, so
for this reproduction we pass the per-DP-replica batch size instead.
"""

import argparse
import json
import os
from pathlib import Path

import yaml
from chakra.src.third_party.utils.protolib import encodeMessage
from Model.Transformer import Transformer
from Orchestrator.MegatronLM import MegatronLM


def write_comm_groups(comm_groups, path):
    with open(path / "comm_groups.json", "w") as f:
        json.dump(comm_groups, f, indent=2, sort_keys=True)


def write_nodes(nodes, name, path):
    path.mkdir(parents=True, exist_ok=True)
    for npu_id, npu_nodes in nodes.items():
        with open(path / f"{name}.{npu_id}.et", "wb") as et:
            for node in npu_nodes:
                encodeMessage(et, node)


def validate_config(cfg):
    if cfg["model"]["batch_size"] < cfg["model"]["num_microbatches"]:
        raise ValueError("per-replica batch_size must be >= num_microbatches")
    if cfg["model"]["num_layers"] % cfg["parallelism"]["pp_size"] != 0:
        raise ValueError("num_layers must be divisible by pp_size")


def output_name(cfg):
    model = cfg["model"]
    parallelism = cfg["parallelism"]
    scale_pct = int(model["scale"] * 100)
    return (
        f"{model['name']}_{parallelism['dp_size']}dp_"
        f"{parallelism['pp_size']}pp_{parallelism['tp_size']}tp_"
        f"{model['batch_size']}Bperdp_{model['sequence_len']}S_"
        f"{model['vocab_size']}V_{model['hidden_size']}d_"
        f"{model['bytes_per_val']}b_{scale_pct}scale"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--output-root", default="output")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)

    name = output_name(cfg)
    out_dir = Path(args.output_root) / name
    et_dir = out_dir / "et"
    os.makedirs(et_dir, exist_ok=True)

    model = Transformer(cfg)
    orchestrator = MegatronLM(model, cfg)

    write_comm_groups(orchestrator.generate_comm_groups(), out_dir)
    write_nodes(orchestrator.exec(), name, et_dir)
    print(out_dir)


if __name__ == "__main__":
    main()

