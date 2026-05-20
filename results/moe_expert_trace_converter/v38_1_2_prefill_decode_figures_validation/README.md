# V38.1/2 Prefill-Informed Decode OCS Figures + Validation

This is an inference-only MoE communication study. It uses `trace[0]` prefill as the selection signal and evaluates on `trace[1:]` decode only.

## What Is Plotted

- Figure 1A: raw decode native ASTRA communication time in ms.
- Figure 1B: native ASTRA decode time normalised by fair universal static = 1.0.
- Figure 2: prefill-informed OCS reconfiguration penalty sensitivity.
- Figure 3: prefill-to-decode predictability.

## Claims Supported

- The three required HF expert-selection workloads were parsed with prefill/decode separation.
- The six plotted methods were evaluated with native ASTRA analytical congestion-aware GraphTopology.
- Fluid link-load values are retained only as lower-bound/explanation columns.
- Prefill-informed OCS uses prefill only; decode oracle is labelled as an upper bound.
- Optical methods use the same degree-4, 400Gb/s/link, ECMP4 budget.

## Claims Not Supported

- This is not MoE training.
- This is not full serving latency.
- This is not native ASTRA in-run topology swap.
- This is not a paper-final figure set.

## Key Validation Summary

- `qwen_mmlu_machine_learning`: OCS vs fair static = False, 1us = False, 10us = False, 25ms = False; prefill candidate `random_regular_seed_11`.
- `deepseek_mmlu_machine_learning`: OCS vs fair static = False, 1us = False, 10us = False, 25ms = False; prefill candidate `random_regular_seed_11`.
- `qwen_livecodebench_execution`: OCS vs fair static = False, 1us = False, 10us = False, 25ms = False; prefill candidate `random_regular_seed_4`.

## Files

- `plotted_values.csv/json`: values used by figures.
- `native_astra_timing_table.csv/json`: native ASTRA dispatch/combine run results.
- `validation_summary.json`: compact pass/fail summary.
- `trace_validation_table.csv`: trace and byte-conservation validation.
- `no_leakage_validation_table.csv`: selection/evaluation split for every method.
- `topology_validation_table.csv`: topology budget and graph checks.
- `why_win_table.csv`: bottleneck, hop-count, hot-link, and gain/loss explanation metrics.
- `penalty_table.csv`: reconfiguration penalty sensitivity.
- `figures/*.png` and `figures/*.pdf`: preliminary supervisor figures.
