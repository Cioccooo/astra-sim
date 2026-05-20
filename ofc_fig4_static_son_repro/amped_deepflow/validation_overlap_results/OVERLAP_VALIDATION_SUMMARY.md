# Overlap Validation Summary

## Scope
This package validates the current frozen mainline using the trusted scheduler-explicit DP and backward-TP overlap metrics. Quantitative overlap claims are taken from `training_time_breakdown_overlap.txt`; run settings are taken from `config_summary.txt`; `training_time_breakdown.txt` is used only to read `Total time to train (s)` because that value is not duplicated in the overlap breakdown file.

## Matrix
- Workloads: GPT3 175B, Megatron 530B
- Modes: lump, layerwise, bucketed(4 layers)
- OCS reconfiguration settings: 10 us, 500 us
- Total runs: 12

## Trusted Invariants
- All 12 runs collected successfully: yes
- DP communication raw = hidden + exposed residual max abs: 1.648e-12
- DP reconfiguration raw = hidden + exposed residual max abs: 5.542e-17

## Representative Mechanism Evidence
- Help case: GPT3 175B / layerwise / 10 us -> `snippets/mechanism_help_snippet.csv`
- Reduced-benefit case: GPT3 175B / bucketed(4L) / 500 us -> `snippets/mechanism_reduced_benefit_snippet.csv`

## Plot Files
- `plots/GPT3_175B_comm_exposed_total.png`
- `plots/GPT3_175B_dp_comm_breakdown_10us.png`
- `plots/GPT3_175B_dp_comm_breakdown_500us.png`
- `plots/GPT3_175B_total_time.png`
- `plots/Megatron_530B_comm_exposed_total.png`
- `plots/Megatron_530B_dp_comm_breakdown_10us.png`
- `plots/Megatron_530B_dp_comm_breakdown_500us.png`
- `plots/Megatron_530B_total_time.png`

## Notes on Interpretation
- The main claim is about trusted scheduler-explicit DP overlap and backward-TP effects.
- PP rows in the overlap CSV remain useful diagnostic context, but they are not treated here as a fully unified global wall-clock truth.
- If exposed DP communication falls while total time does not improve, the difference comes from the wider critical path, including non-DP communication, shared-fabric waits, and backward waiting for TP completion.
