# OFC Fig. 4 ASTRA-sim Sanity Check

Goal: reproduce the first SON/static optical network bar from Fig. 4 of
`Fangxiao Dong_ofc_2026.pdf` using the ASTRA-sim trace flow:

`MLSynth -> Chakra ET -> ASTRA-sim analytical backend`

## Target From AMPeD/DeepFlow

Source:

`/Users/dfx/Python/astra-sim/ofc_fig4_static_son_repro/amped_deepflow/output_files/DP_1_16_TP_8_1_PP_1_8_8/dp_lump_pp_off_tp_off/2026-05-03_13-23-02_training_time_breakdown.txt`

Fig. 4 bar:

- Model: GPT-3 175B
- Network: SON/static optical network
- Parallelism: DP-TP-PP = 16-8-8
- GPUs: 1024
- Training tokens: 300G
- Global batch: 1024
- Sequence length: 2048

Total training iterations:

`300e9 / (1024 * 2048) = 143051.1474609375`

Per-iteration targets:

- Total: `2.751292969 s`
- Compute: `2.159071214 s`
- Communication: `0.590810392 s`
- Pipeline bubble: `0.001411363 s`

## ASTRA Trace Setup

MLSynth config:

`/Users/dfx/Python/astra-sim/ofc_fig4_astra_repro/mlsynth_gpt3_175b_son.yaml`

Important semantic choices:

- MLSynth's current MegatronLM code uses `batch_size` per DP replica, so the config uses `batch_size: 64`, because global batch is `1024` and DP is `16`.
- `num_microbatches: 64` corresponds to AMPeD `microbatch_size=1` for the per-DP local batch.
- The generated trace uses `dp=16`, `pp=8`, `tp=8`, giving 1024 ranks.
- Postprocessing divides compute ops by TP=8, because MLSynth emits full-layer FLOPs while TP shards GEMMs.
- Postprocessing appends one optimizer/weight-update compute node per rank, because AMPeD Fig. 4 includes weight-update compute and MLSynth does not.

Generated ASTRA input:

`/Users/dfx/Python/astra-sim/ofc_fig4_astra_repro/astra_ready/gpt3_175b_fig4_son_16dp_8pp_8tp`

Trace summary:

- Workload traces: `1024`
- Process groups: `192`
- Process group sizes: `8` for TP groups, `16` for DP groups
- MLSynth output size: about `699M`
- ASTRA-ready output size: about `689M`

## Run 1: Physical-ish SON Parameters

System:

`system_h100_effective_roofline.json`

Network:

`network_son_1024_hgx8_ring128.yml`

This keeps the nominal paper/network interpretation:

- Intra-node Switch: `400 GB/s`
- Inter-node SON ring: `25 GB/s` per GPU/NIC rail, from `1.6 Tb/s per 8-GPU node`
- Peak/effective GPU perf: `778.2417908321892 TFLOP/s`

ASTRA command:

```bash
/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware \
  --workload-configuration="$D/workload" \
  --comm-group-configuration="$D/comm_groups.json" \
  --system-configuration=/Users/dfx/Python/astra-sim/ofc_fig4_astra_repro/system_h100_effective_roofline.json \
  --network-configuration=/Users/dfx/Python/astra-sim/ofc_fig4_astra_repro/network_son_1024_hgx8_ring128.yml \
  --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json
```

Result:

- Finished ranks: `1024 / 1024`
- Max wall time: `4.598284373 s`
- GPU time: `3.541791140 s`
- Max exposed communication: `1.056493233 s`
- Return code: `133` after all ranks finished, consistent with the ASTRA post-run SIGTRAP seen in earlier large Chakra runs.

Interpretation:

The trace and topology are runnable, but the result is about `1.67x` slower than
AMPed/DeepFlow. The mismatch is mostly from MLSynth/ASTRA modeling TP collectives
and compute more explicitly than the AMPeD Fig. 4 model.

## Run 2: AMPeD-Calibrated Effective Parameters

System:

`system_h100_amped_calibrated.json`

Network:

`network_son_1024_hgx8_ring128_amped_calibrated.yml`

Calibration:

- Peak/effective GPU perf increased to `1276.6 TFLOP/s` so ASTRA GPU time matches AMPeD compute.
- Effective bandwidths increased to `[714.0, 44.7] GB/s` while preserving the same `Switch x Ring` topology and process groups.
- This is an effective-parameter calibration, not a raw physical-link claim.

Result:

- Finished ranks: `1024 / 1024`
- Max wall time: `2.791398546 s`
- GPU time: `2.159148334 s`
- Max exposed communication: `0.632250212 s`
- Return code: `133` after all ranks finished.

Comparison with AMPeD target:

| Metric | AMPeD target | ASTRA calibrated | Difference |
|---|---:|---:|---:|
| Total iteration time | `2.751293 s` | `2.791399 s` | `+1.46%` |
| Compute/GPU time | `2.159071 s` | `2.159148 s` | `+0.00%` |
| Exposed communication | `0.590810 s` | `0.632250 s` | `+7.01%` |

## Current ASTRA Limitation

`AstraSim_Analytical_Congestion_Aware` cannot run this faithful 2D topology:

```text
[Error] (network/analytical/congestion_aware) only support 1-dim topology
```

So, with the current open-source ASTRA-sim tree on this Mac:

- Faithful `HGX node Switch + static optical Ring` topology works only with `Congestion_Unaware`.
- To use `Congestion_Aware`, we would need either a 1D flattened approximation or implement/port the multi-dimensional congestion-aware extension mentioned in ACOS.

## STG Status

STG is installed locally at:

`/Users/dfx/Python/symbolic_tensor_graph`

Existing local STG tests are recorded at:

`/Users/dfx/Python/astra-sim/stg_generated_tests/astra_results.tsv`

Summary:

- STG can generate Chakra ETs for individual/simple strategies such as DP-only, TP-only, PP-only, and DP+PP.
- Current STG mixed DP+TP traces validate process-group membership, but fail in this ASTRA-sim tree at runtime.
- Example local results:
  - `dp8`: PASS
  - `tp8`: PASS
  - `dp2_pp4`: PASS
  - `dp2_tp4`: FAIL
  - `dp2_tp2_sp2`: FAIL
- Because Fig. 4 requires DP+TP+PP together, MLSynth is currently the reliable trace generator for this reproduction.

Conclusion:

The current verified path for this Fig. 4 bar is `MLSynth -> Chakra -> ASTRA-sim`.
STG should be treated as a future cross-check after fixing or working around the
mixed DP+TP runtime failure in the local ASTRA/STG integration.
