# V38-FINAL Prefill-Informed Decode OCS for MoE Inference

## Scope

This is inference-only. It uses `trace[0]` as prefill and `trace[1:]` as decode. Prefill may select topology, placement, or server circuits; decode is the evaluation target. No ASTRA C++ core changes, no figures, no all-path ECMP, and no native in-run topology swap claims.

## Compact Result

```json
{
  "final_ranking_table": [
    {
      "workload": "qwen_livecodebench_execution",
      "stage_evaluated": "decode",
      "selection_signal": "prefill only",
      "batch_size": "all_requests",
      "source_policy": "block_by_token",
      "expert_placement": "block",
      "architecture": "gpu_level_prefill_informed_ocs",
      "best_static_method": "fair_universal_static",
      "best_ocs_method": "gpu_level_prefill_informed_ocs",
      "best_static_time": 90156923,
      "best_ocs_time": 84320844,
      "ocs_gain_vs_fair_static": 6.473245543218018,
      "ocs_gain_vs_son": 71.46853914640509,
      "reconfiguration_penalty": "0us in ranking; see penalty table",
      "oracle_gap": 1.8638689147845817,
      "astra_validated": "no",
      "interpretation": "fluid-only inference decode audit"
    },
    {
      "workload": "qwen_mmlu_zh_cn_anatomy",
      "stage_evaluated": "decode",
      "selection_signal": "prefill only",
      "batch_size": "all_requests",
      "source_policy": "block_by_token",
      "expert_placement": "block",
      "architecture": "gpu_level_prefill_informed_ocs",
      "best_static_method": "fair_universal_static",
      "best_ocs_method": "gpu_level_prefill_informed_ocs",
      "best_static_time": 23484280,
      "best_ocs_time": 22276026,
      "ocs_gain_vs_fair_static": 5.144948024806381,
      "ocs_gain_vs_son": 72.42900970269741,
      "reconfiguration_penalty": "0us in ranking; see penalty table",
      "oracle_gap": 0.0,
      "astra_validated": "no",
      "interpretation": "fluid-only inference decode audit"
    }
  ],
  "prefill_decode_predictability": [
    {
      "workload": "qwen_livecodebench_execution",
      "request_count": 479,
      "prefill_tokens": 52341,
      "decode_tokens": 61312,
      "prefill_top1": 0.02485706965817855,
      "decode_top1": 0.021571369007628925,
      "top8_overlap": 0.375,
      "top16_overlap": 0.4375,
      "spearman": 0.5421969419520234
    },
    {
      "workload": "qwen_mmlu_zh_cn_anatomy",
      "request_count": 135,
      "prefill_tokens": 1569,
      "decode_tokens": 17280,
      "prefill_top1": 0.018690757088808428,
      "decode_top1": 0.014795960771276595,
      "top8_overlap": 0.625,
      "top16_overlap": 0.75,
      "spearman": 0.9138150064090825
    }
  ],
  "final_diagnosis": {
    "q1_prefill_predicts_decode": "yes/moderately",
    "q2_prefill_informed_ocs_beats_fair_static": {
      "qwen_livecodebench_execution": true,
      "qwen_mmlu_zh_cn_anatomy": true
    },
    "q3_batching_source_locality": "see batch_source_locality_table; synthetic source policies are labelled",
    "q4_placement_vs_topology": "see expert_placement_table; placement changes hotness but static baselines receive the same placement",
    "q5_server_regional_vs_gpu_ocs": "see GPU-level vs server-level table; server regional hybrid is often the strongest OCS candidate if residual EPS is allowed",
    "q6_mixnet_greedy_vs_static_expander": "reported per alpha; negative cases are kept",
    "q7_penalty_survival": "see penalty table for 1us/10us/25ms",
    "q8_closest_case_if_not_winning": "see final ranking table",
    "q9_story": "The strongest defensible story is selected from A-E based on final_ranking_table; do not claim training."
  },
  "limitations": {
    "timing_engine": "fluid link-load scoring only; native ASTRA skipped for V38 because decode traces are large and in-run topology swap is unsupported",
    "no_training_claim": true,
    "no_real_serving_latency_claim": true,
    "no_figures": true,
    "ecmp_max_paths": 4
  }
}
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
