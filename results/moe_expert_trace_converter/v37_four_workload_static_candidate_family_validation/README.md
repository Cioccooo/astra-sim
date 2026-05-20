# V3.7 Four-Workload Static Candidate-Family Validation

## Scope

This extends the V36 aggregated native ASTRA pipeline from one workload to four workloads. It does not implement W=4 dynamic reconfiguration, generate paper figures, use all-path ECMP, or switch to token/layer-level traces.

## Final Answers

1. Do all four workloads run through the aggregated native ASTRA pipeline? **True.**
2. Are all calibrated selections leakage-free? **True.**
3. Are all topologies under the same degree/bandwidth budget? **True.**
4. Which candidate type does calibrated select for each workload? See cross-workload summary.
5. Which candidate type does oracle select for each workload? See cross-workload summary.
6. Does calibrated beat SON? **{'qwen_mmlu_machine_learning': True, 'qwen_livecodebench_execution': True, 'qwen_mmlu_zh_cn_anatomy': True, 'deepseek_livecodebench_execution': True}.**
7. Does calibrated beat fixed random? **{'qwen_mmlu_machine_learning': True, 'qwen_livecodebench_execution': True, 'qwen_mmlu_zh_cn_anatomy': True, 'deepseek_livecodebench_execution': True}.**
8. Does calibrated beat median random? **{'qwen_mmlu_machine_learning': True, 'qwen_livecodebench_execution': True, 'qwen_mmlu_zh_cn_anatomy': True, 'deepseek_livecodebench_execution': True}.**
9. Is the gain mainly workload-aware, greedy hot-pair-aware, or random-regular topology-family strength? See candidate-family interpretation.
10. Which traffic fingerprints predict gain? See cross-workload summary and traffic fingerprints.
11. Is it safe to proceed to RON W=4 segmented validation next? **True.**

## Cross-Workload Summary

```json
[
  {
    "workload": "qwen_mmlu_machine_learning",
    "traffic_gini": 0.08914874853730617,
    "top16_pair_share": 0.022588174010007265,
    "traffic_interpretation": "broad / near-uniform",
    "calibrated_candidate_type": "random_regular",
    "oracle_candidate_type": "random_regular",
    "son_cycles": 88076037,
    "fixed_random_cycles": 74956922,
    "median_random_cycles": 75536328,
    "best_random_cycles": 70863167,
    "calibrated_cycles": 72070031,
    "oracle_cycles": 70863167,
    "calibrated_gain_vs_son_percent": 18.17294072847533,
    "calibrated_gain_vs_fixed_random_percent": 3.851400141537295,
    "oracle_gap_vs_calibrated_percent": 1.6745712236477324,
    "main_explanation": "workload-selected random-regular topology search"
  },
  {
    "workload": "qwen_livecodebench_execution",
    "traffic_gini": 0.0809597951408056,
    "top16_pair_share": 0.023144648128160097,
    "traffic_interpretation": "broad / near-uniform",
    "calibrated_candidate_type": "random_regular",
    "oracle_candidate_type": "random_regular",
    "son_cycles": 1231292825,
    "fixed_random_cycles": 1064751815,
    "median_random_cycles": 1020941313,
    "best_random_cycles": 1013003659,
    "calibrated_cycles": 1013003659,
    "oracle_cycles": 1013003659,
    "calibrated_gain_vs_son_percent": 17.72845269361494,
    "calibrated_gain_vs_fixed_random_percent": 4.860114373226027,
    "oracle_gap_vs_calibrated_percent": 0.0,
    "main_explanation": "workload-selected random-regular topology search"
  },
  {
    "workload": "qwen_mmlu_zh_cn_anatomy",
    "traffic_gini": 0.156996757996986,
    "top16_pair_share": 0.02796755102838233,
    "traffic_interpretation": "moderately structured",
    "calibrated_candidate_type": "random_regular",
    "oracle_candidate_type": "random_regular",
    "son_cycles": 35448255,
    "fixed_random_cycles": 34009401,
    "median_random_cycles": 31716059,
    "best_random_cycles": 29664496,
    "calibrated_cycles": 29664496,
    "oracle_cycles": 29664496,
    "calibrated_gain_vs_son_percent": 16.316061256047725,
    "calibrated_gain_vs_fixed_random_percent": 12.775599899568945,
    "oracle_gap_vs_calibrated_percent": 0.0,
    "main_explanation": "workload-selected random-regular topology search"
  },
  {
    "workload": "deepseek_livecodebench_execution",
    "traffic_gini": 0.024928475575193662,
    "top16_pair_share": 0.01756774031951795,
    "traffic_interpretation": "broad / near-uniform",
    "calibrated_candidate_type": "random_regular",
    "oracle_candidate_type": "random_regular",
    "son_cycles": 732821604,
    "fixed_random_cycles": 595923213,
    "median_random_cycles": 606547850,
    "best_random_cycles": 587268884,
    "calibrated_cycles": 587268884,
    "oracle_cycles": 587268884,
    "calibrated_gain_vs_son_percent": 19.861958108975184,
    "calibrated_gain_vs_fixed_random_percent": 1.4522557287930316,
    "oracle_gap_vs_calibrated_percent": 0.0,
    "main_explanation": "workload-selected random-regular topology search"
  }
]
```

## Candidate-Family Interpretation

```json
{
  "qwen_mmlu_machine_learning": {
    "workload": "qwen_mmlu_machine_learning",
    "calibrated_candidate": "random_regular_seed_16",
    "calibrated_candidate_type": "random_regular",
    "oracle_candidate": "random_regular_seed_22",
    "oracle_candidate_type": "random_regular",
    "traffic_fingerprint": "broad / near-uniform",
    "torus_rank_by_eval_fluid": 34,
    "greedy_calibration_rank_by_eval_fluid": 35,
    "greedy_evaluation_rank_by_eval_fluid": 21,
    "fixed_seed0_rank_by_eval_fluid": 16,
    "calibrated_rank_by_eval_fluid": 2,
    "oracle_rank_by_eval_fluid": 1,
    "calibrated_beats_son": true,
    "calibrated_beats_fixed_random": true,
    "oracle_beats_calibrated": true,
    "calibrated_gain_vs_son_percent": 18.17294072847533,
    "calibrated_gain_vs_fixed_random_percent": 3.851400141537295,
    "oracle_gap_vs_calibrated_percent": 1.6745712236477324,
    "main_explanation": "workload-selected random-regular topology search",
    "median_random_name": "random_regular_seed_14",
    "best_random_name": null
  },
  "qwen_livecodebench_execution": {
    "workload": "qwen_livecodebench_execution",
    "calibrated_candidate": "random_regular_seed_22",
    "calibrated_candidate_type": "random_regular",
    "oracle_candidate": "random_regular_seed_22",
    "oracle_candidate_type": "random_regular",
    "traffic_fingerprint": "broad / near-uniform",
    "torus_rank_by_eval_fluid": 35,
    "greedy_calibration_rank_by_eval_fluid": 31,
    "greedy_evaluation_rank_by_eval_fluid": 33,
    "fixed_seed0_rank_by_eval_fluid": 15,
    "calibrated_rank_by_eval_fluid": 1,
    "oracle_rank_by_eval_fluid": 1,
    "calibrated_beats_son": true,
    "calibrated_beats_fixed_random": true,
    "oracle_beats_calibrated": false,
    "calibrated_gain_vs_son_percent": 17.72845269361494,
    "calibrated_gain_vs_fixed_random_percent": 4.860114373226027,
    "oracle_gap_vs_calibrated_percent": 0.0,
    "main_explanation": "workload-selected random-regular topology search",
    "median_random_name": "random_regular_seed_12",
    "best_random_name": null
  },
  "qwen_mmlu_zh_cn_anatomy": {
    "workload": "qwen_mmlu_zh_cn_anatomy",
    "calibrated_candidate": "random_regular_seed_16",
    "calibrated_candidate_type": "random_regular",
    "oracle_candidate": "random_regular_seed_16",
    "oracle_candidate_type": "random_regular",
    "traffic_fingerprint": "moderately structured",
    "torus_rank_by_eval_fluid": 33,
    "greedy_calibration_rank_by_eval_fluid": 35,
    "greedy_evaluation_rank_by_eval_fluid": 34,
    "fixed_seed0_rank_by_eval_fluid": 21,
    "calibrated_rank_by_eval_fluid": 1,
    "oracle_rank_by_eval_fluid": 1,
    "calibrated_beats_son": true,
    "calibrated_beats_fixed_random": true,
    "oracle_beats_calibrated": false,
    "calibrated_gain_vs_son_percent": 16.316061256047725,
    "calibrated_gain_vs_fixed_random_percent": 12.775599899568945,
    "oracle_gap_vs_calibrated_percent": 0.0,
    "main_explanation": "workload-selected random-regular topology search",
    "median_random_name": "random_regular_seed_14",
    "best_random_name": null
  },
  "deepseek_livecodebench_execution": {
    "workload": "deepseek_livecodebench_execution",
    "calibrated_candidate": "random_regular_seed_22",
    "calibrated_candidate_type": "random_regular",
    "oracle_candidate": "random_regular_seed_22",
    "oracle_candidate_type": "random_regular",
    "traffic_fingerprint": "broad / near-uniform",
    "torus_rank_by_eval_fluid": 35,
    "greedy_calibration_rank_by_eval_fluid": 33,
    "greedy_evaluation_rank_by_eval_fluid": 8,
    "fixed_seed0_rank_by_eval_fluid": 17,
    "calibrated_rank_by_eval_fluid": 1,
    "oracle_rank_by_eval_fluid": 1,
    "calibrated_beats_son": true,
    "calibrated_beats_fixed_random": true,
    "oracle_beats_calibrated": false,
    "calibrated_gain_vs_son_percent": 19.861958108975184,
    "calibrated_gain_vs_fixed_random_percent": 1.4522557287930316,
    "oracle_gap_vs_calibrated_percent": 0.0,
    "main_explanation": "workload-selected random-regular topology search",
    "median_random_name": "random_regular_seed_24",
    "best_random_name": null
  }
}
```

## Workload Results

```json
[
  {
    "workload": {
      "id": "qwen_mmlu_machine_learning",
      "label": "Qwen MMLU machine_learning",
      "path": "/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/mmlu/machine_learning"
    },
    "available": true,
    "trace_parse": {
      "files_found": 112,
      "files_used": 112,
      "moe_layer_count": 94,
      "inferred_num_experts": 128,
      "malformed_records": 0,
      "full": {
        "request_ids": [
          "6835",
          "6836",
          "6837",
          "6838",
          "6839",
          "6840",
          "6841",
          "6842",
          "6843",
          "6844",
          "6845",
          "6846",
          "6847",
          "6848",
          "6849",
          "6850",
          "6851",
          "6852",
          "6853",
          "6854",
          "6855",
          "6856",
          "6857",
          "6858",
          "6859",
          "6860",
          "6861",
          "6862",
          "6863",
          "6864",
          "6865",
          "6866",
          "6867",
          "6868",
          "6869",
          "6870",
          "6871",
          "6872",
          "6873",
          "6874",
          "6875",
          "6876",
          "6877",
          "6878",
          "6879",
          "6880",
          "6881",
          "6882",
          "6883",
          "6884",
          "6885",
          "6886",
          "6887",
          "6888",
          "6889",
          "6890",
          "6891",
          "6892",
          "6893",
          "6894",
          "6895",
          "6896",
          "6897",
          "6898",
          "6899",
          "6900",
          "6901",
          "6902",
          "6903",
          "6904",
          "6905",
          "6906",
          "6907",
          "6908",
          "6909",
          "6910",
          "6911",
          "6912",
          "6913",
          "6914",
          "6915",
          "6916",
          "6917",
          "6918",
          "6919",
          "6920",
          "6921",
          "6922",
          "6923",
          "6924",
          "6925",
          "6926",
          "6927",
          "6928",
          "6929",
          "6930",
          "6931",
          "6932",
          "6933",
          "6934",
          "6935",
          "6936",
          "6937",
          "6938",
          "6939",
          "6940",
          "6941",
          "6942",
          "6943",
          "6944",
          "6945",
          "6946"
        ],
        "request_count": 112,
        "prefill_input_tokens": 3812,
        "selected_expert_events": 2866624,
        "theoretical_dispatch_bytes": 23483383808,
        "theoretical_combine_bytes": 23483383808,
        "local_dispatch_bytes_excluded": 734199808,
        "local_combine_bytes_excluded": 734199808,
        "remote_dispatch_bytes_retained": 22749184000,
        "remote_combine_bytes_retained": 22749184000,
        "byte_conservation_pass": true,
        "dispatch_checksum": "3be34923153d26650dcf8773405ffb4cfdf2ebb122da73bfa23b4798491991f3",
        "combine_checksum": "80e799f5ca37a9fafd7fef465bb32667ad58115d2728042589c98885e16635fd"
      },
      "calibration": {
        "request_ids": [
          "6835",
          "6836",
          "6837",
          "6838",
          "6839",
          "6840",
          "6841",
          "6842",
          "6843",
          "6844",
          "6845",
          "6846"
        ],
        "request_count": 12,
        "prefill_input_tokens": 439,
        "selected_expert_events": 330128,
        "theoretical_dispatch_bytes": 2704408576,
        "theoretical_combine_bytes": 2704408576,
        "local_dispatch_bytes_excluded": 84729856,
        "local_combine_bytes_excluded": 84729856,
        "remote_dispatch_bytes_retained": 2619678720,
        "remote_combine_bytes_retained": 2619678720,
        "byte_conservation_pass": true,
        "dispatch_checksum": "f4cca47bdde8640fd580b27b3e9f8369ae979a51ec7db16ab172ca0e92743a39",
        "combine_checksum": "d94710399125e4bffe862defd91cbdcdf18fa985dcf42b0989d1d0dd6312cdb5"
      },
      "evaluation": {
        "request_ids": [
          "6847",
          "6848",
          "6849",
          "6850",
          "6851",
          "6852",
          "6853",
          "6854",
          "6855",
          "6856",
          "6857",
          "6858",
          "6859",
          "6860",
          "6861",
          "6862",
          "6863",
          "6864",
          "6865",
          "6866",
          "6867",
          "6868",
          "6869",
          "6870",
          "6871",
          "6872",
          "6873",
          "6874",
          "6875",
          "6876",
          "6877",
          "6878",
          "6879",
          "6880",
          "6881",
          "6882",
          "6883",
          "6884",
          "6885",
          "6886",
          "6887",
          "6888",
          "6889",
          "6890",
          "6891",
          "6892",
          "6893",
          "6894",
          "6895",
          "6896",
          "6897",
          "6898",
          "6899",
          "6900",
          "6901",
          "6902",
          "6903",
          "6904",
          "6905",
          "6906",
          "6907",
          "6908",
          "6909",
          "6910",
          "6911",
          "6912",
          "6913",
          "6914",
          "6915",
          "6916",
          "6917",
          "6918",
          "6919",
          "6920",
          "6921",
          "6922",
          "6923",
          "6924",
          "6925",
          "6926",
          "6927",
          "6928",
          "6929",
          "6930",
          "6931",
          "6932",
          "6933",
          "6934",
          "6935",
          "6936",
          "6937",
          "6938",
          "6939",
          "6940",
          "6941",
          "6942",
          "6943",
          "6944",
          "6945",
          "6946"
        ],
        "request_count": 100,
        "prefill_input_tokens": 3373,
        "selected_expert_events": 2536496,
        "theoretical_dispatch_bytes": 20778975232,
        "theoretical_combine_bytes": 20778975232,
        "local_dispatch_bytes_excluded": 649469952,
        "local_combine_bytes_excluded": 649469952,
        "remote_dispatch_bytes_retained": 20129505280,
        "remote_combine_bytes_retained": 20129505280,
        "byte_conservation_pass": true,
        "dispatch_checksum": "484f2a52ab9e785fb9bec5b14e3460e923d8edd48e7d992b4155b7c80cb017b4",
        "combine_checksum": "8082004614f2dfdb95e13215549708d1e216bcf5ddf47c041312f64b8575c055"
      }
    },
    "split": {
      "calibration_request_count": 12,
      "evaluation_request_count": 100,
      "calibration_request_ids": [
        "6835",
        "6836",
        "6837",
        "6838",
        "6839",
        "6840",
        "6841",
        "6842",
        "6843",
        "6844",
        "6845",
        "6846"
      ],
      "evaluation_request_ids": [
        "6847",
        "6848",
        "6849",
        "6850",
        "6851",
        "6852",
        "6853",
        "6854",
        "6855",
        "6856",
        "6857",
        "6858",
        "6859",
        "6860",
        "6861",
        "6862",
        "6863",
        "6864",
        "6865",
        "6866",
        "6867",
        "6868",
        "6869",
        "6870",
        "6871",
        "6872",
        "6873",
        "6874",
        "6875",
        "6876",
        "6877",
        "6878",
        "6879",
        "6880",
        "6881",
        "6882",
        "6883",
        "6884",
        "6885",
        "6886",
        "6887",
        "6888",
        "6889",
        "6890",
        "6891",
        "6892",
        "6893",
        "6894",
        "6895",
        "6896",
        "6897",
        "6898",
        "6899",
        "6900",
        "6901",
        "6902",
        "6903",
        "6904",
        "6905",
        "6906",
        "6907",
        "6908",
        "6909",
        "6910",
        "6911",
        "6912",
        "6913",
        "6914",
        "6915",
        "6916",
        "6917",
        "6918",
        "6919",
        "6920",
        "6921",
        "6922",
        "6923",
        "6924",
        "6925",
        "6926",
        "6927",
        "6928",
        "6929",
        "6930",
        "6931",
        "6932",
        "6933",
        "6934",
        "6935",
        "6936",
        "6937",
        "6938",
        "6939",
        "6940",
        "6941",
        "6942",
        "6943",
        "6944",
        "6945",
        "6946"
      ],
      "calibration_rule": "front ceil(10%) requests"
    },
    "anti_leakage": {
      "calibrated_uses_calibration_only": true,
      "oracle_uses_evaluation_only": true,
      "oracle_reference_only": true
    },
    "traffic_fingerprint": {
      "calibration_dispatch": {
        "total_remote_bytes": 2619678720,
        "nonzero_gpu_pairs": 868,
        "top1_share": 0.001960692340166049,
        "top4_share": 0.007376831308535422,
        "top8_share": 0.014325249777194052,
        "top16_share": 0.027824944884844504,
        "gini": 0.11997523321580765,
        "entropy_bits": 9.726011529138006,
        "message_bytes_min": 958464,
        "message_bytes_median": 3022848.0,
        "message_bytes_mean": 3018063.0414746543,
        "message_bytes_max": 5136384,
        "interpretation": "broad / near-uniform"
      },
      "calibration_combine": {
        "total_remote_bytes": 2619678720,
        "nonzero_gpu_pairs": 868,
        "top1_share": 0.001960692340166049,
        "top4_share": 0.007376831308535422,
        "top8_share": 0.014325249777194052,
        "top16_share": 0.027824944884844504,
        "gini": 0.11997523321580765,
        "entropy_bits": 9.726011529138002,
        "message_bytes_min": 958464,
        "message_bytes_median": 3022848.0,
        "message_bytes_mean": 3018063.0414746543,
        "message_bytes_max": 5136384,
        "interpretation": "broad / near-uniform"
      },
      "evaluation_dispatch": {
        "total_remote_bytes": 20129505280,
        "nonzero_gpu_pairs": 992,
        "top1_share": 0.0015399547862112188,
        "top4_share": 0.005937616366496216,
        "top8_share": 0.011602973284795998,
        "top16_share": 0.022588174010007265,
        "gini": 0.08914874853730617,
        "entropy_bits": 9.936601247209483,
        "message_bytes_min": 13320192,
        "message_bytes_median": 20094976.0,
        "message_bytes_mean": 20291840,
        "message_bytes_max": 30998528,
        "interpretation": "broad / near-uniform"
      },
      "evaluation_combine": {
        "total_remote_bytes": 20129505280,
        "nonzero_gpu_pairs": 992,
        "top1_share": 0.0015399547862112188,
        "top4_share": 0.005937616366496216,
        "top8_share": 0.011602973284795998,
        "top16_share": 0.022588174010007265,
        "gini": 0.08914874853730617,
        "entropy_bits": 9.936601247209476,
        "message_bytes_min": 13320192,
        "message_bytes_median": 20094976.0,
        "message_bytes_mean": 20291840,
        "message_bytes_max": 30998528,
        "interpretation": "broad / near-uniform"
      }
    },
    "candidate_pool": {
      "candidate_count": 35,
      "random_candidate_count": 32,
      "candidate_types": {
        "greedy_calibration": 1,
        "greedy_evaluation": 1,
        "random_regular": 32,
        "torus": 1
      },
      "graph_budget_pass": true,
      "all_candidate_graphs_valid": true
    },
    "candidate_scores": {
      "calibration_top12": [
        {
          "name": "random_regular_seed_16",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 77485397
        },
        {
          "name": "random_regular_seed_22",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 79163391
        },
        {
          "name": "random_regular_seed_3",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 81089195
        },
        {
          "name": "random_regular_seed_19",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 81855147
        },
        {
          "name": "random_regular_seed_27",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 84490923
        },
        {
          "name": "random_regular_seed_18",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 84752385
        },
        {
          "name": "random_regular_seed_13",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 85124438
        },
        {
          "name": "random_regular_seed_9",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 85803007
        },
        {
          "name": "random_regular_seed_30",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 87787520
        },
        {
          "name": "random_regular_seed_25",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 88010069
        },
        {
          "name": "random_regular_seed_20",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 91353772
        },
        {
          "name": "random_regular_seed_14",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 91495084
        }
      ],
      "evaluation_top12": [
        {
          "name": "random_regular_seed_22",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 530889387
        },
        {
          "name": "random_regular_seed_16",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 559740245
        },
        {
          "name": "random_regular_seed_20",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 595292844
        },
        {
          "name": "random_regular_seed_18",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 602815831
        },
        {
          "name": "random_regular_seed_19",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 615293612
        },
        {
          "name": "random_regular_seed_13",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 616904705
        },
        {
          "name": "random_regular_seed_25",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 619573248
        },
        {
          "name": "random_regular_seed_29",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 620749484
        },
        {
          "name": "random_regular_seed_30",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 630147072
        },
        {
          "name": "random_regular_seed_3",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 635793407
        },
        {
          "name": "random_regular_seed_9",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 644166997
        },
        {
          "name": "random_regular_seed_17",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 646270295
        }
      ],
      "evaluation_all": [
        {
          "name": "random_regular_seed_22",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 530889387
        },
        {
          "name": "random_regular_seed_16",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 559740245
        },
        {
          "name": "random_regular_seed_20",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 595292844
        },
        {
          "name": "random_regular_seed_18",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 602815831
        },
        {
          "name": "random_regular_seed_19",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 615293612
        },
        {
          "name": "random_regular_seed_13",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 616904705
        },
        {
          "name": "random_regular_seed_25",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 619573248
        },
        {
          "name": "random_regular_seed_29",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 620749484
        },
        {
          "name": "random_regular_seed_30",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 630147072
        },
        {
          "name": "random_regular_seed_3",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 635793407
        },
        {
          "name": "random_regular_seed_9",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 644166997
        },
        {
          "name": "random_regular_seed_17",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 646270295
        },
        {
          "name": "random_regular_seed_31",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 646653270
        },
        {
          "name": "random_regular_seed_2",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 658284543
        },
        {
          "name": "random_regular_seed_8",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 658412201
        },
        {
          "name": "random_regular_seed_0",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 663841451
        },
        {
          "name": "random_regular_seed_14",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 663965695
        },
        {
          "name": "random_regular_seed_12",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 667076608
        },
        {
          "name": "random_regular_seed_24",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 671512574
        },
        {
          "name": "random_regular_seed_27",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 671531690
        },
        {
          "name": "evaluation_greedy",
          "candidate_type": "greedy_evaluation",
          "score_max_link_load_bytes": 672135851
        },
        {
          "name": "random_regular_seed_11",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 681358677
        },
        {
          "name": "random_regular_seed_15",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 699373567
        },
        {
          "name": "random_regular_seed_5",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 700613293
        },
        {
          "name": "random_regular_seed_23",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 708517888
        },
        {
          "name": "random_regular_seed_1",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 708746581
        },
        {
          "name": "random_regular_seed_7",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 715705686
        },
        {
          "name": "random_regular_seed_6",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 718203561
        },
        {
          "name": "random_regular_seed_4",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 722173955
        },
        {
          "name": "random_regular_seed_21",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 763559254
        },
        {
          "name": "random_regular_seed_26",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 765840726
        },
        {
          "name": "random_regular_seed_10",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 776919722
        },
        {
          "name": "random_regular_seed_28",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 937758037
        },
        {
          "name": "son_torus",
          "candidate_type": "torus",
          "score_max_link_load_bytes": 1171061422
        },
        {
          "name": "calibration_greedy",
          "candidate_type": "greedy_calibration",
          "score_max_link_load_bytes": 1274859523
        }
      ]
    },
    "selected": {
      "ron_calibrated": "random_regular_seed_16",
      "ron_calibrated_type": "random_regular",
      "ron_oracle": "random_regular_seed_22",
      "ron_oracle_type": "random_regular",
      "fixed_random": "random_regular_seed_0",
      "median_random": "random_regular_seed_14",
      "best_random": "random_regular_seed_22",
      "representatives_run_in_astra": [
        "son_torus",
        "random_regular_seed_0",
        "random_regular_seed_14",
        "random_regular_seed_22",
        "random_regular_seed_16",
        "calibration_greedy",
        "evaluation_greedy"
      ],
      "roles": {
        "son_torus": "son",
        "random_regular_seed_0": "fixed_random",
        "random_regular_seed_14": "median_random",
        "random_regular_seed_22": "ron_oracle",
        "random_regular_seed_16": "ron_calibrated",
        "calibration_greedy": "greedy_calibration",
        "evaluation_greedy": "greedy_evaluation"
      }
    },
    "candidate_audits_all_budget_only": {
      "son_torus": {
        "name": "son_torus_4x8_32gpu",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "calibration_greedy": {
        "name": "ron_calibration_greedy_degree4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "evaluation_greedy": {
        "name": "ron_evaluation_greedy_degree4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_0": {
        "name": "ron_random_regular_seed_0",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_1": {
        "name": "ron_random_regular_seed_1",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_2": {
        "name": "ron_random_regular_seed_2",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_3": {
        "name": "ron_random_regular_seed_3",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_4": {
        "name": "ron_random_regular_seed_4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_5": {
        "name": "ron_random_regular_seed_5",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_6": {
        "name": "ron_random_regular_seed_6",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_7": {
        "name": "ron_random_regular_seed_7",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_8": {
        "name": "ron_random_regular_seed_8",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_9": {
        "name": "ron_random_regular_seed_9",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_10": {
        "name": "ron_random_regular_seed_10",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_11": {
        "name": "ron_random_regular_seed_11",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_12": {
        "name": "ron_random_regular_seed_12",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_13": {
        "name": "ron_random_regular_seed_13",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_14": {
        "name": "ron_random_regular_seed_14",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_15": {
        "name": "ron_random_regular_seed_15",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_16": {
        "name": "ron_random_regular_seed_16",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_17": {
        "name": "ron_random_regular_seed_17",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_18": {
        "name": "ron_random_regular_seed_18",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_19": {
        "name": "ron_random_regular_seed_19",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_20": {
        "name": "ron_random_regular_seed_20",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_21": {
        "name": "ron_random_regular_seed_21",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_22": {
        "name": "ron_random_regular_seed_22",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_23": {
        "name": "ron_random_regular_seed_23",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_24": {
        "name": "ron_random_regular_seed_24",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_25": {
        "name": "ron_random_regular_seed_25",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_26": {
        "name": "ron_random_regular_seed_26",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_27": {
        "name": "ron_random_regular_seed_27",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_28": {
        "name": "ron_random_regular_seed_28",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_29": {
        "name": "ron_random_regular_seed_29",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_30": {
        "name": "ron_random_regular_seed_30",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_31": {
        "name": "ron_random_regular_seed_31",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      }
    },
    "candidate_audits_representatives": {
      "son_torus": {
        "name": "son_torus_4x8_32gpu",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 3.096774193548387,
        "diameter": 6,
        "ecmp_path_count_distribution_cap4": {
          "1": 256,
          "2": 192,
          "3": 128,
          "4": 416
        }
      },
      "random_regular_seed_0": {
        "name": "ron_random_regular_seed_0",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.560483870967742,
        "diameter": 4,
        "ecmp_path_count_distribution_cap4": {
          "1": 590,
          "2": 200,
          "3": 84,
          "4": 118
        }
      },
      "random_regular_seed_14": {
        "name": "ron_random_regular_seed_14",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.5745967741935485,
        "diameter": 4,
        "ecmp_path_count_distribution_cap4": {
          "1": 618,
          "2": 206,
          "3": 82,
          "4": 86
        }
      },
      "random_regular_seed_22": {
        "name": "ron_random_regular_seed_22",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.5,
        "diameter": 4,
        "ecmp_path_count_distribution_cap4": {
          "1": 600,
          "2": 230,
          "3": 66,
          "4": 96
        }
      },
      "random_regular_seed_16": {
        "name": "ron_random_regular_seed_16",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.473790322580645,
        "diameter": 4,
        "ecmp_path_count_distribution_cap4": {
          "1": 654,
          "2": 216,
          "3": 54,
          "4": 68
        }
      },
      "calibration_greedy": {
        "name": "ron_calibration_greedy_degree4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.7620967741935485,
        "diameter": 6,
        "ecmp_path_count_distribution_cap4": {
          "1": 636,
          "2": 184,
          "3": 88,
          "4": 84
        }
      },
      "evaluation_greedy": {
        "name": "ron_evaluation_greedy_degree4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.5806451612903225,
        "diameter": 5,
        "ecmp_path_count_distribution_cap4": {
          "1": 668,
          "2": 200,
          "3": 68,
          "4": 56
        }
      }
    },
    "native_astra_results": {
      "dispatch": {
        "son_torus": {
          "label": "qwen_mmlu_machine_learning_dispatch_son_torus",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.03541091701481491,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/network_configs/son_torus.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_dispatch_son_torus.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_dispatch_son_torus.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 43197644,
          "cycles_count": 32
        },
        "random_regular_seed_0": {
          "label": "qwen_mmlu_machine_learning_dispatch_random_regular_seed_0",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.024200750049203634,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/network_configs/random_regular_seed_0.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_dispatch_random_regular_seed_0.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_dispatch_random_regular_seed_0.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 36073603,
          "cycles_count": 32
        },
        "random_regular_seed_14": {
          "label": "qwen_mmlu_machine_learning_dispatch_random_regular_seed_14",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.023956542019732296,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/network_configs/random_regular_seed_14.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_dispatch_random_regular_seed_14.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_dispatch_random_regular_seed_14.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 36892825,
          "cycles_count": 32
        },
        "random_regular_seed_22": {
          "label": "qwen_mmlu_machine_learning_dispatch_random_regular_seed_22",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.02386383304838091,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/network_configs/random_regular_seed_22.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_dispatch_random_regular_seed_22.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_dispatch_random_regular_seed_22.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 34720999,
          "cycles_count": 32
        },
        "random_regular_seed_16": {
          "label": "qwen_mmlu_machine_learning_dispatch_random_regular_seed_16",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.02394720900338143,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/network_configs/random_regular_seed_16.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_dispatch_random_regular_seed_16.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_dispatch_random_regular_seed_16.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 35427999,
          "cycles_count": 32
        },
        "calibration_greedy": {
          "label": "qwen_mmlu_machine_learning_dispatch_calibration_greedy",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.02402487490326166,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/network_configs/calibration_greedy.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_dispatch_calibration_greedy.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_dispatch_calibration_greedy.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 47093642,
          "cycles_count": 32
        },
        "evaluation_greedy": {
          "label": "qwen_mmlu_machine_learning_dispatch_evaluation_greedy",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.023723665974102914,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/network_configs/evaluation_greedy.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_dispatch_evaluation_greedy.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_dispatch_evaluation_greedy.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 41276301,
          "cycles_count": 32
        }
      },
      "combine": {
        "son_torus": {
          "label": "qwen_mmlu_machine_learning_combine_son_torus",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.026274792035110295,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/network_configs/son_torus.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_combine_son_torus.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_combine_son_torus.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 44878393,
          "cycles_count": 32
        },
        "random_regular_seed_0": {
          "label": "qwen_mmlu_machine_learning_combine_random_regular_seed_0",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.024454374914057553,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/network_configs/random_regular_seed_0.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_combine_random_regular_seed_0.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_combine_random_regular_seed_0.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 38883319,
          "cycles_count": 32
        },
        "random_regular_seed_14": {
          "label": "qwen_mmlu_machine_learning_combine_random_regular_seed_14",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.023631791002117097,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/network_configs/random_regular_seed_14.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_combine_random_regular_seed_14.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_combine_random_regular_seed_14.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 38643503,
          "cycles_count": 32
        },
        "random_regular_seed_22": {
          "label": "qwen_mmlu_machine_learning_combine_random_regular_seed_22",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.023923874949105084,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/network_configs/random_regular_seed_22.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_combine_random_regular_seed_22.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_combine_random_regular_seed_22.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 36142168,
          "cycles_count": 32
        },
        "random_regular_seed_16": {
          "label": "qwen_mmlu_machine_learning_combine_random_regular_seed_16",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.024184665991924703,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/network_configs/random_regular_seed_16.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_combine_random_regular_seed_16.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_combine_random_regular_seed_16.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 36642032,
          "cycles_count": 32
        },
        "calibration_greedy": {
          "label": "qwen_mmlu_machine_learning_combine_calibration_greedy",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.02465558296535164,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/network_configs/calibration_greedy.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_combine_calibration_greedy.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_combine_calibration_greedy.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 49704388,
          "cycles_count": 32
        },
        "evaluation_greedy": {
          "label": "qwen_mmlu_machine_learning_combine_evaluation_greedy",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.024586416082456708,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_machine_learning/network_configs/evaluation_greedy.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_combine_evaluation_greedy.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_machine_learning_combine_evaluation_greedy.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 40905739,
          "cycles_count": 32
        }
      }
    },
    "native_astra_totals": {
      "son_torus": {
        "candidate_type": "torus",
        "dispatch_cycles": 43197644,
        "combine_cycles": 44878393,
        "total_cycles": 88076037,
        "dispatch_fluid_cycles": 21812718,
        "combine_fluid_cycles": 20296847,
        "total_fluid_cycles": 42109565,
        "astra_over_fluid_total": 2.0915921833911133,
        "success": true,
        "runtime_s": 0.06168570904992521,
        "role": "son"
      },
      "random_regular_seed_0": {
        "candidate_type": "random_regular",
        "dispatch_cycles": 36073603,
        "combine_cycles": 38883319,
        "total_cycles": 74956922,
        "dispatch_fluid_cycles": 12180226,
        "combine_fluid_cycles": 12365010,
        "total_fluid_cycles": 24545236,
        "astra_over_fluid_total": 3.0538277163030743,
        "success": true,
        "runtime_s": 0.04865512496326119,
        "role": "fixed_random"
      },
      "random_regular_seed_14": {
        "candidate_type": "random_regular",
        "dispatch_cycles": 36892825,
        "combine_cycles": 38643503,
        "total_cycles": 75536328,
        "dispatch_fluid_cycles": 12367324,
        "combine_fluid_cycles": 12271423,
        "total_fluid_cycles": 24638747,
        "astra_over_fluid_total": 3.0657536278123234,
        "success": true,
        "runtime_s": 0.047588333021849394,
        "role": "median_random"
      },
      "random_regular_seed_22": {
        "candidate_type": "random_regular",
        "dispatch_cycles": 34720999,
        "combine_cycles": 36142168,
        "total_cycles": 70863167,
        "dispatch_fluid_cycles": 9819780,
        "combine_fluid_cycles": 9888585,
        "total_fluid_cycles": 19708365,
        "astra_over_fluid_total": 3.5955883199849406,
        "success": true,
        "runtime_s": 0.047787707997485995,
        "role": "ron_oracle"
      },
      "random_regular_seed_16": {
        "candidate_type": "random_regular",
        "dispatch_cycles": 35427999,
        "combine_cycles": 36642032,
        "total_cycles": 72070031,
        "dispatch_fluid_cycles": 10425974,
        "combine_fluid_cycles": 10234629,
        "total_fluid_cycles": 20660603,
        "astra_over_fluid_total": 3.488283038012008,
        "success": true,
        "runtime_s": 0.048131874995306134,
        "role": "ron_calibrated"
      },
      "calibration_greedy": {
        "candidate_type": "greedy_calibration",
        "dispatch_cycles": 47093642,
        "combine_cycles": 49704388,
        "total_cycles": 96798030,
        "dispatch_fluid_cycles": 22927894,
        "combine_fluid_cycles": 23746109,
        "total_fluid_cycles": 46674003,
        "astra_over_fluid_total": 2.0739174653607493,
        "success": true,
        "runtime_s": 0.0486804578686133,
        "role": "greedy_calibration"
      },
      "evaluation_greedy": {
        "candidate_type": "greedy_evaluation",
        "dispatch_cycles": 41276301,
        "combine_cycles": 40905739,
        "total_cycles": 82182040,
        "dispatch_fluid_cycles": 12519505,
        "combine_fluid_cycles": 12518399,
        "total_fluid_cycles": 25037904,
        "astra_over_fluid_total": 3.2823051002991304,
        "success": true,
        "runtime_s": 0.04831008205655962,
        "role": "greedy_evaluation"
      }
    },
    "fluid_lower_bound": {
      "dispatch": {
        "son_torus": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 3.0,
          "selected_path_count_mean": 2.7096774193548385,
          "selected_path_count_max": 4,
          "average_hop_count": 3.5952380952380953,
          "byte_weighted_average_hop_count": 3.100443795109504,
          "max_link_load_bytes": 1171061422,
          "median_link_load_bytes": 456530258.5,
          "average_link_load_bytes": 487581248,
          "fluid_cycles": 21812718,
          "hot_links": [
            {
              "src": 2,
              "dst": 1,
              "bytes": 1171061422
            },
            {
              "src": 3,
              "dst": 2,
              "bytes": 1126604801
            },
            {
              "src": 1,
              "dst": 0,
              "bytes": 1126311939
            },
            {
              "src": 4,
              "dst": 3,
              "bytes": 1072188759
            },
            {
              "src": 0,
              "dst": 7,
              "bytes": 959610882
            },
            {
              "src": 5,
              "dst": 4,
              "bytes": 949595480
            },
            {
              "src": 0,
              "dst": 1,
              "bytes": 944317785
            },
            {
              "src": 1,
              "dst": 2,
              "bytes": 905024857
            }
          ]
        },
        "random_regular_seed_0": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7278225806451613,
          "selected_path_count_max": 4,
          "average_hop_count": 2.911318553092182,
          "byte_weighted_average_hop_count": 2.559305555272941,
          "max_link_load_bytes": 653920939,
          "median_link_load_bytes": 406492841.5,
          "average_link_load_bytes": 402480896,
          "fluid_cycles": 12180226,
          "hot_links": [
            {
              "src": 11,
              "dst": 12,
              "bytes": 653920939
            },
            {
              "src": 1,
              "dst": 19,
              "bytes": 639986345
            },
            {
              "src": 27,
              "dst": 11,
              "bytes": 617734826
            },
            {
              "src": 12,
              "dst": 11,
              "bytes": 599078230
            },
            {
              "src": 19,
              "dst": 1,
              "bytes": 588075009
            },
            {
              "src": 11,
              "dst": 16,
              "bytes": 569738583
            },
            {
              "src": 8,
              "dst": 14,
              "bytes": 561061206
            },
            {
              "src": 13,
              "dst": 31,
              "bytes": 558490966
            }
          ]
        },
        "random_regular_seed_14": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6330645161290323,
          "selected_path_count_max": 4,
          "average_hop_count": 2.8987654320987652,
          "byte_weighted_average_hop_count": 2.57171594671203,
          "max_link_load_bytes": 663965695,
          "median_link_load_bytes": 399968597.0,
          "average_link_load_bytes": 404432576,
          "fluid_cycles": 12367324,
          "hot_links": [
            {
              "src": 14,
              "dst": 29,
              "bytes": 663965695
            },
            {
              "src": 24,
              "dst": 1,
              "bytes": 654398804
            },
            {
              "src": 30,
              "dst": 29,
              "bytes": 639618389
            },
            {
              "src": 1,
              "dst": 24,
              "bytes": 636851541
            },
            {
              "src": 29,
              "dst": 14,
              "bytes": 633746772
            },
            {
              "src": 11,
              "dst": 3,
              "bytes": 600887297
            },
            {
              "src": 22,
              "dst": 7,
              "bytes": 577706668
            },
            {
              "src": 7,
              "dst": 22,
              "bytes": 573026988
            }
          ]
        },
        "random_regular_seed_22": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.655241935483871,
          "selected_path_count_max": 4,
          "average_hop_count": 2.8270401948842876,
          "byte_weighted_average_hop_count": 2.5001165954139135,
          "max_link_load_bytes": 527195478,
          "median_link_load_bytes": 390178475.5,
          "average_link_load_bytes": 393172736,
          "fluid_cycles": 9819780,
          "hot_links": [
            {
              "src": 5,
              "dst": 10,
              "bytes": 527195478
            },
            {
              "src": 10,
              "dst": 5,
              "bytes": 520151723
            },
            {
              "src": 6,
              "dst": 17,
              "bytes": 517053099
            },
            {
              "src": 31,
              "dst": 13,
              "bytes": 511524864
            },
            {
              "src": 15,
              "dst": 12,
              "bytes": 508682923
            },
            {
              "src": 4,
              "dst": 0,
              "bytes": 505860096
            },
            {
              "src": 21,
              "dst": 20,
              "bytes": 503712426
            },
            {
              "src": 18,
              "dst": 1,
              "bytes": 498936491
            }
          ]
        },
        "random_regular_seed_16": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.532258064516129,
          "selected_path_count_max": 4,
          "average_hop_count": 2.763157894736842,
          "byte_weighted_average_hop_count": 2.470562811963951,
          "max_link_load_bytes": 559740245,
          "median_link_load_bytes": 386918058.5,
          "average_link_load_bytes": 388525056,
          "fluid_cycles": 10425974,
          "hot_links": [
            {
              "src": 27,
              "dst": 0,
              "bytes": 559740245
            },
            {
              "src": 19,
              "dst": 22,
              "bytes": 547669333
            },
            {
              "src": 22,
              "dst": 19,
              "bytes": 535906304
            },
            {
              "src": 30,
              "dst": 12,
              "bytes": 530917376
            },
            {
              "src": 24,
              "dst": 10,
              "bytes": 519312725
            },
            {
              "src": 15,
              "dst": 0,
              "bytes": 518716074
            },
            {
              "src": 31,
              "dst": 20,
              "bytes": 516102144
            },
            {
              "src": 11,
              "dst": 12,
              "bytes": 514799617
            }
          ]
        },
        "calibration_greedy": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6169354838709677,
          "selected_path_count_max": 4,
          "average_hop_count": 3.158354114713217,
          "byte_weighted_average_hop_count": 2.7562337849964287,
          "max_link_load_bytes": 1230931964,
          "median_link_load_bytes": 366501548.5,
          "average_link_load_bytes": 433450176,
          "fluid_cycles": 22927894,
          "hot_links": [
            {
              "src": 31,
              "dst": 0,
              "bytes": 1230931964
            },
            {
              "src": 26,
              "dst": 25,
              "bytes": 1118633301
            },
            {
              "src": 0,
              "dst": 31,
              "bytes": 1104627031
            },
            {
              "src": 25,
              "dst": 24,
              "bytes": 969542997
            },
            {
              "src": 7,
              "dst": 0,
              "bytes": 948995414
            },
            {
              "src": 25,
              "dst": 26,
              "bytes": 947455997
            },
            {
              "src": 0,
              "dst": 7,
              "bytes": 940249770
            },
            {
              "src": 24,
              "dst": 25,
              "bytes": 909587111
            }
          ]
        },
        "evaluation_greedy": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.5080645161290323,
          "selected_path_count_max": 4,
          "average_hop_count": 2.879679144385027,
          "byte_weighted_average_hop_count": 2.5655504300600476,
          "max_link_load_bytes": 672135851,
          "median_link_load_bytes": 403861844.5,
          "average_link_load_bytes": 403462976,
          "fluid_cycles": 12519505,
          "hot_links": [
            {
              "src": 5,
              "dst": 25,
              "bytes": 672135851
            },
            {
              "src": 6,
              "dst": 7,
              "bytes": 646345387
            },
            {
              "src": 13,
              "dst": 19,
              "bytes": 639297535
            },
            {
              "src": 26,
              "dst": 27,
              "bytes": 638780073
            },
            {
              "src": 27,
              "dst": 26,
              "bytes": 631215446
            },
            {
              "src": 5,
              "dst": 4,
              "bytes": 629322410
            },
            {
              "src": 2,
              "dst": 23,
              "bytes": 625699499
            },
            {
              "src": 7,
              "dst": 0,
              "bytes": 618306903
            }
          ]
        }
      },
      "combine": {
        "son_torus": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 3.0,
          "selected_path_count_mean": 2.7096774193548385,
          "selected_path_count_max": 4,
          "average_hop_count": 3.5952380952380953,
          "byte_weighted_average_hop_count": 3.100443795109504,
          "max_link_load_bytes": 1089678681,
          "median_link_load_bytes": 444004692.5,
          "average_link_load_bytes": 487581248,
          "fluid_cycles": 20296847,
          "hot_links": [
            {
              "src": 1,
              "dst": 0,
              "bytes": 1089678681
            },
            {
              "src": 3,
              "dst": 2,
              "bytes": 1088266925
            },
            {
              "src": 0,
              "dst": 7,
              "bytes": 1044350295
            },
            {
              "src": 2,
              "dst": 1,
              "bytes": 1041542489
            },
            {
              "src": 1,
              "dst": 2,
              "bytes": 999461550
            },
            {
              "src": 4,
              "dst": 3,
              "bytes": 992342018
            },
            {
              "src": 0,
              "dst": 1,
              "bytes": 978022403
            },
            {
              "src": 7,
              "dst": 6,
              "bytes": 958519980
            }
          ]
        },
        "random_regular_seed_0": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7278225806451613,
          "selected_path_count_max": 4,
          "average_hop_count": 2.911318553092182,
          "byte_weighted_average_hop_count": 2.559305555272941,
          "max_link_load_bytes": 663841451,
          "median_link_load_bytes": 403146411.0,
          "average_link_load_bytes": 402480896,
          "fluid_cycles": 12365010,
          "hot_links": [
            {
              "src": 12,
              "dst": 11,
              "bytes": 663841451
            },
            {
              "src": 19,
              "dst": 1,
              "bytes": 649020076
            },
            {
              "src": 11,
              "dst": 27,
              "bytes": 634856107
            },
            {
              "src": 11,
              "dst": 12,
              "bytes": 588379477
            },
            {
              "src": 1,
              "dst": 19,
              "bytes": 578238464
            },
            {
              "src": 16,
              "dst": 11,
              "bytes": 571405654
            },
            {
              "src": 22,
              "dst": 24,
              "bytes": 558513494
            },
            {
              "src": 8,
              "dst": 14,
              "bytes": 550987094
            }
          ]
        },
        "random_regular_seed_14": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6330645161290323,
          "selected_path_count_max": 4,
          "average_hop_count": 2.8987654320987652,
          "byte_weighted_average_hop_count": 2.57171594671203,
          "max_link_load_bytes": 658817023,
          "median_link_load_bytes": 400150868.0,
          "average_link_load_bytes": 404432576,
          "fluid_cycles": 12271423,
          "hot_links": [
            {
              "src": 29,
              "dst": 14,
              "bytes": 658817023
            },
            {
              "src": 1,
              "dst": 24,
              "bytes": 655564117
            },
            {
              "src": 24,
              "dst": 1,
              "bytes": 638031188
            },
            {
              "src": 14,
              "dst": 29,
              "bytes": 636482901
            },
            {
              "src": 29,
              "dst": 30,
              "bytes": 632431957
            },
            {
              "src": 3,
              "dst": 11,
              "bytes": 611860482
            },
            {
              "src": 7,
              "dst": 22,
              "bytes": 588499628
            },
            {
              "src": 8,
              "dst": 28,
              "bytes": 577363285
            }
          ]
        },
        "random_regular_seed_22": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.655241935483871,
          "selected_path_count_max": 4,
          "average_hop_count": 2.8270401948842876,
          "byte_weighted_average_hop_count": 2.5001165954139135,
          "max_link_load_bytes": 530889387,
          "median_link_load_bytes": 389011115.0,
          "average_link_load_bytes": 393172736,
          "fluid_cycles": 9888585,
          "hot_links": [
            {
              "src": 5,
              "dst": 10,
              "bytes": 530889387
            },
            {
              "src": 17,
              "dst": 6,
              "bytes": 523236011
            },
            {
              "src": 0,
              "dst": 4,
              "bytes": 516335616
            },
            {
              "src": 10,
              "dst": 5,
              "bytes": 515720533
            },
            {
              "src": 20,
              "dst": 21,
              "bytes": 514345643
            },
            {
              "src": 13,
              "dst": 31,
              "bytes": 509290496
            },
            {
              "src": 12,
              "dst": 15,
              "bytes": 508693163
            },
            {
              "src": 23,
              "dst": 0,
              "bytes": 491819690
            }
          ]
        },
        "random_regular_seed_16": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.532258064516129,
          "selected_path_count_max": 4,
          "average_hop_count": 2.763157894736842,
          "byte_weighted_average_hop_count": 2.470562811963951,
          "max_link_load_bytes": 549467477,
          "median_link_load_bytes": 388509355.5,
          "average_link_load_bytes": 388525056,
          "fluid_cycles": 10234629,
          "hot_links": [
            {
              "src": 0,
              "dst": 27,
              "bytes": 549467477
            },
            {
              "src": 22,
              "dst": 19,
              "bytes": 541543765
            },
            {
              "src": 12,
              "dst": 11,
              "bytes": 541442049
            },
            {
              "src": 19,
              "dst": 22,
              "bytes": 541356032
            },
            {
              "src": 16,
              "dst": 22,
              "bytes": 516208641
            },
            {
              "src": 0,
              "dst": 15,
              "bytes": 515392171
            },
            {
              "src": 12,
              "dst": 30,
              "bytes": 513785855
            },
            {
              "src": 24,
              "dst": 10,
              "bytes": 513538048
            }
          ]
        },
        "calibration_greedy": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6169354838709677,
          "selected_path_count_max": 4,
          "average_hop_count": 3.158354114713217,
          "byte_weighted_average_hop_count": 2.7562337849964287,
          "max_link_load_bytes": 1274859523,
          "median_link_load_bytes": 362618878.0,
          "average_link_load_bytes": 433450176,
          "fluid_cycles": 23746109,
          "hot_links": [
            {
              "src": 0,
              "dst": 31,
              "bytes": 1274859523
            },
            {
              "src": 25,
              "dst": 26,
              "bytes": 1080849744
            },
            {
              "src": 31,
              "dst": 0,
              "bytes": 1059800400
            },
            {
              "src": 26,
              "dst": 25,
              "bytes": 983164928
            },
            {
              "src": 7,
              "dst": 0,
              "bytes": 973771437
            },
            {
              "src": 25,
              "dst": 24,
              "bytes": 944642729
            },
            {
              "src": 24,
              "dst": 25,
              "bytes": 932193617
            },
            {
              "src": 0,
              "dst": 7,
              "bytes": 914130261
            }
          ]
        },
        "evaluation_greedy": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.5080645161290323,
          "selected_path_count_max": 4,
          "average_hop_count": 2.879679144385027,
          "byte_weighted_average_hop_count": 2.5655504300600476,
          "max_link_load_bytes": 672076459,
          "median_link_load_bytes": 402875050.0,
          "average_link_load_bytes": 403462976,
          "fluid_cycles": 12518399,
          "hot_links": [
            {
              "src": 25,
              "dst": 5,
              "bytes": 672076459
            },
            {
              "src": 7,
              "dst": 6,
              "bytes": 641718955
            },
            {
              "src": 27,
              "dst": 26,
              "bytes": 638155434
            },
            {
              "src": 19,
              "dst": 13,
              "bytes": 637089792
            },
            {
              "src": 26,
              "dst": 27,
              "bytes": 630627668
            },
            {
              "src": 23,
              "dst": 2,
              "bytes": 621347499
            },
            {
              "src": 4,
              "dst": 5,
              "bytes": 620733099
            },
            {
              "src": 1,
              "dst": 27,
              "bytes": 611061077
            }
          ]
        }
      }
    },
    "tiny_subchunk_audit": {
      "dispatch": {
        "son_torus": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 3330048,
          "subchunk_bytes_median": 5734400.0,
          "subchunks_total": 2688,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 3.0,
          "selected_path_count_mean": 2.7096774193548385,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_0": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 3330048,
          "subchunk_bytes_median": 9670656.0,
          "subchunks_total": 1714,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7278225806451613,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_14": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 3405824,
          "subchunk_bytes_median": 10256384.0,
          "subchunks_total": 1620,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6330645161290323,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_22": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 3540992,
          "subchunk_bytes_median": 10121216.0,
          "subchunks_total": 1642,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.655241935483871,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_16": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 3661824,
          "subchunk_bytes_median": 11167744.0,
          "subchunks_total": 1520,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.532258064516129,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "calibration_greedy": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 3497984,
          "subchunk_bytes_median": 10385408.0,
          "subchunks_total": 1604,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6169354838709677,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "evaluation_greedy": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 3997696,
          "subchunk_bytes_median": 11773952.0,
          "subchunks_total": 1496,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.5080645161290323,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        }
      },
      "combine": {
        "son_torus": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 3330048,
          "subchunk_bytes_median": 5734400.0,
          "subchunks_total": 2688,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 3.0,
          "selected_path_count_mean": 2.7096774193548385,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_0": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 3330048,
          "subchunk_bytes_median": 9670656.0,
          "subchunks_total": 1714,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7278225806451613,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_14": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 3405824,
          "subchunk_bytes_median": 10256384.0,
          "subchunks_total": 1620,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6330645161290323,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_22": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 3540992,
          "subchunk_bytes_median": 10121216.0,
          "subchunks_total": 1642,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.655241935483871,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_16": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 3661824,
          "subchunk_bytes_median": 11167744.0,
          "subchunks_total": 1520,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.532258064516129,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "calibration_greedy": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 3497984,
          "subchunk_bytes_median": 10385408.0,
          "subchunks_total": 1604,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6169354838709677,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "evaluation_greedy": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 3997696,
          "subchunk_bytes_median": 11773952.0,
          "subchunks_total": 1496,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.5080645161290323,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        }
      }
    },
    "candidate_family_interpretation": {
      "workload": "qwen_mmlu_machine_learning",
      "calibrated_candidate": "random_regular_seed_16",
      "calibrated_candidate_type": "random_regular",
      "oracle_candidate": "random_regular_seed_22",
      "oracle_candidate_type": "random_regular",
      "traffic_fingerprint": "broad / near-uniform",
      "torus_rank_by_eval_fluid": 34,
      "greedy_calibration_rank_by_eval_fluid": 35,
      "greedy_evaluation_rank_by_eval_fluid": 21,
      "fixed_seed0_rank_by_eval_fluid": 16,
      "calibrated_rank_by_eval_fluid": 2,
      "oracle_rank_by_eval_fluid": 1,
      "calibrated_beats_son": true,
      "calibrated_beats_fixed_random": true,
      "oracle_beats_calibrated": true,
      "calibrated_gain_vs_son_percent": 18.17294072847533,
      "calibrated_gain_vs_fixed_random_percent": 3.851400141537295,
      "oracle_gap_vs_calibrated_percent": 1.6745712236477324,
      "main_explanation": "workload-selected random-regular topology search",
      "median_random_name": "random_regular_seed_14",
      "best_random_name": null
    },
    "validation_pass": {
      "byte_conservation": true,
      "graph_budget": true,
      "graphs_valid": true,
      "native_runs": true,
      "no_tiny_subchunk_risk": true
    }
  },
  {
    "workload": {
      "id": "qwen_livecodebench_execution",
      "label": "Qwen LiveCodeBench execution",
      "path": "/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/livecodebench/execution"
    },
    "available": true,
    "trace_parse": {
      "files_found": 479,
      "files_used": 479,
      "moe_layer_count": 94,
      "inferred_num_experts": 128,
      "malformed_records": 0,
      "full": {
        "request_ids": [
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "7",
          "8",
          "9",
          "10",
          "11",
          "12",
          "13",
          "14",
          "15",
          "16",
          "17",
          "18",
          "19",
          "20",
          "21",
          "22",
          "23",
          "24",
          "25",
          "26",
          "27",
          "28",
          "29",
          "30",
          "31",
          "32",
          "33",
          "34",
          "35",
          "36",
          "37",
          "38",
          "39",
          "40",
          "41",
          "42",
          "43",
          "44",
          "45",
          "46",
          "47",
          "48",
          "49",
          "50",
          "51",
          "52",
          "53",
          "54",
          "55",
          "56",
          "57",
          "58",
          "59",
          "60",
          "61",
          "62",
          "63",
          "64",
          "65",
          "66",
          "67",
          "68",
          "69",
          "70",
          "71",
          "72",
          "73",
          "74",
          "75",
          "76",
          "77",
          "78",
          "79",
          "80",
          "81",
          "82",
          "83",
          "84",
          "85",
          "86",
          "87",
          "88",
          "89",
          "90",
          "91",
          "92",
          "93",
          "94",
          "95",
          "96",
          "97",
          "98",
          "99",
          "100",
          "101",
          "102",
          "103",
          "104",
          "105",
          "106",
          "107",
          "108",
          "109",
          "110",
          "111",
          "112",
          "113",
          "114",
          "115",
          "116",
          "117",
          "118",
          "119",
          "120",
          "121",
          "122",
          "123",
          "124",
          "125",
          "126",
          "127",
          "128",
          "129",
          "130",
          "131",
          "132",
          "133",
          "134",
          "135",
          "136",
          "137",
          "138",
          "139",
          "140",
          "141",
          "142",
          "143",
          "144",
          "145",
          "146",
          "147",
          "148",
          "149",
          "150",
          "151",
          "152",
          "153",
          "154",
          "155",
          "156",
          "157",
          "158",
          "159",
          "160",
          "161",
          "162",
          "163",
          "164",
          "165",
          "166",
          "167",
          "168",
          "169",
          "170",
          "171",
          "172",
          "173",
          "174",
          "175",
          "176",
          "177",
          "178",
          "179",
          "180",
          "181",
          "182",
          "183",
          "184",
          "185",
          "186",
          "187",
          "188",
          "189",
          "190",
          "191",
          "192",
          "193",
          "194",
          "195",
          "196",
          "197",
          "198",
          "199",
          "200",
          "201",
          "202",
          "203",
          "204",
          "205",
          "206",
          "207",
          "208",
          "209",
          "210",
          "211",
          "212",
          "213",
          "214",
          "215",
          "216",
          "217",
          "218",
          "219",
          "220",
          "221",
          "222",
          "223",
          "224",
          "225",
          "226",
          "227",
          "228",
          "229",
          "230",
          "231",
          "232",
          "233",
          "234",
          "235",
          "236",
          "237",
          "238",
          "239",
          "240",
          "241",
          "242",
          "243",
          "244",
          "245",
          "246",
          "247",
          "248",
          "249",
          "250",
          "251",
          "252",
          "253",
          "254",
          "255",
          "256",
          "257",
          "258",
          "259",
          "260",
          "261",
          "262",
          "263",
          "264",
          "265",
          "266",
          "267",
          "268",
          "269",
          "270",
          "271",
          "272",
          "273",
          "274",
          "275",
          "276",
          "277",
          "278",
          "279",
          "280",
          "281",
          "282",
          "283",
          "284",
          "285",
          "286",
          "287",
          "288",
          "289",
          "290",
          "291",
          "292",
          "293",
          "294",
          "295",
          "296",
          "297",
          "298",
          "299",
          "300",
          "301",
          "302",
          "303",
          "304",
          "305",
          "306",
          "307",
          "308",
          "309",
          "310",
          "311",
          "312",
          "313",
          "314",
          "315",
          "316",
          "317",
          "318",
          "319",
          "320",
          "321",
          "322",
          "323",
          "324",
          "325",
          "326",
          "327",
          "328",
          "329",
          "330",
          "331",
          "332",
          "333",
          "334",
          "335",
          "336",
          "337",
          "338",
          "339",
          "340",
          "341",
          "342",
          "343",
          "344",
          "345",
          "346",
          "347",
          "348",
          "349",
          "350",
          "351",
          "352",
          "353",
          "354",
          "355",
          "356",
          "357",
          "358",
          "359",
          "360",
          "361",
          "362",
          "363",
          "364",
          "365",
          "366",
          "367",
          "368",
          "369",
          "370",
          "371",
          "372",
          "373",
          "374",
          "375",
          "376",
          "377",
          "378",
          "379",
          "380",
          "381",
          "382",
          "383",
          "384",
          "385",
          "386",
          "387",
          "388",
          "389",
          "390",
          "391",
          "392",
          "393",
          "394",
          "395",
          "396",
          "397",
          "398",
          "399",
          "400",
          "401",
          "402",
          "403",
          "404",
          "405",
          "406",
          "407",
          "408",
          "409",
          "410",
          "411",
          "412",
          "413",
          "414",
          "415",
          "416",
          "417",
          "418",
          "419",
          "420",
          "421",
          "422",
          "423",
          "424",
          "425",
          "426",
          "427",
          "428",
          "429",
          "430",
          "431",
          "432",
          "433",
          "434",
          "435",
          "436",
          "437",
          "438",
          "439",
          "440",
          "441",
          "442",
          "443",
          "444",
          "445",
          "446",
          "447",
          "448",
          "449",
          "450",
          "451",
          "452",
          "453",
          "454",
          "455",
          "456",
          "457",
          "458",
          "459",
          "460",
          "461",
          "462",
          "463",
          "464",
          "465",
          "466",
          "467",
          "468",
          "469",
          "470",
          "471",
          "472",
          "473",
          "474",
          "475",
          "476",
          "477",
          "478"
        ],
        "request_count": 479,
        "prefill_input_tokens": 52341,
        "selected_expert_events": 39360432,
        "theoretical_dispatch_bytes": 322440658944,
        "theoretical_combine_bytes": 322440658944,
        "local_dispatch_bytes_excluded": 10098122752,
        "local_combine_bytes_excluded": 10098122752,
        "remote_dispatch_bytes_retained": 312342536192,
        "remote_combine_bytes_retained": 312342536192,
        "byte_conservation_pass": true,
        "dispatch_checksum": "e666c368c1e8dc83ed65b08324f275d29f0046215499bfdcb562eae65094cadc",
        "combine_checksum": "3b41f6d9b8a440276dc65d3b748abf068791f0ad5620f7e849c9493bac76bac3"
      },
      "calibration": {
        "request_ids": [
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "7",
          "8",
          "9",
          "10",
          "11",
          "12",
          "13",
          "14",
          "15",
          "16",
          "17",
          "18",
          "19",
          "20",
          "21",
          "22",
          "23",
          "24",
          "25",
          "26",
          "27",
          "28",
          "29",
          "30",
          "31",
          "32",
          "33",
          "34",
          "35",
          "36",
          "37",
          "38",
          "39",
          "40",
          "41",
          "42",
          "43",
          "44",
          "45",
          "46",
          "47"
        ],
        "request_count": 48,
        "prefill_input_tokens": 4478,
        "selected_expert_events": 3367456,
        "theoretical_dispatch_bytes": 27586199552,
        "theoretical_combine_bytes": 27586199552,
        "local_dispatch_bytes_excluded": 867598336,
        "local_combine_bytes_excluded": 867598336,
        "remote_dispatch_bytes_retained": 26718601216,
        "remote_combine_bytes_retained": 26718601216,
        "byte_conservation_pass": true,
        "dispatch_checksum": "4404aa6ce79700279e61d4fcb357044fc1032ec1d91e9560fc080ce7dd2bf66e",
        "combine_checksum": "b44b41122bfbcc199a1d31b63e2127f1dd35ed66c2e4085b6ff8f8afa8168ba3"
      },
      "evaluation": {
        "request_ids": [
          "48",
          "49",
          "50",
          "51",
          "52",
          "53",
          "54",
          "55",
          "56",
          "57",
          "58",
          "59",
          "60",
          "61",
          "62",
          "63",
          "64",
          "65",
          "66",
          "67",
          "68",
          "69",
          "70",
          "71",
          "72",
          "73",
          "74",
          "75",
          "76",
          "77",
          "78",
          "79",
          "80",
          "81",
          "82",
          "83",
          "84",
          "85",
          "86",
          "87",
          "88",
          "89",
          "90",
          "91",
          "92",
          "93",
          "94",
          "95",
          "96",
          "97",
          "98",
          "99",
          "100",
          "101",
          "102",
          "103",
          "104",
          "105",
          "106",
          "107",
          "108",
          "109",
          "110",
          "111",
          "112",
          "113",
          "114",
          "115",
          "116",
          "117",
          "118",
          "119",
          "120",
          "121",
          "122",
          "123",
          "124",
          "125",
          "126",
          "127",
          "128",
          "129",
          "130",
          "131",
          "132",
          "133",
          "134",
          "135",
          "136",
          "137",
          "138",
          "139",
          "140",
          "141",
          "142",
          "143",
          "144",
          "145",
          "146",
          "147",
          "148",
          "149",
          "150",
          "151",
          "152",
          "153",
          "154",
          "155",
          "156",
          "157",
          "158",
          "159",
          "160",
          "161",
          "162",
          "163",
          "164",
          "165",
          "166",
          "167",
          "168",
          "169",
          "170",
          "171",
          "172",
          "173",
          "174",
          "175",
          "176",
          "177",
          "178",
          "179",
          "180",
          "181",
          "182",
          "183",
          "184",
          "185",
          "186",
          "187",
          "188",
          "189",
          "190",
          "191",
          "192",
          "193",
          "194",
          "195",
          "196",
          "197",
          "198",
          "199",
          "200",
          "201",
          "202",
          "203",
          "204",
          "205",
          "206",
          "207",
          "208",
          "209",
          "210",
          "211",
          "212",
          "213",
          "214",
          "215",
          "216",
          "217",
          "218",
          "219",
          "220",
          "221",
          "222",
          "223",
          "224",
          "225",
          "226",
          "227",
          "228",
          "229",
          "230",
          "231",
          "232",
          "233",
          "234",
          "235",
          "236",
          "237",
          "238",
          "239",
          "240",
          "241",
          "242",
          "243",
          "244",
          "245",
          "246",
          "247",
          "248",
          "249",
          "250",
          "251",
          "252",
          "253",
          "254",
          "255",
          "256",
          "257",
          "258",
          "259",
          "260",
          "261",
          "262",
          "263",
          "264",
          "265",
          "266",
          "267",
          "268",
          "269",
          "270",
          "271",
          "272",
          "273",
          "274",
          "275",
          "276",
          "277",
          "278",
          "279",
          "280",
          "281",
          "282",
          "283",
          "284",
          "285",
          "286",
          "287",
          "288",
          "289",
          "290",
          "291",
          "292",
          "293",
          "294",
          "295",
          "296",
          "297",
          "298",
          "299",
          "300",
          "301",
          "302",
          "303",
          "304",
          "305",
          "306",
          "307",
          "308",
          "309",
          "310",
          "311",
          "312",
          "313",
          "314",
          "315",
          "316",
          "317",
          "318",
          "319",
          "320",
          "321",
          "322",
          "323",
          "324",
          "325",
          "326",
          "327",
          "328",
          "329",
          "330",
          "331",
          "332",
          "333",
          "334",
          "335",
          "336",
          "337",
          "338",
          "339",
          "340",
          "341",
          "342",
          "343",
          "344",
          "345",
          "346",
          "347",
          "348",
          "349",
          "350",
          "351",
          "352",
          "353",
          "354",
          "355",
          "356",
          "357",
          "358",
          "359",
          "360",
          "361",
          "362",
          "363",
          "364",
          "365",
          "366",
          "367",
          "368",
          "369",
          "370",
          "371",
          "372",
          "373",
          "374",
          "375",
          "376",
          "377",
          "378",
          "379",
          "380",
          "381",
          "382",
          "383",
          "384",
          "385",
          "386",
          "387",
          "388",
          "389",
          "390",
          "391",
          "392",
          "393",
          "394",
          "395",
          "396",
          "397",
          "398",
          "399",
          "400",
          "401",
          "402",
          "403",
          "404",
          "405",
          "406",
          "407",
          "408",
          "409",
          "410",
          "411",
          "412",
          "413",
          "414",
          "415",
          "416",
          "417",
          "418",
          "419",
          "420",
          "421",
          "422",
          "423",
          "424",
          "425",
          "426",
          "427",
          "428",
          "429",
          "430",
          "431",
          "432",
          "433",
          "434",
          "435",
          "436",
          "437",
          "438",
          "439",
          "440",
          "441",
          "442",
          "443",
          "444",
          "445",
          "446",
          "447",
          "448",
          "449",
          "450",
          "451",
          "452",
          "453",
          "454",
          "455",
          "456",
          "457",
          "458",
          "459",
          "460",
          "461",
          "462",
          "463",
          "464",
          "465",
          "466",
          "467",
          "468",
          "469",
          "470",
          "471",
          "472",
          "473",
          "474",
          "475",
          "476",
          "477",
          "478"
        ],
        "request_count": 431,
        "prefill_input_tokens": 47863,
        "selected_expert_events": 35992976,
        "theoretical_dispatch_bytes": 294854459392,
        "theoretical_combine_bytes": 294854459392,
        "local_dispatch_bytes_excluded": 9230524416,
        "local_combine_bytes_excluded": 9230524416,
        "remote_dispatch_bytes_retained": 285623934976,
        "remote_combine_bytes_retained": 285623934976,
        "byte_conservation_pass": true,
        "dispatch_checksum": "6785561e2cc52bc6a29cfabba53da5ffec80d6a93391d3253e40591dd349da72",
        "combine_checksum": "efa4c1335c967bc5fc223ea4ecb3e8f71d067a31149e25717268c23c3991a19b"
      }
    },
    "split": {
      "calibration_request_count": 48,
      "evaluation_request_count": 431,
      "calibration_request_ids": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
        "40",
        "41",
        "42",
        "43",
        "44",
        "45",
        "46",
        "47"
      ],
      "evaluation_request_ids": [
        "48",
        "49",
        "50",
        "51",
        "52",
        "53",
        "54",
        "55",
        "56",
        "57",
        "58",
        "59",
        "60",
        "61",
        "62",
        "63",
        "64",
        "65",
        "66",
        "67",
        "68",
        "69",
        "70",
        "71",
        "72",
        "73",
        "74",
        "75",
        "76",
        "77",
        "78",
        "79",
        "80",
        "81",
        "82",
        "83",
        "84",
        "85",
        "86",
        "87",
        "88",
        "89",
        "90",
        "91",
        "92",
        "93",
        "94",
        "95",
        "96",
        "97",
        "98",
        "99",
        "100",
        "101",
        "102",
        "103",
        "104",
        "105",
        "106",
        "107",
        "108",
        "109",
        "110",
        "111",
        "112",
        "113",
        "114",
        "115",
        "116",
        "117",
        "118",
        "119",
        "120",
        "121",
        "122",
        "123",
        "124",
        "125",
        "126",
        "127",
        "128",
        "129",
        "130",
        "131",
        "132",
        "133",
        "134",
        "135",
        "136",
        "137",
        "138",
        "139",
        "140",
        "141",
        "142",
        "143",
        "144",
        "145",
        "146",
        "147",
        "148",
        "149",
        "150",
        "151",
        "152",
        "153",
        "154",
        "155",
        "156",
        "157",
        "158",
        "159",
        "160",
        "161",
        "162",
        "163",
        "164",
        "165",
        "166",
        "167",
        "168",
        "169",
        "170",
        "171",
        "172",
        "173",
        "174",
        "175",
        "176",
        "177",
        "178",
        "179",
        "180",
        "181",
        "182",
        "183",
        "184",
        "185",
        "186",
        "187",
        "188",
        "189",
        "190",
        "191",
        "192",
        "193",
        "194",
        "195",
        "196",
        "197",
        "198",
        "199",
        "200",
        "201",
        "202",
        "203",
        "204",
        "205",
        "206",
        "207",
        "208",
        "209",
        "210",
        "211",
        "212",
        "213",
        "214",
        "215",
        "216",
        "217",
        "218",
        "219",
        "220",
        "221",
        "222",
        "223",
        "224",
        "225",
        "226",
        "227",
        "228",
        "229",
        "230",
        "231",
        "232",
        "233",
        "234",
        "235",
        "236",
        "237",
        "238",
        "239",
        "240",
        "241",
        "242",
        "243",
        "244",
        "245",
        "246",
        "247",
        "248",
        "249",
        "250",
        "251",
        "252",
        "253",
        "254",
        "255",
        "256",
        "257",
        "258",
        "259",
        "260",
        "261",
        "262",
        "263",
        "264",
        "265",
        "266",
        "267",
        "268",
        "269",
        "270",
        "271",
        "272",
        "273",
        "274",
        "275",
        "276",
        "277",
        "278",
        "279",
        "280",
        "281",
        "282",
        "283",
        "284",
        "285",
        "286",
        "287",
        "288",
        "289",
        "290",
        "291",
        "292",
        "293",
        "294",
        "295",
        "296",
        "297",
        "298",
        "299",
        "300",
        "301",
        "302",
        "303",
        "304",
        "305",
        "306",
        "307",
        "308",
        "309",
        "310",
        "311",
        "312",
        "313",
        "314",
        "315",
        "316",
        "317",
        "318",
        "319",
        "320",
        "321",
        "322",
        "323",
        "324",
        "325",
        "326",
        "327",
        "328",
        "329",
        "330",
        "331",
        "332",
        "333",
        "334",
        "335",
        "336",
        "337",
        "338",
        "339",
        "340",
        "341",
        "342",
        "343",
        "344",
        "345",
        "346",
        "347",
        "348",
        "349",
        "350",
        "351",
        "352",
        "353",
        "354",
        "355",
        "356",
        "357",
        "358",
        "359",
        "360",
        "361",
        "362",
        "363",
        "364",
        "365",
        "366",
        "367",
        "368",
        "369",
        "370",
        "371",
        "372",
        "373",
        "374",
        "375",
        "376",
        "377",
        "378",
        "379",
        "380",
        "381",
        "382",
        "383",
        "384",
        "385",
        "386",
        "387",
        "388",
        "389",
        "390",
        "391",
        "392",
        "393",
        "394",
        "395",
        "396",
        "397",
        "398",
        "399",
        "400",
        "401",
        "402",
        "403",
        "404",
        "405",
        "406",
        "407",
        "408",
        "409",
        "410",
        "411",
        "412",
        "413",
        "414",
        "415",
        "416",
        "417",
        "418",
        "419",
        "420",
        "421",
        "422",
        "423",
        "424",
        "425",
        "426",
        "427",
        "428",
        "429",
        "430",
        "431",
        "432",
        "433",
        "434",
        "435",
        "436",
        "437",
        "438",
        "439",
        "440",
        "441",
        "442",
        "443",
        "444",
        "445",
        "446",
        "447",
        "448",
        "449",
        "450",
        "451",
        "452",
        "453",
        "454",
        "455",
        "456",
        "457",
        "458",
        "459",
        "460",
        "461",
        "462",
        "463",
        "464",
        "465",
        "466",
        "467",
        "468",
        "469",
        "470",
        "471",
        "472",
        "473",
        "474",
        "475",
        "476",
        "477",
        "478"
      ],
      "calibration_rule": "front ceil(10%) requests"
    },
    "anti_leakage": {
      "calibrated_uses_calibration_only": true,
      "oracle_uses_evaluation_only": true,
      "oracle_reference_only": true
    },
    "traffic_fingerprint": {
      "calibration_dispatch": {
        "total_remote_bytes": 26718601216,
        "nonzero_gpu_pairs": 992,
        "top1_share": 0.0015845236679024807,
        "top4_share": 0.0062022695971360835,
        "top8_share": 0.0122567566075986,
        "top16_share": 0.024114009666575502,
        "gini": 0.08607341316098585,
        "entropy_bits": 9.936364135077891,
        "message_bytes_min": 14712832,
        "message_bytes_median": 26927104.0,
        "message_bytes_mean": 26934073.80645161,
        "message_bytes_max": 42336256,
        "interpretation": "broad / near-uniform"
      },
      "calibration_combine": {
        "total_remote_bytes": 26718601216,
        "nonzero_gpu_pairs": 992,
        "top1_share": 0.0015845236679024807,
        "top4_share": 0.0062022695971360835,
        "top8_share": 0.0122567566075986,
        "top16_share": 0.024114009666575502,
        "gini": 0.08607341316098585,
        "entropy_bits": 9.936364135077888,
        "message_bytes_min": 14712832,
        "message_bytes_median": 26927104.0,
        "message_bytes_mean": 26934073.80645161,
        "message_bytes_max": 42336256,
        "interpretation": "broad / near-uniform"
      },
      "evaluation_dispatch": {
        "total_remote_bytes": 285623934976,
        "nonzero_gpu_pairs": 992,
        "top1_share": 0.0014586044829716616,
        "top4_share": 0.005825469438126085,
        "top8_share": 0.011625240637760298,
        "top16_share": 0.023144648128160097,
        "gini": 0.0809597951408056,
        "entropy_bits": 9.938458417268459,
        "message_bytes_min": 187252736,
        "message_bytes_median": 288866304.0,
        "message_bytes_mean": 287927353.8064516,
        "message_bytes_max": 416612352,
        "interpretation": "broad / near-uniform"
      },
      "evaluation_combine": {
        "total_remote_bytes": 285623934976,
        "nonzero_gpu_pairs": 992,
        "top1_share": 0.0014586044829716616,
        "top4_share": 0.005825469438126085,
        "top8_share": 0.011625240637760298,
        "top16_share": 0.023144648128160097,
        "gini": 0.0809597951408056,
        "entropy_bits": 9.938458417268452,
        "message_bytes_min": 187252736,
        "message_bytes_median": 288866304.0,
        "message_bytes_mean": 287927353.8064516,
        "message_bytes_max": 416612352,
        "interpretation": "broad / near-uniform"
      }
    },
    "candidate_pool": {
      "candidate_count": 35,
      "random_candidate_count": 32,
      "candidate_types": {
        "greedy_calibration": 1,
        "greedy_evaluation": 1,
        "random_regular": 32,
        "torus": 1
      },
      "graph_budget_pass": true,
      "all_candidate_graphs_valid": true
    },
    "candidate_scores": {
      "calibration_top12": [
        {
          "name": "random_regular_seed_22",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 766321322
        },
        {
          "name": "random_regular_seed_16",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 776056148
        },
        {
          "name": "random_regular_seed_3",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 781292204
        },
        {
          "name": "random_regular_seed_13",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 785333591
        },
        {
          "name": "random_regular_seed_18",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 791670102
        },
        {
          "name": "random_regular_seed_30",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 793621163
        },
        {
          "name": "random_regular_seed_20",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 807032150
        },
        {
          "name": "random_regular_seed_19",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 807783082
        },
        {
          "name": "random_regular_seed_25",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 810772480
        },
        {
          "name": "random_regular_seed_27",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 816141652
        },
        {
          "name": "random_regular_seed_31",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 824916651
        },
        {
          "name": "random_regular_seed_9",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 829990912
        }
      ],
      "evaluation_top12": [
        {
          "name": "random_regular_seed_22",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 7912759978
        },
        {
          "name": "random_regular_seed_30",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8303930026
        },
        {
          "name": "random_regular_seed_3",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8407738369
        },
        {
          "name": "random_regular_seed_19",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8410722304
        },
        {
          "name": "random_regular_seed_18",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8411119617
        },
        {
          "name": "random_regular_seed_31",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8413817514
        },
        {
          "name": "random_regular_seed_13",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8455315456
        },
        {
          "name": "random_regular_seed_20",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8499765251
        },
        {
          "name": "random_regular_seed_16",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8534774442
        },
        {
          "name": "random_regular_seed_25",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8585405782
        },
        {
          "name": "random_regular_seed_27",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8675629055
        },
        {
          "name": "random_regular_seed_9",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8914812928
        }
      ],
      "evaluation_all": [
        {
          "name": "random_regular_seed_22",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 7912759978
        },
        {
          "name": "random_regular_seed_30",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8303930026
        },
        {
          "name": "random_regular_seed_3",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8407738369
        },
        {
          "name": "random_regular_seed_19",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8410722304
        },
        {
          "name": "random_regular_seed_18",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8411119617
        },
        {
          "name": "random_regular_seed_31",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8413817514
        },
        {
          "name": "random_regular_seed_13",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8455315456
        },
        {
          "name": "random_regular_seed_20",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8499765251
        },
        {
          "name": "random_regular_seed_16",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8534774442
        },
        {
          "name": "random_regular_seed_25",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8585405782
        },
        {
          "name": "random_regular_seed_27",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8675629055
        },
        {
          "name": "random_regular_seed_9",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 8914812928
        },
        {
          "name": "random_regular_seed_17",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 9062804822
        },
        {
          "name": "random_regular_seed_2",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 9087383552
        },
        {
          "name": "random_regular_seed_0",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 9146754390
        },
        {
          "name": "random_regular_seed_29",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 9245788160
        },
        {
          "name": "random_regular_seed_12",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 9274816512
        },
        {
          "name": "random_regular_seed_24",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 9316932950
        },
        {
          "name": "random_regular_seed_11",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 9423623510
        },
        {
          "name": "random_regular_seed_23",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 9658855423
        },
        {
          "name": "random_regular_seed_4",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 9700845568
        },
        {
          "name": "random_regular_seed_8",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 9702531754
        },
        {
          "name": "random_regular_seed_7",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 9858377729
        },
        {
          "name": "random_regular_seed_14",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 9885915135
        },
        {
          "name": "random_regular_seed_1",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 10247073791
        },
        {
          "name": "random_regular_seed_5",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 10288056323
        },
        {
          "name": "random_regular_seed_10",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 10383949140
        },
        {
          "name": "random_regular_seed_6",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 10592658775
        },
        {
          "name": "random_regular_seed_21",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 10659614037
        },
        {
          "name": "random_regular_seed_15",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 11046027945
        },
        {
          "name": "calibration_greedy",
          "candidate_type": "greedy_calibration",
          "score_max_link_load_bytes": 11289929729
        },
        {
          "name": "random_regular_seed_26",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 11920238591
        },
        {
          "name": "evaluation_greedy",
          "candidate_type": "greedy_evaluation",
          "score_max_link_load_bytes": 12325253121
        },
        {
          "name": "random_regular_seed_28",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 14033988950
        },
        {
          "name": "son_torus",
          "candidate_type": "torus",
          "score_max_link_load_bytes": 16651745966
        }
      ]
    },
    "selected": {
      "ron_calibrated": "random_regular_seed_22",
      "ron_calibrated_type": "random_regular",
      "ron_oracle": "random_regular_seed_22",
      "ron_oracle_type": "random_regular",
      "fixed_random": "random_regular_seed_0",
      "median_random": "random_regular_seed_12",
      "best_random": "random_regular_seed_22",
      "representatives_run_in_astra": [
        "son_torus",
        "random_regular_seed_0",
        "random_regular_seed_12",
        "random_regular_seed_22",
        "calibration_greedy",
        "evaluation_greedy"
      ],
      "roles": {
        "son_torus": "son",
        "random_regular_seed_0": "fixed_random",
        "random_regular_seed_12": "median_random",
        "random_regular_seed_22": "ron_oracle",
        "calibration_greedy": "greedy_calibration",
        "evaluation_greedy": "greedy_evaluation"
      }
    },
    "candidate_audits_all_budget_only": {
      "son_torus": {
        "name": "son_torus_4x8_32gpu",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "calibration_greedy": {
        "name": "ron_calibration_greedy_degree4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "evaluation_greedy": {
        "name": "ron_evaluation_greedy_degree4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_0": {
        "name": "ron_random_regular_seed_0",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_1": {
        "name": "ron_random_regular_seed_1",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_2": {
        "name": "ron_random_regular_seed_2",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_3": {
        "name": "ron_random_regular_seed_3",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_4": {
        "name": "ron_random_regular_seed_4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_5": {
        "name": "ron_random_regular_seed_5",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_6": {
        "name": "ron_random_regular_seed_6",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_7": {
        "name": "ron_random_regular_seed_7",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_8": {
        "name": "ron_random_regular_seed_8",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_9": {
        "name": "ron_random_regular_seed_9",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_10": {
        "name": "ron_random_regular_seed_10",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_11": {
        "name": "ron_random_regular_seed_11",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_12": {
        "name": "ron_random_regular_seed_12",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_13": {
        "name": "ron_random_regular_seed_13",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_14": {
        "name": "ron_random_regular_seed_14",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_15": {
        "name": "ron_random_regular_seed_15",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_16": {
        "name": "ron_random_regular_seed_16",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_17": {
        "name": "ron_random_regular_seed_17",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_18": {
        "name": "ron_random_regular_seed_18",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_19": {
        "name": "ron_random_regular_seed_19",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_20": {
        "name": "ron_random_regular_seed_20",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_21": {
        "name": "ron_random_regular_seed_21",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_22": {
        "name": "ron_random_regular_seed_22",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_23": {
        "name": "ron_random_regular_seed_23",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_24": {
        "name": "ron_random_regular_seed_24",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_25": {
        "name": "ron_random_regular_seed_25",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_26": {
        "name": "ron_random_regular_seed_26",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_27": {
        "name": "ron_random_regular_seed_27",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_28": {
        "name": "ron_random_regular_seed_28",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_29": {
        "name": "ron_random_regular_seed_29",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_30": {
        "name": "ron_random_regular_seed_30",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_31": {
        "name": "ron_random_regular_seed_31",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      }
    },
    "candidate_audits_representatives": {
      "son_torus": {
        "name": "son_torus_4x8_32gpu",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 3.096774193548387,
        "diameter": 6,
        "ecmp_path_count_distribution_cap4": {
          "1": 256,
          "2": 192,
          "3": 128,
          "4": 416
        }
      },
      "random_regular_seed_0": {
        "name": "ron_random_regular_seed_0",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.560483870967742,
        "diameter": 4,
        "ecmp_path_count_distribution_cap4": {
          "1": 590,
          "2": 200,
          "3": 84,
          "4": 118
        }
      },
      "random_regular_seed_12": {
        "name": "ron_random_regular_seed_12",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.556451612903226,
        "diameter": 4,
        "ecmp_path_count_distribution_cap4": {
          "1": 616,
          "2": 204,
          "3": 66,
          "4": 106
        }
      },
      "random_regular_seed_22": {
        "name": "ron_random_regular_seed_22",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.5,
        "diameter": 4,
        "ecmp_path_count_distribution_cap4": {
          "1": 600,
          "2": 230,
          "3": 66,
          "4": 96
        }
      },
      "calibration_greedy": {
        "name": "ron_calibration_greedy_degree4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.625,
        "diameter": 5,
        "ecmp_path_count_distribution_cap4": {
          "1": 600,
          "2": 196,
          "3": 94,
          "4": 102
        }
      },
      "evaluation_greedy": {
        "name": "ron_evaluation_greedy_degree4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.596774193548387,
        "diameter": 5,
        "ecmp_path_count_distribution_cap4": {
          "1": 620,
          "2": 210,
          "3": 66,
          "4": 96
        }
      }
    },
    "native_astra_results": {
      "dispatch": {
        "son_torus": {
          "label": "qwen_livecodebench_execution_dispatch_son_torus",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.038851041928865016,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/network_configs/son_torus.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_dispatch_son_torus.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_dispatch_son_torus.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 613192078,
          "cycles_count": 32
        },
        "random_regular_seed_0": {
          "label": "qwen_livecodebench_execution_dispatch_random_regular_seed_0",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.025646665948443115,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/network_configs/random_regular_seed_0.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_dispatch_random_regular_seed_0.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_dispatch_random_regular_seed_0.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 517304558,
          "cycles_count": 32
        },
        "random_regular_seed_12": {
          "label": "qwen_livecodebench_execution_dispatch_random_regular_seed_12",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.0261000000173226,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/network_configs/random_regular_seed_12.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_dispatch_random_regular_seed_12.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_dispatch_random_regular_seed_12.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 489497686,
          "cycles_count": 32
        },
        "random_regular_seed_22": {
          "label": "qwen_livecodebench_execution_dispatch_random_regular_seed_22",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.025641791988164186,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/network_configs/random_regular_seed_22.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_dispatch_random_regular_seed_22.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_dispatch_random_regular_seed_22.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 492012015,
          "cycles_count": 32
        },
        "calibration_greedy": {
          "label": "qwen_livecodebench_execution_dispatch_calibration_greedy",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.026406291988678277,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/network_configs/calibration_greedy.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_dispatch_calibration_greedy.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_dispatch_calibration_greedy.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 575767951,
          "cycles_count": 32
        },
        "evaluation_greedy": {
          "label": "qwen_livecodebench_execution_dispatch_evaluation_greedy",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.025742041994817555,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/network_configs/evaluation_greedy.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_dispatch_evaluation_greedy.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_dispatch_evaluation_greedy.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 568566873,
          "cycles_count": 32
        }
      },
      "combine": {
        "son_torus": {
          "label": "qwen_livecodebench_execution_combine_son_torus",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.027926458977162838,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/network_configs/son_torus.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_combine_son_torus.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_combine_son_torus.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 618100747,
          "cycles_count": 32
        },
        "random_regular_seed_0": {
          "label": "qwen_livecodebench_execution_combine_random_regular_seed_0",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.025468082982115448,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/network_configs/random_regular_seed_0.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_combine_random_regular_seed_0.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_combine_random_regular_seed_0.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 547447257,
          "cycles_count": 32
        },
        "random_regular_seed_12": {
          "label": "qwen_livecodebench_execution_combine_random_regular_seed_12",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.02571299998089671,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/network_configs/random_regular_seed_12.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_combine_random_regular_seed_12.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_combine_random_regular_seed_12.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 531443627,
          "cycles_count": 32
        },
        "random_regular_seed_22": {
          "label": "qwen_livecodebench_execution_combine_random_regular_seed_22",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.025614708079956472,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/network_configs/random_regular_seed_22.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_combine_random_regular_seed_22.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_combine_random_regular_seed_22.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 520991644,
          "cycles_count": 32
        },
        "calibration_greedy": {
          "label": "qwen_livecodebench_execution_combine_calibration_greedy",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.025670542032457888,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/network_configs/calibration_greedy.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_combine_calibration_greedy.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_combine_calibration_greedy.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 611330646,
          "cycles_count": 32
        },
        "evaluation_greedy": {
          "label": "qwen_livecodebench_execution_combine_evaluation_greedy",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.025498208007775247,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_livecodebench_execution/network_configs/evaluation_greedy.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_combine_evaluation_greedy.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_livecodebench_execution_combine_evaluation_greedy.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 602203079,
          "cycles_count": 32
        }
      }
    },
    "native_astra_totals": {
      "son_torus": {
        "candidate_type": "torus",
        "dispatch_cycles": 613192078,
        "combine_cycles": 618100747,
        "total_cycles": 1231292825,
        "dispatch_fluid_cycles": 310162938,
        "combine_fluid_cycles": 281167589,
        "total_fluid_cycles": 591330527,
        "astra_over_fluid_total": 2.0822412657210916,
        "success": true,
        "runtime_s": 0.06677750090602785,
        "role": "son"
      },
      "random_regular_seed_0": {
        "candidate_type": "random_regular",
        "dispatch_cycles": 517304558,
        "combine_cycles": 547447257,
        "total_cycles": 1064751815,
        "dispatch_fluid_cycles": 168109893,
        "combine_fluid_cycles": 170371576,
        "total_fluid_cycles": 338481469,
        "astra_over_fluid_total": 3.1456724001632126,
        "success": true,
        "runtime_s": 0.05111474893055856,
        "role": "fixed_random"
      },
      "random_regular_seed_12": {
        "candidate_type": "random_regular",
        "dispatch_cycles": 489497686,
        "combine_cycles": 531443627,
        "total_cycles": 1020941313,
        "dispatch_fluid_cycles": 172756919,
        "combine_fluid_cycles": 169162712,
        "total_fluid_cycles": 341919631,
        "astra_over_fluid_total": 2.9859101977095897,
        "success": true,
        "runtime_s": 0.05181299999821931,
        "role": "median_random"
      },
      "random_regular_seed_22": {
        "candidate_type": "random_regular",
        "dispatch_cycles": 492012015,
        "combine_cycles": 520991644,
        "total_cycles": 1013003659,
        "dispatch_fluid_cycles": 144216473,
        "combine_fluid_cycles": 147386639,
        "total_fluid_cycles": 291603112,
        "astra_over_fluid_total": 3.473912373747232,
        "success": true,
        "runtime_s": 0.05125650006812066,
        "role": "ron_oracle"
      },
      "calibration_greedy": {
        "candidate_type": "greedy_calibration",
        "dispatch_cycles": 575767951,
        "combine_cycles": 611330646,
        "total_cycles": 1187098597,
        "dispatch_fluid_cycles": 210291328,
        "combine_fluid_cycles": 210291328,
        "total_fluid_cycles": 420582656,
        "astra_over_fluid_total": 2.8225096305445367,
        "success": true,
        "runtime_s": 0.052076834021136165,
        "role": "greedy_calibration"
      },
      "evaluation_greedy": {
        "candidate_type": "greedy_evaluation",
        "dispatch_cycles": 568566873,
        "combine_cycles": 602203079,
        "total_cycles": 1170769952,
        "dispatch_fluid_cycles": 229575729,
        "combine_fluid_cycles": 226808738,
        "total_fluid_cycles": 456384467,
        "astra_over_fluid_total": 2.5653150723904896,
        "success": true,
        "runtime_s": 0.0512402500025928,
        "role": "greedy_evaluation"
      }
    },
    "fluid_lower_bound": {
      "dispatch": {
        "son_torus": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 3.0,
          "selected_path_count_mean": 2.7096774193548385,
          "selected_path_count_max": 4,
          "average_hop_count": 3.5952380952380953,
          "byte_weighted_average_hop_count": 3.097258482662996,
          "max_link_load_bytes": 16651745966,
          "median_link_load_bytes": 6429947562.0,
          "average_link_load_bytes": 6911337152,
          "fluid_cycles": 310162938,
          "hot_links": [
            {
              "src": 1,
              "dst": 0,
              "bytes": 16651745966
            },
            {
              "src": 2,
              "dst": 1,
              "bytes": 16309566126
            },
            {
              "src": 3,
              "dst": 2,
              "bytes": 15267551918
            },
            {
              "src": 4,
              "dst": 3,
              "bytes": 13835534338
            },
            {
              "src": 0,
              "dst": 7,
              "bytes": 13583994882
            },
            {
              "src": 0,
              "dst": 1,
              "bytes": 12687117657
            },
            {
              "src": 5,
              "dst": 4,
              "bytes": 12674140846
            },
            {
              "src": 7,
              "dst": 0,
              "bytes": 12667392686
            }
          ]
        },
        "random_regular_seed_0": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7278225806451613,
          "selected_path_count_max": 4,
          "average_hop_count": 2.911318553092182,
          "byte_weighted_average_hop_count": 2.5575668793071618,
          "max_link_load_bytes": 9025331201,
          "median_link_load_bytes": 5665627476.5,
          "average_link_load_bytes": 5707049344,
          "fluid_cycles": 168109893,
          "hot_links": [
            {
              "src": 19,
              "dst": 1,
              "bytes": 9025331201
            },
            {
              "src": 27,
              "dst": 11,
              "bytes": 8964857172
            },
            {
              "src": 12,
              "dst": 11,
              "bytes": 8952169130
            },
            {
              "src": 22,
              "dst": 24,
              "bytes": 8668686337
            },
            {
              "src": 11,
              "dst": 27,
              "bytes": 8508317013
            },
            {
              "src": 24,
              "dst": 22,
              "bytes": 8440654507
            },
            {
              "src": 11,
              "dst": 12,
              "bytes": 8396829354
            },
            {
              "src": 22,
              "dst": 0,
              "bytes": 7808557056
            }
          ]
        },
        "random_regular_seed_12": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.659274193548387,
          "selected_path_count_max": 4,
          "average_hop_count": 2.8979343863912517,
          "byte_weighted_average_hop_count": 2.5588398312256713,
          "max_link_load_bytes": 9274816512,
          "median_link_load_bytes": 5738111316.0,
          "average_link_load_bytes": 5709889856,
          "fluid_cycles": 172756919,
          "hot_links": [
            {
              "src": 3,
              "dst": 30,
              "bytes": 9274816512
            },
            {
              "src": 20,
              "dst": 0,
              "bytes": 9217318228
            },
            {
              "src": 14,
              "dst": 0,
              "bytes": 8554130092
            },
            {
              "src": 30,
              "dst": 3,
              "bytes": 8540657664
            },
            {
              "src": 30,
              "dst": 15,
              "bytes": 8324284416
            },
            {
              "src": 0,
              "dst": 20,
              "bytes": 8305815554
            },
            {
              "src": 12,
              "dst": 7,
              "bytes": 8175338841
            },
            {
              "src": 17,
              "dst": 7,
              "bytes": 8016115030
            }
          ]
        },
        "random_regular_seed_22": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.655241935483871,
          "selected_path_count_max": 4,
          "average_hop_count": 2.8270401948842876,
          "byte_weighted_average_hop_count": 2.4978446319491687,
          "max_link_load_bytes": 7742562986,
          "median_link_load_bytes": 5452514987.0,
          "average_link_load_bytes": 5573782912,
          "fluid_cycles": 144216473,
          "hot_links": [
            {
              "src": 21,
              "dst": 20,
              "bytes": 7742562986
            },
            {
              "src": 23,
              "dst": 0,
              "bytes": 7728256341
            },
            {
              "src": 4,
              "dst": 0,
              "bytes": 7630581760
            },
            {
              "src": 18,
              "dst": 1,
              "bytes": 7512014848
            },
            {
              "src": 20,
              "dst": 21,
              "bytes": 7453405184
            },
            {
              "src": 14,
              "dst": 20,
              "bytes": 7390672896
            },
            {
              "src": 5,
              "dst": 10,
              "bytes": 7360295595
            },
            {
              "src": 31,
              "dst": 13,
              "bytes": 7334775466
            }
          ]
        },
        "calibration_greedy": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6955645161290323,
          "selected_path_count_max": 4,
          "average_hop_count": 2.9785969084423307,
          "byte_weighted_average_hop_count": 2.6293664096431724,
          "max_link_load_bytes": 11289929727,
          "median_link_load_bytes": 5694251349.0,
          "average_link_load_bytes": 5867265472,
          "fluid_cycles": 210291328,
          "hot_links": [
            {
              "src": 19,
              "dst": 20,
              "bytes": 11289929727
            },
            {
              "src": 0,
              "dst": 20,
              "bytes": 9923609940
            },
            {
              "src": 20,
              "dst": 0,
              "bytes": 9670227968
            },
            {
              "src": 14,
              "dst": 13,
              "bytes": 9513989461
            },
            {
              "src": 17,
              "dst": 16,
              "bytes": 9173357911
            },
            {
              "src": 31,
              "dst": 0,
              "bytes": 8903053311
            },
            {
              "src": 8,
              "dst": 7,
              "bytes": 8794523649
            },
            {
              "src": 2,
              "dst": 1,
              "bytes": 8781950293
            }
          ]
        },
        "evaluation_greedy": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6350806451612903,
          "selected_path_count_max": 4,
          "average_hop_count": 2.93711467324291,
          "byte_weighted_average_hop_count": 2.596446851410806,
          "max_link_load_bytes": 12325253121,
          "median_link_load_bytes": 5617422677.5,
          "average_link_load_bytes": 5793807552,
          "fluid_cycles": 229575729,
          "hot_links": [
            {
              "src": 1,
              "dst": 0,
              "bytes": 12325253121
            },
            {
              "src": 3,
              "dst": 22,
              "bytes": 10510217899
            },
            {
              "src": 0,
              "dst": 1,
              "bytes": 9990772053
            },
            {
              "src": 14,
              "dst": 13,
              "bytes": 9860924074
            },
            {
              "src": 10,
              "dst": 11,
              "bytes": 9716922368
            },
            {
              "src": 22,
              "dst": 3,
              "bytes": 9699511638
            },
            {
              "src": 4,
              "dst": 3,
              "bytes": 9491364523
            },
            {
              "src": 3,
              "dst": 4,
              "bytes": 9437781333
            }
          ]
        }
      },
      "combine": {
        "son_torus": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 3.0,
          "selected_path_count_mean": 2.7096774193548385,
          "selected_path_count_max": 4,
          "average_hop_count": 3.5952380952380953,
          "byte_weighted_average_hop_count": 3.097258482662996,
          "max_link_load_bytes": 15095070040,
          "median_link_load_bytes": 6497003518.0,
          "average_link_load_bytes": 6911337152,
          "fluid_cycles": 281167589,
          "hot_links": [
            {
              "src": 3,
              "dst": 2,
              "bytes": 15095070040
            },
            {
              "src": 1,
              "dst": 0,
              "bytes": 14776579417
            },
            {
              "src": 0,
              "dst": 1,
              "bytes": 14716449454
            },
            {
              "src": 0,
              "dst": 7,
              "bytes": 14684037806
            },
            {
              "src": 2,
              "dst": 1,
              "bytes": 14423126701
            },
            {
              "src": 1,
              "dst": 2,
              "bytes": 14212924078
            },
            {
              "src": 4,
              "dst": 3,
              "bytes": 14072155480
            },
            {
              "src": 7,
              "dst": 6,
              "bytes": 13273754968
            }
          ]
        },
        "random_regular_seed_0": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7278225806451613,
          "selected_path_count_max": 4,
          "average_hop_count": 2.911318553092182,
          "byte_weighted_average_hop_count": 2.5575668793071618,
          "max_link_load_bytes": 9146754390,
          "median_link_load_bytes": 5635881302.5,
          "average_link_load_bytes": 5707049344,
          "fluid_cycles": 170371576,
          "hot_links": [
            {
              "src": 11,
              "dst": 27,
              "bytes": 9146754390
            },
            {
              "src": 1,
              "dst": 19,
              "bytes": 8914110464
            },
            {
              "src": 11,
              "dst": 12,
              "bytes": 8750369451
            },
            {
              "src": 22,
              "dst": 24,
              "bytes": 8602356395
            },
            {
              "src": 12,
              "dst": 11,
              "bytes": 8564120235
            },
            {
              "src": 24,
              "dst": 22,
              "bytes": 8520087553
            },
            {
              "src": 27,
              "dst": 11,
              "bytes": 8240039253
            },
            {
              "src": 19,
              "dst": 1,
              "bytes": 7905181698
            }
          ]
        },
        "random_regular_seed_12": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.659274193548387,
          "selected_path_count_max": 4,
          "average_hop_count": 2.8979343863912517,
          "byte_weighted_average_hop_count": 2.5588398312256713,
          "max_link_load_bytes": 9081853951,
          "median_link_load_bytes": 5725687807.0,
          "average_link_load_bytes": 5709889856,
          "fluid_cycles": 169162712,
          "hot_links": [
            {
              "src": 30,
              "dst": 3,
              "bytes": 9081853951
            },
            {
              "src": 0,
              "dst": 20,
              "bytes": 9037759830
            },
            {
              "src": 0,
              "dst": 14,
              "bytes": 8759564972
            },
            {
              "src": 3,
              "dst": 30,
              "bytes": 8748775425
            },
            {
              "src": 15,
              "dst": 30,
              "bytes": 8525701119
            },
            {
              "src": 20,
              "dst": 0,
              "bytes": 8480569344
            },
            {
              "src": 7,
              "dst": 12,
              "bytes": 8112848213
            },
            {
              "src": 7,
              "dst": 17,
              "bytes": 8035366229
            }
          ]
        },
        "random_regular_seed_22": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.655241935483871,
          "selected_path_count_max": 4,
          "average_hop_count": 2.8270401948842876,
          "byte_weighted_average_hop_count": 2.4978446319491687,
          "max_link_load_bytes": 7912759978,
          "median_link_load_bytes": 5430720513.5,
          "average_link_load_bytes": 5573782912,
          "fluid_cycles": 147386639,
          "hot_links": [
            {
              "src": 20,
              "dst": 21,
              "bytes": 7912759978
            },
            {
              "src": 0,
              "dst": 4,
              "bytes": 7776217088
            },
            {
              "src": 0,
              "dst": 23,
              "bytes": 7386447190
            },
            {
              "src": 1,
              "dst": 18,
              "bytes": 7344412672
            },
            {
              "src": 18,
              "dst": 1,
              "bytes": 7340795904
            },
            {
              "src": 20,
              "dst": 14,
              "bytes": 7265382400
            },
            {
              "src": 13,
              "dst": 31,
              "bytes": 7257154218
            },
            {
              "src": 21,
              "dst": 20,
              "bytes": 7225845759
            }
          ]
        },
        "calibration_greedy": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6955645161290323,
          "selected_path_count_max": 4,
          "average_hop_count": 2.9785969084423307,
          "byte_weighted_average_hop_count": 2.6293664096431724,
          "max_link_load_bytes": 11289929729,
          "median_link_load_bytes": 5601204564.0,
          "average_link_load_bytes": 5867265472,
          "fluid_cycles": 210291328,
          "hot_links": [
            {
              "src": 20,
              "dst": 19,
              "bytes": 11289929729
            },
            {
              "src": 20,
              "dst": 0,
              "bytes": 10015274326
            },
            {
              "src": 13,
              "dst": 14,
              "bytes": 9781511510
            },
            {
              "src": 0,
              "dst": 20,
              "bytes": 9586194432
            },
            {
              "src": 0,
              "dst": 31,
              "bytes": 9075032064
            },
            {
              "src": 1,
              "dst": 2,
              "bytes": 8866637142
            },
            {
              "src": 12,
              "dst": 11,
              "bytes": 8682441388
            },
            {
              "src": 20,
              "dst": 13,
              "bytes": 8672844459
            }
          ]
        },
        "evaluation_greedy": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6350806451612903,
          "selected_path_count_max": 4,
          "average_hop_count": 2.93711467324291,
          "byte_weighted_average_hop_count": 2.596446851410806,
          "max_link_load_bytes": 12176701440,
          "median_link_load_bytes": 5500812628.0,
          "average_link_load_bytes": 5793807552,
          "fluid_cycles": 226808738,
          "hot_links": [
            {
              "src": 0,
              "dst": 1,
              "bytes": 12176701440
            },
            {
              "src": 22,
              "dst": 3,
              "bytes": 10413146796
            },
            {
              "src": 1,
              "dst": 0,
              "bytes": 10208953686
            },
            {
              "src": 11,
              "dst": 10,
              "bytes": 10108915713
            },
            {
              "src": 3,
              "dst": 22,
              "bytes": 9756448086
            },
            {
              "src": 13,
              "dst": 14,
              "bytes": 9703412395
            },
            {
              "src": 4,
              "dst": 3,
              "bytes": 9490294103
            },
            {
              "src": 3,
              "dst": 4,
              "bytes": 9417362091
            }
          ]
        }
      }
    },
    "tiny_subchunk_audit": {
      "dispatch": {
        "son_torus": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 47276032,
          "subchunk_bytes_median": 79655936.0,
          "subchunks_total": 2688,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 3.0,
          "selected_path_count_mean": 2.7096774193548385,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_0": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 47276032,
          "subchunk_bytes_median": 135933952.0,
          "subchunks_total": 1714,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7278225806451613,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_12": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 47276032,
          "subchunk_bytes_median": 145420288.0,
          "subchunks_total": 1646,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.659274193548387,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_22": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 47714304,
          "subchunk_bytes_median": 145293312.0,
          "subchunks_total": 1642,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.655241935483871,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "calibration_greedy": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 47476736,
          "subchunk_bytes_median": 136687616.0,
          "subchunks_total": 1682,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6955645161290323,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "evaluation_greedy": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 47276032,
          "subchunk_bytes_median": 144433152.0,
          "subchunks_total": 1622,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6350806451612903,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        }
      },
      "combine": {
        "son_torus": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 47276032,
          "subchunk_bytes_median": 79655936.0,
          "subchunks_total": 2688,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 3.0,
          "selected_path_count_mean": 2.7096774193548385,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_0": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 47276032,
          "subchunk_bytes_median": 135933952.0,
          "subchunks_total": 1714,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7278225806451613,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_12": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 47276032,
          "subchunk_bytes_median": 145420288.0,
          "subchunks_total": 1646,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.659274193548387,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_22": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 47714304,
          "subchunk_bytes_median": 145293312.0,
          "subchunks_total": 1642,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.655241935483871,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "calibration_greedy": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 47476736,
          "subchunk_bytes_median": 136687616.0,
          "subchunks_total": 1682,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6955645161290323,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "evaluation_greedy": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 47276032,
          "subchunk_bytes_median": 144433152.0,
          "subchunks_total": 1622,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6350806451612903,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        }
      }
    },
    "candidate_family_interpretation": {
      "workload": "qwen_livecodebench_execution",
      "calibrated_candidate": "random_regular_seed_22",
      "calibrated_candidate_type": "random_regular",
      "oracle_candidate": "random_regular_seed_22",
      "oracle_candidate_type": "random_regular",
      "traffic_fingerprint": "broad / near-uniform",
      "torus_rank_by_eval_fluid": 35,
      "greedy_calibration_rank_by_eval_fluid": 31,
      "greedy_evaluation_rank_by_eval_fluid": 33,
      "fixed_seed0_rank_by_eval_fluid": 15,
      "calibrated_rank_by_eval_fluid": 1,
      "oracle_rank_by_eval_fluid": 1,
      "calibrated_beats_son": true,
      "calibrated_beats_fixed_random": true,
      "oracle_beats_calibrated": false,
      "calibrated_gain_vs_son_percent": 17.72845269361494,
      "calibrated_gain_vs_fixed_random_percent": 4.860114373226027,
      "oracle_gap_vs_calibrated_percent": 0.0,
      "main_explanation": "workload-selected random-regular topology search",
      "median_random_name": "random_regular_seed_12",
      "best_random_name": null
    },
    "validation_pass": {
      "byte_conservation": true,
      "graph_budget": true,
      "graphs_valid": true,
      "native_runs": true,
      "no_tiny_subchunk_risk": true
    }
  },
  {
    "workload": {
      "id": "qwen_mmlu_zh_cn_anatomy",
      "label": "Qwen MMLU_ZH_CN anatomy",
      "path": "/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/mmlu_ZH_CN/anatomy"
    },
    "available": true,
    "trace_parse": {
      "files_found": 135,
      "files_used": 135,
      "moe_layer_count": 94,
      "inferred_num_experts": 128,
      "malformed_records": 0,
      "full": {
        "request_ids": [
          "100",
          "101",
          "102",
          "103",
          "104",
          "105",
          "106",
          "107",
          "108",
          "109",
          "110",
          "111",
          "112",
          "113",
          "114",
          "115",
          "116",
          "117",
          "118",
          "119",
          "120",
          "121",
          "122",
          "123",
          "124",
          "125",
          "126",
          "127",
          "128",
          "129",
          "130",
          "131",
          "132",
          "133",
          "134",
          "135",
          "136",
          "137",
          "138",
          "139",
          "140",
          "141",
          "142",
          "143",
          "144",
          "145",
          "146",
          "147",
          "148",
          "149",
          "150",
          "151",
          "152",
          "153",
          "154",
          "155",
          "156",
          "157",
          "158",
          "159",
          "160",
          "161",
          "162",
          "163",
          "164",
          "165",
          "166",
          "167",
          "168",
          "169",
          "170",
          "171",
          "172",
          "173",
          "174",
          "175",
          "176",
          "177",
          "178",
          "179",
          "180",
          "181",
          "182",
          "183",
          "184",
          "185",
          "186",
          "187",
          "188",
          "189",
          "190",
          "191",
          "192",
          "193",
          "194",
          "195",
          "196",
          "197",
          "198",
          "199",
          "200",
          "201",
          "202",
          "203",
          "204",
          "205",
          "206",
          "207",
          "208",
          "209",
          "210",
          "211",
          "212",
          "213",
          "214",
          "215",
          "216",
          "217",
          "218",
          "219",
          "220",
          "221",
          "222",
          "223",
          "224",
          "225",
          "226",
          "227",
          "228",
          "229",
          "230",
          "231",
          "232",
          "233",
          "234"
        ],
        "request_count": 135,
        "prefill_input_tokens": 1569,
        "selected_expert_events": 1179888,
        "theoretical_dispatch_bytes": 9665642496,
        "theoretical_combine_bytes": 9665642496,
        "local_dispatch_bytes_excluded": 309125120,
        "local_combine_bytes_excluded": 309125120,
        "remote_dispatch_bytes_retained": 9356517376,
        "remote_combine_bytes_retained": 9356517376,
        "byte_conservation_pass": true,
        "dispatch_checksum": "5a7bf0aaef602ad3f5fd0fbb95d84237018de99ec23131c1a99d37023587add7",
        "combine_checksum": "90d4f71cc07a8bb73fafc8629d87e5fc3aee8cfad138821aeb95ae30b4c53a50"
      },
      "calibration": {
        "request_ids": [
          "100",
          "101",
          "102",
          "103",
          "104",
          "105",
          "106",
          "107",
          "108",
          "109",
          "110",
          "111",
          "112",
          "113"
        ],
        "request_count": 14,
        "prefill_input_tokens": 198,
        "selected_expert_events": 148896,
        "theoretical_dispatch_bytes": 1219756032,
        "theoretical_combine_bytes": 1219756032,
        "local_dispatch_bytes_excluded": 39845888,
        "local_combine_bytes_excluded": 39845888,
        "remote_dispatch_bytes_retained": 1179910144,
        "remote_combine_bytes_retained": 1179910144,
        "byte_conservation_pass": true,
        "dispatch_checksum": "245c83898926d56dd6b8593ef5c777b54a0c2659f6746aac2e99d34bc40d5a24",
        "combine_checksum": "0bdf6fc4c4ceacc447a6e3916cca83005d65ab2a89971027995a77cc5787a386"
      },
      "evaluation": {
        "request_ids": [
          "114",
          "115",
          "116",
          "117",
          "118",
          "119",
          "120",
          "121",
          "122",
          "123",
          "124",
          "125",
          "126",
          "127",
          "128",
          "129",
          "130",
          "131",
          "132",
          "133",
          "134",
          "135",
          "136",
          "137",
          "138",
          "139",
          "140",
          "141",
          "142",
          "143",
          "144",
          "145",
          "146",
          "147",
          "148",
          "149",
          "150",
          "151",
          "152",
          "153",
          "154",
          "155",
          "156",
          "157",
          "158",
          "159",
          "160",
          "161",
          "162",
          "163",
          "164",
          "165",
          "166",
          "167",
          "168",
          "169",
          "170",
          "171",
          "172",
          "173",
          "174",
          "175",
          "176",
          "177",
          "178",
          "179",
          "180",
          "181",
          "182",
          "183",
          "184",
          "185",
          "186",
          "187",
          "188",
          "189",
          "190",
          "191",
          "192",
          "193",
          "194",
          "195",
          "196",
          "197",
          "198",
          "199",
          "200",
          "201",
          "202",
          "203",
          "204",
          "205",
          "206",
          "207",
          "208",
          "209",
          "210",
          "211",
          "212",
          "213",
          "214",
          "215",
          "216",
          "217",
          "218",
          "219",
          "220",
          "221",
          "222",
          "223",
          "224",
          "225",
          "226",
          "227",
          "228",
          "229",
          "230",
          "231",
          "232",
          "233",
          "234"
        ],
        "request_count": 121,
        "prefill_input_tokens": 1371,
        "selected_expert_events": 1030992,
        "theoretical_dispatch_bytes": 8445886464,
        "theoretical_combine_bytes": 8445886464,
        "local_dispatch_bytes_excluded": 269279232,
        "local_combine_bytes_excluded": 269279232,
        "remote_dispatch_bytes_retained": 8176607232,
        "remote_combine_bytes_retained": 8176607232,
        "byte_conservation_pass": true,
        "dispatch_checksum": "52e7a825429f8e4cf1c3c79223727ec27ff58d7c93bda91ab54e629e9feff965",
        "combine_checksum": "e5f86f0d56cdd826d1989fe5e58cc5725ececfd04de9e1eba79bf59081409a34"
      }
    },
    "split": {
      "calibration_request_count": 14,
      "evaluation_request_count": 121,
      "calibration_request_ids": [
        "100",
        "101",
        "102",
        "103",
        "104",
        "105",
        "106",
        "107",
        "108",
        "109",
        "110",
        "111",
        "112",
        "113"
      ],
      "evaluation_request_ids": [
        "114",
        "115",
        "116",
        "117",
        "118",
        "119",
        "120",
        "121",
        "122",
        "123",
        "124",
        "125",
        "126",
        "127",
        "128",
        "129",
        "130",
        "131",
        "132",
        "133",
        "134",
        "135",
        "136",
        "137",
        "138",
        "139",
        "140",
        "141",
        "142",
        "143",
        "144",
        "145",
        "146",
        "147",
        "148",
        "149",
        "150",
        "151",
        "152",
        "153",
        "154",
        "155",
        "156",
        "157",
        "158",
        "159",
        "160",
        "161",
        "162",
        "163",
        "164",
        "165",
        "166",
        "167",
        "168",
        "169",
        "170",
        "171",
        "172",
        "173",
        "174",
        "175",
        "176",
        "177",
        "178",
        "179",
        "180",
        "181",
        "182",
        "183",
        "184",
        "185",
        "186",
        "187",
        "188",
        "189",
        "190",
        "191",
        "192",
        "193",
        "194",
        "195",
        "196",
        "197",
        "198",
        "199",
        "200",
        "201",
        "202",
        "203",
        "204",
        "205",
        "206",
        "207",
        "208",
        "209",
        "210",
        "211",
        "212",
        "213",
        "214",
        "215",
        "216",
        "217",
        "218",
        "219",
        "220",
        "221",
        "222",
        "223",
        "224",
        "225",
        "226",
        "227",
        "228",
        "229",
        "230",
        "231",
        "232",
        "233",
        "234"
      ],
      "calibration_rule": "front ceil(10%) requests"
    },
    "anti_leakage": {
      "calibrated_uses_calibration_only": true,
      "oracle_uses_evaluation_only": true,
      "oracle_reference_only": true
    },
    "traffic_fingerprint": {
      "calibration_dispatch": {
        "total_remote_bytes": 1179910144,
        "nonzero_gpu_pairs": 403,
        "top1_share": 0.004165740946456343,
        "top4_share": 0.01639219062430571,
        "top8_share": 0.03237475005554321,
        "top16_share": 0.06337480559875583,
        "gini": 0.16120170152428218,
        "entropy_bits": 8.589601106988367,
        "message_bytes_min": 524288,
        "message_bytes_median": 2940928,
        "message_bytes_mean": 2927816.734491315,
        "message_bytes_max": 4915200,
        "interpretation": "moderately structured"
      },
      "calibration_combine": {
        "total_remote_bytes": 1179910144,
        "nonzero_gpu_pairs": 403,
        "top1_share": 0.004165740946456343,
        "top4_share": 0.01639219062430571,
        "top8_share": 0.03237475005554321,
        "top16_share": 0.06337480559875583,
        "gini": 0.16120170152428218,
        "entropy_bits": 8.58960110698837,
        "message_bytes_min": 524288,
        "message_bytes_median": 2940928,
        "message_bytes_mean": 2927816.734491315,
        "message_bytes_max": 4915200,
        "interpretation": "moderately structured"
      },
      "evaluation_dispatch": {
        "total_remote_bytes": 8176607232,
        "nonzero_gpu_pairs": 992,
        "top1_share": 0.0018364506908481037,
        "top4_share": 0.007167467671755228,
        "top8_share": 0.014171628489932583,
        "top16_share": 0.02796755102838233,
        "gini": 0.156996757996986,
        "entropy_bits": 9.898718585421038,
        "message_bytes_min": 3735552,
        "message_bytes_median": 8101888.0,
        "message_bytes_mean": 8242547.612903226,
        "message_bytes_max": 15015936,
        "interpretation": "moderately structured"
      },
      "evaluation_combine": {
        "total_remote_bytes": 8176607232,
        "nonzero_gpu_pairs": 992,
        "top1_share": 0.0018364506908481037,
        "top4_share": 0.007167467671755228,
        "top8_share": 0.014171628489932583,
        "top16_share": 0.02796755102838233,
        "gini": 0.156996757996986,
        "entropy_bits": 9.898718585421033,
        "message_bytes_min": 3735552,
        "message_bytes_median": 8101888.0,
        "message_bytes_mean": 8242547.612903226,
        "message_bytes_max": 15015936,
        "interpretation": "moderately structured"
      }
    },
    "candidate_pool": {
      "candidate_count": 35,
      "random_candidate_count": 32,
      "candidate_types": {
        "greedy_calibration": 1,
        "greedy_evaluation": 1,
        "random_regular": 32,
        "torus": 1
      },
      "graph_budget_pass": true,
      "all_candidate_graphs_valid": true
    },
    "candidate_scores": {
      "calibration_top12": [
        {
          "name": "random_regular_seed_16",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 51148799
        },
        {
          "name": "random_regular_seed_29",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 55938389
        },
        {
          "name": "random_regular_seed_23",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 55993002
        },
        {
          "name": "random_regular_seed_1",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 57868972
        },
        {
          "name": "random_regular_seed_10",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 57996630
        },
        {
          "name": "random_regular_seed_31",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 58301099
        },
        {
          "name": "random_regular_seed_19",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 59512832
        },
        {
          "name": "random_regular_seed_30",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 59713536
        },
        {
          "name": "random_regular_seed_22",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 59737430
        },
        {
          "name": "random_regular_seed_20",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 60052821
        },
        {
          "name": "random_regular_seed_11",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 60890454
        },
        {
          "name": "random_regular_seed_24",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 60958722
        }
      ],
      "evaluation_top12": [
        {
          "name": "random_regular_seed_16",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 237959850
        },
        {
          "name": "random_regular_seed_22",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 238675286
        },
        {
          "name": "random_regular_seed_3",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 253990913
        },
        {
          "name": "random_regular_seed_30",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 256699734
        },
        {
          "name": "random_regular_seed_20",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 257002156
        },
        {
          "name": "random_regular_seed_27",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 260126721
        },
        {
          "name": "random_regular_seed_29",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 266767701
        },
        {
          "name": "random_regular_seed_25",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 267553449
        },
        {
          "name": "random_regular_seed_31",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 268281856
        },
        {
          "name": "random_regular_seed_17",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 268953600
        },
        {
          "name": "random_regular_seed_13",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 269387774
        },
        {
          "name": "random_regular_seed_9",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 270213804
        }
      ],
      "evaluation_all": [
        {
          "name": "random_regular_seed_16",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 237959850
        },
        {
          "name": "random_regular_seed_22",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 238675286
        },
        {
          "name": "random_regular_seed_3",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 253990913
        },
        {
          "name": "random_regular_seed_30",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 256699734
        },
        {
          "name": "random_regular_seed_20",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 257002156
        },
        {
          "name": "random_regular_seed_27",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 260126721
        },
        {
          "name": "random_regular_seed_29",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 266767701
        },
        {
          "name": "random_regular_seed_25",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 267553449
        },
        {
          "name": "random_regular_seed_31",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 268281856
        },
        {
          "name": "random_regular_seed_17",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 268953600
        },
        {
          "name": "random_regular_seed_13",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 269387774
        },
        {
          "name": "random_regular_seed_9",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 270213804
        },
        {
          "name": "random_regular_seed_24",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 272990209
        },
        {
          "name": "random_regular_seed_1",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 275732478
        },
        {
          "name": "random_regular_seed_11",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 277121024
        },
        {
          "name": "random_regular_seed_18",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 279583403
        },
        {
          "name": "random_regular_seed_14",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 285455019
        },
        {
          "name": "random_regular_seed_19",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 286421673
        },
        {
          "name": "random_regular_seed_2",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 288204117
        },
        {
          "name": "random_regular_seed_23",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 289599490
        },
        {
          "name": "random_regular_seed_0",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 292528129
        },
        {
          "name": "random_regular_seed_12",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 295677951
        },
        {
          "name": "random_regular_seed_4",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 298975233
        },
        {
          "name": "random_regular_seed_8",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 301636265
        },
        {
          "name": "random_regular_seed_15",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 303289684
        },
        {
          "name": "random_regular_seed_5",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 313035437
        },
        {
          "name": "random_regular_seed_21",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 313687381
        },
        {
          "name": "random_regular_seed_7",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 317853013
        },
        {
          "name": "random_regular_seed_6",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 322643968
        },
        {
          "name": "random_regular_seed_10",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 342915071
        },
        {
          "name": "random_regular_seed_28",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 351398571
        },
        {
          "name": "random_regular_seed_26",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 367877462
        },
        {
          "name": "son_torus",
          "candidate_type": "torus",
          "score_max_link_load_bytes": 472693422
        },
        {
          "name": "evaluation_greedy",
          "candidate_type": "greedy_evaluation",
          "score_max_link_load_bytes": 483956735
        },
        {
          "name": "calibration_greedy",
          "candidate_type": "greedy_calibration",
          "score_max_link_load_bytes": 531868331
        }
      ]
    },
    "selected": {
      "ron_calibrated": "random_regular_seed_16",
      "ron_calibrated_type": "random_regular",
      "ron_oracle": "random_regular_seed_16",
      "ron_oracle_type": "random_regular",
      "fixed_random": "random_regular_seed_0",
      "median_random": "random_regular_seed_14",
      "best_random": "random_regular_seed_16",
      "representatives_run_in_astra": [
        "son_torus",
        "random_regular_seed_0",
        "random_regular_seed_14",
        "random_regular_seed_16",
        "calibration_greedy",
        "evaluation_greedy"
      ],
      "roles": {
        "son_torus": "son",
        "random_regular_seed_0": "fixed_random",
        "random_regular_seed_14": "median_random",
        "random_regular_seed_16": "ron_oracle",
        "calibration_greedy": "greedy_calibration",
        "evaluation_greedy": "greedy_evaluation"
      }
    },
    "candidate_audits_all_budget_only": {
      "son_torus": {
        "name": "son_torus_4x8_32gpu",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "calibration_greedy": {
        "name": "ron_calibration_greedy_degree4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "evaluation_greedy": {
        "name": "ron_evaluation_greedy_degree4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_0": {
        "name": "ron_random_regular_seed_0",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_1": {
        "name": "ron_random_regular_seed_1",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_2": {
        "name": "ron_random_regular_seed_2",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_3": {
        "name": "ron_random_regular_seed_3",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_4": {
        "name": "ron_random_regular_seed_4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_5": {
        "name": "ron_random_regular_seed_5",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_6": {
        "name": "ron_random_regular_seed_6",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_7": {
        "name": "ron_random_regular_seed_7",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_8": {
        "name": "ron_random_regular_seed_8",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_9": {
        "name": "ron_random_regular_seed_9",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_10": {
        "name": "ron_random_regular_seed_10",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_11": {
        "name": "ron_random_regular_seed_11",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_12": {
        "name": "ron_random_regular_seed_12",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_13": {
        "name": "ron_random_regular_seed_13",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_14": {
        "name": "ron_random_regular_seed_14",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_15": {
        "name": "ron_random_regular_seed_15",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_16": {
        "name": "ron_random_regular_seed_16",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_17": {
        "name": "ron_random_regular_seed_17",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_18": {
        "name": "ron_random_regular_seed_18",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_19": {
        "name": "ron_random_regular_seed_19",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_20": {
        "name": "ron_random_regular_seed_20",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_21": {
        "name": "ron_random_regular_seed_21",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_22": {
        "name": "ron_random_regular_seed_22",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_23": {
        "name": "ron_random_regular_seed_23",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_24": {
        "name": "ron_random_regular_seed_24",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_25": {
        "name": "ron_random_regular_seed_25",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_26": {
        "name": "ron_random_regular_seed_26",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_27": {
        "name": "ron_random_regular_seed_27",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_28": {
        "name": "ron_random_regular_seed_28",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_29": {
        "name": "ron_random_regular_seed_29",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_30": {
        "name": "ron_random_regular_seed_30",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_31": {
        "name": "ron_random_regular_seed_31",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      }
    },
    "candidate_audits_representatives": {
      "son_torus": {
        "name": "son_torus_4x8_32gpu",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 3.096774193548387,
        "diameter": 6,
        "ecmp_path_count_distribution_cap4": {
          "1": 256,
          "2": 192,
          "3": 128,
          "4": 416
        }
      },
      "random_regular_seed_0": {
        "name": "ron_random_regular_seed_0",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.560483870967742,
        "diameter": 4,
        "ecmp_path_count_distribution_cap4": {
          "1": 590,
          "2": 200,
          "3": 84,
          "4": 118
        }
      },
      "random_regular_seed_14": {
        "name": "ron_random_regular_seed_14",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.5745967741935485,
        "diameter": 4,
        "ecmp_path_count_distribution_cap4": {
          "1": 618,
          "2": 206,
          "3": 82,
          "4": 86
        }
      },
      "random_regular_seed_16": {
        "name": "ron_random_regular_seed_16",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.473790322580645,
        "diameter": 4,
        "ecmp_path_count_distribution_cap4": {
          "1": 654,
          "2": 216,
          "3": 54,
          "4": 68
        }
      },
      "calibration_greedy": {
        "name": "ron_calibration_greedy_degree4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.711693548387097,
        "diameter": 5,
        "ecmp_path_count_distribution_cap4": {
          "1": 570,
          "2": 222,
          "3": 96,
          "4": 104
        }
      },
      "evaluation_greedy": {
        "name": "ron_evaluation_greedy_degree4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.7903225806451615,
        "diameter": 6,
        "ecmp_path_count_distribution_cap4": {
          "1": 556,
          "2": 190,
          "3": 100,
          "4": 146
        }
      }
    },
    "native_astra_results": {
      "dispatch": {
        "son_torus": {
          "label": "qwen_mmlu_zh_cn_anatomy_dispatch_son_torus",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.02885612496174872,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/network_configs/son_torus.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_dispatch_son_torus.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_dispatch_son_torus.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 17754176,
          "cycles_count": 32
        },
        "random_regular_seed_0": {
          "label": "qwen_mmlu_zh_cn_anatomy_dispatch_random_regular_seed_0",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.02613699994981289,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/network_configs/random_regular_seed_0.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_dispatch_random_regular_seed_0.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_dispatch_random_regular_seed_0.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 16883356,
          "cycles_count": 32
        },
        "random_regular_seed_14": {
          "label": "qwen_mmlu_zh_cn_anatomy_dispatch_random_regular_seed_14",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.025634292047470808,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/network_configs/random_regular_seed_14.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_dispatch_random_regular_seed_14.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_dispatch_random_regular_seed_14.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 15619520,
          "cycles_count": 32
        },
        "random_regular_seed_16": {
          "label": "qwen_mmlu_zh_cn_anatomy_dispatch_random_regular_seed_16",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.02555666700936854,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/network_configs/random_regular_seed_16.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_dispatch_random_regular_seed_16.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_dispatch_random_regular_seed_16.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 14766984,
          "cycles_count": 32
        },
        "calibration_greedy": {
          "label": "qwen_mmlu_zh_cn_anatomy_dispatch_calibration_greedy",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.025410625035874546,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/network_configs/calibration_greedy.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_dispatch_calibration_greedy.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_dispatch_calibration_greedy.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 19117129,
          "cycles_count": 32
        },
        "evaluation_greedy": {
          "label": "qwen_mmlu_zh_cn_anatomy_dispatch_evaluation_greedy",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.026080540963448584,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/network_configs/evaluation_greedy.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_dispatch_evaluation_greedy.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_dispatch_evaluation_greedy.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 19541056,
          "cycles_count": 32
        }
      },
      "combine": {
        "son_torus": {
          "label": "qwen_mmlu_zh_cn_anatomy_combine_son_torus",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.02786375000141561,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/network_configs/son_torus.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_combine_son_torus.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_combine_son_torus.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 17694079,
          "cycles_count": 32
        },
        "random_regular_seed_0": {
          "label": "qwen_mmlu_zh_cn_anatomy_combine_random_regular_seed_0",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.02770824998151511,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/network_configs/random_regular_seed_0.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_combine_random_regular_seed_0.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_combine_random_regular_seed_0.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 17126045,
          "cycles_count": 32
        },
        "random_regular_seed_14": {
          "label": "qwen_mmlu_zh_cn_anatomy_combine_random_regular_seed_14",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.026256332988850772,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/network_configs/random_regular_seed_14.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_combine_random_regular_seed_14.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_combine_random_regular_seed_14.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 16096539,
          "cycles_count": 32
        },
        "random_regular_seed_16": {
          "label": "qwen_mmlu_zh_cn_anatomy_combine_random_regular_seed_16",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.026156332925893366,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/network_configs/random_regular_seed_16.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_combine_random_regular_seed_16.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_combine_random_regular_seed_16.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 14897512,
          "cycles_count": 32
        },
        "calibration_greedy": {
          "label": "qwen_mmlu_zh_cn_anatomy_combine_calibration_greedy",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.025588707998394966,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/network_configs/calibration_greedy.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_combine_calibration_greedy.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_combine_calibration_greedy.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 19633564,
          "cycles_count": 32
        },
        "evaluation_greedy": {
          "label": "qwen_mmlu_zh_cn_anatomy_combine_evaluation_greedy",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.026414875057525933,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/qwen_mmlu_zh_cn_anatomy/network_configs/evaluation_greedy.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_combine_evaluation_greedy.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/qwen_mmlu_zh_cn_anatomy_combine_evaluation_greedy.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 21246070,
          "cycles_count": 32
        }
      }
    },
    "native_astra_totals": {
      "son_torus": {
        "candidate_type": "torus",
        "dispatch_cycles": 17754176,
        "combine_cycles": 17694079,
        "total_cycles": 35448255,
        "dispatch_fluid_cycles": 7934443,
        "combine_fluid_cycles": 8804601,
        "total_fluid_cycles": 16739044,
        "astra_over_fluid_total": 2.117698896065988,
        "success": true,
        "runtime_s": 0.05671987496316433,
        "role": "son"
      },
      "random_regular_seed_0": {
        "candidate_type": "random_regular",
        "dispatch_cycles": 16883356,
        "combine_cycles": 17126045,
        "total_cycles": 34009401,
        "dispatch_fluid_cycles": 5448761,
        "combine_fluid_cycles": 5360336,
        "total_fluid_cycles": 10809097,
        "astra_over_fluid_total": 3.1463683784131087,
        "success": true,
        "runtime_s": 0.053845249931328,
        "role": "fixed_random"
      },
      "random_regular_seed_14": {
        "candidate_type": "random_regular",
        "dispatch_cycles": 15619520,
        "combine_cycles": 16096539,
        "total_cycles": 31716059,
        "dispatch_fluid_cycles": 5317014,
        "combine_fluid_cycles": 5265477,
        "total_fluid_cycles": 10582491,
        "astra_over_fluid_total": 2.9970315117678816,
        "success": true,
        "runtime_s": 0.05189062503632158,
        "role": "median_random"
      },
      "random_regular_seed_16": {
        "candidate_type": "random_regular",
        "dispatch_cycles": 14766984,
        "combine_cycles": 14897512,
        "total_cycles": 29664496,
        "dispatch_fluid_cycles": 4429550,
        "combine_fluid_cycles": 4432347,
        "total_fluid_cycles": 8861897,
        "astra_over_fluid_total": 3.347420535354902,
        "success": true,
        "runtime_s": 0.051712999935261905,
        "role": "ron_oracle"
      },
      "calibration_greedy": {
        "candidate_type": "greedy_calibration",
        "dispatch_cycles": 19117129,
        "combine_cycles": 19633564,
        "total_cycles": 38750693,
        "dispatch_fluid_cycles": 9707425,
        "combine_fluid_cycles": 9906819,
        "total_fluid_cycles": 19614244,
        "astra_over_fluid_total": 1.9756404070429632,
        "success": true,
        "runtime_s": 0.05099933303426951,
        "role": "greedy_calibration"
      },
      "evaluation_greedy": {
        "candidate_type": "greedy_evaluation",
        "dispatch_cycles": 19541056,
        "combine_cycles": 21246070,
        "total_cycles": 40787126,
        "dispatch_fluid_cycles": 9014396,
        "combine_fluid_cycles": 8582725,
        "total_fluid_cycles": 17597121,
        "astra_over_fluid_total": 2.317829490403572,
        "success": true,
        "runtime_s": 0.05249541602097452,
        "role": "greedy_evaluation"
      }
    },
    "fluid_lower_bound": {
      "dispatch": {
        "son_torus": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 3.0,
          "selected_path_count_mean": 2.7096774193548385,
          "selected_path_count_max": 4,
          "average_hop_count": 3.5952380952380953,
          "byte_weighted_average_hop_count": 3.091957788684939,
          "max_link_load_bytes": 425977175,
          "median_link_load_bytes": 191890430.0,
          "average_link_load_bytes": 197513472,
          "fluid_cycles": 7934443,
          "hot_links": [
            {
              "src": 0,
              "dst": 7,
              "bytes": 425977175
            },
            {
              "src": 3,
              "dst": 2,
              "bytes": 423528451
            },
            {
              "src": 1,
              "dst": 0,
              "bytes": 423427415
            },
            {
              "src": 2,
              "dst": 1,
              "bytes": 399005699
            },
            {
              "src": 0,
              "dst": 1,
              "bytes": 387913047
            },
            {
              "src": 1,
              "dst": 2,
              "bytes": 383922862
            },
            {
              "src": 4,
              "dst": 3,
              "bytes": 375965697
            },
            {
              "src": 5,
              "dst": 4,
              "bytes": 353850711
            }
          ]
        },
        "random_regular_seed_0": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7278225806451613,
          "selected_path_count_max": 4,
          "average_hop_count": 2.911318553092182,
          "byte_weighted_average_hop_count": 2.5682236923178654,
          "max_link_load_bytes": 292528129,
          "median_link_load_bytes": 162883926.0,
          "average_link_load_bytes": 164057472,
          "fluid_cycles": 5448761,
          "hot_links": [
            {
              "src": 19,
              "dst": 1,
              "bytes": 292528129
            },
            {
              "src": 11,
              "dst": 27,
              "bytes": 276256085
            },
            {
              "src": 22,
              "dst": 24,
              "bytes": 244454058
            },
            {
              "src": 12,
              "dst": 11,
              "bytes": 236087980
            },
            {
              "src": 1,
              "dst": 19,
              "bytes": 235788287
            },
            {
              "src": 11,
              "dst": 12,
              "bytes": 231741439
            },
            {
              "src": 14,
              "dst": 8,
              "bytes": 230849194
            },
            {
              "src": 31,
              "dst": 13,
              "bytes": 229747371
            }
          ]
        },
        "random_regular_seed_14": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6330645161290323,
          "selected_path_count_max": 4,
          "average_hop_count": 2.8987654320987652,
          "byte_weighted_average_hop_count": 2.5696633975239473,
          "max_link_load_bytes": 285455019,
          "median_link_load_bytes": 160026282.5,
          "average_link_load_bytes": 164149440,
          "fluid_cycles": 5317014,
          "hot_links": [
            {
              "src": 14,
              "dst": 29,
              "bytes": 285455019
            },
            {
              "src": 24,
              "dst": 1,
              "bytes": 277477375
            },
            {
              "src": 29,
              "dst": 30,
              "bytes": 272748543
            },
            {
              "src": 29,
              "dst": 14,
              "bytes": 269436927
            },
            {
              "src": 1,
              "dst": 24,
              "bytes": 269279233
            },
            {
              "src": 22,
              "dst": 7,
              "bytes": 263379627
            },
            {
              "src": 1,
              "dst": 18,
              "bytes": 233400320
            },
            {
              "src": 5,
              "dst": 1,
              "bytes": 232776365
            }
          ]
        },
        "random_regular_seed_16": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.532258064516129,
          "selected_path_count_max": 4,
          "average_hop_count": 2.763157894736842,
          "byte_weighted_average_hop_count": 2.4731410319991265,
          "max_link_load_bytes": 237809665,
          "median_link_load_bytes": 154076501.5,
          "average_link_load_bytes": 157983616,
          "fluid_cycles": 4429550,
          "hot_links": [
            {
              "src": 21,
              "dst": 2,
              "bytes": 237809665
            },
            {
              "src": 0,
              "dst": 27,
              "bytes": 232966827
            },
            {
              "src": 18,
              "dst": 2,
              "bytes": 228477610
            },
            {
              "src": 17,
              "dst": 1,
              "bytes": 225688918
            },
            {
              "src": 22,
              "dst": 19,
              "bytes": 220567551
            },
            {
              "src": 27,
              "dst": 0,
              "bytes": 215770453
            },
            {
              "src": 19,
              "dst": 22,
              "bytes": 215540394
            },
            {
              "src": 0,
              "dst": 15,
              "bytes": 212717568
            }
          ]
        },
        "calibration_greedy": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7318548387096775,
          "selected_path_count_max": 4,
          "average_hop_count": 3.080325960419092,
          "byte_weighted_average_hop_count": 2.7146237780790106,
          "max_link_load_bytes": 521163435,
          "median_link_load_bytes": 158780076.0,
          "average_link_load_bytes": 173409472,
          "fluid_cycles": 9707425,
          "hot_links": [
            {
              "src": 23,
              "dst": 1,
              "bytes": 521163435
            },
            {
              "src": 26,
              "dst": 3,
              "bytes": 405841237
            },
            {
              "src": 20,
              "dst": 0,
              "bytes": 387848877
            },
            {
              "src": 31,
              "dst": 0,
              "bytes": 380024832
            },
            {
              "src": 29,
              "dst": 30,
              "bytes": 361094483
            },
            {
              "src": 0,
              "dst": 4,
              "bytes": 328671234
            },
            {
              "src": 1,
              "dst": 23,
              "bytes": 328204288
            },
            {
              "src": 3,
              "dst": 26,
              "bytes": 326880595
            }
          ]
        },
        "evaluation_greedy": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.8346774193548387,
          "selected_path_count_max": 4,
          "average_hop_count": 3.1747252747252745,
          "byte_weighted_average_hop_count": 2.7651577313772577,
          "max_link_load_bytes": 483956735,
          "median_link_load_bytes": 156368215.5,
          "average_link_load_bytes": 176637568,
          "fluid_cycles": 9014396,
          "hot_links": [
            {
              "src": 19,
              "dst": 18,
              "bytes": 483956735
            },
            {
              "src": 17,
              "dst": 7,
              "bytes": 437196116
            },
            {
              "src": 2,
              "dst": 3,
              "bytes": 411475970
            },
            {
              "src": 3,
              "dst": 2,
              "bytes": 406363481
            },
            {
              "src": 18,
              "dst": 19,
              "bytes": 371970731
            },
            {
              "src": 1,
              "dst": 2,
              "bytes": 367284907
            },
            {
              "src": 2,
              "dst": 1,
              "bytes": 346908673
            },
            {
              "src": 13,
              "dst": 7,
              "bytes": 320070999
            }
          ]
        }
      },
      "combine": {
        "son_torus": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 3.0,
          "selected_path_count_mean": 2.7096774193548385,
          "selected_path_count_max": 4,
          "average_hop_count": 3.5952380952380953,
          "byte_weighted_average_hop_count": 3.091957788684939,
          "max_link_load_bytes": 472693422,
          "median_link_load_bytes": 183157760.5,
          "average_link_load_bytes": 197513472,
          "fluid_cycles": 8804601,
          "hot_links": [
            {
              "src": 2,
              "dst": 1,
              "bytes": 472693422
            },
            {
              "src": 1,
              "dst": 0,
              "bytes": 465083735
            },
            {
              "src": 3,
              "dst": 2,
              "bytes": 389057881
            },
            {
              "src": 4,
              "dst": 3,
              "bytes": 379654830
            },
            {
              "src": 7,
              "dst": 0,
              "bytes": 377970007
            },
            {
              "src": 0,
              "dst": 1,
              "bytes": 373341527
            },
            {
              "src": 0,
              "dst": 7,
              "bytes": 371666263
            },
            {
              "src": 7,
              "dst": 6,
              "bytes": 360955907
            }
          ]
        },
        "random_regular_seed_0": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7278225806451613,
          "selected_path_count_max": 4,
          "average_hop_count": 2.911318553092182,
          "byte_weighted_average_hop_count": 2.5682236923178654,
          "max_link_load_bytes": 287780865,
          "median_link_load_bytes": 160657748.0,
          "average_link_load_bytes": 164057472,
          "fluid_cycles": 5360336,
          "hot_links": [
            {
              "src": 1,
              "dst": 19,
              "bytes": 287780865
            },
            {
              "src": 27,
              "dst": 11,
              "bytes": 269348181
            },
            {
              "src": 13,
              "dst": 31,
              "bytes": 247591594
            },
            {
              "src": 24,
              "dst": 22,
              "bytes": 241283754
            },
            {
              "src": 19,
              "dst": 1,
              "bytes": 239476737
            },
            {
              "src": 12,
              "dst": 11,
              "bytes": 236283905
            },
            {
              "src": 8,
              "dst": 14,
              "bytes": 236272299
            },
            {
              "src": 22,
              "dst": 24,
              "bytes": 232591361
            }
          ]
        },
        "random_regular_seed_14": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6330645161290323,
          "selected_path_count_max": 4,
          "average_hop_count": 2.8987654320987652,
          "byte_weighted_average_hop_count": 2.5696633975239473,
          "max_link_load_bytes": 282688170,
          "median_link_load_bytes": 160182612.5,
          "average_link_load_bytes": 164149440,
          "fluid_cycles": 5265477,
          "hot_links": [
            {
              "src": 29,
              "dst": 14,
              "bytes": 282688170
            },
            {
              "src": 1,
              "dst": 24,
              "bytes": 276029440
            },
            {
              "src": 30,
              "dst": 29,
              "bytes": 275187710
            },
            {
              "src": 14,
              "dst": 29,
              "bytes": 272119808
            },
            {
              "src": 7,
              "dst": 22,
              "bytes": 268378794
            },
            {
              "src": 24,
              "dst": 1,
              "bytes": 268210174
            },
            {
              "src": 18,
              "dst": 1,
              "bytes": 235345919
            },
            {
              "src": 1,
              "dst": 31,
              "bytes": 234580649
            }
          ]
        },
        "random_regular_seed_16": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.532258064516129,
          "selected_path_count_max": 4,
          "average_hop_count": 2.763157894736842,
          "byte_weighted_average_hop_count": 2.4731410319991265,
          "max_link_load_bytes": 237959850,
          "median_link_load_bytes": 155131221.0,
          "average_link_load_bytes": 157983616,
          "fluid_cycles": 4432347,
          "hot_links": [
            {
              "src": 27,
              "dst": 0,
              "bytes": 237959850
            },
            {
              "src": 2,
              "dst": 21,
              "bytes": 235403264
            },
            {
              "src": 2,
              "dst": 18,
              "bytes": 234668715
            },
            {
              "src": 1,
              "dst": 17,
              "bytes": 223950166
            },
            {
              "src": 19,
              "dst": 22,
              "bytes": 222851072
            },
            {
              "src": 30,
              "dst": 12,
              "bytes": 213794817
            },
            {
              "src": 15,
              "dst": 0,
              "bytes": 212860928
            },
            {
              "src": 22,
              "dst": 19,
              "bytes": 212671146
            }
          ]
        },
        "calibration_greedy": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7318548387096775,
          "selected_path_count_max": 4,
          "average_hop_count": 3.080325960419092,
          "byte_weighted_average_hop_count": 2.7146237780790106,
          "max_link_load_bytes": 531868331,
          "median_link_load_bytes": 161469439.5,
          "average_link_load_bytes": 173409472,
          "fluid_cycles": 9906819,
          "hot_links": [
            {
              "src": 1,
              "dst": 23,
              "bytes": 531868331
            },
            {
              "src": 3,
              "dst": 26,
              "bytes": 423064917
            },
            {
              "src": 0,
              "dst": 20,
              "bytes": 384062125
            },
            {
              "src": 30,
              "dst": 29,
              "bytes": 375731540
            },
            {
              "src": 0,
              "dst": 31,
              "bytes": 370599936
            },
            {
              "src": 4,
              "dst": 0,
              "bytes": 331374595
            },
            {
              "src": 23,
              "dst": 1,
              "bytes": 326148097
            },
            {
              "src": 26,
              "dst": 3,
              "bytes": 314928468
            }
          ]
        },
        "evaluation_greedy": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.8346774193548387,
          "selected_path_count_max": 4,
          "average_hop_count": 3.1747252747252745,
          "byte_weighted_average_hop_count": 2.7651577313772577,
          "max_link_load_bytes": 460781568,
          "median_link_load_bytes": 156334422.5,
          "average_link_load_bytes": 176637568,
          "fluid_cycles": 8582725,
          "hot_links": [
            {
              "src": 18,
              "dst": 19,
              "bytes": 460781568
            },
            {
              "src": 7,
              "dst": 17,
              "bytes": 432741716
            },
            {
              "src": 3,
              "dst": 2,
              "bytes": 412151811
            },
            {
              "src": 2,
              "dst": 3,
              "bytes": 404774230
            },
            {
              "src": 19,
              "dst": 18,
              "bytes": 390238891
            },
            {
              "src": 2,
              "dst": 1,
              "bytes": 369289901
            },
            {
              "src": 1,
              "dst": 2,
              "bytes": 344463361
            },
            {
              "src": 7,
              "dst": 13,
              "bytes": 332152152
            }
          ]
        }
      }
    },
    "tiny_subchunk_audit": {
      "dispatch": {
        "son_torus": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 964608,
          "subchunk_bytes_median": 2352128.0,
          "subchunks_total": 2688,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 3.0,
          "selected_path_count_mean": 2.7096774193548385,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_0": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 964608,
          "subchunk_bytes_median": 3637248.0,
          "subchunks_total": 1714,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7278225806451613,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_14": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 997376,
          "subchunk_bytes_median": 4165632.0,
          "subchunks_total": 1620,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6330645161290323,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_16": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 1038336,
          "subchunk_bytes_median": 4653056.0,
          "subchunks_total": 1520,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.532258064516129,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "calibration_greedy": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 983040,
          "subchunk_bytes_median": 3805184.0,
          "subchunks_total": 1718,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7318548387096775,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "evaluation_greedy": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 952320,
          "subchunk_bytes_median": 3241984.0,
          "subchunks_total": 1820,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.8346774193548387,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        }
      },
      "combine": {
        "son_torus": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 964608,
          "subchunk_bytes_median": 2352128.0,
          "subchunks_total": 2688,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 3.0,
          "selected_path_count_mean": 2.7096774193548385,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_0": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 964608,
          "subchunk_bytes_median": 3637248.0,
          "subchunks_total": 1714,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7278225806451613,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_14": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 997376,
          "subchunk_bytes_median": 4165632.0,
          "subchunks_total": 1620,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.6330645161290323,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_16": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 1038336,
          "subchunk_bytes_median": 4653056.0,
          "subchunks_total": 1520,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.532258064516129,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "calibration_greedy": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 983040,
          "subchunk_bytes_median": 3805184.0,
          "subchunks_total": 1718,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7318548387096775,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "evaluation_greedy": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 952320,
          "subchunk_bytes_median": 3241984.0,
          "subchunks_total": 1820,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.8346774193548387,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        }
      }
    },
    "candidate_family_interpretation": {
      "workload": "qwen_mmlu_zh_cn_anatomy",
      "calibrated_candidate": "random_regular_seed_16",
      "calibrated_candidate_type": "random_regular",
      "oracle_candidate": "random_regular_seed_16",
      "oracle_candidate_type": "random_regular",
      "traffic_fingerprint": "moderately structured",
      "torus_rank_by_eval_fluid": 33,
      "greedy_calibration_rank_by_eval_fluid": 35,
      "greedy_evaluation_rank_by_eval_fluid": 34,
      "fixed_seed0_rank_by_eval_fluid": 21,
      "calibrated_rank_by_eval_fluid": 1,
      "oracle_rank_by_eval_fluid": 1,
      "calibrated_beats_son": true,
      "calibrated_beats_fixed_random": true,
      "oracle_beats_calibrated": false,
      "calibrated_gain_vs_son_percent": 16.316061256047725,
      "calibrated_gain_vs_fixed_random_percent": 12.775599899568945,
      "oracle_gap_vs_calibrated_percent": 0.0,
      "main_explanation": "workload-selected random-regular topology search",
      "median_random_name": "random_regular_seed_14",
      "best_random_name": null
    },
    "validation_pass": {
      "byte_conservation": true,
      "graph_budget": true,
      "graphs_valid": true,
      "native_runs": true,
      "no_tiny_subchunk_risk": true
    }
  },
  {
    "workload": {
      "id": "deepseek_livecodebench_execution",
      "label": "DeepSeek LiveCodeBench execution",
      "path": "/Users/dfx/Python/trace/cognitivecomputations/DeepSeek-R1-AWQ/livecodebench/execution"
    },
    "available": true,
    "trace_parse": {
      "files_found": 440,
      "files_used": 440,
      "moe_layer_count": 58,
      "inferred_num_experts": 256,
      "malformed_records": 0,
      "full": {
        "request_ids": [
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "7",
          "8",
          "9",
          "10",
          "11",
          "12",
          "13",
          "14",
          "15",
          "16",
          "17",
          "18",
          "19",
          "20",
          "21",
          "22",
          "23",
          "24",
          "25",
          "26",
          "27",
          "28",
          "29",
          "30",
          "31",
          "32",
          "33",
          "34",
          "35",
          "36",
          "37",
          "38",
          "39",
          "40",
          "41",
          "42",
          "43",
          "44",
          "45",
          "46",
          "47",
          "48",
          "49",
          "50",
          "51",
          "52",
          "53",
          "54",
          "55",
          "56",
          "57",
          "58",
          "59",
          "60",
          "61",
          "62",
          "63",
          "64",
          "65",
          "66",
          "67",
          "68",
          "69",
          "70",
          "71",
          "72",
          "73",
          "74",
          "75",
          "76",
          "77",
          "78",
          "79",
          "80",
          "81",
          "82",
          "83",
          "84",
          "85",
          "86",
          "87",
          "88",
          "89",
          "90",
          "91",
          "92",
          "93",
          "94",
          "95",
          "96",
          "97",
          "98",
          "99",
          "100",
          "101",
          "102",
          "103",
          "104",
          "105",
          "106",
          "107",
          "108",
          "109",
          "110",
          "111",
          "112",
          "113",
          "114",
          "115",
          "116",
          "117",
          "118",
          "119",
          "120",
          "121",
          "122",
          "123",
          "124",
          "125",
          "126",
          "127",
          "128",
          "129",
          "130",
          "131",
          "132",
          "133",
          "134",
          "135",
          "136",
          "137",
          "138",
          "139",
          "160",
          "161",
          "162",
          "163",
          "164",
          "165",
          "166",
          "167",
          "168",
          "169",
          "170",
          "171",
          "172",
          "173",
          "174",
          "175",
          "176",
          "177",
          "178",
          "179",
          "180",
          "181",
          "182",
          "183",
          "184",
          "185",
          "186",
          "187",
          "188",
          "189",
          "190",
          "191",
          "192",
          "193",
          "194",
          "195",
          "196",
          "197",
          "198",
          "199",
          "200",
          "201",
          "202",
          "203",
          "204",
          "205",
          "206",
          "207",
          "208",
          "209",
          "210",
          "211",
          "212",
          "213",
          "214",
          "215",
          "216",
          "217",
          "218",
          "219",
          "220",
          "221",
          "222",
          "223",
          "224",
          "225",
          "226",
          "227",
          "228",
          "229",
          "230",
          "231",
          "232",
          "233",
          "234",
          "235",
          "236",
          "237",
          "238",
          "239",
          "240",
          "241",
          "242",
          "243",
          "244",
          "245",
          "246",
          "247",
          "248",
          "249",
          "250",
          "251",
          "252",
          "253",
          "254",
          "255",
          "256",
          "257",
          "258",
          "259",
          "260",
          "261",
          "262",
          "263",
          "264",
          "265",
          "266",
          "267",
          "268",
          "269",
          "270",
          "271",
          "272",
          "273",
          "274",
          "275",
          "276",
          "277",
          "278",
          "279",
          "280",
          "281",
          "282",
          "283",
          "284",
          "285",
          "286",
          "287",
          "288",
          "289",
          "290",
          "291",
          "292",
          "293",
          "294",
          "295",
          "296",
          "297",
          "298",
          "299",
          "300",
          "301",
          "302",
          "303",
          "304",
          "305",
          "306",
          "307",
          "308",
          "309",
          "310",
          "311",
          "312",
          "313",
          "314",
          "315",
          "316",
          "317",
          "318",
          "319",
          "320",
          "321",
          "322",
          "323",
          "324",
          "325",
          "326",
          "327",
          "328",
          "329",
          "330",
          "331",
          "332",
          "333",
          "334",
          "335",
          "336",
          "337",
          "338",
          "339",
          "340",
          "341",
          "342",
          "343",
          "344",
          "345",
          "346",
          "347",
          "348",
          "349",
          "350",
          "351",
          "352",
          "353",
          "354",
          "355",
          "356",
          "357",
          "358",
          "359",
          "360",
          "361",
          "362",
          "363",
          "364",
          "365",
          "366",
          "367",
          "368",
          "369",
          "370",
          "371",
          "372",
          "373",
          "374",
          "375",
          "376",
          "377",
          "378",
          "379",
          "380",
          "381",
          "382",
          "383",
          "384",
          "385",
          "386",
          "387",
          "388",
          "389",
          "390",
          "391",
          "392",
          "393",
          "394",
          "395",
          "396",
          "397",
          "398",
          "399",
          "400",
          "401",
          "402",
          "403",
          "404",
          "405",
          "406",
          "407",
          "408",
          "409",
          "410",
          "411",
          "412",
          "413",
          "414",
          "415",
          "416",
          "417",
          "418",
          "419",
          "420",
          "421",
          "422",
          "423",
          "424",
          "425",
          "426",
          "427",
          "428",
          "429",
          "430",
          "431",
          "432",
          "433",
          "434",
          "435",
          "436",
          "437",
          "438",
          "439",
          "440",
          "441",
          "442",
          "443",
          "444",
          "445",
          "446",
          "447",
          "448",
          "449",
          "450",
          "451",
          "452",
          "453",
          "454",
          "455",
          "456",
          "457",
          "458",
          "459"
        ],
        "request_count": 440,
        "prefill_input_tokens": 50526,
        "selected_expert_events": 23444064,
        "theoretical_dispatch_bytes": 192053772288,
        "theoretical_combine_bytes": 192053772288,
        "local_dispatch_bytes_excluded": 6004613120,
        "local_combine_bytes_excluded": 6004613120,
        "remote_dispatch_bytes_retained": 186049159168,
        "remote_combine_bytes_retained": 186049159168,
        "byte_conservation_pass": true,
        "dispatch_checksum": "68b123e4d9eb2b01b45330d7dea0cc3470b066328f55f9d17174ece6ef2ce5b3",
        "combine_checksum": "acad04e84497b064f6770b31925b6ac1ac71c1a345a07a5a5483e810d6e7e00e"
      },
      "calibration": {
        "request_ids": [
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "7",
          "8",
          "9",
          "10",
          "11",
          "12",
          "13",
          "14",
          "15",
          "16",
          "17",
          "18",
          "19",
          "20",
          "21",
          "22",
          "23",
          "24",
          "25",
          "26",
          "27",
          "28",
          "29",
          "30",
          "31",
          "32",
          "33",
          "34",
          "35",
          "36",
          "37",
          "38",
          "39",
          "40",
          "41",
          "42",
          "43"
        ],
        "request_count": 44,
        "prefill_input_tokens": 4305,
        "selected_expert_events": 1997520,
        "theoretical_dispatch_bytes": 16363683840,
        "theoretical_combine_bytes": 16363683840,
        "local_dispatch_bytes_excluded": 516005888,
        "local_combine_bytes_excluded": 516005888,
        "remote_dispatch_bytes_retained": 15847677952,
        "remote_combine_bytes_retained": 15847677952,
        "byte_conservation_pass": true,
        "dispatch_checksum": "0619f0b42a6af32a32d6262b18eb36b57055305c73c09002c75837a7f24bdd6c",
        "combine_checksum": "d975618d118001bdb1e133d3039e2c76daa9138accfd551d2867b90dcd52d4fa"
      },
      "evaluation": {
        "request_ids": [
          "44",
          "45",
          "46",
          "47",
          "48",
          "49",
          "50",
          "51",
          "52",
          "53",
          "54",
          "55",
          "56",
          "57",
          "58",
          "59",
          "60",
          "61",
          "62",
          "63",
          "64",
          "65",
          "66",
          "67",
          "68",
          "69",
          "70",
          "71",
          "72",
          "73",
          "74",
          "75",
          "76",
          "77",
          "78",
          "79",
          "80",
          "81",
          "82",
          "83",
          "84",
          "85",
          "86",
          "87",
          "88",
          "89",
          "90",
          "91",
          "92",
          "93",
          "94",
          "95",
          "96",
          "97",
          "98",
          "99",
          "100",
          "101",
          "102",
          "103",
          "104",
          "105",
          "106",
          "107",
          "108",
          "109",
          "110",
          "111",
          "112",
          "113",
          "114",
          "115",
          "116",
          "117",
          "118",
          "119",
          "120",
          "121",
          "122",
          "123",
          "124",
          "125",
          "126",
          "127",
          "128",
          "129",
          "130",
          "131",
          "132",
          "133",
          "134",
          "135",
          "136",
          "137",
          "138",
          "139",
          "160",
          "161",
          "162",
          "163",
          "164",
          "165",
          "166",
          "167",
          "168",
          "169",
          "170",
          "171",
          "172",
          "173",
          "174",
          "175",
          "176",
          "177",
          "178",
          "179",
          "180",
          "181",
          "182",
          "183",
          "184",
          "185",
          "186",
          "187",
          "188",
          "189",
          "190",
          "191",
          "192",
          "193",
          "194",
          "195",
          "196",
          "197",
          "198",
          "199",
          "200",
          "201",
          "202",
          "203",
          "204",
          "205",
          "206",
          "207",
          "208",
          "209",
          "210",
          "211",
          "212",
          "213",
          "214",
          "215",
          "216",
          "217",
          "218",
          "219",
          "220",
          "221",
          "222",
          "223",
          "224",
          "225",
          "226",
          "227",
          "228",
          "229",
          "230",
          "231",
          "232",
          "233",
          "234",
          "235",
          "236",
          "237",
          "238",
          "239",
          "240",
          "241",
          "242",
          "243",
          "244",
          "245",
          "246",
          "247",
          "248",
          "249",
          "250",
          "251",
          "252",
          "253",
          "254",
          "255",
          "256",
          "257",
          "258",
          "259",
          "260",
          "261",
          "262",
          "263",
          "264",
          "265",
          "266",
          "267",
          "268",
          "269",
          "270",
          "271",
          "272",
          "273",
          "274",
          "275",
          "276",
          "277",
          "278",
          "279",
          "280",
          "281",
          "282",
          "283",
          "284",
          "285",
          "286",
          "287",
          "288",
          "289",
          "290",
          "291",
          "292",
          "293",
          "294",
          "295",
          "296",
          "297",
          "298",
          "299",
          "300",
          "301",
          "302",
          "303",
          "304",
          "305",
          "306",
          "307",
          "308",
          "309",
          "310",
          "311",
          "312",
          "313",
          "314",
          "315",
          "316",
          "317",
          "318",
          "319",
          "320",
          "321",
          "322",
          "323",
          "324",
          "325",
          "326",
          "327",
          "328",
          "329",
          "330",
          "331",
          "332",
          "333",
          "334",
          "335",
          "336",
          "337",
          "338",
          "339",
          "340",
          "341",
          "342",
          "343",
          "344",
          "345",
          "346",
          "347",
          "348",
          "349",
          "350",
          "351",
          "352",
          "353",
          "354",
          "355",
          "356",
          "357",
          "358",
          "359",
          "360",
          "361",
          "362",
          "363",
          "364",
          "365",
          "366",
          "367",
          "368",
          "369",
          "370",
          "371",
          "372",
          "373",
          "374",
          "375",
          "376",
          "377",
          "378",
          "379",
          "380",
          "381",
          "382",
          "383",
          "384",
          "385",
          "386",
          "387",
          "388",
          "389",
          "390",
          "391",
          "392",
          "393",
          "394",
          "395",
          "396",
          "397",
          "398",
          "399",
          "400",
          "401",
          "402",
          "403",
          "404",
          "405",
          "406",
          "407",
          "408",
          "409",
          "410",
          "411",
          "412",
          "413",
          "414",
          "415",
          "416",
          "417",
          "418",
          "419",
          "420",
          "421",
          "422",
          "423",
          "424",
          "425",
          "426",
          "427",
          "428",
          "429",
          "430",
          "431",
          "432",
          "433",
          "434",
          "435",
          "436",
          "437",
          "438",
          "439",
          "440",
          "441",
          "442",
          "443",
          "444",
          "445",
          "446",
          "447",
          "448",
          "449",
          "450",
          "451",
          "452",
          "453",
          "454",
          "455",
          "456",
          "457",
          "458",
          "459"
        ],
        "request_count": 396,
        "prefill_input_tokens": 46221,
        "selected_expert_events": 21446544,
        "theoretical_dispatch_bytes": 175690088448,
        "theoretical_combine_bytes": 175690088448,
        "local_dispatch_bytes_excluded": 5488607232,
        "local_combine_bytes_excluded": 5488607232,
        "remote_dispatch_bytes_retained": 170201481216,
        "remote_combine_bytes_retained": 170201481216,
        "byte_conservation_pass": true,
        "dispatch_checksum": "71fd5c96d15656316bd4f2ceb5c42a4759079c95fc1f47a8161b949c514a1655",
        "combine_checksum": "5bc297bba213fb9023b62a4232807ecce3397e65429d81de591465ab1a81c2d6"
      }
    },
    "split": {
      "calibration_request_count": 44,
      "evaluation_request_count": 396,
      "calibration_request_ids": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
        "40",
        "41",
        "42",
        "43"
      ],
      "evaluation_request_ids": [
        "44",
        "45",
        "46",
        "47",
        "48",
        "49",
        "50",
        "51",
        "52",
        "53",
        "54",
        "55",
        "56",
        "57",
        "58",
        "59",
        "60",
        "61",
        "62",
        "63",
        "64",
        "65",
        "66",
        "67",
        "68",
        "69",
        "70",
        "71",
        "72",
        "73",
        "74",
        "75",
        "76",
        "77",
        "78",
        "79",
        "80",
        "81",
        "82",
        "83",
        "84",
        "85",
        "86",
        "87",
        "88",
        "89",
        "90",
        "91",
        "92",
        "93",
        "94",
        "95",
        "96",
        "97",
        "98",
        "99",
        "100",
        "101",
        "102",
        "103",
        "104",
        "105",
        "106",
        "107",
        "108",
        "109",
        "110",
        "111",
        "112",
        "113",
        "114",
        "115",
        "116",
        "117",
        "118",
        "119",
        "120",
        "121",
        "122",
        "123",
        "124",
        "125",
        "126",
        "127",
        "128",
        "129",
        "130",
        "131",
        "132",
        "133",
        "134",
        "135",
        "136",
        "137",
        "138",
        "139",
        "160",
        "161",
        "162",
        "163",
        "164",
        "165",
        "166",
        "167",
        "168",
        "169",
        "170",
        "171",
        "172",
        "173",
        "174",
        "175",
        "176",
        "177",
        "178",
        "179",
        "180",
        "181",
        "182",
        "183",
        "184",
        "185",
        "186",
        "187",
        "188",
        "189",
        "190",
        "191",
        "192",
        "193",
        "194",
        "195",
        "196",
        "197",
        "198",
        "199",
        "200",
        "201",
        "202",
        "203",
        "204",
        "205",
        "206",
        "207",
        "208",
        "209",
        "210",
        "211",
        "212",
        "213",
        "214",
        "215",
        "216",
        "217",
        "218",
        "219",
        "220",
        "221",
        "222",
        "223",
        "224",
        "225",
        "226",
        "227",
        "228",
        "229",
        "230",
        "231",
        "232",
        "233",
        "234",
        "235",
        "236",
        "237",
        "238",
        "239",
        "240",
        "241",
        "242",
        "243",
        "244",
        "245",
        "246",
        "247",
        "248",
        "249",
        "250",
        "251",
        "252",
        "253",
        "254",
        "255",
        "256",
        "257",
        "258",
        "259",
        "260",
        "261",
        "262",
        "263",
        "264",
        "265",
        "266",
        "267",
        "268",
        "269",
        "270",
        "271",
        "272",
        "273",
        "274",
        "275",
        "276",
        "277",
        "278",
        "279",
        "280",
        "281",
        "282",
        "283",
        "284",
        "285",
        "286",
        "287",
        "288",
        "289",
        "290",
        "291",
        "292",
        "293",
        "294",
        "295",
        "296",
        "297",
        "298",
        "299",
        "300",
        "301",
        "302",
        "303",
        "304",
        "305",
        "306",
        "307",
        "308",
        "309",
        "310",
        "311",
        "312",
        "313",
        "314",
        "315",
        "316",
        "317",
        "318",
        "319",
        "320",
        "321",
        "322",
        "323",
        "324",
        "325",
        "326",
        "327",
        "328",
        "329",
        "330",
        "331",
        "332",
        "333",
        "334",
        "335",
        "336",
        "337",
        "338",
        "339",
        "340",
        "341",
        "342",
        "343",
        "344",
        "345",
        "346",
        "347",
        "348",
        "349",
        "350",
        "351",
        "352",
        "353",
        "354",
        "355",
        "356",
        "357",
        "358",
        "359",
        "360",
        "361",
        "362",
        "363",
        "364",
        "365",
        "366",
        "367",
        "368",
        "369",
        "370",
        "371",
        "372",
        "373",
        "374",
        "375",
        "376",
        "377",
        "378",
        "379",
        "380",
        "381",
        "382",
        "383",
        "384",
        "385",
        "386",
        "387",
        "388",
        "389",
        "390",
        "391",
        "392",
        "393",
        "394",
        "395",
        "396",
        "397",
        "398",
        "399",
        "400",
        "401",
        "402",
        "403",
        "404",
        "405",
        "406",
        "407",
        "408",
        "409",
        "410",
        "411",
        "412",
        "413",
        "414",
        "415",
        "416",
        "417",
        "418",
        "419",
        "420",
        "421",
        "422",
        "423",
        "424",
        "425",
        "426",
        "427",
        "428",
        "429",
        "430",
        "431",
        "432",
        "433",
        "434",
        "435",
        "436",
        "437",
        "438",
        "439",
        "440",
        "441",
        "442",
        "443",
        "444",
        "445",
        "446",
        "447",
        "448",
        "449",
        "450",
        "451",
        "452",
        "453",
        "454",
        "455",
        "456",
        "457",
        "458",
        "459"
      ],
      "calibration_rule": "front ceil(10%) requests"
    },
    "anti_leakage": {
      "calibrated_uses_calibration_only": true,
      "oracle_uses_evaluation_only": true,
      "oracle_reference_only": true
    },
    "traffic_fingerprint": {
      "calibration_dispatch": {
        "total_remote_bytes": 15847677952,
        "nonzero_gpu_pairs": 992,
        "top1_share": 0.001244746142605107,
        "top4_share": 0.004947969301086414,
        "top8_share": 0.009827188088482429,
        "top16_share": 0.019473453772516438,
        "gini": 0.04654819405590362,
        "entropy_bits": 9.94938649210314,
        "message_bytes_min": 12828672,
        "message_bytes_median": 15863808.0,
        "message_bytes_mean": 15975481.806451613,
        "message_bytes_max": 19726336,
        "interpretation": "broad / near-uniform"
      },
      "calibration_combine": {
        "total_remote_bytes": 15847677952,
        "nonzero_gpu_pairs": 992,
        "top1_share": 0.001244746142605107,
        "top4_share": 0.004947969301086414,
        "top8_share": 0.009827188088482429,
        "top16_share": 0.019473453772516438,
        "gini": 0.04654819405590362,
        "entropy_bits": 9.949386492103141,
        "message_bytes_min": 12828672,
        "message_bytes_median": 15863808.0,
        "message_bytes_mean": 15975481.806451613,
        "message_bytes_max": 19726336,
        "interpretation": "broad / near-uniform"
      },
      "evaluation_dispatch": {
        "total_remote_bytes": 170201481216,
        "nonzero_gpu_pairs": 992,
        "top1_share": 0.00110961647719342,
        "top4_share": 0.0044225826157454065,
        "top8_share": 0.008816912222376884,
        "top16_share": 0.01756774031951795,
        "gini": 0.024928475575193662,
        "entropy_bits": 9.952832047187371,
        "message_bytes_min": 153370624,
        "message_bytes_median": 171737088.0,
        "message_bytes_mean": 171574073.80645162,
        "message_bytes_max": 188858368,
        "interpretation": "broad / near-uniform"
      },
      "evaluation_combine": {
        "total_remote_bytes": 170201481216,
        "nonzero_gpu_pairs": 992,
        "top1_share": 0.00110961647719342,
        "top4_share": 0.0044225826157454065,
        "top8_share": 0.008816912222376884,
        "top16_share": 0.01756774031951795,
        "gini": 0.024928475575193662,
        "entropy_bits": 9.952832047187368,
        "message_bytes_min": 153370624,
        "message_bytes_median": 171737088.0,
        "message_bytes_mean": 171574073.80645162,
        "message_bytes_max": 188858368,
        "interpretation": "broad / near-uniform"
      }
    },
    "candidate_pool": {
      "candidate_count": 35,
      "random_candidate_count": 32,
      "candidate_types": {
        "greedy_calibration": 1,
        "greedy_evaluation": 1,
        "random_regular": 32,
        "torus": 1
      },
      "graph_budget_pass": true,
      "all_candidate_graphs_valid": true
    },
    "candidate_scores": {
      "calibration_top12": [
        {
          "name": "random_regular_seed_22",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 435348139
        },
        {
          "name": "random_regular_seed_16",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 435776853
        },
        {
          "name": "random_regular_seed_20",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 453698219
        },
        {
          "name": "random_regular_seed_27",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 462367404
        },
        {
          "name": "random_regular_seed_30",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 463890432
        },
        {
          "name": "random_regular_seed_25",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 465528150
        },
        {
          "name": "random_regular_seed_19",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 471969791
        },
        {
          "name": "random_regular_seed_18",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 475742891
        },
        {
          "name": "random_regular_seed_29",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 476885674
        },
        {
          "name": "random_regular_seed_3",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 477660501
        },
        {
          "name": "random_regular_seed_9",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 479976789
        },
        {
          "name": "random_regular_seed_31",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 480047104
        }
      ],
      "evaluation_top12": [
        {
          "name": "random_regular_seed_22",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 4546917035
        },
        {
          "name": "random_regular_seed_16",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 4738105344
        },
        {
          "name": "random_regular_seed_30",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 4797622272
        },
        {
          "name": "random_regular_seed_20",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 4804684460
        },
        {
          "name": "random_regular_seed_25",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 4912482986
        },
        {
          "name": "random_regular_seed_3",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 4944676180
        },
        {
          "name": "random_regular_seed_27",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 4979590485
        },
        {
          "name": "evaluation_greedy",
          "candidate_type": "greedy_evaluation",
          "score_max_link_load_bytes": 4989766315
        },
        {
          "name": "random_regular_seed_9",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5021302784
        },
        {
          "name": "random_regular_seed_13",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5036135767
        },
        {
          "name": "random_regular_seed_18",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5076097708
        },
        {
          "name": "random_regular_seed_31",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5121055402
        }
      ],
      "evaluation_all": [
        {
          "name": "random_regular_seed_22",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 4546917035
        },
        {
          "name": "random_regular_seed_16",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 4738105344
        },
        {
          "name": "random_regular_seed_30",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 4797622272
        },
        {
          "name": "random_regular_seed_20",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 4804684460
        },
        {
          "name": "random_regular_seed_25",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 4912482986
        },
        {
          "name": "random_regular_seed_3",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 4944676180
        },
        {
          "name": "random_regular_seed_27",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 4979590485
        },
        {
          "name": "evaluation_greedy",
          "candidate_type": "greedy_evaluation",
          "score_max_link_load_bytes": 4989766315
        },
        {
          "name": "random_regular_seed_9",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5021302784
        },
        {
          "name": "random_regular_seed_13",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5036135767
        },
        {
          "name": "random_regular_seed_18",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5076097708
        },
        {
          "name": "random_regular_seed_31",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5121055402
        },
        {
          "name": "random_regular_seed_29",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5188706985
        },
        {
          "name": "random_regular_seed_19",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5201433940
        },
        {
          "name": "random_regular_seed_17",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5257287681
        },
        {
          "name": "random_regular_seed_23",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5283319810
        },
        {
          "name": "random_regular_seed_0",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5292974763
        },
        {
          "name": "random_regular_seed_24",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5407435434
        },
        {
          "name": "random_regular_seed_12",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5443044693
        },
        {
          "name": "random_regular_seed_2",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5463958869
        },
        {
          "name": "random_regular_seed_11",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5618401962
        },
        {
          "name": "random_regular_seed_6",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5639548248
        },
        {
          "name": "random_regular_seed_4",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5734491479
        },
        {
          "name": "random_regular_seed_14",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5770833919
        },
        {
          "name": "random_regular_seed_7",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5827568981
        },
        {
          "name": "random_regular_seed_5",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5860470104
        },
        {
          "name": "random_regular_seed_8",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5888183635
        },
        {
          "name": "random_regular_seed_1",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 5890140842
        },
        {
          "name": "random_regular_seed_15",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 6069186559
        },
        {
          "name": "random_regular_seed_10",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 6076104021
        },
        {
          "name": "random_regular_seed_21",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 6272271701
        },
        {
          "name": "random_regular_seed_26",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 6483965952
        },
        {
          "name": "calibration_greedy",
          "candidate_type": "greedy_calibration",
          "score_max_link_load_bytes": 7236025003
        },
        {
          "name": "random_regular_seed_28",
          "candidate_type": "random_regular",
          "score_max_link_load_bytes": 7975064235
        },
        {
          "name": "son_torus",
          "candidate_type": "torus",
          "score_max_link_load_bytes": 9124854445
        }
      ]
    },
    "selected": {
      "ron_calibrated": "random_regular_seed_22",
      "ron_calibrated_type": "random_regular",
      "ron_oracle": "random_regular_seed_22",
      "ron_oracle_type": "random_regular",
      "fixed_random": "random_regular_seed_0",
      "median_random": "random_regular_seed_24",
      "best_random": "random_regular_seed_22",
      "representatives_run_in_astra": [
        "son_torus",
        "random_regular_seed_0",
        "random_regular_seed_24",
        "random_regular_seed_22",
        "calibration_greedy",
        "evaluation_greedy"
      ],
      "roles": {
        "son_torus": "son",
        "random_regular_seed_0": "fixed_random",
        "random_regular_seed_24": "median_random",
        "random_regular_seed_22": "ron_oracle",
        "calibration_greedy": "greedy_calibration",
        "evaluation_greedy": "greedy_evaluation"
      }
    },
    "candidate_audits_all_budget_only": {
      "son_torus": {
        "name": "son_torus_4x8_32gpu",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "calibration_greedy": {
        "name": "ron_calibration_greedy_degree4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "evaluation_greedy": {
        "name": "ron_evaluation_greedy_degree4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_0": {
        "name": "ron_random_regular_seed_0",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_1": {
        "name": "ron_random_regular_seed_1",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_2": {
        "name": "ron_random_regular_seed_2",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_3": {
        "name": "ron_random_regular_seed_3",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_4": {
        "name": "ron_random_regular_seed_4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_5": {
        "name": "ron_random_regular_seed_5",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_6": {
        "name": "ron_random_regular_seed_6",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_7": {
        "name": "ron_random_regular_seed_7",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_8": {
        "name": "ron_random_regular_seed_8",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_9": {
        "name": "ron_random_regular_seed_9",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_10": {
        "name": "ron_random_regular_seed_10",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_11": {
        "name": "ron_random_regular_seed_11",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_12": {
        "name": "ron_random_regular_seed_12",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_13": {
        "name": "ron_random_regular_seed_13",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_14": {
        "name": "ron_random_regular_seed_14",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_15": {
        "name": "ron_random_regular_seed_15",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_16": {
        "name": "ron_random_regular_seed_16",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_17": {
        "name": "ron_random_regular_seed_17",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_18": {
        "name": "ron_random_regular_seed_18",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_19": {
        "name": "ron_random_regular_seed_19",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_20": {
        "name": "ron_random_regular_seed_20",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_21": {
        "name": "ron_random_regular_seed_21",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_22": {
        "name": "ron_random_regular_seed_22",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_23": {
        "name": "ron_random_regular_seed_23",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_24": {
        "name": "ron_random_regular_seed_24",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_25": {
        "name": "ron_random_regular_seed_25",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_26": {
        "name": "ron_random_regular_seed_26",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_27": {
        "name": "ron_random_regular_seed_27",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_28": {
        "name": "ron_random_regular_seed_28",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_29": {
        "name": "ron_random_regular_seed_29",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_30": {
        "name": "ron_random_regular_seed_30",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      },
      "random_regular_seed_31": {
        "name": "ron_random_regular_seed_31",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true
      }
    },
    "candidate_audits_representatives": {
      "son_torus": {
        "name": "son_torus_4x8_32gpu",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 3.096774193548387,
        "diameter": 6,
        "ecmp_path_count_distribution_cap4": {
          "1": 256,
          "2": 192,
          "3": 128,
          "4": 416
        }
      },
      "random_regular_seed_0": {
        "name": "ron_random_regular_seed_0",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.560483870967742,
        "diameter": 4,
        "ecmp_path_count_distribution_cap4": {
          "1": 590,
          "2": 200,
          "3": 84,
          "4": 118
        }
      },
      "random_regular_seed_24": {
        "name": "ron_random_regular_seed_24",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.568548387096774,
        "diameter": 5,
        "ecmp_path_count_distribution_cap4": {
          "1": 646,
          "2": 216,
          "3": 76,
          "4": 54
        }
      },
      "random_regular_seed_22": {
        "name": "ron_random_regular_seed_22",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.5,
        "diameter": 4,
        "ecmp_path_count_distribution_cap4": {
          "1": 600,
          "2": 230,
          "3": 66,
          "4": 96
        }
      },
      "calibration_greedy": {
        "name": "ron_calibration_greedy_degree4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.6975806451612905,
        "diameter": 5,
        "ecmp_path_count_distribution_cap4": {
          "1": 578,
          "2": 214,
          "3": 104,
          "4": 96
        }
      },
      "evaluation_greedy": {
        "name": "ron_evaluation_greedy_degree4",
        "edge_circuit_count": 64,
        "degree_distribution": {
          "4": 32
        },
        "connected_components": 1,
        "duplicate_edges": 0,
        "self_loops": 0,
        "valid": true,
        "same_degree_bandwidth_budget_as_son": true,
        "average_shortest_path_length": 2.504032258064516,
        "diameter": 4,
        "ecmp_path_count_distribution_cap4": {
          "1": 684,
          "2": 174,
          "3": 62,
          "4": 72
        }
      }
    },
    "native_astra_results": {
      "dispatch": {
        "son_torus": {
          "label": "deepseek_livecodebench_execution_dispatch_son_torus",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.04747487499844283,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/network_configs/son_torus.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_dispatch_son_torus.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_dispatch_son_torus.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 363800546,
          "cycles_count": 32
        },
        "random_regular_seed_0": {
          "label": "deepseek_livecodebench_execution_dispatch_random_regular_seed_0",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.026111582992598414,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/network_configs/random_regular_seed_0.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_dispatch_random_regular_seed_0.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_dispatch_random_regular_seed_0.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 297948127,
          "cycles_count": 32
        },
        "random_regular_seed_24": {
          "label": "deepseek_livecodebench_execution_dispatch_random_regular_seed_24",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.024991833022795618,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/network_configs/random_regular_seed_24.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_dispatch_random_regular_seed_24.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_dispatch_random_regular_seed_24.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 301499665,
          "cycles_count": 32
        },
        "random_regular_seed_22": {
          "label": "deepseek_livecodebench_execution_dispatch_random_regular_seed_22",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.024164333008229733,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/network_configs/random_regular_seed_22.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_dispatch_random_regular_seed_22.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_dispatch_random_regular_seed_22.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 286605502,
          "cycles_count": 32
        },
        "calibration_greedy": {
          "label": "deepseek_livecodebench_execution_dispatch_calibration_greedy",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.024499166989699006,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/network_configs/calibration_greedy.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_dispatch_calibration_greedy.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_dispatch_calibration_greedy.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 348205362,
          "cycles_count": 32
        },
        "evaluation_greedy": {
          "label": "deepseek_livecodebench_execution_dispatch_evaluation_greedy",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.02358291600830853,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/chakra_traces/evaluation_dispatch/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/network_configs/evaluation_greedy.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_dispatch_evaluation_greedy.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_dispatch_evaluation_greedy.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 331771221,
          "cycles_count": 32
        }
      },
      "combine": {
        "son_torus": {
          "label": "deepseek_livecodebench_execution_combine_son_torus",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.026692875078879297,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/network_configs/son_torus.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_combine_son_torus.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_combine_son_torus.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 369021058,
          "cycles_count": 32
        },
        "random_regular_seed_0": {
          "label": "deepseek_livecodebench_execution_combine_random_regular_seed_0",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.023799958056770265,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/network_configs/random_regular_seed_0.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_combine_random_regular_seed_0.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_combine_random_regular_seed_0.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 297975086,
          "cycles_count": 32
        },
        "random_regular_seed_24": {
          "label": "deepseek_livecodebench_execution_combine_random_regular_seed_24",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.02420587500091642,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/network_configs/random_regular_seed_24.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_combine_random_regular_seed_24.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_combine_random_regular_seed_24.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 305048185,
          "cycles_count": 32
        },
        "random_regular_seed_22": {
          "label": "deepseek_livecodebench_execution_combine_random_regular_seed_22",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.023973042028956115,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/network_configs/random_regular_seed_22.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_combine_random_regular_seed_22.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_combine_random_regular_seed_22.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 300663382,
          "cycles_count": 32
        },
        "calibration_greedy": {
          "label": "deepseek_livecodebench_execution_combine_calibration_greedy",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.024191166972741485,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/network_configs/calibration_greedy.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_combine_calibration_greedy.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_combine_calibration_greedy.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 346846310,
          "cycles_count": 32
        },
        "evaluation_greedy": {
          "label": "deepseek_livecodebench_execution_combine_evaluation_greedy",
          "returncode": 0,
          "success": true,
          "runtime_s": 0.02454141597263515,
          "command": "/Users/dfx/Python/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware --workload-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/chakra_traces/evaluation_combine/workload --system-configuration=/Users/dfx/Python/astra-sim/examples/system/native_collectives/Ring_4chunks.json --remote-memory-configuration=/Users/dfx/Python/astra-sim/examples/remote_memory/analytical/no_memory_expansion.json --network-configuration=/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/workloads/deepseek_livecodebench_execution/network_configs/evaluation_greedy.yml",
          "stdout": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_combine_evaluation_greedy.stdout.txt",
          "stderr": "/Users/dfx/Python/astra-sim/results/moe_expert_trace_converter/v37_four_workload_static_candidate_family_validation/runs/deepseek_livecodebench_execution_combine_evaluation_greedy.stderr.txt",
          "stderr_tail": [],
          "max_cycles": 344468372,
          "cycles_count": 32
        }
      }
    },
    "native_astra_totals": {
      "son_torus": {
        "candidate_type": "torus",
        "dispatch_cycles": 363800546,
        "combine_cycles": 369021058,
        "total_cycles": 732821604,
        "dispatch_fluid_cycles": 169659385,
        "combine_fluid_cycles": 169963658,
        "total_fluid_cycles": 339623043,
        "astra_over_fluid_total": 2.157749949846601,
        "success": true,
        "runtime_s": 0.07416775007732213,
        "role": "son"
      },
      "random_regular_seed_0": {
        "candidate_type": "random_regular",
        "dispatch_cycles": 297948127,
        "combine_cycles": 297975086,
        "total_cycles": 595923213,
        "dispatch_fluid_cycles": 96990254,
        "combine_fluid_cycles": 98589337,
        "total_fluid_cycles": 195579591,
        "astra_over_fluid_total": 3.0469601145653282,
        "success": true,
        "runtime_s": 0.04991154104936868,
        "role": "fixed_random"
      },
      "random_regular_seed_24": {
        "candidate_type": "random_regular",
        "dispatch_cycles": 301499665,
        "combine_cycles": 305048185,
        "total_cycles": 606547850,
        "dispatch_fluid_cycles": 100625584,
        "combine_fluid_cycles": 100721333,
        "total_fluid_cycles": 201346917,
        "astra_over_fluid_total": 3.0124516383829207,
        "success": true,
        "runtime_s": 0.04919770802371204,
        "role": "median_random"
      },
      "random_regular_seed_22": {
        "candidate_type": "random_regular",
        "dispatch_cycles": 286605502,
        "combine_cycles": 300663382,
        "total_cycles": 587268884,
        "dispatch_fluid_cycles": 84692929,
        "combine_fluid_cycles": 82963574,
        "total_fluid_cycles": 167656503,
        "astra_over_fluid_total": 3.5028100520502923,
        "success": true,
        "runtime_s": 0.04813737503718585,
        "role": "ron_oracle"
      },
      "calibration_greedy": {
        "candidate_type": "greedy_calibration",
        "dispatch_cycles": 348205362,
        "combine_cycles": 346846310,
        "total_cycles": 695051672,
        "dispatch_fluid_cycles": 134781468,
        "combine_fluid_cycles": 133097712,
        "total_fluid_cycles": 267879180,
        "astra_over_fluid_total": 2.5946461087420083,
        "success": true,
        "runtime_s": 0.04869033396244049,
        "role": "greedy_calibration"
      },
      "evaluation_greedy": {
        "candidate_type": "greedy_evaluation",
        "dispatch_cycles": 331771221,
        "combine_cycles": 344468372,
        "total_cycles": 676239593,
        "dispatch_fluid_cycles": 92724444,
        "combine_fluid_cycles": 92941640,
        "total_fluid_cycles": 185666084,
        "astra_over_fluid_total": 3.6422354499597245,
        "success": true,
        "runtime_s": 0.04812433198094368,
        "role": "greedy_evaluation"
      }
    },
    "fluid_lower_bound": {
      "dispatch": {
        "son_torus": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 3.0,
          "selected_path_count_mean": 2.7096774193548385,
          "selected_path_count_max": 4,
          "average_hop_count": 3.5952380952380953,
          "byte_weighted_average_hop_count": 3.096834373063321,
          "max_link_load_bytes": 9108518916,
          "median_link_load_bytes": 3880822784.0,
          "average_link_load_bytes": 4117857792,
          "fluid_cycles": 169659385,
          "hot_links": [
            {
              "src": 3,
              "dst": 2,
              "bytes": 9108518916
            },
            {
              "src": 1,
              "dst": 0,
              "bytes": 9097055576
            },
            {
              "src": 2,
              "dst": 1,
              "bytes": 8968181763
            },
            {
              "src": 4,
              "dst": 3,
              "bytes": 8519285422
            },
            {
              "src": 0,
              "dst": 7,
              "bytes": 8173845165
            },
            {
              "src": 5,
              "dst": 4,
              "bytes": 7860413784
            },
            {
              "src": 1,
              "dst": 2,
              "bytes": 7726289581
            },
            {
              "src": 0,
              "dst": 1,
              "bytes": 7702715736
            }
          ]
        },
        "random_regular_seed_0": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7278225806451613,
          "selected_path_count_max": 4,
          "average_hop_count": 2.911318553092182,
          "byte_weighted_average_hop_count": 2.5612003014167706,
          "max_link_load_bytes": 5207124652,
          "median_link_load_bytes": 3377119232.0,
          "average_link_load_bytes": 3405625664,
          "fluid_cycles": 96990254,
          "hot_links": [
            {
              "src": 1,
              "dst": 19,
              "bytes": 5207124652
            },
            {
              "src": 19,
              "dst": 1,
              "bytes": 5107158358
            },
            {
              "src": 11,
              "dst": 27,
              "bytes": 5101848577
            },
            {
              "src": 22,
              "dst": 24,
              "bytes": 5096494422
            },
            {
              "src": 11,
              "dst": 12,
              "bytes": 4949385898
            },
            {
              "src": 12,
              "dst": 11,
              "bytes": 4937795584
            },
            {
              "src": 27,
              "dst": 11,
              "bytes": 4919046143
            },
            {
              "src": 24,
              "dst": 22,
              "bytes": 4824119297
            }
          ]
        },
        "random_regular_seed_24": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.534274193548387,
          "selected_path_count_max": 4,
          "average_hop_count": 2.889618922470434,
          "byte_weighted_average_hop_count": 2.5683138989210335,
          "max_link_load_bytes": 5402294954,
          "median_link_load_bytes": 3347645782.5,
          "average_link_load_bytes": 3415084608,
          "fluid_cycles": 100625584,
          "hot_links": [
            {
              "src": 23,
              "dst": 2,
              "bytes": 5402294954
            },
            {
              "src": 27,
              "dst": 3,
              "bytes": 5374564350
            },
            {
              "src": 2,
              "dst": 23,
              "bytes": 5340962133
            },
            {
              "src": 3,
              "dst": 27,
              "bytes": 5327891116
            },
            {
              "src": 16,
              "dst": 19,
              "bytes": 5285640874
            },
            {
              "src": 10,
              "dst": 12,
              "bytes": 5101086721
            },
            {
              "src": 25,
              "dst": 22,
              "bytes": 5059069952
            },
            {
              "src": 8,
              "dst": 7,
              "bytes": 5046386689
            }
          ]
        },
        "random_regular_seed_22": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.655241935483871,
          "selected_path_count_max": 4,
          "average_hop_count": 2.8270401948842876,
          "byte_weighted_average_hop_count": 2.4995278330163413,
          "max_link_load_bytes": 4546917035,
          "median_link_load_bytes": 3310818645.5,
          "average_link_load_bytes": 3323619840,
          "fluid_cycles": 84692929,
          "hot_links": [
            {
              "src": 20,
              "dst": 21,
              "bytes": 4546917035
            },
            {
              "src": 5,
              "dst": 10,
              "bytes": 4540666539
            },
            {
              "src": 17,
              "dst": 6,
              "bytes": 4368040620
            },
            {
              "src": 6,
              "dst": 17,
              "bytes": 4333776215
            },
            {
              "src": 18,
              "dst": 1,
              "bytes": 4332105728
            },
            {
              "src": 1,
              "dst": 18,
              "bytes": 4243129003
            },
            {
              "src": 10,
              "dst": 5,
              "bytes": 4229470208
            },
            {
              "src": 21,
              "dst": 20,
              "bytes": 4162626901
            }
          ]
        },
        "calibration_greedy": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.715725806451613,
          "selected_path_count_max": 4,
          "average_hop_count": 3.016451233842538,
          "byte_weighted_average_hop_count": 2.697353236928483,
          "max_link_load_bytes": 7236025003,
          "median_link_load_bytes": 3593529685.5,
          "average_link_load_bytes": 3586668096,
          "fluid_cycles": 134781468,
          "hot_links": [
            {
              "src": 17,
              "dst": 7,
              "bytes": 7236025003
            },
            {
              "src": 31,
              "dst": 20,
              "bytes": 6985945086
            },
            {
              "src": 7,
              "dst": 17,
              "bytes": 6890403158
            },
            {
              "src": 20,
              "dst": 31,
              "bytes": 6641945256
            },
            {
              "src": 16,
              "dst": 17,
              "bytes": 6563100672
            },
            {
              "src": 17,
              "dst": 16,
              "bytes": 6201556310
            },
            {
              "src": 29,
              "dst": 30,
              "bytes": 6116326741
            },
            {
              "src": 8,
              "dst": 30,
              "bytes": 6101835093
            }
          ]
        },
        "evaluation_greedy": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.5181451612903225,
          "selected_path_count_max": 4,
          "average_hop_count": 2.814077025232404,
          "byte_weighted_average_hop_count": 2.504112088302638,
          "max_link_load_bytes": 4978105685,
          "median_link_load_bytes": 3311057578.0,
          "average_link_load_bytes": 3329715520,
          "fluid_cycles": 92724444,
          "hot_links": [
            {
              "src": 8,
              "dst": 3,
              "bytes": 4978105685
            },
            {
              "src": 3,
              "dst": 8,
              "bytes": 4904333994
            },
            {
              "src": 25,
              "dst": 3,
              "bytes": 4554513749
            },
            {
              "src": 3,
              "dst": 25,
              "bytes": 4513295701
            },
            {
              "src": 1,
              "dst": 2,
              "bytes": 4499493547
            },
            {
              "src": 22,
              "dst": 23,
              "bytes": 4465903615
            },
            {
              "src": 23,
              "dst": 22,
              "bytes": 4449945601
            },
            {
              "src": 6,
              "dst": 7,
              "bytes": 4433630549
            }
          ]
        }
      },
      "combine": {
        "son_torus": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 3.0,
          "selected_path_count_mean": 2.7096774193548385,
          "selected_path_count_max": 4,
          "average_hop_count": 3.5952380952380953,
          "byte_weighted_average_hop_count": 3.096834373063321,
          "max_link_load_bytes": 9124854445,
          "median_link_load_bytes": 3826614612.5,
          "average_link_load_bytes": 4117857792,
          "fluid_cycles": 169963658,
          "hot_links": [
            {
              "src": 2,
              "dst": 1,
              "bytes": 9124854445
            },
            {
              "src": 1,
              "dst": 0,
              "bytes": 9003031896
            },
            {
              "src": 3,
              "dst": 2,
              "bytes": 8923991383
            },
            {
              "src": 4,
              "dst": 3,
              "bytes": 8519685464
            },
            {
              "src": 0,
              "dst": 7,
              "bytes": 8221136217
            },
            {
              "src": 0,
              "dst": 1,
              "bytes": 7842577752
            },
            {
              "src": 1,
              "dst": 2,
              "bytes": 7621527555
            },
            {
              "src": 5,
              "dst": 4,
              "bytes": 7565636952
            }
          ]
        },
        "random_regular_seed_0": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7278225806451613,
          "selected_path_count_max": 4,
          "average_hop_count": 2.911318553092182,
          "byte_weighted_average_hop_count": 2.5612003014167706,
          "max_link_load_bytes": 5292974763,
          "median_link_load_bytes": 3432089940.0,
          "average_link_load_bytes": 3405625664,
          "fluid_cycles": 98589337,
          "hot_links": [
            {
              "src": 19,
              "dst": 1,
              "bytes": 5292974763
            },
            {
              "src": 11,
              "dst": 27,
              "bytes": 5047908353
            },
            {
              "src": 12,
              "dst": 11,
              "bytes": 5037361835
            },
            {
              "src": 1,
              "dst": 19,
              "bytes": 5029545302
            },
            {
              "src": 24,
              "dst": 22,
              "bytes": 5004395862
            },
            {
              "src": 27,
              "dst": 11,
              "bytes": 4946499583
            },
            {
              "src": 22,
              "dst": 24,
              "bytes": 4913348609
            },
            {
              "src": 11,
              "dst": 12,
              "bytes": 4832395263
            }
          ]
        },
        "random_regular_seed_24": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.534274193548387,
          "selected_path_count_max": 4,
          "average_hop_count": 2.889618922470434,
          "byte_weighted_average_hop_count": 2.5683138989210335,
          "max_link_load_bytes": 5407435434,
          "median_link_load_bytes": 3356880554.0,
          "average_link_load_bytes": 3415084608,
          "fluid_cycles": 100721333,
          "hot_links": [
            {
              "src": 27,
              "dst": 3,
              "bytes": 5407435434
            },
            {
              "src": 2,
              "dst": 23,
              "bytes": 5402673835
            },
            {
              "src": 23,
              "dst": 2,
              "bytes": 5342973270
            },
            {
              "src": 3,
              "dst": 27,
              "bytes": 5291358208
            },
            {
              "src": 19,
              "dst": 16,
              "bytes": 5208113835
            },
            {
              "src": 10,
              "dst": 12,
              "bytes": 5138432001
            },
            {
              "src": 16,
              "dst": 19,
              "bytes": 4994020692
            },
            {
              "src": 12,
              "dst": 10,
              "bytes": 4971339777
            }
          ]
        },
        "random_regular_seed_22": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.655241935483871,
          "selected_path_count_max": 4,
          "average_hop_count": 2.8270401948842876,
          "byte_weighted_average_hop_count": 2.4995278330163413,
          "max_link_load_bytes": 4454073003,
          "median_link_load_bytes": 3288306347.0,
          "average_link_load_bytes": 3323619840,
          "fluid_cycles": 82963574,
          "hot_links": [
            {
              "src": 10,
              "dst": 5,
              "bytes": 4454073003
            },
            {
              "src": 21,
              "dst": 20,
              "bytes": 4400655018
            },
            {
              "src": 17,
              "dst": 6,
              "bytes": 4381295959
            },
            {
              "src": 18,
              "dst": 1,
              "bytes": 4373621418
            },
            {
              "src": 6,
              "dst": 17,
              "bytes": 4327643819
            },
            {
              "src": 5,
              "dst": 10,
              "bytes": 4311361536
            },
            {
              "src": 20,
              "dst": 21,
              "bytes": 4298282324
            },
            {
              "src": 1,
              "dst": 18,
              "bytes": 4195538944
            }
          ]
        },
        "calibration_greedy": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.715725806451613,
          "selected_path_count_max": 4,
          "average_hop_count": 3.016451233842538,
          "byte_weighted_average_hop_count": 2.697353236928483,
          "max_link_load_bytes": 7145629014,
          "median_link_load_bytes": 3560845653.0,
          "average_link_load_bytes": 3586668096,
          "fluid_cycles": 133097712,
          "hot_links": [
            {
              "src": 17,
              "dst": 7,
              "bytes": 7145629014
            },
            {
              "src": 7,
              "dst": 17,
              "bytes": 6970542764
            },
            {
              "src": 31,
              "dst": 20,
              "bytes": 6861392553
            },
            {
              "src": 20,
              "dst": 31,
              "bytes": 6766280701
            },
            {
              "src": 17,
              "dst": 16,
              "bytes": 6605316097
            },
            {
              "src": 30,
              "dst": 29,
              "bytes": 6333179221
            },
            {
              "src": 16,
              "dst": 17,
              "bytes": 6159650132
            },
            {
              "src": 8,
              "dst": 30,
              "bytes": 6157780991
            }
          ]
        },
        "evaluation_greedy": {
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.5181451612903225,
          "selected_path_count_max": 4,
          "average_hop_count": 2.814077025232404,
          "byte_weighted_average_hop_count": 2.504112088302638,
          "max_link_load_bytes": 4989766315,
          "median_link_load_bytes": 3311057577.5,
          "average_link_load_bytes": 3329715520,
          "fluid_cycles": 92941640,
          "hot_links": [
            {
              "src": 8,
              "dst": 3,
              "bytes": 4989766315
            },
            {
              "src": 3,
              "dst": 8,
              "bytes": 4891063636
            },
            {
              "src": 3,
              "dst": 25,
              "bytes": 4599962965
            },
            {
              "src": 2,
              "dst": 1,
              "bytes": 4544586412
            },
            {
              "src": 23,
              "dst": 22,
              "bytes": 4509102080
            },
            {
              "src": 7,
              "dst": 6,
              "bytes": 4474785111
            },
            {
              "src": 25,
              "dst": 3,
              "bytes": 4468233557
            },
            {
              "src": 12,
              "dst": 13,
              "bytes": 4417219241
            }
          ]
        }
      }
    },
    "tiny_subchunk_audit": {
      "dispatch": {
        "son_torus": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 38342656,
          "subchunk_bytes_median": 44808192.0,
          "subchunks_total": 2688,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 3.0,
          "selected_path_count_mean": 2.7096774193548385,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_0": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 38991872,
          "subchunk_bytes_median": 84164608.0,
          "subchunks_total": 1714,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7278225806451613,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_24": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 39176192,
          "subchunk_bytes_median": 88182784.0,
          "subchunks_total": 1522,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.534274193548387,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_22": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 39176192,
          "subchunk_bytes_median": 85995520.0,
          "subchunks_total": 1642,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.655241935483871,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "calibration_greedy": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 39120896,
          "subchunk_bytes_median": 84344832.0,
          "subchunks_total": 1702,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.715725806451613,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "evaluation_greedy": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 39157760,
          "subchunk_bytes_median": 89341952.0,
          "subchunks_total": 1506,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.5181451612903225,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        }
      },
      "combine": {
        "son_torus": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 38342656,
          "subchunk_bytes_median": 44808192.0,
          "subchunks_total": 2688,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 3.0,
          "selected_path_count_mean": 2.7096774193548385,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_0": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 38991872,
          "subchunk_bytes_median": 84164608.0,
          "subchunks_total": 1714,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.7278225806451613,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_24": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 39176192,
          "subchunk_bytes_median": 88182784.0,
          "subchunks_total": 1522,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.534274193548387,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "random_regular_seed_22": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 39176192,
          "subchunk_bytes_median": 85995520.0,
          "subchunks_total": 1642,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.655241935483871,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "calibration_greedy": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 39120896,
          "subchunk_bytes_median": 84344832.0,
          "subchunks_total": 1702,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.715725806451613,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        },
        "evaluation_greedy": {
          "one_cycle_threshold_bytes": 54,
          "subchunk_bytes_min": 39157760,
          "subchunk_bytes_median": 89341952.0,
          "subchunks_total": 1506,
          "subchunks_lt_54B": 0,
          "subchunks_lt_128B": 0,
          "subchunks_lt_256B": 0,
          "selected_path_count_min": 1,
          "selected_path_count_median": 1.0,
          "selected_path_count_mean": 1.5181451612903225,
          "selected_path_count_max": 4,
          "zero_delay_risk": false
        }
      }
    },
    "candidate_family_interpretation": {
      "workload": "deepseek_livecodebench_execution",
      "calibrated_candidate": "random_regular_seed_22",
      "calibrated_candidate_type": "random_regular",
      "oracle_candidate": "random_regular_seed_22",
      "oracle_candidate_type": "random_regular",
      "traffic_fingerprint": "broad / near-uniform",
      "torus_rank_by_eval_fluid": 35,
      "greedy_calibration_rank_by_eval_fluid": 33,
      "greedy_evaluation_rank_by_eval_fluid": 8,
      "fixed_seed0_rank_by_eval_fluid": 17,
      "calibrated_rank_by_eval_fluid": 1,
      "oracle_rank_by_eval_fluid": 1,
      "calibrated_beats_son": true,
      "calibrated_beats_fixed_random": true,
      "oracle_beats_calibrated": false,
      "calibrated_gain_vs_son_percent": 19.861958108975184,
      "calibrated_gain_vs_fixed_random_percent": 1.4522557287930316,
      "oracle_gap_vs_calibrated_percent": 0.0,
      "main_explanation": "workload-selected random-regular topology search",
      "median_random_name": "random_regular_seed_24",
      "best_random_name": null
    },
    "validation_pass": {
      "byte_conservation": true,
      "graph_budget": true,
      "graphs_valid": true,
      "native_runs": true,
      "no_tiny_subchunk_risk": true
    }
  }
]
```

## Anti-Overclaiming

Can claim:

- Static topology candidate selection can be evaluated across multiple HF-derived MoE prefill workloads.
- RON calibrated/oracle can be compared against SON under the same degree/bandwidth budget.
- Random-regular controls help separate topology-family effects from workload-aware selection.

Cannot claim:

- W=4 dynamic reconfiguration works.
- Real serving latency.
- Physical transparent OCS modelling.
- Token/layer-level execution timing.
- Generality beyond these workloads.
