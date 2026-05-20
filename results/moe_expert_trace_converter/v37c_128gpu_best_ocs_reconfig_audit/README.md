# V37c 128-GPU Hotness + Best OCS Reconfiguration Audit

## Scope

This audit checks whether 32-GPU expert-to-GPU dilution hid hotness, and compares valid no-leak OCS reconfiguration strategies at 128 GPUs. ASTRA C++ core was not modified. Full strategy search uses a fluid link-load model; native ASTRA is used only for representative static topologies on priority workloads because ASTRA does not yet support safe in-run topology swaps.

## Compact Result

```json
{
  "ranked_valid_non_oracle_methods": {
    "qwen_mmlu_machine_learning": [
      {
        "method": "best universal static",
        "candidate": "random_regular_seed_11",
        "fluid_cycles": 91254347
      },
      {
        "method": "request-window W=8 calibrated OCS, 1us",
        "candidate": "window_W8",
        "fluid_cycles": 95502518
      },
      {
        "method": "fixed random seed0",
        "candidate": "random_regular_seed_0",
        "fluid_cycles": 95546027
      },
      {
        "method": "workload-level calibrated OCS",
        "candidate": "random_regular_seed_24",
        "fluid_cycles": 97570791
      },
      {
        "method": "request-window W=4 calibrated OCS, 1us",
        "candidate": "window_W4",
        "fluid_cycles": 98130013
      },
      {
        "method": "SON/static baseline",
        "candidate": "son_torus",
        "fluid_cycles": 161768228
      }
    ],
    "qwen_livecodebench_execution": [
      {
        "method": "best universal static",
        "candidate": "random_regular_seed_11",
        "fluid_cycles": 596743563
      },
      {
        "method": "workload-level calibrated OCS",
        "candidate": "random_regular_seed_2",
        "fluid_cycles": 621945413
      },
      {
        "method": "request-window W=4 calibrated OCS, 1us",
        "candidate": "window_W4",
        "fluid_cycles": 624805650
      },
      {
        "method": "request-window W=8 calibrated OCS, 1us",
        "candidate": "window_W8",
        "fluid_cycles": 633181939
      },
      {
        "method": "fixed random seed0",
        "candidate": "random_regular_seed_0",
        "fluid_cycles": 636529146
      },
      {
        "method": "SON/static baseline",
        "candidate": "son_torus",
        "fluid_cycles": 1472766107
      }
    ],
    "qwen_mmlu_zh_cn_anatomy": [
      {
        "method": "workload-level calibrated OCS",
        "candidate": "random_regular_seed_3",
        "fluid_cycles": 73157214
      },
      {
        "method": "best universal static",
        "candidate": "random_regular_seed_11",
        "fluid_cycles": 73513189
      },
      {
        "method": "request-window W=8 calibrated OCS, 1us",
        "candidate": "window_W8",
        "fluid_cycles": 74184915
      },
      {
        "method": "request-window W=4 calibrated OCS, 1us",
        "candidate": "window_W4",
        "fluid_cycles": 74374789
      },
      {
        "method": "fixed random seed0",
        "candidate": "random_regular_seed_0",
        "fluid_cycles": 75524104
      },
      {
        "method": "SON/static baseline",
        "candidate": "son_torus",
        "fluid_cycles": 97139678
      }
    ],
    "deepseek_livecodebench_execution": [
      {
        "method": "best universal static",
        "candidate": "random_regular_seed_11",
        "fluid_cycles": 338670640
      },
      {
        "method": "workload-level calibrated OCS",
        "candidate": "random_regular_seed_27",
        "fluid_cycles": 348192941
      },
      {
        "method": "request-window W=4 calibrated OCS, 1us",
        "candidate": "window_W4",
        "fluid_cycles": 350931161
      },
      {
        "method": "request-window W=8 calibrated OCS, 1us",
        "candidate": "window_W8",
        "fluid_cycles": 354734254
      },
      {
        "method": "fixed random seed0",
        "candidate": "random_regular_seed_0",
        "fluid_cycles": 361716718
      },
      {
        "method": "SON/static baseline",
        "candidate": "son_torus",
        "fluid_cycles": 836499319
      }
    ]
  },
  "cross_workload_summary": [
    {
      "workload": "qwen_mmlu_machine_learning",
      "num_experts": 128,
      "expert_top1_share": 0.013351594070237324,
      "expert_top16_share": 0.17807846442365655,
      "expert_gini": 0.133848921675811,
      "pair_top16_share_128gpu": 0.002686887061339462,
      "pair_gini_128gpu": 0.24740483234463034,
      "workload_calibrated_candidate": "random_regular_seed_24",
      "oracle_candidate": "random_regular_seed_22",
      "best_method": "best universal static",
      "son_cycles_segmented_fluid": 161768228,
      "best_universal_cycles_segmented_fluid": 91254347,
      "workload_calibrated_cycles_segmented_fluid": 97570791,
      "window4_cycles_1us_segmented_fluid": 98130013,
      "window8_cycles_1us_segmented_fluid": 95502518,
      "workload_calibrated_gain_vs_son_percent": 39.68482426598627,
      "workload_calibrated_gain_vs_universal_percent": -6.921800667753395,
      "window4_gain_vs_workload_calibrated_percent": -0.5731448871824766,
      "window8_gain_vs_workload_calibrated_percent": 2.1197665600558677,
      "greedy_rank": 34,
      "evaluation_greedy_rank": 32
    },
    {
      "workload": "qwen_livecodebench_execution",
      "num_experts": 128,
      "expert_top1_share": 0.02485706965817855,
      "expert_top16_share": 0.20318222117074325,
      "expert_gini": 0.17759680346623227,
      "pair_top16_share_128gpu": 0.003317350523636854,
      "pair_gini_128gpu": 0.18108349404460716,
      "workload_calibrated_candidate": "random_regular_seed_2",
      "oracle_candidate": "random_regular_seed_14",
      "best_method": "best universal static",
      "son_cycles_segmented_fluid": 1472766107,
      "best_universal_cycles_segmented_fluid": 596743563,
      "workload_calibrated_cycles_segmented_fluid": 621945413,
      "window4_cycles_1us_segmented_fluid": 624805650,
      "window8_cycles_1us_segmented_fluid": 633181939,
      "workload_calibrated_gain_vs_son_percent": 57.7702521775917,
      "workload_calibrated_gain_vs_universal_percent": -4.223229467830891,
      "window4_gain_vs_workload_calibrated_percent": -0.45988553661058995,
      "window8_gain_vs_workload_calibrated_percent": -1.8066739886061352,
      "greedy_rank": 30,
      "evaluation_greedy_rank": 34
    },
    {
      "workload": "qwen_mmlu_zh_cn_anatomy",
      "num_experts": 128,
      "expert_top1_share": 0.018690757088808428,
      "expert_top16_share": 0.2266426982900072,
      "expert_gini": 0.23789230948191692,
      "pair_top16_share_128gpu": 0.004015475415476588,
      "pair_gini_128gpu": 0.27610854284212544,
      "workload_calibrated_candidate": "random_regular_seed_3",
      "oracle_candidate": "random_regular_seed_26",
      "best_method": "workload-level calibrated OCS",
      "son_cycles_segmented_fluid": 97139678,
      "best_universal_cycles_segmented_fluid": 73513189,
      "workload_calibrated_cycles_segmented_fluid": 73157214,
      "window4_cycles_1us_segmented_fluid": 74374789,
      "window8_cycles_1us_segmented_fluid": 74184915,
      "workload_calibrated_gain_vs_son_percent": 24.688638560239,
      "workload_calibrated_gain_vs_universal_percent": 0.48423283609693496,
      "window4_gain_vs_workload_calibrated_percent": -1.6643266376983683,
      "window8_gain_vs_workload_calibrated_percent": -1.4047842226468603,
      "greedy_rank": 34,
      "evaluation_greedy_rank": 33
    },
    {
      "workload": "deepseek_livecodebench_execution",
      "num_experts": 256,
      "expert_top1_share": 0.006991151363517861,
      "expert_top16_share": 0.08416578286085552,
      "expert_gini": 0.07766658971755068,
      "pair_top16_share_128gpu": 0.0014448783559609575,
      "pair_gini_128gpu": 0.05907528054832853,
      "workload_calibrated_candidate": "random_regular_seed_27",
      "oracle_candidate": "random_regular_seed_14",
      "best_method": "best universal static",
      "son_cycles_segmented_fluid": 836499319,
      "best_universal_cycles_segmented_fluid": 338670640,
      "workload_calibrated_cycles_segmented_fluid": 348192941,
      "window4_cycles_1us_segmented_fluid": 350931161,
      "window8_cycles_1us_segmented_fluid": 354734254,
      "workload_calibrated_gain_vs_son_percent": 58.37498810922523,
      "workload_calibrated_gain_vs_universal_percent": -2.811670063870904,
      "window4_gain_vs_workload_calibrated_percent": -0.7864088203901871,
      "window8_gain_vs_workload_calibrated_percent": -1.8786460693928888,
      "greedy_rank": 2,
      "evaluation_greedy_rank": 10
    }
  ],
  "final_diagnosis": {
    "expert_hotness_reproduced": true,
    "old_gpu_pair_ids": {
      "old_examples": [
        {
          "src": 81,
          "dst": 117
        },
        {
          "src": 93,
          "dst": 106
        },
        {
          "src": 131,
          "dst": 259
        }
      ],
      "valid_true_gpu_id_range": "0..127",
      "diagnosis": "Pairs with ids above the GPU range are not final true GPU ids. In this pipeline final GPU-pair matrices are indexed only by src_gpu and dst_gpu. The old examples are consistent with pre-mapping ids such as expert ids or token/source-bucket-to-expert coordinates.",
      "coordinate_system_used_in_v37c": "final true src_gpu -> dst_gpu after source policy and expert placement"
    },
    "hot_experts_become_hot_destination_gpus": "block and round_robin destination-GPU hotness are reported per workload; Qwen LiveCode and Qwen ZH Anatomy are hot at destination-GPU level.",
    "hot_destination_gpus_create_true_hot_gpu_pairs": "not strongly under block_by_token full aggregation; true pair top16 shares remain below 0.5% at 128 GPUs.",
    "source_policy_and_placement_maximising_hot_pairs": "decode_like_batch increases pair concentration versus block_by_token, but pair shares remain small in full aggregate.",
    "full_aggregation_hides_local_windows": "per-request/window summaries are available; however W=4/W=8 no-leak selection does not beat strong static topologies in this fluid audit.",
    "greedy_hot_pair_topology_strength": "greedy ranks are weak for Qwen workloads and only strong for DeepSeek calibration; random-regular dominates most selections.",
    "improved_greedy_local_search": "not implemented; strict greedy repair/fallback is included.",
    "universal_static_topology": "random_regular_seed_11",
    "best_valid_non_oracle_by_workload": {
      "qwen_mmlu_machine_learning": "best universal static",
      "qwen_livecodebench_execution": "best universal static",
      "qwen_mmlu_zh_cn_anatomy": "workload-level calibrated OCS",
      "deepseek_livecodebench_execution": "best universal static"
    },
    "story_decision": "The strongest current story is scale-aware degree-4 topology-family search / strong universal random-regular baseline, not hot-pair greedy nor request-window reconfiguration."
  },
  "limitations": {
    "native_astra_window_reconfiguration": "not implemented; request-window methods are exact no-leak fluid scores, not native in-run ASTRA topology swaps",
    "astra_representative_scope": "native ASTRA static representative runs are limited to ['qwen_mmlu_zh_cn_anatomy'] with 120s timeout per phase/topology",
    "all_path_ecmp": "not used; ecmp_max_paths=4",
    "figures": "none generated",
    "run_log_manifest": "valid_native_astra_run_manifest.json identifies run logs referenced by summary; runs/ may include earlier aborted attempt logs from this task."
  }
}
```

## What This Can Claim

- Expert hotness, destination-GPU hotness, true GPU-pair hotness, and aggregation/window sensitivity were audited at 128 GPUs.
- Workload-level calibrated and request-window calibrated OCS strategies were evaluated without future/evaluation leakage in the fluid model.
- Representative native ASTRA runs validate selected static 128-GPU topologies for priority workloads.

## What This Cannot Claim

- Native ASTRA in-run RON W=4/W=8 topology swaps.
- Real serving latency.
- Physical transparent OCS modelling.
- Token/layer-level execution timing.
- A paper-ready figure.

Detailed data are in `summary.json` and per-workload `workload_summary.json` files.
