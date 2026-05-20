# V3.6 Static RON One-Workload Validation

## Scope

This adds static RON calibrated/oracle topology generation to the V35 aggregated native ASTRA pipeline. It is a validation step, not a final paper result.

## RON Construction

- Candidate pool: SON 4x8 torus, calibration-greedy degree-4 graph, evaluation-greedy degree-4 graph, and 32 deterministic random regular degree-4 graphs.
- RON calibrated selection uses **only calibration dispatch+combine traffic**.
- RON oracle selection uses evaluation dispatch+combine traffic and is labelled oracle/reference only.
- Edges are bidirectional optical circuits.
- Degree is enforced at 4 for every GPU.
- Per-link bandwidth is 400 Gb/s; per-GPU budget is degree 4 x 400 Gb/s = 1.6 Tb/s.
- Routing is native GraphTopology ECMP with `ecmp_max_paths=4`.

## Final Answers

1. Does V36 match the V35 full-workload aggregate before splitting? **True.**
2. Does RON calibrated use only calibration traffic? **True.**
3. Does RON oracle act only as an oracle reference? **True.**
4. Are SON, RON calibrated, and RON oracle under the same degree/bandwidth budget? **True.**
5. Are all RON graphs connected and valid? **True.**
6. Do RON calibrated/oracle run successfully in native ASTRA? **True.**
7. Does RON calibrated beat SON ECMP4 on evaluation traffic? **True.**
8. Does RON oracle beat SON ECMP4 on evaluation traffic? **True.**
9. If RON helps, why? See gain/loss explanation below.
10. If RON does not help, why not? See gain/loss explanation below.
11. Is it safe to proceed to four-workload static RON validation next? **True.**

## Calibration / Evaluation Split

```json
{
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
  "calibration_fraction_requested": 0.1,
  "calibration_rule": "front ceil(10%) requests"
}
```

## V35 Consistency

```json
{
  "files_used_match": true,
  "prefill_tokens_match": true,
  "moe_layer_count_match": true,
  "selected_events_match": true,
  "theoretical_dispatch_bytes_match": true,
  "local_bytes_match": true,
  "remote_bytes_match": true,
  "dispatch_checksum_match": true,
  "combine_checksum_match": true,
  "matches_v35": true
}
```

## Traffic Audit

```json
{
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
    "message_bytes_max": 5136384
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
    "message_bytes_max": 5136384
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
    "message_bytes_max": 30998528
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
    "message_bytes_max": 30998528
  }
}
```

## Graph Audit

```json
{
  "son_torus_ecmp4": {
    "name": "son_torus_4x8_32gpu",
    "node_count": 32,
    "gpu_count": 32,
    "edge_circuit_count": 64,
    "degree_distribution": {
      "4": 32
    },
    "degree_min": 4,
    "degree_max": 4,
    "connected_components": 1,
    "component_sizes": [
      32
    ],
    "duplicate_edges": 0,
    "self_loops": 0,
    "per_gpu_optical_bandwidth_tbps": 1.6,
    "per_link_bandwidth_gbps": 400,
    "same_degree_bandwidth_budget_as_son": true,
    "valid": true
  },
  "ron_calibrated_ecmp4": {
    "name": "ron_random_regular_seed_16",
    "node_count": 32,
    "gpu_count": 32,
    "edge_circuit_count": 64,
    "degree_distribution": {
      "4": 32
    },
    "degree_min": 4,
    "degree_max": 4,
    "connected_components": 1,
    "component_sizes": [
      32
    ],
    "duplicate_edges": 0,
    "self_loops": 0,
    "per_gpu_optical_bandwidth_tbps": 1.6,
    "per_link_bandwidth_gbps": 400,
    "same_degree_bandwidth_budget_as_son": true,
    "valid": true
  },
  "ron_oracle_ecmp4": {
    "name": "ron_random_regular_seed_22",
    "node_count": 32,
    "gpu_count": 32,
    "edge_circuit_count": 64,
    "degree_distribution": {
      "4": 32
    },
    "degree_min": 4,
    "degree_max": 4,
    "connected_components": 1,
    "component_sizes": [
      32
    ],
    "duplicate_edges": 0,
    "self_loops": 0,
    "per_gpu_optical_bandwidth_tbps": 1.6,
    "per_link_bandwidth_gbps": 400,
    "same_degree_bandwidth_budget_as_son": true,
    "valid": true
  }
}
```

## Graph Quality

```json
{
  "son_torus_ecmp4": {
    "average_shortest_path_length": 3.096774193548387,
    "diameter": 6,
    "shortest_path_length_distribution": {
      "1": 128,
      "2": 224,
      "3": 256,
      "4": 224,
      "5": 128,
      "6": 32
    },
    "unreachable_pairs": 0,
    "ecmp_path_count_distribution_cap4": {
      "1": 256,
      "2": 192,
      "3": 128,
      "4": 416
    },
    "ecmp_path_count_min_cap4": 1,
    "ecmp_path_count_median_cap4": 3.0,
    "ecmp_path_count_mean_cap4": 2.7096774193548385,
    "ecmp_path_count_max_cap4": 4
  },
  "ron_calibrated_ecmp4": {
    "average_shortest_path_length": 2.473790322580645,
    "diameter": 4,
    "shortest_path_length_distribution": {
      "1": 128,
      "2": 334,
      "3": 462,
      "4": 68
    },
    "unreachable_pairs": 0,
    "ecmp_path_count_distribution_cap4": {
      "1": 654,
      "2": 216,
      "3": 54,
      "4": 68
    },
    "ecmp_path_count_min_cap4": 1,
    "ecmp_path_count_median_cap4": 1.0,
    "ecmp_path_count_mean_cap4": 1.532258064516129,
    "ecmp_path_count_max_cap4": 4
  },
  "ron_oracle_ecmp4": {
    "average_shortest_path_length": 2.5,
    "diameter": 4,
    "shortest_path_length_distribution": {
      "1": 128,
      "2": 330,
      "3": 444,
      "4": 90
    },
    "unreachable_pairs": 0,
    "ecmp_path_count_distribution_cap4": {
      "1": 600,
      "2": 230,
      "3": 66,
      "4": 96
    },
    "ecmp_path_count_min_cap4": 1,
    "ecmp_path_count_median_cap4": 1.0,
    "ecmp_path_count_mean_cap4": 1.655241935483871,
    "ecmp_path_count_max_cap4": 4
  }
}
```

## Tiny-Subchunk Audit

```json
{
  "dispatch": {
    "son_torus_ecmp4": {
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
    "ron_calibrated_ecmp4": {
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
    "ron_oracle_ecmp4": {
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
    }
  },
  "combine": {
    "son_torus_ecmp4": {
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
    "ron_calibrated_ecmp4": {
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
    "ron_oracle_ecmp4": {
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
    }
  }
}
```

## Native ASTRA Timing

```json
{
  "son_torus_ecmp4": {
    "dispatch_cycles": 43197644,
    "combine_cycles": 44878393,
    "total_cycles": 88076037,
    "dispatch_fluid_cycles": 21812718,
    "combine_fluid_cycles": 20296847,
    "total_fluid_cycles": 42109565,
    "astra_over_fluid_total": 2.0915921833911133,
    "success": true,
    "runtime_s": 0.07198358396999538
  },
  "ron_calibrated_ecmp4": {
    "dispatch_cycles": 35427999,
    "combine_cycles": 36642032,
    "total_cycles": 72070031,
    "dispatch_fluid_cycles": 10425974,
    "combine_fluid_cycles": 10234629,
    "total_fluid_cycles": 20660603,
    "astra_over_fluid_total": 3.488283038012008,
    "success": true,
    "runtime_s": 0.04947491595521569
  },
  "ron_oracle_ecmp4": {
    "dispatch_cycles": 34720999,
    "combine_cycles": 36142168,
    "total_cycles": 70863167,
    "dispatch_fluid_cycles": 9819780,
    "combine_fluid_cycles": 9888585,
    "total_fluid_cycles": 19708365,
    "astra_over_fluid_total": 3.5955883199849406,
    "success": true,
    "runtime_s": 0.049288124893791974
  }
}
```

## Fluid Lower Bound

```json
{
  "dispatch": {
    "son_torus_ecmp4": {
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
    "ron_calibrated_ecmp4": {
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
    "ron_oracle_ecmp4": {
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
    }
  },
  "combine": {
    "son_torus_ecmp4": {
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
    "ron_calibrated_ecmp4": {
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
    "ron_oracle_ecmp4": {
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
    }
  }
}
```

## Gain / Loss Explanation

```json
{
  "ron_calibrated_beats_son": true,
  "ron_oracle_beats_son": true,
  "total_cycles": {
    "son_torus_ecmp4": 88076037,
    "ron_calibrated_ecmp4": 72070031,
    "ron_oracle_ecmp4": 70863167
  },
  "calibrated_gain_vs_son_percent": 18.17294072847533,
  "oracle_gain_vs_son_percent": 19.543193116193454,
  "calibrated_vs_oracle_gap_percent": 1.6745712236477324,
  "max_link_load_change_vs_son": {
    "son_torus_ecmp4": {
      "dispatch_max_link_load_bytes": 1171061422,
      "combine_max_link_load_bytes": 1089678681,
      "dispatch_change_vs_son_percent": 0.0,
      "combine_change_vs_son_percent": 0.0
    },
    "ron_calibrated_ecmp4": {
      "dispatch_max_link_load_bytes": 559740245,
      "combine_max_link_load_bytes": 549467477,
      "dispatch_change_vs_son_percent": 52.20231539657021,
      "combine_change_vs_son_percent": 49.57527511727101
    },
    "ron_oracle_ecmp4": {
      "dispatch_max_link_load_bytes": 527195478,
      "combine_max_link_load_bytes": 530889387,
      "dispatch_change_vs_son_percent": 54.98139823446426,
      "combine_change_vs_son_percent": 51.28018963234172
    }
  },
  "hop_count_change_vs_son": {
    "son_torus_ecmp4": {
      "dispatch_byte_weighted_avg_hop": 3.100443795109504,
      "combine_byte_weighted_avg_hop": 3.100443795109504
    },
    "ron_calibrated_ecmp4": {
      "dispatch_byte_weighted_avg_hop": 2.470562811963951,
      "combine_byte_weighted_avg_hop": 2.470562811963951
    },
    "ron_oracle_ecmp4": {
      "dispatch_byte_weighted_avg_hop": 2.5001165954139135,
      "combine_byte_weighted_avg_hop": 2.5001165954139135
    }
  },
  "selected_candidates": {
    "ron_calibrated": "random_regular_seed_16",
    "ron_oracle": "random_regular_seed_22"
  },
  "top_candidate_scores": {
    "calibrated": [
      {
        "name": "random_regular_seed_16",
        "score_max_link_load_bytes": 77485397
      },
      {
        "name": "random_regular_seed_22",
        "score_max_link_load_bytes": 79163391
      },
      {
        "name": "random_regular_seed_3",
        "score_max_link_load_bytes": 81089195
      },
      {
        "name": "random_regular_seed_19",
        "score_max_link_load_bytes": 81855147
      },
      {
        "name": "random_regular_seed_27",
        "score_max_link_load_bytes": 84490923
      },
      {
        "name": "random_regular_seed_18",
        "score_max_link_load_bytes": 84752385
      },
      {
        "name": "random_regular_seed_13",
        "score_max_link_load_bytes": 85124438
      },
      {
        "name": "random_regular_seed_9",
        "score_max_link_load_bytes": 85803007
      }
    ],
    "oracle": [
      {
        "name": "random_regular_seed_22",
        "score_max_link_load_bytes": 530889387
      },
      {
        "name": "random_regular_seed_16",
        "score_max_link_load_bytes": 559740245
      },
      {
        "name": "random_regular_seed_20",
        "score_max_link_load_bytes": 595292844
      },
      {
        "name": "random_regular_seed_18",
        "score_max_link_load_bytes": 602815831
      },
      {
        "name": "random_regular_seed_19",
        "score_max_link_load_bytes": 615293612
      },
      {
        "name": "random_regular_seed_13",
        "score_max_link_load_bytes": 616904705
      },
      {
        "name": "random_regular_seed_25",
        "score_max_link_load_bytes": 619573248
      },
      {
        "name": "random_regular_seed_29",
        "score_max_link_load_bytes": 620749484
      }
    ]
  },
  "interpretation": "RON helps if it reduces max hot-link load and/or byte-weighted hop count without increasing degree/bandwidth. Negative results are valid and indicate weak skew or poor calibration transfer."
}
```

## What V36 Can Claim

- Static RON graph generation and native ASTRA execution work for one real aggregated workload.
- RON calibrated/oracle are compared with SON ECMP4 under the same degree/bandwidth budget.

## What V36 Cannot Claim

- W=4 dynamic reconfiguration works.
- Results generalise to all workloads.
- Real serving latency.
- Physical transparent OCS modelling.
- Token/layer-level execution timing.
