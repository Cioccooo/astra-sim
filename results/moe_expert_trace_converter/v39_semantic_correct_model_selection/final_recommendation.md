# V39 Final Recommendation

## Decision

Use the **optical circuit/capacity reference model** for the main optical-only
comparison. Use native ASTRA GraphTopology for EN electrical reference and for a
separate packet-routing sensitivity figure.

## Why EN Was Faster Than SON Before

The current native ASTRA GraphTopology figure compared electrical packet
folded-Clos EN against SON represented as GPU-as-router multi-hop torus. EN has
switch nodes and shorter routes; SON had long GPU transit paths. That is a
packet-routing sensitivity, not transparent optical OCS semantics.

## Main Paper-Direction Figure

Use Figure A:

- `figures/figure_A1_optical_only_raw.png`
- `figures/figure_A2_optical_only_normalized.png`

## Supervisor Discussion

Use Figure A and Figure B. Show Figure C only to explain why the earlier ASTRA
GraphTopology result was not the right optical main model.

## Final Answers

```json
{
  "q1_model_for_EN": "Native ASTRA GraphTopology is semantically appropriate for electrical packet folded-Clos EN.",
  "q2_model_for_optical": "The optical circuit/capacity reference is more semantically correct than current GraphTopology for transparent optical SON/RON/OCS.",
  "q3_current_ASTRA_SON_RON_valid_for_optical_claims": false,
  "q4_EN_same_figure_policy": "Separate optical-only Figure A from mixed-reference Figure B; never present EN and optical methods as same-budget without labels.",
  "q5_prefill_ocs_beats_fair_static_optical_only": {
    "qwen_mmlu_machine_learning": false,
    "deepseek_mmlu_machine_learning": true,
    "qwen_livecodebench_execution": false
  },
  "q6_oracle_correct_upper_bound": {
    "qwen_mmlu_machine_learning": true,
    "deepseek_mmlu_machine_learning": true,
    "qwen_livecodebench_execution": true
  },
  "q7_why_astra_and_reference_differ": "ASTRA GraphTopology models hop-by-hop packet/store-and-forward; optical reference charges sender, receiver, and optical resource capacity without GPU forwarding semantics.",
  "q8_supervisor_discussion_figure": "Figure A plus Figure B, with Figure C only as sensitivity.",
  "q9_paper_direction_figure": "Figure A if the paper targets optical OCS capacity fabrics; Figure C only for packet-routing sensitivity appendix.",
  "q10_next_step": "Design/prototype native circuit-aware ASTRA backend if ASTRA-native optical results are required."
}
```
