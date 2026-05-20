# V39.1 Supervisor Preliminary Summary

## Routing

V39 SON/torus uses **ECMP-4 over equal-cost shortest paths**. `current` and
`ecmp4` are the same rule. All-shortest-path ECMP is reported only as optimistic
sensitivity.

## Stage

Figure A evaluates **decode `trace[1:]`**. Prefill `trace[0]` is used only for
topology selection in prefill-informed OCS.

## Oracle

Oracle is an upper bound and satisfies oracle <= fair and oracle <= prefill. It
is close to fair because the bottleneck is optical resource load and many
degree-4 random-regular candidates are near-equivalent.

## Safe Figures

- Figure A: safe preliminary optical-only controlled comparison.
- Figure B: safe only with explicit mixed-semantics label.
- Figure C: packet-routing sensitivity only.
- Figure D: trace predictability diagnostic.

```json
{
  "routing_rule_used_for_v39_son": "ECMP-4 over equal-cost shortest paths; current == ecmp4",
  "torus_gap_survives_ecmp4": {
    "qwen_mmlu_machine_learning": 3.4901827936033483,
    "deepseek_mmlu_machine_learning": 3.6563431632884806,
    "qwen_livecodebench_execution": 3.505988883738431
  },
  "figure_A_evaluates": "decode trace[1:]",
  "topology_selection": "prefill trace[0] only for prefill-informed OCS",
  "oracle_and_fair_close_reason": "dominant bottleneck is optical_resource, and the random-regular candidate pool contains many near-equivalent expander-like graphs",
  "prefill_ocs_beats_fair_static": {
    "qwen_mmlu_machine_learning": false,
    "deepseek_mmlu_machine_learning": true,
    "qwen_livecodebench_execution": false
  },
  "safe_supervisor_figures": {
    "Figure A": "safe as preliminary optical-only controlled comparison",
    "Figure B": "safe only as clearly labelled EN electrical reference vs optical reference",
    "Figure C": "safe only as packet-routing sensitivity, not optical main result",
    "Figure D": "safe as trace predictability diagnostic"
  },
  "optional_qwen_zh_status": {
    "workload": "qwen_mmlu_zh_cn_anatomy_optional",
    "label": "Qwen MMLU_ZH_CN anatomy optional",
    "trace_path": "/Users/dfx/Python/trace/Qwen/Qwen3-235B-A22B-FP8/mmlu_ZH_CN/anatomy",
    "included_in_v39_figures": false,
    "request_count": 135,
    "prefill_token_count_trace0": 1569,
    "decode_token_count_trace1_plus": 17280,
    "figure_A_evaluates": "decode trace[1:]",
    "topology_selection_signal": "prefill trace[0] only for prefill-informed OCS",
    "evaluation_signal": "decode trace[1:] only",
    "decode_exists_non_empty": true,
    "note": "optional check only; cheap to include in future but omitted from V39 to keep required set fixed"
  }
}
```
