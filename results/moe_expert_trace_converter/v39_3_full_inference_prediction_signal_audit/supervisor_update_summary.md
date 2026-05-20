# V39.3 Supervisor Update

## Main Takeaway

After including warm-up/default traffic and reconfiguration penalties, OCS
control gains are much weaker than the remaining-decode-only view.

## Best Non-Oracle Method at 1us Penalty

```json
{
  "qwen_mmlu_machine_learning": {
    "best_non_oracle_method": "prefill-warmup OCS",
    "best_non_oracle_normalised": 1.142262084384056,
    "best_non_oracle_beats_fair_static": false,
    "oracle_normalised": 1.0
  },
  "deepseek_mmlu_machine_learning": {
    "best_non_oracle_method": "phase-warmup OCS",
    "best_non_oracle_normalised": 1.0869668483395214,
    "best_non_oracle_beats_fair_static": false,
    "oracle_normalised": 0.9782666270813651
  },
  "qwen_livecodebench_execution": {
    "best_non_oracle_method": "prefill-warmup OCS",
    "best_non_oracle_normalised": 0.9820622057369703,
    "best_non_oracle_beats_fair_static": true,
    "oracle_normalised": 0.9704197196267572
  }
}
```

## Interpretation

This does change the V39.2 conclusion: prefill/decode prediction may help in a
remaining-traffic view, but full inference accounting makes the OCS advantage
harder to defend unless the control policy improves a large fraction of traffic
or reconfiguration is amortised over larger batches/windows.
