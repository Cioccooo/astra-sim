# Phase III Summary

## Goal

Phase III upgraded backward-path realism within the existing coarse analytical scheduler.

The main purpose was to move backward scheduling away from coarse or scaffold-anchored placement and toward a more credible explicit scheduler for backward compute, backward TP communication, and DP readiness timing. Phase III did this without rewriting raw communication formulas and without introducing a full unified scheduler across all parallelism dimensions.

Phase III is not a full execution-trace simulator. It is also not a full unified TP/DP/PP scheduler. The result is a more realistic coarse analytical backward scheduler, not final system truth.

## Phase III Sub-Stages

### P3.1: Backward Compute Split

Objective:
- Make the internal structure of each backward layer more explicit.

What changed:
- Each backward layer was split into two coarse sequential sub-phases:
  - `BWD_COMP_PRE_TP`
  - `BWD_COMP_POST_TP`
- The end of `BWD_COMP_PRE_TP` became the explicit TP-ready point.

What remained unchanged:
- Total backward compute time per layer remained exactly unchanged.
- Frozen Phase II DP totals remained untouched in this round.

Why it mattered:
- This provided an explicit internal backward structure that later stages could build on, instead of treating the whole backward layer as a single opaque compute window.

### P3.2: Backward TP Scheduler-Explicit Placement

Objective:
- Replace scaffold or anchored backward TP placement with scheduler-explicit timing.

What changed:
- Backward TP communication moved from scaffolded placement to scheduler-explicit placement.
- Two separate TP resources were used:
  - TP intra
  - TP inter
- Backward TP rows in `time_series_overlap.csv` became `scheduler_explicit`.

What remained unchanged:
- TP raw timing formulas remained unchanged.
- DP totals remained frozen in this round.

Why it mattered:
- Backward TP ordering became an explicit result of scheduling logic rather than a placement heuristic anchored to layer ends.
- TP scheduler-level contention, including TP inter scheduling contention, became visible once backward TP was explicitly scheduled, even before any shared TP/DP inter-node fabric budget was introduced.

### P3.3: Finer DP-Ready Point Inside the Backward Structure

Objective:
- Make DP readiness more precise inside the backward compute structure.

What changed:
- `BWD_COMP_POST_TP` was refined into:
  - `BWD_COMP_POST_TP_PRE_DP`
  - `BWD_COMP_POST_TP_POST_DP`
- The DP-ready point moved to the end of `BWD_COMP_POST_TP_PRE_DP`.
- Stage B local-slack hiding then consumed this finer compute structure.

What remained unchanged:
- Raw DP communication totals remained unchanged.
- Raw DP reconfiguration totals remained unchanged.
- The Stage B hiding rule itself was not rewritten; only the readiness timestamps feeding it were refined.

Why it mattered:
- Hidden versus exposed DP communication could now change because the scheduler had a finer DP-ready point, rather than because of any raw formula change.
- This was also the stage where Stage B's local consumable-slack mechanism became numerically meaningful under the refined backward structure.

### P3.4: TP-Blocking Backward Stretching

Objective:
- Make the next lower backward layer wait until the prior layer TP work is complete.

What changed:
- The next lower layer could no longer start immediately after natural compute completion if the prior layer TP had not finished.
- Same-layer TP and compute overlap remained allowed.
- Explicit `BWD_WAIT_FOR_TP` rows were introduced in `time_series_overlap.csv`.
- Backward stretching became directly visible in the overlap timeline.

What remained unchanged:
- Raw DP totals remained unchanged.
- Stage B slack stayed compute-window based.
- Wait gaps did not count as Stage B slack.
- This introduced cross-layer TP blocking only; it did not remove the allowed same-layer overlap between TP communication and the layer's own post-TP compute.

Why it mattered:
- The backward timeline could now stretch when TP completion became the blocking dependency, making the backward path more realistic than the earlier scaffold-heavy model.

## Files Changed During Phase III

Primary implementation file:
- [overlap_scheduler.py](/Users/dfx/Python/LLM_analytical_tools/amped_deepflow/overlap_scheduler.py)

Additional notes:
- `training.py` was intentionally not the main modification site for Phase III logic.
- `time_domain` was not the authoritative path for Phase III behavior.
- The old `time_series.csv` was not the authoritative overlap-analysis output for Phase III.
- Phase III analysis relied primarily on:
  - `time_series_overlap.csv`
  - `training_time_breakdown_overlap.txt`

[performance_model.py](/Users/dfx/Python/LLM_analytical_tools/AMPeD/amped/performance_model.py) was updated where necessary to consume overlap-scheduler outputs consistently in reported totals and exposed-time accounting, while the main scheduling logic remained in [overlap_scheduler.py](/Users/dfx/Python/LLM_analytical_tools/amped_deepflow/overlap_scheduler.py).

## Key Outputs

- `training_time_breakdown_overlap.txt`
  - overlap-aware totals for DP communication and related scheduling effects
- `time_series_overlap.csv`
  - overlap-aware event ordering and diagnostics for backward-path analysis
- `config_summary.txt`
  - effective scheduler settings and run configuration used for each output

These files together formed the main Phase III evidence path.

## Validation Summary

- P3.1 + P3.2 preserved all frozen DP totals.
- Backward TP rows moved from `scaffold_anchored` placement to `scheduler_explicit` placement.
- TP scheduler-level contention, including TP inter scheduling contention, became visible once backward TP was explicitly scheduled, even before any shared TP/DP inter-node fabric budget was introduced.
- P3.3 kept raw DP communication and raw DP reconfiguration unchanged.
- P3.3 changed hidden versus exposed DP communication through finer DP-ready timing, not through raw formula changes.
- P3.4 introduced explicit `BWD_WAIT_FOR_TP` rows.
- P3.4 kept raw DP totals unchanged while allowing backward stretching.
- Bucketed cases showed clearer exposure increases under P3.4 because larger DP units were easier to push out of available compute slack.
- Layerwise cases could still remain fully hidden even when `BWD_WAIT_FOR_TP` gaps existed, because the individual DP units were small enough to fit inside the remaining compute slack.

## What Phase III Made More Realistic

- Backward TP ordering became scheduler-derived rather than scaffold-anchored.
- DP readiness became finer than coarse layer-end readiness.
- Backward compute could now be stretched by TP completion.
- `time_series_overlap.csv` became a much stronger backward-path analysis trace than the earlier scaffold-heavy timeline outputs.

These improvements increased backward-path realism while still staying within a coarse analytical scheduling framework.

## What Remained Outside Phase III Scope

- No shared TP-inter / DP inter-node fabric budget yet.
- No PP explicit micro-batch scheduler yet.
- No full-system topology state machine.
- No global unified scheduler across TP / DP / PP / reconfiguration.
- Still a coarse analytical scheduler, not a kernel-trace simulator.

## Final Conclusion

Phase III should be regarded as a completed backward realism upgrade. It materially improved the earlier scaffold-heavy backward model by making backward TP scheduling explicit, refining DP readiness inside backward compute, and allowing TP completion to stretch the backward timeline. At the same time, it remained a bounded coarse analytical scheduler rather than a full-system execution-truth model.
