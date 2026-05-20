# Trace Semantics and Scope

## 1. Current Status of the Simulator
The current simulator mainline is considered complete for research use as a coarse analytical scheduler.

Within that scope, the mainline now includes:

- scheduler-explicit DP overlap logic
- scheduler-explicit backward TP logic
- shared-fabric TP/DP contention and feedback
- explicit PP ordering and PP wait/bubble diagnostics

This mainline is intended for research-oriented timing analysis at a coarse scheduling level. It is not a packet-level, cycle-level, or full-system execution-trace simulator.

## 2. Current Status of `time_series_overlap.csv`
The current overlap CSV is strong for:

- DP overlap analysis
- backward TP scheduling analysis
- shared-fabric TP/DP contention analysis
- backward TP-blocking analysis

It is also useful for:

- PP diagnostic annotation
- stage-by-stage and micro-batch-by-micro-batch inspection

At the same time, the current overlap CSV remains a mixed trace rather than a fully unified global wall-clock trace.

More precisely:

- `FWD_PP_STAGE_COMPUTE` and `BWD_PP_STAGE_COMPUTE` are explicit PP diagnostic/envelope rows
- they are not yet authoritative global-gating envelopes for the full inner TP/DP/layer timeline

This means the PP rows are informative and scheduler-derived, but they do not yet globally gate all inner backward and overlap events into a single unified wall-clock schedule.

## 3. Why Phase 5C Was Not Adopted as a Mainline Change
Phase 5C was not adopted as a mainline change because, under the current frozen GPipe-like interpretation, enforcing PP global-gating would do more than fix a small integration issue.

It would:

- redefine the trace semantics
- potentially shift backward start times substantially
- disturb the meaning of already frozen Phase II, Phase III, and Phase IV results

For that reason, Phase 5C is treated as an alternative modeling direction rather than a required bug fix.

The current mixed-trace semantics are therefore a documented scope choice, not a silently ignored inconsistency in the mainline research model.

## 4. Known Limitations
The following remain known limitations of the current mainline:

- PP is not yet globally gating the inner timeline
- PP traffic is not merged into the frozen Phase 4 TP/DP shared-fabric model
- legacy PP bubble accounting remains partly intact
- the simulator is still not a full-system final-truth execution simulator

Double-counting between legacy bubble accounting and the new explicit `FWD_PP_BUBBLE` / `BWD_PP_BUBBLE` rows is avoided because the explicit rows are diagnostic-only and do not contribute to the authoritative total-time accounting path.

These limitations are real, but they do not invalidate the frozen TP/DP scheduler paths that the main optical-interconnect analysis depends on.

## 5. Why the Current Mainline Is Still Acceptable
The current mainline remains acceptable because the primary metrics for the optical interconnect study are derived from scheduler-explicit lanes that are not invalidated by the PP mixed-trace limitation.

In particular, the following remain grounded in explicit scheduler paths:

- DP hidden and exposed communication
- backward TP scheduling effects
- shared-fabric contention and TP-delay feedback

Those quantities are derived from the frozen DP, TP, and shared-fabric scheduler logic rather than from the PP diagnostic envelope rows.

## 6. Clarification of the P3.3 Split Parameter
The P3.3 split parameter is:

- `BACKWARD_POST_TP_PRE_DP_ALPHA = 0.5`

It is defined in:

- `amped_deepflow/overlap_scheduler.py`

This parameter is a fixed modeling assumption. It is not exposed as a user-facing runtime parameter in the current mainline, and it is not presented as a physically derived constant.

Its modeling role is to place the DP-ready point halfway through the old `POST_TP` compute region by default, so that:

- `BWD_COMP_POST_TP_PRE_DP` ends at the coarse DP-ready point
- `BWD_COMP_POST_TP_POST_DP` remains as the same-layer compute tail after DP becomes ready

This is a coarse structural assumption used to refine readiness timing, not a claimed hardware-truth decomposition.

## 7. Future Work / Optional Extensions
The following remain future-work options rather than current required fixes:

- an optional alternative globally gated PP wall-clock trace mode
- a 1F1B PP schedule
- an interleaved 1F1B PP schedule
- optional PP integration into a shared inter-node fabric model

These are meaningful extensions, but they are not prerequisites for freezing the current mainline coarse analytical scheduler.
