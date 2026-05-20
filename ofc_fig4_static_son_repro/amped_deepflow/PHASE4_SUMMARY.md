# Phase IV Summary

## 1. Goal

Phase IV upgraded the simulator from optimistic separate inter-node lanes toward a more realistic shared-fabric model for backward TP-inter and DP communication.

Phase 4A was the arbitration-only stage:
- it introduced explicit shared inter-node fabric arbitration between `BWD_TP_COMM_INTER` and `BWD_DP_COMM`
- it did not yet feed fabric-delayed TP completion back into backward compute stretching

Phase 4B closed that feedback loop:
- the actual delayed completion time of `BWD_TP_COMM_INTER`, including explicit fabric wait, was fed back into the backward TP-blocking dependency

Phase IV is a shared-resource realism upgrade. It is not:
- a raw timing rewrite
- a PP scheduler phase
- a full global scheduler
- a full execution-trace simulator

## 2. What Was Frozen Before Phase IV

Phase IV was built on top of frozen earlier work:

- Phase 2 / 2.5
  - explicit DP scheduler
  - `lump` / `layerwise` / `bucketed`
  - Stage A topology reuse and stricter reconfiguration timing
  - Stage B local consumable-slack hiding
- Phase III
  - backward compute split
  - backward TP scheduler-explicit placement
  - finer DP-ready split
  - TP-blocking backward stretching

These remained the prerequisites and were not rewritten in Phase IV.

## 3. Phase 4A: Shared Inter-Node Fabric Arbitration

### 3.1 Purpose

Phase 4A introduced one shared inter-node fabric resource for:
- `BWD_TP_COMM_INTER`
- `BWD_DP_COMM`

This was intentionally an arbitration-only stage. Fabric-delayed TP completion was not yet fed back into backward compute stretching.

### 3.2 Key Model Change

Phase 4A added:
- one explicit shared inter-node fabric timeline
- explicit service arbitration for TP-inter and DP communication on that shared resource

What remained separate:
- TP intra stayed on its own resource

Arbitration policy:
- non-preemptive FCFS by request time
- exact-timestamp tie-break: TP first

Request time definitions:
- TP-inter request time:
  - when TP inter becomes ready after TP intra completes
- DP request time:
  - when the DP unit first becomes eligible for inter-node communication service after:
    - DP-ready is reached
    - any required Stage A reconfiguration is complete
    - prior DP-lane ordering constraints are satisfied

Important constraints:
- no time travel
- compute-slack semantics remained unchanged
- Stage B slack remained compute-window based

### 3.3 Explicit CSV Visibility

Phase 4A made shared-fabric waiting explicit in `time_series_overlap.csv` with rows such as:
- `BWD_TP_WAIT_FOR_FABRIC`
- `BWD_DP_WAIT_FOR_FABRIC`

These are non-compute waits and remain distinct from:
- `BWD_WAIT_FOR_TP`

This distinction is important because fabric wait and TP-blocking wait are different mechanisms.

### 3.4 4A Validation Outcome

Main conclusions:
- raw DP totals remained unchanged
- the backward compute timeline remained unchanged relative to frozen Phase III / P3.4
- explicit TP/DP shared-fabric contention appeared in the overlap CSV
- both directions were observed:
  - TP blocks DP
  - in at least one case, DP blocks TP

Phase 4A therefore established explicit shared-fabric contention, but it was still an intermediate arbitration-only stage rather than the final physically closed model.

## 4. Phase 4B: Closing the Feedback Loop

### 4.1 Purpose

Phase 4B closed the physical loop opened in 4A:
- the actual delayed end time of `BWD_TP_COMM_INTER`
- including any `BWD_TP_WAIT_FOR_FABRIC`
- now feeds back into the backward TP-blocking dependency

### 4.2 Key Model Change

The core dependency became:

- `natural_next_start[i-1] = end(BWD_COMP_POST_TP_POST_DP[i])`
- `tp_blocking_end_old =` pre-fabric-delay TP completion
- `tp_blocking_end_new =` actual delayed TP completion after shared-fabric wait
- `actual_next_start[i-1] = max(natural_next_start[i-1], tp_blocking_end_new)`

Important semantic points:
- same-layer TP/compute overlap remained allowed
- fabric wait and `BWD_WAIT_FOR_TP` remained distinct
- neither `BWD_TP_WAIT_FOR_FABRIC` nor `BWD_WAIT_FOR_TP` counts as Stage B slack
- raw TP and DP formulas remained unchanged

### 4.3 4B Validation Outcome

Main conclusions:
- raw DP totals still remained unchanged
- at least one sentinel case showed the full feedback chain:
  - DP holds fabric
  - TP waits for fabric
  - TP finishes later
  - `BWD_WAIT_FOR_TP` grows
  - the next lower-layer compute starts later
- total training time became non-decreasing relative to 4A
- within this scope, the model is now physically closed for:
  - shared-fabric TP delay
  - delayed TP completion
  - backward compute stretching

Important clarification:
- part of the 4A -> 4B total-time change came from an accounting correction that now includes explicit `BWD_WAIT_FOR_TP` idle in `Total time to train`
- only some cases also showed an additional true 4B feedback-loop stretching effect
- the clearest sentinel case was Meg530 `bucketed(4L)`

## 5. Files Changed in Phase IV

Code files changed across Phase IV:
- [performance_model.py](/Users/dfx/Python/LLM_analytical_tools/AMPeD/amped/performance_model.py)
- [overlap_scheduler.py](/Users/dfx/Python/LLM_analytical_tools/amped_deepflow/overlap_scheduler.py)

Summary of the kinds of changes made:
- TP/DP arbitration handling
- delayed TP completion feedback into backward stretching
- overlap CSV wait-row generation
- total-time accounting related to explicit waits

## 6. What Stayed Unchanged

The following remained frozen across Phase IV:
- TP raw timing formulas
- DP raw communication formulas
- DP raw reconfiguration formulas
- TP intra as a separate resource
- Stage A semantics
- Stage B hiding rule semantics
- Phase III backward sub-phase structure
- the old `time_series.csv` / `time_domain` path

## 7. What Is Now Trusted

Within the current coarse analytical modeling scope, the following are now trusted:

- explicit DP scheduling
- explicit backward TP scheduling
- explicit shared inter-node arbitration between TP-inter and DP
- explicit fabric wait visibility in `time_series_overlap.csv`
- explicit backward stretching caused by delayed TP completion
- phase-aware overlap accounting with raw / hidden / exposed DP terms
- the overlap CSV as the main authoritative diagnostic trace for this coarse model

This trust statement is limited to the current coarse analytical scope, not to final full-system truth.

## 8. What Still Remains Approximate / Out of Scope

- PP is not yet explicit or scheduler-driven at micro-batch level
- no unified global scheduler across TP / DP / PP / reconfiguration
- no packet-level or cycle-level network model
- no full topology state machine
- no PP sharing with the shared inter-node fabric yet
- not a full-system execution trace

## 9. Recommended One-Paragraph Conclusion

Phase IV is complete and can be frozen. It upgrades the model from optimistic separate inter-node lanes to a coarse but physically more credible shared-fabric model for backward TP/DP interaction. Phase 4A established explicit TP/DP inter-node contention, and Phase 4B closed the loop by feeding fabric-delayed TP completion back into backward compute stretching. Within its intended scope, this makes the simulator substantially more credible for TP/DP inter-node interaction analysis, while PP explicit scheduling remains a major remaining gap.
