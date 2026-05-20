# Phase 5 Summary

## 1. Scope and Goal
Phase 5 upgraded pipeline-parallel (PP) modeling from a largely scaffolded timing view to a much more explicit coarse scheduler view.

Phase 5A made PP send/recv events and stage-by-stage, micro-batch-by-micro-batch ordering explicit in the overlap trace. Phase 5B then added explicit PP wait, bubble, and idle attribution on top of that ordering.

Phase 5 did not introduce a full-system unified scheduler. It also did not merge PP traffic into the frozen Phase 4 TP/DP shared inter-node fabric model. The goal was to make PP event ordering and PP idle structure much more explicit and diagnostically useful while keeping the already frozen TP/DP logic intact.

## 2. What Was Completed in Phase 5A
Phase 5A introduced explicit PP event rows into `time_series_overlap.csv`, including:

- `FWD_PP_RECV`
- `FWD_PP_STAGE_COMPUTE`
- `FWD_PP_SEND`
- `BWD_PP_RECV`
- `BWD_PP_STAGE_COMPUTE`
- `BWD_PP_SEND`

It also made stage identity and micro-batch identity explicit in the overlap CSV through dedicated `Stage index` and `Microbatch index` columns. This made it possible to follow a PP chain directly as `recv -> compute -> send` for a specific stage and micro-batch.

Stage-local compute ordering became explicit as well. Within a single PP stage, local compute was scheduled as mutually exclusive work, and this was verified during validation. No stage-local overlap was found between forward PP compute blocks or backward PP compute blocks, and no forward PP compute block overlapped a backward PP compute block in the same stage.

The coarse pipeline policy explicitized in Phase 5A was GPipe-like fill-drain: all forward micro-batches first, followed by all backward micro-batches. This matched the frozen model more closely than introducing a different pipeline policy would have.

Phase 5A primarily improved PP ordering realism in the overlap CSV. It did not rewrite authoritative total-time accounting.

## 3. What Was Completed in Phase 5B
Phase 5B added explicit PP wait, bubble, and idle attribution on top of the frozen Phase 5A PP ordering.

The implemented explicit event families included:

- `FWD_PP_WAIT_FOR_RECV` / `BWD_PP_WAIT_FOR_RECV`
- `FWD_PP_WAIT_FOR_STAGE` / `BWD_PP_WAIT_FOR_STAGE`
- `FWD_PP_BUBBLE` / `BWD_PP_BUBBLE`

This made PP-side idle structure visible in `time_series_overlap.csv` instead of leaving PP timing as only explicit send/recv ordering plus coarse legacy bubble accounting.

Phase 5B made startup-side, mid-pipeline, and drain-side PP wait/bubble structure much easier to inspect. It also made stage-local queueing visible when PP-side communication readiness and local stage availability did not align.

Legacy PP bubble accounting was left partly intact to avoid double-counting, and frozen TP/DP total-time accounting was not disturbed. Phase 5B therefore improved PP attribution and diagnostics without fully replacing the legacy PP total-time accounting path.

## 4. Files Changed
Across Phase 5, the changed source files were:

- `amped_deepflow/overlap_scheduler.py`
- `AMPeD/amped/performance_model.py`

More precisely:

- Phase 5A changed `amped_deepflow/overlap_scheduler.py`
- Phase 5B changed `amped_deepflow/overlap_scheduler.py`
- Phase 5B also updated `AMPeD/amped/performance_model.py` where necessary to consume the explicit PP wait/bubble outputs consistently in overlap-oriented diagnostics and reporting

`training.py` was not the main modification site for Phase 5. `time_domain` and old `time_series.csv` were not the authoritative path for Phase 5. The main overlap-aware artifacts remained `time_series_overlap.csv` and `training_time_breakdown_overlap.txt`.

## 5. What Stayed Frozen
The following remained frozen through Phase 5:

- TP raw timing formulas
- DP raw communication formulas
- DP raw reconfiguration formulas
- Stage A and Stage B logic
- Phase 4A / 4B TP/DP shared-fabric model
- P3.1–P3.4 backward realism logic

Phase 5 refined PP event ordering and PP idle attribution around these frozen components. It did not rewrite them.

## 6. Validation Summary
Phase 5 validation used:

- `GPT3_175B_NEW_ocs.json` with `DP_1_16_TP_8_2_PP_1_8_8`
- `megatron_530B_ocs.json` with `DP_1_9_TP_8_2_PP_1_35_8`

For each workload, validation covered:

- `layerwise`
- `bucketed(4L)`
- `ocs_reconf_us = 10.0`

Validation established the following:

- TP/DP raw totals stayed unchanged
- stage-local PP compute overlap was not found
- PP rows became substantially less scaffold-dependent
- explicit PP send/recv ordering is now visible
- explicit PP wait/bubble attribution is now visible
- total-time accounting was not rewritten in a way that breaks frozen TP/DP logic

In Phase 5A, PP ordering became scheduler-explicit without changing frozen TP/DP totals or frozen end-to-end totals. In Phase 5B, PP wait/bubble rows became explicit, and overlap-aware reporting gained explicit PP wait/bubble diagnostics. Legacy PP bubble accounting was left partly intact to avoid double-counting, while frozen TP/DP total-time accounting was not disturbed.

## 7. What Phase 5 Improved
Phase 5 made several aspects of PP behavior more physically credible within the current coarse analytical scope:

- PP recv/compute/send chains are now explicit
- stage-by-stage, micro-batch-by-micro-batch sequencing is explicit
- PP startup-side, mid-pipeline, and drain-side wait/bubble structure is more visible
- the overlap CSV is much more useful diagnostically for PP analysis

This is especially important for backward-path interpretation, because PP-side blocking and PP-side idle can now be inspected directly rather than inferred from scaffolded placement.

## 8. What Phase 5 Did NOT Do
Phase 5 did not yet provide:

- a full-system unified scheduler
- packet-level or cycle-level truth
- PP sharing the frozen Phase 4 TP/DP fabric model
- a single final-truth global execution DAG

It also did not replace the simulator with a full execution-trace engine. The Phase 5 overlap CSV remains a mixed trace rather than a fully unified global wall-clock trace, even though its PP ordering and PP idle annotation are much stronger than before. The result remains a coarse analytical scheduler, albeit a substantially more explicit and diagnostically credible one for PP behavior.

## 9. Final Status
Phase 5A and 5B together make PP explicit enough for the simulator's current mainline coarse scheduler to be considered complete for the intended research use in this project. The resulting `time_series_overlap.csv` is now substantially more rigorous and informative for PP analysis than the earlier scaffold-heavy view. At the same time, the simulator remains a coarse analytical scheduler rather than a full-system final-truth execution simulator.
