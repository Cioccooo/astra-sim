# Phase II Extras Summary

This document records the extra work added after the original Phase II DP-overlap core, so Phase II can be frozen cleanly before any later stage.

## 1. What Phase II Originally Solved

The original Phase II goal was to replace the old DP overlap ratio correction with an explicit per-step DP scheduler while keeping the legacy baseline path available.

The original Phase II core also included explicit DP-related OCS reconfiguration accounting with raw / hidden / exposed totals.

Core Phase II semantics:

- `lump`: one aggregate no-overlap DP unit per optimization step; legacy-safe baseline behavior
- `layerwise`: one DP unit per backward layer; readiness is the backward completion of that layer
- `bucketed`: real byte-based bucketing; each bucket becomes ready when the last backward layer that fills it completes

The Phase II schedule unit is:

- one optimization step

Full-training totals are derived as:

- `number_of_batches × per-step scheduled totals`

## 2. What Extra Work Was Added After The Original Phase II

### Stage A: DP-lane-local topology reuse

Stage A introduced a DP-lane-local topology state for reconfiguration decisions:

- same-topology consecutive DP units reuse the current DP-lane topology
- only topology transitions pay raw DP-related OCS reconfiguration
- the first DP unit in each step is conservatively charged one transition cost

This is a conservative DP-lane-local approximation, not full-system topology truth.

### Stage A: stricter reconfiguration timing / hiding

Stage A also tightened DP-related OCS reconfiguration timing:

- reconfiguration can start only when the DP unit is queue head and ready
- reconfiguration hiding uses only the local remaining backward-compute slack after that point

This was later stress-tested with a trigger case to confirm that the stricter rule can produce meaningful exposed reconfiguration when reconfiguration is made large enough. That stress case was a mechanism-validation case, not a realism claim.

### Stage B: local consumable slack for DP communication hiding

Stage B replaced the old implicit `compute_end` hiding check with an explicit local slack mechanism:

- each DP unit can hide communication only under backward-compute slack that remains after it is queue head and ready
- slack is modeled as a consumable interval pool
- earlier DP units consume slack; later units cannot reuse it

Raw DP communication, bucket construction, and Stage A reconfiguration rules were not changed.

Important current observation:

- under the current continuous backward-compute window model, the Stage B mechanism is explicit and stricter in definition, but in the tested cases it is numerically equivalent to the old rule

### New overlap-aware CSV

Added a new overlap-aware CSV:

- `time_series_overlap.csv`

This file is separate from the old `time_series.csv` and does not patch it.

It includes:

- one event per row
- explicit `BWD_DP_RECONFIG` and `BWD_DP_COMM` events
- start / end / duration / bytes / collective / parallelism / locality / degree
- event provenance and ordering source
- `topology_signature`
- previous-event fields for transition analysis

### Improved overlap-aware breakdown output

Added a dedicated Phase II overlap-aware breakdown file:

- `training_time_breakdown_overlap.txt`

This keeps the legacy `training_time_breakdown.txt` separate from the overlap-aware accounting.

### Config additions

Added under `system_architecture_parameters`:

- `dp_overlap_mode`
- `dp_bucket_size_bytes`

Current meaning of `dp_bucket_size_bytes = 0`:

- use one layer’s gradient bytes as the effective bucket target size
- under the current Phase II uniform-layer-size assumption, this makes `bucketed` schedule-equivalent to `layerwise`

### CLI override additions

Added CLI overrides in `training.py`:

- `--dp-overlap-mode lump|layerwise|bucketed`
- `--dp-bucket-size-bytes <bytes>`
- `--ocs-reconf-us <microseconds>`

CLI overrides apply only to the per-run runtime mirror and do not leak across runs.

### Effective scheduler settings in config summary

Added an `EFFECTIVE SCHEDULER SETTINGS` section in `config_summary.txt`, including:

- requested DP bucket size
- effective bucket size after clamp
- gradient bytes per layer
- schedule unit
- full-training derivation rule

### Naming and output cleanup

Added overlap-mode-aware subdirectories under the existing parallelism directory, for example:

- `dp_lump_pp_off_tp_off`
- `dp_layerwise_pp_off_tp_off`
- `dp_bucketed_pp_off_tp_off`

This keeps different DP overlap modes separated without encoding too many numeric settings in the folder name.

## 3. Current Outputs And What Each One Means

### Authoritative Phase II outputs

- `training_time_breakdown_overlap.txt`
  - authoritative overlap-aware totals
  - contains full-training DP raw / hidden / exposed communication
  - contains full-training DP raw / hidden / exposed reconfiguration
  - contains communication-related raw / hidden / exposed totals

- `config_summary.txt`
  - authoritative companion file for run settings
  - use it to confirm effective scheduler settings and bucket-size clamping

### Useful Phase II analysis output

- `time_series_overlap.csv`
  - the best current event-ordering file for Phase II analysis
  - DP lane timing is explicit
  - TP / PP portions still include scaffolded ordering in places
  - it is a high-confidence transitional trace, not a full final system-truth trace
  - use it for sequence inspection, transition counting, topology candidate analysis, and gap analysis

### Legacy / diagnostic outputs

- `training_time_breakdown.txt`
  - legacy baseline-style breakdown
  - intentionally kept separate from overlap-aware accounting

- `time_series.csv`
  - legacy timeline
  - not Phase-II-authoritative

- `summary_deepflow.txt`
  - DeepFlow kernel / GEMM summary only
  - not an overlap result

- `mat_dims_amped.txt`
  - AMPeD-to-DeepFlow GEMM mapping only

- `AmpedTraining.txt`
  - legacy-style AMPeD summary text

## 4. Validation Summary

Key validations already completed:

- DP mode comparison:
  - `lump` behaves as the no-overlap baseline
  - `layerwise` reduces exposed DP communication relative to `lump`
  - `bucketed` differs from `layerwise` when the effective bucket size is genuinely larger than one layer

- bucket-size sweep:
  - confirmed requested vs effective bucket size behavior
  - confirmed `0` and `1 layer` collapse to the same effective bucket in the current model
  - confirmed larger buckets often reduce raw communication slightly but increase exposed communication

- cross-parallelism comparison:
  - checked both `DP_1_16_TP_8_1_PP_1_8_8`
  - and `DP_1_16_TP_8_2_PP_1_8_8`
  - current DP-overlap behavior remains logically consistent across both mappings

- bandwidth sensitivity:
  - weaker inter-node communication increases raw DP communication and exposed DP communication as expected

- reconfiguration-time sweep:
  - raw reconfiguration scales linearly with the configured reconfiguration time
  - small reconfiguration values stay effectively hidden
  - sufficiently large values eventually produce exposed reconfiguration

- Stage A trigger validation:
  - a dedicated stress / trigger case (`ocs_reconf_us = 5730000.0`) was used to prove that the stricter Stage A reconfiguration hiding rule is actually triggerable
  - this was mechanism validation, not a realism claim about that parameter value

- Stage B validation:
  - the local consumable-slack mechanism was implemented and checked on the requested mappings and bucket settings
  - in the tested cases, results were numerically unchanged because the current backward-compute windows are continuous and make the old and new communication-hiding rules equivalent in those cases

## 5. What Is Now Trusted

The following can now be treated as trustworthy within the current Phase II modeling scope:

- explicit per-step DP unit scheduling
- `lump`, `layerwise`, and byte-based `bucketed` semantics
- full-training DP raw / hidden / exposed communication totals
- full-training DP-related raw / hidden / exposed reconfiguration totals
- DP-lane-local topology reuse logic from Stage A
- stricter DP-related OCS reconfiguration timing / hiding from Stage A
- overlap-aware output accounting in `training_time_breakdown_overlap.txt`
- effective scheduler settings reported in `config_summary.txt`
- DP-lane event ordering in `time_series_overlap.csv`

## 6. What Is Still Approximate / Limited

Important limitations that remain true:

- DP-lane-local topology truth is not full-system topology truth
- TP / PP ordering in `time_series_overlap.csv` is still partly scaffolded
- cross-lane adjacency in the overlap CSV is not the same thing as a true dependency edge
- Stage B makes DP communication hiding mechanism-explicit, but under the current continuous backward windows it may be numerically equivalent to the old rule in tested cases
- backward compute is still represented by coarse layer-level windows, not kernel-trace-level execution
- this remains a coarse analytical scheduler, not a full execution-trace simulator

## 7. Recommended Final One-Paragraph Conclusion

Phase II is now sufficient to freeze as a completed stage if it is described carefully as a coarse but explicit DP-overlap scheduler with full-training raw / hidden / exposed accounting, DP-lane-local topology-aware OCS reconfiguration, and an overlap-aware event CSV; it should also be stated clearly that full-system topology truth, TP / PP overlap truth, and kernel-trace-level timing remain outside the current Phase II scope.
