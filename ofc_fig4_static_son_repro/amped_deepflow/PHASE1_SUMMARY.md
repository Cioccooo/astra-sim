# Phase I Summary

This document records the completion state of Phase I of the staged `amped_deepflow` overlap refactor.

Phase I is intentionally limited to structural preparation only.

## Goal

Prepare the codebase for a later compute-communication overlap model without changing any existing simulator behavior.

## Scope

Phase I includes exactly two code tasks:

1. Split raw OCS-related timing primitives out of the existing DP inter-node communication path.
2. Add a minimal overlap scheduler skeleton for future DP-overlap work.

Phase I does **not** implement overlap scheduling.

## What Was Changed

### Step 1: New raw timing primitives in `PerformanceModel`

File:

- [performance_model.py](/Users/dfx/Python/LLM_analytical_tools/AMPeD/amped/performance_model.py)

Added methods:

- `ocs_base_alpha(steps: int) -> float`
- `ocs_reconfiguration_time_raw() -> float`
- `dp_allreduce_inter_raw() -> float`

Purpose:

- expose pure OCS startup alpha without reconfiguration
- expose raw per-transition OCS reconfiguration time
- expose raw DP inter-node communication duration with the same old DP algorithm logic, but with OCS reconfiguration split out

Important constraints kept:

- `_alpha_ocs()` was left unchanged
- existing aggregate formulas were left unchanged
- no existing total-time, baseline, or output path was rewired to use the new methods

### Step 2: Minimal overlap scheduler skeleton

File:

- [overlap_scheduler.py](/Users/dfx/Python/LLM_analytical_tools/amped_deepflow/overlap_scheduler.py)

Added dataclasses:

- `ComputeWindow`
- `DPCommWindow`
- `ScheduleResult`

Added function stubs:

- `build_backward_compute_windows(inputs, deepflow_outputs)`
- `build_dp_comm_windows(inputs, perf_model)`
- `schedule_dp_overlap(compute_windows, dp_windows)`
- `summarize(result)`

Current behavior:

- every function stub raises `NotImplementedError`
- no overlap logic exists yet

## Files Changed

Code files changed in Phase I:

- [performance_model.py](/Users/dfx/Python/LLM_analytical_tools/AMPeD/amped/performance_model.py)
- [overlap_scheduler.py](/Users/dfx/Python/LLM_analytical_tools/amped_deepflow/overlap_scheduler.py)

Documentation added in Phase I completion:

- [PHASE1_SUMMARY.md](/Users/dfx/Python/LLM_analytical_tools/amped_deepflow/PHASE1_SUMMARY.md)

## What Was Explicitly Not Changed

The following were intentionally left untouched:

- [training.py](/Users/dfx/Python/LLM_analytical_tools/amped_deepflow/training.py)
- `time_domain`
- TP overlap logic
- PP overlap logic
- ZeroDP overlap logic
- config schema
- CLI flags
- output naming
- output directory layout
- timestamp behavior

## Primitive Origins

The new raw primitives were derived from existing code as follows:

- `ocs_base_alpha()`:
  derived from the old `_alpha_ocs()` startup-latency term
- `ocs_reconfiguration_time_raw()`:
  derived from the old `_alpha_ocs()` amortized reconfiguration term
- `dp_allreduce_inter_raw()`:
  derived from `communication_time_backwards_DP_all_reduce_inter()`, preserving the same branch structure, step logic, and payload/beta logic

## Validation Performed

Phase I was validated with a short compatibility check.

### Import check

Confirmed that the modified code imports successfully in the project Python environment.

### Backward compatibility

Ran one existing baseline configuration before and after the Phase I code changes in isolated copies of the repository and compared all generated output artifacts.

Configuration used:

- `GPT3_175B_generate.json` with active run config `DP_1_16_TP_8_1_PP_1_8_8`

Comparison method:

- normalized output filenames by removing timestamp prefixes
- compared the generated file sets
- compared file contents for:
  `AmpedTraining.txt`, `config_summary.txt`, `mat_dims_amped.txt`,
  `summary_deepflow.txt`, `time_series.csv`, and
  `training_time_breakdown.txt`

Result:

- output file sets matched
- file contents matched
- existing baseline behavior remained unchanged

### Primitive sanity checks

Confirmed:

- raw OCS reconfiguration matches `ocs_reconf_per_round / ocs_slot_microbatches`
- `_alpha_ocs(steps)` is exactly equal to:
  `ocs_base_alpha(steps) + steps * ocs_reconfiguration_time_raw()`
- the new raw DP inter-node primitive preserves the old DP algorithm logic and differs from the old OCS aggregate only by removal of the explicit reconfiguration contribution

### No unintended integration

Confirmed that no existing baseline, aggregate, or output path calls the new primitives yet.

## Phase I End State

At the end of Phase I:

- the current simulator behavior is unchanged
- raw OCS and raw DP inter-node timing pieces are now available
- a DP-overlap scheduler skeleton exists
- overlap scheduling itself is still unimplemented

This is the intended handoff point into Phase II.
