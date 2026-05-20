# Phase II Summary

This document records the completion state of Phase II of the staged `amped_deepflow` overlap refactor.

Phase II is limited to DP overlap plus DP-related OCS reconfiguration.

## Goal

Replace ratio-based DP overlap estimation with an explicit DP scheduler while preserving the old baseline path.

## What Changed

### Step 1: DP scheduler implementation

Files:

- [performance_model.py](/Users/dfx/Python/LLM_analytical_tools/AMPeD/amped/performance_model.py)
- [overlap_scheduler.py](/Users/dfx/Python/LLM_analytical_tools/amped_deepflow/overlap_scheduler.py)

Added:

- explicit backward compute windows
- explicit DP communication windows
- explicit DP OCS reconfiguration windows
- explicit per-unit `ready`, `start`, and `end`
- explicit `raw`, `hidden`, and `exposed` communication totals
- explicit `raw`, `hidden`, and `exposed` OCS reconfiguration totals

### Step 2: DP mode semantics

The three DP modes now mean:

- `lump`:
  one aggregate no-overlap DP unit for the whole optimization step, preserving the legacy-safe baseline behavior
- `layerwise`:
  one DP unit per backward layer, with readiness at that layer's backward completion
- `bucketed`:
  real byte-based buckets, with readiness at the last backward layer that fills each bucket

### Step 3: Config cleanup

Added config controls under `system_architecture_parameters`:

- `dp_overlap_mode`
- `dp_bucket_size_bytes`

Defaults:

- `dp_overlap_mode = 0`
- `dp_bucket_size_bytes = 0`

These defaults preserve legacy-safe behavior unless DP overlap is explicitly enabled.

Exact meaning of `dp_bucket_size_bytes = 0`:

- in bucketed mode, `0` means "use one layer's gradient bytes as the bucket target size"
- because Phase II currently assumes uniform per-layer gradient size, this yields exactly one layer per bucket
- in the current model, `bucketed` with `dp_bucket_size_bytes = 0` is therefore schedule-equivalent to `layerwise`
- this is a Phase II model-specific conclusion, not a general statement about arbitrary models with non-uniform layer gradient sizes

### Step 4: CLI overrides

The main Phase II controls can also be overridden from the `training.py` CLI:

- `--dp-overlap-mode lump`
- `--dp-overlap-mode layerwise`
- `--dp-overlap-mode bucketed`
- `--dp-bucket-size-bytes <bytes>`
- `--ocs-reconf-us <microseconds>`

Examples:

```bash
python training.py --config GPT3_175B_NEW_ocs.json --dp-overlap-mode bucketed --dp-bucket-size-bytes 25000000 --ocs-reconf-us 10.0
python training.py --config GPT3_175B_NEW_ocs.json --dp-overlap-mode layerwise --ocs-reconf-us 1.0
```

Each run rebuilds the runtime mirror from the source config first, then applies CLI overrides only to that runtime mirror. The runtime mirror is restored to the non-overridden per-run base state after the run, so CLI overrides from one run cannot leak into the next.

## Scheduling Definitions

For each DP unit:

- `ready`:
  earliest time the unit's gradients are available
- `start`:
  actual scheduled start of DP work
- `end`:
  actual scheduled finish of DP work
- `raw communication`:
  full DP communication duration before hiding
- `hidden communication`:
  communication scheduled before backward compute ends
- `exposed communication`:
  communication left on the critical path
- `raw / hidden / exposed reconfiguration`:
  the same accounting for DP-related OCS reconfiguration

## Schedule Scale

The explicit schedule unit in Phase II is:

- one optimization step

Full-training totals are derived exactly within the model assumptions by:

- `full-run total = number_of_batches × per-step scheduled total`

## Output Status

The authoritative Phase II result is:

- `training_time_breakdown.txt`

This file now contains the scheduler-based DP totals for:

- raw / hidden / exposed DP communication
- raw / hidden / exposed DP-related OCS reconfiguration

Useful companion output:

- `config_summary.txt`

This file should be used to confirm which Phase II controls were active for the run, including:

- `dp_overlap_mode`
- `dp_bucket_size_bytes`
- `ocs_reconf_per_round`

Still legacy / diagnostic only in Phase II:

- `time_series.csv`
- `summary_deepflow.txt`
- `mat_dims_amped.txt`
- `AmpedTraining.txt`

Current limitation of those legacy outputs:

- `time_series.csv` still comes from the old `training.py` / `time_domain` path and does not reflect the new scheduler-based DP overlap logic
- `summary_deepflow.txt` is still only the DeepFlow GEMM/kernel summary and does not report DP overlap or hidden / exposed OCS terms

## What Is Exact In Phase II

- explicit DP unit readiness
- explicit DP unit start and finish times
- explicit DP communication hiding vs exposure
- explicit DP OCS reconfiguration hiding vs exposure
- full-run totals derived from the explicit per-step schedule

## What Is Still Simplified / Conservative

- backward compute is modeled as uniform per-layer windows from the current AMPeD layer-level compute model
- DP communication uses one conservative inter-node resource timeline
- only DP and DP-related OCS are covered in Phase II
- `lump` is a baseline-preserving aggregate DP unit, not a physical single giant collective

## Key Validation Results

- default baseline behavior remains unchanged when overlap mode is left at `0`
- `layerwise` now produces explicit readiness-based scheduling
- `bucketed` is genuinely different from `layerwise` when bucket size changes
- `raw = hidden + exposed` holds for DP communication totals
- `raw = hidden + exposed` holds for DP-related OCS reconfiguration totals
- full-run totals are reported on one consistent scale
- the new DP overlap path no longer depends on sampled ratio correction

Concrete checked example:

- with 4-layer buckets in the active GPT-3 configuration, the last bucket contains layers `[3, 2, 1, 0]`
- it becomes ready at `2.9549313235 s`
- its raw communication is `0.0170497804 s`
- no backward compute remains after readiness, so its communication is fully exposed

## Caveats

- larger buckets can reduce raw communication slightly by reducing collective startup count
- larger buckets can still increase exposed communication because readiness moves later and leaves less backward compute available for hiding
- OCS reconfiguration can be hidden only when it is scheduled before the corresponding gradients become ready
