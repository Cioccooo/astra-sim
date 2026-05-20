# ASTRA-sim + Chakra/STG/MLSynth Capability Matrix

Date: 2026-05-06

## Short Answer

With the current local toolchain, ASTRA-sim can execute the communication
patterns needed for MoE/expert parallelism, context/sequence-parallel style
communication, and forward-only inference traces if those traces are generated
correctly.

The limiting factor is not ASTRA-sim's trace executor. The limiting factor is
the trace generator layer:

- Chakra is only the trace format and tooling.
- MLSynth is usable for dense training today and has an incomplete MoE layer.
- STG exposes MoE/EP and SP/CP knobs, but the local MoE generation route is not
  currently reliable end-to-end.

## Evidence From Local Code

### ASTRA-sim Trace Executor

ASTRA-sim can issue these Chakra communication nodes:

- Point-to-point `COMM_SEND_NODE`
- Point-to-point `COMM_RECV_NODE`
- Collective `ALL_REDUCE`
- Collective `ALL_TO_ALL`
- Collective `ALL_GATHER`
- Collective `REDUCE_SCATTER`

Relevant code:

- `astra-sim/workload/Workload.cc`: `issue_comm()` dispatches send/recv/collective nodes.
- `astra-sim/workload/Workload.cc`: `issue_coll_comm()` maps `ALL_TO_ALL`,
  `ALL_GATHER`, `REDUCE_SCATTER`, and `ALL_REDUCE` into system calls.

One caveat: `BROADCAST` is not modeled as a native broadcast yet; it is replayed
using node runtime.

### MLSynth

MLSynth contains `TransformerMoeLayer`, and that layer emits EP all-to-all when
`ep_size > 1`.

Important limitation in the source:

- The class docstring says `Incomplete`.
- It has TODOs for routing and imbalanced all-to-all collectives.
- It models balanced token shuffle, not dynamic expert imbalance.
- The current `TransformerMoe` model signature does not plug cleanly into the
  existing `MegatronLM` orchestrator, which passes `npu_id` and `pg_name`.
- Existing `MegatronLM.generate_comm_groups()` generates DP and TP groups, not
  EP groups.

Conclusion: MLSynth can be extended into a good MoE/EP generator, but it is not
turnkey for production MoE today.

### STG

STG exposes the relevant CLI knobs:

- `--ep`: expert parallel degree
- `--experts`: number of experts
- `--kexperts`: selected experts per token
- `--sp`: sequence parallel degree
- `--model_type moe`

Internally, STG defines the `--sp` dimension with a symbol named `cp`, so its
sequence-parallel path overlaps with context-parallel modeling terminology.

The MoE path distributes over `[dp, tp, spp/cp, ep]`, so the symbolic model has
the right axes. However, our local previous end-to-end test
`moe_dp2_tp2_ep2` failed during trace generation, so I would not treat STG MoE
as ready without fixes.

## New Sanity Checks

I added a tiny local generator:

- `capability_explore/generate_moe_ep_trace.py`

It uses MLSynth's `TransformerMoeLayer` directly to generate tiny Chakra ETs
with balanced expert-parallel all-to-all. Then I reused our existing
MLSynth-to-ASTRA postprocessor to rewrite string process-group names into the
numeric process-group IDs that ASTRA currently expects.

### MoE/EP Training-Like Trace

Config:

- 4 ranks
- EP size 4
- 1 MoE layer
- forward + backward
- balanced all-to-all token shuffle
- network: `inputs/network/hgx_h100_4gpu.yml`
- system: `examples/system/native_collectives/HGX-H100-validated.json`

Result:

- ASTRA exit code: 0
- finished ranks: 4/4
- wall time: 45,357 cycles
- exposed communication: 45,351 cycles

Log:

- `capability_explore/tiny_moe_ep/astra_unaware.log`

### MoE/EP Forward-Only Trace

Config:

- same as above
- forward only

Result:

- ASTRA exit code: 0
- finished ranks: 4/4
- wall time: 22,679 cycles
- exposed communication: 22,676 cycles

Log:

- `capability_explore/tiny_moe_ep_forward_only/astra_unaware.log`

This proves that forward-only/inference-shaped traces can be executed by ASTRA.
It does not yet prove accurate autoregressive decoding with KV-cache dynamics.

## Capability Decisions

### MoE And Expert Parallelism

Status: feasible, not turnkey.

ASTRA can run EP all-to-all. The small test proves this. For a real MoE study,
we need to improve the trace generator:

- Add EP process-group generation.
- Add MoE orchestrator support for DP/TP/PP/EP combinations.
- Add router/top-k token distribution.
- Add capacity factor, dropped/padded tokens, and load imbalance.
- Optionally support per-destination all-to-all sizes via custom send/recv traces
  if ASTRA's uniform all-to-all is too coarse.

Best route:

- Start from MLSynth for controlled analytical MoE generation.
- Use Chakra ET as the common output format.
- Feed ASTRA-sim analytical backend for fast iteration.

### Context Parallelism

Status: partially feasible.

ASTRA can execute the required primitive operations if represented in Chakra:

- all-gather
- reduce-scatter
- all-to-all
- point-to-point ring exchange

STG has `--sp`, internally named `cp`, and prior local `sp8` traces ran through
ASTRA. That supports sequence/context-sharded modeling at a coarse level.

But true context parallelism is algorithm-specific. Ring Attention, Ulysses,
Megatron CP, and DeepSpeed-Ulysses-style sequence all-to-all are not identical.
To make results credible, the trace generator must explicitly encode the chosen
CP algorithm's schedule.

Best route:

- For quick studies, use STG/MLSynth-style sequence-parallel approximations.
- For paper-grade CP, implement the exact CP schedule as Chakra collectives or
  explicit send/recv rounds.

### Inference

Status: feasible for prefill/forward-only; incomplete for realistic decode.

ASTRA does not care whether a trace is training or inference. If the trace has
only forward compute and communication, ASTRA will run it.

The new forward-only MoE/EP sanity check completed successfully.

What is missing for realistic LLM inference:

- Prefill vs decode phases.
- Autoregressive token loop.
- KV-cache read/write tensor sizes.
- Batch scheduling, continuous batching, request arrivals.
- Tensor/pipeline/expert parallel communication during decode.
- Optional attention algorithm details for context parallel decode.

STG has `models/transformer_forward_only.py`, but
`models/transformer_inference.py` currently raises `NotImplementedError`, so
decode is not ready out of the box.

## Recommended Roadmap

1. Build a small MLSynth-based generator we control.
2. Keep dense GPT training route as the validated baseline.
3. Add forward-only mode for prefill.
4. Add EP groups and balanced MoE all-to-all.
5. Add router imbalance model for MoE.
6. Add exact context-parallel schedules as selectable templates.
7. Add decode/KV-cache inference mode.
8. Validate each new mode against simple analytical formulas before comparing
   with AMPeD/DeepFlow or paper figures.

