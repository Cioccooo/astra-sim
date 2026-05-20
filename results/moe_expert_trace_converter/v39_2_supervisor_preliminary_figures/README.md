# V39.2 Preliminary Supervisor Figures

## What Each Figure Shows

1. Figure 1 compares prefill-informed OCS against strong static optical
   baselines. This is the first figure to show.
2. Figure 2 shows that expander-like optical topologies outperform SON torus
   under ECMP-4.
3. Figure 3 shows that the torus gap is routing-assumption sensitive.
4. Figure 4 explains why oracle and prefill-informed OCS are close to fair
   universal static.

## What These Figures Can Claim

- Figure 1/2 are optical circuit reference results, not native ASTRA optical
  results.
- The evaluated stage is decode `trace[1:]`.
- Prefill-informed OCS uses only prefill `trace[0]` for topology selection.
- Optical methods use the same degree-4, 400Gb/s/circuit, 1.6Tb/s/GPU budget.

## What These Figures Cannot Claim

- They are not paper-final.
- They are not full serving latency.
- They are not MoE training.
- They are not native ASTRA optical-circuit results.

## Why Figure A/1 Is Main

It is the controlled optical-only comparison. It does not mix EN packet
GraphTopology with optical circuit reference bars.

## Why V39 Figure B Is Reference Only

EN is native ASTRA packet folded-Clos electrical reference. Optical methods use
the optical circuit/capacity reference. This is explicitly mixed semantics and
should not be presented as same-model fairness.

## Why Current ASTRA GraphTopology Is Sensitivity Only

Current ASTRA GraphTopology models packet/store-and-forward routing. SON/RON can
include intermediate GPUs on routes, so it is not transparent optical OCS
semantics.

## Key Caveat To Say Out Loud

Prefill-informed OCS only wins DeepSeek MMLU slightly; Qwen MMLU ties and Qwen
LiveCodeBench loses slightly. The robust result is that expander-like optical
topologies beat torus under ECMP-4, but the torus gap shrinks under optimistic
all-shortest routing.
