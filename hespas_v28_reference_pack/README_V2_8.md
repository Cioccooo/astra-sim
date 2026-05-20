# V2.8 Fair EN / SON / RON with Torus ECMP and Overhead Sensitivity

These results are from our custom analytical evaluator, not ASTRA-sim. ASTRA-sim was only used for earlier smoke tests.

The pipeline is:

HF expert-selection JSON -> parser -> block-by-token source mapping -> block expert placement -> dispatch/combine traffic matrix -> analytical network timing model -> CSV/figures.

V2.8 keeps V2.7's prefill-only workload assumptions and adds:

- `SON torus deterministic`: old deterministic shortest-path torus.
- `SON torus ECMP`: same torus topology and bandwidth, but splits traffic across equal-cost shortest paths.
- EN sensitivity variants: ECMP imbalance factors and latency-sensitive/conservative variants.
- RON W=4 reconfiguration sensitivity: 0us, 1us, 10us per evaluated request.

Main fair comparison:

- `RON calibrated` vs `SON torus ECMP`
- `RON W=4 0us/1us` vs `SON torus ECMP`

Diagnostic baselines:

- `SON ring`: weak degree-2 optical baseline.
- `SON torus deterministic`: shows the penalty from deterministic shortest-path tie-breaking.
- `EN ECMP-imbalance 1.3x`: sensitivity, not the main EN baseline.

Multi-hop optical semantics:

The SON/RON graph-routing model should be interpreted as an abstract optical switching fabric path. If intermediate graph hops are interpreted as GPU/NIC forwarding through intermediate GPUs, that is a serious modelling limitation and the optical multi-hop results should not be treated as physically valid without a forwarding/switching model.

Why EN beat SON in V2.7:

V2.7 gave EN ideal ECMP over a Clos abstraction, while SON torus used deterministic shortest paths. That deterministic torus routing could overload a small number of torus edges. SON ring also has degree 2 but 800Gb/s per link, which can outperform a 400Gb/s/link torus for some traffic patterns despite longer paths. Therefore EN beating SON in some V2.7 cases was mostly a combination of weak SON routing/topology and EN idealisation, not a definitive real-world result.

Do not double count:

Max-link-load / bandwidth already accounts for serialization. Additional EN latency and imbalance rows are sensitivity analyses, not the main model.
