# V39.3 Full-Inference Prediction-Signal Audit

This audit compares OCS control policies on full inference communication:
prefill + decode + observation/default-static phases + reconfiguration penalty.

Model: optical circuit/capacity reference. This is not full serving latency and
not native ASTRA optical execution.

Main penalty for figures: 1us per reconfiguration. Tables include 0us, 1us,
10us, and 25ms.

Methods:

- fair universal static baseline
- phase-warmup OCS
- prefill-warmup OCS
- previous-request OCS
- oracle upper bound

Important: every method includes observed/warm-up traffic and reconfiguration
penalty. Results should not be compared to V39.2 remaining-decode-only bars
without this caveat.
