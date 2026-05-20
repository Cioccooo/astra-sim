# V39.3b Full-Inference Accounting Fix

This version fixes V39.3's accounting issue: prefill and decode are evaluated
sequentially as `eval(prefill) + eval(decode)`, not by merging both stages into
one payload.

The key added method is `full-prefill-informed decode OCS`, which is the
V39.2-compatible full-inference policy:

1. run full prefill under fair universal static,
2. select topology using full prefill only,
3. pay reconfiguration penalty,
4. run decode under the selected topology.

Model: optical circuit/capacity reference. This is inference-only communication
time, not full serving latency and not native ASTRA optical execution.
