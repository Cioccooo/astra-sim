# V39 Modelling Note

## A. EN Electrical Reference

EN is an electrical packet folded-Clos / EPS-style reference. Native ASTRA
GraphTopology is appropriate here because it models graph routing, link queues,
serialization, multi-hop forwarding, and ECMP-like path splitting. EN must be
labelled as an electrical reference unless bandwidth and topology are explicitly
normalised against optical methods.

## B. Current SON/RON GraphTopology

The current ASTRA GraphTopology representation of SON/RON uses rank/GPU nodes as
graph vertices. If a route is `GPU -> GPU -> GPU`, the current backend treats it
as packet/store-and-forward routing through intermediate GPUs. This is a valid
packet-routing sensitivity, but it is not transparent optical OCS semantics.

## C. Intended Optical SON/RON/OCS

The intended optical model is a circuit/capacity fabric. Topology controls which
optical resources/circuits carry traffic. Intermediate optical resources consume
capacity but should not imply GPU packet forwarding. Source and destination
injection limits must be included.

The V39 optical reference timing is:

```text
T_phase = max(
  max_src_out_bytes / B_src,
  max_dst_in_bytes / B_dst,
  max_optical_resource_load / B_link
)
```

with `B_src = B_dst = degree * 400Gb/s = 1.6Tb/s` and `B_link = 400Gb/s`.
Dispatch and combine are sequential.

## D. Figure Policy

- Figure A: optical-only controlled comparison. Use optical circuit reference.
- Figure B: EN electrical reference vs optical methods. Label mixed semantics.
- Figure C: packet-routing ASTRA GraphTopology sensitivity. Do not use as main
  optical OCS result.
- Never mix semantics without labels.
