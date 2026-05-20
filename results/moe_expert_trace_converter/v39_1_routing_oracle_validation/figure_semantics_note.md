# Figure Semantics Note

## Figure A

Optical-only controlled comparison. Uses the optical circuit/capacity reference
model. All optical methods share degree=4, 400Gb/s per circuit, 1.6Tb/s per GPU,
ECMP-4 over equal shortest paths, decode evaluation, and prefill-only topology
selection for prefill-informed OCS.

## Figure B

Reference only. EN is native ASTRA packet folded-Clos electrical reference.
Optical bars use the optical circuit/capacity reference. This is explicitly
mixed semantics and must not be described as same-model fairness.

## Figure C

Packet-routing sensitivity only. Native ASTRA GraphTopology treats SON/RON as
packet/store-and-forward graphs; intermediate GPUs appear on SON routes. This is
not the main optical OCS result.
