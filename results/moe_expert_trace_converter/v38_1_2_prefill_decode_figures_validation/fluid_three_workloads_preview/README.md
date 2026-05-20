# Fluid-Only Three-Workload Preview

This directory contains a fluid-only lower-bound preview for the three required workloads. It is intentionally not the native ASTRA main result.

Method logic:
- topology selection and timing are both fluid-only for this preview;
- prefill-informed OCS is selected using trace[0] prefill only;
- decode evaluation uses trace[1:] only;
- oracle uses decode/evaluation traffic and is an upper bound only.

Use these figures only for trend discussion. The native ASTRA result remains the stricter result and was negative versus fair universal static.
