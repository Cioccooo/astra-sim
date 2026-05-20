# HESPAS V2.8 Reference Pack

This folder is a small historical reference pack copied from the local private
HESPAS checkout. It is not a full HESPAS mirror.

Purpose:

- Preserve the V2.8 analytical EN / SON / RON reference used by early ASTRA-side
  reproduction attempts.
- Support provenance for `tools/v29_astrasim_reproduction_attempt.py` and
  `tools/v30_astrasim_custom_topology_attempt.py`.
- Document the pre-V31 analytical baseline without importing the full private
  HESPAS repository.

Included files:

- `evaluate_en_son_ron_v28.py`
- `README_V2_8.md`
- `v28_full_summary.csv`
- `v28_full_summary.json`
- `validation.json`
- `fig_teacher_32gpu_fair_subset.csv`
- `topology_routing_summary.csv`

Later V31+ ASTRA / GraphTopology / optical-reference work does not require this
pack to run; it is retained only for historical comparison and reproducibility.
