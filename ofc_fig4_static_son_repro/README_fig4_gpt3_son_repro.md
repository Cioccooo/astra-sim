# OFC 2026 Figure 4 GPT-3 175B SON Reproduction

Target bar:

- Paper: `Fangxiao Dong_ofc_2026.pdf`
- Figure: Fig. 4, first model group
- Model: GPT-3 175B
- Network: SON, static optical network
- Parallelism: DP-TP-PP = 16-8-8
- Scale: 1024 GPUs
- Inter-node bandwidth: 1.6 Tb/s per node, represented as 8 NICs x 200 Gb/s

Local setup:

- Reproduction directory: `/Users/dfx/Python/astra-sim/ofc_fig4_static_son_repro`
- Copied source: `amped_deepflow`
- Symlinked dependencies: `AMPeD`, `DeepFlow`
- Patched copied `amped_deepflow/training.py` to run only:
  `DP_1_16_TP_8_1_PP_1_8_8`
- Config used:
  `amped_deepflow/amped_backups/ofc_config/GPT3_175B_NEW_opt.json`

Dependency notes on macOS:

- The original `.venv` under `/Users/dfx/Python/LLM_analytical_tools/amped_deepflow/.venv`
  is a Linux ELF environment and cannot run on macOS.
- A local macOS venv was created at:
  `amped_deepflow/.venv`
- Required pins:
  `setuptools<81`, because AMPeD imports deprecated `pkg_resources`.
  `ruamel.yaml<0.18`, because DeepFlow uses the pre-0.18 `yaml.load(..., Loader=...)` API.

Run command:

```bash
cd /Users/dfx/Python/astra-sim/ofc_fig4_static_son_repro/amped_deepflow
.venv/bin/python training.py --config ofc_config/GPT3_175B_NEW_opt.json
```

Latest output parsed from:

`output_files/DP_1_16_TP_8_1_PP_1_8_8/dp_lump_pp_off_tp_off/2026-05-03_13-23-02_training_time_breakdown.txt`

Result:

- Total training time: `393575.6161622606 s` = `109.326560 h`
- Total computation time: `308857.6145763703 s` = `85.793782 h`
- Total communication time: `84516.1044883728 s` = `23.476696 h`
- Pipeline bubble wait: `201.8970975174765 s` = `0.056083 h`

Interpretation:

This is a close sanity-check reproduction of the GPT-3 175B SON bar in Fig. 4.
The visual bar in the PDF is approximately compute 86 h plus communication 23-25 h,
for roughly 109-111 h total.

