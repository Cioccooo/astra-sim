# training.py 总结与重点详解

## 一、training.py 在做什么（总结）

**training.py** 是 amped_deepflow 的**唯一入口脚本**，做三件事：

1. **配置与循环**：解析 `--config`，对每个并行配置名（如 `DP_1_16_TP_8_1_PP_1_8_8`）调用 `update_configs()` 改写 JSON，并设置本轮的输出目录 `output_files/{RESULT_DIR}`。

2. **串联五步流水线**：  
   **AMPeD**（解析型训练时间）→ **mat_dims_ampedToDF**（从 AMPeD 推出每层 GEMM 维度）→ **DeepFlow**（按维度算 GEMM/reduction 时间）→ **cal_time**（在内存里合并）→ **time_domain**（按时间顺序模拟并写出时间线 CSV）。  
   每一步的输入都来自上一步的输出，形成一条单向数据链。

3. **写结果**：把每步产生的文本/CSV 写到 `output_files/{RESULT_DIR}/`（config_summary、AmpedTraining、training_time_breakdown、mat_dims_amped、summary_deepflow、time_series.csv 等）。

整体上，**training.py = 入口 + 配置更新 + 五步流水线编排 + 输出路径管理**；具体计算由 AMPeD、DeepFlow 和本文件内的几个类完成。

---

## 二、最重要的几个部分与运作方式

下面按“在流程中的顺序”说明：每个部分做什么、输入输出是什么、内部如何运作。

---

### 1. 入口：`if __name__ == "__main__"` 与配置循环

**位置**：文件末尾。

**在做什么**：

- 用 `argparse` 解析 **`--config`**（必填），指定要用来更新并行等字段的 JSON 路径。
- 定义 **`itrers`**：一串配置名字，例如 `"DP_1_16_TP_8_1_PP_1_8_8"`。当前启用的是其中一组（如 Megatron 1T 对应的一条）。
- 对 **`itrers`** 里每个 **`cSim`**：
  1. 调用 **`update_configs(cSim, args.config)`**，把该配置写进 JSON（并同步到 `amped_backups/config.json`）。
  2. 设置 **`RESULT_DIR = cSim`**、**`OUTPUT_PATH = output_files/{RESULT_DIR}`**（全局变量，后面所有写文件都用它）。
  3. 依次执行：**amped_exec** → **mat_dims_ampedToDF** → **deepflow_exec** → **cal_time** → **time_domain**，并打印当前执行到哪一步。

**衔接**：`update_configs` 保证本轮用的 config 与 `cSim` 一致；后面的 `Inputs()` 会读到这份 config，所以整条流水线都基于同一并行配置。

---

### 2. update_configs(cSim, config_file)

**作用**：根据配置名字符串，把 DP/TP/PP 的 intra/inter 和每节点 GPU 数写进 JSON，并固定一组“有效性能比例”，避免被 AMPeD 重算覆盖。

**运作**：

- **解析 cSim**：例如 `"DP_1_16_TP_8_1_PP_1_8_8"` 按 `_` 拆成  
  `DP_intra=1, DP_inter=16, TP_intra=8, TP_inter=1, PP_intra=1, PP_inter=8, intraGPUs=8`。
- **读 JSON**：从 `config_file` 解析出的路径读入（并支持 `amped_backups/config_sets/config.json` 等），得到 `data`。
- **改 mapping_parameters**：  
  `intra_node_data_parallel_degree`、`inter_node_data_parallel_degree`、  
  `intra_node_tensor_parallel_degree`、`inter_node_tensor_parallel_degree`、  
  `intra_node_pipeline_parallel_degree`、`inter_node_pipeline_parallel_degree`  
  设为上面解析出的整型。
- **改 system_architecture_parameters**：  
  `number_of_accelerators_per_node`、`number_of_network_cards_per_node` = `intraGPUs`；  
  四个 `effective_perf_perc_*` 设为 0.7，且 **`calculated: false`**，防止 AMPeD 用公式覆盖。
- **写回**：用 `_dump_pretty` 写回“原始 config 文件”和 **`amped_backups/config.json`**，保证 AMPeD 读到的就是本轮配置。

**衔接**：之后 `Inputs()` 会读 `amped_backups/config.json`，所以 **AMPeD 整条链都基于这一份配置**。

---

### 3. amped_exec（AMPeD 阶段）

**作用**：用当前 config 跑一遍 AMPeD 解析型性能模型，得到“训练时间分解”和 `perf_model`，并写出 config_summary、AmpedTraining、training_time_breakdown；产出 **AMPeD** 对象供后续步骤使用。

**构造与入口**：  
`__init__(training=True)` 只调 **`self.main()`**。  
`main()` 里：

1. **建 Inputs**：`inputs = Inputs()`（读当前 `amped_backups/config.json` 和 amped_backups 的 lookup_tables），得到完整 **parameters**。
2. **算时间**：调用 **`self.calc_time(inputs, inputs.parameters["context"], False, False)`**，得到三个量：  
   `computetime_fwd_pass`、`commtime_fwd_pass`、`commtime_pipeline_bubble`（以及训练时还有反向与气泡的对应量）。

**calc_time 内部运作**（核心）：

- 若为推理，会改 `inputs.parameters` 的 context、tokens_to_train、summarization_len 等，并重新算一遍依赖参数（`calculate_parameter`、`Parameters(...)`）。
- **`perf_model = PerformanceModel(inputs)`**：用 AMPeD 的 performance_model，只依赖 `inputs.parameters`。
- 用 perf_model 算 **per-layer** 时间：
  - 前向：`compute_time_forward_pass()` 除以 DP×TP×PP → per-layer 计算时间；`communication_time_forward_pass()` 为前向通信；再根据 PP 和 microbatches 算 **pipeline bubble**。
  - 训练时还有：反向计算+权重更新、DP all-reduce + 反向通信、以及反向的 pipeline bubble。
- 再乘以 **`number_of_batches` × `layers`**，得到**总**的 computetime、commtime、commtime_pipeline_bubble，并 **`self.perf_model = perf_model`**，**`self.breakdown`** 在写 training_time_breakdown 时由 `training_string_training_time_breakdown()` 填满（调用 perf_model 的各种 `total_*`）。
- 返回值是 **三个标量**：总计算时间、总通信时间、总流水线气泡时间。

**main() 后半段**：

- 把“全量参数”写成 **config_summary.txt**；把 Stage 0（和若有）Generation 各 stage 的 compute/comm/bubble 写成 **AmpedTraining.txt**（或 AmpedInference.txt）；用 **training_string_training_time_breakdown()** 生成 **training_time_breakdown.txt**（总时间、各通信/计算分项、TFLOPS 等）。
- **`self.inputs = inputs`**，这样 **AMPeD 对象** 上就有 **inputs、breakdown、perf_model、timeStamp**，供后面 mat_dims、deepflow、cal_time、time_domain 使用。

**衔接**：**AMPeD** 作为第一个“结果对象”传给 **mat_dims_ampedToDF(amped=AMPeD)**，并继续传到 **deepflow_exec(AMPeD, dims)**、**cal_time(AMPeD, ...)**、**time_domain(AMPeD, ...)**。

---

### 4. mat_dims_ampedToDF（维度提取）

**作用**：从 AMPeD 的 parameters 推导出**每层、每个 GEMM 的矩阵维度 (M, N, K) 和并行类型 (CR/RC)**，并写出 mat_dims_amped.txt；产出 **dims** 列表供 DeepFlow 使用。

**运作**：

- **输入**：`amped`（AMPeD 对象），主要用 **`amped.inputs.parameters`** 里的 B, D, S, h, nheads, h_MLP1, h_MLP2, N_DP, N_PP。
- **mmm_breakup(B, D, S, h, nheads, h_MLP1, h_MLP2, N_DP, N_PP)**：  
  按 Transformer 一层 6 个 GEMM 的公式，算出 6 组 **[M, N, K, "CR"/"RC", level_name]**：  
  X.W=KQV, Q.K=R, R.V=Z, Z.W=Y, Y.WL1=O1, O1.WL2=O2。  
  训练时考虑 DP/PP 对 batch/seq 的切分（例如 M 为 `B*S/N_DP/N_PP` 的倍数等）。
- **输出**：**`self.dims`**（list，下标即层内 GEMM 序号），以及 **mat_dims_amped.txt**（每行一组维度信息）。
- **main(dims)** 在 **deepflow_exec** 里会遍历的就是这个 **dims**。

**衔接**：**dims** 作为第二个关键数据传给 **deepflow_exec(AMPeD, mat_dims.dims)**。

---

### 5. deepflow_exec（DeepFlow 阶段）

**作用**：对 **dims** 里每一个 (M, N, K, 并行类型) 调用 DeepFlow 的 **TimeCalculation**，得到该 GEMM 的 **gemm_time** 和 **reduction_time**，汇总成 **deepflow_outputs**，并写 **summary_deepflow.txt**。

**运作**：

- **初始化**：保存 `amped`、`dims`、`TP_DEGREE`、`CONFIG_DIR`（v100.yaml 所在目录）、`OUTDIR`（= OUTPUT_PATH）；清空 **`self.deepflow_outputs`**，然后 **`self.main(dims)`**。
- **deepflow_function(...)**（单次 GEMM）：
  - **exp_config**：传入的是 `f"{CONFIG_DIR}/v100.yaml"` 路径；  
    **`exp_config = config.parse_config(exp_path)`**（DeepFlow 的 config）读 YAML，得到 FullConfig。
  - **TC = TimeCalculation(exp_config)**：DeepFlow 的 perf 模型，内部用 V100 的核、内存、网络等。
  - 根据 **并行类型 t（CR/RC）** 和 **kp1, kp2**：
    - 无并行 (kp1==1 and kp2==1)：**TC.getCf(m, k, n)** → [gemm_time, 0]。
    - CR：**TC.getDistGEMM_f_kp1(m, k, n, kp1, "Cf_CR")**。
    - RC：**TC.getDistGEMM_f_kp2(m, k, n, kp1, kp2, "Cf_RC")**。  
  返回 **[gemm_time, reduction_time]**。
- **main(dims)**：
  - 对 **dims** 的每个下标 **i**：根据是否为 training、dims[i][3] 是 CR 还是 RC，设置 t、kp1、kp2（训练时 CR 用 TP_DEGREE 和 1，RC 用 TP_DEGREE 和 TP_DEGREE），调用 **deepflow_function** 得到 **temp_outputs**。
  - 把 temp_outputs 加上层名、M,N,K 等信息 append 到 **deepflow_outputs**；若是 MHA 的四个 GEMM 或 FFN 的两个，则累加到 **mha_GEMMtime/mha_reduction** 或 **ffn_GEMMtime/ffn_reduction**；在每层最后一个 GEMM（O1.WL2=O2）后，往 deepflow_outputs 里追加两条汇总：**[mha_GEMMtime, mha_reduction, "MHA", ...]** 和 **[ffn_GEMMtime, ffn_reduction, "FFN", ...]**，再清零 MHA/FFN 累加器。
  - 同时把每行写入 **summary_deepflow.txt**（Layer, M, N, K, t, GEMM time, reduction time）。

**衔接**：**deepflow_outputs** 和 **AMPeD** 一起传给 **cal_time(AMPeD, DeepFlow.deepflow_outputs)** 和 **time_domain(AMPeD, DeepFlow.deepflow_outputs)**。  
time_domain 里会用 **deepflow_outputs[6][0]** 和 **deepflow_outputs[7][0]** 作为 MHA/FFN 的汇总时间（索引 6、7 对应每层追加的 MHA、FFN 两条）。

---

### 6. cal_time（后处理，仅内存）

**作用**：把 **DeepFlow 的 GEMM/reduction 时间** 与 **AMPeD 的 breakdown**（通信、权重更新、流水线气泡）在内存里合并，得到“用 DeepFlow 细化 GEMM 后的总时间”视图；不写文件。

**运作**：

- **time_from_GEMM()**：遍历 **deepflow_outputs**，对每个元素取 `[0]`（gemm_time）、`[1]`（reduction_time）累加；再乘 2（前向+反向），得到 **t_FW_BW**、**t_REDUCTION**。
- **main(inputs)**：  
  **总时间 = layers × nbatch × t_FW_BW**  
  **+ breakdown["Total communication time forward pass (s)"]**  
  **+ breakdown["Total communication time backward pass (s)"]**  
  **+ breakdown["Computation time weight updates (s)"]**  
  **+ breakdown["Waiting Time due to pipeline bubbles (s)"]**。  
  即：用 DeepFlow 的 GEMM 时间替代 AMPeD 里对应部分，其余仍用 AMPeD 的通信与气泡。  
  结果可在 debug 时打印，或存到 **return_outputs**（当前主流程未再用）。

**衔接**：逻辑上为“时间视图合并”；后续 **time_domain** 不直接读 cal_time 的返回值，而是用 AMPeD 的 perf_model + deepflow_outputs 的 MHA/FFN 时间自己算时间线。

---

### 7. time_domain（时间线生成与写出）

**作用**：按“时间顺序”模拟多 batch、多层的 **前向 → 反向 → PP 开销 → DP → zeroDP → 权重更新**，把每一步的（Layer, Type, start/end, duration, Bytes, Collective type, Parallelism, Locality, Degree）压入 **timeline**，最后写出 **time_series.csv**。

**运作**：

- **初始化**：从 **amped** 取 **inputs**、**perf_model**；从 **deepFlow_res**（即 deepflow_outputs）取 **deepflow_mhatime**、**deepflow_ffntime**（即上面说的 [6][0]、[7][0]）。  
  预计算 **linear_throughput**（reciprocal_of_OPS）、**non_linear_throughput**（C_NONLIN），以及 B、uB、D、S、layers、N_DP、N_TP、N_PP 等，供各 pass 使用。
- **main(inputs, perf)**：
  - 对 **NR_BATCHS_TO_PROCESS** 个 batch（默认 2）：
    - 对每一层：**forward_pass**（用 perf 的通信时间、DeepFlow 的 MHA/FFN 时间等往 **self.timeline** 里 append 一条记录）；若 PP>1 且在“PP 边界层”，再 **pp_overhead_FWD**。
    - 对每一层：**backward_pass**；同样在 PP 边界 **pp_overhead_BWD**。
    - 若 DP>1 或 PP>1：**dp_overhead**；若 TP>1 或 PP>1：**zeroDP**。
    - 对每一层：**weight_update**。
  - 最后 **total_timeline()**：把 **self.timeline** 写成 **{OUTPUT_PATH}/{timeStamp}time_series.csv**（表头 + 每行一条时间事件）。
- **forward_pass / backward_pass / pp_overhead_* / dp_overhead / zeroDP / weight_update**：每个函数根据当前层/阶段算 duration、通信量、collective 类型等，更新 **self.start**，并 append **[layer_name, type, start, end, duration, bytes, collective, parallelism, locality, degree]** 到 **self.timeline**。

**衔接**：读 **AMPeD** 和 **deepflow_outputs**，不再产生给后续步骤的数据；输出只有 **time_series.csv**，供人或其他工具分析时间线。

---

## 三、前后衔接总览（数据流）

```
__main__
  → update_configs(cSim)           → 写 amped_backups/config.json
  → OUTPUT_PATH = output_files/{cSim}
  → AMPeD = amped_exec()            → inputs, breakdown, perf_model, 3 个 txt
  → mat_dims = mat_dims_ampedToDF(AMPeD)  → dims, mat_dims_amped.txt
  → DeepFlow = deepflow_exec(AMPeD, dims) → deepflow_outputs, summary_deepflow.txt
  → cal_time(AMPeD, deepflow_outputs)      → 仅内存
  → time_domain(AMPeD, deepflow_outputs)   → time_series.csv
```

- **config** 由 update_configs 统一写好，Inputs/AMPeD 只读。
- **AMPeD** 贯穿整条链；**dims** 连接 AMPeD 与 DeepFlow；**deepflow_outputs** 连接 DeepFlow 与 cal_time、time_domain。
- 所有文件都写向 **OUTPUT_PATH**，由每轮开头的 **RESULT_DIR = cSim** 决定。

以上即为 training.py 的总结以及其中最重要部分的详细运作方式与衔接关系。
