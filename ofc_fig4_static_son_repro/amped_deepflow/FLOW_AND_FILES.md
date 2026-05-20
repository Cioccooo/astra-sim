# amped_deepflow：Input / Output 与完整运行流程

本文档整理 amped_deepflow 的**输入**、**输出**以及**代码从头到尾的执行流程**，并说明重要文件与关键代码的作用。

---

## 一、总体架构与依赖

- **入口脚本**：`training.py`（`if __name__ == "__main__"`）
- **外部依赖**（通过 `sys.path`）：
  - `../AMPeD`：AMPeD 分析模型（Inputs、PerformanceModel、save_GEMM_breakdown 等）
  - `../DeepFlow`：DeepFlow 性能模型（`config`、`perf.TimeCalculation`）
- **本地模块**：
  - `amped_backups.inputs`：封装后的 Inputs / 参数计算
  - `config`、`perf` 来自 **DeepFlow**（不是 amped_backups）

---

## 二、Input（输入）

### 1. 主配置：`config.json`

- **实际被读的路径**：`amped_backups/config.json`  
  （可由 `training.py` 的 `update_configs()` 从你指定的 `--config` 同步写入）
- **内容结构**：
  - **neural_network_training_parameters**：模型与训练参数（batch_size、context、layers、dimensionality、lookup 表名/行等）
  - **mapping_parameters**：并行与映射（DP/TP/PP 的 intra/inter、tile_block_size 等）
  - **system_architecture_parameters**：系统与网络（带宽、延迟、每节点卡数、effective_perf_perc_* 等）
  - **accelerator_architecture_parameters**：加速器规格（频率、核心数、FMA、内存等，可来自 lookup 表）

其中很多项为 `"calculated": true`，由 `amped_backups/inputs.py` 中的 `calculate_functions` 根据其它参数计算得到。

### 2. Lookup 表

- **文件**：`amped_backups/lookup_tables.json`
- **作用**：为 config 里 `from_lookup_table: true` 的字段提供取值（如 Megatron 1T、H100 SXM5 等行）。

### 3. 命令行

- `training.py` 主流程：
  - `--config`：指定要用来**更新**并行等配置的 JSON（会同步写入 `amped_backups/config.json`）。
- `amped_backups/inputs.py` 的 Inputs 还支持：
  - `--config`：覆盖默认的 config 路径
  - `--GEMM`、`--compute_graph`：保存 GEMM 分解与计算图（若 AMPeD 侧启用）

### 4. DeepFlow 硬件配置（YAML）

- **路径**：`deepflow_configs/v100.yaml`（在 `training.py` 中写死为 `DEEPFLOW_CONFIG_PATH` 下的 `v100.yaml`）
- **作用**：DeepFlow 的 `config.parse_config(exp_path)` 读取该 YAML，得到模型参数、软件参数、工艺/核心/内存层次、网络拓扑等，供 **perf.py（DeepFlow）** 的 `TimeCalculation` 做 GEMM/通信时间估算。

---

## 三、Output（输出）与 output_files

所有结果写入 **`output_files/`**，子目录名 = 当前配置名（如 `DP_1_16_TP_8_1_PP_1_8_8`）。  
路径在 `training.py` 中为：  
`OUTPUT_PATH = f"/home/fd420/LLM_analytical_tools/amped_deepflow/output_files/{RESULT_DIR}"`，其中 `RESULT_DIR` 来自当前迭代的配置字符串（如 `cSim`）。

### 按生成阶段分类

| 阶段 | 文件/内容 | 说明 |
|------|------------|------|
| **1) update_configs** | 无新文件 | 只改写 `amped_backups/config.json`（及你给的 `--config` 文件） |
| **2) AMPeD (amped_exec)** | `{timeStamp}config_summary.txt` | 全量参数的可读摘要 |
| | `AmpedTraining.txt` / `AmpedInference.txt` | 各 Stage 的 compute/communication/pipeline bubble 时间 |
| | `{timeStamp}training_time_breakdown.txt` | 训练时间分解（总时间、前向/反向/通信/流水线气泡等） |
| **3) mat_dims_ampedToDF** | `{timeStamp}mat_dims_amped.txt` | 每层 GEMM 的 M,N,K 与类型（CR/RC）、层名，供 DeepFlow 使用 |
| **4) DeepFlow (deepflow_exec)** | `{timeStamp}summary_deepflow.txt` | 每层 GEMM 的 M,N,K,t, GEMM time, reduction time；以及 MHA/FFN 汇总 |
| **5) cal_time** | 无单独文件 | 在内存中把 DeepFlow 的 GEMM/reduction 时间与 AMPeD 的 breakdown 结合，用于后续 time_domain |
| **6) time_domain** | `{timeStamp}time_series.csv` | 时间线：Layer, Type, start/end time, duration, Bytes, Collective type, Parallelism, Locality, Degree |

可选（代码中已注释）：
- `individual_timeline()`：可生成 `{timeStamp}time_series_GPU_{id}.csv` 等 per-GPU 时间线
- `mat_dims_astrasim.csv` / astrasim 相关：由 `astrasim_workload` 生成，主流程未调用

---

## 四、代码运行流程（从头到尾）

### 1. 入口与配置更新（training.py __main__）

1. 解析 `--config`。
2. 对每个配置字符串（如 `DP_1_16_TP_8_1_PP_1_8_8`）：
   - 调用 **`update_configs(cSim, args.config)`**：
     - 从配置名解析 DP_intra/inter、TP_intra/inter、PP_intra/inter、intraGPUs；
     - 更新 JSON 中 `mapping_parameters` 和 `system_architecture_parameters`（含 `number_of_accelerators_per_node`、`number_of_network_cards_per_node`）；
     - 将 `effective_perf_perc_*` 设为 0.7 且 `calculated: false`；
     - 写回 `args.config` 和 **`amped_backups/config.json`**。
3. 设置 **`OUTPUT_PATH = output_files/{RESULT_DIR}`**（`RESULT_DIR = cSim`）。

### 2. AMPeD 阶段：`amped_exec(training=True)`

- **类**：`amped_exec`（在 training.py 内）
- **流程**：
  1. **`main()`**：
     - 使用 **`Inputs()`** 读 **`amped_backups/config.json`**（及 lookup_tables），得到 `inputs.parameters`。
     - **`calc_time(inputs, ...)`** 内：
       - 用 `Parameters` + 依赖计算所有 calculated 参数；
       - 创建 **`PerformanceModel(inputs)`**（AMPeD 的 `amped.performance_model.PerformanceModel`）；
       - 计算 per-layer 前向/反向计算与通信时间、流水线气泡等，并汇总为 computetime、commtime、commtime_pipeline_bubble。
     - 写 **`config_summary.txt`**、**`AmpedTraining.txt`**、**`training_time_breakdown.txt`** 到 `OUTPUT_PATH`。
  2. 得到 **`self.inputs`**、**`self.breakdown`**、**`self.perf_model`**，供后续步骤使用。

重要文件/逻辑：
- **amped_backups/inputs.py**：`Inputs` 读 config + lookup_tables，按依赖顺序计算 `calculate_functions` 中的参数，得到 `parameters`。
- **AMPeD** 的 **PerformanceModel**：根据 parameters 计算前向/反向/通信/流水线气泡等时间，并汇总为 `breakdown` 字典。

### 3. 从 AMPeD 到 GEMM 维度：`mat_dims_ampedToDF(amped=AMPeD, training=True)`

- **类**：`mat_dims_ampedToDF`（定义在 training.py 内，与独立脚本 `mat_dims_mapedToDF.py` 逻辑对应）
- **作用**：根据 `AMPeD.inputs.parameters`（B, D, S, h, nheads, h_MLP1, h_MLP2, N_DP, N_PP）生成每层 GEMM 的 **(M, N, K, 并行类型, 层名)**。
- **流程**：
  - **`main(amped.inputs)`** → **`mmm_breakup(...)`**：
    - 训练时 6 个 level：X.W=KQV, Q.K=R, R.V=Z, Z.W=Y, Y.WL1=O1, O1.WL2=O2；
    - 每个 level 对应一组 [M, N, K, "CR"/"RC", level_name]；
    - 写入 **`{timeStamp}mat_dims_amped.txt`**，并设置 **`self.dims`**（list/dict 形式的维度列表）。
- **输出**：**`mat_dims.dims`** 作为 DeepFlow 的输入。

### 4. DeepFlow 阶段：`deepflow_exec(AMPeD, mat_dims.dims)`

- **类**：`deepflow_exec`（在 training.py 内）
- **依赖**：**DeepFlow** 的 **`config`**、**`perf.TimeCalculation`**（即 **DeepFlow/perf.py**）。
- **流程**：
  1. 对 **dims** 中每个 GEMM：
     - 根据 M,N,K 与并行类型（CR/RC）、TP 度选择 **`deepflow_function(...)`**：
       - **`exp_config = config.parse_config(exp_path)`** 读 **`deepflow_configs/v100.yaml`**；
       - 创建 **`TimeCalculation(exp_config)`**，可选 **`TC.updateParams(..., m, n, k, t, kp1, kp2, ...)`**；
       - 若 **validating_GEMM**：
         - 无并行：**`TC.getCf(m,k,n)`**；
         - CR：**`TC.getDistGEMM_f_kp1(m,k,n,kp1,"Cf_CR")`**；
         - RC：**`TC.getDistGEMM_f_kp2(m,k,n,kp1,kp2,"Cf_RC")`**；
       - 得到 **[gemm_time, reduction_time]**，并附带 MHA/FFN 聚合。
     - 将结果追加到 **`self.deepflow_outputs`**，并写入 **`{timeStamp}summary_deepflow.txt`**。
  2. **输出**：**`DeepFlow.deepflow_outputs`**（每层 GEMM/reduction 时间等），供 **cal_time** 和 **time_domain** 使用。

重要文件：
- **DeepFlow/perf.py**：`TimeCalculation` 使用 DeepFlow 的 Model、Core、MemoryHierarchy、Network 等，根据 M,N,K 和并行策略计算 GEMM 与 reduction 时间。
- **DeepFlow/config.py**：`parse_config(filename)` 解析 v100.yaml，返回 FullConfig（model/sw/tech/system/memory/network 等）。

### 5. 后处理时间：`cal_time(AMPeD, DeepFlow.deepflow_outputs)`

- **类**：`cal_time`（在 training.py 内）
- **作用**：用 DeepFlow 的 GEMM/reduction 时间替换/结合 AMPeD 的 breakdown，得到“按 GEMM 细化后的”总时间（含 FW+BW、reduction、与 AMPeD 的通信/流水线气泡等）。  
- **逻辑**：**`time_from_GEMM()`** 遍历 `deepflow_outputs` 累加 gemm_time 和 reduction_time，再按层数和 batch 与 **`self.breakdown`** 中的通信、weight update、pipeline bubble 组合。  
- **输出**：仅内存中的 **`return_outputs`**，不直接写文件；其时间观被 **time_domain** 间接使用（time_domain 用 AMPeD 的 perf_model 与 deepflow 的 MHA/FFN 时间做时间线）。

### 6. 时间线：`time_domain(AMPeD, DeepFlow.deepflow_outputs)`

- **类**：`time_domain`（在 training.py 内）
- **作用**：按 batch 与层循环，模拟前向、反向、PP 开销、DP、zeroDP、weight_update，把每步的 **Layer, Type, start/end, duration, Bytes, Collective type, Parallelism, Locality, Degree** 压入 **`self.timeline`**，并写入 CSV。
- **流程**：
  - 从 **AMPeD** 取 **inputs**、**perf_model**，从 **deepflow_outputs** 取 MHA/FFN 时间；
  - **`main(inputs, perf)`**：对 `NR_BATCHS_TO_PROCESS` 个 batch，每层依次调用 **forward_pass**、**pp_overhead_FWD**、**backward_pass**、**pp_overhead_BWD**、**dp_overhead**、**zeroDP**、**weight_update**，更新 `self.timeline`；
  - **`total_timeline()`**：将 **`self.timeline`** 写入 **`{OUTPUT_PATH}/{timeStamp}time_series.csv`**。
- **可选**：**`individual_timeline(total_GPUs, ...)`** 可写 per-GPU 的 CSV（当前主流程中已注释）。

---

## 五、重要文件与职责小结

| 文件 | 作用 |
|------|------|
| **training.py** | 主入口；定义 `update_configs`、`amped_exec`、`mat_dims_ampedToDF`、`deepflow_exec`、`cal_time`、`time_domain`；串联 AMPeD → mat_dims → DeepFlow → cal_time → time_domain，并写 output_files。 |
| **amped_backups/config.json** | AMPeD 与 mapping 的主配置；可由 `update_configs()` 从 `--config` 同步写入。 |
| **amped_backups/inputs.py** | 读 config + lookup_tables，计算所有 calculated 参数，提供 **Inputs** 和 **Parameters**。 |
| **amped_backups/lookup_tables.json** | 为 config 中 from_lookup_table 的项提供数值。 |
| **deepflow_configs/v100.yaml** | DeepFlow 的硬件/模型/软件/调度配置，供 **config.parse_config** 和 **perf.TimeCalculation** 使用。 |
| **DeepFlow/perf.py** | **TimeCalculation**：按 M,N,K 与并行策略算 GEMM 与 reduction 时间（getCf、getDistGEMM_f_kp1/kp2 等）。 |
| **DeepFlow/config.py** | **parse_config**：解析 YAML 为 FullConfig。 |
| **mat_dims_mapedToDF.py** | 独立脚本版的 AMPeD → GEMM 维度映射；与 training.py 内 **mat_dims_ampedToDF** 逻辑一致，主流程使用 training.py 内嵌类。 |
| **astrasim_workload.py** | 生成 AstraSim 兼容的 workload（mat_dims_astrasim 等）；主流程未调用，可选用于网络仿真。 |

---

## 六、数据流简图

```
--config (可选) ──┐
                 ├── update_configs() ──► amped_backups/config.json
amped_backups/   │
lookup_tables.json
                 │
amped_backups/config.json ──► Inputs() ──► amped_exec.main()
                                        │
                                        ├── PerformanceModel ──► breakdown, perf_model
                                        ├── 写 config_summary.txt, AmpedTraining.txt, training_time_breakdown.txt
                                        ▼
                              mat_dims_ampedToDF(amped) ──► dims, mat_dims_amped.txt
                                        │
deepflow_configs/v100.yaml ──► config.parse_config
                                        │
                                        ▼
                              deepflow_exec(amped, dims) ──► TimeCalculation.getCf / getDistGEMM_f_*
                                        │
                                        ├── summary_deepflow.txt
                                        ▼
                              cal_time(amped, deepflow_outputs)  [仅内存]
                                        │
                                        ▼
                              time_domain(amped, deepflow_outputs) ──► time_series.csv
```

以上即为 amped_deepflow 的 input、output 与从运行到结束的完整流程及重要文件说明。
