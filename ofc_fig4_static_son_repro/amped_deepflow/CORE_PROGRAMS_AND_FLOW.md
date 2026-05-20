# amped_deepflow 核心程序与前后衔接

## 一、最重要的程序（按主流程顺序）

| 序号 | 程序/模块 | 位置 | 一句话作用 |
|------|-----------|------|------------|
| 1 | **training.py** | 项目根目录 | 主入口：读配置、串起 AMPeD → 维度提取 → DeepFlow → 后处理 → 时间线，并写 output_files |
| 2 | **amped_backups/config.json** | amped_backups/ | 主配置：模型/并行/系统/加速器参数（很多项由 inputs 计算得出） |
| 3 | **amped_backups/inputs.py** | amped_backups/ | 读 config + lookup 表，按依赖计算所有参数，提供 Inputs / Parameters |
| 4 | **AMPeD/amped/performance_model.py** | 外部 AMPeD 仓库 | 解析型性能模型：根据 parameters 算前向/反向/通信/流水线气泡等时间 |
| 5 | **mat_dims_ampedToDF**（类在 training.py 内） | training.py | 从 AMPeD 的 parameters 推出每层 GEMM 的 M,N,K 与并行类型，供 DeepFlow 用 |
| 6 | **DeepFlow/perf.py** | 外部 DeepFlow 仓库 | 按 M,N,K 和并行策略算单次 GEMM 与 reduction 时间（getCf / getDistGEMM_f_*） |
| 7 | **DeepFlow/config.py** + **deepflow_configs/v100.yaml** | DeepFlow + 本地 | 解析硬件/模型 YAML，供 perf.TimeCalculation 使用 |
| 8 | **cal_time**（类在 training.py 内） | training.py | 把 DeepFlow 的 GEMM/reduction 时间与 AMPeD 的 breakdown 结合（内存中） |
| 9 | **time_domain**（类在 training.py 内） | training.py | 按 batch/层模拟前向→反向→PP/DP→weight_update，生成并写出 time_series.csv |

**说明**：  
- 「程序」包含：入口脚本、配置、本地/外部 Python 模块、以及 training.py 里定义的核心类。  
- 真正“可执行入口”只有 **training.py**；其余要么被它 import 和调用，要么是它读写的配置/数据。

---

## 二、每个程序分别在做什么

### 1. training.py（主入口与编排）

- **入口**：`if __name__ == "__main__"`  
  - 解析 `--config`，对每个配置名（如 `DP_1_16_TP_8_1_PP_1_8_8`）执行一整条流水线。
- **update_configs()**  
  - 根据配置名改写 JSON 的 DP/TP/PP、每节点 GPU 数等，并同步到 `amped_backups/config.json`。
- **amped_exec**  
  - 用 Inputs 读配置 → PerformanceModel 算时间 → 写 config_summary.txt、AmpedTraining.txt、training_time_breakdown.txt；得到 `AMPeD`（含 inputs、breakdown、perf_model）。
- **mat_dims_ampedToDF**  
  - 输入：`AMPeD`；输出：每层 GEMM 的 `dims` 和 mat_dims_amped.txt。
- **deepflow_exec**  
  - 输入：`AMPeD`、`dims`；对每个 (M,N,K) 调 DeepFlow 的 TimeCalculation；输出：deepflow_outputs、summary_deepflow.txt。
- **cal_time**  
  - 输入：AMPeD、deepflow_outputs；在内存中合并 GEMM 时间与 AMPeD breakdown（不写文件）。
- **time_domain**  
  - 输入：AMPeD、deepflow_outputs；按层/ batch 模拟各阶段，写 time_series.csv。

### 2. amped_backups/config.json

- 定义：模型/训练参数（batch、context、layers、精度等）、映射参数（DP/TP/PP 的 intra/inter）、系统（带宽、延迟、每节点卡数）、加速器（或来自 lookup）。
- 很多字段标 `"calculated": true`，由 **inputs.py** 的公式算出来；`update_configs()` 会改写其中并行与部分系统参数。

### 3. amped_backups/inputs.py

- **Inputs**：读 config.json（及 amped_backups 的 lookup_tables.json），按依赖顺序计算 `calculate_functions` 里所有项，得到 **parameters**（字典式访问）。
- **Parameters**：对 parameters 的封装，供后续所有模块使用。
- **CalculateFunctionsDependencyMapping**：决定参数计算顺序。
- 输出：**Inputs 实例**，其 `.parameters` 被 AMPeD 的 PerformanceModel 和 training.py 里所有步骤使用。

### 4. AMPeD/amped/performance_model.py（外部）

- **PerformanceModel(inputs)**：只依赖 `inputs.parameters`（`self.p`）。
- 做三件事：  
  - **计算时间**：前向/反向/权重更新（MAC、non-linear、精度、FMA 宽度）。  
  - **通信时间**：TP/PP/DP 的 intra/inter（Ring/OCS、α–β 模型）。  
  - **流水线气泡**、**总训练时间**、**总 MACs/TFLOPS** 等汇总。
- 提供 per-layer 的 `compute_time_forward_pass()`、`communication_time_forward_pass()`、`backward_*`、`weight_update_time()` 等，以及 `total_*` 系列（总时间、总通信等）。
- 在 amped_deepflow 里：**只被 amped_exec 和 time_domain 调用**，不直接读文件；输入来自 amped_backups 的 Inputs。

### 5. mat_dims_ampedToDF（training.py 内类）

- 输入：**AMPeD**（主要用 `AMPeD.inputs.parameters`：B, D, S, h, nheads, h_MLP1, h_MLP2, N_DP, N_PP）。
- 逻辑：按 Transformer 每层 6 个 GEMM（X.W=KQV, Q.K=R, R.V=Z, Z.W=Y, Y.WL1=O1, O1.WL2=O2）算 M,N,K 和 CR/RC，考虑 DP/PP 切分。
- 输出：**dims**（列表，每项 [M, N, K, "CR"/"RC", layer_name]）、以及 **mat_dims_amped.txt**。
- **dims** 直接传给 **deepflow_exec**。

### 6. DeepFlow/perf.py（外部）

- **TimeCalculation(exp_config)**：exp_config 来自 **config.parse_config(v100.yaml)**，包含模型、核心、内存层次、网络等。
- 对每个 (M, N, K) 与并行类型（CR/RC）、kp1/kp2：  
  - 无并行：**getCf(m,k,n)**；  
  - CR：**getDistGEMM_f_kp1(..., "Cf_CR")**；  
  - RC：**getDistGEMM_f_kp2(..., "Cf_RC")**。  
  返回 **[gemm_time, reduction_time]**。
- 在 amped_deepflow 里：被 **deepflow_exec.deepflow_function()** 在循环里调用，汇总成 **deepflow_outputs**，并写 **summary_deepflow.txt**。

### 7. DeepFlow/config.py + deepflow_configs/v100.yaml

- **config.parse_config(filename)**：读 YAML，转成 FullConfig（model_param、sw_param、tech_param、memory_hierarchy、network_topology 等）。
- **v100.yaml**：写死路径在 training.py 的 DEEPFLOW_CONFIG_PATH 下，描述 V100 的核、内存、网络等，供 **perf.TimeCalculation** 做 GEMM/reduction 时间估算。
- 与 amped 的 config.json 是两套配置：amped 管“训练/并行/系统”，DeepFlow 管“单卡/单次 GEMM 的硬件与调度”。

### 8. cal_time（training.py 内类）

- 输入：**AMPeD**、**DeepFlow.deepflow_outputs**。
- 做：用 **time_from_GEMM()** 把 deepflow_outputs 里所有 GEMM/reduction 时间累加，再按层数和 batch 与 AMPeD 的 **breakdown**（通信、weight update、pipeline bubble）组合，得到更细的总时间观。
- 输出：仅内存（如 `return_outputs`），**不写文件**；为后续 time_domain 提供时间数据基础（time_domain 仍主要用 AMPeD 的 perf_model + deepflow 的 MHA/FFN 时间）。

### 9. time_domain（training.py 内类）

- 输入：**AMPeD**、**DeepFlow.deepflow_outputs**。
- 做：从 AMPeD 取 inputs、perf_model，从 deepflow_outputs 取 MHA/FFN 时间；按 **NR_BATCHS_TO_PROCESS** 个 batch、每层依次执行 **forward_pass → pp_overhead_FWD → backward_pass → pp_overhead_BWD → dp_overhead → zeroDP → weight_update**，把每步的（Layer, Type, start/end, duration, Bytes, Collective type, Parallelism, Locality, Degree）压入 **timeline**。
- 输出：**total_timeline()** 把 timeline 写入 **{OUTPUT_PATH}/{timeStamp}time_series.csv**。
- 可选：**individual_timeline()** 可写 per-GPU 的 CSV（主流程中已注释）。

---

## 三、从头到尾如何前后连接（数据流）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. 入口 (training.py __main__)                                               │
│    --config → update_configs(cSim) → 写 amped_backups/config.json            │
│    OUTPUT_PATH = output_files/{RESULT_DIR}                                   │
└───────────────────────────────────────────┬─────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. AMPeD 阶段                                                                │
│    amped_backups/config.json + lookup_tables                                  │
│         → Inputs() [amped_backups/inputs.py]                                 │
│         → parameters                                                        │
│         → PerformanceModel(inputs) [AMPeD/amped/performance_model.py]        │
│         → amped_exec: breakdown, perf_model, inputs                          │
│    写出: config_summary.txt, AmpedTraining.txt, training_time_breakdown.txt  │
└───────────────────────────────────────────┬─────────────────────────────────┘
                                             │ AMPeD (inputs, breakdown, perf_model)
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. 维度提取                                                                  │
│    mat_dims_ampedToDF(AMPeD, training=True)                                  │
│    用 AMPeD.inputs.parameters → 每层 GEMM 的 M,N,K, CR/RC, layer_name       │
│    输出: dims, mat_dims_amped.txt                                           │
└───────────────────────────────────────────┬─────────────────────────────────┘
                                             │ dims
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. DeepFlow 阶段                                                             │
│    deepflow_configs/v100.yaml → config.parse_config [DeepFlow/config.py]     │
│    deepflow_exec(AMPeD, dims):                                               │
│      对 dims 中每个 (M,N,K,t,kp1,kp2) 调用                                   │
│        TimeCalculation(exp_config) [DeepFlow/perf.py]                        │
│        → getCf / getDistGEMM_f_kp1 / getDistGEMM_f_kp2                       │
│    输出: deepflow_outputs, summary_deepflow.txt                              │
└───────────────────────────────────────────┬─────────────────────────────────┘
                                             │ AMPeD, deepflow_outputs
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. 后处理 (仅内存)                                                           │
│    cal_time(AMPeD, deepflow_outputs)                                         │
│    合并 GEMM/reduction 与 AMPeD breakdown → 更细总时间                        │
└───────────────────────────────────────────┬─────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. 时间线                                                                    │
│    time_domain(AMPeD, deepflow_outputs)                                      │
│    用 AMPeD.perf_model + deepflow MHA/FFN 时间                               │
│    按 batch/层 模拟 forward/backward/PP/DP/weight_update → timeline          │
│    输出: {timeStamp}time_series.csv                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

**衔接要点**：

- **config.json** 驱动 **Inputs**，**Inputs.parameters** 驱动 **PerformanceModel** 和 **mat_dims_ampedToDF**。
- **AMPeD** 作为“中间结果”贯穿：先给 **mat_dims_ampedToDF**，再和 **dims** 一起给 **deepflow_exec**，最后和 **deepflow_outputs** 一起给 **cal_time** 和 **time_domain**。
- **dims** 是 AMPeD → DeepFlow 的桥梁；**deepflow_outputs** 是 DeepFlow → cal_time / time_domain 的桥梁。
- 所有写文件都进 **output_files/{RESULT_DIR}/**，由 training.py 在每轮循环开始时设置的 **OUTPUT_PATH** 决定。

---

## 四、总结表（程序 ↔ 输入/输出）

| 程序/模块 | 主要输入 | 主要输出 |
|-----------|----------|----------|
| update_configs | 配置名 cSim、--config 文件 | 更新后的 amped_backups/config.json |
| Inputs (inputs.py) | config.json, lookup_tables.json | parameters（供全局使用） |
| PerformanceModel (AMPeD) | inputs（.parameters） | 各类 compute/comm/total_* 时间 |
| amped_exec | Inputs → PerformanceModel | AMPeD 对象；config_summary, AmpedTraining, training_time_breakdown |
| mat_dims_ampedToDF | AMPeD | dims, mat_dims_amped.txt |
| deepflow_exec | AMPeD, dims；v100.yaml | deepflow_outputs, summary_deepflow.txt |
| cal_time | AMPeD, deepflow_outputs | 内存中的合并时间（无文件） |
| time_domain | AMPeD, deepflow_outputs | time_series.csv |

上面就是 amped_deepflow 里最重要的程序、各自在做什么、以及从头到尾如何前后连接。
