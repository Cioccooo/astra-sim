# performance_model.py 总结与重点详解

**文件位置**：`AMPeD/amped/performance_model.py`（外部 AMPeD 仓库，amped_deepflow 通过 `from amped.performance_model import PerformanceModel` 使用）

---

## 一、performance_model.py 在做什么（总结）

**performance_model.py** 实现 AMPeD 的**解析型性能模型**：在给定 **Inputs**（其 `parameters` 来自 config + inputs 计算）下，**不跑真实训练**，只用公式估算**分布式 Transformer 训练**的：

1. **计算时间**：前向、反向、权重更新（按 MAC、non-linear ops、精度、FMA 宽度等）；
2. **通信时间**：张量并行（TP）节点内/节点间、流水线并行（PP）节点内/节点间、数据并行（DP）梯度 all-reduce 节点内/节点间、以及 MoE 相关通信；
3. **流水线气泡等待时间**；
4. **总训练时间**（秒/天/年）以及 **总 MACs、TFLOPS、TFLOPS/GPU** 等汇总指标。

所有公式都只依赖 **`self.p`**（即 `inputs.parameters`）；通信部分采用 **α–β 模型**（延迟 α + 数据量/带宽 β），并支持 **电/固定光（Ring）** 与 **OCS（Raben­seifner 等）** 两种算法分支。  
在 amped_deepflow 里，该类由 **amped_exec** 和 **time_domain** 调用，用于生成 training_time_breakdown 和时间线。

---

## 二、最重要的几个部分与运作方式

下面按「初始化 → 基础吞吐 → 计算时间 → 通信时间 → 流水线气泡 → 总时间与汇总」顺序说明。

---

### 1. 初始化与参数来源：`__init__(self, inputs: Inputs)`

**作用**：只保存「参数视图」，并设少量可覆盖的常数。

**运作**：

- **`self.p = inputs.parameters`**：之后所有方法都通过 `self.p["..."]` 读配置（batch、layers、DP/TP/PP、带宽、延迟、MACs、精度等）。不持有 `inputs` 引用也可，只要 `parameters` 是字典式访问即可（amped_deepflow 里传入的是 amped_backups 的 Inputs，其 `.parameters` 满足这一点）。
- **`self.W_FU_MAC = value(USER, 32)`**：MAC 功能单元位宽，用于把「MAC 数 × 精度」转成「等效周期/时间」；默认 32，可用 `value(USER, ...)` 覆盖。
- **`self.W_FU_NONLIN = value(USER, 32)`**：非线性 op 的位宽。
- **`self.M_f_DP = value(USER, 0.25)`**：Zero-DP 等权重量通信的额外系数，前向/反向通信时间会乘 **(1 + M_f_DP)**。

**衔接**：后续所有 **per-layer / per-batch** 的时间都只依赖 `self.p` 和这三个属性。

---

### 2. 辅助函数（通信 α–β 与带宽）

用于把「跨节点/节点内」的带宽和延迟统一成可插拔的 **α（步数×延迟）** 和 **β（数据量/带宽）**，并支持 Ring 与 OCS 两种路径。

- **`_param(k, default)`**：安全取 `self.p[k]`，缺键或值为 None 时返回 default。
- **`_bw_elec()`**：电/固定光路径的**有效带宽** = `inter_node_bandwidth / electrical_oversub_factor`（默认 oversub=2）。
- **`_bw_ocs()`**：OCS 路径的有效带宽 = `inter_node_bandwidth * ocs_parallel_circuits_per_node * ocs_wdm_lanes`（无 oversub 折减）。
- **`_alpha_ring(steps)`**：Ring 的 **α** = `steps * inter_node_latency`（每步固定延迟）。
- **`_alpha_ocs(steps)`**：OCS 的 **α** = `steps * (inter_node_latency_OCS + ocs_reconf_per_round / ocs_slot_microbatches)`，把重构开销摊到每步。

**衔接**：TP/PP/DP 的 **inter** 通信时间都按「α + β」计算，并根据 `tp_inter_algo`、`dp_inter_algo`、`pp_inter_use_ocs` 等在 Ring 与 OCS 之间选择。

---

### 3. 基础吞吐：`reciprocal_of_OPS()` 与 `C_NONLIN()`

**作用**：把「算力与效率」转成「每 OP 耗时」和「每个非线性 op 耗时」，供所有计算时间公式使用。

- **`reciprocal_of_OPS()`**（C_MAC，单位 s/OP）：  
  **1 / (frequency × number_of_cores × functional_units_per_core × efficiency × functional_unit_hardware_8bit_MAC_per_cycle)**  
  当前代码里 **efficiency** 用 `value(USER, 0.51)` 固定；注释里也可用 **`efficiency_given_ubatch_size()`**（根据 microbatch/minibatch 拟合的曲线，上限 0.91）。
- **`C_NONLIN()`**（s/nonlin）：  
  **1 / (frequency × non_linear_functional_units_per_core × non_linear_functional_unit_hardware_8bit_NLIN_per_cycle)**  
  即每个非线性 op 的耗时。

**衔接**：前向/反向/权重更新的计算时间 = **(MAC 数 × 精度相关系数 × reciprocal_of_OPS) + (non_linear op 数 × 精度相关系数 × C_NONLIN)**。

---

### 4. 计算时间（前向 / 反向 / 权重更新）

**4.1 `compute_time_forward_pass()`（U_f，单位 s）**

- **MAC 部分**：  
  **reciprocal_of_OPS() × (total_attention_sublayer_MAC_operations + total_MLP_sublayer_MAC_operations + gating_MAC_operations/2 + decoder_NMACs) × ceil(weight_precision / W_FU_MAC)**
- **Non-linear 部分**：  
  **C_NONLIN() × (non_linear_operations_for_attention_sublayer + non_linear_operations_for_MLP_sublayer) × ceil(activation_precision / W_FU_NONLIN)**
- 含义：**单层、单 batch** 的前向计算时间（未除 DP/TP/PP）；training.py 里会再除以 DP×TP×PP 得到 per-layer 的「每副本」时间。

**4.2 `compute_time_backward_pass()`（U_b）**

- 结构相同：MAC 用 **(total_attention + total_MLP + gating/2)**，精度取 **max(weight_precision, gradient_precision)**；non-linear 用 **weight_precision**。
- 表示单层、单 batch 的反向计算时间。

**4.3 `weight_update_time()`（U_w）**

- **reciprocal_of_OPS() × (total_attention + total_MLP + gating_MAC_operations)**，无 non-linear 项。
- 单层、单 batch 的权重更新计算时间。

**衔接**：三者都被 **time_spent_in_multi_GPU()** 和 **total_computation_time_*** 使用；**total_*** 再乘 **layers × number_of_batches** 并除以 **DP×TP×PP** 得到「总计算时间」类输出。

---

### 5. 通信时间（前向）

**5.1 总前向通信：`communication_time_forward_pass()`（M_f）**

- ** (1 + M_f_DP) × (forward_tensor_model_intra() + forward_tensor_parallel_inter() + 2×MoE_overhead_per_layer_fw_pass() + max(forward_pipeline_parallel_intra(), forward_pipeline_parallel_inter()))**
- 即：**单层** 的 TP（节点内+节点间）+ MoE + PP（节点内/间取大）前向通信，再乘 Zero-DP 系数。

**5.2 `forward_tensor_model_intra()`（M_f_TP_intra）**

- **n_topo_intra** = 2×(TP_intra - 1)/TP_intra（Ring 的步数系数）。  
- **n_act_intra**：若 TP_intra>1，为 2×activations_volume_per_layer_batch / number_of_nodes_required；否则 0。  
- 返回：**inter_accelerator_latency × n_topo_intra × TP_intra + n_act_intra × activation_precision × n_topo_intra / inter_accelerator_bandwidth**（α + β，节点内）。

**5.3 `forward_tensor_parallel_inter()`**

- 若 **inter_node_tensor_parallel_degree ≤ 1** 返回 0。  
- 否则 **n_topo = 2×(p-1)/p**，**n_act = 2×activations_volume_per_layer_batch**（整层激活参与跨节点 TP）。  
- 按 **tp_inter_algo**：  
  - **0（Ring）**：β = n_act×activation_precision×n_topo / _bw_elec()，α = _alpha_ring(steps=2×(p-1))。  
  - **1（OCS/Rabenseifner）**：β 用 _bw_ocs()，α = _alpha_ocs(steps=2×ceil(log2(p)))。  
- 返回 **α + β**。

**5.4 `forward_pipeline_parallel_intra()` / `forward_pipeline_parallel_inter()`**

- **Intra**：若 PP_intra==1 为 0；否则 **(latency + activations_volume×activation_precision/bandwidth) / layers**（每层摊到的节点内 PP 通信）。  
- **Inter**：若 PP_inter==1 为 0；否则 **(α + β) / layers**，β = n_bits/_bw_elec() 或 _bw_ocs()，α 用 ring 或 OCS 一步；由 **pp_inter_use_ocs** 选择。

**5.5 MoE**：`MoE_overhead_per_layer_fw_pass()` 按 expert 数和节点数分摊激活通信量，仅当 **expert_flag** 非 0 时非零。

**衔接**：前向总通信被 **communication_time_forward_pass()** 汇总；再通过 **total_forward_*** 乘 **layers × number_of_batches** 得到「总前向通信」各类输出。

---

### 6. 通信时间（反向与 DP all-reduce）

**6.1 总反向通信：`communication_time_backward_pass()`（M_b）**

- **(1 + M_f_DP) × (backward_tensor_model_intra() + backward_tensor_parallel_inter() + 2×MoE_overhead_per_layer_bw_pass() + max(backward_pipeline_parallel_intra(), backward_pipeline_parallel_inter()))**
- 结构与前向对称，只是数据量用 **error_volume_per_layer_batch × gradient_precision**，公式形式与 **forward_*** 一致（α–β，intra/inter，Ring/OCS）。

**6.2 DP 梯度 all-reduce：`communication_time_backwards_DP_all_reduce()`**

- **r_non_overlapping × (communication_time_backwards_DP_all_reduce_intra() + communication_time_backwards_DP_all_reduce_inter())**  
  **r_non_overlapping** 默认 1（可 value(USER,...) 覆盖）。

**6.3 `communication_time_backwards_DP_all_reduce_intra()`**

- **n_topo_intra** = 2×(DP_intra - 1)/DP_intra；  
- **n_gradients_intra** = number_of_parameters_per_layer / tp_total（TP 下每卡只持有一份参数 shard），DP_intra>1 时非零。  
- 返回：**inter_accelerator_latency × n_topo_intra × DP_intra + n_gradients_intra × gradient_precision × n_topo_intra / inter_accelerator_bandwidth**。

**6.4 `communication_time_backwards_DP_all_reduce_inter()`**

- P = inter_node_data_parallel_degree；P≤1 返回 0。  
- **vol** = number_of_parameters_per_layer / tp_total（每卡参与跨节点 all-reduce 的梯度量）；**n_topo** = 2×(P-1)/P。  
- 按 **dp_inter_algo**：  
  - **0（Ring）**：β = vol×gradient_precision×n_topo / _bw_elec()，α = _alpha_ring(steps=2×(P-1))。  
  - **1（Raben­seifner/OCS）**：β 用 _bw_ocs()，α = _alpha_ocs(steps=2×ceil(log2(P)))。  
- 返回 **α + β**。  
- 首次调用时会打印一次 ELEC vs OCS 的 tβ/tα/t 自检信息（可选调试）。

**衔接**：DP all-reduce 与反向 TP/PP 一起，被 **time_spent_in_multi_GPU()** 和 **total_communication_time_backward_pass()**、**total_all_reduce_gradients_for_DP_*** 使用。

---

### 7. 流水线气泡：`waiting_time_due_to_pipeline_bubbles()`

**作用**：估算因流水线并行导致的「空闲等待」时间（每 minibatch 内、per-layer 视角）。

**公式**：  
**r_non_overlapping × (pipeline_parallel_degree - 1) × ( (compute_time_forward_pass + compute_time_backward_pass) / (DP×TP×PP×layers) + communication_time_forward_pass + communication_time_backward_pass ) / number_of_microbatches_per_minibatch**

- **(PP-1)**：流水线 stage 数减 1 带来的气泡轮数。  
- 括号内：每个 stage 的「计算+通信」时间（per-layer 摊到的计算 + 整层的前向/反向通信）。  
- 再除以 **microbatch 数**，得到按 microbatch 摊的等待时间。

**衔接**：被 **time_spent_in_multi_GPU()** 加进总时间；**total_waiting_time_due_to_pipeline_bubbles()** = layers × number_of_batches × 上式。

---

### 8. 总时间与入口：`time_spent_in_multi_GPU()` 与 `total_time_to_train()`

**8.1 `time_spent_in_multi_GPU()`（C_multi_accel，单位 s）**

- **number_of_batches × layers × ( (compute_time_forward_pass + compute_time_backward_pass + weight_update_time) / (DP×TP×PP) + communication_time_forward_pass + communication_time_backwards_DP_all_reduce() + communication_time_backward_pass + waiting_time_due_to_pipeline_bubbles() )**
- 含义：所有 batch、所有层上，**每层摊到的计算时间**（已除 DP×TP×PP）+ **每层前向通信** + **每层 DP all-reduce** + **每层反向通信** + **每层流水线气泡**，再乘 batch×layer 数，得到**总墙上时钟时间**。

**8.2 `total_time_to_train()`**

- **max(time_spent_in_multi_GPU(), value(USER, 0)) + value(USER, 0)**  
  即主输出就是 **time_spent_in_multi_GPU()**，留了两个 value(USER, 0) 方便加偏移或下界。

**衔接**：training.py 的 **calc_time**、**training_string_training_time_breakdown()** 会用到 **total_time_to_train()** 以及下面各种 **total_***；**time_domain** 则用 **reciprocal_of_OPS()**、**C_NONLIN()** 和各 **forward_*/backward_***、**communication_time_*** 等逐阶段拼时间线。

---

### 9. 汇总输出：`total_*` 系列

**作用**：把「per-layer、per-batch」的量乘上 **layers × number_of_batches**，或把已汇总的分项相加，得到 training_time_breakdown 里需要的各项总时间 / 指标。

- **总计算**：`total_computation_time_forward_pass`、`total_computation_time_backward_pass`、`total_computation_time_weight_updates`、`total_computation_time()`（三者之和）。
- **总前向通信**：`total_forward_tensor_model_intra`、`total_forward_tensor_parallel_inter`、`total_forward_pipeline_parallelism`、`total_forward_zero_DP`、`total_MoE_overhead_fw_pass`、`total_communication_time_forward_pass()`（前几项之和）。
- **总反向通信**：`total_backward_*` 与 `total_communication_time_backward_pass()` 同理。
- **DP all-reduce**：`total_all_reduce_gradients_for_DP_intra`、`total_all_reduce_gradients_for_inter`、`total_all_reduce_gradients_for_DP()`。
- **总通信**：`total_communication_time()` = 前向总通信 + 反向总通信 + DP all-reduce 总时间。
- **流水线气泡**：`total_waiting_time_due_to_pipeline_bubbles()`。
- **算力指标**：`total_MACs()`、`total_TFLOPS()`、`total_TFLOPS_per_second()`、`total_TFLOPS_per_second_per_gpu()`、`total_TFLOPS_per_second_per_gpu_peak()`。
- **MoE**：`total_MoE_gating_network_compute_overhead()`、`total_MoE_overhead()`。
- **日历时间**：`total_time_to_train_days()`、`total_time_to_train_years()`。

**衔接**：**training_string_training_time_breakdown()** 里用这些键填字典并生成 **training_time_breakdown.txt**；**string_training_time_breakdown()** 若被调用也会用同一批 total_*。

---

## 三、数据流与调用关系简图

```
inputs.parameters (self.p)
        │
        ├── reciprocal_of_OPS(), C_NONLIN()
        │         │
        │         └── compute_time_forward/backward_pass(), weight_update_time()
        │
        ├── _bw_elec(), _bw_ocs(), _alpha_ring(), _alpha_ocs()
        │         │
        │         └── forward/backward_tensor_model_intra(), forward/backward_tensor_parallel_inter()
        │               forward/backward_pipeline_parallel_intra/inter()
        │               communication_time_backwards_DP_all_reduce_intra/inter()
        │
        ├── communication_time_forward_pass(), communication_time_backward_pass()
        │   communication_time_backwards_DP_all_reduce()
        │
        ├── waiting_time_due_to_pipeline_bubbles()
        │
        └── time_spent_in_multi_GPU() → total_time_to_train()
                    │
                    └── total_*() 系列 → training_time_breakdown / string_training_time_breakdown
```

- **Per-layer 计算/通信** 只依赖 **self.p** 和上述辅助函数。  
- **总时间** = batch×layer × ( 计算/(DP×TP×PP) + 各通信 + 气泡 )。  
- **total_*** 要么是 layers×batches × 某 per-layer 量，要么是若干 total_* 分项之和。

---

## 四、与 amped_deepflow 的衔接

- **amped_exec.calc_time()**：`perf_model = PerformanceModel(inputs)`，用 **compute_time_forward/backward_pass**、**communication_time_forward_pass**、**weight_update_time**、**communication_time_backwards_DP_all_reduce**、**communication_time_backward_pass**、**waiting_time_due_to_pipeline_bubbles** 以及 **pipeline_parallel_degree**、**number_of_microbatches_per_minibatch** 等，算出 per-layer 时间并乘 batch×layer 得到总 compute/comm/bubble；再在 **training_string_training_time_breakdown()** 里调用所有 **total_*** 填 breakdown 字典并写 **training_time_breakdown.txt**。
- **time_domain**：用 **perf_model.reciprocal_of_OPS()**、**C_NONLIN()** 以及各 **forward_***、**backward_***、**communication_time_*** 等，按 batch/层模拟前向、反向、PP、DP、zeroDP、weight_update，往 **timeline** 里追加事件并写 **time_series.csv**。

以上即为 **performance_model.py** 的总结以及其中最重要部分的运作方式与衔接关系。
