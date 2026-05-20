# perf.py (DeepFlow) 总结与重点详解

**文件位置**：`DeepFlow/perf.py`（外部 DeepFlow 仓库；amped_deepflow 通过 `sys.path.insert(1, "../DeepFlow")` 后 `from perf import TimeCalculation` 使用）

---

## 一、perf.py 在做什么（总结）

**perf.py** 实现 DeepFlow 的**单次 GEMM 与分布式 reduction/allgather 的解析型性能模型**。在给定**实验配置**（来自 `config.parse_config(v100.yaml)`，包含模型、核心、内存层次、网络、调度等）下，对**任意 (M, N, K)** 的矩阵乘：

1. **单卡 GEMM 时间**：通过 roofline 模型（算力与多级内存带宽/延迟）和多种 dataflow、tile 组合，取最优时间。
2. **分布式情形**：按 **CR（Column-Row）** 或 **RC（Row-Column）** 切分与 **kp1、kp2** 并行度，先算**局部 GEMM 时间**，再算 **reduction 或 all-gather** 的通信时间（Ring 拓扑、带宽/延迟）。

在 amped_deepflow 里，**training.py** 的 **deepflow_exec** 对 **mat_dims_ampedToDF** 给出的每个 (M, N, K, t, kp1, kp2) 调用：

- **getCf(m, k, n)**：无并行，只返回 GEMM 时间（reduction 视为 0）。
- **getDistGEMM_f_kp1(m, k, n, kp1, "Cf_CR")**：CR 切分，返回 [GEMM_time, reduction_time]。
- **getDistGEMM_f_kp2(m, k, n, kp1, kp2, "Cf_RC")**：RC 切分，返回 [GEMM_time, reduction_time]。

这些返回值被汇总成 **deepflow_outputs**，并写入 **summary_deepflow.txt**，供 **cal_time** 和 **time_domain** 使用。

---

## 二、最重要的几个部分与运作方式

下面按「初始化 → 参数覆盖 → GEMM 时间 → Reduction/AllGather 时间 → 对外接口」顺序说明。

---

### 1. 初始化：`__init__(self, exp_config)`

**作用**：从 **exp_config**（FullConfig，由 `config.parse_config(v100.yaml)` 得到）加载**模型、软件、硬件、系统、调度**等，为后续 GEMM 与通信计时提供参数。

**运作**：

- **Model(exp_config)**：batch_size(B)、vocab_size(V)、num_layers(L)、hidden_dim(D)、seq_len(S)、num_gates(G) 等，用于 LSTM/Transformer 相关路径；在 amped_deepflow 的「只算 GEMM」路径下，这些可被 **updateParams** 覆盖为调用方传入的 (m,n,k) 对应维度。
- **Core(exp_config)**：**self.th**（算力，getThroughput）、**FMA_width**、**dataflow**（wst/ast/ost/best），用于 roofline 和 GEMM 的 dataflow 选择。
- **MemoryHierarchy(exp_config)**：**memLayer**（多级内存的带宽、延迟、容量）、**num_levels**；**tileSpace** = **generateTileSpace()**（每级内存的 tile 维度组合），用于 GEMM 的 tile 枚举。
- **Network(exp_config)**：**intra_throughput/latency**、**inter_throughput/latency**；再根据 **inter_derate、intra_derate、par2cross** 得到 derated 带宽，并赋给 **IBK1/LLK1、IBK2/LLK2、IBD/LLD、IBL/LLL**（K 维并行、K 维第二方向、DP、LP 用的带宽和延迟）。**par2cross** 决定各并行维度走 inter 还是 intra。
- **Parallelism(exp_config)**：**findParallelStrategy()** 得到 **dp、lp、kp_hidden_dim1/2、kp_*_type**（1=CR，2=RC）等，用于默认的分布式 GEMM；amped_deepflow 调用时多用 **updateParams** 覆盖为当前 (m,n,k,t,kp1,kp2)。
- **miniB** = ceil(B / dp)。
- **validating_GEMM** = True：表示「只验证 GEMM 时间」，getCf 返回 getGEMMTime 的 (time, order, tile)，getDistGEMM_f_* 返回 [GEMM_time, reduction_time]。

**衔接**：所有 **getGEMMTime、getR** 都依赖 **self.th、memLayer、IBK1/LLK1 等**；**getGEMMTime** 还依赖 **tileSpace、dataflow**。

---

### 2. 参数覆盖：`updateParams(self, debug, m, n, k, t, kp1, kp2, dp, lp, gemm, batch_size, hidden_dim, seq_len, vocab_size, num_layer)`

**作用**：用调用方传入的 (m, n, k) 和并行策略 (t, kp1, kp2, dp, lp) 覆盖 **B, D, S, V, L** 以及 **kp_hidden_dim1/2、kp_*_type**，使后续 **getCf / getDistGEMM_f_*** 按「当前层」的维度和并行度算时间。  
在 amped_deepflow 里，**deepflow_exec** 对每个 dim 调用 **deepflow_function** 时**没有**传 **args_input=True**，因此**不会**调用 **updateParams**，而是直接把 (m, n, k) 传给 **getCf / getDistGEMM_f_kp1 / getDistGEMM_f_kp2**；此时 **getGEMMTime(m, k, n, name)** 或 **getGEMMTime(m, k//dim1, n, name)** 等用的是**传入的维度**，模型里的 B、D、S 等仅影响默认 LSTM 路径，对「只算 GEMM」的调用无影响。

**运作**（若被调用）：

- 设置 **B=batch_size, D=hidden_dim, S=seq_len, V=vocab_size, L=num_layer**，以及 **dp, kp_hidden_dim1, kp_hidden_dim2**，**kp_*_type** 根据 **t** 设为 1(CR) 或 2(RC)。
- 若 **validating_v100** 为 True，会对 **IBK1、IBK2、IBD、IBL** 做 **util.scale_down(..., "kp1"/"kp2"/"dp"/"lp")**，模拟 V100 内链路不对称。

**衔接**：amped_deepflow 当前逻辑下不依赖 updateParams，直接靠 **getCf(m,k,n)** 和 **getDistGEMM_f_kp1/kp2(m,k,n,kp1[,kp2],name)** 的**参数 (m,n,k,kp1,kp2)** 决定单次 GEMM 与 reduction 的维度。

---

### 3. GEMM 时间：`getGEMMTime(dim1, dim2, dim3, name)` 与 roofline

**作用**：对维度 **(dim1, dim2, dim3)**（对应 GEMM 的 M×K、K×N 或分片后的维度），在**多种 (order, tile)** 组合下算 GEMM 时间，取**最小**；单次时间由 **roofline(flop, mem_access)** 加 kernel launch overhead 得到。

**运作**：

- **generateOrder(dim1, dim2, dim3, name)**：根据 **dataflow**（wst/ast/ost/best）生成若干 **(dim2, dim3, dim1)** 等排列，表示矩阵维度的遍历顺序。
- 对每个 **order_dims** 和 **tileSpace** 中的 **tile_dims**：
  - **GEMM(order_dims, tile_dims, name)**：根据 order 和 tile 算 **GEMM_flop** = dim1×dim3×(2×dim2-1)，以及**各级内存的访问量 num_accesses**（由 **getNumAccesses** 按 tile 和 dataflow 的 reuse 算）。
  - **roofline(GEMM_flop, mem_access, name)**：对每一级内存，**inflection_point = th / mem_bw**，**comp_int = flop / num_mem**；若 **comp_int < inflection_point** 取 **mem-bound**：**time = num_mem/mem_bw + mem_latency**，否则 **compute-bound**：**time = flop/th**；最终 **max(time)** 作为该级时间，再对各级取 **max** 作为该 (order, tile) 的时间。
  - **GEMM_time = roofline(...) + self.O**（O 为 kernel launch overhead）。
- **best_tile = min(tile2time, key=tile2time.get)**，返回 **(best_time, best_order, best_tile)**。  
  **getCf** 在 **validating_GEMM** 时直接返回该三元组，调用方取 **[0]** 作为 GEMM 时间。

**衔接**：**getCf**、**getDistGEMM_f_kp1**、**getDistGEMM_f_kp2** 内部都调用 **getGEMMTime**，只是传入的维度不同（无并行：m,k,n；kp1：m, k//dim1, n；kp2：m//dim1, k, n//dim2）。

---

### 4. Reduction / AllGather 时间：`getR(Dim0, Dim1, p, ib, ll, partial, allReduce, name)`

**作用**：在 **Ring** 拓扑假设下，估算 **partial/full reduction** 或 **all-gather** 的通信与本地准备时间；**p** 为参与节点数，**ib** 为带宽，**ll** 为延迟。

**参数含义**：

- **Dim0, Dim1**：张量两维，数据量 **precision × Dim0 × Dim1**（bit）。
- **p**：并行度（如 kp1 或 kp2）。
- **ib, ll**：带宽（bit/s）与延迟（s）；来自 **IBK1/LLK1** 或 **IBK2/LLK2**（由 par2cross 和 derate 决定）。
- **partial**：True 表示每节点只保留 1/p 的归约结果（partial reduction）；False 表示全量。
- **allReduce**：True 表示 all-reduce（先 reduce-scatter 再 all-gather）；False 表示仅 all-gather。

**运作**：

- 若 **p == 1**，返回 0。
- **threshold** 当前为 0，故走 **else** 分支：
  - **factor** = 1（partial 或非 allReduce）或 2（full allReduce，两阶段）。
  - **mem_access**：roofline(0, 2×precision×Dim0×Dim1/p) 模拟「收发前后」的内存访问时间。
  - **data_transfer** = ( (precision×Dim0×Dim1/p)/ib + mem_access + ll ) × factor × (p-1)，即 Ring 上 (p-1) 步的传输与延迟。
  - **data_prep**：每步本地归约的 flop 与内存访问，**roofline(data_prep_comp, data_prep_mem) + O**，共 (p-1) 步；仅当 **allReduce** 时计入。
  - **concat_time**：all-gather 结束后的 concat 内存访问；仅当 **not allReduce** 时计入。
- 返回 **data_transfer + (data_prep if allReduce else 0) + (concat_time if not allReduce else 0)**。

**衔接**：**getDistGEMM_f_kp1** 用 **getR(..., partial=True, allReduce=True)** 得到 reduction 时间；**getDistGEMM_f_kp2** 用 **getR(..., partial=False, allReduce=False)** 得到 all-gather 时间。

---

### 5. 对外接口：`getCf(m, n, k)`、`getDistGEMM_f_kp1`、`getDistGEMM_f_kp2`

**5.1 getCf(m, n, k)**

- **GEMM_time = self.getGEMMTime(m, k, n, "Cf")**：即 (M, K, N)，对应 C = A×B，A(m,k)，B(k,n)。
- 若 **validating_GEMM**：直接 **return GEMM_time**（三元组）；amped_deepflow 侧取 **t_gemm_time[0]**，并构造 **[gemm_time, 0]**（无 reduction）。
- 否则还会加 pointwise 时间（bias + non-linear）后返回标量。

**5.2 getDistGEMM_f_kp1(m, k, n, dim1, name)**（CR：沿 K 列切分）

- **GEMM_time = self.getGEMMTime(m, k//dim1, n, name)**：每个副本算的是 (m, k/dim1, n) 的局部 GEMM。
- **reduction_time = self.getR(Dim0=m, Dim1=n, p=dim1, ib=IBK1, ll=LLK1, partial=True, allReduce=True, name=name)**：在 dim1 个副本间做 partial all-reduce，得到完整 (m, n) 结果。
- 返回 **[GEMM_time[0], reduction_time]**。

**5.3 getDistGEMM_f_kp2(m, k, n, dim1, dim2, name)**（RC：沿 M 行、N 列切分）

- **GEMM_time = self.getGEMMTime(m//dim1, k, n//dim2, name)**：每个副本算 (m/dim1, k, n/dim2)。
- **reduction_time = self.getR(Dim0=m//dim1, Dim1=n, p=dim2, ib=IBK2, ll=LLK2, partial=False, allReduce=False, name=name)**：dim2 路 all-gather，拼出 (m/dim1, n)。
- 返回 **[GEMM_time[0], reduction_time]**。

**衔接**：training.py 的 **deepflow_function** 根据 **t**（CR/RC）和 **kp1、kp2** 调用上述三者之一，得到 **[gemm_time, reduction_time]**，并写入 **deepflow_outputs** 和 **summary_deepflow.txt**。

---

## 三、数据流与调用关系简图

```
exp_config (v100.yaml)
    │
    ├── Model, Core, MemoryHierarchy, Network, Parallelism
    │         │
    │         └── th, memLayer, IBK1/LLK1, IBK2/LLK2, tileSpace, dataflow
    │
    ├── getGEMMTime(dim1, dim2, dim3, name)
    │         │
    │         ├── generateOrder → GEMM(order, tile) → flop, mem_access
    │         └── roofline(flop, mem_access) + O → best_time
    │
    ├── getR(Dim0, Dim1, p, ib, ll, partial, allReduce)
    │         └── data_transfer + data_prep/concat (Ring)
    │
    ├── getCf(m, n, k)                    → (GEMM_time,) 或 [gemm, 0]
    ├── getDistGEMM_f_kp1(m, k, n, dim1, name) → [GEMM_time[0], reduction_time]
    └── getDistGEMM_f_kp2(m, k, n, dim1, dim2, name) → [GEMM_time[0], reduction_time]
```

- **单卡 GEMM**：维度 → **getGEMMTime** → 多 (order, tile) 下 **GEMM + roofline** → 取 min。
- **分布式**：局部 GEMM 维度（k//dim1 或 m//dim1, n//dim2）→ **getGEMMTime**；reduction/allgather 维度与 p、ib、ll → **getR**。
- amped_deepflow 只使用 **getCf**、**getDistGEMM_f_kp1**、**getDistGEMM_f_kp2** 的返回值，不直接依赖 LSTM/embedding 等其它接口。

---

## 四、与 amped_deepflow 的衔接

- **deepflow_exec.deepflow_function()**：对每个 (m, n, k, t, kp1, kp2) 调用 **TimeCalculation(exp_config)**，再根据 **t**：
  - **kp1==1 and kp2==1**：**TC.getCf(m, k, n)** → **[t_gemm_time[0], 0]**。
  - **t=="CR"**：**TC.getDistGEMM_f_kp1(m, k, n, kp1, "Cf_CR")** → **[gemm_time, reduction_time]**。
  - **t=="RC"**：**TC.getDistGEMM_f_kp2(m, k, n, kp1, kp2, "Cf_RC")** → **[gemm_time, reduction_time]**。
- 得到的 **[gemm_time, reduction_time]** 被 append 到 **deepflow_outputs**，并写入 **summary_deepflow.txt**；**cal_time** 用其累加并和 AMPeD breakdown 合并；**time_domain** 用其 MHA/FFN 汇总做时间线。

以上即为 **DeepFlow/perf.py** 的总结以及其中最重要部分的运作方式与衔接关系。
