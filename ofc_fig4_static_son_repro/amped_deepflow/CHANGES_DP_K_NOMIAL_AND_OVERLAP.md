# 需要修改的代码位置：DP k-nomial 与 通信–计算 Overlap

## 一、功能一：DP 从 Rabenseifner 2⌈log₂ P_DP⌉ 改为 WDM k-nomial 2⌈log_k P_DP⌉  

### 1.1 涉及文件与函数

| 文件 | 函数 | 说明 |
|------|------|------|
| **AMPeD/amped/performance_model.py** | `communication_time_backwards_DP_all_reduce_inter` | 跨节点 DP 反向 all-reduce；当前唯一使用 2⌈log₂ P⌉ 的 DP 路径 |

**说明**：节点内 DP（intra）当前为 ring：`n_topo_intra = 2*(P-1)/P`，未使用 log₂；若将来要对 intra 也做 k-nomial，需在 `communication_time_backwards_DP_all_reduce_intra` 中增加分支。

---

### 1.2 具体行号与当前逻辑（AMPeD/amped/performance_model.py）

- **约 298–356 行**：`communication_time_backwards_DP_all_reduce_inter` 整段。

**当前公式（Rabenseifner / OCS 分支）：**

```python
# 约 334-338 行
steps_ring = int(2 * (P - 1))
steps_rab  = int(2 * math.ceil(math.log2(P)))   # ← 这里：2⌈log₂ P⌉
t_alpha_elec = float(self._alpha_ring(steps=steps_ring))
t_alpha_ocs  = float(self._alpha_ocs(steps=steps_rab))

# 约 348-351 行（algo == 1 时）
if algo == 1:  # OCS / Rabenseifner
    beta  = (vol * ...) / self._bw_ocs()
    alpha = self._alpha_ocs(steps=steps_rab)   # ← 使用 steps_rab
```

**需要改的内容：**

1. 引入 **k**（k-nomial / WDM 的 k），例如从 `self.p` 或 `_param` 读取（如 `dp_k_nomial` 或 `wdm_k`，默认可设为 2 以保持与当前一致）。
2. 将  
   `steps_rab = int(2 * math.ceil(math.log2(P)))`  
   改为  
   `steps_rab = int(2 * math.ceil(math.log(P) / math.log(k)))`  
   即 **2⌈log_k P_DP⌉**（P_DP 即此处的 `P = inter_node_data_parallel_degree`）。
3. 保证 k=2 时与现有行为一致：`log(P)/log(2) = log2(P)`。

**无需改的 DP 相关位置：**

- `communication_time_backwards_DP_all_reduce_intra`（约 276–295 行）：当前为 ring，无 log₂，若不做 intra 的 k-nomial 则不必改。
- 第 175 行、第 431 行：属于 **TP**（tensor parallel）的 `forward_tensor_parallel_inter` / `backward_tensor_parallel_inter`，不是 DP，本次仅改 DP。

---

## 二、功能二：通信时间与计算时间 Overlap（一部分通信隐藏在计算后面）

### 2.1 涉及文件与函数

| 文件 | 函数 | 说明 |
|------|------|------|
| **AMPeD/amped/performance_model.py** | `time_spent_in_multi_GPU` | 总时间公式：目前为 **纯相加**，无 overlap |
| **AMPeD/amped/performance_model.py** | `communication_time_backwards_DP_all_reduce` | 已有 `r_non_overlapping`，可用来表示“未与计算重叠的 DP 通信比例” |

---

### 2.2 总时间当前公式（无 overlap）

**约 59–69 行**：`time_spent_in_multi_GPU`

```python
def time_spent_in_multi_GPU(self):
    return (
        self.p["number_of_batches"] * self.p["layers"]
        * ((self.compute_time_forward_pass() + self.compute_time_backward_pass() + self.weight_update_time())
           / (self.p["data_parallel_degree"] * self.p["tensor_parallel_degree"] * self.p["pipeline_parallel_degree"])
           + self.communication_time_forward_pass() + self.communication_time_backwards_DP_all_reduce()
           + self.communication_time_backward_pass() + self.waiting_time_due_to_pipeline_bubbles())
    )
```

即：**每 layer 时间 = (计算时间 / 并行度) + 前向通信 + DP all-reduce + 反向通信 + pipeline bubble**，全部线性相加，没有任何一部分通信被“藏”在计算后面。

---

### 2.3 需要改的内容（overlap 建模）

**位置**：仍在 `time_spent_in_multi_GPU()`（约 59–69 行）及（可选）其调用的通信项。

**思路（二选一或组合）：**

1. **在总时间公式里做 overlap**  
   - 先算：  
     - `T_comp` = (compute_time_forward_pass + compute_time_backward_pass + weight_update_time) / 并行度  
     - `T_comm` = communication_time_forward_pass + communication_time_backwards_DP_all_reduce() + communication_time_backward_pass  
   - 再对 `T_comm` 做“可被计算隐藏”的建模，例如：  
     - 引入参数 `comm_overlap_fraction` ∈ [0,1]：  
       `T_comm_effective = T_comm * (1 - comm_overlap_fraction)`  
       然后  
       `per_layer = T_comp + T_comm_effective + waiting_time_due_to_pipeline_bubbles()`  
     - 或：可隐藏量 = min(T_comm, T_comp * overlap_capacity)，  
       `T_comm_effective = max(0, T_comm - min(T_comm, T_comp * overlap_capacity))`  
   - 在 `time_spent_in_multi_GPU()` 里用上述 `per_layer` 再乘 `number_of_batches * layers`。

2. **利用现有 `r_non_overlapping`（仅对 DP all-reduce）**  
   - **约 265–274 行**：`communication_time_backwards_DP_all_reduce()` 中有  
     `r_non_overlapping = value(USER, 1)`，  
     当前含义是“DP all-reduce 时间的缩放系数”。  
   - 若把 `r_non_overlapping` 视为“**未**与计算重叠的比例”（1=完全不重叠，0=完全重叠），则当前已经是“有效 DP 通信时间 = r_non_overlapping * (intra + inter)”。  
   - 因此：  
     - 若只对 **DP all-reduce** 做 overlap，只需在配置/输入里把 `r_non_overlapping` 设为 <1（或在代码里用新参数如 `dp_comm_overlap_fraction` 算出 `r_non_overlapping = 1 - dp_comm_overlap_fraction`）即可，**无需改总时间公式结构**。  
     - 若要对 **全部通信**（前向 + DP + 反向）做统一 overlap，则仍需在 **`time_spent_in_multi_GPU()`** 里先汇总各项通信时间，再按上面 1 的方式做一次整体 overlap 后与计算、bubble 相加。

---

### 2.4 相关行号汇总（overlap）

| 位置 | 行号（约） | 内容 |
|------|------------|------|
| 总时间入口 | 59-69 | `time_spent_in_multi_GPU()`：当前 compute + comm 纯相加 |
| DP all-reduce 系数 | 266-274 | `communication_time_backwards_DP_all_reduce()`：`r_non_overlapping = value(USER, 1)` |
| Pipeline bubble 系数 | 517-530 | `waiting_time_due_to_pipeline_bubbles()`：内部也有 `r_non_overlapping`，与 DP 的独立 |

---

## 三、小结表

| 功能 | 文件 | 函数 | 关键行号 | 修改要点 |
|------|------|------|----------|----------|
| **DP 2⌈log₂ P⌉ → 2⌈log_k P⌉** | AMPeD/amped/performance_model.py | `communication_time_backwards_DP_all_reduce_inter` | 334-337, 349-351 | 引入 k，`steps_rab = 2*ceil(log(P)/log(k))` |
| **Comm–comp overlap** | AMPeD/amped/performance_model.py | `time_spent_in_multi_GPU` | 59-69 | 在总时间中把部分通信隐藏在计算后（或利用 `r_non_overlapping` 仅对 DP） |

所有需改代码均在 **AMPeD** 仓库的 **amped/performance_model.py** 中；amped_deepflow 侧仅通过调用该模型得到时间，无需改 DP 公式或 overlap 逻辑。
