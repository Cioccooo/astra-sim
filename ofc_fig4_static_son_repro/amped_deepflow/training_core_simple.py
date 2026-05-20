# training_core_simply.py — 精炼版：仅保留流水线主干（入口 → update_configs → AMPeD → mat_dims → DeepFlow → cal_time → time_domain stub）
# 使用: python training_core_simply.py --config amped_backups/config.json

import os
import sys
sys.path.insert(1, "../AMPeD")
sys.path.insert(1, "../DeepFlow")

import argparse
import json
import config
import perf
from perf import TimeCalculation
from amped.common import time_prefix
from amped.performance_model import PerformanceModel
from amped_backups.inputs import CalculateFunctionsDependencyMapping, Inputs, Parameters

DEBUG_PRINT = False
PROMPT_LEN = 2048
GENERATION_LEN = 0
DEEPFLOW_CONFIG_PATH = "/home/fd420/LLM_analytical_tools/amped_deepflow/deepflow_configs"
RESULT_DIR = "DP_2_1_TP_2_1_PP_1_1"
OUTPUT_PATH = f"/home/fd420/LLM_analytical_tools/amped_deepflow/output_files/{RESULT_DIR}"


# ----- 1) cal_time -----
class cal_time:
    def __init__(self, amped, deepFlow_outputs) -> None:
        self.amped = amped
        self.inputs = self.amped.inputs
        self.breakdown = self.amped.breakdown
        self.deepflow_outputs = deepFlow_outputs
        self.debug = DEBUG_PRINT
        self.main(self.inputs)
        self.return_outputs = []

    def time_from_GEMM(self):
        t_elapsed = sum(g[0] for g in self.deepflow_outputs)
        t_reduction_elapsed = sum(g[1] for g in self.deepflow_outputs)
        return [t_elapsed * 2.0, t_reduction_elapsed * 2.0]  # FW + BW

    def main(self, inputs):
        if self.debug:
            print("**** Computing GEMM timings using DeepFlow *****\n", self.breakdown)
        t_FW_BW, t_REDUCTION = self.time_from_GEMM()
        nbatch = 1
        time = (
            int(inputs.parameters["layers"]) * nbatch * t_FW_BW
            + float(self.breakdown["Total communication time forward pass (s)"])
            + float(self.breakdown["Total communication time backward pass (s)"])
            + float(self.breakdown["Computation time weight updates (s)"])
            + float(self.breakdown["Waiting Time due to pipeline bubbles (s)"])
        )
        if self.debug:
            print("Total time after DeepFlow computation: ", time)


# ----- 2) deepflow_exec -----
class deepflow_exec:
    def __init__(self, amped, dims, training=True) -> None:
        self.debug = DEBUG_PRINT
        self.amped = amped
        self.inputs = self.amped.inputs
        self.TP_DEGREE = self.inputs.parameters["tensor_parallel_degree"]
        self.CONFIG_DIR = DEEPFLOW_CONFIG_PATH
        self.OUTDIR = OUTPUT_PATH
        self.training = training
        self.deepflow_outputs = []
        self.main(dims)

    def deepflow_function(self, exp_config, exp_dir, debug, m, n, k, t, kp1, kp2, gemm, batch_size=2048,
                          hidden_dim=19968, seq_len=20, vocab_size=800000, num_layer=2, dp=None, lp=None, lev=None, args_input=False):
        exp_path = os.path.expandvars(os.path.expanduser(exp_config))
        exp_config = config.parse_config(exp_path)
        TC = TimeCalculation(exp_config)
        if args_input:
            TC.updateParams(debug, m, n, k, t, kp1, kp2, dp, lp, gemm,
                            batch_size, hidden_dim, seq_len, vocab_size, num_layer)
        if TC.validating_GEMM:
            if kp1 == 1 and kp2 == 1:
                t_gemm_time = TC.getCf(m, k, n)
                gemm_time = [t_gemm_time[0], 0]
            elif t == "CR":
                gemm_time = TC.getDistGEMM_f_kp1(m, k, n, kp1, "Cf_CR")
            elif t == "RC":
                gemm_time = TC.getDistGEMM_f_kp2(m, k, n, kp1, kp2, "Cf_RC")
            else:
                print("Incorrect parallelism type, CR: Column-Row, RC: Row-Column")
                sys.exit()
            return [gemm_time[0], gemm_time[1]]

    def main(self, dims):
        mha_GEMMtime = mha_reduction = ffn_GEMMtime = ffn_reduction = 0
        output_file = f"{self.OUTDIR}/{self.amped.timeStamp}summary_deepflow.txt"
        with open(output_file, "w") as f:
            f.write("Layer, M, N, K, t, GEMM time, reduction time\n")
        for i in range(len(dims)):
            if not self.training:
                if dims[i][0] < self.TP_DEGREE:
                    temp_outputs = self.deepflow_function(
                        exp_config=f"{self.CONFIG_DIR}/v100.yaml", exp_dir=f"{self.OUTDIR}/LLM", debug=True,
                        m=dims[i][0], n=dims[i][1], k=dims[i][2], t="RC", kp1=1, kp2=self.TP_DEGREE, gemm=True, lev=dims[i][4],
                    )
                else:
                    temp_outputs = self.deepflow_function(
                        exp_config=f"{self.CONFIG_DIR}/v100.yaml", exp_dir=f"{self.OUTDIR}/LLM", debug=True,
                        m=dims[i][0], n=dims[i][1], k=dims[i][2], t="RC", kp1=self.TP_DEGREE, kp2=1, gemm=True, lev=dims[i][4],
                    )
            else:
                if dims[i][3] == "CR":
                    temp_outputs = self.deepflow_function(
                        exp_config=f"{self.CONFIG_DIR}/v100.yaml", exp_dir=f"{self.OUTDIR}/LLM", debug=True,
                        m=dims[i][0], n=dims[i][1], k=dims[i][2], t="CR", kp1=self.TP_DEGREE, kp2=1, gemm=True, lev=dims[i][4],
                    )
                else:
                    temp_outputs = self.deepflow_function(
                        exp_config=f"{self.CONFIG_DIR}/v100.yaml", exp_dir=f"{self.OUTDIR}/LLM", debug=True,
                        m=dims[i][0], n=dims[i][1], k=dims[i][2], t="RC", kp1=self.TP_DEGREE, kp2=self.TP_DEGREE, gemm=True, lev=dims[i][4],
                    )
            temp_outputs.extend([dims[i][4], dims[i][0], dims[i][1], dims[i][2]])
            mha_names = ("X.W=KQV", "Q.K=R", "R.V=Z", "Z.W=Y")
            ffn_names = ("Y.WL1=O1", "O1.WL2=O2")
            if dims[i][4] in mha_names:
                mha_GEMMtime += float(temp_outputs[0])
                mha_reduction += float(temp_outputs[1])
            elif dims[i][4] in ffn_names:
                ffn_GEMMtime += float(temp_outputs[0])
                ffn_reduction += float(temp_outputs[1])
            self.deepflow_outputs.append(temp_outputs)
            with open(output_file, "a") as f:
                f.write(f"{dims[i][4]}, {dims[i][0]}, {dims[i][1]}, {dims[i][2]}, {dims[i][3]}, {temp_outputs[0]}, {temp_outputs[1]}\n")
            if dims[i][4] == "O1.WL2=O2":
                self.deepflow_outputs.append([mha_GEMMtime, mha_reduction, "MHA", 0, 0, 0])
                self.deepflow_outputs.append([ffn_GEMMtime, ffn_reduction, "FFN", 0, 0, 0])
                with open(output_file, "a") as f:
                    f.write(f"MHA, 0, 0, 0,CR, {mha_GEMMtime}, {mha_reduction}\n")
                    f.write(f"FFN, 0, 0, 0,RC, {ffn_GEMMtime}, {ffn_reduction}\n")
                mha_GEMMtime = mha_reduction = ffn_GEMMtime = ffn_reduction = 0


# ----- 3) time_domain（精炼为 stub，仅保留接口） -----
class time_domain:
    def __init__(self, amped, deepFlow_res) -> None:
        self.amped = amped
        self.deepFlow_compute = deepFlow_res
        if DEBUG_PRINT:
            print("time_domain (stub): amped + deepFlow_res stored, skip timeline generation.")


# ----- 4) amped_exec（仅保留 training 单阶段） -----
class amped_exec:
    def __init__(self, training=True) -> None:
        self.debug = DEBUG_PRINT
        self.inputs = []
        self.breakdown = {}
        self.training = training
        self.timeStamp = ""
        self.main()

    def calc_time(self, inputs, seqLen, flag_gen=False, debug=False, iter=1):
        if not self.training:
            inputs.parameters["context"] = seqLen
            inputs.parameters["tokens_to_train"] = seqLen
            inputs.parameters["summarization_len"] = iter if flag_gen else seqLen
        inputs.dependency_mapping = CalculateFunctionsDependencyMapping()
        for parameter_name in inputs.temp_parameters_to_calculate:
            if parameter_name not in inputs.temp_parameters_dict:
                inputs.calculate_parameter(parameter_name, inputs.temp_parameters_dict)
        inputs.parameters = Parameters(inputs, inputs.temp_parameters_dict, inputs.dependency_mapping)
        inputs.transformer = inputs.config["neural_network_training_parameters"]["lookup_config"]["lookup_table_row"]
        inputs.accelerator = inputs.config["accelerator_architecture_parameters"]["lookup_config"]["lookup_table_row"]
        perf_model = PerformanceModel(inputs)
        p = perf_model.p
        perLayer_computetime_fwd = (perf_model.compute_time_forward_pass()) / (p["data_parallel_degree"] * p["tensor_parallel_degree"] * p["pipeline_parallel_degree"])
        perLayer_commtime_fwd = perf_model.communication_time_forward_pass()
        perLayer_bubble_fwd = (p["pipeline_parallel_degree"] - 1) * (
            (perf_model.compute_time_forward_pass()) / (p["data_parallel_degree"] * p["tensor_parallel_degree"] * p["pipeline_parallel_degree"] * p["layers"])
            + perf_model.communication_time_forward_pass()
        ) / p["number_of_microbatches_per_minibatch"]
        if self.training:
            perLayer_computetime_bwd = (perf_model.compute_time_backward_pass() + perf_model.weight_update_time()) / (
                p["data_parallel_degree"] * p["tensor_parallel_degree"] * p["pipeline_parallel_degree"])
            perLayer_commtime_bwd = perf_model.communication_time_backwards_DP_all_reduce() + perf_model.communication_time_backward_pass()
            perLayer_bubble_bwd = (p["pipeline_parallel_degree"] - 1) * (
                (perf_model.compute_time_backward_pass()) / (p["data_parallel_degree"] * p["tensor_parallel_degree"] * p["pipeline_parallel_degree"] * p["layers"])
                + perf_model.communication_time_backward_pass()
            ) / p["number_of_microbatches_per_minibatch"]
        nbatch, layers = p["number_of_batches"], p["layers"]
        computetime = nbatch * layers * perLayer_computetime_fwd
        commtime = nbatch * layers * perLayer_commtime_fwd
        commtime_pipeline_bubble = nbatch * layers * perLayer_bubble_fwd
        if self.training:
            computetime += nbatch * layers * perLayer_computetime_bwd
            commtime += nbatch * layers * perLayer_commtime_bwd
            commtime_pipeline_bubble += nbatch * layers * perLayer_bubble_bwd
        self.perf_model = perf_model
        return [computetime, commtime, commtime_pipeline_bubble]

    def training_string_training_time_breakdown(self):
        pm = self.perf_model
        pairs = {
            "Total time to train (s)": pm.total_time_to_train(),
            "Total time to train (days)": pm.total_time_to_train_days(),
            "Computation time forward pass (s)": pm.total_computation_time_forward_pass(),
            "Computation time backward pass (s)": pm.total_computation_time_backward_pass(),
            "Computation time weight updates (s)": pm.total_computation_time_weight_updates(),
            "Total computation time (s)": pm.total_computation_time(),
            "Total communication time forward pass (s)": pm.total_communication_time_forward_pass(),
            "Total communication time backward pass (s)": pm.total_communication_time_backward_pass(),
            "Total communication time (s)": pm.total_communication_time(),
            "Waiting Time due to pipeline bubbles (s)": pm.total_waiting_time_due_to_pipeline_bubbles(),
        }
        self.breakdown = pairs
        longest = len(max(pairs.keys(), key=len))
        return "TRAINING TIME BREAKDOWN\n\n" + "\n".join(f"{k:-<{longest}} {v}" for k, v in pairs.items())

    def temp_save_as(self, filename: str, content: str, encoding=None, saveDir=""):
        saveDir = saveDir or "output_files"
        if not os.path.isdir(saveDir):
            os.mkdir(saveDir)
        self.timeStamp = time_prefix()
        open(f"{saveDir}/{self.timeStamp}{filename}", "w", encoding=encoding).write(content)

    def main(self):
        inputs = Inputs()
        if not self.training:
            inputs.parameters["batch_size"] = 1
            inputs.parameters["number_of_microbatches_per_minibatch"] = 1
        computetime_fwd, commtime_fwd, commtime_bubble = self.calc_time(inputs, inputs.parameters["context"], False, False)
        summary = "FULL CONFIGURATION\n\n" + inputs.parameters.to_string_structured()
        self.temp_save_as("config_summary.txt", summary, saveDir=OUTPUT_PATH)
        out_file = f"{OUTPUT_PATH}/AmpedInference.txt" if not self.training else f"{OUTPUT_PATH}/AmpedTraining.txt"
        with open(out_file, "w") as f:
            f.write(f"Stage,0,Compute time (s),{computetime_fwd},Communication time (s),{commtime_fwd},Pipeline bubble time (s),{commtime_bubble}\n")
        self.inputs = inputs
        self.temp_save_as("training_time_breakdown.txt", self.training_string_training_time_breakdown(), saveDir=OUTPUT_PATH)


# ----- 5) mat_dims_ampedToDF -----
class mat_dims_ampedToDF:
    def __init__(self, amped, training=True) -> None:
        self.dims = {}
        self.training = training
        self.amped = amped
        self.main(self.amped.inputs)

    def mmm_breakup(self, B, D, S, h, nheads, h_MLP1, h_MLP2, N_DP, N_PP):
        numlevels = 6 * (GENERATION_LEN + 1) if not self.training else 6
        levels = ["X.W=KQV", "Q.K=R", "R.V=Z", "Z.W=Y", "Y.WL1=O1", "O1.WL2=O2"]
        if not self.training:
            S = PROMPT_LEN
        dims = {}
        dims[0] = [int(3 * B * S / N_DP / N_PP), D, h * nheads, "CR", levels[0]]
        dims[1] = [int(B * S / N_DP / N_PP), h * nheads, S, "CR", levels[1]]
        dims[2] = [int(B * S / N_DP / N_PP), S, h * nheads, "CR", levels[2]]
        dims[3] = [int(B * S / N_DP / N_PP), D, D, "CR", levels[3]]
        dims[4] = [int(B * S / N_DP / N_PP), D, h_MLP1, "RC", levels[4]]
        dims[5] = [int(B * S / N_DP / N_PP), h_MLP1, h_MLP2, "RC", levels[5]]
        if not self.training:
            S = 1
            for i in range(1, GENERATION_LEN):
                dims[0 + 6 * i] = [int(3 * B * S / N_DP / N_PP), D, h * nheads, "CR", levels[0]]
                dims[1 + 6 * i] = [S, int(B * S * (i + PROMPT_LEN) / N_DP / N_PP), h * nheads, "CR", levels[1]]
                dims[2 + 6 * i] = [S, h * nheads, int(B * S * (i + PROMPT_LEN) / N_DP / N_PP), "CR", levels[2]]
                dims[3 + 6 * i] = [int(B * S / N_DP / N_PP), D, D, "CR", levels[3]]
                dims[4 + 6 * i] = [int(B * S / N_DP / N_PP), D, h_MLP1, "RC", levels[4]]
                dims[5 + 6 * i] = [int(B * S / N_DP / N_PP), h_MLP1, h_MLP2, "RC", levels[5]]
        if not os.path.isdir(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)
        with open(f"{OUTPUT_PATH}/{self.amped.timeStamp}mat_dims_amped.txt", "w") as file:
            for i in range(len(dims)):
                tmp = str([dims[i]]).replace("[", "").replace("]", "").replace(",", " ")
                file.write(tmp + "\n")
        self.dims = dims

    def main(self, inputs):
        B = int(inputs.parameters["batch_size"])
        D = int(inputs.parameters["dimensionality"])
        S = int(inputs.parameters["context"])
        h = int(inputs.parameters["hidden_layer_dimension_for_attention_sublayers"])
        nheads = int(inputs.parameters["attention_heads"])
        h_MLP1 = int(inputs.parameters["hidden_layer_dimension_MLP_1"])
        h_MLP2 = int(inputs.parameters["hidden_layer_dimension_MLP_2"])
        N_DP = int(inputs.parameters["data_parallel_degree"])
        N_PP = int(inputs.parameters["pipeline_parallel_degree"])
        return self.mmm_breakup(B, D, S, h, nheads, h_MLP1, h_MLP2, N_DP, N_PP)


# ----- 6) update_configs -----
def _resolve_config_path(p):
    if os.path.isabs(p) and os.path.exists(p):
        return p
    repo_root = os.path.dirname(__file__)
    cand = os.path.join(repo_root, "amped_backups", p)
    if os.path.exists(cand):
        return cand
    cand = os.path.join(os.getcwd(), p)
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"Config not found: {p}")


def _dump_pretty(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, separators=(", ", ": "), sort_keys=False)
        f.write("\n")


def update_configs(cSim, config_file):
    _, DP_intra, DP_inter, _, TP_intra, TP_inter, _, PP_intra, PP_inter, intraGPUs = cSim.split("_")
    DP_intra, DP_inter = int(DP_intra), int(DP_inter)
    TP_intra, TP_inter = int(TP_intra), int(TP_inter)
    PP_intra, PP_inter = int(PP_intra), int(PP_inter)
    intraGPUs = int(intraGPUs)
    src_cfg_path = _resolve_config_path(config_file)
    with open(src_cfg_path, "r") as f:
        data = json.load(f)
    mp = data["mapping_parameters"]["parameters"]
    mp["intra_node_data_parallel_degree"]["value"] = DP_intra
    mp["inter_node_data_parallel_degree"]["value"] = DP_inter
    mp["intra_node_tensor_parallel_degree"]["value"] = TP_intra
    mp["inter_node_tensor_parallel_degree"]["value"] = TP_inter
    mp["intra_node_pipeline_parallel_degree"]["value"] = PP_intra
    mp["inter_node_pipeline_parallel_degree"]["value"] = PP_inter
    sp = data["system_architecture_parameters"]["parameters"]
    sp["number_of_accelerators_per_node"]["value"] = intraGPUs
    sp["number_of_network_cards_per_node"]["value"] = intraGPUs
    for k in ("effective_perf_perc_K_Q_V", "effective_perf_perc_attention", "effective_perf_perc_output", "effective_perf_perc_MLP"):
        sp[k]["value"] = 0.7
        sp[k]["calculated"] = False
    _dump_pretty(src_cfg_path, data)
    repo_root = os.path.dirname(__file__)
    amped_cfg = os.path.join(repo_root, "amped_backups", "config.json")
    _dump_pretty(amped_cfg, data)


# ----- 7) 入口 -----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="AMPeD config JSON path.")
    args = parser.parse_args()

    itrers = ["DP_1_16_TP_8_1_PP_1_8_8"]  # 可改为多配置列表

    for cSim in itrers:
        print(f"Processing: {cSim}")
        update_configs(cSim, args.config)
        RESULT_DIR = cSim
        OUTPUT_PATH = f"/home/fd420/LLM_analytical_tools/amped_deepflow/output_files/{RESULT_DIR}"

        AMPeD = amped_exec(training=True)
        mat_dims = mat_dims_ampedToDF(amped=AMPeD, training=True)
        DeepFlow = deepflow_exec(AMPeD, mat_dims.dims)
        cal_time(AMPeD, DeepFlow.deepflow_outputs)
        time_domain(AMPeD, DeepFlow.deepflow_outputs)
