# training_core.py: 从 training.py 提取的核心流水线
# 运行前请确保 sys.path 包含 ../AMPeD 与 ../DeepFlow，且 amped_backups/config.json 存在
# 使用方式: python training_core.py --config <path_to_config.json>
#
# 核心部分索引（约）:
#   1) cal_time           ~45
#   2) deepflow_exec      ~89
#   3) time_domain        ~233
#   4) amped_exec         ~1494
#   5) mat_dims_ampedToDF ~1844
#   6) update_configs     ~1977
#   7) if __name__        ~2056

from math import ceil
import os
import sys

sys.path.insert(1, "../AMPeD")
sys.path.insert(1, "../DeepFlow")

import argparse
import csv
import json
import pickle
import glob
import shutil

import click
import config
import numpy as np
import pandas as pd
import perf
from amped import save_GEMM_breakdown
from amped.common import time_prefix
from amped.performance_model import PerformanceModel
from perf import TimeCalculation

from amped_backups.inputs import CalculateFunctionsDependencyMapping, Inputs, Parameters

PROMPT_LEN = 2048
GENERATION_LEN = 0
DEBUG_PRINT = False
NR_BATCHS_TO_PROCESS = 2
LEN_BUFFER_BEFORE_WRITE = 1024

DEEPFLOW_CONFIG_PATH = (
    "/home/fd420/LLM_analytical_tools/amped_deepflow/deepflow_configs"
)
RESULT_DIR = "DP_2_1_TP_2_1_PP_1_1"
OUTPUT_PATH = f"/home/fd420/LLM_analytical_tools/amped_deepflow/output_files/{RESULT_DIR}"


# ---------------------------------------------------------------------------
# 1) cal_time
# ---------------------------------------------------------------------------
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
        t_elapsed = 0.0
        t_reduction_elapsed = 0.0
        for gemmTimes in self.deepflow_outputs:
            gemmTime = gemmTimes[0]
            reductionTime = gemmTimes[1]
            t_elapsed += gemmTime
            t_reduction_elapsed += reductionTime
        t_elapsed = t_elapsed * 2.0  # FW + BW
        t_reduction_elapsed = t_reduction_elapsed * 2.0
        return [t_elapsed, t_reduction_elapsed]

    def main(self, inputs):
        if self.debug:
            print("**** Computing GEMM timings using DeepFlow *****\n")
            print(self.breakdown)
        [t_FW_BW, t_REDUCTION] = self.time_from_GEMM()
        if self.debug:
            print("t_FW_BW:", t_FW_BW)
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


# ---------------------------------------------------------------------------
# 2) deepflow_exec
# ---------------------------------------------------------------------------
class deepflow_exec:
    def __init__(self, amped, dims, training=True) -> None:
        self.debug = DEBUG_PRINT
        self.amped = amped
        self.inputs = self.amped.inputs
        self.TP_DEGREE = self.inputs.parameters["tensor_parallel_degree"]
        self.CONFIG_DIR = DEEPFLOW_CONFIG_PATH
        self.OUTDIR = OUTPUT_PATH
        self.training = training
        if self.debug:
            print(f"TP degree is set to : {self.TP_DEGREE}")
            print(f"Config: {self.CONFIG_DIR}")
            print(f"Output dir: {self.OUTDIR}")
        self.deepflow_outputs = []
        self.main(dims)

    def deepflow_function(
        self,
        exp_config,
        exp_dir,
        debug,
        m,
        n,
        k,
        t,
        kp1,
        kp2,
        gemm,
        batch_size=2048,
        hidden_dim=19968,
        seq_len=20,
        vocab_size=800000,
        num_layer=2,
        dp=None,
        lp=None,
        lev=None,
        args_input=False,
    ):
        exp_path = os.path.expandvars(os.path.expanduser(exp_config))
        exp_config = config.parse_config(exp_path)
        TC = TimeCalculation(exp_config)
        if args_input:
            TC.updateParams(
                debug, m, n, k, t, kp1, kp2, dp, lp, gemm,
                batch_size, hidden_dim, seq_len, vocab_size, num_layer,
            )
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
        mha_GEMMtime = 0
        mha_reduction = 0
        ffn_GEMMtime = 0
        ffn_reduction = 0
        output_file = f"{self.OUTDIR}/{self.amped.timeStamp}summary_deepflow.txt"
        with open(output_file, "w") as f:
            f.write(f"Layer, M, N, K, t, GEMM time, reduction time\n")
        for i in range(len(dims)):
            if not self.training:
                if dims[i][0] < self.TP_DEGREE:
                    temp_outputs = self.deepflow_function(
                        exp_config=f"{self.CONFIG_DIR}/v100.yaml",
                        exp_dir=f"{self.OUTDIR}/LLM",
                        debug=True,
                        m=dims[i][0], n=dims[i][1], k=dims[i][2],
                        t="RC", kp1=1, kp2=self.TP_DEGREE,
                        gemm=True, lev=dims[i][4],
                    )
                else:
                    temp_outputs = self.deepflow_function(
                        exp_config=f"{self.CONFIG_DIR}/v100.yaml",
                        exp_dir=f"{self.OUTDIR}/LLM",
                        debug=True,
                        m=dims[i][0], n=dims[i][1], k=dims[i][2],
                        t="RC", kp1=self.TP_DEGREE, kp2=1,
                        gemm=True, lev=dims[i][4],
                    )
            else:
                if dims[i][3] == "CR":
                    temp_outputs = self.deepflow_function(
                        exp_config=f"{self.CONFIG_DIR}/v100.yaml",
                        exp_dir=f"{self.OUTDIR}/LLM",
                        debug=True,
                        m=dims[i][0], n=dims[i][1], k=dims[i][2],
                        t="CR", kp1=self.TP_DEGREE, kp2=1,
                        gemm=True, lev=dims[i][4],
                    )
                else:
                    temp_outputs = self.deepflow_function(
                        exp_config=f"{self.CONFIG_DIR}/v100.yaml",
                        exp_dir=f"{self.OUTDIR}/LLM",
                        debug=True,
                        m=dims[i][0], n=dims[i][1], k=dims[i][2],
                        t="RC", kp1=self.TP_DEGREE, kp2=self.TP_DEGREE,
                        gemm=True, lev=dims[i][4],
                    )
            temp_outputs.append(dims[i][4])
            temp_outputs.append(dims[i][0])
            temp_outputs.append(dims[i][1])
            temp_outputs.append(dims[i][2])
            if (dims[i][4] == "X.W=KQV" or dims[i][4] == "Q.K=R" or dims[i][4] == "R.V=Z" or dims[i][4] == "Z.W=Y"):
                mha_GEMMtime += float(temp_outputs[0])
                mha_reduction += float(temp_outputs[1])
            elif dims[i][4] == "Y.WL1=O1" or dims[i][4] == "O1.WL2=O2":
                ffn_GEMMtime += float(temp_outputs[0])
                ffn_reduction += float(temp_outputs[1])
            self.deepflow_outputs.append(temp_outputs)
            with open(output_file, "a") as f:
                f.write(
                    f"{dims[i][4]}, {dims[i][0]}, {dims[i][1]}, {dims[i][2]}, {dims[i][3]}, {temp_outputs[0]}, {temp_outputs[1]}\n"
                )
            if self.debug:
                print(
                    f"Dimension:{dims[i]}\n\tGEMM time: {self.deepflow_outputs[i][0]}\tReduction time: {self.deepflow_outputs[i][1]}\n"
                )
            if dims[i][4] == "O1.WL2=O2":
                self.deepflow_outputs.append([mha_GEMMtime, mha_reduction, "MHA", 0, 0, 0])
                self.deepflow_outputs.append([ffn_GEMMtime, ffn_reduction, "FFN", 0, 0, 0])
                with open(output_file, "a") as f:
                    f.write(f"MHA, 0, 0, 0,CR, {mha_GEMMtime}, {mha_reduction}\n")
                    f.write(f"FFN, 0, 0, 0,RC, {ffn_GEMMtime}, {ffn_reduction}\n")
                if self.debug:
                    print(f"Multi head attention:\n\tGEMM time: {mha_GEMMtime}\tReduction time: {mha_reduction}\n")
                    print(f"Feed-Forward layer:\n\tGEMM time: {ffn_GEMMtime}\tReduction time: {ffn_reduction}\n")
                mha_GEMMtime = 0
                mha_reduction = 0
                ffn_GEMMtime = 0
                ffn_reduction = 0


# ---------------------------------------------------------------------------
# 3) time_domain
# ---------------------------------------------------------------------------
class time_domain:
    def __init__(self, amped, deepFlow_res) -> None:
        self.amped = amped
        self.inputs = self.amped.inputs
        self.perf = self.amped.perf_model

        self.debug = DEBUG_PRINT
        self.timeline = []
        self.buffer_write_to_file = {}
        self.start = 0.0

        self.deepFlow_compute = deepFlow_res

        if self.debug:
            print(self.deepFlow_compute)

        self.linear_throughput = self.perf.reciprocal_of_OPS()
        self.non_linear_throughput = self.perf.C_NONLIN()

        self.B = int(self.inputs.parameters["batch_size"])
        self.uB = int(self.inputs.parameters["microbatch_size"])
        self.D = int(self.inputs.parameters["dimensionality"])
        self.S = int(self.inputs.parameters["context"])
        self.sum_len = int(self.inputs.parameters["summarization_len"])
        self.h = int(
            self.inputs.parameters["hidden_layer_dimension_for_attention_sublayers"]
        )
        self.nheads = int(self.inputs.parameters["attention_heads"])
        self.h_MLP1 = int(self.inputs.parameters["hidden_layer_dimension_MLP_1"])
        self.h_MLP2 = int(self.inputs.parameters["hidden_layer_dimension_MLP_2"])
        self.N_DP = int(self.inputs.parameters["data_parallel_degree"])
        self.N_PP = int(self.inputs.parameters["pipeline_parallel_degree"])
        self.N_TP = int(self.inputs.parameters["tensor_parallel_degree"])

        self.N_TP_INTRA = int(
            self.inputs.parameters["intra_node_tensor_parallel_degree"]
        )
        self.N_TP_INTER = int(
            self.inputs.parameters["inter_node_tensor_parallel_degree"]
        )

        self.N_PP_INTRA = int(
            self.inputs.parameters["intra_node_pipeline_parallel_degree"]
        )
        self.N_PP_INTER = int(
            self.inputs.parameters["inter_node_pipeline_parallel_degree"]
        )

        self.N_DP_INTRA = int(self.inputs.parameters["intra_node_data_parallel_degree"])
        self.N_DP_INTER = int(self.inputs.parameters["inter_node_data_parallel_degree"])

        self.N_TOTAL_GPUS = int(self.inputs.parameters["total_number_of_accelerators"])
        self.N_TOTAL_GPUS_PERNODE = int(
            self.inputs.parameters["number_of_accelerators_per_node"]
        )

        self.nrOfLayers = int(self.perf.p["layers"])
        self.nrOfBatch = int(self.perf.p["number_of_batches"])

        # print(f"Parsed data info:")
        print(f"Intrastructure:")
        print(f"\tTotal number of GPUs: {self.N_TOTAL_GPUS}")
        print(f"\tGPUs per node: {self.N_TOTAL_GPUS_PERNODE}")
        print(f"Parallelism:")
        print(
            f"\tData Parallel: {self.N_DP} [intra:{self.N_DP_INTRA}, inter:{self.N_DP_INTER}]"
        )
        print(
            f"\tTensor Parallel: {self.N_TP} [intra:{self.N_TP_INTRA}, inter:{self.N_TP_INTER}]"
        )
        print(
            f"\tPipeline Parallel: {self.N_PP} [intra:{self.N_PP_INTRA}, inter:{self.N_PP_INTER}]"
        )

        self.weight_precision = self.inputs.parameters["weight_precision"]
        self.gradient_precision = self.inputs.parameters["gradient_precision"]
        self.activation_precision = self.inputs.parameters["activation_precision"]
        self.optimizerstate_precision = self.inputs.parameters[
            "optimizer_state_precision"
        ]

        self.total_profiles = 0
        self.deepflow_mhatime = self.deepFlow_compute[6][0]
        self.deepflow_ffntime = self.deepFlow_compute[7][0]

        self.main(self.inputs, self.perf)

    def total_timeline(self, fileName=""):
        if fileName == "":
            fileName = f"{OUTPUT_PATH}/{self.amped.timeStamp}time_series.csv"
        else:
            fileName = f"{OUTPUT_PATH}/{fileName}"

        _file = open(fileName, "w")
        _file.write(
            f"Total batches, {self.nrOfBatch}, Processed Batches, {NR_BATCHS_TO_PROCESS}\n"
        )
        _file.write(
            f"Layer, Type, start time, end time, duration, Bytes to be transferred, Collective type, Parallelism, Locality, Degree"
            + "\n"
        )
        for x in range(len(self.timeline)):
            # if x % 10 == 0:
            # print(f"Writing step {x}: {self.timeline[x][0]}")
            cList = self.timeline[x]
            # print(cList)
            if cList[4] != 0:
                tmp = str(cList).replace("[", "").replace("]", "")  # .replace(",", " ")
                _file.write(tmp + "\n")

    def individual_timeline(self, total_GPUs, start_id=-1, end_id=-1, _fileName=""):
        if start_id == -1:
            start_id = 0

        if end_id == -1:
            end_id = total_GPUs

        _total_profiles = end_id - start_id
        self.total_profiles = _total_profiles

        sourceGPU = 0
        destGPU = 1
        fileNames = []

        for cGPU in range(start_id, end_id):
            if _fileName == "":
                fileName = (
                    f"{OUTPUT_PATH}/{self.amped.timeStamp}time_series_GPU_{cGPU}.csv"
                )
            else:
                fileName = f"{OUTPUT_PATH}/{_fileName}_GPU_{cGPU}"

            self.buffer_write_to_file[f"GPU_{cGPU}"] = {
                "fileName": fileName,
                "len": 0,
                "data": "",
            }
            # print("")

            fileNames.append(fileName)
            with open(fileName, "w") as _file:
                _file.write(
                    f"Layer, Type, start time, end time, duration, Bytes to be transferred, Collective type, Parallelism, Locality, Degree, SRC, DEST"
                    + "\n"
                )

        # for key in self.buffer_write_to_file.keys():
        #     print(key)

        # print("\n" * 2)
        for x in range(len(self.timeline)):
            tmpStr = len(self.timeline) - 1
            sys.stdout.write(f"Processing layers: {x}/{tmpStr}\r")
            sys.stdout.flush()

            cList = self.timeline[x]

            layerName = cList[0]
            layerType = cList[1]
            startTime = float(cList[2])
            endTime = float(cList[3])
            layerDuration = float(cList[4])
            parallelismType = cList[7]
            # print(cList)

            if layerDuration != 0:
                if cList[6] == "ALLREDUCE":
                    if parallelismType == "TP":
                        self.for_TP(cList, start_id, end_id)
                    elif parallelismType == "DP":
                        self.for_DP(cList, start_id, end_id)
                else:
                    parallelismType = cList[7]
                    if parallelismType == "PP":
                        self.for_PP(cList, start_id, end_id)
                    else:
                        nList = [
                            cList[0],
                            cList[1],
                            cList[2],
                            cList[3],
                            cList[4],
                            cList[5],
                            cList[6],
                            cList[7],
                            cList[8],
                        ]
                        tmp = (
                            str(nList).replace("[", "").replace("]", "")
                        )  # .replace(",", " ")
                        for cGPU in range(start_id, end_id):
                            # print(tmp)
                            self.buffer_write_to_file[f"GPU_{cGPU}"]["data"] += (
                                tmp + "\n"
                            )
                            self.buffer_write_to_file[f"GPU_{cGPU}"]["len"] += 1
                            self.save_individual_files()
        self.save_individual_files(True)

        print(f"Processed individual profiles for {self.total_profiles} GPUs")
        print()

    def animate(self, _done=False):
        printMsg = ""
        UP = "\x1b[3A"
        CLR = "\x1b[0K"
        cnt = self.total_profiles
        for key in self.buffer_write_to_file.keys():
            value = self.buffer_write_to_file[key]
            _len = value["len"]
            if _done:
                printMsg += "".join([f"{UP}" * cnt, f"Processing {key} --> Done\n"])
                # printMsg += f"Processing {key} --> Current buffer length {_len}\r"
            else:
                printMsg += "".join(
                    [
                        f"{UP}" * cnt,
                        f"Processing {key} --> Current buffer length {_len}\n",
                    ]
                )
                # printMsg += f"Processing {key} --> Current buffer length {_len}"
            cnt -= 1

        # printMsg += "\r"

        # print(printMsg, end="", flush=True)
        print(f"{printMsg}")
        # sys.stdout.write(printMsg)
        # sys.stdout.flush()

    def save_individual_files(self, flag_leftover=False):
        # self.animate()
        for key in self.buffer_write_to_file.keys():
            value = self.buffer_write_to_file[key]
            _len = value["len"]
            if flag_leftover:
                # print(f"{key} --> Writing lines: {_len}")
                fileName = value["fileName"]
                _data = value["data"]
                with open(fileName, "a") as _file:
                    _file.write(_data + "\n")
                self.buffer_write_to_file[key]["data"] = ""
                self.buffer_write_to_file[key]["len"] = 0
            else:
                if value["len"] > LEN_BUFFER_BEFORE_WRITE:
                    # print(f"{key} --> Writing lines: {_len}")
                    fileName = value["fileName"]
                    _data = value["data"]
                    with open(fileName, "a") as _file:
                        _file.write(_data + "\n")
                    self.buffer_write_to_file[key]["data"] = ""
                    self.buffer_write_to_file[key]["len"] = 0
                    # self.animate(True)
                    # print(f"Processing {key} --> Done                               ")

    def temp_helper_for_individual(self, cGPU, _data, _fileName=""):
        if _fileName == "":
            fileName = f"{OUTPUT_PATH}/{self.amped.timeStamp}time_series_GPU_{cGPU}.csv"
        else:
            fileName = f"{OUTPUT_PATH}/{_fileName}_GPU_{cGPU}"

        with open(fileName, "a") as _file:
            _file.write(_data + "\n")

    def for_PP(self, cList, start_id, end_id):
        parallelismType = cList[7]
        parallelismLocality = cList[8]
        group_size_intra = self.N_PP_INTRA * self.N_TP_INTRA
        group_size_inter = self.N_TOTAL_GPUS_PERNODE * self.N_TP_INTER
        for sourceGPU in range(start_id, end_id):
            if parallelismLocality == "INTER":
                destGPU = sourceGPU + group_size_inter
                if destGPU >= self.N_TOTAL_GPUS:
                    destGPU = destGPU - self.N_TOTAL_GPUS

            elif parallelismLocality == "INTRA":
                destGPU = sourceGPU + 1
                if (destGPU % group_size_intra) == 0:
                    destGPU = destGPU - group_size_intra

            nList = [
                cList[0],
                cList[1],
                cList[2],
                cList[3],
                cList[4],
                cList[5],
                cList[6],
                cList[7],
                cList[8],
                cList[9],
                sourceGPU,
                destGPU,
            ]
            tmp = str(nList).replace("[", "").replace("]", "")  # .replace(",", " ")
            self.buffer_write_to_file[f"GPU_{sourceGPU}"]["data"] += tmp + "\n"
            self.buffer_write_to_file[f"GPU_{sourceGPU}"]["len"] += 1
            self.save_individual_files()
            # self.temp_helper_for_individual(sourceGPU, tmp)

    def for_TP(self, cList, start_id, end_id):
        layerName = cList[0]
        layerType = cList[1]
        startTime = float(cList[2])
        endTime = float(cList[3])
        layerDuration = float(cList[4])
        commVolume = float(cList[5])
        # collectiveType = cList[6]
        parallelismType = cList[7]
        parallelismLocality = cList[8]
        nDegree = cList[9]
        collectiveType = "P2P"

        arr_steps = 2 * (nDegree - 1)
        nStartTime = startTime
        lDuration = layerDuration / arr_steps
        nEndTime = nStartTime + lDuration
        dVolume = commVolume / nDegree

        group_size_intra = self.N_TP_INTRA
        group_size_inter = self.N_TOTAL_GPUS_PERNODE

        for sourceGPU in range(start_id, end_id):
            if parallelismLocality == "INTER":
                destGPU = sourceGPU + group_size_inter
                if destGPU >= self.N_TOTAL_GPUS:
                    destGPU = destGPU - self.N_TOTAL_GPUS

            elif parallelismLocality == "INTRA":
                destGPU = sourceGPU + 1
                if (destGPU % group_size_intra) == 0:
                    destGPU = destGPU - group_size_intra

            for x in range(arr_steps):
                lName = f"{layerName}_{x}"
                nStartTime = startTime
                nEndTime = nStartTime + lDuration
                nList = [
                    lName,
                    layerType,
                    nStartTime,
                    nEndTime,
                    lDuration,
                    dVolume,
                    collectiveType,
                    parallelismType,
                    parallelismLocality,
                    nDegree,
                    sourceGPU,
                    destGPU,
                ]

                tmp = str(nList).replace("[", "").replace("]", "")  # .replace(",", " ")
                nStartTime = nEndTime
                self.buffer_write_to_file[f"GPU_{sourceGPU}"]["data"] += tmp + "\n"
                self.buffer_write_to_file[f"GPU_{sourceGPU}"]["len"] += 1
                self.save_individual_files()

                # self.temp_helper_for_individual(sourceGPU, tmp)

    def for_DP(self, cList, start_id, end_id):
        layerName = cList[0]
        layerType = cList[1]
        startTime = float(cList[2])
        endTime = float(cList[3])
        layerDuration = float(cList[4])
        commVolume = float(cList[5])
        # collectiveType = cList[6]
        parallelismType = cList[7]
        parallelismLocality = cList[8]
        nDegree = cList[9]
        collectiveType = "P2P"

        arr_steps = 2 * (nDegree - 1)
        lDuration = layerDuration / arr_steps
        nStartTime = startTime
        nEndTime = nStartTime + lDuration
        dVolume = commVolume / nDegree

        if parallelismType == "DP":
            group_size_intra = self.N_TP_INTRA * self.N_PP_INTRA
            group_size_inter = (
                self.N_TOTAL_GPUS_PERNODE * self.N_TP_INTER * self.N_PP_INTER
            )

        for sourceGPU in range(start_id, end_id):
            if parallelismLocality == "INTER":
                destGPU = sourceGPU + group_size_inter
                if destGPU >= self.N_TOTAL_GPUS:
                    destGPU = destGPU - self.N_TOTAL_GPUS

            elif parallelismLocality == "INTRA":
                destGPU = sourceGPU + 1
                if (destGPU % group_size_intra) == 0:
                    destGPU = destGPU - group_size_intra

            for x in range(arr_steps):
                lName = f"{layerName}_{x}"
                nStartTime = startTime
                nEndTime = nStartTime + lDuration
                nList = [
                    lName,
                    layerType,
                    nStartTime,
                    nEndTime,
                    lDuration,
                    dVolume,
                    collectiveType,
                    parallelismType,
                    parallelismLocality,
                    nDegree,
                    sourceGPU,
                    destGPU,
                ]

                tmp = str(nList).replace("[", "").replace("]", "")  # .replace(",", " ")
                nStartTime = nEndTime

                # print(tmp)
                self.buffer_write_to_file[f"GPU_{sourceGPU}"]["data"] += tmp + "\n"
                self.buffer_write_to_file[f"GPU_{sourceGPU}"]["len"] += 1
                self.save_individual_files()
                # self.temp_helper_for_individual(sourceGPU, tmp)

    def main(self, inputs, perf):
        self.start = 0
        self.timeline = []

        PP_switch_layer = self.nrOfLayers / self.N_PP

        for z in range(NR_BATCHS_TO_PROCESS):
            if self.debug:
                print(f"Processing batch {z}")
            for i in range(self.nrOfLayers):
                self.forward_pass(inputs, perf)

                if self.N_PP > 1:
                    if i == PP_switch_layer - 1 or i == self.nrOfLayers - 1:
                        self.pp_overhead_FWD(inputs, perf)

            for i in range(self.nrOfLayers):
                self.backward_pass(inputs, perf)

                if self.N_PP > 1:
                    if i == PP_switch_layer - 1 or i == self.nrOfLayers - 1:
                        self.pp_overhead_BWD(inputs, perf)

            if self.N_DP > 1 or self.N_PP > 1:
                self.dp_overhead(inputs, perf)

            if self.N_TP > 1 or self.N_PP > 1:
                self.zeroDP(inputs, perf)

            for i in range(self.nrOfLayers):
                self.weight_update(inputs, perf)

        # if self.debug:
        #    print("Done processing all batches")

        self.total_timeline()
        # _file = open(f"{OUTPUT_PATH}/{self.amped.timeStamp}time_series.csv", "w")
        # _file.write(
        #     f"Total batches, {self.nrOfBatch}, Processed Batches, {NR_BATCHS_TO_PROCESS}\n"
        # )
        # _file.write(
        #     f"Layer, Type, start time, end time, duration, Bytes to be transferred, Collective type, Parallelism, Locality, Degree"
        #     + "\n"
        # )
        # for x in range(len(self.timeline)):
        #     # if x % 10 == 0:
        #     # print(f"Writing step {x}: {self.timeline[x][0]}")
        #     cList = self.timeline[x]
        #     # print(cList)
        #     if cList[4] != 0:
        #         tmp = str(cList).replace("[", "").replace("]", "")  # .replace(",", " ")
        #         _file.write(tmp + "\n")

        # self.individual_timeline(total_GPUs=self.N_TOTAL_GPUS)  # 输出 per GPU csv [NEW]

        # _file = open(
        #     f"{OUTPUT_PATH}/{self.amped.timeStamp}_time_series_single_GPU.csv", "w"
        # )
        # _file.write(
        #     f"Layer, Type, start time, end time, duration, Bytes to be transferred, Collective type, Parallelism, Locality, Degree, SRC, DEST"
        #     + "\n"
        # )
        # sourceGPU = 0
        # destGPU = 1
        # for x in range(len(self.timeline)):
        #
        #     cList = self.timeline[x]
        #
        #     layerName = cList[0]
        #     layerType = cList[1]
        #     startTime = float(cList[2])
        #     endTime = float(cList[3])
        #     layerDuration = float(cList[4])
        #
        #     if layerDuration != 0:
        #         if cList[6] == "ALLREDUCE":
        #             # print(cList)
        #             commVolume = float(cList[5])
        #             collectiveType = cList[6]
        #             parallelismType = cList[7]
        #             parallelismLocality = cList[8]
        #             nDegree = cList[9]
        #
        #             nStartTime = startTime
        #             for x in range(6):
        #                 lName = f"{layerName}_{x}"
        #                 collectiveType = "P2P"
        #                 lDuration = layerDuration / 6
        #                 nEndTime = nStartTime + lDuration
        #                 dVolume = commVolume / nDegree
        #
        #                 if parallelismLocality == "INTER":
        #                     destGPU = sourceGPU + 1
        #                     if destGPU >= self.N_TOTAL_GPUS_PERNODE:
        #                         destGPU = destGPU - self.N_TOTAL_GPUS_PERNODE
        #                 elif parallelismLocality == "INTRA":
        #                     destGPU = sourceGPU + 1
        #                     if destGPU >= self.N_TOTAL_GPUS:
        #                         destGPU = destGPU - self.N_TOTAL_GPUS
        #
        #                 nList = [
        #                     lName,
        #                     layerType,
        #                     nStartTime,
        #                     nEndTime,
        #                     lDuration,
        #                     dVolume,
        #                     collectiveType,
        #                     parallelismType,
        #                     parallelismLocality,
        #                     nDegree,
        #                     sourceGPU,
        #                     destGPU,
        #                 ]
        #
        #                 tmp = (
        #                     str(nList).replace("[", "").replace("]", "")
        #                 )  # .replace(",", " ")
        #                 nStartTime = nEndTime
        #         else:
        #             parallelismType = cList[7]
        #             parallelismLocality = cList[8]
        #             if parallelismType == "PP":
        #                 if parallelismLocality == "INTRA":
        #                     # There is a bug here:
        #                     # If total GPUs used are less than GPUs per node then it is useless
        #                     # BUT not a problem for new but will come into picture later.
        #                     destGPU = sourceGPU + (
        #                         self.N_TOTAL_GPUS_PERNODE / self.N_PP_INTRA
        #                     )
        #                     if destGPU >= self.N_TOTAL_GPUS_PERNODE:
        #                         destGPU = destGPU - self.N_TOTAL_GPUS_PERNODE
        #                 elif parallelismLocality == "INTER":
        #                     destGPU = sourceGPU + (self.N_TOTAL_GPUS / self.N_PP_INTER)
        #                     if destGPU >= self.N_TOTAL_GPUS:
        #                         destGPU = destGPU - self.N_TOTAL_GPUS
        #                 nList = [
        #                     cList[0],
        #                     cList[1],
        #                     cList[2],
        #                     cList[3],
        #                     cList[4],
        #                     cList[5],
        #                     cList[6],
        #                     cList[7],
        #                     cList[8],
        #                     cList[9],
        #                     sourceGPU,
        #                     destGPU,
        #                 ]
        #             else:
        #                 nList = [
        #                     cList[0],
        #                     cList[1],
        #                     cList[2],
        #                     cList[3],
        #                     cList[4],
        #                     cList[5],
        #                     cList[6],
        #                     cList[7],
        #                     cList[8],
        #                 ]
        #             tmp = (
        #                 str(nList).replace("[", "").replace("]", "")
        #             )  # .replace(",", " ")
        #
        #         _file.write(tmp + "\n")

    def zeroDP(self, inputs, perf):
        if self.debug:
            print("Adding ZeroDP")

        gradient_volume_intra = (
            (perf.p["number_of_parameters_per_layer"])
            * (perf.p["gradient_precision"] / 8)
            * (self.nrOfLayers)
        )  # NOTE: Changed from batch size to microbatch size

        gradient_volume_inter = (
            (perf.p["number_of_parameters_per_layer"])
            * (perf.p["gradient_precision"] / 8)
            * (self.nrOfLayers)
        )  # NOTE: Changed from batch size to microbatch size

        if self.debug:
            print(
                f"gradientVolume: {(gradient_volume_intra / (self.N_TP_INTRA)) / (1024 * 1024)}"
            )

        TP_comm_intra_volume = gradient_volume_intra / self.N_DP_INTER

        nDegree = 0
        if self.N_DP_INTRA > 1:
            nDegree = nDegree + self.N_DP_INTRA
        if self.N_TP_INTRA > 1:
            nDegree = nDegree + self.N_TP_INTRA
        if self.N_PP_INTRA > 1:
            nDegree = nDegree + self.N_PP_INTRA

        if nDegree == 0:
            nDegree = 1

        if self.debug:
            print(f"Test nDrgree = {nDegree}")
        TP_comm_intra_volume = (TP_comm_intra_volume) / (
            nDegree
        )  # TODO: Verify this, since
        if self.N_PP_INTRA > 1 and self.N_TP_INTRA > 1:
            TP_comm_intra_volume = TP_comm_intra_volume / (1.5 * 1.5)
        TP_comm_intra_time = perf.backward_tensor_model_intra()

        TP_comm_inter_volume = gradient_volume_intra / self.N_DP_INTRA

        nDegree = 0
        if self.N_DP_INTER > 1:
            nDegree = nDegree + self.N_DP_INTER
        if self.N_TP_INTER > 1:
            nDegree = nDegree + self.N_TP_INTER
        if self.N_PP_INTER > 1:
            nDegree = nDegree + self.N_PP_INTER

        if nDegree == 0:
            nDegree = 1

        TP_comm_inter_volume = (gradient_volume_inter) / (nDegree)
        if self.N_PP_INTER > 1 and self.N_TP_INTER > 1:
            TP_comm_inter_volume = TP_comm_inter_volume / (1.5 * 1.5)
        TP_comm_inter_time = perf.backward_tensor_parallel_inter()

        if self.debug:
            print(f"ZERO-DP intra volume: {(TP_comm_intra_volume) / (1024 * 1024)}")

        self.timeline.append(
            [
                "FWD_DP_COMM_INTRA_ZERO",
                "Comm",
                self.start,
                self.update_time(TP_comm_intra_time),
                TP_comm_intra_time,
                TP_comm_intra_volume if TP_comm_intra_time > 0 else 0,
                "ALLREDUCE",
                "DP",
                "INTRA",
                self.N_TP_INTRA,
            ]
        )

        self.timeline.append(
            [
                "FWD_DP_COMM_INTER_ZERO",
                "Comm",
                self.start,
                self.update_time(TP_comm_inter_time),
                TP_comm_inter_time,
                TP_comm_inter_volume if TP_comm_inter_time > 0 else 0,
                "ALLREDUCE",
                "DP",
                "INTER",
                self.N_TP_INTER,
            ]
        )

    def forward_pass(self, inputs, perf):
        MHA_macs = inputs.parameters["total_attention_sublayer_MAC_operations"]
        FFN_macs = inputs.parameters["total_MLP_sublayer_MAC_operations"]
        MHA_nonlinear = inputs.parameters[
            "non_linear_operations_for_attention_sublayer"
        ]
        FFN_nonlinear = inputs.parameters["non_linear_operations_for_MLP_sublayer"]

        compute_time_linear = self.deepflow_mhatime
        compute_time_non_linear = (
            (MHA_nonlinear * self.non_linear_throughput)
            * ceil(perf.p["activation_precision"] / perf.W_FU_NONLIN)
        ) / (self.N_TP * self.N_PP)
        compute_time_MHA = compute_time_linear + compute_time_non_linear

        """
        Compute time from AMPeD
        compute_time_MHA =  ((MHA_macs*self.linear_throughput) * \
                            ceil(perf.p["weight_precision"]) / perf.W_FU_MAC) + \
                            ((MHA_nonlinear * self.non_linear_throughput) * \
                            ceil(perf.p["activation_precision"] / perf.W_FU_NONLIN))
        compute_time_MHA = compute_time_MHA / (self.N_TP * self.N_PP)"""

        compute_time_linear = self.deepflow_ffntime
        compute_time_non_linear = (
            (FFN_nonlinear * self.non_linear_throughput)
            * ceil(perf.p["activation_precision"] / perf.W_FU_NONLIN)
        ) / (self.N_TP * self.N_PP)
        compute_time_FFN = compute_time_linear + compute_time_non_linear

        """
        Compute time from AMPeD
        compute_time_FFN =  ((FFN_macs*self.linear_throughput)* \
                            ceil(perf.p["weight_precision"]) / perf.W_FU_MAC) + \
                            ((FFN_nonlinear * self.non_linear_throughput)* \
                            ceil(perf.p["activation_precision"] / perf.W_FU_NONLIN))
        compute_time_FFN = compute_time_FFN / (self.N_TP * self.N_PP)"""

        #####################################################################
        # Communication
        activation_volume = (self.uB * self.h * self.nheads * self.S) * (
            perf.p["activation_precision"] / 8
        )  # NOTE: Changed from batch size to microbatch size
        # print(f"activationVolume: {activation_volume/(1024*1024)}")
        # TP_comm_intra_volume = (activation_volume / self.N_TP_INTER)
        TP_comm_intra_volume = activation_volume / (
            self.N_TP_INTER
        )  # TODO: Verify this, since
        TP_comm_intra_time = perf.forward_tensor_model_intra()

        TP_comm_inter_volume = activation_volume / self.N_TP_INTRA
        TP_comm_inter_time = perf.forward_tensor_parallel_inter()

        """PP_comm_intra_volume = (activation_volume) # / perf.p["layers"]) # TODO: Check if this is true !!!
        PP_comm_intra_time = perf.forward_pipeline_parallel_intra()
        PP_comm_inter_time = perf.forward_pipeline_parallel_inter()

        if PP_comm_intra_time >= PP_comm_inter_time:
            PP_COMM_INTRA_time = PP_comm_intra_time
            PP_COMM_INTRA_volume = PP_comm_intra_volume
            PP_COMM_INTER_time = 0
            PP_COMM_INTER_volume = 0
        else:
            PP_COMM_INTRA_time = 0
            PP_COMM_INTRA_volume = 0
            PP_COMM_INTER_time = PP_comm_inter_time
            PP_COMM_INTER_volume = PP_comm_intra_volume"""

        # Can add MoE and DP here as well
        MoE_comm_time = perf.MoE_overhead_per_layer_fw_pass()
        DP_comm_time = perf.M_f_DP
        # Time line start
        self.timeline.append(
            [
                "FWD_MHA",
                "Compute",
                self.start,
                self.update_time(compute_time_MHA),
                compute_time_MHA,
                "NONE",
                "NONE",
                "NONE",
                0,
            ]
        )

        self.timeline.append(
            [
                "FWD_TP_COMM_INTRA",
                "Comm",
                self.start,
                self.update_time(TP_comm_intra_time / 2),
                TP_comm_intra_time / 2,
                TP_comm_intra_volume if TP_comm_intra_time > 0 else 0,
                "ALLREDUCE",
                "TP",
                "INTRA",
                self.N_TP_INTRA,
            ]
        )

        self.timeline.append(
            [
                "FWD_TP_COMM_INTER",
                "Comm",
                self.start,
                self.update_time(TP_comm_inter_time / 2),
                TP_comm_inter_time / 2,
                TP_comm_inter_volume if TP_comm_inter_time > 0 else 0,
                "ALLREDUCE",
                "TP",
                "INTER",
                self.N_TP_INTER,
            ]
        )

        self.timeline.append(
            [
                "FWD_FFN",
                "Compute",
                self.start,
                self.update_time(compute_time_FFN),
                compute_time_FFN,
                "NONE",
                "NONE",
                "NONE",
                0,
            ]
        )

        self.timeline.append(
            [
                "FWD_TP_COMM_INTRA",
                "Comm",
                self.start,
                self.update_time(TP_comm_intra_time / 2),
                TP_comm_intra_time / 2,
                TP_comm_intra_volume if TP_comm_intra_time > 0 else 0,
                "ALLREDUCE",
                "TP",
                "INTRA",
                self.N_TP_INTRA,
            ]
        )

        self.timeline.append(
            [
                "FWD_TP_COMM_INTER",
                "Comm",
                self.start,
                self.update_time(TP_comm_inter_time / 2),
                TP_comm_inter_time / 2,
                TP_comm_inter_volume if TP_comm_inter_time > 0 else 0,
                "ALLREDUCE",
                "TP",
                "INTER",
                self.N_TP_INTER,
            ]
        )

        """self.timeline.append([
            "FWD_PP_COMM_INTRA",
            "Comm",
            self.start,
            self.update_time(PP_COMM_INTRA_time),
            PP_COMM_INTRA_time,
            PP_COMM_INTRA_volume if PP_COMM_INTRA_time > 0 else 0,
            "P2P",
            "PP",
            "INTRA",
            self.N_PP_INTRA
        ])

        self.timeline.append([
            "FWD_PP_COMM_INTER",
            "Comm",
            self.start,
            self.update_time(PP_COMM_INTER_time),
            PP_COMM_INTER_time,
            PP_COMM_INTER_volume if PP_COMM_INTER_time > 0 else 0,
            "P2P",
            "PP",
            "INTER",
            self.N_PP_INTER
        ])"""

    def pp_overhead_FWD(self, inputs, perf):
        activation_volume = (self.uB * self.h * self.nheads * self.S) * (
            perf.p["activation_precision"] / 8
        )  # NOTE: Changed from batch size to microbatch size

        PP_comm_intra_volume = (
            # / perf.p["layers"]) # TODO: Check if this is true !!!
            activation_volume
        )
        PP_comm_intra_time = perf.forward_pipeline_parallel_intra()
        PP_comm_inter_time = perf.forward_pipeline_parallel_inter()

        if PP_comm_intra_time >= PP_comm_inter_time:
            PP_COMM_INTRA_time = PP_comm_intra_time
            PP_COMM_INTRA_volume = PP_comm_intra_volume
            PP_COMM_INTER_time = 0
            PP_COMM_INTER_volume = 0
        else:
            PP_COMM_INTRA_time = 0
            PP_COMM_INTRA_volume = 0
            PP_COMM_INTER_time = PP_comm_inter_time
            PP_COMM_INTER_volume = PP_comm_intra_volume

        self.timeline.append(
            [
                "FWD_PP_COMM_INTRA",
                "Comm",
                self.start,
                self.update_time(PP_COMM_INTRA_time),
                PP_COMM_INTRA_time,
                PP_COMM_INTRA_volume if PP_COMM_INTRA_time > 0 else 0,
                "P2P",
                "PP",
                "INTRA",
                self.N_PP_INTRA,
            ]
        )

        self.timeline.append(
            [
                "FWD_PP_COMM_INTER",
                "Comm",
                self.start,
                self.update_time(PP_COMM_INTER_time),
                PP_COMM_INTER_time,
                PP_COMM_INTER_volume if PP_COMM_INTER_time > 0 else 0,
                "P2P",
                "PP",
                "INTER",
                self.N_PP_INTER,
            ]
        )

    def update_time(self, duration):
        endTime = self.start + duration
        self.start = endTime
        return endTime

    def backward_pass(self, inputs, perf):
        MHA_macs = inputs.parameters["total_attention_sublayer_MAC_operations"]
        FFN_macs = inputs.parameters["total_MLP_sublayer_MAC_operations"]
        MHA_nonlinear = inputs.parameters[
            "non_linear_operations_for_attention_sublayer"
        ]
        FFN_nonlinear = inputs.parameters["non_linear_operations_for_MLP_sublayer"]

        compute_time_linear = self.deepflow_mhatime
        compute_time_non_linear = (
            (MHA_nonlinear * self.non_linear_throughput)
            * ceil(perf.p["weight_precision"] / perf.W_FU_NONLIN)
        ) / (self.N_TP * self.N_PP)
        compute_time_MHA = compute_time_linear + compute_time_non_linear

        """
        Compute time from AMPeD
        compute_time_MHA =  ((MHA_macs*self.linear_throughput) * ceil(max(perf.p["weight_precision"], perf.p["gradient_precision"])) / perf.W_FU_MAC) + ((MHA_nonlinear * self.non_linear_throughput) * ceil(perf.p["weight_precision"] / perf.W_FU_NONLIN))

        compute_time_MHA = compute_time_MHA/ (self.N_TP * self.N_PP)"""

        compute_time_linear = self.deepflow_ffntime
        compute_time_non_linear = (
            (FFN_nonlinear * self.non_linear_throughput)
            * ceil(perf.p["weight_precision"] / perf.W_FU_NONLIN)
        ) / (self.N_TP * self.N_PP)
        compute_time_FFN = compute_time_linear + compute_time_non_linear

        """
        Compute time from AMPeD
        compute_time_FFN =  ((FFN_macs*self.linear_throughput)* ceil(max(perf.p["weight_precision"], perf.p["gradient_precision"])) / perf.W_FU_MAC) + ((FFN_nonlinear * self.non_linear_throughput)* ceil(perf.p["weight_precision"] / perf.W_FU_NONLIN))

        compute_time_FFN = compute_time_FFN/ (self.N_TP * self.N_PP)"""

        #####################################################################
        # Communication
        error_volume_per_layer_batch = (self.uB * self.S * self.h * self.nheads) * (
            perf.p["gradient_precision"] / 8
        )
        # print(f"errorVolume: {error_volume_per_layer_batch/(1024*1024)}")
        TP_comm_intra_volume = error_volume_per_layer_batch / self.N_TP_INTER
        TP_comm_intra_time = perf.backward_tensor_model_intra()

        TP_comm_inter_volume = error_volume_per_layer_batch / self.N_TP_INTRA
        TP_comm_inter_time = perf.backward_tensor_parallel_inter()

        """PP_comm_intra_volume = (error_volume_per_layer_batch) # / perf.p["layers"])
        PP_comm_intra_time = perf.backward_pipeline_parallel_intra()
        PP_comm_inter_time = perf.backward_pipeline_parallel_inter()

        if PP_comm_intra_time >= PP_comm_inter_time:
            PP_COMM_INTRA_time = PP_comm_intra_time
            PP_COMM_INTRA_volume = PP_comm_intra_volume
            PP_COMM_INTER_time = 0
            PP_COMM_INTER_volume = 0
        else:
            PP_COMM_INTRA_time = 0
            PP_COMM_INTRA_volume = 0
            PP_COMM_INTER_time = PP_comm_inter_time
            PP_COMM_INTER_volume = PP_comm_intra_volume"""

        # Can add MoE and DP here as well
        MoE_comm_time = perf.MoE_overhead_per_layer_bw_pass()
        DP_comm_time = perf.M_f_DP
        # Time line start
        self.timeline.append(
            [
                "BWD_MHA",
                "Compute",
                self.start,
                self.update_time(compute_time_MHA),
                compute_time_MHA,
                "NONE",
                "NONE",
                "NONE",
                0,
            ]
        )

        self.timeline.append(
            [
                "BWD_TP_COMM_INTRA",
                "Comm",
                self.start,
                self.update_time(TP_comm_intra_time / 2),
                TP_comm_intra_time / 2,
                TP_comm_intra_volume if TP_comm_intra_time > 0 else 0,
                "ALLREDUCE",
                "TP",
                "INTRA",
                self.N_TP_INTRA,
            ]
        )

        self.timeline.append(
            [
                "BWD_TP_COMM_INTER",
                "Comm",
                self.start,
                self.update_time(TP_comm_inter_time / 2),
                TP_comm_inter_time / 2,
                TP_comm_inter_volume if TP_comm_inter_time > 0 else 0,
                "ALLREDUCE",
                "TP",
                "INTER",
                self.N_TP_INTER,
            ]
        )

        self.timeline.append(
            [
                "BWD_FFN",
                "Compute",
                self.start,
                self.update_time(compute_time_FFN),
                compute_time_FFN,
                "NONE",
                "NONE",
                "NONE",
                0,
            ]
        )

        self.timeline.append(
            [
                "BWD_TP_COMM_INTRA",
                "Comm",
                self.start,
                self.update_time(TP_comm_intra_time / 2),
                TP_comm_intra_time / 2,
                TP_comm_intra_volume if TP_comm_intra_time > 0 else 0,
                "ALLREDUCE",
                "TP",
                "INTRA",
                self.N_TP_INTRA,
            ]
        )

        self.timeline.append(
            [
                "BWD_TP_COMM_INTER",
                "Comm",
                self.start,
                self.update_time(TP_comm_inter_time / 2),
                TP_comm_inter_time / 2,
                TP_comm_inter_volume if TP_comm_inter_time > 0 else 0,
                "ALLREDUCE",
                "TP",
                "INTER",
                self.N_TP_INTER,
            ]
        )

        """self.timeline.append([
            "BWD_PP_COMM_INTRA",
            "Comm",
            self.start,
            self.update_time(PP_COMM_INTRA_time),
            PP_COMM_INTRA_time,
            PP_COMM_INTRA_volume if PP_COMM_INTRA_time > 0 else 0,
            "P2P",
            "PP",
            "INTRA",
            self.N_PP_INTRA
        ])

        self.timeline.append([
            "BWD_PP_COMM_INTER",
            "Comm",
            self.start,
            self.update_time(PP_COMM_INTER_time),
            PP_COMM_INTER_time,
            PP_COMM_INTER_volume if PP_COMM_INTER_time > 0 else 0,
            "P2P",
            "PP",
            "INTER",
            self.N_PP_INTER
        ])"""

    def pp_overhead_BWD(self, inputs, perf):
        error_volume_per_layer_batch = (self.uB * self.S * self.h * self.nheads) * (
            perf.p["gradient_precision"] / 8
        )

        # / perf.p["layers"])
        PP_comm_intra_volume = error_volume_per_layer_batch
        PP_comm_intra_time = perf.backward_pipeline_parallel_intra()
        PP_comm_inter_time = perf.backward_pipeline_parallel_inter()

        if PP_comm_intra_time >= PP_comm_inter_time:
            PP_COMM_INTRA_time = PP_comm_intra_time
            PP_COMM_INTRA_volume = PP_comm_intra_volume
            PP_COMM_INTER_time = 0
            PP_COMM_INTER_volume = 0
        else:
            PP_COMM_INTRA_time = 0
            PP_COMM_INTRA_volume = 0
            PP_COMM_INTER_time = PP_comm_inter_time
            PP_COMM_INTER_volume = PP_comm_intra_volume

        self.timeline.append(
            [
                "BWD_PP_COMM_INTRA",
                "Comm",
                self.start,
                self.update_time(PP_COMM_INTRA_time),
                PP_COMM_INTRA_time,
                PP_COMM_INTRA_volume if PP_COMM_INTRA_time > 0 else 0,
                "P2P",
                "PP",
                "INTRA",
                self.N_PP_INTRA,
            ]
        )

        self.timeline.append(
            [
                "BWD_PP_COMM_INTER",
                "Comm",
                self.start,
                self.update_time(PP_COMM_INTER_time),
                PP_COMM_INTER_time,
                PP_COMM_INTER_volume if PP_COMM_INTER_time > 0 else 0,
                "P2P",
                "PP",
                "INTER",
                self.N_PP_INTER,
            ]
        )

    def dp_overhead(self, inputs, perf):
        number_of_parameters_per_layer = (
            inputs.parameters["number_of_parameters_per_layer"]
            * (perf.p["gradient_precision"] / 8)
            * perf.p["layers"]
        )

        DP_comm_intra_volume = (number_of_parameters_per_layer) / self.N_DP_INTER

        if self.N_TP_INTRA > 1 or self.N_PP_INTRA > 1:
            nDegree = 0
            if self.N_DP_INTRA > 1:
                nDegree = nDegree + self.N_DP_INTRA
            if self.N_TP_INTRA > 1:
                nDegree = nDegree + self.N_TP_INTRA
            if self.N_PP_INTRA > 1:
                nDegree = nDegree + self.N_PP_INTRA

            if nDegree == 0:
                nDegree = 1

            DP_comm_intra_volume = DP_comm_intra_volume / (nDegree)
            DP_comm_intra_volume = DP_comm_intra_volume * (1.5)

        if self.debug:
            print(f"DP volume : {DP_comm_intra_volume / (1024 * 1024)}")
        DP_comm_intra_time = perf.communication_time_backwards_DP_all_reduce_intra()

        DP_comm_inter_volume = (number_of_parameters_per_layer) / self.N_DP_INTRA

        if self.N_TP_INTER > 1 or self.N_PP_INTER > 1:
            nDegree = 0
            if self.N_DP_INTER > 1:
                nDegree = nDegree + self.N_DP_INTER
            if self.N_TP_INTER > 1:
                nDegree = nDegree + self.N_TP_INTER
            if self.N_PP_INTER > 1:
                nDegree = nDegree + self.N_PP_INTER

            if nDegree == 0:
                nDegree = 1

            DP_comm_inter_volume = DP_comm_inter_volume / (nDegree)
            DP_comm_inter_volume = DP_comm_inter_volume * (1.5)

        DP_comm_inter_time = perf.communication_time_backwards_DP_all_reduce_inter()

        self.timeline.append(
            [
                "BWD_DP_COMM_INTRA",
                "Comm",
                self.start,
                self.update_time(DP_comm_intra_time),
                DP_comm_intra_time,
                DP_comm_intra_volume if DP_comm_intra_time > 0 else 0,
                "ALLREDUCE",
                "DP",
                "INTRA",
                self.N_DP_INTRA,
            ]
        )

        self.timeline.append(
            [
                "BWD_DP_COMM_INTER",
                "Comm",
                self.start,
                self.update_time(DP_comm_inter_time),
                DP_comm_inter_time,
                DP_comm_inter_volume if DP_comm_inter_time > 0 else 0,
                "ALLREDUCE",
                "DP",
                "INTER",
                self.N_DP_INTER,
            ]
        )

    def weight_update(self, inputs, perf):
        compute_weight_update = perf.weight_update_time()
        compute_weight_update = compute_weight_update / (
            self.N_DP * self.N_TP * self.N_PP
        )

        self.timeline.append(
            [
                "Weight_update",
                "Compute",
                self.start,
                self.update_time(compute_weight_update),
                compute_weight_update,
                "NONE",
                "NONE",
                "NONE",
                0,
            ]
        )


# ---------------------------------------------------------------------------
# 4) amped_exec (AMPeD 阶段)
# ---------------------------------------------------------------------------
class amped_exec:
    def __init__(self, training=True) -> None:
        self.debug = DEBUG_PRINT
        self.inputs = []
        self.breakdown = {}
        # self.single_epoc = astrasim_workload()
        self.training = training
        self.timeStamp = ""

        self.main()

    def calc_time(self, inputs, seqLen, flag_gen=False, debug=False, iter=1):
        if not self.training:
            inputs.parameters["context"] = seqLen
            inputs.parameters["tokens_to_train"] = seqLen

            if flag_gen:
                inputs.parameters["summarization_len"] = iter
            else:
                inputs.parameters["summarization_len"] = seqLen

        inputs.dependency_mapping = CalculateFunctionsDependencyMapping()

        for parameter_name in inputs.temp_parameters_to_calculate:
            if parameter_name not in inputs.temp_parameters_dict:
                inputs.calculate_parameter(parameter_name, inputs.temp_parameters_dict)

        inputs.parameters = Parameters(
            inputs, inputs.temp_parameters_dict, inputs.dependency_mapping
        )  # the main property used in other files
        inputs.transformer = inputs.config["neural_network_training_parameters"][
            "lookup_config"
        ]["lookup_table_row"]
        inputs.accelerator = inputs.config["accelerator_architecture_parameters"][
            "lookup_config"
        ]["lookup_table_row"]

        perf_model = PerformanceModel(inputs)

        perLayer_computetime_fwd_pass = (perf_model.compute_time_forward_pass()) / (
            perf_model.p["data_parallel_degree"]
            * perf_model.p["tensor_parallel_degree"]
            * perf_model.p["pipeline_parallel_degree"]
        )
        perLayer_commtime_fwd_pass = perf_model.communication_time_forward_pass()
        perLayer_commtime_pipeline_bubble = (
            (perf_model.p["pipeline_parallel_degree"] - 1)
            * (
                (perf_model.compute_time_forward_pass())
                / (
                    perf_model.p["data_parallel_degree"]
                    * perf_model.p["tensor_parallel_degree"]
                    * perf_model.p["pipeline_parallel_degree"]
                    * perf_model.p["layers"]
                )
                + perf_model.communication_time_forward_pass()
            )
            / perf_model.p["number_of_microbatches_per_minibatch"]
        )

        if self.training:
            perLayer_computetime_bwd_pass = (
                # perf_model.compute_time_forward_pass()
                # + perf_model.compute_time_backward_pass()
                perf_model.compute_time_backward_pass() # new
                + perf_model.weight_update_time()
            ) / (
                perf_model.p["data_parallel_degree"]
                * perf_model.p["tensor_parallel_degree"]
                * perf_model.p["pipeline_parallel_degree"]
            )
            
            perLayer_commtime_bwd_pass = (
                perf_model.communication_time_backwards_DP_all_reduce()
                + perf_model.communication_time_backward_pass()
            )
            # perLayer_commtime_bwd_pass = perf_model.communication_time_backward_pass()  # new

            perLayer_commtime_pipeline_bubble_bwd = (
                (perf_model.p["pipeline_parallel_degree"] - 1)
                * (
                    (perf_model.compute_time_backward_pass())
                    / (
                        perf_model.p["data_parallel_degree"]
                        * perf_model.p["tensor_parallel_degree"]
                        * perf_model.p["pipeline_parallel_degree"]
                        * perf_model.p["layers"]
                    )
                    + perf_model.communication_time_backward_pass()
                )
                / perf_model.p["number_of_microbatches_per_minibatch"]
            )

        if debug:
            print(
                f"Query FLOP : {2 * inputs.parameters['query_MAC_operations'] * inputs.parameters['attention_heads']}"
            )
            print(
                f"Key FLOP : {2 * inputs.parameters['key_MAC_operations'] * inputs.parameters['attention_heads']}"
            )
            print(
                f"Value FLOP : {2 * inputs.parameters['value_MAC_operations'] * inputs.parameters['attention_heads']}"
            )
            print(
                f"MHA FLOP : {2 * inputs.parameters['self_attention_MAC_operations'] * inputs.parameters['attention_heads']}"
            )
            print(
                f"Wout FLOP : {2 * inputs.parameters['attention_sublayer_output_MAC_operations']}"
            )
            print(
                f"FFN FLOP : {2 * inputs.parameters['total_MLP_sublayer_MAC_operations']}"
            )

        computetime = (
            perf_model.p["number_of_batches"]
            * perf_model.p["layers"]
            * perLayer_computetime_fwd_pass
        )
        commtime = (
            perf_model.p["number_of_batches"]
            * perf_model.p["layers"]
            * perLayer_commtime_fwd_pass
        )
        commtime_pipeline_bubble = (
            perf_model.p["number_of_batches"]
            * perf_model.p["layers"]
            * perLayer_commtime_pipeline_bubble
        )

        if self.training:
            computetime += (
                perf_model.p["number_of_batches"]
                * perf_model.p["layers"]
                * perLayer_computetime_bwd_pass
            )
            commtime += (
                perf_model.p["number_of_batches"]
                * perf_model.p["layers"]
                * perLayer_commtime_bwd_pass
            )
            commtime_pipeline_bubble += (
                perf_model.p["number_of_batches"]
                * perf_model.p["layers"]
                * perLayer_commtime_pipeline_bubble_bwd
            )

        # self.single_epoc.layer_update(inputs, perf_model)
        self.perf_model = perf_model

        return [computetime, commtime, commtime_pipeline_bubble]

    def temp_string_training_time_breakdown(
        self, inferenceTime, computeTime, commTime, waitingTime
    ):
        pairs = {
            "Total time to train (s)": inferenceTime,
            "Total time to train (days)": inferenceTime / 3600 / 24,
            "Total time to train (years)": inferenceTime / 3600 / 24 / 365,
            "Computation time forward pass (s)": computeTime,
            "Computation time backward pass (s)": 0,
            "Computation time weight updates (s)": 0,
            "Total computation time (s)": computeTime,
            "Communication time forward pass: tensor parallelism intra node (s)": 0,
            "Communication time forward pass: tensor parallelism inter node (s)": 0,
            "Communication time forward pass: pipeline parallelism (s)": 0,
            "Communication time forward pass: Zero-DP (ad-hoc model gathering) (s)": 0,
            "Total communication time forward pass (s)": commTime,
            "Communication time backward pass: tensor parallelism intra node (s)": 0,
            "Communication time backward pass: tensor parallelism inter node (s)": 0,
            "Communication time backward pass: pipeline parallelism (s)": 0,
            "Communication time backward pass: Zero-DP (ad-hoc model gathering) (s)": 0,
            "Total communication time backward pass (s)": 0,
            "Communication time: all-reduce gradients for DP intra node (s)": 0,
            "Communication time: all-reduce gradients for DP inter node (s)": 0,
            "Communication time: all-reduce gradients for DP total (s)": 0,
            "Total communication time (s)": commTime,
            "Waiting Time due to pipeline bubbles (s)": waitingTime,
            "total MACs": 0,
            "TFLOPS": 0,
            "total TFLOPS/sec": 0,
            "TFLOP/sec/GPU": 0,
            "TFLOP/sec/GPU (peak)": 0,
            "MoE FW pass communication overhead time (s)": 0,
            "MoE BW pass communication overhead time (s)": 0,
            "MoE Gating network: compute overhead time (s)": 0,
            "Total MoE communication overhead time": 0,
        }

        longest_label_length = len(max(pairs.keys(), key=len))
        self.breakdown = pairs
        breakdown = "TRAINING TIME BREAKDOWN\n\n"
        return breakdown + "\n".join(
            [f"{label:-<{longest_label_length}} {val}" for label, val in pairs.items()]
        )

    def training_string_training_time_breakdown(self):
        pairs = {
            "Total time to train (s)": self.perf_model.total_time_to_train(),
            "Total time to train (days)": self.perf_model.total_time_to_train_days(),
            "Total time to train (years)": self.perf_model.total_time_to_train_years(),
            "Computation time forward pass (s)": self.perf_model.total_computation_time_forward_pass(),
            "Computation time backward pass (s)": self.perf_model.total_computation_time_backward_pass(),
            "Computation time weight updates (s)": self.perf_model.total_computation_time_weight_updates(),
            "Total computation time (s)": self.perf_model.total_computation_time(),
            "Communication time forward pass: tensor parallelism intra node (s)": self.perf_model.total_forward_tensor_model_intra(),
            "Communication time forward pass: tensor parallelism inter node (s)": self.perf_model.total_forward_tensor_parallel_inter(),
            "Communication time forward pass: pipeline parallelism (s)": self.perf_model.total_forward_pipeline_parallelism(),
            "Communication time forward pass: Zero-DP (ad-hoc model gathering) (s)": self.perf_model.total_forward_zero_DP(),
            "Total communication time forward pass (s)": self.perf_model.total_communication_time_forward_pass(),
            "Communication time backward pass: tensor parallelism intra node (s)": self.perf_model.total_backward_tensor_model_intra(),
            "Communication time backward pass: tensor parallelism inter node (s)": self.perf_model.total_backward_tensor_parallel_inter(),
            "Communication time backward pass: pipeline parallelism (s)": self.perf_model.total_backward_pipeline_parallelism(),
            "Communication time backward pass: Zero-DP (ad-hoc model gathering) (s)": self.perf_model.total_backward_zero_DP(),
            "Total communication time backward pass (s)": self.perf_model.total_communication_time_backward_pass(),
            "Communication time: all-reduce gradients for DP intra node (s)": self.perf_model.total_all_reduce_gradients_for_DP_intra(),
            "Communication time: all-reduce gradients for DP inter node (s)": self.perf_model.total_all_reduce_gradients_for_inter(),
            "Communication time: all-reduce gradients for DP total (s)": self.perf_model.total_all_reduce_gradients_for_DP(),
            "Total communication time (s)": self.perf_model.total_communication_time(),
            "Waiting Time due to pipeline bubbles (s)": self.perf_model.total_waiting_time_due_to_pipeline_bubbles(),
            "total MACs": self.perf_model.total_MACs(),
            "TFLOPS": self.perf_model.total_TFLOPS(),
            "total TFLOPS/sec": self.perf_model.total_TFLOPS_per_second(),
            "TFLOP/sec/GPU": self.perf_model.total_TFLOPS_per_second_per_gpu(),
            "TFLOP/sec/GPU (peak)": self.perf_model.total_TFLOPS_per_second_per_gpu_peak(),
            "MoE FW pass communication overhead time (s)": self.perf_model.total_MoE_overhead_fw_pass(),
            "MoE BW pass communication overhead time (s)": self.perf_model.total_MoE_overhead_bw_pass(),
            "MoE Gating network: compute overhead time (s)": self.perf_model.total_MoE_gating_network_compute_overhead(),
            "Total MoE communication overhead time": self.perf_model.total_MoE_overhead(),
        }

        longest_label_length = len(max(pairs.keys(), key=len))

        self.breakdown = pairs
        breakdown = "TRAINING TIME BREAKDOWN\n\n"
        return breakdown + "\n".join(
            [f"{label:-<{longest_label_length}} {val}" for label, val in pairs.items()]
        )

    def main(self):
        # Amped inference script
        inputs = Inputs()

        # self.resultDir = f"DP_{}_{}_TP_{}_{}_PP_{}_{}"

        if not self.training:
            inputs.parameters["batch_size"] = 1
            inputs.parameters["number_of_microbatches_per_minibatch"] = 1

        ############################################################################

        # Prompt/ Summarization stage
        if not self.training:
            [computetime_fwd_pass, commtime_fwd_pass, commtime_pipeline_bubble] = (
                self.calc_time(inputs, PROMPT_LEN, False, False)
            )
        else:
            [computetime_fwd_pass, commtime_fwd_pass, commtime_pipeline_bubble] = (
                self.calc_time(inputs, inputs.parameters["context"], False, False)
            )

        summary = "FULL CONFIGURATION\n\n" + inputs.parameters.to_string_structured()
        self.temp_save_as("config_summary.txt", summary, saveDir=OUTPUT_PATH)

        if not self.training:
            file = open(f"{OUTPUT_PATH}/AmpedInference.txt", "w")
        else:
            file = open(f"{OUTPUT_PATH}/AmpedTraining.txt", "w")

        if self.debug:
            print(f"Stage : 0")
            print(f"Compute time (s) : {computetime_fwd_pass}")
            print(f"Communication time (s) : {commtime_fwd_pass}")
            print(f"Pipeline bubble time (s) : {commtime_pipeline_bubble}\n\n")

        tmp = f"Stage,0,Compute time (s),{computetime_fwd_pass},Communication time (s), {commtime_fwd_pass},Pipeline bubble time (s),{commtime_pipeline_bubble}\n"
        file.write(tmp)

        overall_computetime = computetime_fwd_pass
        overall_commtime = commtime_fwd_pass
        overall_pipeline_bubble = commtime_pipeline_bubble

        if not self.training:
            # Generation stage
            for i in range(PROMPT_LEN + 1, PROMPT_LEN + GENERATION_LEN + 1):
                print(f"Stage : {i - PROMPT_LEN}")

                [computetime_fwd_pass, commtime_fwd_pass, commtime_pipeline_bubble] = (
                    self.calc_time(inputs, 1, True, False, i)
                )

                print(f"\nCompute time (s) : {computetime_fwd_pass}")
                print(f"Communication time (s) : {commtime_fwd_pass}")
                print(f"Pipeline bubble time (s) : {commtime_pipeline_bubble}")

                tmp = f"Stage,{i - PROMPT_LEN},Compute time (s),{computetime_fwd_pass},Communication time (s), {commtime_fwd_pass},Pipeline bubble time (s),{commtime_pipeline_bubble}\n"
                file.write(tmp)

                overall_computetime += computetime_fwd_pass
                overall_commtime += commtime_fwd_pass
                overall_pipeline_bubble += commtime_pipeline_bubble

        if self.debug:
            print(f"Overall timings")
            print(f"Compute time(s) : {overall_computetime}")
            print(f"Communication time (s) : {overall_commtime}")
            print(f"Pipeline bubble time (s) : {overall_pipeline_bubble}")

        tmp = f"Stage,Z,Compute time (s),{overall_computetime},Communication time (s), {overall_commtime},Pipeline bubble time (s),{overall_pipeline_bubble}\n"
        file.write(tmp)

        ############################################################################
        # if inputs.commandline_args.GEMM:
        #    save_GEMM_breakdown(inputs, inputs.commandline_args.compute_graph)
        self.inputs = inputs

        if not self.training:
            self.temp_save_as(
                "training_time_breakdown.txt",
                self.temp_string_training_time_breakdown(
                    overall_computetime + overall_commtime + overall_pipeline_bubble,
                    overall_computetime,
                    overall_commtime,
                    overall_pipeline_bubble,
                ),
                saveDir=OUTPUT_PATH,
            )
        else:
            self.temp_save_as(
                "training_time_breakdown.txt",
                self.training_string_training_time_breakdown(),
                saveDir=OUTPUT_PATH,
            )

    def temp_save_as(
        self, filename: str, content: str, encoding: str | None = None, saveDir=""
    ):
        if saveDir == "":
            saveDir = "output_files"

        if not os.path.isdir(saveDir):
            os.mkdir(saveDir)
        self.timeStamp = time_prefix()
        open(f"{saveDir}/{self.timeStamp}{filename}", "w", encoding=encoding).write(
            content
        )


# ---------------------------------------------------------------------------
# 5) mat_dims_ampedToDF
# ---------------------------------------------------------------------------
class mat_dims_ampedToDF:
    def __init__(self, amped, training=True) -> None:
        self.dims = {}
        self.debug = DEBUG_PRINT
        self.training = training
        if self.debug:
            print("Starting mat dims amped to DF script...")
        self.amped = amped
        self.main(self.amped.inputs)
        self.SUB_VOCAB = 1024

    def mmm_breakup(self, B, D, S, h, nheads, h_MLP1, h_MLP2, N_DP, N_PP):
        mmm = {}
        dims = {}
        # deepflow_outputs = {}
        if not self.training:
            numlevels = 6 * (GENERATION_LEN + 1)
        else:
            numlevels = 6

        levels = ["X.W=KQV", "Q.K=R", "R.V=Z", "Z.W=Y", "Y.WL1=O1", "O1.WL2=O2"]
        # print("matrix dimensions accounting for all heads & batched dimension")
        if not self.training:
            S = PROMPT_LEN
        else:
            # Do nothing since S is already context
            pass

        # Summarization/Training
        dims[0] = [
            int(3 * B * S / N_DP / N_PP),
            D,
            h * nheads,
            "CR",
            levels[0],
        ]  # factor 3 due to K+Q+V  Columnwise Q = [Q1, Q2] etc
        dims[1] = [
            int(B * S / N_DP / N_PP),
            h * nheads,
            S,
            "CR",
            levels[1],
        ]  # This seem off !!!
        dims[2] = [int(B * S / N_DP / N_PP), S, h * nheads, "CR", levels[2]]
        dims[3] = [int(B * S / N_DP / N_PP), D, D, "CR", levels[3]]
        dims[4] = [
            int(B * S / N_DP / N_PP),
            D,
            h_MLP1,
            "RC",
            levels[4],
        ]  # Row wise split, WL1 = [WL1 ; WL2]
        dims[5] = [int(B * S / N_DP / N_PP), h_MLP1, h_MLP2, "RC", levels[5]]

        if not self.training:
            # Generation
            S = 1
            for i in range(1, GENERATION_LEN):
                dims[0 + (6 * i)] = [
                    int(3 * B * S / N_DP / N_PP),
                    D,
                    h * nheads,
                    "CR",
                    levels[0],
                ]  # factor 3 due to K+Q+V
                dims[1 + (6 * i)] = [
                    S,
                    int(B * S * (i + PROMPT_LEN) / N_DP / N_PP),
                    h * nheads,
                    "CR",
                    levels[1],
                ]
                dims[2 + (6 * i)] = [
                    S,
                    h * nheads,
                    int(B * S * (i + PROMPT_LEN) / N_DP / N_PP),
                    "CR",
                    levels[2],
                ]
                dims[3 + (6 * i)] = [int(B * S / N_DP / N_PP), D, D, "CR", levels[3]]
                dims[4 + (6 * i)] = [
                    int(B * S / N_DP / N_PP),
                    D,
                    h_MLP1,
                    "RC",
                    levels[4],
                ]
                dims[5 + (6 * i)] = [
                    int(B * S / N_DP / N_PP),
                    h_MLP1,
                    h_MLP2,
                    "RC",
                    levels[5],
                ]

        if self.debug:
            print("levels:", levels)
            print("writting the matrix dimensions ...")

        file = open(f"{OUTPUT_PATH}/{self.amped.timeStamp}mat_dims_amped.txt", "w")
        # file.write('#'+str(levels)+'\n')
        for i in range(len(dims)):
            mmm[i] = []
            if self.debug:
                print(f"Gathered {dims[i]}")
            mmm[i].append(dims[i])
            # deepflow_outputs[i] = []
            # print(mmm[i])
            tmp = str(mmm[i]).replace("[", "").replace("]", "").replace(",", " ")
            # print(tmp)
            file.write(tmp + "\n")

        self.dims = dims

    def main(self, inputs):
        print("**** Creating GEMMs from AMPeD ****\n")

        B = int(inputs.parameters["batch_size"])
        D = int(inputs.parameters["dimensionality"])
        S = int(inputs.parameters["context"])
        sum_len = int(inputs.parameters["summarization_len"])
        h = int(inputs.parameters["hidden_layer_dimension_for_attention_sublayers"])
        nheads = int(inputs.parameters["attention_heads"])
        h_MLP1 = int(inputs.parameters["hidden_layer_dimension_MLP_1"])
        h_MLP2 = int(inputs.parameters["hidden_layer_dimension_MLP_2"])
        N_DP = int(inputs.parameters["data_parallel_degree"])
        N_PP = int(inputs.parameters["pipeline_parallel_degree"])
        return self.mmm_breakup(B, D, S, h, nheads, h_MLP1, h_MLP2, N_DP, N_PP)


# ---------------------------------------------------------------------------
# 6) update_configs 及辅助函数
# ---------------------------------------------------------------------------
def _resolve_config_path(p):
    # 优先：绝对路径已存在
    if os.path.isabs(p) and os.path.exists(p):
        return p
    # 其次：项目内 amped_backups/ 下
    repo_root = os.path.dirname(__file__)
    cand = os.path.join(repo_root, "amped_backups", p)
    if os.path.exists(cand):
        return cand
    # 再次：当前工作目录
    cand = os.path.join(os.getcwd(), p)
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"Config not found: {p}")

def _dump_pretty(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data, f,
            ensure_ascii=False,
            indent=4,
            separators=(", ", ": "),
            sort_keys=False
        )
        f.write("\n")  # 末尾换行，方便 diff
        
def update_configs(cSim, config_file):
    # 例: DP_1_1_TP_4_1_PP_1_1_4
    _, DP_intra, DP_inter, _, TP_intra, TP_inter, _, PP_intra, PP_inter, intraGPUs = cSim.split("_")

    # 转成 int，避免写成字符串
    DP_intra  = int(DP_intra);  DP_inter  = int(DP_inter)
    TP_intra  = int(TP_intra);  TP_inter  = int(TP_inter)
    PP_intra  = int(PP_intra);  PP_inter  = int(PP_inter)
    intraGPUs = int(intraGPUs)

    # 解析真实可用的配置路径
    src_cfg_path = _resolve_config_path(config_file)

    # 读取、修改
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
    sp["number_of_accelerators_per_node"]["value"]  = intraGPUs
    sp["number_of_network_cards_per_node"]["value"] = intraGPUs


    sp = data["system_architecture_parameters"]["parameters"]
    for k in (
        "effective_perf_perc_K_Q_V",
        "effective_perf_perc_attention",
        "effective_perf_perc_output",
        "effective_perf_perc_MLP",
    ):
        sp[k]["value"] = 0.7
        sp[k]["calculated"] = False  # 关键：禁止 AMPeD 重算覆盖
    print("[EFF] set to 0.7 (calculated=False)")

    # 1) 写回“原始文件”的真实路径（不是传入的原字符串）
    _dump_pretty(src_cfg_path, data)

    # 2) 同时覆盖 AMPeD 实际读取的 amped_backups/config.json
    repo_root = os.path.dirname(__file__)
    amped_cfg = os.path.join(repo_root, "amped_backups", "config.json")
    _dump_pretty(amped_cfg, data)


# ---------------------------------------------------------------------------
# 7) 入口 if __name__ == "__main__"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path or filename of the AMPeD config JSON to update (also mirrored into amped_backups/config.json).")
    args = parser.parse_args()

    itrers = [
        "DP_1_4_TP_4_1_PP_1_1_4",
        "DP_1_1_TP_4_1_PP_1_1_4",
        "DP_1_1_TP_1_1_PP_4_1_4",
        "DP_2_1_TP_2_1_PP_1_1_4",
        "DP_2_1_TP_1_1_PP_2_1_4",
        "DP_1_1_TP_2_1_PP_2_1_4",
    ]

    itrers = [
        "DP_1_4_TP_2_1_PP_1_2_2", # 16
        "DP_1_2_TP_2_2_PP_1_2_2", # 16
        "DP_1_2_TP_4_1_PP_1_2_4", # 16
        "DP_1_4_TP_2_1_PP_2_1_4", # 16
        "DP_2_1_TP_2_1_PP_2_1_8", # 16
        "DP_1_2_TP_4_1_PP_1_8_4", # 64
        "DP_1_4_TP_4_1_PP_1_4_4", # 64
    ]

    itrers = [
        "DP_1_64_TP_8_1_PP_1_2_8", # 1024
        "DP_1_32_TP_8_2_PP_1_2_8", # 1024
        "DP_1_32_TP_16_1_PP_1_2_16", # 1024
        "DP_1_64_TP_8_1_PP_2_1_16", # 1024
        "DP_1_32_TP_8_1_PP_1_32_8", # 8192
        "DP_1_32_TP_16_1_PP_1_16_8", # 8192
        "DP_1_64_TP_16_1_PP_1_8_8", # 8192
    ]

    itrers = [
        "DP_1_64_TP_8_1_PP_1_2_8", # 1024
        "DP_1_32_TP_8_2_PP_1_2_8", # 1024
        "DP_1_32_TP_8_1_PP_1_4_8", # 1024
        "DP_1_64_TP_4_1_PP_2_2_8", # 1024
        "DP_2_128_TP_2_1_PP_2_1_8", # 1024
        "DP_1_32_TP_2_1_PP_4_4_8", # 1024 calculon no overlap
        "DP_2_128_TP_4_1_PP_1_1_8", # 1024 calculon
    ]

    itrers = [
        #"DP_1_16_TP_8_1_PP_1_8_8", # 1024 GPT3-175B
        #"DP_1_6_TP_8_1_PP_1_64_8", # 3072 Megatron-1T
        #"DP_1_9_TP_8_1_PP_1_35_8", # 2520 Megatron-530B
        #"DP_1_15_TP_8_1_PP_1_16_8", # 1920 Megatron-310B
        "DP_1_2_TP_8_1_PP_1_8_8", # 1920 DeepNet/OPUS-100
    ]

    # itrers = [
    #     "DP_1_4_TP_8_1_PP_1_8_8", # 256 GPT3-175
    #     "DP_1_16_TP_8_1_PP_1_8_8", # 1024 GPT3-175
    #     "DP_1_64_TP_8_1_PP_1_8_8", # 4096 GPT3-175
    #     "DP_1_256_TP_8_1_PP_1_8_8", # 16384 GPT3-175
    #     "DP_1_1024_TP_8_1_PP_1_8_8", # 65536 GPT3-175
    # ]

    # itrers = [
    #     "DP_1_1_TP_8_1_PP_1_8_8", # 128 GPT3-175
    #     "DP_1_2_TP_8_1_PP_1_8_8", # 128 GPT3-175
    #     "DP_1_4_TP_8_1_PP_1_8_8", # 256 GPT3-175
    #     "DP_1_8_TP_8_1_PP_1_8_8", # 512 GPT3-175
    #     "DP_1_16_TP_8_1_PP_1_8_8", # 1024 GPT3-175
    #     "DP_1_32_TP_8_1_PP_1_8_8", # 2048 GPT3-175
    #     "DP_1_64_TP_8_1_PP_1_8_8", # 4096 GPT3-175
    # ]

    # itrers = [ # self-designed test
    #     "DP_1_8_TP_8_1_PP_1_4_8",   # 256
    #     "DP_1_16_TP_8_1_PP_1_8_8",   # 1024
    #     "DP_1_32_TP_8_1_PP_1_16_8",  # 4096
    #     "DP_1_64_TP_8_1_PP_1_32_8",  # 16384
    #     "DP_1_128_TP_8_1_PP_1_64_8",  # 65536
    # ]

    # itrers = [ # calculon_new
    #     "DP_1_64_TP_2_1_PP_4_2_8",   # 1024
    #     "DP_1_128_TP_2_1_PP_4_4_8",  # 4096
    #     "DP_1_256_TP_4_1_PP_2_8_8",  # 16384
    # ]

    itrers = [ # self-designed test
        "DP_1_16_TP_8_1_PP_1_2_8",   # 256
        "DP_1_64_TP_8_1_PP_1_2_8",  # 1024
        "DP_1_256_TP_8_1_PP_1_2_8",  # 4096
        "DP_1_1024_TP_8_1_PP_1_2_8",  # 16384
        "DP_1_4096_TP_8_1_PP_1_2_8",  # 65536
        "DP_1_16384_TP_8_1_PP_1_2_8",  # 262144
    ]

    itrers = [ # GPT3 175B
        "DP_1_16_TP_8_1_PP_1_8_8",   # 1024
    ]

    itrers = [ # megatron 310B
        "DP_1_15_TP_8_1_PP_1_16_8",   # 1920
    ]

    itrers = [ # megatron 530B
        "DP_1_9_TP_8_1_PP_1_35_8",   # 2520
    ]

    itrers = [ # megatron 1TB
        "DP_1_6_TP_8_1_PP_1_64_8",   # 3072
    ]

    # itrers = [ # self-designed test
    #     "DP_1_16_TP_8_1_PP_1_2_8",   # 256
    #     "DP_1_16_TP_8_4_PP_1_2_8",  # 1024
    #     "DP_1_16_TP_8_16_PP_1_2_8",  # 4096
    #     "DP_1_16_TP_8_64_PP_1_2_8",  # 16384
    #     "DP_1_16_TP_8_256_PP_1_2_8",  # 65536
    #     "DP_1_16_TP_8_1024_PP_1_2_8",  # 262144
    # ]

    # itrers = [ # self-designed test
    #     "DP_1_32_TP_8_1_PP_1_1_8",  # 256
    #     "DP_1_32_TP_8_1_PP_1_4_8",  # 1024
    #     "DP_1_32_TP_8_1_PP_1_16_8",  # 4096
    #     "DP_1_32_TP_8_1_PP_1_64_8",  # 16384
    #     "DP_1_32_TP_8_1_PP_1_256_8",  # 65536
    #     "DP_1_32_TP_8_1_PP_1_1024_8",  # 262144
    # ]

    # itrers = []
    # # TODO: Need to write custom parallelism for simulations
    # for numGPUs in range(8, 64, 8):
    #     _halves = int(numGPUs / 2)
    #     _quarters = int(numGPUs / 4)

    #     itrers.append(f"DP_{_quarters}_1_TP_{_quarters}_1_PP_{_quarters}_1_{numGPUs}")

    #     itrers.append(f"DP_{numGPUs}_1_TP_1_1_PP_1_1_{numGPUs}")
    #     itrers.append(f"DP_1_1_TP_{numGPUs}_1_PP_1_1_{numGPUs}")
    #     itrers.append(f"DP_1_1_TP_1_1_PP_{numGPUs}_1_{numGPUs}")

    #     itrers.append(f"DP_{_halves}_1_TP_{_quarters}_1_PP_1_1_{numGPUs}")
    #     itrers.append(f"DP_{_quarters}_1_TP_{_halves}_1_PP_1_1_{numGPUs}")

    #     itrers.append(f"DP_{_halves}_1_TP_1_1_PP_{_quarters}_1_{numGPUs}")
    #     itrers.append(f"DP_{_quarters}_1_TP_1_1_PP_{_halves}_1_{numGPUs}")

    #     itrers.append(f"DP_1_1_TP_{_halves}_1_PP_{_quarters}_1_{numGPUs}")
    #     itrers.append(f"DP_1_1_TP_{_quarters}_1_PP_{_halves}_1_{numGPUs}")

    totalSims = len(itrers)
    _cnt = 0
    for cSim in itrers:
        _cnt += 1

        print(f"\033[1m\033[95mProcessing ({_cnt}/{totalSims}): {cSim}\033[0m")
        # update_configs(cSim, "amped_backups/config.json")
        update_configs(cSim, args.config)

        RESULT_DIR = cSim
        # OUTPUT_PATH = f"/imec/scratch/dtpatha/patel23/amped_deepflow/amped_deepflow/forNetwork/{RESULT_DIR}"
        OUTPUT_PATH = f"/home/fd420/LLM_analytical_tools/amped_deepflow/output_files/{RESULT_DIR}"

        # First call amped
        print(f"\033[1m\033[92m\tExecuting: AMPeD\033[0m")
        AMPeD = amped_exec(training=True)

        # 紧跟在 AMPeD = amped_exec(training=True) 后面，或在 amped_exec.main() 里 Inputs() 之后
        p = AMPeD.inputs.parameters
        print(">>> total_gpus=", int(p["total_number_of_accelerators"]),  # 👈 这里转 int
            " | DP=", p["data_parallel_degree"], "(intra:", p["intra_node_data_parallel_degree"],
            "inter:", p["inter_node_data_parallel_degree"], ")",
            " | TP=", p["tensor_parallel_degree"], "(intra:", p["intra_node_tensor_parallel_degree"],
            "inter:", p["inter_node_tensor_parallel_degree"], ")",
            " | PP=", p["pipeline_parallel_degree"], "(intra:", p["intra_node_pipeline_parallel_degree"],
            "inter:", p["inter_node_pipeline_parallel_degree"], ")")

        # WIP: To represent workload an astra-sim input file
        # tAstra = astrasim_workload(amped=AMPeD, result_dir=OUTPUT_PATH)

        # Save the class to read it quick for plotting
        # picklTag = f"{OUTPUT_PATH}/{AMPeD.timeStamp}_"
        # with open(f"{picklTag}_pickl_AMPeD.pkl", "wb") as outp:
        #    pickle.dump(AMPeD, outp, pickle.HIGHEST_PROTOCOL)

        # then, mat_dims_ampedToDF
        print(f"\033[1m\033[92m\tExecuting: Extracting mapping from AMPeD\033[0m")
        mat_dims = mat_dims_ampedToDF(amped=AMPeD, training=True)

        # Save the class to read it quick for plotting
        # with open(f"{picklTag}_pickl_mat_dims.pkl", "wb") as outp:
        #    pickle.dump(mat_dims, outp, pickle.HIGHEST_PROTOCOL)

        # then, run.sh / deepflow exec / perf.py
        print(f"\033[1m\033[92m\tExecuting: DeepFlow\033[0m")
        DeepFlow = deepflow_exec(AMPeD, mat_dims.dims)

        # Save the class to read it quick for plotting
        # with open(f"{picklTag}_pickl_DeepFlow.pkl", "wb") as outp:
        #    pickle.dump(DeepFlow, outp, pickle.HIGHEST_PROTOCOL)

        # then, cal_time.py
        print(f"\033[1m\033[92m\tExecuting: Post process - AMPeD + DeepFlow\033[0m")
        cal_time(AMPeD, DeepFlow.deepflow_outputs)
        # Time domain list
        print(f"\033[1m\033[92m\tExecuting: Timelines\033[0m")
        time_domain(AMPeD, DeepFlow.deepflow_outputs)
        # break
