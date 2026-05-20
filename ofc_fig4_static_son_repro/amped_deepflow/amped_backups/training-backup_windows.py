import sys
import os
from math import ceil
sys.path.insert(1, '../AMPeD')
sys.path.insert(1, '../DeepFlow')

# imports for deepflow
import perf
import config
from perf import TimeCalculation
import click

# imports fo mat_dims_ampedToDF AND cal_time
import numpy as np
import argparse
import pandas as pd

# Imports for amped
from amped import save_GEMM_breakdown
from amped.common import time_prefix
from amped.performance_model import PerformanceModel
# from amped.inputs import Inputs, CalculateFunctionsDependencyMapping, Parameters
from amped_backups.inputs import Inputs, CalculateFunctionsDependencyMapping, Parameters

PROMPT_LEN = 2048
GENERATION_LEN = 0
DEBUG_PRINT = True

DEEPFLOW_CONFIG_PATH="/imec/scratch/dtpatha/patel23/amped_deepflow/amped_deepflow/deepflow_configs"
OUTPUT_PATH="/imec/scratch/dtpatha/patel23/amped_deepflow/amped_deepflow/output_files"


class cal_time():
    def __init__(self, amped, deepFlow_outputs) -> None:

        self.amped = amped
        self.inputs = self.amped.inputs
        self.breakdown = self.amped.breakdown
        self.deepflow_outputs = deepFlow_outputs
        self.debug = DEBUG_PRINT
        self.main(self.inputs)
        self.return_outputs = []

    def time_from_GEMM(self):

        # Directory containing the files
        t_elapsed =0.0
        t_reduction_elapsed = 0.0

        for gemmTimes in self.deepflow_outputs:
            gemmTime = gemmTimes[0]
            reductionTime = gemmTimes[1]

            t_elapsed += gemmTime
            t_reduction_elapsed += reductionTime

        t_elapsed = t_elapsed*2.0 #FW pass + BW pass (weight update time is added seperately)
        t_reduction_elapsed = t_reduction_elapsed * 2.0
        # print("********** FW-BW pass time ************", t_elapsed)
        return [t_elapsed, t_reduction_elapsed]

    def main(self, inputs):
        print("**** Computing GEMM timings using DeepFlow *****")
        print(self.breakdown)

        [t_FW_BW, t_REDUCTION] = self.time_from_GEMM()
        if self.debug:
            print("t_FW_BW:", t_FW_BW)
        # nbatch = int(inputs.parameters["tokens_to_train"])/(int(inputs.parameters["context"])\
        #                                    *int(inputs.parameters["batch_size"]))

        nbatch = 1
        time = int(inputs.parameters["layers"])*nbatch*t_FW_BW + float(self.breakdown["Total communication time forward pass (s)"])\
            +float(self.breakdown["Total communication time backward pass (s)"])\
            +float(self.breakdown["Computation time weight updates (s)"]) \
            +float(self.breakdown["Waiting Time due to pipeline bubbles (s)"])

        print("\n\nTotal time after DeepFlow computation:\n\t", time)

class deepflow_exec():
    def __init__(self, amped, dims, training = True) -> None:
        self.debug = DEBUG_PRINT
        self.amped = amped
        self.inputs = self.amped.inputs
        self.TP_DEGREE = self.inputs.parameters["tensor_parallel_degree"]
        self.CONFIG_DIR=DEEPFLOW_CONFIG_PATH
        self.OUTDIR=OUTPUT_PATH

        self.training = training

        if self.debug:
            print(f"TP degree is set to : {self.TP_DEGREE}")
            print(f"Config: {self.CONFIG_DIR}")
            print(f"Output dir: {self.OUTDIR}")
        self.deepflow_outputs = []

        self.main(dims)

    def deepflow_function(self, exp_config, exp_dir, debug, m, n, k, t, kp1, kp2, gemm, batch_size=2048, hidden_dim=19968, seq_len=20, vocab_size=800000, num_layer=2, dp=None, lp=None, lev=None, args_input=False):
        exp_path = os.path.expandvars(os.path.expanduser(exp_config))
        exp_config = config.parse_config(exp_path)
        #output_file = exp_dir + "/%s_summary_l%s_m%s_n%s_k%s.txt" %(self.amped.timeStamp, lev, m, n, k) ##Output dir should be created manually

        TC = TimeCalculation(exp_config)
        if args_input:
            TC.updateParams(debug, m, n, k, t, kp1, kp2, dp, lp, gemm, 
                        batch_size, hidden_dim, seq_len, vocab_size, num_layer)

        # Report GEMM time on fw path
        if TC.validating_GEMM:

            if kp1 == 1 and kp2 ==1: #no parallelism
                t_gemm_time = TC.getCf(m, k, n)
                gemm_time = [t_gemm_time[0], 0]
            elif t == 'CR':
                gemm_time = TC.getDistGEMM_f_kp1(m, k, n, kp1, "Cf_CR")
            elif t == 'RC':
                gemm_time = TC.getDistGEMM_f_kp2(m, k, n, kp1, kp2, "Cf_RC")
            else:
                print("Incorrect parallelism type, CR: Column-Row, RC: Row-Column")
                sys.exit()

            #with open(output_file, "w") as f:
            #    f.write("GEMM Time: {}\n".format(gemm_time[0]))
            #    f.write("Reduction Time: {}\n".format(gemm_time[1]))
            return [gemm_time[0], gemm_time[1]]

    def main(self, dims):

        mha_GEMMtime = 0
        mha_reduction = 0
        ffn_GEMMtime = 0
        ffn_reduction = 0

        output_file = f"{self.OUTDIR}/{self.amped.timeStamp}summary_deepflow.txt" ##Output dir should be created manually
        with open(output_file, "w") as f:
            f.write(f"Layer, M, N, K, t, GEMM time, reduction time\n")

        for i in range(len(dims)):
            # if self.debug:
            #    print(f"Evaluating GEMM of M:{dims[i][0]} N:{dims[i][1]} K:{dims[i][2]}")
            if not self.training:
                if dims[i][0] < self.TP_DEGREE:
                    temp_outputs = self.deepflow_function(
                        exp_config=f"{self.CONFIG_DIR}/v100.yaml",
                        exp_dir=f"{self.OUTDIR}/LLM",
                        debug=True,
                        m=dims[i][0],
                        n=dims[i][1],
                        k=dims[i][2],
                        t="RC",
                        kp1=1,
                        kp2=self.TP_DEGREE,
                        gemm=True,
                        lev=dims[i][4],
                    )
                else:
                    temp_outputs = self.deepflow_function(
                        exp_config=f"{self.CONFIG_DIR}/v100.yaml",
                        exp_dir=f"{self.OUTDIR}/LLM",
                        debug=True,
                        m=dims[i][0],
                        n=dims[i][1],
                        k=dims[i][2],
                        t="RC",
                        kp1=self.TP_DEGREE,
                        kp2=1,
                        gemm=True,
                        lev=dims[i][4],
                    )
            else:
                if dims[i][3] == 'CR':
                    temp_outputs = self.deepflow_function(
                        exp_config=f"{self.CONFIG_DIR}/v100.yaml",
                        exp_dir=f"{self.OUTDIR}/LLM",
                        debug=True,
                        m=dims[i][0],
                        n=dims[i][1],
                        k=dims[i][2],
                        t="CR",
                        kp1=self.TP_DEGREE,
                        kp2=1,
                        gemm=True,
                        lev=dims[i][4]
                    )
                else:
                    temp_outputs = self.deepflow_function(
                        exp_config=f"{self.CONFIG_DIR}/v100.yaml",
                        exp_dir=f"{self.OUTDIR}/LLM",
                        debug=True,
                        m=dims[i][0],
                        n=dims[i][1],
                        k=dims[i][2],
                        t="RC",
                        kp1=self.TP_DEGREE,
                        kp2=self.TP_DEGREE,
                        gemm=True,
                        lev=dims[i][4],
                    )
                
            temp_outputs.append(dims[i][4])
            temp_outputs.append(dims[i][0])
            temp_outputs.append(dims[i][1])
            temp_outputs.append(dims[i][2])

            if dims[i][4] == "X.W=KQV" or dims[i][4] == "Q.K=R" or dims[i][4] == "R.V=Z" or dims[i][4] == "Z.W=Y":
                #print(f"DEBUG: MHA {dims[i][4]}\tGEMM time : {temp_outputs[0]}\tReduction time: {temp_outputs[1]}")
                mha_GEMMtime += float(temp_outputs[0])
                mha_reduction += float(temp_outputs[1])
            elif dims[i][4] == "Y.WL1=O1" or dims[i][4] == "O1.WL2=O2":
                #print(f"DEBUG: FFN {dims[i][4]}\tGEMM time : {temp_outputs[0]}\tReduction time: {temp_outputs[1]}")
                ffn_GEMMtime += float(temp_outputs[0])
                ffn_reduction += float(temp_outputs[1])

            # print(f"\n{temp_outputs}\n")

            self.deepflow_outputs.append(temp_outputs)

            with open(output_file, "a") as f:
                f.write(f"{dims[i][4]}, {dims[i][0]}, {dims[i][1]}, {dims[i][2]}, {dims[i][3]}, {temp_outputs[0]}, {temp_outputs[1]}\n")

            if self.debug:
                print(f"Dimension:{dims[i]}\n\tGEMM time: {self.deepflow_outputs[i][0]}\tReduction time: {self.deepflow_outputs[i][1]}\n")

            if dims[i][4] == "O1.WL2=O2":
                self.deepflow_outputs.append([mha_GEMMtime, mha_reduction, 'MHA', 0, 0, 0])
                self.deepflow_outputs.append([ffn_GEMMtime, ffn_reduction, 'FFN', 0, 0, 0])
                
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


class time_domain():
    def __init__(self, amped, deepFlow_res) -> None:

        self.amped = amped
        self.inputs = self.amped.inputs
        self.perf = self.amped.perf_model

        self.debug =DEBUG_PRINT
        self.timeline = []
        self.start = 0.0

        self.deepFlow_compute = deepFlow_res

        if self.debug:
            print(self.deepFlow_compute)

        self.linear_throughput = self.perf.reciprocal_of_OPS()
        self.non_linear_throughput = self.perf.C_NONLIN()

        self.B = int(self.inputs.parameters['batch_size'])
        self.D = int(self.inputs.parameters['dimensionality'])
        self.S = int(self.inputs.parameters['context'])
        self.sum_len = int(self.inputs.parameters['summarization_len'])
        self.h = int(self.inputs.parameters['hidden_layer_dimension_for_attention_sublayers'])
        self.nheads = int(self.inputs.parameters['attention_heads'])
        self.h_MLP1 = int(self.inputs.parameters['hidden_layer_dimension_MLP_1'])
        self.h_MLP2 = int(self.inputs.parameters['hidden_layer_dimension_MLP_2'])
        self.N_DP = int(self.inputs.parameters['data_parallel_degree'])
        self.N_PP = int(self.inputs.parameters['pipeline_parallel_degree'])
        self.N_TP = int(self.inputs.parameters["tensor_parallel_degree"])

        self.N_TP_INTRA = int(self.inputs.parameters['intra_node_tensor_parallel_degree'])
        self.N_TP_INTER = int(self.inputs.parameters['inter_node_tensor_parallel_degree'])

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

        print(f"Parsed data info:")
        print(f"\tIntrastructure:")
        print(f"\t\tTotal number of GPUs: {self.N_TOTAL_GPUS}")
        print(f"\t\tGPUs per node: {self.N_TOTAL_GPUS_PERNODE}")
        print(f"\tParallelism:")
        print(f"\t\tData Parallel: {self.N_DP} [intra:{self.N_DP_INTRA}, inter:{self.N_DP_INTER}]")
        print(f"\t\tTensor Parallel: {self.N_TP} [intra:{self.N_TP_INTRA}, inter:{self.N_TP_INTER}]")
        print(f"\t\tPipeline Parallel: {self.N_PP} [intra:{self.N_PP_INTRA}, inter:{self.N_PP_INTER}]")

        self.weight_precision = self.inputs.parameters["weight_precision"]
        self.gradient_precision = self.inputs.parameters["gradient_precision"]
        self.activation_precision = self.inputs.parameters["activation_precision"]
        self.optimizerstate_precision = self.inputs.parameters[
            "optimizer_state_precision"
        ]

        self.deepflow_mhatime = self.deepFlow_compute[6][0]
        self.deepflow_ffntime = self.deepFlow_compute[7][0]

        self.main(self.inputs, self.perf)

    def main(self, inputs, perf):

        self.start = 0
        self.timeline = []

        for z in range(20):
            if self.debug:
                print(f"Processing batch {z}")
            for i in range(self.nrOfLayers):
                self.forward_pass(inputs, perf)

            for i in range(self.nrOfLayers):
                self.backward_pass(inputs, perf)
                self.weight_update(inputs, perf)

        # if self.debug:
        #    print("Done processing all batches")
        _file = open(f"output_files/{self.amped.timeStamp}time_series.csv","w")
        _file.write(f"Total batches, {self.nrOfBatch}, Processed Batches, 20\n")
        _file.write(f"Layer, Type, start time, end time, duration, Bytes to be transferred, Collective type, Parallelism, Locality, Degree"+'\n')
        for x in range(len(self.timeline)):
            # if x % 10 == 0:
            # print(f"Writing step {x}: {self.timeline[x][0]}")
            cList = self.timeline[x]
            # print(cList)
            if cList[4] != 0:
                tmp = str(cList).replace("[", "").replace("]","")#.replace(",", " ")
                _file.write(tmp+'\n')

        _file = open(f"output_files/{self.amped.timeStamp}_time_series_single_GPU.csv","w")
        _file.write(f"Layer, Type, start time, end time, duration, Bytes to be transferred, Collective type, Parallelism, Locality, Degree, SRC, DEST"+'\n')
        sourceGPU = 0
        destGPU = 1
        for x in range(len(self.timeline)):

            cList = self.timeline[x]

            layerName = cList[0]
            layerType = cList[1]
            startTime = float(cList[2])
            endTime = float(cList[3])
            layerDuration = float(cList[4])

            if layerDuration != 0:
                if cList[6] == "ALLREDUCE":
                    # print(cList)
                    commVolume = float(cList[5])
                    collectiveType = cList[6]
                    parallelismType = cList[7]
                    parallelismLocality = cList[8]
                    nDegree = cList[9]

                    nStartTime = startTime
                    for x in range(6):
                        lName = f"{layerName}_{x}"
                        collectiveType = "P2P"                    
                        lDuration = layerDuration/6
                        nEndTime = nStartTime + lDuration
                        dVolume = commVolume /nDegree

                        if parallelismLocality == "INTER":
                            destGPU = sourceGPU + 1
                            if destGPU >= self.N_TOTAL_GPUS_PERNODE:
                                destGPU = destGPU - self.N_TOTAL_GPUS_PERNODE
                        elif parallelismLocality == "INTRA":
                            destGPU = sourceGPU + 1
                            if destGPU >= self.N_TOTAL_GPUS:
                                destGPU = destGPU - self.N_TOTAL_GPUS

                        nList = [lName, layerType, nStartTime, nEndTime, lDuration, dVolume, collectiveType, parallelismType, parallelismLocality, nDegree, sourceGPU, destGPU]

                        tmp = str(nList).replace("[", "").replace("]","")#.replace(",", " ")
                        nStartTime = nEndTime
                else:
                    parallelismType = cList[7]
                    parallelismLocality = cList[8]
                    if parallelismType == "PP":
                        if parallelismLocality == "INTRA":
                            # There is a bug here:
                            # If total GPUs used are less than GPUs per node then it is useless
                            # BUT not a problem for new but will come into picture later.
                            destGPU = sourceGPU + (self.N_TOTAL_GPUS_PERNODE/self.N_PP_INTRA)
                            if destGPU >= self.N_TOTAL_GPUS_PERNODE:
                                destGPU = destGPU - self.N_TOTAL_GPUS_PERNODE
                        elif parallelismLocality == "INTER":
                            destGPU = sourceGPU + (self.N_TOTAL_GPUS/self.N_PP_INTER)
                            if destGPU >= self.N_TOTAL_GPUS:
                                destGPU = destGPU - self.N_TOTAL_GPUS
                        nList = [cList[0], cList[1], cList[2], cList[3], cList[4], cList[5], cList[6], cList[7], cList[8], cList[9], sourceGPU, destGPU]
                    else:
                        nList = [cList[0], cList[1], cList[2], cList[3], cList[4], cList[5], cList[6], cList[7], cList[8]]
                    tmp = str(nList).replace("[", "").replace("]","")#.replace(",", " ")

                _file.write(tmp+'\n')

    def forward_pass(self, inputs, perf):

        MHA_macs=inputs.parameters["total_attention_sublayer_MAC_operations"]
        FFN_macs=inputs.parameters["total_MLP_sublayer_MAC_operations"]
        MHA_nonlinear=inputs.parameters["non_linear_operations_for_attention_sublayer"]
        FFN_nonlinear=inputs.parameters["non_linear_operations_for_MLP_sublayer"]

        compute_time_linear = self.deepflow_mhatime
        compute_time_non_linear = ((MHA_nonlinear * self.non_linear_throughput) * ceil(perf.p["activation_precision"] / perf.W_FU_NONLIN))/ (self.N_TP * self.N_PP)
        compute_time_MHA = compute_time_linear + compute_time_non_linear

        """
        Compute time from AMPeD
        compute_time_MHA =  ((MHA_macs*self.linear_throughput) * \
                            ceil(perf.p["weight_precision"]) / perf.W_FU_MAC) + \
                            ((MHA_nonlinear * self.non_linear_throughput) * \
                            ceil(perf.p["activation_precision"] / perf.W_FU_NONLIN))
        compute_time_MHA = compute_time_MHA / (self.N_TP * self.N_PP)"""

        compute_time_linear = self.deepflow_ffntime
        compute_time_non_linear = ((FFN_nonlinear * self.non_linear_throughput)* ceil(perf.p["activation_precision"] / perf.W_FU_NONLIN))/ (self.N_TP * self.N_PP)
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
        activation_volume = (self.B * self.h * self.S) * (
            perf.p["activation_precision"] / 8
        )
        TP_comm_intra_volume = (activation_volume / perf.p["number_of_nodes_required"])* \
            perf.p["activation_precision"]
        TP_comm_intra_time = perf.forward_tensor_model_intra()

        TP_comm_inter_volume = (activation_volume / perf.p["accelerators_per_node_required"])* \
            perf.p["activation_precision"]
        TP_comm_inter_time = perf.forward_tensor_parallel_inter()

        PP_comm_intra_volume = (activation_volume) #* perf.p["activation_precision"]
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

        # Can add MoE and DP here as well
        MoE_comm_time = perf.MoE_overhead_per_layer_fw_pass()
        DP_comm_time = perf.M_f_DP
        # Time line start
        self.timeline.append([
            "FWD_MHA",
            "Compute",
            self.start,
            self.update_time(compute_time_MHA),
            compute_time_MHA,
            "NONE",
            "NONE",
            "NONE",
            0
        ])

        self.timeline.append([
            "FWD_TP_COMM_INTRA",
            "Comm",
            self.start,
            self.update_time(TP_comm_intra_time/2),
            TP_comm_intra_time/2,
            TP_comm_intra_volume/2 if TP_comm_intra_time > 0 else 0,
            "ALLREDUCE",
            "TP",
            "INTRA",
            self.N_TP_INTRA
        ])

        self.timeline.append([
            "FWD_TP_COMM_INTER",
            "Comm",
            self.start,
            self.update_time(TP_comm_inter_time/2),
            TP_comm_inter_time/2,
            TP_comm_inter_volume/2 if TP_comm_inter_time > 0 else 0,
            "ALLREDUCE",
            "TP",
            "INTER",
            self.N_TP_INTER
        ])

        self.timeline.append([
            "FWD_FFN",
            "Compute",
            self.start,
            self.update_time(compute_time_FFN),
            compute_time_FFN,
            "NONE",
            "NONE",
            "NONE",
            0

        ])

        self.timeline.append([
            "FWD_TP_COMM_INTRA",
            "Comm",
            self.start,
            self.update_time(TP_comm_intra_time/2),
            TP_comm_intra_time/2,
            TP_comm_intra_volume/2 if TP_comm_intra_time > 0 else 0,
            "ALLREDUCE",
            "TP",
            "INTRA",
            self.N_TP_INTRA
        ])

        self.timeline.append([
            "FWD_TP_COMM_INTER",
            "Comm",
            self.start,
            self.update_time(TP_comm_inter_time/2),
            TP_comm_inter_time/2,
            TP_comm_inter_volume/2 if TP_comm_inter_time > 0 else 0,
            "ALLREDUCE",
            "TP",
            "INTER",
            self.N_TP_INTER
        ])

        self.timeline.append([
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
        ])

    def update_time(self, duration):
        endTime = self.start + duration
        self.start = endTime
        return endTime

    def backward_pass(self, inputs, perf):
        MHA_macs=inputs.parameters["total_attention_sublayer_MAC_operations"]
        FFN_macs=inputs.parameters["total_MLP_sublayer_MAC_operations"]
        MHA_nonlinear=inputs.parameters["non_linear_operations_for_attention_sublayer"]
        FFN_nonlinear=inputs.parameters["non_linear_operations_for_MLP_sublayer"]

        compute_time_linear = self.deepflow_mhatime
        compute_time_non_linear = ((MHA_nonlinear * self.non_linear_throughput) * ceil(perf.p["weight_precision"] / perf.W_FU_NONLIN))/ (self.N_TP * self.N_PP)
        compute_time_MHA = compute_time_linear + compute_time_non_linear

        """
        Compute time from AMPeD
        compute_time_MHA =  ((MHA_macs*self.linear_throughput) * ceil(max(perf.p["weight_precision"], perf.p["gradient_precision"])) / perf.W_FU_MAC) + ((MHA_nonlinear * self.non_linear_throughput) * ceil(perf.p["weight_precision"] / perf.W_FU_NONLIN))

        compute_time_MHA = compute_time_MHA/ (self.N_TP * self.N_PP)"""

        compute_time_linear = self.deepflow_ffntime
        compute_time_non_linear = ((FFN_nonlinear * self.non_linear_throughput)* ceil(perf.p["weight_precision"] / perf.W_FU_NONLIN))/ (self.N_TP * self.N_PP)
        compute_time_FFN = compute_time_linear + compute_time_non_linear

        """
        Compute time from AMPeD
        compute_time_FFN =  ((FFN_macs*self.linear_throughput)* ceil(max(perf.p["weight_precision"], perf.p["gradient_precision"])) / perf.W_FU_MAC) + ((FFN_nonlinear * self.non_linear_throughput)* ceil(perf.p["weight_precision"] / perf.W_FU_NONLIN))

        compute_time_FFN = compute_time_FFN/ (self.N_TP * self.N_PP)"""

        #####################################################################
        # Communication
        error_volume_per_layer_batch = self.B * self.S * self.h
        TP_comm_intra_volume = (error_volume_per_layer_batch / perf.p["number_of_nodes_required"])* perf.p["gradient_precision"]
        TP_comm_intra_time = perf.backward_tensor_model_intra()

        TP_comm_inter_volume = (error_volume_per_layer_batch / perf.p["accelerators_per_node_required"])* perf.p["gradient_precision"]
        TP_comm_inter_time = perf.backward_tensor_parallel_inter()

        PP_comm_intra_volume = (error_volume_per_layer_batch)* perf.p["gradient_precision"]
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

        # Can add MoE and DP here as well
        MoE_comm_time = perf.MoE_overhead_per_layer_bw_pass()
        DP_comm_time = perf.M_f_DP
        # Time line start
        self.timeline.append([
            "BWD_MHA",
            "Compute",
            self.start,
            self.update_time(compute_time_MHA),
            compute_time_MHA,
            "NONE",
            "NONE",
            "NONE",
            0

        ])

        self.timeline.append([
            "BWD_TP_COMM_INTRA",
            "Comm",
            self.start,
            self.update_time(TP_comm_intra_time/2),
            TP_comm_intra_time/2,
            TP_comm_intra_volume/2 if TP_comm_intra_time > 0 else 0,
            "ALLREDUCE",
            "TP",
            "INTRA",
            self.N_TP_INTRA
        ])

        self.timeline.append([
            "BWD_TP_COMM_INTER",
            "Comm",
            self.start,
            self.update_time(TP_comm_inter_time/2),
            TP_comm_inter_time/2,
            TP_comm_inter_volume/2 if TP_comm_inter_time > 0 else 0,
            "ALLREDUCE",
            "TP",
            "INTER",
            self.N_TP_INTER
        ])

        self.timeline.append([
            "BWD_FFN",
            "Compute",
            self.start,
            self.update_time(compute_time_FFN),
            compute_time_FFN,
            "NONE",
            "NONE",
            "NONE",
            0

        ])

        self.timeline.append([
            "BWD_TP_COMM_INTRA",
            "Comm",
            self.start,
            self.update_time(TP_comm_intra_time/2),
            TP_comm_intra_time/2,
            TP_comm_intra_volume/2 if TP_comm_intra_time > 0 else 0,
            "ALLREDUCE",
            "TP",
            "INTRA",
            self.N_TP_INTRA
        ])

        self.timeline.append([
            "BWD_TP_COMM_INTER",
            "Comm",
            self.start,
            self.update_time(TP_comm_inter_time/2),
            TP_comm_inter_time/2,
            TP_comm_inter_volume/2 if TP_comm_inter_time > 0 else 0,
            "ALLREDUCE",
            "TP",
            "INTER",
            self.N_TP_INTER
        ])

        self.timeline.append([
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
        ])

        number_of_parameters_per_layer = inputs.parameters["number_of_parameters_per_layer"]
        DP_comm_intra_volume = (number_of_parameters_per_layer / perf.p["number_of_nodes_required"])* perf.p["gradient_precision"]
        DP_comm_intra_time = perf.communication_time_backwards_DP_all_reduce_intra()

        DP_comm_inter_volume = (number_of_parameters_per_layer / perf.p["accelerators_per_node_required"])* perf.p["gradient_precision"]
        DP_comm_inter_time = perf.communication_time_backwards_DP_all_reduce_inter()

        self.timeline.append([
            "BWD_DP_COMM_INTRA",
            "Comm",
            self.start,
            self.update_time(DP_comm_intra_time),
            DP_comm_intra_time,
            DP_comm_intra_volume if DP_comm_intra_time > 0 else 0,
            "ALLREDUCE",
            "DP",
            "INTRA",
            self.N_DP_INTRA
        ])

        self.timeline.append([
            "BWD_DP_COMM_INTER",
            "Comm",
            self.start,
            self.update_time(DP_comm_inter_time),
            DP_comm_inter_time,
            DP_comm_inter_volume if DP_comm_inter_time > 0 else 0,
            "ALLREDUCE",
            "DP",
            "INTER",
            self.N_DP_INTER
        ])

    def weight_update(self, inputs, perf):

        compute_weight_update = perf.weight_update_time()
        compute_weight_update = compute_weight_update/ (self.N_DP * self.N_TP * self.N_PP)

        self.timeline.append([
            "Weight_update",
            "Compute",
            self.start,
            self.update_time(compute_weight_update),
            compute_weight_update,
            "NONE",
            "NONE",
            "NONE",
            0
        ])


class astrasim_workload():
    def __init__(self) -> None:
        
        # Gather data for astra-sim workload
        # TP comm will occur after layernorm1 and layernorm2 and is ARR
        # PP comm will occue after layernorm2 after TP and is P2P
        # DP comm will occur after PP and is ARR
        self.sublayers = ["X.W=Q", "X.W=K", "X.W=V", "Q.K=R", "softmax1", "R.V=Z", "Concat", "Z.W=O", "layernorm1", "O.WL1=O1", "GELU", "O1.WL2=O2", "layernorm2"]
        self.sublayers_MGT = [  "attention_layer", 
                                "TP_comm_mha", 
                                "feedForwardNetwork", 
                                "TP_comm_FFN",
                                "DP_comm"]
        self.sublayers_params = [   'layer',                    # Layer name
                                    'rsvd_var',                 # Reserved variable
                                    'fwd_pass_compute_time',    # compute time in ns
                                    'fwd_pass_comm_type',       # communication type ["NONE", "ALLREDUCE", "ALLGATHER"]
                                    'fwd_pass_comm_size',       # communication volume size in bytes
                                    'input_grad_compute_time',  # compute time in ns
                                    'input_grad_comm_type',     # communication type ["NONE", "ALLREDUCE", "ALLGATHER"]
                                    'input_grad_comm_size',     # communication volume size in bytes
                                    'weight_grad_compute_time', # compute time in ns
                                    'weight_grad_comm_type',    # communication type ["NONE", "ALLREDUCE", "ALLGATHER"]
                                    'weight_grad_comm_size',    # communication volume size in bytes
                                    'delay_after_collectives',  # Additional delay after each collectives
                                    'fwd_pass_MACs',            # AP: Forward pass MACs
                                    'input_grad_MACs',          # AP: Inpute gradient MACs
                                    'weight_grad_MACs']         # AP: Weight gradient MACs
        self.epoc = {}
        self.main()
        
    def main(self):
        
        for sublayer in self.sublayers:
            for param in self.sublayers_params:
                if param == 'layer' or param == 'fwd_pass_comm_type' or param == 'input_grad_comm_type' or param == 'weight_grad_comm_type':
                    # self.epoc[sublayer][param] = ''
                    pass
                else:
                    # self.epoc[sublayer][param] = 0
                    pass
    
    def MGT_layer_update(self, inputs, perf):
        MHA_macs = inputs.parameters['total_attention_sublayer_MAC_operations'] * inputs.parameters['attention_heads']
        
        self.single_layer_update(   layer = "attention_layer", 
                                    rsvd_var = -1, 
                                    fwd_pass_compute_time = self.amped_compute_time(
                                        sparam=inputs.parameters["weight_precision"],
                                        MACs=MHA_macs,
                                        cmac=perf.reciprocal_of_OPS(),
                                        wfumac = perf.W_FU_MAC,
                                        TP_degree=inputs.parameters["tensor_parallel_degree"]
                                        ),  # will come from DeepFlow
                                    fwd_pass_comm_type = "None", 
                                    fwd_pass_comm_size = 0,
                                    input_grad_compute_time = self.amped_compute_time(
                                        sparam=inputs.parameters["gradient_precision"],
                                        MACs=MHA_macs,
                                        cmac=perf.reciprocal_of_OPS(),
                                        wfumac = perf.W_FU_MAC,
                                        TP_degree=inputs.parameters["tensor_parallel_degree"]
                                        ), 
                                    input_grad_comm_type = "None", 
                                    input_grad_comm_size = 0, 
                                    weight_grad_compute_time = (MHA_macs * perf.reciprocal_of_OPS())/inputs.parameters["tensor_parallel_degree"], 
                                    weight_grad_comm_type = "None", 
                                    weight_grad_comm_size = 0, 
                                    delay_after_collectives = 10, 
                                    fwd_pass_MACs = MHA_macs, 
                                    input_grad_MACs = MHA_macs, 
                                    weight_grad_MACs = MHA_macs)
        
        FWD_PASS_COMM_VOLUME=(2 * perf.p["activations_volume_per_layer_batch"] / perf.p["number_of_nodes_required"])if perf.p["intra_node_tensor_parallel_degree"] > 1 else 0
        
        self.single_layer_update(   layer = "TP_comm_MHA", 
                                    rsvd_var = -1, 
                                    fwd_pass_compute_time = 0,  # will come from DeepFlow
                                    fwd_pass_comm_type = "ALLREDUCE", 
                                    fwd_pass_comm_size = 2 * perf.p["activations_volume_per_layer_batch"] / perf.p["number_of_nodes_required"],
                                    input_grad_compute_time = 0, 
                                    input_grad_comm_type = "ALLREDUCE", 
                                    input_grad_comm_size = 0, 
                                    weight_grad_compute_time = 0, 
                                    weight_grad_comm_type = 0, 
                                    weight_grad_comm_size = 0, 
                                    delay_after_collectives = 10, 
                                    fwd_pass_MACs = 0, 
                                    input_grad_MACs = 0, 
                                    weight_grad_MACs = 0)
        
        FFN_macs = inputs.parameters['total_MLP_sublayer_MAC_operations'] * inputs.parameters['attention_heads']
        self.single_layer_update(   layer = "feedForwardNetwork", 
                                    rsvd_var = -1, 
                                    fwd_pass_compute_time = self.amped_compute_time(
                                        sparam=inputs.parameters["weight_precision"],
                                        MACs=FFN_macs,
                                        cmac=perf.reciprocal_of_OPS(),
                                        wfumac = perf.W_FU_MAC,
                                        TP_degree=inputs.parameters["tensor_parallel_degree"]
                                        ),  # will come from DeepFlow
                                    fwd_pass_comm_type = "None", 
                                    fwd_pass_comm_size = 0,
                                    input_grad_compute_time = self.amped_compute_time(
                                        sparam=inputs.parameters["gradient_precision"],
                                        MACs=FFN_macs,
                                        cmac=perf.reciprocal_of_OPS(),
                                        wfumac = perf.W_FU_MAC,
                                        TP_degree=inputs.parameters["tensor_parallel_degree"]
                                        ), 
                                    input_grad_comm_type = "None", 
                                    input_grad_comm_size = 0, 
                                    weight_grad_compute_time = (FFN_macs * perf.reciprocal_of_OPS())/inputs.parameters["tensor_parallel_degree"], 
                                    weight_grad_comm_type = "None", 
                                    weight_grad_comm_size = 0, 
                                    delay_after_collectives = 10, 
                                    fwd_pass_MACs = FFN_macs, 
                                    input_grad_MACs = FFN_macs, 
                                    weight_grad_MACs = FFN_macs)
        
        self.single_layer_update(   layer = "TP_comm_FFN", 
                                    rsvd_var = -1, 
                                    fwd_pass_compute_time = 0,  # will come from DeepFlow
                                    fwd_pass_comm_type = "ALLREDUCE", 
                                    fwd_pass_comm_size = 0,
                                    input_grad_compute_time = 0, 
                                    input_grad_comm_type = "ALLREDUCE", 
                                    input_grad_comm_size = 0, 
                                    weight_grad_compute_time = 0, 
                                    weight_grad_comm_type = 0, 
                                    weight_grad_comm_size = 0, 
                                    delay_after_collectives = 10, 
                                    fwd_pass_MACs = MHA_macs, 
                                    input_grad_MACs = MHA_macs, 
                                    weight_grad_MACs = MHA_macs)

        self.single_layer_update(   layer = "DP_comm", 
                                    rsvd_var = -1, 
                                    fwd_pass_compute_time = 0,  # will come from DeepFlow
                                    fwd_pass_comm_type = "NONE", 
                                    fwd_pass_comm_size = 0,
                                    input_grad_compute_time = 0, 
                                    input_grad_comm_type = "NONE", 
                                    input_grad_comm_size = 0, 
                                    weight_grad_compute_time = 0, 
                                    weight_grad_comm_type = "ALLREDUCE", 
                                    weight_grad_comm_size = 0, 
                                    delay_after_collectives = 10, 
                                    fwd_pass_MACs = MHA_macs, 
                                    input_grad_MACs = MHA_macs, 
                                    weight_grad_MACs = MHA_macs)
        
        pass
    
    def amped_compute_time(self, sparam, MACs, cmac, wfumac, TP_degree):
        return ((MACs*cmac*(sparam/wfumac))/TP_degree)*(10**9)
    
    def layer_update(self, inputs, perf):
        
        query_macs = inputs.parameters['query_MAC_operations'] * inputs.parameters['attention_heads']
        self.single_layer_update(   layer = "X.W=Q", 
                                    rsvd_var = -1, 
                                    fwd_pass_compute_time = self.amped_compute_time(
                                        sparam=inputs.parameters["weight_precision"],
                                        MACs=query_macs,
                                        cmac=perf.reciprocal_of_OPS(),
                                        wfumac = perf.W_FU_MAC,
                                        TP_degree=inputs.parameters["tensor_parallel_degree"]
                                        ),  # will come from DeepFlow
                                    fwd_pass_comm_type = "None", 
                                    fwd_pass_comm_size = 0,
                                    input_grad_compute_time = self.amped_compute_time(
                                        sparam=inputs.parameters["gradient_precision"],
                                        MACs=query_macs,
                                        cmac=perf.reciprocal_of_OPS(),
                                        wfumac = perf.W_FU_MAC,
                                        TP_degree=inputs.parameters["tensor_parallel_degree"]
                                        ), 
                                    input_grad_comm_type = "None", 
                                    input_grad_comm_size = 0, 
                                    weight_grad_compute_time = (query_macs * perf.reciprocal_of_OPS())/inputs.parameters["tensor_parallel_degree"], 
                                    weight_grad_comm_type = 0, 
                                    weight_grad_comm_size = 0, 
                                    delay_after_collectives = 10, 
                                    fwd_pass_MACs = query_macs, 
                                    input_grad_MACs = query_macs, 
                                    weight_grad_MACs = query_macs)
        
        """
        self.single_layer_update(   layer = "X.W=Q", 
                                    rsvd_var = -1, 
                                    fwd_pass_compute_time, 
                                    fwd_pass_comm_type, 
                                    fwd_pass_comm_size,
                                    input_grad_compute_time, 
                                    input_grad_comm_type, 
                                    input_grad_comm_size, 
                                    weight_grad_compute_time , 
                                    weight_grad_comm_type, 
                                    weight_grad_comm_size, 
                                    delay_after_collectives, 
                                    fwd_pass_MACs, 
                                    input_grad_MACs, 
                                    weight_grad_MACs)
        """

    
    def single_layer_update(self, layer, rsvd_var, fwd_pass_compute_time, fwd_pass_comm_type, fwd_pass_comm_size, 
                     input_grad_compute_time, input_grad_comm_type, input_grad_comm_size, 
                     weight_grad_compute_time , weight_grad_comm_type, weight_grad_comm_size, 
                     delay_after_collectives, 
                     fwd_pass_MACs, input_grad_MACs, weight_grad_MACs):
        pass
        """
        self.epoc[layer]["layer"] = layer
        self.epoc[layer]['rsvd_var'] = rsvd_var
        self.epoc[layer]['fwd_pass_compute_time'] = fwd_pass_compute_time
        self.epoc[layer]['fwd_pass_comm_type'] = fwd_pass_comm_type
        self.epoc[layer]['fwd_pass_comm_size'] = fwd_pass_comm_size
        self.epoc[layer]['input_grad_compute_time'] = input_grad_compute_time
        self.epoc[layer]['input_grad_comm_type'] = input_grad_comm_type
        self.epoc[layer]['input_grad_comm_size'] = input_grad_comm_size
        self.epoc[layer]['weight_grad_compute_time'] = weight_grad_compute_time
        self.epoc[layer]['weight_grad_comm_type'] = weight_grad_comm_type
        self.epoc[layer]['weight_grad_comm_size'] = weight_grad_comm_size
        self.epoc[layer]['delay_after_collectives'] = delay_after_collectives
        self.epoc[layer]['fwd_pass_MACs'] = fwd_pass_MACs
        self.epoc[layer]['input_grad_MACs'] = input_grad_MACs
        self.epoc[layer]['weight_grad_MACs'] = weight_grad_MACs
        """


class amped_exec():
    def __init__(self, training=True) -> None:
        self.debug = DEBUG_PRINT
        self.inputs = []
        self.breakdown = {}
        self.single_epoc = astrasim_workload()
        self.training = training
        self.timeStamp = ''

        self.main()

    def calc_time(self, inputs, seqLen, flag_gen = False, debug = False, iter=1):

        if not self.training:
            inputs.parameters['context'] = seqLen
            inputs.parameters['tokens_to_train'] = seqLen

            if flag_gen:
                inputs.parameters['summarization_len'] = iter
            else:
                inputs.parameters['summarization_len'] = seqLen

        inputs.dependency_mapping = CalculateFunctionsDependencyMapping()

        for parameter_name in inputs.temp_parameters_to_calculate:
            if parameter_name not in inputs.temp_parameters_dict:
                inputs.calculate_parameter(parameter_name, inputs.temp_parameters_dict)

        inputs.parameters = Parameters(inputs, inputs.temp_parameters_dict, inputs.dependency_mapping)  # the main property used in other files
        inputs.transformer = inputs.config["neural_network_training_parameters"]["lookup_config"]["lookup_table_row"]
        inputs.accelerator = inputs.config["accelerator_architecture_parameters"]["lookup_config"]["lookup_table_row"]

        perf_model = PerformanceModel(inputs)

        perLayer_computetime_fwd_pass = (perf_model.compute_time_forward_pass()) / (perf_model.p["data_parallel_degree"] * perf_model.p["tensor_parallel_degree"] * perf_model.p["pipeline_parallel_degree"])
        perLayer_commtime_fwd_pass = perf_model.communication_time_forward_pass()
        perLayer_commtime_pipeline_bubble = ((perf_model.p["pipeline_parallel_degree"] - 1)
            * ((perf_model.compute_time_forward_pass())
               / (perf_model.p["data_parallel_degree"] * perf_model.p["tensor_parallel_degree"] * perf_model.p["pipeline_parallel_degree"]
                  * perf_model.p["layers"])
               + perf_model.communication_time_forward_pass())
            / perf_model.p["number_of_microbatches_per_minibatch"])

        if self.training:
            perLayer_computetime_bwd_pass = (perf_model.compute_time_forward_pass() + perf_model.compute_time_backward_pass() + perf_model.weight_update_time()) / (perf_model.p["data_parallel_degree"] * perf_model.p["tensor_parallel_degree"] * perf_model.p["pipeline_parallel_degree"])
            perLayer_commtime_bwd_pass = perf_model.communication_time_backwards_DP_all_reduce() + perf_model.communication_time_backward_pass()
            perLayer_commtime_pipeline_bubble_bwd = ((perf_model.p["pipeline_parallel_degree"] - 1)
            * ((perf_model.compute_time_backward_pass())
               / (perf_model.p["data_parallel_degree"] * perf_model.p["tensor_parallel_degree"] * perf_model.p["pipeline_parallel_degree"]
                  * perf_model.p["layers"])
               + perf_model.communication_time_backward_pass())
            / perf_model.p["number_of_microbatches_per_minibatch"])

        if debug:
            print(f"Query FLOP : {2 * inputs.parameters['query_MAC_operations'] * inputs.parameters['attention_heads']}")
            print(f"Key FLOP : {2 * inputs.parameters['key_MAC_operations'] * inputs.parameters['attention_heads']}")
            print(f"Value FLOP : {2 * inputs.parameters['value_MAC_operations'] * inputs.parameters['attention_heads']}")
            print(f"MHA FLOP : {2 * inputs.parameters['self_attention_MAC_operations'] * inputs.parameters['attention_heads']}")    
            print(f"Wout FLOP : {2 * inputs.parameters['attention_sublayer_output_MAC_operations']}")
            print(f"FFN FLOP : {2 * inputs.parameters['total_MLP_sublayer_MAC_operations']}")

        computetime = perf_model.p["number_of_batches"] * perf_model.p["layers"] * perLayer_computetime_fwd_pass
        commtime = perf_model.p["number_of_batches"] * perf_model.p["layers"] * perLayer_commtime_fwd_pass
        commtime_pipeline_bubble = perf_model.p["number_of_batches"] * perf_model.p["layers"] * perLayer_commtime_pipeline_bubble

        if self.training:
            computetime += perf_model.p["number_of_batches"] * perf_model.p["layers"] * perLayer_computetime_bwd_pass
            commtime += perf_model.p["number_of_batches"] * perf_model.p["layers"] * perLayer_commtime_bwd_pass
            commtime_pipeline_bubble += perf_model.p["number_of_batches"] * perf_model.p["layers"] * perLayer_commtime_pipeline_bubble_bwd

        self.single_epoc.layer_update(inputs, perf_model)
        self.perf_model = perf_model

        return [computetime, commtime, commtime_pipeline_bubble]

    def temp_string_training_time_breakdown(self, inferenceTime, computeTime, commTime, waitingTime):
        pairs = {
            "Total time to train (s)": inferenceTime,
            "Total time to train (days)": inferenceTime/3600/24,
            "Total time to train (years)": inferenceTime/3600/24/365,
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
            "Total MoE communication overhead time": 0
        }

        longest_label_length = len(max(pairs.keys(), key=len))
        self.breakdown = pairs
        breakdown = "TRAINING TIME BREAKDOWN\n\n"
        return breakdown + "\n".join([f"{label :-<{longest_label_length}} {val}" for label, val in pairs.items()])

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
            "Total MoE communication overhead time": self.perf_model.total_MoE_overhead()
        }

        longest_label_length = len(max(pairs.keys(), key=len))

        self.breakdown = pairs
        breakdown = "TRAINING TIME BREAKDOWN\n\n"
        return breakdown + "\n".join([f"{label :-<{longest_label_length}} {val}" for label, val in pairs.items()])

    def main(self):
        # Amped inference script
        inputs = Inputs()

        if not self.training:
            inputs.parameters['batch_size'] = 1
            inputs.parameters['number_of_microbatches_per_minibatch'] = 1

        ############################################################################

        # Prompt/ Summarization stage
        if not self.training:
            [computetime_fwd_pass, commtime_fwd_pass, commtime_pipeline_bubble] \
                = self.calc_time(inputs, PROMPT_LEN, False , False)
        else:
            [computetime_fwd_pass, commtime_fwd_pass, commtime_pipeline_bubble] \
                = self.calc_time(inputs, inputs.parameters['context'], False , False)

        summary = "FULL CONFIGURATION\n\n" + inputs.parameters.to_string_structured()
        self.temp_save_as("config_summary.txt", summary)

        if not self.training:
            file = open(f"{OUTPUT_PATH}/AmpedInference.txt", "w")
        else:
            file = open(f"{OUTPUT_PATH}/AmpedTraining.txt", "w")

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
            for i in range(PROMPT_LEN+1, PROMPT_LEN+GENERATION_LEN+1):

                print(f"Stage : {i-PROMPT_LEN}")

                [computetime_fwd_pass, commtime_fwd_pass, commtime_pipeline_bubble] \
                    = self.calc_time(inputs, 1, True , False, i)

                print(f"\nCompute time (s) : {computetime_fwd_pass}")
                print(f"Communication time (s) : {commtime_fwd_pass}")
                print(f"Pipeline bubble time (s) : {commtime_pipeline_bubble}")

                tmp = f"Stage,{i-PROMPT_LEN},Compute time (s),{computetime_fwd_pass},Communication time (s), {commtime_fwd_pass},Pipeline bubble time (s),{commtime_pipeline_bubble}\n"
                file.write(tmp)

                overall_computetime += computetime_fwd_pass
                overall_commtime += commtime_fwd_pass
                overall_pipeline_bubble += commtime_pipeline_bubble

        print(f"\n\nOverall timings")
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
            self.temp_save_as("training_time_breakdown.txt", \
                self.temp_string_training_time_breakdown(\
                    overall_computetime+overall_commtime+overall_pipeline_bubble,
                    overall_computetime, 
                    overall_commtime, 
                    overall_pipeline_bubble))
        else:
            self.temp_save_as("training_time_breakdown.txt", \
                self.training_string_training_time_breakdown())

    def temp_save_as(self, filename: str, content: str, encoding: str | None = None):
        if not os.path.isdir("output_files"):
            os.mkdir("output_files")
        self.timeStamp = time_prefix()
        open(f"output_files/{self.timeStamp}{filename}", "w", encoding=encoding).write(
            content
        )

class mat_dims_ampedToDF():
    def __init__(self, amped, training=True) -> None:
        self.dims = {}
        self.debug = DEBUG_PRINT
        self.training = training
        if self.debug:
            print("Starting mat dims amped to DF script...")
        self.amped = amped
        self.main(self.amped.inputs)

    def mmm_breakup(self, B, D, S, h, nheads, h_MLP1, h_MLP2, N_DP, N_PP):
        mmm =  {}
        dims = {}
        # deepflow_outputs = {}
        if not self.training:
            numlevels = 6*(GENERATION_LEN+1)
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
        dims[0]=[int(3*B*S/N_DP/N_PP), D, h*nheads, 'CR', levels[0]] #factor 3 due to K+Q+V  Columnwise Q = [Q1, Q2] etc
        dims[1]=[int(B*S/N_DP/N_PP), h*nheads, S, 'CR', levels[1]] # This seem off !!!
        dims[2]=[int(B*S/N_DP/N_PP), S, h*nheads, 'CR', levels[2]]
        dims[3]=[int(B*S/N_DP/N_PP), D, D, 'CR', levels[3]]
        dims[4]=[int(B*S/N_DP/N_PP), D, h_MLP1, 'RC', levels[4]] # Row wise split, WL1 = [WL1 ; WL2]
        dims[5]=[int(B*S/N_DP/N_PP), h_MLP1, h_MLP2, 'RC', levels[5]]

        if not self.training:
            # Generation
            S = 1
            for i in range(1, GENERATION_LEN):
                dims[0+(6*i)]=[int(3*B*S/N_DP/N_PP), D, h*nheads, 'CR', levels[0]] #factor 3 due to K+Q+V
                dims[1+(6*i)]=[S, int(B*S*(i+PROMPT_LEN)/N_DP/N_PP), h*nheads, 'CR', levels[1]]
                dims[2+(6*i)]=[S, h*nheads ,int(B*S*(i+PROMPT_LEN)/N_DP/N_PP), 'CR', levels[2]]
                dims[3+(6*i)]=[int(B*S/N_DP/N_PP), D, D, 'CR', levels[3]]
                dims[4+(6*i)]=[int(B*S/N_DP/N_PP), D, h_MLP1, 'RC', levels[4]]
                dims[5+(6*i)]=[int(B*S/N_DP/N_PP), h_MLP1, h_MLP2, 'RC', levels[5]]

        if self.debug:
            print("levels:",levels)
            print("writting the matrix dimensions ...")

        file = open(f"{OUTPUT_PATH}/{self.amped.timeStamp}mat_dims_amped.txt","w")
        # file.write('#'+str(levels)+'\n')
        for i in range(len(dims)):
            mmm[i]=[]
            if self.debug:
                print(f"Gathered {dims[i]}")
            mmm[i].append(dims[i])
            # deepflow_outputs[i] = []
            # print(mmm[i])
            tmp = str(mmm[i]).replace("[", "").replace("]","").replace(",", " ")
            # print(tmp)
            file.write(tmp+'\n')

        self.dims = dims

    def main(self, inputs):
        print("**** Creating GEMMs from AMPeD ****")

        B = int(inputs.parameters['batch_size'])
        D = int(inputs.parameters['dimensionality'])
        S = int(inputs.parameters['context'])
        sum_len = int(inputs.parameters['summarization_len'])
        h = int(inputs.parameters['hidden_layer_dimension_for_attention_sublayers'])
        nheads = int(inputs.parameters['attention_heads'])
        h_MLP1 = int(inputs.parameters['hidden_layer_dimension_MLP_1'])
        h_MLP2 = int(inputs.parameters['hidden_layer_dimension_MLP_2'])
        N_DP = int(inputs.parameters['data_parallel_degree'])
        N_PP = int(inputs.parameters['pipeline_parallel_degree'])
        return self.mmm_breakup(B, D, S, h, nheads, h_MLP1, h_MLP2, N_DP, N_PP)


if __name__ == '__main__':

    # First call amped
    AMPeD = amped_exec(training=True)
    # then, mat_dims_ampedToDF
    mat_dims = mat_dims_ampedToDF(amped=AMPeD, training=True)
    # then, run.sh / deepflow exec / perf.py
    DeepFlow = deepflow_exec(AMPeD, mat_dims.dims)
    # then, cal_time.py
    cal_time(AMPeD, DeepFlow.deepflow_outputs)
    # Time domain list
    time_domain(AMPeD, DeepFlow.deepflow_outputs)
