import sys
import os
sys.path.insert(1, '../AMPeD')
sys.path.insert(1, '../DeepFlow')

#imports for deepflow
import perf
import config
from perf import TimeCalculation
import click

# imports fo mat_dims_ampedToDF AND cal_time
import numpy as np
import argparse
import pandas as pd

PROMPT_LEN = 2048
GENERATION_LEN = 10


class cal_time():
    def __init__(self, inputs, outputs, deepFlow_outputs) -> None:
        self.inputs = inputs
        self.breakdown = outputs
        self.deepflow_outputs = deepFlow_outputs
        
        self.main(self.inputs)

    def time_from_GEMM(self):
        # Directory containing the files
        t_elapsed =0.0
        
        for gemmTimes in self.deepflow_outputs:
            gemmTime = gemmTimes[0]
            reductionTime = gemmTimes[1]
            
            t_elapsed += gemmTime
            
        t_elapsed = t_elapsed*2.0 #FW pass + BW pass (weight update time is added seperately)
        #print("********** FW-BW pass time ************", t_elapsed)
        return t_elapsed
    
    def main(self, inputs):
        print("**** Computing GEMM timings using DeepFlow *****")
        
        #------------------------reading timings from AMPeD------------------------------------------------------------
        print(self.breakdown)

        t_FW_BW = self.time_from_GEMM()
        #print("t_FW_BW:", t_FW_BW)
        nbatch = int(inputs.parameters["tokens_to_train"])/(int(inputs.parameters["context"])\
                                            *int(inputs.parameters["batch_size"]))
        nbatch = 1
        time = int(inputs.parameters["layers"])*nbatch*t_FW_BW + float(self.breakdown["Total communication time forward pass (s)"])\
            +float(self.breakdown["Total communication time backward pass (s)"])\
            +float(self.breakdown["Computation time weight updates (s)"]) \
            +float(self.breakdown["Waiting Time due to pipeline bubbles (s)"])

        print("total time:", time)

class deepflow_exec():
    def __init__(self, inputs, dims) -> None:
        self.debug = True
        self.inputs = inputs
        self.TP_DEGREE = inputs.parameters["intra_node_tensor_parallel_degree"]
        self.CONFIG_DIR="/imec/scratch/dtpatha/patel23/amped_deepflow/amped_deepflow/deepflow_configs"
        self.OUTDIR="/imec/scratch/dtpatha/patel23/amped_deepflow/amped_deepflow/output_files"
        
        if self.debug:
            print(f"TP degree is set to : {self.TP_DEGREE}")
            print(f"Config: {self.CONFIG_DIR}")
            print(f"Output dir: {self.OUTDIR}")
        self.deepflow_outputs = []
        self.main(dims)
    
    def deepflow_function(self, exp_config, exp_dir, debug, m, n, k, t, kp1, kp2, gemm, batch_size=2048, hidden_dim=19968, seq_len=20, vocab_size=800000, num_layer=2, dp=None, lp=None, lev=None, args_input=False):
        exp_path = os.path.expandvars(os.path.expanduser(exp_config))
        exp_config = config.parse_config(exp_path)
        output_file = exp_dir + "/summary_l%s_m%s_n%s_k%s.txt" %(lev, m, n, k) ##Output dir should be created manually

        TC = TimeCalculation(exp_config)
        if args_input:
            TC.updateParams(debug, m, n, k, t, kp1, kp2, dp, lp, gemm, 
                        batch_size, hidden_dim, seq_len, vocab_size, num_layer)

        #Report GEMM time on fw path
        if TC.validating_GEMM:
            
            if kp1 == 1 and kp2 ==1: #no parallelism
                gemm_time = TC.getCf(m, k, n)
            elif t == 'CR':
                gemm_time = TC.getDistGEMM_f_kp1(m, k, n, kp1, "Cf_CR")
            elif t == 'RC':
                gemm_time = TC.getDistGEMM_f_kp2(m, k, n, kp1, kp2, "Cf_RC")
            else:
                print("Incorrect parallelism type, CR: Column-Row, RC: Row-Column")
                sys.exit()
            
            with open(output_file, "w") as f:
                f.write("GEMM Time: {}\n".format(gemm_time[0]))
                f.write("Reduction Time: {}\n".format(gemm_time[1]))
            return [gemm_time[0], gemm_time[1]]
        
    def main(self, dims):
        for i in range(len(dims)):
            #if self.debug:
            #    print(f"Evaluating GEMM of M:{dims[i][0]} N:{dims[i][1]} K:{dims[i][2]}")
            if dims[i][0] < self.TP_DEGREE:
                temp_outputs = self.deepflow_function(exp_config=f"{self.CONFIG_DIR}/v100.yaml", exp_dir=f"{self.OUTDIR}/LLM", debug=True, m=dims[i][0], n=dims[i][1], k=dims[i][2], t='RC', kp1=1, kp2=self.TP_DEGREE, gemm=True)
            else:
                temp_outputs = self.deepflow_function(exp_config=f"{self.CONFIG_DIR}/v100.yaml", exp_dir=f"{self.OUTDIR}/LLM", debug=True, m=dims[i][0], n=dims[i][1], k=dims[i][2], t='RC', kp1=self.TP_DEGREE, kp2=1, gemm=True)
           
            print(temp_outputs)
            self.deepflow_outputs.append(temp_outputs)

            if self.debug:
                print(f"Dimension:{dims[i]}\t\tGEMM time/Reduction time: {self.deepflow_outputs[i]}")
    
    
# Imports for amped
from amped import save_GEMM_breakdown
from amped.common import save_as
from amped.performance_model import PerformanceModel
# from amped.inputs import Inputs, CalculateFunctionsDependencyMapping, Parameters
from amped_backups.inputs import Inputs, CalculateFunctionsDependencyMapping, Parameters


class amped_exec():
    def __init__(self) -> None:
        self.debug = True
        self.inputs = []
        self.breakdown = {}
        
        self.main()
    
    def calc_time(self, inputs, seqLen, flag_gen = False, debug = False, iter=1):
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
        perLayer_commtime_pipeline_bubble = perf_model.waiting_time_due_to_pipeline_bubbles()

        if debug:
            print(f"Query FLOP : {2 * inputs.parameters['query_MAC_operations'] * inputs.parameters['attention_heads']}")
            print(f"Key FLOP : {2 * inputs.parameters['key_MAC_operations'] * inputs.parameters['attention_heads']}")
            print(f"Value FLOP : {2 * inputs.parameters['value_MAC_operations'] * inputs.parameters['attention_heads']}")
            print(f"MHA FLOP : {2 * inputs.parameters['self_attention_MAC_operations'] * inputs.parameters['attention_heads']}")    
            print(f"Wout FLOP : {2 * inputs.parameters['attention_sublayer_output_MAC_operations']}")
            print(f"FFN FLOP : {2 * inputs.parameters['total_MLP_sublayer_MAC_operations']}")
        
        computetime_fwd_pass = perf_model.p["number_of_batches"] * perf_model.p["layers"] * perLayer_computetime_fwd_pass
        commtime_fwd_pass = perf_model.p["number_of_batches"] * perf_model.p["layers"] * perLayer_commtime_fwd_pass
        commtime_pipeline_bubble = perf_model.p["number_of_batches"] * perf_model.p["layers"] * perLayer_commtime_pipeline_bubble
        
        return [computetime_fwd_pass, commtime_fwd_pass, commtime_pipeline_bubble]

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

    def main(self):
        # Amped inference script
        inputs = Inputs()
        inputs.parameters['batch_size'] = 1
        inputs.parameters['number_of_microbatches_per_minibatch'] = 1
        inputs.parameters['intra_node_tensor_parallel_degree'] = 16

        ############################################################################

        # Prompt/ Summarization stage
        [computetime_fwd_pass, commtime_fwd_pass, commtime_pipeline_bubble] \
            = self.calc_time(inputs, PROMPT_LEN, False , False)

        summary = "FULL CONFIGURATION\n\n" + inputs.parameters.to_string_structured()
        save_as("config_summary.txt", summary)

        file = open("AmpedInference.txt", "w")

        print(f"Stage : 0")
        print(f"Compute time (s) : {computetime_fwd_pass}")
        print(f"Communication time (s) : {commtime_fwd_pass}")
        print(f"Pipeline bubble time (s) : {commtime_pipeline_bubble}\n\n")
        
        tmp = f"Stage,0,Compute time (s),{computetime_fwd_pass},Communication time (s), {commtime_fwd_pass},Pipeline bubble time (s),{commtime_pipeline_bubble}\n"
        file.write(tmp)

        overall_computetime = computetime_fwd_pass
        overall_commtime = commtime_fwd_pass
        overall_pipeline_bubble = commtime_pipeline_bubble
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
        #if inputs.commandline_args.GEMM:
        #    save_GEMM_breakdown(inputs, inputs.commandline_args.compute_graph)
        self.inputs = inputs
        
        save_as("training_time_breakdown.txt", \
            self.temp_string_training_time_breakdown(\
                overall_computetime+overall_commtime+overall_pipeline_bubble,
                overall_computetime, 
                overall_commtime, 
                overall_pipeline_bubble))

class mat_dims_ampedToDF():
    def __init__(self, inputs) -> None:
        self.dims = {}
        self.debug = True
        if self.debug:
            print("Starting mat dims amped to DF script...")
        self.main(inputs)

    def mmm_breakup(self, B, D, S, h, nheads, h_MLP1, h_MLP2, N_DP, N_PP):
        mmm =  {}
        dims = {}
        # deepflow_outputs = {}
        numlevels = 6*(GENERATION_LEN+1)
        levels = ["X.W=KQV", "Q.K=R", "R.V=Z", "Z.W=O", "O.WL1=O1", "O1.WL2=O2"]
        #print("matrix dimensions accounting for all heads & batched dimension")
        S = PROMPT_LEN

        # Summarization
        dims[0]=[int(3*B*S/N_DP/N_PP), D, h*nheads] #factor 3 due to K+Q+V
        dims[1]=[int(B*S/N_DP/N_PP), h*nheads, S] # This seem off !!!
        dims[2]=[int(B*S/N_DP/N_PP), S, h*nheads]
        dims[3]=[int(B*S/N_DP/N_PP), D, D]
        dims[4]=[int(B*S/N_DP/N_PP), D, h_MLP1]
        dims[5]=[int(B*S/N_DP/N_PP), h_MLP1, h_MLP2]

        # Generation
        S = 1
        for i in range(1, GENERATION_LEN):
            dims[0+(6*i)]=[int(3*B*S/N_DP/N_PP), D, h*nheads] #factor 3 due to K+Q+V
            dims[1+(6*i)]=[S, int(B*S*(i+PROMPT_LEN)/N_DP/N_PP), h*nheads]
            dims[2+(6*i)]=[S, h*nheads ,int(B*S*(i+PROMPT_LEN)/N_DP/N_PP)]
            dims[3+(6*i)]=[int(B*S/N_DP/N_PP), D, D]
            dims[4+(6*i)]=[int(B*S/N_DP/N_PP), D, h_MLP1]
            dims[5+(6*i)]=[int(B*S/N_DP/N_PP), h_MLP1, h_MLP2]

        #print("levels:",levels)
        #print("writting the matrix dimensions ...")
        file = open("mat_dims_amped.txt","w")
        #file.write('#'+str(levels)+'\n')
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
    AMPeD = amped_exec()
    # then, mat_dims_ampedToDF
    mat_dims = mat_dims_ampedToDF(AMPeD.inputs)
    # then, run.sh / deepflow exec / perf.py
    DeepFlow = deepflow_exec(AMPeD.inputs, mat_dims.dims)
    # then, cal_time.py
    cal_time(AMPeD.inputs, AMPeD.breakdown, DeepFlow.deepflow_outputs)
