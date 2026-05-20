PROMPT_LEN = 2048
GENERATION_LEN = 0
DEBUG_PRINT = True

import csv


class astrasim_workload:
    def __init__(self, amped, result_dir) -> None:

        # Gather data for astra-sim workload
        # TP comm will occur after layernorm1 and layernorm2 and is ARR
        # PP comm will occue after layernorm2 after TP and is P2P
        # DP comm will occur after PP and is ARR
        self.SUB_VOCAB = 1024
        self.training = True
        self.debug = True
        self.amped = amped
        self.sublayers = [
            "InputEmbeddings",
            "LayerNorm1",
            "X.W=Q",
            "X.W=K",
            "X.W=V",
            "Q.K=U",
            "softmax1",
            "U.V=Y",
            "Concat1",
            "Resudal1",
            "Z.W=O",
            "layernorm1",
            "O.WL1=O1",
            "GELU",
            "O1.WL2=O2",
            "layernorm2",
        ]
        self.sublayers_MGT = [
            "attention_layer",
            "TP_comm_mha",
            "feedForwardNetwork",
            "TP_comm_FFN",
            "DP_comm",
        ]
        self.sublayers_params = [
            "layer",  # Layer name
            "rsvd_var",  # Reserved variable
            "fwd_pass_compute_time",  # compute time in ns
            # communication type ["NONE", "ALLREDUCE", "ALLGATHER"]
            "fwd_pass_comm_type",
            "fwd_pass_comm_size",  # communication volume size in bytes
            "input_grad_compute_time",  # compute time in ns
            # communication type ["NONE", "ALLREDUCE", "ALLGATHER"]
            "input_grad_comm_type",
            "input_grad_comm_size",  # communication volume size in bytes
            "weight_grad_compute_time",  # compute time in ns
            # communication type ["NONE", "ALLREDUCE", "ALLGATHER"]
            "weight_grad_comm_type",
            "weight_grad_comm_size",  # communication volume size in bytes
            "delay_after_collectives",  # Additional delay after each collectives
            "M",  # AP: Forward pass MACs
            "N",  # AP: Inpute gradient MACs
            "K",
        ]  # AP: Weight gradient MACs

        self.layers = []
        self.epoc = {}
        self.resultDir = result_dir
        # self.main()
        self.extractParams()

    def extractParams(self):

        inputs = self.amped.inputs

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

        self.astrasim_mmm_breakup(B, D, S, h, nheads, h_MLP1, h_MLP2, N_DP, N_PP)
        self.astrasim_mmm(B, D, S, h, nheads, h_MLP1, h_MLP2, N_DP, N_PP)

    def tempLayer(self, _name, _M, _N, _K):

        _tDict = {
            "layer": _name,  # Layer name
            "rsvd_var": -1,  # Reserved variable
            "fwd_pass_compute_time": 0,  # compute time in ns
            # communication type ["NONE", "ALLREDUCE", "ALLGATHER"]
            "fwd_pass_comm_type": "NONE",
            "fwd_pass_comm_size": 0,  # communication volume size in bytes
            "input_grad_compute_time": 0,  # compute time in ns
            # communication type ["NONE", "ALLREDUCE", "ALLGATHER"]
            "input_grad_comm_type": "NONE",
            "input_grad_comm_size": 0,  # communication volume size in bytes
            "weight_grad_compute_time": 0,  # compute time in ns
            # communication type ["NONE", "ALLREDUCE", "ALLGATHER"]
            "weight_grad_comm_type": "NONE",
            "weight_grad_comm_size": 0,  # communication volume size in bytes
            "delay_after_collectives": 0,  # Additional delay after each collectives
            "M": _M,  # AP: Forward pass MACs
            "N": _N,  # AP: Inpute gradient MACs
            "K": _K,
        }  # AP: Weight gradient MACs

        return _tDict

    def astrasim_mmm_breakup(self, B, D, S, h, nheads, h_MLP1, h_MLP2, N_DP, N_PP):
        dims = []
        # deepflow_outputs = {}
        if not self.training:
            numlevels = 6 * (GENERATION_LEN + 1)
        else:
            numlevels = 6

        if not self.training:
            S = PROMPT_LEN
        else:
            # Do nothing since S is already context
            pass

        # Summarization/Training
        dims.append(
            self.tempLayer(
                "InputEmbedding", int(B * S / N_DP / N_PP), self.SUB_VOCAB, D
            )
        )
        dims.append(
            self.tempLayer(
                "LayerNorm1",
                int(B * S / N_DP / N_PP),
                1,
                D,
            )
        )
        dims.append(
            self.tempLayer(
                "Qi",
                int(B * S / N_DP / N_PP),
                D,
                h * nheads,
            )
        )
        dims.append(
            self.tempLayer(
                "Ki",
                int(B * S / N_DP / N_PP),
                D,
                h * nheads,
            )
        )
        dims.append(
            self.tempLayer(
                "Ki",
                int(B * S / N_DP / N_PP),
                D,
                h * nheads,
            )
        )
        dims.append(
            self.tempLayer(
                "Ui",
                int(B * S / N_DP / N_PP),
                h * nheads,
                S,
            )
        )
        dims.append(
            self.tempLayer(
                "Yi",
                int(B * S / N_DP / N_PP),
                S,
                h * nheads,
            )
        )
        dims.append(
            self.tempLayer(
                "Concat",
                int(B * S / N_DP / N_PP),
                h * nheads,
                D,
            )
        )

        dims.append(
            self.tempLayer(
                "Residual1",
                int(B * S / N_DP / N_PP),
                1,
                D,
            )
        )
        dims.append(
            self.tempLayer(
                "LayerNorm2",
                int(B * S / N_DP / N_PP),
                1,
                D,
            )
        )

        dims.append(
            self.tempLayer(
                "FFN1",
                int(B * S / N_DP / N_PP),
                D,
                h_MLP1,
            )
        )
        dims.append(
            self.tempLayer(
                "FFN2",
                int(B * S / N_DP / N_PP),
                h_MLP1,
                h_MLP2,
            )
        )
        dims.append(
            self.tempLayer(
                "Residual2",
                int(B * S / N_DP / N_PP),
                1,
                h * nheads,
            )
        )
        dims.append(
            self.tempLayer(
                "OutputEmbedding",
                int(B * S / N_DP / N_PP),
                D,
                self.SUB_VOCAB,
            )
        )

        if self.debug:
            print("writting the matrix dimensions ...")

        print(
            f"Saving astrasim compatible workload file at {self.resultDir}/{self.amped.timeStamp}mat_dims_astrasim.txt"
        )
        fileName = f"{self.resultDir}/{self.amped.timeStamp}mat_dims_astrasim.csv"
        # file.write('#'+str(levels)+'\n')

        with open(fileName, "w", newline="") as file:
            writer = csv.writer(file)
            for _cLayer in dims:
                if self.debug:
                    print(f"Gathered {_cLayer}")

                # tmp = str(mmm[i]).replace("[", "").replace("]", "").replace(",", " ")
                # print(tmp)
                # file.write(tmp + "\n")
                writer.writerow(_cLayer.values())

        self.dims = dims


    def astrasim_mmm(self, B, D, S, h, nheads, h_MLP1, h_MLP2, N_DP, N_PP):
        dims = []
        # deepflow_outputs = {}
        if not self.training:
            numlevels = 6 * (GENERATION_LEN + 1)
        else:
            numlevels = 6

        if not self.training:
            S = PROMPT_LEN
        else:
            # Do nothing since S is already context
            pass

        # Summarization/Training
        dims.append(
            self.tempLayer(
                "InputEmbedding", int(B * S), self.SUB_VOCAB, D
            )
        )
        dims.append(
            self.tempLayer(
                "LayerNorm1",
                int(B * S),
                1,
                D,
            )
        )
        dims.append(
            self.tempLayer(
                "Qi",
                int(B * S),
                D,
                h * nheads,
            )
        )
        dims.append(
            self.tempLayer(
                "Ki",
                int(B * S),
                D,
                h * nheads,
            )
        )
        dims.append(
            self.tempLayer(
                "Ki",
                int(B * S),
                D,
                h * nheads,
            )
        )
        dims.append(
            self.tempLayer(
                "Ui",
                int(B * S),
                h * nheads,
                S,
            )
        )
        dims.append(
            self.tempLayer(
                "Yi",
                int(B * S),
                S,
                h * nheads,
            )
        )
        dims.append(
            self.tempLayer(
                "Concat",
                int(B * S),
                h * nheads,
                D,
            )
        )

        dims.append(
            self.tempLayer(
                "Residual1",
                int(B * S),
                1,
                D,
            )
        )
        dims.append(
            self.tempLayer(
                "LayerNorm2",
                int(B * S),
                1,
                D,
            )
        )

        dims.append(
            self.tempLayer(
                "FFN1",
                int(B * S),
                D,
                h_MLP1,
            )
        )
        dims.append(
            self.tempLayer(
                "FFN2",
                int(B * S),
                h_MLP1,
                h_MLP2,
            )
        )
        dims.append(
            self.tempLayer(
                "Residual2",
                int(B * S),
                1,
                h * nheads,
            )
        )
        dims.append(
            self.tempLayer(
                "OutputEmbedding",
                int(B * S),
                D,
                self.SUB_VOCAB,
            )
        )

        if self.debug:
            print("writting the matrix dimensions ...")

        print(
            f"Saving astrasim compatible workload file at {self.resultDir}/{self.amped.timeStamp}mat_dims_astrasim.txt"
        )
        fileName = f"{self.resultDir}/{self.amped.timeStamp}mat_dims_astrasim_all.csv"
        # file.write('#'+str(levels)+'\n')

        with open(fileName, "w", newline="") as file:
            writer = csv.writer(file)
            for _cLayer in dims:
                if self.debug:
                    print(f"Gathered {_cLayer}")

                # tmp = str(mmm[i]).replace("[", "").replace("]", "").replace(",", " ")
                # print(tmp)
                # file.write(tmp + "\n")
                writer.writerow(_cLayer.values())

        self.dims = dims

    def main(self):

        for sublayer in self.sublayers:
            for param in self.sublayers_params:
                if (
                    param == "layer"
                    or param == "fwd_pass_comm_type"
                    or param == "input_grad_comm_type"
                    or param == "weight_grad_comm_type"
                ):
                    # self.epoc[sublayer][param] = ''
                    pass
                else:
                    # self.epoc[sublayer][param] = 0
                    pass

    def MGT_layer_update(self, inputs, perf):
        MHA_macs = (
            inputs.parameters["total_attention_sublayer_MAC_operations"]
            * inputs.parameters["attention_heads"]
        )

        self.single_layer_update(
            layer="attention_layer",
            rsvd_var=-1,
            fwd_pass_compute_time=self.amped_compute_time(
                sparam=inputs.parameters["weight_precision"],
                MACs=MHA_macs,
                cmac=perf.reciprocal_of_OPS(),
                wfumac=perf.W_FU_MAC,
                TP_degree=inputs.parameters["tensor_parallel_degree"],
            ),  # will come from DeepFlow
            fwd_pass_comm_type="None",
            fwd_pass_comm_size=0,
            input_grad_compute_time=self.amped_compute_time(
                sparam=inputs.parameters["gradient_precision"],
                MACs=MHA_macs,
                cmac=perf.reciprocal_of_OPS(),
                wfumac=perf.W_FU_MAC,
                TP_degree=inputs.parameters["tensor_parallel_degree"],
            ),
            input_grad_comm_type="None",
            input_grad_comm_size=0,
            weight_grad_compute_time=(MHA_macs * perf.reciprocal_of_OPS())
            / inputs.parameters["tensor_parallel_degree"],
            weight_grad_comm_type="None",
            weight_grad_comm_size=0,
            delay_after_collectives=10,
            fwd_pass_MACs=MHA_macs,
            input_grad_MACs=MHA_macs,
            weight_grad_MACs=MHA_macs,
        )

        FWD_PASS_COMM_VOLUME = (
            (
                2
                * perf.p["activations_volume_per_layer_batch"]
                / perf.p["number_of_nodes_required"]
            )
            if perf.p["intra_node_tensor_parallel_degree"] > 1
            else 0
        )

        self.single_layer_update(
            layer="TP_comm_MHA",
            rsvd_var=-1,
            fwd_pass_compute_time=0,  # will come from DeepFlow
            fwd_pass_comm_type="ALLREDUCE",
            fwd_pass_comm_size=2
            * perf.p["activations_volume_per_layer_batch"]
            / perf.p["number_of_nodes_required"],
            input_grad_compute_time=0,
            input_grad_comm_type="ALLREDUCE",
            input_grad_comm_size=0,
            weight_grad_compute_time=0,
            weight_grad_comm_type=0,
            weight_grad_comm_size=0,
            delay_after_collectives=10,
            fwd_pass_MACs=0,
            input_grad_MACs=0,
            weight_grad_MACs=0,
        )

        FFN_macs = (
            inputs.parameters["total_MLP_sublayer_MAC_operations"]
            * inputs.parameters["attention_heads"]
        )
        self.single_layer_update(
            layer="feedForwardNetwork",
            rsvd_var=-1,
            fwd_pass_compute_time=self.amped_compute_time(
                sparam=inputs.parameters["weight_precision"],
                MACs=FFN_macs,
                cmac=perf.reciprocal_of_OPS(),
                wfumac=perf.W_FU_MAC,
                TP_degree=inputs.parameters["tensor_parallel_degree"],
            ),  # will come from DeepFlow
            fwd_pass_comm_type="None",
            fwd_pass_comm_size=0,
            input_grad_compute_time=self.amped_compute_time(
                sparam=inputs.parameters["gradient_precision"],
                MACs=FFN_macs,
                cmac=perf.reciprocal_of_OPS(),
                wfumac=perf.W_FU_MAC,
                TP_degree=inputs.parameters["tensor_parallel_degree"],
            ),
            input_grad_comm_type="None",
            input_grad_comm_size=0,
            weight_grad_compute_time=(FFN_macs * perf.reciprocal_of_OPS())
            / inputs.parameters["tensor_parallel_degree"],
            weight_grad_comm_type="None",
            weight_grad_comm_size=0,
            delay_after_collectives=10,
            fwd_pass_MACs=FFN_macs,
            input_grad_MACs=FFN_macs,
            weight_grad_MACs=FFN_macs,
        )

        self.single_layer_update(
            layer="TP_comm_FFN",
            rsvd_var=-1,
            fwd_pass_compute_time=0,  # will come from DeepFlow
            fwd_pass_comm_type="ALLREDUCE",
            fwd_pass_comm_size=0,
            input_grad_compute_time=0,
            input_grad_comm_type="ALLREDUCE",
            input_grad_comm_size=0,
            weight_grad_compute_time=0,
            weight_grad_comm_type=0,
            weight_grad_comm_size=0,
            delay_after_collectives=10,
            fwd_pass_MACs=MHA_macs,
            input_grad_MACs=MHA_macs,
            weight_grad_MACs=MHA_macs,
        )

        self.single_layer_update(
            layer="DP_comm",
            rsvd_var=-1,
            fwd_pass_compute_time=0,  # will come from DeepFlow
            fwd_pass_comm_type="NONE",
            fwd_pass_comm_size=0,
            input_grad_compute_time=0,
            input_grad_comm_type="NONE",
            input_grad_comm_size=0,
            weight_grad_compute_time=0,
            weight_grad_comm_type="ALLREDUCE",
            weight_grad_comm_size=0,
            delay_after_collectives=10,
            fwd_pass_MACs=MHA_macs,
            input_grad_MACs=MHA_macs,
            weight_grad_MACs=MHA_macs,
        )

        pass

    def amped_compute_time(self, sparam, MACs, cmac, wfumac, TP_degree):
        return ((MACs * cmac * (sparam / wfumac)) / TP_degree) * (10**9)

    def layer_update(self, inputs, perf):

        query_macs = (
            inputs.parameters["query_MAC_operations"]
            * inputs.parameters["attention_heads"]
        )
        self.single_layer_update(
            layer="X.W=Q",
            rsvd_var=-1,
            fwd_pass_compute_time=self.amped_compute_time(
                sparam=inputs.parameters["weight_precision"],
                MACs=query_macs,
                cmac=perf.reciprocal_of_OPS(),
                wfumac=perf.W_FU_MAC,
                TP_degree=inputs.parameters["tensor_parallel_degree"],
            ),  # will come from DeepFlow
            fwd_pass_comm_type="None",
            fwd_pass_comm_size=0,
            input_grad_compute_time=self.amped_compute_time(
                sparam=inputs.parameters["gradient_precision"],
                MACs=query_macs,
                cmac=perf.reciprocal_of_OPS(),
                wfumac=perf.W_FU_MAC,
                TP_degree=inputs.parameters["tensor_parallel_degree"],
            ),
            input_grad_comm_type="None",
            input_grad_comm_size=0,
            weight_grad_compute_time=(query_macs * perf.reciprocal_of_OPS())
            / inputs.parameters["tensor_parallel_degree"],
            weight_grad_comm_type=0,
            weight_grad_comm_size=0,
            delay_after_collectives=10,
            fwd_pass_MACs=query_macs,
            input_grad_MACs=query_macs,
            weight_grad_MACs=query_macs,
        )

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

    def single_layer_update(
        self,
        layer,
        rsvd_var,
        fwd_pass_compute_time,
        fwd_pass_comm_type,
        fwd_pass_comm_size,
        input_grad_compute_time,
        input_grad_comm_type,
        input_grad_comm_size,
        weight_grad_compute_time,
        weight_grad_comm_type,
        weight_grad_comm_size,
        delay_after_collectives,
        fwd_pass_MACs,
        input_grad_MACs,
        weight_grad_MACs,
    ):
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
