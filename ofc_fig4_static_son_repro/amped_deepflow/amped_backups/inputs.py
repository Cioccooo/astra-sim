import argparse
import json
import os
import pkg_resources

from amped.common import CalculateFunction, ConfigType, ParametersDict, ParameterCategory, LookupTablesType
from amped.units import parse_quantity

# 放在 import 后、calculate_functions 定义之前
EFFECTIVE_PERF_KEYS = (
    "effective_perf_perc_K_Q_V",
    "effective_perf_perc_attention",
    "effective_perf_perc_output",
    "effective_perf_perc_MLP",
)

# This dictionary maps parameters to be calculated to the function that calculates them.
calculate_functions: dict[str, CalculateFunction] = {
    
    "minibatch_size":
        lambda p: p["batch_size"] / p["data_parallel_degree"],
    "microbatch_size":
        lambda p: p["minibatch_size"] / p["number_of_microbatches_per_minibatch"],
    "hidden_layer_dimension_MLP_1":
        lambda p: p["dimensionality"],
    "activations_volume_per_layer_batch":
        lambda p: p["batch_size"] * p["context"] * p["hidden_layer_dimension_for_attention_sublayers"],
    "number_of_batches":
        lambda p: p["samples"] / p["batch_size"],
    "samples":
        lambda p: p["tokens_to_train"] / p["context"],
    "query_MAC_operations":
        lambda p: p["batch_size"] * p["context"] * p["hidden_layer_dimension_for_attention_sublayers"] * p["dimensionality"],
    "key_MAC_operations":
        lambda p: p["batch_size"] * p["context"] * p["hidden_layer_dimension_for_attention_sublayers"] * p["dimensionality"],
    "value_MAC_operations":
        lambda p: p["batch_size"] * p["context"] * p["hidden_layer_dimension_for_attention_sublayers"] * p["dimensionality"],
    "self_attention_MAC_operations":
        lambda p: 2 * p["batch_size"] * p["context"] * p["summarization_len"] * p["hidden_layer_dimension_for_attention_sublayers"],
    "multihead_MAC_operations":
        lambda p: p["attention_heads"] * (p["query_MAC_operations"] + p["key_MAC_operations"] + p["value_MAC_operations"] + p["self_attention_MAC_operations"]),
    "attention_sublayer_output_MAC_operations":
        lambda p: p["batch_size"] * p["context"] * p["dimensionality"] ** 2,
    "total_attention_sublayer_MAC_operations":
        lambda p: p["multihead_MAC_operations"] + p["attention_sublayer_output_MAC_operations"],
    "total_MLP_sublayer_MAC_operations":
        lambda p: 2 * p["batch_size"] * p["context"] * p["hidden_layer_dimension_MLP_1"] * p["hidden_layer_dimension_MLP_2"],
    "non_linear_operations_for_attention_sublayer":
        lambda p: p["batch_size"] * p["context"] * p["hidden_layer_dimension_for_attention_sublayers"] * p["attention_heads"],
    "non_linear_operations_for_MLP_sublayer":
        lambda p: 2 * p["batch_size"] * (p["hidden_layer_dimension_MLP_1"] + p["hidden_layer_dimension_MLP_2"]),
    "error_volume_per_layer_batch":
        lambda p: p["batch_size"] * p["context"] * p["hidden_layer_dimension_for_attention_sublayers"],
    "dim_query_weight":
        lambda p: p["dimensionality"],
    "dim_key_weight":
        lambda p: p["dimensionality"],
    "dim_value_weight":
        lambda p: p["dimensionality"],
    "projection_weight":
        lambda p: p["dimensionality"],
    "feedforward_bias_1":
        lambda p: p["dimensionality"],
    "feedforward_bias_2":
        lambda p: p["dimensionality"],
    "feedforward_scale_1":
        lambda p: p["dimensionality"],
    "feedforward_scale_2":
        lambda p: p["dimensionality"],
    "number_of_parameters_per_layer":
        lambda p: (
            (p["dim_query_weight"] * p["key_query_value_length"] * p["attention_heads"]
             + p["dim_key_weight"] * p["key_query_value_length"] * p["attention_heads"]
             + p["dim_value_weight"] * p["key_query_value_length"] * p["attention_heads"]
             + p["projection_weight"] * p["key_query_value_length"] * p["attention_heads"]
             + p["feedforward_bias_1"] + p["feedforward_bias_2"] + p["feedforward_scale_1"]
             + p["feedforward_scale_2"] + 2 * p["hidden_layer_dimension_MLP_2"] * p["hidden_layer_dimension_MLP_1"])
            + p["embedding_layer_parameters"] / p["layers"]
        ),
    "number_of_parameters_per_expert":
        lambda p: p["number_of_parameters_per_layer"] * p["layers"],
    "number_of_MoE_layers":
        lambda p: p["layers"] // 2,
    "expert_flag":
        lambda p: int(p["number_of_experts"] > 1),
    "number_of_gating_parameters":
        lambda p: p["expert_flag"] * p["dimensionality"] * p["number_of_experts"] * p["number_of_MoE_layers"],
    "number_of_expert_parameters":
        lambda p: (p["number_of_experts"] - 1) * p["number_of_MoE_layers"] * 2 * p["hidden_layer_dimension_MLP_1"] * p["hidden_layer_dimension_MLP_2"],
    "gating_MAC_operations":
        lambda p: p["context"] * (p["dimensionality"] - 1) * p["batch_size"] * p["expert_flag"] * p["number_of_experts"] * p["number_of_MoE_layers"],
    "total_number_of_parameters_without_embedding":
        lambda p: p["number_of_parameters_per_layer"] * p["layers"] + p["number_of_gating_parameters"] + p["number_of_expert_parameters"],
    "embedding_layer_parameters":
        lambda p: p["number_of_tokens_in_vocabulary"] * p["dimensionality"],
    "decoder_NMACs":
        lambda p: p["batch_size"] * p["number_of_tokens_in_vocabulary"] * p["dimensionality"] * p["decoder_flag"],
    "decoder_flag":
        lambda p: p["minibatch_size"] * p["embedding_layer_parameters"] * p["query_MAC_operations"],

    "number_of_microbatches_per_minibatch":
        lambda p: p["pipeline_parallel_degree"],
    "accelerators_per_node_required":
        lambda p: p["intra_node_data_parallel_degree"] * p["intra_node_tensor_parallel_degree"] * p["intra_node_pipeline_parallel_degree"],
    "data_parallel_degree":
        lambda p: p["intra_node_data_parallel_degree"] * p["inter_node_data_parallel_degree"],
    "tensor_parallel_degree":
        lambda p: p["intra_node_tensor_parallel_degree"] * p["inter_node_tensor_parallel_degree"],
    "pipeline_parallel_degree":
        lambda p: p["intra_node_pipeline_parallel_degree"] * p["inter_node_pipeline_parallel_degree"],
    "number_of_accelerators_required":
        lambda p: p["data_parallel_degree"] * p["tensor_parallel_degree"] * p["pipeline_parallel_degree"],
    "number_of_nodes_required":
        lambda p: p["number_of_accelerators_required"] // p["accelerators_per_node_required"],
    "required_memory_per_accelerator_weights":
        lambda p: (
            (p["weight_precision"] + p["gradient_precision"] + 3 * p["optimizer_state_precision"])
            * p["total_number_of_parameters_without_embedding"]
            / (8 * p["data_parallel_degree"] * p["tensor_parallel_degree"] * p["pipeline_parallel_degree"]) / 1000000000
        ),
    "required_memory_per_accelerator_activations":
        lambda p: (
            p["activation_precision"] * p["layers"] *
            (p["non_linear_operations_for_attention_sublayer"] + p["non_linear_operations_for_MLP_sublayer"]
             + 2 * p["error_volume_per_layer_batch"])
            / p["number_of_microbatches_per_minibatch"] / p["data_parallel_degree"]
            / (8 * p["data_parallel_degree"] * p["tensor_parallel_degree"] * p["pipeline_parallel_degree"]) / 1000000000
        ),
    "total_required_memory_per_accelerator":
        lambda p: p["required_memory_per_accelerator_weights"] + p["required_memory_per_accelerator_activations"],
    "min_required_memory_per_accelerator":
        lambda p: (
            (p["weight_precision"] + p["gradient_precision"] / 2) * p["total_number_of_parameters_without_embedding"]
            / (8 * p["data_parallel_degree"] * p["tensor_parallel_degree"] * p["pipeline_parallel_degree"]) / 1000000000
            + p["required_memory_per_accelerator_activations"]
        ),
    "tile_block_size":
        lambda p: 1,

    "number_of_network_cards_per_node":
        lambda p: p["accelerators_per_node_required"],
    "inter_node_bandwidth":
        lambda p: p["number_of_network_cards_per_node"] * p["bandwidth_per_network_card"],
    "total_intra_node_bandwidth":
        lambda p: p["inter_accelerator_bandwidth"] * p["number_of_accelerators_per_node"],
    "number_of_nodes":
        lambda p: p["number_of_accelerators_required"] / p["number_of_accelerators_per_node"],
    "total_number_of_accelerators":
        lambda p: p["number_of_accelerators_per_node"] * p["number_of_nodes"],
    "compute_intensity_K_Q_V":
        lambda p: p["dimensionality"] / (p["dimensionality"] / p["tile_block_size"] + 1) / 2,
    "compute_intensity_self_attention":
        lambda p: p["microbatch_size"] * p["context"] / (p["microbatch_size"] * p["context"] / p["tile_block_size"] + 1) / 2,
    "compute_intensity_attention_output":
        lambda p: p["dimensionality"] / (p["dimensionality"] / p["tile_block_size"] + 1) / 2,
    "compute_intensity_MLP":
        lambda p: p["hidden_layer_dimension_MLP_1"] / (p["hidden_layer_dimension_MLP_1"] / p["tile_block_size"] + 1) / (p["activation_precision"] / 8),
    "main_mem_bw_A100":
        lambda p: 350,
    "compute_intensity_weight_updates":
        lambda p: 1/3,
    "compute_intensity_NonLin_ops":
        lambda p: 0,
    "parallelization_degree_K_Q_V":
        lambda p: (
            p["microbatch_size"] * p["context"] * p["hidden_layer_dimension_for_attention_sublayers"]
            / p["tile_block_size"] / p["tile_block_size"]
        ),
    "parallelization_degree_attention":
        lambda p: (
            p["microbatch_size"] * p["context"] * p["hidden_layer_dimension_for_attention_sublayers"]
            / p["tile_block_size"] / p["tile_block_size"]
        ),
    "parallelization_degree_output":
        lambda p: (
            p["microbatch_size"] * p["context"] * p["dimensionality"] / p["bigger_tile_block_size"]
            / p["bigger_tile_block_size"]
        ),
    "parallelization_degree_MLP":
        lambda p: (
            p["microbatch_size"] * p["context"] * p["hidden_layer_dimension_MLP_2"] / p["bigger_tile_block_size"]
            / p["bigger_tile_block_size"]
        ),
    "effective_perf_perc_K_Q_V":
        lambda p: 1 if p["parallelization_degree_K_Q_V"] > 100 else p["parallelization_degree_K_Q_V"] / 100,
    "effective_perf_perc_attention":
        lambda p: 1 if p["parallelization_degree_attention"] > 100 else p["parallelization_degree_attention"] / 100,
    "effective_perf_perc_output":
        lambda p: 1 if p["parallelization_degree_output"] > 100 else p["parallelization_degree_output"] / 100,
    "effective_perf_perc_MLP":
        lambda p: 1 if p["parallelization_degree_MLP"] > 100 else p["parallelization_degree_MLP"] / 100,

    "peak_compute_8bit_MACs":
        lambda p: p["frequency"] * p["number_of_cores"] * p["functional_units_per_core"] * p["functional_unit_hardware_8bit_MAC_per_cycle"],
    "peak_compute_8bit_OPS":
        lambda p: p["peak_compute_8bit_MACs"] * p["OPS_per_MAC"],
    "peak_compute_8bit_NLINs":
        lambda p: (
            p["frequency"] * p["number_of_cores"] * p["non_linear_functional_units_per_core"]
            * p["non_linear_functional_unit_hardware_8bit_NLIN_per_cycle"]
        ),
}


class Inputs:
    """Processes the inputs to the model.

    Has property 'parameters' which contains all the calculated and user-supplied parameters.
    """

    def __init__(self, lookup_overrides: dict[str, str] = None, config_override: ConfigType = None, config_override_path: str = None):
        """Creates a new Inputs object which contains all parameters necessary for computation.

        :param lookup_overrides: a dict with lookup table overrides e.g. {"transformer_network_parameters": "GPT-3 XL", "accelerator_specifications": "V100"}.
            If left empty, the lookup tables specified in the config file will be used.
        :param config_override: a ConfigType dict to be used instead of reading and using config.json
        :param config_override_path: a filepath for a config file to be used instead of config.json
        """

        commandline_arg_parser = argparse.ArgumentParser(
            description='An Analytical Model for Performance in Distributed Training of Transformers',
            epilog='Check README.md for more info.'
        )
        commandline_arg_parser.add_argument(
            "--config", required=False, metavar="<path>", type=str,
            help="Specify filepath for a config to be used instead of config.json"
        )
        commandline_arg_parser.add_argument(
            "--GEMM", required=False, action="store_true", help="Save a GEMM breakdown in gemm_breakdown.txt"
        )
        commandline_arg_parser.add_argument(
            "--compute_graph", required=False, action="store_true",
            help="Save a compute graph in visual_graph.txt (--GEMM must be present)"
        )
        self.commandline_args = commandline_arg_parser.parse_args()

        if config_override is None:
            if config_override_path:
                config_path = config_override_path
            elif self.commandline_args.config:
                config_path = self.commandline_args.config
            else:
                config_path = "config.json"
            config_path = pkg_resources.resource_filename(__name__, config_path)
            with open(config_path, "r") as config_file:
                config: ConfigType = json.load(config_file)
        else:
            config = config_override
        self.config = config

        self.override_lookups(lookup_overrides)

        with open(pkg_resources.resource_filename(__name__, "lookup_tables.json"), "r") as lookup_tables_file:
            lookup_tables: LookupTablesType = json.load(lookup_tables_file)

        parameters_dict: ParametersDict = {}  # a dict for aggregating all the parameters

        parameters_to_calculate: list[str] = []  # a list which will contain the names of the parameters that must be calculated

        for parameter_category_name, parameter_category in config.items():
            for parameter_name, parameter_object in parameter_category["parameters"].items():
                if parameter_object.get("from_lookup_table") is True:  # 1
                    if "lookup_config" not in parameter_category:
                        raise Exception(f"Parameter {parameter_name} gets value from lookup table, "
                                        f"but {parameter_category_name} has no lookup config!")

                    lookup_config = parameter_category["lookup_config"]
                    lookup_table_name = lookup_config["lookup_table_name"]
                    lookup_table_row = lookup_config["lookup_table_row"]
                    lookup_property_name = parameter_name
                    if "lookup_name" in parameter_object:
                        lookup_property_name = parameter_object["lookup_name"]

                    if lookup_table_name not in lookup_tables:
                        raise Exception(f"lookup_tables.json doesn't have lookup table '{lookup_table_name}'!")
                    if lookup_table_row not in lookup_tables[lookup_table_name]:
                        raise Exception(f"Lookup table '{lookup_table_name}' in lookup_tables.json doesn't have element '{lookup_table_row}'!")
                    if lookup_property_name not in lookup_tables[lookup_table_name][lookup_table_row]:
                        raise Exception(
                            f"Element '{lookup_table_row}' of lookup table '{lookup_table_name}' in lookup_tables.json "
                            f"doesn't have property '{lookup_property_name}'!"
                        )

                    parameters_dict[parameter_name] = parse_quantity(
                        lookup_tables[lookup_table_name][lookup_table_row][lookup_property_name]
                    )

                elif parameter_object.get("calculated") is True:  # 2
                    parameters_to_calculate.append(parameter_name)
                    if parameter_name not in calculate_functions:
                        raise Exception(f"Parameter to be calculated '{parameter_name}' does not have a function that "
                                        f"calculates it in 'calculate_functions' in inputs.py!")

                else:  # 3
                    if "value" not in parameter_object:
                        raise Exception(f"Parameter '{parameter_name}' does not have a 'value' property in config.json")
                    if parameter_object["value"] is None:
                        raise Exception(
                            f"Parameter '{parameter_name}' has null as value in config.json, "
                            f"but is not calculated or looked up in a table!"
                        )
                    parameters_dict[parameter_name] = parse_quantity(parameter_object["value"])

        # --- Guard: respect manual efficiency overrides from config.json ---
        # 如果用户在 JSON 里把 effective_perf_perc_* 标为 calculated:false 且给了 value，
        # 则强制采用该数值，并从计算图中移除这几个参数，避免被内部公式重算覆盖。
        try:
            eff_cat = config["system_architecture_parameters"]["parameters"]
            for k in EFFECTIVE_PERF_KEYS:
                pobj = eff_cat.get(k)
                if pobj is not None and pobj.get("calculated") is False:
                    # 1) 用 JSON 里的 value 覆盖进 parameters_dict
                    if "value" in pobj and pobj["value"] is not None:
                        parameters_dict[k] = parse_quantity(pobj["value"])
                    else:
                        raise Exception(f"Parameter '{k}' is marked calculated:false but has no 'value' in config.json")

                    # 2) 从计算列表里移除，防止再次计算覆盖
                    if k in parameters_to_calculate:
                        parameters_to_calculate.remove(k)
                    if k in calculate_functions:
                        del calculate_functions[k]
        except KeyError:
            # 若结构不在 config（不影响其它参数），忽略
            pass
        # -------------------------------------------------------------------

        # if the following isn't done, the recalculation methods in Parameters will recalculate these parameters
        # which cannot happen because they were set to not be calculated in config.json
        for key in [k for k in calculate_functions if k not in parameters_to_calculate]:
            del calculate_functions[key]

        # AP changes ###########################
        self.temp_parameters_dict = parameters_dict
        self.temp_parameters_to_calculate = parameters_to_calculate

        #######################################

        self.dependency_mapping = CalculateFunctionsDependencyMapping()

        for parameter_name in parameters_to_calculate:
            if parameter_name not in parameters_dict:
                self.calculate_parameter(parameter_name, parameters_dict)

        self.parameters = Parameters(self, parameters_dict, self.dependency_mapping)  # the main property used in other files
        self.transformer = config["neural_network_training_parameters"]["lookup_config"]["lookup_table_row"]
        self.accelerator = config["accelerator_architecture_parameters"]["lookup_config"]["lookup_table_row"]

    def override_lookups(self, lookup_overrides: dict[str, str]):
        if lookup_overrides is None:
            return

        for parameter_category in self.config.values():
            if "lookup_config" in parameter_category:
                table_name = parameter_category["lookup_config"]["lookup_table_name"]
                if table_name in lookup_overrides:
                    parameter_category["lookup_config"]["lookup_table_row"] = lookup_overrides[table_name]

    def calculate_parameter(self, parameter_name: str, parameters_dict: dict[str, int | float]):
        for dependant in self.dependency_mapping.mapping[parameter_name]:
            if dependant not in parameters_dict:
                self.calculate_parameter(dependant, parameters_dict)
        parameters_dict[parameter_name] = calculate_functions[parameter_name](parameters_dict)


class CalculateFunctionsDependencyMapping:
    """Maps calculated parameters onto the parameters their calculation depends on."""

    def __init__(self):
        """Creates a dependency mapping that maps calculated parameters onto the parameters their calculation depends on.
        Is used internally."""

        self.mapping: dict[str, set[str]] = {}  # maps parameter to set of parameters it depends on to be calculated
        self.__current_dependencies: set[str] = set()

        for key, func in calculate_functions.items():
            func(self)
            self.mapping[key] = self.__current_dependencies
            self.__current_dependencies = set()

    def __getitem__(self, key):
        self.__current_dependencies.add(key)
        return 1


class Parameters:
    """Object that contains all the parameters of the AMPeD model."""

    def __init__(self, inputs: Inputs, parameters_dict: ParametersDict, dependency_mapping: CalculateFunctionsDependencyMapping):
        """Creates an object that contains all the parameters of the AMPeD model.

        :param inputs: the Inputs object which created these parameters
        :param parameters_dict: a dictionary mapping each parameter's name to its value
        :param dependency_mapping: the CalculateFunctionsDependencyMapping object
        """

        self.inputs = inputs
        self.parameters_dict = parameters_dict
        self.dependency_mapping = dependency_mapping

    def __getitem__(self, key):
        value = self.parameters_dict.get(key)
        if key is None:
            raise KeyError(f"'{key}' is an unknown parameter!")
        return value

    def __setitem__(self, key, value):
        self.parameters_dict[key] = value
        self.__recalculate_affected_parameters(key)

    def set_multiple(self, *pairs: tuple[str, int | float]):
        """Simultaneously sets multiple parameters.

        This results in faster recalculation than by changing each parameter individually if many parameters are set."""

        for pair in pairs:
            self.parameters_dict[pair[0]] = pair[1]
        self.__recalculate_affected_parameters_multiple({pair[0] for pair in pairs})

    def __recalculate_affected_parameters(self, changed_parameter: str):
        """Recursively recalculates all parameters which depend on the changed parameter given.

        :param changed_parameter: the parameter whose value has changed
        """

        for parameter, dependencies in self.dependency_mapping.mapping.items():
            if changed_parameter in dependencies:
                self.parameters_dict[parameter] = calculate_functions[parameter](self)
                self.__recalculate_affected_parameters(parameter)

    def __recalculate_affected_parameters_multiple(self, changed_parameters: set[str], affected: list[set[str]] = None):
        """Recursively recalculates all parameters which depend on the changed parameters given.

        Tries to minimize total amount of recalculations.
        :param changed_parameters: the set of parameters whose values have changed
        :param affected: [only used in recursive calls] the list of affected parameter sets, in order of dependence
            e.g. parameters in affected[1] depend on those in affected[0]
        """

        if affected is None:
            affected = []

        if len(changed_parameters) == 0:
            for group in affected:
                for affected_parameter in group:
                    self.parameters_dict[affected_parameter] = calculate_functions[affected_parameter](self)
        else:
            next_changed_parameters = set()

            for parameter, dependencies in self.dependency_mapping.mapping.items():  # for each parameter and its dependencies
                if any([c in dependencies for c in changed_parameters]):  # if any of changed_parameters are in dependencies
                    next_changed_parameters.add(parameter)

                    # the following ensures 'parameter' is, in the end, only present in one set inside affected
                    groups_containing_parameter = [group for group in affected if parameter in group]
                    for group in groups_containing_parameter:
                        group.remove(parameter)

            self.__recalculate_affected_parameters_multiple(next_changed_parameters, affected + [next_changed_parameters])

    def to_string_structured(self):
        """Returns a string in which all the parameters and their values are represented per parameter category."""

        s = f"Transformer: {self.inputs.transformer} | Accelerator: {self.inputs.accelerator}\n\n"
        config: ConfigType = self.inputs.config
        i = 1
        for category_name, category in config.items():
            s += (
                f"{i}: " + category_name.replace("_", " ").upper() + "\n"
                + self.__category_table_string(category)
                + "\n\n"
            )
            i += 1
        return s[:-1]

    def __category_table_string(self, category: ParameterCategory):
        longest_key_len = len(max(category["parameters"].keys(), key=len))
        longest_val_len = max([len(str(self.parameters_dict[k])) for k in category["parameters"].keys()])
        table = ""
        for key in category["parameters"]:
            table += f"{key:{longest_key_len+1}}: {self.parameters_dict[key]:{longest_val_len}}"
            description = category['parameters'][key].get('description')
            if description:
                table += f"  # {description}"
            table += "\n"

        return table[:-1]

    def copy(self):
        return Parameters(self.inputs, self.parameters_dict.copy(), self.dependency_mapping)
