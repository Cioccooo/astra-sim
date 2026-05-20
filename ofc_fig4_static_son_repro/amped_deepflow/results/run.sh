#!/bin/bash

TP_intra=$(grep "intra_node_tensor_parallel_degree" $1 | awk -F ':' '{gsub(/[^0-9]/, "", $2); print $2}' | tr -d '[:space:]')
echo "TP intra degree: $TP_intra"

# Input data file
input_file="/imec/scratch/dtpatha/patel23/DeepFlow/scripts/amped_scripts/mat_dims_amped.txt"

# Relevant directories
RUNDIR="/imec/scratch/dtpatha/patel23/DeepFlow"
CONFIG_DIR="/imec/scratch/dtpatha/patel23/DeepFlow/configs/new-configs"
OUTDIR="/imec/scratch/dtpatha/patel23/DeepFlow/results/output"

# Remove the old files
rm -rf "$OUTDIR"/LLM/*

# Read lines from the input file
while IFS= read -r line; do
    # Extract values from the line
    value1=$(echo "$line" | awk '{print $1}')
    value2=$(echo "$line" | awk '{print $2}')
    value3=$(echo "$line" | awk '{print $3}')

    # Submit a job using python perf.py with extracted values as arguments
    
    if (( value1 < TP_intra ))
    then
        python "$RUNDIR"/perf.py --exp_config "$CONFIG_DIR"/v100.yaml --exp_dir "$OUTDIR"/LLM/ --debug True --gemm True --t RC --kp1 1 --kp2 $TP_intra --m "$value1" --n "$value2" --k "$value3" >> DeepFlow_results.txt
    else
        python "$RUNDIR"/perf.py --exp_config "$CONFIG_DIR"/v100.yaml --exp_dir "$OUTDIR"/LLM/ --debug True --gemm True --t RC --kp1 $TP_intra --kp2 1 --m "$value1" --n "$value2" --k "$value3" >> DeepFlow_results.txt
    fi;
    
    #python "$RUNDIR"/perf.py --exp_config "$CONFIG_DIR"/v100.yaml --exp_dir "$OUTDIR"/LLM/ --debug True --gemm True --t CR --kp1 $TP_intra --kp2 1 --m "$value1" --n "$value2" --k "$value3"

    # You can add sleep between job submissions if needed
    # sleep 1
done < "$input_file"
