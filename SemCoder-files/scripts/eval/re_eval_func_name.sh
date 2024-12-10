#!/bin/bash

###################################################################################
# Hardware: 1x A6000 48GB GPU, or any other GPUs with at least 48GB memory
# Note: We use the default hyperparameters provided by the corresponding benchmark.
# To reproduce the results reported in the paper, do not change it.
###################################################################################

# export CUDA_VISIBLE_DEVICES=0
# export PYTHONPATH=$PYTHONPATH:/home/rc3593/SemCoder

CRUXEVAL_HOME="/home/rc3593/cruxeval"
SEMCODER_HOME=$(pwd)
MODEL=semcoder/semcoder_s_1030 # semcoder/semcoder_s_1030

cd $CRUXEVAL_HOME/evaluation;

root_dir="${SEMCODER_HOME}/output_dir/func_name/cruxeval"

model_name=$(basename $MODEL)

for method_dir in "${root_dir}"/*/; do
    if [ -d "${method_dir}" ]; then
        method_name=$(basename "${method_dir}")
        echo "Re-evaluating CRUXEval-I for ${method_name}"
        OPT_BASE="${root_dir}/${method_name}/cruxeval_input"
        
        direct_pred_dir=${OPT_BASE}/${model_name}_direct
        monologue_pred_dir=${OPT_BASE}/${model_name}_monologue

        echo "Evaluating results: direct prediction..."
        
        python evaluate_generations.py \
            --generations_path ${direct_pred_dir}/generations.json \
            --scored_results_path ${direct_pred_dir}/scored_results.json \
            --mode input \
            --method "${method_name}" \
            2>&1 | tee ${direct_pred_dir}/eval.log
        
        echo "Evaluating results: monologue prediction..."
        
        python evaluate_generations.py \
            --generations_path ${monologue_pred_dir}/generations.json \
            --scored_results_path ${monologue_pred_dir}/scored_results.json \
            --mode input \
            --method "${method_name}" \
            2>&1 | tee ${monologue_pred_dir}/eval.log

        echo "Re-evaluating CRUXEval-O for ${method_name}"
        OPT_BASE="${root_dir}/${method_name}/cruxeval_output"

        direct_pred_dir=${OPT_BASE}/${model_name}_direct
        monologue_pred_dir=${OPT_BASE}/${model_name}_monologue

        echo "Evaluating results: direct prediction..."
        
        python evaluate_generations.py \
            --generations_path ${direct_pred_dir}/generations.json \
            --scored_results_path ${direct_pred_dir}/scored_results.json \
            --mode output \
            --method "${method_name}" \
            2>&1 | tee ${direct_pred_dir}/eval.log
        
        echo "Evaluating results: monologue prediction..."
        
        python evaluate_generations.py \
            --generations_path ${monologue_pred_dir}/generations.json \
            --scored_results_path ${monologue_pred_dir}/scored_results.json \
            --mode output \
            --method "${method_name}" \
            2>&1 | tee ${monologue_pred_dir}/eval.log
    fi
done
