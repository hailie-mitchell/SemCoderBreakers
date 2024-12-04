#!/bin/bash

###################################################################################
# Hardware: 1x A6000 48GB GPU, or any other GPUs with at least 48GB memory
# Note: We use the default hyperparameters provided by the corresponding benchmark.
# To reproduce the results reported in the paper, do not change it.
###################################################################################

# SCRIPT USED FOR ABLATION WITH DEEPSEEK-CODER MODELS

export CUDA_VISIBLE_DEVICES=0

CRUXEVAL_HOME="/home/hm3075/cruxeval"
SEMCODER_HOME=$(pwd)
RESULT_INPUT_DIR="${SEMCODER_HOME}/output/cruxeval/func_name/FuncRenameSynonymSub/cruxeval_input"    # UPDATE HERE
RESULT_OUTPUT_DIR="${SEMCODER_HOME}/output/cruxeval/func_name/FuncRenameSynonymSub/cruxeval_output"  # UPDATE HERE                                                                 # UPDATE HERE
# MODEL=semcoder/semcoder_s_1030 # semcoder/semcoder_s_1030
MODEL_1=deepseek-ai/deepseek-coder-6.7b-base
MODEL_2=deepseek-ai/deepseek-coder-6.7b-instruct

########################### 
# CRUXEval-I: run inference
###########################

export PYTHONPATH=$PYTHONPATH:/home/hm3075/SemCoder

model_name=$(basename $MODEL_1)

direct_pred_dir=${RESULT_INPUT_DIR}/${model_name}_direct
monologue_pred_dir=${RESULT_INPUT_DIR}/${model_name}_monologue

mkdir -p ${direct_pred_dir}
mkdir -p ${monologue_pred_dir}

echo "Evaluating model: ${model_name} on CRUXEval-I (direct prediction)..."

python experiments/run_cruxeval.py \
    --model $MODEL_1 \
    --trust_remote_code \
    --tasks input_prediction \
    --batch_size 1 \
    --n_samples 1 \
    --max_length_generation 4096 \
    --precision fp16 \
    --limit 200 \
    --temperature 0.0 \
    --save_generations \
    --save_generations_path ${direct_pred_dir}/results.json \
    --start 0 \
    --end 200 \
    --shuffle \
    --tensor_parallel_size 1

echo "Evaluating model: ${model_name} on CRUXEval-I with SemCoder Monologue..."

python experiments/run_cruxeval.py \
    --model $MODEL_1 \
    --trust_remote_code \
    --tasks input_prediction \
    --batch_size 1 \
    --n_samples 1 \
    --max_length_generation 4096 \
    --precision fp16 \
    --limit 200 \
    --temperature 0.0 \
    --save_generations \
    --save_generations_path ${monologue_pred_dir}/results.json \
    --start 0 \
    --end 200 \
    --shuffle \
    --cot \
    --tensor_parallel_size 1

########################## 
# CRUXEval-I: Report score
##########################

echo "Reporting score for model: ${model_name}..."

python experiments/cruxeval_combine_generations.py --gen_dir ${direct_pred_dir}
python experiments/process_cruxeval.py --task i --gen_dir ${direct_pred_dir}
python experiments/cruxeval_combine_generations.py --gen_dir ${monologue_pred_dir}
python experiments/process_cruxeval.py --task i --gen_dir ${monologue_pred_dir}

cd $CRUXEVAL_HOME/evaluation;

echo "Evaluating results: direct prediction..."

python evaluate_generations.py \
    --generations_path ${direct_pred_dir}/generations.json \
    --scored_results_path ${direct_pred_dir}/scored_results.json \
    --mode input \
    2>&1 | tee ${direct_pred_dir}/eval.log

echo "Evaluating results: monologue prediction..."

python evaluate_generations.py \
    --generations_path ${monologue_pred_dir}/generations.json \
    --scored_results_path ${monologue_pred_dir}/scored_results.json \
    --mode input \
    2>&1 | tee ${monologue_pred_dir}/eval.log

########################### 
# CRUXEval-O: run inference
###########################

cd $SEMCODER_HOME;

model_name=$(basename $MODEL_1)

direct_pred_dir=${RESULT_OUTPUT_DIR}/${model_name}_direct
monologue_pred_dir=${RESULT_OUTPUT_DIR}/${model_name}_monologue

mkdir -p ${direct_pred_dir}
mkdir -p ${monologue_pred_dir}

echo "Evaluating model: ${model_name} on CRUXEval-O (direct prediction)..."

python experiments/run_cruxeval.py \
    --model $MODEL_1 \
    --trust_remote_code \
    --tasks output_prediction \
    --batch_size 1 \
    --n_samples 1 \
    --max_length_generation 4096 \
    --precision fp16 \
    --limit 200 \
    --temperature 0.0 \
    --save_generations \
    --save_generations_path ${direct_pred_dir}/results.json \
    --start 0 \
    --end 200 \
    --shuffle \
    --tensor_parallel_size 1

echo "Evaluating model: ${model_name} on CRUXEval-O with SemCoder Forward Monologue..."

python experiments/run_cruxeval.py \
    --model $MODEL_1 \
    --trust_remote_code \
    --tasks output_prediction \
    --batch_size 1 \
    --n_samples 1 \
    --max_length_generation 4096 \
    --precision fp16 \
    --limit 200 \
    --temperature 0.0 \
    --save_generations \
    --save_generations_path ${monologue_pred_dir}/results.json \
    --start 0 \
    --end 200 \
    --shuffle \
    --cot \
    --tensor_parallel_size 1

########################## 
# CRUXEval-O: Report score
##########################
echo "Reporting score for model: ${model_name}...";

python experiments/cruxeval_combine_generations.py --gen_dir ${direct_pred_dir}
python experiments/process_cruxeval.py --task o --gen_dir ${direct_pred_dir}
python experiments/cruxeval_combine_generations.py --gen_dir ${monologue_pred_dir}
python experiments/process_cruxeval.py --task o --gen_dir ${monologue_pred_dir}

cd $CRUXEVAL_HOME/evaluation;

echo "Evaluating results: direct prediction..."

python evaluate_generations.py \
    --generations_path ${direct_pred_dir}/generations.json \
    --scored_results_path ${direct_pred_dir}/scored_results.json \
    --mode output \
    2>&1 | tee ${direct_pred_dir}/eval.log

echo "Evaluating results: monologue prediction..."

python evaluate_generations.py \
    --generations_path ${monologue_pred_dir}/generations.json \
    --scored_results_path ${monologue_pred_dir}/scored_results.json \
    --mode output \
    2>&1 | tee ${monologue_pred_dir}/eval.log



# REPEAT ALL THAT WITH INSTRUCT MODEL

cd $SEMCODER_HOME;
MODEL=deepseek-ai/deepseek-coder-6.7b-instruct
model_name=$(basename $MODEL_2)

direct_pred_dir=${RESULT_INPUT_DIR}/${model_name}_direct
monologue_pred_dir=${RESULT_INPUT_DIR}/${model_name}_monologue

mkdir -p ${direct_pred_dir}
mkdir -p ${monologue_pred_dir}

echo "Evaluating model: ${model_name} on CRUXEval-I (direct prediction)..."

python experiments/run_cruxeval.py \
    --model $MODEL_2 \
    --trust_remote_code \
    --tasks input_prediction \
    --batch_size 1 \
    --n_samples 1 \
    --max_length_generation 4096 \
    --precision fp16 \
    --limit 200 \
    --temperature 0.0 \
    --save_generations \
    --save_generations_path ${direct_pred_dir}/results.json \
    --start 0 \
    --end 200 \
    --shuffle \
    --tensor_parallel_size 1

echo "Evaluating model: ${model_name} on CRUXEval-I with SemCoder Monologue..."

python experiments/run_cruxeval.py \
    --model $MODEL_2 \
    --trust_remote_code \
    --tasks input_prediction \
    --batch_size 1 \
    --n_samples 1 \
    --max_length_generation 4096 \
    --precision fp16 \
    --limit 200 \
    --temperature 0.0 \
    --save_generations \
    --save_generations_path ${monologue_pred_dir}/results.json \
    --start 0 \
    --end 200 \
    --shuffle \
    --cot \
    --tensor_parallel_size 1

########################## 
# CRUXEval-I: Report score
##########################

echo "Reporting score for model: ${model_name}..."

python experiments/cruxeval_combine_generations.py --gen_dir ${direct_pred_dir}
python experiments/process_cruxeval.py --task i --gen_dir ${direct_pred_dir}
python experiments/cruxeval_combine_generations.py --gen_dir ${monologue_pred_dir}
python experiments/process_cruxeval.py --task i --gen_dir ${monologue_pred_dir}

cd $CRUXEVAL_HOME/evaluation;

echo "Evaluating results: direct prediction..."

python evaluate_generations.py \
    --generations_path ${direct_pred_dir}/generations.json \
    --scored_results_path ${direct_pred_dir}/scored_results.json \
    --mode input \
    2>&1 | tee ${direct_pred_dir}/eval.log

echo "Evaluating results: monologue prediction..."

python evaluate_generations.py \
    --generations_path ${monologue_pred_dir}/generations.json \
    --scored_results_path ${monologue_pred_dir}/scored_results.json \
    --mode input \
    2>&1 | tee ${monologue_pred_dir}/eval.log

########################### 
# CRUXEval-O: run inference
###########################

cd $SEMCODER_HOME;

model_name=$(basename $MODEL_2)

direct_pred_dir=${RESULT_OUTPUT_DIR}/${model_name}_direct
monologue_pred_dir=${RESULT_OUTPUT_DIR}/${model_name}_monologue

mkdir -p ${direct_pred_dir}
mkdir -p ${monologue_pred_dir}

echo "Evaluating model: ${model_name} on CRUXEval-O (direct prediction)..."

python experiments/run_cruxeval.py \
    --model $MODEL_2 \
    --trust_remote_code \
    --tasks output_prediction \
    --batch_size 1 \
    --n_samples 1 \
    --max_length_generation 4096 \
    --precision fp16 \
    --limit 200 \
    --temperature 0.0 \
    --save_generations \
    --save_generations_path ${direct_pred_dir}/results.json \
    --start 0 \
    --end 200 \
    --shuffle \
    --tensor_parallel_size 1

echo "Evaluating model: ${model_name} on CRUXEval-O with SemCoder Forward Monologue..."

python experiments/run_cruxeval.py \
    --model $MODEL_2 \
    --trust_remote_code \
    --tasks output_prediction \
    --batch_size 1 \
    --n_samples 1 \
    --max_length_generation 4096 \
    --precision fp16 \
    --limit 200 \
    --temperature 0.0 \
    --save_generations \
    --save_generations_path ${monologue_pred_dir}/results.json \
    --start 0 \
    --end 200 \
    --shuffle \
    --cot \
    --tensor_parallel_size 1

########################## 
# CRUXEval-O: Report score
##########################
echo "Reporting score for model: ${model_name}...";

python experiments/cruxeval_combine_generations.py --gen_dir ${direct_pred_dir}
python experiments/process_cruxeval.py --task o --gen_dir ${direct_pred_dir}
python experiments/cruxeval_combine_generations.py --gen_dir ${monologue_pred_dir}
python experiments/process_cruxeval.py --task o --gen_dir ${monologue_pred_dir}

cd $CRUXEVAL_HOME/evaluation;

echo "Evaluating results: direct prediction..."

python evaluate_generations.py \
    --generations_path ${direct_pred_dir}/generations.json \
    --scored_results_path ${direct_pred_dir}/scored_results.json \
    --mode output \
    2>&1 | tee ${direct_pred_dir}/eval.log

echo "Evaluating results: monologue prediction..."

python evaluate_generations.py \
    --generations_path ${monologue_pred_dir}/generations.json \
    --scored_results_path ${monologue_pred_dir}/scored_results.json \
    --mode output \
    2>&1 | tee ${monologue_pred_dir}/eval.log

