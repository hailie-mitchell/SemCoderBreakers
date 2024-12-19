#!/bin/bash

# Define the base directories
RECODE_DIR="../recode/datasets/perturbed/humaneval/full"
VLLM_DIR="output_dir/eval/evalplus/deepseek_instruct_2048"

# Define the folders to iterate through
folders=("natgen" "nlaugmenter" "func_name" "format")

# Iterate through each folder
for folder in "${folders[@]}"; do
    # Find all .jsonl files in the current folder
    for file in "$RECODE_DIR/$folder"/*.jsonl; do
        # Extract the filename without path
        filename=$(basename "$file")
        
        # Construct the command
        command="HUMANEVAL_OVERRIDE_PATH=\"$file\" evalplus.codegen --model "deepseek-ai/deepseek-coder-6.7b-instruct" --greedy --dataset humaneval --backend vllm --root $VLLM_DIR/$folder"
        
        # Execute the command
        echo "Executing: $command"
        eval "$command"
    done
done


