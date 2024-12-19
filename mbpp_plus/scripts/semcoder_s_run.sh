#!/bin/bash

# Input directory path containing the perturbed datasets
SEMCODER_DIR="/home/preethiprakash2001/SemCoderBreakers/mbpp_plus/perturbed-datasets/partial"

# Output directory to store the generated solutions
OUTPUT_PATH="output_dir/semcoder_s"

# Loop through the folders for each perturbation type
folders=("natgen" "func_name" "format" "nlaugmenter")

for folder in "${folders[@]}"; do
    for file in "$SEMCODER_DIR/$folder"/*.jsonl; do
        filename=$(basename "$file")
        
        # Define the output path based on the folder and filename
        output_file="$OUTPUT_PATH/$folder/$filename"
        
        # Create directory if it doesn't exist
        mkdir -p "$(dirname "$output_file")"

        # Define the command to generate the solutions for the SemCoder model
        command="PYTHONPATH=\$PYTHONPATH:\$(pwd)/src python /home/preethiprakash2001/SemCoder/experiments/run_evalplus.py \
--model_key deepseek-ai/deepseek-coder-6.7b-base \
--model_name_or_path semcoder/semcoder_s_1030 \
--dataset mbpp \
--save_path $output_file \
--input_path $file \
--n_batches 1 \
--n_problems_per_batch 1 \
--n_samples_per_problem 1 \
--max_new_tokens 2028 \
--top_p 0.9 \
--temperature 0"
        
        echo "Executing: $command"
        eval "$command"

        # Clear CUDA cache using Python
        echo "Clearing CUDA cache..."
        python3 - <<EOF
import torch
print("Clearing CUDA cache...")
torch.cuda.empty_cache()
torch.cuda.synchronize()
print("CUDA cache cleared successfully.")
EOF

    done
done
