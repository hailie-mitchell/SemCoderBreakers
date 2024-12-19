#!/bin/bash

# Base directories
INPUT_BASE_DIR="/home/rg3637/recode/datasets/perturbed/humaneval/full"
OUTPUT_BASE_DIR="output_dir/eval/evalplus/semcoder_2048"

# Folders to iterate through
FOLDERS=("format" "nlaugmenter" "natgen")

# Function to process files in a folder
process_folder() {
    local folder=$1
    local input_dir="${INPUT_BASE_DIR}/${folder}"
    local output_dir="${OUTPUT_BASE_DIR}/${folder}"

    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Process each .jsonl file in the input directory
    for input_file in "$input_dir"/*.jsonl; do
        if [ -f "$input_file" ]; then
            # Extract filename without path
            filename=$(basename "$input_file")
            # Construct output file path
            output_file="${output_dir}/${filename}"

            # Check if output file already exists
            if [ -f "$output_file" ]; then
                echo "Skipping: $input_file (output already exists)"
                continue
            fi

            echo "Processing: $input_file"
            echo "Output to: $output_file"

            # Run the command
            PYTHONPATH=$PYTHONPATH:$(pwd)/src python /home/rg3637/SemCoder/experiments/run_evalplus.py \
                --model_key deepseek-ai/deepseek-coder-6.7b-base \
                --model_name_or_path semcoder/semcoder_1030 \
                --dataset humaneval \
                --save_path "$output_file" \
                --n_batches 1 \
                --n_problems_per_batch 1 \
                --n_samples_per_problem 1 \
                --max_new_tokens 2048 \
                --top_p 0.9 \
                --temperature 0 \
                --input_data_path "$input_file"
        fi
    done
}

# Main execution
for folder in "${FOLDERS[@]}"; do
    process_folder "$folder"
done
