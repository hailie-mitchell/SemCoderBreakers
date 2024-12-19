#!/bin/bash

# Base directory for input files
INPUT_BASE_DIR="output_dir/eval/evalplus/deepseek_instruct_2048"

# Folders to process
FOLDERS=("nlaugmenter", "natgen", "format", "func_name")

# Dataset name
DATASET="humaneval"

# Function to process files in a folder
process_folder() {
    local folder=$1
    local input_dir="${INPUT_BASE_DIR}/${folder}"

    # Process each .jsonl file in the input directory
    for input_file in "$input_dir"/*.jsonl; do
        if [ -f "$input_file" ]; then
            echo "Processing: $input_file"

            # Run sanitize command
            evalplus.sanitize --samples "$input_file"

            # Set OUTPUT_PATH (input file with -sanitized.jsonl appended)
            OUTPUT_PATH="${input_file%.jsonl}-sanitized.jsonl"

            # Run evaluate command
            evalplus.evaluate --dataset humaneval --samples "$OUTPUT_PATH" --parallel 64 --i-just-wanna-run 2>&1 | tee "${input_dir}/evalplus_${DATASET}_results.log"
        fi
    done
}

# Main execution
for folder in "${FOLDERS[@]}"; do
    process_folder "$folder"
done
