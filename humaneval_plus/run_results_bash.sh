#!/bin/bash

# Base paths
base_perturbed_path="/home/rg3637/SemCoder/output_dir/eval/evalplus/semcoder_s_2048"
nominal_data="/home/rg3637/SemCoder/output_dir/eval/evalplus/semcoder_s_2048/nominal-sanitized_eval_results.json"

# Output CSV file
output_csv="semcoder_s_metrics-plus.csv"

# Create the CSV header
echo "Category,Technique,RR@1,RD@1,RP@1" > "$output_csv"

# Folders to process
folders=("natgen" "func_name" "format" "nlaugmenter")

# Loop through each folder
for folder in "${folders[@]}"; do
    # Find all JSON files in the current folder
    find "${base_perturbed_path}/${folder}" -name "*.json" | while read -r perturbed_file; do
        # Extract the filename without extension
        filename=$(basename "$perturbed_file" .json)
        
        # Run the Python script and capture the output
        output=$(python ./calculate-metrics.py \
            --perturbed_data_list "$perturbed_file" \
            --nominal_data "$nominal_data" \
            --output_data /dev/null)
        
        # Append the result to the CSV file
        echo "${folder},${output}" >> "$output_csv"
        
        echo "Processed: $perturbed_file"
    done
done

echo "CSV file created: $output_csv"
