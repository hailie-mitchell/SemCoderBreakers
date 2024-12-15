""" This is the main python file to run our robustness benchmark.
Please run python run_robust.py --help to check detailed usage.

public models: [ "codegen-350M-multi", "codegen-2B-multi", "codegen-6B-multi",
    "codegen-350M-mono", "codegen-2B-mono", "codegen-6B-mono",
    "incoder-1B", "incoder-6B", 
    "gpt-j-6B",
    "codet5-base", "codet5-large"
]
Evaluated datasets: ["humaneval", "mbpp", "mbjp", "mbjsp", "mbphp", "mbrbp", "mbkp"]
"""

from __future__ import annotations
import os
import json
import argparse
# from config import *
import csv
import random
import numpy as np
from collections import Counter, defaultdict
from perturb import read_config

cwd = os.getcwd()


def run_cmd(cmd):
    """ A help function to run the command
    """
    print(f"=== {cmd} ===")
    os.system(cmd)


def read_json(file_name):
    """ A help funtion to load data json files
    """
    data = []
    if not os.path.exists(file_name):
        print(f"Warning: {file_name} not exists, skip!")
        return
    with open(file_name, 'r') as input_file:
        for line in input_file:
            data.append(json.loads(line))
    return data


def read_and_reformat_json(input_file):
    # Load the entire JSON file
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    # List to store reformatted data
    reformatted_data_list = []

    # Iterate through the 'eval' section (e.g., HumanEval/1, HumanEval/2, etc.)
    for task_key, tasks in data['eval'].items():
        for task in tasks:
            # Extract the task_id and plus_status
            task_id = task.get('task_id')
            plus_status = task.get('plus_status', 'fail')

            # Determine if the task passed or failed
            passed = True if plus_status == 'pass' else False

            # Create a new dictionary in the desired format
            reformatted_data = {
                "task_id": task_id,
                "passed": passed
            }

            # Append the reformatted data to the list
            reformatted_data_list.append(reformatted_data)

    return reformatted_data_list



def read_passatk(file):
    f = open(file, "r")
    line = f.readlines()[0].replace("\'", "\"")
    data = json.loads(line)
    f.close()
    return data["pass@1"]


def calculate_passatk(data):
    length = len(data)
    cnt = 0
    for d in data:
        if d["passed"]:
            cnt += 1
    return cnt / length


def calculate_relative(nominal_data, perturbed_data):
    length = len(nominal_data)
    cnt = 0
    perturbed_lookup = {d["task_id"]: d["passed"] for d in perturbed_data}

    for d1 in nominal_data:
        task_id = d1["task_id"]
        if task_id in perturbed_lookup and d1["passed"] != perturbed_lookup[task_id]:
            cnt += 1

    return cnt / length


def report_results_updated(nominal_data_path, perturbed_data_paths):
    """Report results comparing a single nominal dataset with multiple perturbed datasets.
    Calculates multiple metrics: passatk, drop, and relative.
    """

    # Read the nominal dataset
    if not os.path.exists(nominal_data_path):
        print(f"Nominal dataset {nominal_data_path} missing, skipping...")
        return
    nominal_data = read_and_reformat_json(nominal_data_path)
    
    full_data = [["Dataset", "Pass@1", "Drop", "Relative"]]

    # Add the nominal dataset's Pass@k metric to the report
    nominal_passatk = calculate_passatk(nominal_data)
    full_data.append(["nominal", nominal_passatk, "N/A", "N/A"])

    # Iterate through perturbed datasets and compute the metrics for each
    for perturbed_data_path in perturbed_data_paths:
        if not os.path.exists(perturbed_data_path):
            print(f"Perturbed dataset {perturbed_data_path} missing, skipping...")
            continue
        
        # Read the perturbed dataset
        perturbed_data = read_and_reformat_json(perturbed_data_path)
        
        # Calculate all metrics for each perturbed dataset
        pass_value = calculate_passatk(perturbed_data)
        drop_value = (nominal_passatk - pass_value)/nominal_passatk
        relative_value = calculate_relative(nominal_data, perturbed_data)
        full_data.append([perturbed_data_path, pass_value, drop_value, relative_value])
        
    # Define the CSV file path with perturbed dataset file name and suffix
    csv_path = f"csv/deepseek_base/format-plus-report.csv"
        
    # Ensure the directory exists
    if not os.path.exists("csv"):
        os.mkdir("csv")
        
    # Write results to CSV
    with open(csv_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(full_data)
        
    print(f"Results saved to {csv_path}")



if __name__ == '__main__':
    """ The main function for using our robustness benchmark
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('status', type=str, choices=['perturb', 'report_results_updated', 'create_partial', 'nominal', 'subset', 'exec', 'analysis', 'report', "report_coarse", "report_finegrained"], help='The funcitons enabled by our benchmark')
    parser.add_argument('--nominal_data', type=str, help="Path to the nominal dataset (for 'report_results_updated').")
    parser.add_argument('--perturbed_data', type=str, nargs='+', help="Paths to the perturbed datasets (for 'report_results_updated').")
    

    
    args = parser.parse_args()
    print(args)
    
    if args.status == "nominal":
        evaluate_nominal(args)
    elif args.status == "create_partial":
        create_nominal_partial_datasets(args)
    elif args.status == "subset":
        create_subset(args)
    elif args.status == "perturb":
        create_perturbed_datasets(args)
    elif args.status == "exec":
        evaluate_perturbed_datasets(args)
    elif args.status == "analysis":
        print_sample_analysis(args)
    elif args.status == "report":
        if args.metric == "all":
            for metric in ["passatk", "drop", "relative"]:
                args.metric = metric
                report_results(args)
        else:
            report_results(args)
    elif args.status == "report_results_updated":
        # Instead of iterating over all metrics, directly call the updated report function
        nominal_data_path = args.nominal_data  # Path to the nominal dataset
        perturbed_data_paths = args.perturbed_data  # List of paths to perturbed datasets
        report_results_updated(nominal_data_path, perturbed_data_paths)
    elif args.status == "report_coarse":
        report_results_coarse(args)
    elif args.status == "report_finegrained":
        report_results_finegrained(args)


