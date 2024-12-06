import os
import json
import argparse
import csv
import copy
import numpy as np


def read_config(config, data):
    with open(config, "r") as config_file:
        config_dict = json.load(config_file)
    NL_AUG_RECIPES = config_dict[data]["NL_AUG_RECIPES"]
    PARTIAL_RECIPES = config_dict[data]["PARTIAL_RECIPES"]
    FUNC_RECIPES = config_dict[data]["FUNC_RECIPES"]
    FORMAT_RECIPES = config_dict[data]["FORMAT_RECIPES"]
    FULL_RECIPES = NL_AUG_RECIPES + PARTIAL_RECIPES + FUNC_RECIPES + FORMAT_RECIPES
    RECIPES = config_dict["RECIPES"]
    for recipe in RECIPES:
        RECIPES[recipe] = eval(RECIPES[recipe])
    DATASET_PATH = config_dict["DATASET_PATH"]
    RANDOM_TRANS = config_dict["RANDOM_TRANS"]
    data_path = config_dict["data_path"]
    output_adv_path = config_dict["output_adv_path"]
    model_generate_path = config_dict["model_generate_path"]
    run_script = config_dict["run_script"]
    return NL_AUG_RECIPES, PARTIAL_RECIPES, FUNC_RECIPES, FORMAT_RECIPES, FULL_RECIPES, RECIPES,\
             DATASET_PATH, RANDOM_TRANS, data_path, output_adv_path, model_generate_path, run_script


def read_json(file_name, ref_file=None):
    """ A helper function to load json data
    """
    assert file_name.endswith(".json"), "File must be json."
    if not os.path.exists(file_name):
        print(f"Warning: {file_name} not exists, skip!")
        return

    with open(file_name, 'r') as input_file:
        if ref_file is None:
            data = json.load(input_file)
            assert "raw_scored_generations" in data.keys(),\
                f"{os.path.basename(file_name)} doesn't seem like a CRUXEval scored result file."
            return data

        data = json.load(input_file)
        assert "raw_scored_generations" in data.keys(),\
            f"{os.path.basename(file_name)} doesn't seem like a CRUXEval scored result file."
        ref_data = read_json(ref_file)["raw_scored_generations"]
        if data["raw_scored_generations"].keys() == ref_data.keys():
            return data

        data = data["raw_scored_generations"]
        sampled_data = {}
        for sample in ref_data.keys():
            assert sample in data.keys(), "Sample not found!"
            sampled_data[sample] = data[sample]
        return {
            "raw_scored_generations": sampled_data,
            "pass_at_1": calculate_passatk(sampled_data) * 100.
        }


def get_worst_dict(perturbed_data_list):
    assert len(perturbed_data_list) >= 1, "Empty data!"
    n = len(next(iter(perturbed_data_list[0].values())))
    worst_dict = {sample: [True] * n for sample in perturbed_data_list[0].keys()}

    for perturbed_data in perturbed_data_list:
        for sample, passed in perturbed_data.items():
            assert sample in worst_dict, "Unexpected sample found."
            assert len(passed) == n, "Unexpected length of scored results."

            for i in range(n):
                worst_dict[sample][i] = worst_dict[sample][i] and passed[i]

    return worst_dict


def get_best_dict(perturbed_data_list):
    assert len(perturbed_data_list) >= 1, "Empty data!"
    n = len(next(iter(perturbed_data_list[0].values())))
    best_dict = {sample: [False] * n for sample in perturbed_data_list[0].keys()}

    for perturbed_data in perturbed_data_list:
        for sample, passed in perturbed_data.items():
            assert sample in best_dict, "Unexpected sample found."
            assert len(passed) == n, "Unexpected length of scored results."

            for i in range(n):
                best_dict[sample][i] = best_dict[sample][i] or passed[i]

    return best_dict


def calculate_passatk(scored_results):
    pass_at_1s = []
    for result in scored_results.values():
        c, n = result.count(True), len(result)
        pass_at_1s.append(pass_at_k(n, c, 1))
    return sum(pass_at_1s) / len(pass_at_1s)


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def transpose_result(perturbed_data_list):
    assert len(perturbed_data_list) >= 1, "Empty data!"
    n = len(next(iter(perturbed_data_list[0].values())))
    transposed = [
        {sample: [] for sample in perturbed_data_list[0].keys()} for _ in range(n)
    ]

    for perturbed_data in perturbed_data_list:
        for sample, passed in perturbed_data.items():
            assert sample in transposed, "Unexpected sample found."
            assert len(passed) == n, "Unexpected length of scored results."

            for i in range(n):
                transposed[i][sample].append(passed[i])

    return transposed


def get_suc_rate(perturbed_results, nominal):
    if perturbed_results is None or len(perturbed_results) == 0:
        return 0.0
    return perturbed_results.count((not nominal)) / len(perturbed_results)


def eval_robustness(args, infer_mode="monologue"):
    results = {}                # for all perturbed results [worst_dict, best_dict]
    nominal_dict = {}           # for nominal result
    nominal_passatk_dict = {}   # for nominal passatk

    nonrobust_stats = args.nonrobust_stats
    nonrobust_dict = {
        "right_to_wrong": {},
        "wrong_to_right": {},
        "right_to_all_wrong": [],
        "wrong_to_all_right": [],
    }                           # for non-robust samples
    raw_results = {}            # for raw (not worst or best) perturbed results

    for model in args.models:
        results[model] = {}
        nominal_dict[model] = {}
        nominal_passatk_dict[model] = {}
        if nonrobust_stats:
            raw_results[model] = {}
        for task in ["input", "output"]:
            results[model][task] = {}
            nominal_dict[model][task] = {}
            nominal_passatk_dict[model][task] = {}
            if nonrobust_stats:
                nonrobust_dir = f"nonrobust_stats/cruxeval_{task}/{model}_{infer_mode}/"
                os.makedirs(nonrobust_dir, exist_ok=True)
                raw_results[model][task] = {}

    NL_AUG_RECIPES, PARTIAL_RECIPES, FUNC_RECIPES, FORMAT_RECIPES, FULL_RECIPES, RECIPES, \
        DATASET_PATH, RANDOM_TRANS, data_path, output_adv_path, model_generate_path, run_script = read_config(args.config, "cruxeval")
    for model in args.models:
        for task in ["input", "output"]:
            print(f"Evaluating {model} on {task} prediction (with {infer_mode} inference)...")
            nominal_data_path = f"nominal/cruxeval_{task}/{model}_{infer_mode}/scored_results.json"

            if not os.path.exists(nominal_data_path):
                print(f"{nominal_data_path} missing, skip...")
                continue
            nominal_data = read_json(nominal_data_path, ref_file=args.samples_like)

            nominal_pass_dict = nominal_data["raw_scored_generations"]
            nominal_passatk = nominal_data["pass_at_1"]

            print(f"nominal pass@1: {nominal_passatk:.2f}")
            nominal_dict[model][task] = nominal_pass_dict
            nominal_passatk_dict[model][task] = nominal_passatk
            nominal_cross_checked = False

            results[model][task] = None
            for aug_method in range(len(RECIPES[args.method])):
                if args.aug_method is not None and aug_method != args.aug_method:
                    # specific aug_method index is given
                    continue
                perturbed_data_list = []
                passatk_list = []
                method_passatk = 0.0
                method_name = RECIPES[args.method][aug_method]

                if args.n_outputs == 1:
                    perturbed_data_path = f"{args.method}/{method_name}/cruxeval_{task}/{model}_{infer_mode}/scored_results.json"
                    if os.path.exists(perturbed_data_path):
                        perturbed_data = read_json(perturbed_data_path, ref_file=args.samples_like)
                        perturbed_data_list.append(perturbed_data["raw_scored_generations"])
                        method_passatk = perturbed_data["pass_at_1"]
                    else:
                        print(f"{perturbed_data_path} not exists, skip..")
                        pass
                else:
                    for seed in range(args.n_outputs):
                        if method_name not in RANDOM_TRANS and seed >= 1: # skip other seeds since they are not random
                            continue
                        perturbed_data_path = f"{args.method}/{method_name}_s{seed}/cruxeval_{task}/{model}_{infer_mode}/scored_results.json"
                        if os.path.exists(perturbed_data_path):
                            perturbed_data = read_json(perturbed_data_path, ref_file=args.samples_like)
                            perturbed_data_list.append(perturbed_data["raw_scored_generations"])
                            passatk_list.append(perturbed_data["pass_at_1"])
                        else:
                            print(f"{perturbed_data_path} not exists, skip..")
                            pass
                    method_passatk = calculate_passatk(get_worst_dict(perturbed_data_list)) * 100.

                if perturbed_data_list:
                    if passatk_list:
                        print(f"\t{RECIPES[args.method][aug_method]} passatk: {passatk_list}, {method_passatk:.2f}")
                    else:
                        print(f"\t{RECIPES[args.method][aug_method]} passatk: {method_passatk:.2f}")
                    # import pdb; pdb.set_trace()
                    # merge results across different aug_method
                    worst_dict = get_worst_dict(perturbed_data_list)
                    best_dict = get_best_dict(perturbed_data_list)
                    assert worst_dict.keys() == best_dict.keys(), "Worst and best dict have different samples."

                    if not nominal_cross_checked:
                        assert len(worst_dict) <= len(nominal_pass_dict),\
                            "Check the scored result file. Perturbed results have more samples than nominal result."
                        if len(worst_dict) < len(nominal_pass_dict):
                            print("Perturbed results have less samples than nominal result.")
                            print("Assume sampling and try to extract the subset from the nominal result:")
                            sampled_nominal_pass_dict = {}
                            for sample in worst_dict.keys():
                                assert sample in nominal_pass_dict.keys(), "Unexpected sample found."
                                sampled_nominal_pass_dict[sample] = nominal_pass_dict[sample]
                            nominal_pass_dict = sampled_nominal_pass_dict
                            nominal_passatk = calculate_passatk(nominal_pass_dict) * 100.
                            nominal_dict[model][task] = nominal_pass_dict
                            nominal_passatk_dict[model][task] = nominal_passatk
                            print(f"nominal sampled; re-evaluated nominal pass@1: {nominal_passatk:.2f}")
                        else:
                            assert nominal_pass_dict.keys() == worst_dict.keys(), "Unexpected sample found."
                        nominal_cross_checked = True

                    if nonrobust_stats:
                        raw_results[model][task][method_name] = transpose_result(perturbed_data_list)

                    if results[model][task] is None:
                        results[model][task] = [worst_dict, best_dict]
                    else:
                        for sample in results[model][task][0].keys():
                            assert sample in worst_dict and sample in best_dict, "Unexpected sample found."
                            # results[model][task][0][sample] = results[model][task][0][sample] and worst_dict[sample]
                            # results[model][task][1][sample] = results[model][task][1][sample] or best_dict[sample]
                            results[model][task][0][sample] = [r and w for r, w in zip(results[model][task][0][sample], worst_dict[sample])]
                            results[model][task][1][sample] = [r or b for r, b in zip(results[model][task][1][sample], best_dict[sample])]
                else:
                    # no data available
                    print(f"No data processed for {method_name}. Skip...")

            if not nominal_cross_checked:
                print("Nominal dataset has not cross checked with perturbed dataset. Result might be inaccurate.")

    # directory = "statitic_jsons"
    # os.makedirs(directory, exist_ok=True)

    # json.dump(nominal_dict, open(f"statitic_jsons/{args.method}_{infer_mode}_nominal.json", "w"))
    # json.dump(results, open(f"statitic_jsons/{args.method}_{infer_mode}_perturbed.json", "w"))
    # json.load(nominal_dict, open(f"statitic_jsons/{args.method}_nominal.json", "r"))

    # reformulate results to csv table
    for task in ["input", "output"]:
        full_data = []
        row = ["nominal"]
        for model in args.models:
            if model in nominal_passatk_dict and task in nominal_passatk_dict[model]:
                row.append(nominal_passatk_dict[model][task])
            else:
                row.append(" ")
        full_data.append(row)

        row = ["passatk"]
        for model in args.models:
            try:
                worst_dict = results[model][task][0]
                assert worst_dict, "Empty result!"
                row.append(calculate_passatk(worst_dict) * 100.)
            except Exception as e:
                row.append(" ")
                print(f"{type(e).__name__}: {str(e)}. Skip this result.")
        full_data.append(row)

        row = ["drop (%)"]
        for model in args.models:
            try:
                worst_dict = results[model][task][0]
                assert worst_dict, "Empty result!"
                passatk = calculate_passatk(worst_dict) * 100.
                nominal_passatk = nominal_passatk_dict[model][task]
                row.append((nominal_passatk - passatk) / nominal_passatk * 100.)                
            except Exception as e:
                row.append(" ")
                print(f"{type(e).__name__}: {str(e)}. Skip this result.")
        full_data.append(row)

        row = ["relative (%)"]
        for model in args.models:
            try:
                n = len(next(iter(nominal_dict[model][task].values())))
                right_to_wrong = {sample: [False] * n for sample in nominal_dict[model][task].keys()}
                wrong_to_right = {sample: [False] * n for sample in nominal_dict[model][task].keys()}

                worst_dict = results[model][task][0]
                best_dict = results[model][task][1]
                assert worst_dict and best_dict, "Empty result!"

                raw_result = raw_results[model][task] if nonrobust_stats else None
                method_list = list(raw_result.keys()) if nonrobust_stats else None
                for i in range(n):
                    nonrobust_dict_stream = copy.deepcopy(nonrobust_dict) if nonrobust_stats else None
                    for sample in worst_dict.keys():
                        nominal = nominal_dict[model][task][sample][i]
                        if worst_dict[sample][i] != nominal:
                            # worst dict difference
                            if nominal:
                                right_to_wrong[sample][i] = True
                                if nonrobust_stats:
                                    nonrobust_dict_stream["right_to_wrong"][sample] = {
                                        "nominal": nominal,
                                        "perturbed": {
                                            aug_method: raw_result[aug_method][i][sample]
                                            for aug_method in method_list
                                        },
                                        "method_succeeded": {
                                            aug_method: get_suc_rate(raw_result[aug_method][i][sample], nominal)
                                            for aug_method in method_list if (not nominal) in raw_result[aug_method][i][sample]
                                        },
                                    }
                                    if set(nonrobust_dict_stream["right_to_wrong"][sample]["method_succeeded"].keys()) == set(method_list):
                                        nonrobust_dict_stream["right_to_all_wrong"].append(sample)
                            else:
                                wrong_to_right[sample][i] = True
                                if nonrobust_stats:
                                    nonrobust_dict_stream["wrong_to_right"][sample] = {
                                        "nominal": nominal,
                                        "perturbed": {
                                            aug_method: raw_result[aug_method][i][sample]
                                            for aug_method in method_list
                                        },
                                        "method_succeeded": {
                                            aug_method: get_suc_rate(raw_result[aug_method][i][sample], nominal)
                                            for aug_method in method_list if (not nominal) in raw_result[aug_method][i][sample]
                                        },
                                    }
                                    # due to worst dict, no need for extra check
                                    nonrobust_dict_stream["wrong_to_all_right"].append(sample)
                        elif best_dict[sample][i] != nominal:
                            # best dict difference
                            wrong_to_right[sample][i] = True
                            if nonrobust_stats:
                                nonrobust_dict_stream["wrong_to_right"][sample] = {
                                    "nominal": nominal,
                                    "perturbed": {
                                        aug_method: raw_result[aug_method][i][sample]
                                        for aug_method in method_list
                                    },
                                    "method_succeeded": {
                                        aug_method: get_suc_rate(raw_result[aug_method][i][sample], nominal)
                                        for aug_method in method_list if (not nominal) in raw_result[aug_method][i][sample]
                                    },
                                }
                                if set(nonrobust_dict_stream["wrong_to_right"][sample]["method_succeeded"].keys()) == set(method_list):
                                    nonrobust_dict_stream["wrong_to_all_right"].append(sample)
                    if nonrobust_stats:
                        dump_path = f"nonrobust_stats/cruxeval_{task}/{model}_{infer_mode}/{args.method}_nonrobust.jsonl"
                        with open(dump_path, "a") as file:
                            json.dump(nonrobust_dict_stream, file)
                            file.write("\n")
                row.append((calculate_passatk(right_to_wrong) + calculate_passatk(wrong_to_right)) * 100.)
            except Exception as e:
                row.append(" ")
                print(f"{type(e).__name__}: {str(e)}. Skip this result.")
        full_data.append(row)

        # header = [args.method] + args.models
        csv_path = f"category_report/{args.method}/{infer_mode}/"
        os.makedirs(csv_path, exist_ok=True)

        if args.aug_method is not None:
            method_name = RECIPES[args.method][args.aug_method]
            header = [f"{args.method}/{method_name}"] + args.models
            csv_path += f"cruxeval_{task}_{method_name}.csv"
        else:
            header = [args.method] + args.models
            csv_path += f"cruxeval_{task}_{args.method}.csv"

        file = open(csv_path, "w")
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(full_data)
        file.close()

    return


if __name__ == '__main__':
    """ The main function for using our robustness benchmark
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=["normal", "nlaugmenter", "natgen", "format", "func_name", "random"], help="The classes of perturbation. Please set method to natgen with status nominal to evaluate nominal partial code.")
    parser.add_argument('--config', type=str, default="../../recode/config.json", help="Path to recode config")
    parser.add_argument('--aug_method', type=int, default=None, help="The detailed augmentation method used with index (index defined in config.json for each method). Default None means running all the perturbations")
    parser.add_argument('--models', type=str, nargs='+', default=["semcoder_1030", "semcoder_s_1030"], help="A list of the models needed to evaluate with")
    parser.add_argument('--mode', type=str, nargs='+', default=["monologue", "direct"], choices=["monologue", "direct"], help="Inference mode of the model")
    parser.add_argument('--nonrobust_stats', action='store_true', help="Show detailed statistics of non-robust samples")
    parser.add_argument('--n_outputs', type=int, default=1, help="The total number of perturbations generated/evaluated with")
    parser.add_argument('--samples_like', type=str, default=None, help="Path to a reference file which you wish to use for sampling the full result")
    args = parser.parse_args()
    print(args)

    for mode in args.mode:
        eval_robustness(args, mode)
