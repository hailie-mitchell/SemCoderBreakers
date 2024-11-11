import os
import json
import argparse
import csv


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


def read_json(file_name):
    """ A help funtion to load data json files
    """
    data = []
    if not os.path.exists(file_name):
        print(f"Warning: {file_name} not exists, skip!")
        return

    with open(file_name, 'r') as input_file:
        if file_name.endswith('.json'):
            # Load a single JSON object from the file
            return json.load(input_file)
        elif file_name.endswith('.jsonl'):
            # Load multiple JSON objects, one per line
            for line in input_file:
                data.append(json.loads(line))
            return data
        else:
            print(f"Warning: {file_name} is neither a JSON nor JSONL file.")
            return


def get_worst_passatk_dict(perturbed_data_list):
    assert len(perturbed_data_list) >= 1, "Empty data!"
    passatk_worst = {sample: True for sample in perturbed_data_list[0].keys()}

    for perturbed_data in perturbed_data_list:
        for sample, passed in perturbed_data.items():
            assert sample in passatk_worst, "Unexpected sample found."
            passatk_worst[sample] = passatk_worst[sample] and passed[0]

    return passatk_worst


def get_best_passatk_dict(perturbed_data_list):
    assert len(perturbed_data_list) >= 1, "Empty data!"
    passatk_best = {sample: False for sample in perturbed_data_list[0].keys()}

    for perturbed_data in perturbed_data_list:
        for sample, passed in perturbed_data.items():
            assert sample in passatk_best, "Unexpected sample found."
            passatk_best[sample] = passatk_best[sample] or passed[0]

    return passatk_best


def calculate_passatk(data):
    length = len(data)
    cnt = 0
    for passed in data.values():
        if passed[0]:
            cnt += 1
    return cnt / length


def calculate_metric(perturbed_data_list, metric, nominal_data):
    """ Get targeted metric numbers
    perturbed_data_list: a list of perturbed data completions, each element is the completion of one seed dataset
    """
    length = len(nominal_data)

    passatk_worst = get_worst_passatk_dict(perturbed_data_list)
    passatk_best = get_best_passatk_dict(perturbed_data_list)

    if metric == "passatk":
        # perturbed pass@k
        passatk_list = []
        for perturbed_data in perturbed_data_list:
            passatk_list.append(calculate_passatk(perturbed_data))

        worst_cnt = 0
        for sample in passatk_worst.keys():
            if passatk_worst[sample]: 
                worst_cnt += 1

        return passatk_list, worst_cnt / length if passatk_list else " ", passatk_worst

    if metric == "drop":
        # (nominal pass@k - perturbed pass@k) / nominal pass@k
        nominal_passatk = calculate_passatk(nominal_data)

        passatk_list = []
        for perturbed_data in perturbed_data_list:
            perturbed_passatk = calculate_passatk(perturbed_data)
            passatk_list.append((nominal_passatk - perturbed_passatk) / nominal_passatk)

        worst_cnt = 0
        for sample in passatk_worst.keys():
            if passatk_worst[sample]: 
                worst_cnt += 1
        perturbed_passatk_worst = worst_cnt / length

        return passatk_list, (nominal_passatk - perturbed_passatk_worst) / nominal_passatk if passatk_list else " ", passatk_worst

    if metric == "relative":
        # (nominal != perturbed) / total prompts
        diffset = []
        relative_list = []

        for perturbed_data in perturbed_data_list:
            relative_cnt = 0
            for sample, passed in perturbed_data.items():
                if nominal_data[sample][0] != passed[0]:
                    relative_cnt += 1
                    diffset.append(sample)
            relative_list.append(relative_cnt / length)

        diffset = set(diffset)
        worst_cnt = 0
        for sample in passatk_worst.keys():
            if nominal_data[sample][0] != passatk_worst[sample][0]:
                worst_cnt += 1
            elif nominal_data[sample][0] != passatk_best[sample][0]:
                worst_cnt += 1
        assert len(diffset) == worst_cnt

        return relative_list, worst_cnt / length  if relative_list else " ", passatk_worst


def eval_per_cat(args):
    results = {}    # for all the perturbed result [worst_dict, best_dict]
    nominal_dict = {}   # for nominal result 
    nominal_passatk_dict = {} # for nominal passatk

    for model in args.models:
        results[model] = {}
        nominal_dict[model] = {}
        nominal_passatk_dict[model] = {}
        for task in ["input", "output"]:
            results[model][task] = {}
            nominal_dict[model][task] = {}
            nominal_passatk_dict[model][task] = {}

    for model in args.models:
        NL_AUG_RECIPES, PARTIAL_RECIPES, FUNC_RECIPES, FORMAT_RECIPES, FULL_RECIPES, RECIPES, \
            DATASET_PATH, RANDOM_TRANS, data_path, output_adv_path, model_generate_path, run_script = read_config(args.config, "cruxeval")
        for task in ["input", "output"]:
            print(f"Evaluating {model} on {task} prediction...")
            # monologue so far
            nominal_data_path = f"nominal/cruxeval_{task}/{model}_monologue/scored_results.json"

            if not os.path.exists(nominal_data_path):
                print(f"{nominal_data_path} missing, skip...")
                continue
            nominal_data = read_json(nominal_data_path)

            nominal_pass_dict = nominal_data["raw_scored_generations"]
            nominal_passatk = nominal_data["pass_at_1"]

            print(f"nominal pass@1: {nominal_passatk:.2f}")
            # Works but in a dirty way
            nominal_dict[model][task] = get_worst_passatk_dict([nominal_pass_dict])
            nominal_passatk_dict[model][task] = nominal_passatk

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
                    perturbed_data_path = f"{args.method}/{method_name}/cruxeval_{task}/{model}_monologue/scored_results.json"
                    if os.path.exists(perturbed_data_path):
                        perturbed_data = read_json(perturbed_data_path)
                        perturbed_data_list.append(perturbed_data["raw_scored_generations"])
                        method_passatk = perturbed_data["pass_at_1"]
                    else:
                        print(f"{perturbed_data_path} not exists, skip..")
                        pass
                else:
                    for seed in range(args.n_outputs):
                        if method_name not in RANDOM_TRANS and seed >= 1: # skip other seeds since they are not random
                            continue
                        perturbed_data_path = f"{args.method}/{method_name}_s{seed}/cruxeval_{task}/{model}_monologue/scored_results.json"
                        if os.path.exists(perturbed_data_path):
                            perturbed_data = read_json(perturbed_data_path)
                            perturbed_data_list.append(perturbed_data["raw_scored_generations"])
                            passatk_list.append(perturbed_data["pass_at_1"])
                        else:
                            print(f"{perturbed_data_path} not exists, skip..")
                            pass

                    _, method_passatk, _ = calculate_metric(perturbed_data_list, "passatk", nominal_pass_dict)
                    method_passatk *= 100.0

                if perturbed_data_list:
                    if passatk_list:
                        print(f"\t{RECIPES[args.method][aug_method]} passatk: {passatk_list}, {method_passatk:.2f}")
                    else:
                        print(f"\t{RECIPES[args.method][aug_method]} passatk: {method_passatk:.2f}")
                    # import pdb; pdb.set_trace()
                    # merge results across different aug_method
                    passatk_worst_dict = get_worst_passatk_dict(perturbed_data_list)
                    passatk_best_dict = get_best_passatk_dict(perturbed_data_list)

                    if results[model][task] is None:
                        results[model][task] = [passatk_worst_dict, passatk_best_dict]
                    else:
                        for sample in results[model][task][0].keys():
                            assert sample in passatk_worst_dict and sample in passatk_best_dict, "Unexpected sample found."
                            results[model][task][0][sample] = results[model][task][0][sample] and passatk_worst_dict[sample]
                            results[model][task][1][sample] = results[model][task][1][sample] or passatk_best_dict[sample]
                else:
                    # no data available
                    # print(f"\t{RECIPES[args.method][aug_method]} passatk: {passatk_list}, {method_passatk}")
                    print(f"No data processed for {method_name}. Skip...")

    # directory = "statitic_jsons"
    # os.makedirs(directory, exist_ok=True)

    json.dump(nominal_dict, open(f"statitic_jsons/{args.method}_nominal.json", "w"))
    json.dump(results, open(f"statitic_jsons/{args.method}_perturbed.json", "w"))
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
            cnt = 0
            total_cnt = 0
            if results[model][task][0]:
                for sample in results[model][task][0].keys():
                    if results[model][task][0][sample]:
                        cnt += 1
                    total_cnt += 1
                row.append(cnt / total_cnt * 100.)
            else:
                row.append(" ")
        full_data.append(row)

        row = ["drop (%)"]
        for model in args.models:
            cnt = 0
            total_cnt = 0
            if results[model][task][0]:
                for sample in results[model][task][0].keys():
                    if results[model][task][0][sample]:
                        cnt += 1
                    total_cnt += 1
                passatk = cnt / total_cnt * 100
                nominal_passatk = nominal_passatk_dict[model][task]
                row.append((nominal_passatk - passatk) / nominal_passatk * 100.)
            else:
                row.append(" ")
        full_data.append(row)
        
        row = ["relative (%)"]
        for model in args.models:
            cnt = 0
            total_cnt = 0
            if results[model][task][0]:
                for sample in results[model][task][0].keys():
                    if results[model][task][0][sample] != nominal_dict[model][task][sample]:
                        # worst dict difference
                        cnt += 1
                    elif results[model][task][1][sample] != nominal_dict[model][task][sample]:
                        # best dict difference
                        cnt += 1
                    total_cnt += 1
                row.append(cnt / total_cnt * 100.)
            else:
                row.append(" ")
        full_data.append(row)

        header = [args.method] + args.models
        csv_path = f"category_report/cruxeval_{task}_{args.method}.csv"
        if not os.path.exists("category_report"):
            os.mkdir("category_report")
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
    parser.add_argument('--granularity', type=str, default='category', choices=['category', 'method'], help='Obtain robustness metrics on specific granularity')
    parser.add_argument('--method', type=str, choices=["normal", "nlaugmenter", "natgen", "format", "func_name", "random"], help="The classes of perturbation. Please set method to natgen with status nominal to evaluate nominal partial code.")
    parser.add_argument('--config', default="config.json", help="path to recode config")
    parser.add_argument('--aug_method', type=int, default=None, help="The detailed augmentation method used with index (index defined in config.json for each method). Default None means running all the perturbations")
    parser.add_argument('--models', nargs='+', default=["semcoder_s_1030"], help="A list of the models needed to evaluate with (or create subset dataset for perturbed dataset, not needed most of the times).")
    parser.add_argument('--n_outputs', type=int, default=1, help="The total number of perturbations generated/evaluated with")
    args = parser.parse_args()
    print(args)
    
    if args.granularity == "category":
        eval_per_cat(args)
    else:
        raise NotImplementedError
