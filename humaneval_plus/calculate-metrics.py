import json
import argparse
import os

def calculate_passatk(data):
    length = len(data["eval"])
    cnt = 0
    for d in data["eval"]:
        d = data["eval"][d][0]
        if d["base_status"] == "pass":
            cnt += 1
    return cnt/length


def estimator(n, c, k):
    # calculate estimated passatk for each input problem
    if n - c < k:
        return 1.
    return 1. - np.prod(1. - k / np.arange(n - c + 1, n + 1))


def calculate_passatk_sampling(data, n=1, k=1):
    completion_id = Counter()
    n_samples = 0
    results = defaultdict(list)

    for d in data:
        task_id = d["task_id"]
        results[task_id].append([completion_id[task_id], d["passed"]])
        completion_id[task_id] += 1
        n_samples += 1

    single_passatk_list = []
    for task_id in results:
        if len(results) * n == len(data):
            c = sum(d[1] for d in results[task_id])
        else:
            assert n <= len(results[task_id])
            c = sum(results[task_id][ni][1] for ni in range(n))
        single_passatk_list.append(estimator(n, c, k))
    return sum(single_passatk_list) / len(single_passatk_list)


def read_into_dict(data):
    data_dict = {}
    for d in data:
        data_dict[d["task_id"]] = d["passed"]
    return data_dict


def get_worst_passatk_dict(perturbed_data_list):
    assert len(perturbed_data_list) >= 1
    passatk_worst = {}
    for pdata in perturbed_data_list[0]["eval"]:
        pdata = perturbed_data_list[0]["eval"][pdata][0]
        passatk_worst[pdata["task_id"]] = True
    for pdata in perturbed_data_list[0]["eval"]:
        pdata = perturbed_data_list[0]["eval"][pdata][0]
        assert pdata["task_id"] in passatk_worst
        passatk_worst[pdata["task_id"]] = passatk_worst[pdata["task_id"]] and pdata["base_status"] == "pass"
    return passatk_worst


def get_worst_passatk_dict_sampling(perturbed_data_list):
    assert len(perturbed_data_list) >= 1
    passatk_worst = defaultdict(list)
    completion_id = Counter()
    for pdata in perturbed_data_list[0]:
        task_id = pdata["task_id"]
        passatk_worst[task_id].append([completion_id[task_id], True])
        completion_id[task_id] += 1
    for perturbed_data in perturbed_data_list:
        completion_id = Counter()
        for pdata in perturbed_data:
            task_id = pdata["task_id"]
            assert task_id in passatk_worst
            passatk_worst[task_id][completion_id[task_id]][1] = passatk_worst[task_id][completion_id[task_id]][1] and \
                                                                pdata["passed"]
            completion_id[task_id] += 1
    return passatk_worst


def get_best_passatk_dict(perturbed_data_list):
    assert len(perturbed_data_list) >= 1
    passatk_best = {}
    for pdata in perturbed_data_list[0]["eval"]:
        pdata = perturbed_data_list[0]["eval"][pdata][0]
        passatk_best[pdata["task_id"]] = False
    for pdata in perturbed_data_list[0]["eval"]:
        pdata = perturbed_data_list[0]["eval"][pdata][0]
        assert pdata["task_id"] in passatk_best
        passatk_best[pdata["task_id"]] = passatk_best[pdata["task_id"]] or pdata["base_status"] == "pass"
    return passatk_best


def get_best_passatk_dict_sampling(perturbed_data_list):
    assert len(perturbed_data_list) >= 1
    passatk_best = defaultdict(list)
    completion_id = Counter()
    for pdata in perturbed_data_list[0]:
        task_id = pdata["task_id"]
        passatk_best[task_id].append([completion_id[task_id], False])
        completion_id[task_id] += 1
    for perturbed_data in perturbed_data_list:
        completion_id = Counter()
        for pdata in perturbed_data:
            task_id = pdata["task_id"]
            assert task_id in passatk_best
            passatk_best[task_id][completion_id[task_id]][1] = passatk_best[task_id][completion_id[task_id]][1] or \
                                                               pdata["passed"]
            completion_id[task_id] += 1
    return passatk_best


def calculate_metric(perturbed_data_list, metric, nominal_data):
    """ Get targeted metric numbers
    perturbed_data_list: a list of perturbed data completions, each element is the completion of one seed dataset
    """

    length = len(nominal_data[0]["eval"])
    # init worst dict
    # passatk_worst = {}
    # for ndata in nominal_data:
    #     passatk_worst[ndata["task_id"]] = True
    passatk_worst = get_worst_passatk_dict(perturbed_data_list)
    passatk_best = get_best_passatk_dict(perturbed_data_list)
    if metric == "passatk":
        # perturbed pass@k
        passatk_list = []
        for perturbed_data in perturbed_data_list:
            passatk_list.append(calculate_passatk(perturbed_data))
        worst_cnt = 0
        for key in passatk_worst:
            if passatk_worst[key]:
                worst_cnt += 1
        return passatk_list, worst_cnt / length if passatk_list else " ", passatk_worst

    if metric == "drop":
        # (nominal pass@k - perturbed pass@k) / nominal pass@k
        nominal_passatk = calculate_passatk(nominal_data[0])
        passatk_list = []
        for perturbed_data in perturbed_data_list:
            perturbed_passatk = calculate_passatk(perturbed_data)
            passatk_list.append((nominal_passatk - perturbed_passatk) / nominal_passatk)
        worst_cnt = 0
        for key in passatk_worst:
            if passatk_worst[key]:
                worst_cnt += 1
        perturbed_passatk_worst = worst_cnt / length
        return passatk_list, (nominal_passatk - perturbed_passatk_worst) / nominal_passatk if passatk_list else " ", passatk_worst

    if metric == "relative":
        # (nominal != perturbed) / total prompts
        diffset = []
        nominal_dict = {}
        for ndata in nominal_data[0]["eval"]:
            ndata = nominal_data[0]["eval"][ndata][0]
            nominal_dict[ndata["task_id"]] = ndata["base_status"]
        relative_list = []

        relative_cnt = 0
        for pdata in perturbed_data_list[0]["eval"]:
            pdata = perturbed_data_list[0]["eval"][pdata][0]
            # print(pdata["task_id"], nominal_dict[pdata["task_id"]], pdata["plus_status"])
            if nominal_dict[pdata["task_id"]] != pdata["base_status"]:
                relative_cnt += 1
                diffset.append(pdata["task_id"])

        relative_list.append(relative_cnt / length)
        diffset = set(diffset)
        worst_cnt = 0
        for key in passatk_worst:

            if int(nominal_dict[key] == "pass") != int(passatk_worst[key]):
                # print(1, nominal_dict[key] == "pass", passatk_worst[key], passatk_best[key])
                worst_cnt += 1
            elif int(nominal_dict[key] == "pass") != int(passatk_best[key]):
                # print(2, nominal_dict[key] == "pass", passatk_worst[key], passatk_best[key])
                worst_cnt += 1
        assert len(diffset) == worst_cnt
        return relative_list, worst_cnt / length if relative_list else " "
        # rcMinus = 0
        # rcPlus = 0
        # for ndata in nominal_data[0]["eval"]:
        #     pdata = perturbed_data_list[0]["eval"][ndata][0]
        #     ndata = nominal_data[0]["eval"][ndata][0]
        #     if ndata["plus_status"] == "pass" and pdata["plus_status"] == "fail":
        #         rcMinus += 1
        #     if ndata["plus_status"] == "fail" and pdata["plus_status"] == "pass":
        #         rcPlus += 1
        # return (rcMinus + rcPlus)/length



    if metric == "attack_success":
        # (nominal correct & perturbed incorrect) / nominal correct
        nominal_dict = {}
        correct_cnt = 0
        for ndata in nominal_data:
            nominal_dict[ndata["task_id"]] = ndata["passed"]
            if ndata["passed"]:
                correct_cnt += 1
        success_list = []
        for perturbed_data in perturbed_data_list:
            success_cnt = 0
            for pdata in perturbed_data:
                if nominal_dict[pdata["task_id"]] and not pdata["passed"]:
                    success_cnt += 1
            success_list.append(success_cnt / correct_cnt)
        worst_cnt = 0
        for key in passatk_worst:
            if nominal_dict[key] and not passatk_worst[key]:
                worst_cnt += 1
        return success_list, worst_cnt / correct_cnt if success_list else " ", passatk_worst

def parse_multiple_json_objects(file_path):
    json_objects = []
    with open(file_path, 'r') as file:
        content = file.read()
        # Split the content by newlines or other appropriate delimiters
        for line in content.splitlines():
            if line.strip():
                try:
                    json_object = json.loads(line)
                    json_objects.append(json_object)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line}. Error: {e}")
    return json_objects


def create_parser():
    parser = argparse.ArgumentParser(description='Process JSON data paths.')
    parser.add_argument('--perturbed_data_list', type=str, required=True,
                        help='Path to the perturbed data list JSON file')
    parser.add_argument('--nominal_data', type=str, required=True,
                        help='Path to the nominal data JSON file')
    parser.add_argument('--output_data', type=str, required=True,
                        help='Path to the nominal data JSON file')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    perturbed_data_list_path = args.perturbed_data_list
    nominal_data_path = args.nominal_data
    output_results_path = args.output_data

    perturbed_data_list = parse_multiple_json_objects(perturbed_data_list_path)
    nominal_data = parse_multiple_json_objects(nominal_data_path)
    passatk = calculate_metric(perturbed_data_list, "passatk", nominal_data)[0][0]
    rr = calculate_metric(perturbed_data_list, "relative", nominal_data)[0][0]
    drop = calculate_metric(perturbed_data_list, "drop", nominal_data)[0][0]
    with open(output_results_path, 'w') as f:
        f.write(f'passatk: {passatk}\n')
        f.write(f'rr: {rr}\n')
        f.write(f'drop: {drop}\n')
    
    print(f"{os.path.basename(perturbed_data_list_path).split('-')[0]},{rr},{drop},{passatk}")

