import json
import ast
import sys
import argparse
import logging
import os

DATAPATH = "nominal/cruxeval_partial.jsonl"

def main(datapath=DATAPATH, logfile=None):

    # check if datapath exists
    assert os.path.isfile(datapath), f"Dataset file {datapath} does not exist"
    
    # set up logfile
    if logfile is not None:
        logfile = logfile.split(".")[0] + ".log"
        print(f'Logging to file {logfile}.')
        logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.info("Validating data in file: %s" % datapath)
    samples = []
    with open(datapath, "r") as file_name:
        for line in file_name:
            samples.append(json.loads(line))

    total, total_right, errors = 0, 0, 0
    wrong_samples = []
    for sample in samples:
        sample_in = sample["input"]
        sample_out = sample["output"]
        prefix = sample["prompt"]
        suffix = sample["canonical_solution"]
        func_name = sample["entry_point"]
        code = prefix + suffix
        if "id" in sample: sample_id = sample["id"]
        elif "task_id" in sample: sample_id = sample["task_id"]
        else: sample_id = "-1"

        total += 1
        func_dict = {}

        try:
            exec(code, func_dict)
            eval_str = func_name + "(" + sample_in + ")"
            eval_ret = eval(eval_str, func_dict)
            if eval_ret == eval(sample_out, {}): 
                total_right += 1
            else:
                sample["err_out"] = eval_ret
                wrong_samples.append(sample)

        except Exception as e:
            errors += 1
            log_err = "EXECUTION ERROR\ncode:\n" + code
            log_err += "\nid:\t" + sample_id
            log_err += "\ninput:\t" + sample_in
            log_err += "\noutput:\t" + sample_out
            log_err += "\nerror:\n" + str(e) + "\n"
            logging.info(log_err)

    total_wrong = len(wrong_samples)
    logging.info("all samples validated!")
    logging.info(f"samples correct: {total_right} / {total} ({total_right/total*100}%)")
    logging.info(f"samples with incorrect output: {total_wrong} / {total} ({total_wrong/total*100}%)")
    logging.info(f"samples with execution errors: {errors} / {total} ({errors/total*100}%)")
    if len(wrong_samples) > 0:
        logging.info("wrong samples:")
        for sample in wrong_samples:
            if "id" in sample: sample_id = sample["id"]
            elif "task_id" in sample: sample_id = sample["task_id"]
            else: sample_id = "-100"
            log_str = "code:\n" + sample["prompt"] + sample["canonical_solution"]
            log_str += "\nid:\t" + sample_id 
            log_str += "\ninput:\t" + sample["input"]
            log_str += "\noutput:\t" + sample["output"]
            log_str += "\nout:\t" + sample["err_out"] + "\n"
            logging.info(log_str)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Validate dataset samples return expected output when passed given input.")
    parser.add_argument("--data_path", default=DATAPATH, help="Filepath to dataset for validaiton.")
    parser.add_argument("--log_file", help="Path to log file to log output.")
    args = parser.parse_args()

    main(args.data_path, args.log_file)

