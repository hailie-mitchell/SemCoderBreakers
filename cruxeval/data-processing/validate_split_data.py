import json
import ast

DATAPATH = "cruxeval_partial.jsonl"

samples = []
with open(DATAPATH, "r") as f:
    for line in f:
        samples.append(json.loads(line))

total_wrong, total_right = 0, 0
wrong_samples = []
for sample in samples:
    sample_in = sample["input"]
    sample_out = sample["output"]
    prefix = sample["prompt"]
    suffix = sample["canonical_solution"]
    
    if (prefix + suffix) == sample["code"]:
        total_right += 1

    else:
        exec(prefix + suffix)
    
        try:
            out = f(ast.literal_eval(sample_in))
        except:
            try:
                out = f(*ast.literal_eval(sample_in))
            except:
                pass

        try:
            assert out == ast.literal_eval(sample_out)
            total_right += 1
        except:
            total_wrong += 1
            sample["out_wrong"] = out
            wrong_samples.append(sample)

print("all samples validated!")
print("total correct:\t", total_right)
print("total incorrect:\t", total_wrong)

for sample in wrong_samples:
    print("input:\t", sample["input"])
    print("output:\t", sample["output"])
    print("incorrect output:\t", sample["out_wrong"])
    print("function:\n", (sample["prompt"] + sample["canonical_solution"]))
    print("\n\n")

