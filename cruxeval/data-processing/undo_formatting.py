import json

DATAPATH = "cruxeval_partial.jsonl"
NEW_DATAPATH = "cruxeval_perturbed_formatted.jsonl"

with open(DATAPATH, "r") as infile, open(NEW_DATAPATH, "w") as outfile:
    for line in infile:
        sample = json.loads(line.strip())
        
        sample["code"] = sample["prompt"] + sample["canonical_solution"]
        sample.pop("prompt")
        sample.pop("canonical_solution")
        sample.pop("entry_point")
        sample.pop("partial")

        outfile.write(json.dumps(sample) + "\n")
