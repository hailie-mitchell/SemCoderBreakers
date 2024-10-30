import json

DATAPATH = "cruxeval.jsonl"
NEW_DATAPATH = "cruxeval_formatted.jsonl"

with open(DATAPATH, "r") as infile, open(NEW_DATAPATH, "w") as outfile:
    for line in infile:
        sample = json.loads(line.strip())
        sample["entry_point"] = "f"
        sample["prompt"] = sample["code"]
        sample["canonical_solution"] = ""

        """
        # insert halfline flag to split: recode perturb.py create_partial_code()
        temp_code = sample["code"]
        if temp_code[-1] != "\n": temp_code += "\n"
        tmp, cnt = 0, 0
        while tmp != -1:
            start = tmp + 1
            tmp = temp_code.find('\n', start)
            if tmp == -1: break
            cnt += 1

        assert cnt > 0
        half = cnt // 2
        tmp = 0
        while half > 0:
            idx = temp_code.find('\n', tmp)
            tmp = idx + 1
            half -= 1

        prefix = temp_code[:idx + 1]
        suffix = temp_code[idx + 1:]

        post_indent_buffer = ""
        for ch in temp_code[idx + 1:]:
            if ch in [" ", "\t"]:
                post_indent_buffer += ch 
            else:
                break
        pre_indent_buffer = ""
        for ch in temp_code[temp_code[: idx].rfind('\n') + 1: idx]:
            if ch in [" ", "\t"]:
                pre_indent_buffer += ch
            else:
                break

        indent_buffer = post_indent_buffer if len(pre_indent_buffer) < len(post_indent_buffer) else pre_indent_buffer
        split_flag = indent_buffer + "# print('@@this is the line to split##')\n"
        sample["partial"] = prefix + split_flag + suffix
        """

        outfile.write(json.dumps(sample) + "\n")

