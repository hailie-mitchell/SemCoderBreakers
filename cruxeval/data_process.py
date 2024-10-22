import json
import sys


def for_perturb(input_file, output_file):
    """Prepare CRUXEval for code perturbation
    here instead of rewriting functions in ReCode
    we cheat them by formatting CRUXEval as code completion tasks"""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            # Change 'id' to 'task_id'
            data['task_id'] = f"CRUXEval/{data.pop('id').split('_')[1]}"
            # Split 'code' into 'prompt' and 'canonical_solution'
            header, doc, body = sep(data.pop('code'), 'f')
            data['prompt'] = header + doc
            data['canonical_solution'] = body
            # Add 'entry_point'
            data['entry_point'] = 'f'   # not safe, but actually works

            # TODO: change function names to more descriptive ones, e.g. func_to_analyze
            # By doing so, we can also implement FUNC_RECIPES

            # Write the modified data to the output file
            json.dump(data, outfile)
            outfile.write('\n')


def for_eval(input_file, output_file):
    """Process perturbed CRUXEval to original format"""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            # Merge 'prompt' and 'canonical_solution' back to code (necessary)
            data['code'] = data.pop('prompt') + data.pop('canonical_solution')
            
            # Other optional operations
            # Change 'task_id' to 'id'
            data['id'] = f"sample_{data.pop('task_id').split('/')[1]}"
            # Remove 'entry_point'
            data.pop('entry_point', None)

            # Write the modified data to the output file
            json.dump(data, outfile)
            outfile.write('\n')


def sep(code, entry_point):
    """ core function to seperate function signature (header) ||| docstring ||| code
    here entry point is the main function we need to complete
    copy from natgen/utils.py (no revision, may replace with import)
    """
    single_doc = code.find("\'\'\'")
    double_doc = code.find("\"\"\"")
    if single_doc == -1:
        doc_type = "\"\"\""
    elif double_doc == -1:
        doc_type = "\'\'\'"
    elif single_doc != -1 and double_doc != -1:
        doc_type = "\"\"\""
    else:
        print("doc_type not supported!")
        exit()
    header_end = code.find('\n', code.find(entry_point))
    header = code[:header_end + 1]
    doc_begin = code.find(doc_type, header_end)
    doc_end = code.find(f"{doc_type}\n", doc_begin + 3)
    # doc_begin != -1 and doc_end != -1, means no docstring in the code, just return "" for docstring
    doc = code[header_end+1 : doc_end+4] if doc_begin != -1 and doc_end != -1 else ""
    code = code[doc_end+4:] if doc_begin != -1 and doc_end != -1 else code[header_end+1:]
    return header, doc, code


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python data_process.py <function_name> <input_file> <output_file>")
        sys.exit(1)

    function_name = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    if function_name == "for_perturb":
        for_perturb(input_file, output_file)
    elif function_name == "for_eval":
        for_eval(input_file, output_file)
    else:
        print("Invalid function name. Use 'for_perturb' or 'for_eval'.")
