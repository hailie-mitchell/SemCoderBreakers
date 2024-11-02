import json
import argparse


ALT_FUNC_NAME = "operation_to_perform"


def rename_function(code):
    lines = code.splitlines()
    for i in range(len(lines)):
        if lines[i].strip().startswith('def f('):
            lines[i] = lines[i].replace('def f', f'def {ALT_FUNC_NAME}', 1)
    return '\n'.join(lines)


def for_perturb(input_file, output_file, rename_func=False):
    """Prepare CRUXEval for code perturbation
    here instead of rewriting functions in ReCode
    we cheat them by formatting CRUXEval as code completion tasks"""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)

            # Change 'id' to 'task_id'
            data['task_id'] = f"CRUXEval/{data.pop('id').split('_')[1]}"

            code = data.pop('code')
            # Rename function if requested
            code = rename_function(code) if rename_func else code
            entry_point = ALT_FUNC_NAME if rename_func else 'f'

            # Split 'code' into 'prompt' and 'canonical_solution'
            header, doc, body = sep(code, entry_point)
            data['prompt'] = header + doc
            data['canonical_solution'] = body

            # Add 'entry_point'
            data['entry_point'] = entry_point

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
    parser = argparse.ArgumentParser(description="Process data with specified function.")
    parser.add_argument('function', choices=['for_perturb', 'for_eval'], 
                        help="Function to execute: 'for_perturb' or 'for_eval'.")
    parser.add_argument('input_file', help="Input file path.")
    parser.add_argument('output_file', help="Output file path.")
    parser.add_argument('--rename_func', type=bool, default=False,
                        help="Specify whether to rename (True or False). Default is False.")

    args = parser.parse_args()

    if args.function == "for_perturb":
        for_perturb(args.input_file, args.output_file, args.rename_func)
    elif args.function == "for_eval":
        for_eval(args.input_file, args.output_file)
