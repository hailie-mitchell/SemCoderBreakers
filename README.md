# Evaluating Robustness of SemCoder Against Semantic-Preserving Perturbations

Course project for the COMS_6998E Generative Models for Code course.

In this SemCoderBreakers project, we perform an ablation study on the [SemCoder](https://arxiv.org/pdf/2406.01006) model, which learns comprehensive code semantics to enable Code LLMs to handle complex problems such as reasoning about code execution for downstream tasks like debugging and self-refinement. Although SemCoder outperforms models such as StarCoder and CodeLlama on popular benchmarks such as EvalPlus and CRUXEval, the robustness of SemCoder against semantic-preserving perturbations is unexplored.

In this study, we compare the robustness of SemCoder relative to DeepSeekCoder, the base model without Monologue Reasoning. Perturbed versions of the EvalPlus and CRUXEVal datasets will be used to examine the effect of Monologue Reasoning on model robustness. Our robustness evaluation is based on and builds from [ReCode](https://arxiv.org/pdf/2212.10264).

## Original Dataset Retrieval
The original MBPP+ dataset is retrieved from https://github.com/evalplus/mbppplus_release.
The original HumanEval+ dataset is retrieved from https://github.com/evalplus/humanevalplus_release.
The original CRUXEval dataset is retrieved from https://github.com/facebookresearch/cruxeval/blob/main/data/cruxeval.jsonl.

## Creating Perturbations
The MBPP+, HumanEval+ and CRUXEval datasets were perturbed using [ReCode](https://arxiv.org/pdf/2212.10264), a benchmark providing general perturbations on docstrings, function names, and codes. 

The installation steps provided in the [ReCode repository's](https://github.com/amazon-science/recode) README were first followed. We updated the dataset paths in the config.json file to map to our new MBPP+, HumanEval+, and CRUXEval datasets.

### MBPP+ & HumanEval+ Perturbations
Next, we used the [perturb] option of Recode to create the perturbed versions of the datasets.

For natural language perturbations, we ran the following command:
```
python run_robust.py perturb nlaugmenter --datasets [dataset_name]
```

For function rename perturbations, we ran the following command:
```
python run_robust.py perturb func_name --datasets [dataset_name]
```

For code syntax perturbations, we ran the following command:
```
python run_robust.py perturb natgen --datasets [dataset_name]
```

For code format transformations, we ran the following command:
```
python run_robust.py perturb format --datasets [dataset_name]
```

Note that for MBPP+ and HumanEval+ benchmarks, we wanted to assess the code generation ability of SemCoder when presented with a perturbed code prefix. Therefore, we performed the function rename, code syntax, and code format perturbations on partial code.

To perturb the partial code before performing the perturbations, we ran the following command:
```
python run_robust.py create_partial natgen --datasets mbpp humaneval
```

To verify the perturbations modified the input as intended, we ran the following command to assess the perturbations on a case-by-case basis:
```
python run_robust.py analysis [perturbation_name] --aug_method [index] --models None
```

### CRUXEval Perturbations
To perturb the CRUXEval dataset, we started by modifying the format of the original dataset so that it can be handled by the ReCode framework.
Since all samples in the CRUXEval dataset have source code with function names `f`, we can optionally change all of the samples to have an alternate function name in order to later perform function name perturbations.
To reformat the CRUXEval dataset, we run the following script from our own repository:
```
cd SemCoderBreakers/cruxeval/nominal
python data_process.py for_perturb [path_to_original_dataset] [path_to_save_reformatted_dataset] --rename_func
```

To then apply the ReCode perturbations, we must update some of the ReCode framework to handle the CRUXEval dataset.
The modified ReCode files are procided in the directory `SemCoderBreakers/recode`.
We can then apply the func_name, natgen, and format perturbations from the ReCode framework to the reformatted CRUXEval dataset with a simple script.
First copy the script into the ReCode repository, and then run it:
```
cp SemCoderBreakers/cruxeval/command.sh [path_to_recode_repo]/command.sh
cd [path_to_recode_repo]
bash command.sh
```

Once the CRUXEval datasets have been perturbed, we validate that the sample inputs associated with each sample still produce the sample output in each sample when running the perturbed source code.
We also reformat the perturbed datasets back to the format of the original CRUXEval dataset so that a model can be evaluated on the perturbed datasets.
The validation can be run on either format of the perturbed dataset by modifying the `format` argument when running the validation script:
```
cd SemCoderBreakers/cruxeval
python validate_split_data.py --data_path [path_to_perturbed dataset] --format recode
python nominal/data_process.py for_eval [path_to_perturbed_dataset] [path_to_save_perturbed_reformatted_dataset]
python validate_split_data.py --data_path [path_to_perturbed_reformatted_dataset] --format eval
```

## SemCoder Experiments
### MBPP+ & HumanEval+ Experiments
Next, we use the SemCoder and DeepSeekCoder models to assess their robustness in code generation and execution reasoning against semantic-preserving perturbations of NL and code inputs.

First, the installation steps provided in the [SemCoder repository's](https://github.com/ARiSE-Lab/SemCoder) README were first followed.

Then, the run_evalplus.py scripts in the [experiments](https://github.com/ARiSE-Lab/SemCoder/tree/main/experiments) directory were updated for our analyses. It was modified such that the raw problems were extracted from the perturbed datasets rather than the unchanged datasets.

Then, the following command was run to generate solutions to the perturbed EvalPlus prompts using SemCoder. These settings remained consistent across all experiments run in our study.
```
run_evalplus.py --model_key deepseek-ai/deepseek-coder-6.7b-base --model_name_or_path semcoder/semcoder_1030 --dataset [dataset_name] --save_path [output_path] --n_batches 1     --n_problems_per_batch 1 --n_samples_per_problem 5     --max_new_tokens 100 --top_p 0.9 --temperature 0 --input_data_path [input_path]
```

The following command was run to generate solutions to the perturbed EvalPlus prompts using SemCoder-S. Only the model_name_or_path argument is changed.
```
run_evalplus.py --model_key deepseek-ai/deepseek-coder-6.7b-base --model_name_or_path semcoder/semcoder_s_1030 --dataset [dataset_name] --save_path [output_path] --n_batches 1     --n_problems_per_batch 1 --n_samples_per_problem 5     --max_new_tokens 100 --top_p 0.9 --temperature 0 --input_data_path [input_path]
```

To run the DeepSeekCoder models, we used the [EvalPlus repository](https://github.com/evalplus/evalplus). We first followed the installation steps provided in the README.

Then, we run the following command to generate the evaluation logs for each code generation by the DeepSeekCoder-Base model.
```
MBPP_OVERRIDE_PATH= [generated_solutions_path] evalplus.codegen --model "deepseek-ai/deepseek-coder-6.7b-base" --greedy --root [output_directory] --dataset mbpp --backend hf
```

Similarly, we run the following command to generate the evaluation logs for each code generation by the DeepSeekCoder-Instruct model. Only the model argument has been changed.
```
MBPP_OVERRIDE_PATH= [generated_solutions_path] evalplus.codegen --model "deepseek-ai/deepseek-coder-6.7b-instruct" --greedy --root [output_directory] --dataset mbpp --backend hf
```

We also used the EvalPlus repository to generate the Robust Pass@1, Robust Drop@1, and Robust Relative@1 scores and save them as a CSV file. To do this, we made some modifications to the run_robust.py file found in the EvalPlus repository. The updated version of the run_robust.py file can be found in the [mbpp_plus directory of this repository](https://github.com/hailie-mitchell/SemCoderBreakers/tree/main/mbpp_plus).

Then, we simply ran the following command to generate the comprehensive metrics:
```
python run_robust.py report_results --nominal_data [nominal_dataset_path] --perturbed_data [perturbed_dataset_1_path] [perturbed_dataset_2_path] [perturbed_dataset_3_path] ...
```

### CRUXEval Experiments
To run experiments on the CRUXEval datasets for SemCoder and DeepSeek-Coder models, we use the SemCoder repository. All modifications we made can be found in our own fork of the SemCoder directory, included as a [submodule here](SemCoder).
To generate model responses on the perturbed CRUXEval datasets, the SemCoder script `SemCoder/scripts/eval/eval_cruxeval.sh` was changed to use temperature 0.
The `SemCoder/experiments/cruxeval_utils.py` file was updated to point to the perturbed dataset file of interest.
Additionally, we modified the prompting in the file `SemCoder/experiments/cruxeval_prompts.py` so that prompts referred to a function rather than `function f` since we renamed all functions in the CRUXEval dataset.
Then we generated SemCoder model responses with the SemCoder script:
```
cd SemCoder
bash scripts/eval/eval_cruxeval.sh
```
And we generated DeepSeek-Coder responses with the command: 
```
bash scripts/eval/eval_cruxeval_deepseek.sh
```

Additional details about robustness evaluations of model generations for the CRUXEval datasets are available in our [cruxeval evaluation documentation](cruxeval/results/README.md).
