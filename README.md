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

## SemCoder
Next, we use the [SemCoder](https://arxiv.org/pdf/2406.01006) model to assess its robustness in code generation and execution reasoning against semantic-preserving perturbations of NL and code inputs.

The installation steps provided in the [SemCoder repository's](https://github.com/ARiSE-Lab/SemCoder) README were first followed.

The run_cruxeval.py and run_evalplus.py scripts in the [experiments](https://github.com/ARiSE-Lab/SemCoder/tree/main/experiments) directory were updated for our analyses. It was modified such that the raw problems were extracted from the perturbed datasets rather than the unchanged datasets.

Then, the following command was run to generate solutions to the perturbed EvalPlus prompts using SemCoder. These settings remained consistent across all experiments run in our study. Note that the CRUXEval generations were computed using the same command but run_cruxeval.py was used.
```
run_evalplus.py --model_key deepseek-ai/deepseek-coder-6.7b-base --dataset [dataset_name] --save_path [output_path] --n_batches 1     --n_problems_per_batch 1 --n_samples_per_problem 5     --max_new_tokens 100 --top_p 0.9 --temperature 0 --input_data_path [input_path]
```