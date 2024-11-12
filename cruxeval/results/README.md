## Robustness Evaluation

Since execution reasoning is not implemented in ReCode, we follow the evaluation pipline for CRUXEval in [SemCoder](https://github.com/ARiSE-Lab/SemCoder/tree/main/experiments) to obtain the regular pass@k metrics.

Then, to calculate robustness metrics $`\text{RP}_{s}@k`$, $`\text{RD}_{s}@k`$, and $`\text{RR}_{s}@k`$ defined in [ReCode paper](https://arxiv.org/abs/2212.10264), we create `eval_robustness.py` specifically for CRUXEval.

To run the robustness evaluation, use the following example command:
```sh
python eval_robustness.py --granularity category --method format --config ../../recode/config.json\
--models semcoder_s_1030 --n_outputs 1
# we have some default value, so it is equivalent to:
# python eval_robustness.py --method format --config ../../recode/config.json
```

To get robustness metrics for a specific perturbation method, please add an `--aug_method` argument in your command:
```sh
python eval_robustness.py --granularity category --method format --config ../../recode/config.json\
--aug_method 0 --models semcoder_s_1030 --n_outputs 1
# or simplified as 
# python eval_robustness.py --method format --config ../../recode/config.json --aug_method 0
```

If you're using direct prediction for CRUXEval in SemCoder pipline, you can specify the `--mode` argument to obtain the robustness metrics for that:
```sh
python eval_robustness.py --granularity category --method format --config ../../recode/config.json\
--models semcoder_s_1030 --mode direct --n_outputs 1
# or simplified as 
# python eval_robustness.py --method format --config ../../recode/config.json --mode direct
##
# and you may specify the method as well
# python eval_robustness.py --method format --config ../../recode/config.json --mode direct --aug_method 0
```
