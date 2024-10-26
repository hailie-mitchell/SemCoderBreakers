#!/bin/bash
# Please run under recode repo

# do this if there is no such dir in your recode repo
mkdir datasets/perturbed

# create partial
python run_robust.py create_partial natgen --datasets cruxeval

# initial attempt to perturb CRUXEval
python run_robust.py perturb natgen --aug_method 0 --datasets cruxeval
