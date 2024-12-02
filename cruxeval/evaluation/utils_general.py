# Copyright (c) Meta Platforms, Inc. and affiliates.

import re
import numpy as np
from utils_execute import check_correctness

def get_func_name(code):
    func_name = re.search(r'def\s+([^\s(]+)\s*\(', code).group(1)
    return func_name

def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def evaluate_score(args):
    gs, (c, i, o), mode = args
    func_name = get_func_name(c)

    execution_results = []
    for g in gs:
        if mode == "input" and f"{func_name}(" not in g:
            pass
        elif mode == "output" and f"{func_name}({i})" in g:
            pass
        else:
            code_to_execute = f"{c}\nassert {o} == {g}"
            execution_results.append(check_correctness(code_to_execute, 3))
    if True not in execution_results:
        execution_results = [False] * len(gs)
    return execution_results