import re

import numpy as np
import pandas as pd
from scipy import stats

report_files = {
    'given': 'dice_counterfactual_report_all_given.txt',
    'learned': 'dice_counterfactual_report_all_learned.txt',
    'vanilla': 'dice_counterfactual_report_all_vanilla.txt'
}


def parse_raw_report(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    blocks = re.split(r'={5,}', content)
    data = {}
    for block in blocks:
        idx_m = re.search(r'Index:\s+(\d+)', block)
        gow_m = re.search(r'Gower:\s+([\d\.]+)', block)
        if idx_m and gow_m:
            idx = int(idx_m.group(1))
            sparsity = len(re.findall(r'^\s+-\s+', block, re.MULTILINE))
            data[idx] = {'gower': float(gow_m.group(1)), 'sparsity': sparsity}
    return data


raw_results = {name: parse_raw_report(path) for name, path in report_files.items()}
common_idx = sorted(list(set(raw_results['given'].keys()) &
                         set(raw_results['learned'].keys()) &
                         set(raw_results['vanilla'].keys())))


print("\n--- STATISTISCHE SIGNIFIKANZ (Wilcoxon-Test) ---")
test_pairs = [('learned', 'given'), ('learned', 'vanilla'), ('given', 'vanilla')]

for metric in ['gower', 'sparsity']:
    print(f"\nMetrik: {metric.upper()}")
    for m1, m2 in test_pairs:
        d1 = [raw_results[m1][i][metric] for i in common_idx]
        d2 = [raw_results[m2][i][metric] for i in common_idx]
        stat, p = stats.wilcoxon(d1, d2)
        sig = "*** (p<0.001)" if p < 0.001 else "** (p<0.01)" if p < 0.05 else "n.s."
        print(f"  {m1.capitalize()} vs {m2.capitalize()}: p = {p:.4e} {sig}")