import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import re

file_names = ['dice_summary_given.txt', 'dice_summary_learned.txt', 'dice_summary_vanilla.txt']


def parse_dice_summary(file_name):
    with open(file_name, 'r') as f:
        content = f.read()

    method_match = re.search(r'=== DiCE Counterfactual Summary \((.*?)\) ===', content)
    method = method_match.group(1).capitalize() if method_match else "Unknown"

    def extract_value(label):
        match = re.search(fr'{re.escape(label)}:\s+([\d\.]+)', content)
        return float(match.group(1)) if match else None

    return {
        'Method': method,
        'Gower Proximity': extract_value('Average Gower proximity'),
        'Sparsity': extract_value('Average sparsity'),
        'Diversity': extract_value('Average diversity (Gower)'),
        'Plausibility': extract_value('Average kNN plausibility (Gower)'),
        'Runtime (s)': extract_value('Average runtime (successful)')
    }


all_data = [parse_dice_summary(f) for f in file_names]
df = pd.DataFrame(all_data)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
fig.suptitle('Comparison of DiCE Methods', fontsize=16)

metrics = ['Gower Proximity', 'Sparsity', 'Runtime (s)', 'Diversity', 'Plausibility']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, metric in enumerate(metrics):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    bars = ax.bar(df['Method'], df[metric], color=colors, alpha=0.8)
    ax.set_title(metric)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.4f}', ha='center', va='bottom')

axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('dice_summary_plot.png')
