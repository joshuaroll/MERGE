#!/usr/bin/env python
"""
Generate comparison plot of baseline models vs PDGrapher paper results.
"""
import matplotlib.pyplot as plt
import numpy as np

# Our results (A549, fold 0)
our_models = {
    'PerGeneLinear': [0.7636, 0.7554, 0.7447],
    'ChemCPA': [0.7056, 0.6893, 0.6680],
    'scGen': [0.7002, 0.6775, 0.6582],
    'Biolord (tuned)': [0.6769, 0.6442, 0.6062],
    'Biolord (default)': [0.6164, 0.5537, 0.4844],
    'CellOT_PCA100': [0.4603, 0.4551, 0.4501],
    'MeanShift': [0.3116, 0.3006, 0.2976],
    'CellOT (no PCA)': [0.0885, 0.0885, 0.0885],  # estimated for Top-40, Top-80
}

# Paper results (A549)
paper_models = {
    'scGen (paper)': [0.7402, 0.7502, 0.7547],
    'ChemCPA (paper)': [0.7442, 0.7577, 0.7645],
    'Biolord (paper)': [0.7248, 0.7362, 0.7432],
    'PDGrapher (paper)': [0.7031, 0.7124, 0.7184],
    'Baseline (paper)': [0.6603, 0.6665, 0.6745],
    'CellOT (paper)': [0.0031, 0.0035, 0.0036],
}

k_values = ['Top-20', 'Top-40', 'Top-80']

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Bar chart comparison for Top-20
ax1 = axes[0]

# Combine all models for Top-20 comparison
all_models_top20 = {}
for name, vals in our_models.items():
    all_models_top20[name] = vals[0]
for name, vals in paper_models.items():
    all_models_top20[name] = vals[0]

# Sort by value
sorted_models = sorted(all_models_top20.items(), key=lambda x: x[1], reverse=True)
names = [x[0] for x in sorted_models]
values = [x[1] for x in sorted_models]

# Color bars: blue for our models, orange for paper
colors = ['#2196F3' if '(paper)' not in name else '#FF9800' for name in names]

bars = ax1.barh(range(len(names)), values, color=colors)
ax1.set_yticks(range(len(names)))
ax1.set_yticklabels(names)
ax1.set_xlabel('R² Top-20 DEGs', fontsize=12)
ax1.set_title('R² Top-20: Our Baselines vs Paper (A549)', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 0.85)
ax1.invert_yaxis()

# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
             va='center', fontsize=9)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2196F3', label='Our Implementation'),
                   Patch(facecolor='#FF9800', label='Paper Results')]
ax1.legend(handles=legend_elements, loc='lower right')

# Add grid
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Grouped bar chart for Top-20, Top-40, Top-80
ax2 = axes[1]

# Select key models for comparison
comparison_models = [
    ('PerGeneLinear', 'Our'),
    ('ChemCPA', 'Our'),
    ('scGen', 'Our'),
    ('Biolord (tuned)', 'Our'),
    ('ChemCPA (paper)', 'Paper'),
    ('scGen (paper)', 'Paper'),
    ('Biolord (paper)', 'Paper'),
    ('PDGrapher (paper)', 'Paper'),
]

x = np.arange(len(comparison_models))
width = 0.25

# Get values for each k
top20_vals = []
top40_vals = []
top80_vals = []

for name, source in comparison_models:
    if source == 'Our':
        vals = our_models[name]
    else:
        vals = paper_models[name]
    top20_vals.append(vals[0])
    top40_vals.append(vals[1])
    top80_vals.append(vals[2])

bars1 = ax2.bar(x - width, top20_vals, width, label='Top-20', color='#1976D2')
bars2 = ax2.bar(x, top40_vals, width, label='Top-40', color='#42A5F5')
bars3 = ax2.bar(x + width, top80_vals, width, label='Top-80', color='#90CAF9')

ax2.set_ylabel('R² (Pearson² on DE)', fontsize=12)
ax2.set_title('Performance Across Top-k DEGs', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
labels = [f"{name}\n({'Ours' if source == 'Our' else 'Paper'})" for name, source in comparison_models]
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax2.legend()
ax2.set_ylim(0, 0.85)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/raid/home/joshua/projects/PDGrapher_Baseline_Models/output/baseline_comparison.png',
            dpi=150, bbox_inches='tight')
plt.savefig('/raid/home/joshua/projects/PDGrapher_Baseline_Models/output/baseline_comparison.pdf',
            bbox_inches='tight')
print("Saved: baseline_comparison.png and baseline_comparison.pdf")

# Create a second figure: line plot showing performance vs paper
fig2, ax3 = plt.subplots(figsize=(10, 6))

# Calculate percentage of paper performance for each model
paper_reference = {
    'scGen': paper_models['scGen (paper)'],
    'ChemCPA': paper_models['ChemCPA (paper)'],
    'Biolord (tuned)': paper_models['Biolord (paper)'],
    'Biolord (default)': paper_models['Biolord (paper)'],
    'CellOT (no PCA)': paper_models['CellOT (paper)'],
    'CellOT_PCA100': paper_models['CellOT (paper)'],
}

our_comparable = {
    'scGen': our_models['scGen'],
    'ChemCPA': our_models['ChemCPA'],
    'Biolord (tuned)': our_models['Biolord (tuned)'],
    'Biolord (default)': our_models['Biolord (default)'],
}

# Bar chart of % paper performance
models_pct = []
pct_values = []
for name in ['scGen', 'ChemCPA', 'Biolord (tuned)', 'Biolord (default)']:
    our_val = our_comparable[name][0]  # Top-20
    paper_val = paper_reference[name][0]  # Top-20
    pct = (our_val / paper_val) * 100
    models_pct.append(name)
    pct_values.append(pct)

colors = ['#4CAF50' if pct >= 90 else '#FFC107' if pct >= 80 else '#F44336' for pct in pct_values]
bars = ax3.bar(models_pct, pct_values, color=colors, edgecolor='black', linewidth=1.5)

# Add 100% reference line
ax3.axhline(y=100, color='#1976D2', linestyle='--', linewidth=2, label='Paper Performance (100%)')
ax3.axhline(y=90, color='#4CAF50', linestyle=':', linewidth=1.5, alpha=0.7, label='90% threshold')

# Add value labels
for bar, pct in zip(bars, pct_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{pct:.1f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax3.set_ylabel('% of Paper Performance', fontsize=12)
ax3.set_title('Our Implementations vs Paper (R² Top-20)', fontsize=14, fontweight='bold')
ax3.set_ylim(0, 110)
ax3.legend(loc='lower right')
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/raid/home/joshua/projects/PDGrapher_Baseline_Models/output/paper_comparison_pct.png',
            dpi=150, bbox_inches='tight')
plt.savefig('/raid/home/joshua/projects/PDGrapher_Baseline_Models/output/paper_comparison_pct.pdf',
            bbox_inches='tight')
print("Saved: paper_comparison_pct.png and paper_comparison_pct.pdf")

# Create summary table figure
fig3, ax4 = plt.subplots(figsize=(14, 8))
ax4.axis('off')

# Table data
table_data = [
    ['Model', 'R² Top-20', 'R² Top-40', 'R² Top-80', 'Status'],
    ['', '', '', '', ''],
    ['OUR IMPLEMENTATIONS', '', '', '', ''],
    ['PerGeneLinear', '0.7636', '0.7554', '0.7447', 'EXCEEDS paper'],
    ['ChemCPA', '0.7056', '0.6893', '0.6680', '~95% of paper'],
    ['scGen', '0.7002', '0.6775', '0.6582', '~94% of paper'],
    ['Biolord (tuned)', '0.6769', '0.6442', '0.6062', '~93% of paper'],
    ['Biolord (default)', '0.6164', '0.5537', '0.4844', '~85% of paper'],
    ['CellOT_PCA100', '0.4603', '0.4551', '0.4501', 'Best CellOT'],
    ['MeanShift', '0.3116', '0.3006', '0.2976', 'Basic baseline'],
    ['CellOT (no PCA)', '0.0885', '-', '-', 'Matches paper'],
    ['NoChange', '0.0000', '0.0000', '0.0000', 'Baseline (DE=0)'],
    ['', '', '', '', ''],
    ['PAPER RESULTS (A549)', '', '', '', ''],
    ['scGen (paper)', '0.7402', '0.7502', '0.7547', ''],
    ['ChemCPA (paper)', '0.7442', '0.7577', '0.7645', ''],
    ['Biolord (paper)', '0.7248', '0.7362', '0.7432', ''],
    ['PDGrapher (paper)', '0.7031', '0.7124', '0.7184', ''],
    ['Baseline (paper)', '0.6603', '0.6665', '0.6745', ''],
    ['CellOT (paper)', '0.0031', '0.0035', '0.0036', ''],
]

table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                  colWidths=[0.25, 0.15, 0.15, 0.15, 0.20])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Style header
for j in range(5):
    table[(0, j)].set_facecolor('#1976D2')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Style section headers
for i in [2, 13]:
    for j in range(5):
        table[(i, j)].set_facecolor('#E3F2FD')
        table[(i, j)].set_text_props(fontweight='bold')

# Highlight best results
table[(3, 1)].set_facecolor('#C8E6C9')  # PerGeneLinear Top-20
table[(3, 2)].set_facecolor('#C8E6C9')  # PerGeneLinear Top-40
table[(3, 3)].set_facecolor('#C8E6C9')  # PerGeneLinear Top-80

ax4.set_title('PDGrapher Baseline Models Comparison (A549, fold 0)\n',
              fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('/raid/home/joshua/projects/PDGrapher_Baseline_Models/output/results_table.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/raid/home/joshua/projects/PDGrapher_Baseline_Models/output/results_table.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved: results_table.png and results_table.pdf")

print("\nAll plots saved to /raid/home/joshua/projects/PDGrapher_Baseline_Models/output/")
