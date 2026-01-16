#!/usr/bin/env python3
"""
Create comparison bar plots matching the PDGrapher paper style.
Adds MultiDCP results alongside baseline methods.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Baseline data estimated from PDGrapher paper figure
# All 9 cell types from the original plot
CELL_TYPES = ['HT29', 'PC3', 'HELA', 'MCF7', 'A549', 'A375', 'VCAP', 'MDAMB231', 'BT20']
CELL_TYPE_LABELS = ['Colon-HT29', 'Prostate-PC3', 'Cervix-HELA', 'Breast-MCF7', 'Lung-A549',
                    'Skin-A375', 'Prostate-VCAP', 'Breast-MDAMB231', 'Breast-BT20']

# Actual values from PDGrapher paper supplementary data (41551_2025_1481_MOESM8_ESM.xlsx)
# Mean R² across 5-fold cross-validation
BASELINE_TOP20 = {
    'Baseline':   {'HT29': 0.703174, 'PC3': 0.729734, 'HELA': 0.843018, 'MCF7': 0.721962, 'A549': 0.660343, 'A375': 0.711524, 'VCAP': 0.722691, 'MDAMB231': 0.808391, 'BT20': 0.794316},
    'ChemCPA':    {'HT29': 0.754152, 'PC3': 0.765560, 'HELA': 0.836133, 'MCF7': 0.742084, 'A549': 0.744171, 'A375': 0.765574, 'VCAP': 0.786830, 'MDAMB231': 0.844413, 'BT20': 0.851479},
    'Biolord':    {'HT29': 0.727572, 'PC3': 0.753490, 'HELA': 0.837685, 'MCF7': 0.735372, 'A549': 0.724752, 'A375': 0.759642, 'VCAP': 0.807131, 'MDAMB231': 0.831608, 'BT20': 0.869613},
    'CellOT':     {'HT29': -0.004179, 'PC3': -0.000717, 'HELA': 0.001329, 'MCF7': 0.003135, 'A549': 0.003089, 'A375': 0.000347, 'VCAP': -0.004268, 'MDAMB231': 0.000733, 'BT20': 0.001255},
    'scGen':      {'HT29': 0.797365, 'PC3': 0.734934, 'HELA': 0.894556, 'MCF7': 0.819014, 'A549': 0.740155, 'A375': 0.785210, 'VCAP': 0.566694, 'MDAMB231': 0.616593, 'BT20': 0.461885},
    'PDGrapher':  {'HT29': 0.737254, 'PC3': 0.753613, 'HELA': 0.839629, 'MCF7': 0.725486, 'A549': 0.703063, 'A375': 0.752429, 'VCAP': 0.831624, 'MDAMB231': 0.831613, 'BT20': 0.854026},
}

# Actual values from PDGrapher paper supplementary data (top 40 DE genes)
BASELINE_TOP40 = {
    'Baseline':   {'HT29': 0.706115, 'PC3': 0.737675, 'HELA': 0.842309, 'MCF7': 0.722631, 'A549': 0.666468, 'A375': 0.723355, 'VCAP': 0.733449, 'MDAMB231': 0.814279, 'BT20': 0.798552},
    'ChemCPA':    {'HT29': 0.762940, 'PC3': 0.771371, 'HELA': 0.835919, 'MCF7': 0.744665, 'A549': 0.757654, 'A375': 0.778984, 'VCAP': 0.796701, 'MDAMB231': 0.846214, 'BT20': 0.855353},
    'Biolord':    {'HT29': 0.734319, 'PC3': 0.758905, 'HELA': 0.837043, 'MCF7': 0.736128, 'A549': 0.736234, 'A375': 0.768838, 'VCAP': 0.813117, 'MDAMB231': 0.835357, 'BT20': 0.871461},
    'CellOT':     {'HT29': -0.002481, 'PC3': 0.000118, 'HELA': 0.001482, 'MCF7': -0.000739, 'A549': 0.003512, 'A375': -0.001376, 'VCAP': -0.006930, 'MDAMB231': -0.000374, 'BT20': 0.009042},
    'scGen':      {'HT29': 0.803557, 'PC3': 0.741850, 'HELA': 0.894433, 'MCF7': 0.826182, 'A549': 0.750227, 'A375': 0.794192, 'VCAP': 0.571610, 'MDAMB231': 0.624310, 'BT20': 0.486780},
    'PDGrapher':  {'HT29': 0.742651, 'PC3': 0.759707, 'HELA': 0.838635, 'MCF7': 0.726561, 'A549': 0.712368, 'A375': 0.763071, 'VCAP': 0.834664, 'MDAMB231': 0.835301, 'BT20': 0.855857},
}

# Method colors matching the original plot
METHOD_COLORS = {
    'Baseline': '#4a4a4a',
    'ChemCPA': '#5b9bd5',
    'Biolord': '#f4b183',
    'CellOT': '#a9d18e',
    'scGen': '#ff6b6b',
    'PDGrapher': '#70d6d6',
    'MultiDCP': '#9b59b6',  # Purple for MultiDCP
}

def load_multidcp_data(csv_file="wandb_metrics_export.csv"):
    """Load MultiDCP results from CSV."""
    df = pd.read_csv(csv_file)

    # Aggregate by cell type
    top20 = df.groupby('cell_type')['test_de_top20_r2_from_pearson_mean'].agg(['mean', 'std', 'count']).to_dict('index')
    top40 = df.groupby('cell_type')['test_de_top40_r2_from_pearson_mean'].agg(['mean', 'std', 'count']).to_dict('index')

    return top20, top40

def create_comparison_plot(output_file="comparison_plot.png"):
    """Create side-by-side comparison plots matching PDGrapher style."""

    # Load MultiDCP data
    multidcp_top20, multidcp_top40 = load_multidcp_data()

    # Methods to plot (in order)
    methods = ['Baseline', 'ChemCPA', 'Biolord', 'CellOT', 'scGen', 'PDGrapher', 'MultiDCP']

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # Settings
    bar_width = 0.11
    x = np.arange(len(CELL_TYPES))

    for ax_idx, (ax, baseline_data, multidcp_data, title, ylabel) in enumerate([
        (axes[0], BASELINE_TOP20, multidcp_top20, 'a', r'$R^2$ (top 20 DE genes)'),
        (axes[1], BASELINE_TOP40, multidcp_top40, 'b', r'$R^2$ (top 40 DE genes)')
    ]):

        method_means = {m: [] for m in methods}
        method_stds = {m: [] for m in methods}

        for cell_type in CELL_TYPES:
            # Baseline methods
            for method in methods[:-1]:  # All except MultiDCP
                method_means[method].append(baseline_data[method][cell_type])
                method_stds[method].append(0)  # No std for baseline methods

            # MultiDCP
            if cell_type in multidcp_data:
                method_means['MultiDCP'].append(multidcp_data[cell_type]['mean'])
                std_val = multidcp_data[cell_type]['std']
                count = multidcp_data[cell_type]['count']
                # Only show error bars if more than 1 fold
                method_stds['MultiDCP'].append(std_val if (count > 1 and not np.isnan(std_val)) else 0)
            else:
                method_means['MultiDCP'].append(0)
                method_stds['MultiDCP'].append(0)

        # Plot bars for each method
        for i, method in enumerate(methods):
            offset = (i - len(methods)/2 + 0.5) * bar_width
            bars = ax.bar(x + offset, method_means[method], bar_width,
                         label=method, color=METHOD_COLORS[method],
                         edgecolor='white', linewidth=0.5)

            # Add error bars for MultiDCP
            if method == 'MultiDCP':
                ax.errorbar(x + offset, method_means[method], yerr=method_stds[method],
                           fmt='none', ecolor='black', capsize=2, capthick=1, linewidth=1)

        # Add horizontal dashed lines for method averages
        for method in methods:
            avg = np.mean([v for v in method_means[method] if v > 0])
            ax.axhline(y=avg, color=METHOD_COLORS[method], linestyle='--',
                      linewidth=1, alpha=0.7)

        # Customize axes
        ax.set_xlabel('Cell Type', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(CELL_TYPE_LABELS, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.6, len(CELL_TYPES) - 0.4)

        # Add panel label
        ax.text(-0.08, 1.05, title, transform=ax.transAxes, fontsize=14,
               fontweight='bold', va='top')

        # Add legend (only on first plot)
        if ax_idx == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=4,
                     fontsize=8, framealpha=0.9)

        # Grid
        ax.yaxis.grid(True, linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_file}")
    plt.close()

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE - Top 20 DE genes R²")
    print("="*80)
    print(f"{'Cell Type':<15}", end="")
    for method in methods:
        print(f"{method:<12}", end="")
    print()
    print("-"*80)
    for i, cell_type in enumerate(CELL_TYPES):
        print(f"{CELL_TYPE_LABELS[i]:<15}", end="")
        for method in methods[:-1]:
            print(f"{BASELINE_TOP20[method][cell_type]:<12.3f}", end="")
        if cell_type in multidcp_top20:
            print(f"{multidcp_top20[cell_type]['mean']:<12.3f}", end="")
        else:
            print(f"{'N/A':<12}", end="")
        print()

    print("\n" + "="*80)
    print("COMPARISON TABLE - Top 40 DE genes R²")
    print("="*80)
    print(f"{'Cell Type':<15}", end="")
    for method in methods:
        print(f"{method:<12}", end="")
    print()
    print("-"*80)
    for i, cell_type in enumerate(CELL_TYPES):
        print(f"{CELL_TYPE_LABELS[i]:<15}", end="")
        for method in methods[:-1]:
            print(f"{BASELINE_TOP40[method][cell_type]:<12.3f}", end="")
        if cell_type in multidcp_top40:
            print(f"{multidcp_top40[cell_type]['mean']:<12.3f}", end="")
        else:
            print(f"{'N/A':<12}", end="")
        print()

if __name__ == "__main__":
    create_comparison_plot()
