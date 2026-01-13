#!/usr/bin/env python3
"""
Create comparison bar plot for top 80 DE genes matching the PDGrapher paper style.
Adds MultiDCP results alongside baseline methods.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# All 9 cell types from the original plot
CELL_TYPES = ['HT29', 'PC3', 'HELA', 'MCF7', 'A549', 'A375', 'VCAP', 'MDAMB231', 'BT20']
CELL_TYPE_LABELS = ['Colon-HT29', 'Prostate-PC3', 'Cervix-HELA', 'Breast-MCF7', 'Lung-A549',
                    'Skin-A375', 'Prostate-VCAP', 'Breast-MDAMB231', 'Breast-BT20']

# Actual values from PDGrapher paper supplementary data (top 80 DE genes)
BASELINE_TOP80 = {
    'Baseline':   {'HT29': 0.711950, 'PC3': 0.746398, 'HELA': 0.844283, 'MCF7': 0.723575, 'A549': 0.674543, 'A375': 0.734747, 'VCAP': 0.741733, 'MDAMB231': 0.822106, 'BT20': 0.820575},
    'ChemCPA':    {'HT29': 0.769967, 'PC3': 0.776830, 'HELA': 0.836894, 'MCF7': 0.744965, 'A549': 0.764489, 'A375': 0.791160, 'VCAP': 0.800741, 'MDAMB231': 0.850846, 'BT20': 0.871604},
    'Biolord':    {'HT29': 0.740439, 'PC3': 0.764849, 'HELA': 0.837704, 'MCF7': 0.735377, 'A549': 0.743163, 'A375': 0.777956, 'VCAP': 0.815875, 'MDAMB231': 0.841267, 'BT20': 0.886093},
    'CellOT':     {'HT29': -0.000554, 'PC3': -0.000303, 'HELA': 0.001899, 'MCF7': 0.000447, 'A549': 0.003587, 'A375': -0.003145, 'VCAP': -0.005547, 'MDAMB231': -0.002449, 'BT20': 0.002469},
    'scGen':      {'HT29': 0.808667, 'PC3': 0.748368, 'HELA': 0.896050, 'MCF7': 0.829272, 'A549': 0.754721, 'A375': 0.801198, 'VCAP': 0.570052, 'MDAMB231': 0.626216, 'BT20': 0.518770},
    'PDGrapher':  {'HT29': 0.747726, 'PC3': 0.765991, 'HELA': 0.839279, 'MCF7': 0.725970, 'A549': 0.718426, 'A375': 0.773122, 'VCAP': 0.835943, 'MDAMB231': 0.841452, 'BT20': 0.872491},
}

# Method colors matching the original plot
METHOD_COLORS = {
    'Baseline': '#4a4a4a',
    'ChemCPA': '#5b9bd5',
    'Biolord': '#f4b183',
    'CellOT': '#a9d18e',
    'scGen': '#ff6b6b',
    'PDGrapher': '#70d6d6',
    'MultiDCP': '#9b59b6',
}

def load_multidcp_data(csv_file="wandb_metrics_export.csv"):
    """Load MultiDCP results from CSV."""
    df = pd.read_csv(csv_file)
    top80 = df.groupby('cell_type')['test_de_top80_r2_from_pearson_mean'].agg(['mean', 'std', 'count']).to_dict('index')
    return top80

def create_top80_plot(output_file="comparison_plot_top80.png"):
    """Create comparison plot for top 80 DE genes."""

    # Load MultiDCP data
    multidcp_top80 = load_multidcp_data()

    # Methods to plot (in order)
    methods = ['Baseline', 'ChemCPA', 'Biolord', 'CellOT', 'scGen', 'PDGrapher', 'MultiDCP']

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Settings
    bar_width = 0.11
    x = np.arange(len(CELL_TYPES))

    method_means = {m: [] for m in methods}
    method_stds = {m: [] for m in methods}

    for cell_type in CELL_TYPES:
        # Baseline methods
        for method in methods[:-1]:  # All except MultiDCP
            method_means[method].append(BASELINE_TOP80[method][cell_type])
            method_stds[method].append(0)

        # MultiDCP
        if cell_type in multidcp_top80:
            method_means['MultiDCP'].append(multidcp_top80[cell_type]['mean'])
            std_val = multidcp_top80[cell_type]['std']
            count = multidcp_top80[cell_type]['count']
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
    ax.set_ylabel(r'$R^2$ (top 80 DE genes)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CELL_TYPE_LABELS, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.6, len(CELL_TYPES) - 0.4)

    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=4, fontsize=8, framealpha=0.9)

    # Grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_file}")
    plt.close()

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE - Top 80 DE genes R²")
    print("="*80)
    print(f"{'Cell Type':<15}", end="")
    for method in methods:
        print(f"{method:<12}", end="")
    print()
    print("-"*80)
    for i, cell_type in enumerate(CELL_TYPES):
        print(f"{CELL_TYPE_LABELS[i]:<15}", end="")
        for method in methods[:-1]:
            print(f"{BASELINE_TOP80[method][cell_type]:<12.3f}", end="")
        if cell_type in multidcp_top80:
            print(f"{multidcp_top80[cell_type]['mean']:<12.3f}", end="")
        else:
            print(f"{'N/A':<12}", end="")
        print()

if __name__ == "__main__":
    create_top80_plot()
