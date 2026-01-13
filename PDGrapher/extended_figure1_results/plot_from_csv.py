#!/usr/bin/env python3
"""
Create bar plots from exported wandb CSV data.
Aggregates across folds to show mean ± std for each cell type.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

# Configuration
METRICS = [
    "test_de_top20_r2_from_pearson_mean",
    "test_de_top40_r2_from_pearson_mean"
]

def plot_metrics(df, output_prefix="metrics"):
    """Create bar plots for each metric, grouped by cell type with error bars."""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 6)
    
    for metric in METRICS:
        # Aggregate by cell type: mean and std across folds
        agg_df = df.groupby('cell_type')[metric].agg(['mean', 'std', 'count']).reset_index()
        agg_df = agg_df.sort_values('mean', ascending=False)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(agg_df))
        bars = ax.bar(x, agg_df['mean'], yerr=agg_df['std'], 
                      capsize=5, alpha=0.8, color='steelblue', 
                      error_kw={'linewidth': 2, 'elinewidth': 1.5})
        
        # Customize
        ax.set_xlabel('Cell Type', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} by Cell Type\n(Mean ± Std across folds)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(agg_df['cell_type'], rotation=45, ha='right')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Add value labels on bars
        for i, (mean_val, std_val, count) in enumerate(zip(agg_df['mean'], agg_df['std'], agg_df['count'])):
            ax.text(i, mean_val + std_val + 0.01, f'{mean_val:.3f}\n(n={int(count)})', 
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save
        output_file = f"{output_prefix}_{metric}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {output_file}")
        
        plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    for metric in METRICS:
        print(f"\n{metric}:")
        summary = df.groupby('cell_type')[metric].agg(['mean', 'std', 'count'])
        summary = summary.sort_values('mean', ascending=False)
        print(summary.to_string())

def main():
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "wandb_metrics_export.csv"
    
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} runs")
    print(f"Cell types: {sorted(df['cell_type'].unique())}")
    
    # Filter out rows with missing metrics
    df_clean = df.dropna(subset=METRICS)
    print(f"\nAfter filtering NaN values: {len(df_clean)} runs")
    
    if len(df_clean) == 0:
        print("\nERROR: No runs found with the specified metrics!")
        print("Available columns:")
        print(df.columns.tolist())
        return
    
    print("\nGenerating plots...")
    plot_metrics(df_clean)
    
    print("\n✓ Done! Check the generated PNG files.")

if __name__ == "__main__":
    main()
