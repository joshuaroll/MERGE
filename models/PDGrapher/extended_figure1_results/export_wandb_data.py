#!/usr/bin/env python3
"""
Export wandb data to CSV - run this on your server where wandb is configured.
Then use the plotting script on the exported CSV.
"""

import wandb
import pandas as pd
import json
import re
from datetime import datetime

# Configuration
PROJECT = "joshroll/MultiDCP_AE_DE"
OUTPUT_FILE = "wandb_metrics_export.csv"
GROUP_NAME = "MultiDCP_Comparison"

def parse_run_name(name):
    """Parse run name like 'multidcp_de_A375_fold1_sd343' to extract cell_type and fold."""
    # Pattern: multidcp_de_{cell_type}_fold{fold}_sd{seed}
    match = re.match(r'multidcp_de_(.+)_fold(\d+)_sd\d+', name)
    if match:
        return match.group(1), int(match.group(2))
    return 'unknown', None

def export_wandb_data():
    """Pull runs from specified group and export to CSV."""
    api = wandb.Api()

    # Filter runs by group name
    runs = api.runs(PROJECT, filters={"group": GROUP_NAME})
    runs_list = list(runs)

    print(f"Found {len(runs_list)} runs in group '{GROUP_NAME}'")

    data = []
    for run in runs_list:
        config = run.config
        summary = run.summary._json_dict

        # Parse cell_type and fold from run name
        cell_type, fold = parse_run_name(run.name)

        row = {
            'run_id': run.id,
            'run_name': run.name,
            'cell_type': cell_type,
            'fold': fold,
            'test_de_top20_r2_from_pearson_mean': summary.get('test_de_top20_r2_from_pearson_mean', None),
            'test_de_top40_r2_from_pearson_mean': summary.get('test_de_top40_r2_from_pearson_mean', None),
            'test_de_top80_r2_from_pearson_mean': summary.get('test_de_top80_r2_from_pearson_mean', None),
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Exported {len(df)} runs to {OUTPUT_FILE}")
    print(f"\nPreview:")
    print(df.head(10))
    print(f"\nCell types: {sorted(df['cell_type'].unique())}")

    return df

if __name__ == "__main__":
    export_wandb_data()
