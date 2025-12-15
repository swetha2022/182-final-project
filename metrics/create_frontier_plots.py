import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from matplotlib.ticker import PercentFormatter

def read_csv_to_dict(filename):
    data = []
    with open(filename, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def create_frontier_plot(finetune_opt):
    """
    Create a frontier plot showing finetuning accuracy vs pretraining accuracy.
    
    Args:
        finetune_opt: "muon" or "adam" to specify which finetuning optimizer to plot
    """
    # Find all files with "all" prefix in raw_data directory
    raw_data_dir = "metrics/raw_data"
    all_files = [f for f in os.listdir(raw_data_dir) if f.startswith("all_") and f.endswith(".csv")]
    
    # Filter files by finetune optimizer
    pretrain_acc_file = None
    test_acc_file = None
    
    for f in all_files:
        if f"finetune_{finetune_opt}" in f:
            if "pretrain_acc" in f:
                pretrain_acc_file = os.path.join(raw_data_dir, f)
            elif "test_acc" in f:
                test_acc_file = os.path.join(raw_data_dir, f)
    
    if not pretrain_acc_file or not test_acc_file:
        raise FileNotFoundError(f"Could not find 'all' files for finetune optimizer: {finetune_opt}")
    
    # Read both CSV files
    pretrain_data = read_csv_to_dict(pretrain_acc_file)
    test_data = read_csv_to_dict(test_acc_file)
    
    # Get the last epoch (last row)
    last_pretrain_row = pretrain_data[-1]
    last_test_row = test_data[-1]
    
    # Extract all group columns (runs)
    # Each group has columns: mean, MIN, MAX, _step, etc.
    # We want the mean columns for pretrain_accuracy and test/accuracy
    pretrain_accs = []
    finetune_accs = []
    
    # Find all unique groups (runs) by extracting group prefix from columns
    # Format: "Group: alexnet-pretrain-{pretrain_opt}-finetune-{finetune_opt}-lr{lr}-preratio{ratio}-group - test/pretrain_accuracy"
    group_prefixes = {}
    
    for col in last_pretrain_row.keys():
        if "pretrain_accuracy" in col and "__MIN" not in col and "__MAX" not in col and "_step" not in col:
            # Extract group prefix (everything before " - test/")
            if " - test/" in col:
                group_prefix = col.split(" - test/")[0]
                group_prefixes[group_prefix] = col
    
    for group_prefix, pretrain_col in group_prefixes.items():
        # Find corresponding test accuracy column
        # Format: "{group_prefix} - test/accuracy"
        test_col = f"{group_prefix} - test/accuracy"
        
        if test_col in last_test_row.keys():
            try:
                pretrain_acc = float(last_pretrain_row[pretrain_col])
                finetune_acc = float(last_test_row[test_col])
                
                # Only add if both values are valid (not NaN)
                if not np.isnan(pretrain_acc) and not np.isnan(finetune_acc):
                    pretrain_accs.append(pretrain_acc)
                    finetune_accs.append(finetune_acc)
            except (ValueError, KeyError) as e:
                continue
    
    if len(pretrain_accs) == 0:
        raise ValueError(f"No valid data points found for finetune optimizer: {finetune_opt}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot points
    ax.scatter(finetune_accs, pretrain_accs, s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    
    # Set labels
    ax.set_xlabel('Finetuning Accuracy', fontsize=12)
    ax.set_ylabel('Pretraining Accuracy', fontsize=12)
    ax.set_title(f'Accuracy Frontier - {finetune_opt.capitalize()} Finetuning', fontsize=14)
    
    # Format axes as percentages
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Set axis ranges to 0-100%
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Set equal aspect ratio for better visualization
    ax.set_aspect('equal', adjustable='box')
    
    # Save the plot
    output_file = f"metrics/plots/frontier_{finetune_opt}.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"Plot saved to {output_file}")
    print(f"Plotted {len(pretrain_accs)} data points")

if __name__ == "__main__":
    create_frontier_plot("adam")

