import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

def read_csv_to_dict(filename):
    data = []
    with open(filename, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def clean_column_name(c):
    finetune_idx = c.find("finetune")
    group_idx = c.find("group")
    c = c[finetune_idx:group_idx].rstrip("- ").strip()
    return c

def get_pretrain_ratio(c):
    pretrain_idx = c.find("pretrain")
    c = c[pretrain_idx:]
    return c

def create_forgetting_graph(pretrain_opt):
    files_to_read = ["metrics/raw_data/pretrain_acc.csv"]
    if pretrain_opt == "adam":
        files_to_read.append("metrics/raw_data/pretrain_adam_finetune_adam_pretrain_acc.csv")
        files_to_read.append("metrics/raw_data/pretrain_adam_finetune_muon_pretrain_acc.csv")
    if pretrain_opt == "muon":
        files_to_read.append("metrics/raw_data/pretrain_muon_finetune_muon_pretrain_acc.csv")
        files_to_read.append("metrics/raw_data/pretrain_muon_finetune_adam_pretrain_acc.csv")

    pretrain_mean_acc, pretrain_min_acc, pretrain_max_acc = 0, 0, 0
    mean_accs, min_accs, max_accs = {}, {}, {}
    for file_to_read in files_to_read:        
        data = read_csv_to_dict(file_to_read)
        last_data = data[-1]
        opt_columns = {c: last_data[c] for c in last_data.keys() if pretrain_opt in c and "accuracy" in c}
        for c, v in opt_columns.items():
            if "finetune" in c:
                if "MIN" in c:
                    min_accs[clean_column_name(c)] =float(v)
                elif "MAX" in c:
                    max_accs[clean_column_name(c)] = float(v)
                else:
                    mean_accs[clean_column_name(c)] = float(v)
            else:
                if "MIN" in c:
                    pretrain_min_acc = float(v)
                elif "MAX" in c:
                    pretrain_max_acc = float(v)
                else:
                    pretrain_mean_acc = float(v)

    keys = list(mean_accs.keys())
    if pretrain_opt == "muon":
        preratio_05_keys = [keys[0], keys[3]]
        preratio_01_keys = [keys[1], keys[4]]
        # preratio_00_keys = [keys[2], keys[5]]
        graphs = [preratio_05_keys, preratio_01_keys] # , preratio_00_keys]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    elif pretrain_opt == "adam":
        preratio_05_keys = [keys[2], keys[0]]
        preratio_01_keys = [keys[3], keys[1],]
        graphs = [preratio_05_keys, preratio_01_keys]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))


    
    # Adjust right margin based on pretrain_opt to prevent legend overlap
    if pretrain_opt == "adam":
        right_margin = 0.78  # More space for legend with 2 subplots
    else:
        right_margin = 0.78  # Less space needed with 3 subplots
    fig.subplots_adjust(wspace=0.4, top=0.92, right=right_margin)
    bar_colors = ['#1f77b4', '#ff7f0e']
    
    for idx, (keys, ax) in enumerate(zip(graphs, axes)):
        labels = []
        mean_forgets = []
        std_devs = []
        for k in keys:
            labels.append(k)

            mean_forget = pretrain_mean_acc - mean_accs[k]
            mean_forgets.append(mean_forget)

            min_forget = pretrain_min_acc - max_accs[k]
            max_forget = pretrain_max_acc - min_accs[k]
            std_devs.append(((max_forget - min_forget)) / 4)

        ax.set_axisbelow(True)  # Put grid lines behind the bars
        ax.bar(labels, mean_forgets, yerr=std_devs, capsize=6, color=bar_colors[:len(labels)])

        pretrain_ratio = get_pretrain_ratio(keys[0])
        # Position text based on pretrain_opt: left for muon, right for adam
        if pretrain_opt == "adam":
            x_pos, halign = 0.96, 'right'
        else:
            x_pos, halign = 0.04, 'left'
        ax.text(x_pos, 0.97, f"Pretrain Ratio: {pretrain_ratio}%", 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', horizontalalignment=halign,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        # Remove x-axis labels
        ax.set_xticklabels([])
        # Format y-axis as percentages
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    axes[0].set_ylabel(r"$F_{\text{pre,post}} = A_{\text{pre,pre}} - A_{\text{pre,post}}$", fontsize=12)
    fig.suptitle(f"Catastrophic Forgetting - {pretrain_opt.capitalize()} Pretraining", fontsize=14, y=1.0)
    
    # Create legend handles
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Muon Finetuning'),
        Patch(facecolor='#ff7f0e', label='Adam Finetuning')
    ]
    # Adjust legend position based on pretrain_opt
    if pretrain_opt == "adam":
        legend_anchor = (0.97, 0.5)  # Further left to avoid overlap
    else:
        legend_anchor = (0.97, 0.5)  # Original position for muon
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=legend_anchor)
    
    plt.savefig(f"metrics/plots/forgetting_{pretrain_opt}.png", bbox_inches='tight')

if __name__ == "__main__":
    create_forgetting_graph(pretrain_opt="muon")