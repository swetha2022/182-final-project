import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

# --------------------------
# LOAD METRICS
# --------------------------

optimizers = ["adamw", "sgd", "muon"]
metrics = ["pre", "post"]

output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)

results = {}

# Load arrays: each file contains accuracy values across seeds
for opt in optimizers:
    opt_dir = f"finetune_outputs_{opt}"
    results[opt] = {}
    for metric in metrics:
        file_path = os.path.join(opt_dir, f"imagenet_{metric}_accs.npy")
        arr = np.load(file_path)
        results[opt][metric] = arr[:4]      # first 4 seeds only


# --------------------------
# COMPUTE STATISTICS
# --------------------------

# mean pre / post
pre_mean =  {opt: np.mean(results[opt]["pre"])  for opt in optimizers}
post_mean = {opt: np.mean(results[opt]["post"]) for opt in optimizers}

# min / max (to get error bar range)
pre_min  = {opt: np.min(results[opt]["pre"])  for opt in optimizers}
pre_max  = {opt: np.max(results[opt]["pre"])  for opt in optimizers}
post_min = {opt: np.min(results[opt]["post"]) for opt in optimizers}
post_max = {opt: np.max(results[opt]["post"]) for opt in optimizers}

# forgetting = A_pre(pretrain) - A_pre(post-finetune)
forgetting = {
    opt: pre_mean[opt] - post_mean[opt]
    for opt in optimizers
}

# Build label-indexed dicts the plotting code expects
mean_accs = {f"{opt}_post": post_mean[opt] for opt in optimizers}
min_accs  = {f"{opt}_post": post_min[opt]  for opt in optimizers}
max_accs  = {f"{opt}_post": post_max[opt]  for opt in optimizers}


print(pre_min)
print(pre_max)
print(post_min)
print(post_max)
print(forgetting)
print(mean_accs)
print(min_accs)
print(max_accs)
# --------------------------
# PLOTTING STYLE
# --------------------------

optimizer_colors = {
    "adamw": "#ff7f0e",
    "sgd":   "#55a868",
    "muon":  "#1f77b4",
}

comparisons = [
    ("adamw", "sgd"),
    ("adamw", "muon")
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.32, right=0.80)

i = 0
for ax, (opt1, opt2) in zip(axes, comparisons):

    # find the 2 relevant keys e.g. ['adamw_post', 'sgd_post']
    selected = [k for k in mean_accs.keys() if opt1 in k or opt2 in k]

    labels = selected
    mean_forgets = []
    std_devs = []

    for k in labels:
        opt = k.replace("_post", "")    # "adamw_post" → "adamw"

        # forgetting = mean_pre - mean_post
        mean_fg = pre_mean[opt] - post_mean[opt]

        # error bars:
        # range of forgetting = (pre_min - post_max) to (pre_max - post_min)
        lower = (pre_min[opt] - post_max[opt])
        upper = (pre_max[opt] - post_min[opt])
        std = (upper - lower) / 4

        mean_forgets.append(mean_fg)
        std_devs.append(std)

    # colors based on optimizer names
    bar_colors = [
        optimizer_colors[k.replace("_post", "")]
        for k in labels
    ]

    # plot bars
    ax.bar(labels, mean_forgets, yerr=std_devs, capsize=6, color=bar_colors)

    # style
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0)
    ax.set_xticklabels([])
    
    if i == 0:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f%%'))
        ax.set_ylabel(r"$F_{\text{pre,post}} = A_{\text{pre,pre}} - A_{\text{pre,post}}$", fontsize=12)
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d%%'))

    i += 1

# global title
fig.suptitle("Catastrophic Forgetting — AdamW Pretraining", fontsize=15)

# legend
legend_elems = [
    Patch(color=optimizer_colors["adamw"], label="AdamW Finetuning"),
    Patch(color=optimizer_colors["sgd"],   label="SGD Finetuning"),
    Patch(color=optimizer_colors["muon"],  label="Muon Finetuning"),
]
fig.legend(handles=legend_elems, loc='center right', bbox_to_anchor=(0.98, 0.5))

plt.savefig("plots/forgetting.png", bbox_inches="tight")
plt.close()
