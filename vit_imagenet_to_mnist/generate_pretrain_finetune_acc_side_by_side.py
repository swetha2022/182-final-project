import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# --------------------------
# Config
# --------------------------
optimizer_colors = {
    "adamw": "#ff7f0e",
    "sgd":   "#55a868",
    "muon":  "#1f77b4",
}
folders = {
    "adamw": "finetune_outputs_adamw",
    "sgd": "finetune_outputs_sgd",
    "muon": "finetune_outputs_muon",
}

output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Load data
# --------------------------
optimizers_all = ["adamw", "sgd", "muon"]
data = {}
for opt in optimizers_all:
    folder = folders[opt]
    imagenet_file = os.path.join(folder, "seed_42_imagenet_val_accs_interval.npy")
    mnist_file = os.path.join(folder, "seed_42_mnist_val_accs_interval.npy")
    
    imagenet_accs = np.load(imagenet_file)
    mnist_accs = np.load(mnist_file)

    imagenet_accs = [acc[1] for acc in imagenet_accs] 
    imagenet_accs = [np.float64(81.068)] + imagenet_accs
    mnist_accs = [acc[1] for acc in mnist_accs] 
    mnist_accs = [np.float64(13.070)] + mnist_accs  
    
    data[opt] = {
        "imagenet": imagenet_accs,
        "mnist": mnist_accs
    }

# --------------------------
# Plot
# --------------------------
fig, axes = plt.subplots(1, 2, figsize=(14,6))  # auto-scaling for each subplot

# Define the optimizer combinations for each subplot
subplot_combos = [
    ("AdamW vs SGD", ["adamw", "sgd"]),
    ("AdamW vs Muon", ["adamw", "muon"])
]

i = 0 
for ax, (title, optimizers) in zip(axes, subplot_combos):
    for opt in optimizers:
        x = data[opt]["mnist"]
        y = data[opt]["imagenet"]
        
        if opt == "adamw":
            legend_label = "AdamW Finetuning"
        elif opt == "sgd":
            legend_label = "SGD Finetuning"
        else:
            legend_label = "Muon Finetuning"
        
        # Plot line connecting points with faded color
        rgba_color = to_rgba(optimizer_colors[opt], alpha=0.5)
        ax.plot(x, y, color=rgba_color, linestyle='-', marker='o', label=legend_label)
        
        # Plot points in full color
        ax.scatter(x, y, color=optimizer_colors[opt])
    
    ax.set_xlabel("MNIST Finetune accuracy")
    if i == 0: 
        ax.set_ylabel("ImageNet Pretrain accuracy")
    ax.grid(True)
    ax.legend(loc="lower left")  # always bottom-left

    i += 1

# Add a single shared title
fig.suptitle("ImageNet vs MNIST Accuracy — AdamW Pretraining", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle

# Save figure
plt.savefig(os.path.join(output_dir, "pretrain_vs_finetune_side_by_side_sharedtitle.png"), dpi=300)
plt.show()
