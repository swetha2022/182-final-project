import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# --------------------------
# Config
# --------------------------
optimizers = ["adamw", "sgd"]
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
data = {}
for opt in optimizers:
    folder = folders[opt]
    imagenet_file = os.path.join(folder, "seed_42_imagenet_val_accs_interval.npy")
    mnist_file = os.path.join(folder, "seed_42_mnist_val_accs_interval.npy")
    
    imagenet_accs = np.load(imagenet_file)
    mnist_accs = np.load(mnist_file)

    imagenet_accs = [acc[1] for acc in imagenet_accs] 
    imagenet_accs = [np.float64(81.068)] + imagenet_accs
    mnist_accs = [acc[1] for acc in mnist_accs] 
    if opt == "sgd":
        mnist_accs = [np.float64(13.070)] + mnist_accs
    elif opt == "adamw":
        mnist_accs = [np.float64(13.070)] + mnist_accs
    else:
        mnist_accs = [np.float64(13.070)] + mnist_accs  
    
    data[opt] = {
        "imagenet": imagenet_accs,
        "mnist": mnist_accs
    }

# --------------------------
# Plot
# --------------------------
plt.figure(figsize=(7,6))

for opt in optimizers:

    x = data[opt]["mnist"]
    y = data[opt]["imagenet"]
    
    legend_label = "AdamW Finetuning" if opt == "adamw" else "SGD Finetuning"
    
    # Plot the line connecting points with a faded version
    rgba_color = to_rgba(optimizer_colors[opt], alpha=0.3)
    plt.plot(x, y, color=rgba_color, linestyle='-', marker='o', label=legend_label)
    
    # Plot the points in full color without extra legend entries
    plt.scatter(x, y, color=optimizer_colors[opt])

plt.xlabel("MNIST Finetune accuracy")
plt.ylabel("ImageNet Pretrain accuracy")
plt.title("ImageNet vs MNIST Accuracy — AdamW Pretraining")
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()

# Save figure
plt.savefig(os.path.join(output_dir, "pretrain_vs_finetune_accuracy_adam_sgd.png"), dpi=300)

plt.show()
