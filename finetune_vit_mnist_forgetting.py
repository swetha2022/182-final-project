#!/usr/bin/env python3
"""
finetune_vit_mnist_forgetting.py

Fine-tune torchvision ViT-B/16 (pretrained on ImageNet) on MNIST, evaluate ImageNet before (Apre_pre)
and after (Apre_post) fine-tuning to measure catastrophic forgetting. Repeat over multiple seeds,
compute mean ± sd, and plot results.

Usage (quick MNIST-only smoke test, skip ImageNet eval):
    python finetune_vit_mnist_forgetting.py --no-imagenet-eval --seeds 42 7 --epochs 2 --batch-size 128 --output-dir ./out_test

Full run (requires ImageNet val as ImageFolder):
    python finetune_vit_mnist_forgetting.py \
      --imagenet-root /path/to/imagenet_root --imagenet-val-dir ILSVRC2012_img_val \
      --seeds 42 1337 7 --epochs 10 --batch-size 128 --output-dir ./finetune_outputs \
      --optimizer adamw

Notes:
- This script ALWAYS uses torchvision pretrained ViT-B/16 as the starting backbone.
- No custom checkpoint loading logic (keeps initialization simple).
- Muon optimizer implemented here for convenience (may be slower).
"""
import argparse
import os
import random
import time
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

# ----------------------------
# Muon optimizer (simple reimplementation)
# ----------------------------
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    # work in float for stability
    X = G.clone().float()
    X = X / (X.norm() + eps)
    transposed = False
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)

class Muon(optim.Optimizer):
    def __init__(self, params, lr=0.005, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim > 2:
                    g_view = g.view(g.size(0), -1)
                else:
                    g_view = g
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g_view)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g_view)
                if nesterov:
                    g_use = g_view.add(buf, alpha=momentum)
                else:
                    g_use = buf
                update_matrix = zeropower_via_newtonschulz5(g_use, steps=ns_steps)
                if g.ndim > 2:
                    update_matrix = update_matrix.view_as(p)
                # scale a bit to avoid extremely small updates
                try:
                    scale = max(1.0, p.size(0) / max(1.0, p.size(1))) ** 0.5
                except Exception:
                    scale = 1.0
                update_matrix *= scale
                p.data.add_(update_matrix, alpha=-lr)

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate(model, loader, criterion, device, limit_batches=None):
    model.eval()
    total = 0
    correct = 0
    correct_top5 = 0
    running_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if limit_batches is not None and i >= limit_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            if outputs.size(1) >= 5:
                _, top5 = torch.topk(outputs, 5, dim=1)
                correct_top5 += (top5 == labels.view(-1, 1)).sum().item()
    if total == 0:
        return float('nan'), float('nan'), float('nan')
    loss = running_loss / (i + 1)
    acc = 100.0 * correct / total
    acc5 = 100.0 * correct_top5 / total if total > 0 else 0.0
    return loss, acc, acc5

# ----------------------------
# Single-seed run
# ----------------------------
def run_single_seed(seed, args, device, use_wandb=False):
    set_seed(seed)
    print(f"\n=== Running seed {seed} on device {device} ===")

    # transforms
    imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    imagenet_val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        imagenet_normalize
    ])
    mnist_train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        imagenet_normalize
    ])
    mnist_val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        imagenet_normalize
    ])

    # dataloaders
    mnist_train = datasets.MNIST(root=args.mnist_root, train=True, download=True, transform=mnist_train_transform)
    mnist_val = datasets.MNIST(root=args.mnist_root, train=False, download=True, transform=mnist_val_transform)
    mnist_train_loader = DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True)
    mnist_val_loader = DataLoader(mnist_val, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

    imagenet_val_loader = None
    if not args.no_imagenet_eval:
        if args.imagenet_root is None:
            raise ValueError("--imagenet-root is required when not using --no-imagenet-eval")
        imagenet_val = datasets.ImageFolder(os.path.join(args.imagenet_root, args.imagenet_val_dir),
                                            transform=imagenet_val_transform)
        imagenet_val_loader = DataLoader(imagenet_val, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True)

    # --------- Model: ALWAYS use torchvision pretrained ViT-B/16 ----------
    print("Loading torchvision ViT-B/16 pretrained weights (ImageNet).")
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model = model.to(device)

    # Keep a copy of the original ImageNet head to use for pre/post evaluations
    head_imagenet = deepcopy(model.heads.head).to(device)

    # Replace head with MNIST head
    feat_dim = model.heads.head.in_features
    head_mnist = nn.Linear(feat_dim, args.mnist_num_classes).to(device)
    model.heads.head = head_mnist
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Evaluate ImageNet baseline BEFORE fine-tuning (Apre_pre)
    pre_loss, pre_acc, pre_acc5 = (float('nan'), float('nan'), float('nan'))
    if not args.no_imagenet_eval:
        model.heads.head = head_imagenet
        model = model.to(device)
        pre_loss, pre_acc, pre_acc5 = validate(model, imagenet_val_loader, criterion, device,
                                                limit_batches=args.imagenet_limit_batches)
        print(f"[seed {seed}] ImageNet baseline acc: {pre_acc:.2f}% (loss {pre_loss:.4f})")
        # restore MNIST head for fine-tuning
        model.heads.head = head_mnist
        model = model.to(device)

    # --------- Optimizer selection ----------
    opt_name = args.optimizer.lower()
    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif opt_name == "muon":
        optimizer = Muon(model.parameters(), lr=args.muon_lr, momentum=args.muon_momentum, nesterov=True,
                         ns_steps=args.muon_ns_steps)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    scheduler = None
    if args.scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --------- Fine-tune on MNIST ----------
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        pbar = tqdm(mnist_train_loader, desc=f"Seed {seed} Train E{epoch+1}/{args.epochs}", leave=False)
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_total += labels.size(0)
            running_correct += (preds == labels).sum().item()
            pbar.set_postfix(loss=running_loss / (len(pbar)), acc=100.0 * running_correct / running_total)
            global_step += 1

        train_loss = running_loss / (len(mnist_train_loader))
        train_acc = 100.0 * running_correct / running_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # validate on MNIST
        val_loss, val_acc, _ = validate(model, mnist_val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if scheduler is not None:
            scheduler.step()

        print(f"[seed {seed}] Epoch {epoch+1}/{args.epochs} - train_acc {train_acc:.2f}% - val_acc {val_acc:.2f}%")

    # After fine-tuning: evaluate ImageNet head AGAIN (Apre_post)
    post_loss, post_acc, post_acc5 = (float('nan'), float('nan'), float('nan'))
    if not args.no_imagenet_eval:
        model.heads.head = head_imagenet
        model = model.to(device)
        post_loss, post_acc, post_acc5 = validate(model, imagenet_val_loader, criterion, device,
                                                   limit_batches=args.imagenet_limit_batches)
        print(f"[seed {seed}] ImageNet post-finetune acc: {post_acc:.2f}% (loss {post_loss:.4f})")

    # Save per-seed curves
    os.makedirs(args.output_dir, exist_ok=True)
    seed_prefix = os.path.join(args.output_dir, f"seed_{seed}")
    np.save(seed_prefix + "_train_losses.npy", np.array(train_losses))
    np.save(seed_prefix + "_train_accs.npy", np.array(train_accs))
    np.save(seed_prefix + "_val_losses.npy", np.array(val_losses))
    np.save(seed_prefix + "_val_accs.npy", np.array(val_accs))

    # per-seed plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.arange(1, len(train_losses)+1), train_losses, marker='o', label='train_loss')
    axes[0].plot(np.arange(1, len(val_losses)+1), val_losses, marker='o', label='val_loss')
    axes[0].set_title(f"Seed {seed} Losses")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(np.arange(1, len(train_accs)+1), train_accs, marker='o', label='train_acc')
    axes[1].plot(np.arange(1, len(val_accs)+1), val_accs, marker='o', label='val_acc')
    axes[1].set_title(f"Seed {seed} Accuracies")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(seed_prefix + "_curves.png")
    plt.close(fig)

    return {
        'seed': seed,
        'imagenet_pre': {'loss': pre_loss, 'acc': pre_acc, 'acc5': pre_acc5},
        'imagenet_post': {'loss': post_loss, 'acc': post_acc, 'acc5': post_acc5},
        'mnist_curves': {'train_losses': train_losses, 'train_accs': train_accs, 'val_losses': val_losses, 'val_accs': val_accs}
    }

# ----------------------------
# Aggregate across seeds and plotting
# ----------------------------
def aggregate_and_plot(results, args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ImageNet pre/post
    pre_accs = []
    post_accs = []
    for r in results:
        pre = r['imagenet_pre']['acc']
        post = r['imagenet_post']['acc']
        if not np.isnan(pre):
            pre_accs.append(pre)
        if not np.isnan(post):
            post_accs.append(post)

    if len(pre_accs) == 0 or len(post_accs) == 0:
        print("Missing ImageNet evaluations across seeds; cannot compute catastrophic forgetting metric.")
    else:
        pre_arr = np.array(pre_accs)
        post_arr = np.array(post_accs)
        errors_pre = 100.0 - pre_arr
        errors_post = 100.0 - post_arr
        diffs = errors_pre - errors_post  # Apre_pre - Apre_post (percentage points)
        mean_diff = np.mean(diffs)
        sd_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else 0.0
        print(f"Apre_pre - Apre_post across {len(diffs)} seeds: mean={mean_diff:.4f} pp, sd={sd_diff:.4f} pp")

        # bar plot
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(0, mean_diff, yerr=sd_diff, align='center', capsize=10)
        ax.set_xticks([0])
        ax.set_xticklabels([f"Apre_pre - Apre_post\n(n={len(diffs)})"])
        ax.set_ylabel("Difference in error (percentage points)")
        ax.set_title("Catastrophic Forgetting: Apre_pre - Apre_post (mean ± sd)")
        ax.axhline(0, color='black', linewidth=0.8)
        fig.tight_layout()
        out_path = os.path.join(args.output_dir, "Apre_pre_minus_Apre_post_mean_sd.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved catastrophic forgetting plot to {out_path}")

    # Aggregate MNIST curves across seeds (mean ± sd per epoch)
    max_epochs = 0
    for r in results:
        max_epochs = max(max_epochs, len(r['mnist_curves']['train_accs']))
    if max_epochs == 0:
        print("No MNIST curves found to aggregate.")
        return

    train_acc_mat = np.full((len(results), max_epochs), np.nan)
    val_acc_mat = np.full((len(results), max_epochs), np.nan)
    train_loss_mat = np.full((len(results), max_epochs), np.nan)
    val_loss_mat = np.full((len(results), max_epochs), np.nan)

    for i, r in enumerate(results):
        ta = r['mnist_curves']['train_accs']
        va = r['mnist_curves']['val_accs']
        tl = r['mnist_curves']['train_losses']
        vl = r['mnist_curves']['val_losses']
        train_acc_mat[i, :len(ta)] = ta
        val_acc_mat[i, :len(va)] = va
        train_loss_mat[i, :len(tl)] = tl
        val_loss_mat[i, :len(vl)] = vl

    epochs = np.arange(1, max_epochs+1)
    train_acc_mean = np.nanmean(train_acc_mat, axis=0)
    train_acc_sd = np.nanstd(train_acc_mat, axis=0, ddof=1)
    val_acc_mean = np.nanmean(val_acc_mat, axis=0)
    val_acc_sd = np.nanstd(val_acc_mat, axis=0, ddof=1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(epochs, train_acc_mean, label='train_acc_mean', marker='o')
    ax.fill_between(epochs, train_acc_mean - train_acc_sd, train_acc_mean + train_acc_sd, alpha=0.25)
    ax.plot(epochs, val_acc_mean, label='val_acc_mean', marker='o')
    ax.fill_between(epochs, val_acc_mean - val_acc_sd, val_acc_mean + val_acc_sd, alpha=0.25)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("MNIST Accuracy (mean ± sd across seeds)")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(args.output_dir, "mnist_accuracy_mean_sd.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved aggregated MNIST accuracy plot to {out_path}")

    # Save raw results
    save_path = os.path.join(args.output_dir, "all_results.npy")
    np.save(save_path, results)
    print(f"Saved raw results to {save_path}")

# ----------------------------
# Argument parser
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ViT on MNIST, evaluate ImageNet pre/post, aggregate over seeds")
    parser.add_argument("--imagenet-root", type=str, default="/home/swetha/vision_transformer/data/ImageNet_1K", help="Root folder for ImageNet data (if doing ImageNet eval).")
    parser.add_argument("--imagenet-val-dir", type=str, default="ILSVRC2012_img_val", help="ImageNet validation dir name inside imagenet-root.")
    parser.add_argument("--no-imagenet-eval", action="store_true", help="If set, skip ImageNet pre/post evaluation.")
    parser.add_argument("--imagenet-limit-batches", type=int, default=None, help="If set, limit ImageNet eval to N batches (for quick runs).")
    parser.add_argument("--mnist-root", type=str, default="./data", help="MNIST root dir")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 7], help="Random seeds to run (space-separated)")
    parser.add_argument("--epochs", type=int, default=10, help="Fine-tune epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--mnist-num-classes", type=int, default=10, help="MNIST classes")
    parser.add_argument("--imagenet-num-classes", type=int, default=1000, help="ImageNet classes")
    parser.add_argument("--output-dir", type=str, default="./finetune_outputs", help="Where to save curves and plots")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    parser.add_argument("--scheduler", action="store_true", help="Use CosineAnnealingLR scheduler")
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer: adam | sgd | adamw | muon")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--muon-lr", type=float, default=0.005, help="Muon lr")
    parser.add_argument("--muon-momentum", type=float, default=0.95, help="Muon momentum")
    parser.add_argument("--muon-ns-steps", type=int, default=5, help="Muon Newton-Schulz steps")
    parser.add_argument("--use-wandb", action="store_true", help="Log metrics to wandb if available")
    return parser.parse_args()

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("Using device:", device)

    use_wandb = False
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project="vit-finetune-mnist")
            use_wandb = True
        except Exception as e:
            print("Warning: wandb requested but not available:", e)
            use_wandb = False

    results = []
    start_time = time.time()
    for seed in args.seeds:
        res = run_single_seed(seed, args, device, use_wandb=use_wandb)
        results.append(res)
        if use_wandb:
            import wandb
            wandb.log({
                f"seed_{seed}/imagenet_pre_acc": res['imagenet_pre']['acc'],
                f"seed_{seed}/imagenet_post_acc": res['imagenet_post']['acc']
            })

    elapsed = time.time() - start_time
    print(f"\nAll seeds completed in {elapsed/60.0:.2f} minutes.")

    aggregate_and_plot(results, args)

if __name__ == "__main__":
    main()