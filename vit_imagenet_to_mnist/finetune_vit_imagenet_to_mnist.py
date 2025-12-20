#!/usr/bin/env python3
"""
finetune_vit_mnist_forgetting.py

Fine-tune torchvision ViT-B/16 (pretrained on ImageNet) on MNIST, evaluate ImageNet before (Apre_pre)
and after (Apre_post) fine-tuning to measure catastrophic forgetting. Repeat over multiple seeds,
compute mean ± sd, and plot results (step-based).

Usage (quick MNIST-only smoke test, skip ImageNet eval):
    python finetune_vit_mnist_forgetting.py --no-imagenet-eval --seeds 42 7 --steps 90 --batch-size 128 --output-dir ./out_test
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
    """
    Compute matrix 'zero-power' approximation via Newton-Schulz iterations.
    Accepts 2-D matrices and 1-D vectors. For 1-D inputs, we treat them as
    (n,1) matrices, run the algorithm, then squeeze back to 1-D.
    """
    # remember original shape so we can return same-shaped tensor
    orig_shape = G.shape
    was_vector = False
    if G.ndim == 1:
        # treat vector as column matrix (n,1)
        G2 = G.unsqueeze(1)
        was_vector = True
    else:
        G2 = G

    assert G2.ndim == 2, "zeropower_via_newtonschulz5 expects 1D or 2D input"

    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G2.clone().float()
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    X = X.to(G.dtype)

    if was_vector:
        X = X.squeeze(1)
        X = X.view(orig_shape)

    return X


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

    # Model: pretrained ViT-B/16
    print("Loading torchvision ViT-B/16 pretrained weights (ImageNet).")
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model = model.to(device)
    head_imagenet = deepcopy(model.heads.head).to(device)
    feat_dim = model.heads.head.in_features
    head_mnist = nn.Linear(feat_dim, args.mnist_num_classes).to(device)
    model.heads.head = head_mnist
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # ImageNet baseline BEFORE fine-tuning
    pre_loss, pre_acc, pre_acc5 = (float('nan'), float('nan'), float('nan'))
    if not args.no_imagenet_eval:
        model.heads.head = head_imagenet
        model = model.to(device)
        pre_loss, pre_acc, pre_acc5 = validate(model, imagenet_val_loader, criterion, device,
                                                limit_batches=args.imagenet_limit_batches)
        print(f"[seed {seed}] ImageNet baseline acc: {pre_acc:.2f}% (loss {pre_loss:.4f})")
        model.heads.head = head_mnist
        model = model.to(device)

    # Optimizer
    opt_name = args.optimizer.lower()
    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif opt_name == "muon":
        optimizer = Muon(model.parameters(), lr=args.muon_lr, momentum=args.momentum, nesterov=True,
                         ns_steps=args.muon_ns_steps)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    scheduler = None
    if args.scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # --------- Fine-tune on MNIST (step-based) ----------
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    steps_list = []
    global_step = 0
    eval_interval = 90 
    mnist_val_accs_per_interval = []
    mnist_val_losses_per_interval = []
    imagenet_val_accs_per_interval = []
    imagenet_val_losses_per_interval = []

    while global_step < args.steps:
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        pbar = tqdm(mnist_train_loader, desc=f"Seed {seed} Training", leave=False)
        for imgs, labels in pbar:
            if global_step >= args.steps or global_step == 0:
                val_loss, val_acc, _ = validate(model, mnist_val_loader, criterion, device)
                mnist_val_accs_per_interval.append((global_step, val_acc))
                
                if not args.no_imagenet_eval:
                    model.heads.head = head_imagenet
                    model = model.to(device)
                    _, imagenet_acc, _ = validate(model, imagenet_val_loader, criterion, device,
                                                limit_batches=args.imagenet_limit_batches)
                    imagenet_val_accs_per_interval.append((global_step, imagenet_acc))
                    model.heads.head = head_mnist
                    model = model.to(device)
                print(f"[seed {seed}] Step {global_step}/{args.steps} - val_acc {val_acc:.3f}% - MNIST val, ImageNet val {imagenet_acc:.3f}%")
                break
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
            global_step += 1
            steps_list.append(global_step)

            # --- record per-step loss & accuracy ---
            train_losses.append(loss.item())
            train_accs.append(100.0 * running_correct / running_total)

            pbar.set_postfix(loss=train_losses[-1],
                            acc=train_accs[-1])
            
            if global_step % eval_interval == 0:
                val_loss, val_acc, _ = validate(model, mnist_val_loader, criterion, device)
                mnist_val_accs_per_interval.append((global_step, val_acc))
                
                if not args.no_imagenet_eval:
                    model.heads.head = head_imagenet
                    model = model.to(device)
                    _, imagenet_acc, _ = validate(model, imagenet_val_loader, criterion, device,
                                                limit_batches=args.imagenet_limit_batches)
                    imagenet_val_accs_per_interval.append((global_step, imagenet_acc))
                    model.heads.head = head_mnist
                    model = model.to(device)
                print(f"[seed {seed}] Step {global_step}/{args.steps} - val_acc {val_acc:.2f}% - MNIST val, ImageNet val {imagenet_acc:.2f}%")

        # Validate **per epoch** (optional, keeps val curves lower-resolution)
        val_loss, val_acc, _ = validate(model, mnist_val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if scheduler is not None:
            scheduler.step()

        print(f"[seed {seed}] Step {global_step}/{args.steps} - last_train_acc {train_accs[-1]:.2f}% - val_acc {val_acc:.2f}%")

    # ImageNet post-finetune
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
    np.save(seed_prefix + "_steps.npy", np.array(steps_list))

    np.save(seed_prefix + "_mnist_val_accs_interval.npy", np.array(mnist_val_accs_per_interval))
    np.save(seed_prefix + "_mnist_val_losses_interval.npy", np.array(mnist_val_losses_per_interval))
    if not args.no_imagenet_eval:
        np.save(seed_prefix + "_imagenet_val_accs_interval.npy", np.array(imagenet_val_accs_per_interval))
        np.save(seed_prefix + "_imagenet_val_losses_interval.npy", np.array(imagenet_val_losses_per_interval))


    # Plot step-aligned curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps_list[:len(train_losses)], train_losses, marker='o', label='train_loss')
    axes[0].plot(steps_list[:len(val_losses)], val_losses, marker='o', label='val_loss')
    axes[0].set_title(f"Seed {seed} Losses")
    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(steps_list[:len(train_accs)], train_accs, marker='o', label='train_acc')
    axes[1].plot(steps_list[:len(val_accs)], val_accs, marker='o', label='val_acc')
    axes[1].set_title(f"Seed {seed} Accuracies")
    axes[1].set_xlabel("Training Steps")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(seed_prefix + "_curves_steps.png")
    plt.close(fig)

    return {
        'seed': seed,
        'imagenet_pre': {'loss': pre_loss, 'acc': pre_acc, 'acc5': pre_acc5},
        'imagenet_post': {'loss': post_loss, 'acc': post_acc, 'acc5': post_acc5},
        'mnist_curves': {'train_losses': train_losses, 'train_accs': train_accs,
                         'val_losses': val_losses, 'val_accs': val_accs,
                         'steps': steps_list}
    }

# ----------------------------
# Aggregate plotting
# ----------------------------
def aggregate_and_plot(results, args):
    plt.figure(figsize=(8,5))
    max_steps = max([len(r['mnist_curves']['steps']) for r in results])
    steps = np.arange(1, max_steps+1)
    all_train_accs = []
    all_imagenet_pre_accs = []
    all_imagenet_post_accs = []
    for r in results:
        step_curve = r['mnist_curves']['steps']
        train_curve = np.interp(steps, step_curve, r['mnist_curves']['train_accs'])
        all_train_accs.append(train_curve)
        all_imagenet_pre_accs.append(r['imagenet_pre']['acc'])
        all_imagenet_post_accs.append(r['imagenet_post']['acc'])

    prefix = os.path.join(args.output_dir, f"imagenet")
    np.save(prefix + "_pre_accs.npy", np.array(all_imagenet_pre_accs))
    np.save(prefix + "_post_accs.npy", np.array(all_imagenet_post_accs))

    train_mean = np.mean(all_train_accs, axis=0)
    train_std = np.std(all_train_accs, axis=0)
    plt.plot(steps, train_mean, label='train_acc')
    plt.fill_between(steps, train_mean-train_std, train_mean+train_std, alpha=0.2)
    plt.xlabel("Training Steps")
    plt.ylabel(f"Training Accuracy with {args.optimizer} (%)")
    plt.title("MNIST Accuracy Across Seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "aggregate_acc_steps.png"))
    plt.close()

# ----------------------------
# Argument parser
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ViT on MNIST, evaluate ImageNet pre/post, aggregate over seeds")
    parser.add_argument("--imagenet-root", type=str, default="/raid/users/celinet/data")
    parser.add_argument("--imagenet-val-dir", type=str, default="ILSVRC2012_img_val")
    parser.add_argument("--no-imagenet-eval", action="store_true")
    parser.add_argument("--imagenet-limit-batches", type=int, default=None)
    parser.add_argument("--mnist-root", type=str, default="/raid/users/celinet/data")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mnist-num-classes", type=int, default=10)
    parser.add_argument("--imagenet-num-classes", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="./finetune_outputs_muon_test")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--optimizer", type=str, default="muon")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--muon-lr", type=float, default=0.003)
    parser.add_argument("--muon-ns-steps", type=int, default=3)
    parser.add_argument("--use-wandb", action="store_true")
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

    results = []
    start_time = time.time()
    for seed in args.seeds:
        res = run_single_seed(seed, args, device, use_wandb=use_wandb)
        results.append(res)
        if use_wandb:
            wandb.log({
                f"seed_{seed}/imagenet_pre_acc": res['imagenet_pre']['acc'],
                f"seed_{seed}/imagenet_post_acc": res['imagenet_post']['acc'],
            })

    aggregate_and_plot(results, args)
    elapsed = time.time() - start_time
    print(f"\nAll seeds done in {elapsed/60:.1f} minutes. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
