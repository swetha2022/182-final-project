#!/usr/bin/env python3
"""
CIFAR-10 Continual Learning Training Script
"""

import os
import sys
import random
import glob
import argparse
from datetime import datetime
from typing import Sequence, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import alexnet, AlexNet_Weights
from tqdm.auto import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

def print_cuda_info():
    print("\n" + "="*50)
    print("CUDA DEVICE INFORMATION")
    print("="*50)
    try:
        if torch.cuda.is_available():
            print(f"✓ CUDA is available!")
            print(f"  Device Name: {torch.cuda.get_device_name(0)}")
            print(f"  Device Count: {torch.cuda.device_count()}")
            print(f"  Current Device Index: {torch.cuda.current_device()}")
            try:
                print(f"  CUDA Version: {torch.version.cuda}")
            except:
                print(f"  CUDA Version: Unable to determine")
        else:
            print("✗ CUDA is not available. Using CPU.")
    except Exception as e:
        print(f"Error checking CUDA: {e}")
    print("="*50 + "\n")

class FilteredDataset(Dataset):
    def __init__(self, base_dataset: Dataset, included_classes: Sequence[int]):
        self.base = base_dataset
        self.included_classes = list(included_classes)
        self.class_to_new = {c: i for i, c in enumerate(self.included_classes)}
        if hasattr(base_dataset, 'targets'):
            self.indices = [i for i, label in enumerate(base_dataset.targets) if label in self.class_to_new]
        else:
            self.indices = [i for i, (_, y) in enumerate(self.base) if int(y) in self.class_to_new]
        self.labels = [self.class_to_new[int(self.base.targets[i])] if hasattr(self.base, 'targets') else self.class_to_new[int(self.base[i][1])] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, y = self.base[real_idx]
        new_y = self.class_to_new[int(y)]
        return x, new_y

def compute_weight_l2_norm(model):
    """Compute L2 norm of all model parameters."""
    total_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            param_norm = param.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def parse_holdout_classes(holdout_arg: str) -> Tuple[List[int], List[int]]:
    """Parse holdout classes and return (included_classes, holdout_classes)."""
    all_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    if holdout_arg is None or holdout_arg.strip() == "":
        return all_classes, []

    parts = [p.strip() for p in holdout_arg.replace(",", " ").split()]
    holdout_classes = sorted({int(p) for p in parts})

    if not all(0 <= c <= 9 for c in holdout_classes):
        raise ValueError("CIFAR-10 classes must be integers in [0,9].")

    included_classes = sorted([c for c in all_classes if c not in holdout_classes])

    if len(included_classes) == 0:
        raise ValueError("Cannot hold out all classes. At least one class must be included for training.")

    return included_classes, holdout_classes

def get_model(num_classes=10, pretrained=False, device=None):
    if pretrained:
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    else:
        model = alexnet(weights=None)
    
    last_linear = model.classifier[6]
    in_features = last_linear.in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    
    if device is not None:
        model = model.to(device)
    
    return model

def create_optimizer(optimizer_name: str, params, lr: float, momentum: float = 0.9, weight_decay: float = 5e-4,
                     beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, nesterov: bool = True, ns_steps: int = 5):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return optim.Adam(params, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    elif optimizer_name == 'muon':
        from train_alexnet_muon import MuonWithAuxAdam
        
        muon_params = [p for p in params if p.ndim >= 2]
        adam_params = [p for p in params if p.ndim < 2]
        
        param_groups = []
        if len(muon_params) > 0:
            param_groups.append({
                'params': muon_params,
                'use_muon': True,
                'lr': lr,
                'momentum': momentum,
                'weight_decay': weight_decay,
            })
        if len(adam_params) > 0:
            param_groups.append({
                'params': adam_params,
                'use_muon': False,
                'lr': lr,
                'betas': (beta1, beta2),
                'eps': eps,
                'weight_decay': weight_decay,
            })
        
        if len(param_groups) == 0:
            raise ValueError("No parameters to optimize")
        else:
            return MuonWithAuxAdam(param_groups)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Supported: 'sgd', 'adam', 'muon'")
        
def get_data_loaders(included_classes: Sequence[int], batch_size=128, num_workers=0):
    cifar10_mean = [0.4914, 0.4822, 0.4465]
    cifar10_std = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])

    train_base = datasets.CIFAR10(root="./data", train=True, transform=train_transform, download=True)
    val_base = datasets.CIFAR10(root="./data", train=False, transform=val_transform, download=True)

    train_ds = FilteredDataset(train_base, included_classes)
    val_ds = FilteredDataset(val_base, included_classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

def save_checkpoint(state, path):
    torch.save(state, path)

def evaluate(model, device, loader, criterion, desc="Evaluating"):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0

def train(
    holdout_classes=None,
    epochs=8,
    batch_size=128,
    lr=1e-3,
    momentum=0.9,
    weight_decay=5e-4,
    seed=42,
    no_cuda=False,
    freeze_features=False,
    pretrained=False,
    resume=None,
    output_dir="/scratch/current/celinet/182-cifar-checkpoints",
    num_workers=0,
    lr_step=0,
    lr_gamma=0.1,
    use_wandb=True,
    wandb_project="continual-learning-cifar10",
    wandb_name=None,
    wandb_entity="182-research-project",
    optimizer_name='sgd',
    optimizer_beta1=0.9,
    optimizer_beta2=0.999,
    optimizer_eps=1e-8,
    device=None
):
    """Train AlexNet on CIFAR-10 with optional class holdout."""
    if device is None:
        device = torch.device("cuda:6" if torch.cuda.is_available() and not no_cuda else "cpu")
    print("Using device:", device)

    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    included_classes, holdout_classes_list = parse_holdout_classes(holdout_classes)
    print("Training on classes:", included_classes)
    if holdout_classes_list:
        print("Holding out classes (for later fine-tuning):", holdout_classes_list)
    num_classes = len(included_classes)
    print("Number of classes (K):", num_classes)

    train_loader, val_loader = get_data_loaders(included_classes=included_classes, batch_size=batch_size, num_workers=num_workers)

    if pretrained:
        print("Using ImageNet pretrained weights")
    else:
        print("Training from scratch (no pretrained weights)")

    model = get_model(num_classes=num_classes, pretrained=pretrained, device=device)

    if freeze_features:
        for p in model.features.parameters():
            p.requires_grad = False
        print("Feature extractor frozen. Only classifier will be trained.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = create_optimizer(
        optimizer_name=optimizer_name,
        params=trainable_params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        beta1=optimizer_beta1,
        beta2=optimizer_beta2,
        eps=optimizer_eps
    )
    print(f"Using optimizer: {optimizer_name.upper()}")

    scheduler = None
    if lr_step > 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    criterion = nn.CrossEntropyLoss()

    train_initialized_wandb = False

    if use_wandb and WANDB_AVAILABLE:
        wandb_already_init = wandb.run is not None

        if not wandb_already_init:
            if wandb_name is None:
                classes_str = '_'.join(map(str, included_classes))
                holdout_str = '_'.join(map(str, holdout_classes_list)) if holdout_classes_list else "all"
                wandb_name = f"classes_{classes_str}_holdout_{holdout_str}"

            init_kwargs = {
                "project": wandb_project,
                "name": wandb_name,
                "config": {
                    "holdout_classes": holdout_classes_list if holdout_classes_list else "all",
                    "included_classes": included_classes,
                    "num_classes": num_classes,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                    "seed": seed,
                    "freeze_features": freeze_features,
                    "pretrained": pretrained,
                    "lr_step": lr_step,
                    "lr_gamma": lr_gamma,
                    "device": str(device),
                    "optimizer_name": optimizer_name,
                }
            }
            if wandb_entity is not None:
                init_kwargs["entity"] = wandb_entity

            wandb.init(**init_kwargs)
            train_initialized_wandb = True

    start_epoch = 0
    if resume is not None and os.path.isfile(resume):
        checkpoint = torch.load(resume, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("Loaded optimizer state from resume checkpoint.")
            except Exception as e:
                print("Warning: failed to load optimizer state:", e)
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resumed from {resume} (epoch {start_epoch})")

    best_val_acc = 0.0
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    weight_norms = []

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            avg_loss = running_loss / total if total > 0 else 0.0
            acc = correct / total if total > 0 else 0.0
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.4f}'})

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0

        val_loss, val_acc = evaluate(model, device, val_loader, criterion, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        weight_norms.append(compute_weight_l2_norm(model))

        if use_wandb and WANDB_AVAILABLE and train_initialized_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            if scheduler is not None:
                log_dict["learning_rate"] = scheduler.get_last_lr()[0]
            wandb.log(log_dict)

        if scheduler is not None:
            scheduler.step()

        classes_str = '_'.join(map(str, included_classes))
        holdout_str = '_'.join(map(str, holdout_classes_list)) if holdout_classes_list else "all"
        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_acc": train_acc,
            "val_acc": val_acc,
            "included_classes": included_classes,
            "holdout_classes": holdout_classes_list,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
        }
        ckpt_name = os.path.join(output_dir, f"alexnet_cifar10_classes_{classes_str}_holdout_{holdout_str}_{now}_epoch{epoch+1}.pth.tar")
        save_checkpoint(ckpt, ckpt_name)
        print("Saved checkpoint:", ckpt_name)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_name = os.path.join(output_dir, f"alexnet_cifar10_classes_{classes_str}_holdout_{holdout_str}_best.pth.tar")
            save_checkpoint(ckpt, best_name)
            print("New best model saved to", best_name)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs_range, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, 'b-', label='Train Acc', linewidth=2)
    plt.plot(epochs_range, val_accs, 'r-', label='Val Acc', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"training_curves_{classes_str}_holdout_{holdout_str}_{now}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {plot_path}")

    if use_wandb and WANDB_AVAILABLE and train_initialized_wandb:
        wandb.log({
            "best_val_acc": best_val_acc,
            "training_curves": wandb.Image(plot_path)
        })
        wandb.finish()

    print("Training complete. Best val acc:", best_val_acc)
    
    return {
        'val_accs': val_accs,
        'weight_norms': weight_norms,
        'best_val_acc': best_val_acc
    }

def main():
    parser = argparse.ArgumentParser(description='Train AlexNet on CIFAR-10 with continual learning')
    parser.add_argument('--holdout-classes', type=str, default=None,
                        help='Comma-separated list of classes to hold out (e.g., "5,6,7,8,9")')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='CUDA device (e.g., "cuda:6") or "cpu"')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'muon'],
                        help='Optimizer to use')
    parser.add_argument('--output-dir', type=str, default='/scratch/current/celinet/182-cifar-checkpoints',
                        help='Output directory for checkpoints and plots')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb-project', type=str, default='continual-learning-cifar10',
                        help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default='182-research-project',
                        help='Wandb entity name')
    
    args = parser.parse_args()
    
    print_cuda_info()
    
    device = None
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:6")
    else:
        device = torch.device("cpu")
    
    train(
        holdout_classes=args.holdout_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
        optimizer_name=args.optimizer,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity
    )

if __name__ == "__main__":
    main()

