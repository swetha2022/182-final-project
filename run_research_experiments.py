#!/usr/bin/env python3
"""
Run research experiments for continual learning
This script runs multiple experiments and generates research plots
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cifar_continual_learning import (
    get_model, get_data_loaders, evaluate, create_optimizer, compute_weight_l2_norm,
    FilteredDataset, train
)
try:
    from train_alexnet_muon import Muon
except ImportError:
    Muon = None
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import glob
import argparse
from typing import List

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def evaluate_initial_classes(model, device, loader, criterion, all_classes, initial_classes, desc="Evaluating"):
    """Evaluate model on initial classes, remapping labels to all_classes indices."""
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    label_map = {i: all_classes.index(initial_classes[i]) for i in range(len(initial_classes))}

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            remapped_labels = torch.tensor([label_map[int(l)] for l in labels], device=device)

            outputs = model(images)
            loss = criterion(outputs, remapped_labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == remapped_labels).sum().item()
            total += labels.size(0)
    return running_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0

def evaluate_new_classes(model, device, loader, criterion, all_classes, new_classes, desc="Evaluating"):
    """Evaluate model on new classes, remapping labels to all_classes indices."""
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    label_map = {i: all_classes.index(new_classes[i]) for i in range(len(new_classes))}

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            remapped_labels = torch.tensor([label_map[int(l)] for l in labels], device=device)

            outputs = model(images)
            loss = criterion(outputs, remapped_labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == remapped_labels).sum().item()
            total += labels.size(0)
    return running_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0

def continual_learning_experiment_with_replay(
    initial_classes: list,
    new_classes: list,
    epochs_task1: int = 5,
    epochs_task2: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    lr_task2: float = None,
    device=None,
    output_dir: str = "/scratch/current/celinet/182-cifar-checkpoints",
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "continual-learning-cifar10",
    wandb_name: str = None,
    wandb_entity: str = "182-research-project",
    freeze_features_task2: bool = False,
    use_lwf: bool = False,
    lwf_alpha: float = 1.0,
    lwf_temperature: float = 2.0,
    replay_buffer_size_per_class: int = 30,
    optimizer_name_task1: str = 'sgd',
    optimizer_name_task2: str = 'sgd',
    momentum_task1: float = 0.9,
    weight_decay_task1: float = 5e-4,
    optimizer_beta1_task1: float = 0.9,
    optimizer_beta2_task1: float = 0.95,
    optimizer_eps_task1: float = 1e-8,
    momentum_task2: float = 0.9,
    weight_decay_task2: float = 5e-4,
    optimizer_beta1_task2: float = 0.9,
    optimizer_beta2_task2: float = 0.95,
    optimizer_eps_task2: float = 1e-8,
):
    if device is None:
        device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

    print("="*60)
    print("CONTINUAL LEARNING EXPERIMENT")
    print("="*60)
    print(f"Initial task classes: {initial_classes}")
    print(f"New task classes: {new_classes}")
    print(f"Device: {device}")
    print("="*60)

    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    print("Loading data loaders...", flush=True)
    all_classes = sorted(initial_classes + new_classes)
    new_loader_train, new_loader_val = get_data_loaders(included_classes=new_classes, batch_size=batch_size, num_workers=4)
    initial_loader_train, initial_loader_val = get_data_loaders(included_classes=initial_classes, batch_size=batch_size, num_workers=4)
    print("Data loaders ready!", flush=True)
    criterion = nn.CrossEntropyLoss()

    print("\n" + "="*60)
    print("TASK 1: Training on initial classes", initial_classes)
    print("="*60)
    task1_output_dir = os.path.join(output_dir, "task1")
    os.makedirs(task1_output_dir, exist_ok=True)

    all_cifar_classes = list(range(10))
    holdout_for_task1 = [c for c in all_cifar_classes if c not in initial_classes]
    holdout_str = ','.join(map(str, holdout_for_task1)) if holdout_for_task1 else None

    lr_task1 = lr
    if optimizer_name_task1.lower() == 'adam':
        lr_task1 = 5e-5
    elif optimizer_name_task1.lower() == 'sgd':
        lr_task1 = 5e-3
    elif optimizer_name_task1.lower() == 'muon':
        lr_task1 = 5e-4
    
    pretrain_results = train(holdout_classes=holdout_str, epochs=epochs_task1, batch_size=batch_size, lr=lr_task1,
          momentum=momentum_task1, weight_decay=weight_decay_task1,
          seed=seed, no_cuda=(device.type == "cpu"), pretrained=False, output_dir=task1_output_dir,
          num_workers=4, use_wandb=False,
          optimizer_name=optimizer_name_task1,
          optimizer_beta1=optimizer_beta1_task1,
          optimizer_beta2=optimizer_beta2_task1,
          optimizer_eps=optimizer_eps_task1,
          lr_step=0,
          device=device)
    
    pretrain_weight_norms = pretrain_results.get('weight_norms', []) if pretrain_results else []

    best_checkpoints = glob.glob(os.path.join(task1_output_dir, f"*best.pth.tar"))
    best_checkpoint_path = best_checkpoints[0] if best_checkpoints else None

    model_task1_teacher = get_model(num_classes=len(initial_classes), pretrained=False, device=device)
    if best_checkpoint_path:
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model_task1_teacher.load_state_dict(checkpoint["model_state_dict"])
    model_task1_teacher.eval()

    model = get_model(num_classes=len(initial_classes), pretrained=False, device=device)
    if best_checkpoint_path:
        model.load_state_dict(checkpoint["model_state_dict"])

    initial_val_loss_before, initial_val_acc_before = evaluate(model_task1_teacher, device, initial_loader_val, criterion,
                                                               desc="Initial classes (before fine-tuning)")
    print(f"\nInitial classes before fine-tuning - Loss: {initial_val_loss_before:.4f}, Acc: {initial_val_acc_before:.4f}")

    num_initial_classes = len(initial_classes)
    num_new_classes = len(new_classes)
    num_all_classes = num_initial_classes + num_new_classes

    last_linear = model.classifier[6]
    in_features = last_linear.in_features
    old_weight = last_linear.weight.data.clone()
    old_bias = last_linear.bias.data.clone() if last_linear.bias is not None else None

    new_classifier = nn.Linear(in_features, num_all_classes).to(device)
    
    if freeze_features_task2:
        print(f"Expanded classifier {num_initial_classes} -> {num_all_classes} classes (linear probe)")
    else:
        for old_idx, c in enumerate(initial_classes):
            new_global_idx = all_classes.index(c)
            new_classifier.weight.data[new_global_idx] = old_weight[old_idx]
            if old_bias is not None:
                new_classifier.bias.data[new_global_idx] = old_bias[old_idx]
        print(f"Expanded classifier {num_initial_classes} -> {num_all_classes} classes (fine-tuning)")
    
    model.classifier[6] = new_classifier
    model = model.to(device)

    if freeze_features_task2:
        for p in model.features.parameters():
            p.requires_grad = False
        print("Feature extractor frozen for Task 2")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if lr_task2 is not None:
        lr_finetune = lr_task2
    else:
        if optimizer_name_task2.lower() == 'adam':
            lr_finetune = 1e-6
        elif optimizer_name_task2.lower() == 'sgd':
            lr_finetune = 1e-4
        elif optimizer_name_task2.lower() == 'muon':
            lr_finetune = 1e-4
        else:
            lr_finetune = lr
    
    optimizer = create_optimizer(
        optimizer_name=optimizer_name_task2,
        params=trainable_params,
        lr=lr_finetune,
        momentum=momentum_task2,
        weight_decay=weight_decay_task2,
        beta1=optimizer_beta1_task2,
        beta2=optimizer_beta2_task2,
        eps=optimizer_eps_task2
    )
    print(f"Using optimizer for Task 2: {optimizer_name_task2.upper()} with lr={lr_finetune}")
    
    scheduler = None

    replay_images, replay_labels = [], []
    for class_idx in initial_classes:
        all_class_indices = [i for i, y in enumerate(initial_loader_train.dataset.labels) if y == class_idx]
        selected_indices = np.random.choice(all_class_indices, size=min(replay_buffer_size_per_class, len(all_class_indices)), replace=False)
        for idx in selected_indices:
            img, label = initial_loader_train.dataset[idx]
            replay_images.append(img)
            replay_labels.append(label)
    replay_images = torch.stack(replay_images).to(device)
    replay_labels = torch.tensor(replay_labels, device=device)
    replay_loader = DataLoader(TensorDataset(replay_images, replay_labels), batch_size=batch_size, shuffle=True)
    replay_iterator = iter(replay_loader)

    if use_lwf:
        initial_distillation_loader_for_lwf, _ = get_data_loaders(included_classes=initial_classes, batch_size=batch_size, num_workers=4)
        distillation_data_iterator = iter(initial_distillation_loader_for_lwf)
        criterion_kl = nn.KLDivLoss(reduction='batchmean')

    initial_val_accs_during = []
    finetune_weight_norms = []
    finetune_val_accs = []

    for epoch in range(epochs_task2):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        new_data_iterator = iter(new_loader_train)
        num_batches = len(new_loader_train)
        pbar = tqdm(range(num_batches), desc=f"Task2 Epoch {epoch+1}/{epochs_task2}")

        for _ in pbar:
            images_new, labels_new_local = next(new_data_iterator)
            images_new, labels_new_local = images_new.to(device), labels_new_local.to(device)
            label_map_new_classes_to_global = {i: all_classes.index(new_classes[i]) for i in range(len(new_classes))}
            labels_new_global = torch.tensor([label_map_new_classes_to_global[int(l)] for l in labels_new_local], device=device)

            try:
                images_old, labels_old = next(replay_iterator)
            except StopIteration:
                replay_iterator = iter(replay_loader)
                images_old, labels_old = next(replay_iterator)
            images_old, labels_old = images_old.to(device), labels_old.to(device)

            images_combined = torch.cat([images_new, images_old], dim=0)
            labels_combined = torch.cat([labels_new_global, labels_old], dim=0)

            optimizer.zero_grad()
            outputs_combined = model(images_combined)
            total_loss = criterion(outputs_combined, labels_combined)

            if use_lwf:
                try:
                    images_teacher, _ = next(distillation_data_iterator)
                except StopIteration:
                    distillation_data_iterator = iter(initial_distillation_loader_for_lwf)
                    images_teacher, _ = next(distillation_data_iterator)
                images_teacher = images_teacher.to(device)
                with torch.no_grad():
                    teacher_logits = model_task1_teacher(images_teacher)
                student_logits_full = model(images_teacher)
                initial_class_indices = [all_classes.index(c) for c in initial_classes]
                student_logits_for_initial = student_logits_full[:, initial_class_indices]
                lwf_loss = criterion_kl(F.log_softmax(student_logits_for_initial / lwf_temperature, dim=1),
                                        F.softmax(teacher_logits / lwf_temperature, dim=1)) * (lwf_temperature**2)
                total_loss = total_loss + lwf_alpha * lwf_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * images_combined.size(0)
            preds = outputs_combined.argmax(dim=1)
            correct += (preds == labels_combined).sum().item()
            total += labels_combined.size(0)

            avg_loss = running_loss / total if total > 0 else 0.0
            acc = correct / total if total > 0 else 0.0
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.4f}'})

        model.eval()
        new_val_loss, new_val_acc = evaluate_new_classes(model, device, new_loader_val, criterion, all_classes, new_classes, desc="Task2 Val (New)")
        initial_val_loss, initial_val_acc = evaluate_initial_classes(model, device, initial_loader_val, criterion, all_classes, initial_classes, desc="Task2 Val (Initial)")

        initial_val_accs_during.append(initial_val_acc)
        finetune_weight_norms.append(compute_weight_l2_norm(model))
        finetune_val_accs.append(new_val_acc)
        
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}: Learning rate used = {current_lr:.2e}")
            scheduler.step()

    # FINAL EVALUATION
    initial_val_loss_after, initial_val_acc_after = evaluate_initial_classes(model, device, initial_loader_val, criterion, all_classes, initial_classes, desc="Initial classes (after)")
    new_val_loss_after, new_val_acc_after = evaluate_new_classes(model, device, new_loader_val, criterion, all_classes, new_classes, desc="New classes (after)")
    forgetting = initial_val_acc_before - initial_val_acc_after
    print(f"\nCatastrophic Forgetting: {forgetting:.4f} ({initial_val_acc_before:.4f} -> {initial_val_acc_after:.4f})")

    return {
        'initial_acc_before': initial_val_acc_before,
        'initial_acc_after': initial_val_acc_after,
        'new_acc_after': new_val_acc_after,
        'forgetting': forgetting,
        'initial_val_accs_during_finetune': initial_val_accs_during,
        'finetune_val_accs': finetune_val_accs,
        'weight_norms_pretrain': pretrain_weight_norms,
        'weight_norms_finetune': finetune_weight_norms,
        'model': model
    }

def run_multiple_experiments_with_error_bars(
    initial_classes: List[int],
    new_classes: List[int],
    num_runs: int = 5,
    base_seed: int = 42,
    epochs_task1: int = 5,
    epochs_task2: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    lr_task2: float = None,
    device=None,
    output_dir: str = "/scratch/current/celinet/182-cifar-checkpoints",
    optimizer_name_task1: str = 'sgd',
    optimizer_name_task2: str = 'sgd',
    momentum_task1: float = 0.9,
    weight_decay_task1: float = 5e-4,
    momentum_task2: float = 0.9,
    weight_decay_task2: float = 5e-4,
    freeze_features_task2: bool = False,
    use_lwf: bool = False,
    lwf_alpha: float = 1.0,
    lwf_temperature: float = 2.0,
    replay_buffer_size_per_class: int = 30,
):
    print("=" * 80)
    print(f"RUNNING {num_runs} EXPERIMENTS WITH DIFFERENT SEEDS")
    print("=" * 80)
    print(f"Initial classes: {initial_classes}")
    print(f"New classes: {new_classes}")
    print(f"Base seed: {base_seed}")
    print("=" * 80)

    all_results = {
        'initial_acc_before': [],
        'initial_acc_after': [],
        'new_acc_after': [],
        'forgetting': [],
        'relative_forgetting': [],
        'initial_val_accs_during_finetune': [],
        'finetune_val_accs': [],
        'weight_norms_pretrain': [],
        'weight_norms_finetune': [],
    }

    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        print(f"\n{'='*80}")
        print(f"RUN {run_idx + 1}/{num_runs} (seed={seed})")
        print(f"{'='*80}")

        try:
            result = continual_learning_experiment_with_replay(
                initial_classes=initial_classes,
                new_classes=new_classes,
                epochs_task1=epochs_task1,
                epochs_task2=epochs_task2,
                batch_size=batch_size,
                lr=lr,
                lr_task2=lr_task2,
                device=device,
                output_dir=output_dir,
                seed=seed,
                use_wandb=False,
                freeze_features_task2=freeze_features_task2,
                use_lwf=use_lwf,
                lwf_alpha=lwf_alpha,
                lwf_temperature=lwf_temperature,
                replay_buffer_size_per_class=replay_buffer_size_per_class,
                optimizer_name_task1=optimizer_name_task1,
                optimizer_name_task2=optimizer_name_task2,
                momentum_task1=momentum_task1,
                weight_decay_task1=weight_decay_task1,
                momentum_task2=momentum_task2,
                weight_decay_task2=weight_decay_task2,
            )

            all_results['initial_acc_before'].append(result['initial_acc_before'])
            all_results['initial_acc_after'].append(result['initial_acc_after'])
            all_results['new_acc_after'].append(result['new_acc_after'])
            forgetting = result['forgetting']
            all_results['forgetting'].append(forgetting)
            relative_forgetting = forgetting / max(result['initial_acc_before'], 1e-8)
            all_results['relative_forgetting'].append(relative_forgetting)
            all_results['initial_val_accs_during_finetune'].append(result['initial_val_accs_during_finetune'])
            all_results['finetune_val_accs'].append(result['finetune_val_accs'])
            all_results['weight_norms_pretrain'].append(result['weight_norms_pretrain'])
            all_results['weight_norms_finetune'].append(result['weight_norms_finetune'])

            print(f"\nRun {run_idx + 1} Results:")
            print(f"  Initial acc before: {result['initial_acc_before']:.4f}")
            print(f"  Initial acc after:  {result['initial_acc_after']:.4f}")
            print(f"  New acc after:      {result['new_acc_after']:.4f}")
            print(f"  Forgetting:         {forgetting:.4f}")
            print(f"  Relative forgetting:{relative_forgetting:.4f}")

        except Exception as e:
            print(f"ERROR in run {run_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print("AGGREGATED RESULTS ACROSS ALL RUNS")
    print(f"{'='*80}")

    aggregated = {}
    for key, values in all_results.items():
        if len(values) > 0 and key not in ['initial_val_accs_during_finetune', 'finetune_val_accs', 'weight_norms_pretrain', 'weight_norms_finetune']:
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
            }
            print(f"\n{key}:")
            print(f"  Mean: {aggregated[key]['mean']:.4f} ± {aggregated[key]['std']:.4f}")

    optimizer_combo = f"{optimizer_name_task1.upper()}->{optimizer_name_task2.upper()}"
    experiment_data = {
        'relative_forgetting': all_results['relative_forgetting'],
        'initial_acc_before': all_results['initial_acc_before'],
        'initial_acc_after': all_results['initial_acc_after'],
        'new_acc_after': all_results['new_acc_after'],
        'initial_val_accs_during_finetune': all_results['initial_val_accs_during_finetune'],
        'finetune_val_accs': all_results['finetune_val_accs'],
        'weight_norms_pretrain': all_results['weight_norms_pretrain'],
        'weight_norms_finetune': all_results['weight_norms_finetune']
    }
    
    return {
        'aggregated': aggregated,
        'individual_runs': all_results,
        'num_successful_runs': len(all_results['initial_acc_before']),
        'experiment_data': experiment_data,
        'optimizer_combo': optimizer_combo
    }

def create_research_plots(all_experiment_results, output_dir="./plots", use_wandb=True, wandb_project="continual-learning-cifar10", wandb_entity=None):
    """Create research-quality plots from collected experiment results."""
    os.makedirs(output_dir, exist_ok=True)
    
    if use_wandb and WANDB_AVAILABLE:
        if wandb.run is None:
            wandb.init(project=wandb_project, entity=wandb_entity, name="research_plots_summary")
    
    optimizer_combos = [combo for combo in all_experiment_results.keys() 
                       if combo in all_experiment_results and 
                       'relative_forgetting' in all_experiment_results[combo] and
                       len(all_experiment_results[combo]['relative_forgetting']) > 0]
    n_combos = len(optimizer_combos)
    
    relative_forgetting_means = []
    relative_forgetting_stds = []
    initial_acc_before_means = []
    initial_acc_after_means = []
    new_acc_after_means = []
    
    for combo in optimizer_combos:
        data = all_experiment_results[combo]
        relative_forgetting_means.append(np.mean(data['relative_forgetting']))
        relative_forgetting_stds.append(np.std(data['relative_forgetting']))
        initial_acc_before_means.append(np.mean(data['initial_acc_before']))
        initial_acc_after_means.append(np.mean(data['initial_acc_after']))
        new_acc_after_means.append(np.mean(data['new_acc_after']))
    
    pretrain_groups = {
        'SGD': [],
        'Adam': [],
        'Muon': []
    }
    
    for combo in optimizer_combos:
        pretrain_opt = combo.split('->')[0]
        pretrain_opt_lower = pretrain_opt.lower()
        if pretrain_opt_lower == 'sgd':
            pretrain_opt_normalized = 'SGD'
        elif pretrain_opt_lower == 'adam':
            pretrain_opt_normalized = 'Adam'
        elif pretrain_opt_lower == 'muon':
            pretrain_opt_normalized = 'Muon'
        else:
            pretrain_opt_normalized = pretrain_opt.capitalize()
        
        if pretrain_opt_normalized in pretrain_groups:
            pretrain_groups[pretrain_opt_normalized].append(combo)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_finetune_opts = set()
    for combo in optimizer_combos:
        if combo in all_experiment_results:
            data = all_experiment_results[combo]
            if 'relative_forgetting' in data and len(data['relative_forgetting']) > 0:
                finetune_opt = combo.split('->')[1]
                all_finetune_opts.add(finetune_opt)
    all_finetune_opts = sorted(list(all_finetune_opts))
    
    group_names = [g for g in pretrain_groups.keys() if len(pretrain_groups[g]) > 0]
    n_groups = len(group_names)
    n_bars_per_group = len(all_finetune_opts)
    
    if n_groups == 0 or n_bars_per_group == 0:
        print("Warning: No data available for relative forgetting plot")
        return
    
    width = 0.8 / n_bars_per_group if n_bars_per_group > 0 else 0.4
    
    finetune_colors = {
        'SGD': '#228B22', 'sgd': '#228B22',
        'Adam': '#ff7f0e', 'ADAM': '#ff7f0e', 'adam': '#ff7f0e',
        'Muon': '#1f77b4', 'muon': '#1f77b4', 'MUON': '#1f77b4'
    }
    
    x = np.arange(n_groups)
    group_width = 0.8
    
    for finetune_idx, finetune_opt in enumerate(all_finetune_opts):
        means = []
        stds = []
        for group_name in group_names:
            combo_variants = [
                f'{group_name.upper()}->{finetune_opt}',
                f'{group_name}->{finetune_opt}',
                f'{group_name.upper()}->{finetune_opt.upper()}',
            ]
            
            data = None
            for combo in combo_variants:
                if combo in all_experiment_results:
                    data = all_experiment_results[combo]
                    break
            
            if data is None:
                target_combo_lower = f'{group_name.upper()}->{finetune_opt}'.lower()
                for key in all_experiment_results.keys():
                    if key.lower() == target_combo_lower:
                        data = all_experiment_results[key]
                        break
                
            if data and 'relative_forgetting' in data and len(data['relative_forgetting']) > 0:
                means.append(np.mean(data['relative_forgetting']))
                stds.append(np.std(data['relative_forgetting']))
            else:
                means.append(0)
                stds.append(0)
        
        bar_width = group_width / n_bars_per_group
        offset = (finetune_idx - (n_bars_per_group - 1) / 2) * bar_width
        color = finetune_colors.get(finetune_opt, 
                   finetune_colors.get(finetune_opt.upper(), 
                   finetune_colors.get(finetune_opt.lower(), '#808080')))
        bars = ax.bar(x + offset, means, bar_width, yerr=stds, 
                     label=f'{finetune_opt} Finetune',
                     capsize=3, alpha=0.8, 
                     color=color,
                     edgecolor='black', linewidth=1.2)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'{g} Pretraining' for g in group_names], fontsize=11, fontweight='bold')
    
    ax.legend(loc='best', fontsize=10)
    
    ax.set_ylabel('Relative Forgetting', fontsize=12, fontweight='bold')
    ax.set_xlabel('Pretraining Optimizer', fontsize=12, fontweight='bold')
    ax.set_title('Relative Forgetting: Grouped by Pretraining Optimizer', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plot_path1 = os.path.join(output_dir, 'relative_forgetting_bar.png')
    plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
    plt.close()
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"research_plots/relative_forgetting_bar": wandb.Image(plot_path1)})
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for i, combo in enumerate(optimizer_combos):
        if combo not in all_experiment_results:
            continue
        data = all_experiment_results[combo]
        # Check if data exists and has values
        if 'initial_acc_after' not in data or len(data['initial_acc_after']) == 0:
            continue
        table_data.append([
            combo,
            f"{np.mean(data['initial_acc_after']):.4f} ± {np.std(data['initial_acc_after']):.4f}",
            f"{np.mean(data['new_acc_after']):.4f} ± {np.std(data['new_acc_after']):.4f}",
            f"{np.mean(data['relative_forgetting']):.4f} ± {np.std(data['relative_forgetting']):.4f}"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Optimizer\nCombo', 'Initial Acc\nAfter Finetune', 
                              'New Acc\nAfter Finetune', 'Relative\nForgetting'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(len(optimizer_combos) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    plot_path2 = os.path.join(output_dir, 'summary_table.png')
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    plt.close()
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"research_plots/summary_table": wandb.Image(plot_path2)})
    
    # Plot 3: Scatter plot of pretrain validation acc (y-axis) vs finetune validation acc (x-axis)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_combos))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, combo in enumerate(optimizer_combos):
        if combo not in all_experiment_results:
            continue
        data = all_experiment_results[combo]
        has_initial = 'initial_val_accs_during_finetune' in data and len(data['initial_val_accs_during_finetune']) > 0
        has_new = 'finetune_val_accs' in data and len(data['finetune_val_accs']) > 0
        
        if has_initial and has_new:
            # Average across runs for initial classes (pretrain validation set during finetuning)
            if isinstance(data['initial_val_accs_during_finetune'][0], list):
                initial_accs = np.mean(data['initial_val_accs_during_finetune'], axis=0)
            else:
                initial_accs = data['initial_val_accs_during_finetune']
            
            # Average across runs for new classes (finetune validation set)
            if isinstance(data['finetune_val_accs'][0], list):
                new_accs = np.mean(data['finetune_val_accs'], axis=0)
            else:
                new_accs = data['finetune_val_accs']
            
            # Plot scatter: each point is (new_acc, initial_acc) for each epoch
            min_len = min(len(initial_accs), len(new_accs))
            if min_len > 0:
                ax.scatter(new_accs[:min_len], initial_accs[:min_len], 
                          color=colors[i], marker=markers[i % len(markers)], 
                          s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                          label=combo)
                
                # Optionally add lines connecting points to show trajectory
                ax.plot(new_accs[:min_len], initial_accs[:min_len], 
                       color=colors[i], alpha=0.3, linewidth=1, linestyle='--')
    
    ax.set_xlabel('Finetune Validation Accuracy (New Classes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pretrain Validation Accuracy (Initial Classes)', fontsize=12, fontweight='bold')
    ax.set_title('Pretrain vs Finetune Validation Accuracy During Fine-tuning', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plot_path3 = os.path.join(output_dir, 'pretrain_finetune_acc.png')
    plt.savefig(plot_path3, dpi=300, bbox_inches='tight')
    plt.close()
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"research_plots/pretrain_finetune_acc": wandb.Image(plot_path3)})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    optimizer_colors = {
        'adam': '#ff7f0e',
        'ADAM': '#ff7f0e',
        'sgd': '#228B22',
        'SGD': '#228B22',
        'muon': '#1f77b4',
        'MUON': '#1f77b4'
    }
    
    max_pretrain_epochs_weight = 0
    all_weight_norms = []
    seen_optimizers = set()
    
    for i, combo in enumerate(optimizer_combos):
        if combo not in all_experiment_results:
            continue
        data = all_experiment_results[combo]
        if 'weight_norms_pretrain' in data and 'weight_norms_finetune' in data and len(data['weight_norms_pretrain']) > 0:
            if isinstance(data['weight_norms_pretrain'][0], list):
                weight_norms_pretrain = np.mean(data['weight_norms_pretrain'], axis=0)
            else:
                weight_norms_pretrain = data['weight_norms_pretrain']
            
            if isinstance(data['weight_norms_finetune'][0], list):
                weight_norms_finetune = np.mean(data['weight_norms_finetune'], axis=0)
            else:
                weight_norms_finetune = data['weight_norms_finetune']
            
            parts = combo.split('->')
            pretrain_opt = parts[0].strip().lower() if len(parts) > 0 else 'sgd'
            finetune_opt = parts[1].strip().lower() if len(parts) > 1 else 'sgd'
            
            pretrain_color = optimizer_colors.get(pretrain_opt, optimizer_colors.get(pretrain_opt.upper(), '#808080'))
            finetune_color = optimizer_colors.get(finetune_opt, optimizer_colors.get(finetune_opt.upper(), '#808080'))
            
            epochs_pretrain = range(1, len(weight_norms_pretrain) + 1)
            all_weight_norms.extend(weight_norms_pretrain)
            max_pretrain_epochs_weight = max(max_pretrain_epochs_weight, len(weight_norms_pretrain))
            pretrain_label = pretrain_opt.capitalize() if pretrain_opt not in seen_optimizers else ''
            if pretrain_opt not in seen_optimizers:
                seen_optimizers.add(pretrain_opt)
            ax.plot(epochs_pretrain, weight_norms_pretrain, '-', color=pretrain_color, linewidth=2, label=pretrain_label)
            
            epochs_finetune = range(len(weight_norms_pretrain) + 1, len(weight_norms_pretrain) + len(weight_norms_finetune) + 1)
            all_weight_norms.extend(weight_norms_finetune)
            if len(weight_norms_pretrain) > 0 and len(weight_norms_finetune) > 0:
                connect_epochs = [len(weight_norms_pretrain), len(weight_norms_pretrain) + 1]
                connect_values = [weight_norms_pretrain[-1], weight_norms_finetune[0]]
                ax.plot(connect_epochs, connect_values, '-', color=pretrain_color, linewidth=2, alpha=0.5)
            finetune_label = finetune_opt.capitalize() if finetune_opt not in seen_optimizers else ''
            if finetune_opt not in seen_optimizers:
                seen_optimizers.add(finetune_opt)
            ax.plot(epochs_finetune, weight_norms_finetune, '-', color=finetune_color, linewidth=2, label=finetune_label)
    
    if max_pretrain_epochs_weight > 0:
        ax.axvline(x=max_pretrain_epochs_weight + 1, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Finetuning Start')
    
    if len(all_weight_norms) > 0:
        y_min = min(all_weight_norms)
        y_max = max(all_weight_norms)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('L2 Norm of Model Weights', fontsize=12, fontweight='bold')
    ax.set_title('Weight Space Geometry: L2 Norm Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plot_path4 = os.path.join(output_dir, 'weight_geometry.png')
    plt.savefig(plot_path4, dpi=300, bbox_inches='tight')
    plt.close()
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"research_plots/weight_geometry": wandb.Image(plot_path4)})
    
    print(f"Research plots saved to {output_dir}")
    print(f"  - {plot_path1}")
    print(f"  - {plot_path2}")
    print(f"  - {plot_path3}")
    print(f"  - {plot_path4}")
    if use_wandb and WANDB_AVAILABLE:
        print("Research plots logged to wandb")

def plot_relative_forgetting_by_strategy(strategy_results, output_dir="./plots", use_wandb=True, wandb_project="continual-learning-cifar10", wandb_entity=None):
    """
    Create bar graphs comparing relative forgetting across optimizer mismatches and mitigation strategies.
    Creates two plots: one for SGD pretraining, one for Adam pretraining.
    
    Args:
        strategy_results: Dict mapping strategy names to experiment results
            Format: {
                'baseline': {
                    'SGD->SGD': {'relative_forgetting': [...], ...},
                    'SGD->Adam': {'relative_forgetting': [...], ...},
                    ...
                },
                'lwf': {
                    'SGD->SGD': {'relative_forgetting': [...], ...},
                    ...
                },
                'freeze': {
                    'SGD->SGD': {'relative_forgetting': [...], ...},
                    ...
                }
            }
        output_dir: Directory to save plots
        use_wandb: Whether to log plots to wandb
        wandb_project: Wandb project name
        wandb_entity: Wandb entity name
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if use_wandb and WANDB_AVAILABLE:
        if wandb.run is None:
            wandb.init(project=wandb_project, entity=wandb_entity, name="strategy_comparison")
    
    strategies = list(strategy_results.keys())
    
    optimizer_colors = {
        'adam': '#ff7f0e',
        'ADAM': '#ff7f0e',
        'sgd': '#228B22',
        'SGD': '#228B22',
        'muon': '#1f77b4',
        'MUON': '#1f77b4'
    }
    
    pretrain_groups = {
        'SGD': ['SGD->SGD', 'SGD->Adam'],
        'Adam': ['Adam->Adam', 'Adam->SGD']
    }
    
    for pretrain_opt, combos in pretrain_groups.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n_combos = len(combos)
        n_strategies = len(strategies)
        x = np.arange(n_combos)
        width = 0.8 / n_strategies
        
        for i, strategy in enumerate(strategies):
            means = []
            stds = []
            bar_colors = []
            for combo in combos:
                if combo in strategy_results[strategy]:
                    data = strategy_results[strategy][combo]['relative_forgetting']
                    means.append(np.mean(data))
                    stds.append(np.std(data))
                    finetune_opt = combo.split('->')[1].strip().lower() if '->' in combo else 'sgd'
                    bar_color = optimizer_colors.get(finetune_opt, optimizer_colors.get(finetune_opt.upper(), '#808080'))
                    bar_colors.append(bar_color)
                else:
                    means.append(0)
                    stds.append(0)
                    bar_colors.append('#808080')
            
            offset = (i - n_strategies/2 + 0.5) * width
            bars = ax.bar(x + offset, means, width, yerr=stds, label=strategy.capitalize(), 
                         capsize=3, alpha=0.8, color=bar_colors, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Finetuning Optimizer', fontsize=12, fontweight='bold')
        ax.set_ylabel('Relative Forgetting', fontsize=12, fontweight='bold')
        ax.set_title(f'Relative Forgetting: {pretrain_opt} Pretraining', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([combo.split('->')[1] for combo in combos])
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f'relative_forgetting_{pretrain_opt.lower()}_pretraining.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({f"research_plots/relative_forgetting_{pretrain_opt.lower()}_pretraining": wandb.Image(plot_path)})
        
        plt.close()
        print(f"Strategy comparison plot ({pretrain_opt} pretraining) saved to {plot_path}")
    
    if use_wandb and WANDB_AVAILABLE:
        print("Strategy comparison plots logged to wandb")

def main():
    parser = argparse.ArgumentParser(description='Run research experiments for continual learning')
    parser.add_argument('--initial-classes', type=str, default='0,1,2,3,4',
                        help='Comma-separated initial classes (e.g., "0,1,2,3,4")')
    parser.add_argument('--new-classes', type=str, default='5,6,7,8,9',
                        help='Comma-separated new classes (e.g., "5,6,7,8,9")')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of runs with different seeds')
    parser.add_argument('--epochs-task1', type=int, default=20,
                        help='Epochs for task 1 (pretraining)')
    parser.add_argument('--epochs-task2', type=int, default=10,
                        help='Epochs for task 2 (finetuning)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for task 1 (pretraining)')
    parser.add_argument('--lr-task2', type=float, default=None,
                        help='Learning rate for task 2 (fine-tuning). If None, uses optimizer-specific defaults (Adam=5e-5, SGD=5e-2, Muon=1e-3)')
    parser.add_argument('--optimizer-task1', type=str, default='sgd', choices=['sgd', 'adam', 'muon'],
                        help='Optimizer for task 1')
    parser.add_argument('--optimizer-task2', type=str, default='sgd', choices=['sgd', 'adam', 'muon'],
                        help='Optimizer for task 2')
    parser.add_argument('--device', type=str, default=None,
                        help='CUDA device (e.g., "cuda:6") or "cpu"')
    parser.add_argument('--output-dir', type=str, default='/scratch/current/celinet/182-cifar-checkpoints',
                        help='Output directory')
    parser.add_argument('--plots-dir', type=str, default='./plots',
                        help='Directory for plots')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='continual-learning-cifar10',
                        help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default='182-research-project',
                        help='Wandb entity name')
    parser.add_argument('--run-strategy-comparison', action='store_true',
                        help='Run experiments for all strategies (baseline, lwf, freeze) and create strategy comparison plots')
    parser.add_argument('--use-lwf', action='store_true',
                        help='Use Learning without Forgetting (LwF) strategy')
    parser.add_argument('--lwf-alpha', type=float, default=1.0,
                        help='LwF alpha parameter')
    parser.add_argument('--lwf-temperature', type=float, default=2.0,
                        help='LwF temperature parameter')
    parser.add_argument('--freeze-features', action='store_true',
                        help='Freeze feature extractor during fine-tuning (linear probe)')
    parser.add_argument('--replay-buffer-size', type=int, default=30,
                        help='Replay buffer size per class')
    
    args = parser.parse_args()
    
    initial_classes = [int(x.strip()) for x in args.initial_classes.split(',')]
    new_classes = [int(x.strip()) for x in args.new_classes.split(',')]
    
    device = None
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:6")
    else:
        device = torch.device("cpu")
    
    # If running strategy comparison, run all strategies and optimizer combinations
    if args.run_strategy_comparison:
        print("="*80)
        print("RUNNING STRATEGY COMPARISON EXPERIMENTS")
        print("="*80)
        
        strategy_results = {}
        # optimizer_combos = [
        #     ('sgd', 'sgd'),
        #     ('sgd', 'adam'),
        #     ('sgd', 'muon'),
        #     ('adam', 'adam'),
        #     ('adam', 'sgd'),
        #     ('adam', 'muon'),
        #     ('muon', 'muon'),
        #     ('muon', 'sgd'),
        #     ('muon', 'adam')
        # ]
        optimizer_combos = [
            ('sgd', 'sgd'),
            ('sgd', 'adam'),
            ('adam', 'adam'),
            ('adam', 'sgd'),
        ]
        strategies = [
            # ('baseline', {'use_lwf': False, 'freeze_features_task2': False}),
            ('lwf', {'use_lwf': True, 'freeze_features_task2': False, 'lwf_alpha': args.lwf_alpha, 'lwf_temperature': args.lwf_temperature}),
            ('freeze', {'use_lwf': False, 'freeze_features_task2': True})
        ]
        
        for strategy_name, strategy_params in strategies:
            strategy_results[strategy_name] = {}
            print(f"\n{'='*80}")
            print(f"Running experiments for strategy: {strategy_name.upper()}")
            print(f"{'='*80}")
            
            for opt1, opt2 in optimizer_combos:
                print(f"\n{'='*80}")
                print(f"Running {opt1.upper()} -> {opt2.upper()} with {strategy_name}")
                print(f"{'='*80}")
                
                result = run_multiple_experiments_with_error_bars(
                    initial_classes=initial_classes,
                    new_classes=new_classes,
                    num_runs=args.num_runs,
                    base_seed=42,
                    epochs_task1=args.epochs_task1,
                    epochs_task2=args.epochs_task2,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    lr_task2=args.lr_task2,
                    device=device,
                    output_dir=args.output_dir,
                    optimizer_name_task1=opt1,
                    optimizer_name_task2=opt2,
                    momentum_task1=0.9 if opt1 in ['sgd', 'muon'] else 0.0,
                    momentum_task2=0.9 if opt2 in ['sgd', 'muon'] else 0.0,
                    use_lwf=strategy_params.get('use_lwf', False),
                    lwf_alpha=strategy_params.get('lwf_alpha', 1.0),
                    lwf_temperature=strategy_params.get('lwf_temperature', 2.0),
                    freeze_features_task2=strategy_params.get('freeze_features_task2', False),
                    replay_buffer_size_per_class=args.replay_buffer_size,
                )
                
                optimizer_combo = result['optimizer_combo']
                strategy_results[strategy_name][optimizer_combo] = result['experiment_data']
        
        # Create strategy comparison plots
        print(f"\n{'='*80}")
        print("Creating strategy comparison plots...")
        print(f"{'='*80}")
        plot_relative_forgetting_by_strategy(
            strategy_results,
            output_dir=args.plots_dir,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity
        )
        
    else:
        print("="*80)
        print("RUNNING ALL OPTIMIZER COMBINATIONS FOR RESEARCH PLOTS")
        print("="*80)
        
        # optimizer_combos = [
        #     ('muon', 'muon'),
        #     ('muon', 'sgd'),
        #     ('muon', 'adam'),
        #     ('sgd', 'sgd'),
        #     ('sgd', 'adam'),
        #     ('sgd', 'muon'),
        #     ('adam', 'adam'),
        #     ('adam', 'sgd'),
        #     ('adam', 'muon')
        # ]
        optimizer_combos = [
            ('sgd', 'sgd'),
            ('sgd', 'adam'),
            ('adam', 'adam'),
            ('adam', 'sgd'),
        ]
        all_experiment_results = {}
        
        for opt1, opt2 in optimizer_combos:
            print(f"\n{'='*80}")
            print(f"Running experiments: {opt1.upper()} -> {opt2.upper()}")
            print(f"{'='*80}")
            
            result = run_multiple_experiments_with_error_bars(
                initial_classes=initial_classes,
                new_classes=new_classes,
                num_runs=args.num_runs,
                base_seed=42,
                epochs_task1=args.epochs_task1,
                epochs_task2=args.epochs_task2,
                batch_size=args.batch_size,
                lr=args.lr,
                lr_task2=args.lr_task2,
                device=device,
                output_dir=args.output_dir,
                optimizer_name_task1=opt1,
                optimizer_name_task2=opt2,
                momentum_task1=0.9 if opt1 in ['sgd', 'muon'] else 0.0,
                momentum_task2=0.9 if opt2 in ['sgd', 'muon'] else 0.0,
                use_lwf=args.use_lwf,
                lwf_alpha=args.lwf_alpha,
                lwf_temperature=args.lwf_temperature,
                freeze_features_task2=args.freeze_features,
                replay_buffer_size_per_class=args.replay_buffer_size,
            )
            
            # Store results for plotting
            all_experiment_results[result['optimizer_combo']] = result['experiment_data']
        
        # Create research plots with all optimizer combinations
        print(f"\n{'='*80}")
        print("Creating research plots with all optimizer combinations...")
        print(f"{'='*80}")
        create_research_plots(
            all_experiment_results,
            output_dir=args.plots_dir,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity
        )

if __name__ == "__main__":
    main()

