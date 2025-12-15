import os
import re
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def find_matching_runs(pretrain_folder, model_type, lr, weight_decay):
    """
    Find all run folders matching the specified model type, learning rate, and weight decay.
    
    Args:
        pretrain_folder: Path to pretrain_omniglot folder
        model_type: Model type (e.g., "alexnet")
        lr: Learning rate (float)
        weight_decay: Weight decay (float)
    
    Returns:
        List of folder paths matching the criteria
    """
    matching_runs = []
    pretrain_path = Path(pretrain_folder)
    
    if not pretrain_path.exists():
        raise ValueError(f"Pretrain folder does not exist: {pretrain_folder}")
    
    # Extract learning rate and weight decay from folder names using regex
    # Pattern: lr0.00005 or lr0.0005, wd0.0001 or wd0.00001
    lr_pattern = re.compile(r'lr([\d.]+)')
    wd_pattern = re.compile(r'wd([\d.]+)')
    
    for folder in pretrain_path.iterdir():
        if not folder.is_dir():
            continue
        
        folder_name = folder.name
        
        # Check if folder matches model type
        if not folder_name.startswith(f"{model_type}-"):
            continue
        
        # Extract learning rate and weight decay from folder name
        lr_match = lr_pattern.search(folder_name)
        wd_match = wd_pattern.search(folder_name)
        
        if lr_match and wd_match:
            folder_lr = float(lr_match.group(1))
            folder_wd = float(wd_match.group(1))
            
            # Match with tolerance for floating point comparison
            if abs(folder_lr - lr) < 1e-8 and abs(folder_wd - weight_decay) < 1e-8:
                matching_runs.append(folder)
    
    return matching_runs


def find_matching_finetune_runs(finetune_folder, model_type, pretrain_optimizer, finetune_optimizer, 
                                 finetune_lr, pretrain_ratio):
    """
    Find all finetuning run folders matching the specified criteria.
    
    Args:
        finetune_folder: Path to finetune_omniglot folder
        model_type: Model type (e.g., "alexnet")
        pretrain_optimizer: Pretraining optimizer (e.g., "adam" or "muon")
        finetune_optimizer: Finetuning optimizer (e.g., "adam" or "muon")
        finetune_lr: Finetuning learning rate (float)
        pretrain_ratio: Pretrain ratio (float)
    
    Returns:
        List of folder paths matching the criteria
    """
    matching_runs = []
    finetune_path = Path(finetune_folder)
    
    if not finetune_path.exists():
        raise ValueError(f"Finetune folder does not exist: {finetune_folder}")
    
    # Pattern: alexnet-pretrain-adam-finetune-adam-lr0.00005-preratio0.05-seed30
    pattern = re.compile(
        rf'{model_type}-pretrain-{pretrain_optimizer}-finetune-{finetune_optimizer}-lr([\d.]+)-preratio([\d.]+)-seed(\d+)'
    )
    
    for folder in finetune_path.iterdir():
        if not folder.is_dir():
            continue
        
        folder_name = folder.name
        match = pattern.match(folder_name)
        
        if match:
            folder_lr = float(match.group(1))
            folder_ratio = float(match.group(2))
            
            # Match with tolerance for floating point comparison
            if abs(folder_lr - finetune_lr) < 1e-8 and abs(folder_ratio - pretrain_ratio) < 1e-8:
                matching_runs.append(folder)
    
    return matching_runs


def load_checkpoint(checkpoint_path, device):
    """
    Load a model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Model state dict
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    return state_dict


def calculate_weight_changes(prev_state, curr_state, learning_rate):
    """
    Calculate the change in weights between two checkpoints, normalized by learning rate.
    
    Args:
        prev_state: Previous checkpoint state dict
        curr_state: Current checkpoint state dict
        learning_rate: Learning rate to normalize the weight changes by
    
    Returns:
        Dictionary of weight changes (delta_theta / learning_rate) for each parameter
    """
    delta_theta = {}
    for key in prev_state.keys():
        # if key == "classifier.6.weight":
        # if key in curr_state:
        #     print(key)
        if key != "classifier.6.weight" and key != "classifier.6.bias":
            # Calculate change and normalize by learning rate
            delta_theta[key] = (curr_state[key] - prev_state[key]) / learning_rate
    return delta_theta


def compute_delta_norm(delta_theta, norm_type, device):
    """
    Compute the norm of weight changes using compute_parameter_norm logic.
    
    Args:
        delta_theta: Dictionary of weight changes
        norm_type: Type of norm ('inf' or 'rms')
        device: Device for computation
    
    Returns:
        Scalar norm value
    """
    norm_type_lower = norm_type.lower()
    parameter_norms = []
    
    for key, param in delta_theta.items():
        if param.numel() == 0:
            continue
        
        # Determine if parameter is a matrix (2D or higher) or vector (1D)
        is_matrix = param.dim() >= 2
        
        if norm_type_lower == 'rms':
            if is_matrix:
                if param.dim() > 2:
                    param_2d = param.view(-1, param.shape[-1])
                    param_norm = torch.linalg.matrix_norm(param_2d, ord=2)
                    param_norm = param_norm * np.sqrt(param.shape[-2] / param.shape[-1])
                else:
                    param_norm = torch.linalg.matrix_norm(param, ord=2)
                    param_norm = param_norm * np.sqrt(param.shape[0] / param.shape[1])
            else:
                param_norm = torch.norm(param, p=2)
                param_norm = param_norm * np.sqrt(1.0 / param.shape[0])
        elif norm_type_lower in ['inf', 'infinity']:
            param_norm = torch.norm(param, p=float('inf'))
        else:
            param_norm = torch.norm(param, p=2)
        
        # Ensure param_norm is a scalar
        if param_norm.dim() > 0 or param_norm.numel() > 1:
            param_norm = param_norm.flatten()[0]
        param_norm_value = param_norm.item() if isinstance(param_norm, torch.Tensor) else float(param_norm)
        param_norm = torch.tensor(param_norm_value, device=device)
        
        parameter_norms.append(param_norm)
    
    if len(parameter_norms) == 0:
        return torch.tensor(0.0)
    
    # Average all parameter norms
    return torch.stack(parameter_norms).mean()


def process_run(run_folder, device, learning_rate, initial_checkpoint=None, initial_epoch=None):
    """
    Process a single run and calculate weight change norms between consecutive checkpoints.
    
    Args:
        run_folder: Path to run folder containing checkpoints
        device: Device for computation
        learning_rate: Learning rate to normalize weight changes by
        initial_checkpoint: Optional path to initial checkpoint (for finetuning, this would be the last pretrain checkpoint)
        initial_epoch: Optional initial epoch (for finetuning, this would be the last pretrain epoch)
    
    Returns:
        Dictionary with epochs and corresponding norm values for inf and rms
    """
    # Find all checkpoint files
    checkpoint_files = sorted(run_folder.glob("checkpoint_*.pt"), 
                              key=lambda x: int(re.search(r'checkpoint_(\d+)\.pt', x.name).group(1)))
    
    if len(checkpoint_files) == 0:
        print(f"Warning: Run {run_folder.name} has no checkpoints, skipping")
        return None
    
    epochs = []
    inf_norms = []
    rms_norms = []
    
    # Load first checkpoint (either from initial_checkpoint or first file in folder)
    if initial_checkpoint is not None:
        # For finetuning, skip the first checkpoint because dimension changes
        # Start from the first checkpoint in the folder (which will be compared to the second)
        if len(checkpoint_files) < 2:
            print(f"Warning: Finetuning run {run_folder.name} has less than 2 checkpoints, skipping")
            return None
        # Load first checkpoint as previous state (skip comparing initial_checkpoint to first)
        prev_state = load_checkpoint(checkpoint_files[0], device)
        prev_epoch = initial_epoch if initial_epoch is not None else 0
        # Start processing from the second checkpoint
        checkpoint_files_to_process = checkpoint_files[1:]
    else:
        if len(checkpoint_files) < 2:
            print(f"Warning: Run {run_folder.name} has less than 2 checkpoints, skipping")
            return None
        prev_state = load_checkpoint(checkpoint_files[0], device)
        prev_epoch = int(re.search(r'checkpoint_(\d+)\.pt', checkpoint_files[0].name).group(1))
        checkpoint_files_to_process = checkpoint_files[1:]
        
        # For pretraining, add the first checkpoint epoch with zero change (no previous checkpoint to compare)
        epochs.append(prev_epoch)
        inf_norms.append(0.0)
        rms_norms.append(0.0)
    
    # Process consecutive checkpoints
    for checkpoint_file in tqdm(checkpoint_files_to_process, desc=f"Processing {run_folder.name}"):
        checkpoint_epoch = int(re.search(r'checkpoint_(\d+)\.pt', checkpoint_file.name).group(1))
        curr_state = load_checkpoint(checkpoint_file, device)
        
        # Calculate weight changes (normalized by learning rate)
        delta_theta = calculate_weight_changes(prev_state, curr_state, learning_rate)
        
        # Compute norms
        inf_norm = compute_delta_norm(delta_theta, 'inf', device)
        rms_norm = compute_delta_norm(delta_theta, 'rms', device)
        
        # If initial_checkpoint is provided (finetuning), offset epochs by initial_epoch
        if initial_checkpoint is not None and initial_epoch is not None:
            curr_epoch = checkpoint_epoch + initial_epoch
        else:
            curr_epoch = checkpoint_epoch
        
        epochs.append(curr_epoch)
        inf_norms.append(inf_norm.item())
        rms_norms.append(rms_norm.item())
        
        prev_state = curr_state
        prev_epoch = curr_epoch
    
    return {
        'epochs': epochs,
        'inf_norms': inf_norms,
        'rms_norms': rms_norms
    }


def aggregate_runs(run_results):
    """
    Aggregate results across all runs to compute mean, min, and max.
    
    Args:
        run_results: List of dictionaries from process_run
    
    Returns:
        Dictionary with aggregated statistics
    """
    # Find all unique epochs
    all_epochs = set()
    for result in run_results:
        if result is not None:
            all_epochs.update(result['epochs'])
    
    all_epochs = sorted(all_epochs)
    
    # Aggregate norms at each epoch
    inf_norms_by_epoch = defaultdict(list)
    rms_norms_by_epoch = defaultdict(list)
    
    for result in run_results:
        if result is None:
            continue
        for epoch, inf_norm, rms_norm in zip(result['epochs'], result['inf_norms'], result['rms_norms']):
            inf_norms_by_epoch[epoch].append(inf_norm)
            rms_norms_by_epoch[epoch].append(rms_norm)
    
    # Compute statistics
    epochs = []
    inf_means = []
    inf_mins = []
    inf_maxs = []
    rms_means = []
    rms_mins = []
    rms_maxs = []
    
    for epoch in all_epochs:
        epochs.append(epoch)
        inf_means.append(np.mean(inf_norms_by_epoch[epoch]))
        inf_mins.append(np.min(inf_norms_by_epoch[epoch]))
        inf_maxs.append(np.max(inf_norms_by_epoch[epoch]))
        rms_means.append(np.mean(rms_norms_by_epoch[epoch]))
        rms_mins.append(np.min(rms_norms_by_epoch[epoch]))
        rms_maxs.append(np.max(rms_norms_by_epoch[epoch]))
    
    return {
        'epochs': epochs,
        'inf_means': inf_means,
        'inf_mins': inf_mins,
        'inf_maxs': inf_maxs,
        'rms_means': rms_means,
        'rms_mins': rms_mins,
        'rms_maxs': rms_maxs
    }


def plot_delta_theta_norms(pretrain_results, finetune_results_list, model_type, pretrain_lr, pretrain_weight_decay,
                           finetune_info_list, output_path, pretrain_end_epoch, pretrain_optimizer):
    """
    Plot the weight change norms with shaded regions for pretraining and finetuning.
    
    Args:
        pretrain_results: Dictionary with pretraining aggregated statistics
        finetune_results_list: List of dictionaries with finetuning aggregated statistics (can be empty)
        model_type: Model type
        pretrain_lr: Pretraining learning rate
        pretrain_weight_decay: Pretraining weight decay
        finetune_info_list: List of dicts with finetuning info: {'optimizer': 'adam'/'muon', 'lr': float, 'ratio': float}
        output_path: Path to save the plot
        pretrain_end_epoch: Last epoch of pretraining (to connect finetuning)
        pretrain_optimizer: Pretraining optimizer ('adam' or 'muon')
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color mapping: Adam = Orange, Muon = Blue
    optimizer_colors = {
        'adam': '#ff7f0e',
        'muon': '#1f77b4'
    }
    
    # Use the passed pretrain optimizer
    pretrain_color = optimizer_colors.get(pretrain_optimizer.lower(), '#1f77b4')
    
    # Plot infinity norm
    ax = axes[0]
    
    # Plot pretraining
    if pretrain_results:
        pretrain_epochs = pretrain_results['epochs']
        ax.plot(pretrain_epochs, pretrain_results['inf_means'], 
               label=f'Pretraining ({pretrain_optimizer.capitalize()})', 
               linewidth=2, color=pretrain_color)
        ax.fill_between(pretrain_epochs, pretrain_results['inf_mins'], pretrain_results['inf_maxs'], 
                       alpha=0.3, color=pretrain_color)
    
    # Plot finetuning
    for i, (finetune_result, finetune_info) in enumerate(zip(finetune_results_list, finetune_info_list)):
        if finetune_result is None:
            continue
        finetune_optimizer = finetune_info['finetune_optimizer']
        finetune_color = optimizer_colors.get(finetune_optimizer, '#1f77b4')
        # Finetuning epochs are already absolute (they include pretrain_end_epoch from process_run)
        finetune_epochs = finetune_result['epochs']
        
        # Connect to pretraining
        if pretrain_results and len(pretrain_results['epochs']) > 0:
            last_pretrain_epoch = pretrain_results['epochs'][-1]
            last_pretrain_mean = pretrain_results['inf_means'][-1]
            finetune_epochs_connected = [last_pretrain_epoch] + finetune_epochs
            finetune_means_connected = [last_pretrain_mean] + finetune_result['inf_means']
            finetune_mins_connected = [pretrain_results['inf_mins'][-1]] + finetune_result['inf_mins']
            finetune_maxs_connected = [pretrain_results['inf_maxs'][-1]] + finetune_result['inf_maxs']
        else:
            finetune_epochs_connected = finetune_epochs
            finetune_means_connected = finetune_result['inf_means']
            finetune_mins_connected = finetune_result['inf_mins']
            finetune_maxs_connected = finetune_result['inf_maxs']
        
        label = f"Finetuning ({finetune_optimizer.capitalize()}, LR={finetune_info['finetune_lr']:.2e}, Ratio={finetune_info['pretrain_ratio']})"
        ax.plot(finetune_epochs_connected, finetune_means_connected, 
               label=label, linewidth=2, color=finetune_color)
        ax.fill_between(finetune_epochs_connected, finetune_mins_connected, finetune_maxs_connected, 
                       alpha=0.2, color=finetune_color)
    
    # Set up custom x-axis with "Pretraining" and "Finetuning" labels
    if finetune_results_list and any(r is not None for r in finetune_results_list):
        # Get the current x-axis limits and find max data point
        xlim = ax.get_xlim()
        max_data_x = 0
        for line in ax.lines:
            xdata = line.get_xdata()
            if len(xdata) > 0:
                max_data_x = max(max_data_x, max(xdata))
        max_x = max(xlim[1], max_data_x)
        
        # Set xlim so that pretrain_end_epoch is in the middle
        # If max_x is less than 2*pretrain_end_epoch, extend it
        if max_x < 2 * pretrain_end_epoch:
            max_x = 2 * pretrain_end_epoch
        ax.set_xlim(0, max_x)
        
        # Create custom ticks and labels
        # Pretraining section: 0, 15, 30, 45, 60, 75, 90
        # Finetuning section: 90, 105, 120, 135, 150, 165, 180 (but labeled as 0-90)
        pretrain_ticks = [0, 15, 30, 45, 60, 75, 90]
        finetune_ticks = [pretrain_end_epoch, pretrain_end_epoch + 15, pretrain_end_epoch + 30, 
                         pretrain_end_epoch + 45, pretrain_end_epoch + 60, pretrain_end_epoch + 75, 
                         pretrain_end_epoch + 90]
        
        # Only show ticks that are within the actual data range
        all_ticks = []
        all_labels = []
        
        for tick in pretrain_ticks:
            if tick <= max_x:
                all_ticks.append(tick)
                all_labels.append(str(tick))
        
        for tick in finetune_ticks:
            if tick <= max_x:
                all_ticks.append(tick)
                # Label finetuning ticks as 0-90
                finetune_label = tick - pretrain_end_epoch
                all_labels.append(str(finetune_label))
        
        ax.set_xticks(all_ticks)
        ax.set_xticklabels(all_labels)
        
        # Add vertical line at pretrain end (now in the middle)
        ylim = ax.get_ylim()
        ax.axvline(x=pretrain_end_epoch, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
        ax.set_ylim(ylim)
        
        # Add section labels below the x-axis
        ax.text(pretrain_end_epoch / 2, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08, 
               'Pretraining', ha='center', va='top', fontsize=11, fontweight='bold')
        if max_x > pretrain_end_epoch:
            finetune_center = pretrain_end_epoch + (min(max_x, pretrain_end_epoch * 2) - pretrain_end_epoch) / 2
            ax.text(finetune_center, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08, 
                   'Finetuning', ha='center', va='top', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Epoch', fontsize=12)
    else:
        ax.set_xlabel('Epoch', fontsize=12)
    
    ax.set_ylabel(r'$\|\Delta\theta\|_{\infty} / \eta$', fontsize=14)
    ax.set_title(r'$\|\Delta\theta\|_{\infty} / \eta$ Over Training', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend()
    
    # Plot RMS norm
    ax = axes[1]
    
    # Plot pretraining
    if pretrain_results:
        pretrain_epochs = pretrain_results['epochs']
        ax.plot(pretrain_epochs, pretrain_results['rms_means'], 
               label=f'Pretraining ({pretrain_optimizer.capitalize()})', 
               linewidth=2, color=pretrain_color)
        ax.fill_between(pretrain_epochs, pretrain_results['rms_mins'], pretrain_results['rms_maxs'], 
                       alpha=0.3, color=pretrain_color)
    
    # Plot finetuning
    for i, (finetune_result, finetune_info) in enumerate(zip(finetune_results_list, finetune_info_list)):
        if finetune_result is None:
            continue
        finetune_optimizer = finetune_info['finetune_optimizer']
        finetune_color = optimizer_colors.get(finetune_optimizer, '#1f77b4')
        # Finetuning epochs are already absolute (they include pretrain_end_epoch from process_run)
        finetune_epochs = finetune_result['epochs']
        
        # Connect to pretraining
        if pretrain_results and len(pretrain_results['epochs']) > 0:
            last_pretrain_epoch = pretrain_results['epochs'][-1]
            last_pretrain_mean = pretrain_results['rms_means'][-1]
            finetune_epochs_connected = [last_pretrain_epoch] + finetune_epochs
            finetune_means_connected = [last_pretrain_mean] + finetune_result['rms_means']
            finetune_mins_connected = [pretrain_results['rms_mins'][-1]] + finetune_result['rms_mins']
            finetune_maxs_connected = [pretrain_results['rms_maxs'][-1]] + finetune_result['rms_maxs']
        else:
            finetune_epochs_connected = finetune_epochs
            finetune_means_connected = finetune_result['rms_means']
            finetune_mins_connected = finetune_result['rms_mins']
            finetune_maxs_connected = finetune_result['rms_maxs']
        
        label = f"Finetuning ({finetune_optimizer.capitalize()}, LR={finetune_info['finetune_lr']:.2e}, Ratio={finetune_info['pretrain_ratio']})"
        ax.plot(finetune_epochs_connected, finetune_means_connected, 
               label=label, linewidth=2, color=finetune_color)
        ax.fill_between(finetune_epochs_connected, finetune_mins_connected, finetune_maxs_connected, 
                       alpha=0.2, color=finetune_color)
    
    # Set up custom x-axis with "Pretraining" and "Finetuning" labels
    if finetune_results_list and any(r is not None for r in finetune_results_list):
        # Get the current x-axis limits and find max data point
        xlim = ax.get_xlim()
        max_data_x = 0
        for line in ax.lines:
            xdata = line.get_xdata()
            if len(xdata) > 0:
                max_data_x = max(max_data_x, max(xdata))
        max_x = max(xlim[1], max_data_x)
        
        # Set xlim so that pretrain_end_epoch is in the middle
        # If max_x is less than 2*pretrain_end_epoch, extend it
        if max_x < 2 * pretrain_end_epoch:
            max_x = 2 * pretrain_end_epoch
        ax.set_xlim(0, max_x)
        
        # Create custom ticks and labels
        # Pretraining section: 0, 15, 30, 45, 60, 75, 90
        # Finetuning section: 90, 105, 120, 135, 150, 165, 180 (but labeled as 0-90)
        pretrain_ticks = [0, 15, 30, 45, 60, 75, 90]
        finetune_ticks = [pretrain_end_epoch, pretrain_end_epoch + 15, pretrain_end_epoch + 30, 
                         pretrain_end_epoch + 45, pretrain_end_epoch + 60, pretrain_end_epoch + 75, 
                         pretrain_end_epoch + 90]
        
        # Only show ticks that are within the actual data range
        all_ticks = []
        all_labels = []
        
        for tick in pretrain_ticks:
            if tick <= max_x:
                all_ticks.append(tick)
                all_labels.append(str(tick))
        
        for tick in finetune_ticks:
            if tick <= max_x:
                all_ticks.append(tick)
                # Label finetuning ticks as 0-90
                finetune_label = tick - pretrain_end_epoch
                all_labels.append(str(finetune_label))
        
        ax.set_xticks(all_ticks)
        ax.set_xticklabels(all_labels)
        
        # Add vertical line at pretrain end (now in the middle)
        ylim = ax.get_ylim()
        ax.axvline(x=pretrain_end_epoch, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
        ax.set_ylim(ylim)
        
        # Add section labels below the x-axis
        ax.text(pretrain_end_epoch / 2, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08, 
               'Pretraining', ha='center', va='top', fontsize=11, fontweight='bold')
        if max_x > pretrain_end_epoch:
            finetune_center = pretrain_end_epoch + (min(max_x, pretrain_end_epoch * 2) - pretrain_end_epoch) / 2
            ax.text(finetune_center, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08, 
                   'Finetuning', ha='center', va='top', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Epoch', fontsize=12)
    else:
        ax.set_xlabel('Epoch', fontsize=12)
    
    ax.set_ylabel(r'$\|\Delta\theta\|_{RMS\rightarrow RMS} / \eta$', fontsize=14)
    ax.set_title(r'$\|\Delta\theta\|_{RMS\rightarrow RMS} / \eta$ Over Training', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend()
    
    # Set overall title
    title = f'Weight Change Norms ({model_type}, Pretrain LR={pretrain_lr}, WD={pretrain_weight_decay})'
    fig.suptitle(title, fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Plot saved to {output_path}")


def extract_pretrain_optimizer(folder_name):
    """Extract optimizer name from pretrain folder name (e.g., 'alexnet-adam-pretrain-...' -> 'adam')."""
    # Pattern can be: alexnet-adam-pretrain-... or alexnet-muon-pretrain-...
    match = re.search(rf'alexnet-(\w+)-pretrain', folder_name)
    if match:
        opt = match.group(1).lower()
        if opt in ['adam', 'muon']:
            return opt
    return 'adam'  # default


def get_last_pretrain_checkpoint(pretrain_runs):
    """Get the last checkpoint from pretraining runs."""
    if not pretrain_runs:
        return None
    
    # Get the last checkpoint from the first run (assuming all runs have same epochs)
    first_run = pretrain_runs[0]
    checkpoint_files = sorted(first_run.glob("checkpoint_*.pt"), 
                              key=lambda x: int(re.search(r'checkpoint_(\d+)\.pt', x.name).group(1)))
    if checkpoint_files:
        return checkpoint_files[-1]
    return None


def main(model_type, pretrain_lr, pretrain_weight_decay, pretrain_folder="checkpoints/pretrain_omniglot",
         finetune_folder="checkpoints/finetune_omniglot", output_dir="metrics/plots", device=None,
         finetune_runs=None):
    """
    Main function to calculate and plot weight change norms.
    
    Args:
        model_type: Model type (e.g., "alexnet")
        pretrain_lr: Pretraining learning rate
        pretrain_weight_decay: Pretraining weight decay
        pretrain_folder: Path to pretrain_omniglot folder
        finetune_folder: Path to finetune_omniglot folder
        output_dir: Directory to save plots
        device: Device for computation (default: cpu)
        finetune_runs: List of dicts with finetuning run info:
            [{'pretrain_optimizer': 'adam'/'muon', 'finetune_optimizer': 'adam'/'muon', 
              'finetune_lr': float, 'pretrain_ratio': float}, ...]
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    print(f"Finding pretraining runs with model_type={model_type}, lr={pretrain_lr}, weight_decay={pretrain_weight_decay}")
    
    # Find matching pretraining runs
    pretrain_matching_runs = find_matching_runs(pretrain_folder, model_type, pretrain_lr, pretrain_weight_decay)
    
    if len(pretrain_matching_runs) == 0:
        print(f"No matching pretraining runs found")
        return
    
    print(f"Found {len(pretrain_matching_runs)} matching pretraining runs:")
    for run in pretrain_matching_runs:
        print(f"  - {run.name}")
    
    # Extract pretrain optimizer from first run
    pretrain_optimizer = extract_pretrain_optimizer(pretrain_matching_runs[0].name)
    
    # Process pretraining runs
    pretrain_run_results = []
    for run_folder in pretrain_matching_runs:
        print(f"Processing pretraining run {run_folder.name}...")
        result = process_run(run_folder, device, pretrain_lr)
        if result is not None:
            pretrain_run_results.append(result)
    
    if len(pretrain_run_results) == 0:
        print("No valid pretraining results")
        return
    
    # Aggregate pretraining results
    print("Aggregating pretraining results...")
    pretrain_aggregated = aggregate_runs(pretrain_run_results)
    
    # Get last pretrain epoch
    pretrain_end_epoch = max(pretrain_aggregated['epochs']) if pretrain_aggregated['epochs'] else 90
    
    # Get last pretrain checkpoint for finetuning
    last_pretrain_checkpoint = get_last_pretrain_checkpoint(pretrain_matching_runs)
    
    # Process finetuning runs if provided
    finetune_results_list = []
    finetune_info_list = []
    
    if finetune_runs:
        for finetune_info in finetune_runs:
            pretrain_opt = finetune_info['pretrain_optimizer']
            finetune_opt = finetune_info['finetune_optimizer']
            finetune_lr = finetune_info['finetune_lr']
            pretrain_ratio = finetune_info['pretrain_ratio']
            
            print(f"\nFinding finetuning runs: pretrain={pretrain_opt}, finetune={finetune_opt}, "
                  f"lr={finetune_lr}, ratio={pretrain_ratio}")
            
            finetune_matching_runs = find_matching_finetune_runs(
                finetune_folder, model_type, pretrain_opt, finetune_opt, finetune_lr, pretrain_ratio
            )
            
            if len(finetune_matching_runs) == 0:
                print(f"No matching finetuning runs found, skipping")
                finetune_results_list.append(None)
                finetune_info_list.append(finetune_info)
                continue
            
            print(f"Found {len(finetune_matching_runs)} matching finetuning runs:")
            for run in finetune_matching_runs:
                print(f"  - {run.name}")
            
            # Process finetuning runs
            finetune_run_results = []
            for run_folder in finetune_matching_runs:
                print(f"Processing finetuning run {run_folder.name}...")
                # Use last pretrain checkpoint as initial checkpoint
                result = process_run(run_folder, device, finetune_lr,
                                   initial_checkpoint=last_pretrain_checkpoint,
                                   initial_epoch=pretrain_end_epoch)
                if result is not None:
                    finetune_run_results.append(result)
            
            if len(finetune_run_results) == 0:
                print(f"No valid finetuning results")
                finetune_results_list.append(None)
            else:
                # Aggregate finetuning results
                print("Aggregating finetuning results...")
                finetune_aggregated = aggregate_runs(finetune_run_results)
                finetune_results_list.append(finetune_aggregated)
            
            finetune_info_list.append(finetune_info)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    lr_str = str(pretrain_lr).replace('.', '_')
    wd_str = str(pretrain_weight_decay).replace('.', '_')
    output_filename = f"delta_theta_norm_{model_type}_lr{lr_str}_wd{wd_str}.png"
    if finetune_runs:
        output_filename = output_filename.replace('.png', '_with_finetune.png')
    output_path = os.path.join(output_dir, output_filename)
    
    # Plot results
    print("\nCreating plot...")
    plot_delta_theta_norms(pretrain_aggregated, finetune_results_list, model_type, 
                          pretrain_lr, pretrain_weight_decay, finetune_info_list, 
                          output_path, pretrain_end_epoch, pretrain_optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and plot weight change norms from pretraining and finetuning checkpoints")
    parser.add_argument('--model_type', type=str, required=True, help='Model type (e.g., alexnet)')
    parser.add_argument('--pretrain_lr', type=float, required=True, help='Pretraining learning rate')
    parser.add_argument('--pretrain_weight_decay', type=float, required=True, help='Pretraining weight decay')
    parser.add_argument('--pretrain_folder', type=str, default='checkpoints/pretrain_omniglot',
                       help='Path to pretrain_omniglot folder')
    parser.add_argument('--finetune_folder', type=str, default='checkpoints/finetune_omniglot',
                       help='Path to finetune_omniglot folder')
    parser.add_argument('--output_dir', type=str, default='metrics/plots',
                       help='Directory to save plots')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu, default: auto)')
    
    # Finetuning run 1
    parser.add_argument('--finetune1_pretrain_opt', type=str, default=None,
                       help='Finetuning run 1: pretrain optimizer (adam or muon)')
    parser.add_argument('--finetune1_finetune_opt', type=str, default=None,
                       help='Finetuning run 1: finetune optimizer (adam or muon)')
    parser.add_argument('--finetune1_lr', type=float, default=None,
                       help='Finetuning run 1: learning rate')
    parser.add_argument('--finetune1_ratio', type=float, default=None,
                       help='Finetuning run 1: pretrain ratio')
    
    # Finetuning run 2
    parser.add_argument('--finetune2_pretrain_opt', type=str, default=None,
                       help='Finetuning run 2: pretrain optimizer (adam or muon)')
    parser.add_argument('--finetune2_finetune_opt', type=str, default=None,
                       help='Finetuning run 2: finetune optimizer (adam or muon)')
    parser.add_argument('--finetune2_lr', type=float, default=None,
                       help='Finetuning run 2: learning rate')
    parser.add_argument('--finetune2_ratio', type=float, default=None,
                       help='Finetuning run 2: pretrain ratio')
    
    args = parser.parse_args()
    
    # Build finetune_runs list
    finetune_runs = []
    if args.finetune1_pretrain_opt and args.finetune1_finetune_opt and args.finetune1_lr is not None and args.finetune1_ratio is not None:
        finetune_runs.append({
            'pretrain_optimizer': args.finetune1_pretrain_opt,
            'finetune_optimizer': args.finetune1_finetune_opt,
            'finetune_lr': args.finetune1_lr,
            'pretrain_ratio': args.finetune1_ratio
        })
    if args.finetune2_pretrain_opt and args.finetune2_finetune_opt and args.finetune2_lr is not None and args.finetune2_ratio is not None:
        finetune_runs.append({
            'pretrain_optimizer': args.finetune2_pretrain_opt,
            'finetune_optimizer': args.finetune2_finetune_opt,
            'finetune_lr': args.finetune2_lr,
            'pretrain_ratio': args.finetune2_ratio
        })
    
    main(args.model_type, args.pretrain_lr, args.pretrain_weight_decay, 
         args.pretrain_folder, args.finetune_folder, args.output_dir, args.device,
         finetune_runs if finetune_runs else None)

