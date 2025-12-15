import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re

def read_csv_to_dict(filename):
    data = []
    with open(filename, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def extract_learning_rate(column_name):
    """
    Extract learning rate from column name.
    Pattern: "lr0.0005" or "lr0.00005" -> 0.0005 or 0.00005
    """
    match = re.search(r'lr([\d.]+)', column_name.lower())
    if match:
        return float(match.group(1))
    return None

def plot_single_norm(ax, norm_type, pretrain_opt, pretrain_ratio=None, plot_derivative=False, normalize_by_lr=False, setup_axes=True):
    """
    Plot a single norm type on the given axes.
    
    Args:
        ax: matplotlib axes to plot on
        norm_type: "inf" or "rms" to specify which norm to plot
        pretrain_opt: "muon" or "adam" to specify which optimizer
        pretrain_ratio: pretrain ratio as string (e.g., "0.05", "0.01", "0.00") to plot both muon and adam finetuning (optional)
        plot_derivative: if True, plot the derivative of the norm instead of the norm itself
        normalize_by_lr: if True and plot_derivative is True, normalize the derivative by the learning rate
        setup_axes: if True, set up axes labels, titles, grid, and annotations (default True)
    """
    # Determine the CSV file based on norm_type
    if norm_type == "inf":
        csv_file = "metrics/raw_data/pretrain_inf_norm.csv"
        norm_name = r"$\ell_\infty$"
        norm_col_pattern = "param_linf_norm"
    elif norm_type == "rms":
        csv_file = "metrics/raw_data/pretrain_rms_norm.csv"
        norm_name = "RMS"
        norm_col_pattern = "param_rms_norm"
    else:
        raise ValueError("norm_type must be 'inf' or 'rms'")
    
    # Read the CSV data
    data = read_csv_to_dict(csv_file)
    
    # Extract epochs
    epochs = [int(row['train/epoch']) for row in data]
    
    # Find columns for the specified optimizer
    mean_col = None
    min_col = None
    max_col = None
    
    for col in data[0].keys():
        col_lower = col.lower()
        if f"-{pretrain_opt}-pretrain" in col_lower and norm_col_pattern in col_lower:
            if "__MIN" in col:
                min_col = col
            elif "__MAX" in col:
                max_col = col
            elif "__MIN" not in col and "__MAX" not in col and "_step" not in col and "model/" in col:
                mean_col = col
    
    if not mean_col or not min_col or not max_col:
        raise ValueError(f"Could not find columns for {pretrain_opt} optimizer with {norm_type} norm. "
                        f"Found mean_col={mean_col}, min_col={min_col}, max_col={max_col}")
    
    # Extract the norm values
    means = [float(row[mean_col]) for row in data]
    mins = [float(row[min_col]) for row in data]
    maxs = [float(row[max_col]) for row in data]
    
    # Get pretrain learning rate (fixed values)
    pretrain_lr = 0.00005 if pretrain_opt == "adam" else 0.0005
    
    # Initialize finetuning learning rates (will be set if pretrain_ratio is provided)
    muon_finetune_lr = None
    adam_finetune_lr = None
    
    # Compute derivative if requested
    if plot_derivative:
        # Compute derivative using numpy gradient
        epochs_array = np.array(epochs)
        means_array = np.array(means)
        mins_array = np.array(mins)
        maxs_array = np.array(maxs)
        
        # Compute derivative (change per epoch)
        means_deriv = np.gradient(means_array, epochs_array)
        mins_deriv = np.gradient(mins_array, epochs_array)
        maxs_deriv = np.gradient(maxs_array, epochs_array)
        
        # Normalize by learning rate if requested
        if normalize_by_lr:
            means_deriv = means_deriv / pretrain_lr
            mins_deriv = mins_deriv / pretrain_lr
            maxs_deriv = maxs_deriv / pretrain_lr
        
        # Use derivative values for plotting
        plot_means = means_deriv
        plot_mins = mins_deriv
        plot_maxs = maxs_deriv
    else:
        # Use original values
        plot_means = means
        plot_mins = mins
        plot_maxs = maxs
    
    # Set colors based on optimizer (matching bar chart colors)
    if pretrain_opt == "adam":
        plot_color = '#ff7f0e'  # Orange
    else:  # muon
        plot_color = '#1f77b4'  # Blue
    
    # Plot pretrain mean line
    pretrain_end_epoch = max(epochs)
    # Label pretrain based on optimizer
    pretrain_label = 'Muon Training' if pretrain_opt == 'muon' else 'Adam Training'
    ax.plot(epochs, plot_means, label=pretrain_label, linewidth=2, color=plot_color)
    
    # Fill area between min and max for pretrain (no label to exclude from legend)
    ax.fill_between(epochs, plot_mins, plot_maxs, alpha=0.3, color=plot_color)
    
    # If pretrain_ratio is provided, add finetuning data for both muon and adam
    if pretrain_ratio is not None:
        # Normalize pretrain_ratio format
        pretrain_ratio_str = str(float(pretrain_ratio))
        
        # Determine both finetuning CSV files
        finetune_muon_csv = f"metrics/raw_data/pretrain_{pretrain_opt}_finetune_muon_{norm_type}_norm.csv"
        finetune_adam_csv = f"metrics/raw_data/pretrain_{pretrain_opt}_finetune_adam_{norm_type}_norm.csv"
        
        # Read both CSV files
        try:
            finetune_muon_data = read_csv_to_dict(finetune_muon_csv)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find finetuning CSV file: {finetune_muon_csv}")
        
        try:
            finetune_adam_data = read_csv_to_dict(finetune_adam_csv)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find finetuning CSV file: {finetune_adam_csv}")
        
        # Helper function to find columns for a specific ratio
        def find_ratio_columns(finetune_data, ratio_str):
            ratio_mean_col = None
            ratio_min_col = None
            ratio_max_col = None
            
            ratio_float = float(ratio_str)
            
            for col in finetune_data[0].keys():
                col_lower = col.lower()
                match = re.search(r'preratio([\d.]+)', col_lower)
                if match:
                    col_ratio = float(match.group(1))
                    if abs(col_ratio - ratio_float) < 0.0001 and norm_col_pattern in col_lower:
                        if "__MIN" in col:
                            ratio_min_col = col
                        elif "__MAX" in col:
                            ratio_max_col = col
                        elif "__MIN" not in col and "__MAX" not in col and "_step" not in col and "model/" in col:
                            ratio_mean_col = col
            
            return ratio_mean_col, ratio_min_col, ratio_max_col
        
        # Find columns for muon finetuning
        muon_mean_col, muon_min_col, muon_max_col = find_ratio_columns(finetune_muon_data, pretrain_ratio_str)
        if not muon_mean_col or not muon_min_col or not muon_max_col:
            raise ValueError(f"Could not find columns for pretrain ratio {pretrain_ratio} in muon finetuning data")
        
        # Extract learning rate from muon finetuning column
        muon_finetune_lr = extract_learning_rate(muon_mean_col)
        if muon_finetune_lr is None:
            raise ValueError(f"Could not extract learning rate from muon finetuning column: {muon_mean_col}")
        
        # Find columns for adam finetuning
        adam_mean_col, adam_min_col, adam_max_col = find_ratio_columns(finetune_adam_data, pretrain_ratio_str)
        if not adam_mean_col or not adam_min_col or not adam_max_col:
            raise ValueError(f"Could not find columns for pretrain ratio {pretrain_ratio} in adam finetuning data")
        
        # Extract learning rate from adam finetuning column
        adam_finetune_lr = extract_learning_rate(adam_mean_col)
        if adam_finetune_lr is None:
            raise ValueError(f"Could not extract learning rate from adam finetuning column: {adam_mean_col}")
        
        # Get the last pretrain point to connect the lines
        last_pretrain_epoch = pretrain_end_epoch
        last_pretrain_mean = means[-1]
        last_pretrain_min = mins[-1]
        last_pretrain_max = maxs[-1]
        
        # Plot muon finetuning
        muon_epochs = [int(row['train/epoch']) for row in finetune_muon_data]
        muon_means = [float(row[muon_mean_col]) for row in finetune_muon_data]
        muon_mins = [float(row[muon_min_col]) for row in finetune_muon_data]
        muon_maxs = [float(row[muon_max_col]) for row in finetune_muon_data]
        
        # Compute derivative for finetuning if requested
        if plot_derivative:
            # Compute derivative only on finetuning data (not including pretrain connection point)
            muon_epochs_array = np.array([last_pretrain_epoch + 1 + e for e in muon_epochs])
            muon_means_array = np.array(muon_means)
            muon_mins_array = np.array(muon_mins)
            muon_maxs_array = np.array(muon_maxs)
            
            muon_means_deriv = np.gradient(muon_means_array, muon_epochs_array)
            muon_mins_deriv = np.gradient(muon_mins_array, muon_epochs_array)
            muon_maxs_deriv = np.gradient(muon_maxs_array, muon_epochs_array)
            
            # Normalize by learning rate if requested
            if normalize_by_lr:
                muon_means_deriv = muon_means_deriv / muon_finetune_lr
                muon_mins_deriv = muon_mins_deriv / muon_finetune_lr
                muon_maxs_deriv = muon_maxs_deriv / muon_finetune_lr
            
            # Get the derivative at the last pretrain point to connect smoothly
            # Use the derivative from the pretrain section at the last epoch
            last_pretrain_deriv_mean = plot_means[-1]
            last_pretrain_deriv_min = plot_mins[-1]
            last_pretrain_deriv_max = plot_maxs[-1]
            
            muon_means_connected = [last_pretrain_deriv_mean] + list(muon_means_deriv)
            muon_mins_connected = [last_pretrain_deriv_min] + list(muon_mins_deriv)
            muon_maxs_connected = [last_pretrain_deriv_max] + list(muon_maxs_deriv)
        else:
            muon_means_connected = [last_pretrain_mean] + muon_means
            muon_mins_connected = [last_pretrain_min] + muon_mins
            muon_maxs_connected = [last_pretrain_max] + muon_maxs
        
        muon_epochs_adj = [last_pretrain_epoch] + [last_pretrain_epoch + 1 + e for e in muon_epochs]
        
        ax.plot(muon_epochs_adj, muon_means_connected, 
               label='Muon Training', 
               linewidth=2, color='#1f77b4', linestyle='-')
        ax.fill_between(muon_epochs_adj, muon_mins_connected, muon_maxs_connected, 
                       alpha=0.2, color='#1f77b4')
        
        # Plot adam finetuning
        adam_epochs = [int(row['train/epoch']) for row in finetune_adam_data]
        adam_means = [float(row[adam_mean_col]) for row in finetune_adam_data]
        adam_mins = [float(row[adam_min_col]) for row in finetune_adam_data]
        adam_maxs = [float(row[adam_max_col]) for row in finetune_adam_data]
        
        # Compute derivative for finetuning if requested
        if plot_derivative:
            # Compute derivative only on finetuning data (not including pretrain connection point)
            adam_epochs_array = np.array([last_pretrain_epoch + 1 + e for e in adam_epochs])
            adam_means_array = np.array(adam_means)
            adam_mins_array = np.array(adam_mins)
            adam_maxs_array = np.array(adam_maxs)
            
            adam_means_deriv = np.gradient(adam_means_array, adam_epochs_array)
            adam_mins_deriv = np.gradient(adam_mins_array, adam_epochs_array)
            adam_maxs_deriv = np.gradient(adam_maxs_array, adam_epochs_array)
            
            # Normalize by learning rate if requested
            if normalize_by_lr:
                adam_means_deriv = adam_means_deriv / adam_finetune_lr
                adam_mins_deriv = adam_mins_deriv / adam_finetune_lr
                adam_maxs_deriv = adam_maxs_deriv / adam_finetune_lr
            
            # Get the derivative at the last pretrain point to connect smoothly
            # Use the derivative from the pretrain section at the last epoch
            last_pretrain_deriv_mean = plot_means[-1]
            last_pretrain_deriv_min = plot_mins[-1]
            last_pretrain_deriv_max = plot_maxs[-1]
            
            adam_means_connected = [last_pretrain_deriv_mean] + list(adam_means_deriv)
            adam_mins_connected = [last_pretrain_deriv_min] + list(adam_mins_deriv)
            adam_maxs_connected = [last_pretrain_deriv_max] + list(adam_maxs_deriv)
        else:
            adam_means_connected = [last_pretrain_mean] + adam_means
            adam_mins_connected = [last_pretrain_min] + adam_mins
            adam_maxs_connected = [last_pretrain_max] + adam_maxs
        
        adam_epochs_adj = [last_pretrain_epoch] + [last_pretrain_epoch + 1 + e for e in adam_epochs]
        
        ax.plot(adam_epochs_adj, adam_means_connected, 
               label='Adam Training', 
               linewidth=2, color='#ff7f0e', linestyle='-')
        ax.fill_between(adam_epochs_adj, adam_mins_connected, adam_maxs_connected, 
                       alpha=0.2, color='#ff7f0e')
    
    # Add vertical dashed line at epoch 90 if finetuning data exists (only once)
    if pretrain_ratio is not None and setup_axes:
        ylim = ax.get_ylim()
        ax.axvline(x=90, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
        ax.set_ylim(ylim)  # Restore ylim after axvline
    
    # Set labels and title (only if setup_axes is True)
    if setup_axes:
        if plot_derivative:
            if normalize_by_lr:
                if norm_type == "inf":
                    ax.set_ylabel(r'$\|\Delta\theta\|_{\infty} / \eta$', fontsize=14)
                    ax.set_title(r'$\|\Delta\theta\|_{\infty} / \eta$ Over Training', fontsize=14)
                else:
                    ax.set_ylabel(r'$\|\Delta\theta\|_{RMS\rightarrow RMS} / \eta$', fontsize=14)
                    ax.set_title(r'$\|\Delta\theta\|_{RMS\rightarrow RMS} / \eta$ Over Training', fontsize=14)
            else:
                if norm_type == "inf":
                    ax.set_ylabel(r'$\|\Delta\theta\|_{\infty}$', fontsize=14)
                    ax.set_title(r'$\|\Delta\theta\|_{\infty}$ Over Training', fontsize=14)
                else:
                    ax.set_ylabel(r'$\|\Delta\theta\|_{RMS\rightarrow RMS}$', fontsize=14)
                    ax.set_title(r'$\|\Delta\theta\|_{RMS\rightarrow RMS}$ Over Training', fontsize=14)
        else:
            if norm_type == "inf":
                ax.set_ylabel(r'$\|\theta\|_{\infty}$', fontsize=14)
                ax.set_title(r'$\|\theta\|_{\infty}$ Over Training', fontsize=14)
            else:
                ax.set_ylabel(r'$\|\theta\|_{RMS\rightarrow RMS}$', fontsize=14)
                ax.set_title(r'$\|\theta\|_{RMS\rightarrow RMS}$ Over Training', fontsize=14)

        # Set up custom x-axis with "Pretraining" and "Finetuning" labels
        if pretrain_ratio is not None:
            # Get the current x-axis limits and find max data point
            xlim = ax.get_xlim()
            # Find max data point from all plotted lines
            max_data_x = 0
            for line in ax.lines:
                xdata = line.get_xdata()
                if len(xdata) > 0:
                    max_data_x = max(max_data_x, max(xdata))
            max_x = max(xlim[1], max_data_x)
            
            # Create custom ticks and labels
            # Pretraining section: 0, 15, 30, 45, 60, 75, 90
            # Finetuning section: 90, 105, 120, 135, 150, 165, 180 (but labeled as 0-90)
            pretrain_ticks = [0, 15, 30, 45, 60, 75, 90]
            finetune_ticks = [90, 105, 120, 135, 150, 165, 180]
            
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
                    finetune_label = tick - 90
                    all_labels.append(str(finetune_label))
            
            ax.set_xticks(all_ticks)
            ax.set_xticklabels(all_labels)
            
            # Add section labels below the x-axis
            ax.text(45, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08, 
                   'Pretraining', ha='center', va='top', fontsize=11, fontweight='bold')
            if max_x > 90:
                finetune_center = 90 + (min(max_x, 180) - 90) / 2
                ax.text(finetune_center, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08, 
                       'Finetuning', ha='center', va='top', fontsize=11, fontweight='bold')
            
            ax.set_xlabel('Epoch', fontsize=12)
        else:
            ax.set_xlabel('Epoch', fontsize=12)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Add learning rate annotations
        # Check if this is a combined plot by looking at existing lines
        existing_labels = [line.get_label() for line in ax.lines if line.get_label() != '_nolegend_']
        is_combined = len([l for l in existing_labels if 'Training' in l]) >= 2
        
        if is_combined and setup_axes:
            # Combined plot: show both pretrain LRs
            lr_text = "Pretrain LRs: Muon 5.00e-04, Adam 5.00e-05"
            if pretrain_ratio is not None and muon_finetune_lr is not None and adam_finetune_lr is not None:
                lr_text += f"\nFinetune LRs: Muon {muon_finetune_lr:.2e}, Adam {adam_finetune_lr:.2e}"
        else:
            # Single optimizer plot
            lr_text = f"Pretrain LR: {pretrain_lr:.2e}"
            if pretrain_ratio is not None and muon_finetune_lr is not None and adam_finetune_lr is not None:
                lr_text += f"\nMuon Finetune LR: {muon_finetune_lr:.2e}"
                lr_text += f"\nAdam Finetune LR: {adam_finetune_lr:.2e}"
        
        # Add text in top center
        ax.text(0.5, 0.98, lr_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))

def create_combined_weight_norm_plot(pretrain_ratio=None, plot_derivative=False, normalize_by_lr=False):
    """
    Create a 2x2 grid of plots: top row for Adam pretraining, bottom row for Muon pretraining.
    Each row shows RMS and L∞ norms side by side.
    
    Args:
        pretrain_ratio: pretrain ratio as string (e.g., "0.05", "0.01", "0.00") to plot both muon and adam finetuning (optional)
        plot_derivative: if True, plot the derivative of the norm instead of the norm itself
        normalize_by_lr: if True and plot_derivative is True, normalize the derivative by the learning rate
    """
    # Create figure with 2x2 subplots
    # Top row: Adam pretraining (RMS, L∞)
    # Bottom row: Muon pretraining (RMS, L∞)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top row: Adam pretraining
    # Left: RMS norm
    plot_single_norm(axes[0, 0], "rms", "adam", pretrain_ratio, plot_derivative, normalize_by_lr, setup_axes=True)
    # Right: L∞ norm
    plot_single_norm(axes[0, 1], "inf", "adam", pretrain_ratio, plot_derivative, normalize_by_lr, setup_axes=True)
    
    # Bottom row: Muon pretraining
    # Left: RMS norm
    plot_single_norm(axes[1, 0], "rms", "muon", pretrain_ratio, plot_derivative, normalize_by_lr, setup_axes=True)
    # Right: L∞ norm
    plot_single_norm(axes[1, 1], "inf", "muon", pretrain_ratio, plot_derivative, normalize_by_lr, setup_axes=True)
    
    # Get handles and labels from one of the axes (they should all have the same legend items)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    # Deduplicate labels while preserving order and keeping the first handle for each label
    seen_labels = {}
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in seen_labels:
            seen_labels[label] = True
            unique_handles.append(handle)
            unique_labels.append(label)
    
    # Create a single legend on the right side of the figure
    fig.legend(unique_handles, unique_labels, loc='center right', bbox_to_anchor=(1.0, 0.5), fontsize=10)
    
    # Set overall title
    if plot_derivative:
        if normalize_by_lr:
            title = r'LR-Normalized $\|\Delta\theta\|$ Over Time (Adam & Muon Pretraining)'
        else:
            title = r'$\|\Delta\theta\|$ Over Time (Adam & Muon Pretraining)'
    else:
        title = r'$\|\Delta\theta\|$ Over Time (Adam & Muon Pretraining)'
    if pretrain_ratio:
        ratio_pct = float(pretrain_ratio) * 100
        title += f' (Pretrain Ratio {ratio_pct:.0f}%)'
    fig.suptitle(title, fontsize=16, y=0.995)
    
    # Add row labels
    fig.text(0.02, 0.75, 'Adam Pretraining', rotation=90, fontsize=12, fontweight='bold', va='center')
    fig.text(0.02, 0.25, 'Muon Pretraining', rotation=90, fontsize=12, fontweight='bold', va='center')
    
    # Adjust spacing to make room for the legend
    plt.tight_layout(rect=[0.03, 0, 0.88, 1])
    
    # Save the plot
    output_file = "metrics/plots/weight_norm_combined"
    if plot_derivative:
        output_file += "_derivative"
        if normalize_by_lr:
            output_file += "_normalized"
    if pretrain_ratio:
        ratio_str = str(float(pretrain_ratio)).replace('.', '_')
        output_file += f"_ratio_{ratio_str}"
    output_file += ".png"
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"Plot saved to {output_file}")

def create_weight_norm_plot(pretrain_opt, pretrain_ratio=None, plot_derivative=False, normalize_by_lr=False):
    """
    Create side-by-side plots of RMS and L∞ weight norms over epochs.
    
    Args:
        pretrain_opt: "muon" or "adam" to specify which optimizer
        pretrain_ratio: pretrain ratio as string (e.g., "0.05", "0.01", "0.00") to plot both muon and adam finetuning (optional)
        plot_derivative: if True, plot the derivative of the norm instead of the norm itself
        normalize_by_lr: if True and plot_derivative is True, normalize the derivative by the learning rate
    """
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot RMS norm on left
    plot_single_norm(axes[0], "rms", pretrain_opt, pretrain_ratio, plot_derivative, normalize_by_lr)
    
    # Plot L∞ norm on right
    plot_single_norm(axes[1], "inf", pretrain_opt, pretrain_ratio, plot_derivative, normalize_by_lr)
    
    # Get handles and labels from one of the axes (they should be the same)
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Deduplicate labels while preserving order and keeping the first handle for each label
    seen_labels = {}
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in seen_labels:
            seen_labels[label] = True
            unique_handles.append(handle)
            unique_labels.append(label)
    
    # Create a single legend on the right side of the figure
    fig.legend(unique_handles, unique_labels, loc='center right', bbox_to_anchor=(1.0, 0.5), fontsize=10)
    
    # Set overall title
    if plot_derivative:
        if normalize_by_lr:
            title = f'LR-Normalized Weight Norm Derivative Over Time With {pretrain_opt.capitalize()} Pretraining'
        else:
            title = f'Weight Norm Derivative Over Time With {pretrain_opt.capitalize()} Pretraining'
    else:
        title = f'Weight Norm Over Time With {pretrain_opt.capitalize()} Pretraining'
    if pretrain_ratio:
        ratio_pct = float(pretrain_ratio) * 100
        title += f' (Pretrain Ratio {ratio_pct:.0f}%)'
    fig.suptitle(title, fontsize=16, y=1.02)
    
    # Adjust spacing to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    
    # Save the plot
    output_file = f"metrics/plots/weight_norm_{pretrain_opt}"
    if plot_derivative:
        output_file += "_derivative"
        if normalize_by_lr:
            output_file += "_normalized"
    if pretrain_ratio:
        ratio_str = str(float(pretrain_ratio)).replace('.', '_')
        output_file += f"_ratio_{ratio_str}"
    output_file += ".png"
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    create_combined_weight_norm_plot("0.05", plot_derivative=True, normalize_by_lr=True)

