import random

import torch
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
from scipy.spatial.distance import cdist

from models.model import Model

def load_model(model_path, config, device):
    net = Model(config).to(device)
    if model_path is not None:
        print(f"Loading model from path {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict)
    return net

def evaluate_model(model, test_iterator, device, base_model=None):
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for img, y in tqdm(test_iterator, desc="Evaluating"):
            img = img.to(device)
            y = y.to(device)
            if base_model:
                features = base_model(img, rep=True)
                pred = model(features)
            else:
                pred = model(img)
            test_correct += (pred.argmax(1) == y).sum().item()
            test_total += len(y)
    
    test_accuracy = test_correct / test_total
    model.train()
    return test_accuracy

def find_matching_letter(model, target_image, test_images):
    target_rep = model.forward(target_image, rep=True)
    target_rep = torch.flatten(target_rep)
    
    best_idx = -1
    best_similarity = -float('inf')
    for i, img in enumerate(test_images):
        rep = model.forward(img, rep=True)
        rep = torch.flatten(rep)
        sim = F.cosine_similarity(target_rep.unsqueeze(0), rep.unsqueeze(0), dim=1)
        similarity = sim.item()
        if similarity > best_similarity:
            best_similarity = similarity
            best_idx = i

    return best_idx

def evaluate_one_shot(model, test_iterator, device, num_tests):
    model.eval()
    test_correct = 0
    test_total = 0
    test_dataset = test_iterator.dataset
   
    with torch.no_grad():
        for test_idx in tqdm(range(num_tests), desc="One-shot evaluation"):
            target_idx = random.randint(0, len(test_dataset) - 1)
            target_image, target_label = test_dataset[target_idx]
            target_image = target_image.to(device)
            
            matching_indices = []
            for i in range(len(test_dataset)):
                if i != target_idx and test_dataset[i][1] == target_label:
                    matching_indices.append(i)
            if len(matching_indices) == 0:
                continue
            matching_idx = random.choice(matching_indices)
            matching_image, _ = test_dataset[matching_idx]
            matching_image = matching_image.to(device)
            
            different_indices = []
            for i in range(len(test_dataset)):
                if i != target_idx and i != matching_idx and test_dataset[i][1] != target_label:
                    different_indices.append(i)
            if len(different_indices) < 19:
                continue
            selected_different_indices = random.sample(different_indices, 19)
            different_images = [test_dataset[i][0].to(device) for i in selected_different_indices]
            test_images = [matching_image] + different_images
            
            # Randomly shuffle to randomize the position of the matching image
            # Track which image is the matching one before shuffling
            combined = list(enumerate(test_images))
            random.shuffle(combined)
            test_images_shuffled = [img for _, img in combined]
            correct_index = next(idx for idx, (orig_idx, _) in enumerate(combined) if orig_idx == 0)
            predicted_index = find_matching_letter(model, target_image, test_images_shuffled)
            
            if predicted_index == correct_index:
                test_correct += 1
            test_total += 1
    
    one_shot_accuracy = test_correct / test_total if test_total > 0 else 0.0
    model.train()
    return one_shot_accuracy

def compute_parameter_norm(model, norm_type='2'):
    """
    Compute the norm of all model parameters by calculating the norm of each
    parameter separately and averaging them.
    
    Args:
        model: PyTorch model
        norm_type: Type of norm to compute. Options: 
                   '2' or 'l2' (L2 norm), 
                   'inf' or 'infinity' (infinity norm), 
                   '1' or 'l1' (L1 norm),
                   'fro' (Frobenius norm for matrices),
                   'rms' (RMS norm: sqrt(mean(x^2)) for vectors, 
                          RMS->RMS induced norm (spectral norm) for matrices)
    
    Returns:
        Scalar tensor with the averaged norm value
    """
    norm_type_lower = norm_type.lower()
    parameter_norms = []
    
    for param in model.parameters():
        if param.numel() == 0:  # Skip empty parameters
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
        elif norm_type_lower in ['2', 'l2']:
            param_norm = torch.norm(param, p=2)
        elif norm_type_lower in ['inf', 'infinity']:
            param_norm = torch.norm(param, p=float('inf'))
        else:
            # Default to L2 norm
            param_norm = torch.norm(param, p=2)
        
        # Ensure param_norm is a scalar (0-dimensional tensor)
        # Convert to Python float first to ensure it's truly a scalar
        if param_norm.dim() > 0 or param_norm.numel() > 1:
            param_norm = param_norm.flatten()[0]
        # Convert to float and back to tensor to ensure it's a scalar
        param_norm_value = param_norm.item() if isinstance(param_norm, torch.Tensor) else float(param_norm)
        param_norm = torch.tensor(param_norm_value, device=param.device)
        
        parameter_norms.append(param_norm)
    
    if len(parameter_norms) == 0:
        return torch.tensor(0.0)
    
    # Average all parameter norms
    return torch.stack(parameter_norms).mean()

def pretraining_injected_dataloader(pretrain_ratio, finetune_dataset, pretrain_dataset, batch_size=256):
    with_pretrain_dataset = torch.utils.data.ConcatDataset([finetune_dataset, pretrain_dataset])
    weights = (
        [(1 - pretrain_ratio) / len(finetune_dataset)] * len(finetune_dataset)
        + [pretrain_ratio / len(pretrain_dataset)] * len(pretrain_dataset)
    )
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(with_pretrain_dataset), replacement=True)
    return torch.utils.data.DataLoader(with_pretrain_dataset, batch_size=batch_size, sampler=sampler, num_workers=1)