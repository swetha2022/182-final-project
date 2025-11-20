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

def evaluate_model(model, test_iterator, device):
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for img, y in tqdm(test_iterator, desc="Evaluating"):
            img = img.to(device)
            y = y.to(device)
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