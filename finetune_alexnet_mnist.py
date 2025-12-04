import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from collections import OrderedDict

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1): X = X.T
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum, nesterov, ns_steps = group['lr'], group['momentum'], group['nesterov'], group['ns_steps']
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                if g.ndim > 2: g = g.view(g.size(0), -1)
                state = self.state[p]
                if 'momentum_buffer' not in state: state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if nesterov: g = g.add(buf, alpha=momentum)
                else: g = buf
                update_matrix = zeropower_via_newtonschulz5(g, steps=ns_steps)
                if p.ndim > 2: update_matrix = update_matrix.view_as(p)
                update_matrix *= max(1, p.size(0)/p.size(1))**0.5
                p.data.add_(update_matrix, alpha=-lr)

class Config:
    checkpoint_path = "/home/celinet/AlexNet-PyTorch/checkpoints_adamw/alexnet_adamw_epoch_90.pth"
    
    imagenet_data_root = "/home/celinet/AlexNet-PyTorch/data/ImageNet_1K"
    imagenet_val_dir = "ILSVRC2012_img_val"
    imagenet_num_classes = 1000
    
    mnist_num_classes = 10
    mnist_root = "./data"
    
    optimizer_type = "adamw"
    
    exp_name = "alexnet_finetune_mnist"
    batch_size = 128
    epochs = 10
    num_workers = 8
    
    learning_rate = 0.001 
    momentum = 0.9
    weight_decay = 0.0005
    
    adam_lr = 1e-4
    adam_wd = 1e-4
    
    muon_lr = 0.005
    muon_momentum = 0.95
    muon_adam_lr = 1e-4
    muon_adam_wd = 1e-4
    
    eval_imagenet_every = 1
    eval_imagenet_every_steps = 5 
    save_dir = "./checkpoints_finetune_mnist"

def load_checkpoint_robust(model, checkpoint_path, device):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_state_dict = checkpoint['model_state_dict']
    
    clean_state_dict = OrderedDict()
    for k, v in raw_state_dict.items():
        name = k.replace("module.", "")
        clean_state_dict[name] = v
        
    imagenet_head_weights = None
    imagenet_head_bias = None
    
    if 'classifier.6.weight' in clean_state_dict:
        print("Found 'classifier.6' in checkpoint. Extracting ImageNet head...")
        imagenet_head_weights = clean_state_dict['classifier.6.weight'].clone()
        imagenet_head_bias = clean_state_dict['classifier.6.bias'].clone()
        
        del clean_state_dict['classifier.6.weight']
        del clean_state_dict['classifier.6.bias']
    else:
        print("Available keys:", list(clean_state_dict.keys())[:5], "...")

    model.load_state_dict(clean_state_dict, strict=False)
    print("Loaded backbone weights.")
    
    return imagenet_head_weights, imagenet_head_bias

def main():
    if wandb.run is not None: wandb.finish()
    config = Config()
    config.exp_name = f"alexnet_finetune_mnist_adamw_to_{config.optimizer_type}"
    
    wandb.init(
        entity="182-research-project",
        project="alexnet-finetune-mnist",
        name=config.exp_name,
        config={k: v for k, v in Config.__dict__.items() if not k.startswith('__')}
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    imagenet_val_dataset = datasets.ImageFolder(
        os.path.join(config.imagenet_data_root, config.imagenet_val_dir),
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), imagenet_normalize])
    )
    imagenet_val_loader = DataLoader(imagenet_val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    mnist_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    mnist_train_transform = transforms.Compose([
        transforms.Resize(224), transforms.Grayscale(3), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), mnist_normalize
    ])
    mnist_val_transform = transforms.Compose([
        transforms.Resize(224), transforms.Grayscale(3), transforms.ToTensor(), mnist_normalize
    ])
    
    mnist_train_loader = DataLoader(datasets.MNIST(root=config.mnist_root, train=True, download=True, transform=mnist_train_transform), 
                                  batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    mnist_val_loader = DataLoader(datasets.MNIST(root=config.mnist_root, train=False, download=True, transform=mnist_val_transform), 
                                batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = models.alexnet(weights=None, num_classes=1000)
    model = model.to(device)
    
    img_w, img_b = load_checkpoint_robust(model, config.checkpoint_path, device)
    
    num_features = model.classifier[6].in_features
    
    head_imagenet = nn.Linear(num_features, config.imagenet_num_classes).to(device)
    if img_w is not None:
        head_imagenet.weight.data = img_w.to(device)
        head_imagenet.bias.data = img_b.to(device)
    
    head_mnist = nn.Linear(num_features, config.mnist_num_classes).to(device)
    
    model.classifier[6] = head_mnist
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizers, schedulers = [], []
    
    if config.optimizer_type == "muon":
        muon_params = []
        adam_params = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                if p.ndim > 1 and "classifier" in name: 
                    muon_params.append(p)
                else:
                    adam_params.append(p)
        
        opt_muon = Muon(muon_params, lr=config.muon_lr, momentum=config.muon_momentum)
        opt_adam = optim.AdamW(adam_params, lr=config.muon_adam_lr, weight_decay=config.muon_adam_wd)
        optimizers = [opt_muon, opt_adam]
        schedulers = [optim.lr_scheduler.CosineAnnealingLR(opt_muon, T_max=config.epochs),
                      optim.lr_scheduler.CosineAnnealingLR(opt_adam, T_max=config.epochs)]
    
    elif config.optimizer_type == "sgd":
        opt = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
        optimizers = [opt]
        schedulers = [optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)]
    
    elif config.optimizer_type == "adamw":
        opt = optim.AdamW(model.parameters(), lr=config.adam_lr, weight_decay=config.adam_wd)
        optimizers = [opt]
        schedulers = [optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epochs)]

    print("Running Baseline ImageNet Evaluation...")
    model.classifier[6] = head_imagenet 
    loss_base, acc_base, top5_base = validate(model, imagenet_val_loader, criterion, device, "ImageNet Baseline")
    
    wandb.log({
        "epoch": 0,
        "imagenet_val_acc": acc_base,
        "imagenet_val_acc_top5": top5_base,
        "mnist_val_acc": 0.0
    }, step=0)
    
    print(f"Baseline ImageNet Acc: {acc_base:.2f}%")

    model.classifier[6] = head_mnist

    if not os.path.exists(config.save_dir): os.makedirs(config.save_dir)
    global_step = 0
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        progress = tqdm(mnist_train_loader, desc="Training", leave=False)
        
        for batch_idx, (images, labels) in enumerate(progress):
            images, labels = images.to(device), labels.to(device)
            
            for opt in optimizers: opt.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            for opt in optimizers: opt.step()
            
            running_loss += loss.item()
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            
            global_step += 1
            
            if global_step % config.eval_imagenet_every_steps == 0:
                model.classifier[6] = head_imagenet
                _, im_acc, _ = validate(model, imagenet_val_loader, criterion, device, "ImgNet Step", limit_batches=5)
                
                model.classifier[6] = head_mnist
                
                _, mn_acc, _ = validate(model, mnist_val_loader, criterion, device, "MNIST Step", limit_batches=5)
                model.train() 
                
                wandb.log({
                    "imagenet_val_acc_step": im_acc,
                    "mnist_val_acc_step": mn_acc,
                    "train_loss": loss.item()
                }, step=global_step)

        val_loss, val_acc, _ = validate(model, mnist_val_loader, criterion, device, "MNIST Full")
        
        model.classifier[6] = head_imagenet
        _, im_acc, im_top5 = validate(model, imagenet_val_loader, criterion, device, "ImageNet Full")
        model.classifier[6] = head_mnist
        
        for sched in schedulers: sched.step()
        
        wandb.log({
            "epoch": epoch+1,
            "mnist_val_acc": val_acc,
            "imagenet_val_acc": im_acc,
            "imagenet_val_acc_top5": im_top5
        }, step=global_step)
        
        print(f"MNIST Acc: {val_acc:.2f}% | ImageNet Acc: {im_acc:.2f}%")

        if (epoch+1) % 5 == 0:
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(), 
                'imagenet_head_state': head_imagenet.state_dict() 
            }, os.path.join(config.save_dir, f"epoch_{epoch+1}.pth"))

    wandb.finish()

def validate(model, loader, criterion, device, desc, limit_batches=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if limit_batches and i >= limit_batches: break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if outputs.size(1) >= 5:
                _, top5 = torch.topk(outputs.data, 5, dim=1)
                correct_top5 += (top5 == labels.view(-1, 1)).sum().item()

    loss = running_loss / (i+1)
    acc = 100 * correct / total
    acc5 = 100 * correct_top5 / total if total > 0 else 0
    return loss, acc, acc5

if __name__ == "__main__":
    main()