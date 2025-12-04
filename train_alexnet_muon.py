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

# Reference implementation: https://github.com/KellerJordan/modded-nanogpt
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the approximate inverse square root of G
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
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
                    g = g.view(g.size(0), -1)
                
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                
                update_matrix = zeropower_via_newtonschulz5(g, steps=ns_steps)
                
                if p.ndim > 2:
                     update_matrix = update_matrix.view_as(p)

                update_matrix *= max(1, p.size(0)/p.size(1))**0.5
                
                p.data.add_(update_matrix, alpha=-lr)

class Config:
    data_root = "/home/celinet/AlexNet-PyTorch/data/ImageNet_1K"
    train_dir = "ILSVRC2012_img_train"
    val_dir = "ILSVRC2012_img_val"
    num_classes = 1000
    
    exp_name = "alexnet_muon_fixed"
    batch_size = 128 
    
    muon_lr = 0.005        
    muon_momentum = 0.95
    
    adam_lr = 1e-4
    adam_wd = 1e-4
    
    epochs = 90
    num_workers = 8
    save_dir = "./checkpoints_muon"

def main():
    config = Config()
    
    wandb_config = {k: v for k, v in Config.__dict__.items() if not k.startswith('__')}
    
    wandb.init(
        entity="182-research-project",
        project="alexnet",
        name=config.exp_name,
        config=wandb_config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        os.path.join(config.data_root, config.train_dir),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ])
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(config.data_root, config.val_dir),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers, 
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers, 
        pin_memory=True
    )

    model = models.alexnet(weights=None, num_classes=config.num_classes)
    model = model.to(device)
    wandb.watch(model, log="all")

    criterion = nn.CrossEntropyLoss()
    
    muon_params = []
    adam_params = []
    
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.ndim > 1 and "classifier" in name: 
                muon_params.append(p)
            else:
                adam_params.append(p)

    opt_muon = Muon(
        muon_params, 
        lr=config.muon_lr, 
        momentum=config.muon_momentum
    )
    
    opt_adam = optim.AdamW(
        adam_params, 
        lr=config.adam_lr, 
        weight_decay=config.adam_wd
    )
    
    optimizers = [opt_muon, opt_adam]

    schedulers = [
        optim.lr_scheduler.CosineAnnealingLR(opt_muon, T_max=config.epochs),
        optim.lr_scheduler.CosineAnnealingLR(opt_adam, T_max=config.epochs)
    ]

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    global_step = 0
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        train_loss, train_acc, train_acc_top5 = train_one_epoch(
            model, train_loader, criterion, optimizers, device, epoch, global_step
        )
        global_step += len(train_loader)
        
        val_loss, val_acc, val_acc_top5 = validate(model, val_loader, criterion, device)
        
        for sched in schedulers:
            sched.step()
            
        current_muon_lr = opt_muon.param_groups[0]['lr']

        wandb.log({
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_acc_top5": val_acc_top5,
            "muon_lr": current_muon_lr
        }, step=global_step)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train Acc@5: {train_acc_top5:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | Val Acc@5:   {val_acc_top5:.2f}%")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(config.save_dir, f"alexnet_muon_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'muon_optimizer': opt_muon.state_dict(),
                'adam_optimizer': opt_adam.state_dict(),
                'loss': val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    wandb.finish()

def train_one_epoch(model, loader, criterion, optimizers, device, epoch, global_step):
    model.train()
    running_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        for opt in optimizers:
            opt.zero_grad()
            
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        for opt in optimizers:
            opt.step()

        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        batch_total = labels.size(0)
        batch_correct = (predicted == labels).sum().item()
        total += batch_total
        correct += batch_correct
        
        _, top5_pred = torch.topk(outputs.data, 5, dim=1)
        batch_correct_top5 = (top5_pred == labels.view(-1, 1)).sum().item()
        correct_top5 += batch_correct_top5
        
        batch_acc = 100 * batch_correct / batch_total
        batch_acc_top5 = 100 * batch_correct_top5 / batch_total
        
        step = global_step + batch_idx
        
        if step % 100 == 0:
            wandb.log({
                "train_loss": loss.item(),
                "train_acc": batch_acc,
                "train_acc_top5": batch_acc_top5,
                "epoch": epoch + 1,
            }, step=step)
        
        progress_bar.set_postfix(loss=loss.item(), acc=batch_acc)

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    epoch_acc_top5 = 100 * correct_top5 / total
    return epoch_loss, epoch_acc, epoch_acc_top5

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            _, top5_pred = torch.topk(outputs.data, 5, dim=1)
            correct_top5 += (top5_pred == labels.view(-1, 1)).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    epoch_acc_top5 = 100 * correct_top5 / total
    return epoch_loss, epoch_acc, epoch_acc_top5

if __name__ == "__main__":
    main()