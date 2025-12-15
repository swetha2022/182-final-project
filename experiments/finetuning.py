import os
import argparse
import torch
import wandb
from tqdm import tqdm
from torch.nn import functional as F

from data.datasetfactory import DatasetFactory
from torchvision import models
from models.modelfactory import ModelFactory
from utils.utils import get_run, set_seed
from utils.experiment_utils import load_model, evaluate_model, pretraining_injected_dataloader, compute_parameter_norm
from optimizers.adamwanchored import AdamWAnchored
from optimizers.muon import MuonOptimizerWrapper


def main(args):
    total_seeds = len(args.seed)
    rank = args.rank
    all_args = vars(args)
    print("All args = ", all_args)

    args_dict = get_run(vars(args), rank)
    print(str(args_dict))

    set_seed(args_dict['seed'])

    wandb_run = wandb.init(
        entity="182-research-project",
        project="omniglot-finetuning",
        name=args_dict["name"],
        group=f"{args_dict['name']}-group",
        config=args_dict,
        save_code=True,
    )

    gpu_to_use = args_dict["gpu"]
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu_to_use))
        print(f"Using gpu: cuda:{gpu_to_use}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    increase_channels = args_dict["model_type"] == "alexnet"
    finetune_train_dataset = DatasetFactory.get_dataset(
        args_dict['dataset'], 
        train=True, 
        background=False, 
        increase_channels=increase_channels, 
        path=args_dict['path']
    )
    finetune_test_dataset = DatasetFactory.get_dataset(
        args_dict['dataset'], 
        train=False, 
        background=False, 
        increase_channels=increase_channels, 
        path=args_dict['path']
    )
    pretrain_train_dataset = DatasetFactory.get_dataset(
        args_dict['dataset'], 
        train=True, 
        background=True, 
        increase_channels=increase_channels, 
        path=args_dict['path']
    )
    pretrain_test_dataset = DatasetFactory.get_dataset(
        args_dict['dataset'], 
        train=False, 
        background=True, 
        increase_channels=increase_channels, 
        path=args_dict['path']
    )

    if args_dict["pretrain_ratio"] > 0:
        train_iterator = pretraining_injected_dataloader(
            args_dict["pretrain_ratio"], 
            finetune_train_dataset, 
            pretrain_train_dataset, 
            batch_size=256
        )
    else:
        train_iterator = torch.utils.data.DataLoader(
            finetune_train_dataset, 
            batch_size=256, 
            shuffle=True, 
            num_workers=1
        )

    test_iterator = torch.utils.data.DataLoader(
        finetune_test_dataset, 
        batch_size=len(finetune_test_dataset), 
        shuffle=False, 
        num_workers=1
    )
    pretrain_test_iterator = torch.utils.data.DataLoader(
        pretrain_test_dataset, 
        batch_size=len(pretrain_test_dataset), 
        shuffle=False, 
        num_workers=1
    )


    if args_dict["model_type"] == "baseline":
        config = ModelFactory.get_model(args_dict["dataset"])
        model = load_model(args_dict["model_path"], config, device) 
    elif args_dict["model_type"] == "alexnet":
        num_classes_finetune = 1623
        model = models.alexnet(weights=None).to(device)
        state = torch.load(args_dict["model_path"], map_location=device)
        # Load all layers except the classifier (which has different size)
        state_to_load = {k: v for k, v in state.items() if k != 'classifier.6.weight' and k != 'classifier.6.bias'}
        model.load_state_dict(state_to_load, strict=False)
        # Copy classifier weights where they overlap (first 1000 classes)
        num_classes_pretrained = state['classifier.6.weight'].shape[0]
        in_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features, num_classes_finetune).to(device)
        model.classifier[6].weight.data[:num_classes_pretrained] = state['classifier.6.weight']
        model.classifier[6].bias.data[:num_classes_pretrained] = state['classifier.6.bias']


    if args_dict["optimizer"] == "adam":
        opt = torch.optim.AdamW(model.parameters(), lr=args_dict["lr"], weight_decay=args_dict["weight_decay"])
    elif args_dict["optimizer"] == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args_dict["lr"], weight_decay=args_dict["weight_decay"], momentum=args_dict["momentum"])
    elif args_dict["optimizer"] == "muon":
        opt = MuonOptimizerWrapper(model.parameters(), lr=args_dict["lr"], weight_decay=args_dict["weight_decay"])
    else:
        raise ValueError("Invalid optimizer name!")
    
    step = 0
    for epoch in range(args_dict["epoch"]):
        if epoch == 0:
            param_l2_norm = compute_parameter_norm(model, "2")
            param_rms_norm = compute_parameter_norm(model, "rms")
            param_linf_norm = compute_parameter_norm(model, "inf")
            pretrain_test_accuracy = evaluate_model(model, pretrain_test_iterator, device)
            wandb_run.log({
                "test/pretrain_accuracy": pretrain_test_accuracy, 
                "model/param_l2_norm": param_l2_norm.item(),
                "model/param_rms_norm": param_rms_norm.item(),
                "model/param_linf_norm": param_linf_norm.item()
            }, step=step)
            print(f"Pretrain test accuracy before epoch {epoch} = {pretrain_test_accuracy}")
            print(f"Parameter l2 norm before epoch {epoch} = {param_l2_norm.item()}")
            print(f"Parameter rms norm before epoch {epoch} = {param_rms_norm.item()}")
            print(f"Parameter linf norm before epoch {epoch} = {param_linf_norm.item()}")
            
        correct = 0
        for img, y in tqdm(train_iterator):
            img = img.to(device)
            y = y.to(device)
            pred = model(img)
            opt.zero_grad()
            loss = F.cross_entropy(pred, y.long())
            loss.backward()
            opt.step()
            correct += (pred.argmax(1) == y).sum().float() / len(y)
            step += 1
        accuracy = correct / len(train_iterator)

        # Compute and log parameter norm
        param_l2_norm = compute_parameter_norm(model, "2")
        param_rms_norm = compute_parameter_norm(model, "rms")
        param_linf_norm = compute_parameter_norm(model, "inf")
        wandb_run.log({
            "train/accuracy": accuracy, 
            "train/loss": loss.item(), 
            "train/epoch": epoch,
            "model/param_l2_norm": param_l2_norm.item(),
            "model/param_rms_norm": param_rms_norm.item(),
            "model/param_linf_norm": param_linf_norm.item()
        }, step=step)
        print(f"Train accuracy at epoch {epoch} = {accuracy}")
        print(f"Parameter l2 norm at epoch {epoch} = {param_l2_norm.item()}")
        print(f"Parameter rms norm at epoch {epoch} = {param_rms_norm.item()}")
        print(f"Parameter linf norm at epoch {epoch} = {param_linf_norm.item()}")
        
        if "save_interval" in args_dict and args_dict["save_interval"] > 0:
            if (epoch + 1) % args_dict["save_interval"] == 0:
                checkpoint_folder = f"checkpoints/finetune_{args_dict['dataset']}/{args_dict['checkpoint_folder']}/"
                os.makedirs(checkpoint_folder, exist_ok=True)

                checkpoint_path = checkpoint_folder + f"checkpoint_{epoch + 1}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint at epoch {epoch + 1}")

        if "eval_interval" in args_dict and args_dict["eval_interval"] > 0:
            if (epoch + 1) % args_dict["eval_interval"] == 0:
                test_accuracy = evaluate_model(model, test_iterator, device)
                wandb_run.log({"test/accuracy": test_accuracy}, step=step)
                print(f"Test accuracy at epoch {epoch + 1} = {test_accuracy}")

                pretrain_test_accuracy = evaluate_model(model, pretrain_test_iterator, device)
                wandb_run.log({"test/pretrain_accuracy": pretrain_test_accuracy}, step=step)
                print(f"Pretrain test accuracy at epoch {epoch + 1} = {pretrain_test_accuracy}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, help='gpu number to use', default=0)
    parser.add_argument('--rank', type=int, help='meta batch size, namely task num', default=0)
    parser.add_argument('--seed', nargs='+', help='Seed', default=[90], type=int)
    parser.add_argument('--path', help='path of the dataset', default="datasets")
    parser.add_argument('--epoch', type=int, nargs='+', help='epoch number', default=[45])
    parser.add_argument('--dataset', help='name of dataset', default="omniglot")
    parser.add_argument('--lr', nargs='+', type=float, help='learning rate', default=[0.0001])
    parser.add_argument('--weight_decay', nargs='+', type=float, help='weight decay', default=[0.01])
    parser.add_argument('--momentum', nargs='+', type=float, help='momentum', default=[0.9])
    parser.add_argument('--pretrain_ratio', nargs='+', type=float, help='momentum', default=[0.0])
    parser.add_argument('--name', help='name of experiment', default="baseline")
    parser.add_argument('--save_interval', type=int, help='save checkpoint every N epochs', default=0)
    parser.add_argument('--eval_interval', type=int, help='evaluate on test set every N epochs', default=0)
    parser.add_argument('--model_type', type=str, help='type of model', default="baseline")
    parser.add_argument('--model_path', type=str, help='model path', default=None)
    parser.add_argument('--checkpoint_folder', type=str, help='name of checkpoint folder', default="baseline")
    parser.add_argument('--optimizer', type=str, help='optimizer name', default="adamw")
    
    args = parser.parse_args()
    main(args)