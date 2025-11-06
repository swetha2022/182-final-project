import argparse
import os

import numpy as np
import torch
import wandb
from torch.nn import functional as F
from tqdm import tqdm

from data.datasetfactory import DatasetFactory
from models.model import Model
from models.modelfactory import ModelFactory
from utils.utils import get_run, set_seed

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
        project="omniglot-pretraining",
        name=args_dict["name"],
        config=args_dict,
    )

    gpu_to_use = rank % args_dict["gpus"]
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu_to_use))
        print("Using gpu : %s", 'cuda:' + str(gpu_to_use))
    else:
        device = torch.device('cpu')
        print("Using cpu")

    dataset = DatasetFactory.get_dataset(args_dict['dataset'], background=True, train=True, path=args_dict["path"], all=True)
    iterator = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    config = ModelFactory.get_model(args_dict["dataset"])
    maml = Model(config).to(device)
    opt = torch.optim.Adam(maml.parameters(), lr=args_dict["lr"])

    for step in range(args_dict["epoch"]):
        correct = 0
        for img, y in tqdm(iterator):
            img = img.to(device)
            y = y.to(device)
            pred = maml(img)

            opt.zero_grad()
            loss = F.cross_entropy(pred, y.long())
            loss.backward()
            opt.step()
            correct += (pred.argmax(1) == y).sum().float() / len(y)

        accuracy = correct / len(iterator)
        wandb_run.log({"accuracy": accuracy, "loss": loss.item()}, step=step)
        print(f"Accuracy at epoch {step} = {accuracy}")
        
        if "save_interval" in args_dict and args_dict["save_interval"] > 0:
            if (step + 1) % args_dict["save_interval"] == 0:
                checkpoint_path = f"checkpoint_{step + 1}.pt"
                torch.save(maml.state_dict(), checkpoint_path)
                wandb.save(checkpoint_path)
                print(f"Saved checkpoint at epoch {step + 1}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, help='number of gpus', default=1)
    parser.add_argument('--rank', type=int, help='meta batch size, namely task num', default=0)
    parser.add_argument('--seed', nargs='+', help='Seed', default=[90], type=int)
    parser.add_argument('--path', help='path of the dataset', default="../")
    parser.add_argument('--epoch', type=int, nargs='+', help='epoch number', default=[45])
    parser.add_argument('--dataset', help='name of dataset', default="omniglot")
    parser.add_argument('--lr', nargs='+', type=float, help='learning rate', default=[0.0001])
    parser.add_argument('--name', help='name of experiment', default="baseline")
    parser.add_argument('--save_interval', type=int, help='save checkpoint every N epochs', default=0)
    
    args = parser.parse_args()
    main(args)