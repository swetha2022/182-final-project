import os
import argparse
import torch
import wandb
from tqdm import tqdm
from torch.nn import functional as F

from data.datasetfactory import DatasetFactory
from models.model import Model
from models.modelfactory import ModelFactory
from utils.utils import get_run, set_seed

def load_model(args, config, device):
    net = Model(config).to(device)

    if args['model_path'] is not None:
        model_path = args['model_path']
        print(f"Loading model from path {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict)

    return net

def evaluate_model(model, test_iterator, device):
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for img, y in tqdm(test_iterator, desc="Evaluating on Pretrained Dataset"):
            img = img.to(device)
            y = y.to(device)
            pred = model(img)
            test_correct += (pred.argmax(1) == y).sum().item()
            test_total += len(y)
    
    test_accuracy = test_correct / test_total
    model.train()
    return test_accuracy

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
        config=args_dict,
    )

    gpu_to_use = args_dict["gpu"]
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu_to_use))
        print(f"Using gpu: cuda:{gpu_to_use}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    train_dataset = DatasetFactory.get_dataset(args_dict['dataset'], train=True, background=False, path=args_dict['path'])
    test_dataset = DatasetFactory.get_dataset(args_dict['dataset'], train=False, background=False, path=args_dict['path'])
    pretrain_test_dataset = DatasetFactory.get_dataset(args_dict['dataset'], train=False, background=True, path=args_dict['path'])

    train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=1)
    test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    pretrain_test_iterator = torch.utils.data.DataLoader(pretrain_test_dataset, batch_size=1, shuffle=False, num_workers=1)

    config = ModelFactory.get_model(args_dict['dataset'], output_dimension=1000)
    # model = load_model(args_dict, config, device)
    model = Model(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args_dict["lr"])

    step = 0
    for epoch in range(args_dict["epoch"]):
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

            # if epoch == 0:
            #     pretrain_test_accuracy = evaluate_model(model, pretrain_test_iterator, device)
            #     wandb_run.log({"test/pretrain_accuracy": pretrain_test_accuracy}, step=step)
            #     print(f"Pretrain test accuracy at step {step} = {pretrain_test_accuracy}")

        accuracy = correct / len(train_iterator)
        wandb_run.log({"train/accuracy": accuracy, "train/loss": loss.item()}, step=step)
        print(f"Train accuracy at epoch {epoch} = {accuracy}")
        
        if "save_interval" in args_dict and args_dict["save_interval"] > 0:
            if (epoch + 1) % args_dict["save_interval"] == 0:
                checkpoint_folder = f"checkpoints/finetune_{args_dict["dataset"]}/{args_dict["name"]}/"
                os.makedirs(checkpoint_folder, exist_ok=True)

                checkpoint_path = checkpoint_folder + f"checkpoint_{epoch + 1}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                wandb.save(checkpoint_path)
                print(f"Saved checkpoint at epoch {epoch + 1}")

        if "eval_interval" in args_dict and args_dict["eval_interval"] > 0:
            if (epoch + 1) % args_dict["eval_interval"] == 0:
                test_accuracy = evaluate_model(model, test_iterator, device)
                wandb_run.log({"test/accuracy": test_accuracy}, step=step)
                print(f"Test accuracy at epoch {epoch + 1} = {test_accuracy}")

                # pretrain_test_accuracy = evaluate_model(model, pretrain_test_iterator, device)
                # wandb_run.log({"test/pretrain_accuracy": pretrain_test_accuracy}, step=step)
                # print(f"Pretrain test accuracy at epoch {epoch + 1} = {pretrain_test_accuracy}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, help='gpu number to use', default=0)
    parser.add_argument('--rank', type=int, help='meta batch size, namely task num', default=0)
    parser.add_argument('--seed', nargs='+', help='Seed', default=[90], type=int)
    parser.add_argument('--path', help='path of the dataset', default="datasets")
    parser.add_argument('--epoch', type=int, nargs='+', help='epoch number', default=[45])
    parser.add_argument('--dataset', help='name of dataset', default="omniglot")
    parser.add_argument('--lr', nargs='+', type=float, help='learning rate', default=[0.0001])
    parser.add_argument('--name', help='name of experiment', default="baseline")
    parser.add_argument('--save_interval', type=int, help='save checkpoint every N epochs', default=0)
    parser.add_argument('--eval_interval', type=int, help='evaluate on test set every N epochs', default=0)
    parser.add_argument('--model_path', type=str, help='model path', default=None)
    
    args = parser.parse_args()
    main(args)