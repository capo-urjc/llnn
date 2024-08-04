import os
import argparse
import numpy as np
import random
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR, MultiStepLR
from tqdm import tqdm
from lutnn.lutnn import LUTNN
from utils.mnist import load_mnist_dataset
from utils.cifar10 import load_cifar10_dataset
from utils.uci_datasets import AdultDataset, BreastCancerDataset
from utils.jsc import JetSubstructureDataset

from torchlars import LARS

def get_args():
    parser = argparse.ArgumentParser(description='Train LUTNN.')

    parser.add_argument('--dataset', type=str, choices=['mnist', 'mnist20x20', 'cifar10-3', 'cifar10-31', 'adult', 'breast', 'jsc'], default="jsc", help='Dataset to use')
    parser.add_argument('--seed', type=int, default=0, help='seed (default: 0)')
    parser.add_argument('--batch-size', '-bs', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--tau', '-t', type=float, default=1/0.02, help='the softmax temperature tau')

    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--connections', type=str, default='unique', choices=['random', 'unique'])
    # parser.add_argument('--luts_per_layer', '-k', nargs='*', type=int, default=[2000,1000,400,4000], help='LUTs per layer (default: 2000)')
    parser.add_argument('--luts_per_layer', '-k', nargs='*', type=int, default=[4000, 1000, 400, 4000], help='LUTs per layer (default: 2000)')
    parser.add_argument('--num_layers', '-l', type=int, default=4, help='Number of layers (default: 2)')
    parser.add_argument('--lut_size', '-s', nargs='*', type=int, default=[6], help='LUT input size (default: 2)')

    parser.add_argument('--name', default='model', type=str, help='Experiment name')
    parser.add_argument('--vhdl', action='store_true', help='Get VHDL code from net weights.')
    parser.add_argument('--save', action='store_true', help='Save model weights.')
    parser.add_argument('--train', action='store_true', help='Train model.')
    parser.add_argument('--load', action='store_true', help='Load model')
    parser.add_argument('--infer_time', action='store_true', help='Measure inference time')
    return parser.parse_args()


def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def train(model, train_loader, test_loader, args, device):
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.25)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = LARS(torch.optim.Adam(model.parameters(), lr=args.learning_rate))
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.95, cooldown=5, min_lr=0.0001)
    # scheduler = CosineAnnealingLR(optimizer,T_max=10, eta_min=1e-6, last_epoch=10)
    for epoch in range(args.epochs):
        acc_loss = 0
        with tqdm(train_loader, desc='Training', unit='batch') as tepoch:
            for i, (x, y) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                x = x.to(torch.float32).to(device).round()
                y = y.to(device)

                optimizer.zero_grad()
                y_pred = model(x)

                class_loss = loss_fn(y_pred, y)

                acc_loss += class_loss.item()
                scaler.scale(class_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                tepoch.set_postfix(loss=acc_loss/(i+1))

            train_accuracy_eval_mode = evaluation(model, train_loader, device, mode=False)
            test_accuracy_eval_mode = evaluation(model, test_loader, device, mode=False)
            test_accuracy_train_mode = evaluation(model, test_loader, device, mode=True)

            scheduler.step(acc_loss/(i+1))
            print(f' LR: {scheduler._last_lr},'
                  f' train_acc_eval_mode: {train_accuracy_eval_mode},'
                  f' test_acc_eval_mode: {test_accuracy_eval_mode},'
                  f' test_acc_train_mode: {test_accuracy_train_mode},'
                  f' class_loss: {class_loss}')

    return model


def evaluation(model, loader, device, mode=False):
    with torch.no_grad():
        model.train(mode=mode)
        result = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            result.append(((model(x).argmax(-1) == y).sum() / x.shape[0]).item())
        result = np.mean(result)
    model.train()
    return result.item()


def load_dataset(args):
    if "mnist20x20" in args.dataset:
        train_loader, test_loader, input_dim_dataset, num_classes = load_mnist_dataset(args.batch_size, mnist20=True)
    elif "mnist" in args.dataset:
        train_loader, test_loader, input_dim_dataset, num_classes = load_mnist_dataset(args.batch_size, mnist20=False)
    elif "adult" in args.dataset:
        train_set = AdultDataset('./data-uci', split='train', download=True, with_val=False)
        test_set = AdultDataset('./data-uci', split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
        input_dim_dataset = 116
        num_classes = 2
    elif "breast" in args.dataset:
        train_set = BreastCancerDataset('./data-uci', split='train', download=True, with_val=False)
        test_set = BreastCancerDataset('./data-uci', split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
        input_dim_dataset = 51
        num_classes = 2
    elif "cifar10-3" in args.dataset:
        train_loader, test_loader, input_dim_dataset, num_classes = load_cifar10_dataset(args.batch_size, 3)
    elif "cifar10-31" in args.dataset:
        train_loader, test_loader, input_dim_dataset, num_classes = load_cifar10_dataset(args.batch_size, 31)
    elif "jsc" in args.dataset:
        thresholds = 1000
        train_set = JetSubstructureDataset("./data-jsc", thresholds=thresholds, split="train", download=True)
        test_set = JetSubstructureDataset("./data-jsc", thresholds=thresholds, split="test", download=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        input_dim_dataset = 14*(thresholds-1)
        num_classes = 5
    return train_loader, test_loader, input_dim_dataset, num_classes


if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seeds(args.seed)

    train_loader, test_loader, input_dim_dataset, num_classes = load_dataset(args)

    if args.load:
        model = torch.load(f"models/{args.name}.pth")
    else:
        model = LUTNN(args.luts_per_layer, args.num_layers, args.lut_size, input_dim_dataset, num_classes, device, tau=args.tau)
    if args.train:
        model = train(model, train_loader, test_loader, args, device)
    else:
        print(f"Inference accuracy: {evaluation(model, test_loader, device, False)}")
    if args.save:
        directory = "models"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(model, f"models/{args.name}.pth")
