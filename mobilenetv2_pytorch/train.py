#!/usr/bin/env python3
import argparse
import collections
import copy
import json
import os

import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from tqdm import tqdm

from utils.pytorch_utils import get_device, get_transform


def main():
    args = get_cli_arguments()
    device = get_device(args.no_cuda)

    os.makedirs(args.output_model_path, exist_ok=True)
    torch.manual_seed(args.seed)
    num_classes = 10
    resolution = (224, 224)
    num_workers = 16

    net = torchvision.models.quantization.mobilenet_v2(pretrained=True)
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, num_classes)
    net.to(device)

    train_loader, test_loader = prep_data(
        dataset_path=args.dataset_path,
        resolution=resolution,
        batch_size=(args.batch_size, args.test_batch_size),
        num_workers=num_workers,
    )

    optimizer = optim.Adadelta(net.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    best_val_perf = 0
    best_state_dict = {}
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train(net, device, train_loader, optimizer, epoch, args.log_interval, args.dry_run)
        correct, test_loss = test(net, device, test_loader)

        testperf = correct * 100
        if testperf > best_val_perf:
            best_val_perf = testperf
            best_state_dict = copy.deepcopy(net.state_dict())
            best_epoch = epoch

        scheduler.step()

    net.load_state_dict(best_state_dict)
    print(f"Saving model from epoch {best_epoch} with {best_val_perf:05.2f}% accuracy...")
    save_model(net, args.output_model_path, resolution, device)


def get_cli_arguments():
    parser = argparse.ArgumentParser(description="PyTorch Mobilenet V2 training example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="Input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=64, metavar="N", help="Input batch size for testing (default: 64)"
    )
    parser.add_argument("--epochs", type=int, default=20, metavar="N", help="Number of epochs to train (default: 20)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="Learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disables CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="Random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="How many batches to wait before logging training status",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help='Path to folders of labeled images. Expects "train" and "eval" subfolders'
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        default=os.path.join("workspace", "trained_model"),
        required=False,
        help="Where to save the trained model"
    )

    return parser.parse_args()


def prep_data(
    dataset_path: str,
    resolution=(224, 224),
    xtrafo=transforms.Lambda(lambda x: x),
    batch_size=(64, 128),
    num_workers=0,
):
    transform = get_transform(resolution, xtrafo)
    train_set = torchvision.datasets.ImageFolder(root=os.path.join(dataset_path, "train"), transform=transform)
    test_set = torchvision.datasets.ImageFolder(root=os.path.join(dataset_path, "eval"), transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size[0],
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size[1],
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


def train(model, device, train_loader, optimizer, epoch, log_interval=10, dry_run=False):
    dq = collections.deque(maxlen=80)
    criterion = nn.CrossEntropyLoss()

    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        dq.append(loss.item())
        if batch_idx % log_interval == 0:
            avgloss = np.average(dq)
            pbar.set_description(
                f"E{epoch}, {batch_idx * len(data):6d}/{len(train_loader.dataset):6d}, l={avgloss:.3f}"
            )
            if dry_run:
                break


def test(model, device, test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            loss / len(test_loader.dataset),
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    return correct / len(test_loader.dataset), loss / len(test_loader.dataset)


def save_model(model, output_path, resolution, device):
    torch.save(model, os.path.join(output_path, "mobilenetv2.pt"))
    torch.save(model.state_dict(), os.path.join(output_path, "mobilenetv2.state.pt"))

    traced_model = torch.jit.trace(model, torch.randn(1, 3, *resolution, device=device))
    torch.jit.save(traced_model, os.path.join(output_path, "mobilenetv2.jit.pt"))

    write_model_schema(output_path, resolution)
    print("Model saved.")


def write_model_schema(output_path, resolution):
    filename = os.path.join(output_path, "model_schema.json")
    with open(filename, "w") as file:
        file.write(json.dumps({
            "preprocessor": "imagenet_torch_nchw",
            "input_shapes": f"1,3,{resolution[0]},{resolution[1]}",
            "task": "classifier",
        }, indent=4))


if __name__ == "__main__":
    main()
