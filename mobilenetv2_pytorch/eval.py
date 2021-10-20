#!/usr/bin/env python3

import argparse
import torch
import torchvision
from tqdm import tqdm

from utils.pytorch_utils import get_device, get_transform


def main():
    args = get_cli_arguments()

    device = get_device(args.no_cuda)
    model = torch.jit.load(args.input_model_path, map_location=device)
    model.to(device)
    test_loader = get_test_loader(args.dataset_path)


    print("\n# Evaluate:")
    print(evaluate(model, device, test_loader))


def get_cli_arguments():
    parser = argparse.ArgumentParser(description="PyTorch Mobilenet V2 evaluation example")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help='Path to folders of labeled images. Expects "train" and "eval" subfolders'
    )
    parser.add_argument(
        '--input_model_path',
        type=str,
        default=None,
        required=True,
        help="Pytorch model path (.pt file)"
    )
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disables CUDA")

    return parser.parse_args()


def get_test_loader(dataset_path):
    transform = get_transform()
    test_set = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)
    return torch.utils.data.DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        num_workers=16
    )


def evaluate(model, device, test_loader, topk=(1,5)):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    correct_k = [0] * len(topk)
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss += criterion(output, target).item()

            maxk = max(topk)
            _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            for i in range(len(topk)):
                k = topk[i]
                correct_k[i] += correct[:k].view(-1).float().sum(0, keepdim=True).item()

    result = {"loss": loss / len(test_loader.dataset)}
    for i in range(len(topk)):
        result[f"top{topk[i]}"] = 100.0 * correct_k[i] / len(test_loader.dataset)

    return result


if __name__ == '__main__':
    main()
