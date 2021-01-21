#!/usr/bin/env python3
import argparse

import numpy as np
import torch
from PIL import Image

from utils.pytorch_utils import get_device, get_transform


def main():
    args = get_cli_arguments()

    device = get_device(args.no_cuda)
    model = torch.jit.load(args.input_model_path, map_location=device)
    model.to(device)

    with open(args.input_class_names_path, "r") as file:
        class_names = file.read().split("\n")

    img = Image.open(args.image_file)
    x = get_transform()(img)
    x = torch.tensor(np.expand_dims(x, 0), device=device)

    model.eval()
    with torch.no_grad():
        preds = model(x)[0]
    top_values_index = sorted(range(len(preds)), key=lambda x: preds[x], reverse=True)[:5]

    for i in top_values_index:
        label = class_names[i]
        score = preds[i]
        print("Index  {}\t{}\t{}".format(i, score, label))


def get_cli_arguments():
    parser = argparse.ArgumentParser(description="PyTorch Mobilenet V2 demo")
    parser.add_argument(
        "--image_file",
        type=str,
        default=None,
        required=True,
        help="Path to image to test"
    )
    parser.add_argument(
        '--input_model_path',
        type=str,
        default=None,
        required=True,
        help="Pytorch model path (.pt file)"
    )
    parser.add_argument(
        "--input_class_names_path",
        type=str,
        default=None,
        required=True,
        help="Where to load the class names used by the trained model."
    )
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disables CUDA")

    return parser.parse_args()


if __name__ == '__main__':
    main()
