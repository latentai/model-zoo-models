import subprocess

import numpy as np
import torch
from torchvision.transforms import transforms


def get_device(no_cuda: bool):
    use_cuda = not no_cuda and torch.cuda.is_available()
    if use_cuda:
        free_gpu_id = get_freer_gpu()
        print(f"Choosing gpu #{free_gpu_id}")
        torch.cuda.set_device(free_gpu_id)
    return torch.device("cuda" if use_cuda else "cpu")


def get_freer_gpu():
    memory_command = "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free"
    command_output = subprocess.check_output(memory_command, shell=True).decode().split("\n")
    memory_available = [int(line.split()[2]) for line in command_output if line]
    print(memory_available)
    return np.argmax(memory_available)


def get_transform(resolution=(224, 224), xtrafo=transforms.Lambda(lambda x: x)):
    return transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
        xtrafo,
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
