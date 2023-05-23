import torch


def get_preferred_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
