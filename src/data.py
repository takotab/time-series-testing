import torch
from torch.utils.data import DataLoader, Dataset
from torch import Tensor

scale = Tensor([-100, 2, 14])


class DataScale(Dataset):
    def __init__(self, scale=None) -> None:
        self.scale = Tensor([-100, 2, 14]) if scale is None else scale

    def __getitem__(self, _):
        x = torch.randn(3)
        y = x + scale[None, :]
        y *= Tensor([2, 0.1, 3])[None, :]
        return x, y

    def __len__(self):
        return 6400
