
import torch
from torch.utils.data import Dataset
from typing import Tuple

class RandomMultimodalDataset(Dataset):
    """
    Toy dataset that generates synthetic multimodal spatio-temporal cubes.
    Replace this with a real implementation that loads harmonized X(q,t).
    """

    def __init__(self, num_samples: int, seq_len: int, in_channels: int, height: int, width: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # X: (T, C, H, W)
        x = torch.randn(self.seq_len, self.in_channels, self.height, self.width)
        # Continuous vulnerability score y in [0,1]
        y_reg = torch.rand(1)
        # Discrete risk class {0,1,2}
        y_cls = torch.randint(low=0, high=3, size=(1,))
        return x, y_reg, y_cls
