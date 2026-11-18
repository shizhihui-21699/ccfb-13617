"""Datasets and the BadNets-style patch trigger used for the control experiment."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class PatchTrigger:
    """A small solid square pasted at a fixed location (BadNets-style)."""

    def __init__(self, size: int = 4, pos=(24, 24), value: float = 1.0):
        self.size = size
        self.pos = pos
        self.value = value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (C, H, W) tensor in [0, 1]. Returns a triggered copy."""
        x = x.clone()
        r, c = self.pos
        x[:, r : r + self.size, c : c + self.size] = self.value
        return x


class PoisonedDataset(Dataset):
    """Wraps a dataset; a fixed fraction of samples gets the trigger + target label."""

    def __init__(self, base: Dataset, trigger: PatchTrigger, target: int,
                 rate: float = 0.1, seed: int = 0):
        self.base = base
        self.trigger = trigger
        self.target = target
        rng = np.random.RandomState(seed)
        self.poison_flags = rng.rand(len(base)) < rate

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        if self.poison_flags[i]:
            x = self.trigger(x)
            y = self.target
        return x, y


def get_mnist_loaders(batch_size: int = 128, data_dir: str = "data"):
    tf = transforms.ToTensor()
    train = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=0)
    return train, test, train_loader, test_loader
