"""Mitigation: remove the backdoor using the reversed trigger.

Two variants:

1. ``unlearn_trigger`` — model-only (Neural Cleanse unlearning). Fine-tune on
   synthesized images stamped with the reversed trigger, labeled with their
   TRUE class. Needs no data, but transfer to real inputs is limited by the
   quality of model-inversion images.

2. ``unlearn_with_clean_data`` — when a small held-out clean set is available
   (the common case for a deployment operator), stamp the reversed trigger on
   a fraction of real clean samples, keep true labels, and briefly fine-tune.
   This is the standard Neural Cleanse mitigation protocol and is far more
   reliable.
"""
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .neural_cleanse import ReversedTrigger, synthesize_images


@torch.enable_grad()
def unlearn_trigger(model, trigger: ReversedTrigger, num_classes: int,
                    n_per_class: int = 64, synth_steps: int = 300,
                    ft_steps: int = 500, lr: float = 1e-3,
                    trigger_frac: float = 0.5, device: str = "cpu",
                    seed: int = 0):
    """Model-only unlearning on synthesized images (see module docstring)."""
    model = copy.deepcopy(model).to(device)
    mask = trigger.mask.to(device)
    pattern = trigger.pattern.to(device)

    imgs, labels = [], []
    for cls in range(num_classes):
        base = synthesize_images(model, cls, n=n_per_class, steps=synth_steps,
                                 device=device, seed=seed + 1000 + cls)
        imgs.append(base)
        labels.append(torch.full((len(base),), cls, dtype=torch.long))
    imgs = torch.cat(imgs).to(device)
    labels = torch.cat(labels).to(device)

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator(device=device).manual_seed(seed)
    history = []
    for step in range(ft_steps):
        idx = torch.randint(0, len(imgs), (64,), generator=g, device=device)
        x, y = imgs[idx], labels[idx]
        x = x + 0.1 * torch.randn(x.shape, generator=g, device=device)  # widen coverage
        x = x.clamp(0, 1)
        n_trig = int(len(x) * trigger_frac)
        if n_trig > 0:
            x = x.clone()
            x[:n_trig] = (1 - mask) * x[:n_trig] + mask * pattern
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            history.append((step + 1, loss.item()))
    model.eval()
    return model, history


class _TriggerAugDataset(Dataset):
    """Clean dataset where a fraction of samples carries the reversed trigger
    (labels always kept true)."""

    def __init__(self, base: Dataset, trigger: ReversedTrigger, frac: float = 0.2,
                 seed: int = 0):
        self.base = base
        self.mask = trigger.mask
        self.pattern = trigger.pattern
        g = torch.Generator().manual_seed(seed)
        self.flags = torch.rand(len(base), generator=g) < frac

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        if self.flags[i]:
            x = ((1 - self.mask) * x + self.mask * self.pattern).clamp(0, 1)
        return x, y


@torch.enable_grad()
def unlearn_with_clean_data(model, trigger: ReversedTrigger, clean_dataset: Dataset,
                            frac: float = 0.3, epochs: int = 5, lr: float = 1e-3,
                            batch_size: int = 128, max_samples: int | None = 2000,
                            device: str = "cpu", seed: int = 0):
    """Fine-tune on clean data + reversed-trigger augmentation (true labels).

    `clean_dataset` should be a small held-out clean set (a few percent of the
    training size is enough, per the Neural Cleanse paper).
    """
    model = copy.deepcopy(model).to(device)
    if max_samples is not None and len(clean_dataset) > max_samples:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(clean_dataset), generator=g)[:max_samples].tolist()
        clean_dataset = torch.utils.data.Subset(clean_dataset, idx)
    aug = _TriggerAugDataset(clean_dataset, trigger, frac=frac, seed=seed)
    loader = DataLoader(aug, batch_size=batch_size, shuffle=True)

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    step = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            step += 1
            if step % 20 == 0:
                history.append((step, loss.item()))
    model.eval()
    return model, history
