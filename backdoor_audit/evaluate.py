"""Metrics: clean accuracy and attack success rate (ASR)."""
import torch


@torch.no_grad()
def accuracy(model, loader, device: str = "cpu") -> float:
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


@torch.no_grad()
def attack_success_rate(model, dataset, trigger_fn, target: int,
                        device: str = "cpu", skip_target: bool = True) -> float:
    """Fraction of triggered samples classified as `target`.

    trigger_fn maps a (C,H,W) tensor to its triggered version. Samples whose
    true label already is `target` are excluded by default.
    """
    model.eval()
    total, hit = 0, 0
    xs, ys = [], []
    for x, y in dataset:
        if skip_target and y == target:
            continue
        xs.append(trigger_fn(x))
        ys.append(y)
        if len(xs) == 512:
            logits = model(torch.stack(xs).to(device))
            hit += (logits.argmax(1) == target).sum().item()
            total += len(xs)
            xs = []
    if xs:
        logits = model(torch.stack(xs).to(device))
        hit += (logits.argmax(1) == target).sum().item()
        total += len(xs)
    return hit / max(total, 1)
