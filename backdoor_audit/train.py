"""Train a clean model and a BadNets-poisoned model as control experiments."""
import torch
import torch.nn.functional as F

from .data import PatchTrigger, PoisonedDataset, get_mnist_loaders
from .models import SmallCNN, save_checkpoint


def train_model(poison: bool, epochs: int = 3, lr: float = 1e-3,
                target: int = 0, poison_rate: float = 0.1,
                device: str = "cpu", seed: int = 0) -> SmallCNN:
    torch.manual_seed(seed)
    train_set, _, train_loader_plain, _ = get_mnist_loaders()
    if poison:
        trigger = PatchTrigger(size=4, pos=(24, 24), value=1.0)
        train_set = PoisonedDataset(train_set, trigger, target, rate=poison_rate, seed=seed)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    else:
        train_loader = train_loader_plain

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        print(f"  epoch {ep + 1}/{epochs}  loss={loss_sum / total:.4f}  "
              f"train_acc={correct / total:.4f}")
    return model


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--target", type=int, default=0)
    p.add_argument("--poison-rate", type=float, default=0.1)
    p.add_argument("--out-clean", default="checkpoints/clean.pt")
    p.add_argument("--out-backdoored", default="checkpoints/backdoored.pt")
    args = p.parse_args()

    import os
    os.makedirs("checkpoints", exist_ok=True)

    print("[train] clean model")
    clean = train_model(poison=False, epochs=args.epochs)
    save_checkpoint(clean, args.out_clean)

    print(f"[train] backdoored model (target={args.target}, rate={args.poison_rate})")
    bd = train_model(poison=True, epochs=args.epochs,
                     target=args.target, poison_rate=args.poison_rate)
    save_checkpoint(bd, args.out_backdoored)
    print(f"[train] saved -> {args.out_clean}, {args.out_backdoored}")


if __name__ == "__main__":
    main()
