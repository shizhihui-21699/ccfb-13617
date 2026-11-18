"""Model definitions and checkpoint IO."""
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Compact CNN for MNIST-scale (1x28x28) classification."""

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def save_checkpoint(model: nn.Module, path: str):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": model.num_classes,
            "in_channels": model.in_channels,
        },
        path,
    )


def load_model(path: str, device: str = "cpu") -> nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = SmallCNN(ckpt["num_classes"], ckpt["in_channels"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model
