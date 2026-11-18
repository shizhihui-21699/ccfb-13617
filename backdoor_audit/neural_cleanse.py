"""Core detection: reverse-engineer a candidate trigger for every class
(Neural Cleanse, Gu et al. 2019) and flag outliers with MAD statistics.

No training data is required: base images are synthesized from the model
itself by input-space optimization (class-conditional model inversion).
"""
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


def _tv(x: torch.Tensor) -> torch.Tensor:
    """Total variation regularizer, keeps synthesized images piecewise smooth."""
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + \
           (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()


@torch.enable_grad()
def synthesize_images(model, cls: int, n: int = 32, steps: int = 300,
                      lr: float = 0.1, tv_weight: float = 1e-3,
                      device: str = "cpu", seed: int = 0) -> torch.Tensor:
    """Synthesize n images the model confidently classifies as `cls`."""
    g = torch.Generator(device=device).manual_seed(seed + cls)
    imgs = torch.rand(n, model.in_channels, 28, 28, generator=g,
                      device=device, requires_grad=True)
    opt = torch.optim.Adam([imgs], lr=lr)
    target = torch.full((n,), cls, dtype=torch.long, device=device)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(imgs), target) + tv_weight * _tv(imgs)
        loss.backward()
        opt.step()
        with torch.no_grad():
            imgs.clamp_(0, 1)
    return imgs.detach()


@dataclass
class ReversedTrigger:
    target: int
    mask: torch.Tensor      # (1, H, W) in [0, 1]
    pattern: torch.Tensor   # (C, H, W) in [0, 1]
    l1_norm: float
    asr: float              # attack success rate on synthesized base images

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, H, W) -> triggered copy."""
        return (1 - self.mask) * x + self.mask * self.pattern


@torch.enable_grad()
def reverse_engineer_trigger(model, target: int, base_images: torch.Tensor,
                             steps: int = 500, lr: float = 0.1,
                             lam: float = 1e-3, lam_max: float = 0.2,
                             adapt_every: int = 20, asr_target: float = 0.99,
                             device: str = "cpu") -> ReversedTrigger:
    """Optimize mask+pattern so that stamped images are classified as `target`,
    with an L1 penalty keeping the mask small.

    The penalty weight `lam` adapts (as in the original Neural Cleanse code):
    while the attack already succeeds, lam grows so the mask keeps shrinking;
    when the attack breaks, lam relaxes. Backdoored classes — where even a
    tiny mask flips the prediction — end up with a drastically smaller mask.
    """
    c, h, w = base_images.shape[1:]
    base_images = base_images.to(device)
    mask_raw = torch.full((1, h, w), -2.0, device=device, requires_grad=True)  # sigmoid ~ 0.12
    pattern_raw = torch.zeros(c, h, w, device=device, requires_grad=True)
    opt = torch.optim.Adam([mask_raw, pattern_raw], lr=lr)
    labels = torch.full((len(base_images),), target, dtype=torch.long, device=device)

    cur_lam = lam
    for step in range(steps):
        opt.zero_grad()
        m = torch.sigmoid(mask_raw)
        p = torch.sigmoid(pattern_raw)
        x = (1 - m) * base_images + m * p
        loss = F.cross_entropy(model(x), labels) + cur_lam * m.abs().sum()
        loss.backward()
        opt.step()
        if (step + 1) % adapt_every == 0:
            with torch.no_grad():
                asr = (model(x).argmax(1) == labels).float().mean().item()
            if asr >= asr_target:
                cur_lam = min(cur_lam * 1.5, lam_max)
            else:
                cur_lam = max(cur_lam / 1.5, lam)

    with torch.no_grad():
        m = torch.sigmoid(mask_raw)
        p = torch.sigmoid(pattern_raw)
        x = (1 - m) * base_images + m * p
        asr = (model(x).argmax(1) == labels).float().mean().item()
    return ReversedTrigger(target=target, mask=m.cpu(), pattern=p.cpu(),
                           l1_norm=m.abs().sum().item(), asr=asr)


@dataclass
class ScanResult:
    triggers: dict = field(default_factory=dict)   # target -> ReversedTrigger
    anomaly_index: dict = field(default_factory=dict)  # target -> float
    verdict_target: int | None = None              # flagged class, or None if clean

    @property
    def backdoored(self) -> bool:
        return self.verdict_target is not None


def mad_anomaly_indices(norms: np.ndarray) -> np.ndarray:
    """Median Absolute Deviation outlier score; > 2 is the classic threshold."""
    med = np.median(norms)
    mad = np.median(np.abs(norms - med))
    return np.abs(norms - med) / (1.4826 * mad + 1e-8)


def scan_model(model, num_classes: int | None = None,
               classes: list[int] | None = None,
               n_base: int = 32, synth_steps: int = 300,
               opt_steps: int = 500, threshold: float = 2.0,
               gap_ratio: float = 0.7, min_asr: float = 0.95,
               device: str = "cpu", seed: int = 0,
               verbose: bool = True) -> ScanResult:
    """Reverse-engineer a trigger per class and flag MAD outliers.

    `classes` optionally restricts the scan to a subset (coarse-to-fine
    strategy for models with many classes, e.g. face recognition).
    """
    model.eval()
    num_classes = num_classes or model.num_classes
    classes = classes if classes is not None else list(range(num_classes))

    result = ScanResult()
    # Shared pool of synthesized images, n_base per class. For each candidate
    # target t the trigger must flip images of the OTHER classes to t —
    # using class-t images as base would make the "attack" trivially succeed.
    pool = {}
    for cls in classes:
        pool[cls] = synthesize_images(model, cls, n=n_base, steps=synth_steps,
                                      device=device, seed=seed)
    for cls in classes:
        base = torch.cat([pool[c] for c in classes if c != cls])
        trig = reverse_engineer_trigger(model, cls, base, steps=opt_steps,
                                        device=device)
        result.triggers[cls] = trig
        if verbose:
            print(f"  class {cls:>3}: L1(mask)={trig.l1_norm:8.2f}  "
                  f"reversed-ASR={trig.asr:.3f}")

    norms = np.array([result.triggers[c].l1_norm for c in classes])
    idxs = mad_anomaly_indices(norms)
    for c, a in zip(classes, idxs):
        result.anomaly_index[c] = float(a)
    # Only flag outliers on the small side (a backdoor trigger is *smaller*
    # than any benign feature needed to flip a class).
    med = np.median(norms)
    for c, a in zip(classes, idxs):
        if a > threshold and result.triggers[c].l1_norm < med:
            result.verdict_target = c
            break
    # Secondary check: with few classes the MAD is noisy, so also flag a class
    # whose reversed trigger is drastically smaller than the rest of the
    # distribution (relative-gap rule), provided its attack actually works.
    if result.verdict_target is None and len(classes) >= 5:
        i_min = int(np.argmin(norms))
        rest = np.delete(norms, i_min)
        if (norms[i_min] < gap_ratio * np.median(rest)
                and result.triggers[classes[i_min]].asr >= min_asr):
            result.verdict_target = classes[i_min]
    return result
