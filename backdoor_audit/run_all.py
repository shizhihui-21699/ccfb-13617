"""End-to-end CLI.

  demo  — train a clean and a backdoored model, scan both, mitigate the
          backdoored one, and report accuracy/ASR before and after.
  scan  — audit an existing checkpoint: reverse-engineer triggers and
            decide whether the model is backdoored (no training data needed).
"""
import argparse
import json
import os

import torch
from torchvision.utils import save_image

from .data import PatchTrigger, get_mnist_loaders
from .evaluate import accuracy, attack_success_rate
from .mitigate import unlearn_trigger, unlearn_with_clean_data
from .models import load_model
from .neural_cleanse import scan_model
from .train import train_model
from .models import save_checkpoint

DEVICE = "cpu"


def _print_scan(name, result):
    print(f"\n[scan] {name}")
    for c in sorted(result.triggers):
        t = result.triggers[c]
        a = result.anomaly_index[c]
        flag = "  <-- OUTLIER" if result.verdict_target == c else ""
        print(f"  class {c:>3}: L1={t.l1_norm:8.2f}  anomaly_index={a:5.2f}{flag}")
    if result.backdoored:
        print(f"  VERDICT: BACKDOORED, target class = {result.verdict_target}")
    else:
        print("  VERDICT: no backdoor detected")


def _save_trigger_fig(result, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for c, t in result.triggers.items():
        # visualize mask * pattern (the effective trigger stamp)
        stamp = (t.mask * t.pattern).clamp(0, 1)
        save_image(stamp, os.path.join(out_dir, f"{name}_trigger_class{c}.png"))


def cmd_scan(args):
    model = load_model(args.model, DEVICE)
    classes = (list(range(args.classes_start, args.classes_end))
               if args.classes_end is not None else None)
    result = scan_model(model, classes=classes, n_base=args.n_base,
                        synth_steps=args.synth_steps, opt_steps=args.opt_steps,
                        device=DEVICE)
    _print_scan(args.model, result)
    _save_trigger_fig(result, args.out_dir, "scan")
    report = {
        "model": args.model,
        "backdoored": result.backdoored,
        "target": result.verdict_target,
        "l1_norms": {c: t.l1_norm for c, t in result.triggers.items()},
        "anomaly_index": result.anomaly_index,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "scan_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[scan] report + trigger images -> {args.out_dir}/")


def _cached_scan(model, name, args):
    """Scan results are deterministic (fixed seeds) and expensive — cache them."""
    path = f"reports/scan_{name}.pt"
    if os.path.exists(path) and not args.rescan:
        print(f"[scan] {name}: loaded cached result from {path}")
        return torch.load(path, weights_only=False)
    res = scan_model(model, device=DEVICE, n_base=args.n_base,
                     synth_steps=args.synth_steps, opt_steps=args.opt_steps)
    os.makedirs("reports", exist_ok=True)
    torch.save(res, path)
    return res


def cmd_demo(args):
    os.makedirs("checkpoints", exist_ok=True)
    target = args.target
    real_trigger = PatchTrigger(size=4, pos=(24, 24), value=1.0)

    # 1. control models -----------------------------------------------------
    if os.path.exists(args.clean_ckpt) and os.path.exists(args.bd_ckpt) and not args.retrain:
        print("[demo] loading existing checkpoints")
        clean = load_model(args.clean_ckpt, DEVICE)
        backdoored = load_model(args.bd_ckpt, DEVICE)
    else:
        print("[demo] training clean model")
        clean = train_model(poison=False, epochs=args.epochs)
        save_checkpoint(clean, args.clean_ckpt)
        print(f"[demo] training backdoored model (target={target})")
        backdoored = train_model(poison=True, epochs=args.epochs, target=target)
        save_checkpoint(backdoored, args.bd_ckpt)

    _, test_set, _, test_loader = get_mnist_loaders()

    print("\n=== baseline metrics ===")
    print(f"clean      : acc={accuracy(clean, test_loader, DEVICE):.4f}")
    bd_acc = accuracy(backdoored, test_loader, DEVICE)
    bd_asr = attack_success_rate(backdoored, test_set, real_trigger, target, DEVICE)
    print(f"backdoored : acc={bd_acc:.4f}  ASR(real trigger)={bd_asr:.4f}")

    # 2. scan both models (cached: the scan is the expensive part) -----------
    print("\n=== Neural Cleanse scan (model-only, no training data) ===")
    res_clean = _cached_scan(clean, "clean", args)
    _print_scan("clean model", res_clean)
    res_bd = _cached_scan(backdoored, "backdoored", args)
    _print_scan("backdoored model", res_bd)
    _save_trigger_fig(res_bd, "reports", "backdoored")
    _save_trigger_fig(res_clean, "reports", "clean")

    # 3. mitigate ------------------------------------------------------------
    if not res_bd.backdoored:
        print("\n[demo] no backdoor found in backdoored model; skipping mitigation")
        return
    reversed_trig = res_bd.triggers[res_bd.verdict_target]

    def rev_apply(x):
        return ((1 - reversed_trig.mask) * x + reversed_trig.mask * reversed_trig.pattern).clamp(0, 1)

    print("\n=== mitigation A: model-only unlearning (synthesized data) ===")
    fixed_a, hist = unlearn_trigger(backdoored, reversed_trig,
                                    num_classes=backdoored.num_classes,
                                    device=DEVICE)
    for step, loss in hist:
        print(f"  ft step {step:>4}: loss={loss:.4f}")
    a_acc = accuracy(fixed_a, test_loader, DEVICE)
    a_asr = attack_success_rate(fixed_a, test_set, real_trigger, target, DEVICE)
    a_asr_rev = attack_success_rate(fixed_a, test_set, rev_apply, target, DEVICE)

    print("\n=== mitigation B: unlearning with a small held-out clean set ===")
    print("  (2000 clean test samples, 30% stamped with reversed trigger, true labels)")
    fixed_b, hist = unlearn_with_clean_data(backdoored, reversed_trig, test_set,
                                            frac=0.3, epochs=5, lr=1e-3,
                                            max_samples=2000, device=DEVICE)
    b_acc = accuracy(fixed_b, test_loader, DEVICE)
    b_asr = attack_success_rate(fixed_b, test_set, real_trigger, target, DEVICE)
    b_asr_rev = attack_success_rate(fixed_b, test_set, rev_apply, target, DEVICE)

    print("\n=== summary ===")
    print(f"{'model':<28}{'clean acc':>12}{'ASR(real)':>12}{'ASR(reversed)':>15}")
    print(f"{'backdoored (before)':<28}{bd_acc:>12.4f}{bd_asr:>12.4f}{'':>15}")
    print(f"{'A: unlearn (model-only)':<28}{a_acc:>12.4f}{a_asr:>12.4f}{a_asr_rev:>15.4f}")
    print(f"{'B: unlearn (+clean data)':<28}{b_acc:>12.4f}{b_asr:>12.4f}{b_asr_rev:>15.4f}")
    print(f"clean-model reference acc: {accuracy(clean, test_loader, DEVICE):.4f}")
    save_checkpoint(fixed_a, "checkpoints/backdoored_mitigated_model_only.pt")
    save_checkpoint(fixed_b, "checkpoints/backdoored_mitigated.pt")
    print("[demo] mitigated models -> checkpoints/backdoored_mitigated*.pt")
    print("[demo] reversed trigger images -> reports/")


def main():
    p = argparse.ArgumentParser(prog="backdoor_audit")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="full control experiment")
    d.add_argument("--epochs", type=int, default=3)
    d.add_argument("--target", type=int, default=0)
    d.add_argument("--n-base", type=int, default=32)
    d.add_argument("--synth-steps", type=int, default=300)
    d.add_argument("--opt-steps", type=int, default=500)
    d.add_argument("--retrain", action="store_true")
    d.add_argument("--rescan", action="store_true", help="ignore cached scan results")
    d.add_argument("--clean-ckpt", default="checkpoints/clean.pt")
    d.add_argument("--bd-ckpt", default="checkpoints/backdoored.pt")
    d.set_defaults(fn=cmd_demo)

    s = sub.add_parser("scan", help="audit an existing checkpoint")
    s.add_argument("--model", required=True)
    s.add_argument("--n-base", type=int, default=32)
    s.add_argument("--synth-steps", type=int, default=300)
    s.add_argument("--opt-steps", type=int, default=500)
    s.add_argument("--classes-start", type=int, default=0)
    s.add_argument("--classes-end", type=int, default=None,
                   help="scan classes [start, end) only (coarse-to-fine)")
    s.add_argument("--out-dir", default="reports")
    s.set_defaults(fn=cmd_scan)

    args = p.parse_args()
    torch.manual_seed(0)
    args.fn(args)


if __name__ == "__main__":
    main()
