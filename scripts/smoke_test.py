"""Fast sanity check for the training/eval stack.

Runs the real model, losses, optimizer, checkpoint round-trip and metric code on a
few random batches so device/shape errors surface in seconds instead of after a
full epoch. Documented in REPRODUCIBILITY.md as the fast pre-run check.

    python scripts/smoke_test.py
"""
import os
import sys
import tempfile

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.full_model import SleepStagingModel  # noqa: E402
from src.train.losses import FocalLoss, WeightedCrossEntropyLoss  # noqa: E402

FAILURES = []


def check(name, fn):
    try:
        detail = fn()
        print(f"  PASS  {name}" + (f"   {detail}" if detail else ""))
    except Exception as exc:
        FAILURES.append(name)
        print(f"  FAIL  {name}")
        print(f"        {type(exc).__name__}: {exc}")


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    print(f"config: {cfg_path}")
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    print(f"device: {device}  ({gpu})")

    L = cfg["data"]["sequence_length"]
    B = 8
    state = {}

    model = SleepStagingModel(config=cfg).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}  (budget 48,000)")
    print(f"sequence_length: {L}s   lr: {cfg['train']['lr']}   weight_decay: {cfg['train']['weight_decay']}")
    print()

    x = torch.randn(B, L, 100, device=device)
    y = torch.randint(0, 5, (B, L), device=device)

    def forward():
        state["logits"] = model(x)
        got = tuple(state["logits"].shape)
        assert got == (B, L, 5), f"expected {(B, L, 5)}, got {got}"
        return f"output {got}"

    def focal():
        alpha = np.array([0.018, 0.492, 0.077, 0.232, 0.181])
        crit = FocalLoss(gamma=cfg["train"]["focal_gamma"], alpha=alpha).to(device)
        state["loss"] = crit(state["logits"], y)
        assert torch.isfinite(state["loss"]), "loss is not finite"
        return f"loss {state['loss'].item():.4f}"

    def backward():
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
        )
        opt.zero_grad()
        state["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        return "gradients flowed, step taken"

    def weighted_ce():
        crit = WeightedCrossEntropyLoss(weights=np.array([0.02, 0.5, 0.08, 0.23, 0.18])).to(device)
        loss = crit(model(x), y)
        assert torch.isfinite(loss), "loss is not finite"
        return f"loss {loss.item():.4f}"

    def checkpoint():
        path = os.path.join(tempfile.gettempdir(), "_smoke_ckpt.pth")
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": torch.optim.AdamW(model.parameters()).state_dict(),
                "val_f1": 0.5,
                "config": cfg,
            },
            path,
        )
        ck = torch.load(path, map_location=device, weights_only=False)
        fresh = SleepStagingModel(config=cfg)
        fresh.load_state_dict(ck["model_state_dict"])
        os.remove(path)
        return "save -> load -> load_state_dict"

    def eval_batch1():
        # evaluate.py uses batch_size=1; make sure BatchNorm is happy in eval mode
        model.eval()
        with torch.no_grad():
            out = model(x[:1])
        model.train()
        assert tuple(out.shape) == (1, L, 5), out.shape
        return f"batch-size-1 output {tuple(out.shape)}"

    def metrics():
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(x), dim=-1)
        model.train()
        yt = y.cpu().numpy().flatten()
        yp = preds.cpu().numpy().flatten()
        acc = accuracy_score(yt, yp)
        kappa = cohen_kappa_score(yt, yp)
        macro = f1_score(yt, yp, average="macro", zero_division=0)
        per_class = f1_score(yt, yp, average=None, zero_division=0, labels=[0, 1, 2, 3, 4])
        assert len(per_class) == 5, f"expected 5 per-class scores, got {len(per_class)}"
        return f"acc={acc:.3f} kappa={kappa:.3f} macroF1={macro:.3f}"

    check("model forward", forward)
    check("focal loss (the one that crashed)", focal)
    check("backward + optimizer step", backward)
    check("weighted cross-entropy loss", weighted_ce)
    check("checkpoint round-trip", checkpoint)
    check("eval mode, batch size 1", eval_batch1)
    check("metrics path", metrics)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {', '.join(FAILURES)}")
        print("Do NOT start the long run yet.")
        return 1
    print("ALL CHECKS PASSED - safe to start the full training run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
