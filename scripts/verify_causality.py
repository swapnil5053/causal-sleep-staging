"""Verify that the complete pipeline is causal, and write a citable report.

The claim the paper makes is about the whole path, not the network alone: a raw EEG
sample arriving at time t+k must be unable to change any output at or before t, through
normalization and through the model. This script checks that empirically on the real
code, on trained weights when they are available, and writes the outcome to a markdown
file that can be referenced from the paper and re-run by a reviewer.

    python scripts/verify_causality.py
    python scripts/verify_causality.py --config configs/sleep78_streaming_causal.yaml \
        --checkpoint checkpoints_78streaming_causal_s42/best_model_fold_0.pth
    python scripts/verify_causality.py --subject data/processed78_streaming/subject_00.npz

Exit status is 0 when every check passes and 1 otherwise, so it can gate a run.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.normalization import StreamingZScore, causal_rolling_zscore  # noqa: E402
from src.model.full_model import SleepStagingModel  # noqa: E402


class Check:
    def __init__(self, name, passed, detail):
        self.name = name
        self.passed = bool(passed)
        self.detail = detail

    @property
    def status(self):
        return "PASS" if self.passed else "FAIL"


def build_model(config, checkpoint_path, causal):
    cfg = {"model": dict(config["model"])}
    cfg["model"]["causal"] = causal
    model = SleepStagingModel(config=cfg)
    loaded = "randomly initialised weights"
    if checkpoint_path and os.path.exists(checkpoint_path) and causal == config["model"]["causal"]:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        loaded = f"trained weights from {checkpoint_path} (epoch {state.get('epoch', '?')})"
    model.eval()
    return model, loaded


def forward(model, normalized, sample_rate):
    seconds = len(normalized) // sample_rate
    windows = normalized[: seconds * sample_rate].reshape(seconds, sample_rate)
    tensor = torch.from_numpy(windows).float().unsqueeze(0)
    with torch.no_grad():
        return model(tensor)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/sleep78_streaming_causal.yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="optional trained checkpoint; a random model is used otherwise")
    ap.add_argument("--subject", default=None,
                    help="optional processed .npz whose labels/length shape the synthetic test")
    ap.add_argument("--seconds", type=int, default=240, help="length of the probe signal")
    ap.add_argument("--split", type=int, default=120,
                    help="second at which the future perturbation begins")
    ap.add_argument("--out", default="results/causality_verification.md")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    sample_rate = int(config["data"]["resample_rate"])
    norm_cfg = config["data"].get("normalization", {"method": "epoch_zscore"})
    method = norm_cfg.get("method", "causal_rolling")
    window_seconds = float(norm_cfg.get("window_seconds", 30))
    window_samples = int(round(window_seconds * sample_rate))
    eps = float(norm_cfg.get("eps", 1e-8))

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    n_samples = args.seconds * sample_rate
    cut = args.split * sample_rate
    if not 0 < cut < n_samples:
        raise SystemExit("--split must fall strictly inside --seconds")

    signal = rng.normal(scale=2e-5, size=n_samples)
    signal += 1e-5 * np.sin(2 * np.pi * np.arange(n_samples) / (60.0 * sample_rate))

    perturbed = signal.copy()
    perturbed[cut:] = perturbed[cut:] * 50.0 + 1e-3

    checks = []

    # ---------------------------------------------------------------- preprocessing
    if method == "causal_rolling":
        a = causal_rolling_zscore(signal, window_samples=window_samples, eps=eps)
        b = causal_rolling_zscore(perturbed, window_samples=window_samples, eps=eps)
        identical = np.array_equal(a[:cut], b[:cut])
        checks.append(Check(
            "Normalization ignores future samples",
            identical,
            f"max |difference| before sample {cut:,} = {np.abs(a[:cut] - b[:cut]).max():.3e}"))

        online = StreamingZScore(window_samples=window_samples, eps=eps).process(signal)
        delta = float(np.abs(a - online).max())
        checks.append(Check(
            "Offline arrays equal the online, sample-at-a-time filter",
            delta < 1e-9,
            f"max |offline - streaming| = {delta:.3e} over {n_samples:,} samples"))

        checks.append(Check(
            "Normalized output is finite through the cold start",
            bool(np.isfinite(a).all()) and float(np.abs(a).max()) <= np.sqrt(window_samples - 1) + 1e-9,
            f"max |z| = {np.abs(a).max():.3f}, bound sqrt(window-1) = {np.sqrt(window_samples - 1):.3f}"))
    else:
        a = np.zeros(n_samples)
        b = np.zeros(n_samples)
        checks.append(Check(
            "Normalization ignores future samples",
            False,
            f"config uses '{method}', which reads the whole 30 s epoch. "
            f"The pipeline is NOT end-to-end causal under this configuration."))

    # ------------------------------------------------------------------ whole path
    # Guard against citing a PASS obtained from the ablation arm: this script certifies the
    # causal configuration, so a config with `causal: false` is reported as out of scope.
    configured_causal = bool(config["model"]["causal"])
    checks.append(Check(
        "Configuration declares a causal model",
        configured_causal,
        "`causal: true`" if configured_causal else
        "`causal: false` - this is the non-causal ablation arm. End-to-end causality is not "
        "claimed for it; run this script against the causal config instead."))

    model, weights_note = build_model(config, args.checkpoint, causal=True)
    out_a = forward(model, a, sample_rate)
    out_b = forward(model, b, sample_rate)
    before = torch.equal(out_a[:, : args.split, :], out_b[:, : args.split, :])
    after_differs = not torch.equal(out_a[:, args.split:, :], out_b[:, args.split:, :])
    max_before = float((out_a[:, : args.split, :] - out_b[:, : args.split, :]).abs().max())

    checks.append(Check(
        "Raw sample -> normalization -> model: no future leakage",
        before,
        f"max |logit difference| for seconds 0-{args.split - 1} = {max_before:.3e} "
        f"(bit-identical required)"))
    checks.append(Check(
        "The probe actually perturbs later outputs (test is not vacuous)",
        after_differs,
        f"seconds {args.split}+ differ as expected"))

    # -------------------------------------------------------------------- control
    control, _ = build_model(config, None, causal=False)
    c_a = forward(control, a, sample_rate)
    c_b = forward(control, b, sample_rate)
    leaks = not torch.equal(c_a[:, : args.split, :], c_b[:, : args.split, :])
    checks.append(Check(
        "Non-causal control does leak (so the ablation is a real contrast)",
        leaks,
        "removing the masks and left-padding lets future signal reach earlier outputs"))

    # ------------------------------------------------------------ optional real data
    subject_note = "not used"
    if args.subject and os.path.exists(args.subject):
        data = np.load(args.subject)
        stored_method = str(data["normalization_method"]) if "normalization_method" in data.files else "unknown (legacy file)"
        finite = bool(np.isfinite(data["x"]).all())
        checks.append(Check(
            "Processed subject file is finite and carries preprocessing metadata",
            finite and stored_method != "unknown (legacy file)",
            f"{os.path.basename(args.subject)}: {len(data['x']):,} seconds, "
            f"normalization='{stored_method}', dtype={data['x'].dtype}"))
        subject_note = f"{args.subject} (normalization='{stored_method}')"

    # ---------------------------------------------------------------------- report
    all_passed = all(c.passed for c in checks)
    lines = []
    w = lines.append
    w("# Causality verification\n")
    w("Generated by `python scripts/verify_causality.py`. Each check perturbs the input at a")
    w("future time and requires every earlier value to be **bit-identical**, not merely close.\n")
    w(f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    w(f"- Config: `{args.config}`")
    w(f"- Normalization: `{method}`, trailing window {window_seconds:g} s")
    w(f"- Model weights: {weights_note}")
    w(f"- Probe: {args.seconds} s of synthetic EEG at {sample_rate} Hz, "
      f"perturbed from second {args.split}")
    w(f"- Processed subject checked: {subject_note}\n")
    w("| Check | Result | Detail |")
    w("|---|---|---|")
    for c in checks:
        w(f"| {c.name} | **{c.status}** | {c.detail} |")
    w("")
    if all_passed:
        w("**All checks passed.** For this configuration the prediction at second *t* is a")
        w("function of the raw signal up to second *t* only, through both preprocessing and the")
        w("network, and the offline arrays used for training are exactly what a sample-at-a-time")
        w("streaming implementation produces.")
    else:
        w("**At least one check failed.** Do not describe this configuration as end-to-end")
        w("causal until the failing rows above are resolved.")
    w("")
    text = "\n".join(lines) + "\n"

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    open(args.out, "w").write(text)

    print(text)
    print(f"written to {args.out}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
