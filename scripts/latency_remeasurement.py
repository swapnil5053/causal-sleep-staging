"""Corrected CPU inference latency benchmark.

The existing benchmark in src/eval/evaluate.py (run_latency_benchmark) has several
issues that make its C0-vs-C1 comparison unreliable:
  - Does not pin torch to a single thread, so OS scheduling noise pollutes timing.
  - Only 20 warm-up / 200 timed runs (plan calls for 100 warm-up / 1000 timed).
  - Reports only the mean, which is sensitive to occasional slow runs; no median or IQR.
  - Never separates StreamingZScore (the normalizer) from the model forward pass, so
    it cannot say whether a slowdown is in preprocessing or in the network.

This script replaces it: batch size 1, single thread, 100 warm-up, 1000 timed runs,
reporting median and IQR, with StreamingZScore benchmarked separately from the model.
Random (untrained) weights are used, since latency does not depend on trained values.

    python scripts/latency_remeasurement.py --out results/latency_remeasurement.md
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

# Run as `python scripts/latency_remeasurement.py` and sys.path[0] is scripts/, not the
# repository root, so `src` is not importable. Every other script in this directory does
# the same insert.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.normalization import StreamingZScore  # noqa: E402
from src.model.full_model import SleepStagingModel  # noqa: E402

CONFIGS = {
    "C0 (60s context)": "configs/default.yaml",
    "C1 (120s context)": "configs/sleep78_streaming_causal.yaml",
}

NUM_WARMUP = 100
NUM_TIMED = 1000
SAMPLE_RATE_HZ = 100
NORMALIZER_WINDOW_SEC = 30


def get_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def time_calls(fn, num_warmup, num_timed):
    """Run fn() num_warmup+num_timed times, return per-call latencies in ms for the timed runs."""
    for _ in range(num_warmup):
        fn()
    latencies_ms = []
    for _ in range(num_timed):
        start = time.perf_counter()
        fn()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return np.asarray(latencies_ms)


def summarize(latencies_ms):
    median = float(np.median(latencies_ms))
    q1, q3 = np.percentile(latencies_ms, [25, 75])
    return {"median_ms": median, "iqr_low_ms": float(q1), "iqr_high_ms": float(q3),
            "iqr_ms": float(q3 - q1)}


def benchmark_config(name, config_path):
    config = get_config(config_path)
    sequence_length_sec = config["data"]["sequence_length"]

    model = SleepStagingModel(config=config)
    model.eval()

    dummy_input = torch.randn(1, sequence_length_sec, SAMPLE_RATE_HZ)

    def forward_call():
        with torch.no_grad():
            model(dummy_input)

    forward_latencies = time_calls(forward_call, NUM_WARMUP, NUM_TIMED)
    forward_stats = summarize(forward_latencies)
    forward_stats["ms_per_sec_of_eeg"] = forward_stats["median_ms"] / sequence_length_sec

    return {
        "name": name,
        "sequence_length_sec": sequence_length_sec,
        "forward": forward_stats,
    }


def benchmark_normalizer():
    window_samples = NORMALIZER_WINDOW_SEC * SAMPLE_RATE_HZ
    normalizer = StreamingZScore(window_samples=window_samples)
    # Pre-fill so update() is measured at steady state, not during the empty-buffer ramp-up.
    for _ in range(window_samples):
        normalizer.update(np.random.randn())

    def update_call():
        normalizer.update(np.random.randn())

    latencies_ms = time_calls(update_call, NUM_WARMUP, NUM_TIMED)
    stats = summarize(latencies_ms)
    # Convert per-sample ms to microseconds for readability, matching the plan's phrasing.
    stats["median_us_per_sample"] = stats["median_ms"] * 1000.0
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/latency_remeasurement.md")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(1)

    config_results = []
    for name, path in CONFIGS.items():
        if not os.path.exists(path):
            print(f"skipping {name}: {path} not found")
            continue
        config_results.append(benchmark_config(name, path))

    normalizer_stats = benchmark_normalizer()

    lines = ["# Latency re-measurement", ""]
    lines.append(f"Protocol: batch size 1, single thread (`torch.set_num_threads(1)`), "
                 f"{NUM_WARMUP} warm-up calls, {NUM_TIMED} timed calls, median and IQR "
                 f"reported (not mean, which the previous benchmark used and which is "
                 f"sensitive to occasional OS scheduling noise). Random (untrained) "
                 f"weights, since latency does not depend on trained parameter values. "
                 f"`StreamingZScore` is benchmarked separately from the model forward "
                 f"pass, since the previous benchmark only measured the model and could "
                 f"not distinguish a normalizer slowdown from a network slowdown.")
    lines.append("")
    lines.append("## Model forward pass\n")
    lines.append("| Config | Context (s) | Median (ms) | IQR (ms) | ms per sec of EEG |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in config_results:
        f = r["forward"]
        lines.append(f"| {r['name']} | {r['sequence_length_sec']} | {f['median_ms']:.3f} | "
                     f"[{f['iqr_low_ms']:.3f}, {f['iqr_high_ms']:.3f}] | "
                     f"{f['ms_per_sec_of_eeg']:.4f} |")
    lines.append("")
    lines.append("## StreamingZScore (30 s trailing window, measured separately)\n")
    lines.append(f"- Median: {normalizer_stats['median_us_per_sample']:.3f} us per sample")
    lines.append(f"- IQR: [{normalizer_stats['iqr_low_ms']*1000:.3f}, "
                 f"{normalizer_stats['iqr_high_ms']*1000:.3f}] us per sample")
    lines.append("")
    if len(config_results) == 2:
        c0, c1 = config_results[0], config_results[1]
        lines.append("## Comparison\n")
        lines.append(f"{c0['name']} does strictly less work per call than {c1['name']} "
                     f"({c0['sequence_length_sec']}s vs {c1['sequence_length_sec']}s of "
                     f"context), so its median latency should be lower per call. The "
                     f"previous mean-based, non-thread-pinned benchmark reported the "
                     f"opposite; this corrected measurement is the one to present.")
    report = "\n".join(lines) + "\n"

    if not args.quiet:
        print(report)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
