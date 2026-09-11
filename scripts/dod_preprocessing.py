"""Preprocess DOD-H into the same processed format the Sleep-EDF pipeline produces.

The point of this script is that the evaluation-protocol result should be reproducible on a
second dataset, and reproducing it requires only that the two arms be trained and scored
identically on that dataset. Nothing about the architecture, the training recipe or the two
protocols changes; only the source of the EEG does.

Three decisions are forced by the data and each is recorded in the manifest:

  Channel. DOD-H carries twelve derivations and DOD-O carries eight, and the only frontal
  derivations present in both cohorts are F3-M2, F3-F4, F4-O2 and F3-O1. F3-M2 is used, so
  that the same channel choice extends to DOD-O without a second decision. Note this is not
  Fpz-Cz: a model trained on Sleep-EDF is not expected to transfer to it without adaptation,
  and cross-dataset transfer is not what this script is for.

  Sampling rate. DOD is recorded at 250 Hz and the MRCNN kernels are defined in samples
  (50 = 0.5 s, 400 = 4 s) on the assumption of 100 Hz. The signal is resampled to 100 Hz
  rather than the kernels retuned, so the architecture is unchanged between datasets.

  Wake trimming. Sleep-EDF cassette recordings run about 20 hours with Wake at 68%, which is
  why that pipeline trims. DOD records are 5-10 hours with Wake at 12%, so trimming is not
  applied and `wake_trim_minutes` is recorded as absent. Applying the Sleep-EDF setting here
  would remove real sleep-adjacent Wake rather than daytime recording.

Amplitude units differ too (mV here, uV in Sleep-EDF), which the trailing z-score removes:
the normalizer divides by a trailing standard deviation, so the pipeline is scale-invariant
and no unit conversion is needed. That is worth knowing rather than assuming, so the script
checks it.

Usage:
    python scripts/dod_preprocessing.py --h5_dir data/dod/dodh \
        --processed_dir data/processed_dodh --all
    python scripts/dod_preprocessing.py --h5_dir data/dod/dodh --limit 3   # validate first
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import h5py
import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.normalization import normalize_signal  # noqa: E402

STAGE_NAMES = ("W", "N1", "N2", "N3", "REM")
SOURCE_RATE = 250
TARGET_RATE = 100
EPOCH_SECONDS = 30


def load_record(path, channel):
    """Return (signal at TARGET_RATE, per-epoch labels) for one DOD record."""
    with h5py.File(path, "r") as h:
        key = f"signals/eeg/{channel}"
        if key not in h:
            available = sorted(h["signals/eeg"].keys())
            raise KeyError(f"{os.path.basename(path)} has no {channel}; has {available}")
        raw = h[key][:].astype(np.float64)
        labels = h["hypnogram"][:].astype(np.int64)

    # 250 -> 100 Hz. resample_poly applies an anti-alias filter, which decimation alone
    # would not; 2/5 is exact so no resampling artefacts from a fractional ratio.
    signal = resample_poly(raw, TARGET_RATE, SOURCE_RATE).astype(np.float32)

    expected = len(labels) * EPOCH_SECONDS * TARGET_RATE
    if len(signal) < expected:
        raise ValueError(f"{os.path.basename(path)}: signal shorter than hypnogram "
                         f"({len(signal)} < {expected})")
    return signal[:expected], labels


def build_subject(signal, labels, norm_method, norm_window, norm_eps):
    """Normalize, drop unscored epochs, and return (x, y, segment_starts).

    Order matters and follows the Sleep-EDF pipeline exactly. The whole recording is
    normalized first, as a continuous signal, and only then are unscored epochs removed.
    Normalizing each surviving stretch separately would reset the trailing statistic at every
    gap, which a device would not do: the signal is continuous in time whether or not a
    scorer labelled it. Getting this backwards would make the two datasets incomparable.
    """
    spe = EPOCH_SECONDS * TARGET_RATE
    keep = labels >= 0
    if not keep.any():
        raise ValueError("no scored epochs")

    window_samples = int(round(norm_window * TARGET_RATE))
    normalized = normalize_signal(signal, method=norm_method,
                                  window_samples=window_samples, eps=norm_eps)
    per_second = np.asarray(normalized, dtype=np.float32).reshape(-1, TARGET_RATE)

    # Keep the seconds belonging to scored epochs, and record where each contiguous stretch
    # begins in the OUTPUT index space, so a window can be refused if it spans a dropped gap.
    xs, ys, starts, written = [], [], [], 0
    run_start = None
    for i in range(len(labels) + 1):
        scored = i < len(labels) and keep[i]
        if scored and run_start is None:
            run_start = i
        elif not scored and run_start is not None:
            xs.append(per_second[run_start * EPOCH_SECONDS:i * EPOCH_SECONDS])
            ys.append(np.repeat(labels[run_start:i], EPOCH_SECONDS))
            starts.append(written)
            written += (i - run_start) * EPOCH_SECONDS
            run_start = None

    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    assert len(x) == len(y) == written, (len(x), len(y), written)
    return x, y, np.array(sorted(set(starts)), dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5_dir", default="data/dod/dodh")
    ap.add_argument("--processed_dir", default="data/processed_dodh")
    ap.add_argument("--channel", default="F3_M2")
    ap.add_argument("--normalization", default="causal_rolling")
    ap.add_argument("--window_seconds", type=int, default=30)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N records, to validate before committing")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.h5_dir) if f.endswith(".h5"))
    if not files:
        sys.exit(f"no .h5 files in {args.h5_dir}")
    if args.limit:
        files = files[:args.limit]
    elif not args.all:
        sys.exit("pass --all to process every record, or --limit N to validate first")

    os.makedirs(args.processed_dir, exist_ok=True)
    print(f"{len(files)} records | channel {args.channel} | "
          f"{SOURCE_RATE} -> {TARGET_RATE} Hz | normalization {args.normalization}"
          f"@{args.window_seconds}s | no wake trimming\n")

    manifest_subjects, total = {}, np.zeros(5, dtype=np.int64)
    for idx, fname in enumerate(files):
        sub_id = f"{idx:02d}"
        signal, labels = load_record(os.path.join(args.h5_dir, fname), args.channel)
        x, y, starts = build_subject(signal, labels, args.normalization,
                                     args.window_seconds, args.eps)

        # The normalizer should have removed the amplitude-unit difference against Sleep-EDF.
        # Check rather than assume: a scale left in would silently shift the input distribution.
        sd = float(np.std(x))
        if not 0.5 < sd < 2.0:
            print(f"  WARNING: subject {sub_id} normalized s.d. {sd:.3f} is far from 1; "
                  f"check the channel and units before training on this.")

        np.savez_compressed(
            os.path.join(args.processed_dir, f"subject_{sub_id}.npz"),
            x=x, y=y,
            normalization_method=np.array(args.normalization),
            normalization_window_seconds=np.array(args.window_seconds),
            normalization_eps=np.array(args.eps),
            wake_trim_minutes=np.array(-1),
            resample_rate=np.array(TARGET_RATE),
            segment_starts=starts,
        )
        counts = np.bincount(y, minlength=5)
        total += counts
        manifest_subjects[sub_id] = {
            "source_file": fname, "seconds": int(len(x)),
            "segments": int(len(starts)), "normalized_sd": round(sd, 4),
            "class_counts": {n: int(c) for n, c in zip(STAGE_NAMES, counts)},
        }
        print(f"  subject_{sub_id}  {len(x):>8,} s  {len(starts)} segment(s)  "
              f"sd {sd:.3f}  <- {fname}")

    n = int(total.sum())
    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "DOD-H", "h5_dir": args.h5_dir, "processed_dir": args.processed_dir,
        "target_channel": args.channel, "source_rate": SOURCE_RATE,
        "resample_rate": TARGET_RATE, "wake_trim_minutes": None,
        "normalization": {"method": args.normalization,
                          "window_seconds": args.window_seconds, "eps": args.eps},
        "end_to_end_causal_preprocessing": args.normalization == "causal_rolling",
        "n_records": len(files), "n_subjects_written": len(manifest_subjects),
        "total_seconds": n,
        "class_distribution": {name: {"seconds": int(c), "fraction": round(float(c) / n, 6)}
                               for name, c in zip(STAGE_NAMES, total)},
        "subjects": manifest_subjects,
    }
    with open(os.path.join(args.processed_dir, "preprocessing_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{len(manifest_subjects)} subjects, {n / 3600:.1f} h total")
    print("class balance: " + "  ".join(
        f"{name} {int(c) / n:.1%}" for name, c in zip(STAGE_NAMES, total)))
    print(f"manifest: {os.path.join(args.processed_dir, 'preprocessing_manifest.json')}")
    print("\nDOD-H is far more N2-heavy and far less Wake-heavy than wake-trimmed Sleep-EDF, "
          "so the focal-loss class weights should be recomputed for it rather than reused.")


if __name__ == "__main__":
    main()
