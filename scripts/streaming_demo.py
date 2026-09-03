"""Stage one recording the way a device would: one raw sample in, one label per second out.

This is the deployment claim made executable. Raw EEG samples are pushed through
``StreamingZScore`` one at a time, buffered into one-second frames, and classified, with
nothing downstream of the current sample ever consulted. Nothing here batches a night,
looks ahead, or re-reads a sample it has already consumed.

    # no data required: synthesises a recording with a stage transition
    python scripts/streaming_demo.py --synthetic --verify

    # a processed subject (the normalizer already ran offline; see the note it prints)
    python scripts/streaming_demo.py --subject data/processed78_streaming/subject_00.npz \
        --checkpoint checkpoints_78streaming_causal_s42/best_model_fold_0.pth \
        --start 3600 --minutes 6 --verify

    # the whole path, from the raw EDF
    python scripts/streaming_demo.py --edf data/raw/SC4001E0-PSG.edf \
        --hypnogram data/raw/SC4001EC-Hypnogram.edf \
        --checkpoint checkpoints_78streaming_causal_s42/best_model_fold_0.pth

Two emission modes:

``--mode window`` (default)
    Accumulate ``sequence_length`` seconds, run the model once, emit that many labels.
    This is exactly what ``src/eval/evaluate.py`` does, so ``--verify`` can prove the
    streaming path reproduces the archived evaluation predictions sample for sample.

``--mode rolling``
    Emit a label every second from the trailing ``sequence_length`` seconds. This is how a
    wearable would actually run. It does *not* match ``evaluate.py``, and the difference is
    the window-boundary artefact named on the limitations slide: under ``window`` the model
    restarts with no context every ``sequence_length`` seconds, under ``rolling`` it never
    does.

Without ``--checkpoint`` the model runs on randomly initialised weights. The labels are then
meaningless but every other property demonstrated here - causality, latency, and the
equivalence checked by ``--verify`` - is independent of the weights and still holds.

Exit status is 0 when the run (and ``--verify``, if requested) succeeds, 1 otherwise.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.normalization import StreamingZScore  # noqa: E402
from src.model.full_model import SleepStagingModel  # noqa: E402

STAGES = ["W", "N1", "N2", "N3", "REM"]


def hms(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


# --------------------------------------------------------------------------- recordings

def synthetic_recording(sample_rate, minutes=10, seed=0):
    """A raw-scale recording with one stage transition, so the demo runs with no data.

    Amplitudes and rhythms are only loosely EEG-like; this exists to exercise the path,
    not to be staged correctly. The returned signal is in volts, matching what MNE hands
    back from a Sleep-EDF EDF.
    """
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * sample_rate)
    t = np.arange(n) / sample_rate
    half = n // 2

    signal = rng.normal(scale=8e-6, size=n)
    # first half: deeper sleep, slow high-amplitude activity plus periodic spindle bursts
    signal[:half] += 3.0e-5 * np.sin(2 * np.pi * 1.2 * t[:half])
    spindles = ((t[:half] % 20.0) < 1.0)
    signal[:half][spindles] += 2.0e-5 * np.sin(2 * np.pi * 13.0 * t[:half][spindles])
    # second half: lighter, faster, lower amplitude
    signal[half:] += 8.0e-6 * np.sin(2 * np.pi * 7.5 * t[half:])

    labels = np.empty(n // sample_rate, dtype=np.int64)
    labels[: half // sample_rate] = 2      # N2
    labels[half // sample_rate:] = 4       # REM
    return signal, labels


def load_edf(edf_path, hypnogram_path, target_channel, sample_rate):
    """Load a raw recording exactly as ``src/data/preprocessing`` does, without normalizing."""
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(f"--edf needs MNE: {exc}. pip install -r requirements.txt")

    from src.data.preprocessing import find_eeg_channel

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    channel = find_eeg_channel(raw.ch_names, target_channel)
    raw.pick([channel]) if hasattr(raw, "pick") else raw.pick_channels([channel])
    if raw.info["sfreq"] != sample_rate:
        raw.resample(sample_rate, verbose=False)
    signal = raw.get_data()[0]

    labels = None
    if hypnogram_path:
        annotations = mne.read_annotations(hypnogram_path)
        labels = np.full(len(signal) // sample_rate, -1, dtype=np.int64)
        mapping = [("Sleep stage W", 0), ("Sleep stage 1", 1), ("Sleep stage 2", 2),
                   ("Sleep stage 3", 3), ("Sleep stage 4", 3), ("Sleep stage R", 4)]
        for annotation in annotations:
            stage = next((v for k, v in mapping if k in annotation["description"]), -1)
            if stage != -1:
                start = max(0, int(annotation["onset"]))
                end = min(len(labels), int(annotation["onset"]) + int(annotation["duration"]))
                labels[start:end] = stage
    return signal, labels, channel


# ------------------------------------------------------------------------- the streamer

class StreamingSleepStager:
    """Sample-at-a-time front end for :class:`SleepStagingModel`.

    Holds a trailing normalizer, the current partial second, and at most ``context``
    seconds of normalized history. No method on this class can read a sample it has not
    already been handed.
    """

    def __init__(self, model, sample_rate, context, window_samples, eps=1e-8):
        self.model = model
        self.sample_rate = int(sample_rate)
        self.context = int(context)
        self.normalizer = StreamingZScore(window_samples=window_samples, eps=eps)
        self.reset()

    def reset(self):
        self.normalizer.reset()
        self._partial = []
        self._history = deque(maxlen=self.context)
        self.seconds_seen = 0

    def push_sample(self, value):
        """Normalize one raw sample. Returns the completed second, or None."""
        self._partial.append(self.normalizer.update(value))
        if len(self._partial) < self.sample_rate:
            return None
        # float32 at exactly the point the offline pipeline casts, so the streaming and
        # archived arrays are the same numbers and not merely close ones.
        second = np.asarray(self._partial, dtype=np.float32)
        self._partial = []
        self._history.append(second)
        self.seconds_seen += 1
        return second

    def push_second(self, second):
        """Accept an already-normalized second (used with a processed .npz)."""
        self._history.append(np.asarray(second, dtype=np.float32))
        self.seconds_seen += 1

    def set_context(self, seconds):
        """Seat the trailing context directly, for replaying a chosen part of a recording.

        Deliberately not used by the demo or by ``--verify``: seating the buffer from an
        offline slice would bypass everything this class exists to exercise, and would make
        the equivalence check compare one array slice against the same array slice.
        """
        self._history = deque(seconds, maxlen=self.context)

    def reset_history(self):
        """Drop buffered seconds, keeping the normalizer state."""
        self._history = deque(maxlen=self.context)

    def context_filled(self):
        """True once a full context window has been pushed in."""
        return len(self._history) == self.context

    def warm_up(self, runs=5):
        """Run a few forwards so the timing below measures steady state, not first-call cost."""
        dummy = [np.zeros(self.sample_rate, dtype=np.float32)] * self.context
        for _ in range(runs):
            self._forward(dummy)

    def _forward(self, seconds):
        tensor = torch.from_numpy(np.stack(seconds)).unsqueeze(0)
        with torch.no_grad():
            return self.model(tensor)[0]

    def predict_trailing(self):
        """Label the most recent second from the trailing context. Rolling mode."""
        logits = self._forward(list(self._history))
        return logits[-1]

    def predict_window(self):
        """Label a full context window in one pass. Window mode; matches evaluate.py."""
        assert len(self._history) == self.context, "window mode needs a full context"
        return self._forward(list(self._history))


# ------------------------------------------------------------------------------ helpers

def build_model(config, checkpoint_path):
    model = SleepStagingModel(config=config)
    note = "randomly initialised weights (labels are meaningless; the path is not)"
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise SystemExit(f"checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        note = f"trained weights from {checkpoint_path} (epoch {state.get('epoch', '?')})"
    model.eval()
    return model, note


def batched_predictions(x, model, context):
    """Reproduce ``src/eval/evaluate.py``: non-overlapping windows, argmax per second."""
    preds = []
    with torch.no_grad():
        for start in range(0, len(x) - context + 1, context):
            window = torch.from_numpy(x[start:start + context]).float().unsqueeze(0)
            preds.append(torch.argmax(model(window), dim=-1)[0].numpy())
    return np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)


def find_transition(labels, context, minutes, sample_rate=1):
    """Pick a segment start (in seconds) whose window contains a stage change."""
    if labels is None:
        return 0
    span = int(minutes * 60)
    changes = np.flatnonzero(np.diff(labels) != 0)
    changes = changes[(changes > span // 2) & (changes < len(labels) - span // 2)]
    if len(changes) == 0:
        return 0
    start = int(changes[len(changes) // 2]) - span // 2
    return max(0, (start // context) * context)


# --------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true",
                        help="synthesise a recording with a stage transition; needs no data")
    source.add_argument("--subject", help="a processed subject_XX.npz")
    source.add_argument("--edf", help="a raw *-PSG.edf")

    ap.add_argument("--hypnogram", help="the matching *-Hypnogram.edf, for agreement figures")
    ap.add_argument("--config", default="configs/sleep78_streaming_causal.yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="trained weights; random weights are used otherwise")
    ap.add_argument("--mode", choices=("window", "rolling"), default="window",
                    help="window reproduces evaluate.py; rolling is true per-second streaming")
    ap.add_argument("--start", type=int, default=None,
                    help="first second to print (default: a segment containing a transition). "
                         "The stream always runs from second 0 regardless, so the trailing "
                         "statistic is warmed up exactly as it is in the archived pipeline.")
    ap.add_argument("--minutes", type=float, default=6.0, help="how much to print")
    ap.add_argument("--verify", action="store_true",
                    help="also assert the streaming labels equal evaluate.py's, over the "
                         "whole recording")
    ap.add_argument("--quiet", action="store_true", help="summary only, no per-second lines")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    sample_rate = int(config["data"]["resample_rate"])
    context = int(config["data"]["sequence_length"])
    norm_cfg = config["data"].get("normalization", {})
    window_samples = int(round(float(norm_cfg.get("window_seconds", 30)) * sample_rate))
    eps = float(norm_cfg.get("eps", 1e-8))

    if norm_cfg.get("method") != "causal_rolling":
        print(f"WARNING: {args.config} normalizes with '{norm_cfg.get('method')}', which is "
              f"not streaming-safe. Use a streaming config for an honest demonstration.")

    model, weights_note = build_model(config, args.checkpoint)
    stager = StreamingSleepStager(model, sample_rate, context, window_samples, eps)

    print("=" * 72)
    print("STREAMING SLEEP STAGING - one raw sample in, one label per second out")
    print("=" * 72)
    print(f"Config        : {args.config}")
    print(f"Model         : {weights_note}")
    print(f"Context       : {context} s   Normalizer: trailing {window_samples / sample_rate:g} s "
          f"at {sample_rate} Hz")
    print(f"Emission mode : {args.mode}")

    # ------------------------------------------------------------------ ingest the signal
    labels = None
    normalizer_note = None
    normalize_micros = None

    if args.subject:
        data = np.load(args.subject)
        x = np.asarray(data["x"], dtype=np.float32)
        labels = np.asarray(data["y"], dtype=np.int64)
        stored = str(data["normalization_method"]) if "normalization_method" in data.files else "unknown"
        print(f"Source        : {args.subject} ({len(x):,} s, normalization='{stored}')")
        normalizer_note = (
            "the trailing z-score was applied offline when this file was written; "
            "scripts/verify_causality.py proves that array equals what StreamingZScore\n"
            "                produces sample by sample, so the model half is streamed here and the "
            "normalizer half is streamed there")
        # Time the normalizer on this many samples anyway, so the throughput line is measured
        # rather than quoted.
        probe = x[: min(len(x), 600)].reshape(-1)
        filt = StreamingZScore(window_samples=window_samples, eps=eps)
        started = time.perf_counter()
        for value in probe:
            filt.update(value)
        normalize_micros = (time.perf_counter() - started) / len(probe) * 1e6
        for second in x:
            stager.push_second(second)
    else:
        if args.synthetic:
            raw, labels = synthetic_recording(sample_rate, minutes=10)
            print(f"Source        : synthetic recording, {len(raw) / sample_rate / 60:.0f} min, "
                  f"one stage transition at the midpoint")
        else:
            raw, labels, channel = load_edf(args.edf, args.hypnogram,
                                            config["data"]["target_channel"], sample_rate)
            print(f"Source        : {args.edf} (channel '{channel}', "
                  f"{len(raw) / sample_rate / 3600:.1f} h)")

        seconds = []
        started = time.perf_counter()
        for value in raw:
            second = stager.push_sample(value)
            if second is not None:
                seconds.append(second)
        normalize_micros = (time.perf_counter() - started) / len(raw) * 1e6
        x = np.stack(seconds) if seconds else np.empty((0, sample_rate), dtype=np.float32)
        normalizer_note = (f"{len(raw):,} raw samples pushed through StreamingZScore one at a "
                           f"time, {normalize_micros:.2f} us each")
        if labels is not None:
            labels = labels[: len(x)]

    if len(x) < context:
        raise SystemExit(f"recording is {len(x)} s, shorter than the {context} s context")
    print(f"Normalizer    : {normalizer_note}")

    # ------------------------------------------------------------------ stream the labels
    start = args.start if args.start is not None else find_transition(labels, context, args.minutes)
    start = max(0, min(start, len(x) - context))
    span = min(int(args.minutes * 60), len(x) - start)
    if args.mode == "window":
        # Clamp first, then align, and floor the span to whole windows. Aligning before the
        # clamp lets the clamp knock start back off the grid whenever len(x) is not a
        # multiple of the context, and the printed labels would then not be the ones
        # evaluate.py produces for those seconds.
        start = (start // context) * context
        span = (span // context) * context
        if span == 0:
            raise SystemExit(f"--minutes {args.minutes} covers {int(args.minutes * 60)} s, "
                             f"less than one {context} s window; nothing would be emitted")
        assert start % context == 0, "window mode must emit on evaluate.py's window grid"
    else:
        start = max(start, context - 1)                # need a full trailing context
        span = min(span, len(x) - start)
        if span <= 0:
            raise SystemExit("nothing to emit: the trailing context reaches the end of the "
                             "recording")

    print(f"Printing      : {hms(start)} to {hms(start + span)} "
          f"({span} s of {len(x):,} s streamed)")
    print("-" * 72)

    stager.warm_up()

    emitted, forward_times = [], []
    previous = None
    last = min(start + span, len(x))
    # Seconds are pushed in one at a time and the trailing buffer fills on its own. Seating
    # the buffer from a slice instead would make the run a re-slice of the offline array,
    # and --verify below could then never fail.
    stager.reset_history()
    if args.mode == "window":
        for second in range(start, last):
            stager.push_second(x[second])
            if stager.context_filled() and (second - start + 1) % context == 0:
                began = time.perf_counter()
                logits = stager.predict_window()
                forward_times.append((time.perf_counter() - began) * 1000.0)
                probabilities = torch.softmax(logits, dim=-1).numpy()
                window_start = second - context + 1
                for offset in range(context):
                    emitted.append((window_start + offset, probabilities[offset]))
    else:
        # Fill the trailing context first, then label every second from it.
        for second in range(max(0, start - context + 1), start):
            stager.push_second(x[second])
        for second in range(start, last):
            stager.push_second(x[second])
            if not stager.context_filled():
                continue
            began = time.perf_counter()
            logits = stager.predict_trailing()
            forward_times.append((time.perf_counter() - began) * 1000.0)
            emitted.append((second, torch.softmax(logits, dim=-1).numpy()))

    correct = 0
    scored = 0
    for second, probabilities in emitted:
        stage = int(np.argmax(probabilities))
        truth = int(labels[second]) if labels is not None and labels[second] >= 0 else None
        if truth is not None:
            scored += 1
            correct += stage == truth
        if not args.quiet:
            marker = "  <- stage change" if previous is not None and stage != previous else ""
            truth_text = f"  scored={STAGES[truth]}" if truth is not None else ""
            distribution = " ".join(f"{name}={p:.2f}" for name, p in zip(STAGES, probabilities))
            print(f"  {hms(second)}  {STAGES[stage]:>3}  p={probabilities[stage]:.2f}"
                  f"{truth_text}   [{distribution}]{marker}")
        previous = stage

    # ------------------------------------------------------------------------- throughput
    print("-" * 72)
    forward_ms = float(np.median(forward_times))
    seconds_per_forward = context if args.mode == "window" else 1
    compute_ms_per_second = forward_ms / seconds_per_forward + normalize_micros * sample_rate / 1000.0
    print(f"Normalizer    : {normalize_micros:.2f} us per raw sample "
          f"({normalize_micros * sample_rate / 1000.0:.3f} ms per second of EEG)")
    print(f"Model         : {forward_ms:.2f} ms per forward (median of {len(forward_times)}), "
          f"covering {seconds_per_forward} s")
    print(f"Total         : {compute_ms_per_second:.3f} ms of compute per second of EEG "
          f"= {1000.0 / compute_ms_per_second:.0f}x real time, single CPU thread")
    if args.mode == "window":
        print("                This is the batch cost: one forward is amortised over the whole")
        print("                context. A device labelling every second pays the rolling cost")
        print("                instead - run --mode rolling for the number to quote as latency.")
    else:
        print("                This is the deployment cost, recomputing the full context every")
        print("                second. Caching the convolution and attention state between")
        print("                seconds would cut it; that work is not done here.")
    if scored:
        print(f"Agreement     : {correct / scored:.1%} over {scored} scored seconds"
              + ("" if args.checkpoint else "  (random weights - ignore this number)"))

    # ----------------------------------------------------------------------------- verify
    status = 0
    if args.verify:
        print("-" * 72)
        print("VERIFY: streaming labels vs src/eval/evaluate.py over the whole recording")
        # Stream the whole recording second by second, letting the trailing buffer fill and
        # tile on its own, and require the result to equal what evaluate.py's window slicing
        # produces. Nothing here re-slices x, so a buffering or alignment error fails here.
        stager.reset_history()
        streamed = []
        for index in range(len(x)):
            stager.push_second(x[index])
            if stager.context_filled() and (index + 1) % context == 0:
                streamed.append(torch.argmax(stager.predict_window(), dim=-1).numpy())
        streamed = np.concatenate(streamed) if streamed else np.empty(0, dtype=np.int64)
        reference = batched_predictions(x, model, context)
        compared = min(len(streamed), len(reference))
        identical = (len(streamed) == len(reference)
                     and np.array_equal(streamed[:compared], reference[:compared]))
        mismatches = int((streamed[:compared] != reference[:compared]).sum())
        print(f"  seconds streamed : {len(streamed):,} (buffered one second at a time)")
        print(f"  seconds compared : {compared:,}")
        print(f"  mismatches       : {mismatches}")
        print(f"  result           : {'PASS - identical' if identical else 'FAIL'}")
        if not identical:
            status = 1
        else:
            print("  Labels produced by the incremental buffer match the offline evaluation")
            print("  window for window, so the archived numbers are what this path emits.")
            if args.subject:
                print("  Note: this file was normalized offline, so only the model half is")
                print("  streamed here. Run with --edf for the raw-to-label path.")

    print("=" * 72)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
