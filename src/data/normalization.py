"""Streaming-safe signal normalization utilities.

Two implementations of the same statistic are kept here on purpose:

``causal_rolling_zscore``
    Vectorised, prefix-sum based. Used by the offline preprocessing script.

``StreamingZScore``
    A sample-at-a-time online filter with O(window) state. This is what a wearable
    would actually run. It exists so the offline pipeline can be *proved* equivalent
    to a deployable online one (see ``scripts/verify_causality.py``), which is the
    claim the paper makes about the preprocessing step.

Both use the trailing window ``signal[max(0, t-window+1) : t+1]`` and therefore
cannot see the future by construction.
"""

from collections import deque

import numpy as np


def causal_rolling_zscore(signal, window_samples, eps=1e-8):
    """Normalize a 1-D signal using only the current and preceding samples.

    At sample ``t``, the mean and variance are computed over
    ``signal[max(0, t-window_samples+1):t+1]``. Consequently, changing any
    sample after ``t`` cannot change the normalized value at ``t``.

    The output is bounded: within a trailing window holding ``n`` samples the
    largest attainable magnitude is ``sqrt(n - 1)``, so no infinities or NaNs can
    be produced even during the warm-up at the start of a recording.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {signal.shape}")
    if window_samples < 1:
        raise ValueError("window_samples must be at least 1")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if signal.size == 0:
        return signal.copy()

    prefix = np.concatenate(([0.0], np.cumsum(signal, dtype=np.float64)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(signal * signal, dtype=np.float64)))
    end = np.arange(1, signal.size + 1)
    start = np.maximum(0, end - window_samples)
    count = end - start
    rolling_sum = prefix[end] - prefix[start]
    rolling_sum_sq = prefix_sq[end] - prefix_sq[start]
    mean = rolling_sum / count
    variance = np.maximum(rolling_sum_sq / count - mean * mean, 0.0)
    std = np.sqrt(variance)

    normalized = np.zeros_like(signal)
    valid = std >= eps
    normalized[valid] = (signal[valid] - mean[valid]) / std[valid]
    return normalized


class StreamingZScore:
    """Online, sample-at-a-time equivalent of :func:`causal_rolling_zscore`.

    Holds only the trailing window, so memory is O(window_samples) and each update
    is O(1). This is the reference for what runs on a device.

        filt = StreamingZScore(window_samples=3000)
        for sample in incoming_eeg:
            z = filt.update(sample)

    ``reset()`` clears the state, matching the offline behaviour of restarting the
    statistic at the beginning of every recording.
    """

    def __init__(self, window_samples, eps=1e-8):
        if window_samples < 1:
            raise ValueError("window_samples must be at least 1")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.window_samples = int(window_samples)
        self.eps = float(eps)
        self.reset()

    def reset(self):
        """Drop all history. Call this at the start of each new recording."""
        self._buffer = deque()
        self._sum = 0.0
        self._sum_sq = 0.0

    def update(self, sample):
        """Push one raw sample, return its normalized value."""
        value = float(sample)
        self._buffer.append(value)
        self._sum += value
        self._sum_sq += value * value
        if len(self._buffer) > self.window_samples:
            dropped = self._buffer.popleft()
            self._sum -= dropped
            self._sum_sq -= dropped * dropped

        count = len(self._buffer)
        mean = self._sum / count
        variance = max(self._sum_sq / count - mean * mean, 0.0)
        std = variance ** 0.5
        if std < self.eps:
            return 0.0
        return (value - mean) / std

    def process(self, signal):
        """Run :meth:`update` over a 1-D array and return the normalized array."""
        signal = np.asarray(signal, dtype=np.float64)
        if signal.ndim != 1:
            raise ValueError(f"signal must be 1-D, got shape {signal.shape}")
        return np.array([self.update(v) for v in signal], dtype=np.float64)


def normalize_signal(signal, method="causal_rolling", window_samples=3000, eps=1e-8):
    """Apply an explicitly selected normalization method to a recording."""
    if method == "causal_rolling":
        return causal_rolling_zscore(signal, window_samples=window_samples, eps=eps)
    if method == "none":
        return np.asarray(signal, dtype=np.float64).copy()
    raise ValueError(
        f"Unsupported continuous normalization method '{method}'. "
        "Use 'causal_rolling' or 'none'."
    )
