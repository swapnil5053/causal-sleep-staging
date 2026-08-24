"""Streaming-safe signal normalization utilities."""

import numpy as np


def causal_rolling_zscore(signal, window_samples, eps=1e-8):
    """Normalize a 1-D signal using only the current and preceding samples.

    At sample ``t``, the mean and variance are computed over
    ``signal[max(0, t-window_samples+1):t+1]``. Consequently, changing any
    sample after ``t`` cannot change the normalized value at ``t``.
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
