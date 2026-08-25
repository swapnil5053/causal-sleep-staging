"""End-to-end causality: raw EEG sample -> normalization -> model -> per-second logits.

The existing tests prove the two halves separately: ``test_preprocessing`` shows the
normalizer never reads ahead, ``test_model_behavior`` shows the layers never read ahead.
Neither shows that the *composition* is causal, which is the claim the paper makes.

These tests run the real preprocessing normalization and the real model over one signal,
perturb the raw signal at a future sample, and require every earlier output to be
bit-identical. They also require the offline vectorised normalizer to agree exactly with
the sample-at-a-time streaming filter a device would run.
"""
import unittest

import numpy as np
import torch

from src.data.normalization import StreamingZScore, causal_rolling_zscore
from src.model.full_model import SleepStagingModel

SAMPLE_RATE = 100
WINDOW_SECONDS = 30


def _config(causal=True):
    return {
        "model": {
            "mrcnn_channels_1": 16,
            "mrcnn_channels_2": 16,
            "tcn_channels": [32, 32, 32],
            "tcn_kernel_size": 3,
            "tcn_dilations": [1, 2, 4],
            "tcn_dropout": 0.2,
            "attn_num_heads": 4,
            "attn_dropout": 0.1,
            "num_classes": 5,
            "causal": causal,
        }
    }


def _pipeline(raw_signal, model):
    """Full path: raw 1-D signal -> causal rolling z-score -> (1, seconds, 100) -> logits."""
    normalized = causal_rolling_zscore(
        raw_signal, window_samples=WINDOW_SECONDS * SAMPLE_RATE
    )
    seconds = len(normalized) // SAMPLE_RATE
    windows = normalized[: seconds * SAMPLE_RATE].reshape(seconds, SAMPLE_RATE)
    tensor = torch.from_numpy(windows).float().unsqueeze(0)
    with torch.no_grad():
        return model(tensor)


class EndToEndCausalityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.rng = np.random.default_rng(23)
        self.seconds = 120
        self.split_second = 60
        self.signal = self.rng.normal(scale=2e-5, size=self.seconds * SAMPLE_RATE)

    def test_future_raw_samples_cannot_change_earlier_predictions(self):
        model = SleepStagingModel(config=_config(causal=True))
        model.eval()

        perturbed = self.signal.copy()
        cut = self.split_second * SAMPLE_RATE
        # a large amplitude and offset change, i.e. exactly what a rolling statistic
        # computed over the future would leak
        perturbed[cut:] = perturbed[cut:] * 50.0 + 1e-3

        original = _pipeline(self.signal, model)
        changed = _pipeline(perturbed, model)

        torch.testing.assert_close(
            original[:, : self.split_second, :],
            changed[:, : self.split_second, :],
            rtol=0,
            atol=0,
        )

    def test_perturbation_does_reach_later_predictions(self):
        """Guards against a vacuous pass (e.g. a model that ignores its input)."""
        model = SleepStagingModel(config=_config(causal=True))
        model.eval()

        perturbed = self.signal.copy()
        cut = self.split_second * SAMPLE_RATE
        perturbed[cut:] = perturbed[cut:] * 50.0 + 1e-3

        original = _pipeline(self.signal, model)
        changed = _pipeline(perturbed, model)

        self.assertFalse(
            torch.equal(
                original[:, self.split_second:, :], changed[:, self.split_second:, :]
            ),
            "the perturbation had no effect anywhere, so the test proves nothing",
        )

    def test_noncausal_configuration_does_leak(self):
        """The ablation arm must actually see the future, or it is not a control."""
        model = SleepStagingModel(config=_config(causal=False))
        model.eval()

        perturbed = self.signal.copy()
        cut = self.split_second * SAMPLE_RATE
        perturbed[cut:] = perturbed[cut:] * 50.0 + 1e-3

        original = _pipeline(self.signal, model)
        changed = _pipeline(perturbed, model)

        self.assertFalse(
            torch.equal(
                original[:, : self.split_second, :], changed[:, : self.split_second, :]
            ),
            "causal: false should allow future signal to reach earlier outputs",
        )

    def test_offline_normalizer_matches_online_streaming_filter(self):
        """The archived arrays must equal what a device computes sample by sample."""
        window = WINDOW_SECONDS * SAMPLE_RATE
        offline = causal_rolling_zscore(self.signal, window_samples=window)
        online = StreamingZScore(window_samples=window).process(self.signal)
        np.testing.assert_allclose(offline, online, rtol=1e-9, atol=1e-9)

    def test_streaming_filter_resets_between_recordings(self):
        window = WINDOW_SECONDS * SAMPLE_RATE
        filt = StreamingZScore(window_samples=window)
        first = filt.process(self.signal[:2000])
        filt.reset()
        second = filt.process(self.signal[:2000])
        np.testing.assert_array_equal(first, second)

    def test_normalized_output_is_finite_including_warmup(self):
        """A cold start divides by the std of very few samples; it must stay bounded."""
        window = WINDOW_SECONDS * SAMPLE_RATE
        normalized = causal_rolling_zscore(self.signal, window_samples=window)
        self.assertTrue(np.isfinite(normalized).all())
        # |z| within a trailing window of n samples cannot exceed sqrt(n - 1)
        self.assertLessEqual(np.abs(normalized).max(), np.sqrt(window - 1) + 1e-9)


if __name__ == "__main__":
    unittest.main()
