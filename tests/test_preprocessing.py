import unittest

import numpy as np

from src.data.normalization import causal_rolling_zscore, normalize_signal


class CausalNormalizationTests(unittest.TestCase):
    def test_future_perturbation_cannot_change_past_outputs(self):
        rng = np.random.default_rng(7)
        signal = rng.normal(size=500)
        changed = signal.copy()
        changed[300:] += rng.normal(loc=100.0, scale=20.0, size=200)

        original_output = causal_rolling_zscore(signal, window_samples=80)
        changed_output = causal_rolling_zscore(changed, window_samples=80)

        np.testing.assert_array_equal(original_output[:300], changed_output[:300])
        self.assertFalse(np.array_equal(original_output[300:], changed_output[300:]))

    def test_matches_direct_trailing_window_calculation(self):
        signal = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        actual = causal_rolling_zscore(signal, window_samples=3)
        expected = []
        for index, value in enumerate(signal):
            window = signal[max(0, index - 2):index + 1]
            std = window.std()
            expected.append(0.0 if std < 1e-8 else (value - window.mean()) / std)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_constant_signal_is_finite_and_zero(self):
        actual = causal_rolling_zscore(np.ones(20), window_samples=10)
        np.testing.assert_array_equal(actual, np.zeros(20))
        self.assertTrue(np.isfinite(actual).all())

    def test_rejects_unknown_method(self):
        with self.assertRaisesRegex(ValueError, "Unsupported continuous normalization"):
            normalize_signal(np.arange(5.0), method="epoch")


if __name__ == "__main__":
    unittest.main()
