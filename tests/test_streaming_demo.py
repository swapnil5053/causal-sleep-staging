"""The sample-at-a-time front end, and the equivalence check built on it.

The point of this class is that a device can produce the archived numbers without ever
holding a night in memory. That claim is only worth making if the demo actually streams,
so the tests here check the buffer behaves incrementally and, above all, that the
equivalence check can FAIL. A verification that cannot fail proves nothing, and an earlier
version of this script compared one slice of an offline array against the same slice.
"""
import importlib.util
import os
import unittest

import numpy as np
import torch

from src.data.normalization import StreamingZScore, causal_rolling_zscore
from src.model.full_model import SleepStagingModel

SAMPLE_RATE = 100
CONTEXT = 30


def _load_demo():
    """Load the script by path; scripts/ is not a package."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "scripts", "streaming_demo.py")
    spec = importlib.util.spec_from_file_location("streaming_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demo = _load_demo()


def _model():
    torch.manual_seed(11)
    model = SleepStagingModel(config={"model": {
        "mrcnn_channels_1": 16, "mrcnn_channels_2": 16,
        "tcn_channels": [32, 32, 32], "tcn_kernel_size": 3, "tcn_dilations": [1, 2, 4],
        "tcn_dropout": 0.2, "attn_num_heads": 4, "attn_dropout": 0.1,
        "num_classes": 5, "causal": True}})
    model.eval()
    return model


def _stager(model):
    return demo.StreamingSleepStager(model, sample_rate=SAMPLE_RATE, context=CONTEXT,
                                     window_samples=SAMPLE_RATE * 30)


class BufferTests(unittest.TestCase):
    def setUp(self):
        self.model = _model()
        self.stager = _stager(self.model)

    def test_a_second_is_only_emitted_once_it_is_complete(self):
        for index in range(SAMPLE_RATE - 1):
            self.assertIsNone(self.stager.push_sample(float(index)))

        self.assertIsNotNone(self.stager.push_sample(0.0))

    def test_context_fills_incrementally_and_then_holds_steady(self):
        for index in range(CONTEXT - 1):
            self.stager.push_second(np.zeros(SAMPLE_RATE, dtype=np.float32))
            self.assertFalse(self.stager.context_filled())

        self.stager.push_second(np.zeros(SAMPLE_RATE, dtype=np.float32))
        self.assertTrue(self.stager.context_filled())

        for _ in range(10):                       # the deque must not grow past the context
            self.stager.push_second(np.zeros(SAMPLE_RATE, dtype=np.float32))
            self.assertEqual(len(self.stager._history), CONTEXT)

    def test_reset_history_keeps_the_normalizer_running(self):
        """A new window restarts the model's context, not the trailing z-score.

        Compared against an independently advanced filter rather than merely asserting the
        value changed: a reset normalizer would also change the value, so "different" does
        not distinguish the two and would pass even if reset_history wiped the filter.
        """
        reference = StreamingZScore(window_samples=SAMPLE_RATE * 30)
        for value in range(500):
            self.stager.push_sample(float(value))
            reference.update(float(value))

        self.stager.reset_history()

        self.assertFalse(self.stager.context_filled())
        self.assertAlmostEqual(self.stager.normalizer.update(1.0), reference.update(1.0),
                               places=12)

    def test_streamed_normalization_equals_the_offline_array(self):
        rng = np.random.default_rng(4)
        raw = rng.normal(scale=2e-5, size=SAMPLE_RATE * 40)
        seconds = [s for s in (self.stager.push_sample(v) for v in raw) if s is not None]
        streamed = np.concatenate(seconds)
        offline = causal_rolling_zscore(raw, window_samples=SAMPLE_RATE * 30)

        np.testing.assert_allclose(streamed, offline[:len(streamed)], rtol=1e-6, atol=1e-6)

    def test_trailing_prediction_labels_only_the_latest_second(self):
        """It must be the newest second's logits, not the oldest and not a constant.

        Shape alone would pass for predict_window()[0], labelling the oldest second in the
        buffer, which is a causality bug, and for a hard-coded zero vector.
        """
        rng = np.random.default_rng(9)
        for _ in range(CONTEXT):
            self.stager.push_second(rng.normal(size=SAMPLE_RATE).astype(np.float32))

        trailing = self.stager.predict_trailing()
        window = self.stager.predict_window()

        self.assertEqual(tuple(trailing.shape), (5,))
        torch.testing.assert_close(trailing, window[-1])
        self.assertFalse(torch.allclose(trailing, window[0]))

    def test_the_trailing_prediction_moves_when_the_newest_second_changes(self):
        """Guards against the buffer being read at a fixed offset."""
        rng = np.random.default_rng(10)
        history = [rng.normal(size=SAMPLE_RATE).astype(np.float32) for _ in range(CONTEXT)]
        for second in history:
            self.stager.push_second(second)
        before = self.stager.predict_trailing()

        self.stager.reset_history()
        for second in history[:-1]:
            self.stager.push_second(second)
        self.stager.push_second((history[-1] * 5.0 + 3.0).astype(np.float32))

        self.assertFalse(torch.allclose(before, self.stager.predict_trailing()))


class EquivalenceTests(unittest.TestCase):
    """The incremental buffer must reproduce evaluate.py's window slicing, and must not
    be able to pass when it does not."""

    def setUp(self):
        self.model = _model()
        rng = np.random.default_rng(6)
        self.x = rng.normal(size=(CONTEXT * 6, SAMPLE_RATE)).astype(np.float32)

    def _stream(self, transform=None):
        stager = _stager(self.model)
        stager.reset_history()
        out = []
        for index in range(len(self.x)):
            second = self.x[index] if transform is None else transform(index, self.x[index])
            if second is None:
                continue
            stager.push_second(second)
            if stager.context_filled() and (index + 1) % CONTEXT == 0:
                out.append(torch.argmax(stager.predict_window(), dim=-1).numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

    def test_incremental_buffer_matches_batched_prediction(self):
        reference = demo.batched_predictions(self.x, self.model, CONTEXT)

        np.testing.assert_array_equal(self._stream(), reference)

    def test_a_corrupted_buffer_is_detected(self):
        reference = demo.batched_predictions(self.x, self.model, CONTEXT)
        corrupted = self._stream(transform=lambda i, s: s * 4.0 + 2.0)

        self.assertFalse(np.array_equal(corrupted, reference))

    def test_a_dropped_second_is_detected(self):
        reference = demo.batched_predictions(self.x, self.model, CONTEXT)
        dropped = self._stream(transform=lambda i, s: None if i == 40 else s)

        self.assertFalse(len(dropped) == len(reference)
                         and np.array_equal(dropped, reference))


if __name__ == "__main__":
    unittest.main()
