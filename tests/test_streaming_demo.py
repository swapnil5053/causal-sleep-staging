"""The streaming front end must be the deployed path and the evaluated path at once.

A demo that merely looks plausible on stage is worth nothing; these tests pin the two
properties the demo is used to claim. First, that pushing raw samples one at a time
reproduces the arrays the offline pipeline writes. Second, that the labels it emits are the
same labels ``src/eval/evaluate.py`` produced for the archived numbers - so the kappa on the
slide is the number this path would give, not a different one obtained by batching a night.
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from src.data.normalization import causal_rolling_zscore  # noqa: E402
from src.model.full_model import SleepStagingModel  # noqa: E402
from streaming_demo import (  # noqa: E402
    StreamingSleepStager,
    batched_predictions,
    find_transition,
    synthetic_recording,
)

SAMPLE_RATE = 100
CONTEXT = 30
WINDOW_SAMPLES = 30 * SAMPLE_RATE

CONFIG = {
    "model": {
        "mrcnn_channels_1": 8,
        "mrcnn_channels_2": 8,
        "tcn_channels": [16, 16],
        "tcn_kernel_size": 3,
        "tcn_dilations": [1, 2],
        "tcn_dropout": 0.0,
        "attn_num_heads": 2,
        "attn_dropout": 0.0,
        "num_classes": 5,
        "causal": True,
    }
}


def build_model():
    torch.manual_seed(0)
    model = SleepStagingModel(config=CONFIG)
    model.eval()
    return model


def stream(stager, raw):
    """Push a raw signal sample by sample; return the completed seconds."""
    seconds = [second for second in (stager.push_sample(v) for v in raw) if second is not None]
    return np.stack(seconds)


class StreamingFrontEndTests(unittest.TestCase):
    def setUp(self):
        self.raw, self.labels = synthetic_recording(SAMPLE_RATE, minutes=2)
        self.model = build_model()
        self.stager = StreamingSleepStager(self.model, SAMPLE_RATE, CONTEXT, WINDOW_SAMPLES)

    def test_sample_at_a_time_matches_the_offline_arrays(self):
        streamed = stream(self.stager, self.raw)
        offline = causal_rolling_zscore(self.raw, window_samples=WINDOW_SAMPLES)
        offline = np.asarray(offline.reshape(-1, SAMPLE_RATE), dtype=np.float32)

        self.assertEqual(streamed.shape, offline.shape)
        # Prefix sums and a running accumulator differ in the last bits of a float64; the
        # claim is that the stored float32 arrays are the same numbers.
        np.testing.assert_allclose(streamed, offline, rtol=0, atol=1e-6)

    def test_streaming_labels_equal_the_evaluation_labels(self):
        x = stream(self.stager, self.raw)

        streamed = []
        for start in range(0, len(x) - CONTEXT + 1, CONTEXT):
            self.stager.set_context(x[start:start + CONTEXT])
            streamed.append(torch.argmax(self.stager.predict_window(), dim=-1).numpy())
        streamed = np.concatenate(streamed)

        np.testing.assert_array_equal(streamed, batched_predictions(x, self.model, CONTEXT))

    def test_a_future_sample_cannot_change_an_emitted_second(self):
        """The property the whole project claims, asserted on the streaming object itself."""
        cut = 60 * SAMPLE_RATE
        perturbed = self.raw.copy()
        perturbed[cut:] = perturbed[cut:] * 50.0 + 1e-3

        clean = stream(self.stager, self.raw)
        self.stager.reset()
        dirty = stream(self.stager, perturbed)

        emitted_before = cut // SAMPLE_RATE
        np.testing.assert_array_equal(clean[:emitted_before], dirty[:emitted_before])
        self.assertFalse(np.array_equal(clean[emitted_before:], dirty[emitted_before:]),
                         "the perturbation must actually change later seconds, or the test "
                         "proves nothing")

    def test_rolling_prediction_reads_only_the_trailing_context(self):
        x = stream(self.stager, self.raw)
        second = 80

        self.stager.set_context(x[second - CONTEXT + 1:second + 1])
        expected = self.stager.predict_trailing()

        future = x.copy()
        future[second + 1:] = 0.0
        self.stager.set_context(future[second - CONTEXT + 1:second + 1])
        self.assertTrue(torch.equal(expected, self.stager.predict_trailing()))

    def test_reset_clears_the_normalizer_between_recordings(self):
        first = stream(self.stager, self.raw)
        self.stager.reset()
        second = stream(self.stager, self.raw)
        np.testing.assert_array_equal(first, second)


class SegmentSelectionTests(unittest.TestCase):
    def test_chosen_segment_contains_a_stage_change(self):
        labels = np.concatenate([np.full(600, 2), np.full(600, 4)]).astype(np.int64)
        start = find_transition(labels, context=CONTEXT, minutes=6.0)
        window = labels[start:start + 360]
        self.assertGreater(len(set(window.tolist())), 1)
        self.assertEqual(start % CONTEXT, 0)

    def test_no_labels_falls_back_to_the_start(self):
        self.assertEqual(find_transition(None, context=CONTEXT, minutes=6.0), 0)


if __name__ == "__main__":
    unittest.main()
