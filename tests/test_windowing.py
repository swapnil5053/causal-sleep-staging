"""Windowing and evaluation-mode behaviour.

Two properties matter here. The archived windowing must be reachable unchanged, because
every published number depends on it, and the options that change it must do exactly what
they say: drop windows that span a discontinuity, cover the trailing seconds, and evaluate
against a rolling buffer rather than a fresh one.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np
import torch

from src.data.dataset import SleepDataset, segment_starts
from src.eval.evaluate import predict_streaming, predict_windowed
from src.model.full_model import SleepStagingModel

SECONDS_PER_EPOCH = 30


def _config():
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
            "causal": True,
        }
    }


class WindowingTestCase(unittest.TestCase):
    seconds = {"00": 300, "01": 270}
    starts = {"00": [0, 120], "01": [0]}

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="cap-windows-")
        rng = np.random.default_rng(3)
        for sub_id, n in cls.seconds.items():
            np.savez_compressed(
                os.path.join(cls.dir, f"subject_{sub_id}.npz"),
                x=rng.normal(size=(n, 100)).astype(np.float32),
                y=rng.integers(0, 5, size=n).astype(np.int64),
                normalization_method=np.array("causal_rolling"),
                normalization_window_seconds=np.array(30),
                segment_starts=np.array(cls.starts[sub_id], dtype=np.int64),
            )
        # A subject written before segment metadata existed.
        np.savez_compressed(
            os.path.join(cls.dir, "subject_99.npz"),
            x=rng.normal(size=(300, 100)).astype(np.float32),
            y=rng.integers(0, 5, size=300).astype(np.int64),
            normalization_method=np.array("causal_rolling"),
            normalization_window_seconds=np.array(30),
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)


class DatasetWindowingTests(WindowingTestCase):
    def test_defaults_reproduce_the_archived_windowing(self):
        dataset = SleepDataset(self.dir, ["00"], seq_len=120, stride=120)

        # 300 seconds, stride 120 -> starts at 0 and 120; the last 60 s are not covered.
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.windows_dropped_at_boundaries, 0)

    def test_cover_tail_adds_a_final_window_flush_with_the_end(self):
        dataset = SleepDataset(self.dir, ["00"], seq_len=120, stride=120, cover_tail=True)
        source = np.load(os.path.join(self.dir, "subject_00.npz"))["x"]

        self.assertEqual(len(dataset), 3)
        np.testing.assert_array_equal(dataset.windows[-1], source[-120:])

    def test_cover_tail_is_a_no_op_when_the_stride_already_reaches_the_end(self):
        exact = SleepDataset(self.dir, ["01"], seq_len=90, stride=90)
        covered = SleepDataset(self.dir, ["01"], seq_len=90, stride=90, cover_tail=True)

        self.assertEqual(len(exact), len(covered))

    def test_boundaries_drop_windows_that_span_a_join(self):
        plain = SleepDataset(self.dir, ["00"], seq_len=120, stride=60)
        masked = SleepDataset(self.dir, ["00"], seq_len=120, stride=60,
                              respect_boundaries=True)

        # Subject 00 restarts at second 120, so the window at 60 spans the join.
        self.assertEqual(len(plain) - len(masked), 1)
        self.assertEqual(masked.windows_dropped_at_boundaries, 1)

    def test_a_window_starting_exactly_on_a_boundary_is_kept(self):
        masked = SleepDataset(self.dir, ["00"], seq_len=120, stride=120,
                              respect_boundaries=True)

        self.assertEqual(len(masked), 2)

    def test_legacy_files_refuse_boundary_masking_instead_of_ignoring_it(self):
        with self.assertRaisesRegex(ValueError, "segment metadata"):
            SleepDataset(self.dir, ["99"], seq_len=120, stride=60, respect_boundaries=True)

    def test_segment_starts_defaults_to_one_segment_for_legacy_files(self):
        legacy = np.load(os.path.join(self.dir, "subject_99.npz"))

        self.assertIsNone(segment_starts(legacy, 300))

    def test_window_subjects_line_up_with_the_windows(self):
        dataset = SleepDataset(self.dir, ["00", "01"], seq_len=120, stride=120)

        self.assertEqual(len(dataset.window_subjects), len(dataset))
        self.assertEqual(list(dataset.window_subjects), ["00", "00", "01", "01"])


class StreamingEvaluationTests(WindowingTestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.model = SleepStagingModel(config=_config())
        self.model.eval()
        self.device = torch.device("cpu")

    def test_streaming_covers_every_second_of_every_subject(self):
        result = predict_streaming(
            self.model, self.dir, ["00", "01"], seq_len=120, stream_stride=30,
            device=self.device)
        total = self.seconds["00"] + self.seconds["01"]

        self.assertEqual(len(result["y_pred"]), total)
        self.assertEqual(len(result["y_true"]), total)
        self.assertEqual(result["logits"].shape, (total, 5))
        self.assertEqual(sum(1 for s in result["subject"] if s == "00"),
                         self.seconds["00"])
        np.testing.assert_array_equal(result["second_index"][:self.seconds["00"]],
                                      np.arange(self.seconds["00"]))

    def test_streaming_targets_match_the_stored_labels(self):
        result = predict_streaming(
            self.model, self.dir, ["01"], seq_len=120, stream_stride=30, device=self.device)
        stored = np.load(os.path.join(self.dir, "subject_01.npz"))["y"]

        np.testing.assert_array_equal(result["y_true"], stored)

    def test_full_stride_reduces_to_the_windowed_tiling(self):
        """With stride == seq_len the rolling buffer is the archived evaluation again."""
        dataset = SleepDataset(self.dir, ["00"], seq_len=120, stride=120)
        windowed = predict_windowed(self.model, dataset, self.device)["y_pred"]
        streamed = predict_streaming(
            self.model, self.dir, ["00"], seq_len=120, stream_stride=120,
            device=self.device)["y_pred"]

        np.testing.assert_array_equal(streamed[:len(windowed)], windowed)

    def test_a_shorter_stride_changes_the_predictions_it_inherits_context_for(self):
        """Guards against a vacuous pass: more history must actually reach the model."""
        coarse = predict_streaming(
            self.model, self.dir, ["00"], seq_len=120, stream_stride=120,
            device=self.device)["y_pred"]
        fine = predict_streaming(
            self.model, self.dir, ["00"], seq_len=120, stream_stride=30,
            device=self.device)["y_pred"]

        # The opening window is identical in both; the rest is decided with more history.
        np.testing.assert_array_equal(coarse[:120], fine[:120])
        self.assertFalse(np.array_equal(coarse[120:], fine[120:]))

    def test_streaming_output_length_stays_epoch_aligned(self):
        """30 s aggregation reshapes the prediction stream, so it must divide evenly."""
        result = predict_streaming(
            self.model, self.dir, ["00", "01"], seq_len=120, stream_stride=45,
            device=self.device)

        self.assertEqual(len(result["y_pred"]) % SECONDS_PER_EPOCH, 0)

    def test_rejects_a_stride_longer_than_the_context(self):
        with self.assertRaisesRegex(ValueError, "stream_stride"):
            predict_streaming(self.model, self.dir, ["00"], seq_len=120, stream_stride=121,
                              device=self.device)


if __name__ == "__main__":
    unittest.main()
