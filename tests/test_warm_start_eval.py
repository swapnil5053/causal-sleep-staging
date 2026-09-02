"""Warm and cold scoring must cover exactly the same seconds, differing only in context.

If the two modes scored different seconds, the difference between them would confound context
with which parts of the night were included, and the whole measurement would be worthless.
These tests pin the second-level bookkeeping down on synthetic arrays, where the correct answer
is known, rather than on model output.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from warm_start_eval import cold_seconds, common_span, score, warm_seconds  # noqa: E402

SEQ_LEN = 12
STRIDE = 3


def ramp_windows(n_windows, start, step):
    """Windows whose values are the absolute second index, so misalignment is visible."""
    return [np.arange(start + i * step, start + i * step + SEQ_LEN) for i in range(n_windows)]


class ColdAssemblyTests(unittest.TestCase):
    def test_windows_concatenate_in_order_per_subject(self):
        subjects = ["07", "07", "19"]
        windows = ramp_windows(3, 0, SEQ_LEN)
        windows[2] = np.arange(SEQ_LEN)                      # subject 19 restarts at 0

        out = cold_seconds(subjects, windows, SEQ_LEN)
        np.testing.assert_array_equal(out["07"], np.arange(2 * SEQ_LEN))
        np.testing.assert_array_equal(out["19"], np.arange(SEQ_LEN))

    def test_wrong_window_length_is_rejected(self):
        with self.assertRaises(ValueError):
            cold_seconds(["07"], [np.arange(SEQ_LEN - 1)], SEQ_LEN)

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            cold_seconds(["07", "19"], [np.arange(SEQ_LEN)], SEQ_LEN)


class WarmAssemblyTests(unittest.TestCase):
    def test_kept_positions_are_the_absolute_seconds_they_claim(self):
        """Window i spans [i*stride, i*stride+seq_len); its tail must be the right seconds."""
        n = 5
        subjects = ["07"] * n
        windows = ramp_windows(n, 0, STRIDE)

        offset, out = warm_seconds(subjects, windows, SEQ_LEN, STRIDE)
        self.assertEqual(offset, SEQ_LEN - STRIDE)
        expected = np.arange(offset, offset + n * STRIDE)
        np.testing.assert_array_equal(out["07"], expected)

    def test_every_scored_second_carries_the_minimum_context(self):
        """The first kept position of a window sits `seq_len - stride` in from its start."""
        offset, _ = warm_seconds(["07"], ramp_windows(1, 0, STRIDE), SEQ_LEN, STRIDE)
        self.assertEqual(offset, SEQ_LEN - STRIDE)
        self.assertGreaterEqual(offset, 1, "a warm stride must leave some context")

    def test_stride_equal_to_seq_len_degenerates_to_cold(self):
        subjects = ["07", "07"]
        windows = ramp_windows(2, 0, SEQ_LEN)
        offset, warm = warm_seconds(subjects, windows, SEQ_LEN, SEQ_LEN)
        cold = cold_seconds(subjects, windows, SEQ_LEN)
        self.assertEqual(offset, 0)
        np.testing.assert_array_equal(warm["07"], cold["07"])

    def test_invalid_strides_are_rejected(self):
        for stride in (0, -1, SEQ_LEN + 1):
            with self.subTest(stride=stride), self.assertRaises(ValueError):
                warm_seconds(["07"], ramp_windows(1, 0, STRIDE), SEQ_LEN, stride)


class CommonSpanTests(unittest.TestCase):
    def test_both_modes_end_up_on_identical_absolute_seconds(self):
        n_warm = 6
        cold = cold_seconds(["07"] * 2, ramp_windows(2, 0, SEQ_LEN), SEQ_LEN)
        offset, warm = warm_seconds(["07"] * n_warm, ramp_windows(n_warm, 0, STRIDE),
                                    SEQ_LEN, STRIDE)

        cold_c, warm_c = common_span(cold, warm, offset)
        self.assertEqual(len(cold_c["07"]), len(warm_c["07"]))
        # Both arrays carry the absolute second index, so equality proves alignment.
        np.testing.assert_array_equal(cold_c["07"], warm_c["07"])

    def test_a_subject_warm_never_reached_is_dropped_not_misaligned(self):
        cold = {"07": np.arange(SEQ_LEN)}
        warm = {"07": np.arange(0)}
        cold_c, warm_c = common_span(cold, warm, SEQ_LEN - STRIDE)
        self.assertEqual(cold_c, {})
        self.assertEqual(warm_c, {})

    def test_truncates_to_the_shorter_of_the_two(self):
        cold = {"07": np.arange(20)}
        warm = {"07": np.arange(9, 60)}                       # runs past cold
        cold_c, warm_c = common_span(cold, warm, 9)
        self.assertEqual(len(cold_c["07"]), 11)
        np.testing.assert_array_equal(cold_c["07"], warm_c["07"])


class ScoreTests(unittest.TestCase):
    def test_perfect_predictions_score_one(self):
        targets = np.array([0, 1, 2, 3, 4, 0, 1])
        m = score(targets, targets)
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["kappa"], 1.0)
        self.assertEqual(m["n_seconds"], 7)

    def test_per_class_f1_covers_all_five_stages(self):
        targets = np.array([0, 0, 1, 1])
        self.assertEqual(len(score(targets, targets)["per_class_f1"]), 5)


if __name__ == "__main__":
    unittest.main()
