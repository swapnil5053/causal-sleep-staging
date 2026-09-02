"""Calibration and transition-response arithmetic, on cases with a known answer.

Both scripts will eventually run on real predictions nobody can check by eye, so the arithmetic
is pinned here against hand-constructed inputs: a deliberately over-confident model, a model
that responds to a transition after a known number of seconds, and the degenerate cases that
would otherwise produce a confident-looking number from nothing.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from calibration import (  # noqa: E402
    analyse,
    brier_score,
    calibration_error,
    reliability,
)
from transition_response import (  # noqa: E402
    NOT_DETECTED,
    epoch_majority,
    find_transitions,
    paired_gain,
    response_latency,
    summarise,
)


def onehot_probs(preds, confidence, num_classes=5):
    """Probabilities that predict `preds` with exactly `confidence`, rest spread evenly."""
    preds = np.asarray(preds)
    rest = (1.0 - confidence) / (num_classes - 1)
    p = np.full((len(preds), num_classes), rest)
    p[np.arange(len(preds)), preds] = confidence
    return p


class ReliabilityTests(unittest.TestCase):
    def test_perfect_calibration_has_zero_error(self):
        # 100 predictions at confidence 0.8, exactly 80 of them correct
        confidence = np.full(100, 0.8)
        correct = np.array([True] * 80 + [False] * 20)
        rows = reliability(confidence, correct, n_bins=10)
        ece, mce = calibration_error(rows, 100)
        self.assertAlmostEqual(ece, 0.0, places=9)
        self.assertAlmostEqual(mce, 0.0, places=9)

    def test_over_confidence_is_reported_with_the_right_magnitude(self):
        confidence = np.full(100, 0.9)
        correct = np.array([True] * 50 + [False] * 50)      # 0.9 claimed, 0.5 achieved
        ece, mce = calibration_error(reliability(confidence, correct), 100)
        self.assertAlmostEqual(ece, 0.4, places=9)
        self.assertAlmostEqual(mce, 0.4, places=9)

    def test_empty_bins_are_omitted_not_counted_as_zero(self):
        confidence = np.full(10, 0.95)
        rows = reliability(confidence, np.ones(10, dtype=bool), n_bins=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][2], 10)

    def test_confidence_of_exactly_one_lands_in_the_top_bin(self):
        rows = reliability(np.array([1.0]), np.array([True]), n_bins=10)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0][1], 1.0)

    def test_no_predictions_is_not_an_error(self):
        self.assertEqual(reliability(np.array([]), np.array([], dtype=bool)), [])
        self.assertTrue(np.isnan(calibration_error([], 0)[0]))

    def test_mismatched_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            reliability(np.zeros(3), np.zeros(4, dtype=bool))


class BrierTests(unittest.TestCase):
    def test_perfect_confident_prediction_scores_zero(self):
        targets = np.array([0, 3, 4])
        p = np.zeros((3, 5))
        p[np.arange(3), targets] = 1.0
        self.assertAlmostEqual(brier_score(p, targets), 0.0)

    def test_confidently_wrong_scores_two(self):
        targets = np.array([0])
        p = np.zeros((1, 5))
        p[0, 1] = 1.0
        self.assertAlmostEqual(brier_score(p, targets), 2.0)


class AnalyseTests(unittest.TestCase):
    def test_over_confident_model_is_flagged_by_the_summary(self):
        preds = np.array([0] * 100)
        targets = np.array([0] * 50 + [1] * 50)
        r = analyse(onehot_probs(preds, 0.9), targets)
        self.assertAlmostEqual(r["accuracy"], 0.5)
        self.assertAlmostEqual(r["mean_confidence"], 0.9)
        self.assertGreater(r["mean_confidence"] - r["accuracy"], 0.3)
        self.assertEqual(len(r["per_stage"]), 5)


class TransitionTests(unittest.TestCase):
    def test_transitions_are_the_indices_of_the_new_stage(self):
        targets = np.array([2, 2, 2, 4, 4, 0])
        np.testing.assert_array_equal(find_transitions(targets), [3, 5])

    def test_a_subject_boundary_is_not_a_transition(self):
        targets = np.array([2, 2, 4, 4])
        subjects = np.array(["07", "07", "19", "19"])
        self.assertEqual(len(find_transitions(targets, subjects)), 0)

    def test_latency_counts_seconds_until_the_new_stage_appears(self):
        targets = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        preds = np.array([0, 0, 0, 0, 0, 1, 1, 1])          # switches 2 s late
        transitions = find_transitions(targets)
        np.testing.assert_array_equal(
            response_latency(preds, transitions, targets, horizon=10), [2])

    def test_an_immediate_switch_is_zero_latency(self):
        targets = np.array([0, 0, 1, 1])
        preds = np.array([0, 0, 1, 1])
        self.assertEqual(response_latency(preds, find_transitions(targets), targets, 10)[0], 0)

    def test_a_miss_is_marked_not_detected_rather_than_scored(self):
        targets = np.array([0, 0, 1, 1, 1])
        preds = np.array([0, 0, 0, 0, 0])
        latencies = response_latency(preds, find_transitions(targets), targets, horizon=10)
        self.assertEqual(latencies[0], NOT_DETECTED)
        self.assertEqual(summarise(latencies)["detected"], 0)

    def test_a_later_transition_cannot_lend_credit_to_an_earlier_one(self):
        # stage 1 is never predicted during its own span; the 1 after the next transition
        # must not be counted as a late detection of it
        targets = np.array([0, 0, 1, 1, 2, 2, 2])
        preds = np.array([0, 0, 0, 0, 1, 1, 1])
        latencies = response_latency(preds, find_transitions(targets), targets, horizon=10)
        self.assertEqual(latencies[0], NOT_DETECTED)

    def test_summary_statistics_ignore_misses(self):
        s = summarise(np.array([1, 3, NOT_DETECTED, 5]))
        self.assertEqual(s["transitions"], 4)
        self.assertEqual(s["detected"], 3)
        self.assertAlmostEqual(s["median"], 3.0)
        self.assertAlmostEqual(s["within_5s"], 1.0)


class PairedGainTests(unittest.TestCase):
    """Detection rates differ between series, so the comparison has to be paired."""

    def test_only_transitions_both_detected_are_compared(self):
        per_second = np.array([1, 2, NOT_DETECTED, 4])
        epoch = np.array([5, NOT_DETECTED, 7, 10])
        gain = paired_gain(per_second, epoch)
        self.assertEqual(gain["n"], 2)                    # indices 0 and 3 only
        self.assertAlmostEqual(gain["median_gain"], 5.0)  # (5-1) and (10-4)
        self.assertAlmostEqual(gain["earlier_fraction"], 1.0)

    def test_a_miss_in_either_series_cannot_inflate_the_gain(self):
        # If misses leaked in as their sentinel value the gain would be wildly negative.
        gain = paired_gain(np.array([1, NOT_DETECTED]), np.array([3, 50]))
        self.assertEqual(gain["n"], 1)
        self.assertAlmostEqual(gain["median_gain"], 2.0)

    def test_no_common_detections_reports_nothing_rather_than_zero(self):
        gain = paired_gain(np.array([NOT_DETECTED]), np.array([4]))
        self.assertEqual(gain["n"], 0)
        self.assertTrue(np.isnan(gain["median_gain"]))

    def test_misaligned_arrays_are_rejected(self):
        with self.assertRaises(ValueError):
            paired_gain(np.array([1, 2]), np.array([1]))


class EpochMajorityTests(unittest.TestCase):
    def test_a_whole_epoch_takes_its_majority_stage(self):
        preds = np.array([0] * 10 + [2] * 20)               # 30 s, majority is 2
        np.testing.assert_array_equal(epoch_majority(preds), np.full(30, 2))

    def test_epoch_voting_delays_the_response(self):
        """The point of the comparison: 30 s reading cannot respond mid-epoch."""
        targets = np.concatenate([np.zeros(30, dtype=int), np.ones(60, dtype=int)])
        preds = targets.copy()                               # per-second model is perfect
        transitions = find_transitions(targets)
        per_second = response_latency(preds, transitions, targets, 60)[0]
        epoch = response_latency(epoch_majority(preds), transitions, targets, 60)[0]
        self.assertEqual(per_second, 0)
        self.assertGreaterEqual(epoch, per_second)

    def test_a_trailing_partial_epoch_is_passed_through(self):
        preds = np.array([1] * 30 + [3] * 7)
        out = epoch_majority(preds)
        self.assertEqual(len(out), 37)
        np.testing.assert_array_equal(out[30:], [3] * 7)


if __name__ == "__main__":
    unittest.main()
