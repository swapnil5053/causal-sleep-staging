"""Analyses that read archived predictions.

These scripts are the reason evaluation saves its predictions at all, so their arithmetic
has to be checked rather than eyeballed: a confidence interval or a detection latency is
only worth quoting if the code behind it is right.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from scripts.analyze_predictions import (apply_prior_correction, bootstrap_over_subjects,
                                         confusion, load_folds, metrics,
                                         metrics_from_confusion, per_subject_confusions)
from scripts.boundary_latency import epoch_majority, first_sustained, subject_transitions

N_CLASSES = 5
SECONDS_PER_EPOCH = 30


def write_predictions(path, subjects, y_true, y_pred, logits=None, fold=0):
    payload = dict(
        subject=np.asarray(subjects, dtype=str),
        y_true=np.asarray(y_true, dtype=np.int8),
        y_pred=np.asarray(y_pred, dtype=np.int8),
        fold=np.array(fold),
        seq_len=np.array(120),
        stream_stride=np.array(-1),
    )
    if logits is not None:
        payload["logits"] = np.asarray(logits, dtype=np.float16)
    np.savez_compressed(path, **payload)


class ConfusionMetricTests(unittest.TestCase):
    """The bootstrap sums 5x5 matrices instead of re-scoring seconds; it must still agree."""

    def setUp(self):
        rng = np.random.default_rng(11)
        self.y_true = rng.integers(0, N_CLASSES, size=4000)
        self.y_pred = np.where(rng.random(4000) < 0.6, self.y_true,
                               rng.integers(0, N_CLASSES, size=4000))

    def test_matches_sklearn(self):
        direct = metrics(self.y_true, self.y_pred)
        matrix = metrics_from_confusion(confusion(self.y_true, self.y_pred))

        for key in ("accuracy", "kappa", "macro_f1"):
            with self.subTest(metric=key):
                self.assertAlmostEqual(direct[key], matrix[key], places=10)

    def test_confusion_rows_are_true_labels(self):
        matrix = confusion(self.y_true, self.y_pred)

        np.testing.assert_array_equal(matrix.sum(axis=1),
                                      np.bincount(self.y_true, minlength=N_CLASSES))
        np.testing.assert_array_equal(matrix.sum(axis=0),
                                      np.bincount(self.y_pred, minlength=N_CLASSES))

    def test_matrices_are_additive_across_subjects(self):
        """Summing per-subject matrices must equal the matrix of the pooled data."""
        half = len(self.y_true) // 2
        pooled = confusion(self.y_true, self.y_pred)
        summed = (confusion(self.y_true[:half], self.y_pred[:half])
                  + confusion(self.y_true[half:], self.y_pred[half:]))

        np.testing.assert_array_equal(pooled, summed)

    def test_perfect_prediction_scores_one(self):
        result = metrics_from_confusion(confusion(self.y_true, self.y_true))

        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["kappa"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], 1.0)

    def test_empty_matrix_is_not_a_crash(self):
        result = metrics_from_confusion(np.zeros((N_CLASSES, N_CLASSES)))

        self.assertTrue(np.isnan(result["kappa"]))


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.codes = np.repeat(np.arange(8), 500).astype(np.int32)
        self.y_true = rng.integers(0, N_CLASSES, size=4000)
        self.y_pred = np.where(rng.random(4000) < 0.7, self.y_true,
                               rng.integers(0, N_CLASSES, size=4000))

    def test_interval_brackets_the_point_estimate(self):
        point = metrics(self.y_true, self.y_pred)
        interval = bootstrap_over_subjects(self.codes, self.y_true, self.y_pred,
                                           n_boot=500, seed=0)

        for key in ("accuracy", "kappa", "macro_f1"):
            low, high = interval[key]
            with self.subTest(metric=key):
                self.assertLessEqual(low, point[key])
                self.assertGreaterEqual(high, point[key])

    def test_is_reproducible_for_a_given_seed(self):
        first = bootstrap_over_subjects(self.codes, self.y_true, self.y_pred,
                                        n_boot=200, seed=3)
        second = bootstrap_over_subjects(self.codes, self.y_true, self.y_pred,
                                         n_boot=200, seed=3)

        self.assertEqual(first, second)

    def test_a_single_subject_gives_no_interval(self):
        """One subject carries no between-subject information, so refuse rather than lie."""
        lone = np.zeros(100, dtype=np.int32)

        self.assertIsNone(bootstrap_over_subjects(lone, self.y_true[:100],
                                                  self.y_pred[:100], n_boot=50))

    def test_per_subject_matrices_match_direct_scoring(self):
        matrices = per_subject_confusions(self.codes, self.y_true, self.y_pred, 8)

        for code in range(8):
            mask = self.codes == code
            np.testing.assert_array_equal(matrices[code],
                                          confusion(self.y_true[mask], self.y_pred[mask]))


class PriorCorrectionTests(unittest.TestCase):
    def test_shifting_toward_a_class_makes_it_more_likely(self):
        logits = np.zeros((1, N_CLASSES))
        uniform = np.full(N_CLASSES, 0.2)
        skewed = np.array([0.9, 0.025, 0.025, 0.025, 0.025])

        corrected = apply_prior_correction(logits, uniform, skewed)

        self.assertEqual(int(corrected.argmax()), 0)

    def test_identical_priors_leave_predictions_untouched(self):
        rng = np.random.default_rng(2)
        logits = rng.normal(size=(50, N_CLASSES))
        prior = np.array([0.4, 0.1, 0.3, 0.1, 0.1])

        corrected = apply_prior_correction(logits, prior, prior)

        np.testing.assert_allclose(corrected, logits, atol=1e-12)


class LoadFoldsTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="cap-preds-")
        for fold in range(2):
            write_predictions(os.path.join(self.dir, f"fold_{fold}_predictions.npz"),
                              subjects=["00"] * 60 + ["01"] * 60,
                              y_true=np.zeros(120), y_pred=np.zeros(120), fold=fold)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_reads_every_fold_in_a_directory(self):
        folds = load_folds([self.dir])

        self.assertEqual(len(folds), 2)
        self.assertEqual(sorted(f["fold"] for f in folds), [0, 1])

    def test_subject_ids_are_namespaced_so_folds_cannot_collide(self):
        """Subject "00" appears in both files and must not be merged into one record."""
        folds = load_folds([self.dir])
        codes = np.concatenate([f["subject_code"] for f in folds])

        self.assertEqual(len(np.unique(codes)), 4)

    def test_subjects_do_not_collide_across_directories(self):
        """Two runs each holding subject "00" are two records, not one."""
        other = tempfile.mkdtemp(prefix="cap-preds2-")
        try:
            write_predictions(os.path.join(other, "fold_0_predictions.npz"),
                              subjects=["00"] * 60 + ["01"] * 60,
                              y_true=np.zeros(120), y_pred=np.zeros(120), fold=0)
            folds = load_folds([self.dir, other])
            codes = np.concatenate([f["subject_code"] for f in folds])

            self.assertEqual(len(np.unique(codes)), 6)
        finally:
            shutil.rmtree(other, ignore_errors=True)

    def test_the_same_file_is_never_loaded_twice(self):
        """A directory plus one of its own files must not double-count that file."""
        one_file = os.path.join(self.dir, "fold_0_predictions.npz")

        self.assertEqual(len(load_folds([self.dir, one_file])), 2)

    def test_missing_path_raises_rather_than_returning_nothing(self):
        with self.assertRaises(FileNotFoundError):
            load_folds([os.path.join(self.dir, "does_not_exist")])

    def test_a_file_without_predictions_is_rejected(self):
        broken = os.path.join(self.dir, "fold_9_predictions.npz")
        np.savez_compressed(broken, subject=np.array(["00"]))

        with self.assertRaisesRegex(ValueError, "missing"):
            load_folds([broken])


class BoundaryLatencyTests(unittest.TestCase):
    def test_epoch_majority_votes_within_each_epoch(self):
        predictions = np.concatenate([np.full(20, 0), np.full(10, 2),   # epoch 0 -> 0
                                      np.full(5, 1), np.full(25, 3)])   # epoch 1 -> 3

        np.testing.assert_array_equal(epoch_majority(predictions), [0, 3])

    def test_epoch_majority_ignores_a_partial_trailing_epoch(self):
        predictions = np.zeros(75, dtype=int)

        self.assertEqual(len(epoch_majority(predictions)), 2)

    def test_first_sustained_needs_the_whole_hold_window(self):
        stream = np.array([0, 1, 1, 0, 1, 1, 1, 1])

        self.assertEqual(first_sustained(stream, 1, 0, len(stream), hold=3), 4)
        self.assertIsNone(first_sustained(stream, 2, 0, len(stream), hold=1))

    def test_detects_a_change_before_the_epoch_system_can_report_it(self):
        """Hand-checked case: model switches at second 50, scored change is at second 60.

        The run starts at 50 and must be held for 5 s, so the earliest causal report is
        second 54, which is 6 s before the scored change. Crediting second 50 would charge
        the per-second system nothing for evidence while charging the epoch system a whole
        epoch, and would overstate the comparison by hold - 1 seconds.
        """
        y_true = np.concatenate([np.full(60, 0), np.full(60, 2)])
        y_pred = np.concatenate([np.full(50, 0), np.full(70, 2)])

        rows = subject_transitions(y_true, y_pred, hold=5, max_early=30, max_late=60)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["stage"], 2)
        self.assertEqual(rows[0]["per_second"], -6)    # confirmed at second 54
        self.assertEqual(rows[0]["epoch"], 30)         # epoch 2 lands at second 90
        self.assertFalse(rows[0]["already_active"])

    def test_the_hold_window_is_charged_to_the_per_second_system(self):
        """A longer confirmation window must push the reported detection later."""
        y_true = np.concatenate([np.full(60, 0), np.full(60, 2)])
        y_pred = np.concatenate([np.full(50, 0), np.full(70, 2)])

        short = subject_transitions(y_true, y_pred, hold=1, max_early=30, max_late=60)
        long = subject_transitions(y_true, y_pred, hold=11, max_early=30, max_late=60)

        self.assertEqual(short[0]["per_second"], -10)
        self.assertEqual(long[0]["per_second"], 0)

    def test_flags_a_stage_the_model_was_already_calling(self):
        """A model stuck on the incoming stage must not be credited with early detection."""
        y_true = np.concatenate([np.full(60, 0), np.full(60, 2)])
        y_pred = np.full(120, 2)

        rows = subject_transitions(y_true, y_pred, hold=5, max_early=30, max_late=60)

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["already_active"])

    def test_flags_a_run_that_starts_partway_through_the_window(self):
        """The guard looks at the whole run-up, not just its first few seconds."""
        y_true = np.concatenate([np.full(60, 0), np.full(60, 2)])
        # Model switches at second 35, i.e. 25 s of the 30 s run-up already call stage 2.
        y_pred = np.concatenate([np.full(35, 0), np.full(85, 2)])

        rows = subject_transitions(y_true, y_pred, hold=5, max_early=30, max_late=60)

        self.assertTrue(rows[0]["already_active"])
        self.assertAlmostEqual(rows[0]["prior_share"], 25 / 30)

    def test_a_genuinely_early_detection_is_not_flagged(self):
        y_true = np.concatenate([np.full(60, 0), np.full(60, 2)])
        y_pred = np.concatenate([np.full(55, 0), np.full(65, 2)])

        rows = subject_transitions(y_true, y_pred, hold=5, max_early=30, max_late=60)

        self.assertFalse(rows[0]["already_active"])

    def test_changes_at_a_splice_are_skipped(self):
        """A label change where a night was joined is not a transition in time."""
        y_true = np.concatenate([np.full(60, 0), np.full(60, 2)])
        y_pred = np.concatenate([np.full(50, 0), np.full(70, 2)])
        segment_start = np.zeros(120, dtype=bool)
        segment_start[60] = True

        with_flag = subject_transitions(y_true, y_pred, 5, 30, 60,
                                        segment_start=segment_start)
        without = subject_transitions(y_true, y_pred, 5, 30, 60)

        self.assertEqual(with_flag, [])
        self.assertEqual(len(without), 1)

    def test_reports_no_detection_when_the_stage_never_appears(self):
        y_true = np.concatenate([np.full(60, 0), np.full(60, 2)])
        y_pred = np.zeros(120, dtype=int)

        rows = subject_transitions(y_true, y_pred, hold=5, max_early=30, max_late=60)

        self.assertIsNone(rows[0]["per_second"])
        self.assertIsNone(rows[0]["epoch"])

    def test_a_recording_without_changes_yields_nothing(self):
        y_true = np.zeros(300, dtype=int)

        self.assertEqual(subject_transitions(y_true, y_true, 5, 30, 60), [])

    def test_changes_without_room_for_the_search_window_are_skipped(self):
        """A change 10 s from the end cannot be searched fairly, so it is dropped."""
        y_true = np.concatenate([np.full(60, 0), np.full(30, 2)])
        y_pred = y_true.copy()

        self.assertEqual(subject_transitions(y_true, y_pred, 5, 30, 60), [])


if __name__ == "__main__":
    unittest.main()
