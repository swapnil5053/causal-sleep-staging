"""Per-subject scoring has to attribute every prediction to the right night.

The per-subject numbers are only worth having if the attribution is exact, so these tests
check the mapping from windows to subjects and the guards on the paired test that stop it
comparing rows which are not actually paired.
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from src.data.dataset import SleepDataset  # noqa: E402
from src.eval.evaluate import format_subject_row, per_subject_metrics  # noqa: E402
from subject_paired_test import analyse, read_subject_metrics  # noqa: E402

HEADER = "fold,subject,n_seconds,accuracy,kappa,macro_f1,f1_W,f1_N1,f1_N2,f1_N3,f1_REM\n"


def write_subject_csv(directory, rows):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "test_subject_metrics.csv"), "w") as handle:
        handle.write(HEADER)
        for fold, subject, kappa in rows:
            handle.write(f"{fold},{subject},1000,0.8000,{kappa},0.7000,"
                         f"0.7,0.4,0.8,0.7,0.6\n")


class WindowAttributionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        self.lengths = {"07": 300, "19": 180}
        for subject, seconds in self.lengths.items():
            np.savez(
                os.path.join(self.directory, f"subject_{subject}.npz"),
                x=rng.normal(size=(seconds, 100)).astype(np.float32),
                y=rng.integers(0, 5, size=seconds).astype(np.int64),
                normalization_method=np.array("causal_rolling"),
                normalization_window_seconds=np.array(30),
            )

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_one_subject_label_per_window_in_dataset_order(self):
        seq_len = 60
        dataset = SleepDataset(self.directory, ["07", "19"], seq_len=seq_len, stride=seq_len)

        self.assertEqual(len(dataset.window_subjects), len(dataset))
        expected = ["07"] * (self.lengths["07"] // seq_len) + ["19"] * (self.lengths["19"] // seq_len)
        self.assertEqual(list(dataset.window_subjects), expected)

    def test_expanded_labels_line_up_with_the_predictions(self):
        """The exact expansion evaluate.py performs, on the same objects."""
        seq_len = 60
        dataset = SleepDataset(self.directory, ["07", "19"], seq_len=seq_len, stride=seq_len)
        per_second = np.repeat(dataset.window_subjects, seq_len)

        self.assertEqual(len(per_second), len(dataset) * seq_len)
        self.assertEqual(int((per_second == "07").sum()),
                         (self.lengths["07"] // seq_len) * seq_len)


class PerSubjectMetricTests(unittest.TestCase):
    def test_subjects_are_scored_independently(self):
        subjects = np.array(["07"] * 4 + ["19"] * 4)
        targets = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        preds = np.array([0, 1, 2, 3, 0, 0, 0, 0])       # perfect, then poor

        rows = {row["subject"]: row for row in per_subject_metrics(subjects, targets, preds)}
        self.assertEqual(set(rows), {"07", "19"})
        self.assertAlmostEqual(rows["07"]["accuracy"], 1.0)
        self.assertAlmostEqual(rows["19"]["accuracy"], 0.25)
        self.assertGreater(rows["07"]["kappa"], rows["19"]["kappa"])
        self.assertEqual(rows["07"]["n_seconds"], 4)

    def test_single_stage_night_has_no_kappa_and_writes_an_empty_field(self):
        subjects = np.array(["07"] * 4)
        targets = np.array([2, 2, 2, 2])
        preds = np.array([2, 2, 2, 3])

        row = per_subject_metrics(subjects, targets, preds)[0]
        self.assertTrue(np.isnan(row["kappa"]))
        self.assertEqual(format_subject_row(0, row).split(",")[4], "")

    def test_subject_order_follows_the_test_set(self):
        subjects = np.array(["19"] * 2 + ["07"] * 2)
        targets = np.array([0, 1, 0, 1])
        rows = per_subject_metrics(subjects, targets, targets)
        self.assertEqual([r["subject"] for r in rows], ["19", "07"])


class PairedTestGuardTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.causal = os.path.join(self.root, "causal")
        self.noncausal = os.path.join(self.root, "noncausal")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_pairs_by_subject_and_reports_the_difference(self):
        write_subject_csv(self.causal, [(0, "07", "0.60"), (0, "19", "0.70"), (1, "23", "0.65")])
        write_subject_csv(self.noncausal, [(0, "07", "0.64"), (0, "19", "0.72"), (1, "23", "0.66")])

        result = analyse("t", self.causal, self.noncausal, "kappa")
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mean"], (-0.04 - 0.02 - 0.01) / 3, places=6)
        self.assertEqual(result["loses"], 3)

    def test_mismatched_subjects_are_refused(self):
        write_subject_csv(self.causal, [(0, "07", "0.60"), (0, "19", "0.70")])
        write_subject_csv(self.noncausal, [(0, "07", "0.64"), (0, "23", "0.72")])

        with self.assertRaises(SystemExit) as caught:
            analyse("t", self.causal, self.noncausal, "kappa")
        self.assertIn("not paired", str(caught.exception))

    def test_different_partitions_are_refused(self):
        write_subject_csv(self.causal, [(0, "07", "0.60"), (0, "19", "0.70")])
        write_subject_csv(self.noncausal, [(0, "07", "0.64"), (3, "19", "0.72")])

        with self.assertRaises(SystemExit) as caught:
            analyse("t", self.causal, self.noncausal, "kappa")
        self.assertIn("different folds", str(caught.exception))

    def test_duplicated_rows_are_refused(self):
        write_subject_csv(self.causal, [(0, "07", "0.60"), (0, "07", "0.61")])
        with self.assertRaises(SystemExit) as caught:
            read_subject_metrics(self.causal, "kappa")
        self.assertIn("more than once", str(caught.exception))

    def test_empty_kappa_rows_are_skipped_not_read_as_zero(self):
        write_subject_csv(self.causal, [(0, "07", "0.60"), (0, "19", "")])
        values, _ = read_subject_metrics(self.causal, "kappa")
        self.assertEqual(set(values), {"07"})


if __name__ == "__main__":
    unittest.main()
