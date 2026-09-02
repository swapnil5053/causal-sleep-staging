"""A recovered split is only useful if it is provably the split the model was trained on.

The recovery regenerates a partition and checks it against the test-subject list each archived
report already carries. These tests cover the check itself, and the refusal to write when it
fails - a plausible-looking split that is not the real one would be worse than no file.
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import yaml  # noqa: E402

from src.data.dataset import get_cv_splits  # noqa: E402
from recover_splits import (  # noqa: E402
    DEFAULT_RUNS,
    archived_test_subjects,
    reconstruct,
    write_split,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_report(directory, fold, test_subjects):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"fold_{fold}_test_report.txt"), "w") as handle:
        handle.write(f"Fold {fold} held-out test results\n")
        handle.write(f"Test subjects: {test_subjects!r}\n")
        handle.write("Cohen's Kappa:    0.6237\n")


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.subjects = [f"{i:02d}" for i in range(40)]
        self.seed = 43
        for fold in range(5):
            _, _, test = get_cv_splits(self.subjects, 5, fold, self.seed)
            write_report(self.directory, fold, test)

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_reads_the_archived_test_subjects(self):
        archived = archived_test_subjects(self.directory)
        self.assertEqual(sorted(archived), [0, 1, 2, 3, 4])
        self.assertEqual(sorted(s for group in archived.values() for s in group), self.subjects)

    def test_correct_seed_reproduces_every_fold(self):
        splits, subjects, mismatches = reconstruct(self.directory, self.seed)
        self.assertEqual(mismatches, [])
        self.assertEqual(subjects, self.subjects)
        for fold, (train, val, test) in splits.items():
            self.assertEqual((train, val, test), get_cv_splits(self.subjects, 5, fold, self.seed))
            self.assertTrue(set(train).isdisjoint(val))
            self.assertTrue(set(train).isdisjoint(test))
            self.assertEqual(set(train + val + test), set(self.subjects))

    def test_wrong_seed_is_detected_rather_than_written(self):
        _, _, mismatches = reconstruct(self.directory, self.seed + 1)
        self.assertTrue(mismatches, "a wrong seed must not silently reproduce the folds")

    def test_missing_test_subject_line_is_an_error(self):
        with open(os.path.join(self.directory, "fold_0_test_report.txt"), "w") as handle:
            handle.write("Fold 0 held-out test results\nCohen's Kappa: 0.62\n")
        with self.assertRaises(ValueError):
            archived_test_subjects(self.directory)

    def test_written_file_round_trips_and_records_the_seed(self):
        splits, _, _ = reconstruct(self.directory, self.seed)
        train, val, test = splits[0]
        path, did_write = write_split(self.directory, 0, train, val, test, self.seed, force=False)
        self.assertTrue(did_write)

        loaded = yaml.safe_load(open(path))
        self.assertEqual(loaded["test_subjects"], test)
        self.assertEqual(loaded["train_subjects"], train)
        self.assertEqual(loaded["val_subjects"], val)
        self.assertEqual(loaded["split_seed"], self.seed)

    def test_an_existing_file_is_not_overwritten_without_force(self):
        splits, _, _ = reconstruct(self.directory, self.seed)
        train, val, test = splits[0]
        write_split(self.directory, 0, train, val, test, self.seed, force=False)

        _, did_write = write_split(self.directory, 0, [], [], [], self.seed, force=False)
        self.assertFalse(did_write)
        path = os.path.join(self.directory, "split_fold_0.yaml")
        self.assertEqual(yaml.safe_load(open(path))["test_subjects"], test)

        _, did_write = write_split(self.directory, 0, [], [], [], self.seed, force=True)
        self.assertTrue(did_write)


class ArchivedRunTests(unittest.TestCase):
    """The real thing: every archived streaming run must reproduce from its own seed."""

    def test_every_archived_streaming_run_reproduces(self):
        checked = 0
        for run_dir, seed in DEFAULT_RUNS:
            path = os.path.join(REPO, run_dir)
            if not os.path.isdir(path):
                continue
            with self.subTest(run=run_dir):
                splits, subjects, mismatches = reconstruct(path, seed)
                self.assertEqual(mismatches, [], f"{run_dir} does not reproduce at seed {seed}")
                self.assertEqual(len(subjects), 78)
                self.assertEqual(len(splits), 5)
                checked += 1
        self.assertTrue(checked, "no archived streaming runs found to check")


if __name__ == "__main__":
    unittest.main()
