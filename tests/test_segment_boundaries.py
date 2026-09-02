"""A window must not span a join between two recordings, or a dropped-epoch gap.

Unscored epochs are removed and a subject's two nights are concatenated, so consecutive rows of
the stored array are not always consecutive in time. A window covering such a join asks the
model to read across a jump that never happened, and at 120 s of context that is a sizeable
fraction of the training set. These tests cover the detection and, just as importantly, that
files predating the metadata still load exactly as before.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from src.data.dataset import SleepDataset


def write_subject(directory, subject, seconds, segment_starts=None):
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(abs(hash(subject)) % 2**31)
    payload = {
        "x": rng.normal(size=(seconds, 100)).astype(np.float32),
        "y": rng.integers(0, 5, size=seconds).astype(np.int64),
        "normalization_method": np.array("causal_rolling"),
        "normalization_window_seconds": np.array(30),
    }
    if segment_starts is not None:
        payload["segment_starts"] = np.asarray(segment_starts, dtype=np.int64)
    np.savez(os.path.join(directory, f"subject_{subject}.npz"), **payload)


class SegmentAwareWindowingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_exactly_the_straddling_windows_are_dropped(self):
        """Not just the count: the surviving windows must be the right ones."""
        seconds, seq_len, stride, boundary = 200, 20, 10, 95
        write_subject(self.directory, "07", seconds, segment_starts=[0, boundary])
        dataset = SleepDataset(self.directory, ["07"], seq_len=seq_len, stride=stride,
                               respect_segments=True)

        source = np.load(os.path.join(self.directory, "subject_07.npz"))["x"]
        all_starts = list(range(0, seconds - seq_len + 1, stride))
        expected_kept = [s for s in all_starts if not s < boundary < s + seq_len]
        expected_dropped = [s for s in all_starts if s < boundary < s + seq_len]

        self.assertEqual(expected_dropped, [80, 90])
        self.assertEqual(dataset.windows_dropped_at_segment_boundaries, len(expected_dropped))
        self.assertEqual(len(dataset), len(expected_kept))

        # Recover each surviving window's true start by matching its contents.
        recovered = []
        for window in dataset.windows:
            matches = [s for s in all_starts if np.array_equal(source[s:s + seq_len], window)]
            self.assertEqual(len(matches), 1, "window content must identify one start")
            recovered.append(matches[0])
        self.assertEqual(recovered, expected_kept)

    def test_permissive_mode_keeps_every_window(self):
        write_subject(self.directory, "07", 200, segment_starts=[0, 95])
        strict = SleepDataset(self.directory, ["07"], seq_len=20, stride=10,
                              respect_segments=True)
        loose = SleepDataset(self.directory, ["07"], seq_len=20, stride=10,
                             respect_segments=False)
        self.assertLess(len(strict), len(loose))
        self.assertEqual(loose.windows_dropped_at_segment_boundaries, 0)

    def test_a_boundary_exactly_on_a_window_edge_is_allowed(self):
        """A window ending where the next segment begins does not span the join."""
        write_subject(self.directory, "07", 120, segment_starts=[0, 60])
        dataset = SleepDataset(self.directory, ["07"], seq_len=60, stride=60,
                               respect_segments=True)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.windows_dropped_at_segment_boundaries, 0)

    def test_the_leading_zero_boundary_is_not_treated_as_a_join(self):
        write_subject(self.directory, "07", 120, segment_starts=[0])
        dataset = SleepDataset(self.directory, ["07"], seq_len=30, stride=30,
                               respect_segments=True)
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.windows_dropped_at_segment_boundaries, 0)

    def test_archived_files_without_metadata_load_unchanged(self):
        """The compatibility guarantee: no segment key means the old behaviour, exactly."""
        write_subject(self.directory, "07", 120, segment_starts=None)
        strict = SleepDataset(self.directory, ["07"], seq_len=30, stride=15,
                              respect_segments=True)
        loose = SleepDataset(self.directory, ["07"], seq_len=30, stride=15,
                             respect_segments=False)

        self.assertEqual(len(strict), len(loose))
        self.assertEqual(strict.windows_dropped_at_segment_boundaries, 0)
        self.assertEqual(strict.subjects_without_segment_metadata, ["07"])
        np.testing.assert_array_equal(strict.windows, loose.windows)

    def test_subject_attribution_survives_dropping(self):
        write_subject(self.directory, "07", 200, segment_starts=[0, 95])
        write_subject(self.directory, "19", 120, segment_starts=[0])
        dataset = SleepDataset(self.directory, ["07", "19"], seq_len=20, stride=10)

        self.assertEqual(len(dataset.window_subjects), len(dataset))
        self.assertEqual(set(dataset.window_subjects), {"07", "19"})


if __name__ == "__main__":
    unittest.main()
