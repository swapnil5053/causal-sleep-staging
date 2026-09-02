"""The cross-validation partition must be controlled separately from initialisation.

Before `split_seed` existed, `train.seed` did both jobs, so a "seed 43 replication" also
moved every subject between folds. These tests pin the separation down: that the shipped
configs agree on one partition, that the code honours it, and that omitting the key still
reproduces the archived behaviour exactly.
"""
import glob
import os
import unittest

import yaml

from src.data.dataset import get_cv_splits
from src.train.train import resolve_seeds

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


class SeedSeparationTests(unittest.TestCase):
    def setUp(self):
        self.subject_ids = [f"{subject:02d}" for subject in range(78)]
        self.configs = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))
        self.assertTrue(self.configs, "no configs found")

    def _load(self, path):
        with open(path, "r") as handle:
            return yaml.safe_load(handle)

    def test_every_config_declares_a_split_seed(self):
        for path in self.configs:
            with self.subTest(config=os.path.basename(path)):
                self.assertIn("split_seed", self._load(path)["train"])

    def test_all_configs_share_one_subject_partition(self):
        split_seeds = {self._load(p)["train"]["split_seed"] for p in self.configs}
        self.assertEqual(
            split_seeds, {42},
            "seed variants must differ only in initialisation, so every config has to "
            f"name the same partition; found {sorted(split_seeds)}")

    def test_split_seed_falls_back_to_seed_for_legacy_configs(self):
        legacy = {"train": {"seed": 43}}
        self.assertEqual(resolve_seeds(legacy), (43, 43))

    def test_split_seed_is_used_when_present(self):
        config = {"train": {"seed": 43, "split_seed": 42}}
        self.assertEqual(resolve_seeds(config), (43, 42))

    def test_command_line_overrides_win(self):
        class Args:
            seed = 7
            split_seed = 42

        config = {"train": {"seed": 43, "split_seed": 44}}
        self.assertEqual(resolve_seeds(config, Args()), (7, 42))

    def test_seed_variants_now_share_folds_but_did_not_before(self):
        """The bug and the fix, side by side, on the folds themselves."""
        for fold in range(5):
            with self.subTest(fold=fold):
                fixed = [
                    get_cv_splits(self.subject_ids, num_folds=5, fold_idx=fold, seed=42)
                    for _ in (42, 43, 44)
                ]
                self.assertEqual(fixed[0], fixed[1])
                self.assertEqual(fixed[1], fixed[2])

        # The archived runs took the split from the training seed, so their folds moved.
        before = {seed: get_cv_splits(self.subject_ids, num_folds=5, fold_idx=0, seed=seed)[2]
                  for seed in (42, 43, 44)}
        self.assertNotEqual(before[42], before[43])
        self.assertNotEqual(before[42], before[44])


if __name__ == "__main__":
    unittest.main()
