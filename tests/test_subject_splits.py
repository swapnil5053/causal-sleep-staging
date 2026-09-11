import unittest

from src.data.dataset import get_cv_splits, get_subject_splits, resolve_split_seed


class SubjectSplitTests(unittest.TestCase):
    def setUp(self):
        self.subject_ids = [f"{subject:02d}" for subject in range(20)]

    def test_holdout_splits_are_disjoint_and_complete(self):
        train, validation, test = get_subject_splits(
            self.subject_ids,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        self.assertTrue(set(train).isdisjoint(validation))
        self.assertTrue(set(train).isdisjoint(test))
        self.assertTrue(set(validation).isdisjoint(test))
        self.assertEqual(set(train + validation + test), set(self.subject_ids))

    def test_cross_validation_fold_sets_are_disjoint(self):
        for fold_index in range(5):
            with self.subTest(fold=fold_index):
                train, validation, test = get_cv_splits(
                    self.subject_ids,
                    num_folds=5,
                    fold_idx=fold_index,
                    seed=42,
                )

                self.assertTrue(set(train).isdisjoint(validation))
                self.assertTrue(set(train).isdisjoint(test))
                self.assertTrue(set(validation).isdisjoint(test))
                self.assertEqual(set(train + validation + test), set(self.subject_ids))

    def test_every_subject_is_tested_once_across_five_folds(self):
        test_subjects = []

        for fold_index in range(5):
            _, _, fold_test_subjects = get_cv_splits(
                self.subject_ids,
                num_folds=5,
                fold_idx=fold_index,
                seed=42,
            )
            test_subjects.extend(fold_test_subjects)

        self.assertCountEqual(test_subjects, self.subject_ids)
        self.assertEqual(len(test_subjects), len(set(test_subjects)))

    def test_same_seed_produces_same_split(self):
        first = get_cv_splits(self.subject_ids, num_folds=5, fold_idx=2, seed=42)
        second = get_cv_splits(self.subject_ids, num_folds=5, fold_idx=2, seed=42)

        self.assertEqual(first, second)

    def test_different_seeds_produce_different_partitions(self):
        """The reason data.split_seed exists: the seed really does move the folds."""
        with_42 = get_cv_splits(self.subject_ids, num_folds=5, fold_idx=0, seed=42)[2]
        with_43 = get_cv_splits(self.subject_ids, num_folds=5, fold_idx=0, seed=43)[2]

        self.assertNotEqual(set(with_42), set(with_43))


class SplitSeedResolutionTests(unittest.TestCase):
    """`data.split_seed` decides the partition; the training seed decides initialisation."""

    def setUp(self):
        self.subject_ids = [f"{subject:02d}" for subject in range(20)]

    def test_defaults_to_the_training_seed(self):
        """Configs written before split_seed existed must keep their archived splits."""
        for seed in (42, 43, 44):
            with self.subTest(seed=seed):
                config = {"data": {"seed": seed}, "train": {"seed": seed}}
                self.assertEqual(resolve_split_seed(config), seed)

    def test_split_seed_overrides_the_training_seed(self):
        config = {"data": {"seed": 42, "split_seed": 42}, "train": {"seed": 44}}

        self.assertEqual(resolve_split_seed(config), 42)

    def test_training_seed_alone_cannot_move_the_folds(self):
        """Pinning split_seed must make three seeds share one partition."""
        pinned = [
            get_cv_splits(
                self.subject_ids,
                num_folds=5,
                fold_idx=0,
                seed=resolve_split_seed({"data": {"split_seed": 42}, "train": {"seed": seed}}),
            )
            for seed in (42, 43, 44)
        ]

        self.assertEqual(pinned[0], pinned[1])
        self.assertEqual(pinned[1], pinned[2])

    def test_missing_train_section_falls_back_to_the_data_seed(self):
        self.assertEqual(resolve_split_seed({"data": {"seed": 7}}), 7)


if __name__ == "__main__":
    unittest.main()
