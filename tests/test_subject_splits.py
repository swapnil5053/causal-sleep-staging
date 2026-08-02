import unittest

from src.data.dataset import get_cv_splits, get_subject_splits


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


if __name__ == "__main__":
    unittest.main()
