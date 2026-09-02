# Seed / split overlap simulation

`get_cv_splits()` takes a single `seed` argument that controls both the subject shuffle and (elsewhere) weight initialisation. This simulates 5-fold splits over 78 synthetic subjects at seeds 42, 43, 44 and measures how much the *test set* of each fold index changes across seeds. If seeds only changed initialisation, overlap would be 100%. It is not.

## Test-subject overlap per fold, across seed pairs

| Fold | 42-43 | 42-44 | 43-44 | Fold size |
|---|---|---|---|---:|
| 0 | 4/16 | 2/16 | 3/16 | 16 |
| 1 | 4/16 | 2/16 | 2/16 | 16 |
| 2 | 3/16 | 3/16 | 4/16 | 16 |
| 3 | 2/15 | 2/15 | 3/15 | 15 |
| 4 | 1/15 | 4/15 | 2/15 | 15 |

Average test-subject overlap across all folds and seed pairs: **2.7 subjects** out of an average fold size of 15.6 (fold sizes: [16, 16, 16, 15, 15]) -- about 18% overlap.

This confirms that `seed` conflates weight initialisation with the subject split: results reported for "fold 0" under different seeds are not the same held-out subjects, so per-seed "fold 0" numbers are not directly comparable across seeds. The fix (already applied going forward: see the separated `split_seed` / `init_seed` config change) is to fix the split seed independently of the initialisation seed, so repeated-seed runs share folds and only initialisation varies.
