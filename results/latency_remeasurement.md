# Latency re-measurement

Protocol: batch size 1, single thread (`torch.set_num_threads(1)`), 100 warm-up calls, 1000 timed calls, median and IQR reported (not mean, which the previous benchmark used and which is sensitive to occasional OS scheduling noise). Random (untrained) weights, since latency does not depend on trained parameter values. `StreamingZScore` is benchmarked separately from the model forward pass, since the previous benchmark only measured the model and could not distinguish a normalizer slowdown from a network slowdown.

## Model forward pass

| Config | Context (s) | Median (ms) | IQR (ms) | ms per sec of EEG |
|---|---:|---:|---:|---:|
| C0 (60s context) | 60 | 1.298 | [1.290, 1.311] | 0.0216 |
| C1 (120s context) | 120 | 3.248 | [3.214, 3.290] | 0.0271 |

## StreamingZScore (30 s trailing window, measured separately)

- Median: 0.792 us per sample
- IQR: [0.791, 0.833] us per sample

## Comparison

C0 (60s context) does strictly less work per call than C1 (120s context) (60s vs 120s of context), so its median latency should be lower per call. The previous mean-based, non-thread-pinned benchmark reported the opposite; this corrected measurement is the one to present.
