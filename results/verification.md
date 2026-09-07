# Verification and benchmarks

Two things that need checking rather than analysing: whether the system is actually causal,
and whether it is fast enough to run in real time.

---

## 1. Causality verification

Each check perturbs the input at a future time and requires every earlier value to be
**bit-identical**, not merely close. Written by `scripts/verify_causality.py`, which exits
non-zero on failure so it can gate a commit.

- Config: `configs/sleep78_streaming_causal.yaml`
- Normalization: `causal_rolling`, trailing window 30 s
- Model weights: randomly initialised — causality does not depend on trained values
- Probe: 240 s of synthetic EEG at 100 Hz, perturbed from second 120

| Check | Result | Detail |
|---|---|---|
| Normalization ignores future samples | **PASS** | max \|difference\| before sample 12,000 = 0.000e+00 |
| Offline arrays equal the online, sample-at-a-time filter | **PASS** | max \|offline − streaming\| = 5.240e−14 over 24,000 samples |
| Normalized output is finite through the cold start | **PASS** | max \|z\| = 4.932, bound √(window−1) = 54.763 |
| Configuration declares a causal model | **PASS** | `causal: true` |
| Raw sample → normalization → model: no future leakage | **PASS** | max \|logit difference\| for seconds 0–119 = 0.000e+00 |
| The probe actually perturbs later outputs | **PASS** | seconds 120+ differ as expected |
| Non-causal control does leak | **PASS** | removing the masks and left-padding lets future signal reach earlier outputs |

**All checks pass.** For this configuration the prediction at second *t* is a function of
the raw signal up to second *t* only, through both preprocessing and the network, and the
offline arrays used for training are exactly what a sample-at-a-time streaming
implementation produces.

The last two rows matter as much as the first five. A test that can never fail always
passes; the perturbation must be shown to move later outputs, and the non-causal ablation
must be shown to fail the same test. Both hold, so the pass is a real result rather than a
vacuous one.

---

## 2. CPU inference latency

Batch size 1, single thread (`torch.set_num_threads(1)`), 100 warm-up calls, 1000 timed
calls. Median and IQR reported rather than the mean, which is sensitive to occasional OS
scheduling noise. Random untrained weights, since latency does not depend on trained
parameter values. The normalizer is benchmarked separately from the model forward pass, so
a normalizer slowdown cannot be mistaken for a network one.

Measured on an Intel Core i9-14900HX.

### Model forward pass

| Config | Context (s) | Median (ms) | IQR (ms) | ms per s of EEG | Margin vs 3 ms/s |
|---|---:|---:|---:|---:|---:|
| C0 | 60 | 1.298 | [1.290, 1.311] | 0.0216 | 139× |
| C1 | 120 | 3.248 | [3.214, 3.290] | 0.0271 | 111× |

### StreamingZScore, 30 s trailing window

Median 0.792 µs per sample, IQR [0.791, 0.833] — negligible against the forward pass.

### Note on the earlier benchmark

C0 does strictly less work per call than C1 — 60 s of context against 120 s — so its median
latency must be lower. The previous benchmark reported the opposite. It used the mean
without thread pinning and generated its random input samples inside the timed region.
Both faults are corrected here; these are the numbers to report.

C2 has not been re-measured under this protocol and is omitted rather than quoted from the
old run.
