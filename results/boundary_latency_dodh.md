# Stage-change reporting latency

Read from 5 archived prediction file(s). A stage change counts as reported once the new stage has been held for 10 consecutive seconds, so the report lands on the last second of that run. Search window: 30 s before to 300 s after the scored change.

Both systems are charged for the evidence they need. The per-second system reports at the end of its hold window; the 30-second system cannot report epoch *k* until epoch *k* has elapsed, so it carries a structural floor of 30 s on any change that begins an epoch. Negative values mean the change was reported before the scored boundary.

**2615 scored stage changes.**

In 557 of them (21.3%) the per-second model was already emitting the incoming stage when the search window opened, 30 s before the scored change. Those are more likely false positives during the outgoing stage than genuine early detections, so the clean subset below excludes them.

## Overall

| System | Detected | Median delay | Mean delay | IQR |
|---|---:|---:|---:|---|
| Per-second (1 Hz) | 2314 / 2615 | 17 s | 33.5 s | -2 to 41 s |
| 30-second majority | 2265 / 2615 | 30 s | 52.3 s | 30 to 60 s |

### Excluding changes the model was already calling

| System | Detected | Median delay | Mean delay | IQR |
|---|---:|---:|---:|---|
| Per-second (1 Hz) | 1758 / 2058 | 25 s | 48.8 s | 12 to 62 s |
| 30-second majority | 1708 / 2058 | 30 s | 69.1 s | 30 to 90 s |

This is the conservative reading, and the one to quote.

## Head to head

Restricted to the 1695 changes both systems reported, excluding any the model was already calling before the window opened.

| Outcome | Count | Share |
|---|---:|---:|
| Per-second reports earlier | 1621 | 95.6% |
| Same second | 9 | 0.5% |
| 30-second reports earlier | 65 | 3.8% |

Median time saved by the per-second output: **19 s** (mean 21.1 s).

## By stage entered

| Stage | Changes | Per-second median | 30-second median | Difference |
|---|---:|---:|---:|---:|
| W | 520 | 20 s | 30 s | +10 s |
| N1 | 615 | 21 s | 30 s | +9 s |
| N2 | 699 | 39 s | 60 s | +21 s |
| N3 | 115 | 42 s | 60 s | +18 s |
| REM | 109 | 35 s | 60 s | +25 s |

## Reading this

A negative or small positive per-second median means the model commits to the new stage inside the epoch the technician assigned it to, which an epoch-level system cannot do at all. That is the concrete argument for the finer output rate, expressed in seconds rather than in agreement.

Two caveats belong with any figure quoted from this table. The scored boundary is itself only located to the nearest 30 s, so these numbers describe latency against the *scored* transition and not against the physiological one. And a hold requirement trades detection speed against false alarms; sweep `--hold` before quoting a single number.

