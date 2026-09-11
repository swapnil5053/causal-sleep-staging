# Stage-change reporting latency

Read from 5 archived prediction file(s). A stage change counts as reported once the new stage has been held for 10 consecutive seconds, so the report lands on the last second of that run. Search window: 30 s before to 300 s after the scored change.

Both systems are charged for the evidence they need. The per-second system reports at the end of its hold window; the 30-second system cannot report epoch *k* until epoch *k* has elapsed, so it carries a structural floor of 30 s on any change that begins an epoch. Negative values mean the change was reported before the scored boundary.

**20002 scored stage changes.**

In 6625 of them (33.1%) the per-second model was already emitting the incoming stage when the search window opened, 30 s before the scored change. Those are more likely false positives during the outgoing stage than genuine early detections, so the clean subset below excludes them.

## Overall

| System | Detected | Median delay | Mean delay | IQR |
|---|---:|---:|---:|---|
| Per-second (1 Hz) | 16510 / 20002 | 10 s | 29.2 s | -21 to 42 s |
| 30-second majority | 16184 / 20002 | 30 s | 46.7 s | 0 to 60 s |

### Excluding changes the model was already calling

| System | Detected | Median delay | Mean delay | IQR |
|---|---:|---:|---:|---|
| Per-second (1 Hz) | 9898 / 13377 | 33 s | 59.7 s | 14 to 79 s |
| 30-second majority | 9562 / 13377 | 60 s | 78.8 s | 30 to 90 s |

This is the conservative reading, and the one to quote.

## Head to head

Restricted to the 9469 changes both systems reported, excluding any the model was already calling before the window opened.

| Outcome | Count | Share |
|---|---:|---:|
| Per-second reports earlier | 8973 | 94.8% |
| Same second | 56 | 0.6% |
| 30-second reports earlier | 440 | 4.6% |

Median time saved by the per-second output: **19 s** (mean 20.7 s).

## By stage entered

| Stage | Changes | Per-second median | 30-second median | Difference |
|---|---:|---:|---:|---:|
| W | 3002 | 31 s | 60 s | +29 s |
| N1 | 4005 | 26 s | 60 s | +34 s |
| N2 | 4266 | 41 s | 60 s | +19 s |
| N3 | 1183 | 38 s | 60 s | +22 s |
| REM | 921 | 35 s | 60 s | +25 s |

## Reading this

A negative or small positive per-second median means the model commits to the new stage inside the epoch the technician assigned it to, which an epoch-level system cannot do at all. That is the concrete argument for the finer output rate, expressed in seconds rather than in agreement.

Two caveats belong with any figure quoted from this table. The scored boundary is itself only located to the nearest 30 s, so these numbers describe latency against the *scored* transition and not against the physiological one. And a hold requirement trades detection speed against false alarms; sweep `--hold` before quoting a single number.

