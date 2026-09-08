# v10 patch: context length and the fixed-buffer objection

Four edits. All numbers computed from `results/lat_*_b240.csv` and `lat_*_b360.csv`, no retraining.

The patch does two jobs. It adds a clean, defensible result (the causal model's exact
context-response curve, and an optimum at 145 s rather than the 120 s it was trained at). And
it states honestly that the obvious objection to the latency argument — that the fixed 120 s
buffer forces a trade a real offline system would not accept — is *not* settled by these
weights, and names the experiment that would settle it.

Conceding this is the right call. A reviewer who raises the objection and finds it already
raised, quantified as far as it can be, and answered with a specific next experiment, reads a
careful paper. One who raises it and finds silence reads an oversight.

---

## Edit 1 — new subsection, insert after Section VI-C (the latency subsection)

```latex
\subsection{Context Length and the Fixed-Buffer Objection}
\label{sec:buffer}

Section~\ref{sec:latency} measures latency within a buffer of fixed length, so permitting $L$
seconds of lookahead necessarily costs $L$ seconds of history. An offline system would not
accept that trade: it would enlarge the buffer and take both. Whether the result survives a
larger buffer is therefore a fair question, and we address it as far as these weights allow.

For the causal arm the question is answered exactly. Left-padded convolutions, left-aligned
pooling and a causal attention mask together make the output at buffer position $p$ a function
of inputs $0 \ldots p$ only, so scoring position $p$ of a long window is bit-identical to
scoring the last position of a window of length $p+1$. Evaluating one long buffer therefore
yields the model's entire context-response curve at once, and that curve cannot depend on the
buffer it was measured in. Table~\ref{tab:context} confirms this empirically: measured in
buffers of 120, 240 and 360~s, $\kappa$ at any fixed history agrees to within $0.0035$.

\begin{table}[t]
\caption{Causal Arm: $\kappa$ Against Available History (Sleep-EDF-78, C1, Buffer 360~s)}
\label{tab:context}
\centering
\begin{tabular}{lccccccc}
\toprule
History (s) & 30 & 60 & 90 & 120 & \textbf{145} & 240 & 360 \\
\midrule
$\kappa$ & 0.658 & 0.690 & 0.701 & 0.704 & \textbf{0.705} & 0.698 & 0.683 \\
\bottomrule
\end{tabular}
\end{table}

The curve rises steeply to about 90~s, is nearly flat from 90 to 180~s, peaks at
approximately 145~s, and declines beyond. The optimum is located independently at 145~s in the
240~s buffer and 146~s in the 360~s buffer. Two things follow. A deployed causal system should
use a buffer of roughly 145~s rather than the 120~s used here, which is worth about
$0.005\,\kappa$ at no latency cost. And the decline past 180~s is at least partly a train/test
length mismatch, since these weights were trained on 120~s sequences; it does not establish
that a model trained at longer context would not benefit.

For the non-causal arm the same experiment is not interpretable, and it is worth being explicit
about why. Enlarging the buffer necessarily lengthens the sequence its unmasked attention
operates over, and unlike the causal arm it has no mask limiting the effective span.
Table~\ref{tab:contamination} isolates the effect: holding history fixed and increasing only
the available lookahead reduces $\kappa$ monotonically.

\begin{table}[t]
\caption{Non-Causal Arm: $\kappa$ at Fixed History as the Buffer Grows}
\label{tab:contamination}
\centering
\begin{tabular}{lccc}
\toprule
 & \multicolumn{3}{c}{Buffer} \\
History & 120~s & 240~s & 360~s \\
\midrule
\hphantom{0}60~s & 0.6942 & 0.6714 & 0.6550 \\
\hphantom{0}90~s & 0.6971 & 0.6820 & 0.6666 \\
120~s            & 0.6870 & 0.6886 & 0.6757 \\
\bottomrule
\end{tabular}
\end{table}

Additional lookahead cannot itself degrade a model that is free to ignore it, so this is the
signature of attention operating over sequences longer than those it was trained on rather than
a property of lookahead. Consequently the natural test, giving the non-causal arm 120~s of
history and 120~s of lookahead in a 240~s buffer, returns $\kappa = 0.689$, which is below the
causal arm's $0.704$ at the same history with no lookahead, but which we do not treat as
decisive: the figure is depressed by an amount these experiments cannot separate from the
comparison.

We therefore state the scope of Section~\ref{sec:latency}'s result precisely. Within the buffer
both arms were trained for, the causal model at zero latency is not beaten by the non-causal
model at any latency. Whether a non-causal model \emph{trained} at a longer context, and so able
to use full history and full lookahead together, would exceed it is untested here and is the
clearest experiment this analysis calls for. It requires retraining both arms at the longer
context and is left to future work.
```

---

## Edit 2 — Limitations, replace the normalization-window paragraph

Replace:

> The normalization window is a free parameter. A 30~s trailing window was the only setting
> trained, chosen to match the epoch length it replaces, so whether the results depend on it is
> untested.

with:

```latex
Two free parameters are untested. The normalization window was fixed at 30~s to match the epoch
length it replaces, and no other setting was trained. The context length was fixed at 120~s;
Section~\ref{sec:buffer} shows post hoc that approximately 145~s would have been better for the
causal arm, and that the question of whether a non-causal model trained at longer context could
use history and lookahead together to exceed the causal arm at zero latency is open. Both
require retraining and neither is resolved here.
```

---

## Edit 3 — Discussion, replace the sentence bounding the claim

Replace:

> Second, the non-causal arm here is our own matched twin, not a state-of-the-art offline model,
> and the buffer is fixed at 120~s. A different architecture or a longer buffer could move the
> crossover.

with:

```latex
Second, the non-causal arm here is our own matched twin rather than a state-of-the-art offline
model, so the comparison bounds what causality costs within this capacity class and not in
general. Third, the buffer is fixed at the length both arms were trained for.
Section~\ref{sec:buffer} shows that the causal arm's context-response curve can be recovered
exactly from a longer buffer, because masking makes it buffer-invariant, but that the same
manoeuvre does not work for the non-causal arm: enlarging its buffer lengthens the sequence its
unmasked attention spans, and the resulting degradation cannot be separated from the effect
under study. The claim is therefore scoped to the trained buffer, and a non-causal arm trained
at longer context is the experiment that would extend or overturn it.
```

---

## Edit 4 — Abstract, one clause

After "the causal model reaches $\kappa = 0.700$ with zero delay while its non-causal twin
cannot exceed $\kappa = 0.698$ at any delay", insert:

```latex
within the context buffer both were trained for;
```

so the sentence reads "...cannot exceed $\kappa = 0.698$ at any delay within the context buffer
both were trained for, and the apparent REM concentration disappears."

That one clause is what stops the claim being overstated, and it costs nothing rhetorically.

---

## What this changes about priorities

The buffer question is now the paper's clearest open experiment, and it is a *training* run:
both arms at 240~s context, one seed, five folds, ten runs. It directly defends the central
claim.

That makes it a better use of GPU than either the matched 30-run sweep (tighter error bars on a
claim that is no longer the headline) or a causal-only ctx300 run (Table~\ref{tab:context}
already suggests the available gain is about $0.005\,\kappa$). Not needed for the preprint.
Worth having before a journal submission.
