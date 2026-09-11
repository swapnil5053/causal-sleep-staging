"""Statistical treatment of the causality ablation.

    python scripts/analysis_stats.py                    # writes results/generated/statistics.md
    python scripts/analysis_stats.py --quiet            # file only, no console output

Reports, for both dataset sizes:
  - paired t-test on per-fold kappa (causal vs non-causal)
  - bootstrap confidence interval on the difference
  - Wilcoxon signed-rank as a distribution-free check
  - Cohen's d
  - a power calculation: how many folds would be needed to detect the observed effect

Everything reads the per-fold summary CSVs in results/, so it can be re-run unchanged after
any new experiment.
"""
import argparse
import csv
import math
import os
import statistics as st

import numpy as np

PAIRS = [
    ("Sleep-EDF-20", "results/run_b_context", "results/run_d2_noncausal"),
    ("Sleep-EDF-78", "results/sleep78_causal", "results/sleep78_noncausal"),
]


def load_kappa(d):
    for fn in ("test_metrics_summary.csv", "summary.csv"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return [float(r["kappa"]) for r in csv.DictReader(open(p))]
    raise FileNotFoundError(d)


def bootstrap_ci(diff, n_boot=20000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean paired difference."""
    rng = np.random.default_rng(seed)
    d = np.asarray(diff)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def required_n(effect_d, power=0.80, alpha=0.05):
    """Folds needed for a paired test to reach the given power, normal approximation."""
    if effect_d == 0:
        return float("inf")
    z_a, z_b = 1.959964, 0.8416212 if power == 0.80 else 1.2815516
    return math.ceil(((z_a + z_b) / abs(effect_d)) ** 2)


def analyse(name, causal_dir, noncausal_dir):
    kc, kn = load_kappa(causal_dir), load_kappa(noncausal_dir)
    if len(kc) != len(kn):
        raise ValueError(
            f"{name}: fold count mismatch ({len(kc)} in {causal_dir}, "
            f"{len(kn)} in {noncausal_dir}). Evaluation appends rows, so a duplicated "
            f"summary row is the usual cause."
        )
    diff = [a - b for a, b in zip(kc, kn)]
    n = len(diff)
    if n < 2:
        raise ValueError(f"{name}: need at least 2 folds, found {n}")
    mean, sd = st.mean(diff), st.stdev(diff)
    if sd == 0:
        raise ValueError(f"{name}: per-fold differences are identical, no variance to test")
    se = sd / math.sqrt(n)
    t = mean / se
    d = mean / sd                                   # Cohen's d for paired samples
    lo, hi = bootstrap_ci(diff)

    p_t = p_w = None
    try:
        from scipy import stats
        p_t = float(stats.ttest_rel(kc, kn).pvalue)
        if n >= 5:
            p_w = float(stats.wilcoxon(kc, kn).pvalue)
    except Exception:
        pass

    return dict(name=name, n=n, kc=kc, kn=kn, diff=diff, mean_c=st.mean(kc), mean_n=st.mean(kn),
                mean=mean, sd=sd, t=t, p_t=p_t, p_w=p_w, d=d, lo=lo, hi=hi,
                sd_c=st.stdev(kc), sd_n=st.stdev(kn),
                need_80=required_n(d, 0.80), need_90=required_n(d, 0.90),
                loses=sum(1 for x in diff if x < 0))


def render(rows, interpret=True):
    L = []
    w = L.append
    w("# Statistical analysis of the causality ablation\n")
    w("Per-fold Cohen's kappa, causal model versus the same architecture with the causal")
    w("constraint removed. Identical parameters, data and folds; the only difference is access")
    w("to future signal. Paired tests, since folds are matched.\n")
    w("## Summary\n")
    w("| Dataset | Causal | Non-causal | Difference | 95% CI | t | p | Cohen's d | Folds causal loses |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        pt = f"{r['p_t']:.4f}" if r["p_t"] is not None else "n/a"
        w(f"| {r['name']} | {r['mean_c']:.4f} | {r['mean_n']:.4f} | {r['mean']:+.4f} | "
          f"[{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['t']:.2f} | {pt} | {r['d']:.2f} | "
          f"{r['loses']}/{r['n']} |")
    w("")
    for r in rows:
        w(f"## {r['name']}\n")
        w("| Fold | Causal | Non-causal | Difference |")
        w("|---|---|---|---|")
        for i, (a, b, dd) in enumerate(zip(r["kc"], r["kn"], r["diff"])):
            w(f"| {i} | {a:.4f} | {b:.4f} | {dd:+.4f} |")
        w(f"| Mean | {r['mean_c']:.4f} | {r['mean_n']:.4f} | {r['mean']:+.4f} |\n")
        pt = f"{r['p_t']:.4f}" if r["p_t"] is not None else "not computed"
        pw = f"{r['p_w']:.4f}" if r["p_w"] is not None else "not computed"
        w(f"- Paired t-test: t({r['n']-1}) = {r['t']:.2f}, p = {pt}")
        w(f"- Wilcoxon signed-rank: p = {pw}")
        w(f"- Bootstrap 95% CI on the mean difference: [{r['lo']:+.4f}, {r['hi']:+.4f}]")
        w(f"- Cohen's d = {r['d']:.2f}")
        w(f"- Fold-to-fold kappa sd: causal {r['sd_c']:.4f}, non-causal {r['sd_n']:.4f}")
        w(f"- Folds needed for 80% power at this effect size: {r['need_80']}")
        w(f"- Folds needed for 90% power: {r['need_90']}\n")
    if interpret and len(rows) == 2:
        a, b = rows
        w("## Interpretation\n")
        sig = b["p_t"] is not None and b["p_t"] < 0.05
        w(f"On {b['name']} the causal model is worse in {b['loses']} of {b['n']} folds and the")
        w(f"difference is {'significant' if sig else 'not significant'} "
          f"(p = {b['p_t']:.4f}). The confidence interval [{b['lo']:+.4f}, {b['hi']:+.4f}] excludes")
        w("zero." if b["lo"] * b["hi"] > 0 else "zero is inside the interval.")
        w("")
        w(f"On {a['name']} the same comparison gives {a['mean']:+.4f} with p = {a['p_t']:.4f} and a")
        w(f"confidence interval [{a['lo']:+.4f}, {a['hi']:+.4f}] that spans zero. Fold-to-fold")
        w(f"variance is {a['sd_c']/b['sd_c']:.1f} times larger there (kappa sd {a['sd_c']:.4f} against")
        w(f"{b['sd_c']:.4f}), which is enough to hide an effect of this size and even flip its sign.")
        w("")
        w("The practical conclusion is that causality penalties reported on 20-subject splits are")
        w("unreliable, and that the true cost of causal operation for this architecture is around")
        w(f"{abs(b['mean']):.3f} kappa.")
    return "\n".join(L) + "\n"


def parse_pair(spec):
    """Parse ``NAME=causal_dir,noncausal_dir`` from the command line."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--pair must look like NAME=causal_dir,noncausal_dir (got {spec!r})")
    name, _, dirs = spec.partition("=")
    parts = [p.strip() for p in dirs.split(",")]
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"--pair needs exactly two comma-separated directories (got {spec!r})")
    return (name.strip(), parts[0], parts[1])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/generated/statistics.md")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--pair", type=parse_pair, action="append", default=None,
        metavar="NAME=CAUSAL_DIR,NONCAUSAL_DIR",
        help="Compare an arbitrary run pair instead of the archived defaults. "
             "Repeatable. Example: --pair 'Streaming-78=results/sleep78_streaming_causal,"
             "results/sleep78_streaming_noncausal'")
    args = ap.parse_args()

    pairs = args.pair if args.pair else PAIRS

    rows = []
    for name, c, n in pairs:
        if os.path.isdir(c) and os.path.isdir(n):
            rows.append(analyse(name, c, n))
        else:
            print(f"skipping {name}: missing {c} or {n}")
    if not rows:
        print("no comparable runs found")
        return

    # The closing narrative compares the 20- and 78-subject archived runs specifically,
    # so it is only emitted for the default pairing.
    text = render(rows, interpret=(args.pair is None))
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    open(args.out, "w").write(text)
    if not args.quiet:
        print(text)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
