"""Figures for the two-dataset evaluation-protocol analysis.

Two figures, sized for a single IEEE column at 300 dpi, each with one panel per dataset:

  fig_latency.png    kappa against permitted latency, both arms, Sleep-EDF-78 and DOD-H
  fig_perclass.png   per-class cost under batch tiling versus under the streaming protocol

The latency panels read the CSVs written by scripts/latency_sweep.py. The per-class panels
read the same per-fold summary CSVs the paper's tables are computed from, so the figure and
the table cannot drift apart: both are the mean over three seeds x five folds.

Series are distinguished by line style and marker as well as hue, so the figures survive
greyscale printing and colour-vision deficiency; the palette is validated for CVD separation.

    python scripts/make_paper_figures.py
"""

import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CAUSAL = "#1665A8"      # blue
NONCAUSAL = "#C2622D"   # rust
INK = "#1A1A1A"
MUTED = "#6B6B6B"
GRID = "#DCDCDC"

CLASSES = ("W", "N1", "N2", "N3", "REM")
SEEDS = (42, 43, 44)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def read_curve(path):
    with open(path) as f:
        return {int(r["latency_s"]): r for r in csv.DictReader(f)}


def read_folds(patterns):
    """Mean of every numeric column over every fold row matched by the patterns.

    Patterns are globbed rather than listed so a missing seed fails loudly (no files, no
    figure) instead of silently changing what the mean is taken over.
    """
    rows = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if not matched:
            raise FileNotFoundError(pattern)
        for path in matched:
            with open(path) as f:
                rows += list(csv.DictReader(f))
    keys = [k for k in rows[0] if k != "fold"]
    return {k: sum(float(r[k]) for r in rows) / len(rows) for k in keys}


def smooth(values, window):
    """Centred moving average over latency.

    The sweep scores each latency on a 1-in-`stride` subsample of seconds, so latencies
    sharing a residue class modulo the stride share a subsample and the raw curve carries a
    sawtooth of exactly that period. Averaging over a full stride period removes it by
    construction rather than by taste. The window shrinks at the ends so no point is dropped.
    """
    out, n, half = [], len(values), window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)


def latency_panel(ax, causal, noncausal, title, window=5, show_xlabel=True):
    lat = sorted(causal)
    craw = {L: float(causal[L]["kappa"]) for L in lat}
    nraw = {L: float(noncausal[L]["kappa"]) for L in lat}
    ck = smooth([craw[L] for L in lat], window)
    nk = smooth([nraw[L] for L in lat], window)
    cd, nd = dict(zip(lat, ck)), dict(zip(lat, nk))

    # Curves are drawn smoothed, because the sampling sawtooth is an artefact of the stride.
    # Every annotated number is read off the raw curve, so the figure quotes exactly the
    # values the text does rather than the smoother's version of them.
    cross = next((L for L in lat if nraw[L] > craw[L]), None)
    c_best = max(lat, key=lambda L: craw[L])
    n_best = max(lat, key=lambda L: nraw[L])

    style_axes(ax)
    ax.plot(lat, ck, color=CAUSAL, linewidth=1.6, label="Causal", zorder=3)
    ax.plot(lat, nk, color=NONCAUSAL, linewidth=1.6, linestyle="--", label="Non-causal",
            zorder=3)
    if cross is not None:
        ax.axvline(cross, color=MUTED, linewidth=0.7, linestyle=":", zorder=1)

    ax.set_title(title, loc="left", pad=3)
    if show_xlabel:
        ax.set_xlabel("Permitted latency $L$ (s)\n"
                      "larger $L$: less history, more lookahead", labelpad=2)
    ax.set_ylabel("Cohen's $\\kappa$")
    ax.set_xlim(-3, 122)
    ax.set_ylim(0.21, 0.76)
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.03), frameon=False,
              handlelength=2.2, fontsize=6.8)

    # The decision region is a narrow band at the top of the full range, so it gets its own
    # inset. The full range stays visible because the causal collapse at large L is the
    # mechanism being demonstrated, not a distraction.
    zx = [L for L in lat if L <= 60]
    lo = min(min(cd[L] for L in zx), min(nd[L] for L in zx))
    hi = max(max(cd[L] for L in zx), max(nd[L] for L in zx))
    pad = (hi - lo) * 0.30
    axi = ax.inset_axes([0.36, 0.11, 0.60, 0.40])
    axi.plot(zx, [cd[L] for L in zx], color=CAUSAL, linewidth=1.3)
    axi.plot(zx, [nd[L] for L in zx], color=NONCAUSAL, linewidth=1.3, linestyle="--")
    if cross is not None and cross <= 60:
        axi.axvline(cross, color=MUTED, linewidth=0.7, linestyle=":")
        axi.annotate(f"crossover {cross} s", xy=(cross, lo - pad * 0.15),
                     fontsize=5.5, color=MUTED,
                     ha="left" if cross < 30 else "center", va="bottom")
    axi.plot([0], [craw[0]], marker="o", markersize=3.4, color=CAUSAL,
             markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    axi.plot([n_best], [nraw[n_best]], marker="s", markersize=3.4, color=NONCAUSAL,
             markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    axi.annotate(f"{craw[0]:.4f} at $L$=0", xy=(0, craw[0]), xytext=(4, 3),
                 textcoords="offset points", fontsize=5.5, color=CAUSAL, ha="left")
    axi.annotate(f"{nraw[n_best]:.4f} at {n_best} s", xy=(n_best, nraw[n_best]),
                 xytext=(-1, 3), textcoords="offset points", fontsize=5.5,
                 color=NONCAUSAL, ha="center")
    axi.set_ylim(lo - pad * 0.5, hi + pad)
    axi.set_xlim(-2, 62)
    axi.set_xticks([0, 20, 40, 60])
    axi.tick_params(labelsize=5.5, length=2, pad=1)
    axi.grid(True, color=GRID, linewidth=0.4)
    axi.set_axisbelow(True)
    for s in axi.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(0.5)
    axi.set_title("detail, $L \\leq 60$ s", fontsize=5.5, color=MUTED, pad=2)
    return cross, c_best, n_best


def perclass_panel(ax, tiled, streaming, title, show_xlabel=True, show_legend=False):
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    y = range(len(CLASSES))
    h = 0.34
    ax.barh([v + h / 2 for v in y], tiled, height=h, color=NONCAUSAL,
            label="Batch tiling (all buffer positions)", zorder=3)
    ax.barh([v - h / 2 for v in y], streaming, height=h, color=CAUSAL,
            label="Streaming protocol (deployment)", zorder=3)
    ax.axvline(0, color=INK, linewidth=0.8, zorder=4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(CLASSES)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", pad=3)
    if show_xlabel:
        ax.set_xlabel("F1 the causal model gives up   ($\\leftarrow$ causal better)")
    lim = max(max(map(abs, tiled)), max(map(abs, streaming))) * 1.6
    ax.set_xlim(-lim, lim)
    if show_legend:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=1, frameon=False,
                  fontsize=6.8, handlelength=1.4, borderpad=0, labelspacing=0.35)
    for i, (t, s) in enumerate(zip(tiled, streaming)):
        off = lim * 0.02
        ax.text(t + (off if t >= 0 else -off), i + h / 2, f"{t:+.3f}", va="center",
                ha="left" if t >= 0 else "right", fontsize=6.0, color=MUTED)
        ax.text(s + (off if s >= 0 else -off), i - h / 2, f"{s:+.3f}", va="center",
                ha="left" if s >= 0 else "right", fontsize=6.0, color=MUTED)


def perclass_deltas(causal_dirs, noncausal_dirs, summary):
    c = read_folds([os.path.join(d, summary) for d in causal_dirs])
    n = read_folds([os.path.join(d, summary) for d in noncausal_dirs])
    return [n[f"f1_{k}"] - c[f"f1_{k}"] for k in CLASSES]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--sleepedf_runs", default="logs_78streaming_{arm}_s{seed}",
                    help="directory template holding the Sleep-EDF per-fold summaries")
    ap.add_argument("--dodh_runs", default="logs_dodh_{arm}_s{seed}")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    def runs(template, arm):
        return [template.format(arm=arm, seed=s) for s in SEEDS]

    os.makedirs(args.out, exist_ok=True)
    r = args.results

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.0))
    a = latency_panel(axes[0], read_curve(os.path.join(r, "latency_causal_s5.csv")),
                      read_curve(os.path.join(r, "latency_noncausal_s5.csv")),
                      "(a) Sleep-EDF-78", show_xlabel=False)
    b = latency_panel(axes[1], read_curve(os.path.join(r, "latency_dodh_causal.csv")),
                      read_curve(os.path.join(r, "latency_dodh_noncausal.csv")),
                      "(b) DOD-H")
    fig.subplots_adjust(hspace=0.34)
    out = os.path.join(args.out, "fig_latency.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    for name, (cross, cb, nb) in (("Sleep-EDF-78", a), ("DOD-H", b)):
        print(f"  {name}: crossover {cross}s, causal best L={cb}, non-causal best L={nb}")

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.4))
    for ax, (name, template, first) in zip(axes, (
            ("(a) Sleep-EDF-78", args.sleepedf_runs, True),
            ("(b) DOD-H", args.dodh_runs, False))):
        tiled = perclass_deltas(runs(template, "causal"), runs(template, "noncausal"),
                                "test_metrics_summary.csv")
        stream = perclass_deltas(runs(template, "causal"), runs(template, "noncausal"),
                                 "test_metrics_summary_streaming30.csv")
        perclass_panel(ax, tiled, stream, name, show_xlabel=not first, show_legend=first)
        print(f"  {name} tiled    : "
              + ", ".join(f"{k} {v:+.4f}" for k, v in zip(CLASSES, tiled)))
        print(f"  {name} streaming: "
              + ", ".join(f"{k} {v:+.4f}" for k, v in zip(CLASSES, stream)))
    fig.subplots_adjust(hspace=0.40)
    out = os.path.join(args.out, "fig_perclass.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
