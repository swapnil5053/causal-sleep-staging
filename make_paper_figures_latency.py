"""Figures for the latency analysis.

Two figures, sized for a single IEEE column at 300 dpi:

  fig_latency.png    kappa against permitted latency, both arms
  fig_perclass.png   per-class cost under batch tiling versus at deployment latency

Both read the CSVs written by scripts/latency_sweep.py. Series are distinguished by
line style and marker as well as hue, so the figures survive greyscale printing and
colour-vision deficiency; the palette itself is validated for CVD separation.

    python make_paper_figures.py
"""

import csv
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
RESULTS = "results"

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


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {int(r["latency_s"]): r for r in rows}


def smooth(values, window):
    """Centred moving average over latency.

    The sweep scores each latency on a 1-in-`stride` subsample of seconds, so latencies
    sharing a residue class modulo the stride share a subsample and the raw curve carries
    a sawtooth of exactly that period. Averaging over a full stride period removes it by
    construction rather than by taste. The window shrinks at the ends so no point is
    dropped.
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


def fig_latency(c, n, out, window=5):
    lat = sorted(c)
    ck = smooth([float(c[L]["kappa"]) for L in lat], window)
    nk = smooth([float(n[L]["kappa"]) for L in lat], window)
    cd = dict(zip(lat, ck))
    nd = dict(zip(lat, nk))

    cross = next((L for L in lat if nd[L] > cd[L]), None)
    c_best = max(lat, key=lambda L: cd[L])
    n_best = max(lat, key=lambda L: nd[L])
    c = {L: {"kappa": cd[L]} for L in lat}
    n = {L: {"kappa": nd[L]} for L in lat}

    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    style_axes(ax)

    ax.plot(lat, ck, color=CAUSAL, linewidth=1.6, linestyle="-", label="Causal", zorder=3)
    ax.plot(lat, nk, color=NONCAUSAL, linewidth=1.6, linestyle="--", label="Non-causal",
            zorder=3)
    if cross is not None:
        ax.axvline(cross, color=MUTED, linewidth=0.7, linestyle=":", zorder=1)

    ax.set_xlabel("Permitted latency $L$ (s)\n"
                  "larger $L$: less history, more lookahead", labelpad=2)
    ax.set_ylabel("Cohen's $\\kappa$")
    ax.set_xlim(-3, 122)
    ax.set_ylim(0.28, 0.74)
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02), frameon=False,
              handlelength=2.2)

    # The decision region is a narrow band at the top of the full range, so it gets its
    # own inset. The full range stays visible because the causal collapse at large L is
    # the mechanism being demonstrated, not a distraction.
    zx = [L for L in lat if L <= 60]
    axi = ax.inset_axes([0.36, 0.10, 0.58, 0.40])
    axi.plot(zx, [float(c[L]["kappa"]) for L in zx], color=CAUSAL, linewidth=1.4)
    axi.plot(zx, [float(n[L]["kappa"]) for L in zx], color=NONCAUSAL, linewidth=1.4,
             linestyle="--")
    if cross is not None:
        axi.axvline(cross, color=MUTED, linewidth=0.7, linestyle=":")
        axi.annotate(f"crossover\n{cross} s", xy=(cross, 0.7045), fontsize=5.8,
                     color=MUTED, ha="center", va="top")
    axi.plot([c_best], [float(c[c_best]["kappa"])], marker="o", markersize=3.8,
             color=CAUSAL, markeredgecolor="white", markeredgewidth=0.7, zorder=5)
    axi.plot([n_best], [float(n[n_best]["kappa"])], marker="s", markersize=3.8,
             color=NONCAUSAL, markeredgecolor="white", markeredgewidth=0.7, zorder=5)
    axi.annotate(f"{float(c[c_best]['kappa']):.3f}", xy=(c_best, float(c[c_best]["kappa"])),
                 xytext=(3, 1), textcoords="offset points", fontsize=5.8, color=CAUSAL)
    axi.annotate(f"{float(n[n_best]['kappa']):.3f} at {n_best} s",
                 xy=(n_best, float(n[n_best]["kappa"])), xytext=(-2, -9),
                 textcoords="offset points", fontsize=5.8, color=NONCAUSAL, ha="center")
    axi.set_ylim(0.6825, 0.7075)
    axi.set_xlim(-2, 62)
    axi.set_xticks([0, 20, 40, 60])
    axi.set_yticks([0.685, 0.695, 0.705])
    axi.tick_params(labelsize=5.8, length=2, pad=1)
    axi.grid(True, color=GRID, linewidth=0.4)
    axi.set_axisbelow(True)
    for s in axi.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(0.5)
    axi.set_title("detail, $L \\leq 60$ s (5-point moving average)",
                  fontsize=5.8, color=MUTED, pad=2)

    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  (crossover {cross}s, causal best L={c_best}, "
          f"non-causal best L={n_best})")


def fig_perclass(c, n, out):
    # Cost the causal arm pays, positive = causal worse.
    tiled = [sum(float(n[L][f"f1_{k}"]) - float(c[L][f"f1_{k}"]) for L in c) / len(c)
             for k in CLASSES]
    deployed = [float(n[0][f"f1_{k}"]) - float(c[0][f"f1_{k}"]) for k in CLASSES]

    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    style_axes(ax)
    ax.grid(axis="y", visible=False)

    y = range(len(CLASSES))
    h = 0.34
    ax.barh([v + h / 2 for v in y], tiled, height=h, color=NONCAUSAL,
            label="Batch tiling (all buffer positions)", zorder=3)
    ax.barh([v - h / 2 for v in y], deployed, height=h, color=CAUSAL,
            label="At deployment latency ($L$ = 0)", zorder=3)

    ax.axvline(0, color=INK, linewidth=0.8, zorder=4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(CLASSES)
    ax.invert_yaxis()
    ax.set_xlabel("F1 the causal model gives up   ($\\leftarrow$ causal better)")
    lim = max(max(map(abs, tiled)), max(map(abs, deployed))) * 1.55
    ax.set_xlim(-lim, lim)
    # Above the axes: the bars run the full width, so any in-plot legend collides.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=1, frameon=False,
              fontsize=6.8, handlelength=1.4, borderpad=0, labelspacing=0.35)

    for i, (t, d) in enumerate(zip(tiled, deployed)):
        ax.text(t + (0.0016 if t >= 0 else -0.0016), i + h / 2, f"{t:+.3f}",
                va="center", ha="left" if t >= 0 else "right", fontsize=6.3, color=MUTED)
        ax.text(d + (0.0016 if d >= 0 else -0.0016), i - h / 2, f"{d:+.3f}",
                va="center", ha="left" if d >= 0 else "right", fontsize=6.3, color=MUTED)

    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    print("  tiled   :", ", ".join(f"{k} {v:+.4f}" for k, v in zip(CLASSES, tiled)))
    print("  deployed:", ", ".join(f"{k} {v:+.4f}" for k, v in zip(CLASSES, deployed)))


def main():
    c = read(os.path.join(RESULTS, "latency_causal_s5.csv"))
    n = read(os.path.join(RESULTS, "latency_noncausal_s5.csv"))
    os.makedirs("figures", exist_ok=True)
    fig_latency(c, n, os.path.join("figures", "fig_latency.png"))
    fig_perclass(c, n, os.path.join("figures", "fig_perclass.png"))


if __name__ == "__main__":
    main()
