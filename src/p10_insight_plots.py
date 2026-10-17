"""Phase 10 (extra): plots that open up the MECHANISM, not just the outcome.

Built because the Phase 9 plots (tournament progression, final choice) all showed "MSFT wins" without
showing WHY, or what the underlying math is actually doing. These four answer that directly:

1. why_msft_wins.png      — MSFT's standing on all three perspectives vs. its closest rivals: not #1 on
                             any one, but the only company strong across more than one at once.
2. perspective_radar.png  — the same comparison as a radar/spider chart, which makes "consistently good
                             across 3 axes" vs. "spiky, great on one axis, bad on another" visually obvious.
3. embedding_directions.png — the 2-D PCA map of all 30 companies, with the three learned directions
                             (Finance/ESG/Risk) drawn as arrows in the SAME space, so you can see which
                             way "better Finance" actually points among the real companies.
4. worked_debate_waterfall.png — one real pair debate (NVDA vs VZ, balanced investor) as a waterfall:
                             each agent's margin, and the running evidence after each speaker.

All built from files already on disk (directions.npz, embeddings, labels.csv, tournament/debate
outputs) -- no new experiment, no API, just different views of the same real numbers.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common import PERSPECTIVES, PERSP_NAMES, load_companies, load_embeddings, load_labels
from config import DATA, OUT, set_seeds

PCOLORS = {"F": "#4C72B0", "E": "#55A868", "R": "#C44E52"}


def plot_why_msft_wins():
    lab = load_labels(load_companies().index.tolist())
    cols = ["finance_label", "esg_label", "risk_label"]
    # MSFT + the company ranked #1 on each perspective + NVDA (Finance runner-up, Risk worst)
    highlight = ["MSFT"]
    for c in cols:
        top = lab[c].idxmax()
        if top not in highlight:
            highlight.append(top)
    tickers = highlight[:6]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(tickers))
    width = 0.25
    for i, (col, name) in enumerate(zip(cols, ["Finance", "ESG", "Risk"])):
        vals = lab.loc[tickers, col].values
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name,
                      color=PCOLORS[PERSPECTIVES[i]])
        if "MSFT" in tickers:
            bars[tickers.index("MSFT")].set_edgecolor("black")
            bars[tickers.index("MSFT")].set_linewidth(2.2)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(tickers)
    ax.set_ylabel("Label score (z-scored; higher = better)")
    ax.set_title("Why MSFT keeps winning: not #1 on any single perspective,\n"
                "but strong across more than one at once (MSFT bars outlined in black)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for i, t in enumerate(tickers):
        rF, rE, rR = (lab[c].rank(ascending=False)[t] for c in cols)
        ax.text(i, -2.1, f"ranks: F#{int(rF)} E#{int(rE)} R#{int(rR)}", ha="center", fontsize=8, color="#555")
    ax.set_ylim(-2.4, lab.loc[tickers, cols].values.max() + 0.6)
    fig.tight_layout()
    fig.savefig(OUT / "why_msft_wins.png", dpi=150)
    plt.close(fig)
    print("saved why_msft_wins.png")


def plot_radar():
    lab = load_labels(load_companies().index.tolist())
    cols = ["finance_label", "esg_label", "risk_label"]
    tickers = ["MSFT", "NVDA", "XOM", "HD"]  # MSFT vs Finance-leader-but-risky vs ESG-poor vs a safe pick
    colors = ["#C44E52", "#4C72B0", "#8C6D31", "#55A868"]

    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    lo, hi = lab[cols].values.min(), lab[cols].values.max()
    for t, c in zip(tickers, colors):
        vals = lab.loc[t, cols].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=t, color=c)
        ax.fill(angles, vals, alpha=0.08, color=c)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Finance", "ESG", "Risk"], fontsize=11)
    ax.set_ylim(lo - 0.3, hi + 0.3)
    ax.set_title("Consistent vs. spiky: four companies' profile\nacross all three perspectives", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()
    fig.savefig(OUT / "perspective_radar.png", dpi=150)
    plt.close(fig)
    print("saved perspective_radar.png")


def plot_embedding_directions():
    set_seeds()
    Z, names = load_embeddings()  # default tag/variant = the one directions.npz was fit on
    comps = load_companies()
    sectors = [comps.loc[t, "sector"] for t in names]
    d = np.load(DATA / "directions.npz")
    D, dir_tickers = d["D"], list(d["tickers"])
    assert dir_tickers == names, "embedding/direction order mismatch"

    pca = PCA(n_components=2, random_state=0).fit(Z)
    X = pca.transform(Z)
    # project the directions into the SAME 2-D space (components are rows of pca.components_)
    D2 = D @ pca.components_.T
    # Fix the plot's view to the COMPANY cloud's own spread first, independent of arrow length --
    # otherwise a long arrow (ESG's 2-D shadow is actually the largest of the three, 0.65 vs
    # Finance's 0.17) forces matplotlib to auto-expand the axes, shrinking the company cloud down
    # into a corner and pushing the arrow's label into the legend.
    pad = 0.15
    xlo, xhi = X[:, 0].min(), X[:, 0].max()
    ylo, yhi = X[:, 1].min(), X[:, 1].max()
    xr, yr = xhi - xlo, yhi - ylo
    # Origin + arrow scale use the INTERQUARTILE range, not the full min/max -- three banks
    # (JPM/GS/BAC) sit well below the rest of the cloud and would otherwise drag both the
    # arrows' starting point and their length out into that mostly-empty region.
    x_iqr = np.percentile(X[:, 0], 75) - np.percentile(X[:, 0], 25)
    y_iqr = np.percentile(X[:, 1], 75) - np.percentile(X[:, 1], 25)
    target_len = 0.9 * min(x_iqr, y_iqr)
    origin = np.median(X, axis=0)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    uniq = sorted(set(sectors))
    cmap = plt.get_cmap("tab10")
    for si, s in enumerate(uniq):
        idx = [i for i, x in enumerate(sectors) if x == s]
        ax.scatter(X[idx, 0], X[idx, 1], s=55, color=cmap(si), label=s, edgecolor="k", linewidth=0.3, zorder=2, alpha=0.85)
    for i, t in enumerate(names):
        ax.annotate(t, (X[i, 0], X[i, 1]), fontsize=6.5, xytext=(3, 3), textcoords="offset points", alpha=0.8)

    label_offsets = {"Finance": (1.18, 1.18), "ESG": (1.25, 1.0), "Risk": (1.0, 1.3)}
    for k, name in enumerate(["Finance", "ESG", "Risk"]):
        unit_vec = D2[k] / np.linalg.norm(D2[k])
        dx, dy = unit_vec * target_len
        ax.annotate("", xy=(origin[0] + dx, origin[1] + dy), xytext=(origin[0], origin[1]),
                   arrowprops=dict(arrowstyle="-|>", lw=2.8, color=PCOLORS[PERSPECTIVES[k]]), zorder=5)
        lx, ly = label_offsets[name]
        ax.text(origin[0] + dx * lx, origin[1] + dy * ly, name, fontsize=12, fontweight="bold",
               color=PCOLORS[PERSPECTIVES[k]], zorder=6, ha="center",
               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.plot(*origin, "k+", markersize=10, zorder=5)

    ax.set_xlim(xlo - pad * xr, xhi + pad * xr)
    ax.set_ylim(ylo - pad * yr, yhi + pad * yr)
    ax.set_title("Company embeddings with the three learned perspective directions\n"
                "(each direction is a 384-D arrow; shown here as its shadow projected into this 2-D view,\n"
                "all rescaled to the same visible length -- only the ANGLE between arrows is meaningful, not their relative length)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "embedding_directions.png", dpi=150)
    plt.close(fig)
    print("saved embedding_directions.png")


def plot_worked_debate_waterfall():
    tickers = ["NVDA", "VZ"]
    t = json.load(open(OUT / "tournament_balanced.json"))
    # find the real NVDA-vs-VZ (or VZ-vs-NVDA) debate inside the worked trace file instead
    import re
    trace = (OUT / "debate_trace_balanced.txt").read_text(encoding="utf-8")
    block = trace.split("Debate 1: A=NVDA vs B=VZ")[1].split("TOURNAMENT")[0]
    rows = []
    for m in re.finditer(r"speaker \d+: (\w+)\s+beta=([\d.]+).*?margin ([+-][\d.]+)\s+running evidence ([+-][\d.]+)", block):
        rows.append({"agent": m.group(1), "beta": float(m.group(2)), "margin": float(m.group(3)), "running": float(m.group(4))})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    all_vals = [0.0] + df["running"].tolist()
    yspan = max(all_vals) - min(all_vals)
    running = 0.0
    for i, r in df.iterrows():
        color = {"Finance": PCOLORS["F"], "ESG": PCOLORS["E"], "Risk": PCOLORS["R"]}[r["agent"]]
        tiny = abs(r["margin"]) < 0.02 * yspan  # too thin to read a label INSIDE the bar
        ax.bar(i, r["margin"], bottom=running, color=color, width=0.6, edgecolor="black", linewidth=0.6)
        if tiny:
            # callout to the side instead of inside, since the bar itself is barely visible
            ax.annotate(f"{r['margin']:+.4f}  (too small to see)", xy=(i, running + r["margin"] / 2),
                       xytext=(i + 0.45, running - 0.12 * yspan), fontsize=9, color=color, fontweight="bold",
                       arrowprops=dict(arrowstyle="-", color=color, lw=1))
        else:
            ax.text(i, running + r["margin"] / 2, f"{r['margin']:+.4f}", ha="center", va="center",
                   fontsize=9, color="white", fontweight="bold")
        ax.plot([i - 0.3, i + 0.3], [r["running"], r["running"]], color="black", lw=1.4, ls="--", zorder=5)
        running = r["running"]
    ax.axhline(0, color="#999999", lw=1)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{r['agent']}\n(β={r['beta']:.2f})" for _, r in df.iterrows()])
    ax.set_ylabel("Margin contributed (dashed line = running evidence total)")
    ax.set_title("One real debate, step by step: NVDA vs VZ (balanced investor)\n"
                f"final evidence {running:+.4f} → {'NVDA' if running < 0 else 'VZ'} wins")
    pad = 0.18 * yspan
    ax.set_ylim(min(all_vals) - pad, max(all_vals) + pad * 0.4)
    fig.tight_layout()
    fig.savefig(OUT / "worked_debate_waterfall.png", dpi=150)
    plt.close(fig)
    print("saved worked_debate_waterfall.png")


if __name__ == "__main__":
    plot_why_msft_wins()
    plot_radar()
    plot_embedding_directions()
    plot_worked_debate_waterfall()
